import type { MemxStoreBundle } from "../runtime.js";
import { clamp01, normalizeText, objectRecord, randomId, stableHash, truncateText } from "../support.js";
import type {
  BackgroundRecallBundle,
  CandidateHit,
  ConversationChunk,
  ConversationTask,
  EvidenceBundle,
  EvidencePacket,
  EvidencePlanAudit,
  EvidencePlanLayer,
  EvidenceRow,
  LineageRef,
  PromptEvidenceCandidate,
  QueryEvidenceSlot,
  QueryCompileResult,
  RecallQueryShape,
  RecallBudgetPlan,
  RecallControllerTrace,
  RecallObjectiveBudget,
  RecallQualityGateDecision,
  ShallowRecallResult,
  RecallRouteEvaluation,
  RecallSelectionTrace,
  MemoryRecallPlan,
  MemoryPrimaryRouteType,
  MemoryOperationContext,
  MemoryRouteType,
  NormalizedFact,
  RecallNeedLevel,
  RecallProbeDecision,
  RecallProbeSignals,
  RecallProbeThresholds,
  RouteDecision,
  RouteEvidenceCandidate,
  RouteEvidenceDecision,
  RecallSelectionTraceEntry,
  SearchHit,
  SourceSegmentRecord,
  WorkingProjectionRole,
} from "../types.js";
import { isSnapshotFactualStateKey } from "./authority.js";
import { filterBootstrapRows, isBootstrapMemoryContamination } from "./bootstrapFilter.js";
import {
  type CandidateGenerationResult,
  candidateGenerationAuditPayload,
  generateCandidates,
} from "./candidateGeneration.js";
import { assembleEvidencePackets } from "./evidenceAssembler.js";
import { capScoreByEvidenceCoverage, evidenceCoverageForText } from "./evidenceCoverage.js";
import { snapshotMemoryLlmBudgetAudit } from "./llmBudgetAudit.js";
import { sourceRefsFromMaintenanceMetadata, uniqueMaintenanceRefs } from "./maintenanceContract.js";
import {
  buildBackgroundRecallBundle as buildBackgroundRecallBundleFromObjects,
  collectBehavioralGuidance as collectBehavioralGuidanceFromObjects,
  collectAndScheduleMemoryObjects,
  collectAndScheduleMemoryObjectsWithBudget,
  collectMemoryObjects,
  queryMemoryFacts,
  routeHintFromMemoryObjectKind,
  scheduleMemoryObjects,
  topProbeSupportForRoute,
  topScheduledMemoryScore,
  toRouteEvidenceCandidatesFromObjects,
} from "./memoryObjects.js";
import {
  dedupeEvidenceRows,
  formatFactLine,
  lineageFromMetadata,
  normalizeSearchText,
  rowsFromSearchHits,
  shouldSuppressRecallText,
  splitLabelValue,
  toEvidenceRow,
} from "./memoryObjectsHelpers.js";
import {
  createMemorySelectionObjective,
  projectScheduledMemoryObjects,
} from "./memoryObjectsProjection.js";
import { projectNamesMatch, resolveProjectReference } from "./projectIdentity.js";
import { compileQuery, compileQueryWithoutSemanticFallback } from "./queryCompiler.js";
import { roleLabelForHit } from "./reasoner.js";
import {
  type RetrievalAuditOptions,
  buildRecallAuditPayload,
  compareEvidenceRowsChronologically,
  sanitizeFocusedRecallQuery,
  summarizeBackgroundRecallBundle,
} from "./retrieveTracing.js";
import { queryAnchorSupport } from "./semantic/heuristics.js";
import { semanticTextSimilarity } from "./semantic/textSimilarity.js";
import { isQuestionLike } from "./semantics.js";
import { emitContradictionSignals, emitFullRetrievalSignals } from "./signalLedger.js";
import { isAnswerPromptLineRole, normalizeSourceRefs, promptLineRole } from "./sourceRefs.js";
import { SOURCE_SEGMENT_TARGET_CHARS } from "./sourceSegments.js";
import { semanticTaskSummaryText } from "./taskSummary.js";

const PRIMARY_ROUTE_TYPES: MemoryPrimaryRouteType[] = [
  "workflow",
  "factual",
  "temporal",
  "explanatory",
];

type RouteEvidencePack = {
  routeType: MemoryPrimaryRouteType;
  query: string;
  candidates: RouteEvidenceCandidate[];
};

type ExactSnippetCandidate = {
  snippetId: string;
  text: string;
  sourceRef?: string;
  lineage: LineageRef;
  source: "support_ref" | "projected" | "fallback";
  goalScore?: number;
};

export type RecallQueryAnalysis = QueryCompileResult;

export function analyzeRecallQuery(query: string): RecallQueryAnalysis {
  return compileQueryWithoutSemanticFallback(query, "llm-only-recall-analysis-unavailable");
}

function recallProbeThresholds(config: MemoryOperationContext["config"]): RecallProbeThresholds {
  return {
    workflowStrong: config.advanced.recallProbeWorkflowStrongThreshold,
    workflowContinuation: config.advanced.recallProbeWorkflowContinuationThreshold,
    factualStrong: config.advanced.recallProbeFactualStrongThreshold,
    factualShortQuery: config.advanced.recallProbeFactualShortQueryThreshold,
    hybridStrong: config.advanced.recallProbeHybridStrongThreshold,
    hybridModerate: config.advanced.recallProbeHybridModerateThreshold,
    escalate: config.advanced.recallProbeEscalateThreshold,
    continuationEscalate: config.advanced.recallProbeContinuationEscalateThreshold,
  };
}

export function buildBackgroundRecallBundle(
  store: MemxStoreBundle,
  ctx: MemoryOperationContext,
): BackgroundRecallBundle {
  return buildBackgroundRecallBundleFromObjects(store, ctx);
}

export function hasBackgroundRecallMaterial(bundle: BackgroundRecallBundle): boolean {
  return bundle.projectionBlocks.length > 0 || bundle.behavioralGuidance.length > 0;
}

function deriveRecallNeedLevel(params: {
  query: string;
  background: BackgroundRecallBundle;
  plan: MemoryRecallPlan;
  probe: RecallProbeDecision;
  queryAnalysis?: RecallQueryAnalysis;
  legacyTrigger: "plan" | "probe" | "both" | "none";
}): RecallControllerTrace {
  const trimmed = params.query.trim();
  const backgroundContextAvailable =
    hasBackgroundRecallMaterial(params.background) ||
    params.background.tasks.length > 0 ||
    params.background.states.length > 0;
  const queryAnalysis = params.queryAnalysis ?? analyzeRecallQuery(params.query);
  const { queryShape, routeWeights } = queryAnalysis;
  const workflowContinuity = clamp01(params.probe.signals.workflowContinuity);
  const workflowSupport = clamp01(
    Math.max(
      params.probe.signals.workflowSimilarity,
      params.probe.signals.contextAssociation * 0.88,
      workflowContinuity,
    ),
  );
  const factualSupport = clamp01(
    params.probe.signals.factualSimilarity *
      Math.min(1, Math.max(params.probe.signals.factualCandidateCount, 0) / 2) *
      (params.probe.signals.hybridHitCount > 0 ? 0.92 : 0.35),
  );
  const temporalSupport = clamp01(
    params.probe.signals.temporalSimilarity *
      Math.min(1, Math.max(params.probe.signals.hybridHitCount, 0) / 2) *
      0.84,
  );
  const explanatorySupport = clamp01(
    params.probe.signals.explanatorySimilarity *
      Math.min(1, Math.max(params.probe.signals.hybridHitCount, 0) / 2) *
      0.84,
  );
  const routeSupport = Math.max(
    workflowSupport,
    factualSupport,
    temporalSupport,
    explanatorySupport,
  );
  const lowSupportSurface =
    routeSupport < Math.min(0.52, params.probe.thresholds.hybridModerate * 0.72);
  const authoritativeQueryShape =
    queryShape.timeframe !== "timeless" ||
    queryShape.granularity === "exact_detail" ||
    queryShape.evidenceNeed === "workflow_context";
  const continuityAuthority =
    !params.plan.shouldRecall &&
    queryShape.evidenceNeed === "workflow_context" &&
    params.background.tasks.length > 0 &&
    workflowContinuity >= Math.max(0.42, params.probe.thresholds.workflowContinuation * 0.72) &&
    routeSupport >= Math.max(0.62, params.probe.thresholds.hybridModerate * 0.88) &&
    (params.probe.signals.hybridHitCount >= 2 ||
      params.probe.signals.factualCandidateCount >= 1 ||
      params.probe.signals.topProbeDocType === "event") &&
    backgroundContextAvailable;
  const exactDetailAuthority =
    queryShape.granularity === "exact_detail" &&
    (params.probe.signals.hybridHitCount >= 1 || params.probe.signals.factualCandidateCount >= 1);
  const correctedHistoricalAuthority =
    (queryShape.timeframe === "historical" || queryShape.timeframe === "compare") &&
    (params.probe.signals.hybridHitCount >= 1 ||
      params.probe.signals.factualCandidateCount >= 1 ||
      params.probe.signals.topProbeDocType === "event");
  const snapshotAuthority =
    queryShape.timeframe === "current" &&
    (params.probe.signals.factualCandidateCount >= 1 ||
      params.probe.signals.stateAssociation >= 0.42 ||
      params.probe.signals.hybridHitCount >= 1);

  let needLevel: RecallNeedLevel;
  const reasons: string[] = [];
  if (!trimmed) {
    needLevel = "none";
    reasons.push("controller:empty-query");
  } else if (params.plan.shouldRecall) {
    needLevel = "full";
    reasons.push("controller:plan-requires-recall");
  } else if (snapshotAuthority) {
    needLevel = "full";
    reasons.push("controller:snapshot-authority");
  } else if (correctedHistoricalAuthority) {
    needLevel = "full";
    reasons.push("controller:historical-authority");
  } else if (exactDetailAuthority) {
    needLevel = "full";
    reasons.push("controller:exact-detail-authority");
  } else if (continuityAuthority) {
    needLevel = "full";
    reasons.push("controller:continuity-authority");
  } else if (lowSupportSurface) {
    needLevel = backgroundContextAvailable ? "background_only" : "none";
    reasons.push(
      backgroundContextAvailable
        ? "controller:low-support-background-only"
        : "controller:low-support-no-background",
    );
  } else if (authoritativeQueryShape || params.probe.shouldEscalate) {
    needLevel = backgroundContextAvailable || routeSupport >= 0.38 ? "full" : "background_only";
    reasons.push(
      needLevel === "full"
        ? "controller:query-analysis-requires-full"
        : "controller:query-analysis-background-only",
    );
  } else if (backgroundContextAvailable) {
    needLevel = "background_only";
    reasons.push("controller:background-context-only");
  } else {
    needLevel = "none";
    reasons.push("controller:no-memory-need");
  }

  const legacyFullRecall = params.legacyTrigger !== "none";
  const divergence =
    legacyFullRecall && needLevel !== "full"
      ? "legacy_over_recall"
      : !legacyFullRecall && needLevel === "full"
        ? "legacy_under_recall"
        : undefined;

  return {
    needLevel,
    routeHint: params.plan.routeHint ?? params.probe.hintedRoute,
    queryShape,
    routeWeights,
    shouldUseBackground: needLevel !== "none",
    shouldUseShallow: false,
    shouldUseFull: needLevel === "full",
    legacyTrigger: params.legacyTrigger,
    legacyFullRecall,
    divergence,
    reasons,
    signals: {
      routeSupport,
      workflowContinuity,
      backgroundContextAvailable,
      lowSupportSurface,
      backgroundGuidanceCount: params.background.behavioralGuidance.length,
      backgroundStrategyCount: params.background.strategyGuidance.length,
      backgroundTaskCount: params.background.tasks.length,
      backgroundStateCount: params.background.states.length,
    },
  };
}

export function evaluateRecallController(params: {
  query: string;
  background: BackgroundRecallBundle;
  plan: MemoryRecallPlan;
  probe: RecallProbeDecision;
  queryAnalysis?: RecallQueryAnalysis;
  legacyTrigger: "plan" | "probe" | "both" | "none";
}): RecallControllerTrace {
  return deriveRecallNeedLevel(params);
}

type ProbeRouteSurface = {
  route: MemoryPrimaryRouteType;
  scheduled: ReturnType<typeof collectAndScheduleMemoryObjects>;
  support: ReturnType<typeof topProbeSupportForRoute>;
};

type ProbeSurface = {
  probeHits: SearchHit[];
  workflowContinuity: number;
  taskAssociation: number;
  stateAssociation: number;
  contextAssociation: number;
  workflowContextSupport: number;
  factualSupport: number;
  temporalSupport: number;
  explanatorySupport: number;
  routeSurfaces: ProbeRouteSurface[];
  strongestRouteCandidate?: {
    route: MemoryPrimaryRouteType;
    support: number;
    topEntry?: ReturnType<typeof topProbeSupportForRoute>["topEntry"];
  };
  topProbeScore: number;
};

function isPrimaryRouteType(route?: MemoryRouteType): route is MemoryPrimaryRouteType {
  return Boolean(route && PRIMARY_ROUTE_TYPES.includes(route as MemoryPrimaryRouteType));
}

function projectionRolesForRoute(routeType: MemoryPrimaryRouteType): WorkingProjectionRole[] {
  switch (routeType) {
    case "workflow":
      return ["active_task"];
    case "factual":
      return ["user_style"];
    case "temporal":
      return [];
    case "explanatory":
      return [];
  }
}

function hasProjectionRole(
  background: BackgroundRecallBundle,
  role: WorkingProjectionRole,
): boolean {
  return background.projectionBlocks.some((block) => block.role === role);
}

function projectionSupportForRoute(
  query: string,
  background: BackgroundRecallBundle,
  routeType: MemoryPrimaryRouteType,
): number {
  const relevantBlocks = background.projectionBlocks.filter((block) =>
    projectionRolesForRoute(routeType).includes(block.role),
  );
  if (relevantBlocks.length === 0) {
    return 0;
  }
  void query;
  return routeType === "workflow" ? 0.18 : routeType === "factual" ? 0.14 : 0.1;
}

function taskMetadataValue(task: { metadataJson?: Record<string, unknown> }, key: string): string {
  const value = task.metadataJson?.[key];
  return typeof value === "string" && value.trim() ? value.trim() : "";
}

function taskProject(task: { metadataJson?: Record<string, unknown> }): string {
  return taskMetadataValue(task, "project");
}

function uniqueNonEmpty(
  values: Array<string | null | undefined>,
  limit = Number.POSITIVE_INFINITY,
): string[] {
  const seen = new Set<string>();
  const ordered: string[] = [];
  for (const value of values) {
    if (!value) {
      continue;
    }
    const trimmed = value.trim();
    if (!trimmed) {
      continue;
    }
    const normalized = normalizeText(trimmed);
    if (!normalized || seen.has(normalized)) {
      continue;
    }
    seen.add(normalized);
    ordered.push(trimmed);
    if (ordered.length >= limit) {
      break;
    }
  }
  return ordered;
}

function shouldUseExactSnippetSupport(queryAnalysis: QueryCompileResult): boolean {
  return (
    queryAnalysis.evidenceFidelity === "high" ||
    queryAnalysis.queryShape.timeframe === "compare" ||
    queryAnalysis.queryShape.granularity === "exact_detail" ||
    (queryAnalysis.queryShape.evidenceNeed !== "workflow_context" &&
      queryAnalysis.candidateSurfaces.includes("chunk"))
  );
}

function factStatusFromMetadata(
  metadata: Record<string, unknown> | undefined,
): NormalizedFact["status"] | undefined {
  return metadata?.status === "active" ||
    metadata?.status === "superseded" ||
    metadata?.status === "deleted" ||
    metadata?.status === "uncertain"
    ? metadata.status
    : undefined;
}

function isHistoricalFactQuery(queryAnalysis: RecallQueryAnalysis): boolean {
  return (
    queryAnalysis.queryShape.timeframe === "historical" ||
    queryAnalysis.queryShape.timeframe === "compare"
  );
}

function factCandidateVisibleForQuery(
  queryAnalysis: RecallQueryAnalysis,
  metadata: Record<string, unknown> | undefined,
): boolean {
  const status = factStatusFromMetadata(metadata);
  if (!status) {
    return true;
  }
  if (status === "deleted") {
    return false;
  }
  if (status === "superseded" && !isHistoricalFactQuery(queryAnalysis)) {
    return false;
  }
  return true;
}

function evidenceRowFromCandidateHit(hit: CandidateHit, store?: MemxStoreBundle): EvidenceRow {
  const metadata = store ? hydrateFactCandidateMetadata(store, hit) : hit.metadata;
  const observedAt =
    typeof metadata?.observedAt === "string" ? metadata.observedAt : undefined;
  const status = factStatusFromMetadata(metadata);
  const text =
    hit.surface === "fact" &&
    typeof metadata?.canonicalSubject === "string" &&
    typeof metadata?.predicate === "string"
      ? formatFactLine({
          subject: metadata.canonicalSubject,
          predicate: metadata.predicate,
          object:
            typeof metadata.canonicalObject === "string" ? metadata.canonicalObject : undefined,
          status: status === "superseded" || hit.supersededHint ? "superseded" : undefined,
        })
      : normalizeSearchText(hit.text);
  const sourceRef =
    typeof metadata?.sourceRef === "string" && metadata.sourceRef.trim()
      ? metadata.sourceRef.trim()
      : hit.lineage.sourceRef;
  return toEvidenceRow({
    id: hit.lineage.canonicalId ?? hit.lineage.sourceId ?? hit.docId,
    text,
    score: hit.score,
    scope: hit.scope,
    confidence: hit.confidence,
    observedAt,
    sourceRef,
    lineage: {
      ...hit.lineage,
      sourceRef,
    },
  });
}

function factIdFromCandidateHit(hit: CandidateHit): string | undefined {
  if (hit.surface !== "fact") {
    return undefined;
  }
  const candidates = [hit.lineage.canonicalId, hit.lineage.sourceId, hit.docId, hit.candidateId];
  for (const candidate of candidates) {
    if (!candidate) {
      continue;
    }
    return candidate.startsWith("fact:") ? candidate.slice(5) : candidate;
  }
  return undefined;
}

function semanticAssertionMetadata(fact: NormalizedFact): Record<string, unknown> | undefined {
  return objectRecord(fact.objectValueJson?.semanticAssertion);
}

function factSupportSourceRefs(fact: NormalizedFact): string[] {
  const assertion = semanticAssertionMetadata(fact);
  return uniqueMaintenanceRefs([
    fact.sourceRef,
    ...sourceRefsFromMaintenanceMetadata(fact.objectValueJson),
    typeof assertion?.sourceRef === "string" ? assertion.sourceRef : undefined,
  ]);
}

function factSupportText(fact: NormalizedFact): string | undefined {
  const assertion = semanticAssertionMetadata(fact);
  const supportText = typeof assertion?.supportText === "string" ? assertion.supportText.trim() : "";
  if (supportText) {
    return supportText;
  }
  return fact.provenanceText?.trim() || undefined;
}

function hydrateFactCandidateMetadata(
  store: MemxStoreBundle,
  hit: CandidateHit,
): Record<string, unknown> | undefined {
  if (hit.surface !== "fact") {
    return hit.metadata;
  }
  const factId = factIdFromCandidateHit(hit);
  const fact = factId ? store.factRepo.get(factId) : null;
  if (!fact) {
    return hit.metadata;
  }
  const sourceRefs = factSupportSourceRefs(fact);
  const supportText = factSupportText(fact);
  return {
    ...(hit.metadata ?? {}),
    canonicalSubject: fact.canonicalSubject,
    predicate: fact.predicate,
    ...(fact.canonicalObject ? { canonicalObject: fact.canonicalObject } : {}),
    status: fact.status,
    activeHint: fact.status === "active" || fact.status === "uncertain",
    supersededHint: fact.status === "superseded",
    currentnessHint: fact.status === "superseded" ? "historical" : "current",
    observedAt: fact.updatedAt,
    confidence: fact.confidence,
    ...(sourceRefs[0] ? { sourceRef: sourceRefs[0] } : {}),
    ...(sourceRefs.length > 0
      ? {
          sourceRefs,
          supportRefs: sourceRefs,
          supportContentRefs: sourceRefs,
        }
      : {}),
    ...(supportText ? { supportText } : {}),
  };
}

function candidateDisplayText(hit: CandidateHit, metadata: Record<string, unknown> | undefined): string {
  const status = factStatusFromMetadata(metadata);
  if (
    hit.surface === "fact" &&
    typeof metadata?.canonicalSubject === "string" &&
    typeof metadata?.predicate === "string"
  ) {
    return formatFactLine({
      subject: metadata.canonicalSubject,
      predicate: metadata.predicate,
      object:
        typeof metadata.canonicalObject === "string" ? metadata.canonicalObject : undefined,
      status: status === "superseded" || hit.supersededHint ? "superseded" : undefined,
    });
  }
  return hit.text;
}

function evidenceRowFromFactId(store: MemxStoreBundle, factId: string): EvidenceRow | null {
  const fact = store.factRepo.get(factId);
  if (!fact) {
    return null;
  }
  const supportRefs = factSupportSourceRefs(fact);
  const sourceRef = supportRefs[0] ?? fact.sourceRef;
  return toEvidenceRow({
    id: fact.factId,
    text: formatFactLine({
      subject: fact.canonicalSubject,
      predicate: fact.predicate,
      object: fact.canonicalObject ?? undefined,
      objectValueJson: fact.objectValueJson,
      status: fact.status,
    }),
    score: fact.confidence,
    scope: fact.scope,
    confidence: fact.confidence,
    observedAt: fact.updatedAt,
    sourceRef,
    provenance: factSupportText(fact),
    lineage: {
      canonicalKind: "fact",
      canonicalId: fact.factId,
      sourceKind: "fact",
      sourceId: fact.factId,
      sourceRef,
      materializedEpoch: fact.materializedEpoch,
    },
  });
}

function primaryCandidateRowsForSurface(
  result: CandidateGenerationResult,
  surface: CandidateHit["surface"],
  options?: {
    store: MemxStoreBundle;
    queryAnalysis: RecallQueryAnalysis;
  },
): EvidenceRow[] {
  return result.candidates
    .filter((candidate) => candidate.surface === surface && candidate.tier === "primary")
    .map((candidate) => ({
      candidate,
      metadata: options?.store ? hydrateFactCandidateMetadata(options.store, candidate) : candidate.metadata,
    }))
    .filter(
      ({ candidate, metadata }) =>
        candidate.surface !== "fact" ||
        !options ||
        factCandidateVisibleForQuery(options.queryAnalysis, metadata),
    )
    .map(({ candidate }) => evidenceRowFromCandidateHit(candidate, options?.store));
}

function candidateSurfaceAnchorScore(row: EvidenceRow, queryAnchors: string[]): number {
  if (queryAnchors.length === 0) {
    return 0;
  }
  return queryAnchorSupport(row.text, queryAnchors);
}

function prioritizeCandidateRowsForMainSurface(
  rows: EvidenceRow[],
  queryAnchors: string[],
  limit: number,
): EvidenceRow[] {
  return dedupeEvidenceRows(
    rows.slice().sort((left, right) => {
      const rightAnchor = candidateSurfaceAnchorScore(right, queryAnchors);
      const leftAnchor = candidateSurfaceAnchorScore(left, queryAnchors);
      if (rightAnchor !== leftAnchor) {
        return rightAnchor - leftAnchor;
      }
      return (right.score ?? 0) - (left.score ?? 0);
    }),
    limit,
  );
}

function selectComplementaryFactRowsForMainSurface(params: {
  store: MemxStoreBundle;
  queryAnalysis: RecallQueryAnalysis;
  queryAnchors: string[];
  selectedFacts: EvidenceRow[];
  fallbackFacts: EvidenceRow[];
  reserveFactTrace?: RecallSelectionTraceEntry[];
  limit: number;
}): EvidenceRow[] {
  const detailSensitive =
    params.queryAnalysis.answerGranularity === "detail" ||
    params.queryAnalysis.evidenceFidelity === "high";
  if (!detailSensitive || params.queryAnalysis.queryShape.timeframe !== "current") {
    return [];
  }

  const selectedIds = new Set(params.selectedFacts.map((entry) => entry.id));
  const selectedMeta = params.selectedFacts
    .map((row) => {
      const fact = params.store.factRepo.get(row.id);
      if (!fact || fact.predicate !== "reported_detail") {
        return null;
      }
      if (
        fact.objectValueJson?.answerBearing !== true ||
        fact.objectValueJson?.currentnessHint !== "current"
      ) {
        return null;
      }
      return {
        row,
        fact,
        anchorSupport: candidateSurfaceAnchorScore(row, params.queryAnchors),
      };
    })
    .filter((entry): entry is NonNullable<typeof entry> => Boolean(entry))
    .sort((left, right) => {
      const rightScore = right.anchorSupport * 0.7 + (right.row.score ?? 0) * 0.3;
      const leftScore = left.anchorSupport * 0.7 + (left.row.score ?? 0) * 0.3;
      return rightScore - leftScore;
    });
  const seed = selectedMeta[0];
  if (!seed) {
    return [];
  }

  const reserveFactRows = (params.reserveFactTrace ?? [])
    .map((entry) => {
      const factId = entry.objectId.startsWith("fact:") ? entry.objectId.slice(5) : entry.objectId;
      return evidenceRowFromFactId(params.store, factId);
    })
    .filter((entry): entry is EvidenceRow => Boolean(entry));

  return dedupeEvidenceRows(
    [...params.fallbackFacts, ...reserveFactRows],
    Math.max(6, params.limit * 6),
  )
    .filter((row) => !selectedIds.has(row.id))
    .map((row) => {
      const fact = params.store.factRepo.get(row.id);
      if (!fact || fact.predicate !== seed.fact.predicate) {
        return null;
      }
      if (fact.canonicalSubject !== seed.fact.canonicalSubject) {
        return null;
      }
      if (
        fact.objectValueJson?.answerBearing !== true ||
        fact.objectValueJson?.currentnessHint !== "current"
      ) {
        return null;
      }
      const similarity = semanticTextSimilarity(seed.row.text, row.text);
      const anchorSupport = candidateSurfaceAnchorScore(row, params.queryAnchors);
      if (similarity < 0.42) {
        return null;
      }
      return {
        row,
        score: similarity * 0.75 + anchorSupport * 0.25,
      };
    })
    .filter((entry): entry is NonNullable<typeof entry> => Boolean(entry))
    .sort((left, right) => right.score - left.score)
    .slice(0, params.limit)
    .map((entry) => entry.row);
}

function preferredProjectedExactSnippets(params: {
  projectedIds: string[];
  projectedTexts: string[];
  queryAnchors: string[];
  limit: number;
}): ExactSnippetCandidate[] {
  return params.projectedIds
    .map((snippetId, index) => ({
      snippetId,
      text: params.projectedTexts[index] ?? "",
      lineage: {
        sourceKind: "vector_doc" as const,
        sourceId: snippetId,
      },
      source: "projected" as const,
      anchorSupport:
        params.queryAnchors.length > 0
          ? queryAnchorSupport(params.projectedTexts[index] ?? "", params.queryAnchors)
          : 1,
    }))
    .filter((entry) => entry.text.trim().length > 0)
    .filter((entry) => (params.queryAnchors.length > 0 ? entry.anchorSupport >= 0.32 : true))
    .slice(0, params.limit);
}

function buildSourceRefFallbackSnippets(params: {
  store: MemxStoreBundle;
  ctx: MemoryOperationContext;
  sourceRefs: string[];
  limit: number;
  queryAnchors?: string[];
}): ExactSnippetCandidate[] {
  const chunks = params.store.chunkRepo.listBySourceRefs({
    agentId: params.ctx.agentId,
    scopes: params.ctx.scopes,
    sourceRefs: params.sourceRefs,
    limit: Math.max(params.limit * 2, 6),
  });
  const seen = new Set<string>();
  return chunks
    .map((chunk) => ({
      chunk,
      anchorScore:
        params.queryAnchors && params.queryAnchors.length > 0
          ? queryAnchorSupport(chunk.content, params.queryAnchors)
          : 1,
    }))
    .filter((entry) => {
      if (seen.has(entry.chunk.chunkId)) {
        return false;
      }
      seen.add(entry.chunk.chunkId);
      return params.queryAnchors && params.queryAnchors.length > 0
        ? entry.anchorScore >= 0.22
        : true;
    })
    .sort((left, right) => {
      if (right.anchorScore !== left.anchorScore) {
        return right.anchorScore - left.anchorScore;
      }
      return right.chunk.createdAt.localeCompare(left.chunk.createdAt);
    })
    .slice(0, params.limit)
    .map(({ chunk }) => ({
      snippetId: `event:chunk:${chunk.chunkId}`,
      text: truncateText(chunk.content.trim() || chunk.summary.trim(), 280),
      sourceRef: chunk.sourceRef,
      lineage: {
        sourceKind: "chunk" as const,
        sourceId: chunk.chunkId,
        sourceRef: chunk.sourceRef,
      },
      source: "fallback" as const,
    }));
}

function rawScoreTextAgainstEvidenceGoals(params: {
  text: string;
  queryAnalysis: RecallQueryAnalysis;
  surface?: "fact" | "event" | "chunk" | "snippet";
  explicitGoalScore?: number;
}): number {
  const goals = params.queryAnalysis.evidenceGoals ?? [];
  const anchorScore =
    params.queryAnalysis.anchors.length > 0
      ? informativePromptAnchorSupport(params.text, params.queryAnalysis.anchors)
      : 0;
  const goalScore = goals.reduce((best, goal) => {
    const positiveScore = Math.max(
      semanticTextSimilarity(goal.goal, params.text),
      ...goal.positiveQueries.map((query) => semanticTextSimilarity(query, params.text)),
    );
    const focusScore =
      goal.focusAnchors.length > 0
        ? informativePromptAnchorSupport(params.text, goal.focusAnchors)
        : 0;
    const surfaceBoost =
      params.surface && goal.preferredSurfaces.includes(params.surface) ? 0.08 : 0;
    const negativePenalty = Math.max(
      0,
      ...(goal.negativeHints ?? []).map((hint) => semanticTextSimilarity(hint, params.text) * 0.12),
    );
    return Math.max(
      best,
      clamp01(positiveScore * 0.68 + focusScore * 0.18 + surfaceBoost - negativePenalty),
    );
  }, 0);
  const explicitGoalScore = params.explicitGoalScore ?? 0;
  const explicitSupport = Math.max(
    evidenceGoalQuerySemanticSupport(params.queryAnalysis, params.text),
    evidenceGoalFocusSupport(params.queryAnalysis, params.text),
    anchorScore,
  );
  const tailorAdvice = params.queryAnalysis.evidencePlan?.operation.type === "tailor_advice";
  const adjustedExplicitGoalScore = tailorAdvice
    ? Math.min(explicitGoalScore, explicitSupport + 0.18)
    : explicitGoalScore > 0 && explicitSupport < 0.12
      ? Math.min(explicitGoalScore, 0.42)
      : explicitGoalScore;
  const adjustedGoalScore =
    explicitSupport < 0.12
      ? Math.min(goalScore, 0.42)
      : tailorAdvice
        ? Math.min(goalScore, explicitSupport + 0.18)
        : goalScore;
  return clamp01(Math.max(adjustedExplicitGoalScore, adjustedGoalScore, anchorScore * 0.72));
}

function scoreTextAgainstEvidenceGoals(params: {
  text: string;
  queryAnalysis: RecallQueryAnalysis;
  surface?: "fact" | "event" | "chunk" | "snippet";
  explicitGoalScore?: number;
}): number {
  const rawScore = rawScoreTextAgainstEvidenceGoals(params);
  return capScoreByEvidenceCoverage(
    rawScore,
    evidenceCoverageForText(params.queryAnalysis, params.text),
  );
}

function topCandidateGoalScore(hit: CandidateHit): number {
  return Math.max(
    0,
    ...(hit.goalMatches ?? []).map((match) => match.score),
    ...(hit.bridgeMatches ?? []).map((match) => match.score),
  );
}

function rankExactSnippetCandidates(
  candidates: ExactSnippetCandidate[],
  queryAnalysis: RecallQueryAnalysis,
): ExactSnippetCandidate[] {
  const byId = new Map<string, ExactSnippetCandidate>();
  for (const candidate of candidates) {
    const goalScore = scoreTextAgainstEvidenceGoals({
      text: candidate.text,
      queryAnalysis,
      surface: "snippet",
    });
    const scored = {
      ...candidate,
      goalScore,
    };
    const existing = byId.get(candidate.snippetId);
    if (!existing || (scored.goalScore ?? 0) > (existing.goalScore ?? 0)) {
      byId.set(candidate.snippetId, scored);
    }
  }
  return [...byId.values()].sort((left, right) => {
    const scoreDelta = (right.goalScore ?? 0) - (left.goalScore ?? 0);
    if (scoreDelta !== 0) {
      return scoreDelta;
    }
    const sourceRank = { projected: 3, support_ref: 2, fallback: 1 };
    return sourceRank[right.source] - sourceRank[left.source];
  });
}

function candidateSurfaceToPromptSurface(
  surface: CandidateHit["surface"],
): PromptEvidenceCandidate["surface"] | null {
  if (surface === "fact" || surface === "event" || surface === "chunk") {
    return surface;
  }
  if (surface === "state" || surface === "graph") {
    return "fact";
  }
  if (surface === "task") {
    return "snippet";
  }
  return null;
}

function preferredSurfaceSupport(
  queryAnalysis: RecallQueryAnalysis,
  surface: PromptEvidenceCandidate["surface"],
): number {
  const mappedSurface = surface === "snippet" ? "chunk" : surface;
  return (queryAnalysis.evidenceGoals ?? []).some((goal) =>
    goal.preferredSurfaces.includes(mappedSurface),
  )
    ? 0.08
    : 0;
}

function evidenceGoalFocusSupport(queryAnalysis: RecallQueryAnalysis, text: string): number {
  const anchors = uniqueNonEmpty(
    (queryAnalysis.evidenceGoals ?? []).flatMap((goal) => goal.focusAnchors),
  );
  if (anchors.length === 0) {
    return queryAnalysis.anchors.length > 0
      ? informativePromptAnchorSupport(text, queryAnalysis.anchors)
      : 0;
  }
  return informativePromptAnchorSupport(text, anchors);
}

function informativePromptAnchors(anchors: string[]): string[] {
  return uniqueNonEmpty(
    anchors.filter((anchor) => normalizeText(anchor).length > 0),
    12,
  );
}

function informativePromptAnchorSupport(text: string, anchors: string[]): number {
  const usefulAnchors = informativePromptAnchors(anchors);
  if (usefulAnchors.length === 0) {
    return 0;
  }
  const normalizedText = normalizeText(text);
  let best = 0;
  for (const anchor of usefulAnchors) {
    const normalizedAnchor = normalizeText(anchor);
    if (!normalizedAnchor) {
      continue;
    }
    const containment =
      normalizedAnchor.length >= 4 && normalizedText.includes(normalizedAnchor) ? 0.94 : 0;
    best = Math.max(best, containment, semanticTextSimilarity(anchor, text));
  }
  return clamp01(best);
}

function evidenceGoalQuerySemanticSupport(
  queryAnalysis: RecallQueryAnalysis,
  text: string,
): number {
  let best = 0;
  for (const goal of queryAnalysis.evidenceGoals ?? []) {
    for (const query of [goal.goal, ...goal.positiveQueries]) {
      if (!normalizeText(query)) {
        continue;
      }
      best = Math.max(best, semanticTextSimilarity(query, text));
    }
  }
  return clamp01(best);
}

function promptEvidenceExcerptAnchors(queryAnalysis: RecallQueryAnalysis): string[] {
  return informativePromptAnchors([
    ...queryAnalysis.anchors,
    ...(queryAnalysis.evidenceGoals ?? []).flatMap((goal) => goal.focusAnchors),
    ...(queryAnalysis.evidencePlan?.slots ?? []).flatMap((slot) => slot.subjectHints),
    ...(queryAnalysis.evidencePlan?.slots ?? []).flatMap((slot) => slot.relationHints ?? []),
    ...(queryAnalysis.evidencePlan?.slots ?? []).flatMap((slot) => slot.capabilityQueries ?? []),
  ]);
}

type PromptSlotCoverage = NonNullable<PromptEvidenceCandidate["slotCoverage"]>[number];

function evidencePlanLayersForSlot(slot: QueryEvidenceSlot): EvidencePlanLayer[] {
  return [...new Set([...slot.preferredLayers, ...slot.fallbackLayers])];
}

function evidenceSlotRequiredRole(
  slot: QueryEvidenceSlot | undefined,
): PromptEvidenceCandidate["slotEvidenceRole"] | undefined {
  if (!slot) {
    return undefined;
  }
  if (slot.requiredRole) {
    return slot.requiredRole;
  }
  if (
    slot.role === "query_context" ||
    slot.role === "user_resource" ||
    slot.role === "prior_advice" ||
    slot.role === "answer_value" ||
    slot.role === "answer_event" ||
    slot.role === "time_constraint"
  ) {
    return slot.role;
  }
  if (slot.role === "answer_evidence") {
    return "answer_value";
  }
  return undefined;
}

function effectiveSlotHints(hints: string[]): string[] {
  return uniqueNonEmpty(
    hints.filter((hint) => normalizeText(hint).length > 0),
    8,
  );
}

function promptMetadataStringArray(
  metadata: Record<string, unknown> | undefined,
  key: string,
): string[] {
  const value = metadata?.[key];
  return Array.isArray(value)
    ? value.filter((entry): entry is string => typeof entry === "string" && entry.trim().length > 0)
    : [];
}

function promptEvidenceScoringTextFromMetadata(
  text: string,
  metadata: Record<string, unknown> | undefined,
): string {
  if (!metadata) {
    return text;
  }
  const parts = [
    text,
    typeof metadata.canonicalSubject === "string" ? metadata.canonicalSubject : undefined,
    typeof metadata.predicate === "string" ? metadata.predicate : undefined,
    typeof metadata.stateKey === "string" ? metadata.stateKey : undefined,
    typeof metadata.resource === "string" ? metadata.resource : undefined,
    typeof metadata.canonicalObject === "string" ? metadata.canonicalObject : undefined,
    typeof metadata.resourceType === "string"
      ? `resource type: ${metadata.resourceType}`
      : undefined,
    promptMetadataStringArray(metadata, "domains").length > 0
      ? `domains: ${promptMetadataStringArray(metadata, "domains").join(", ")}`
      : undefined,
    promptMetadataStringArray(metadata, "affordances").length > 0
      ? `affordances: ${promptMetadataStringArray(metadata, "affordances").join(", ")}`
      : undefined,
    typeof metadata.supportText === "string" ? `evidence: ${metadata.supportText}` : undefined,
    metadata.signalKind === "resourceAssertion" ? "resource assertion" : undefined,
  ];
  return uniqueNonEmpty(
    parts.filter((part): part is string => Boolean(part)),
    16,
  ).join(" | ");
}

function promptEvidenceScoringText(candidate: PromptEvidenceCandidate): string {
  return uniqueNonEmpty(
    [candidate.text, candidate.scoringText, candidate.rawText].filter(
      (part): part is string => typeof part === "string" && part.trim().length > 0,
    ),
    6,
  ).join(" | ");
}

function slotSubjectFillThreshold(queryAnalysis: RecallQueryAnalysis): number {
  const operationType = queryAnalysis.evidencePlan?.operation.type;
  if (operationType === "tailor_advice") {
    return 0.52;
  }
  const multiSlot =
    (queryAnalysis.evidencePlan?.slots.length ?? 0) > 1 ||
    operationType === "derive" ||
    operationType === "aggregate" ||
    operationType === "relate";
  return multiSlot ? 0.52 : 0.42;
}

function semanticSlotFillThreshold(queryAnalysis: RecallQueryAnalysis): number {
  const operationType = queryAnalysis.evidencePlan?.operation.type;
  if (operationType === "derive" || operationType === "compare") {
    return 0.66;
  }
  if (operationType === "aggregate" || queryAnalysis.answerMode === "count_aggregate") {
    return 0.6;
  }
  if (queryAnalysis.answerMode === "attribute_lookup") {
    return 0.58;
  }
  return queryAnalysis.evidenceFidelity === "high" || queryAnalysis.answerGranularity === "detail"
    ? 0.56
    : 0.52;
}

function slotFieldSupport(
  text: string,
  fields: string[],
): { hits: string[]; missing: string[]; score: number } {
  if (fields.length === 0) {
    return { hits: [], missing: [], score: 1 };
  }
  const normalized = normalizeText(text);
  const hits: string[] = [];
  const missing: string[] = [];
  let concreteFieldCount = 0;
  for (const field of fields) {
    const key = normalizeText(field);
    if (isGenericRequiredField(key)) {
      continue;
    }
    concreteFieldCount += 1;
    const textKey = key.replace(/_/gu, " ");
    const matched = textKey.length >= 3 && normalizedTextContainsExactPhrase(normalized, textKey);
    if (matched) {
      hits.push(field);
    } else {
      missing.push(field);
    }
  }
  return {
    hits,
    missing,
    score:
      concreteFieldCount > 0
        ? clamp01(hits.length / concreteFieldCount)
        : fields.length > 0
          ? 0.15
          : 1,
  };
}

function normalizedTextContainsExactPhrase(
  normalizedText: string,
  normalizedPhrase: string,
): boolean {
  const escaped = normalizedPhrase
    .split(/\s+/u)
    .filter(Boolean)
    .map((part) => part.replace(/[.*+?^${}()|[\]\\]/gu, "\\$&"))
    .join("\\s+");
  if (!escaped) {
    return false;
  }
  return new RegExp(`(^|[^\\p{L}\\p{N}])${escaped}($|[^\\p{L}\\p{N}])`, "u").test(normalizedText);
}

function promptEvidenceNonSubjectSlotHitCount(
  queryAnalysis: RecallQueryAnalysis,
  candidate: PromptEvidenceCandidate,
): number {
  const slotById = new Map(
    (queryAnalysis.evidencePlan?.slots ?? []).map((slot) => [slot.id, slot]),
  );
  const queryAnchors = queryAnalysis.anchors ?? [];
  const focusAnchors = new Set(
    [...queryAnchors, ...(queryAnalysis.evidenceGoals ?? []).flatMap((goal) => goal.focusAnchors)]
      .map((anchor) => normalizeText(anchor))
      .filter(Boolean),
  );
  const hits = new Set<string>();
  const slotCoverage =
    candidate.slotCoverage ??
    slotCoverageForText(queryAnalysis, promptEvidenceScoringText(candidate));
  for (const coverage of slotCoverage) {
    if (!coverage.filled) {
      continue;
    }
    const slot = slotById.get(coverage.slotId);
    const subjectHints = new Set(
      [...(slot?.subjectHints ?? []), ...focusAnchors]
        .map((hint) => normalizeText(hint))
        .filter(Boolean),
    );
    for (const hit of coverage.requiredHits) {
      const normalizedHit = normalizeText(hit);
      if (!normalizedHit || isGenericRequiredField(normalizedHit)) {
        continue;
      }
      if (!subjectHints.has(normalizedHit)) {
        hits.add(normalizedHit);
      }
    }
  }
  return hits.size;
}

function promptEvidenceRequiresAnswerSlotSupport(queryAnalysis: RecallQueryAnalysis): boolean {
  const operationType = queryAnalysis.evidencePlan?.operation.type;
  const answerMode = queryAnalysis.answerMode ?? "single_fact";
  return (
    operationType === "return_value" ||
    answerMode === "single_fact" ||
    answerMode === "attribute_lookup"
  );
}

function isGenericRequiredField(key: string): boolean {
  const normalizedKey = normalizeText(key).replace(/[\s-]+/g, "_");
  return [
    "answer_value",
    "attribute_value",
    "value",
    "countable_item",
    "temporal_marker",
    "preference_or_prior_action",
    "source_evidence",
    "query_context",
  ].includes(normalizedKey);
}

function slotCoverageForText(
  queryAnalysis: RecallQueryAnalysis,
  text: string,
): PromptSlotCoverage[] {
  return (queryAnalysis.evidencePlan?.slots ?? []).map((slot) => {
    const isTailorAdvice = queryAnalysis.evidencePlan?.operation.type === "tailor_advice";
    const slotRole = evidenceSlotRequiredRole(slot);
    const isTailorAdviceMemory =
      isTailorAdvice && (slotRole ? slotRole !== "query_context" : slot.id !== "current_need");
    const subjectHints = effectiveSlotHints(slot.subjectHints);
    const relationHints = effectiveSlotHints([
      ...(slot.relationHints ?? []),
      ...(slot.capabilityQueries ?? []),
    ]);
    const subjectScore =
      subjectHints.length > 0
        ? informativePromptAnchorSupport(text, subjectHints)
        : slot.subjectHints.length > 0
          ? 0
          : 0.5;
    const relationScore =
      relationHints.length > 0
        ? informativePromptAnchorSupport(text, relationHints)
        : (slot.relationHints ?? []).length > 0
          ? 0
          : 0.35;
    const field = slotFieldSupport(text, slot.requiredFields);
    const querySimilarity = semanticTextSimilarity(
      queryAnalysis.focusedQuery || queryAnalysis.queryText,
      text,
    );
    const semanticSupport = Math.max(
      querySimilarity,
      rawScoreTextAgainstEvidenceGoals({ text, queryAnalysis }),
    );
    const subjectThreshold = slotSubjectFillThreshold(queryAnalysis);
    const subjectHits = subjectHints.filter(
      (hint) => informativePromptAnchorSupport(text, [hint]) >= subjectThreshold,
    );
    const relationHits = relationHints.filter(
      (hint) => informativePromptAnchorSupport(text, [hint]) >= 0.52,
    );
    const requiredHits = [...subjectHits, ...relationHits, ...field.hits];
    const fieldHitKeys = new Set(field.hits.map((hit) => normalizeText(hit)));
    const structuralHits = requiredHits.filter((hit) => !fieldHitKeys.has(normalizeText(hit)));
    const coverageScore = isTailorAdviceMemory
      ? clamp01(
          subjectScore * 0.22 + relationScore * 0.46 + field.score * 0.16 + semanticSupport * 0.16,
        )
      : clamp01(subjectScore * 0.58 + relationScore * 0.18 + field.score * 0.24);
    const semanticStructuralSupport =
      subjectScore >= Math.max(0.24, subjectThreshold - 0.18) ||
      relationScore >= 0.32 ||
      structuralHits.length > 0 ||
      coverageScore >= 0.34;
    const semanticFilled =
      semanticSupport >= semanticSlotFillThreshold(queryAnalysis) && semanticStructuralSupport;
    const adviceFilled =
      isTailorAdvice &&
      field.missing.length === 0 &&
      structuralHits.length > 0 &&
      (subjectScore >= 0.34 ||
        relationScore >= 0.42 ||
        (querySimilarity >= 0.38 && (subjectScore >= 0.24 || relationScore >= 0.28)));
    const tailorMemoryFilled =
      isTailorAdviceMemory &&
      field.missing.length === 0 &&
      (relationScore >= 0.36 ||
        semanticSupport >= 0.4 ||
        coverageScore >= 0.34 ||
        (relationHits.length > 0 && Math.max(relationScore, semanticSupport) >= 0.24));
    const filled =
      tailorMemoryFilled ||
      adviceFilled ||
      semanticFilled ||
      (coverageScore >= 0.56 &&
        (subjectHints.length === 0 || subjectScore >= subjectThreshold) &&
        field.missing.length === 0);
    const missingRequired = filled
      ? []
      : [
          ...(isTailorAdviceMemory
            ? []
            : subjectHints.filter(
                (hint) => informativePromptAnchorSupport(text, [hint]) < subjectThreshold,
              )),
          ...field.missing,
        ];
    return {
      slotId: slot.id,
      requiredHits: uniqueNonEmpty(requiredHits),
      missingRequired: uniqueNonEmpty(missingRequired),
      coverageScore,
      filled,
    };
  });
}

function filledSlotIdsForText(queryAnalysis: RecallQueryAnalysis, text: string): string[] {
  return slotCoverageForText(queryAnalysis, text)
    .filter((coverage) => coverage.filled)
    .map((coverage) => coverage.slotId);
}

function isTailorAdviceMemorySlot(queryAnalysis: RecallQueryAnalysis, slotId: string): boolean {
  const slot = (queryAnalysis.evidencePlan?.slots ?? []).find((entry) => entry.id === slotId);
  const requiredRole = evidenceSlotRequiredRole(slot);
  return (
    queryAnalysis.evidencePlan?.operation.type === "tailor_advice" &&
    (requiredRole ? requiredRole !== "query_context" : slotId !== "current_need")
  );
}

function isTailorAdviceResourceSlot(queryAnalysis: RecallQueryAnalysis, slotId: string): boolean {
  const slot = (queryAnalysis.evidencePlan?.slots ?? []).find((entry) => entry.id === slotId);
  const requiredRole = evidenceSlotRequiredRole(slot);
  return (
    queryAnalysis.evidencePlan?.operation.type === "tailor_advice" &&
    (requiredRole === "user_resource" || slotId === "relevant_user_resources")
  );
}

function tailorAdviceQueryContextText(queryAnalysis: RecallQueryAnalysis): string {
  const currentNeedSlot = (queryAnalysis.evidencePlan?.slots ?? []).find(
    (slot) => slot.id === "current_need",
  );
  return normalizeText(
    [
      queryAnalysis.queryText,
      queryAnalysis.focusedQuery,
      ...queryAnalysis.anchors,
      ...(currentNeedSlot?.subjectHints ?? []),
      ...(currentNeedSlot?.relationHints ?? []),
    ].join(" "),
  );
}

function isQueryContextHit(queryAnalysis: RecallQueryAnalysis, hit: string): boolean {
  const normalizedHit = normalizeText(hit);
  if (!normalizedHit || isGenericRequiredField(normalizedHit)) {
    return true;
  }
  const queryContext = tailorAdviceQueryContextText(queryAnalysis);
  return (
    normalizedTextContainsExactPhrase(queryContext, normalizedHit) ||
    semanticTextSimilarity(normalizedHit, queryContext) >= 0.82
  );
}

function slotSemanticSupport(
  queryAnalysis: RecallQueryAnalysis,
  slotId: string,
  text: string,
): number {
  const slot = (queryAnalysis.evidencePlan?.slots ?? []).find((entry) => entry.id === slotId);
  if (!slot) {
    return 0;
  }
  const slotQueries = uniqueNonEmpty(
    [
      slot.description,
      ...(slot.relationHints ?? []),
      ...(slot.capabilityQueries ?? []),
      ...slot.requiredFields.filter((field) => !isGenericRequiredField(normalizeText(field))),
    ],
    16,
  );
  return clamp01(Math.max(0, ...slotQueries.map((query) => semanticTextSimilarity(query, text))));
}

function pruneTailorAdviceMemorySlotCoverage(params: {
  queryAnalysis: RecallQueryAnalysis;
  text: string;
  coverage: PromptSlotCoverage;
}): PromptSlotCoverage {
  if (!isTailorAdviceMemorySlot(params.queryAnalysis, params.coverage.slotId)) {
    return params.coverage;
  }
  const roleSpecificHits = params.coverage.requiredHits.filter(
    (hit) => !isQueryContextHit(params.queryAnalysis, hit),
  );
  if (roleSpecificHits.length > 0) {
    return params.coverage;
  }
  const semanticSupport = evidenceGoalQuerySemanticSupport(params.queryAnalysis, params.text);
  const slotSupport = slotSemanticSupport(
    params.queryAnalysis,
    params.coverage.slotId,
    params.text,
  );
  const requiredSemanticSupport = isTailorAdviceResourceSlot(
    params.queryAnalysis,
    params.coverage.slotId,
  )
    ? 0.34
    : 0.42;
  const roleSupport = Math.max(semanticSupport, slotSupport, params.coverage.coverageScore);
  if (roleSupport >= requiredSemanticSupport) {
    return {
      ...params.coverage,
      missingRequired: [],
      coverageScore: Math.max(params.coverage.coverageScore, roleSupport),
      filled: true,
    };
  }
  return {
    ...params.coverage,
    requiredHits: roleSpecificHits,
    missingRequired: uniqueNonEmpty([
      ...params.coverage.missingRequired,
      ...(params.coverage.requiredHits.length > 0
        ? params.coverage.requiredHits
        : ["role-specific memory evidence"]),
    ]),
    coverageScore: Math.min(params.coverage.coverageScore, 0.28),
    filled: false,
  };
}

function slotCoverageWithCandidateMatches(
  queryAnalysis: RecallQueryAnalysis,
  text: string,
  hit: CandidateHit,
): PromptSlotCoverage[] {
  const baseCoverage = slotCoverageForText(queryAnalysis, text).map((coverage) =>
    pruneTailorAdviceMemorySlotCoverage({
      queryAnalysis,
      text,
      coverage,
    }),
  );
  const slotMatches = hit.slotMatches ?? [];
  if (slotMatches.length === 0) {
    return baseCoverage;
  }
  return baseCoverage.map((coverage) => {
    const bestMatch = slotMatches
      .filter((match) => match.slotId === coverage.slotId)
      .sort((left, right) => right.score - left.score)[0];
    const resourceSlot = isTailorAdviceResourceSlot(queryAnalysis, coverage.slotId);
    const threshold =
      queryAnalysis.evidencePlan?.operation.type === "tailor_advice"
        ? resourceSlot
          ? 0.16
          : 0.34
        : 0.52;
    if (!bestMatch || bestMatch.score < threshold) {
      return coverage;
    }
    const memorySlot = isTailorAdviceMemorySlot(queryAnalysis, coverage.slotId);
    const querySemanticSupport = evidenceGoalQuerySemanticSupport(queryAnalysis, text);
    const specificSupport = memorySlot
      ? querySemanticSupport
      : Math.max(querySemanticSupport, evidenceGoalFocusSupport(queryAnalysis, text));
    const matchedQuerySupport = semanticTextSimilarity(bestMatch.matchedQuery, text);
    const hasConcreteBaseHit = coverage.requiredHits.some(
      (entry) => !isGenericRequiredField(normalizeText(entry)),
    );
    const highConfidenceSemanticMatch =
      bestMatch.score >= 0.86 && hit.retrievalBackend === "embedding";
    const semanticSlotMatch =
      specificSupport >= 0.18 ||
      matchedQuerySupport >= 0.42 ||
      (bestMatch.score >= 0.82 && matchedQuerySupport >= 0.32);
    const structuredResourceMatch =
      resourceSlot && candidateHitHasStructuredResourceSignal(hit) && bestMatch.score >= 0.16;
    const canPromoteCandidateSlotMatch = memorySlot
      ? (hasConcreteBaseHit && specificSupport >= 0.18) ||
        semanticSlotMatch ||
        structuredResourceMatch
      : (hasConcreteBaseHit && specificSupport >= 0.08) ||
        semanticSlotMatch ||
        (highConfidenceSemanticMatch && specificSupport >= 0.12);
    if (!canPromoteCandidateSlotMatch) {
      return {
        ...coverage,
        requiredHits: uniqueNonEmpty([...coverage.requiredHits, bestMatch.matchedQuery]),
        coverageScore: Math.max(coverage.coverageScore, clamp01(bestMatch.score * 0.62)),
      };
    }
    return {
      ...coverage,
      requiredHits: uniqueNonEmpty([...coverage.requiredHits, bestMatch.matchedQuery]),
      missingRequired: [],
      coverageScore: Math.max(coverage.coverageScore, clamp01(bestMatch.score)),
      filled: true,
    };
  });
}

function filledSlotIdsFromCoverage(slotCoverage: PromptSlotCoverage[]): string[] {
  return slotCoverage.filter((coverage) => coverage.filled).map((coverage) => coverage.slotId);
}

function stripPromptEvidenceRolePrefix(text: string): string {
  return text
    .trim()
    .replace(/^(?:\[(?:user|assistant|tool|memory)\]|(?:user|assistant|tool|memory))\s*:\s*/iu, "")
    .replace(/^assistant\s+explanatory\s+response\s*[:.-]?\s*/iu, "")
    .trim();
}

function collapseRepeatedLeadingSentence(text: string): string {
  const trimmed = text.trim();
  const firstSentence = trimmed.split(/(?<=[.!?])\s+/u)[0] ?? "";
  if (firstSentence.length < 24 || firstSentence.length >= trimmed.length) {
    return trimmed;
  }
  const remainder = trimmed.slice(firstSentence.length).trimStart();
  if (remainder.toLowerCase().startsWith(firstSentence.toLowerCase())) {
    return `${firstSentence} ${remainder.slice(firstSentence.length).trimStart()}`.trim();
  }
  return trimmed;
}

function truncatedPrefixStem(text: string): string {
  const trimmed = text.trim();
  if (!/(?:…|\.\.\.)\s*$/u.test(trimmed)) {
    return "";
  }
  const withoutEllipsis = trimmed.replace(/(?:…|\.\.\.)+\s*$/u, "").trim();
  const withoutPartialToken = withoutEllipsis.replace(/\s+\S+$/u, "").trim();
  return withoutPartialToken.length >= 16 ? withoutPartialToken : withoutEllipsis;
}

function cleanPromptEvidenceText(text: string): string {
  const trimmed = text.trim();
  if (!trimmed) {
    return trimmed;
  }
  const lines = trimmed.split(/\r?\n/u);
  const firstLine = stripPromptEvidenceRolePrefix(lines[0] ?? "");
  const rest = lines.slice(1).join("\n").trim();
  if (rest) {
    const cleanedRest = stripPromptEvidenceRolePrefix(rest);
    const firstNormalized = normalizeText(firstLine);
    const restNormalized = normalizeText(cleanedRest);
    const truncatedStemNormalized = normalizeText(truncatedPrefixStem(firstLine));
    const firstRestSentence = cleanedRest.split(/(?<=[.!?])\s+/u)[0] ?? cleanedRest;
    const restStartsWithSummary =
      firstNormalized.length >= 16 && restNormalized.startsWith(firstNormalized);
    const restStartsWithTruncatedSummary =
      truncatedStemNormalized.length >= 16 && restNormalized.startsWith(truncatedStemNormalized);
    const restRepeatsSummary =
      firstNormalized.length >= 24 &&
      semanticTextSimilarity(firstLine, firstRestSentence.slice(0, 260)) >= 0.82;
    const uninformativeSummary =
      /^assistant\s+explanatory\s+response$/iu.test(firstLine) ||
      firstNormalized === "assistant explanatory response";
    if (
      restStartsWithSummary ||
      restStartsWithTruncatedSummary ||
      restRepeatsSummary ||
      uninformativeSummary
    ) {
      return collapseRepeatedLeadingSentence(cleanedRest);
    }
    if (firstLine !== (lines[0] ?? "").trim()) {
      return collapseRepeatedLeadingSentence(`${firstLine}\n${cleanedRest}`.trim());
    }
    return collapseRepeatedLeadingSentence(`${lines[0] ?? ""}\n${cleanedRest}`.trim());
  }
  return collapseRepeatedLeadingSentence(stripPromptEvidenceRolePrefix(trimmed));
}

function rawAssistantExplanatorySummary(text: string): boolean {
  const firstLine = (text.trim().split(/\r?\n/u)[0] ?? "").trim();
  const withoutRole = firstLine.replace(
    /^(?:\[(?:user|assistant|tool|memory)\]|(?:user|assistant|tool|memory))\s*:\s*/iu,
    "",
  );
  return normalizeText(withoutRole) === "assistant explanatory response";
}

function promptEvidenceMetadataForRawText(
  text: string,
  metadata?: Record<string, unknown>,
): Record<string, unknown> | undefined {
  if (!rawAssistantExplanatorySummary(text)) {
    return metadata;
  }
  return {
    ...(metadata ?? {}),
    semanticRole: "assistant_acknowledgement",
  };
}

function promptEvidenceRole(
  candidate: PromptEvidenceCandidate,
): "user" | "assistant" | "tool" | "unknown" {
  const text = (candidate.rawText ?? candidate.text).trim().toLowerCase();
  if (
    /^(?:\[assistant\]|assistant)\s*:/iu.test(text) ||
    text.startsWith("assistant explanatory response")
  ) {
    return "assistant";
  }
  if (/^(?:\[user\]|user)\s*:/iu.test(text)) {
    return "user";
  }
  if (/^(?:\[tool\]|tool)\s*:/iu.test(text)) {
    return "tool";
  }
  return "unknown";
}

function queryAllowsAssistantAuthoredEvidence(queryAnalysis: RecallQueryAnalysis): boolean {
  const slots = queryAnalysis.evidencePlan?.slots ?? [];
  if (slots.some((slot) => slot.role === "prior_advice" || slot.requiredRole === "prior_advice")) {
    return true;
  }
  return (queryAnalysis.semanticBridges ?? []).some((bridge) => bridge.role === "prior_advice");
}

function promptEvidenceHasLiteralFocusAnchor(
  queryAnalysis: RecallQueryAnalysis,
  text: string,
): boolean {
  const anchors = informativePromptAnchors([
    ...queryAnalysis.anchors,
    ...(queryAnalysis.evidenceGoals ?? []).flatMap((goal) => goal.focusAnchors),
  ]);
  const normalizedText = normalizeText(text);
  return anchors.some((anchor) => {
    const normalizedAnchor = normalizeText(anchor);
    return normalizedAnchor.length >= 4 && normalizedText.includes(normalizedAnchor);
  });
}

function promptEvidenceAnswerSpecificity(
  queryAnalysis: RecallQueryAnalysis,
  candidate: PromptEvidenceCandidate,
): number {
  const scoringText = promptEvidenceScoringText(candidate);
  const coverage = candidate.coverage ?? evidenceCoverageForText(queryAnalysis, scoringText);
  const answerSlotSupport = promptEvidenceAnswerSlotStrength(queryAnalysis, candidate);
  const querySemanticSupport = evidenceGoalQuerySemanticSupport(queryAnalysis, scoringText);
  const focusSupport = evidenceGoalFocusSupport(queryAnalysis, scoringText);
  const literalFocus = promptEvidenceHasLiteralFocusAnchor(queryAnalysis, scoringText) ? 0.6 : 0;
  const coverageSupport = coverage.requiredHits.length > 0 ? 0.7 : 0;
  const highSemanticSupport =
    (candidate.semanticScore ?? candidate.goalScore) >= 0.82 || candidate.goalScore >= 0.82
      ? 0.42
      : 0;
  return clamp01(
    Math.max(
      querySemanticSupport,
      focusSupport,
      literalFocus,
      answerSlotSupport,
      coverageSupport,
      highSemanticSupport,
    ),
  );
}

function promptEvidenceAnswerSlotStrength(
  queryAnalysis: RecallQueryAnalysis,
  candidate: PromptEvidenceCandidate,
): number {
  if ((queryAnalysis.evidencePlan?.slots ?? []).length === 0) {
    return 0;
  }
  const nonSubjectHitCount = promptEvidenceNonSubjectSlotHitCount(queryAnalysis, candidate);
  if (nonSubjectHitCount === 0) {
    return 0;
  }
  const slotCoverage = candidate.slotCoverage ?? slotCoverageForText(queryAnalysis, candidate.text);
  const bestSlotCoverage = Math.max(0, ...slotCoverage.map((entry) => entry.coverageScore));
  return clamp01(
    Math.max(0.52, Math.min(0.86, bestSlotCoverage)) + Math.min(0.18, nonSubjectHitCount * 0.055),
  );
}

function promptEvidenceSameSourceQuality(
  queryAnalysis: RecallQueryAnalysis,
  candidate: PromptEvidenceCandidate,
): number {
  const scoringText = promptEvidenceScoringText(candidate);
  const coverage = candidate.coverage ?? evidenceCoverageForText(queryAnalysis, scoringText);
  const coverageSupport = Math.min(0.22, coverage.requiredHits.length * 0.11);
  const detailSupport = Math.min(0.12, normalizeText(candidate.text).length / 900);
  return clamp01(
    promptEvidenceAnswerSpecificity(queryAnalysis, candidate) * 0.42 +
      promptEvidenceAnswerSlotStrength(queryAnalysis, candidate) * 0.24 +
      coverageSupport +
      promptEvidenceAuthorityRank(candidate) * 0.035 +
      detailSupport -
      evidenceTextCompletenessPenalty(candidate.text),
  );
}

function promptEvidenceAuthorityRank(candidate: PromptEvidenceCandidate): number {
  const role = promptEvidenceRole(candidate);
  const sourceRef = candidate.sourceRef ?? "";
  if (role === "assistant") {
    return 0.6;
  }
  if (
    role === "user" &&
    (candidate.surface === "chunk" || candidate.surface === "snippet") &&
    sourceRef
  ) {
    return 5;
  }
  if (candidate.surface === "chunk" || candidate.surface === "snippet") {
    return sourceRef ? 4.4 : 3.8;
  }
  if (role === "user") {
    return 3.8;
  }
  if (candidate.surface === "fact") {
    return 3.2;
  }
  if (candidate.surface === "event") {
    return 2.8;
  }
  return 2;
}

function evidenceTextCompletenessPenalty(text: string): number {
  const trimmed = text.trim();
  if (!trimmed) {
    return 0.18;
  }
  let penalty = 0;
  if (isTruncatedEvidenceText(trimmed)) {
    penalty += 0.24;
  }
  return Math.min(0.32, penalty);
}

function isTruncatedEvidenceText(text: string): boolean {
  const trimmed = text.trim();
  return trimmed.includes("…") || trimmed.endsWith("...");
}

function shouldReplacePromptEvidence(
  existing: PromptEvidenceCandidate,
  candidate: PromptEvidenceCandidate,
  queryAnalysis?: RecallQueryAnalysis,
): boolean {
  if (queryAnalysis && existing.sourceRef && existing.sourceRef === candidate.sourceRef) {
    const candidateQuality = promptEvidenceSameSourceQuality(queryAnalysis, candidate);
    const existingQuality = promptEvidenceSameSourceQuality(queryAnalysis, existing);
    if (candidateQuality >= existingQuality + 0.05) {
      return true;
    }
    if (existingQuality >= candidateQuality + 0.08) {
      return false;
    }
  }
  const existingTruncated = isTruncatedEvidenceText(existing.text);
  const candidateTruncated = isTruncatedEvidenceText(candidate.text);
  if (existingTruncated !== candidateTruncated) {
    return candidateTruncated ? false : candidate.priority >= existing.priority - 0.18;
  }
  if (candidate.goalScore !== existing.goalScore) {
    const authorityDelta =
      promptEvidenceAuthorityRank(candidate) - promptEvidenceAuthorityRank(existing);
    if (authorityDelta >= 1 && candidate.goalScore >= existing.goalScore - 0.08) {
      return true;
    }
    return candidate.goalScore > existing.goalScore;
  }
  const authorityDelta =
    promptEvidenceAuthorityRank(candidate) - promptEvidenceAuthorityRank(existing);
  if (authorityDelta >= 1 && candidate.priority >= existing.priority - 0.14) {
    return true;
  }
  return candidate.priority > existing.priority;
}

function mergePromptEvidenceSourceRefs(
  winner: PromptEvidenceCandidate,
  loser: PromptEvidenceCandidate,
): PromptEvidenceCandidate {
  const mergedSourceRefs = uniqueNonEmpty(
    [
      winner.sourceRef,
      ...(winner.mergedSourceRefs ?? []),
      loser.sourceRef,
      ...(loser.mergedSourceRefs ?? []),
    ].filter((entry): entry is string => typeof entry === "string"),
  );
  return {
    ...winner,
    mergedSourceRefs: mergedSourceRefs.length > 0 ? mergedSourceRefs : undefined,
  };
}

function promptEvidencePriority(params: {
  text: string;
  score?: number;
  surface: PromptEvidenceCandidate["surface"];
  source: PromptEvidenceCandidate["source"];
  tier?: CandidateHit["tier"];
  queryAnalysis: RecallQueryAnalysis;
  explicitGoalScore?: number;
}): { priority: number; goalScore: number; semanticScore: number } {
  const semanticScore = rawScoreTextAgainstEvidenceGoals({
    text: params.text,
    queryAnalysis: params.queryAnalysis,
    surface: params.surface,
    explicitGoalScore: params.explicitGoalScore,
  });
  const goalScore =
    params.queryAnalysis.evidencePlan?.operation.type === "tailor_advice"
      ? semanticScore
      : capScoreByEvidenceCoverage(
          semanticScore,
          evidenceCoverageForText(params.queryAnalysis, params.text),
        );
  const sourceBoost: Record<PromptEvidenceCandidate["source"], number> = {
    candidate: 0.1,
    selected: 0.08,
    support_ref: 0.07,
    projected: 0.06,
    fallback: 0.03,
  };
  const surfaceBoost =
    params.surface === "chunk" || params.surface === "snippet"
      ? params.queryAnalysis.evidenceFidelity === "high" ||
        params.queryAnalysis.answerGranularity === "detail"
        ? 0.06
        : 0.03
      : params.surface === "event"
        ? 0.04
        : 0.02;
  const tierBoost = params.tier === "primary" ? 0.05 : params.tier === "alternate" ? -0.06 : 0;
  const focusSupport = evidenceGoalFocusSupport(params.queryAnalysis, params.text);
  const querySemanticSupport = evidenceGoalQuerySemanticSupport(params.queryAnalysis, params.text);
  const focusPenalty =
    (params.queryAnalysis.evidenceFidelity === "high" ||
      params.queryAnalysis.answerGranularity === "detail") &&
    (params.queryAnalysis.evidenceGoals ?? []).some((goal) => goal.focusAnchors.length > 0) &&
    semanticScore < 0.48 &&
    focusSupport < 0.22
      ? 0.22
      : semanticScore < 0.4 && focusSupport < 0.38
        ? 0.08
        : 0;
  const weakUncoveredGoalPenalty =
    params.queryAnalysis.evidencePlan?.operation.type === "tailor_advice" &&
    params.source === "candidate" &&
    params.score &&
    params.score >= 0.8 &&
    goalScore >= 0.6 &&
    evidenceCoverageForText(params.queryAnalysis, params.text).requiredHits.length === 0 &&
    querySemanticSupport < 0.14 &&
    focusSupport < 0.22
      ? 0.08
      : 0;
  const completenessPenalty = evidenceTextCompletenessPenalty(params.text);
  const slotCoverage = slotCoverageForText(params.queryAnalysis, params.text);
  const topSlotCoverage = Math.max(0, ...slotCoverage.map((coverage) => coverage.coverageScore));
  const filledSlotBoost = slotCoverage.some((coverage) => coverage.filled) ? 0.12 : 0;
  const priority = clamp01(
    goalScore * 0.52 +
      Math.max(0, params.score ?? 0) * 0.08 +
      sourceBoost[params.source] +
      surfaceBoost +
      preferredSurfaceSupport(params.queryAnalysis, params.surface) +
      tierBoost +
      focusSupport * 0.08 +
      querySemanticSupport * 0.22 +
      Math.max(0, topSlotCoverage - 0.45) * 0.08 +
      filledSlotBoost * 0.55 -
      focusPenalty -
      weakUncoveredGoalPenalty -
      completenessPenalty,
  );
  return {
    priority,
    goalScore,
    semanticScore,
  };
}

function promptEvidenceSlotRoleRank(
  role: PromptEvidenceCandidate["slotEvidenceRole"] | undefined,
): number {
  if (role === "user_resource") {
    return 4;
  }
  if (role === "answer_value" || role === "answer_event") {
    return 3;
  }
  if (role === "prior_advice") {
    return 2;
  }
  if (role === "query_context" || role === "time_constraint") {
    return 1;
  }
  return 0;
}

function promptEvidenceSlotRoleForCandidate(
  queryAnalysis: RecallQueryAnalysis,
  candidate: PromptEvidenceCandidate,
): PromptEvidenceCandidate["slotEvidenceRole"] | undefined {
  const slotById = new Map(
    (queryAnalysis.evidencePlan?.slots ?? []).map((slot) => [slot.id, slot]),
  );
  const scoringText = promptEvidenceScoringText(candidate);
  const coverage = candidate.slotCoverage ?? slotCoverageForText(queryAnalysis, scoringText);
  const best = coverage
    .map((entry) => {
      const role = evidenceSlotRequiredRole(slotById.get(entry.slotId));
      return {
        role,
        filled: entry.filled,
        coverageScore: entry.coverageScore,
        rank: promptEvidenceSlotRoleRank(role),
      };
    })
    .filter((entry) => entry.role)
    .sort((left, right) => {
      if (left.filled !== right.filled) {
        return left.filled ? -1 : 1;
      }
      if (right.rank !== left.rank) {
        return right.rank - left.rank;
      }
      return right.coverageScore - left.coverageScore;
    })[0];
  if (!best || (!best.filled && best.coverageScore < 0.36)) {
    return undefined;
  }
  return best.role;
}

function metadataHasStructuredResourceSignal(
  metadata: Record<string, unknown> | undefined,
): boolean {
  if (!metadata) {
    return false;
  }
  return (
    metadata.signalKind === "resourceAssertion" ||
    typeof metadata.resourceType === "string" ||
    (Array.isArray(metadata.affordances) && metadata.affordances.length > 0) ||
    (Array.isArray(metadata.domains) && metadata.domains.length > 0)
  );
}

function textHasStructuredResourceMarker(text: string): boolean {
  const normalized = normalizeText(text);
  return (
    normalized.includes("user.resource") ||
    normalized.includes("has_resource") ||
    normalized.includes("resource_observation") ||
    normalized.includes("affordances") ||
    normalized.includes("resource type") ||
    normalized.includes("signalkind resourceassertion")
  );
}

function promptEvidenceHasStructuredResourceSignal(candidate: PromptEvidenceCandidate): boolean {
  return (
    metadataHasStructuredResourceSignal(candidate.metadata) ||
    textHasStructuredResourceMarker(promptEvidenceScoringText(candidate))
  );
}

function candidateHitHasStructuredResourceSignal(hit: CandidateHit): boolean {
  return (
    metadataHasStructuredResourceSignal(hit.metadata) ||
    textHasStructuredResourceMarker(promptEvidenceScoringTextFromMetadata(hit.text, hit.metadata))
  );
}

function promptEvidenceCapabilitySupport(
  queryAnalysis: RecallQueryAnalysis,
  candidate: PromptEvidenceCandidate,
): number {
  const slotById = new Map(
    (queryAnalysis.evidencePlan?.slots ?? []).map((slot) => [slot.id, slot]),
  );
  const role =
    candidate.slotEvidenceRole ?? promptEvidenceSlotRoleForCandidate(queryAnalysis, candidate);
  const matchedSlots = (
    candidate.slotCoverage ??
    slotCoverageForText(queryAnalysis, promptEvidenceScoringText(candidate))
  )
    .filter((coverage) => coverage.filled || coverage.coverageScore >= 0.36)
    .map((coverage) => slotById.get(coverage.slotId))
    .filter((slot): slot is QueryEvidenceSlot => Boolean(slot))
    .filter((slot) => !role || evidenceSlotRequiredRole(slot) === role);
  const capabilityHints = uniqueNonEmpty(
    matchedSlots.flatMap((slot) =>
      slot.capabilityQueries && slot.capabilityQueries.length > 0
        ? slot.capabilityQueries
        : (slot.relationHints ?? []),
    ),
    12,
  );
  if (capabilityHints.length === 0) {
    return 0;
  }
  const scoringText = promptEvidenceScoringText(candidate);
  const anchorSupport = informativePromptAnchorSupport(scoringText, capabilityHints);
  const semanticSupport = Math.max(
    0,
    ...capabilityHints.map((hint) => semanticTextSimilarity(hint, scoringText)),
  );
  return Math.max(anchorSupport, semanticSupport);
}

function promptEvidenceInjectionLimit(
  queryAnalysis: RecallQueryAnalysis,
  ranked: PromptEvidenceCandidate[],
): number {
  if (ranked.length === 0) {
    return 0;
  }
  const planSlots = queryAnalysis.evidencePlan?.slots ?? [];
  if (
    queryAnalysis.evidencePlan?.operation.type &&
    queryAnalysis.evidencePlan.operation.type !== "tailor_advice" &&
    planSlots.length > 0
  ) {
    const minEvidence = planSlots.reduce((sum, slot) => sum + slot.minEvidence, 0);
    return Math.max(1, Math.min(3, minEvidence, ranked.length));
  }
  if (queryAnalysis.evidencePlan?.operation.type === "tailor_advice") {
    return Math.min(3, ranked.length);
  }
  if (queryAnalysis.answerMode === "count_aggregate") {
    return Math.max(
      1,
      Math.min(3, queryAnalysis.evidenceCoverage?.minProtectedItems ?? 2, ranked.length),
    );
  }
  if (promptEvidenceNeedsCompanions(queryAnalysis)) {
    return Math.min(3, ranked.length);
  }
  return 1;
}

function scorePromptEvidenceForInjection(
  queryAnalysis: RecallQueryAnalysis,
  candidate: PromptEvidenceCandidate,
): PromptEvidenceCandidate {
  const slotEvidenceRole =
    candidate.slotEvidenceRole ?? promptEvidenceSlotRoleForCandidate(queryAnalysis, candidate);
  const semanticScore = candidate.semanticScore ?? candidate.goalScore;
  const scoringText = promptEvidenceScoringText(candidate);
  const slotCoverage = candidate.slotCoverage ?? slotCoverageForText(queryAnalysis, scoringText);
  const bestSlotCoverage = Math.max(0, ...slotCoverage.map((entry) => entry.coverageScore));
  const filledSlotCount = slotCoverage.filter((entry) => entry.filled).length;
  const focusSupport = evidenceGoalFocusSupport(queryAnalysis, scoringText);
  const goalQuerySupport = evidenceGoalQuerySemanticSupport(queryAnalysis, scoringText);
  const bridgeSupport = Math.max(0, ...(candidate.bridgeMatches ?? []).map((match) => match.score));
  const bridgePositiveSupport = Math.max(
    0,
    ...(candidate.bridgeMatches ?? []).map((match) => match.positiveSignalScore),
  );
  const bridgeNegativeSupport = Math.max(
    0,
    ...(candidate.bridgeMatches ?? []).map((match) => match.negativeSignalScore),
  );
  const capabilitySupport = promptEvidenceCapabilitySupport(queryAnalysis, candidate);
  const authority = promptEvidenceAuthorityRank(candidate) / 5;
  const operationType = queryAnalysis.evidencePlan?.operation.type;
  const roleWeight =
    operationType === "tailor_advice"
      ? slotEvidenceRole === "user_resource"
        ? 0.22
        : slotEvidenceRole === "prior_advice"
          ? 0.14
          : slotEvidenceRole === "answer_value" || slotEvidenceRole === "answer_event"
            ? 0.16
            : slotEvidenceRole === "query_context" || slotEvidenceRole === "time_constraint"
              ? -0.08
              : 0
      : slotEvidenceRole === "answer_value" || slotEvidenceRole === "answer_event"
        ? 0.12
        : slotEvidenceRole === "user_resource"
          ? 0.08
          : slotEvidenceRole === "time_constraint"
            ? -0.04
            : 0;
  const sourceTrace =
    candidate.sourceRef || (candidate.mergedSourceRefs?.length ?? 0) > 0 ? 0.04 : 0;
  const structuredResource =
    slotEvidenceRole === "user_resource" &&
    promptEvidenceHasStructuredResourceSignal(candidate) &&
    (capabilitySupport >= 0.2 || goalQuerySupport >= 0.24 || semanticScore >= 0.32)
      ? 0.1
      : 0;
  const resourceCapabilityPenalty =
    slotEvidenceRole === "user_resource" &&
    promptEvidenceHasStructuredResourceSignal(candidate) &&
    capabilitySupport < 0.14 &&
    goalQuerySupport < 0.2 &&
    semanticScore < 0.3
      ? 0.24
      : 0;
  const assistantPenalty =
    promptEvidenceRole(candidate) === "assistant" &&
    !queryAllowsAssistantAuthoredEvidence(queryAnalysis)
      ? 0.16
      : 0;
  const queryContextOnlyPenalty =
    operationType === "tailor_advice" &&
    (slotEvidenceRole === "query_context" ||
      slotEvidenceRole === "time_constraint" ||
      !slotEvidenceRole) &&
    filledSlotCount <= 1 &&
    bestSlotCoverage < 0.52
      ? 0.18
      : 0;
  const incompletePenalty = evidenceTextCompletenessPenalty(candidate.text);
  const injectionScore = clamp01(
    candidate.priority * 0.34 +
      semanticScore * 0.2 +
      bestSlotCoverage * 0.16 +
      Math.min(0.12, filledSlotCount * 0.055) +
      focusSupport * 0.08 +
      goalQuerySupport * 0.12 +
      bridgeSupport * 0.14 +
      bridgePositiveSupport * 0.08 +
      capabilitySupport * 0.16 +
      authority * 0.07 +
      sourceTrace +
      roleWeight +
      structuredResource -
      resourceCapabilityPenalty -
      assistantPenalty -
      queryContextOnlyPenalty -
      bridgeNegativeSupport * 0.12 -
      incompletePenalty,
  );
  return {
    ...candidate,
    slotEvidenceRole,
    injectionScore,
    scoreBreakdown: {
      priority: Number(candidate.priority.toFixed(4)),
      semanticScore: Number(semanticScore.toFixed(4)),
      bestSlotCoverage: Number(bestSlotCoverage.toFixed(4)),
      filledSlotCount,
      focusSupport: Number(focusSupport.toFixed(4)),
      goalQuerySupport: Number(goalQuerySupport.toFixed(4)),
      bridgeSupport: Number(bridgeSupport.toFixed(4)),
      bridgePositiveSupport: Number(bridgePositiveSupport.toFixed(4)),
      bridgeNegativeSupport: Number(bridgeNegativeSupport.toFixed(4)),
      capabilitySupport: Number(capabilitySupport.toFixed(4)),
      authority: Number(authority.toFixed(4)),
      sourceTrace,
      slotRole: slotEvidenceRole ?? "none",
      roleWeight,
      structuredResource,
      resourceCapabilityPenalty,
      assistantPenalty,
      queryContextOnlyPenalty,
      incompletePenalty,
    },
  };
}

function promptEvidenceInjectionSelectionReason(
  candidate: PromptEvidenceCandidate,
  index: number,
): string {
  const role = candidate.slotEvidenceRole ? `:${candidate.slotEvidenceRole}` : "";
  const score =
    typeof candidate.injectionScore === "number" ? `:${candidate.injectionScore.toFixed(3)}` : "";
  return `top-injection-score#${index + 1}${role}${score}`;
}

function promptEvidenceNeedsCompanions(queryAnalysis: RecallQueryAnalysis): boolean {
  const surfaces = new Set(queryAnalysis.candidateSurfaces);
  const planSlots = queryAnalysis.evidencePlan?.slots ?? [];
  return (
    planSlots.length > 1 ||
    planSlots.some((slot) => slot.minEvidence > 1) ||
    queryAnalysis.answerMode === "count_aggregate" ||
    queryAnalysis.answerMode === "multi_evidence" ||
    queryAnalysis.queryShape.timeframe === "compare" ||
    queryAnalysis.queryShape.evidenceNeed === "relation" ||
    (queryAnalysis.evidenceGoals ?? []).length > 1 ||
    (surfaces.has("fact") &&
      surfaces.has("event") &&
      !surfaces.has("state") &&
      !surfaces.has("chunk"))
  );
}

function promptEvidenceFromRow(params: {
  row: EvidenceRow;
  surface: PromptEvidenceCandidate["surface"];
  source: PromptEvidenceCandidate["source"];
  queryAnalysis: RecallQueryAnalysis;
}): PromptEvidenceCandidate {
  const text = cleanPromptEvidenceText(params.row.text);
  const supportText = params.row.provenance?.trim();
  const scoringText = supportText
    ? uniqueNonEmpty([text, supportText], 6).join(" | ")
    : text;
  const scored = promptEvidencePriority({
    text: scoringText,
    score: params.row.score,
    surface: params.surface,
    source: params.source,
    queryAnalysis: params.queryAnalysis,
  });
  return {
    id: params.row.id,
    surface: params.surface,
    text,
    rawText: supportText ? `${params.row.text}\n${supportText}` : params.row.text,
    scoringText: scoringText !== text ? scoringText : undefined,
    sourceRef: params.row.sourceRef,
    mergedSourceRefs: params.row.lineage?.sourceRef
      ? uniqueNonEmpty([params.row.sourceRef ?? "", params.row.lineage.sourceRef])
      : undefined,
    metadata: promptEvidenceMetadataForRawText(params.row.text),
    observedAt: params.row.observedAt,
    excerptAnchors: promptEvidenceExcerptAnchors(params.queryAnalysis),
    lineage: params.row.lineage,
    priority: scored.priority,
    goalScore: scored.goalScore,
    semanticScore: scored.semanticScore,
    coverage: evidenceCoverageForText(params.queryAnalysis, scoringText),
    slotCoverage: slotCoverageForText(params.queryAnalysis, scoringText),
    filledSlotIds: filledSlotIdsForText(params.queryAnalysis, scoringText),
    source: params.source,
    role: "support",
  };
}

function metadataStringArray(metadata: Record<string, unknown> | undefined, key: string): string[] {
  const value = metadata?.[key];
  if (!Array.isArray(value)) {
    return [];
  }
  return value.filter(
    (entry): entry is string => typeof entry === "string" && entry.trim().length > 0,
  );
}

function sourceRefsFromCandidateMetadata(metadata: Record<string, unknown> | undefined): string[] {
  const refs: string[] = [
    ...(typeof metadata?.sourceRef === "string" ? [metadata.sourceRef] : []),
    ...metadataStringArray(metadata, "sourceRefs"),
    ...metadataStringArray(metadata, "supportRefs"),
    ...metadataStringArray(metadata, "supportContentRefs"),
    ...metadataStringArray(metadata, "sourceRefsForExpansion"),
  ];
  const maintenanceContract = metadata?.maintenanceContract;
  if (maintenanceContract && typeof maintenanceContract === "object") {
    refs.push(
      ...metadataStringArray(
        maintenanceContract as Record<string, unknown>,
        "sourceRefsForExpansion",
      ),
    );
  }
  const frame = metadata?.turnSemanticFrame;
  if (frame && typeof frame === "object") {
    const record = frame as Record<string, unknown>;
    refs.push(...metadataStringArray(record, "sourceRefs"));
    for (const key of ["chunkDrafts", "assertionDrafts", "correctionDrafts"]) {
      const entries = record[key];
      if (!Array.isArray(entries)) {
        continue;
      }
      for (const entry of entries) {
        if (!entry || typeof entry !== "object") {
          continue;
        }
        const draft = entry as Record<string, unknown>;
        if (typeof draft.sourceRef === "string") {
          refs.push(draft.sourceRef);
        }
        const lineage = draft.lineage;
        if (lineage && typeof lineage === "object") {
          const sourceRef = (lineage as Record<string, unknown>).sourceRef;
          if (typeof sourceRef === "string") {
            refs.push(sourceRef);
          }
        }
        refs.push(...metadataStringArray(draft, "supportRefs"));
        const supportSpans = draft.supportSpans;
        if (Array.isArray(supportSpans)) {
          for (const span of supportSpans) {
            if (span && typeof span === "object") {
              const sourceRef = (span as Record<string, unknown>).sourceRef;
              if (typeof sourceRef === "string") {
                refs.push(sourceRef);
              }
            }
          }
        }
      }
    }
  }
  return uniqueNonEmpty(refs);
}

function numericSuffix(value: string): number | undefined {
  const match = /(\d+)$/u.exec(value);
  if (!match?.[1]) {
    return undefined;
  }
  const parsed = Number.parseInt(match[1], 10);
  return Number.isFinite(parsed) ? parsed : undefined;
}

function turnPartLike(original: string | undefined, value: number): string {
  const match = /^([^\d]*)(\d+)$/u.exec(original ?? "");
  return match?.[1] ? `${match[1]}${value}` : String(value);
}

function sourceExpansionNeighborRefs(sourceRef: string, window = 2): string[] {
  const parts = sourceRef.split(":").filter(Boolean);
  const refs: string[] = [sourceRef];
  const pushTurnWindow = (
    prefix: string[],
    turn: number,
    suffixes: string[] = [],
    userPrefix?: string[],
    turnTemplate?: string,
  ) => {
    for (let next = Math.max(1, turn - window); next <= turn + window; next += 1) {
      const turnPart = turnPartLike(turnTemplate, next);
      refs.push([...prefix, turnPart].join(":"));
      for (const suffix of suffixes) {
        refs.push([...prefix, turnPart, suffix].join(":"));
      }
      if (userPrefix) {
        refs.push([...userPrefix, turnPart].join(":"));
        for (const suffix of suffixes) {
          refs.push([...userPrefix, turnPart, suffix].join(":"));
        }
      }
    }
  };
  if (parts[0] === "lme" && parts.length >= 4) {
    const turnPart = parts[3] ?? "";
    const turn = numericSuffix(turnPart);
    if (turn !== undefined) {
      pushTurnWindow(parts.slice(0, 3), turn, ["user", "assistant"], undefined, turnPart);
    }
  }
  if (parts[0] === "user" && parts[1] === "lme" && parts.length >= 5) {
    const turnPart = parts[4] ?? "";
    const turn = numericSuffix(turnPart);
    if (turn !== undefined) {
      pushTurnWindow(
        ["lme", parts[2]!, parts[3]!],
        turn,
        ["user", "assistant"],
        undefined,
        turnPart,
      );
    }
  }
  if (parts[0] === "agentmem" && parts.length >= 4) {
    const turnPart = parts[3] ?? "";
    const turn = numericSuffix(turnPart);
    if (turn !== undefined) {
      pushTurnWindow(
        ["agentmem", parts[1]!, parts[2]!],
        turn,
        ["user", "assistant"],
        ["user", "agentmem", parts[1]!, parts[2]!],
        turnPart,
      );
    }
  }
  if (parts[0] === "user" && parts[1] === "agentmem" && parts.length >= 5) {
    const turnPart = parts[4] ?? "";
    const turn = numericSuffix(turnPart);
    if (turn !== undefined) {
      pushTurnWindow(
        ["agentmem", parts[2]!, parts[3]!],
        turn,
        ["user", "assistant"],
        ["user", "agentmem", parts[2]!, parts[3]!],
        turnPart,
      );
    }
  }
  if (parts[0] === "user" && parts[1] === "lme" && parts.length >= 6 && parts[4] === "batch") {
    const batch = numericSuffix(parts[5] ?? "");
    if (batch !== undefined) {
      const prefix = ["lme", parts[2]!, parts[3]!];
      const firstTurn = Math.max(1, batch * 2 - 1);
      const lastTurn = Math.max(firstTurn, batch * 2);
      for (let next = Math.max(1, firstTurn - window); next <= lastTurn + window; next += 1) {
        refs.push([...prefix, String(next)].join(":"));
      }
    }
  }
  return uniqueNonEmpty(refs);
}

function maintenanceObjectExpansionRefs(
  store: MemxStoreBundle,
  sourceRef: string,
  seen = new Set<string>(),
): string[] {
  if (!sourceRef || seen.has(sourceRef)) {
    return [];
  }
  seen.add(sourceRef);
  if (sourceRef.startsWith("abstraction_candidate:")) {
    const candidate = store.abstractionRepo.getById(
      sourceRef.slice("abstraction_candidate:".length),
    );
    if (!candidate) {
      return [];
    }
    const refs = uniqueNonEmpty([
      ...sourceRefsFromMaintenanceMetadata(candidate.metadataJson),
      ...candidate.supportContentRefs,
    ]);
    return uniqueNonEmpty([
      ...refs,
      ...refs.flatMap((ref) => maintenanceObjectExpansionRefs(store, ref, seen)),
    ]);
  }
  if (sourceRef.startsWith("fact:")) {
    const fact = store.factRepo.get(sourceRef.slice("fact:".length));
    if (!fact) {
      return [];
    }
    const refs = factSupportSourceRefs(fact);
    return uniqueNonEmpty([
      ...refs,
      ...refs.flatMap((ref) => maintenanceObjectExpansionRefs(store, ref, seen)),
    ]);
  }
  if (sourceRef.startsWith("event:")) {
    const event = store.eventRepo.get(sourceRef.slice("event:".length));
    return event ? uniqueNonEmpty([event.sourceRef]) : [];
  }
  return [];
}

function sourceExpansionSeedScore(candidate: PromptEvidenceCandidate): number {
  const roleBonus =
    candidate.slotEvidenceRole === "answer_value" || candidate.slotEvidenceRole === "answer_event"
      ? 0.18
      : candidate.slotEvidenceRole === "query_context"
        ? 0.12
        : candidate.slotEvidenceRole === "user_resource" ||
            candidate.slotEvidenceRole === "prior_advice"
          ? 0.1
          : 0;
  const sourceTraceBonus =
    candidate.sourceRef || (candidate.mergedSourceRefs?.length ?? 0) > 0 ? 0.06 : 0;
  return clamp01(
    Math.max(candidate.priority, candidate.goalScore, candidate.semanticScore ?? 0) +
      roleBonus +
      sourceTraceBonus +
      Math.min(0.12, (candidate.filledSlotIds?.length ?? 0) * 0.04),
  );
}

function sourceExpansionBaseRefs(candidate: PromptEvidenceCandidate): string[] {
  return uniqueNonEmpty(
    [
      candidate.sourceRef,
      candidate.lineage?.sourceRef,
      ...(candidate.mergedSourceRefs ?? []),
      ...sourceRefsFromCandidateMetadata(candidate.metadata),
    ].filter((value): value is string => typeof value === "string"),
  );
}

function sourceExpansionRefsForCandidates(
  store: MemxStoreBundle,
  candidates: PromptEvidenceCandidate[],
): string[] {
  const refs: string[] = [];
  const seen = new Set<string>();
  const pushRef = (sourceRef: string) => {
    if (!sourceRef || seen.has(sourceRef)) {
      return;
    }
    seen.add(sourceRef);
    refs.push(sourceRef);
  };

  const sortedSeeds = [...candidates]
    .filter((candidate) => sourceExpansionBaseRefs(candidate).length > 0)
    .sort((left, right) => sourceExpansionSeedScore(right) - sourceExpansionSeedScore(left))
    .slice(0, 40);

  for (const candidate of sortedSeeds) {
    for (const sourceRef of sourceExpansionBaseRefs(candidate)) {
      for (const neighborRef of sourceExpansionNeighborRefs(sourceRef)) {
        pushRef(neighborRef);
      }
      for (const expandedRef of maintenanceObjectExpansionRefs(store, sourceRef)) {
        for (const neighborRef of sourceExpansionNeighborRefs(expandedRef)) {
          pushRef(neighborRef);
        }
      }
    }
    if (refs.length >= 360) {
      break;
    }
  }

  return refs.slice(0, 360);
}

function sourceExpansionChunksForRefs(params: {
  store: MemxStoreBundle;
  ctx: MemoryOperationContext;
  sourceRefs: string[];
}): ConversationChunk[] {
  const chunkById = new Map<string, ConversationChunk>();
  const directSourceRefs: string[] = [];

  const pushChunk = (chunk: ConversationChunk | null) => {
    if (!chunk) {
      return;
    }
    if (chunk.agentId !== params.ctx.agentId || !params.ctx.scopes.includes(chunk.scope)) {
      return;
    }
    if (chunk.dedupStatus !== "active") {
      return;
    }
    chunkById.set(chunk.chunkId, chunk);
    directSourceRefs.push(chunk.sourceRef);
  };

  for (const sourceRef of params.sourceRefs) {
    if (sourceRef.startsWith("chunk:")) {
      pushChunk(params.store.chunkRepo.get(sourceRef.slice("chunk:".length)));
      continue;
    }
    if (sourceRef.startsWith("task:")) {
      const taskId = sourceRef.slice("task:".length).split(":")[0];
      if (taskId) {
        for (const chunk of params.store.chunkRepo.listByTask(taskId)) {
          pushChunk(chunk);
        }
      }
      continue;
    }
    directSourceRefs.push(sourceRef);
  }

  const sourceRefs = uniqueNonEmpty([...params.sourceRefs, ...directSourceRefs]);
  for (const chunk of params.store.chunkRepo.listBySourceRefs({
    agentId: params.ctx.agentId,
    scopes: params.ctx.scopes,
    sourceRefs,
    limit: Math.max(32, sourceRefs.length),
  })) {
    pushChunk(chunk);
  }

  return [...chunkById.values()].sort(
    (left, right) =>
      Date.parse(right.createdAt) - Date.parse(left.createdAt) || right.seq - left.seq,
  );
}

function promptEvidenceFromChunkExpansion(params: {
  chunk: ConversationChunk;
  queryAnalysis: RecallQueryAnalysis;
  seedSourceRefs: string[];
}): PromptEvidenceCandidate {
  const text = cleanPromptEvidenceText(`${params.chunk.role}: ${params.chunk.content}`);
  const scored = promptEvidencePriority({
    text,
    score: 0.64,
    surface: "chunk",
    source: "support_ref",
    queryAnalysis: params.queryAnalysis,
  });
  const mergedSourceRefs = uniqueNonEmpty([params.chunk.sourceRef]);
  const candidate: PromptEvidenceCandidate = {
    id: `source-expansion:${params.chunk.chunkId}`,
    surface: "chunk",
    text,
    rawText: `${params.chunk.role}: ${params.chunk.content}`,
    scoringText: text,
    sourceRef: params.chunk.sourceRef,
    mergedSourceRefs,
    observedAt: params.chunk.createdAt,
    excerptAnchors: promptEvidenceExcerptAnchors(params.queryAnalysis),
    lineage: {
      sourceKind: "chunk",
      sourceId: params.chunk.chunkId,
      sourceRef: params.chunk.sourceRef,
    },
    metadata: {
      sourceExpansion: true,
      authorRole: params.chunk.role,
      neighborOf: [params.chunk.sourceRef],
      turnId: params.chunk.turnId,
      sessionKey: params.chunk.sessionKey,
    },
    priority: scored.priority,
    goalScore: scored.goalScore,
    semanticScore: scored.semanticScore,
    coverage: evidenceCoverageForText(params.queryAnalysis, text),
    slotCoverage: slotCoverageForText(params.queryAnalysis, text),
    filledSlotIds: filledSlotIdsForText(params.queryAnalysis, text),
    source: "support_ref",
    role: "support",
    selectionReason: "source-expansion:neighbor-chunk",
  };
  return {
    ...candidate,
    slotEvidenceRole: promptEvidenceSlotRoleForCandidate(params.queryAnalysis, candidate),
  };
}

function promptEvidenceFromSegmentExpansion(params: {
  segment: SourceSegmentRecord;
  queryAnalysis: RecallQueryAnalysis;
  seedSourceRefs: string[];
}): PromptEvidenceCandidate {
  const text = cleanPromptEvidenceText(`${params.segment.role}: ${params.segment.text}`);
  const scored = promptEvidencePriority({
    text,
    score: 0.64,
    surface: "chunk",
    source: "support_ref",
    queryAnalysis: params.queryAnalysis,
  });
  const candidate: PromptEvidenceCandidate = {
    id: `source-segment-expansion:${params.segment.segmentId}`,
    surface: "chunk",
    text,
    rawText: `${params.segment.role}: ${params.segment.text}`,
    scoringText: text,
    sourceRef: params.segment.parentSourceRef,
    mergedSourceRefs: uniqueNonEmpty([params.segment.parentSourceRef, ...params.seedSourceRefs]),
    observedAt: params.segment.createdAt,
    excerptAnchors: promptEvidenceExcerptAnchors(params.queryAnalysis),
    lineage: {
      sourceKind: "chunk",
      sourceId: params.segment.chunkId,
      sourceRef: params.segment.parentSourceRef,
    },
    metadata: {
      sourceExpansion: true,
      authorRole: params.segment.role,
      neighborOf: [params.segment.parentSourceRef],
      turnId: params.segment.turnId,
      sessionKey: params.segment.sessionKey,
      sourceGroupId: params.segment.sourceGroupId,
      segmentId: params.segment.segmentId,
      segmentIndex: params.segment.segmentIndex,
      charStart: params.segment.charStart,
      charEnd: params.segment.charEnd,
    },
    priority: scored.priority,
    goalScore: scored.goalScore,
    semanticScore: scored.semanticScore,
    coverage: evidenceCoverageForText(params.queryAnalysis, text),
    slotCoverage: slotCoverageForText(params.queryAnalysis, text),
    filledSlotIds: filledSlotIdsForText(params.queryAnalysis, text),
    source: "support_ref",
    role: "support",
    selectionReason: "source-expansion:source-segment",
  };
  return {
    ...candidate,
    slotEvidenceRole: promptEvidenceSlotRoleForCandidate(params.queryAnalysis, candidate),
  };
}

function sourceExpansionPromptEvidenceCandidates(params: {
  store: MemxStoreBundle;
  ctx: MemoryOperationContext;
  queryAnalysis: RecallQueryAnalysis;
  candidates: PromptEvidenceCandidate[];
}): PromptEvidenceCandidate[] {
  const sourceRefs = sourceExpansionRefsForCandidates(params.store, params.candidates);
  if (sourceRefs.length === 0) {
    return [];
  }
  const chunks = sourceExpansionChunksForRefs({
    store: params.store,
    ctx: params.ctx,
    sourceRefs,
  });
  const candidates: PromptEvidenceCandidate[] = [];
  for (const chunk of chunks) {
    const segments =
      chunk.content.length > SOURCE_SEGMENT_TARGET_CHARS
        ? params.store.sourceSegmentRepo.listByChunk(chunk.chunkId)
        : [];
    if (segments.length > 1) {
      candidates.push(
        ...segments.map((segment) =>
          promptEvidenceFromSegmentExpansion({
            segment,
            queryAnalysis: params.queryAnalysis,
            seedSourceRefs: sourceRefs,
          }),
        ),
      );
      continue;
    }
    candidates.push(
      promptEvidenceFromChunkExpansion({
        chunk,
        queryAnalysis: params.queryAnalysis,
        seedSourceRefs: sourceRefs,
      }),
    );
  }
  return candidates;
}

function promptEvidenceSourceRefs(candidate: PromptEvidenceCandidate): string[] {
  return uniqueNonEmpty(
    [
      candidate.sourceRef,
      candidate.lineage?.sourceRef,
      ...(candidate.mergedSourceRefs ?? []),
      ...sourceRefsFromCandidateMetadata(candidate.metadata),
    ].filter((value): value is string => typeof value === "string" && value.length > 0),
  );
}

function sourceRefFamily(sourceRef: string): string {
  const parts = sourceRef.split(":").filter(Boolean);
  const normalizeSessionPart = (part: string) =>
    /^answer_[a-z0-9]+_\d+$/iu.test(part) ? part.replace(/_\d+$/u, "") : part;
  if (parts[0] === "user" && parts[1] === "agentmem" && parts.length >= 4) {
    return `agentmem:${parts[2]}:${normalizeSessionPart(parts[3] ?? "")}`;
  }
  if (parts[0] === "agentmem" && parts.length >= 3) {
    return `agentmem:${parts[1]}:${normalizeSessionPart(parts[2] ?? "")}`;
  }
  if (parts[0] === "user" && parts[1] === "lme" && parts.length >= 4) {
    return `lme:${parts[2]}:${normalizeSessionPart(parts[3] ?? "")}`;
  }
  if (parts[0] === "lme" && parts.length >= 3) {
    return `lme:${parts[1]}:${normalizeSessionPart(parts[2] ?? "")}`;
  }
  return parts.slice(0, Math.min(3, parts.length)).join(":") || sourceRef;
}

function sourceRefTurnIndex(sourceRef: string): number | undefined {
  const parts = sourceRef.split(":").filter(Boolean);
  for (let index = parts.length - 1; index >= 0; index -= 1) {
    const value = numericSuffix(parts[index] ?? "");
    if (value !== undefined) {
      return value;
    }
  }
  return undefined;
}

function promptEvidenceRefsAdjacent(
  left: PromptEvidenceCandidate,
  right: PromptEvidenceCandidate,
  maxDistance = 2,
): boolean {
  for (const leftRef of promptEvidenceSourceRefs(left)) {
    const leftFamily = sourceRefFamily(leftRef);
    const leftTurn = sourceRefTurnIndex(leftRef);
    for (const rightRef of promptEvidenceSourceRefs(right)) {
      if (leftFamily !== sourceRefFamily(rightRef)) {
        continue;
      }
      const rightTurn = sourceRefTurnIndex(rightRef);
      if (leftTurn === undefined || rightTurn === undefined) {
        if (leftRef === rightRef) {
          return true;
        }
        continue;
      }
      if (Math.abs(leftTurn - rightTurn) <= maxDistance) {
        return true;
      }
    }
  }
  return false;
}

function shouldRetainSourceExpansionAnswerCandidate(
  queryAnalysis: RecallQueryAnalysis,
  candidate: PromptEvidenceCandidate,
): boolean {
  if (candidate.metadata?.sourceExpansion !== true) {
    return false;
  }
  if (promptEvidenceRole(candidate) !== "assistant") {
    return false;
  }
  const slotRole =
    candidate.slotEvidenceRole ?? promptEvidenceSlotRoleForCandidate(queryAnalysis, candidate);
  if (slotRole === "answer_value" || slotRole === "answer_event") {
    return true;
  }
  const score = Math.max(
    candidate.injectionScore ?? 0,
    candidate.priority,
    candidate.goalScore,
    candidate.semanticScore ?? 0,
    promptEvidenceAnswerSlotStrength(queryAnalysis, candidate),
    promptEvidenceAnswerSpecificity(queryAnalysis, candidate),
  );
  return score >= 0.42;
}

function keepRetainedPromptEvidenceCandidates(params: {
  queryAnalysis: RecallQueryAnalysis;
  candidates: PromptEvidenceCandidate[];
  cap: number;
}): PromptEvidenceCandidate[] {
  const keyFor = (candidate: PromptEvidenceCandidate) =>
    [
      candidate.surface,
      candidate.id,
      candidate.sourceRef ?? "",
      normalizeText(candidate.text).slice(0, 120),
    ].join("|");
  const result = params.candidates.slice(0, params.cap);
  const seen = new Set(result.map(keyFor));
  const retainedAnswers = params.candidates
    .filter((candidate) =>
      shouldRetainSourceExpansionAnswerCandidate(params.queryAnalysis, candidate),
    )
    .slice(0, 8);
  const retainedContext = params.candidates
    .filter((candidate) =>
      retainedAnswers.some((answer) => {
        if (keyFor(answer) === keyFor(candidate)) {
          return false;
        }
        if (candidate.metadata?.sourceExpansion !== true) {
          return false;
        }
        const role =
          candidate.slotEvidenceRole ??
          promptEvidenceSlotRoleForCandidate(params.queryAnalysis, candidate);
        return (
          promptEvidenceRefsAdjacent(answer, candidate, 2) &&
          (role === "query_context" || promptEvidenceRole(candidate) === "user")
        );
      }),
    )
    .slice(0, 12);

  const reserve = [...retainedAnswers, ...retainedContext];
  for (const candidate of reserve) {
    const key = keyFor(candidate);
    if (seen.has(key)) {
      continue;
    }
    const replaceIndex = [...result]
      .map((entry, index) => ({ entry, index }))
      .reverse()
      .find(
        ({ entry }) => !shouldRetainSourceExpansionAnswerCandidate(params.queryAnalysis, entry),
      )?.index;
    if (replaceIndex === undefined) {
      result.push(candidate);
    } else {
      seen.delete(keyFor(result[replaceIndex]!));
      result[replaceIndex] = candidate;
    }
    seen.add(key);
  }
  return result.slice(0, Math.max(params.cap, reserve.length));
}

function controlPromptEvidenceCandidates(params: {
  store: MemxStoreBundle;
  ctx: MemoryOperationContext;
  queryAnalysis: RecallQueryAnalysis;
}): PromptEvidenceCandidate[] {
  const slots = params.queryAnalysis.evidencePlan?.slots ?? [];
  if (
    !slots.some((slot) =>
      evidencePlanLayersForSlot(slot).some(
        (layer) =>
          layer === "control" ||
          layer === "strategy" ||
          layer === "abstraction" ||
          layer === "belief",
      ),
    )
  ) {
    return [];
  }
  const strategies = params.store.strategyRepo.listByAgent({
    agentId: params.ctx.agentId,
    scopes: params.ctx.scopes,
    stages: ["active", "candidate"],
    limit: 24,
    readEpoch: params.ctx.readEpoch,
  });
  const abstractions = params.store.abstractionRepo.listByAgent({
    agentId: params.ctx.agentId,
    scopes: params.ctx.scopes,
    stages: ["active", "candidate", "probationary"],
    limit: 24,
    readEpoch: params.ctx.readEpoch,
  });
  const beliefs = params.store.beliefRepo
    .listByAgent({
      agentId: params.ctx.agentId,
      limit: 48,
      readEpoch: params.ctx.readEpoch,
    })
    .filter((belief) => params.ctx.scopes.includes(belief.scope));
  const rows = [
    ...strategies.map((entry) => {
      const sourceRefs = uniqueMaintenanceRefs([
        ...sourceRefsFromMaintenanceMetadata(entry.metadataJson),
        ...entry.supportTaskIds.map((taskId) => `task:${taskId}`),
      ]);
      return {
        id: `strategy:${entry.strategyId}`,
        text: `strategy: ${entry.summary}`,
        sourceRef: sourceRefs[0],
        sourceRefs,
        layer: "strategy" as const,
        score: entry.confidence * 0.6 + entry.usefulnessScore * 0.4,
      };
    }),
    ...abstractions.map((entry) => {
      const sourceRefs = uniqueMaintenanceRefs([
        ...sourceRefsFromMaintenanceMetadata(entry.metadataJson),
        ...entry.supportContentRefs,
      ]);
      return {
        id: `abstraction:${entry.candidateId}`,
        text: `${entry.abstractionType}: ${entry.summary}`,
        sourceRef: sourceRefs[0],
        sourceRefs,
        layer: "abstraction" as const,
        score: entry.confidence * 0.6 + entry.usefulnessScore * 0.4,
      };
    }),
    ...beliefs.map((entry) => ({
      id: `belief:${entry.beliefId}`,
      text: `belief: ${entry.semanticKey}`,
      sourceRef: entry.contentRef,
      layer: "belief" as const,
      score: entry.posteriorConfidence * 0.7 + entry.usefulnessScore * 0.3,
    })),
  ];
  return rows
    .map((row) => {
      const text = cleanPromptEvidenceText(row.text);
      const slotCoverage = slotCoverageForText(params.queryAnalysis, text);
      const filledSlotIds = slotCoverage
        .filter((coverage) => coverage.filled)
        .map((coverage) => coverage.slotId);
      const scored = promptEvidencePriority({
        text,
        score: row.score,
        surface: "snippet",
        source: "selected",
        queryAnalysis: params.queryAnalysis,
      });
      return {
        id: row.id,
        surface: "snippet" as const,
        text,
        rawText: row.text,
        sourceRef: row.sourceRef,
        mergedSourceRefs: "sourceRefs" in row ? row.sourceRefs : undefined,
        observedAt: undefined,
        excerptAnchors: promptEvidenceExcerptAnchors(params.queryAnalysis),
        lineage: {
          sourceKind: "alternate" as const,
          sourceId: row.id,
          sourceRef: row.sourceRef,
        },
        priority: scored.priority,
        goalScore: scored.goalScore,
        semanticScore: scored.semanticScore,
        coverage: evidenceCoverageForText(params.queryAnalysis, text),
        slotCoverage,
        filledSlotIds,
        source: "selected" as const,
        role: "support" as const,
      };
    })
    .filter((entry) => (entry.filledSlotIds?.length ?? 0) > 0 || entry.priority >= 0.48)
    .sort((left, right) => right.priority - left.priority)
    .slice(0, 4);
}

function buildPromptEvidenceCandidates(params: {
  store: MemxStoreBundle;
  ctx: MemoryOperationContext;
  queryAnalysis: RecallQueryAnalysis;
  candidateGenerationResult?: CandidateGenerationResult;
  facts: EvidenceRow[];
  events: EvidenceRow[];
  controlEvidence?: PromptEvidenceCandidate[];
  selectedExactSnippets?: ExactSnippetCandidate[];
  suppressedCandidateIds?: Set<string>;
  suppressCandidateEvidenceBeforeMs?: number;
}): PromptEvidenceCandidate[] {
  const candidates: PromptEvidenceCandidate[] = [];
  const excerptAnchors = promptEvidenceExcerptAnchors(params.queryAnalysis);
  const candidateHits = [
    ...(params.candidateGenerationResult?.candidates ?? []),
    ...(params.candidateGenerationResult?.slotCandidates ?? []),
    ...(params.candidateGenerationResult?.bridgeCandidates ?? []),
  ];
  for (const hit of candidateHits) {
    const hitIds = uniqueNonEmpty([
      hit.docId,
      hit.candidateId,
      hit.lineage.canonicalId,
      hit.lineage.sourceId,
    ].filter((id): id is string => typeof id === "string" && id.trim().length > 0));
    if (hitIds.some((id) => params.suppressedCandidateIds?.has(id))) {
      continue;
    }
    const surface = candidateSurfaceToPromptSurface(hit.surface);
    if (!surface) {
      continue;
    }
    const metadata = hydrateFactCandidateMetadata(params.store, hit);
    const observedAt = typeof metadata?.observedAt === "string" ? Date.parse(metadata.observedAt) : Number.NaN;
    if (
      typeof params.suppressCandidateEvidenceBeforeMs === "number" &&
      Number.isFinite(observedAt) &&
      observedAt < params.suppressCandidateEvidenceBeforeMs
    ) {
      continue;
    }
    if (hit.surface === "fact" && !factCandidateVisibleForQuery(params.queryAnalysis, metadata)) {
      continue;
    }
    const text = cleanPromptEvidenceText(candidateDisplayText(hit, metadata));
    const scoringText = promptEvidenceScoringTextFromMetadata(text, metadata);
    const slotCoverage = slotCoverageWithCandidateMatches(params.queryAnalysis, scoringText, hit);
    const scored = promptEvidencePriority({
      text: scoringText,
      score: hit.score,
      surface,
      source: "candidate",
      tier: hit.tier ?? "primary",
      queryAnalysis: params.queryAnalysis,
      explicitGoalScore: topCandidateGoalScore(hit),
    });
    const matchedSlotBoost = slotCoverage.some((coverage) => coverage.filled) ? 0.12 : 0;
    const filledSlotIds = filledSlotIdsFromCoverage(slotCoverage);
    const priority = clamp01(scored.priority + matchedSlotBoost);
    const promptCandidate: PromptEvidenceCandidate = {
      id: hit.docId ?? hit.candidateId,
      surface,
      text,
      rawText: hit.text,
      scoringText,
      sourceRef:
        typeof metadata?.sourceRef === "string" && metadata.sourceRef.trim()
          ? metadata.sourceRef.trim()
          : hit.lineage.sourceRef,
      mergedSourceRefs: uniqueNonEmpty([
        ...sourceRefsFromCandidateMetadata(metadata),
        hit.lineage.sourceRef,
      ]),
      observedAt:
        typeof metadata?.observedAt === "string" ? metadata.observedAt : undefined,
      excerptAnchors,
      lineage: hit.lineage,
      metadata: promptEvidenceMetadataForRawText(hit.text, metadata),
      priority,
      goalScore: scored.goalScore,
      semanticScore: scored.semanticScore,
      coverage: evidenceCoverageForText(params.queryAnalysis, scoringText),
      bridgeMatches: hit.bridgeMatches,
      slotCoverage,
      filledSlotIds,
      source: "candidate",
      role: "support",
    };
    candidates.push({
      ...promptCandidate,
      slotEvidenceRole: promptEvidenceSlotRoleForCandidate(params.queryAnalysis, promptCandidate),
    });
  }
  candidates.push(...(params.controlEvidence ?? []));
  for (const fact of params.facts) {
    candidates.push(
      promptEvidenceFromRow({
        row: fact,
        surface: "fact",
        source: "selected",
        queryAnalysis: params.queryAnalysis,
      }),
    );
  }
  for (const event of params.events) {
    candidates.push(
      promptEvidenceFromRow({
        row: event,
        surface: isChunkEvidenceRow(event) ? "chunk" : "event",
        source: event.lineage?.sourceKind === "chunk" ? "support_ref" : "selected",
        queryAnalysis: params.queryAnalysis,
      }),
    );
  }
  for (const snippet of params.selectedExactSnippets ?? []) {
    const text = cleanPromptEvidenceText(snippet.text);
    const scored = promptEvidencePriority({
      text,
      score: snippet.goalScore,
      surface: "snippet",
      source: snippet.source,
      queryAnalysis: params.queryAnalysis,
      explicitGoalScore: snippet.goalScore,
    });
    candidates.push({
      id: snippet.snippetId,
      surface: "snippet",
      text,
      rawText: snippet.text,
      sourceRef: snippet.sourceRef,
      observedAt: undefined,
      excerptAnchors,
      lineage: snippet.lineage,
      priority: scored.priority,
      goalScore: scored.goalScore,
      semanticScore: scored.semanticScore,
      coverage: evidenceCoverageForText(params.queryAnalysis, text),
      slotCoverage: slotCoverageForText(params.queryAnalysis, text),
      filledSlotIds: filledSlotIdsForText(params.queryAnalysis, text),
      source: snippet.source,
      role: "support",
    });
  }

  candidates.push(
    ...sourceExpansionPromptEvidenceCandidates({
      store: params.store,
      ctx: params.ctx,
      queryAnalysis: params.queryAnalysis,
      candidates,
    }),
  );

  const ordered = candidates.sort((left, right) => {
    const priorityDelta = right.priority - left.priority;
    const nonSubjectHitDelta =
      promptEvidenceNonSubjectSlotHitCount(params.queryAnalysis, right) -
      promptEvidenceNonSubjectSlotHitCount(params.queryAnalysis, left);
    if (
      nonSubjectHitDelta !== 0 &&
      promptEvidenceRequiresAnswerSlotSupport(params.queryAnalysis) &&
      Math.abs(priorityDelta) <= 0.32
    ) {
      return nonSubjectHitDelta;
    }
    const answerSlotDelta =
      promptEvidenceAnswerSlotStrength(params.queryAnalysis, right) -
      promptEvidenceAnswerSlotStrength(params.queryAnalysis, left);
    if (Math.abs(answerSlotDelta) >= 0.08 && Math.abs(priorityDelta) <= 0.3) {
      return answerSlotDelta;
    }
    const specificityDelta =
      promptEvidenceAnswerSpecificity(params.queryAnalysis, right) -
      promptEvidenceAnswerSpecificity(params.queryAnalysis, left);
    if (Math.abs(specificityDelta) >= 0.08 && Math.abs(priorityDelta) <= 0.22) {
      return specificityDelta;
    }
    if (Math.abs(priorityDelta) > 0.03) {
      return priorityDelta;
    }
    if (specificityDelta !== 0) {
      return specificityDelta;
    }
    if ((right.semanticScore ?? 0) !== (left.semanticScore ?? 0)) {
      return (right.semanticScore ?? 0) - (left.semanticScore ?? 0);
    }
    if (priorityDelta !== 0) {
      return priorityDelta;
    }
    if (right.goalScore !== left.goalScore) {
      return right.goalScore - left.goalScore;
    }
    const filledSlotDelta = (right.filledSlotIds?.length ?? 0) - (left.filledSlotIds?.length ?? 0);
    if (filledSlotDelta !== 0) {
      return filledSlotDelta;
    }
    const authorityDelta = promptEvidenceAuthorityRank(right) - promptEvidenceAuthorityRank(left);
    if (authorityDelta !== 0) {
      return authorityDelta;
    }
    const observedDelta = (right.observedAt ?? "").localeCompare(left.observedAt ?? "");
    if (observedDelta !== 0) {
      return observedDelta;
    }
    if (right.priority !== left.priority) {
      return right.priority - left.priority;
    }
    return right.goalScore - left.goalScore;
  });
  const selected: PromptEvidenceCandidate[] = [];
  const dropped: PromptEvidenceCandidate[] = [];
  const seen = new Set<string>();
  for (const candidate of ordered) {
    const key = [
      candidate.sourceRef ?? "",
      candidate.id,
      normalizeText(candidate.text).slice(0, 240),
    ].join("|");
    if (seen.has(key)) {
      const sameSourceIndex = selected.findIndex((entry) => {
        return (
          (entry.sourceRef ?? "") === (candidate.sourceRef ?? "") &&
          entry.id === candidate.id &&
          normalizeText(entry.text).slice(0, 240) === normalizeText(candidate.text).slice(0, 240)
        );
      });
      if (sameSourceIndex >= 0) {
        const existing = selected[sameSourceIndex]!;
        if (shouldReplacePromptEvidence(existing, candidate, params.queryAnalysis)) {
          selected[sameSourceIndex] = mergePromptEvidenceSourceRefs(candidate, existing);
          dropped.push({
            ...existing,
            role: "alternate",
            dropReason: "duplicate-source-or-text",
          });
          continue;
        }
        selected[sameSourceIndex] = mergePromptEvidenceSourceRefs(existing, candidate);
      }
      dropped.push({
        ...candidate,
        role: "alternate",
        dropReason: "duplicate-source-or-text",
      });
      continue;
    }
    seen.add(key);
    selected.push(candidate);
  }
  const ranked = selected
    .map((candidate) => scorePromptEvidenceForInjection(params.queryAnalysis, candidate))
    .sort((left, right) => {
      const scoreDelta = (right.injectionScore ?? 0) - (left.injectionScore ?? 0);
      if (Math.abs(scoreDelta) > 0.0001) {
        return scoreDelta;
      }
      if (right.priority !== left.priority) {
        return right.priority - left.priority;
      }
      return right.goalScore - left.goalScore;
    });
  const injectionLimit = promptEvidenceInjectionLimit(params.queryAnalysis, ranked);
  const selectedWithRoles = ranked.map((candidate, index) => {
    const role =
      index < Math.max(8, injectionLimit) ? ("support" as const) : ("alternate" as const);
    const candidateMissingRequired = candidate.coverage?.missingRequired;
    const missingRequired =
      candidateMissingRequired && candidateMissingRequired.length > 0
        ? candidateMissingRequired
        : evidenceCoverageForText(params.queryAnalysis, candidate.text).missingRequired;
    const selectionReason =
      missingRequired.length > 0
        ? `candidate-stage-missing-required:${missingRequired.join(",")}`
        : `candidate-stage-ranked#${index + 1}`;
    return {
      ...candidate,
      role,
      selectionReason,
      protectionReason: selectionReason,
    };
  });
  const scoredDropped = dropped.map((candidate) => ({
    ...scorePromptEvidenceForInjection(params.queryAnalysis, candidate),
    role: "alternate" as const,
    selectionReason: candidate.dropReason ?? "dropped-before-ranking",
    protectionReason: candidate.dropReason ?? "dropped-before-ranking",
  }));
  return keepRetainedPromptEvidenceCandidates({
    queryAnalysis: params.queryAnalysis,
    candidates: [...selectedWithRoles, ...scoredDropped],
    cap: 64,
  });
}

function buildEvidencePlanAudit(params: {
  queryAnalysis: RecallQueryAnalysis;
  promptEvidence: PromptEvidenceCandidate[];
  candidateGenerationResult?: CandidateGenerationResult;
  injectedEvidence: PromptEvidenceCandidate[];
}): EvidencePlanAudit | undefined {
  const plan = params.queryAnalysis.evidencePlan;
  if (!plan || plan.slots.length === 0) {
    return undefined;
  }
  const statsBySlotLayer = new Map<string, { raw: number; selected: number; alternate: number }>();
  for (const stat of params.candidateGenerationResult?.slotLayerStats ?? []) {
    const key = `${stat.slotId}:${stat.layer}`;
    const current = statsBySlotLayer.get(key) ?? { raw: 0, selected: 0, alternate: 0 };
    current.raw += stat.rawCount;
    current.selected += stat.selectedCount;
    current.alternate += stat.alternateCount;
    statsBySlotLayer.set(key, current);
  }
  return {
    operation: plan.operation,
    slots: plan.slots.map((slot) => {
      const queriedLayers = evidencePlanLayersForSlot(slot);
      const layerCandidateCounts: Partial<Record<EvidencePlanLayer, number>> = {};
      for (const layer of queriedLayers) {
        const stats = statsBySlotLayer.get(`${slot.id}:${layer}`);
        if (stats) {
          layerCandidateCounts[layer] = stats.raw;
        }
      }
      const selectedEvidence = params.promptEvidence.filter((entry) =>
        (entry.slotCoverage ?? []).some(
          (coverage) =>
            coverage.slotId === slot.id && (coverage.filled || coverage.coverageScore >= 0.3),
        ),
      );
      const filledEvidence = selectedEvidence.filter((entry) =>
        (entry.filledSlotIds ?? []).includes(slot.id),
      );
      const protectedEvidence = selectedEvidence.filter(
        (entry) => entry.role === "protected" && !entry.dropReason,
      );
      const injectedEvidence = params.injectedEvidence.filter((entry) =>
        (entry.filledSlotIds ?? []).includes(slot.id),
      );
      const bestCoverage = Math.max(
        0,
        ...selectedEvidence.flatMap((entry) =>
          (entry.slotCoverage ?? [])
            .filter((coverage) => coverage.slotId === slot.id)
            .map((coverage) => coverage.coverageScore),
        ),
      );
      const missingFields = uniqueNonEmpty(
        selectedEvidence.flatMap((entry) =>
          (entry.slotCoverage ?? [])
            .filter((coverage) => coverage.slotId === slot.id)
            .flatMap((coverage) => coverage.missingRequired),
        ),
      );
      return {
        slotId: slot.id,
        description: slot.description,
        queriedLayers,
        layerCandidateCounts,
        filled: filledEvidence.length >= slot.minEvidence,
        missingFields: filledEvidence.length >= slot.minEvidence ? [] : missingFields,
        coverageScore: bestCoverage,
        selectedEvidenceIds: selectedEvidence.map((entry) => entry.id),
        protectedEvidenceIds: protectedEvidence.map((entry) => entry.id),
        injectedEvidenceIds: injectedEvidence.map((entry) => entry.id),
        sourceRefs: uniqueNonEmpty(
          selectedEvidence
            .flatMap((entry) => [entry.sourceRef, ...(entry.mergedSourceRefs ?? [])])
            .filter((ref): ref is string => typeof ref === "string" && ref.length > 0),
        ),
      };
    }),
  };
}

function normalizedPromptEvidenceSourceRefs(entry: PromptEvidenceCandidate) {
  return normalizeSourceRefs([entry.sourceRef, ...(entry.mergedSourceRefs ?? [])]);
}

function renderedPromptLineAudit(bundle: EvidenceBundle) {
  return bundle.evidencePackets
    .filter((packet) => packet.injected && !packet.dropReason)
    .sort(
      (left, right) =>
        (right.grade?.finalScore ?? right.coverage.confidence) -
          (left.grade?.finalScore ?? left.coverage.confidence) ||
        right.coverage.confidence - left.coverage.confidence,
    )
    .slice(0, 6)
    .flatMap((packet) => {
      const evidenceUnitIds = [
        ...(packet.answerUnits ?? []),
        ...(packet.contextUnits ?? []),
        ...(packet.supportUnits ?? []),
      ].map((unit) => unit.unitId);
      const refs = packet.allSourceRefs ?? packet.sourceRefs;
      const lines = (packet.displayLines ?? [packet.primaryText]).filter(
        (line) => line.trim().length > 0,
      );
      return lines.map((line, index) => {
        const role = promptLineRole(line);
        return {
          lineId: `prompt_line:${packet.packetId}:${index + 1}`,
          packetId: packet.packetId,
          role,
          answerLine: isAnswerPromptLineRole(role),
          supportLine: !isAnswerPromptLineRole(role),
          line,
          sourceRefs: refs,
          normalizedSourceRefs: normalizeSourceRefs(refs),
          evidenceUnitIds,
        };
      });
    });
}

function isChunkEvidenceRow(row: EvidenceRow): boolean {
  return row.lineage?.sourceKind === "chunk";
}

function supportRefsFromFactObjectValueJson(
  value: Record<string, unknown> | null | undefined,
): string[] {
  const refs = value?.supportRefs;
  return Array.isArray(refs)
    ? uniqueNonEmpty(refs.filter((entry): entry is string => typeof entry === "string"))
    : [];
}

function supportRefsFromSelectedFacts(
  store: MemxStoreBundle,
  facts: EvidenceRow[],
  queryAnchors: string[],
): string[] {
  const rankedFacts = facts
    .map((factRow) => ({
      factRow,
      anchorScore: candidateSurfaceAnchorScore(factRow, queryAnchors),
      score: factRow.score ?? 0,
    }))
    .sort((left, right) => {
      if (right.anchorScore !== left.anchorScore) {
        return right.anchorScore - left.anchorScore;
      }
      return right.score - left.score;
    });
  const bestAnchorScore = rankedFacts[0]?.anchorScore ?? 0;
  const selectedFacts = rankedFacts
    .filter(
      (entry, index) =>
        entry.anchorScore >= Math.max(0.38, bestAnchorScore - 0.08) ||
        (index === 0 && bestAnchorScore < 0.38),
    )
    // Amount/entity questions often need a small set of complementary facts
    // rather than a single top hit.
    .slice(0, 3)
    .map((entry) => entry.factRow);
  const refs: string[] = [];
  for (const factRow of selectedFacts) {
    const factId =
      factRow.lineage?.canonicalKind === "fact"
        ? (factRow.lineage.canonicalId ?? factRow.lineage.sourceId)
        : factRow.lineage?.sourceKind === "fact"
          ? factRow.lineage.sourceId
          : undefined;
    if (!factId) {
      continue;
    }
    const fact = store.factRepo.get(factId);
    refs.push(...supportRefsFromFactObjectValueJson(fact?.objectValueJson));
  }
  return uniqueNonEmpty(refs);
}

function chunkEvidenceRowFromConversationChunk(
  chunk: ConversationChunk,
  queryAnchors: string[],
): EvidenceRow {
  const text = normalizeSearchText(chunk.content.trim() || chunk.summary.trim());
  const anchorScore = queryAnchors.length > 0 ? queryAnchorSupport(text, queryAnchors) : 0;
  return toEvidenceRow({
    id: chunk.chunkId,
    text,
    score: Math.max(0.36, 0.72 + anchorScore * 0.22),
    scope: chunk.scope,
    confidence: 0.82,
    observedAt: chunk.createdAt,
    sourceRef: chunk.sourceRef,
    provenance: chunk.role,
    lineage: {
      sourceKind: "chunk",
      sourceId: chunk.chunkId,
      sourceRef: chunk.sourceRef,
    },
  });
}

function buildSupportRefChunkRows(params: {
  store: MemxStoreBundle;
  ctx: MemoryOperationContext;
  sourceRefs: string[];
  queryAnchors: string[];
  limit: number;
}): EvidenceRow[] {
  const chunks = params.store.chunkRepo.listBySourceRefs({
    agentId: params.ctx.agentId,
    scopes: params.ctx.scopes,
    sourceRefs: params.sourceRefs,
    limit: Math.max(params.limit * 3, 8),
  });
  return dedupeEvidenceRows(
    chunks
      .map((chunk) => ({
        row: chunkEvidenceRowFromConversationChunk(chunk, params.queryAnchors),
        anchorScore:
          params.queryAnchors.length > 0
            ? queryAnchorSupport(chunk.content, params.queryAnchors)
            : 1,
      }))
      .filter((entry) => (params.queryAnchors.length > 0 ? entry.anchorScore >= 0.22 : true))
      .sort((left, right) => {
        if (right.anchorScore !== left.anchorScore) {
          return right.anchorScore - left.anchorScore;
        }
        return (right.row.score ?? 0) - (left.row.score ?? 0);
      })
      .map((entry) => entry.row),
    params.limit,
  );
}

function queryEntityAnchorTerms(queryAnalysis: RecallQueryAnalysis): string[] {
  return uniqueNonEmpty(
    (queryAnalysis.queryEntities ?? [])
      .map((entity) => entity.name)
      .filter((value): value is string => typeof value === "string" && value.trim().length > 0),
  );
}

function queryAttributeAnchorTerms(queryAnalysis: RecallQueryAnalysis): string[] {
  const entityTerms = new Set(queryEntityAnchorTerms(queryAnalysis).map((entry) => normalizeText(entry)));
  const slotTerms = (queryAnalysis.evidencePlan?.slots ?? []).flatMap((slot) => [
    slot.description,
    ...(slot.relationHints ?? []),
    ...(slot.capabilityQueries ?? []),
    ...slot.requiredFields,
  ]);
  const goalTerms = (queryAnalysis.evidenceGoals ?? []).flatMap((goal) => [
    goal.goal,
    ...(goal.positiveQueries ?? []),
    ...(goal.focusAnchors ?? []),
  ]);
  return uniqueNonEmpty([...(queryAnalysis.anchors ?? []), ...slotTerms, ...goalTerms], 24).filter(
    (term) => {
      const normalized = normalizeText(term);
      return (
        normalized &&
        !entityTerms.has(normalized) &&
        !isGenericRequiredField(normalized) &&
        isMeaningfulRecallAnchor(term)
      );
    },
  );
}

function rowMentionsQueryEntity(row: EvidenceRow, entityTerms: string[]): boolean {
  if (entityTerms.length === 0) {
    return true;
  }
  const normalizedText = normalizeText(row.text);
  return entityTerms.some((term) => {
    const normalized = normalizeText(term);
    return (
      normalized.length > 0 &&
      (normalizedText.includes(normalized) ||
        projectNamesMatch(row.text, term) ||
        semanticTextSimilarity(row.text, term) >= 0.46)
    );
  });
}

function rowSupportsQueryAttribute(row: EvidenceRow, attributeTerms: string[]): boolean {
  if (attributeTerms.length === 0) {
    return true;
  }
  const support = Math.max(
    queryAnchorSupport(row.text, attributeTerms),
    ...attributeTerms.map((term) => semanticTextSimilarity(row.text, term)),
  );
  return support >= 0.24;
}

function observedAtMs(row: EvidenceRow): number {
  const parsed = row.observedAt ? Date.parse(row.observedAt) : Number.NaN;
  return Number.isFinite(parsed) ? parsed : 0;
}

function shouldApplyNewerSourceFactGuard(queryAnalysis: RecallQueryAnalysis): boolean {
  const operation = queryAnalysis.evidencePlan?.operation.type;
  return (
    queryAnalysis.queryShape.timeframe === "current" &&
    (queryAnalysis.queryShape.granularity === "exact_detail" ||
      queryAnalysis.answerGranularity === "detail") &&
    (operation === "return_value" ||
      queryAnalysis.answerMode === "attribute_lookup" ||
      queryAnalysis.answerMode === "single_fact")
  );
}

function newerSourceRowsForCurrentFactGuard(params: {
  facts: EvidenceRow[];
  sourceRows: EvidenceRow[];
  queryAnalysis: RecallQueryAnalysis;
}): EvidenceRow[] {
  if (!shouldApplyNewerSourceFactGuard(params.queryAnalysis) || params.facts.length === 0) {
    return [];
  }
  const newestFactObservedAt = Math.max(0, ...params.facts.map((row) => observedAtMs(row)));
  if (newestFactObservedAt <= 0) {
    return [];
  }
  const entityTerms = queryEntityAnchorTerms(params.queryAnalysis);
  const attributeTerms = queryAttributeAnchorTerms(params.queryAnalysis);
  return dedupeEvidenceRows(
    params.sourceRows
      .filter((row) => observedAtMs(row) > newestFactObservedAt)
      .filter((row) => !isQuestionLike(row.text))
      .filter((row) => rowMentionsQueryEntity(row, entityTerms))
      .filter((row) => rowSupportsQueryAttribute(row, attributeTerms))
      .sort((left, right) => observedAtMs(right) - observedAtMs(left) || (right.score ?? 0) - (left.score ?? 0)),
    3,
  );
}

function mergeRecalledChunkSupport(params: {
  supportRows: EvidenceRow[];
  projectedIds: string[];
  projectedTexts: string[];
  limit: number;
}) {
  const merged: Array<{ id: string; text: string }> = [];
  const seen = new Set<string>();
  for (const row of params.supportRows) {
    const sourceId = row.lineage?.sourceId;
    if (!sourceId || seen.has(sourceId)) {
      continue;
    }
    seen.add(sourceId);
    merged.push({
      id: sourceId,
      text: truncateText(row.text, 280),
    });
    if (merged.length >= params.limit) {
      break;
    }
  }
  if (merged.length < params.limit) {
    for (let index = 0; index < params.projectedIds.length; index += 1) {
      const id = params.projectedIds[index];
      if (!id || seen.has(id)) {
        continue;
      }
      seen.add(id);
      merged.push({
        id,
        text: params.projectedTexts[index] ?? "",
      });
      if (merged.length >= params.limit) {
        break;
      }
    }
  }
  return {
    ids: merged.map((entry) => entry.id),
    texts: merged.map((entry) => entry.text),
  };
}

function shouldUseWorkflowMainSurface(queryAnalysis: QueryCompileResult): boolean {
  return queryAnalysis.queryShape.evidenceNeed === "workflow_context";
}

function shouldRetainTasksInMainSurface(queryAnalysis: QueryCompileResult): boolean {
  if (shouldUseWorkflowMainSurface(queryAnalysis)) {
    return true;
  }
  return queryAnalysis.candidateSurfaces.includes("task");
}

function shouldRetainStatesInMainSurface(queryAnalysis: QueryCompileResult): boolean {
  return (
    shouldUseWorkflowMainSurface(queryAnalysis) || queryAnalysis.candidateSurfaces.includes("state")
  );
}

function shouldApplyCandidateAuthorityToMainSurface(
  route: { routeType: string },
  queryAnalysis: QueryCompileResult,
): boolean {
  if (route.routeType === "factual") {
    return true;
  }
  if (route.routeType !== "mixed") {
    return false;
  }
  if (shouldUseWorkflowMainSurface(queryAnalysis)) {
    return false;
  }
  if (
    !queryAnalysis.candidateSurfaces.includes("fact") &&
    !queryAnalysis.candidateSurfaces.includes("state")
  ) {
    return false;
  }
  return (
    (queryAnalysis.routeWeights.factual ?? 0) >= 0.24 &&
    (queryAnalysis.answerGranularity === "detail" || queryAnalysis.evidenceFidelity === "high")
  );
}

function isQuestionLikeTask(task: {
  title: string;
  summary: string;
  metadataJson?: Record<string, unknown>;
}): boolean {
  const candidateResolution =
    typeof task.metadataJson?.candidateResolution === "string" &&
    task.metadataJson.candidateResolution.trim()
      ? task.metadataJson.candidateResolution.trim()
      : "";
  const stableSummary = semanticTaskSummaryText(task) ?? "";
  return (
    isQuestionLike(task.title) ||
    isQuestionLike(candidateResolution) ||
    isQuestionLike(stableSummary) ||
    isQuestionLike(taskMetadataValue(task, "currentTask"))
  );
}

function resolveChunkMentionedProject(
  chunk: { content: string; summary: string },
  knownProjects: string[],
): string | undefined {
  for (const candidate of knownProjects) {
    if (!candidate) {
      continue;
    }
    if (chunk.content.includes(candidate) || chunk.summary.includes(candidate)) {
      return candidate;
    }
  }
  return undefined;
}

function recentProjectMentionSequence(params: {
  chunks: Array<{ content: string; summary: string }>;
  knownProjects: string[];
}): string[] {
  const mentions: string[] = [];
  for (const chunk of params.chunks) {
    const project = resolveChunkMentionedProject(chunk, params.knownProjects);
    if (!project) {
      continue;
    }
    if (mentions.length > 0 && projectNamesMatch(mentions.at(-1)!, project)) {
      continue;
    }
    mentions.push(project);
  }
  return uniqueNonEmpty(mentions);
}

function isMeaningfulRecallAnchor(anchor: string): boolean {
  const trimmed = anchor.trim();
  const normalized = normalizeText(trimmed);
  if (!normalized) {
    return false;
  }
  if ([...trimmed].some((char) => /\p{Script=Han}/u.test(char))) {
    return trimmed.length >= 2;
  }
  return normalized.length >= 4;
}

function appendAnchorsToQuery(query: string, anchors: string[]): string {
  const missing = anchors.filter((anchor) => !normalizeText(query).includes(normalizeText(anchor)));
  return missing.length > 0 ? `${query} ${missing.join(" ")}`.trim() : query;
}

function resolveReferentialQueryAnchors(
  store: MemxStoreBundle,
  ctx: MemoryOperationContext,
  query: string,
  queryAnalysis?: Pick<QueryCompileResult, "contextExclusions">,
): { anchors: string[]; reasons: string[] } {
  const reasons: string[] = [];
  const recentTasks = uniqueTasksById(
    [
      ...store.taskRepo.listRecent({
        agentId: ctx.agentId,
        scopes: ctx.scopes,
        limit: 10,
        sessionKey: ctx.sessionKey,
      }),
      ...store.taskRepo.listRecent({
        agentId: ctx.agentId,
        scopes: ctx.scopes,
        limit: 10,
      }),
    ].sort((left, right) => right.updatedAt.localeCompare(left.updatedAt)),
  ).filter((task) => !isQuestionLikeTask(task));
  const recentChunks = store.chunkRepo.listRecentActive({
    agentId: ctx.agentId,
    scopes: ctx.scopes,
    limit: 24,
    sessionKey: ctx.sessionKey,
  });
  const activeProjectState = store.stateRepo.get({
    agentId: ctx.agentId,
    scopes: ctx.scopes,
    key: "project.active_project",
    now: ctx.now,
    readEpoch: ctx.readEpoch,
  })[0];
  const authoritativeActiveProject =
    typeof activeProjectState?.valueJson.project === "string" &&
    activeProjectState.valueJson.project.trim()
      ? activeProjectState.valueJson.project.trim()
      : "";
  const projectCandidates = uniqueNonEmpty([
    authoritativeActiveProject,
    ...recentTasks.map((task) => taskProject(task)),
  ]);
  const projectMentionSequence = recentProjectMentionSequence({
    chunks: recentChunks.filter((chunk) => chunk.role === "user"),
    knownProjects: projectCandidates,
  });
  const activeTasks = recentTasks.filter((task) => task.status === "active");
  const activeTask = activeTasks[0] ?? recentTasks[0];
  const activeProject = authoritativeActiveProject || (activeTask ? taskProject(activeTask) : "");
  const anchors: string[] = [];

  const excludesPriorProject = (queryAnalysis?.contextExclusions ?? []).some(
    (exclusion) => exclusion.kind === "prior_project" || exclusion.kind === "prior_context",
  );
  if (!excludesPriorProject && /(?:前一个项目|上一个项目|previous project|last project)/iu.test(query)) {
    const previousProject =
      projectMentionSequence[1] ??
      projectCandidates.find(
        (candidate) => normalizeText(candidate) !== normalizeText(activeProject),
      ) ??
      projectCandidates[1] ??
      "";
    if (previousProject) {
      anchors.push(previousProject);
      reasons.push(`resolved-previous-project:${previousProject}`);
    }
  }

  const sideHints = [...query.matchAll(/([\p{Script=Han}\p{L}\p{N}_-]{1,12})那边/gu)]
    .map((match) => match[1]?.trim() ?? "")
    .filter(
      (hint) =>
        hint &&
        hint !== "这" &&
        hint !== "那" &&
        hint !== "那个" &&
        hint !== "这个" &&
        hint !== "项目" &&
        hint !== "前一个",
    );
  for (const hint of sideHints) {
    const ranked = recentTasks
      .map((task) => {
        const references = uniqueNonEmpty([
          taskProject(task),
          taskMetadataValue(task, "currentTask"),
          task.title,
          typeof task.metadataJson?.candidateResolution === "string"
            ? task.metadataJson.candidateResolution
            : "",
          semanticTaskSummaryText(task) ?? "",
        ]);
        const score = references.reduce(
          (best, reference) =>
            Math.max(
              best,
              reference.includes(hint) || hint.includes(reference)
                ? 0.96
                : queryAnchorSupport(reference, [hint]),
            ),
          0,
        );
        return {
          task,
          references,
          score,
        };
      })
      .filter((entry) => entry.score >= 0.2)
      .sort((left, right) => right.score - left.score);
    const best = ranked[0];
    const second = ranked[1];
    let selectedTask: ConversationTask | undefined = best?.task;
    let selectedReason = best ? `resolved-side-reference:${hint}` : "";
    if (
      !best ||
      (best.score < 0.3 && second && best.score - second.score < 0.08) ||
      best.score < 0.2
    ) {
      selectedTask = activeTasks.length === 1 ? activeTasks[0]! : undefined;
      selectedReason = `resolved-side-reference:fallback-active:${hint}`;
    }
    if (!selectedTask) {
      continue;
    }
    const selected = uniqueNonEmpty([
      taskProject(selectedTask),
      taskMetadataValue(selectedTask, "currentTask"),
      selectedTask.title,
      selectedTask.summary,
    ])[0];
    if (!selected) {
      continue;
    }
    anchors.push(selected);
    reasons.push(`${selectedReason}->${selected}`);
  }

  return {
    anchors: uniqueNonEmpty(anchors),
    reasons,
  };
}

function uniqueTasksById<
  T extends {
    taskId: string;
  },
>(tasks: T[]): T[] {
  const seen = new Set<string>();
  const ordered: T[] = [];
  for (const task of tasks) {
    if (seen.has(task.taskId)) {
      continue;
    }
    seen.add(task.taskId);
    ordered.push(task);
  }
  return ordered;
}

function groundingForRouteEntry(
  routeType: MemoryPrimaryRouteType,
  topEntry?: ReturnType<typeof topProbeSupportForRoute>["topEntry"],
): number {
  if (!topEntry) {
    return 0;
  }
  const kind = topEntry.object.kind;
  const profile = topEntry.object.profile;
  const belief = topEntry.object.belief;
  let base =
    routeType === "workflow"
      ? kind === "task" || kind === "state"
        ? 0.88
        : 0.54
      : routeType === "factual"
        ? kind === "fact"
          ? 0.84
          : 0.48
        : routeType === "temporal"
          ? kind === "event" || kind === "chunk"
            ? 0.86
            : 0.5
          : kind === "graph_path"
            ? 0.92
            : kind === "fact" && topEntry.object.attributes.relational
              ? 0.76
              : 0.52;
  if (topEntry.object.attributes.activeTask) {
    base += 0.04;
  }
  if (belief?.stage === "active") {
    base += 0.04;
  }
  base += (topEntry.graphSupport ?? 0) * 0.08;
  base += profile.connectivity * 0.04;
  return clamp01(base);
}

function candidateCountForRoute(
  scheduled: ReturnType<typeof collectAndScheduleMemoryObjects>,
  topEntry?: ReturnType<typeof topProbeSupportForRoute>["topEntry"],
): number {
  const threshold = Math.max(0.45, (topEntry?.objectiveScore ?? 0) * 0.6);
  return scheduled.filter((entry) => entry.objectiveScore >= threshold).length;
}

async function buildProbeSurface(
  store: MemxStoreBundle,
  ctx: MemoryOperationContext,
  query: string,
  searchQuery: string,
  background: BackgroundRecallBundle,
): Promise<ProbeSurface> {
  const thresholds = recallProbeThresholds(ctx.config);
  const trimmed = query.trim();
  const rawProbeHits = await store.retrievalBackend.hybridSearch({
    agentId: ctx.agentId,
    scopes: ctx.scopes,
    query: searchQuery,
    limit: 6,
    readEpoch: ctx.readEpoch,
  });
  const probeHits = rawProbeHits.filter((hit) => !shouldSuppressRecallText(hit.text));

  const workflowObjective = selectionObjectiveForRoute(
    "workflow",
    searchQuery,
    ctx.now,
    ctx.sessionKey,
  );
  const factualObjective = selectionObjectiveForRoute(
    "factual",
    searchQuery,
    ctx.now,
    ctx.sessionKey,
  );
  const temporalObjective = selectionObjectiveForRoute(
    "temporal",
    searchQuery,
    ctx.now,
    ctx.sessionKey,
  );
  const explanatoryObjective = selectionObjectiveForRoute(
    "explanatory",
    searchQuery,
    ctx.now,
    ctx.sessionKey,
  );

  // Optimization: all 4 routes use the same query and probeHits, so DB data
  // (states, tasks, facts, events, graph) is identical.  Collect once with
  // the most inclusive objective (temporal gets broadTemporal + 8 events,
  // factual gets includeHistorical).  Per-route scoring happens in
  // scheduleMemoryObjects → objectiveScoreForObject.
  const supersetObjective = selectionObjectiveForRoute(
    "temporal",
    searchQuery,
    ctx.now,
    ctx.sessionKey,
  );
  supersetObjective.includeHistorical = factualObjective.includeHistorical;
  const probeObjects = collectMemoryObjects(store, ctx, supersetObjective, probeHits);

  const workflowScheduled = scheduleMemoryObjects(probeObjects, workflowObjective);
  const factualScheduled = scheduleMemoryObjects(probeObjects, factualObjective);
  const temporalScheduled = scheduleMemoryObjects(probeObjects, temporalObjective);
  const explanatoryScheduled = scheduleMemoryObjects(probeObjects, explanatoryObjective);

  const taskAssociation = topScheduledMemoryScore(
    workflowScheduled,
    (entry) => entry.object.kind === "task",
  );
  const stateAssociation = topScheduledMemoryScore(
    workflowScheduled,
    (entry) => entry.object.kind === "state",
  );
  const workflowProbe = topProbeSupportForRoute(workflowScheduled, "workflow");
  const factualProbe = topProbeSupportForRoute(factualScheduled, "factual");
  const temporalProbe = topProbeSupportForRoute(temporalScheduled, "temporal");
  const explanatoryProbe = topProbeSupportForRoute(explanatoryScheduled, "explanatory");
  const workflowSimilarity = workflowProbe.support;
  const factualSimilarity = factualProbe.support;
  const temporalSimilarity = temporalProbe.support;
  const explanatorySimilarity = explanatoryProbe.support;
  const workflowContinuity = clamp01(
    taskAssociation * 0.62 + stateAssociation * 0.26 + workflowSimilarity * 0.12,
  );
  const contextAssociation = computeContextAssociation({
    taskAssociation,
    stateAssociation,
    topProbeScore: Math.max(
      workflowSimilarity,
      factualSimilarity,
      temporalSimilarity,
      explanatorySimilarity,
    ),
  });
  const workflowContextSupport = clamp01(
    workflowSimilarity * 0.72 + contextAssociation * 0.2 + workflowContinuity * 0.08,
  );
  const explanatoryStructureStrength = explanatoryProbe.topEntry
    ? Math.max(
        explanatoryProbe.topEntry.object.profile.relationDensity,
        explanatoryProbe.topEntry.object.profile.connectivity,
      )
    : 0;
  let explanatorySupport = explanatorySimilarity;
  if (
    explanatoryProbe.topEntry &&
    factualProbe.topEntry &&
    explanatoryProbe.topEntry.object.objectId === factualProbe.topEntry.object.objectId &&
    explanatoryStructureStrength >= 0.45 &&
    explanatorySupport + 0.05 >= factualSimilarity
  ) {
    explanatorySupport = clamp01(Math.max(explanatorySupport, factualSimilarity + 0.01));
  }
  const factualSupport = factualSimilarity;
  const temporalSupport = temporalSimilarity;
  const routedExplanatorySupport = explanatorySupport;
  const routeSurfaces: ProbeRouteSurface[] = [
    {
      route: "workflow",
      scheduled: workflowScheduled,
      support: {
        support: workflowContextSupport,
        topEntry: workflowProbe.topEntry,
      },
    },
    {
      route: "factual",
      scheduled: factualScheduled,
      support: {
        support: factualSupport,
        topEntry: factualProbe.topEntry,
      },
    },
    {
      route: "temporal",
      scheduled: temporalScheduled,
      support: {
        support: temporalSupport,
        topEntry: temporalProbe.topEntry,
      },
    },
    {
      route: "explanatory",
      scheduled: explanatoryScheduled,
      support: {
        support: routedExplanatorySupport,
        topEntry: explanatoryProbe.topEntry,
      },
    },
  ];
  const strongestRouteCandidate = [...routeSurfaces]
    .map((entry) => ({
      route: entry.route,
      support: entry.support.support,
      topEntry: entry.support.topEntry,
    }))
    .sort((left, right) => {
      if (right.support !== left.support) {
        return right.support - left.support;
      }
      const rightObjective = right.topEntry?.objectiveScore ?? 0;
      const leftObjective = left.topEntry?.objectiveScore ?? 0;
      return rightObjective - leftObjective;
    })[0];

  return {
    probeHits,
    workflowContinuity,
    taskAssociation,
    stateAssociation,
    contextAssociation,
    workflowContextSupport,
    factualSupport,
    temporalSupport,
    explanatorySupport: routedExplanatorySupport,
    routeSurfaces,
    strongestRouteCandidate,
    topProbeScore: clamp01(strongestRouteCandidate?.support ?? 0),
  };
}

function computeContextAssociation(params: {
  taskAssociation: number;
  stateAssociation: number;
  topProbeScore: number;
}): number {
  return clamp01(
    params.taskAssociation * 0.58 + params.stateAssociation * 0.24 + params.topProbeScore * 0.18,
  );
}

function routeModerateThreshold(
  route: MemoryPrimaryRouteType,
  thresholds: RecallProbeThresholds,
): number {
  switch (route) {
    case "workflow":
      return thresholds.workflowContinuation;
    case "factual":
      return thresholds.factualShortQuery;
    case "temporal":
    case "explanatory":
      return thresholds.hybridModerate;
  }
}

function stripStructuredLabel(text: string): string {
  const trimmed = text.trim();
  const separator = trimmed.indexOf(":");
  if (separator <= 1 || separator > 48) {
    return trimmed;
  }
  const value = trimmed.slice(separator + 1).trim();
  return value || trimmed;
}

function uniqueFocusedFragments(fragments: string[], limit: number): string[] {
  const seen = new Set<string>();
  const result: string[] = [];
  for (const fragment of fragments) {
    const trimmed = fragment.trim();
    if (!trimmed) {
      continue;
    }
    const key = normalizeSearchText(trimmed).toLowerCase();
    if (!key || seen.has(key)) {
      continue;
    }
    seen.add(key);
    result.push(trimmed);
    if (result.length >= limit) {
      break;
    }
  }
  return result;
}

function focusedStateKeyFragment(stateKey?: string): string | undefined {
  switch (stateKey) {
    case "workflow.current_task":
      return "current task";
    case "workflow.next_action":
      return "next action";
    case "workflow.blocker":
      return "workflow blocker";
    case "project.active_project":
      return "active project";
    default:
      return stateKey ? stateKey.replace(/[._]+/gu, " ") : undefined;
  }
}

function buildSummaryEvidenceFragments(
  routeType: MemoryPrimaryRouteType,
  summary: ShallowRecallResult["routeSummaries"][MemoryPrimaryRouteType],
): string[] {
  const fragments: string[] = [];
  const strippedTopText = summary.topText ? stripStructuredLabel(summary.topText) : "";
  if (routeType === "workflow") {
    const stateFragment = focusedStateKeyFragment(summary.topStateKey);
    if (stateFragment) {
      fragments.push(stateFragment);
    }
    if (strippedTopText) {
      fragments.push(strippedTopText);
    }
    return uniqueFocusedFragments(fragments, 3);
  }

  if (routeType === "factual") {
    const factFragment = [summary.topFactSubject, summary.topFactPredicate, summary.topFactObject]
      .filter(Boolean)
      .join(" ");
    if (factFragment) {
      fragments.push(factFragment);
    }
    if (strippedTopText) {
      fragments.push(strippedTopText);
    }
    return uniqueFocusedFragments(fragments, 3);
  }

  if (routeType === "temporal") {
    if (strippedTopText) {
      fragments.push(strippedTopText);
    }
    return uniqueFocusedFragments(fragments, 2);
  }

  if (summary.topGraphNodeNames?.length) {
    fragments.push(summary.topGraphNodeNames.slice(0, 4).join(" "));
  }
  const factFragment = [summary.topFactSubject, summary.topFactPredicate, summary.topFactObject]
    .filter(Boolean)
    .join(" ");
  if (factFragment) {
    fragments.push(factFragment);
  }
  if (strippedTopText) {
    fragments.push(strippedTopText);
  }
  return uniqueFocusedFragments(fragments, 3);
}

function buildProjectionContextFragments(
  background: BackgroundRecallBundle,
  routeType: MemoryPrimaryRouteType,
): string[] {
  return uniqueFocusedFragments(
    background.projectionBlocks
      .filter((block) => projectionRolesForRoute(routeType).includes(block.role))
      .flatMap((block) => block.lines.map(stripStructuredLabel)),
    2,
  );
}

function buildGateFocusedQuery(params: {
  routeType: MemoryPrimaryRouteType;
  query: string;
  searchQuery: string;
  background: BackgroundRecallBundle;
  plan: MemoryRecallPlan;
  probe: RecallProbeDecision;
  shallow: ShallowRecallResult;
}): string {
  const topSummary = params.shallow.routeSummaries[params.routeType];
  const originalSearchQuery = params.searchQuery || params.query;
  const baseQuery = params.probe.focusedQuery || params.plan.focusedQuery || params.searchQuery;
  const evidenceFragments = buildSummaryEvidenceFragments(params.routeType, topSummary);
  if (evidenceFragments.length === 0) {
    return truncateText(originalSearchQuery, 160);
  }
  const projectionContext = buildProjectionContextFragments(params.background, params.routeType);
  const evidenceContext = uniqueFocusedFragments(
    [...evidenceFragments, ...projectionContext],
    params.routeType === "explanatory" ? 4 : 3,
  ).join(" ");
  const includeBaseQuery = params.routeType === "temporal" && evidenceContext.length < 120;
  const composite = [evidenceContext, includeBaseQuery ? baseQuery || params.query : ""]
    .filter(Boolean)
    .join(" ");
  return truncateText(composite || params.query, 160);
}

export async function runShallowRecall(
  store: MemxStoreBundle,
  ctx: MemoryOperationContext,
  query: string,
  searchQuery = query,
  background = buildBackgroundRecallBundle(store, ctx),
): Promise<ShallowRecallResult> {
  const trimmed = query.trim();
  if (!trimmed) {
    return {
      searchQuery,
      topSupport: 0,
      hybridHitCount: 0,
      projectionRoles: background.projectionBlocks.map((block) => block.role),
      routeSummaries: Object.fromEntries(
        PRIMARY_ROUTE_TYPES.map((routeType) => [
          routeType,
          {
            routeType,
            support: 0,
            candidateCount: 0,
            projectionSupport: 0,
            freshness: 0,
            contradictionPressure: 0,
            grounding: 0,
          },
        ]),
      ) as Record<
        MemoryPrimaryRouteType,
        ShallowRecallResult["routeSummaries"][MemoryPrimaryRouteType]
      >,
      reasons: ["shallow:empty-query"],
    };
  }

  const surface = await buildProbeSurface(store, ctx, trimmed, searchQuery, background);
  const routeSummaries = Object.fromEntries(
    surface.routeSurfaces.map((entry) => {
      const projectionSupport = projectionSupportForRoute(trimmed, background, entry.route);
      const projectionWeight =
        entry.route === "workflow"
          ? 0.92
          : entry.route === "factual"
            ? 0.8
            : entry.route === "explanatory"
              ? 0.78
              : 0.72;
      let effectiveSupport = Math.max(entry.support.support, projectionSupport * projectionWeight);
      if (entry.route === "workflow" && surface.workflowContinuity >= 0.42) {
        effectiveSupport = clamp01(
          effectiveSupport + surface.contextAssociation * 0.08 + surface.workflowContinuity * 0.06,
        );
      }
      if (
        entry.route === "workflow" &&
        projectionSupport >= 0.45 &&
        background.projectionBlocks.some(
          (block) => block.role === "active_task" || block.role === "active_blocker",
        )
      ) {
        effectiveSupport = clamp01(effectiveSupport + 0.05);
      }
      const topEntry = entry.support.topEntry;
      return [
        entry.route,
        {
          routeType: entry.route,
          support: effectiveSupport,
          candidateCount: candidateCountForRoute(entry.scheduled, topEntry),
          projectionSupport,
          freshness: topEntry?.object.profile.recencyScore ?? 0,
          contradictionPressure: clamp01(
            Math.max(topEntry?.object.belief?.contradictionScore ?? 0, topEntry?.graphPenalty ?? 0),
          ),
          grounding: Math.max(
            groundingForRouteEntry(entry.route, topEntry),
            projectionSupport * 0.88,
          ),
          topKind: topEntry?.object.kind,
          topObjectId: topEntry?.object.objectId,
          topText: topEntry?.object.row.text,
          topStateKey: topEntry?.object.attributes.stateKey,
          topFactSubject: topEntry?.object.attributes.factSubject,
          topFactPredicate: topEntry?.object.attributes.factPredicate,
          topFactObject: topEntry?.object.attributes.factObject,
          topGraphNodeNames: topEntry?.object.graphNodes?.map((node) => node.name).filter(Boolean),
        } satisfies ShallowRecallResult["routeSummaries"][MemoryPrimaryRouteType],
      ];
    }),
  ) as Record<
    MemoryPrimaryRouteType,
    ShallowRecallResult["routeSummaries"][MemoryPrimaryRouteType]
  >;
  const strongest = [...PRIMARY_ROUTE_TYPES]
    .map((routeType) => routeSummaries[routeType])
    .sort((left, right) => right.support - left.support)[0];

  return {
    searchQuery,
    routeHint: strongest && strongest.support > 0 ? strongest.routeType : undefined,
    topSupport: strongest?.support ?? 0,
    hybridHitCount: surface.probeHits.length,
    projectionRoles: background.projectionBlocks.map((block) => block.role),
    routeSummaries,
    reasons: [
      `shallow:workflow-continuity:${surface.workflowContinuity.toFixed(2)}`,
      `shallow:hits:${surface.probeHits.length}`,
      `shallow:top:${(strongest?.support ?? 0).toFixed(2)}`,
    ],
  };
}

export function evaluateRecallQualityGate(params: {
  query: string;
  searchQuery: string;
  background: BackgroundRecallBundle;
  plan: MemoryRecallPlan;
  probe: RecallProbeDecision;
  controller: RecallControllerTrace;
  shallow: ShallowRecallResult;
}): RecallQualityGateDecision {
  const topRoute = isPrimaryRouteType(params.shallow.routeHint)
    ? params.shallow.routeHint
    : undefined;
  const planJudgmentUnavailable =
    params.plan.judgmentMode === "degraded" || params.plan.judgmentMode === "disabled";
  if (!topRoute) {
    return {
      decision: "downgrade_background",
      focusedQuery: params.searchQuery,
      confidence: 0.94,
      reasons: ["quality:no-route-grounding"],
      metrics: {
        supportDensity: 0,
        routeAgreement: 0,
        contradictionPressure: 0,
        freshness: 0,
        grounding: 0,
        irrelevantSpill: 1,
      },
    };
  }

  const topSummary = params.shallow.routeSummaries[topRoute];
  const routeVotes = [
    params.controller.routeHint,
    params.plan.routeHint,
    params.probe.hintedRoute,
  ].filter(isPrimaryRouteType);
  const routeAgreement =
    routeVotes.length > 0
      ? routeVotes.filter((route) => route === topRoute).length / routeVotes.length
      : clamp01(topSummary.support * 0.75 + topSummary.projectionSupport * 0.25);
  const supportDensity = clamp01(
    topSummary.support * 0.56 +
      Math.min(1, topSummary.candidateCount / 3) * 0.14 +
      topSummary.projectionSupport * 0.1 +
      topSummary.grounding * 0.12 +
      topSummary.freshness * 0.08,
  );
  const contradictionPressure = topSummary.contradictionPressure;
  const freshness = topSummary.freshness;
  const grounding = topSummary.grounding;
  const freshnessFloor =
    topRoute === "workflow"
      ? 0.22
      : topRoute === "factual"
        ? 0.05
        : topRoute === "temporal"
          ? 0.06
          : 0.08;
  const acceptSupportDensityFloor = planJudgmentUnavailable ? 0.5 : 0.6;
  const acceptRouteSupportFloor = planJudgmentUnavailable ? 0.52 : 0.58;
  const downgradeSupportFloor = planJudgmentUnavailable ? 0.34 : 0.42;
  const groundingFloor = planJudgmentUnavailable ? 0.28 : 0.34;
  const coveragePenalty =
    params.shallow.hybridHitCount > 0
      ? 1 -
        clamp01(topSummary.candidateCount / Math.max(1, Math.min(params.shallow.hybridHitCount, 4)))
      : topSummary.projectionSupport >= 0.45
        ? 0.08
        : 0.42;
  const irrelevantSpill = clamp01(
    coveragePenalty * 0.5 +
      (1 - routeAgreement) * 0.2 +
      (1 - grounding) * 0.15 +
      contradictionPressure * 0.15,
  );
  const focusedQuery = buildGateFocusedQuery({
    routeType: topRoute,
    query: params.query,
    searchQuery: params.searchQuery,
    background: params.background,
    plan: params.plan,
    probe: params.probe,
    shallow: params.shallow,
  });
  const baseDecision = {
    routeHint: topRoute,
    focusedQuery,
    metrics: {
      supportDensity,
      routeAgreement,
      contradictionPressure,
      freshness,
      grounding,
      irrelevantSpill,
    },
  };

  if (
    !planJudgmentUnavailable &&
    params.controller.needLevel === "background_only" &&
    routeAgreement < 0.5 &&
    topSummary.projectionSupport >= 0.35
  ) {
    return {
      decision: "downgrade_background",
      confidence: clamp01(0.72 + topSummary.projectionSupport * 0.12),
      reasons: ["quality:background-route-divergence"],
      ...baseDecision,
    };
  }

  if (
    supportDensity < downgradeSupportFloor ||
    contradictionPressure >= 0.82 ||
    irrelevantSpill >= 0.76 ||
    (grounding < groundingFloor && topSummary.support < Math.max(acceptRouteSupportFloor, 0.6))
  ) {
    return {
      decision: "downgrade_background",
      confidence: clamp01(0.62 + Math.max(irrelevantSpill, contradictionPressure) * 0.25),
      reasons: ["quality:downgrade-background"],
      ...baseDecision,
    };
  }

  if (
    routeVotes.length > 0 &&
    routeAgreement < 0.45 &&
    topSummary.support >= (planJudgmentUnavailable ? 0.56 : 0.62) &&
    routeVotes.some((route) => route !== topRoute)
  ) {
    return {
      decision: "accept_full",
      confidence: clamp01(topSummary.support * 0.7 + (1 - routeAgreement) * 0.16),
      reasons: ["quality:route-mismatch-accept-full"],
      ...baseDecision,
    };
  }

  if (
    supportDensity < acceptSupportDensityFloor ||
    topSummary.support < acceptRouteSupportFloor ||
    freshness < freshnessFloor ||
    (params.shallow.hybridHitCount > 0 && topSummary.candidateCount === 0)
  ) {
    return {
      decision: "retry_shallow",
      confidence: clamp01(0.48 + supportDensity * 0.3),
      reasons: ["quality:retry-shallow"],
      ...baseDecision,
    };
  }

  return {
    decision: "accept_full",
    confidence: clamp01(
      supportDensity * 0.42 +
        routeAgreement * 0.18 +
        freshness * 0.1 +
        grounding * 0.18 +
        (1 - contradictionPressure) * 0.12,
    ),
    reasons: ["quality:accept-full"],
    ...baseDecision,
  };
}

export async function runRecallProbe(
  store: MemxStoreBundle,
  ctx: MemoryOperationContext,
  query: string,
  searchQuery = query,
  background = buildBackgroundRecallBundle(store, ctx),
): Promise<RecallProbeDecision> {
  const thresholds = recallProbeThresholds(ctx.config);
  const trimmed = query.trim();
  if (!trimmed) {
    return {
      shouldEscalate: false,
      probeScore: 0,
      reasons: ["probe:empty-query"],
      focusedQuery: searchQuery,
      signals: {
        workflowContinuity: 0,
        taskAssociation: 0,
        stateAssociation: 0,
        contextAssociation: 0,
        workflowSimilarity: 0,
        factualSimilarity: 0,
        temporalSimilarity: 0,
        explanatorySimilarity: 0,
        topProbeScore: 0,
        backgroundTaskCount: background.tasks.length,
        backgroundStateCount: background.states.length,
        factualCandidateCount: 0,
        hybridHitCount: 0,
      },
      thresholds,
    };
  }

  const surface = await buildProbeSurface(store, ctx, trimmed, searchQuery, background);
  const strongestRouteCandidate = surface.strongestRouteCandidate;
  const strongestProbeEntry = strongestRouteCandidate?.topEntry;
  const topProbeScore = surface.topProbeScore;
  const signals: RecallProbeSignals = {
    workflowContinuity: surface.workflowContinuity,
    taskAssociation: surface.taskAssociation,
    stateAssociation: surface.stateAssociation,
    contextAssociation: surface.contextAssociation,
    workflowSimilarity: surface.workflowContextSupport,
    factualSimilarity: surface.factualSupport,
    temporalSimilarity: surface.temporalSupport,
    explanatorySimilarity: surface.explanatorySupport,
    topProbeScore,
    topProbeDocType: strongestProbeEntry?.object.kind,
    backgroundTaskCount: background.tasks.length,
    backgroundStateCount: background.states.length,
    factualCandidateCount:
      surface.routeSurfaces
        .find((entry) => entry.route === "factual")
        ?.scheduled.filter((entry) => entry.object.kind === "fact").length ?? 0,
    hybridHitCount: surface.probeHits.length,
  };

  let probeScore = 0;
  let hintedRoute: MemoryRouteType | undefined;
  const reasons: string[] = [];
  const routeHintFloor = 0.55;
  if (surface.workflowContextSupport >= thresholds.workflowContinuation) {
    reasons.push(
      `probe:workflow-object:${surface.workflowContextSupport.toFixed(2)}/${surface.taskAssociation.toFixed(2)}/${surface.stateAssociation.toFixed(2)}`,
    );
  }
  if (surface.workflowContinuity >= thresholds.workflowContinuation * 0.75) {
    reasons.push(`probe:workflow-continuity:${surface.workflowContinuity.toFixed(2)}`);
  }
  if (
    surface.factualSupport >=
    Math.min(routeModerateThreshold("factual", thresholds), routeHintFloor)
  ) {
    reasons.push(`probe:factual-object:${surface.factualSupport.toFixed(2)}`);
  }
  if (
    surface.temporalSupport >=
    Math.min(routeModerateThreshold("temporal", thresholds), routeHintFloor)
  ) {
    reasons.push(`probe:temporal-object:${surface.temporalSupport.toFixed(2)}`);
  }
  if (
    surface.explanatorySupport >=
    Math.min(routeModerateThreshold("explanatory", thresholds), routeHintFloor)
  ) {
    const explanatoryTopEntry = surface.routeSurfaces.find((entry) => entry.route === "explanatory")
      ?.support.topEntry;
    reasons.push(
      `probe:explanatory-object:${surface.explanatorySupport.toFixed(2)}/${Math.max(
        explanatoryTopEntry?.object.profile.relationDensity ?? 0,
        explanatoryTopEntry?.object.profile.connectivity ?? 0,
      ).toFixed(2)}`,
    );
  }
  if (topProbeScore >= thresholds.hybridModerate) {
    reasons.push(`probe:hybrid-top:${topProbeScore.toFixed(2)}`);
  }

  probeScore = topProbeScore;
  if (
    strongestRouteCandidate &&
    strongestRouteCandidate.support >=
      Math.min(routeModerateThreshold(strongestRouteCandidate.route, thresholds), routeHintFloor)
  ) {
    hintedRoute = strongestRouteCandidate.route;
  } else {
    hintedRoute = strongestProbeEntry
      ? routeHintFromMemoryObjectKind(strongestProbeEntry.object.kind)
      : undefined;
  }

  const shouldEscalate =
    probeScore >= thresholds.escalate ||
    surface.workflowContextSupport >= thresholds.workflowStrong ||
    (probeScore >= thresholds.continuationEscalate &&
      surface.workflowContinuity >= thresholds.workflowContinuation * 0.75);
  const focusedQuery =
    hintedRoute === "workflow" && background.tasks[0] && surface.workflowContinuity >= 0.45
      ? truncateText(`${background.tasks[0].text} ${searchQuery}`.trim(), 160)
      : searchQuery;

  return {
    shouldEscalate,
    probeScore,
    reasons: reasons.length > 0 ? reasons : ["probe:no-strong-evidence"],
    focusedQuery: focusedQuery || searchQuery,
    hintedRoute,
    signals,
    thresholds,
  };
}

function buildFocusedQueries(
  queryAnalysis: RecallQueryAnalysis,
  originalQuery: string,
  searchQuery: string,
): Record<MemoryPrimaryRouteType, string> {
  const normalizedFallback = sanitizeFocusedRecallQuery(
    originalQuery,
    queryAnalysis.focusedQuery || searchQuery,
  );
  return {
    workflow: normalizedFallback,
    factual: normalizedFallback,
    temporal: normalizedFallback,
    explanatory: normalizedFallback,
  };
}

function isSnapshotFactualCandidateSummary(summary: string): boolean {
  const { label } = splitLabelValue(summary);
  return Boolean(label && isSnapshotFactualStateKey(label));
}

function routeRoleFit(
  routeType: MemoryPrimaryRouteType,
  role: RouteEvidenceCandidate["role"],
  summary: string,
): number {
  switch (routeType) {
    case "workflow":
      return role === "state" || role === "task" ? 1 : role === "event" ? 0.72 : 0.42;
    case "factual":
      return role === "fact"
        ? 1
        : role === "state" && isSnapshotFactualCandidateSummary(summary)
          ? 0.94
          : role === "chunk"
            ? 0.66
            : role === "graph"
              ? 0.6
              : 0.4;
    case "temporal":
      return role === "event" || role === "chunk" ? 1 : role === "state" ? 0.64 : 0.4;
    case "explanatory":
      return role === "graph" ? 1 : role === "fact" ? 0.84 : role === "chunk" ? 0.68 : 0.42;
  }
}

function buildScoreDrivenRouteEvidenceDecision(
  routeType: MemoryPrimaryRouteType,
  candidates: RouteEvidenceCandidate[],
  judgmentMode: "degraded" | "disabled" | "score-driven",
): RouteEvidenceDecision {
  if (candidates.length === 0) {
    return {
      relevant: [],
      sufficient: false,
      support: 0,
      reason: `surface ${routeType} evidence: no candidates`,
      judgmentMode,
    };
  }

  const weighted = candidates.map((candidate) => {
    const fit = routeRoleFit(routeType, candidate.role, candidate.summary);
    const confidence =
      typeof candidate.confidence === "number" ? clamp01(candidate.confidence) : 0.6;
    const weightedScore = clamp01(
      candidate.score * (0.72 + fit * 0.28) * (0.88 + confidence * 0.12),
    );
    return {
      candidate,
      fit,
      weightedScore,
      confidence,
    };
  });
  const ordered = [...weighted].sort((left, right) => right.weightedScore - left.weightedScore);
  const top = ordered[0]?.weightedScore ?? 0;
  const topThree = ordered.slice(0, 3);
  const averageTop =
    topThree.reduce((sum, entry) => sum + entry.weightedScore, 0) / Math.max(topThree.length, 1);
  const roleFit =
    topThree.reduce((sum, entry) => sum + entry.fit, 0) / Math.max(topThree.length, 1);
  const confidence =
    topThree.reduce((sum, entry) => sum + entry.confidence, 0) / Math.max(topThree.length, 1);
  const coverage = Math.min(1, ordered.length / 3);
  const aggregateSupport = clamp01(
    top * 0.54 + averageTop * 0.2 + roleFit * 0.12 + confidence * 0.08 + coverage * 0.06,
  );
  const topWeight =
    routeType === "temporal" || routeType === "explanatory"
      ? 0.94
      : routeType === "workflow"
        ? 0.9
        : 0.88;
  const support = clamp01(Math.max(aggregateSupport, top * topWeight));
  const relevantFloor = Math.max(0.42, top * 0.72);
  const relevant = ordered
    .filter((entry) => entry.weightedScore >= relevantFloor)
    .slice(0, 3)
    .map((entry) => entry.candidate.index);

  return {
    relevant,
    sufficient: support >= 0.46 && (top >= 0.44 || relevant.length >= 2),
    support,
    reason: `surface ${routeType} evidence: top=${top.toFixed(2)} avg=${averageTop.toFixed(2)} count=${ordered.length}`,
    judgmentMode,
  };
}

/**
 * Score-based relevance filter that replaces the LLM `filterRelevant` call.
 * Uses the hybrid search score combined with semantic text similarity to
 * re-rank and trim low-relevance hits.
 */
function scoreBasedRelevanceFilter(
  hits: SearchHit[],
  _query: string,
  maxResults: number,
): SearchHit[] {
  const scored = hits.slice(0, maxResults).map((hit) => {
    const combined = hit.score;
    return { hit, combined };
  });
  scored.sort((left, right) => right.combined - left.combined);
  const filtered = scored.filter((entry) => entry.combined >= 0.25);
  if (filtered.length === 0 && scored.length > 0) {
    // Keep at least top 3 to avoid empty recall
    return scored.slice(0, 3).map((entry) => entry.hit);
  }
  return filtered.map((entry) => entry.hit);
}

function candidateRoleShare(
  candidates: RouteEvidenceCandidate[],
  role: RouteEvidenceCandidate["role"],
): number {
  const topCandidates = candidates.slice(0, 3);
  const totalScore = topCandidates.reduce(
    (sum, candidate) => sum + Math.max(candidate.score, 0),
    0,
  );
  if (totalScore <= 0) {
    return 0;
  }
  const matchedScore = topCandidates.reduce(
    (sum, candidate) => sum + (candidate.role === role ? Math.max(candidate.score, 0) : 0),
    0,
  );
  return clamp01(matchedScore / totalScore);
}

function buildPriorWeights(
  queryAnalysis: RecallQueryAnalysis,
): Record<MemoryPrimaryRouteType, number> {
  const weights: Record<MemoryPrimaryRouteType, number> = {
    workflow: 0.25,
    factual: 0.25,
    temporal: 0.25,
    explanatory: 0.25,
  };
  for (const route of PRIMARY_ROUTE_TYPES) {
    if (typeof queryAnalysis.routeWeights[route] === "number") {
      weights[route] = Math.max(weights[route], clamp01(queryAnalysis.routeWeights[route] ?? 0));
    }
  }
  return weights;
}

function allocateBudgetByWeight(
  entries: Array<{
    routeType: MemoryPrimaryRouteType;
    activated: boolean;
    weight: number;
    minBudget: number;
  }>,
  total: number,
): Record<MemoryPrimaryRouteType, number> {
  const allocations: Record<MemoryPrimaryRouteType, number> = {
    workflow: 0,
    factual: 0,
    temporal: 0,
    explanatory: 0,
  };
  const active = entries.filter((entry) => entry.activated);
  const budget = Math.max(0, Math.floor(total));
  if (budget <= 0 || active.length === 0) {
    return allocations;
  }

  const totalWeight = active.reduce((sum, entry) => sum + Math.max(entry.weight, 0.001), 0);
  if (totalWeight <= 0) {
    const evenShare = Math.floor(budget / active.length);
    let remainder = budget - evenShare * active.length;
    for (const entry of active) {
      allocations[entry.routeType] = evenShare + (remainder > 0 ? 1 : 0);
      if (remainder > 0) {
        remainder -= 1;
      }
    }
    return allocations;
  }

  const minBudgetSum = active.reduce((sum, entry) => sum + Math.max(0, entry.minBudget), 0);
  if (minBudgetSum > budget) {
    const provisional = active.map((entry) => {
      const weighted = (budget * Math.max(entry.weight, 0.001)) / totalWeight;
      const floor = Math.floor(weighted);
      return {
        routeType: entry.routeType,
        remainder: weighted - floor,
        budget: floor,
      };
    });
    let remainder = budget - provisional.reduce((sum, entry) => sum + entry.budget, 0);
    provisional
      .sort((left, right) => right.remainder - left.remainder)
      .forEach((entry, index) => {
        allocations[entry.routeType] = entry.budget + (index < remainder ? 1 : 0);
      });
    return allocations;
  }

  for (const entry of active) {
    allocations[entry.routeType] = Math.max(0, entry.minBudget);
  }

  const remainderBudget = budget - minBudgetSum;
  if (remainderBudget <= 0) {
    return allocations;
  }

  const proportional = active.map((entry) => {
    const weighted = (remainderBudget * Math.max(entry.weight, 0.001)) / totalWeight;
    const floor = Math.floor(weighted);
    allocations[entry.routeType] += floor;
    return {
      routeType: entry.routeType,
      remainder: weighted - floor,
    };
  });

  let leftover = budget - active.reduce((sum, entry) => sum + allocations[entry.routeType], 0);
  proportional
    .sort((left, right) => right.remainder - left.remainder)
    .forEach((entry, index) => {
      if (index < leftover) {
        allocations[entry.routeType] += 1;
      }
    });

  return allocations;
}

function buildRouteDecision(
  evaluations: RecallRouteEvaluation[],
  reasons: string[],
  intent?: {
    preferWorkflowDeictic?: boolean;
    preferProjectSnapshot?: boolean;
    preferHistoricalLookup?: boolean;
    preferRelationalLookup?: boolean;
    preferChunkEvidence?: boolean;
  },
): RouteDecision {
  const ordered = [...evaluations].sort((left, right) => right.finalScore - left.finalScore);
  const top = ordered[0];
  const second = ordered[1];

  if (!top) {
    return {
      routeType: "unknown",
      routeConfidence: 0.25,
      reasons: [...reasons, "route:unknown:insufficient-evidence"],
    };
  }

  if (
    top.finalScore < 0.38 &&
    !(intent?.preferChunkEvidence && top.evidenceSupport >= 0.34 && top.candidateCount >= 1)
  ) {
    return {
      routeType: "unknown",
      routeConfidence: Math.max(0.25, top.finalScore),
      reasons: [...reasons, "route:unknown:insufficient-evidence"],
    };
  }

  if (intent?.preferProjectSnapshot) {
    const factual = ordered.find((entry) => entry.routeType === "factual");
    if (
      factual &&
      factual.evidenceSupport >= 0.42 &&
      factual.finalScore >= 0.42 &&
      (top.routeType === "factual" || top.finalScore - factual.finalScore < 0.22)
    ) {
      return {
        routeType: "factual",
        routeConfidence: Math.max(0.5, factual.finalScore),
        reasons: [...reasons, "route:factual:project-profile-snapshot-authoritative"],
      };
    }
  }

  if (intent?.preferHistoricalLookup) {
    const temporal = ordered.find((entry) => entry.routeType === "temporal");
    if (
      temporal &&
      temporal.evidenceSupport >= 0.42 &&
      temporal.finalScore >= 0.42 &&
      (top.routeType === "temporal" || top.finalScore - temporal.finalScore < 0.18)
    ) {
      return {
        routeType: "temporal",
        routeConfidence: Math.max(0.5, temporal.finalScore),
        reasons: [...reasons, "route:temporal:historical-query-authoritative"],
      };
    }
  }

  if (intent?.preferRelationalLookup) {
    const explanatory = ordered.find((entry) => entry.routeType === "explanatory");
    if (
      explanatory &&
      explanatory.evidenceSupport >= 0.38 &&
      explanatory.finalScore >= 0.38 &&
      (top.routeType === "explanatory" || top.finalScore - explanatory.finalScore < 0.18)
    ) {
      return {
        routeType: "explanatory",
        routeConfidence: Math.max(0.5, explanatory.finalScore),
        reasons: [...reasons, "route:explanatory:relation-query-authoritative"],
      };
    }
  }

  const explanatory = ordered.find((entry) => entry.routeType === "explanatory");
  if (
    explanatory &&
    explanatory.graphRoleShare >= 0.28 &&
    explanatory.evidenceSupport >= 0.36 &&
    top.routeType === "workflow" &&
    top.finalScore - explanatory.finalScore < 0.12
  ) {
    return {
      routeType: "explanatory",
      routeConfidence: Math.max(0.5, explanatory.finalScore),
      reasons: [...reasons, "route:explanatory:graph-supported-close-rival"],
    };
  }

  if (intent?.preferWorkflowDeictic) {
    const workflow = ordered.find((entry) => entry.routeType === "workflow");
    if (
      workflow &&
      workflow.evidenceSupport >= 0.48 &&
      workflow.finalScore >= 0.48 &&
      (top.routeType === "workflow" || top.finalScore - workflow.finalScore < 0.16)
    ) {
      return {
        routeType: "workflow",
        routeConfidence: Math.max(0.5, workflow.finalScore),
        reasons: [...reasons, "route:workflow:deictic-reference-authoritative"],
      };
    }
  }

  if (
    second &&
    top.evidenceSufficient &&
    second.evidenceSufficient &&
    top.evidenceSupport >= 0.62 &&
    second.evidenceSupport >= 0.58 &&
    second.evidenceSupport / Math.max(top.evidenceSupport, 0.01) >= 0.8
  ) {
    const collapsedPrimaryReason = collapseMixedRouteIntoPrimary(top, second);
    if (collapsedPrimaryReason) {
      return {
        routeType: top.routeType,
        routeConfidence: Math.max(0.5, top.finalScore),
        reasons: [...reasons, collapsedPrimaryReason],
      };
    }
    return {
      routeType: "mixed",
      routeConfidence: clamp01((top.evidenceSupport + second.evidenceSupport) / 2),
      reasons: [...reasons, `route:mixed:${top.routeType}+${second.routeType}`],
    };
  }

  return {
    routeType: top.routeType,
    routeConfidence: Math.max(0.5, top.finalScore),
    reasons: [...reasons, `route:${top.routeType}:${top.reason}`],
  };
}

function buildQueryDrivenRouteDecision(
  evaluations: RecallRouteEvaluation[],
  queryAnalysis: RecallQueryAnalysis,
  reasons: string[],
): RouteDecision {
  const ranked = PRIMARY_ROUTE_TYPES.map((routeType) => ({
    routeType,
    queryWeight: queryAnalysis.routeWeights[routeType] ?? 0,
    evaluation: evaluations.find((entry) => entry.routeType === routeType) ?? {
      routeType,
      priorWeight: 0,
      evidenceSupport: 0,
      evidenceSufficient: false,
      candidateCount: 0,
      topRole: null,
      graphRoleShare: 0,
      factRoleShare: 0,
      finalScore: 0,
      reason: "query-driven route fallback",
    },
  })).sort((left, right) => {
    if (right.queryWeight !== left.queryWeight) {
      return right.queryWeight - left.queryWeight;
    }
    return right.evaluation.finalScore - left.evaluation.finalScore;
  });
  const top = ranked[0];
  const second = ranked[1];
  if (!top) {
    return {
      routeType: "unknown",
      routeConfidence: 0.25,
      reasons: [...reasons, "route:unknown:missing-query-weights"],
    };
  }
  const lowContextGenericQuery =
    queryAnalysis.turnMode === "mixed" &&
    queryAnalysis.queryShape.timeframe === "timeless" &&
    queryAnalysis.queryShape.evidenceNeed === "canonical_state" &&
    queryAnalysis.evidenceFidelity === "low" &&
    queryAnalysis.answerGranularity === "summary" &&
    queryAnalysis.anchors.length <= 1;
  const confidence = clamp01(
    top.queryWeight * 0.76 +
      top.evaluation.evidenceSupport * 0.16 +
      top.evaluation.finalScore * 0.08,
  );
  if (
    lowContextGenericQuery
      ? top.evaluation.evidenceSupport < 0.28
      : top.queryWeight < 0.22 && top.evaluation.evidenceSupport < 0.28
  ) {
    return {
      routeType: "unknown",
      routeConfidence: Math.max(0.25, confidence),
      reasons: [
        ...reasons,
        lowContextGenericQuery
          ? "route:unknown:generic-query-with-weak-support"
          : "route:unknown:query-weights-insufficient",
      ],
    };
  }
  if (
    second &&
    second.queryWeight >= 0.28 &&
    top.queryWeight - second.queryWeight < 0.08 &&
    top.evaluation.finalScore - second.evaluation.finalScore < 0.12
  ) {
    return {
      routeType: "mixed",
      routeConfidence: clamp01((confidence + second.queryWeight) / 2),
      reasons: [
        ...reasons,
        `route:mixed:query-compiler-close-rival:${top.routeType}+${second.routeType}`,
      ],
    };
  }
  return {
    routeType: top.routeType,
    routeConfidence: Math.max(0.5, confidence),
    reasons: [
      ...reasons,
      `route:${top.routeType}:query-compiler-authority:q=${top.queryWeight.toFixed(2)} support=${top.evaluation.evidenceSupport.toFixed(2)}`,
    ],
  };
}

function collapseMixedRouteIntoPrimary(
  top: RecallRouteEvaluation,
  second: RecallRouteEvaluation,
): string | null {
  if (top.routeType === "explanatory" && second.routeType === "factual") {
    const supportGap = top.evidenceSupport - second.evidenceSupport;
    const finalGap = top.finalScore - second.finalScore;
    const graphLead = top.graphRoleShare - second.graphRoleShare;
    const structuralPrimary = top.graphRoleShare >= 0.28;

    // Treat factual as a companion surface when explanatory already carries the
    // structural dependency chain and materially outranks the factual restatement.
    if (structuralPrimary && graphLead >= 0.08 && (supportGap >= 0.04 || finalGap >= 0.07)) {
      return "route:explanatory:structural-primary-over-factual";
    }
  }

  if (top.routeType === "explanatory" && second.routeType === "temporal") {
    const supportGap = top.evidenceSupport - second.evidenceSupport;
    const finalGap = top.finalScore - second.finalScore;
    const graphLead = top.graphRoleShare - second.graphRoleShare;
    const structuralPrimary = top.graphRoleShare >= 0.32;
    const temporalCompanion = second.topRole === "event" || second.topRole === "chunk";

    // Temporal context often trails explanatory dependency questions with recent
    // events. Keep the route explanatory when the top route still owns the
    // structural chain and the temporal lane is just a narrative companion.
    if (
      temporalCompanion &&
      structuralPrimary &&
      graphLead >= 0.16 &&
      (supportGap >= 0.03 || finalGap >= 0.1)
    ) {
      return "route:explanatory:structural-primary-over-temporal";
    }
  }

  if (top.routeType === "explanatory" && second.routeType === "workflow") {
    const supportGap = top.evidenceSupport - second.evidenceSupport;
    const finalGap = top.finalScore - second.finalScore;
    const graphLead = top.graphRoleShare - second.graphRoleShare;
    const structuralPrimary = top.graphRoleShare >= 0.3;
    const workflowCompanion = second.topRole === "task" || second.topRole === "state";

    // Workflow often shows up as a companion because the active task mentions
    // the same dependency chain. Keep the route explanatory when structural
    // graph evidence is still the clear primary signal.
    if (
      workflowCompanion &&
      structuralPrimary &&
      graphLead >= 0.16 &&
      (supportGap >= 0.05 || finalGap >= 0.1)
    ) {
      return "route:explanatory:structural-primary-over-workflow";
    }
  }

  return null;
}

function reservedObjectiveRoutes(
  evaluations: RecallRouteEvaluation[],
  routeDecision: RouteDecision,
): Set<MemoryPrimaryRouteType> {
  const ordered = [...evaluations].sort((left, right) => right.finalScore - left.finalScore);
  const top = ordered[0];
  const second = ordered[1];
  if (routeDecision.routeType === "unknown") {
    return new Set(PRIMARY_ROUTE_TYPES);
  }
  if (routeDecision.routeType === "mixed") {
    return new Set([top?.routeType, second?.routeType].filter(Boolean) as MemoryPrimaryRouteType[]);
  }
  return new Set(
    routeDecision.routeType === "workflow" ||
      routeDecision.routeType === "factual" ||
      routeDecision.routeType === "temporal" ||
      routeDecision.routeType === "explanatory"
      ? [routeDecision.routeType]
      : [],
  );
}

function buildObjectiveBudgets(
  ctx: MemoryOperationContext,
  evaluations: RecallRouteEvaluation[],
  routeDecision: RouteDecision,
): {
  objectiveBudgets: Record<MemoryPrimaryRouteType, RecallObjectiveBudget>;
  totalObjectBudget: number;
  totalPromptChars: number;
  reservedBackgroundChars: number;
  globalOverflowObjects: number;
} {
  const reservedRoutes = reservedObjectiveRoutes(evaluations, routeDecision);
  const minWeight = clamp01(ctx.config.advanced.recallObjectiveMinWeight);
  const totalObjectBudget = Math.max(4, Math.floor(ctx.config.advanced.recallTotalObjectBudget));
  const promptFloor = Math.max(
    0,
    Math.min(ctx.config.maxInjectedChars, Math.floor(ctx.config.advanced.recallPromptBudgetFloor)),
  );
  const reservedBackgroundChars = Math.max(
    0,
    Math.min(
      Math.max(0, ctx.config.maxInjectedChars - promptFloor),
      Math.floor(ctx.config.advanced.recallBackgroundCharReserve),
    ),
  );
  const totalPromptChars = Math.max(0, ctx.config.maxInjectedChars - reservedBackgroundChars);
  const effectivePromptFloor = Math.max(
    0,
    Math.min(totalPromptChars, Math.floor(ctx.config.advanced.recallPromptBudgetFloor)),
  );

  const seeded = evaluations.map((entry) => {
    const reserved = reservedRoutes.has(entry.routeType);
    const activated =
      routeDecision.routeType === "unknown" ? true : reserved || entry.finalScore >= minWeight;
    const rawScore = activated ? Math.max(entry.finalScore, reserved ? minWeight : 0) : 0;
    return {
      ...entry,
      reserved,
      activated,
      rawScore,
    };
  });
  const active = seeded.filter((entry) => entry.activated);
  const normalizedSeed =
    active.reduce((sum, entry) => sum + entry.rawScore, 0) > 0
      ? seeded
      : seeded.map((entry) => ({
          ...entry,
          activated: true,
          rawScore: 1,
        }));

  const weightTotal = normalizedSeed.reduce((sum, entry) => sum + entry.rawScore, 0);
  const weighted = normalizedSeed.map((entry) => ({
    ...entry,
    weight: weightTotal > 0 ? entry.rawScore / weightTotal : 0,
  }));

  const overflowReserve = Math.min(
    totalObjectBudget,
    Math.floor(totalObjectBudget * clamp01(ctx.config.advanced.recallObjectiveOverflowRatio)),
  );
  const objectAllocations = allocateBudgetByWeight(
    weighted.map((entry) => ({
      routeType: entry.routeType,
      activated: entry.activated,
      weight: entry.weight,
      minBudget: entry.reserved ? 1 : 0,
    })),
    Math.max(0, totalObjectBudget - overflowReserve),
  );
  const promptAllocations = allocateBudgetByWeight(
    weighted.map((entry) => ({
      routeType: entry.routeType,
      activated: entry.activated,
      weight: entry.weight,
      minBudget: entry.activated ? effectivePromptFloor : 0,
    })),
    totalPromptChars,
  );

  return {
    objectiveBudgets: Object.fromEntries(
      weighted.map((entry) => [
        entry.routeType,
        {
          routeType: entry.routeType,
          weight: entry.weight,
          rawScore: entry.rawScore,
          activated: entry.activated,
          minObjects: entry.reserved ? 1 : 0,
          objectBudget: objectAllocations[entry.routeType],
          minPromptChars: entry.activated ? effectivePromptFloor : 0,
          promptChars: promptAllocations[entry.routeType],
          reasons: [
            entry.reserved ? "route-reserve" : "score-activated",
            `score=${entry.finalScore.toFixed(2)}`,
            `support=${entry.evidenceSupport.toFixed(2)}`,
          ],
        } satisfies RecallObjectiveBudget,
      ]),
    ) as Record<MemoryPrimaryRouteType, RecallObjectiveBudget>,
    totalObjectBudget,
    totalPromptChars,
    reservedBackgroundChars,
    globalOverflowObjects:
      totalObjectBudget - Object.values(objectAllocations).reduce((sum, budget) => sum + budget, 0),
  };
}

function selectionObjectiveForRoute(
  routeType: MemoryPrimaryRouteType,
  query: string,
  now: string,
  currentSessionKey?: string,
) {
  const objective = createMemorySelectionObjective(routeType, query, now, currentSessionKey);
  return objective;
}

function buildRouteEvidencePacks(
  store: MemxStoreBundle,
  ctx: MemoryOperationContext,
  focusedQueries: Record<MemoryPrimaryRouteType, string>,
  hybridHits: SearchHit[],
): Record<MemoryPrimaryRouteType, RouteEvidencePack> {
  // Optimization: collect memory objects once with a superset objective (temporal for
  // broadTemporal + maxEvents, plus includeHistorical for factual coverage), then
  // schedule per-route — same pattern as buildProbeSurface (line 399-411).
  const supersetObjective = selectionObjectiveForRoute(
    "temporal",
    focusedQueries.temporal,
    ctx.now,
    ctx.sessionKey,
  );
  supersetObjective.includeHistorical = true;
  const allObjects = collectMemoryObjects(store, ctx, supersetObjective, hybridHits);

  return Object.fromEntries(
    PRIMARY_ROUTE_TYPES.map((route) => {
      const objective = selectionObjectiveForRoute(
        route,
        focusedQueries[route],
        ctx.now,
        ctx.sessionKey,
      );
      return [
        route,
        {
          routeType: route,
          query: focusedQueries[route],
          candidates: toRouteEvidenceCandidatesFromObjects(
            scheduleMemoryObjects(allObjects, objective),
            10,
          ),
        },
      ];
    }),
  ) as Record<MemoryPrimaryRouteType, RouteEvidencePack>;
}

export async function planRecallAllocation(
  store: MemxStoreBundle,
  ctx: MemoryOperationContext,
  query: string,
  searchQuery: string,
  hybridHits: SearchHit[],
  queryAnalysis?: RecallQueryAnalysis,
): Promise<RecallBudgetPlan> {
  const resolvedQueryAnalysis = queryAnalysis ?? analyzeRecallQuery(query);
  const focusedQueries = buildFocusedQueries(resolvedQueryAnalysis, query, searchQuery);
  const packs = buildRouteEvidencePacks(store, ctx, focusedQueries, hybridHits);
  const priorWeights = buildPriorWeights(resolvedQueryAnalysis);
  const queryDrivenAuthority = ctx.config.advanced.enableQueryCompiler;
  const queryBlendWeight = queryDrivenAuthority
    ? 0.72
    : resolvedQueryAnalysis.queryShape.timeframe === "timeless" &&
        resolvedQueryAnalysis.queryShape.granularity === "summary"
      ? 0.2
      : 0.32;
  const priorBlendWeight = queryDrivenAuthority ? 0.06 : 0.1;
  const evidenceBlendWeight = 1 - priorBlendWeight - queryBlendWeight;
  const scoreDrivenEvaluations: RecallRouteEvaluation[] = PRIMARY_ROUTE_TYPES.map((routeType) => {
    const pack = packs[routeType];
    const evidence = buildScoreDrivenRouteEvidenceDecision(
      routeType,
      pack.candidates,
      "score-driven",
    );
    const queryWeight = resolvedQueryAnalysis.routeWeights[routeType] ?? 0;
    const finalScore = clamp01(
      priorWeights[routeType] * priorBlendWeight +
        evidence.support * evidenceBlendWeight +
        queryWeight * queryBlendWeight,
    );
    return {
      routeType,
      priorWeight: priorWeights[routeType],
      evidenceSupport: evidence.support,
      evidenceSufficient: evidence.sufficient,
      candidateCount: pack.candidates.length,
      topRole: pack.candidates[0]?.role ?? null,
      graphRoleShare: candidateRoleShare(pack.candidates, "graph"),
      factRoleShare: candidateRoleShare(pack.candidates, "fact"),
      finalScore,
      reason: `${evidence.reason};query=${queryWeight.toFixed(2)}`,
    };
  });
  const evaluations: RecallRouteEvaluation[] = scoreDrivenEvaluations;

  const ordered = [...evaluations].sort((left, right) => right.finalScore - left.finalScore);
  const reasons = [
    ...PRIMARY_ROUTE_TYPES.map(
      (routeType) =>
        `queryWeight:${routeType}:${(resolvedQueryAnalysis.routeWeights[routeType] ?? 0).toFixed(2)}`,
    ),
    ...ordered.map(
      (entry) =>
        `evidence:${entry.routeType}:final=${entry.finalScore.toFixed(2)} support=${entry.evidenceSupport.toFixed(2)} candidates=${entry.candidateCount}`,
    ),
  ];
  const routeDecision = queryDrivenAuthority
    ? buildQueryDrivenRouteDecision(evaluations, resolvedQueryAnalysis, reasons)
    : buildRouteDecision(evaluations, reasons, {
        preferWorkflowDeictic: resolvedQueryAnalysis.queryShape.evidenceNeed === "workflow_context",
        preferProjectSnapshot: resolvedQueryAnalysis.queryShape.timeframe === "current",
        preferHistoricalLookup:
          resolvedQueryAnalysis.queryShape.timeframe === "historical" ||
          resolvedQueryAnalysis.queryShape.timeframe === "compare",
        preferRelationalLookup: resolvedQueryAnalysis.queryShape.evidenceNeed === "relation",
        preferChunkEvidence:
          resolvedQueryAnalysis.queryShape.evidenceNeed === "chunk" ||
          resolvedQueryAnalysis.evidenceFidelity === "high",
      });
  const budgets = buildObjectiveBudgets(ctx, evaluations, routeDecision);
  return {
    routeDecision,
    focusedQueries,
    routeEvaluations: evaluations,
    objectiveBudgets: budgets.objectiveBudgets,
    totalObjectBudget: budgets.totalObjectBudget,
    totalPromptChars: budgets.totalPromptChars,
    reservedBackgroundChars: budgets.reservedBackgroundChars,
    globalOverflowObjects: budgets.globalOverflowObjects,
  };
}

function renderSection(title: string, lines: string[]): string {
  if (lines.length === 0) {
    return "";
  }
  return `## ${title}\n${lines.join("\n")}`;
}

function truncateLinesToBudget(lines: string[], maxChars: number): string[] {
  if (maxChars <= 0 || lines.length === 0) {
    return [];
  }
  const selected: string[] = [];
  let used = 0;
  for (const line of lines) {
    const normalized = line.trim();
    if (!normalized) {
      continue;
    }
    const addition = (selected.length > 0 ? 1 : 0) + normalized.length;
    if (used + addition <= maxChars) {
      selected.push(normalized);
      used += addition;
      continue;
    }
    if (selected.length === 0) {
      selected.push(truncateText(normalized, Math.max(24, maxChars)));
    }
    break;
  }
  return selected;
}

function renderBudgetedSection(title: string, lines: string[], maxChars: number): string {
  if (maxChars <= 0 || lines.length === 0) {
    return "";
  }
  const titleCost = title.length + 5;
  const trimmedLines = truncateLinesToBudget(lines, Math.max(24, maxChars - titleCost));
  const rendered = renderSection(title, trimmedLines);
  return rendered.length <= maxChars ? rendered : truncateText(rendered, maxChars);
}

function splitBudget(total: number, firstWeight: number): [number, number] {
  if (total <= 0) {
    return [0, 0];
  }
  const first = Math.max(0, Math.min(total, Math.floor(total * firstWeight)));
  return [first, total - first];
}

function deriveProjectionOptions(
  ctx: MemoryOperationContext,
  plan: RecallBudgetPlan,
  query: string,
  route: RouteDecision,
  queryAnalysis?: RecallQueryAnalysis,
): Parameters<typeof projectScheduledMemoryObjects>[1] {
  const queryShape = queryAnalysis?.queryShape;
  const allowHistoricalFacts =
    queryShape?.timeframe === "historical" || queryShape?.timeframe === "compare";
  const workflowBudget = plan.objectiveBudgets.workflow.objectBudget;
  const factualBudget = plan.objectiveBudgets.factual.objectBudget;
  const temporalBudget = plan.objectiveBudgets.temporal.objectBudget;
  const explanatoryBudget = plan.objectiveBudgets.explanatory.objectBudget;
  let [stateLimit, taskLimit] =
    workflowBudget > 0 ? splitBudget(Math.max(2, workflowBudget), 0.5) : [0, 0];
  if (route.routeType === "workflow" && workflowBudget > 0) {
    stateLimit = Math.max(stateLimit, 3);
    taskLimit = Math.max(taskLimit, 1);
  }
  const alternateLimit = Math.max(
    0,
    Math.min(
      6,
      plan.globalOverflowObjects +
        (plan.routeDecision.routeType === "mixed" || plan.routeDecision.routeType === "unknown"
          ? 2
          : 1),
    ),
  );
  const eventLimit = temporalBudget > 0 ? Math.max(1, temporalBudget) : 0;
  return {
    routeType: route.routeType,
    routeConfidence: route.routeConfidence,
    allowHistoricalFacts,
    preferTemporalEvents:
      route.routeType === "temporal" ||
      queryShape?.timeframe === "historical" ||
      queryShape?.timeframe === "compare",
    stateLimit,
    taskLimit,
    factLimit:
      factualBudget > 0 || explanatoryBudget > 0
        ? Math.max(
            factualBudget,
            explanatoryBudget > 0 ? 2 : 0,
            factualBudget > 0 ? 1 : 0,
            allowHistoricalFacts ? 2 : 0,
          )
        : 0,
    eventLimit,
    graphLimit: explanatoryBudget > 0 ? Math.max(1, explanatoryBudget) : 0,
    alternateLimit,
    recallChunkBudget:
      eventLimit > 0
        ? Math.min(
            ctx.config.advanced.recallChunkBudget,
            eventLimit,
            queryAnalysis?.evidenceFidelity === "high" ||
              queryShape?.timeframe === "compare" ||
              queryShape?.granularity === "exact_detail"
              ? 3
              : eventLimit,
          )
        : 0,
  };
}

function sectionPromptBudget(
  bundle: EvidenceBundle,
  routeType: MemoryPrimaryRouteType,
  fallback: number,
): number {
  const planned = bundle.budgetPlan?.objectiveBudgets[routeType]?.promptChars;
  return Math.max(0, planned ?? fallback);
}

function sliceAlternates(bundle: EvidenceBundle): string[] {
  if (
    bundle.routeConfidence >= 0.6 &&
    !bundle.diagnostics.some((entry) => entry.includes("conflict"))
  ) {
    return [];
  }
  return bundle.alternates
    .slice(0, 3)
    .map((entry) => `- Alternate: ${truncateText(entry.text, 180)}`);
}

export async function retrieveEvidence(
  store: MemxStoreBundle,
  ctx: MemoryOperationContext,
  query: string,
  searchQuery = query,
  auditOptions: RetrievalAuditOptions & {
    queryAnalysis?: RecallQueryAnalysis;
  } = {},
): Promise<EvidenceBundle> {
  const behavioralGuidance = collectBehavioralGuidanceFromObjects(store, ctx);
  const queryAnalysis =
    auditOptions.queryAnalysis ??
    (ctx.config.advanced.enableQueryCompiler
      ? await compileQuery({
          query,
          ctx,
          reasoner: store.reasoner,
        })
      : analyzeRecallQuery(query));
  const referentResolution = resolveReferentialQueryAnchors(store, ctx, query, queryAnalysis);
  const retrievalQuery = appendAnchorsToQuery(query, referentResolution.anchors);
  const retrievalSearchQuery = appendAnchorsToQuery(searchQuery, referentResolution.anchors);
  const snapshotFocus = queryAnalysis.queryShape.timeframe === "current";
  const candidateGenerationResult = await generateCandidates(store, ctx, {
    ...queryAnalysis,
    focusedQuery: retrievalSearchQuery,
    anchors: uniqueNonEmpty([...queryAnalysis.anchors, ...referentResolution.anchors]),
  });
  const rawHybridHits = candidateGenerationResult
    ? candidateGenerationResult.searchHits
    : await store.retrievalBackend.hybridSearch({
        agentId: ctx.agentId,
        scopes: ctx.scopes,
        query: retrievalSearchQuery,
        limit: 16,
        readEpoch: ctx.readEpoch,
      });
  let hybridHits = rawHybridHits.filter((hit) => !shouldSuppressRecallText(hit.text));
  if (hybridHits.length !== rawHybridHits.length) {
    // Keep diagnostics local until route decision is resolved.
  }
  const allocationPlan = await planRecallAllocation(
    store,
    ctx,
    retrievalQuery,
    retrievalSearchQuery,
    hybridHits,
    queryAnalysis,
  );
  if (referentResolution.anchors.length > 0) {
    for (const routeType of PRIMARY_ROUTE_TYPES) {
      allocationPlan.focusedQueries[routeType] = appendAnchorsToQuery(
        allocationPlan.focusedQueries[routeType] || retrievalQuery,
        referentResolution.anchors,
      );
    }
  }
  const route = allocationPlan.routeDecision;
  const diagnostics = [...route.reasons, ...referentResolution.reasons];
  if (hybridHits.length !== rawHybridHits.length) {
    diagnostics.push("filtered-bootstrap-memory-noise");
  }
  if (hybridHits.length > 6 && (route.routeType === "mixed" || route.routeConfidence < 0.72)) {
    hybridHits = scoreBasedRelevanceFilter(hybridHits, retrievalQuery, 12);
  }
  const budgetedSelection = collectAndScheduleMemoryObjectsWithBudget(
    store,
    ctx,
    allocationPlan,
    hybridHits,
  );
  const budgetedScheduled = budgetedSelection.scheduled;

  const projected = projectScheduledMemoryObjects(
    budgetedScheduled,
    deriveProjectionOptions(ctx, allocationPlan, retrievalQuery, route, queryAnalysis),
  );
  const allowHistoricalFacts =
    queryAnalysis.queryShape.timeframe === "historical" ||
    queryAnalysis.queryShape.timeframe === "compare";
  const queryAnchors = uniqueNonEmpty(referentResolution.anchors).filter((anchor) =>
    isMeaningfulRecallAnchor(anchor),
  );
  const resolvedReferentAnchors = uniqueNonEmpty(referentResolution.anchors).filter((anchor) =>
    isMeaningfulRecallAnchor(anchor),
  );
  let states = projected.states;
  let tasks = projected.tasks;
  let facts = projected.facts;
  let events = projected.events;
  let graph = projected.graph;
  let alternates = projected.alternates;
  const candidateDrivenAuthority = true;
  const candidateAlternateRows = candidateGenerationResult
    ? filterBootstrapRows(rowsFromSearchHits(candidateGenerationResult.alternateSearchHits))
    : [];

  if (events.length > 1) {
    events = [...events].sort(compareEvidenceRowsChronologically);
  }

  if (allocationPlan.objectiveBudgets.explanatory.activated && graph.nodes.length === 0) {
    diagnostics.push("graph-no-seeds");
  }

  if (alternates.length === 0) {
    alternates = dedupeEvidenceRows(
      [...candidateAlternateRows, ...filterBootstrapRows(rowsFromSearchHits(hybridHits))].filter(
        (entry) => !shouldSuppressRecallText(entry.text),
      ),
      6,
    );
  } else if (candidateAlternateRows.length > 0) {
    alternates = dedupeEvidenceRows([...alternates, ...candidateAlternateRows], 6);
  }

  if (allowHistoricalFacts) {
    const conservativeHistoricalAlternates = queryMemoryFacts({
      store,
      ctx,
      text: retrievalQuery,
      limit: 2,
      includeHistorical: true,
    })
      .filter((fact) => fact.status === "superseded")
      .map((fact) => {
        const supportRefs = factSupportSourceRefs(fact);
        const sourceRef = supportRefs[0] ?? fact.sourceRef;
        return toEvidenceRow({
          id: fact.factId,
          text: formatFactLine({
            subject: fact.canonicalSubject,
            predicate: fact.predicate,
            object: fact.canonicalObject ?? undefined,
            objectValueJson: fact.objectValueJson,
            status: fact.status,
          }),
          score: 0.34,
          scope: fact.scope,
          confidence: fact.confidence,
          sourceRef,
          observedAt: fact.updatedAt,
          provenance: factSupportText(fact),
          lineage: {
            canonicalKind: "fact",
            canonicalId: fact.factId,
            sourceKind: "fact",
            sourceId: fact.factId,
            sourceRef,
            materializedEpoch: fact.materializedEpoch,
          },
        });
      });
    if (conservativeHistoricalAlternates.length > 0) {
      alternates = dedupeEvidenceRows(
        [...alternates, ...conservativeHistoricalAlternates].sort(
          (left, right) => (right.score ?? 0) - (left.score ?? 0),
        ),
        6,
      );
      diagnostics.push("historical-backfill:alternate-only");
    }
  }

  if (snapshotFocus && route.routeType !== "workflow") {
    states = states.filter((entry) => {
      const { label } = splitLabelValue(entry.text);
      return Boolean(label && isSnapshotFactualStateKey(label));
    });
    tasks = [];
    diagnostics.push("snapshot-focus");
  }

  if (
    route.routeType === "workflow" &&
    states.some((entry) => entry.id.endsWith("workflow.current_task")) &&
    (states.some((entry) => entry.id.endsWith("workflow.blocker")) ||
      states.some((entry) => entry.id.endsWith("workflow.next_action")))
  ) {
    tasks = [];
    diagnostics.push("workflow-state-authority");
  }

  states = dedupeEvidenceRows(filterBootstrapRows(states), 4);
  tasks = dedupeEvidenceRows(filterBootstrapRows(tasks), 4);
  const factualMainLimit = shouldUseExactSnippetSupport(queryAnalysis) ? 8 : 6;
  facts = dedupeEvidenceRows(filterBootstrapRows(facts), factualMainLimit);
  events = dedupeEvidenceRows(
    filterBootstrapRows(events),
    Math.max(4, ctx.config.advanced.recallChunkBudget),
  );
  states = states.filter((entry) => !shouldSuppressRecallText(entry.text));
  tasks = tasks.filter((entry) => !shouldSuppressRecallText(entry.text));
  facts = facts.filter((entry) => !shouldSuppressRecallText(entry.text));
  events = events.filter((entry) => !shouldSuppressRecallText(entry.text));
  if (!shouldRetainTasksInMainSurface(queryAnalysis)) {
    tasks = [];
    diagnostics.push("main-surface:task-suppressed");
  }
  if (!shouldRetainStatesInMainSurface(queryAnalysis)) {
    states = [];
    diagnostics.push("main-surface:state-suppressed");
  } else if (!shouldUseWorkflowMainSurface(queryAnalysis)) {
    const factualStates = states.filter((entry) => {
      const { label } = splitLabelValue(entry.text);
      return Boolean(label && isSnapshotFactualStateKey(label));
    });
    if (factualStates.length !== states.length) {
      diagnostics.push("main-surface:workflow-state-suppressed");
    }
    states = factualStates;
  }
  if (!shouldUseWorkflowMainSurface(queryAnalysis)) {
    const factualFacts = facts.filter((entry) => {
      const fact = store.factRepo.get(entry.id);
      return fact?.predicate !== "has_workflow_guidance";
    });
    if (factualFacts.length !== facts.length) {
      diagnostics.push("main-surface:workflow-guidance-suppressed");
    }
    facts = factualFacts;
  }
  const scheduledFactRows = facts.slice();
  let currentFactGuardSourceRows: EvidenceRow[] = [];
  const suppressedPromptEvidenceCandidateIds = new Set<string>();
  let suppressPromptEvidenceBeforeMs: number | undefined;

  if (
    candidateGenerationResult &&
    shouldApplyCandidateAuthorityToMainSurface(route, queryAnalysis)
  ) {
    const candidateFactLimit = shouldUseExactSnippetSupport(queryAnalysis) ? 8 : 6;
    const candidateStateRows = prioritizeCandidateRowsForMainSurface(
      primaryCandidateRowsForSurface(candidateGenerationResult, "state"),
      queryAnchors,
      4,
    );
    const candidateFactRows = prioritizeCandidateRowsForMainSurface(
      primaryCandidateRowsForSurface(candidateGenerationResult, "fact", { store, queryAnalysis }),
      queryAnchors,
      candidateFactLimit,
    );
    const candidateEventRows = prioritizeCandidateRowsForMainSurface(
      primaryCandidateRowsForSurface(candidateGenerationResult, "event"),
      queryAnchors,
      Math.max(4, ctx.config.advanced.recallChunkBudget),
    );
    const candidateChunkRows = prioritizeCandidateRowsForMainSurface(
      primaryCandidateRowsForSurface(candidateGenerationResult, "chunk").filter(
        (entry) => entry.provenance !== "assistant",
      ),
      queryAnchors,
      Math.max(3, Math.min(6, ctx.config.advanced.recallChunkBudget)),
    );
    currentFactGuardSourceRows = [...candidateEventRows, ...candidateChunkRows];

    if (candidateStateRows.length > 0) {
      states = candidateStateRows;
      diagnostics.push("candidate-authority:state-main");
    }
    if (candidateFactRows.length > 0) {
      facts = candidateFactRows;
      diagnostics.push("candidate-authority:fact-main");
    }
    if (
      route.routeType === "factual" &&
      (candidateEventRows.length > 0 || candidateChunkRows.length > 0)
    ) {
      events = dedupeEvidenceRows(
        [...candidateEventRows, ...candidateChunkRows],
        Math.max(4, ctx.config.advanced.recallChunkBudget),
      );
      diagnostics.push("candidate-authority:event-main");
    }
  }

  const newerCurrentSourceRows = newerSourceRowsForCurrentFactGuard({
    facts,
    sourceRows: currentFactGuardSourceRows,
    queryAnalysis,
  });
  if (newerCurrentSourceRows.length > 0) {
    for (const fact of facts) {
      for (const id of uniqueNonEmpty([
        fact.id,
        fact.id.startsWith("fact:") ? fact.id.slice(5) : `fact:${fact.id}`,
        fact.lineage?.canonicalId,
        fact.lineage?.sourceId,
      ])) {
        suppressedPromptEvidenceCandidateIds.add(id);
      }
    }
    suppressPromptEvidenceBeforeMs = Math.min(...newerCurrentSourceRows.map((row) => observedAtMs(row)));
    facts = [];
    events = dedupeEvidenceRows(
      [
        ...newerCurrentSourceRows,
        ...events.filter((row) => observedAtMs(row) >= (suppressPromptEvidenceBeforeMs ?? 0)),
      ],
      Math.max(4, ctx.config.advanced.recallChunkBudget),
    );
    diagnostics.push("newer-source-evidence-suppressed-stale-current-fact");
  }

  const complementaryFactRows = selectComplementaryFactRowsForMainSurface({
    store,
    queryAnalysis,
    queryAnchors,
    selectedFacts: facts,
    fallbackFacts: scheduledFactRows,
    reserveFactTrace: budgetedSelection.trace.reserveSelections.factual,
    limit: 1,
  });
  if (complementaryFactRows.length > 0) {
    // Keep a tightly scoped factual companion when the main fact surface already carries a
    // current answer-bearing detail and a sibling fact contributes another part of the answer.
    facts = dedupeEvidenceRows([...facts, ...complementaryFactRows], factualMainLimit);
    diagnostics.push("candidate-authority:fact-companion");
  }

  const workflowGateAnchors =
    resolvedReferentAnchors.length > 0 ? resolvedReferentAnchors : queryAnchors;
  const hasAnchoredWorkflowTask =
    route.routeType === "workflow" &&
    queryAnalysis.queryShape.evidenceNeed === "workflow_context" &&
    workflowGateAnchors.length > 0 &&
    tasks.some((entry) => queryAnchorSupport(entry.text, workflowGateAnchors) >= 0.58);
  if (hasAnchoredWorkflowTask) {
    tasks = tasks.filter((entry) => queryAnchorSupport(entry.text, workflowGateAnchors) >= 0.58);
    facts = [];
    graph = {
      nodes: [],
      edges: [],
      paths: [],
      pathCandidates: [],
    };
    if (!diagnostics.includes("workflow-anchored-task-authority")) {
      diagnostics.push("workflow-anchored-task-authority");
    }
  }

  const supportLinkedSourceRefs = shouldUseExactSnippetSupport(queryAnalysis)
    ? supportRefsFromSelectedFacts(store, facts, queryAnchors)
    : [];
  const supportLinkedChunkRows = shouldUseExactSnippetSupport(queryAnalysis)
    ? buildSupportRefChunkRows({
        store,
        ctx,
        sourceRefs: supportLinkedSourceRefs,
        queryAnchors,
        limit: Math.max(3, Math.min(6, ctx.config.advanced.recallChunkBudget)),
      })
    : [];
  if (supportLinkedChunkRows.length > 0) {
    events = dedupeEvidenceRows(
      [...supportLinkedChunkRows, ...events.filter((entry) => !isChunkEvidenceRow(entry))],
      Math.max(4, ctx.config.advanced.recallChunkBudget),
    );
    diagnostics.push("source-linked-chunk-rescue:fact-support");
  }

  const exactSnippetFallbacks = shouldUseExactSnippetSupport(queryAnalysis)
    ? buildSourceRefFallbackSnippets({
        store,
        ctx,
        sourceRefs: uniqueNonEmpty([
          ...supportLinkedSourceRefs,
          ...facts.map((entry) => entry.sourceRef ?? ""),
          ...events.map((entry) => entry.sourceRef ?? ""),
        ]),
        limit: 3,
        queryAnchors,
      })
    : [];
  const projectedSnippetCandidates = shouldUseExactSnippetSupport(queryAnalysis)
    ? preferredProjectedExactSnippets({
        projectedIds: projected.recalledChunkIds,
        projectedTexts: projected.recalledChunkTexts,
        queryAnchors,
        limit: Math.min(3, projected.recalledChunkIds.length),
      })
    : [];
  const supportLinkedSnippetCandidates = shouldUseExactSnippetSupport(queryAnalysis)
    ? buildSourceRefFallbackSnippets({
        store,
        ctx,
        sourceRefs: supportLinkedSourceRefs,
        limit: 3,
        queryAnchors,
      }).map((entry) => ({ ...entry, source: "support_ref" as const }))
    : [];
  const rankedExactSnippetCandidates = shouldUseExactSnippetSupport(queryAnalysis)
    ? rankExactSnippetCandidates(
        [
          ...supportLinkedSnippetCandidates,
          ...projectedSnippetCandidates,
          ...exactSnippetFallbacks,
        ],
        queryAnalysis,
      ).slice(0, 3)
    : [];
  const exactSnippetIds = shouldUseExactSnippetSupport(queryAnalysis)
    ? rankedExactSnippetCandidates.map((entry) => entry.snippetId)
    : undefined;
  const exactSnippetTexts = shouldUseExactSnippetSupport(queryAnalysis)
    ? rankedExactSnippetCandidates.map((entry) => entry.text)
    : undefined;
  const selectedExactSnippets = shouldUseExactSnippetSupport(queryAnalysis)
    ? rankedExactSnippetCandidates.length > 0
      ? rankedExactSnippetCandidates.map((entry) => {
          if (entry.source !== "projected") {
            return entry;
          }
          const snippetId = entry.snippetId;
          const matchedHit =
            candidateGenerationResult?.searchHits.find((hit) => hit.docId === snippetId) ??
            rawHybridHits.find((hit) => hit.docId === snippetId);
          return {
            snippetId,
            text: entry.text,
            sourceRef:
              typeof matchedHit?.metadata.sourceRef === "string"
                ? matchedHit.metadata.sourceRef
                : undefined,
            lineage: lineageFromMetadata(matchedHit?.metadata, {
              sourceKind: "vector_doc",
              sourceId: snippetId,
            }) ?? {
              sourceKind: "vector_doc",
              sourceId: snippetId,
            },
            source: "projected" as const,
            goalScore: entry.goalScore,
          };
        })
      : undefined
    : undefined;
  const recalledChunkSupport = mergeRecalledChunkSupport({
    supportRows: supportLinkedChunkRows,
    projectedIds: projected.recalledChunkIds,
    projectedTexts: projected.recalledChunkTexts,
    limit: Math.max(3, Math.min(6, ctx.config.advanced.recallChunkBudget)),
  });
  if (
    shouldUseExactSnippetSupport(queryAnalysis) &&
    projected.recalledChunkIds.length === 0 &&
    exactSnippetFallbacks.length > 0
  ) {
    diagnostics.push("exact-snippet:source-ref-fallback");
  }
  if (shouldUseExactSnippetSupport(queryAnalysis) && supportLinkedSnippetCandidates.length > 0) {
    diagnostics.push("exact-snippet:support-ref-rescue");
  }
  const controlEvidence = controlPromptEvidenceCandidates({
    store,
    ctx,
    queryAnalysis,
  });
  const initialPromptEvidence = buildPromptEvidenceCandidates({
    store,
    ctx,
    queryAnalysis,
    candidateGenerationResult,
    facts,
    events,
    controlEvidence,
    selectedExactSnippets,
    suppressedCandidateIds: suppressedPromptEvidenceCandidateIds,
    suppressCandidateEvidenceBeforeMs: suppressPromptEvidenceBeforeMs,
  });
  const evidenceAssembly = assembleEvidencePackets({
    queryAnalysis,
    candidateGenerationResult,
    promptEvidence: initialPromptEvidence,
    now: ctx.now,
  });
  const promptEvidence = evidenceAssembly.promptEvidence;
  const evidencePlanAudit = buildEvidencePlanAudit({
    queryAnalysis,
    promptEvidence,
    candidateGenerationResult,
    injectedEvidence: promptEvidence.filter(
      (entry) => entry.role === "protected" && !entry.dropReason,
    ),
  });

  const bundle: EvidenceBundle = {
    routeType: route.routeType,
    routeConfidence: route.routeConfidence,
    queryText: query,
    queryAnchors,
    queryShape: queryAnalysis.queryShape,
    routeWeights: queryAnalysis.routeWeights,
    turnMode: queryAnalysis.turnMode,
    snapshotFocus,
    states,
    tasks,
    facts,
    events,
    graph,
    alternates: alternates.slice(0, 6),
    diagnostics,
    behavioralGuidance,
    recalledChunkIds: recalledChunkSupport.ids,
    recalledChunkTexts: recalledChunkSupport.texts,
    promptEvidence,
    evidencePackets: evidenceAssembly.packets,
    evidencePlanAudit,
    evidencePacketAudit: evidenceAssembly.audit,
    selectedExactSnippetIds: exactSnippetIds,
    selectedExactSnippetTexts: exactSnippetTexts,
    selectedExactSnippets,
    budgetPlan: allocationPlan,
    selectionTrace: budgetedSelection.trace,
    renderedBlock: "",
  };

  bundle.renderedBlock = renderEvidenceBundle(bundle, ctx.config.maxInjectedChars);
  const auditId = randomId("audit");
  emitFullRetrievalSignals(store, ctx, {
    auditId,
    bundle,
    scheduled: budgetedSelection.scheduled,
  });
  emitContradictionSignals(store, ctx, {
    query,
    routeType:
      route.routeType === "mixed" || route.routeType === "unknown"
        ? "explanatory"
        : route.routeType,
    graphEdges: bundle.graph.edges,
  });

  if (ctx.config.advanced.enableTelemetryAudit) {
    const activePromptEvidence = bundle.promptEvidence.filter(
      (entry) => !entry.dropReason && entry.role !== "alternate",
    );
    const injectedPromptEvidence = activePromptEvidence.filter((entry) => entry.injected);
    const renderedPromptLines = renderedPromptLineAudit(bundle);
    const evidencePlanAuditForTelemetry = buildEvidencePlanAudit({
      queryAnalysis,
      promptEvidence: bundle.promptEvidence,
      candidateGenerationResult,
      injectedEvidence: injectedPromptEvidence,
    });
    const selected = {
      states: bundle.states.map((entry) => entry.id),
      tasks: bundle.tasks.map((entry) => entry.id),
      facts: bundle.facts.map((entry) => entry.id),
      events: bundle.events.map((entry) => entry.id),
      graphEdges: bundle.graph.edges.map((entry) => entry.edgeId),
      recalledChunks: bundle.recalledChunkIds,
      exactSnippets: bundle.selectedExactSnippetIds ?? [],
      exactSnippetTexts: bundle.selectedExactSnippetTexts ?? [],
      exactSnippetLineage: bundle.selectedExactSnippets ?? [],
      semanticBridges: queryAnalysis.semanticBridges ?? [],
      promptEvidence: bundle.promptEvidence.map((entry) => ({
        id: entry.id,
        surface: entry.surface,
        sourceRef: entry.sourceRef,
        observedAt: entry.observedAt,
        mergedSourceRefs: entry.mergedSourceRefs,
        normalizedSourceRefs: normalizedPromptEvidenceSourceRefs(entry),
        excerptAnchors: entry.excerptAnchors,
        priority: entry.priority,
        injectionScore: entry.injectionScore,
        scoreBreakdown: entry.scoreBreakdown,
        goalScore: entry.goalScore,
        semanticScore: entry.semanticScore,
        coverage: entry.coverage,
        slotCoverage: entry.slotCoverage,
        filledSlotIds: entry.filledSlotIds,
        slotEvidenceRole: entry.slotEvidenceRole,
        bridgeMatches: entry.bridgeMatches,
        injected: entry.injected,
        packetId: entry.packetId,
        metadata: entry.metadata,
        eligibility: entry.eligibility,
        grade: entry.grade,
        blockedBy: entry.blockedBy,
        source: entry.source,
        role: entry.role,
        selectionReason: entry.selectionReason,
        protectionReason: entry.protectionReason,
        dropReason: entry.dropReason,
        text: truncateText(entry.text, 720),
        rawText:
          entry.rawText && entry.rawText !== entry.text
            ? truncateText(entry.rawText, 720)
            : undefined,
        lineage: entry.lineage,
      })),
      slotCandidates: (candidateGenerationResult?.slotCandidates ?? []).map((hit) => ({
        candidateId: hit.candidateId,
        docId: hit.docId,
        surface: hit.surface,
        tier: hit.tier,
        score: hit.score,
        slotMatches: hit.slotMatches,
        goalMatches: hit.goalMatches,
        bridgeMatches: hit.bridgeMatches,
        sourceRef: hit.lineage.sourceRef,
        normalizedSourceRefs: normalizeSourceRefs([hit.lineage.sourceRef]),
        text: truncateText(hit.text, 720),
        lineage: hit.lineage,
      })),
      bridgeCandidates: (candidateGenerationResult?.bridgeCandidates ?? []).map((hit) => ({
        candidateId: hit.candidateId,
        docId: hit.docId,
        surface: hit.surface,
        tier: hit.tier,
        score: hit.score,
        slotMatches: hit.slotMatches,
        goalMatches: hit.goalMatches,
        bridgeMatches: hit.bridgeMatches,
        sourceRef: hit.lineage.sourceRef,
        normalizedSourceRefs: normalizeSourceRefs([hit.lineage.sourceRef]),
        text: truncateText(hit.text, 720),
        lineage: hit.lineage,
      })),
      candidatePool: bundle.promptEvidence
        .filter((entry) => !entry.dropReason)
        .map((entry) => ({
          id: entry.id,
          surface: entry.surface,
          sourceRef: entry.sourceRef,
          mergedSourceRefs: entry.mergedSourceRefs,
          normalizedSourceRefs: normalizedPromptEvidenceSourceRefs(entry),
          metadata: entry.metadata,
          source: entry.source,
          role: entry.role,
          priority: entry.priority,
          injectionScore: entry.injectionScore,
          bridgeMatches: entry.bridgeMatches,
          slotEvidenceRole: entry.slotEvidenceRole,
          selectionReason: entry.selectionReason,
          text: truncateText(entry.text, 360),
        })),
      rankedPromptEvidence: bundle.promptEvidence
        .filter((entry) => !entry.dropReason)
        .sort(
          (left, right) =>
            (right.injectionScore ?? right.priority) - (left.injectionScore ?? left.priority),
        )
        .map((entry) => ({
          id: entry.id,
          surface: entry.surface,
          sourceRef: entry.sourceRef,
          mergedSourceRefs: entry.mergedSourceRefs,
          normalizedSourceRefs: normalizedPromptEvidenceSourceRefs(entry),
          metadata: entry.metadata,
          role: entry.role,
          injected: entry.injected,
          priority: entry.priority,
          injectionScore: entry.injectionScore,
          scoreBreakdown: entry.scoreBreakdown,
          slotEvidenceRole: entry.slotEvidenceRole,
          packetId: entry.packetId,
          eligibility: entry.eligibility,
          grade: entry.grade,
          blockedBy: entry.blockedBy,
          softPenalties: entry.softPenalties,
          hardExclusions: entry.hardExclusions,
          selectionReason: entry.selectionReason,
          text: truncateText(entry.text, 360),
        })),
      evidencePackets: bundle.evidencePackets.filter((packet) => !packet.dropReason).map((packet) => ({
        packetId: packet.packetId,
        slotId: packet.slotId,
        operationType: packet.operationType,
        role: packet.role,
        protected: packet.protected,
        injected: packet.injected,
        layers: packet.layers,
        sourceRefs: packet.sourceRefs,
        normalizedSourceRefs: packet.normalizedSourceRefs,
        supportSourceRefs: packet.supportSourceRefs,
        normalizedSupportSourceRefs: packet.normalizedSupportSourceRefs,
        slotIds: packet.slotIds,
        eligibility: packet.eligibility,
        grade: packet.grade,
        allSourceRefs: packet.allSourceRefs,
        normalizedAllSourceRefs: packet.normalizedAllSourceRefs,
        score: packet.score,
        scoreBreakdown: packet.scoreBreakdown,
        displayLines: packet.displayLines,
        hiddenExactDuplicates: packet.hiddenExactDuplicates,
        answerUnits: packet.answerUnits,
        contextUnits: packet.contextUnits,
        supportUnits: packet.supportUnits,
        selectionReason: packet.selectionReason,
        blockedBy: packet.blockedBy,
        softPenalties: packet.softPenalties,
        hardExclusions: packet.hardExclusions,
        answerCandidate: packet.answerCandidate
          ? {
              id: packet.answerCandidate.id,
              surface: packet.answerCandidate.surface,
              sourceRef: packet.answerCandidate.sourceRef,
              normalizedSourceRefs: normalizedPromptEvidenceSourceRefs(packet.answerCandidate),
              slotEvidenceRole: packet.answerCandidate.slotEvidenceRole,
              text: truncateText(packet.answerCandidate.text, 360),
            }
          : undefined,
        contextCandidates: (packet.contextCandidates ?? []).map((entry) => ({
          id: entry.id,
          surface: entry.surface,
          sourceRef: entry.sourceRef,
          normalizedSourceRefs: normalizedPromptEvidenceSourceRefs(entry),
          slotEvidenceRole: entry.slotEvidenceRole,
          text: truncateText(entry.text, 240),
        })),
        observedAt: packet.observedAt,
        resolvedDate: packet.resolvedDate,
        quantityHint: packet.quantityHint,
        unitHint: packet.unitHint,
        dedupeKey: packet.dedupeKey,
        coverage: packet.coverage,
        protectionReason: packet.protectionReason,
        dropReason: packet.dropReason,
        primaryText: truncateText(packet.primaryText, 720),
        supportingTexts: packet.supportingTexts.map((text) => truncateText(text, 360)),
      })),
      droppedEvidencePackets: bundle.evidencePackets
        .filter((packet) => packet.dropReason)
        .map((packet) => ({
          packetId: packet.packetId,
          slotId: packet.slotId,
          operationType: packet.operationType,
          role: packet.role,
          injected: packet.injected,
          layers: packet.layers,
          sourceRefs: packet.sourceRefs,
          normalizedSourceRefs: packet.normalizedSourceRefs,
          allSourceRefs: packet.allSourceRefs,
          normalizedAllSourceRefs: packet.normalizedAllSourceRefs,
          score: packet.score,
          scoreBreakdown: packet.scoreBreakdown,
          displayLines: packet.displayLines,
          selectionReason: packet.selectionReason,
          blockedBy: packet.blockedBy,
          softPenalties: packet.softPenalties,
          hardExclusions: packet.hardExclusions,
          coverage: packet.coverage,
          dropReason: packet.dropReason,
          primaryText: truncateText(packet.primaryText, 720),
          supportingTexts: packet.supportingTexts.map((text) => truncateText(text, 360)),
        })),
      rankedInjectedPackets: bundle.evidencePackets
        .filter((packet) => packet.injected || packet.protected)
        .sort((left, right) => (right.score ?? 0) - (left.score ?? 0))
        .map((packet) => ({
          packetId: packet.packetId,
          slotId: packet.slotId,
          score: packet.score,
          scoreBreakdown: packet.scoreBreakdown,
          sourceRefs: packet.sourceRefs,
          allSourceRefs: packet.allSourceRefs,
          normalizedSourceRefs: packet.normalizedSourceRefs,
          normalizedAllSourceRefs: packet.normalizedAllSourceRefs,
          displayLines: packet.displayLines,
          selectionReason: packet.selectionReason,
        })),
      scoreCurve: bundle.evidencePackets
        .filter((packet) => !packet.dropReason)
        .sort((left, right) => (right.score ?? 0) - (left.score ?? 0))
        .slice(0, 12)
        .map((packet, index) => ({
          rank: index + 1,
          packetId: packet.packetId,
          injected: packet.injected ?? false,
          score: packet.score,
          finalScore: packet.grade?.finalScore,
          slotId: packet.slotId,
          sourceRefs: packet.sourceRefs,
          normalizedSourceRefs: packet.normalizedSourceRefs,
        })),
      renderedPromptLines,
      injectedEvidence: injectedPromptEvidence.map((entry) => ({
        id: entry.id,
        surface: entry.surface,
        sourceRef: entry.sourceRef,
        mergedSourceRefs: entry.mergedSourceRefs,
        normalizedSourceRefs: normalizedPromptEvidenceSourceRefs(entry),
        metadata: entry.metadata,
        priority: entry.priority,
        injectionScore: entry.injectionScore,
        scoreBreakdown: entry.scoreBreakdown,
        goalScore: entry.goalScore,
        semanticScore: entry.semanticScore,
        coverage: entry.coverage,
        slotCoverage: entry.slotCoverage,
        filledSlotIds: entry.filledSlotIds,
        slotEvidenceRole: entry.slotEvidenceRole,
        bridgeMatches: entry.bridgeMatches,
        packetId: entry.packetId,
        eligibility: entry.eligibility,
        grade: entry.grade,
        blockedBy: entry.blockedBy,
        softPenalties: entry.softPenalties,
        hardExclusions: entry.hardExclusions,
        role: entry.role,
        selectionReason: entry.selectionReason,
        protectionReason: entry.protectionReason,
        text: truncateText(entry.text, 720),
        rawText:
          entry.rawText && entry.rawText !== entry.text
            ? truncateText(entry.rawText, 720)
            : undefined,
      })),
      droppedEvidence: bundle.promptEvidence
        .filter((entry) => entry.dropReason)
        .map((entry) => ({
          id: entry.id,
          surface: entry.surface,
          sourceRef: entry.sourceRef,
          mergedSourceRefs: entry.mergedSourceRefs,
          normalizedSourceRefs: normalizedPromptEvidenceSourceRefs(entry),
          metadata: entry.metadata,
          priority: entry.priority,
          injectionScore: entry.injectionScore,
          scoreBreakdown: entry.scoreBreakdown,
          goalScore: entry.goalScore,
          semanticScore: entry.semanticScore,
          coverage: entry.coverage,
          slotCoverage: entry.slotCoverage,
          filledSlotIds: entry.filledSlotIds,
          slotEvidenceRole: entry.slotEvidenceRole,
          packetId: entry.packetId,
          eligibility: entry.eligibility,
          grade: entry.grade,
          blockedBy: entry.blockedBy,
          softPenalties: entry.softPenalties,
          hardExclusions: entry.hardExclusions,
          selectionReason: entry.selectionReason,
          protectionReason: entry.protectionReason,
          dropReason: entry.dropReason,
          text: truncateText(entry.text, 240),
          rawText:
            entry.rawText && entry.rawText !== entry.text
              ? truncateText(entry.rawText, 240)
              : undefined,
        })),
      sourceExpansion: bundle.promptEvidence
        .filter((entry) => entry.metadata?.sourceExpansion === true)
        .map((entry) => ({
          id: entry.id,
          surface: entry.surface,
          sourceRef: entry.sourceRef,
          mergedSourceRefs: entry.mergedSourceRefs,
          normalizedSourceRefs: normalizedPromptEvidenceSourceRefs(entry),
          neighborOf: Array.isArray(entry.metadata?.neighborOf) ? entry.metadata.neighborOf : [],
          role: entry.role,
          packetId: entry.packetId,
          injectionScore: entry.injectionScore,
          text: truncateText(entry.text, 360),
        })),
      queryCompiler: {
        answerGranularity: queryAnalysis.answerGranularity,
        evidenceFidelity: queryAnalysis.evidenceFidelity,
        answerMode: queryAnalysis.answerMode,
        evidenceCoverage: queryAnalysis.evidenceCoverage,
        candidateSurfaces: queryAnalysis.candidateSurfaces,
        evidenceGoals: queryAnalysis.evidenceGoals,
        suppressedEntities: queryAnalysis.suppressedEntities,
        evidencePlan: queryAnalysis.evidencePlan,
        semanticBridges: queryAnalysis.semanticBridges,
        supportNeed: queryAnalysis.supportNeed,
        ambiguityLevel: queryAnalysis.ambiguityLevel,
        compilerProvenance: queryAnalysis.compilerProvenance,
      },
      candidateGeneration: candidateGenerationResult
        ? candidateGenerationAuditPayload(candidateGenerationResult)
        : undefined,
      evidencePlanAudit: evidencePlanAuditForTelemetry,
      evidencePacketAudit: bundle.evidencePacketAudit,
      selectionAuthority: {
        candidateDrivenAuthority,
        historicalBackfillMode: diagnostics.includes("historical-backfill:primary")
          ? "primary"
          : diagnostics.includes("historical-backfill:alternate-only")
            ? "alternate-only"
            : "none",
        alternatePromotionEnabled: false,
      },
      mainVsAlternate: {
        stateIds: bundle.states.map((entry) => entry.id),
        taskIds: bundle.tasks.map((entry) => entry.id),
        factIds: bundle.facts.map((entry) => entry.id),
        eventIds: bundle.events.map((entry) => entry.id),
        alternateIds: bundle.alternates.map((entry) => entry.id),
      },
      llmBudget: snapshotMemoryLlmBudgetAudit(ctx.llmBudgetAudit),
      recall: buildRecallAuditPayload({
        ...auditOptions,
        recallMode: auditOptions.recallMode ?? "full",
        budgetPlan: allocationPlan,
        selectionTrace: budgetedSelection.trace,
        queryAnalysis,
      }),
    };
    store.auditRepo.recordRetrieval({
      auditId,
      agentId: ctx.agentId,
      sessionKey: ctx.sessionKey,
      scope: ctx.scopes.join(","),
      routeType: bundle.routeType,
      queryText: query,
      queryHash: stableHash([query]),
      selectedItemsJson: selected,
      injectedChars: bundle.renderedBlock.length,
      createdAt: ctx.now,
    });
  }

  return bundle;
}

function sentenceWindowForPromptEvidence(
  text: string,
  anchors: string[],
  maxChars: number,
): string {
  const cleaned = text.trim();
  if (cleaned.length <= maxChars) {
    return cleaned;
  }
  const sentences = cleaned
    .split(/(?<=[.!?])\s+|\n+/u)
    .map((entry) => entry.trim())
    .filter(Boolean);
  const usefulAnchors = informativePromptAnchors(anchors);
  const candidates = sentences.length > 0 ? sentences : [cleaned];
  const best = candidates
    .map((sentence) => ({
      sentence,
      score:
        usefulAnchors.length > 0
          ? informativePromptAnchorSupport(sentence, usefulAnchors)
          : semanticTextSimilarity(sentence, cleaned),
    }))
    .sort((left, right) => right.score - left.score)[0];
  const selected = best?.sentence ?? cleaned;
  if (selected.length <= maxChars) {
    return selected;
  }
  const normalizedSelected = normalizeText(selected);
  const anchor = usefulAnchors.find((entry) => normalizedSelected.includes(normalizeText(entry)));
  const anchorIndex = anchor ? selected.toLowerCase().indexOf(anchor.toLowerCase()) : -1;
  if (anchorIndex >= 0) {
    const start = Math.max(0, anchorIndex - Math.floor(maxChars * 0.45));
    return truncateText(selected.slice(start).trim(), maxChars);
  }
  return truncateText(selected, maxChars);
}

function promptEvidenceLineText(entry: PromptEvidenceCandidate, maxChars: number): string {
  return sentenceWindowForPromptEvidence(entry.text, entry.excerptAnchors ?? [], maxChars);
}

function packetPromptEvidenceEntry(
  bundle: EvidenceBundle,
  packet: EvidencePacket,
): PromptEvidenceCandidate | undefined {
  const sourceRefs = new Set(packet.sourceRefs);
  return bundle.promptEvidence.find((entry) => {
    if (entry.dropReason) {
      return false;
    }
    if (
      entry.filledSlotIds &&
      entry.filledSlotIds.length > 0 &&
      !entry.filledSlotIds.includes(packet.slotId)
    ) {
      return false;
    }
    return [entry.sourceRef, ...(entry.mergedSourceRefs ?? [])].some(
      (sourceRef) => sourceRef && sourceRefs.has(sourceRef),
    );
  });
}

function evidencePacketRenderLine(packet: EvidencePacket, bundle: EvidenceBundle): string {
  const slot = `[slot:${packet.slotId}]`;
  const date = packet.resolvedDate ?? packet.observedAt?.slice(0, 10);
  const datePart = date ? ` [date:${date}]` : "";
  const quantity =
    typeof packet.quantityHint === "number"
      ? ` [quantity:${packet.quantityHint}${packet.unitHint ? ` ${packet.unitHint}` : ""}]`
      : "";
  const matchingEntry = packetPromptEvidenceEntry(bundle, packet);
  const primaryText = matchingEntry
    ? promptEvidenceLineText(matchingEntry, 360)
    : sentenceWindowForPromptEvidence(packet.primaryText, bundle.queryAnchors ?? [], 360);
  return `- ${slot}${datePart}${quantity} ${primaryText}`;
}

export function renderEvidenceBundle(bundle: EvidenceBundle, maxChars: number): string {
  const effectiveBudget = Math.max(
    220,
    Math.min(maxChars, bundle.budgetPlan?.totalPromptChars ?? maxChars),
  );
  const promptEvidenceBudget = Math.max(360, Math.floor(effectiveBudget * 0.86));
  const packetEvidenceLines = bundle.evidencePackets
    .filter((packet) => packet.injected && !packet.dropReason)
    .sort(
      (left, right) =>
        (right.grade?.finalScore ?? right.coverage.confidence) -
        (left.grade?.finalScore ?? left.coverage.confidence),
    )
    .slice(0, 6)
    .flatMap((packet) => {
      const displayLines = (packet.displayLines ?? []).filter((line) => line.trim().length > 0);
      if (displayLines.length > 0) {
        return displayLines.map((line) =>
          line.trim().startsWith("-") ? line.trim() : `- ${line.trim()}`,
        );
      }
      return [evidencePacketRenderLine(packet, bundle)];
    });

  const sections = [
    renderBudgetedSection("Priority Evidence", packetEvidenceLines, promptEvidenceBudget),
  ].filter(Boolean);
  if (sections.length === 0) {
    return "";
  }

  const rendered = ["## Memory Context", ...sections].join("\n\n").trim();
  return rendered.length <= effectiveBudget ? rendered : truncateText(rendered, effectiveBudget);
}
