import type { MemxStoreBundle } from "../runtime.js";
import { clamp01, normalizeName, normalizeText, stableHash, truncateText } from "../support.js";
import type {
  CandidateHit,
  CandidateSurface,
  EvidenceMatchType,
  EvidencePlanLayer,
  LayerCandidateHit,
  MemoryOperationContext,
  NormalizedFact,
  NormalizedEntity,
  QueryCompileResult,
  QueryEvidenceGoal,
  QueryEvidenceSlot,
  QuerySemanticBridge,
  QuerySemanticBridgeMatch,
  RetrievalSearchParams,
  SearchHit,
} from "../types.js";
import { isSnapshotFactualStateKey } from "./authority.js";
import { buildEntityMention, resolveEntityMention } from "./entityResolver.js";
import { capScoreByEvidenceCoverage, evidenceCoverageForText } from "./evidenceCoverage.js";
import { sourceRefsFromMaintenanceMetadata, uniqueMaintenanceRefs } from "./maintenanceContract.js";
import { describeStateValue, formatFactLine, lineageFromMetadata } from "./memoryObjectsHelpers.js";
import { semanticTextSimilarity } from "./semantic/textSimilarity.js";
import { queryAnchorSupport } from "./semantics.js";
import { normalizeSourceRefs } from "./sourceRefs.js";
import {
  stateCurrentnessFromVectorMetadata,
  stateCurrentnessToMetadata,
  stateCurrentnessVectorMetadata,
} from "./stateLifecycle.js";

export type CandidateGenerationSurfaceStats = {
  rawCount: number;
  filteredCount: number;
  alternateCount: number;
  topN: number;
  backendMix: Array<"embedding" | "fts" | "hybrid" | "lexical" | "repo">;
};

export type CandidateGenerationSlotLayerStats = {
  slotId: string;
  layer: EvidencePlanLayer;
  rawCount: number;
  selectedCount: number;
  alternateCount: number;
  topCandidateIds: string[];
};

export type CandidateGenerationResult = {
  candidates: CandidateHit[];
  slotCandidates: CandidateHit[];
  bridgeCandidates: CandidateHit[];
  searchHits: SearchHit[];
  alternateSearchHits: SearchHit[];
  surfaceStats: Partial<Record<CandidateSurface, CandidateGenerationSurfaceStats>>;
  slotLayerStats: CandidateGenerationSlotLayerStats[];
  layerCandidates: LayerCandidateHit[];
  budgets: Partial<Record<CandidateSurface, number>>;
};

const CANDIDATE_GENERATION_CUTOVER_CRITERIA = {
  invariantRegressionMustBeZero: true,
  lineageCompletenessRequired: true,
  budgetAuditCompletenessRequired: true,
  embeddingOffFallbackMustPass: true,
  maxSelectedEvidenceDiffRate: 0.2,
} as const;

const SURFACES: CandidateSurface[] = [
  "state",
  "fact",
  "event",
  "task",
  "chunk",
  "graph",
  "entity_alias",
] as const;

type GoalMatch = NonNullable<CandidateHit["goalMatches"]>[number];
type SlotMatch = NonNullable<CandidateHit["slotMatches"]>[number];
type BridgeMatch = QuerySemanticBridgeMatch;
type GoalScoredSearchHit = SearchHit & {
  goalMatches?: GoalMatch[];
  slotMatches?: SlotMatch[];
  bridgeMatches?: BridgeMatch[];
};

const hybridSearchCache = new WeakMap<MemoryOperationContext, Map<string, Promise<SearchHit[]>>>();

function hybridSearchCacheKey(params: RetrievalSearchParams): string {
  return JSON.stringify({
    agentId: params.agentId,
    scopes: params.scopes,
    limit: params.limit,
    query: params.query,
    readEpoch: params.readEpoch ?? null,
    docKinds: params.docKinds ?? [],
    docTypes: params.docTypes ?? [],
  });
}

function cachedHybridSearch(
  store: MemxStoreBundle,
  ctx: MemoryOperationContext,
  params: RetrievalSearchParams,
): Promise<SearchHit[]> {
  let cache = hybridSearchCache.get(ctx);
  if (!cache) {
    cache = new Map();
    hybridSearchCache.set(ctx, cache);
  }
  const key = hybridSearchCacheKey(params);
  const cached = cache.get(key);
  if (cached) {
    return cached;
  }
  const promise = store.retrievalBackend.hybridSearch(params);
  cache.set(key, promise);
  return promise;
}

function evidenceSlotRequiredRole(
  slot: QueryEvidenceSlot,
): NonNullable<QueryEvidenceSlot["requiredRole"]> | undefined {
  if (slot.requiredRole) {
    return slot.requiredRole;
  }
  if (slot.id === "current_need") {
    return "query_context";
  }
  if (slot.id === "relevant_user_resources") {
    return "user_resource";
  }
  if (slot.id === "prior_advice_or_strategy") {
    return "prior_advice";
  }
  if (slot.role === "answer_value" || slot.role === "answer_evidence") {
    return "answer_value";
  }
  if (
    slot.role === "query_context" ||
    slot.role === "user_resource" ||
    slot.role === "prior_advice" ||
    slot.role === "answer_event" ||
    slot.role === "time_constraint"
  ) {
    return slot.role;
  }
  return undefined;
}

function slotMatchType(slot: QueryEvidenceSlot, matchedQuery: string): EvidenceMatchType {
  const normalizedQuery = normalizeText(matchedQuery);
  const queryMatches = (values: string[] | undefined) =>
    (values ?? []).some((value) => {
      const normalized = normalizeText(value);
      return (
        normalized.length > 0 &&
        (normalizedQuery.includes(normalized) || normalized.includes(normalizedQuery))
      );
    });
  const role = evidenceSlotRequiredRole(slot);
  if (role === "query_context" || role === "time_constraint") {
    return "context";
  }
  if (queryMatches(slot.capabilityQueries)) {
    return "capability";
  }
  if (queryMatches(slot.relationHints)) {
    return "relation";
  }
  if (queryMatches(slot.subjectHints)) {
    return "subject";
  }
  if (role === "answer_value" || role === "answer_event") {
    return "answer_source";
  }
  if (role === "user_resource") {
    return "capability";
  }
  return "context";
}

function informativeAnchorSupport(text: string, anchors: string[]): number {
  if (anchors.length === 0) {
    return 0;
  }
  const normalizedText = normalizeText(text);
  let best = 0;
  for (const anchor of anchors) {
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

function retrievalScopeForSurface(surface: CandidateSurface): {
  docKinds?: Array<"fact" | "event" | "edge" | "state">;
  docTypes?: string[];
} {
  switch (surface) {
    case "state":
      return {
        docKinds: ["state"],
        docTypes: ["state"],
      };
    case "fact":
      return {
        docKinds: ["fact"],
        docTypes: ["fact"],
      };
    case "event":
      return {
        docKinds: ["event"],
        docTypes: ["event"],
      };
    case "task":
      return {
        docKinds: ["state"],
        docTypes: ["task"],
      };
    case "chunk":
    case "snippet":
      return {
        docKinds: ["event"],
        docTypes: ["chunk", "source_segment"],
      };
    case "graph":
      return {
        docKinds: ["edge"],
        docTypes: ["edge"],
      };
    case "entity_alias":
      return {};
  }
}

function surfaceTypeMatches(surface: CandidateSurface, hit: SearchHit): boolean {
  const docType =
    typeof hit.metadata.memxDocType === "string" ? hit.metadata.memxDocType : undefined;
  switch (surface) {
    case "state":
      return docType === "state";
    case "fact":
      return docType === "fact";
    case "event":
      return docType === "event";
    case "task":
      return docType === "task";
    case "chunk":
    case "snippet":
      return docType === "chunk" || docType === "source_segment";
    case "graph":
      return docType === "edge";
    case "entity_alias":
      return false;
  }
}

type CandidateRetention = "primary" | "alternate" | "drop";

function evaluateHitRetention(
  surface: CandidateSurface,
  hit: SearchHit,
  compiled: QueryCompileResult,
  ctx: MemoryOperationContext,
): CandidateRetention {
  if (!surfaceTypeMatches(surface, hit)) {
    return "drop";
  }
  if (
    surface === "fact" &&
    compiled.queryShape.evidenceNeed !== "workflow_context" &&
    hit.metadata.predicate === "has_workflow_guidance"
  ) {
    return "alternate";
  }
  if (
    compiled.evidencePlan?.operation.type === "tailor_advice" &&
    hasStructuredResourceBridge(hit.metadata)
  ) {
    const resourceSlotIds = new Set(
      (compiled.evidencePlan.slots ?? []).filter(isUserResourceSlot).map((slot) => slot.id),
    );
    const bestResourceSlotScore = Math.max(
      0,
      ...((hit as GoalScoredSearchHit).slotMatches ?? [])
        .filter((match) => resourceSlotIds.has(match.slotId))
        .map((match) => match.score),
      ...((hit as GoalScoredSearchHit).bridgeMatches ?? [])
        .filter((match) => match.role === "user_resource")
        .map((match) => match.score),
    );
    if (resourceSlotIds.size > 0 && bestResourceSlotScore < 0.15) {
      return "alternate";
    }
  }
  if (surface === "state") {
    const stateCurrentness = stateCurrentnessFromVectorMetadata(hit.metadata, ctx.now);
    if (stateCurrentness?.hardExclusions.length) {
      return "drop";
    }
    if (stateCurrentness && stateCurrentness.currentnessScore < 0.35) {
      return "alternate";
    }
  }
  if (ctx.sessionKey && compiled.queryShape.evidenceNeed === "workflow_context") {
    const sessionKey =
      typeof hit.metadata.sessionKey === "string" ? hit.metadata.sessionKey : undefined;
    if (
      sessionKey &&
      sessionKey !== ctx.sessionKey &&
      (surface === "task" || surface === "chunk" || surface === "event")
    ) {
      return "alternate";
    }
  }
  const anchors = compiled.anchors.filter((anchor) => anchor.trim().length > 0);
  if (anchors.length > 0 && compiled.queryShape.referentialMode === "deictic") {
    const anchorSupport = queryAnchorSupport(hit.text, anchors);
    if (
      anchorSupport < 0.22 &&
      (surface === "task" || surface === "chunk" || surface === "event")
    ) {
      return compiled.evidenceFidelity === "high" ? "alternate" : "drop";
    }
    if (
      anchorSupport < 0.52 &&
      (surface === "task" || surface === "chunk" || surface === "event")
    ) {
      return "alternate";
    }
  }
  if (compiled.queryShape.timeframe === "current") {
    if (surface === "state") {
      const stateKey =
        typeof hit.metadata.stateKey === "string" ? hit.metadata.stateKey : undefined;
      if (
        stateKey &&
        !isSnapshotFactualStateKey(stateKey) &&
        compiled.queryShape.evidenceNeed !== "workflow_context"
      ) {
        return "alternate";
      }
      return hit.metadata.currentnessHint === "historical" ? "alternate" : "primary";
    }
    if (surface === "fact") {
      return hit.metadata.supersededHint === true ? "alternate" : "primary";
    }
  }
  if (compiled.queryShape.timeframe === "historical") {
    if (surface === "state") {
      return "drop";
    }
  }
  return "primary";
}

function deriveSurfaceBudgets(
  compiled: QueryCompileResult,
  ctx: MemoryOperationContext,
): Partial<Record<CandidateSurface, number>> {
  const max = ctx.config.advanced.candidateSurfaceBudgets;
  const budgets: Partial<Record<CandidateSurface, number>> = {};
  const ambiguityBoost = compiled.ambiguityLevel >= 0.58 ? 1 : 0;
  const fidelityBoost = compiled.evidenceFidelity === "high" ? 1 : 0;
  const supportBoost = compiled.supportNeed >= 0.72 ? 1 : 0;
  const tailorAdvice = compiled.evidencePlan?.operation.type === "tailor_advice";
  for (const surface of compiled.candidateSurfaces) {
    switch (surface) {
      case "state":
        budgets.state =
          compiled.queryShape.timeframe === "current" ? max.state : Math.min(2, max.state);
        break;
      case "fact":
        budgets.fact =
          compiled.queryShape.timeframe === "compare"
            ? Math.min(max.fact, 4 + ambiguityBoost + fidelityBoost)
            : compiled.queryShape.timeframe === "historical"
              ? Math.min(max.fact, 3 + ambiguityBoost + fidelityBoost)
              : Math.min(max.fact, 3 + supportBoost + ambiguityBoost);
        break;
      case "event":
        budgets.event = tailorAdvice
          ? Math.min(max.event, 4 + supportBoost + ambiguityBoost)
          : compiled.queryShape.timeframe === "compare" ||
              compiled.queryShape.timeframe === "historical"
            ? Math.min(max.event, 4 + fidelityBoost + ambiguityBoost)
            : Math.min(max.event, 2 + ambiguityBoost);
        break;
      case "task":
        budgets.task = tailorAdvice
          ? Math.min(max.task, 4 + supportBoost + ambiguityBoost)
          : compiled.queryShape.evidenceNeed === "workflow_context"
            ? Math.min(max.task, 2 + ambiguityBoost)
            : Math.min(2, max.task);
        break;
      case "chunk":
        budgets.chunk = tailorAdvice
          ? Math.min(max.chunk, 5 + supportBoost + ambiguityBoost)
          : compiled.evidenceFidelity === "high" || compiled.answerGranularity === "detail"
            ? Math.min(max.chunk, 4 + ambiguityBoost + supportBoost)
            : Math.min(max.chunk, 2 + supportBoost);
        break;
      case "graph":
        budgets.graph =
          compiled.queryShape.evidenceNeed === "relation"
            ? Math.min(max.graph, 2 + ambiguityBoost)
            : Math.min(2 + ambiguityBoost, max.graph);
        break;
      case "entity_alias":
        budgets.entity_alias =
          compiled.queryShape.evidenceNeed === "relation"
            ? Math.min(max.entityAlias, 1 + ambiguityBoost)
            : Math.min(1, max.entityAlias);
        break;
    }
  }
  return budgets;
}

function toCandidateHit(
  surface: CandidateSurface,
  hit: GoalScoredSearchHit,
  ctx: MemoryOperationContext,
  tier: CandidateHit["tier"],
): CandidateHit | null {
  const lineage = lineageFromMetadata(hit.metadata, {
    sourceKind: "vector_doc",
    sourceId: hit.docId,
  });
  if (!lineage) {
    return null;
  }
  const stateCurrentness =
    surface === "state" ? stateCurrentnessFromVectorMetadata(hit.metadata, ctx.now) : undefined;
  const metadata = stateCurrentness
    ? { ...hit.metadata, ...stateCurrentnessToMetadata(stateCurrentness) }
    : hit.metadata;
  return {
    candidateId: `${surface}:${hit.docId}`,
    surface,
    tier,
    text: hit.text,
    score: hit.score,
    retrievalBackend: hit.backend,
    docId: hit.docId,
    scope:
      typeof hit.metadata.scope === "string" ? hit.metadata.scope : (ctx.scopes[0] ?? "unknown"),
    agentId: ctx.agentId,
    confidence:
      typeof hit.metadata.confidence === "number" ? clamp01(hit.metadata.confidence) : undefined,
    activeHint: stateCurrentness
      ? stateCurrentness.hardExclusions.length === 0
      : hit.metadata.activeHint === true,
    supersededHint: stateCurrentness
      ? stateCurrentness.hardExclusions.includes("superseded-state")
      : hit.metadata.supersededHint === true,
    currentnessHint:
      metadata.currentnessHint === "current" ||
      metadata.currentnessHint === "historical" ||
      metadata.currentnessHint === "compare"
        ? metadata.currentnessHint
        : "unknown",
    lineage,
    goalMatches: hit.goalMatches,
    slotMatches: hit.slotMatches,
    bridgeMatches: hit.bridgeMatches,
    metadata,
  };
}

function mergeCandidateHits(hits: CandidateHit[]): CandidateHit[] {
  const byId = new Map<string, CandidateHit>();
  for (const hit of hits) {
    const key = hit.docId ? `${hit.surface}:${hit.docId}` : hit.candidateId;
    const existing = byId.get(key);
    if (!existing) {
      byId.set(key, hit);
      continue;
    }
    byId.set(key, {
      ...existing,
      ...hit,
      tier: existing.tier === "primary" || hit.tier === "primary" ? "primary" : "alternate",
      score: Math.max(existing.score, hit.score),
      confidence:
        typeof existing.confidence === "number" || typeof hit.confidence === "number"
          ? Math.max(existing.confidence ?? 0, hit.confidence ?? 0)
          : undefined,
      goalMatches: mergeGoalMatches(existing.goalMatches, hit.goalMatches),
      slotMatches: mergeSlotMatches(existing.slotMatches, hit.slotMatches),
      bridgeMatches: mergeBridgeMatches(existing.bridgeMatches, hit.bridgeMatches),
    });
  }
  return [...byId.values()];
}

function mergeGoalMatches(left?: GoalMatch[], right?: GoalMatch[]): GoalMatch[] | undefined {
  const matches = [...(left ?? []), ...(right ?? [])];
  if (matches.length === 0) {
    return undefined;
  }
  const byKey = new Map<string, GoalMatch>();
  for (const match of matches) {
    const key = `${match.goal}\n${match.matchedQuery}`;
    const current = byKey.get(key);
    if (!current || match.score > current.score) {
      byKey.set(key, match);
    }
  }
  return [...byKey.values()].sort((left, right) => right.score - left.score).slice(0, 4);
}

function mergeSlotMatches(left?: SlotMatch[], right?: SlotMatch[]): SlotMatch[] | undefined {
  const matches = [...(left ?? []), ...(right ?? [])];
  if (matches.length === 0) {
    return undefined;
  }
  const byKey = new Map<string, SlotMatch>();
  for (const match of matches) {
    const key = `${match.slotId}\n${match.layer}\n${match.matchedQuery}`;
    const current = byKey.get(key);
    if (!current || match.score > current.score) {
      byKey.set(key, match);
    }
  }
  return [...byKey.values()].sort((left, right) => right.score - left.score).slice(0, 6);
}

function mergeBridgeMatches(
  left?: BridgeMatch[],
  right?: BridgeMatch[],
): BridgeMatch[] | undefined {
  const matches = [...(left ?? []), ...(right ?? [])];
  if (matches.length === 0) {
    return undefined;
  }
  const byKey = new Map<string, BridgeMatch>();
  for (const match of matches) {
    const key = `${match.bridgeId}\n${match.matchedQuery}`;
    const current = byKey.get(key);
    if (!current || match.score > current.score) {
      byKey.set(key, match);
    }
  }
  return [...byKey.values()].sort((left, right) => right.score - left.score).slice(0, 8);
}

function dedupeSearchHits<T extends SearchHit>(hits: T[]): T[] {
  const byDocId = new Map<string, T>();
  for (const hit of hits) {
    const existing = byDocId.get(hit.docId);
    if (!existing) {
      byDocId.set(hit.docId, hit);
      continue;
    }
    const merged = {
      ...(hit.score > existing.score ? hit : existing),
      goalMatches: mergeGoalMatches(
        (existing as GoalScoredSearchHit).goalMatches,
        (hit as GoalScoredSearchHit).goalMatches,
      ),
      slotMatches: mergeSlotMatches(
        (existing as GoalScoredSearchHit).slotMatches,
        (hit as GoalScoredSearchHit).slotMatches,
      ),
      bridgeMatches: mergeBridgeMatches(
        (existing as GoalScoredSearchHit).bridgeMatches,
        (hit as GoalScoredSearchHit).bridgeMatches,
      ),
      score: Math.max(existing.score, hit.score),
    } as T;
    byDocId.set(hit.docId, merged);
  }
  return [...byDocId.values()];
}

function meaningfulCandidateAnchors(compiled: QueryCompileResult): string[] {
  return compiled.anchors
    .map((anchor) => anchor.trim())
    .filter((anchor) => normalizeText(anchor).length >= 4)
    .slice(0, 2);
}

function rerankSurfaceHits(
  hits: GoalScoredSearchHit[],
  surface: CandidateSurface,
  compiled: QueryCompileResult,
): GoalScoredSearchHit[] {
  if (hits.length <= 1) {
    return hits;
  }
  if (
    surface !== "fact" &&
    surface !== "event" &&
    surface !== "task" &&
    surface !== "chunk" &&
    surface !== "graph"
  ) {
    return hits;
  }
  const anchors = meaningfulCandidateAnchors(compiled);
  if (anchors.length === 0) {
    return hits;
  }
  const detailSensitive =
    compiled.answerGranularity === "detail" || compiled.evidenceFidelity === "high";
  return [...hits].sort((left, right) => {
    const rightGoal = topGoalMatchScore(right);
    const leftGoal = topGoalMatchScore(left);
    if (rightGoal !== leftGoal) {
      return rightGoal - leftGoal;
    }
    const rightSlot = topSlotMatchScore(right);
    const leftSlot = topSlotMatchScore(left);
    if (rightSlot !== leftSlot) {
      return rightSlot - leftSlot;
    }
    const rightBridge = topBridgeMatchScore(right);
    const leftBridge = topBridgeMatchScore(left);
    if (rightBridge !== leftBridge) {
      return rightBridge - leftBridge;
    }
    const rightAnchor = queryAnchorSupport(right.text, anchors);
    const leftAnchor = queryAnchorSupport(left.text, anchors);
    if (rightAnchor !== leftAnchor) {
      return rightAnchor - leftAnchor;
    }
    const rightCurrentPenalty =
      surface === "fact" &&
      compiled.queryShape.timeframe === "current" &&
      right.metadata.currentnessHint === "historical"
        ? 0.08
        : 0;
    const leftCurrentPenalty =
      surface === "fact" &&
      compiled.queryShape.timeframe === "current" &&
      left.metadata.currentnessHint === "historical"
        ? 0.08
        : 0;
    const rightScore =
      right.score - rightCurrentPenalty + (detailSensitive ? rightAnchor * 0.14 : 0);
    const leftScore = left.score - leftCurrentPenalty + (detailSensitive ? leftAnchor * 0.14 : 0);
    if (rightScore !== leftScore) {
      return rightScore - leftScore;
    }
    return right.score - left.score;
  });
}

function topGoalMatchScore(hit: GoalScoredSearchHit): number {
  return Math.max(0, ...(hit.goalMatches ?? []).map((match) => match.score));
}

function topSlotMatchScore(hit: GoalScoredSearchHit): number {
  return Math.max(0, ...(hit.slotMatches ?? []).map((match) => match.score));
}

function topBridgeMatchScore(hit: GoalScoredSearchHit): number {
  return Math.max(0, ...(hit.bridgeMatches ?? []).map((match) => match.score));
}

function dedupeCandidateHitsById(hits: CandidateHit[]): CandidateHit[] {
  const byId = new Map<string, CandidateHit>();
  for (const hit of hits) {
    const existing = byId.get(hit.candidateId);
    if (!existing || hit.score > existing.score) {
      byId.set(hit.candidateId, hit);
    }
  }
  return [...byId.values()];
}

function entityExpansionTextMatches(entity: NormalizedEntity, text: string): boolean {
  const haystack = normalizeText(text);
  if (!haystack) {
    return false;
  }
  const aliases = [entity.canonicalName, entity.normalizedName, ...entity.aliases]
    .map((value) => normalizeText(value))
    .filter(Boolean);
  return aliases.some((alias) => alias.length > 0 && haystack.includes(alias));
}

function factSupportText(fact: NormalizedFact): string | undefined {
  const assertion =
    fact.objectValueJson?.semanticAssertion &&
    typeof fact.objectValueJson.semanticAssertion === "object"
      ? (fact.objectValueJson.semanticAssertion as Record<string, unknown>)
      : undefined;
  const assertionSupportText =
    typeof assertion?.supportText === "string" ? assertion.supportText.trim() : "";
  if (assertionSupportText) {
    return assertionSupportText;
  }
  const directSupportText =
    typeof fact.objectValueJson?.supportText === "string" ? fact.objectValueJson.supportText.trim() : "";
  return directSupportText || fact.provenanceText?.trim() || undefined;
}

function factSupportSourceRefs(fact: NormalizedFact): string[] {
  return uniqueMaintenanceRefs([
    ...sourceRefsFromMaintenanceMetadata(fact.objectValueJson),
    fact.sourceRef,
  ]);
}

function factCandidatesForEntity(
  store: MemxStoreBundle,
  ctx: MemoryOperationContext,
  entity: NormalizedEntity,
  limit: number,
): CandidateHit[] {
  const subjects = [
    entity.normalizedName,
    normalizeName(entity.canonicalName),
    ...entity.aliases.map((alias) => normalizeName(alias)),
  ].filter(Boolean);
  const facts = new Map<string, ReturnType<typeof store.factRepo.findActiveBySubject>[number]>();
  for (const scope of ctx.scopes) {
    for (const subject of new Set(subjects)) {
      for (const fact of store.factRepo.findActiveBySubject({
        agentId: ctx.agentId,
        scope,
        canonicalSubject: subject,
      })) {
        facts.set(fact.factId, fact);
      }
    }
  }
  return [...facts.values()]
    .sort(
      (left, right) =>
        right.confidence - left.confidence ||
        Date.parse(right.updatedAt) - Date.parse(left.updatedAt),
    )
    .slice(0, limit)
    .map((fact) => {
      const sourceRefs = factSupportSourceRefs(fact);
      const sourceRef = sourceRefs[0] ?? fact.sourceRef;
      const supportText = factSupportText(fact);
      return {
        candidateId: `fact:${fact.factId}:entity-expansion`,
        surface: "fact",
        tier: "primary",
        text: formatFactLine({
          subject: entity.canonicalName,
          predicate: fact.predicate,
          object: fact.canonicalObject,
          objectValueJson: fact.objectValueJson,
          status: fact.status,
        }),
        score: clamp01(fact.confidence * 0.72 + entity.confidence * 0.18 + 0.1),
        retrievalBackend: "repo",
        docId: fact.factId,
        scope: fact.scope,
        agentId: ctx.agentId,
        confidence: fact.confidence,
        currentnessHint: fact.status === "active" ? "current" : "unknown",
        lineage: {
          sourceKind: "fact",
          sourceId: fact.factId,
          sourceRef,
          canonicalKind: "fact",
          canonicalId: fact.factId,
          ...(typeof fact.materializedEpoch === "number"
            ? { materializedEpoch: fact.materializedEpoch }
            : {}),
        },
        metadata: {
          entityExpansion: true,
          expandedEntityId: entity.entityId,
          sourceRef,
          sourceRefs,
          supportRefs: sourceRefs,
          supportContentRefs: sourceRefs,
          ...(supportText ? { supportText } : {}),
          canonicalSubject: fact.canonicalSubject,
          ...(fact.canonicalObject ? { canonicalObject: fact.canonicalObject } : {}),
          predicate: fact.predicate,
          currentnessHint: fact.status === "active" ? "current" : "unknown",
          memxDocType: "fact",
        },
      };
    });
}

function eventCandidatesForEntity(
  store: MemxStoreBundle,
  ctx: MemoryOperationContext,
  entity: NormalizedEntity,
  limit: number,
): CandidateHit[] {
  const events = new Map<string, ReturnType<typeof store.eventRepo.search>[number]>();
  const queries = uniqueSearchQueries([entity.canonicalName, ...entity.aliases], 4);
  for (const query of queries) {
    for (const event of store.eventRepo.search({
      agentId: ctx.agentId,
      scopes: ctx.scopes,
      text: query,
      limit: Math.max(limit, 3),
      readEpoch: ctx.readEpoch,
    })) {
      events.set(event.eventId, event);
    }
  }
  return [...events.values()]
    .filter((event) => entityExpansionTextMatches(entity, event.text))
    .sort(
      (left, right) =>
        right.confidence - left.confidence ||
        Date.parse(right.observedAt) - Date.parse(left.observedAt),
    )
    .slice(0, limit)
    .map((event) => ({
      candidateId: `event:${event.eventId}:entity-expansion`,
      surface: "event",
      tier: "primary",
      text: event.text,
      score: clamp01(
        event.confidence * 0.54 +
          semanticTextSimilarity(entity.canonicalName, event.text) * 0.34 +
          0.12,
      ),
      retrievalBackend: "repo",
      docId: event.eventId,
      scope: event.scope,
      agentId: ctx.agentId,
      confidence: event.confidence,
      lineage: {
        sourceKind: "event",
        sourceId: event.eventId,
        sourceRef: event.sourceRef,
        canonicalKind: "event",
        canonicalId: event.eventId,
        ...(typeof event.materializedEpoch === "number"
          ? { materializedEpoch: event.materializedEpoch }
          : {}),
      },
      metadata: {
        entityExpansion: true,
        expandedEntityId: entity.entityId,
        sourceRef: event.sourceRef,
        observedAt: event.observedAt,
        memxDocType: "event",
      },
    }));
}

function stateCandidatesForEntity(
  store: MemxStoreBundle,
  ctx: MemoryOperationContext,
  entity: NormalizedEntity,
  limit: number,
): CandidateHit[] {
  const states = store.stateRepo
    .get({
      agentId: ctx.agentId,
      scopes: ctx.scopes,
      now: ctx.now,
      readEpoch: ctx.readEpoch,
    })
    .filter((state) =>
      entityExpansionTextMatches(
        entity,
        `${state.key} ${JSON.stringify(state.valueJson)} ${describeStateValue(state.key, state.valueJson)}`,
      ),
    )
    .sort(
      (left, right) =>
        right.confidence - left.confidence ||
        Date.parse(right.updatedAt) - Date.parse(left.updatedAt),
    )
    .slice(0, limit);
  return states.flatMap((state) => {
    const currentness = stateCurrentnessVectorMetadata(state, ctx.now);
    const expandedSupportRefs = uniqueMaintenanceRefs([
      ...(Array.isArray(currentness.stateSupportRefs)
        ? currentness.stateSupportRefs.filter(
            (entry): entry is string => typeof entry === "string" && entry.trim().length > 0,
          )
        : []),
      ...sourceRefsForMaintenanceObjectRef(store, state.sourceRef),
    ]);
    const enrichedCurrentness = {
      ...currentness,
      ...(expandedSupportRefs.length > 0
        ? {
            stateSupportRefs: expandedSupportRefs,
            sourceRefsForExpansion: expandedSupportRefs,
          }
        : {}),
    };
    const stateCurrentnessScore =
      typeof currentness.stateCurrentnessScore === "number"
        ? currentness.stateCurrentnessScore
        : 0.45;
    const hardExclusions = Array.isArray(currentness.stateCurrentnessHardExclusions)
      ? currentness.stateCurrentnessHardExclusions
      : [];
    if (hardExclusions.length > 0) {
      return [];
    }
    return [
      {
        candidateId: `state:${state.key}:entity-expansion`,
        surface: "state" as const,
        tier: "primary" as const,
        text: `${state.key}: ${describeStateValue(state.key, state.valueJson)}`.trim(),
        score: clamp01(
          state.confidence * 0.4 +
            entity.confidence * 0.18 +
            stateCurrentnessScore * 0.34 +
            0.08,
        ),
        retrievalBackend: "repo" as const,
        docId: state.key,
        scope: state.scope,
        agentId: ctx.agentId,
        confidence: state.confidence,
        currentnessHint:
          currentness.currentnessHint === "current" ||
          currentness.currentnessHint === "historical" ||
          currentness.currentnessHint === "compare"
            ? currentness.currentnessHint
            : "unknown",
        activeHint: currentness.activeHint === true,
        supersededHint: currentness.supersededHint === true,
        lineage: {
          sourceKind: "state",
          sourceId: state.key,
          sourceRef: state.sourceRef,
          canonicalKind: "state",
          canonicalId: state.key,
          ...(typeof state.materializedEpoch === "number"
            ? { materializedEpoch: state.materializedEpoch }
            : {}),
        },
        metadata: {
          entityExpansion: true,
          expandedEntityId: entity.entityId,
          sourceRef: state.sourceRef,
          stateKind: state.stateKind,
          memxDocType: "state",
          stateKey: state.key,
          ...enrichedCurrentness,
        },
      },
    ];
  });
}

function sourceRefsForMaintenanceObjectRef(
  store: MemxStoreBundle,
  ref: string | undefined,
): string[] {
  if (!ref) {
    return [];
  }
  if (ref.startsWith("abstraction_candidate:")) {
    const candidateId = ref.slice("abstraction_candidate:".length);
    const candidate = store.abstractionRepo.getById(candidateId);
    return candidate
      ? uniqueMaintenanceRefs([
          ...sourceRefsFromMaintenanceMetadata(candidate.metadataJson),
          ...candidate.supportContentRefs,
        ])
      : [];
  }
  if (ref.startsWith("fact:")) {
    const fact = store.factRepo.get(ref.slice("fact:".length));
    return fact
      ? uniqueMaintenanceRefs([
          ...sourceRefsFromMaintenanceMetadata(fact.objectValueJson),
          fact.sourceRef,
        ])
      : [];
  }
  if (ref.startsWith("event:")) {
    const event = store.eventRepo.get(ref.slice("event:".length));
    return event ? uniqueMaintenanceRefs([event.sourceRef]) : [];
  }
  return [];
}

function graphCandidatesForEntities(
  store: MemxStoreBundle,
  ctx: MemoryOperationContext,
  entities: NormalizedEntity[],
  limit: number,
): CandidateHit[] {
  if (entities.length === 0) {
    return [];
  }
  const graph = store.graphRepo.expandNeighborhood({
    agentId: ctx.agentId,
    scopes: ctx.scopes,
    seedEntityIds: entities.map((entity) => entity.entityId),
    maxHops: 1,
    maxEdges: Math.max(limit * 2, 8),
    maxNodes: Math.max(limit * 3, 12),
    now: ctx.now,
    readEpoch: ctx.readEpoch,
  });
  const nodes = new Map(graph.nodes.map((node) => [node.nodeId, node]));
  const seedIds = new Set(entities.map((entity) => entity.entityId));
  return graph.edges.slice(0, limit).map((edge, index) => {
    const src = nodes.get(edge.srcNodeId)?.name ?? edge.srcNodeId;
    const dst = nodes.get(edge.dstNodeId)?.name ?? edge.dstNodeId;
    const relation = edge.relationSlot ? `${edge.relType}[${edge.relationSlot}]` : edge.relType;
    const edgeMetadata = edge.metadata ?? {};
    const sourceRefsForExpansion = uniqueMaintenanceRefs([
      ...sourceRefsFromMaintenanceMetadata(edgeMetadata),
      edge.evidenceRef,
      ...sourceRefsForMaintenanceObjectRef(store, edge.evidenceRef),
    ]);
    const seedBoost =
      seedIds.has(edge.srcEntityId ?? edge.srcNodeId) ||
      seedIds.has(edge.dstEntityId ?? edge.dstNodeId)
        ? 0.12
        : 0;
    return {
      candidateId: `graph:${edge.edgeId}:entity-expansion`,
      surface: "graph",
      tier: "primary",
      text: `${src} --${relation}--> ${dst}`,
      score: clamp01(edge.confidence * 0.72 + seedBoost + 0.08 - index * 0.015),
      retrievalBackend: "repo",
      docId: edge.edgeId,
      scope: ctx.scopes[0] ?? "unknown",
      agentId: ctx.agentId,
      confidence: edge.confidence,
      lineage: {
        sourceKind: "graph_edge",
        sourceId: edge.edgeId,
        sourceRef: edge.evidenceRef,
        canonicalKind: "graph_edge",
        canonicalId: edge.edgeId,
      },
      metadata: {
        ...edgeMetadata,
        entityExpansion: true,
        expandedEntityIds: entities.map((entity) => entity.entityId),
        sourceRef: edge.evidenceRef,
        sourceRefsForExpansion,
        supportRefs: sourceRefsForExpansion,
        supportContentRefs: sourceRefsForExpansion,
        relType: edge.relType,
        relationSlot: edge.relationSlot,
        memxDocType: "edge",
      },
    };
  });
}

function entityExpansionCandidates(
  store: MemxStoreBundle,
  ctx: MemoryOperationContext,
  entities: NormalizedEntity[],
  topN: number,
): CandidateHit[] {
  const uniqueEntities = [
    ...new Map(entities.map((entity) => [entity.entityId, entity])).values(),
  ].slice(0, Math.max(3, topN));
  const perEntityLimit = Math.max(2, Math.min(4, topN));
  const candidates = [
    ...graphCandidatesForEntities(store, ctx, uniqueEntities, Math.max(4, topN * 2)),
    ...uniqueEntities.flatMap((entity) =>
      factCandidatesForEntity(store, ctx, entity, perEntityLimit),
    ),
    ...uniqueEntities.flatMap((entity) =>
      eventCandidatesForEntity(store, ctx, entity, perEntityLimit),
    ),
    ...uniqueEntities.flatMap((entity) =>
      stateCandidatesForEntity(store, ctx, entity, perEntityLimit),
    ),
  ];
  return dedupeCandidateHitsById(candidates)
    .sort((left, right) => right.score - left.score)
    .slice(0, Math.max(8, topN * 6));
}

function queryEntitySearchQueries(compiled: QueryCompileResult, limit: number): string[] {
  return uniqueSearchQueries(
    compiled.queryEntities.flatMap((entity) => slotSubjectQueryVariants([entity.name])),
    limit,
  );
}

function resolveQueryEntities(
  store: MemxStoreBundle,
  ctx: MemoryOperationContext,
  compiled: QueryCompileResult,
  limit: number,
): NormalizedEntity[] {
  const querySourceRef = `query:${stableHash([compiled.queryText])}`;
  const entityQueries = queryEntitySearchQueries(compiled, limit);
  const resolvedEntities = entityQueries
    .map((query) =>
      resolveEntityMention(
        store,
        ctx,
        buildEntityMention({
          ctx,
          scope: ctx.scopes[0] ?? "unknown",
          rawText: query,
          semanticRole: "query",
          sourceRef: querySourceRef,
          supportText: compiled.queryText,
          observedAt: ctx.now,
          metadataJson: {
            queryText: compiled.queryText,
            focusedQuery: compiled.focusedQuery,
            generatedFrom: "query-entity-resolution",
          },
        }),
        { createIfMissing: false, persist: false },
      ),
    )
    .filter((result) => result.method !== "uncertain")
    .map((result) => result.entity);
  const linkedEntities = resolvedEntities
    .filter((entity): entity is NonNullable<typeof entity> => Boolean(entity))
    .flatMap((entity) =>
      store.graphRepo
        .linkedEntityIds({ entityId: entity.entityId, limit: 4 })
        .map((entityId) => store.graphRepo.getEntityById(entityId)),
    );
  const seen = new Set<string>();
  return [...resolvedEntities, ...linkedEntities]
    .filter((entity): entity is NormalizedEntity => Boolean(entity))
    .filter((entity) => {
      if (seen.has(entity.entityId)) {
        return false;
      }
      seen.add(entity.entityId);
      return true;
    });
}

function entityFromProfileHit(
  store: MemxStoreBundle,
  hit: SearchHit,
): NormalizedEntity | undefined {
  const entityId = typeof hit.metadata.entityId === "string" ? hit.metadata.entityId : undefined;
  const sourceId = typeof hit.metadata.sourceId === "string" ? hit.metadata.sourceId : undefined;
  return (
    store.graphRepo.getEntityById(
    entityId ?? sourceId ?? hit.docId.replace(/^entity_profile:/u, ""),
    ) ?? undefined
  );
}

async function semanticQueryEntityCandidates(
  store: MemxStoreBundle,
  ctx: MemoryOperationContext,
  compiled: QueryCompileResult,
  limit: number,
): Promise<NormalizedEntity[]> {
  const entities: NormalizedEntity[] = [];
  const queries = queryEntitySearchQueries(compiled, limit).slice(0, Math.max(2, limit));
  for (const query of queries) {
    const hits = ctx.config.advanced.enableEmbeddingCandidates
      ? await cachedHybridSearch(store, ctx, {
          agentId: ctx.agentId,
          scopes: ctx.scopes,
          query,
          limit: Math.max(4, Math.min(12, limit)),
          readEpoch: ctx.readEpoch,
          docKinds: ["entity_profile"],
          docTypes: ["entity_profile"],
        })
      : store.retrievalBackend.keywordSearch({
          agentId: ctx.agentId,
          scopes: ctx.scopes,
          query,
          limit: Math.max(4, Math.min(12, limit)),
          readEpoch: ctx.readEpoch,
          docKinds: ["entity_profile"],
          docTypes: ["entity_profile"],
        });
    for (const hit of hits.slice(0, 4)) {
      const entity = entityFromProfileHit(store, hit);
      if (entity) {
        entities.push(entity);
      }
    }
  }
  return entities;
}

async function resolveQueryEntitiesForRecall(
  store: MemxStoreBundle,
  ctx: MemoryOperationContext,
  compiled: QueryCompileResult,
  limit: number,
): Promise<NormalizedEntity[]> {
  const resolvedEntities = resolveQueryEntities(store, ctx, compiled, limit);
  const semanticEntities = await semanticQueryEntityCandidates(store, ctx, compiled, limit);
  const linkedEntities = [...resolvedEntities, ...semanticEntities]
    .flatMap((entity) =>
      store.graphRepo
        .linkedEntityIds({ entityId: entity.entityId, limit: 4 })
        .map((entityId) => store.graphRepo.getEntityById(entityId)),
    )
    .filter((entity): entity is NormalizedEntity => Boolean(entity));
  const seen = new Set<string>();
  return [...resolvedEntities, ...semanticEntities, ...linkedEntities]
    .filter((entity) => {
      if (seen.has(entity.entityId)) {
        return false;
      }
      seen.add(entity.entityId);
      return true;
    })
    .slice(0, Math.max(4, limit));
}

function selectSurfaceHitsForSlots(
  rankedHits: GoalScoredSearchHit[],
  topN: number,
  compiled: QueryCompileResult,
): GoalScoredSearchHit[] {
  const slots = compiled.evidencePlan?.slots ?? [];
  if (slots.length <= 1 || rankedHits.length <= topN) {
    return rankedHits.slice(0, topN);
  }
  const selected: GoalScoredSearchHit[] = [];
  const selectedDocIds = new Set<string>();
  for (const slot of slots) {
    const hit = rankedHits.find((candidate) => {
      if (selectedDocIds.has(candidate.docId)) {
        return false;
      }
      return (candidate.slotMatches ?? []).some(
        (match) => match.slotId === slot.id && match.score >= 0.2,
      );
    });
    if (!hit) {
      continue;
    }
    selected.push(hit);
    selectedDocIds.add(hit.docId);
    if (selected.length >= topN) {
      return selected;
    }
  }
  for (const hit of rankedHits) {
    if (selectedDocIds.has(hit.docId)) {
      continue;
    }
    selected.push(hit);
    selectedDocIds.add(hit.docId);
    if (selected.length >= topN) {
      break;
    }
  }
  return selected;
}

function lexicalSearchHits(
  store: MemxStoreBundle,
  ctx: MemoryOperationContext,
  compiled: QueryCompileResult,
  surface: CandidateSurface,
  limit: number,
): SearchHit[] {
  const retrievalScope = retrievalScopeForSurface(surface);
  const docs = store.vectorRepo.listDocs({
    agentId: ctx.agentId,
    scopes: ctx.scopes,
    limit: Math.max(limit * 6, 24),
    readEpoch: ctx.readEpoch,
    docKinds: retrievalScope.docKinds,
    docTypes: retrievalScope.docTypes,
  });
  const scored = docs
    .map((doc) => {
      const metadata = doc.metadataJson;
      const hit: SearchHit = {
        docId: doc.docId,
        text: doc.text,
        metadata,
        score: semanticTextSimilarity(compiled.focusedQuery, doc.text),
        backend: "lexical",
      };
      return hit;
    })
    .filter((hit) => surfaceTypeMatches(surface, hit))
    .filter((hit) => evaluateHitRetention(surface, hit, compiled, ctx) !== "drop")
    .filter((hit) => hit.score >= 0.08)
    .sort((left, right) => right.score - left.score);
  return scored.slice(0, limit);
}

function goalsForSurface(
  compiled: QueryCompileResult,
  surface: CandidateSurface,
): QueryEvidenceGoal[] {
  return (compiled.evidenceGoals ?? []).filter(
    (goal) =>
      goal.preferredSurfaces.includes(surface) || (surface === "chunk" && goal.fidelity === "high"),
  );
}

function requiredCoverageAnchors(compiled: QueryCompileResult): string[] {
  const seen = new Set<string>();
  const anchors: string[] = [];
  for (const anchor of compiled.evidenceCoverage?.requiredAnchors ?? []) {
    const trimmed = anchor.trim();
    const key = normalizeText(trimmed);
    if (!key || seen.has(key)) {
      continue;
    }
    seen.add(key);
    anchors.push(trimmed);
  }
  return anchors;
}

function requiredAnchorQueries(compiled: QueryCompileResult): string[] {
  const required = requiredCoverageAnchors(compiled);
  if (required.length === 0) {
    return [];
  }
  const optional = (compiled.evidenceCoverage?.optionalAnchors ?? [])
    .map((anchor) => anchor.trim())
    .filter(Boolean)
    .slice(0, 2);
  const queries = [required.join(" "), [...required, ...optional].join(" "), ...required];
  const seen = new Set<string>();
  return queries.filter((query) => {
    const key = normalizeText(query);
    if (!key || seen.has(key)) {
      return false;
    }
    seen.add(key);
    return true;
  });
}

function requiredAnchorGoal(
  compiled: QueryCompileResult,
  surface: CandidateSurface,
): QueryEvidenceGoal | null {
  const required = requiredCoverageAnchors(compiled);
  if (required.length === 0) {
    return null;
  }
  const surfaceGoal = goalsForSurface(compiled, surface)[0];
  if (surfaceGoal) {
    return {
      ...surfaceGoal,
      focusAnchors: [...required, ...surfaceGoal.focusAnchors],
    };
  }
  return {
    goal: `Find evidence covering ${required.join(", ")}.`,
    positiveQueries: requiredAnchorQueries(compiled),
    focusAnchors: required,
    preferredSurfaces: compiled.candidateSurfaces.includes(surface)
      ? [surface]
      : compiled.candidateSurfaces,
    fidelity: compiled.evidenceFidelity,
  };
}

function attachGoalMatch(
  hit: SearchHit,
  goal: QueryEvidenceGoal,
  matchedQuery: string,
  compiled: QueryCompileResult,
): GoalScoredSearchHit {
  const focusScore =
    goal.focusAnchors.length > 0 ? informativeAnchorSupport(hit.text, goal.focusAnchors) : 0;
  const compilerQuerySupport = Math.max(
    semanticTextSimilarity(matchedQuery, hit.text),
    semanticTextSimilarity(goal.goal, hit.text),
    focusScore,
  );
  const retrievalScore =
    hit.backend === "hybrid" || hit.backend === "embedding"
      ? clamp01(hit.score)
      : Math.min(clamp01(hit.score), compilerQuerySupport);
  const coverage = evidenceCoverageForText(compiled, hit.text);
  const tailorAdvice = compiled.evidencePlan?.operation.type === "tailor_advice";
  const rawMatchScore = tailorAdvice
    ? clamp01(
        compilerQuerySupport +
          Math.min(retrievalScore, 0.28) * (compilerQuerySupport >= 0.18 ? 0.18 : 0.08),
      )
    : coverage.requiredHits.length === 0 && compilerQuerySupport < 0.18
      ? Math.min(Math.max(compilerQuerySupport, retrievalScore), 0.68)
      : Math.max(compilerQuerySupport, retrievalScore);
  const matchScore = tailorAdvice
    ? clamp01(rawMatchScore)
    : capScoreByEvidenceCoverage(clamp01(rawMatchScore), coverage);
  const score = tailorAdvice
    ? Math.max(matchScore, Math.min(hit.score, matchScore + 0.18))
    : Math.max(hit.score, matchScore);
  return {
    ...hit,
    score,
    goalMatches: mergeGoalMatches((hit as GoalScoredSearchHit).goalMatches, [
      {
        goal: goal.goal,
        score: clamp01(matchScore),
        matchedQuery,
        matchType: "answer_source",
      },
    ]),
  };
}

function layerToCandidateSurface(layer: EvidencePlanLayer): CandidateSurface | null {
  switch (layer) {
    case "state":
    case "fact":
    case "event":
    case "task":
    case "chunk":
    case "graph":
    case "entity_alias":
      return layer;
    case "control":
    case "strategy":
    case "abstraction":
    case "belief":
    case "snippet":
      return null;
  }
}

function slotLayers(slot: QueryEvidenceSlot): EvidencePlanLayer[] {
  return [...new Set([...slot.preferredLayers, ...slot.fallbackLayers])];
}

function uniqueSearchQueries(values: string[], limit = 12): string[] {
  const seen = new Set<string>();
  const result: string[] = [];
  for (const value of values) {
    const cleaned = value.trim();
    if (!cleaned) {
      continue;
    }
    const key = normalizeText(cleaned);
    if (!key || seen.has(key)) {
      continue;
    }
    seen.add(key);
    result.push(cleaned);
    if (result.length >= limit) {
      break;
    }
  }
  return result;
}

function effectiveSlotHints(hints: string[]): string[] {
  return uniqueSearchQueries(
    hints.filter((hint) => normalizeText(hint).length > 0),
    10,
  );
}

function slotSubjectQueryVariants(hints: string[]): string[] {
  const variants: string[] = [];
  for (const hint of hints) {
    const trimmed = hint.trim();
    if (!trimmed) {
      continue;
    }
    variants.push(trimmed);
    for (const match of trimmed.matchAll(/\(([^)]+)\)/gu)) {
      if (match[1]?.trim()) {
        variants.push(match[1].trim());
      }
    }
    const withoutParenthetical = trimmed
      .replace(/\([^)]*\)/gu, " ")
      .replace(/\s+/gu, " ")
      .trim();
    if (withoutParenthetical.trim()) {
      variants.push(withoutParenthetical.trim());
    }
  }
  return uniqueSearchQueries(variants, 12);
}

function evidenceGoalQueriesForSlot(
  slot: QueryEvidenceSlot,
  compiled: QueryCompileResult,
): string[] {
  return evidenceGoalsForSlot(slot, compiled).flatMap((goal) => [
    goal.goal,
    ...(goal.positiveQueries ?? []),
  ]);
}

function evidenceGoalsForSlot(
  slot: QueryEvidenceSlot,
  compiled: QueryCompileResult,
): QueryEvidenceGoal[] {
  const goals = compiled.evidenceGoals ?? [];
  if (goals.length === 0) {
    return [];
  }
  const slotText = [
    slot.id,
    slot.description,
    ...slot.subjectHints,
    ...(slot.relationHints ?? []),
    ...(slot.capabilityQueries ?? []),
    ...slot.requiredFields,
  ]
    .filter(Boolean)
    .join(" ");
  const scored = goals
    .map((goal) => {
      const goalText = [goal.goal, ...(goal.positiveQueries ?? []), ...goal.focusAnchors].join(" ");
      return { goal, score: semanticTextSimilarity(goalText, slotText) };
    })
    .sort((left, right) => right.score - left.score);
  const selected = scored.some((entry) => entry.score >= 0.16)
    ? scored.filter((entry) => entry.score >= 0.16).map((entry) => entry.goal)
    : goals;
  return selected;
}

function slotsForSurface(
  compiled: QueryCompileResult,
  surface: CandidateSurface,
): Array<{
  slot: QueryEvidenceSlot;
  layer: EvidencePlanLayer;
}> {
  return (compiled.evidencePlan?.slots ?? []).flatMap((slot) =>
    slotLayers(slot)
      .filter((layer) => layerToCandidateSurface(layer) === surface)
      .map((layer) => ({ slot, layer })),
  );
}

function slotSearchQueries(slot: QueryEvidenceSlot, compiled: QueryCompileResult): string[] {
  const subjects = slot.subjectHints.join(" ").trim();
  const relations = [...(slot.relationHints ?? []), ...(slot.capabilityQueries ?? [])]
    .join(" ")
    .trim();
  const fields = slot.requiredFields.join(" ").trim();
  const isTailorAdvice = compiled.evidencePlan?.operation.type === "tailor_advice";
  if (isTailorAdvice) {
    if (isUserResourceSlot(slot)) {
      const resourceQueries = evidenceGoalsForSlot(slot, compiled).flatMap(
        (goal) => goal.positiveQueries ?? [],
      );
      const subjectFallback =
        (slot.relationHints ?? []).length === 0 ? slotSubjectQueryVariants(slot.subjectHints) : [];
      return uniqueSearchQueries(
        [
          ...subjectFallback,
          ...(slot.relationHints ?? []),
          ...(slot.capabilityQueries ?? []),
          ...resourceQueries,
          compiled.focusedQuery,
        ].filter((query): query is string => typeof query === "string" && query.trim().length > 0),
        10,
      );
    }
    return uniqueSearchQueries(
      [
        ...slotSubjectQueryVariants(slot.subjectHints),
        ...(slot.capabilityQueries ?? []),
        ...evidenceGoalQueriesForSlot(slot, compiled),
        compiled.focusedQuery,
      ].filter((query): query is string => typeof query === "string" && query.trim().length > 0),
      10,
    );
  }
  return uniqueSearchQueries(
    [
      ...slotSubjectQueryVariants(slot.subjectHints),
      [subjects, relations, fields].filter(Boolean).join(" "),
      [subjects, relations].filter(Boolean).join(" "),
      slot.description,
      ...(slot.capabilityQueries ?? []),
      ...evidenceGoalQueriesForSlot(slot, compiled),
      compiled.focusedQuery,
    ].filter((query): query is string => typeof query === "string" && query.trim().length > 0),
    12,
  );
}

function isUserResourceSlot(slot: QueryEvidenceSlot): boolean {
  return (
    slot.requiredRole === "user_resource" ||
    slot.role === "user_resource" ||
    slot.id === "relevant_user_resources"
  );
}

function metadataStringArray(metadata: Record<string, unknown> | undefined, key: string): string[] {
  const value = metadata?.[key];
  return Array.isArray(value)
    ? value.filter((entry): entry is string => typeof entry === "string" && entry.trim().length > 0)
    : [];
}

function resourceBridgeText(text: string, metadata: Record<string, unknown> | undefined): string {
  if (!metadata) {
    return text;
  }
  const parts = [
    typeof metadata.canonicalSubject === "string" ? metadata.canonicalSubject : undefined,
    typeof metadata.predicate === "string" ? metadata.predicate : undefined,
    typeof metadata.resource === "string" ? metadata.resource : undefined,
    typeof metadata.canonicalObject === "string" ? metadata.canonicalObject : undefined,
    typeof metadata.resourceType === "string"
      ? `resource type: ${metadata.resourceType}`
      : undefined,
    metadataStringArray(metadata, "domains").length > 0
      ? `domains: ${metadataStringArray(metadata, "domains").join(", ")}`
      : undefined,
    metadataStringArray(metadata, "affordances").length > 0
      ? `affordances: ${metadataStringArray(metadata, "affordances").join(", ")}`
      : undefined,
    typeof metadata.supportText === "string" ? `evidence: ${metadata.supportText}` : undefined,
    text,
  ];
  return parts.filter(Boolean).join(" | ");
}

function hasStructuredResourceBridge(metadata: Record<string, unknown> | undefined): boolean {
  if (!metadata) {
    return false;
  }
  return (
    metadata.signalKind === "resourceAssertion" ||
    metadata.predicate === "has_resource" ||
    (typeof metadata.stateKey === "string" && metadata.stateKey.includes(".resource."))
  );
}

function hasResourceAffordanceBridge(metadata: Record<string, unknown> | undefined): boolean {
  return (
    metadataStringArray(metadata, "domains").length > 0 ||
    metadataStringArray(metadata, "affordances").length > 0 ||
    (typeof metadata?.resourceType === "string" && metadata.resourceType.trim().length > 0)
  );
}

function bridgePreferredSurfaces(bridge: QuerySemanticBridge, surface: CandidateSurface): boolean {
  return bridge.preferredLayers.some((layer) => layerToCandidateSurface(layer) === surface);
}

function bridgesForSurface(
  compiled: QueryCompileResult,
  surface: CandidateSurface,
): QuerySemanticBridge[] {
  return (compiled.semanticBridges ?? []).filter((bridge) =>
    bridgePreferredSurfaces(bridge, surface),
  );
}

function bridgeMatchType(bridge: QuerySemanticBridge): EvidenceMatchType {
  if (bridge.role === "user_resource" || bridge.evidenceShape === "resource_affordance") {
    return "capability";
  }
  if (bridge.role === "query_context" || bridge.role === "time_constraint") {
    return "context";
  }
  return "answer_source";
}

function bridgeScoringText(text: string, metadata: Record<string, unknown> | undefined): string {
  return resourceBridgeText(text, metadata);
}

function scoreBridgeMatch(
  bridge: QuerySemanticBridge,
  hit: SearchHit,
  matchedQuery: string,
): BridgeMatch {
  const scoringText = bridgeScoringText(hit.text, hit.metadata);
  const positiveSignalScore =
    bridge.positiveSignals.length > 0
      ? informativeAnchorSupport(scoringText, bridge.positiveSignals)
      : 0;
  const negativeSignalScore =
    (bridge.negativeSignals ?? []).length > 0
      ? informativeAnchorSupport(scoringText, bridge.negativeSignals ?? [])
      : 0;
  const querySupport = Math.max(
    semanticTextSimilarity(matchedQuery, scoringText),
    semanticTextSimilarity(bridge.sourceConcept, scoringText),
    positiveSignalScore,
  );
  const embeddingLike = hit.backend === "hybrid" || hit.backend === "embedding";
  const retrievalSupport = embeddingLike
    ? clamp01(hit.score)
    : Math.min(clamp01(hit.score), querySupport);
  const structuredBridgeBoost =
    bridge.evidenceShape === "resource_affordance" && hasResourceAffordanceBridge(hit.metadata)
      ? 0.1
      : 0;
  const negativeSignalWeight =
    bridge.role === "answer_value" ||
    bridge.role === "answer_event" ||
    bridge.role === "user_resource" ||
    bridge.role === "prior_advice"
      ? 0
      : 0.18;
  const score = clamp01(
    querySupport * 0.72 +
      positiveSignalScore * 0.2 +
      retrievalSupport * 0.08 +
      structuredBridgeBoost -
      negativeSignalScore * negativeSignalWeight,
  );
  return {
    bridgeId: bridge.bridgeId,
    sourceConcept: bridge.sourceConcept,
    role: bridge.role,
    evidenceShape: bridge.evidenceShape,
    score,
    matchedQuery,
    positiveSignalScore,
    negativeSignalScore,
  };
}

function attachBridgeMatch(
  hit: SearchHit,
  bridge: QuerySemanticBridge,
  matchedQuery: string,
): GoalScoredSearchHit {
  const match = scoreBridgeMatch(bridge, hit, matchedQuery);
  return {
    ...hit,
    score: Math.max(hit.score, match.score * 0.92),
    bridgeMatches: mergeBridgeMatches((hit as GoalScoredSearchHit).bridgeMatches, [match]),
  };
}

function fieldSupport(text: string, fields: string[]): number {
  if (fields.length === 0) {
    return 1;
  }
  const normalized = normalizeText(text);
  let hits = 0;
  let concreteFieldCount = 0;
  for (const field of fields) {
    const key = normalizeText(field);
    if (!key) {
      continue;
    }
    if (isGenericRequiredField(key)) {
      continue;
    }
    concreteFieldCount += 1;
    if (normalized.includes(key.replace(/_/gu, " "))) {
      hits += 1;
    }
  }
  return concreteFieldCount > 0 ? clamp01(hits / concreteFieldCount) : 0.15;
}

function isGenericRequiredField(key: string): boolean {
  return [
    "answer_value",
    "attribute_value",
    "countable_item",
    "temporal_marker",
    "preference_or_prior_action",
    "source_evidence",
    "query_context",
  ].includes(key);
}

function slotMatchScore(
  text: string,
  slot: QueryEvidenceSlot,
  query: string,
  retrievalScore = 0,
  retrievalBackend?: SearchHit["backend"],
  metadata?: Record<string, unknown>,
): number {
  const subjectHints = effectiveSlotHints(slot.subjectHints);
  const relationHints = effectiveSlotHints([
    ...(slot.relationHints ?? []),
    ...(slot.capabilityQueries ?? []),
  ]);
  const subjectSupport =
    subjectHints.length > 0
      ? informativeAnchorSupport(text, subjectHints)
      : slot.subjectHints.length > 0
        ? 0
        : 0.5;
  const relationSupport =
    relationHints.length > 0
      ? informativeAnchorSupport(text, relationHints)
      : (slot.relationHints ?? []).length > 0
        ? 0
        : 0.35;
  const field = fieldSupport(text, slot.requiredFields);
  const querySimilarity = semanticTextSimilarity(query, text);
  const embeddingLike = retrievalBackend === "hybrid" || retrievalBackend === "embedding";
  const retrievalSupport = embeddingLike
    ? clamp01(retrievalScore)
    : Math.min(clamp01(retrievalScore), querySimilarity);
  const querySupport = Math.max(querySimilarity, retrievalSupport * (embeddingLike ? 0.92 : 0.54));
  const baseScore = clamp01(
    Math.max(subjectSupport, querySupport * 0.82, retrievalSupport * 0.86) * 0.52 +
      relationSupport * 0.2 +
      field * 0.2 +
      Math.max(querySupport, retrievalSupport) * 0.08,
  );
  if (!isUserResourceSlot(slot)) {
    return baseScore;
  }
  const bridgeText = resourceBridgeText(text, metadata);
  const bridgeHints = isUserResourceSlot(slot)
    ? relationHints
    : [...subjectHints, ...relationHints, ...slot.requiredFields];
  const bridgeHintSupport =
    bridgeHints.length > 0 ? informativeAnchorSupport(bridgeText, bridgeHints) : 0;
  const bridgeSimilarity = Math.max(semanticTextSimilarity(query, bridgeText), bridgeHintSupport);
  if (hasStructuredResourceBridge(metadata)) {
    // Structured resources should be judged by their resource affordance bridge,
    // not by generic wording in the surrounding conversation.
    const structuredScore = clamp01(
      bridgeSimilarity * 0.72 +
        field * 0.06 +
        (hasResourceAffordanceBridge(metadata) ? 0.04 : 0) +
        0.03,
    );
    return hasResourceAffordanceBridge(metadata) || bridgeSimilarity >= 0.72
      ? structuredScore
      : Math.min(structuredScore, 0.34);
  }
  const bridgeScore = clamp01(
    bridgeSimilarity * 0.5 +
      retrievalSupport * 0.32 +
      (hasResourceAffordanceBridge(metadata) ? 0.14 : 0),
  );
  const rawScore = Math.max(baseScore, bridgeScore);
  return rawScore;
}

function attachSlotMatch(
  hit: SearchHit,
  slot: QueryEvidenceSlot,
  layer: EvidencePlanLayer,
  matchedQuery: string,
): GoalScoredSearchHit {
  const score = slotMatchScore(hit.text, slot, matchedQuery, hit.score, hit.backend, hit.metadata);
  const matchType = slotMatchType(slot, matchedQuery);
  return {
    ...hit,
    score: Math.max(hit.score, score),
    slotMatches: mergeSlotMatches((hit as GoalScoredSearchHit).slotMatches, [
      {
        slotId: slot.id,
        score,
        matchedQuery,
        layer,
        matchType,
        queryContextOnly:
          matchType === "context" || evidenceSlotRequiredRole(slot) === "query_context",
      },
    ]),
  };
}

async function slotSearchHits(
  store: MemxStoreBundle,
  ctx: MemoryOperationContext,
  compiled: QueryCompileResult,
  surface: CandidateSurface,
  topN: number,
): Promise<{ hits: GoalScoredSearchHit[]; slotStats: CandidateGenerationSlotLayerStats[] }> {
  const retrievalScope = retrievalScopeForSurface(surface);
  const hits: GoalScoredSearchHit[] = [];
  const slotStats: CandidateGenerationSlotLayerStats[] = [];
  for (const { slot, layer } of slotsForSurface(compiled, surface)) {
    const slotLayerHits: GoalScoredSearchHit[] = [];
    const isTailorAdvice = compiled.evidencePlan?.operation.type === "tailor_advice";
    const queryLimit = isTailorAdvice ? 6 : 4;
    for (const query of slotSearchQueries(slot, compiled).slice(0, queryLimit)) {
      const rawLimit = isTailorAdvice ? Math.max(topN * 8, 32) : Math.max(topN * 3, 8);
      const semanticHits = ctx.config.advanced.enableEmbeddingCandidates
        ? await cachedHybridSearch(store, ctx, {
            agentId: ctx.agentId,
            scopes: ctx.scopes,
            query,
            limit: rawLimit,
            readEpoch: ctx.readEpoch,
            docKinds: retrievalScope.docKinds,
            docTypes: retrievalScope.docTypes,
          })
        : store.retrievalBackend.keywordSearch({
            agentId: ctx.agentId,
            scopes: ctx.scopes,
            query,
            limit: rawLimit,
            readEpoch: ctx.readEpoch,
            docKinds: retrievalScope.docKinds,
            docTypes: retrievalScope.docTypes,
          });
      const keywordHits = store.retrievalBackend.keywordSearch({
        agentId: ctx.agentId,
        scopes: ctx.scopes,
        query,
        limit: isTailorAdvice ? Math.max(topN * 6, 32) : Math.max(topN * 8, 32),
        readEpoch: ctx.readEpoch,
        docKinds: retrievalScope.docKinds,
        docTypes: retrievalScope.docTypes,
      });
      const rawHits = dedupeSearchHits([...semanticHits, ...keywordHits]);
      for (const hit of rawHits) {
        if (!surfaceTypeMatches(surface, hit)) {
          continue;
        }
        const scored = attachSlotMatch(hit, slot, layer, query);
        if ((scored.slotMatches?.[0]?.score ?? 0) < 0.16) {
          continue;
        }
        slotLayerHits.push(scored);
      }
    }
    if (isUserResourceSlot(slot)) {
      const inventoryLimit = Math.max(topN * 32, 96);
      const inventoryDocs = store.vectorRepo.listDocs({
        agentId: ctx.agentId,
        scopes: ctx.scopes,
        limit: inventoryLimit,
        readEpoch: ctx.readEpoch,
        docKinds: retrievalScope.docKinds,
        docTypes: retrievalScope.docTypes,
      });
      for (const doc of inventoryDocs) {
        const metadata = doc.metadataJson;
        if (!hasStructuredResourceBridge(metadata)) {
          continue;
        }
        const searchHit: SearchHit = {
          docId: doc.docId,
          text: doc.text,
          metadata,
          score: 0,
          backend: "repo",
        };
        if (!surfaceTypeMatches(surface, searchHit)) {
          continue;
        }
        let bestHit: GoalScoredSearchHit | null = null;
        for (const query of slotSearchQueries(slot, compiled)) {
          const scored = attachSlotMatch(searchHit, slot, layer, query);
          const score = scored.slotMatches?.find((match) => match.slotId === slot.id)?.score ?? 0;
          if (score < 0.15) {
            continue;
          }
          if (!bestHit || score > topSlotMatchScore(bestHit)) {
            bestHit = scored;
          }
        }
        if (bestHit) {
          slotLayerHits.push(bestHit);
        }
      }
    }
    const deduped = dedupeSearchHits(slotLayerHits);
    hits.push(...deduped);
    slotStats.push({
      slotId: slot.id,
      layer,
      rawCount: slotLayerHits.length,
      selectedCount: 0,
      alternateCount: 0,
      topCandidateIds: deduped.slice(0, 4).map((hit) => `${surface}:${hit.docId}`),
    });
  }
  return { hits: dedupeSearchHits(hits), slotStats };
}

function selectSlotCandidateSearchHits(
  hits: GoalScoredSearchHit[],
  compiled: QueryCompileResult,
  surface: CandidateSurface,
): GoalScoredSearchHit[] {
  const selected: GoalScoredSearchHit[] = [];
  const seenDocIds = new Set<string>();
  for (const { slot, layer } of slotsForSurface(compiled, surface)) {
    const threshold = isUserResourceSlot(slot) ? 0.15 : 0.2;
    const perSlot = hits
      .filter((hit) =>
        (hit.slotMatches ?? []).some(
          (match) => match.slotId === slot.id && match.layer === layer && match.score >= threshold,
        ),
      )
      .sort((left, right) => {
        const rightScore = Math.max(
          0,
          ...(right.slotMatches ?? [])
            .filter((match) => match.slotId === slot.id && match.layer === layer)
            .map((match) => match.score),
        );
        const leftScore = Math.max(
          0,
          ...(left.slotMatches ?? [])
            .filter((match) => match.slotId === slot.id && match.layer === layer)
            .map((match) => match.score),
        );
        return rightScore - leftScore || right.score - left.score;
      });
    const limit = isUserResourceSlot(slot) ? 3 : 2;
    for (const hit of perSlot.slice(0, limit)) {
      if (seenDocIds.has(hit.docId)) {
        continue;
      }
      selected.push(hit);
      seenDocIds.add(hit.docId);
    }
  }
  return dedupeSearchHits(selected);
}

async function requiredAnchorSearchHits(
  store: MemxStoreBundle,
  ctx: MemoryOperationContext,
  compiled: QueryCompileResult,
  surface: CandidateSurface,
  topN: number,
): Promise<GoalScoredSearchHit[]> {
  const goal = requiredAnchorGoal(compiled, surface);
  if (!goal) {
    return [];
  }
  const retrievalScope = retrievalScopeForSurface(surface);
  const queries = requiredAnchorQueries(compiled);
  const hits: GoalScoredSearchHit[] = [];
  const rawLimit = Math.max(topN * 16, 64);
  const coverageDocs = store.vectorRepo
    .listDocs({
      agentId: ctx.agentId,
      scopes: ctx.scopes,
      limit: rawLimit,
      readEpoch: ctx.readEpoch,
      docKinds: retrievalScope.docKinds,
      docTypes: retrievalScope.docTypes,
    })
    .map(
      (doc): SearchHit => ({
        docId: doc.docId,
        text: doc.text,
        metadata: doc.metadataJson,
        score: evidenceCoverageForText(compiled, doc.text).coverageScore,
        backend: "lexical",
      }),
    )
    .filter((hit) => surfaceTypeMatches(surface, hit))
    .filter((hit) => evidenceCoverageForText(compiled, hit.text).requiredHits.length > 0)
    .map((hit) => attachGoalMatch(hit, goal, queries[0] ?? goal.goal, compiled));
  hits.push(...coverageDocs);
  for (const query of queries) {
    const semanticHits = ctx.config.advanced.enableEmbeddingCandidates
      ? await cachedHybridSearch(store, ctx, {
          agentId: ctx.agentId,
          scopes: ctx.scopes,
          query,
          limit: rawLimit,
          readEpoch: ctx.readEpoch,
          docKinds: retrievalScope.docKinds,
          docTypes: retrievalScope.docTypes,
        })
      : [];
    const lexicalHits = store.retrievalBackend.keywordSearch({
      agentId: ctx.agentId,
      scopes: ctx.scopes,
      query,
      limit: rawLimit,
      readEpoch: ctx.readEpoch,
      docKinds: retrievalScope.docKinds,
      docTypes: retrievalScope.docTypes,
    });
    const rawHits = dedupeSearchHits([...semanticHits, ...lexicalHits]);
    for (const hit of rawHits) {
      if (!surfaceTypeMatches(surface, hit)) {
        continue;
      }
      const coverage = evidenceCoverageForText(compiled, hit.text);
      const semanticSupport = Math.max(
        semanticTextSimilarity(query, hit.text),
        semanticTextSimilarity(goal.goal, hit.text),
        semanticTextSimilarity(compiled.focusedQuery, hit.text),
      );
      const isTailorAdvice = compiled.evidencePlan?.operation.type === "tailor_advice";
      const embeddingLike = hit.backend === "hybrid" || hit.backend === "embedding";
      const semanticRetrievalSupport = embeddingLike
        ? Math.max(semanticSupport, hit.score)
        : semanticSupport;
      if (!isTailorAdvice && coverage.requiredHits.length === 0 && semanticRetrievalSupport < 0.5) {
        continue;
      }
      if (
        !isTailorAdvice &&
        coverage.missingRequired.length > 0 &&
        compiled.answerMode !== "count_aggregate" &&
        compiled.answerMode !== "multi_evidence" &&
        semanticRetrievalSupport < 0.58
      ) {
        continue;
      }
      if (isTailorAdvice && coverage.requiredHits.length === 0 && semanticRetrievalSupport < 0.18) {
        continue;
      }
      hits.push(attachGoalMatch(hit, goal, query, compiled));
    }
  }
  return dedupeSearchHits(hits);
}

async function goalSearchHits(
  store: MemxStoreBundle,
  ctx: MemoryOperationContext,
  compiled: QueryCompileResult,
  surface: CandidateSurface,
  topN: number,
): Promise<GoalScoredSearchHit[]> {
  const goals = goalsForSurface(compiled, surface);
  if (goals.length === 0) {
    return [];
  }
  const retrievalScope = retrievalScopeForSurface(surface);
  const queries = goals.flatMap((goal) =>
    goal.positiveQueries.slice(0, 3).map((query) => ({ goal, query: query.trim() })),
  );
  const hits: GoalScoredSearchHit[] = [];
  for (const { goal, query } of queries) {
    if (!query) {
      continue;
    }
    const rawLimit = Math.max(topN * 3, topN + 2, 6);
    const rawHits = ctx.config.advanced.enableEmbeddingCandidates
      ? await cachedHybridSearch(store, ctx, {
          agentId: ctx.agentId,
          scopes: ctx.scopes,
          query,
          limit: rawLimit,
          readEpoch: ctx.readEpoch,
          docKinds: retrievalScope.docKinds,
          docTypes: retrievalScope.docTypes,
        })
      : store.retrievalBackend.keywordSearch({
          agentId: ctx.agentId,
          scopes: ctx.scopes,
          query,
          limit: rawLimit,
          readEpoch: ctx.readEpoch,
          docKinds: retrievalScope.docKinds,
          docTypes: retrievalScope.docTypes,
        });
    hits.push(...rawHits.map((hit) => attachGoalMatch(hit, goal, query, compiled)));
  }
  return dedupeSearchHits(hits);
}

async function bridgeSearchHits(
  store: MemxStoreBundle,
  ctx: MemoryOperationContext,
  compiled: QueryCompileResult,
  surface: CandidateSurface,
  topN: number,
): Promise<GoalScoredSearchHit[]> {
  const bridges = bridgesForSurface(compiled, surface);
  if (bridges.length === 0) {
    return [];
  }
  const retrievalScope = retrievalScopeForSurface(surface);
  const hits: GoalScoredSearchHit[] = [];
  for (const bridge of bridges) {
    const queries = uniqueSearchQueries([...bridge.retrievalQueries, ...bridge.positiveSignals], 8);
    for (const query of queries) {
      const rawLimit = Math.max(topN * 5, 16);
      const semanticHits = ctx.config.advanced.enableEmbeddingCandidates
        ? await cachedHybridSearch(store, ctx, {
            agentId: ctx.agentId,
            scopes: ctx.scopes,
            query,
            limit: rawLimit,
            readEpoch: ctx.readEpoch,
            docKinds: retrievalScope.docKinds,
            docTypes: retrievalScope.docTypes,
          })
        : [];
      const lexicalHits = store.retrievalBackend.keywordSearch({
        agentId: ctx.agentId,
        scopes: ctx.scopes,
        query,
        limit: rawLimit,
        readEpoch: ctx.readEpoch,
        docKinds: retrievalScope.docKinds,
        docTypes: retrievalScope.docTypes,
      });
      for (const hit of dedupeSearchHits([...semanticHits, ...lexicalHits])) {
        if (!surfaceTypeMatches(surface, hit)) {
          continue;
        }
        const scored = attachBridgeMatch(hit, bridge, query);
        if ((scored.bridgeMatches?.[0]?.score ?? 0) >= 0.14) {
          hits.push(scored);
        }
      }
    }

    const repoDocs = store.vectorRepo.listDocs({
      agentId: ctx.agentId,
      scopes: ctx.scopes,
      limit: Math.max(topN * 16, 64),
      readEpoch: ctx.readEpoch,
      docKinds: retrievalScope.docKinds,
      docTypes: retrievalScope.docTypes,
    });
    for (const doc of repoDocs) {
      const hit: SearchHit = {
        docId: doc.docId,
        text: doc.text,
        metadata: doc.metadataJson,
        score: 0,
        backend: "repo",
      };
      if (!surfaceTypeMatches(surface, hit)) {
        continue;
      }
      let best: GoalScoredSearchHit | null = null;
      for (const query of queries) {
        const scored = attachBridgeMatch(hit, bridge, query);
        const score = scored.bridgeMatches?.[0]?.score ?? 0;
        if (score < 0.16) {
          continue;
        }
        if (!best || score > topBridgeMatchScore(best)) {
          best = scored;
        }
      }
      if (best) {
        hits.push(best);
      }
    }
  }
  return dedupeSearchHits(hits);
}

async function retrieveSurfaceHits(
  store: MemxStoreBundle,
  ctx: MemoryOperationContext,
  compiled: QueryCompileResult,
  surface: CandidateSurface,
  topN: number,
): Promise<{
  selectedHits: SearchHit[];
  alternateHits: SearchHit[];
  stats: CandidateGenerationSurfaceStats;
  slotStats: CandidateGenerationSlotLayerStats[];
  candidates: CandidateHit[];
  slotCandidates: CandidateHit[];
  bridgeCandidates: CandidateHit[];
}> {
  if (surface === "entity_alias") {
    const entities = await resolveQueryEntitiesForRecall(
      store,
      ctx,
      compiled,
      Math.max(12, topN * 4),
    );
    const seenEntityIds = new Set<string>();
    const entityCandidates: CandidateHit[] = entities
      .filter((entity) => {
        if (seenEntityIds.has(entity.entityId)) {
          return false;
        }
        seenEntityIds.add(entity.entityId);
        return true;
      })
      .slice(0, topN)
      .map((entity, index) => ({
        candidateId: `entity_alias:${entity.entityId}:${index}`,
        surface,
        tier: "primary",
        text: entity.canonicalName,
        score: 1 - index * 0.05,
        retrievalBackend: "repo",
        scope: ctx.scopes[0] ?? "unknown",
        agentId: ctx.agentId,
        confidence: entity.confidence,
        lineage: {
          canonicalKind: "entity",
          canonicalId: entity.entityId,
          sourceKind: "entity_alias",
          sourceId: entity.entityId,
        },
        metadata: {
          entityId: entity.entityId,
          canonicalName: entity.canonicalName,
        },
      }));
    const expansionCandidates = entityExpansionCandidates(
      store,
      ctx,
      entityCandidates
        .map((candidate) =>
          typeof candidate.metadata?.entityId === "string"
            ? store.graphRepo.getEntityById(candidate.metadata.entityId)
            : null,
        )
        .filter((entity): entity is NormalizedEntity => Boolean(entity)),
      topN,
    );
    const candidates = mergeCandidateHits([...entityCandidates, ...expansionCandidates]);
    return {
      selectedHits: [],
      alternateHits: [],
      stats: {
        rawCount: candidates.length,
        filteredCount: candidates.length,
        alternateCount: 0,
        topN,
        backendMix: candidates.length > 0 ? ["repo"] : [],
      },
      slotStats: slotsForSurface(compiled, surface).map(({ slot, layer }) => ({
        slotId: slot.id,
        layer,
        rawCount: candidates.length,
        selectedCount: candidates.length,
        alternateCount: 0,
        topCandidateIds: candidates.slice(0, 4).map((candidate) => candidate.candidateId),
      })),
      candidates,
      slotCandidates: [],
      bridgeCandidates: [],
    };
  }

  const retrievalScope = retrievalScopeForSurface(surface);
  const rawLimit = Math.max(topN * 4, topN + 2, 6);
  const focusedHits = ctx.config.advanced.enableEmbeddingCandidates
    ? await cachedHybridSearch(store, ctx, {
        agentId: ctx.agentId,
        scopes: ctx.scopes,
        query: compiled.focusedQuery,
        limit: rawLimit,
        readEpoch: ctx.readEpoch,
        docKinds: retrievalScope.docKinds,
        docTypes: retrievalScope.docTypes,
      })
    : store.retrievalBackend.keywordSearch({
        agentId: ctx.agentId,
        scopes: ctx.scopes,
        query: compiled.focusedQuery,
        limit: rawLimit,
        readEpoch: ctx.readEpoch,
        docKinds: retrievalScope.docKinds,
        docTypes: retrievalScope.docTypes,
      });
  const slotResult = await slotSearchHits(store, ctx, compiled, surface, topN);
  const bridgeHits = await bridgeSearchHits(store, ctx, compiled, surface, topN);
  const rawHits = dedupeSearchHits([
    ...focusedHits,
    ...(await goalSearchHits(store, ctx, compiled, surface, topN)),
    ...bridgeHits,
    ...(await requiredAnchorSearchHits(store, ctx, compiled, surface, topN)),
    ...slotResult.hits,
  ]);
  const retained = rawHits.map((hit) => ({
    hit,
    retention: evaluateHitRetention(surface, hit, compiled, ctx),
  }));
  const filtered = retained
    .filter((entry) => entry.retention === "primary")
    .map((entry) => entry.hit);
  const alternate = retained
    .filter((entry) => entry.retention === "alternate")
    .map((entry) => entry.hit);
  const rankedFiltered = rerankSurfaceHits(filtered, surface, compiled);
  const rankedAlternate = rerankSurfaceHits(alternate, surface, compiled);
  const lexicalHits =
    filtered.length >= topN ? [] : lexicalSearchHits(store, ctx, compiled, surface, topN);
  const lexicalRetained = lexicalHits.map((hit) => ({
    hit,
    retention: evaluateHitRetention(surface, hit, compiled, ctx),
  }));
  // Lexical fallback may seed alternates/debug surfaces, but it must never
  // become a primary evidence source. Primary candidates must come from
  // embedding/hybrid retrieval or structured repos.
  const lexicalPrimary: SearchHit[] = [];
  const lexicalAlternate = lexicalRetained
    .filter((entry) => entry.retention !== "drop")
    .map((entry) => entry.hit);
  const mergedSelectedHits = selectSurfaceHitsForSlots(
    dedupeSearchHits([...rankedFiltered, ...lexicalPrimary]),
    topN,
    compiled,
  );
  const mergedAlternateHits = dedupeSearchHits([...rankedAlternate, ...lexicalAlternate]).slice(
    0,
    Math.max(1, Math.min(2, topN)),
  );
  const slotCandidateHits = selectSlotCandidateSearchHits(slotResult.hits, compiled, surface);
  const bridgeCandidateHits = bridgeHits
    .filter((hit) => topBridgeMatchScore(hit) >= 0.14)
    .sort(
      (left, right) =>
        topBridgeMatchScore(right) - topBridgeMatchScore(left) || right.score - left.score,
    )
    .slice(0, Math.max(2, Math.min(3, topN)));
  const slotCandidates = slotCandidateHits
    .map((hit) => {
      const retention = evaluateHitRetention(surface, hit, compiled, ctx);
      if (retention === "drop") {
        return null;
      }
      return toCandidateHit(surface, hit, ctx, retention === "alternate" ? "alternate" : "primary");
    })
    .filter((hit): hit is CandidateHit => Boolean(hit));
  const bridgeCandidates = bridgeCandidateHits
    .map((hit) => {
      const retention = evaluateHitRetention(surface, hit, compiled, ctx);
      if (retention === "drop") {
        return null;
      }
      return toCandidateHit(surface, hit, ctx, retention === "alternate" ? "alternate" : "primary");
    })
    .filter((hit): hit is CandidateHit => Boolean(hit));
  const selectedDocIds = new Set([
    ...mergedSelectedHits.map((hit) => hit.docId),
    ...slotCandidates.filter((hit) => hit.tier !== "alternate").map((hit) => hit.docId ?? ""),
    ...bridgeCandidates.filter((hit) => hit.tier !== "alternate").map((hit) => hit.docId ?? ""),
  ]);
  const alternateDocIds = new Set(mergedAlternateHits.map((hit) => hit.docId));
  const rankedCandidates = [...mergedSelectedHits, ...mergedAlternateHits]
    .map((hit) =>
      toCandidateHit(
        surface,
        hit,
        ctx,
        mergedSelectedHits.some((selected) => selected.docId === hit.docId)
          ? "primary"
          : "alternate",
      ),
    )
    .filter((hit): hit is CandidateHit => Boolean(hit));
  return {
    selectedHits: mergedSelectedHits,
    alternateHits: mergedAlternateHits,
    stats: {
      rawCount: rawHits.length + lexicalHits.length,
      filteredCount: filtered.length + lexicalPrimary.length,
      alternateCount: alternate.length + lexicalAlternate.length,
      topN,
      backendMix: [
        ...new Set([...mergedSelectedHits, ...mergedAlternateHits].map((hit) => hit.backend)),
      ],
    },
    slotStats: slotResult.slotStats.map((stats) => ({
      ...stats,
      selectedCount: stats.topCandidateIds.filter((candidateId) =>
        selectedDocIds.has(candidateId.split(":").slice(1).join(":")),
      ).length,
      alternateCount: stats.topCandidateIds.filter((candidateId) =>
        alternateDocIds.has(candidateId.split(":").slice(1).join(":")),
      ).length,
    })),
    candidates: mergeCandidateHits([...rankedCandidates, ...slotCandidates, ...bridgeCandidates]),
    slotCandidates,
    bridgeCandidates,
  };
}

function shouldAttemptSurfaceRecovery(compiled: QueryCompileResult): boolean {
  return (
    compiled.turnMode === "memory_qa" &&
    compiled.queryShape.evidenceNeed !== "workflow_context" &&
    compiled.queryShape.evidenceNeed !== "relation"
  );
}

function recoverySurfacesForCompiled(compiled: QueryCompileResult): CandidateSurface[] {
  const recovery = new Set<CandidateSurface>();
  if (!compiled.candidateSurfaces.includes("event")) {
    recovery.add("event");
  }
  if (
    !compiled.candidateSurfaces.includes("chunk") &&
    (compiled.answerGranularity === "detail" ||
      compiled.evidenceFidelity === "high" ||
      compiled.candidateSurfaces.includes("event") ||
      compiled.queryShape.timeframe === "historical" ||
      compiled.queryShape.timeframe === "compare")
  ) {
    recovery.add("chunk");
  }
  return [...recovery];
}

function recoveryTopN(
  surface: CandidateSurface,
  budgets: Partial<Record<CandidateSurface, number>>,
): number {
  const configured = budgets[surface];
  if (typeof configured === "number" && configured > 0) {
    return configured;
  }
  switch (surface) {
    case "event":
      return 3;
    case "chunk":
      return 2;
    default:
      return 1;
  }
}

function layerCandidateId(prefix: string, id: string): string {
  return id.startsWith(`${prefix}:`) ? id : `${prefix}:${id}`;
}

async function controlLayerCandidateResult(
  store: MemxStoreBundle,
  ctx: MemoryOperationContext,
  compiled: QueryCompileResult,
): Promise<{ stats: CandidateGenerationSlotLayerStats[]; candidates: LayerCandidateHit[] }> {
  const slots = compiled.evidencePlan?.slots ?? [];
  if (slots.length === 0) {
    return { stats: [], candidates: [] };
  }
  const stats: CandidateGenerationSlotLayerStats[] = [];
  const candidates: LayerCandidateHit[] = [];
  const strategies = store.strategyRepo.listByAgent({
    agentId: ctx.agentId,
    scopes: ctx.scopes,
    stages: ["active", "candidate"],
    limit: 64,
    readEpoch: ctx.readEpoch,
  });
  const abstractions = store.abstractionRepo.listByAgent({
    agentId: ctx.agentId,
    scopes: ctx.scopes,
    stages: ["active", "candidate", "probationary"],
    limit: 64,
    readEpoch: ctx.readEpoch,
  });
  const beliefs = store.beliefRepo
    .listByAgent({
      agentId: ctx.agentId,
      limit: 128,
      readEpoch: ctx.readEpoch,
    })
    .filter((belief) => ctx.scopes.includes(belief.scope));
  const queryEntityIds = new Set(
    (await resolveQueryEntitiesForRecall(store, ctx, compiled, Math.max(8, slots.length * 4))).map(
      (entity) => entity.entityId,
    ),
  );
  const entityControlScore = (text: string): number => {
    if (queryEntityIds.size === 0) {
      return 0;
    }
    return [...queryEntityIds].some((entityId) => text.includes(`entity:${entityId}`)) ? 0.94 : 0;
  };
  const layerTexts: Record<
    "strategy" | "abstraction" | "belief" | "control",
    Array<{
      id: string;
      text: string;
      sourceRef?: string;
      sourceRefs?: string[];
      observedAt?: string;
    }>
  > = {
    strategy: strategies.map((entry) => {
      const sourceRefs = uniqueMaintenanceRefs([
        ...sourceRefsFromMaintenanceMetadata(entry.metadataJson),
        ...entry.supportTaskIds.map((taskId) => `task:${taskId}`),
      ]);
      return {
        id: layerCandidateId("strategy", entry.strategyId),
        text: `${entry.domainKey} ${entry.summary}`,
        sourceRef: sourceRefs[0],
        sourceRefs,
        observedAt: entry.updatedAt,
      };
    }),
    abstraction: abstractions.map((entry) => {
      const sourceRefs = uniqueMaintenanceRefs([
        ...sourceRefsFromMaintenanceMetadata(entry.metadataJson),
        ...entry.supportContentRefs,
      ]);
      return {
        id: layerCandidateId("abstraction", entry.candidateId),
        text: `${entry.abstractionType} ${entry.semanticKey} ${entry.summary}`,
        sourceRef: sourceRefs[0],
        sourceRefs,
        observedAt: entry.updatedAt,
      };
    }),
    belief: beliefs.map((entry) => ({
      id: layerCandidateId("belief", entry.beliefId),
      text: `${entry.memoryKind} ${entry.semanticKey} ${JSON.stringify(entry.metadataJson ?? {})}`,
      sourceRef: entry.contentRef,
      sourceRefs: entry.contentRef ? [entry.contentRef] : undefined,
      observedAt: entry.updatedAt,
    })),
    control: [],
  };
  layerTexts.control = [...layerTexts.strategy, ...layerTexts.abstraction, ...layerTexts.belief];
  for (const slot of slots) {
    for (const layer of slotLayers(slot)) {
      if (
        layer !== "control" &&
        layer !== "strategy" &&
        layer !== "abstraction" &&
        layer !== "belief"
      ) {
        continue;
      }
      const rows = layerTexts[layer];
      const scored = rows
        .map((row) => ({
          ...row,
          score: Math.max(
            slotMatchScore(row.text, slot, compiled.focusedQuery),
            entityControlScore(row.text),
          ),
        }))
        .filter((row) => row.score >= 0.18)
        .sort((left, right) => right.score - left.score);
      candidates.push(
        ...scored.slice(0, 8).map((row) => {
          const matchedQuery = slotSearchQueries(slot, compiled)[0] ?? compiled.focusedQuery;
          const matchType = slotMatchType(slot, matchedQuery);
          return {
            id: row.id,
            layer,
            text: row.text,
            sourceRef: row.sourceRef,
            sourceRefs: row.sourceRefs,
            observedAt: row.observedAt,
            score: row.score,
            slotMatches: [
              {
                slotId: slot.id,
                score: row.score,
                matchedQuery,
                matchType,
                queryContextOnly:
                  matchType === "context" || evidenceSlotRequiredRole(slot) === "query_context",
              },
            ],
            lineage: {
              sourceKind: "alternate" as const,
              sourceId: row.id,
              sourceRef: row.sourceRef,
            },
          };
        }),
      );
      stats.push({
        slotId: slot.id,
        layer,
        rawCount: rows.length,
        selectedCount: scored.length,
        alternateCount: 0,
        topCandidateIds: scored.slice(0, 4).map((row) => row.id),
      });
    }
  }
  return { stats, candidates };
}

export async function generateCandidates(
  store: MemxStoreBundle,
  ctx: MemoryOperationContext,
  compiled: QueryCompileResult,
): Promise<CandidateGenerationResult> {
  const budgets = deriveSurfaceBudgets(compiled, ctx);
  const surfaceStats: Partial<Record<CandidateSurface, CandidateGenerationSurfaceStats>> = {};
  const slotLayerStats: CandidateGenerationSlotLayerStats[] = [];
  const candidates: CandidateHit[] = [];
  const slotCandidates: CandidateHit[] = [];
  const bridgeCandidates: CandidateHit[] = [];
  const selectedHits: SearchHit[] = [];
  const alternateSearchHits: SearchHit[] = [];

  for (const surface of SURFACES) {
    const topN = budgets[surface] ?? 0;
    if (topN <= 0) {
      continue;
    }
    const result = await retrieveSurfaceHits(store, ctx, compiled, surface, topN);
    surfaceStats[surface] = result.stats;
    slotLayerStats.push(...result.slotStats);
    candidates.push(...result.candidates);
    slotCandidates.push(...result.slotCandidates);
    bridgeCandidates.push(...result.bridgeCandidates);
    selectedHits.push(...result.selectedHits);
    alternateSearchHits.push(...result.alternateHits);
  }

  const primaryCount = candidates.filter((candidate) => candidate.tier === "primary").length;
  if (primaryCount === 0 && shouldAttemptSurfaceRecovery(compiled)) {
    for (const surface of recoverySurfacesForCompiled(compiled)) {
      const topN = recoveryTopN(surface, budgets);
      if (topN <= 0) {
        continue;
      }
      if (surfaceStats[surface]?.filteredCount || surfaceStats[surface]?.alternateCount) {
        continue;
      }
      const result = await retrieveSurfaceHits(store, ctx, compiled, surface, topN);
      surfaceStats[surface] = result.stats;
      slotLayerStats.push(...result.slotStats);
      candidates.push(...result.candidates);
      slotCandidates.push(...result.slotCandidates);
      bridgeCandidates.push(...result.bridgeCandidates);
      selectedHits.push(...result.selectedHits);
      alternateSearchHits.push(...result.alternateHits);
    }
  }

  const controlLayer = await controlLayerCandidateResult(store, ctx, compiled);

  return {
    candidates: mergeCandidateHits(candidates),
    slotCandidates: mergeCandidateHits(slotCandidates),
    bridgeCandidates: mergeCandidateHits(bridgeCandidates),
    searchHits: dedupeSearchHits(selectedHits),
    alternateSearchHits: dedupeSearchHits(alternateSearchHits),
    surfaceStats,
    slotLayerStats: [...slotLayerStats, ...controlLayer.stats],
    layerCandidates: controlLayer.candidates,
    budgets,
  };
}

export function candidateGenerationAuditPayload(
  result: CandidateGenerationResult,
): Record<string, unknown> {
  const summarizeCandidate = (hit: CandidateHit) => ({
    candidateId: hit.candidateId,
    surface: hit.surface,
    tier: hit.tier ?? "primary",
    text: hit.text,
    baseScore: hit.score,
    retrievalBackend: hit.retrievalBackend,
    docId: hit.docId,
    activeHint: hit.activeHint ?? false,
    supersededHint: hit.supersededHint ?? false,
    currentnessHint: hit.currentnessHint ?? "unknown",
    sourceKind: hit.lineage.sourceKind,
    sourceId: hit.lineage.sourceId,
    sourceRef: hit.lineage.sourceRef,
    normalizedSourceRefs: normalizeSourceRefs([hit.lineage.sourceRef]),
    canonicalKind: hit.lineage.canonicalKind,
    canonicalId: hit.lineage.canonicalId,
    goalMatches: hit.goalMatches,
    slotMatches: hit.slotMatches,
    bridgeMatches: hit.bridgeMatches,
    materializedEpoch:
      typeof hit.metadata?.materializedEpoch === "number"
        ? hit.metadata.materializedEpoch
        : undefined,
  });
  const summarizeLayerCandidate = (hit: LayerCandidateHit) => ({
    id: hit.id,
    layer: hit.layer,
    text: hit.text,
    sourceRef: hit.sourceRef,
    sourceRefs: hit.sourceRefs,
    normalizedSourceRefs: normalizeSourceRefs([hit.sourceRef, ...(hit.sourceRefs ?? [])]),
    observedAt: hit.observedAt,
    score: hit.score,
    slotMatches: hit.slotMatches,
    lineage: hit.lineage,
  });

  return {
    cutoverCriteria: CANDIDATE_GENERATION_CUTOVER_CRITERIA,
    budgets: result.budgets,
    surfaces: Object.fromEntries(
      Object.entries(result.surfaceStats).map(([surface, stats]) => {
        const primaryCandidates = result.candidates
          .filter((hit) => hit.surface === surface && hit.tier !== "alternate")
          .slice(0, 8)
          .map(summarizeCandidate);
        const alternateCandidates = result.candidates
          .filter((hit) => hit.surface === surface && hit.tier === "alternate")
          .slice(0, 4)
          .map(summarizeCandidate);
        return [
          surface,
          {
            ...stats,
            primaryCandidates,
            alternateCandidates,
          },
        ];
      }),
    ),
    candidateCount: result.candidates.length,
    slotCandidateCount: result.slotCandidates.length,
    bridgeCandidateCount: result.bridgeCandidates.length,
    layerCandidateCount: result.layerCandidates.length,
    alternateCandidateCount: result.alternateSearchHits.length,
    slotLayerStats: result.slotLayerStats,
    candidates: result.candidates.slice(0, 24).map(summarizeCandidate),
    slotCandidates: result.slotCandidates.slice(0, 32).map(summarizeCandidate),
    bridgeCandidates: result.bridgeCandidates.slice(0, 32).map(summarizeCandidate),
    layerCandidates: result.layerCandidates.slice(0, 24).map(summarizeLayerCandidate),
  };
}
