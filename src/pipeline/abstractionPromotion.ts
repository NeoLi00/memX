import type { MemxStoreBundle } from "../runtime.js";
import { nowIso, objectRecord, stableHash } from "../support.js";
import type {
  AbstractionCandidateRecord,
  AbstractionCandidateStage,
  AbstractionPromotionStats,
  ConversationTask,
  GraphRelationType,
  MaintenanceBatchMetadata,
  MaintenanceSemanticSource,
  MemoryOperationContext,
  NormalizedState,
  StrategyHypothesisRecord,
  SynthesizedTaskEvent,
  EntityType,
} from "../types.js";
import { refreshEntityProfileDocs } from "./entityProfile.js";
import { buildEntityMention, resolveEntityMention } from "./entityResolver.js";
import {
  buildMaintenanceContractMetadata,
  summarizeMaintenanceContractDiagnostics,
} from "./maintenanceContract.js";
import { buildOutcomeEventCandidate } from "./outcomeHypotheses.js";
import { normalizeGraphRelationType } from "./semantics.js";
import { inferStrategyHypothesisStage } from "./strategyHypotheses.js";
import { writeCandidate } from "./write.js";

const PROMOTION_POLICY_VERSION = "memx-authority-v3";

function currentSourceEpoch(store: MemxStoreBundle, ctx: MemoryOperationContext): number {
  return ctx.readEpoch ?? store.client.currentMemoryEpoch(ctx.agentId);
}

function candidateStageCounts(
  store: MemxStoreBundle,
  agentId: string,
): Record<AbstractionCandidateStage, number> {
  return {
    active: store.abstractionRepo.countByAgent({ agentId, stages: ["active"] }),
    candidate: store.abstractionRepo.countByAgent({ agentId, stages: ["candidate"] }),
    decaying: store.abstractionRepo.countByAgent({ agentId, stages: ["decaying"] }),
    probationary: store.abstractionRepo.countByAgent({ agentId, stages: ["probationary"] }),
    quarantined: store.abstractionRepo.countByAgent({ agentId, stages: ["quarantined"] }),
    superseded: store.abstractionRepo.countByAgent({ agentId, stages: ["superseded"] }),
  };
}

function eligibleForPromotion(candidate: AbstractionCandidateRecord): boolean {
  return candidate.stage === "probationary" || candidate.stage === "active";
}

function stringArray(value: unknown): string[] {
  if (!Array.isArray(value)) {
    return [];
  }
  return value.filter(
    (entry): entry is string => typeof entry === "string" && entry.trim().length > 0,
  );
}

function stringValue(value: unknown): string | undefined {
  return typeof value === "string" && value.trim() ? value.trim() : undefined;
}

function numberValue(value: unknown): number | undefined {
  return typeof value === "number" && Number.isFinite(value) ? value : undefined;
}

function generatedFromValue(value: unknown, fallback: string): string | string[] {
  if (typeof value === "string" && value.trim()) {
    return value.trim();
  }
  if (Array.isArray(value)) {
    const entries = value.filter(
      (entry): entry is string => typeof entry === "string" && entry.trim().length > 0,
    );
    if (entries.length > 0) {
      return entries;
    }
  }
  return fallback;
}

function parseWorkflowDomainKey(candidate: AbstractionCandidateRecord): string | undefined {
  const metadataDomainKey = stringValue(candidate.metadataJson.domainKey);
  if (metadataDomainKey) {
    return metadataDomainKey;
  }
  const prefix = `workflow_pattern:${candidate.scope}:`;
  if (candidate.semanticKey.startsWith(prefix)) {
    return candidate.semanticKey.slice(prefix.length) || undefined;
  }
  return undefined;
}

function inferDerivedStateKind(
  candidate: AbstractionCandidateRecord,
  stateKey: string,
): "session" | "durable" {
  const metadataStateKind = stringValue(candidate.metadataJson.promotedStateKind);
  if (metadataStateKind === "session" || metadataStateKind === "durable") {
    return metadataStateKind;
  }
  if (
    stateKey === "project.active_project" &&
    candidate.confidence >= 0.78 &&
    candidate.stabilityScore >= 0.68 &&
    candidate.contradictionScore <= 0.18
  ) {
    return "durable";
  }
  return "session";
}

function buildPromotionMetadata(
  candidate: AbstractionCandidateRecord,
  params: {
    targetKind: "event" | "graph" | "state" | "strategy";
    targetRef: string;
    sourceRef: string;
    promotedAt: string;
    derivedAtEpoch: number;
    derivedFromKind: string;
    derivedFromIds: string[];
    materializedEpoch?: number;
    extra?: Record<string, unknown>;
  },
): Record<string, unknown> {
  const promotion = objectRecord(candidate.metadataJson.promotion) ?? {};
  const semanticSource =
    (stringValue(candidate.metadataJson.semanticSource) as MaintenanceSemanticSource | undefined) ??
    "upstream_structured";
  const semanticSources = Array.isArray(candidate.metadataJson.semanticSources)
    ? candidate.metadataJson.semanticSources.filter(
        (entry): entry is MaintenanceSemanticSource =>
          typeof entry === "string" && entry.trim().length > 0,
      )
    : undefined;
  const baseMetadata = buildMaintenanceContractMetadata({
    existing: candidate.metadataJson,
    sourceRef: params.sourceRef,
    supportContentRefs: candidate.supportContentRefs,
    supportBeliefIds: candidate.supportBeliefIds,
    derivedFromIds: params.derivedFromIds,
    semanticSource,
    semanticSources,
    authoritySource: "deterministic_aggregated",
    generatedFrom: generatedFromValue(
      candidate.metadataJson.generatedFrom,
      "abstraction_promotion",
    ),
    recallLayer: params.targetKind === "strategy" ? "strategy" : params.targetKind,
    answerEligibleByDefault: params.targetKind !== "strategy",
    materializedEpoch: params.materializedEpoch,
    derivationPolicyVersion: PROMOTION_POLICY_VERSION,
  });
  return {
    ...baseMetadata,
    ...params.extra,
    promotion: {
      ...promotion,
      targetKind: params.targetKind,
      targetRef: params.targetRef,
      sourceRef: params.sourceRef,
      supportContentRefs: candidate.supportContentRefs,
      supportBeliefIds: candidate.supportBeliefIds,
      derivedAtEpoch: params.derivedAtEpoch,
      derivedFromKind: params.derivedFromKind,
      derivedFromIds: params.derivedFromIds,
      derivationPolicyVersion: PROMOTION_POLICY_VERSION,
      firstPromotedAt:
        stringValue(promotion.firstPromotedAt) ??
        stringValue(promotion.promotedAt) ??
        params.promotedAt,
      lastPromotedAt: params.promotedAt,
      ...params.extra,
    },
  };
}

function updateCandidatePromotionMetadata(
  store: MemxStoreBundle,
  candidate: AbstractionCandidateRecord,
  metadataJson: Record<string, unknown>,
  updatedAt: string,
  stage: AbstractionCandidateStage = candidate.stage,
  lineage?: {
    materializedEpoch: number;
    derivedAtEpoch: number;
    derivedFromKind: string;
    derivedFromIds: string[];
  },
): void {
  store.abstractionRepo.upsert({
    ...candidate,
    stage,
    ...(lineage
      ? {
          materializedEpoch: lineage.materializedEpoch,
          derivedFromMinEpoch: Math.min(
            candidate.derivedFromMinEpoch ?? lineage.derivedAtEpoch,
            lineage.derivedAtEpoch,
          ),
          derivedFromMaxEpoch: Math.max(
            candidate.derivedFromMaxEpoch ?? lineage.derivedAtEpoch,
            lineage.derivedAtEpoch,
          ),
          derivedFromKind: lineage.derivedFromKind,
          derivedFromIds: lineage.derivedFromIds,
          derivedAtEpoch: lineage.derivedAtEpoch,
          derivationPolicyVersion: PROMOTION_POLICY_VERSION,
        }
      : {}),
    metadataJson,
    updatedAt,
  });
}

function parseOutcomeHypothesisCandidate(
  store: MemxStoreBundle,
  candidate: AbstractionCandidateRecord,
): {
  task: ConversationTask;
  promotedOutcome: SynthesizedTaskEvent;
  observedAt: string;
  sourceRef: string;
  targetRef: string;
  alreadyEmitted: boolean;
} | null {
  const metadata = objectRecord(candidate.metadataJson);
  const taskId = stringValue(metadata?.taskId);
  const outcomeKey = stringValue(metadata?.outcomeKey);
  const eventType = stringValue(metadata?.eventType);
  const phase = stringValue(metadata?.phase);
  if (!taskId || !outcomeKey || !eventType || !phase) {
    return null;
  }
  const task = store.taskRepo.get(taskId);
  if (!task) {
    return null;
  }
  const promotedOutcome: SynthesizedTaskEvent = {
    eventType,
    summary: candidate.summary,
    phase: phase as SynthesizedTaskEvent["phase"],
    closureScore: numberValue(metadata?.closureScore) ?? candidate.confidence,
    verificationScore: numberValue(metadata?.verificationScore) ?? candidate.usefulnessScore,
    contradictionRisk: numberValue(metadata?.contradictionRisk) ?? candidate.contradictionScore,
    confidence: numberValue(metadata?.confidence) ?? candidate.confidence,
    promotionScore: numberValue(metadata?.promotionScore) ?? candidate.usefulnessScore,
    evidenceChunkIds: stringArray(metadata?.evidenceChunkIds),
    outcomeKey,
  };
  return {
    task,
    promotedOutcome,
    observedAt: stringValue(metadata?.observedAt) ?? candidate.updatedAt,
    sourceRef: `abstraction_candidate:${candidate.candidateId}`,
    targetRef: `task:${task.taskId}:outcome:${outcomeKey}`,
    alreadyEmitted: task.metadataJson.lastEmittedOutcomeKey === outcomeKey,
  };
}

function parseGraphHypothesisCandidate(candidate: AbstractionCandidateRecord): {
  relationType: GraphRelationType;
  relationSlot?: string;
  rawPredicate?: string;
  relationClass?: string;
  sourceName: string;
  targetName: string;
  sourceType?: string;
  targetType?: string;
  observedAt: string;
} | null {
  const metadata = objectRecord(candidate.metadataJson);
  const relationValue = stringValue(metadata?.relationType);
  const normalizedRelation = relationValue ? normalizeGraphRelationType(relationValue) : null;
  const sourceName = stringValue(metadata?.sourceName);
  const targetName = stringValue(metadata?.targetName);
  const relationType = normalizedRelation?.relationType;
  const durableRelationType =
    relationType === "depends_on" ||
    relationType === "blocks" ||
    relationType === "caused_by" ||
    relationType === "uses" ||
    relationType === "part_of" ||
    relationType === "owner_of" ||
    relationType === "supersedes" ||
    relationType === "contradicts" ||
    relationType === "resolved_by" ||
    relationType === "related_to" ||
    relationType === "reads"
      ? relationType
      : null;
  if (!durableRelationType || !sourceName || !targetName) {
    return null;
  }
  return {
    relationType: durableRelationType,
    relationSlot: stringValue(metadata?.relationSlot),
    rawPredicate: normalizedRelation?.rawPredicate,
    relationClass: stringValue(metadata?.relationClass),
    sourceName,
    targetName,
    sourceType: stringValue(metadata?.sourceType),
    targetType: stringValue(metadata?.targetType),
    observedAt: stringValue(metadata?.firstSeenAt) ?? candidate.updatedAt,
  };
}

function entityTypeValue(value: string | undefined): EntityType {
  switch (value) {
    case "person":
    case "project":
    case "tool":
    case "service":
    case "language":
    case "framework":
    case "concept":
    case "organization":
      return value;
    default:
      return "unknown";
  }
}

function shouldReplaceState(
  existing: NormalizedState | undefined,
  candidate: AbstractionCandidateRecord,
  sourceRef: string,
): boolean {
  if (!existing) {
    return true;
  }
  if (existing.sourceRef === sourceRef) {
    return true;
  }
  const lastSeenAt = stringValue(candidate.metadataJson.lastSeenAt) ?? candidate.updatedAt;
  const existingUpdated = Date.parse(existing.updatedAt);
  const candidateObserved = Date.parse(lastSeenAt);
  if (
    Number.isFinite(existingUpdated) &&
    Number.isFinite(candidateObserved) &&
    existingUpdated > candidateObserved &&
    existing.confidence >= candidate.confidence - 0.02
  ) {
    return false;
  }
  return existing.confidence + 0.08 < candidate.confidence;
}

function promoteDerivedState(
  store: MemxStoreBundle,
  ctx: MemoryOperationContext,
  candidate: AbstractionCandidateRecord,
): boolean {
  const stateKey = stringValue(candidate.metadataJson.stateKey);
  const valueJson = objectRecord(candidate.metadataJson.valueJson);
  if (!stateKey || !valueJson) {
    return false;
  }
  const stateKind = inferDerivedStateKind(candidate, stateKey);
  const sourceRef = `abstraction_candidate:${candidate.candidateId}`;
  const sourceEpoch = currentSourceEpoch(store, ctx);
  const existingState = store.stateRepo
    .get({
      agentId: ctx.agentId,
      scopes: [candidate.scope],
      key: stateKey,
      includeExpired: true,
      now: ctx.now,
      readEpoch: sourceEpoch,
    })
    .at(0);
  if (!shouldReplaceState(existingState, candidate, sourceRef)) {
    return false;
  }

  const promotionEpoch = store.client.nextMemoryEpoch(ctx.agentId, ctx.now);
  const promotedState: NormalizedState = {
    key: stateKey,
    valueJson,
    scope: candidate.scope,
    agentId: ctx.agentId,
    stateKind,
    confidence: candidate.confidence,
    sourceRef,
    updatedAt: ctx.now,
    materializedEpoch: promotionEpoch,
    expiresAt:
      stateKind === "session"
        ? store.stateRepo.createExpiry(ctx.now, ctx.config.stateTtlHours)
        : undefined,
  };
  store.stateRepo.upsert(promotedState);

  const targetRef = `${candidate.scope}:${stateKey}`;
  const metadataJson = buildPromotionMetadata(candidate, {
    targetKind: "state",
    targetRef,
    sourceRef,
    promotedAt: ctx.now,
    derivedAtEpoch: sourceEpoch,
    derivedFromKind: "abstraction_candidate",
    derivedFromIds: [candidate.candidateId],
    materializedEpoch: promotionEpoch,
    extra: {
      stateKey,
      stateKind,
      supportLastSeenAt: stringValue(candidate.metadataJson.lastSeenAt) ?? candidate.updatedAt,
    },
  });
  updateCandidatePromotionMetadata(store, candidate, metadataJson, ctx.now, candidate.stage, {
    materializedEpoch: promotionEpoch,
    derivedAtEpoch: sourceEpoch,
    derivedFromKind: "abstraction_candidate",
    derivedFromIds: [candidate.candidateId],
  });
  return true;
}

function promoteGraphHypothesis(
  store: MemxStoreBundle,
  ctx: MemoryOperationContext,
  candidate: AbstractionCandidateRecord,
): boolean {
  const parsed = parseGraphHypothesisCandidate(candidate);
  if (!parsed) {
    return false;
  }
  const sourceRef = `abstraction_candidate:${candidate.candidateId}`;
  const sourceResolution = resolveEntityMention(
    store,
    ctx,
    buildEntityMention({
      ctx,
      scope: candidate.scope,
      rawText: parsed.sourceName,
      proposedType: entityTypeValue(parsed.sourceType),
      semanticRole: parsed.sourceType === "project" ? "project" : "subject",
      sourceRef,
      supportText: candidate.summary,
      observedAt: parsed.observedAt,
      metadataJson: {
        abstractionCandidateId: candidate.candidateId,
        relationType: parsed.relationType,
        relationRole: "source",
      },
    }),
  );
  const targetResolution = resolveEntityMention(
    store,
    ctx,
    buildEntityMention({
      ctx,
      scope: candidate.scope,
      rawText: parsed.targetName,
      proposedType: entityTypeValue(parsed.targetType),
      semanticRole: parsed.targetType === "project" ? "project" : "object",
      sourceRef,
      supportText: candidate.summary,
      observedAt: parsed.observedAt,
      metadataJson: {
        abstractionCandidateId: candidate.candidateId,
        relationType: parsed.relationType,
        relationRole: "target",
      },
    }),
  );
  if (sourceResolution.method === "uncertain" || targetResolution.method === "uncertain") {
    return false;
  }
  const sourceEntity = sourceResolution.entity;
  const targetEntity = targetResolution.entity;
  const sourceEpoch = currentSourceEpoch(store, ctx);
  const promotionEpoch = store.client.nextMemoryEpoch(ctx.agentId, ctx.now);
  const edgeId = stableHash([
    ctx.agentId,
    candidate.scope,
    sourceEntity.entityId,
    parsed.relationType,
    parsed.relationSlot ?? "",
    targetEntity.entityId,
  ]);
  const edgeMetadata = buildPromotionMetadata(candidate, {
    targetKind: "graph",
    targetRef: edgeId,
    sourceRef,
    promotedAt: ctx.now,
    derivedAtEpoch: sourceEpoch,
    derivedFromKind: "abstraction_candidate",
    derivedFromIds: [candidate.candidateId],
    materializedEpoch: promotionEpoch,
    extra: {
      relationType: parsed.relationType,
      ...(parsed.relationSlot ? { relationSlot: parsed.relationSlot } : {}),
      sourceEntityId: sourceEntity.entityId,
      targetEntityId: targetEntity.entityId,
      sourceName: parsed.sourceName,
      targetName: parsed.targetName,
      sourceMentionId: sourceResolution.mention.mentionId,
      targetMentionId: targetResolution.mention.mentionId,
      sourceResolutionMethod: sourceResolution.method,
      targetResolutionMethod: targetResolution.method,
      sourceResolutionConfidence: sourceResolution.confidence,
      targetResolutionConfidence: targetResolution.confidence,
      relationClass: parsed.relationClass,
    },
  });
  store.graphRepo.upsertEdge({
    edgeId,
    srcEntityId: sourceEntity.entityId,
    relType: parsed.relationType,
    dstEntityId: targetEntity.entityId,
    relationSlot: parsed.relationSlot,
    scope: candidate.scope,
    agentId: ctx.agentId,
    confidence: candidate.confidence,
    validFrom: parsed.observedAt,
    evidenceRef: sourceRef,
    rawRelationType: parsed.rawPredicate,
    sourceKind: parsed.relationClass === "observed" ? "extracted" : "synthesized",
    createdAt: ctx.now,
    updatedAt: ctx.now,
    materializedEpoch: promotionEpoch,
    metadataJson: edgeMetadata,
  });
  refreshEntityProfileDocs(store, ctx, [sourceEntity.entityId, targetEntity.entityId]);
  updateCandidatePromotionMetadata(store, candidate, edgeMetadata, ctx.now, candidate.stage, {
    materializedEpoch: promotionEpoch,
    derivedAtEpoch: sourceEpoch,
    derivedFromKind: "abstraction_candidate",
    derivedFromIds: [candidate.candidateId],
  });
  return true;
}

function promoteWorkflowPattern(
  store: MemxStoreBundle,
  ctx: MemoryOperationContext,
  candidate: AbstractionCandidateRecord,
): boolean {
  const domainKey = parseWorkflowDomainKey(candidate);
  if (!domainKey) {
    return false;
  }
  const supportTaskIds = candidate.supportContentRefs
    .filter((ref) => ref.startsWith("task:"))
    .map((ref) => ref.slice("task:".length));
  const strategyId = stableHash([ctx.agentId, candidate.scope, domainKey]);
  const sourceEpoch = currentSourceEpoch(store, ctx);
  const promotionEpoch = store.client.nextMemoryEpoch(ctx.agentId, ctx.now);
  const stage = inferStrategyHypothesisStage({
    confidence: candidate.confidence,
    usefulnessScore: candidate.usefulnessScore,
    stabilityScore: candidate.stabilityScore,
    contradictionScore: candidate.contradictionScore,
    groundedByRoles: stringArray(candidate.metadataJson.groundedByRoles),
    taskPhase: stringValue(candidate.metadataJson.taskPhase),
    explicitInstruction: Boolean(candidate.metadataJson.explicitInstruction),
    groundedResolution: Boolean(candidate.metadataJson.groundedResolution),
  });
  const sourceRef = `abstraction_candidate:${candidate.candidateId}`;
  const strategyMetadata = buildMaintenanceContractMetadata({
    existing: {
      ...candidate.metadataJson,
      promotedFromCandidateId: candidate.candidateId,
      promotionSourceRef: sourceRef,
      supportContentRefs: candidate.supportContentRefs,
      supportBeliefIds: candidate.supportBeliefIds,
      promotedAt: ctx.now,
    },
    sourceRef,
    supportContentRefs: candidate.supportContentRefs,
    supportBeliefIds: candidate.supportBeliefIds,
    derivedFromIds: [candidate.candidateId],
    semanticSource:
      (stringValue(candidate.metadataJson.semanticSource) as
        | MaintenanceSemanticSource
        | undefined) ?? "upstream_structured",
    semanticSources: Array.isArray(candidate.metadataJson.semanticSources)
      ? candidate.metadataJson.semanticSources.filter(
          (entry): entry is MaintenanceSemanticSource =>
            typeof entry === "string" && entry.trim().length > 0,
        )
      : undefined,
    authoritySource: "deterministic_aggregated",
    generatedFrom: generatedFromValue(
      candidate.metadataJson.generatedFrom,
      "workflow_pattern_promotion",
    ),
    recallLayer: "strategy",
    answerEligibleByDefault: false,
    materializedEpoch: promotionEpoch,
    derivationPolicyVersion: PROMOTION_POLICY_VERSION,
  });
  const strategyRecord: StrategyHypothesisRecord = {
    strategyId,
    agentId: ctx.agentId,
    scope: candidate.scope,
    domainKey,
    summary: candidate.summary,
    supportBeliefIds: candidate.supportBeliefIds,
    supportTaskIds,
    confidence: candidate.confidence,
    usefulnessScore: candidate.usefulnessScore,
    stabilityScore: candidate.stabilityScore,
    contradictionScore: candidate.contradictionScore,
    stage,
    derivedFromMinEpoch: sourceEpoch,
    derivedFromMaxEpoch: sourceEpoch,
    materializedEpoch: promotionEpoch,
    derivedFromKind: "abstraction_candidate",
    derivedFromIds: [candidate.candidateId],
    derivedAtEpoch: sourceEpoch,
    derivationPolicyVersion: PROMOTION_POLICY_VERSION,
    metadataJson: strategyMetadata,
    createdAt: ctx.now,
    updatedAt: ctx.now,
  };
  store.strategyRepo.upsert(strategyRecord);

  const metadataJson = buildPromotionMetadata(candidate, {
    targetKind: "strategy",
    targetRef: strategyId,
    sourceRef,
    promotedAt: ctx.now,
    derivedAtEpoch: sourceEpoch,
    derivedFromKind: "abstraction_candidate",
    derivedFromIds: [candidate.candidateId],
    materializedEpoch: promotionEpoch,
    extra: {
      domainKey,
      strategyStage: stage,
      supportTaskIds,
    },
  });
  updateCandidatePromotionMetadata(store, candidate, metadataJson, ctx.now, candidate.stage, {
    materializedEpoch: promotionEpoch,
    derivedAtEpoch: sourceEpoch,
    derivedFromKind: "abstraction_candidate",
    derivedFromIds: [candidate.candidateId],
  });
  return true;
}

function promoteOutcomeHypothesis(
  store: MemxStoreBundle,
  ctx: MemoryOperationContext,
  candidate: AbstractionCandidateRecord,
): boolean {
  const parsed = parseOutcomeHypothesisCandidate(store, candidate);
  if (!parsed) {
    return false;
  }
  if (!parsed.alreadyEmitted) {
    const summary = writeCandidate(
      store,
      ctx,
      buildOutcomeEventCandidate(parsed.task, parsed.promotedOutcome, parsed.observedAt),
    );
    if (summary.events === 0) {
      return false;
    }
  }

  const sourceEpoch = currentSourceEpoch(store, ctx);
  const promotionEpoch = store.client.nextMemoryEpoch(ctx.agentId, ctx.now);
  const metadataJson = buildPromotionMetadata(candidate, {
    targetKind: "event",
    targetRef: parsed.targetRef,
    sourceRef: parsed.sourceRef,
    promotedAt: ctx.now,
    derivedAtEpoch: sourceEpoch,
    derivedFromKind: "abstraction_candidate",
    derivedFromIds: [candidate.candidateId],
    materializedEpoch: promotionEpoch,
    extra: {
      taskId: parsed.task.taskId,
      outcomeKey: parsed.promotedOutcome.outcomeKey,
      eventType: parsed.promotedOutcome.eventType,
      phase: parsed.promotedOutcome.phase,
      evidenceChunkIds: parsed.promotedOutcome.evidenceChunkIds,
    },
  });
  updateCandidatePromotionMetadata(store, candidate, metadataJson, ctx.now, "active", {
    materializedEpoch: promotionEpoch,
    derivedAtEpoch: sourceEpoch,
    derivedFromKind: "abstraction_candidate",
    derivedFromIds: [candidate.candidateId],
  });

  store.taskRepo.update(parsed.task.taskId, {
    metadataJson: {
      ...parsed.task.metadataJson,
      lastEmittedOutcomeKey: parsed.promotedOutcome.outcomeKey,
      lastEmittedOutcomeAt: ctx.now,
      lastPromotedOutcomeHypothesisId: candidate.candidateId,
    },
    updatedAt: ctx.now,
  });
  return true;
}

export function runAbstractionPromotion(
  store: MemxStoreBundle,
  ctx: MemoryOperationContext,
  options: {
    batch?: MaintenanceBatchMetadata;
    candidateIds?: string[];
    deltaTriggered?: boolean;
  } = {},
): AbstractionPromotionStats {
  const runStartedAt = nowIso();
  const runId = store.auditRepo.startMaintenance({
    agentId: ctx.agentId,
    sessionKey: options.batch?.sessionKey ?? ctx.sessionKey,
    jobType: "abstraction-promotion",
    stats: {},
    startedAt: runStartedAt,
  });
  try {
    if (options.batch && options.deltaTriggered === false) {
      const skippedStats: AbstractionPromotionStats = {
        candidatesEvaluated: 0,
        deltaTriggered: false,
        skippedNoRelevantDelta: true,
        activeCandidates: 0,
        probationaryCandidates: 0,
        candidateCandidates: 0,
        decayingCandidates: 0,
        quarantinedCandidates: 0,
        supersededCandidates: 0,
        promotedGraphs: 0,
        promotedOutcomes: 0,
        promotedStates: 0,
        promotedStrategies: 0,
        promotedConcepts: 0,
        skippedCandidates: [],
      };
      store.auditRepo.finishMaintenance({
        runId,
        agentId: ctx.agentId,
        jobType: "abstraction-promotion",
        statsJson: {
          ...(options.batch ? { batch: options.batch } : {}),
          ...skippedStats,
        },
        startedAt: runStartedAt,
        completedAt: nowIso(),
        status: "completed",
      });
      return skippedStats;
    }

    const candidates = options.candidateIds
      ? options.candidateIds
          .map((candidateId) => store.abstractionRepo.getById(candidateId))
          .filter((candidate): candidate is AbstractionCandidateRecord => candidate !== undefined)
      : store.abstractionRepo.listByAgent({
          agentId: ctx.agentId,
          scopes: ctx.scopes,
          limit: 64,
        });
    const stages = candidateStageCounts(store, ctx.agentId);
    const maintenanceContractDiagnostics = summarizeMaintenanceContractDiagnostics(
      candidates.map((candidate) => candidate.metadataJson),
    );
    const stats: AbstractionPromotionStats = {
      candidatesEvaluated: candidates.length,
      ...(options.batch ? { deltaTriggered: options.deltaTriggered !== false } : {}),
      skippedNoRelevantDelta: false,
      activeCandidates: stages.active,
      probationaryCandidates: stages.probationary,
      candidateCandidates: stages.candidate,
      decayingCandidates: stages.decaying,
      quarantinedCandidates: stages.quarantined,
      supersededCandidates: stages.superseded,
      promotedGraphs: 0,
      promotedOutcomes: 0,
      promotedStates: 0,
      promotedStrategies: 0,
      promotedConcepts: 0,
      skippedCandidates: [],
      maintenanceContractDiagnostics,
      recallFacingDiagnostics: {
        recallVisible: maintenanceContractDiagnostics.recallVisibleCount > 0,
        answerEligibleByDefault: maintenanceContractDiagnostics.answerEligibleByDefaultCount > 0,
        sourceRefsForExpansion: maintenanceContractDiagnostics.sourceRefsForExpansion,
        recallLayers: maintenanceContractDiagnostics.recallLayers,
      },
    };

    for (const candidate of candidates) {
      if (!eligibleForPromotion(candidate)) {
        stats.skippedCandidates?.push({
          candidateId: candidate.candidateId,
          abstractionType: candidate.abstractionType,
          reason: "stage_not_promotable",
        });
        continue;
      }
      if (candidate.abstractionType === "outcome_hypothesis") {
        if (promoteOutcomeHypothesis(store, ctx, candidate)) {
          stats.promotedOutcomes += 1;
        } else {
          stats.skippedCandidates?.push({
            candidateId: candidate.candidateId,
            abstractionType: candidate.abstractionType,
            reason: "outcome_not_grounded_or_already_materialized",
          });
        }
        continue;
      }
      if (candidate.abstractionType === "derived_state") {
        if (promoteDerivedState(store, ctx, candidate)) {
          stats.promotedStates += 1;
        } else {
          stats.skippedCandidates?.push({
            candidateId: candidate.candidateId,
            abstractionType: candidate.abstractionType,
            reason: "state_promotion_blocked",
          });
        }
        continue;
      }
      if (candidate.abstractionType === "graph_hypothesis") {
        if (promoteGraphHypothesis(store, ctx, candidate)) {
          stats.promotedGraphs += 1;
        } else {
          stats.skippedCandidates?.push({
            candidateId: candidate.candidateId,
            abstractionType: candidate.abstractionType,
            reason: "graph_promotion_blocked_or_unresolved",
          });
        }
        continue;
      }
      if (candidate.abstractionType === "workflow_pattern") {
        if (promoteWorkflowPattern(store, ctx, candidate)) {
          stats.promotedStrategies += 1;
        } else {
          stats.skippedCandidates?.push({
            candidateId: candidate.candidateId,
            abstractionType: candidate.abstractionType,
            reason: "workflow_pattern_not_grounded_or_invalid",
          });
        }
        continue;
      }
      if (candidate.abstractionType === "concept_candidate") {
        stats.skippedCandidates?.push({
          candidateId: candidate.candidateId,
          abstractionType: candidate.abstractionType,
          reason: "not_promotable_yet",
        });
      }
    }

    store.auditRepo.finishMaintenance({
      runId,
      agentId: ctx.agentId,
      jobType: "abstraction-promotion",
      statsJson: {
        ...(options.batch ? { batch: options.batch } : {}),
        ...stats,
      },
      startedAt: runStartedAt,
      completedAt: nowIso(),
      status: "completed",
    });
    return stats;
  } catch (error) {
    store.auditRepo.finishMaintenance({
      runId,
      agentId: ctx.agentId,
      jobType: "abstraction-promotion",
      statsJson: {
        ...(options.batch ? { batch: options.batch } : {}),
        error: String(error),
      },
      startedAt: runStartedAt,
      completedAt: nowIso(),
      status: "failed",
    });
    throw error;
  }
}
