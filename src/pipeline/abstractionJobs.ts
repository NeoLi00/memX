import type { MemxStoreBundle } from "../runtime.js";
import {
  clamp01,
  normalizeText,
  nowIso,
  objectRecord,
  stableHash,
  truncateText,
} from "../support.js";
import type {
  ConversationTask,
  GraphEvidence,
  GraphEvidenceEdge,
  GraphEvidenceNode,
  GraphTraversalRelationType,
  NormalizedFact,
  NormalizedState,
} from "../types.js";
import type {
  AbstractionCandidateRecord,
  AbstractionCandidateStage,
  AbstractionJobStats,
  MaintenanceBatchMetadata,
  ConversationChunk,
  MaintenanceSemanticSource,
  MemoryCandidateRelationHint,
  MemoryCandidateStructuredHints,
  MemoryCandidateWorkflowHint,
  MemoryBeliefRecord,
  MemoryOperationContext,
} from "../types.js";
import { applyAbstractionRefinement, eligibleForLlmRefinement } from "./abstractionRefinement.js";
import { buildEntityMention, resolveEntityMention } from "./entityResolver.js";
import { buildGraphPathCandidates } from "./graphPathEngine.js";
import { snapshotMemoryLlmBudgetAudit } from "./llmBudgetAudit.js";
import {
  buildMaintenanceContractMetadata,
  summarizeMaintenanceContractDiagnostics,
} from "./maintenanceContract.js";
import { describeStateValue } from "./memoryObjectsHelpers.js";
import { tokenizeSearchTerms } from "./semantic/heuristics.js";
import { semanticTextSimilarity } from "./semantic/textSimilarity.js";
import { canonicalStateKey } from "./semantics.js";
import { contentStructuralComplexity } from "./sourceWeighting.js";
import { deriveWorkflowPatternSummaries } from "./strategyHypotheses.js";

const ABSTRACTION_CANDIDATE_CAP = 12;
const ABSTRACTION_CANDIDATE_BUDGETS: Record<AbstractionCandidateRecord["abstractionType"], number> =
  {
    derived_state: 4,
    workflow_pattern: 4,
    graph_hypothesis: 3,
    concept_candidate: 1,
    outcome_hypothesis: 0,
  };
const DERIVED_STATE_EVENT_WINDOW_DAYS = 14;
const ABSTRACTION_LLM_REFINEMENT_LIMIT = 3;
const CONCEPT_FACT_LIMIT = 48;
const CONCEPT_GRAPH_EDGE_BUDGET = 8;
const CONCEPT_GRAPH_NODE_BUDGET = 10;
const GRAPH_HYPOTHESIS_FACT_LIMIT = 48;
const DERIVED_STATE_SUPPORTED_KEYS = new Set([
  "project.active_project",
  "workflow.current_task",
  "workflow.next_action",
  "workflow.blocker",
]);
const CONCEPT_GENERIC_NAMES = new Set(["assistant", "system", "user"]);
const EXPLICIT_STRATEGY_STOPWORDS = new Set([
  "the",
  "and",
  "for",
  "with",
  "from",
  "that",
  "this",
  "into",
  "when",
  "then",
  "task",
  "issue",
  "problem",
  "work",
  "using",
  "still",
  "need",
  "needs",
  "confirmation",
  "user",
  "decision",
  "summary",
  "处理",
  "问题",
  "任务",
  "需要",
  "确认",
  "当前",
  "用户",
  "总结",
]);
const WORKFLOW_GUIDANCE_PREDICATE = "has_workflow_guidance";
const ADVICE_SIGNAL_PREDICATE = "has_advice_signal";
const CONCEPT_RELATIONAL_FACT_PREDICATES = new Set([
  "depends_on",
  "blocks",
  "caused_by",
  "uses",
  "part_of",
  "owner_of",
  "supersedes",
  "contradicts",
  "resolved_by",
  "related_to",
  "reads",
]);

function relationalFactGraphRelation(
  fact: Pick<NormalizedFact, "predicate" | "objectValueJson">,
): GraphTraversalRelationType | null {
  const graph = objectRecord(fact.objectValueJson?.graph);
  const explicit = stringValue(graph?.relationType);
  if (
    explicit &&
    (CONCEPT_RELATIONAL_FACT_PREDICATES.has(explicit) || explicit === "supported_by")
  ) {
    return explicit as GraphTraversalRelationType;
  }
  if (CONCEPT_RELATIONAL_FACT_PREDICATES.has(fact.predicate)) {
    return fact.predicate as GraphTraversalRelationType;
  }
  if (fact.predicate.startsWith("uses_")) {
    return "uses";
  }
  return null;
}

type StateSupportEntry = {
  scope: string;
  stateKey: string;
  valueJson: Record<string, unknown>;
  valueText: string;
  contentRef: string;
  supportKind: "event" | "task";
  observedAt: string;
  sessionKey?: string;
  confidence: number;
  usefulnessScore: number;
  stabilityScore: number;
  contradictionScore: number;
  beliefId?: string;
};

type ConceptSupportEntry = {
  supportKind: "fact" | "graph_edge";
  contentRef: string;
  relationLabel: string;
  observedAt: string;
  confidence: number;
  usefulnessScore: number;
  stabilityScore: number;
  contradictionScore: number;
  beliefId?: string;
};

type GraphHypothesisRelationClass = "observed" | "inferred";

type GraphHypothesisSupportEntry = {
  scope: string;
  relationType: GraphTraversalRelationType;
  relationSlot?: string;
  relationClass: GraphHypothesisRelationClass;
  sourceName: string;
  targetName: string;
  sourceType: string;
  targetType: string;
  contentRef: string;
  supportKind: "event" | "fact" | "task" | "state";
  observedAt: string;
  sessionKey?: string;
  confidence: number;
  usefulnessScore: number;
  stabilityScore: number;
  contradictionScore: number;
  beliefId?: string;
};

function olderThanDays(days: number, now: string = new Date().toISOString()): string {
  const date = new Date(now);
  if (!Number.isFinite(date.getTime())) {
    const fallback = new Date();
    fallback.setUTCDate(fallback.getUTCDate() - days);
    return fallback.toISOString();
  }
  date.setUTCDate(date.getUTCDate() - days);
  return date.toISOString();
}

function compareAbstractionCandidates(
  left: AbstractionCandidateRecord,
  right: AbstractionCandidateRecord,
): number {
  if (right.confidence !== left.confidence) {
    return right.confidence - left.confidence;
  }
  if (right.usefulnessScore !== left.usefulnessScore) {
    return right.usefulnessScore - left.usefulnessScore;
  }
  return right.stabilityScore - left.stabilityScore;
}

function selectAbstractionCandidatesByType(
  candidatesByType: Partial<
    Record<AbstractionCandidateRecord["abstractionType"], AbstractionCandidateRecord[]>
  >,
): {
  selected: AbstractionCandidateRecord[];
  stats: NonNullable<AbstractionJobStats["candidateSelection"]>;
} {
  const selectedIds = new Set<string>();
  const selected: AbstractionCandidateRecord[] = [];
  const stats: NonNullable<AbstractionJobStats["candidateSelection"]> = {
    cap: ABSTRACTION_CANDIDATE_CAP,
    byType: {},
  };
  for (const [type, budget] of Object.entries(ABSTRACTION_CANDIDATE_BUDGETS) as Array<
    [AbstractionCandidateRecord["abstractionType"], number]
  >) {
    const ranked = [...(candidatesByType[type] ?? [])].sort(compareAbstractionCandidates);
    const picked = ranked.slice(0, Math.max(0, budget));
    for (const candidate of picked) {
      selectedIds.add(candidate.candidateId);
      selected.push(candidate);
    }
    stats.byType[type] = {
      budget,
      available: ranked.length,
      selected: picked.length,
      deferred: Math.max(0, ranked.length - picked.length),
    };
  }

  const leftovers = Object.values(candidatesByType)
    .flat()
    .filter((candidate): candidate is AbstractionCandidateRecord => Boolean(candidate))
    .filter((candidate) => !selectedIds.has(candidate.candidateId))
    .sort(compareAbstractionCandidates);
  for (const candidate of leftovers) {
    if (selected.length >= ABSTRACTION_CANDIDATE_CAP) {
      break;
    }
    selectedIds.add(candidate.candidateId);
    selected.push(candidate);
    const typeStats =
      stats.byType[candidate.abstractionType] ??
      (stats.byType[candidate.abstractionType] = {
        budget: ABSTRACTION_CANDIDATE_BUDGETS[candidate.abstractionType] ?? 0,
        available: 0,
        selected: 0,
        deferred: 0,
      });
    typeStats.selected += 1;
    typeStats.deferred = Math.max(0, typeStats.available - typeStats.selected);
  }

  return { selected: selected.sort(compareAbstractionCandidates), stats };
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

function average(values: number[]): number {
  if (values.length === 0) {
    return 0;
  }
  return values.reduce((sum, value) => sum + value, 0) / values.length;
}

function daysBetween(earlierIso: string, laterIso: string): number {
  const deltaMs = Date.parse(laterIso) - Date.parse(earlierIso);
  return Number.isFinite(deltaMs) && deltaMs > 0 ? deltaMs / 86_400_000 : 0;
}

function buildBeliefMap(beliefs: MemoryBeliefRecord[]): Map<string, MemoryBeliefRecord> {
  const entries = beliefs
    .filter((belief) => belief.contentRef)
    .map((belief) => [`${belief.memoryKind}:${belief.contentRef}`, belief] as const);
  return new Map(entries);
}

function stringValue(value: unknown): string | undefined {
  return typeof value === "string" && value.trim() ? value.trim() : undefined;
}

function eventStructuredHints(
  event: ReturnType<MemxStoreBundle["eventRepo"]["search"]>[number],
): MemoryCandidateStructuredHints | null {
  const metadata = objectRecord(event.metadataJson);
  const structured = objectRecord(metadata?.memxStructuredHints);
  if (!structured) {
    return null;
  }
  return structured as MemoryCandidateStructuredHints;
}

function eventWorkflowHints(
  event: ReturnType<MemxStoreBundle["eventRepo"]["search"]>[number],
): MemoryCandidateWorkflowHint[] {
  const hints = eventStructuredHints(event);
  if (!hints) {
    return [];
  }
  if (Array.isArray(hints.workflows) && hints.workflows.length > 0) {
    return hints.workflows;
  }
  if (hints.workflow) {
    return [hints.workflow];
  }
  return [];
}

function eventRelationHints(
  event: ReturnType<MemxStoreBundle["eventRepo"]["search"]>[number],
): MemoryCandidateRelationHint[] {
  const hints = eventStructuredHints(event);
  if (!hints) {
    return [];
  }
  if (Array.isArray(hints.relations) && hints.relations.length > 0) {
    return hints.relations;
  }
  return hints.relation ? [hints.relation] : [];
}

function graphHypothesisNodeId(name: string, type: string): string {
  return `graph_hypothesis_node:${stableHash([normalizeText(name), type])}`;
}

function graphRelationPhrase(relationType: GraphTraversalRelationType): string {
  switch (relationType) {
    case "depends_on":
      return "depends on";
    case "blocks":
      return "blocks";
    case "caused_by":
      return "is caused by";
    case "uses":
      return "uses";
    case "part_of":
      return "is part of";
    case "owner_of":
      return "owns";
    case "supersedes":
      return "supersedes";
    case "contradicts":
      return "contradicts";
    case "resolved_by":
      return "is resolved by";
    case "related_to":
      return "relates to";
    case "reads":
      return "reads";
    case "supported_by":
      return "is supported by";
    case "derived_from":
      return "is derived from";
    case "updates":
      return "updates";
    case "targets":
      return "targets";
  }
}

function semanticValueKey(key: string, valueJson: Record<string, unknown>): string {
  return normalizeText(`${canonicalStateKey(key)}:${describeStateValue(key, valueJson)}`);
}

function derivedStateSummary(key: string, valueText: string): string {
  switch (canonicalStateKey(key)) {
    case "project.active_project":
      return `The active project appears to be ${valueText}.`;
    case "workflow.current_task":
      return `The current working focus appears to be ${valueText}.`;
    case "workflow.next_action":
      return `The next likely action appears to be ${valueText}.`;
    case "workflow.blocker":
      return `The current blocker appears to be ${valueText}.`;
    default:
      return `The current state appears to be ${valueText}.`;
  }
}

function buildEventSupport(
  beliefMap: Map<string, MemoryBeliefRecord>,
  event: ReturnType<MemxStoreBundle["eventRepo"]["search"]>[number],
): StateSupportEntry | null {
  const workflow = eventWorkflowHints(event)[0];
  if (!workflow) {
    return null;
  }
  const stateKey = canonicalStateKey(workflow.key);
  if (!DERIVED_STATE_SUPPORTED_KEYS.has(stateKey)) {
    return null;
  }
  const belief = beliefMap.get(`event:${event.eventId}`);
  return {
    scope: event.scope,
    stateKey,
    valueJson: workflow.value,
    valueText: describeStateValue(stateKey, workflow.value),
    contentRef: `event:${event.eventId}`,
    supportKind: "event",
    observedAt: event.observedAt,
    sessionKey: event.sessionKey,
    confidence: clamp01(average([workflow.confidence ?? 0.6, belief?.posteriorConfidence ?? 0.58])),
    usefulnessScore: belief?.usefulnessScore ?? 0.42,
    stabilityScore: belief?.stabilityScore ?? 0.48,
    contradictionScore: belief?.contradictionScore ?? 0.08,
    beliefId: belief?.beliefId,
  };
}

function taskMetadataStateEntries(task: ConversationTask): Array<{
  key: string;
  valueJson: Record<string, unknown>;
}> {
  const metadata = task.metadataJson ?? {};
  const entries: Array<{ key: string; valueJson: Record<string, unknown> }> = [];
  if (typeof metadata.project === "string" && metadata.project.trim()) {
    entries.push({
      key: "project.active_project",
      valueJson: { project: metadata.project.trim(), status: "active" },
    });
  }
  if (typeof metadata.currentTask === "string" && metadata.currentTask.trim()) {
    entries.push({
      key: "workflow.current_task",
      valueJson: { task: metadata.currentTask.trim() },
    });
  }
  if (typeof metadata.nextAction === "string" && metadata.nextAction.trim()) {
    entries.push({
      key: "workflow.next_action",
      valueJson: { step: metadata.nextAction.trim() },
    });
  }
  if (typeof metadata.blocker === "string" && metadata.blocker.trim()) {
    entries.push({
      key: "workflow.blocker",
      valueJson: { blocker: metadata.blocker.trim(), status: "blocked" },
    });
  }
  return entries;
}

function buildTaskSupports(
  beliefMap: Map<string, MemoryBeliefRecord>,
  task: ConversationTask,
): StateSupportEntry[] {
  const belief = beliefMap.get(`task:${task.taskId}`);
  return taskMetadataStateEntries(task).map((entry) => ({
    scope: task.scope,
    stateKey: canonicalStateKey(entry.key),
    valueJson: entry.valueJson,
    valueText: describeStateValue(entry.key, entry.valueJson),
    contentRef: `task:${task.taskId}`,
    supportKind: "task",
    observedAt: task.updatedAt,
    sessionKey: task.sessionKey,
    confidence: clamp01(average([belief?.posteriorConfidence ?? 0.7, 0.84])),
    usefulnessScore: belief?.usefulnessScore ?? 0.6,
    stabilityScore: belief?.stabilityScore ?? 0.64,
    contradictionScore: belief?.contradictionScore ?? 0.06,
    beliefId: belief?.beliefId,
  }));
}

function supportDiversityScore(entries: StateSupportEntry[]): number {
  const supportKinds = new Set(entries.map((entry) => entry.supportKind));
  const sessions = new Set(entries.map((entry) => entry.sessionKey).filter(Boolean));
  return clamp01(
    Math.min(0.45, (entries.length - 1) * 0.12) +
      Math.min(0.25, Math.max(0, supportKinds.size - 1) * 0.25) +
      Math.min(0.3, Math.max(0, sessions.size - 1) * 0.18),
  );
}

function temporalPersistenceScore(entries: StateSupportEntry[]): number {
  if (entries.length === 0) {
    return 0;
  }
  const observedAt = entries.map((entry) => entry.observedAt).sort();
  const spreadDays = daysBetween(observedAt[0]!, observedAt.at(-1)!);
  return clamp01(
    Math.min(0.5, spreadDays / 7) +
      Math.min(0.22, Math.max(0, entries.length - 1) * 0.08) +
      (spreadDays >= 1 ? 0.16 : 0),
  );
}

function contradictionPressure(
  entries: StateSupportEntry[],
  siblingGroups: StateSupportEntry[][],
): number {
  const totalConfidence = entries.reduce((sum, entry) => sum + entry.confidence, 0);
  const competingConfidence = siblingGroups
    .filter((group) => group !== entries)
    .reduce(
      (sum, group) => sum + group.reduce((groupSum, entry) => groupSum + entry.confidence, 0),
      0,
    );
  const localConflict =
    totalConfidence + competingConfidence > 0
      ? competingConfidence / (totalConfidence + competingConfidence)
      : 0;
  return clamp01(
    localConflict * 0.75 + average(entries.map((entry) => entry.contradictionScore)) * 0.25,
  );
}

function conceptSupportDiversity(entries: ConceptSupportEntry[]): number {
  const supportKinds = new Set(entries.map((entry) => entry.supportKind));
  const relationFamilies = new Set(entries.map((entry) => entry.relationLabel));
  return clamp01(
    Math.min(0.32, Math.max(0, entries.length - 1) * 0.08) +
      Math.min(0.28, Math.max(0, supportKinds.size - 1) * 0.28) +
      Math.min(0.4, Math.max(0, relationFamilies.size - 1) * 0.12),
  );
}

function conceptTemporalPersistence(entries: ConceptSupportEntry[]): number {
  if (entries.length === 0) {
    return 0;
  }
  const observedAt = entries.map((entry) => entry.observedAt).sort();
  const spreadDays = daysBetween(observedAt[0]!, observedAt.at(-1)!);
  return clamp01(
    Math.min(0.54, spreadDays / 30) +
      Math.min(0.18, Math.max(0, entries.length - 1) * 0.06) +
      (spreadDays >= 3 ? 0.12 : 0),
  );
}

function topLabels(entries: ConceptSupportEntry[], limit = 4): string[] {
  const counts = new Map<string, number>();
  for (const entry of entries) {
    counts.set(entry.relationLabel, (counts.get(entry.relationLabel) ?? 0) + 1);
  }
  return [...counts.entries()]
    .sort((left, right) => right[1] - left[1] || left[0].localeCompare(right[0]))
    .map(([label]) => label)
    .slice(0, limit);
}

function conceptSummary(entityName: string, relationLabels: string[]): string {
  const relationSummary =
    relationLabels.length > 0 ? relationLabels.join(", ") : "stable supporting relations";
  return truncateText(
    `A stable concept cluster appears around ${entityName}, linking ${relationSummary}.`,
    220,
  );
}

function conceptCandidateStage(params: {
  confidence: number;
  usefulnessScore: number;
  stabilityScore: number;
  contradictionScore: number;
  structuralStrength: number;
}): AbstractionCandidateStage {
  if (params.contradictionScore >= 0.46) {
    return "quarantined";
  }
  if (
    params.confidence >= 0.82 &&
    params.usefulnessScore >= 0.64 &&
    params.stabilityScore >= 0.66 &&
    params.structuralStrength >= 0.52
  ) {
    return "probationary";
  }
  return "candidate";
}

function trimGraphHypothesisName(value: string): string {
  return truncateText(value.trim(), 140).trim();
}

function graphHypothesisSupportDiversity(entries: GraphHypothesisSupportEntry[]): number {
  const supportKinds = new Set(entries.map((entry) => entry.supportKind));
  const sessions = new Set(entries.map((entry) => entry.sessionKey).filter(Boolean));
  return clamp01(
    Math.min(0.4, Math.max(0, entries.length - 1) * 0.1) +
      Math.min(0.3, Math.max(0, supportKinds.size - 1) * 0.18) +
      Math.min(0.3, Math.max(0, sessions.size - 1) * 0.16),
  );
}

function graphHypothesisTemporalPersistence(entries: GraphHypothesisSupportEntry[]): number {
  if (entries.length === 0) {
    return 0;
  }
  const observedAt = entries.map((entry) => entry.observedAt).sort();
  const spreadDays = daysBetween(observedAt[0]!, observedAt.at(-1)!);
  return clamp01(
    Math.min(0.5, spreadDays / 10) +
      Math.min(0.18, Math.max(0, entries.length - 1) * 0.08) +
      (spreadDays >= 1 ? 0.14 : 0),
  );
}

function graphHypothesisContradictionPressure(
  entries: GraphHypothesisSupportEntry[],
  siblingGroups: GraphHypothesisSupportEntry[][],
): number {
  const totalConfidence = entries.reduce((sum, entry) => sum + entry.confidence, 0);
  const competingConfidence = siblingGroups
    .filter((group) => group !== entries)
    .reduce(
      (sum, group) => sum + group.reduce((groupSum, entry) => groupSum + entry.confidence, 0),
      0,
    );
  const localConflict =
    totalConfidence + competingConfidence > 0
      ? competingConfidence / (totalConfidence + competingConfidence)
      : 0;
  return clamp01(
    localConflict * 0.72 + average(entries.map((entry) => entry.contradictionScore)) * 0.28,
  );
}

function graphHypothesisSummary(
  relationClass: GraphHypothesisRelationClass,
  relationType: GraphTraversalRelationType,
  sourceName: string,
  targetName: string,
): string {
  const clause = `${sourceName} ${graphRelationPhrase(relationType)} ${targetName}`;
  if (relationClass === "observed") {
    return truncateText(clause, 220);
  }
  return truncateText(`A recurring structure suggests ${clause}.`, 220);
}

function graphHypothesisStage(params: {
  relationClass: GraphHypothesisRelationClass;
  confidence: number;
  usefulnessScore: number;
  stabilityScore: number;
  contradictionScore: number;
  supportCount: number;
  supportDiversity: number;
  temporalPersistence: number;
}): AbstractionCandidateStage {
  if (params.contradictionScore >= 0.5) {
    return "quarantined";
  }
  if (
    params.relationClass === "observed" &&
    params.confidence >= 0.74 &&
    params.usefulnessScore >= 0.5 &&
    params.stabilityScore >= 0.5
  ) {
    return "probationary";
  }
  if (
    params.confidence >= 0.68 &&
    params.supportCount >= 2 &&
    (params.supportDiversity >= 0.28 || params.temporalPersistence >= 0.18)
  ) {
    return "probationary";
  }
  return "candidate";
}

function buildEventGraphSupports(
  beliefMap: Map<string, MemoryBeliefRecord>,
  event: ReturnType<MemxStoreBundle["eventRepo"]["search"]>[number],
): GraphHypothesisSupportEntry[] {
  const relations = eventRelationHints(event);
  if (relations.length === 0) {
    return [];
  }
  const belief = beliefMap.get(`event:${event.eventId}`);
  return relations.map((relation) => ({
    scope: event.scope,
    relationType: relation.predicate,
    relationSlot: relation.relationSlot,
    relationClass: "observed",
    sourceName: trimGraphHypothesisName(relation.subject),
    targetName: trimGraphHypothesisName(relation.object),
    sourceType: "entity",
    targetType: "entity",
    contentRef: `event:${event.eventId}`,
    supportKind: "event",
    observedAt: event.observedAt,
    sessionKey: event.sessionKey,
    confidence: clamp01(average([event.confidence, belief?.posteriorConfidence ?? 0.68])),
    usefulnessScore: belief?.usefulnessScore ?? 0.46,
    stabilityScore: belief?.stabilityScore ?? 0.54,
    contradictionScore: belief?.contradictionScore ?? 0.08,
    beliefId: belief?.beliefId,
  }));
}

function buildFactGraphSupport(
  beliefMap: Map<string, MemoryBeliefRecord>,
  fact: NormalizedFact,
): GraphHypothesisSupportEntry | null {
  const relationType = relationalFactGraphRelation(fact);
  if (!fact.canonicalObject || !relationType) {
    return null;
  }
  const belief = beliefMap.get(`fact:${fact.factId}`);
  return {
    scope: fact.scope,
    relationType,
    relationSlot: stringValue(objectRecord(fact.objectValueJson?.graph)?.relationSlot),
    relationClass: "observed",
    sourceName: trimGraphHypothesisName(fact.canonicalSubject),
    targetName: trimGraphHypothesisName(fact.canonicalObject),
    sourceType: "entity",
    targetType: "entity",
    contentRef: `fact:${fact.factId}`,
    supportKind: "fact",
    observedAt: fact.updatedAt,
    confidence: clamp01(average([fact.confidence, belief?.posteriorConfidence ?? 0.72])),
    usefulnessScore: belief?.usefulnessScore ?? 0.5,
    stabilityScore: belief?.stabilityScore ?? 0.58,
    contradictionScore: belief?.contradictionScore ?? 0.08,
    beliefId: belief?.beliefId,
  };
}

function pushInferredGraphSupport(
  supports: GraphHypothesisSupportEntry[],
  params: {
    scope: string;
    relationType: GraphTraversalRelationType;
    sourceName?: string;
    targetName?: string;
    sourceType: string;
    targetType: string;
    contentRef: string;
    supportKind: "task" | "state";
    observedAt: string;
    sessionKey?: string;
    confidence: number;
    usefulnessScore: number;
    stabilityScore: number;
    contradictionScore: number;
    beliefId?: string;
  },
): void {
  const sourceName = trimGraphHypothesisName(params.sourceName ?? "");
  const targetName = trimGraphHypothesisName(params.targetName ?? "");
  if (!sourceName || !targetName || sourceName === targetName) {
    return;
  }
  supports.push({
    scope: params.scope,
    relationType: params.relationType,
    relationClass: "inferred",
    sourceName,
    targetName,
    sourceType: params.sourceType,
    targetType: params.targetType,
    contentRef: params.contentRef,
    supportKind: params.supportKind,
    observedAt: params.observedAt,
    sessionKey: params.sessionKey,
    confidence: clamp01(params.confidence),
    usefulnessScore: clamp01(params.usefulnessScore),
    stabilityScore: clamp01(params.stabilityScore),
    contradictionScore: clamp01(params.contradictionScore),
    beliefId: params.beliefId,
  });
}

function buildGraphHypothesisCandidate(params: {
  agentId: string;
  scope: string;
  relationType: GraphTraversalRelationType;
  relationSlot?: string;
  relationClass: GraphHypothesisRelationClass;
  sourceName: string;
  targetName: string;
  sourceType: string;
  targetType: string;
  supports: GraphHypothesisSupportEntry[];
  siblingGroups: GraphHypothesisSupportEntry[][];
  now: string;
}): AbstractionCandidateRecord | null {
  const sourceName = trimGraphHypothesisName(params.sourceName);
  const targetName = trimGraphHypothesisName(params.targetName);
  if (!sourceName || !targetName || sourceName === targetName) {
    return null;
  }
  const supportDiversity = graphHypothesisSupportDiversity(params.supports);
  const temporalPersistence = graphHypothesisTemporalPersistence(params.supports);
  const contradictionScore = graphHypothesisContradictionPressure(
    params.supports,
    params.siblingGroups,
  );
  const usefulnessScore = clamp01(average(params.supports.map((entry) => entry.usefulnessScore)));
  const stabilityScore = clamp01(
    average(params.supports.map((entry) => entry.stabilityScore)) * 0.34 +
      temporalPersistence * 0.26 +
      supportDiversity * 0.16 +
      (params.relationClass === "observed" ? 0.12 : 0.06) +
      (1 - contradictionScore) * 0.12,
  );
  const confidence = clamp01(
    average(params.supports.map((entry) => entry.confidence)) * 0.42 +
      supportDiversity * 0.18 +
      temporalPersistence * 0.12 +
      usefulnessScore * 0.1 +
      (params.relationClass === "observed" ? 0.12 : 0.06) +
      (1 - contradictionScore) * 0.06,
  );
  if (confidence < 0.52) {
    return null;
  }
  const stage = graphHypothesisStage({
    relationClass: params.relationClass,
    confidence,
    usefulnessScore,
    stabilityScore,
    contradictionScore,
    supportCount: params.supports.length,
    supportDiversity,
    temporalPersistence,
  });
  const sourceNormalized = normalizeText(sourceName);
  const targetNormalized = normalizeText(targetName);
  const semanticKey = `graph_hypothesis:${params.scope}:${sourceNormalized}:${params.relationType}:${params.relationSlot ?? ""}:${targetNormalized}`;
  return {
    candidateId: stableHash([params.agentId, params.scope, semanticKey]),
    agentId: params.agentId,
    scope: params.scope,
    abstractionType: "graph_hypothesis",
    semanticKey,
    summary: graphHypothesisSummary(
      params.relationClass,
      params.relationType,
      sourceName,
      targetName,
    ),
    supportContentRefs: [...new Set(params.supports.map((entry) => entry.contentRef))],
    supportBeliefIds: [
      ...new Set(
        params.supports
          .map((entry) => entry.beliefId)
          .filter((beliefId): beliefId is string => Boolean(beliefId)),
      ),
    ],
    confidence,
    usefulnessScore,
    stabilityScore,
    contradictionScore,
    stage,
    metadataJson: {
      relationType: params.relationType,
      ...(params.relationSlot ? { relationSlot: params.relationSlot } : {}),
      relationClass: params.relationClass,
      sourceName,
      targetName,
      sourceType: params.sourceType,
      targetType: params.targetType,
      sourceNodeId: graphHypothesisNodeId(sourceName, params.sourceType),
      targetNodeId: graphHypothesisNodeId(targetName, params.targetType),
      supportCount: params.supports.length,
      supportKinds: [...new Set(params.supports.map((entry) => entry.supportKind))].sort(),
      supportDiversity,
      temporalPersistence,
      firstSeenAt: params.supports
        .map((entry) => entry.observedAt)
        .sort()
        .at(0),
      lastSeenAt: params.supports
        .map((entry) => entry.observedAt)
        .sort()
        .at(-1),
      generatedFrom: [...new Set(params.supports.map((entry) => entry.supportKind))],
      semanticSource: "upstream_structured",
      semanticSources: ["upstream_structured", "deterministic_lifecycle"],
      frameworkRound: 4,
    },
    createdAt: params.now,
    updatedAt: params.now,
  };
}

function buildGraphHypothesisCandidates(
  store: MemxStoreBundle,
  ctx: MemoryOperationContext,
  beliefMap: Map<string, MemoryBeliefRecord>,
  recentEvents: ReturnType<MemxStoreBundle["eventRepo"]["search"]>,
  _activeTasks: ConversationTask[],
): AbstractionCandidateRecord[] {
  const recentFacts = store.factRepo.query({
    agentId: ctx.agentId,
    scopes: ctx.scopes,
    limit: GRAPH_HYPOTHESIS_FACT_LIMIT,
    includeHistorical: true,
  });
  const activeStates = store.stateRepo.get({
    agentId: ctx.agentId,
    scopes: ctx.scopes,
    now: ctx.now,
  });
  const supports = [
    ...recentEvents.flatMap((event) => buildEventGraphSupports(beliefMap, event)),
    ...recentFacts
      .map((fact) => buildFactGraphSupport(beliefMap, fact))
      .filter((entry): entry is GraphHypothesisSupportEntry => entry !== null),
  ];
  const grouped = new Map<string, GraphHypothesisSupportEntry[]>();
  for (const entry of supports) {
    const key = `${entry.scope}:${normalizeText(entry.sourceName)}:${entry.relationType}:${entry.relationSlot ?? ""}:${normalizeText(entry.targetName)}`;
    const bucket = grouped.get(key) ?? [];
    bucket.push(entry);
    grouped.set(key, bucket);
  }
  const siblingGroups = new Map<string, GraphHypothesisSupportEntry[][]>();
  for (const group of grouped.values()) {
    const familyKey = `${group[0]!.scope}:${normalizeText(group[0]!.sourceName)}:${group[0]!.relationType}:${group[0]!.relationSlot ?? ""}`;
    const bucket = siblingGroups.get(familyKey) ?? [];
    bucket.push(group);
    siblingGroups.set(familyKey, bucket);
  }
  return [...grouped.values()]
    .map((group) =>
      buildGraphHypothesisCandidate({
        agentId: ctx.agentId,
        scope: group[0]!.scope,
        relationType: group[0]!.relationType,
        relationSlot: group[0]!.relationSlot,
        relationClass: group[0]!.relationClass,
        sourceName: group[0]!.sourceName,
        targetName: group[0]!.targetName,
        sourceType: group[0]!.sourceType,
        targetType: group[0]!.targetType,
        supports: group,
        siblingGroups: siblingGroups.get(
          `${group[0]!.scope}:${normalizeText(group[0]!.sourceName)}:${group[0]!.relationType}:${group[0]!.relationSlot ?? ""}`,
        ) ?? [group],
        now: ctx.now,
      }),
    )
    .filter((candidate): candidate is AbstractionCandidateRecord => candidate !== null);
}

function candidateStage(params: {
  confidence: number;
  contradiction: number;
  supportCount: number;
  supportDiversity: number;
  temporalPersistence: number;
}): AbstractionCandidateStage {
  if (
    params.confidence >= 0.72 &&
    params.contradiction <= 0.28 &&
    params.supportCount >= 3 &&
    (params.supportDiversity >= 0.4 || params.temporalPersistence >= 0.32)
  ) {
    return "probationary";
  }
  return "candidate";
}

function workflowCandidateStage(params: {
  confidence: number;
  usefulnessScore: number;
  stabilityScore: number;
  contradictionScore: number;
  groundedByRoles?: string[];
  taskPhase?: string;
  explicitInstruction?: boolean;
  groundedResolution?: boolean;
}): AbstractionCandidateStage {
  if (params.contradictionScore >= 0.52) {
    return "quarantined";
  }
  if (
    params.confidence >= 0.78 &&
    params.usefulnessScore >= 0.62 &&
    params.stabilityScore >= 0.62
  ) {
    return "probationary";
  }
  const roles = new Set(params.groundedByRoles ?? []);
  const resolvedPhase = params.taskPhase === "resolved" || params.taskPhase === "validated";
  const hasGroundedWorkflowEvidence =
    params.explicitInstruction === true ||
    params.groundedResolution === true ||
    (resolvedPhase && roles.has("user") && (roles.has("tool") || roles.has("assistant")));
  if (
    hasGroundedWorkflowEvidence &&
    params.contradictionScore <= 0.24 &&
    params.confidence >= 0.64 &&
    params.usefulnessScore >= 0.54 &&
    params.stabilityScore >= 0.54
  ) {
    return "probationary";
  }
  return "candidate";
}

function explicitWorkflowCandidateStage(params: {
  confidence: number;
  usefulnessScore: number;
  stabilityScore: number;
  contradictionScore: number;
  explicitInstruction: boolean;
}): AbstractionCandidateStage {
  if (params.contradictionScore >= 0.52) {
    return "quarantined";
  }
  // Explicit user workflow guidance is already semantic-compiler output with
  // source lineage. It should enter probationary strategy memory before it has
  // retrieval-use feedback; later belief signals can promote, demote, or decay it.
  if (
    params.explicitInstruction &&
    params.confidence >= 0.78 &&
    params.stabilityScore >= 0.56 &&
    params.contradictionScore <= 0.18
  ) {
    return "probationary";
  }
  if (
    params.explicitInstruction &&
    params.confidence >= 0.68 &&
    params.usefulnessScore >= 0.52 &&
    params.stabilityScore >= 0.52
  ) {
    return "probationary";
  }
  return workflowCandidateStage(params);
}

function explicitStrategyDomainLabel(summary: string): string {
  const concise = conciseWorkflowDomainPart(summary);
  if (concise) {
    return concise;
  }
  const tokens = tokenizeSearchTerms(summary, EXPLICIT_STRATEGY_STOPWORDS).slice(0, 4);
  if (tokens.length > 0) {
    return tokens.join(" ");
  }
  return truncateText(summary, 48);
}

function conciseWorkflowDomainPart(value: string): string | undefined {
  const trimmed = value
    .replace(/\s+/g, " ")
    .replace(/[。.!！?？]+$/u, "")
    .trim();
  if (!trimmed) {
    return undefined;
  }
  if (/[^\x00-\x7F]/u.test(trimmed)) {
    const phrase = trimmed
      .split(/[，,;；。.!！?？]/u)
      .map((part) => part.trim())
      .find((part) => part.length >= 4 && part.length <= 24);
    if (phrase) {
      return truncateText(phrase, 48);
    }
    if (trimmed.length <= 24) {
      return truncateText(trimmed, 48);
    }
  }
  const wordCount = trimmed.split(/\s+/u).filter(Boolean).length;
  if (wordCount >= 2 && wordCount <= 6 && trimmed.length <= 48) {
    return trimmed;
  }
  return undefined;
}

function workflowDomainLabelFromParts(parts: string[], fallback: string): string {
  for (const part of parts) {
    const concise = conciseWorkflowDomainPart(part);
    if (concise) {
      return concise;
    }
  }
  return explicitStrategyDomainLabel(fallback);
}

function explicitStrategyDomainKey(summary: string): string {
  const label = explicitStrategyDomainLabel(summary);
  const normalizedLabel =
    normalizeText(label)
      .replace(/[^\p{L}\p{N}]+/gu, ".")
      .replace(/^\.+|\.+$/g, "") || "explicit";
  return `explicit.${normalizedLabel}.${stableHash([normalizeText(summary)]).slice(0, 8)}`;
}

function explicitStrategySummary(summary: string, domainLabel: string): string {
  const cleaned = summary.replace(/[。.!！?？]+$/u, "").trim();
  if (!cleaned) {
    return "";
  }
  if (/^when handling\b/i.test(cleaned) || /^prefer:/i.test(cleaned)) {
    return truncateText(cleaned, 240);
  }
  return truncateText(`When handling ${domainLabel || "similar tasks"}, prefer: ${cleaned}.`, 240);
}

function factGuidanceText(fact: NormalizedFact): string | undefined {
  const guidance = objectRecord(fact.objectValueJson?.guidance);
  const guidanceText = stringValue(guidance?.guidanceText);
  if (guidanceText) {
    return guidanceText;
  }
  if (fact.predicate !== ADVICE_SIGNAL_PREDICATE) {
    return undefined;
  }
  const problemContext = stringValue(fact.objectValueJson?.problemContext);
  const assistantRecommendation = stringValue(fact.objectValueJson?.assistantRecommendation);
  const resources = Array.isArray(fact.objectValueJson?.userResources)
    ? fact.objectValueJson.userResources
        .filter((entry): entry is string => typeof entry === "string" && entry.trim().length > 0)
        .slice(0, 4)
    : [];
  const parts = [
    problemContext ? `problem: ${problemContext}` : undefined,
    resources.length > 0 ? `resources: ${resources.join(", ")}` : undefined,
    assistantRecommendation ? `recommendation: ${assistantRecommendation}` : undefined,
  ].filter(Boolean);
  return parts.length > 0 ? truncateText(parts.join(" | "), 240) : undefined;
}

function structuredWorkflowGuidanceSemanticSource(
  fact: NormalizedFact,
): MaintenanceSemanticSource | null {
  const guidance = objectRecord(fact.objectValueJson?.guidance);
  const semanticFamily = stringValue(fact.objectValueJson?.semanticFamily);
  if (
    fact.predicate === WORKFLOW_GUIDANCE_PREDICATE ||
    semanticFamily === "strategy_like" ||
    semanticFamily === "workflow"
  ) {
    return "upstream_structured";
  }
  if (
    stringValue(guidance?.guidanceText) &&
    stringValue(guidance?.reason)?.includes("semantic draft")
  ) {
    return "upstream_structured";
  }
  return null;
}

function isWorkflowGuidanceFact(fact: NormalizedFact): boolean {
  if (fact.canonicalSubject !== "user" || fact.status !== "active") {
    return false;
  }
  return Boolean(factGuidanceText(fact) && structuredWorkflowGuidanceSemanticSource(fact));
}

// Maintenance is only allowed to consume support refs that were already emitted
// by upstream structured outputs. It must not rediscover support by re-querying
// recent task text with lexical similarity.
function explicitWorkflowSupportChunks(
  store: MemxStoreBundle,
  fact: NormalizedFact,
): ConversationChunk[] {
  const guidance = objectRecord(fact.objectValueJson?.guidance);
  const supportRefs = stringSet(
    objectRecord(guidance)?.supportContentRefs ?? fact.objectValueJson?.supportContentRefs,
  );
  if (supportRefs.size === 0) {
    return [];
  }
  const chunks = [...supportRefs]
    .filter((ref) => ref.startsWith("chunk:"))
    .map((ref) => store.chunkRepo.get(ref.slice("chunk:".length)))
    .filter((chunk): chunk is ConversationChunk => Boolean(chunk));
  return uniqueChunks(chunks);
}

function stringSet(value: unknown): Set<string> {
  if (!Array.isArray(value)) {
    return new Set();
  }
  return new Set(
    value.filter((entry): entry is string => typeof entry === "string" && entry.trim().length > 0),
  );
}

function uniqueChunks(chunks: ConversationChunk[]): ConversationChunk[] {
  const seen = new Set<string>();
  const ordered: ConversationChunk[] = [];
  for (const chunk of chunks) {
    if (seen.has(chunk.chunkId)) {
      continue;
    }
    seen.add(chunk.chunkId);
    ordered.push(chunk);
  }
  return ordered;
}

function maintenanceSemanticSourceRank(candidate: AbstractionCandidateRecord): number {
  const semanticSource = stringValue(candidate.metadataJson.semanticSource);
  switch (semanticSource) {
    case "upstream_structured":
      return 4;
    case "embedding_clustered":
      return 3;
    case "deterministic_lifecycle":
      return 2;
    case "llm_upgrade":
      return 1;
    case "lexical_fallback":
      return 0;
    default:
      return 1;
  }
}

function resolutionGroundingSignal(params: {
  task: ConversationTask;
  resolution: string;
  evidenceChunks: ConversationChunk[];
}): {
  grounded: boolean;
  score: number;
  complexity: number;
  semanticSource: MaintenanceSemanticSource;
} {
  const metadata = objectRecord(params.task.metadataJson) ?? {};
  const explicitEvidenceCount = uniqueChunks(params.evidenceChunks).length;
  const hasUserEvidence = params.evidenceChunks.some((chunk) => chunk.role === "user");
  const hasToolEvidence = params.evidenceChunks.some((chunk) => chunk.role === "tool");
  const hasAssistantEvidence = params.evidenceChunks.some((chunk) => chunk.role === "assistant");
  const promotionScore =
    typeof metadata.candidateResolutionPromotionScore === "number" &&
    Number.isFinite(metadata.candidateResolutionPromotionScore)
      ? metadata.candidateResolutionPromotionScore
      : 0;
  const hasOutcomeKey = Boolean(stringValue(metadata.lastEmittedOutcomeKey));
  const complexity = contentStructuralComplexity(params.resolution);
  const score = clamp01(
    (explicitEvidenceCount > 0 ? 0.36 : 0) +
      (hasToolEvidence ? 0.28 : 0) +
      (hasUserEvidence ? 0.12 : 0) +
      (hasAssistantEvidence ? 0.08 : 0) +
      (hasOutcomeKey ? 0.1 : 0) +
      promotionScore * 0.24,
  );
  return {
    grounded:
      (explicitEvidenceCount > 0 || hasToolEvidence || promotionScore >= 0.74 || hasOutcomeKey) &&
      score >= 0.28 &&
      complexity <= 0.42 &&
      !params.resolution.includes("```"),
    score,
    complexity,
    semanticSource: "upstream_structured",
  };
}

function buildExplicitWorkflowPatternCandidates(
  store: MemxStoreBundle,
  ctx: MemoryOperationContext,
  beliefMap: Map<string, MemoryBeliefRecord>,
): AbstractionCandidateRecord[] {
  const recentFacts = store.factRepo.query({
    agentId: ctx.agentId,
    scopes: ctx.scopes,
    limit: 48,
    includeHistorical: false,
  });
  const workflowGuidanceFacts = recentFacts.filter(isWorkflowGuidanceFact);
  const explicitFacts = [
    ...new Map(workflowGuidanceFacts.map((fact) => [fact.factId, fact])).values(),
  ];

  const candidates: AbstractionCandidateRecord[] = [];
  for (const fact of explicitFacts) {
    const objectSummary = stringValue(fact.canonicalObject)?.trim();
    const guidanceText = factGuidanceText(fact);
    const semanticSource = structuredWorkflowGuidanceSemanticSource(fact);
    if (!semanticSource) {
      continue;
    }
    const supportChunks = explicitWorkflowSupportChunks(store, fact);
    const supportSummary =
      supportChunks.length > 0
        ? truncateText(
            supportChunks
              .map((chunk) => chunk.content.trim())
              .filter(Boolean)
              .join("；"),
            240,
          )
        : undefined;
    const summarySource = supportSummary || objectSummary || guidanceText;
    if (!summarySource) {
      continue;
    }
    const belief = beliefMap.get(`fact:${fact.factId}`);
    const explicitInstruction =
      fact.canonicalSubject === "user" &&
      fact.status === "active" &&
      fact.predicate !== ADVICE_SIGNAL_PREDICATE &&
      Boolean(guidanceText);
    const groundedByRoles = supportChunks.length > 0 ? ["user"] : [];
    const contradictionScore = clamp01(belief?.contradictionScore ?? 0.08);
    const usefulnessScore = clamp01(
      (belief?.usefulnessScore ?? 0.58) * 0.64 +
        fact.confidence * 0.18 +
        (belief?.stage === "active" ? 0.18 : 0.08) +
        (explicitInstruction ? 0.08 : 0) +
        (supportChunks.length > 0 ? 0.08 : 0),
    );
    const stabilityScore = clamp01(
      (belief?.stabilityScore ?? 0.62) * 0.58 +
        (belief?.stage === "active" ? 0.18 : 0.08) +
        (1 - contradictionScore) * 0.24 +
        (explicitInstruction ? 0.06 : 0) +
        (supportChunks.length > 0 ? 0.06 : 0),
    );
    const confidence = clamp01(
      average([fact.confidence, belief?.posteriorConfidence ?? fact.confidence]) * 0.72 +
        usefulnessScore * 0.14 +
        (1 - contradictionScore) * 0.14 +
        (explicitInstruction ? 0.1 : 0) +
        (supportChunks.length > 0 ? 0.08 : 0),
    );
    const stage = explicitWorkflowCandidateStage({
      confidence,
      usefulnessScore,
      stabilityScore,
      contradictionScore,
      explicitInstruction,
    });
    const domainLabel = explicitStrategyDomainLabel(summarySource);
    const domainKey = explicitStrategyDomainKey(summarySource);
    const normalizedSummary = explicitStrategySummary(summarySource, domainLabel);
    if (!normalizedSummary) {
      continue;
    }
    candidates.push({
      candidateId: stableHash([ctx.agentId, fact.scope, "explicit_workflow_pattern", fact.factId]),
      agentId: ctx.agentId,
      scope: fact.scope,
      abstractionType: "workflow_pattern",
      semanticKey: `workflow_pattern:${fact.scope}:${domainKey}`,
      summary: normalizedSummary,
      supportContentRefs: [
        `fact:${fact.factId}`,
        ...supportChunks.map((chunk) => `chunk:${chunk.chunkId}`),
      ],
      supportBeliefIds: belief?.beliefId ? [belief.beliefId] : [],
      confidence,
      usefulnessScore,
      stabilityScore,
      contradictionScore,
      stage,
      metadataJson: {
        domainKey,
        domainLabel,
        sourceFactId: fact.factId,
        sourcePredicate: fact.predicate,
        generatedFrom: "explicit_workflow_guidance",
        explicitInstruction,
        groundedByRoles,
        supportCount: 1 + supportChunks.length,
        semanticSource,
        semanticSources: [semanticSource],
        frameworkRound: 3,
      },
      createdAt: ctx.now,
      updatedAt: ctx.now,
    });
  }
  return candidates;
}

function buildGroundedWorkflowPatternCandidates(
  store: MemxStoreBundle,
  ctx: MemoryOperationContext,
  beliefMap: Map<string, MemoryBeliefRecord>,
): AbstractionCandidateRecord[] {
  const recentTasks = store.taskRepo.listRecent({
    agentId: ctx.agentId,
    scopes: ctx.scopes,
    limit: 24,
  });

  const candidates: AbstractionCandidateRecord[] = [];
  for (const task of recentTasks) {
    const metadata = objectRecord(task.metadataJson) ?? {};
    const resolution = stringValue(metadata.candidateResolution);
    const phase =
      stringValue(metadata.candidateResolutionPhase) ?? stringValue(metadata.taskPhase) ?? "";
    if (
      !resolution ||
      (task.status !== "completed" && phase !== "validated" && phase !== "resolved")
    ) {
      continue;
    }

    const belief = beliefMap.get(`task:${task.taskId}`);
    if (belief && (belief.stage === "quarantined" || belief.stage === "superseded")) {
      continue;
    }
    const resolutionTokens = tokenizeSearchTerms(resolution, EXPLICIT_STRATEGY_STOPWORDS);
    if (resolutionTokens.length < 3) {
      continue;
    }

    const chunks = store.chunkRepo.listByTask(task.taskId);
    const evidenceChunkIds = stringSet(metadata.candidateResolutionEvidenceChunkIds);
    const explicitEvidenceChunks = chunks.filter((chunk) => evidenceChunkIds.has(chunk.chunkId));
    const toolEvidenceChunks = chunks.filter((chunk) => chunk.role === "tool");
    const supportChunks = uniqueChunks([...explicitEvidenceChunks, ...toolEvidenceChunks]);
    const supportChunkIds = [
      ...supportChunks
        .filter(
          (chunk) => chunk.role === "user" || chunk.role === "assistant" || chunk.role === "tool",
        )
        .map((chunk) => `chunk:${chunk.chunkId}`),
    ];
    const hasUserEvidence = supportChunks.some((chunk) => chunk.role === "user");
    const hasToolEvidence = supportChunks.some((chunk) => chunk.role === "tool");
    const hasAssistantEvidence = supportChunks.some((chunk) => chunk.role === "assistant");
    const resolutionGrounding = resolutionGroundingSignal({
      task,
      resolution,
      evidenceChunks: supportChunks,
    });
    const hasGroundedResolution = resolutionGrounding.grounded;
    const hasGroundedEvidence =
      hasToolEvidence ||
      (hasAssistantEvidence && hasUserEvidence) ||
      hasGroundedResolution ||
      (belief?.outcomeSupportScore ?? 0) >= 0.72;
    if (!hasGroundedEvidence) {
      continue;
    }

    const contradictionScore = clamp01(belief?.contradictionScore ?? 0.1);
    const usefulnessScore = clamp01(
      (belief?.usefulnessScore ?? 0.58) * 0.44 +
        (hasToolEvidence ? 0.2 : 0.1) +
        (hasUserEvidence ? 0.12 : 0.04) +
        (phase === "resolved" ? 0.12 : 0.08) +
        (hasGroundedResolution ? 0.08 : 0) +
        (1 - contradictionScore) * 0.12,
    );
    const stabilityScore = clamp01(
      (belief?.stabilityScore ?? 0.6) * 0.42 +
        (hasToolEvidence ? 0.18 : 0.06) +
        (hasAssistantEvidence ? 0.08 : 0) +
        (phase === "resolved" ? 0.12 : 0.08) +
        (hasGroundedResolution ? 0.08 : 0) +
        (1 - contradictionScore) * 0.2,
    );
    const confidence = clamp01(
      average([belief?.posteriorConfidence ?? 0.68, task.status === "completed" ? 0.76 : 0.7]) *
        0.56 +
        usefulnessScore * 0.18 +
        stabilityScore * 0.18 +
        (hasToolEvidence ? 0.08 : hasAssistantEvidence ? 0.04 : 0) +
        (hasGroundedResolution ? 0.04 : 0),
    );
    const groundedByRoles = [
      ...(hasUserEvidence ? ["user"] : []),
      ...(hasAssistantEvidence ? ["assistant"] : []),
      ...(hasToolEvidence ? ["tool"] : []),
    ];
    const stage = workflowCandidateStage({
      confidence,
      usefulnessScore,
      stabilityScore,
      contradictionScore,
      groundedByRoles,
      taskPhase: phase,
      groundedResolution: hasGroundedResolution,
    });
    const domainParts = [
      stringValue(metadata.currentTask),
      task.title.trim(),
      stringValue(metadata.project),
    ].filter((value): value is string => Boolean(value));
    const domainSource = domainParts.join(" ").trim();
    const domainLabel = workflowDomainLabelFromParts(domainParts, domainSource || resolution);
    const domainKey = explicitStrategyDomainKey(domainSource || resolution);
    const summary = explicitStrategySummary(resolution, domainLabel);
    if (!summary) {
      continue;
    }

    candidates.push({
      candidateId: stableHash([ctx.agentId, task.scope, "grounded_workflow_pattern", task.taskId]),
      agentId: ctx.agentId,
      scope: task.scope,
      abstractionType: "workflow_pattern",
      semanticKey: `workflow_pattern:${task.scope}:${domainKey}`,
      summary,
      supportContentRefs: [`task:${task.taskId}`, ...supportChunkIds.slice(0, 6)],
      supportBeliefIds: belief?.beliefId ? [belief.beliefId] : [],
      confidence,
      usefulnessScore,
      stabilityScore,
      contradictionScore,
      stage,
      metadataJson: {
        domainKey,
        domainLabel,
        sourceTaskId: task.taskId,
        taskPhase: phase || undefined,
        groundedByRoles,
        groundedResolution: hasGroundedResolution || undefined,
        groundedResolutionScore: Number(resolutionGrounding.score.toFixed(3)),
        groundedResolutionComplexity: Number(resolutionGrounding.complexity.toFixed(3)),
        generatedFrom: "grounded_task_resolution",
        semanticSource: resolutionGrounding.semanticSource,
        semanticSources: [resolutionGrounding.semanticSource],
        frameworkRound: 6,
      },
      createdAt: ctx.now,
      updatedAt: ctx.now,
    });
  }

  return candidates;
}

function conceptSeedNames(facts: NormalizedFact[]): string[] {
  const names = new Map<string, string>();
  for (const fact of facts) {
    for (const name of [fact.canonicalSubject, fact.canonicalObject]) {
      if (!name) {
        continue;
      }
      const normalized = normalizeText(name);
      if (!normalized || CONCEPT_GENERIC_NAMES.has(normalized)) {
        continue;
      }
      if (!names.has(normalized)) {
        names.set(normalized, name);
      }
    }
  }
  return [...names.values()];
}

function factMatchesConceptSeed(fact: NormalizedFact, seedName: string): boolean {
  const normalizedSeed = normalizeText(seedName);
  return (
    normalizeText(fact.canonicalSubject) === normalizedSeed ||
    normalizeText(fact.canonicalObject ?? "") === normalizedSeed
  );
}

function buildConceptSupportEntries(params: {
  seedEntityId: string;
  facts: NormalizedFact[];
  graph: GraphEvidence;
  beliefMap: Map<string, MemoryBeliefRecord>;
}): {
  factSupports: ConceptSupportEntry[];
  edgeSupports: ConceptSupportEntry[];
  supportBeliefIds: string[];
  supportContentRefs: string[];
} {
  const factSupports = params.facts.map((fact) => {
    const belief = params.beliefMap.get(`fact:${fact.factId}`);
    return {
      supportKind: "fact",
      contentRef: `fact:${fact.factId}`,
      relationLabel: fact.predicate,
      observedAt: fact.updatedAt,
      confidence: clamp01(average([fact.confidence, belief?.posteriorConfidence ?? 0.66])),
      usefulnessScore: belief?.usefulnessScore ?? 0.46,
      stabilityScore: belief?.stabilityScore ?? 0.58,
      contradictionScore: belief?.contradictionScore ?? 0.08,
      beliefId: belief?.beliefId,
    } satisfies ConceptSupportEntry;
  });
  const edgeSupports = params.graph.edges.map((edge) => {
    const belief = params.beliefMap.get(`graph_edge:${edge.edgeId}`);
    const isSeedIncident =
      edge.srcEntityId === params.seedEntityId || edge.dstEntityId === params.seedEntityId;
    const contentRef =
      edge.sourceKind === "stored"
        ? `graph_edge:${edge.edgeId}`
        : edge.evidenceRef || `graph_edge:${edge.edgeId}`;
    return {
      supportKind: "graph_edge",
      contentRef,
      relationLabel: edge.relType,
      observedAt: edge.updatedAt ?? "",
      confidence: clamp01(
        average([edge.confidence, belief?.posteriorConfidence ?? (isSeedIncident ? 0.7 : 0.62)]),
      ),
      usefulnessScore: belief?.usefulnessScore ?? (isSeedIncident ? 0.54 : 0.44),
      stabilityScore: belief?.stabilityScore ?? (isSeedIncident ? 0.62 : 0.52),
      contradictionScore:
        belief?.contradictionScore ?? (edge.relType === "contradicts" ? 0.42 : 0.08),
      beliefId: belief?.beliefId,
    } satisfies ConceptSupportEntry;
  });
  const supportBeliefIds = [
    ...new Set(
      [...factSupports, ...edgeSupports]
        .map((entry) => entry.beliefId)
        .filter((beliefId): beliefId is string => Boolean(beliefId)),
    ),
  ];
  const supportContentRefs = [
    ...new Set([...factSupports, ...edgeSupports].map((entry) => entry.contentRef)),
  ];
  return { factSupports, edgeSupports, supportBeliefIds, supportContentRefs };
}

function resolveConceptGraphEntity(params: {
  store: MemxStoreBundle;
  ctx: MemoryOperationContext;
  scope: string;
  name: string;
  observedAt: string;
  sourceRef: string;
  supportText: string;
}): {
  entityId: string;
  canonicalName: string;
  entityType: GraphEvidenceNode["type"];
  normalizedName: string;
  confidence: number;
} {
  const result = resolveEntityMention(
    params.store,
    params.ctx,
    buildEntityMention({
      ctx: params.ctx,
      scope: params.scope,
      rawText: params.name,
      semanticRole: "support",
      sourceRef: params.sourceRef,
      supportText: params.supportText,
      observedAt: params.observedAt,
      metadataJson: {
        generatedFrom: "abstraction-concept-graph-resolution",
      },
    }),
    { createIfMissing: false, persist: false },
  );
  if (result.method !== "uncertain") {
    return result.entity;
  }
  return {
    entityId: graphHypothesisNodeId(params.name, "unknown"),
    canonicalName: params.name.trim(),
    entityType: "unknown",
    normalizedName: normalizeText(params.name),
    confidence: 0.58,
  };
}

function ensureConceptGraphNode(params: {
  store: MemxStoreBundle;
  ctx: MemoryOperationContext;
  scope: string;
  nodes: Map<string, GraphEvidenceNode>;
  name: string;
  observedAt: string;
  sourceRef: string;
  supportText: string;
}): GraphEvidenceNode {
  const entity = resolveConceptGraphEntity(params);
  const existing = params.nodes.get(entity.entityId);
  if (existing) {
    return existing;
  }
  const node: GraphEvidenceNode = {
    nodeId: entity.entityId,
    nodeKind: "entity",
    entityId: entity.entityId,
    name: entity.canonicalName,
    type: entity.entityType,
    confidence: entity.confidence,
    observedAt: params.observedAt,
  };
  params.nodes.set(node.nodeId, node);
  return node;
}

function conceptGraphEdgeKey(
  edge: Pick<GraphEvidenceEdge, "srcNodeId" | "relType" | "dstNodeId">,
): string {
  return `${edge.srcNodeId}:${edge.relType}:${edge.dstNodeId}`;
}

function upsertConceptGraphEdge(
  edges: Map<string, GraphEvidenceEdge>,
  edge: GraphEvidenceEdge,
): void {
  const key = conceptGraphEdgeKey(edge);
  const existing = edges.get(key);
  if (!existing) {
    edges.set(key, edge);
    return;
  }
  const preferred =
    existing.sourceKind === "stored" || edge.sourceKind !== "stored" ? existing : edge;
  const fallback = preferred === existing ? edge : existing;
  edges.set(key, {
    ...preferred,
    confidence: Math.max(existing.confidence, edge.confidence),
    updatedAt:
      Date.parse(existing.updatedAt ?? "") >= Date.parse(edge.updatedAt ?? "")
        ? existing.updatedAt
        : edge.updatedAt,
    evidenceRef: preferred.evidenceRef ?? fallback.evidenceRef,
  });
}

function mergeConceptStoredGraph(params: {
  nodes: Map<string, GraphEvidenceNode>;
  edges: Map<string, GraphEvidenceEdge>;
  graph: GraphEvidence;
}): void {
  for (const node of params.graph.nodes) {
    if (!params.nodes.has(node.nodeId)) {
      params.nodes.set(node.nodeId, node);
    }
  }
  for (const edge of params.graph.edges) {
    upsertConceptGraphEdge(params.edges, edge);
  }
}

function buildConceptGraphEvidence(params: {
  store: MemxStoreBundle;
  ctx: MemoryOperationContext;
  seedName: string;
  scope: string;
  facts: NormalizedFact[];
}): { seedEntityId: string; entityName: string; graph: GraphEvidence } {
  const seedEntity = resolveConceptGraphEntity({
    store: params.store,
    ctx: params.ctx,
    scope: params.scope,
    name: params.seedName,
    observedAt: params.ctx.now,
    sourceRef: `maintenance:concept:${stableHash([params.scope, params.seedName])}`,
    supportText: params.seedName,
  });
  const nodes = new Map<string, GraphEvidenceNode>();
  const edges = new Map<string, GraphEvidenceEdge>();
  ensureConceptGraphNode({
    store: params.store,
    ctx: params.ctx,
    scope: params.scope,
    nodes,
    name: seedEntity.canonicalName,
    observedAt: params.ctx.now,
    sourceRef: `maintenance:concept:${stableHash([params.scope, params.seedName, "seed"])}`,
    supportText: params.seedName,
  });

  for (const fact of params.facts) {
    const relType = relationalFactGraphRelation(fact);
    if (!fact.canonicalObject || !relType) {
      continue;
    }
    const src = ensureConceptGraphNode({
      store: params.store,
      ctx: params.ctx,
      scope: params.scope,
      nodes,
      name: fact.canonicalSubject,
      observedAt: fact.updatedAt,
      sourceRef: `fact:${fact.factId}`,
      supportText: fact.provenanceText,
    });
    const dst = ensureConceptGraphNode({
      store: params.store,
      ctx: params.ctx,
      scope: params.scope,
      nodes,
      name: fact.canonicalObject,
      observedAt: fact.updatedAt,
      sourceRef: `fact:${fact.factId}`,
      supportText: fact.provenanceText,
    });
    const edgeId = stableHash(["concept-fact-edge", fact.factId, src.nodeId, relType, dst.nodeId]);
    upsertConceptGraphEdge(edges, {
      edgeId,
      srcNodeId: src.nodeId,
      srcEntityId: src.nodeId,
      relType,
      dstNodeId: dst.nodeId,
      dstEntityId: dst.nodeId,
      confidence: clamp01(fact.confidence),
      evidenceRef: `fact:${fact.factId}`,
      updatedAt: fact.updatedAt,
      sourceKind: "synthesized",
    });
  }

  if (!seedEntity.entityId.startsWith("graph_hypothesis_node:")) {
    const storedGraph = params.store.graphRepo.expandNeighborhood({
      agentId: params.ctx.agentId,
      scopes: [params.scope],
      seedEntityIds: [seedEntity.entityId],
      maxHops: Math.min(params.ctx.config.graphMaxHops, 2),
      maxEdges: Math.min(params.ctx.config.maxGraphEdges, CONCEPT_GRAPH_EDGE_BUDGET),
      maxNodes: Math.min(params.ctx.config.maxGraphNodes, CONCEPT_GRAPH_NODE_BUDGET),
      now: params.ctx.now,
    });
    mergeConceptStoredGraph({ nodes, edges, graph: storedGraph });
  }

  const edgeList = [...edges.values()].sort(
    (left, right) =>
      right.confidence - left.confidence ||
      (right.updatedAt ? Date.parse(right.updatedAt) : 0) -
        (left.updatedAt ? Date.parse(left.updatedAt) : 0),
  );
  const pathCandidates = buildGraphPathCandidates({
    seedNodeIds: [seedEntity.entityId],
    nodes,
    edges: edgeList,
    now: params.ctx.now,
    maxPaths: Math.max(6, Math.min(CONCEPT_GRAPH_EDGE_BUDGET + 2, 12)),
    maxHops: 2,
  });

  return {
    seedEntityId: seedEntity.entityId,
    entityName: seedEntity.canonicalName,
    graph: {
      nodes: [...nodes.values()],
      edges: edgeList,
      pathCandidates,
      paths: pathCandidates.map((path) => path.summary),
    },
  };
}

function buildConceptCandidate(params: {
  agentId: string;
  scope: string;
  entityId: string;
  entityName: string;
  facts: NormalizedFact[];
  graph: GraphEvidence;
  beliefMap: Map<string, MemoryBeliefRecord>;
  now: string;
}): AbstractionCandidateRecord | null {
  if (params.facts.length < 2) {
    return null;
  }
  const relationalFactCount = params.facts.filter((fact) =>
    Boolean(relationalFactGraphRelation(fact)),
  ).length;
  if (params.graph.edges.length === 0 && relationalFactCount < 2) {
    return null;
  }
  const { factSupports, edgeSupports, supportBeliefIds, supportContentRefs } =
    buildConceptSupportEntries({
      seedEntityId: params.entityId,
      facts: params.facts,
      graph: params.graph,
      beliefMap: params.beliefMap,
    });
  const supports = [...factSupports, ...edgeSupports];
  const relationLabels = topLabels(supports);
  const relationFamilies = new Set(supports.map((entry) => entry.relationLabel));
  const supportDiversity = conceptSupportDiversity(supports);
  const temporalPersistence = conceptTemporalPersistence(supports);
  const contradictionScore = clamp01(average(supports.map((entry) => entry.contradictionScore)));
  const pathScoreAverage = average(params.graph.pathCandidates.map((path) => path.score));
  const relationDensity = clamp01(
    (relationFamilies.size + params.graph.edges.length + relationalFactCount) / 10,
  );
  const graphCoverage = clamp01(
    (params.graph.edges.length + params.graph.nodes.length + params.graph.pathCandidates.length) /
      18,
  );
  const structuralStrength = clamp01(
    relationDensity * 0.34 +
      pathScoreAverage * 0.34 +
      graphCoverage * 0.2 +
      supportDiversity * 0.12,
  );
  if (relationFamilies.size < 2 || structuralStrength < 0.38) {
    return null;
  }
  const usefulnessScore = clamp01(
    average(supports.map((entry) => entry.usefulnessScore)) * 0.58 +
      pathScoreAverage * 0.18 +
      relationDensity * 0.14 +
      supportDiversity * 0.1,
  );
  const stabilityScore = clamp01(
    average(supports.map((entry) => entry.stabilityScore)) * 0.42 +
      temporalPersistence * 0.18 +
      structuralStrength * 0.22 +
      supportDiversity * 0.08 +
      (1 - contradictionScore) * 0.1,
  );
  const confidence = clamp01(
    average(supports.map((entry) => entry.confidence)) * 0.34 +
      structuralStrength * 0.28 +
      supportDiversity * 0.16 +
      usefulnessScore * 0.12 +
      temporalPersistence * 0.06 +
      (1 - contradictionScore) * 0.04,
  );
  if (confidence < 0.56) {
    return null;
  }
  const stage = conceptCandidateStage({
    confidence,
    usefulnessScore,
    stabilityScore,
    contradictionScore,
    structuralStrength,
  });
  const semanticKey = `concept_candidate:${params.scope}:${normalizeText(params.entityName)}`;
  return {
    candidateId: stableHash([params.agentId, params.scope, semanticKey]),
    agentId: params.agentId,
    scope: params.scope,
    abstractionType: "concept_candidate",
    semanticKey,
    summary: conceptSummary(params.entityName, relationLabels),
    supportContentRefs,
    supportBeliefIds,
    confidence,
    usefulnessScore,
    stabilityScore,
    contradictionScore,
    stage,
    metadataJson: {
      entityId: params.entityId,
      entityName: params.entityName,
      factCount: factSupports.length,
      graphEdgeCount: edgeSupports.length,
      relationFamilyCount: relationFamilies.size,
      relationFamilies: [...relationFamilies].sort(),
      supportDiversity,
      temporalPersistence,
      structuralStrength,
      pathCount: params.graph.pathCandidates.length,
      pathScoreAverage,
      topPathSummaries: params.graph.pathCandidates.slice(0, 3).map((path) => path.summary),
      generatedFrom: "fact_graph_clusters",
      semanticSource: "deterministic_lifecycle",
      semanticSources: ["deterministic_lifecycle"],
      frameworkRound: 5,
    },
    createdAt: params.now,
    updatedAt: params.now,
  };
}

function buildConceptCandidates(
  store: MemxStoreBundle,
  ctx: MemoryOperationContext,
  beliefMap: Map<string, MemoryBeliefRecord>,
): AbstractionCandidateRecord[] {
  const recentFacts = store.factRepo.query({
    agentId: ctx.agentId,
    scopes: ctx.scopes,
    limit: CONCEPT_FACT_LIMIT,
    includeHistorical: false,
  });
  const candidates: AbstractionCandidateRecord[] = [];
  for (const seedName of conceptSeedNames(recentFacts)) {
    const scopeFacts = recentFacts.filter(
      (fact) => ctx.scopes.includes(fact.scope) && factMatchesConceptSeed(fact, seedName),
    );
    if (scopeFacts.length < 2) {
      continue;
    }
    const scopeFactGroups = new Map<string, NormalizedFact[]>();
    for (const fact of scopeFacts) {
      const bucket = scopeFactGroups.get(fact.scope) ?? [];
      bucket.push(fact);
      scopeFactGroups.set(fact.scope, bucket);
    }
    for (const [scope, facts] of scopeFactGroups.entries()) {
      const { seedEntityId, entityName, graph } = buildConceptGraphEvidence({
        store,
        ctx,
        seedName,
        scope,
        facts,
      });
      const candidate = buildConceptCandidate({
        agentId: ctx.agentId,
        scope,
        entityId: seedEntityId,
        entityName,
        facts,
        graph,
        beliefMap,
        now: ctx.now,
      });
      if (candidate) {
        candidates.push(candidate);
      }
    }
  }
  return candidates;
}

function buildDerivedStateCandidate(params: {
  agentId: string;
  scope: string;
  stateKey: string;
  valueJson: Record<string, unknown>;
  supports: StateSupportEntry[];
  siblingGroups: StateSupportEntry[][];
  now: string;
}): AbstractionCandidateRecord | null {
  const valueText = describeStateValue(params.stateKey, params.valueJson).trim();
  if (!valueText) {
    return null;
  }
  const supportDiversity = supportDiversityScore(params.supports);
  const temporalPersistence = temporalPersistenceScore(params.supports);
  const contradiction = contradictionPressure(params.supports, params.siblingGroups);
  const usefulnessScore = clamp01(average(params.supports.map((entry) => entry.usefulnessScore)));
  const stabilityScore = clamp01(
    average(params.supports.map((entry) => entry.stabilityScore)) * 0.35 +
      temporalPersistence * 0.4 +
      supportDiversity * 0.15 +
      (1 - contradiction) * 0.1,
  );
  const confidence = clamp01(
    average(params.supports.map((entry) => entry.confidence)) * 0.42 +
      supportDiversity * 0.2 +
      temporalPersistence * 0.16 +
      usefulnessScore * 0.14 +
      (1 - contradiction) * 0.08,
  );
  if (params.supports.length < 2 || confidence < 0.52) {
    return null;
  }
  const stage = candidateStage({
    confidence,
    contradiction,
    supportCount: params.supports.length,
    supportDiversity,
    temporalPersistence,
  });
  const semanticKey = `derived_state:${params.stateKey}:${semanticValueKey(
    params.stateKey,
    params.valueJson,
  )}`;
  const contentRefs = [...new Set(params.supports.map((entry) => entry.contentRef))];
  const beliefIds = [
    ...new Set(
      params.supports
        .map((entry) => entry.beliefId)
        .filter((beliefId): beliefId is string => Boolean(beliefId)),
    ),
  ];
  return {
    candidateId: stableHash([params.agentId, params.scope, semanticKey]),
    agentId: params.agentId,
    scope: params.scope,
    abstractionType: "derived_state",
    semanticKey,
    summary: derivedStateSummary(params.stateKey, valueText),
    supportContentRefs: contentRefs,
    supportBeliefIds: beliefIds,
    confidence,
    usefulnessScore,
    stabilityScore,
    contradictionScore: contradiction,
    stage,
    metadataJson: {
      stateKey: params.stateKey,
      valueJson: params.valueJson,
      valueText,
      supportCount: params.supports.length,
      eventSupportCount: params.supports.filter((entry) => entry.supportKind === "event").length,
      taskSupportCount: params.supports.filter((entry) => entry.supportKind === "task").length,
      supportDiversity,
      temporalPersistence,
      sessionCount: new Set(params.supports.map((entry) => entry.sessionKey).filter(Boolean)).size,
      firstSeenAt: params.supports
        .map((entry) => entry.observedAt)
        .sort()
        .at(0),
      lastSeenAt: params.supports
        .map((entry) => entry.observedAt)
        .sort()
        .at(-1),
      supportKinds: [...new Set(params.supports.map((entry) => entry.supportKind))],
      semanticSource: "upstream_structured",
      semanticSources: ["upstream_structured", "deterministic_lifecycle"],
      frameworkRound: 2,
    },
    createdAt: params.now,
    updatedAt: params.now,
  };
}

async function buildWorkflowPatternCandidates(
  store: MemxStoreBundle,
  ctx: MemoryOperationContext,
  beliefMap: Map<string, MemoryBeliefRecord>,
): Promise<AbstractionCandidateRecord[]> {
  const repeatedCandidates = (await deriveWorkflowPatternSummaries(store, ctx)).map((pattern) => {
    const stage = workflowCandidateStage({
      confidence: pattern.confidence,
      usefulnessScore: pattern.usefulnessScore,
      stabilityScore: pattern.stabilityScore,
      contradictionScore: pattern.contradictionScore,
      groundedByRoles: Array.isArray(pattern.metadataJson.groundedByRoles)
        ? pattern.metadataJson.groundedByRoles.filter(
            (role): role is string => typeof role === "string" && role.trim().length > 0,
          )
        : undefined,
      taskPhase: stringValue(pattern.metadataJson.taskPhase),
      explicitInstruction: Boolean(pattern.metadataJson.explicitInstruction),
      groundedResolution: Boolean(pattern.metadataJson.groundedResolution),
    });
    const semanticKey = `workflow_pattern:${pattern.scope}:${pattern.domainKey}`;
    return {
      candidateId: stableHash([ctx.agentId, pattern.scope, semanticKey]),
      agentId: ctx.agentId,
      scope: pattern.scope,
      abstractionType: "workflow_pattern",
      semanticKey,
      summary: pattern.summary,
      supportContentRefs: pattern.supportTaskIds.map((taskId) => `task:${taskId}`),
      supportBeliefIds: pattern.supportBeliefIds,
      confidence: pattern.confidence,
      usefulnessScore: pattern.usefulnessScore,
      stabilityScore: pattern.stabilityScore,
      contradictionScore: pattern.contradictionScore,
      stage,
      metadataJson: {
        ...pattern.metadataJson,
        domainKey: pattern.domainKey,
        supportTaskIds: pattern.supportTaskIds,
        generatedFrom: "workflow_pattern_candidates",
        frameworkRound: 3,
      },
      createdAt: ctx.now,
      updatedAt: ctx.now,
    } satisfies AbstractionCandidateRecord;
  });
  const explicitCandidates = buildExplicitWorkflowPatternCandidates(store, ctx, beliefMap);
  const groundedCandidates = buildGroundedWorkflowPatternCandidates(store, ctx, beliefMap);
  const merged = new Map<string, AbstractionCandidateRecord>();
  for (const candidate of [...repeatedCandidates, ...explicitCandidates, ...groundedCandidates]) {
    const existing = merged.get(candidate.semanticKey);
    const candidateRank = maintenanceSemanticSourceRank(candidate);
    const existingRank = existing ? maintenanceSemanticSourceRank(existing) : -1;
    if (
      !existing ||
      candidateRank > existingRank ||
      (candidateRank === existingRank &&
        (candidate.confidence > existing.confidence ||
          candidate.usefulnessScore > existing.usefulnessScore))
    ) {
      merged.set(candidate.semanticKey, candidate);
    }
  }
  return [...merged.values()];
}

async function refineCandidatesWithLlm(
  store: MemxStoreBundle,
  candidates: AbstractionCandidateRecord[],
  now: string,
  ctx: MemoryOperationContext,
): Promise<{
  candidates: AbstractionCandidateRecord[];
  considered: number;
  refined: number;
}> {
  if (!store.reasoner.isEnabled()) {
    return { candidates, considered: 0, refined: 0 };
  }

  const refinedCandidates = [...candidates];
  const resolvedModel = store.reasoner.getResolvedJudgeModel();
  let considered = 0;
  let refined = 0;

  for (const [index, candidate] of refinedCandidates.entries()) {
    if (!eligibleForLlmRefinement(candidate) || considered >= ABSTRACTION_LLM_REFINEMENT_LIMIT) {
      continue;
    }
    considered += 1;
    const result = await store.reasoner.judgeAbstractionCandidate(candidate, {
      stage: "maintenance_async",
      audit: ctx.llmBudgetAudit,
    });
    if (!result) {
      continue;
    }
    const refinedCandidate = applyAbstractionRefinement({
      candidate,
      result,
      now,
      resolvedModel,
    });
    if (!refinedCandidate) {
      continue;
    }

    refined += 1;
    refinedCandidates[index] = refinedCandidate;
  }

  return {
    candidates: refinedCandidates,
    considered,
    refined,
  };
}

export async function runAbstractionJobs(
  store: MemxStoreBundle,
  ctx: MemoryOperationContext,
  options: {
    refineWithLlm?: boolean;
    batch?: MaintenanceBatchMetadata;
    deltaTriggered?: boolean;
  } = {
    refineWithLlm: false,
  },
): Promise<AbstractionJobStats> {
  const runStartedAt = nowIso();
  const runId = store.auditRepo.startMaintenance({
    agentId: ctx.agentId,
    sessionKey: options.batch?.sessionKey ?? ctx.sessionKey,
    jobType: "abstraction-jobs",
    stats: {},
    startedAt: runStartedAt,
  });
  try {
    if (options.batch && options.deltaTriggered === false) {
      const skippedStats: AbstractionJobStats = {
        eventsConsidered: 0,
        taskBeliefsConsidered: 0,
        factFamiliesConsidered: 0,
        candidatesMaterialized: 0,
        materializedCandidateIds: [],
        deferredByBudget: 0,
        llmCandidatesConsidered: 0,
        llmCandidatesRefined: 0,
        llmRefinementEnabled: options.refineWithLlm === true,
        deltaTriggered: false,
        skippedNoRelevantDelta: true,
        authoritySources: ["deterministic_aggregated"],
        semanticSources: ["upstream_structured"],
        activeCandidates: 0,
        probationaryCandidates: 0,
        candidateCandidates: 0,
        decayingCandidates: 0,
        quarantinedCandidates: 0,
        supersededCandidates: 0,
      };
      store.auditRepo.finishMaintenance({
        runId,
        agentId: ctx.agentId,
        jobType: "abstraction-jobs",
        statsJson: {
          ...(options.batch ? { batch: options.batch } : {}),
          ...skippedStats,
          llmBudget: snapshotMemoryLlmBudgetAudit(ctx.llmBudgetAudit),
        },
        startedAt: runStartedAt,
        completedAt: nowIso(),
        status: "completed",
      });
      return skippedStats;
    }

    const recentEvents = store.eventRepo.search({
      agentId: ctx.agentId,
      scopes: ctx.scopes,
      ...(options.batch?.sessionKey ? { sessionKey: options.batch.sessionKey } : {}),
      limit: 64,
      since: olderThanDays(
        Math.max(DERIVED_STATE_EVENT_WINDOW_DAYS, ctx.config.episodicDedupWindowDays * 2),
        ctx.now,
      ),
    });
    const recentBeliefs = store.beliefRepo.listByAgent({ agentId: ctx.agentId, limit: 192 });
    const taskBeliefs = recentBeliefs.filter((belief) => belief.memoryKind === "task").slice(0, 24);
    const beliefMap = buildBeliefMap(recentBeliefs);
    const activeTasks = store.taskRepo.listActive({
      agentId: ctx.agentId,
      scopes: ctx.scopes,
      ...(options.batch?.sessionKey ? { sessionKey: options.batch.sessionKey } : {}),
      limit: 24,
    });
    const factFamilies = Number(
      (
        store.client
          .prepare(
            `SELECT COUNT(*) AS count
               FROM (
                 SELECT canonical_subject, predicate
                   FROM facts
                  WHERE agent_id = ?
                    AND scope IN (${ctx.scopes.map(() => "?").join(", ")})
                    AND status IN ('active', 'uncertain')
                  GROUP BY canonical_subject, predicate
                  LIMIT 24
               )`,
          )
          .get(ctx.agentId, ...ctx.scopes) as { count: number } | undefined
      )?.count ?? 0,
    );

    const supports = [
      ...recentEvents
        .map((event) => buildEventSupport(beliefMap, event))
        .filter((entry): entry is StateSupportEntry => entry !== null),
      ...activeTasks.flatMap((task) => buildTaskSupports(beliefMap, task)),
    ];

    const groupedByKey = new Map<string, StateSupportEntry[]>();
    for (const entry of supports) {
      const familyKey = `${entry.scope}:${entry.stateKey}:${semanticValueKey(entry.stateKey, entry.valueJson)}`;
      const bucket = groupedByKey.get(familyKey) ?? [];
      bucket.push(entry);
      groupedByKey.set(familyKey, bucket);
    }

    const siblingGroupsByState = new Map<string, StateSupportEntry[][]>();
    for (const group of groupedByKey.values()) {
      const stateKey = `${group[0]!.scope}:${group[0]!.stateKey}`;
      const bucket = siblingGroupsByState.get(stateKey) ?? [];
      bucket.push(group);
      siblingGroupsByState.set(stateKey, bucket);
    }

    const derivedStateCandidates = [...groupedByKey.values()]
      .map((group) =>
        buildDerivedStateCandidate({
          agentId: ctx.agentId,
          scope: group[0]!.scope,
          stateKey: group[0]!.stateKey,
          valueJson: group[0]!.valueJson,
          supports: group,
          siblingGroups: siblingGroupsByState.get(`${group[0]!.scope}:${group[0]!.stateKey}`) ?? [
            group,
          ],
          now: ctx.now,
        }),
      )
      .filter((candidate): candidate is AbstractionCandidateRecord => candidate !== null)
      .sort((left, right) => {
        if (right.confidence !== left.confidence) {
          return right.confidence - left.confidence;
        }
        return right.usefulnessScore - left.usefulnessScore;
      });
    const workflowPatternCandidates = await buildWorkflowPatternCandidates(store, ctx, beliefMap);
    const graphHypothesisCandidates = buildGraphHypothesisCandidates(
      store,
      ctx,
      beliefMap,
      recentEvents,
      activeTasks,
    );
    const conceptCandidates = buildConceptCandidates(store, ctx, beliefMap);
    const candidates = [
      ...derivedStateCandidates,
      ...workflowPatternCandidates,
      ...graphHypothesisCandidates,
      ...conceptCandidates,
    ].sort(compareAbstractionCandidates);

    // Keep the background pipeline bounded without letting one abstraction type
    // starve the others.
    const candidateSelection = selectAbstractionCandidatesByType({
      derived_state: derivedStateCandidates,
      workflow_pattern: workflowPatternCandidates,
      graph_hypothesis: graphHypothesisCandidates,
      concept_candidate: conceptCandidates,
    });
    const selected = candidateSelection.selected;
    const refinement =
      options.refineWithLlm === false
        ? { candidates: selected, considered: 0, refined: 0 }
        : await refineCandidatesWithLlm(store, selected, ctx.now, ctx);
    const sourceEpoch = ctx.readEpoch ?? store.client.currentMemoryEpoch(ctx.agentId);
    for (const candidate of refinement.candidates) {
      candidate.derivedFromMinEpoch = candidate.derivedFromMinEpoch ?? sourceEpoch;
      candidate.derivedFromMaxEpoch = candidate.derivedFromMaxEpoch ?? sourceEpoch;
      candidate.materializedEpoch = store.client.nextMemoryEpoch(ctx.agentId, ctx.now);
      candidate.derivedFromKind = candidate.derivedFromKind ?? "abstraction_job";
      candidate.derivedFromIds = candidate.derivedFromIds ?? [
        ...candidate.supportContentRefs,
        ...candidate.supportBeliefIds,
      ];
      candidate.derivedAtEpoch = candidate.derivedAtEpoch ?? sourceEpoch;
      candidate.derivationPolicyVersion = candidate.derivationPolicyVersion ?? "memx-authority-v3";
      candidate.metadataJson = buildMaintenanceContractMetadata({
        existing: {
          ...candidate.metadataJson,
          ...(candidate.abstractionType === "concept_candidate"
            ? { promotionBlocker: "not_promotable_yet" }
            : {}),
        },
        sourceRef: `abstraction_candidate:${candidate.candidateId}`,
        supportContentRefs: candidate.supportContentRefs,
        supportBeliefIds: candidate.supportBeliefIds,
        derivedFromIds: candidate.derivedFromIds,
        semanticSource:
          (stringValue(candidate.metadataJson.semanticSource) as
            | MaintenanceSemanticSource
            | undefined) ?? "upstream_structured",
        semanticSources: Array.isArray(candidate.metadataJson.semanticSources)
          ? candidate.metadataJson.semanticSources.filter(
              (source): source is MaintenanceSemanticSource =>
                typeof source === "string" && source.trim().length > 0,
            )
          : undefined,
        authoritySource:
          options.refineWithLlm === true ? "llm_upgrade" : "deterministic_aggregated",
        generatedFrom:
          typeof candidate.metadataJson.generatedFrom === "string" ||
          Array.isArray(candidate.metadataJson.generatedFrom)
            ? (candidate.metadataJson.generatedFrom as string | string[])
            : "abstraction_job",
        recallLayer: "abstraction",
        answerEligibleByDefault: false,
        materializedEpoch: candidate.materializedEpoch,
        derivationPolicyVersion: candidate.derivationPolicyVersion,
      });
      store.abstractionRepo.upsert(candidate);
    }
    const materializedCandidateIds = refinement.candidates.map(
      (candidate) => candidate.candidateId,
    );

    const stages = candidateStageCounts(store, ctx.agentId);
    const semanticSources: MaintenanceSemanticSource[] = [
      ...new Set<MaintenanceSemanticSource>(
        refinement.candidates.flatMap((candidate) => {
          const primary = stringValue(candidate.metadataJson.semanticSource);
          const secondary = Array.isArray(candidate.metadataJson.semanticSources)
            ? candidate.metadataJson.semanticSources.filter(
                (source): source is MaintenanceSemanticSource =>
                  typeof source === "string" && source.trim().length > 0,
              )
            : [];
          return [...(primary ? [primary as MaintenanceSemanticSource] : []), ...secondary];
        }),
      ),
    ];
    const maintenanceContractDiagnostics = summarizeMaintenanceContractDiagnostics(
      refinement.candidates.map((candidate) => candidate.metadataJson),
    );
    const stats: AbstractionJobStats = {
      eventsConsidered: recentEvents.length,
      taskBeliefsConsidered: taskBeliefs.length,
      factFamiliesConsidered: factFamilies,
      candidatesMaterialized: refinement.candidates.length,
      materializedCandidateIds,
      deferredByBudget: Math.max(0, candidates.length - selected.length),
      candidateSelection: candidateSelection.stats,
      llmCandidatesConsidered: refinement.considered,
      llmCandidatesRefined: refinement.refined,
      llmRefinementEnabled: options.refineWithLlm === true,
      deltaTriggered: options.batch ? options.deltaTriggered !== false : undefined,
      skippedNoRelevantDelta: false,
      authoritySources:
        options.refineWithLlm === true
          ? ["deterministic_aggregated", "llm_upgrade"]
          : ["deterministic_aggregated"],
      semanticSources:
        options.refineWithLlm === true
          ? [
              ...new Set<MaintenanceSemanticSource>([
                ...semanticSources,
                "llm_upgrade",
              ]),
            ]
          : semanticSources,
      maintenanceContractDiagnostics,
      recallFacingDiagnostics: {
        recallVisible: maintenanceContractDiagnostics.recallVisibleCount > 0,
        answerEligibleByDefault: maintenanceContractDiagnostics.answerEligibleByDefaultCount > 0,
        sourceRefsForExpansion: maintenanceContractDiagnostics.sourceRefsForExpansion,
        recallLayers: maintenanceContractDiagnostics.recallLayers,
      },
      activeCandidates: stages.active,
      probationaryCandidates: stages.probationary,
      candidateCandidates: stages.candidate,
      decayingCandidates: stages.decaying,
      quarantinedCandidates: stages.quarantined,
      supersededCandidates: stages.superseded,
    };
    store.auditRepo.finishMaintenance({
      runId,
      agentId: ctx.agentId,
      jobType: "abstraction-jobs",
      statsJson: {
        ...(options.batch ? { batch: options.batch } : {}),
        ...stats,
        llmBudget: snapshotMemoryLlmBudgetAudit(ctx.llmBudgetAudit),
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
      jobType: "abstraction-jobs",
      statsJson: {
        ...(options.batch ? { batch: options.batch } : {}),
        error: String(error),
        llmBudget: snapshotMemoryLlmBudgetAudit(ctx.llmBudgetAudit),
      },
      startedAt: runStartedAt,
      completedAt: nowIso(),
      status: "failed",
    });
    throw error;
  }
}
