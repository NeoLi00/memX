import { clamp01, normalizeText, stableHash, truncateText } from "../support.js";
import type {
  EvidenceUnit,
  EvidenceUnitOrigin,
  EvidenceUnitRole,
  EvidencePacket,
  EvidencePacketAudit,
  EvidencePacketSlotAudit,
  EvidencePlanLayer,
  PromptEvidenceCandidate,
  QueryCompileResult,
  QueryEvidenceOperationType,
  QueryEvidenceSlot,
  QuerySemanticBridge,
} from "../types.js";
import type { CandidateGenerationResult } from "./candidateGeneration.js";
import {
  attributeSlotFromPredicate,
  canonicalAttributeSlot,
  predicateMatchesAttributeSlots,
  requestedAttributeSlotsFromText,
} from "./attributeSlots.js";
import { looksLikeBareMemoryUseInstruction } from "./semantic/heuristics.js";
import { semanticTextSimilarity } from "./semantic/textSimilarity.js";
import { normalizeSourceRefs, promptLineRole } from "./sourceRefs.js";
import { stateCurrentnessFromVectorMetadata } from "./stateLifecycle.js";

export type EvidenceAssemblerInput = {
  queryAnalysis: QueryCompileResult;
  candidateGenerationResult?: CandidateGenerationResult;
  promptEvidence: PromptEvidenceCandidate[];
  now?: string;
};

export type EvidenceAssemblerResult = {
  packets: EvidencePacket[];
  promptEvidence: PromptEvidenceCandidate[];
  audit?: EvidencePacketAudit;
};

const HIGH_LEVEL_LAYERS = new Set<EvidencePlanLayer>([
  "control",
  "strategy",
  "abstraction",
  "belief",
  "graph",
  "entity_alias",
]);

const TEMPORAL_REQUIRED_FIELDS = new Set(["temporal_marker", "date", "time", "observed_at"]);
const GENERIC_REQUIRED_FIELDS = new Set([
  "answer_value",
  "attribute_value",
  "countable_item",
  "source_evidence",
  "query_context",
  "temporal_marker",
  "date",
  "time",
  "observed_at",
]);

const HARD_EXCLUSION_PATTERNS = [
  /\b(?:BOOTSTRAP|IDENTITY|MEMORY\.md|USER\.md|debug|trace|stack trace)\b/iu,
  /\b(?:memory search|database inspection|plugin debugging|workspace file inspection)\b/iu,
];

type EvidenceEligibility = NonNullable<EvidencePacket["eligibility"]>;
type EvidenceGrade = NonNullable<EvidencePacket["grade"]>;

function slotLayers(slot: QueryEvidenceSlot): EvidencePlanLayer[] {
  return [...new Set([...slot.preferredLayers, ...slot.fallbackLayers])];
}

function operationType(queryAnalysis: QueryCompileResult): QueryEvidenceOperationType {
  return queryAnalysis.evidencePlan?.operation.type ?? "return_value";
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
  if (slot.role === "answer_event" || slot.role === "time_constraint") {
    return slot.role;
  }
  if (
    slot.role === "query_context" ||
    slot.role === "user_resource" ||
    slot.role === "prior_advice"
  ) {
    return slot.role;
  }
  return undefined;
}

function slotNeedsTemporalField(
  queryAnalysis: QueryCompileResult,
  slot: QueryEvidenceSlot,
): boolean {
  const op = operationType(queryAnalysis);
  if (op === "derive" || op === "compare") {
    return true;
  }
  return slot.requiredFields.some((field) => TEMPORAL_REQUIRED_FIELDS.has(field));
}

function sourceRefsForEntry(entry: PromptEvidenceCandidate): string[] {
  const metadataSourceRef =
    typeof entry.metadata?.sourceRef === "string" ? entry.metadata.sourceRef : undefined;
  const metadataSupportRefs = Array.isArray(entry.metadata?.supportRefs)
    ? entry.metadata.supportRefs.filter(
        (value): value is string => typeof value === "string" && value.trim().length > 0,
      )
    : [];
  const metadataLineage =
    entry.metadata?.lineage && typeof entry.metadata.lineage === "object"
      ? (entry.metadata.lineage as Record<string, unknown>)
      : undefined;
  const metadataLineageSourceRef =
    typeof metadataLineage?.sourceRef === "string" ? metadataLineage.sourceRef : undefined;
  return [
    ...new Set(
      [
        entry.sourceRef,
        entry.lineage?.sourceRef,
        metadataSourceRef,
        metadataLineageSourceRef,
        ...(entry.mergedSourceRefs ?? []),
        ...metadataSupportRefs,
      ].filter((value): value is string => Boolean(value)),
    ),
  ];
}

function supportRefsForEntry(entry: PromptEvidenceCandidate): string[] {
  const metadataSupportRefs = Array.isArray(entry.metadata?.supportRefs)
    ? entry.metadata.supportRefs.filter(
        (value): value is string => typeof value === "string" && value.trim().length > 0,
      )
    : [];
  return [...new Set([...(entry.mergedSourceRefs ?? []), ...metadataSupportRefs])];
}

function bindingSourceRefsForEntry(entry: PromptEvidenceCandidate): string[] {
  if (entry.sourceRef) {
    return [entry.sourceRef];
  }
  if (entry.lineage?.sourceRef) {
    return [entry.lineage.sourceRef];
  }
  if (typeof entry.metadata?.sourceRef === "string") {
    return [entry.metadata.sourceRef];
  }
  const metadataLineage =
    entry.metadata?.lineage && typeof entry.metadata.lineage === "object"
      ? (entry.metadata.lineage as Record<string, unknown>)
      : undefined;
  if (typeof metadataLineage?.sourceRef === "string") {
    return [metadataLineage.sourceRef];
  }
  return [...new Set((entry.mergedSourceRefs ?? []).filter(Boolean))];
}

function entriesShareBindingSourceIdentity(
  left: PromptEvidenceCandidate,
  right: PromptEvidenceCandidate,
): boolean {
  const leftRefs = bindingSourceRefsForEntry(left);
  const rightRefs = bindingSourceRefsForEntry(right);
  if (leftRefs.length > 0 && rightRefs.length > 0) {
    return leftRefs.some((sourceRef) => rightRefs.includes(sourceRef));
  }
  const leftAllRefs = sourceRefsForEntry(left);
  const rightAllRefs = sourceRefsForEntry(right);
  return leftAllRefs.some((sourceRef) => rightAllRefs.includes(sourceRef));
}

function sourceFamilyRef(sourceRef: string): string {
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
    return `user:lme:${parts[2]}:${normalizeSessionPart(parts[3] ?? "")}`;
  }
  if (parts[0] === "lme" && parts.length >= 3) {
    return `lme:${parts[1]}:${normalizeSessionPart(parts[2] ?? "")}`;
  }
  if (parts.length >= 3) {
    return parts.slice(0, 3).join(":");
  }
  return sourceRef.replace(/(?:[:/_-](?:turn|message|chunk|answer)?[:/_-]?\d+)$/iu, "");
}

function sourceTurnIndex(sourceRef: string): number | undefined {
  const match =
    /(?:^|[:/_-])(?:turn|message|chunk|answer)?[:/_-]?(\d+)(?=$|[:/_-](?:user|assistant|tool|memory)$)/iu.exec(
      sourceRef,
    );
  if (!match?.[1]) {
    return undefined;
  }
  const value = Number.parseInt(match[1], 10);
  return Number.isFinite(value) ? value : undefined;
}

function sourceSessionId(sourceRef: string): string | undefined {
  const parts = sourceRef.split(":").filter(Boolean);
  if (parts[0] === "user" && parts[1] === "agentmem" && parts.length >= 4) {
    return parts[3];
  }
  if (parts[0] === "agentmem" && parts.length >= 3) {
    return parts[2];
  }
  if (parts[0] === "user" && parts[1] === "lme" && parts.length >= 4) {
    return parts[3];
  }
  if (parts[0] === "lme" && parts.length >= 3) {
    return parts[2];
  }
  return undefined;
}

function sourceRefsAdjacent(leftRefs: string[], rightRefs: string[], maxDistance = 2): boolean {
  for (const left of leftRefs) {
    const leftFamily = sourceFamilyRef(left);
    const leftTurn = sourceTurnIndex(left);
    for (const right of rightRefs) {
      if (leftFamily !== sourceFamilyRef(right)) {
        continue;
      }
      const rightTurn = sourceTurnIndex(right);
      if (leftTurn === undefined || rightTurn === undefined) {
        if (left === right) {
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

function sourceKey(sourceRefs: string[], fallbackText: string): string {
  return sourceRefs.length > 0
    ? `source:${sourceRefs.sort().join("|")}`
    : `text:${normalizeText(fallbackText).slice(0, 180)}`;
}

function entryKey(entry: PromptEvidenceCandidate): string {
  return [entry.surface, entry.id, entry.sourceRef ?? "", normalizeText(entry.text).slice(0, 80)]
    .filter(Boolean)
    .join("|");
}

function entryAuthorRole(
  entry: PromptEvidenceCandidate,
): "user" | "assistant" | "tool" | "memory" | "unknown" {
  const text = (entry.rawText ?? entry.text).trim().toLowerCase();
  if (/^(?:\[assistant\]|assistant)\s*:/iu.test(text) || text.startsWith("assistant ")) {
    return "assistant";
  }
  if (/^(?:\[user\]|user)\s*:/iu.test(text)) {
    return "user";
  }
  if (/^(?:\[tool\]|tool)\s*:/iu.test(text)) {
    return "tool";
  }
  const roleFromRefs = (sourceRefs: string[]) => {
    const hasUser = sourceRefs.some((ref) => /(?:^|:)user$/iu.test(ref));
    const hasAssistant = sourceRefs.some((ref) => /(?:^|:)assistant$/iu.test(ref));
    const hasTool = sourceRefs.some((ref) => /(?:^|:)tool$/iu.test(ref));
    if (hasUser && !hasAssistant && !hasTool) {
      return "user" as const;
    }
    if (hasAssistant && !hasUser && !hasTool) {
      return "assistant" as const;
    }
    if (hasTool && !hasUser && !hasAssistant) {
      return "tool" as const;
    }
    return undefined;
  };
  // Merged/support refs may include an adjacent assistant answer. The author role
  // belongs to the primary evidence source, so do not let merged refs rewrite it.
  const primaryRole = roleFromRefs(bindingSourceRefsForEntry(entry));
  if (primaryRole) {
    return primaryRole;
  }
  const sourceRole = roleFromRefs(sourceRefsForEntry(entry));
  if (sourceRole) {
    return sourceRole;
  }
  if (sourceRefsForEntry(entry).some((ref) => /(?:^|:)user$/iu.test(ref))) {
    return "user";
  }
  if (sourceRefsForEntry(entry).some((ref) => /(?:^|:)tool$/iu.test(ref))) {
    return "tool";
  }
  if (entry.surface === "fact" || entry.surface === "event") {
    return "memory";
  }
  return "unknown";
}

function evidenceUnitOrigin(entry: PromptEvidenceCandidate): EvidenceUnitOrigin {
  const sourceId = entry.lineage?.sourceId ?? entry.id;
  if (sourceId.startsWith("belief:")) {
    return "belief";
  }
  if (sourceId.startsWith("strategy:")) {
    return "strategy";
  }
  if (entry.metadata?.recallLayer === "belief") {
    return "belief";
  }
  if (entry.metadata?.recallLayer === "strategy") {
    return "strategy";
  }
  if (entry.surface === "chunk") {
    return entry.source === "support_ref" || entry.lineage?.sourceKind === "chunk"
      ? "raw_chunk"
      : "derived_summary";
  }
  if (entry.surface === "fact") {
    return "canonical_fact";
  }
  if (entry.surface === "event") {
    return entry.source === "support_ref" ? "raw_chunk" : "event";
  }
  return "snippet";
}

function evidenceUnitRoles(
  queryAnalysis: QueryCompileResult,
  entry: PromptEvidenceCandidate,
): EvidenceUnitRole[] {
  const role = inferredSlotRole(queryAnalysis, entry);
  const roles: EvidenceUnitRole[] = [];
  if (role) {
    roles.push(role);
  }
  if (entry.source === "support_ref" && !roles.includes("support")) {
    roles.push("support");
  }
  if (roles.length === 0) {
    roles.push("support");
  }
  return [...new Set(roles)];
}

function evidenceUnitFromEntry(
  queryAnalysis: QueryCompileResult,
  entry: PromptEvidenceCandidate,
): EvidenceUnit {
  const sourceRefs = sourceRefsForEntry(entry);
  const supportRefs = supportRefsForEntry(entry);
  const primarySourceRef = sourceRefs[0];
  return {
    unitId: `unit:${stableHash([
      entry.surface,
      entry.id,
      ...sourceRefs,
      normalizeText(entry.text).slice(0, 160),
    ])}`,
    surfaceRefs: [entry.id],
    sourceRefs,
    normalizedSourceRefs: normalizeSourceRefs(sourceRefs),
    supportRefs,
    normalizedSupportRefs: normalizeSourceRefs(supportRefs),
    derivedFromRefs: entry.lineage?.sourceRef ? [entry.lineage.sourceRef] : [],
    normalizedDerivedFromRefs: normalizeSourceRefs(
      entry.lineage?.sourceRef ? [entry.lineage.sourceRef] : [],
    ),
    neighborRefs:
      entry.metadata?.sourceExpansion === true && Array.isArray(entry.metadata.neighborOf)
        ? entry.metadata.neighborOf.filter(
            (value): value is string => typeof value === "string" && value.trim().length > 0,
          )
        : [],
    normalizedNeighborRefs: normalizeSourceRefs(
      entry.metadata?.sourceExpansion === true && Array.isArray(entry.metadata.neighborOf)
        ? entry.metadata.neighborOf
        : [],
    ),
    sessionId: primarySourceRef ? sourceSessionId(primarySourceRef) : undefined,
    turnIndex: primarySourceRef ? sourceTurnIndex(primarySourceRef) : undefined,
    authorRole: entryAuthorRole(entry),
    observedAt: entry.observedAt,
    rawText: entry.rawText ?? entry.text,
    displayText: truncateText(entry.text, 560),
    roles: evidenceUnitRoles(queryAnalysis, entry),
    origin: evidenceUnitOrigin(entry),
  };
}

const reportedQuestionPrefixPattern =
  /^\s*(?:(?:\[user\]|user)\s*:?\s*)?(?:they\s+also\s+ask|user\s+asks?|(?:a|the)\s+reviewer\s+asks?|security\s+review\s+asks?|(?:security\s+)?reviewer\s+asks?|question\s*:)/iu;

function entryIsQuestionLike(entry: PromptEvidenceCandidate): boolean {
  const raw = entry.rawText ?? entry.text;
  const trimmed = raw.trim();
  return /\?\s*$/u.test(trimmed) || reportedQuestionPrefixPattern.test(raw);
}

function entryIsQueryLikeEvidence(
  queryAnalysis: QueryCompileResult,
  entry: PromptEvidenceCandidate,
): boolean {
  const author = entryAuthorRole(entry);
  if (author === "assistant" || author === "tool") {
    return false;
  }
  const text = entry.rawText ?? entry.text;
  const explicitlyFramedAsQuery = reportedQuestionPrefixPattern.test(text);
  if (explicitlyFramedAsQuery) {
    return true;
  }
  const echo = queryEchoScore(queryAnalysis, text);
  return echo >= 0.86 || (entryIsQuestionLike(entry) && echo >= 0.52);
}

function unitIsQuestionLike(queryAnalysis: QueryCompileResult, unit: EvidenceUnit): boolean {
  if (unit.authorRole === "assistant" || unit.authorRole === "tool") {
    return false;
  }
  const text = unit.rawText || unit.displayText;
  return (
    /\?\s*$/u.test(text.trim()) ||
    reportedQuestionPrefixPattern.test(text) ||
    queryEchoScore(queryAnalysis, text) >= 0.86
  );
}

function unitHasAnswerRole(unit: EvidenceUnit): boolean {
  return unit.roles.some(
    (role) =>
      role === "answer_value" ||
      role === "answer_event" ||
      role === "user_resource" ||
      role === "prior_advice",
  );
}

function queryAllowsAssistantAuthoredAnswer(queryAnalysis: QueryCompileResult): boolean {
  const slots = queryAnalysis.evidencePlan?.slots ?? [];
  const adviceOrResourceMode =
    operationType(queryAnalysis) === "tailor_advice" ||
    slots.some(
      (slot) =>
        slot.role === "prior_advice" ||
        slot.requiredRole === "prior_advice" ||
        slot.role === "user_resource" ||
        slot.requiredRole === "user_resource",
    );
  if (adviceOrResourceMode) {
    return true;
  }
  if (
    queryAnalysis.answerMode === "attribute_lookup" ||
    queryAnalysis.queryShape.evidenceNeed === "canonical_state"
  ) {
    return false;
  }
  return true;
}

function unitLooksLikeAssistantNonAnswer(unit: EvidenceUnit): boolean {
  if (unit.authorRole !== "assistant") {
    return false;
  }
  const text = `${unit.rawText || ""}\n${unit.displayText || ""}`;
  return /(?:\b(?:i\s+do\s+not|i\s+don't|cannot|can't)\s+(?:know|answer)\b|\bno\s+(?:information|record|memory)\b|无法(?:回答|确定)|不知道|不清楚|没有(?:关于|相关)?.{0,24}(?:信息|记录|记忆)|未(?:找到|记录)|需要(?:你)?提供.{0,24}(?:文件|路径|信息))/iu.test(
    text,
  );
}

function unitLooksLikeAnswer(queryAnalysis: QueryCompileResult, unit: EvidenceUnit): boolean {
  if (unitLooksLikeAssistantNonAnswer(unit)) {
    return false;
  }
  if (unitIsQuestionLike(queryAnalysis, unit) && !unit.roles.includes("user_resource")) {
    return false;
  }
  if (unit.authorRole === "assistant" && !queryAllowsAssistantAuthoredAnswer(queryAnalysis)) {
    return false;
  }
  if (unitHasAnswerRole(unit)) {
    return true;
  }
  if (unit.authorRole === "assistant" && queryAllowsAssistantAuthoredAnswer(queryAnalysis)) {
    return true;
  }
  return (
    unit.origin === "canonical_fact" ||
    unit.origin === "event" ||
    unit.origin === "raw_chunk" ||
    unit.origin === "snippet"
  );
}

function answerUnitsForMode(
  queryAnalysis: QueryCompileResult,
  units: EvidenceUnit[],
): EvidenceUnit[] {
  if (queryAnalysis.answerMode !== "multi_evidence") {
    return units;
  }
  const canonicalFacts = units.filter((unit) => unit.origin === "canonical_fact");
  return canonicalFacts.length >= 2 ? canonicalFacts : units;
}

function refsForUnit(unit: EvidenceUnit): string[] {
  return [
    ...new Set([
      ...unit.sourceRefs,
      ...(unit.supportRefs ?? []),
      ...(unit.derivedFromRefs ?? []),
      ...(unit.neighborRefs ?? []),
    ]),
  ];
}

function exactDisplayKey(text: string): string {
  return normalizeText(text).replace(/\s+/gu, " ").trim();
}

function compactResourceDisplay(text: string): string | undefined {
  const resource =
    /(?:^|\|\s*)resource\s*=\s*([^|]+)/iu.exec(text)?.[1]?.trim() ??
    /(?:has_resource|user\.resource\.)([^|:(]+)/iu.exec(text)?.[1]?.replace(/[_-]+/gu, " ").trim();
  if (!resource) {
    return undefined;
  }
  const affordances = /affordances\s*=\s*([^|]+)/iu.exec(text)?.[1]?.trim();
  const domains = /domains\s*=\s*([^|]+)/iu.exec(text)?.[1]?.trim();
  const support =
    /supportText\s*=\s*([^|]+)/iu.exec(text)?.[1]?.trim() ??
    /evidence:\s*([^|]+)/iu.exec(text)?.[1]?.trim();
  return truncateText(
    [
      `The user has ${resource}`,
      affordances ? `useful for ${affordances}` : undefined,
      domains ? `domain: ${domains}` : undefined,
      support ? `evidence: ${support}` : undefined,
    ]
      .filter(Boolean)
      .join(" | "),
    360,
  );
}

function displayLabelForUnit(unit: EvidenceUnit): string {
  if (unit.roles.includes("user_resource")) {
    return "resource";
  }
  if (unit.roles.includes("answer_event") || unit.origin === "event") {
    return "event";
  }
  if (unit.roles.includes("query_context") || unit.roles.includes("time_constraint")) {
    return "context";
  }
  return "answer";
}

function displayLabelForAnswerUnit(unit: EvidenceUnit): string {
  if (unit.roles.includes("user_resource")) {
    return "resource";
  }
  if (unit.roles.includes("answer_value")) {
    return "answer";
  }
  if (unit.roles.includes("answer_event") || unit.origin === "event") {
    return "event";
  }
  if (
    unit.roles.includes("query_context") &&
    unit.authorRole !== "assistant" &&
    !unitHasAnswerRole(unit)
  ) {
    return "context";
  }
  return "answer";
}

function unitDisplayText(unit: EvidenceUnit, maxLength = 360): string {
  const raw = unit.displayText || unit.rawText;
  const compactResource = unit.roles.includes("user_resource")
    ? compactResourceDisplay(raw)
    : undefined;
  return truncateText(compactResource ?? raw, maxLength);
}

function classifyPacketUnits(params: {
  queryAnalysis: QueryCompileResult;
  selectedEntry: PromptEvidenceCandidate;
  contextCandidates: PromptEvidenceCandidate[];
}): {
  answerUnits: EvidenceUnit[];
  contextUnits: EvidenceUnit[];
  supportUnits: EvidenceUnit[];
} {
  const selectedUnit = evidenceUnitFromEntry(params.queryAnalysis, params.selectedEntry);
  const contextUnits = params.contextCandidates.map((entry) =>
    evidenceUnitFromEntry(params.queryAnalysis, entry),
  );
  const selectedRole = inferredSlotRole(params.queryAnalysis, params.selectedEntry);
  const selectedAuthor = entryAuthorRole(params.selectedEntry);
  const selectedCanAnswerDespiteQuestion =
    selectedRole === "user_resource" || selectedRole === "prior_advice";
  const selectedIsQuestion =
    !selectedCanAnswerDespiteQuestion && entryIsQueryLikeEvidence(params.queryAnalysis, params.selectedEntry);
  const contextAnswerUnits = contextUnits.filter((unit) =>
    unitLooksLikeAnswer(params.queryAnalysis, unit),
  );

  const selectedAssistantAnswer =
    selectedUnit.authorRole === "assistant" &&
    !selectedIsQuestion &&
    unitLooksLikeAnswer(params.queryAnalysis, selectedUnit);

  if (
    selectedIsQuestion ||
    ((selectedRole === "query_context" || selectedRole === "time_constraint") &&
      !selectedAssistantAnswer)
  ) {
    const answerUnits = answerUnitsForMode(
      params.queryAnalysis,
      contextAnswerUnits.filter(
        (unit) =>
          (unit.authorRole === "assistant" || unitHasAnswerRole(unit)) &&
          !unitIsQuestionLike(params.queryAnalysis, unit),
      ),
    );
    if (answerUnits.length > 0) {
      return {
        answerUnits,
        contextUnits: [selectedUnit, ...contextUnits.filter((unit) => !answerUnits.includes(unit))],
        supportUnits: [],
      };
    }
  }

  if (unitLooksLikeAnswer(params.queryAnalysis, selectedUnit) && !selectedIsQuestion) {
    const answerUnits = answerUnitsForMode(
      params.queryAnalysis,
      params.queryAnalysis.answerMode === "multi_evidence"
        ? [
            selectedUnit,
            ...contextAnswerUnits.filter(
              (unit) =>
                unit.unitId !== selectedUnit.unitId &&
                !unitIsQuestionLike(params.queryAnalysis, unit),
            ),
          ]
        : [selectedUnit],
    );
    return {
      answerUnits,
      contextUnits: contextUnits.filter(
        (unit) =>
          !answerUnits.includes(unit) &&
          (unit.roles.includes("query_context") || unit.roles.includes("time_constraint")),
      ),
      supportUnits: contextUnits.filter(
        (unit) =>
          !answerUnits.includes(unit) &&
          !unit.roles.includes("query_context") &&
          !unit.roles.includes("time_constraint"),
      ),
    };
  }

  return {
    answerUnits: contextAnswerUnits,
    contextUnits: [
      selectedUnit,
      ...contextUnits.filter((unit) => !contextAnswerUnits.includes(unit)),
    ],
    supportUnits: [],
  };
}

function packetDisplayLines(params: {
  queryAnalysis: QueryCompileResult;
  answerUnits: EvidenceUnit[];
  contextUnits: EvidenceUnit[];
  supportUnits: EvidenceUnit[];
}): {
  displayLines: string[];
  hiddenExactDuplicates: NonNullable<EvidencePacket["hiddenExactDuplicates"]>;
} {
  const displayLines: string[] = [];
  const hiddenExactDuplicates: NonNullable<EvidencePacket["hiddenExactDuplicates"]> = [];
  const seen = new Map<string, { displayText: string; sourceRefs: string[] }>();
  const contextText = params.contextUnits
    .map((unit) => unitDisplayText(unit, 220))
    .find((text) => text.length > 0);
  const units =
    params.answerUnits.length > 0
      ? params.answerUnits
      : [...params.contextUnits, ...params.supportUnits].slice(0, 1);

  for (const unit of units) {
    const answerText = unitDisplayText(unit, 420);
    if (!answerText) {
      continue;
    }
    const answerLabel = params.answerUnits.includes(unit)
      ? displayLabelForAnswerUnit(unit)
      : unitIsQuestionLike(params.queryAnalysis, unit) && !unit.roles.includes("user_resource")
        ? "context"
        : displayLabelForUnit(unit);
    const line =
      contextText && semanticTextSimilarity(answerText, contextText) < 0.78
        ? `[${answerLabel}] ${answerText} | [context] ${contextText}`
        : `[${answerLabel}] ${answerText}`;
    const key = exactDisplayKey(line);
    const refs = refsForUnit(unit);
    const existing = seen.get(key);
    if (existing) {
      hiddenExactDuplicates.push({
        displayText: existing.displayText,
        sourceRefs: existing.sourceRefs,
        hiddenSourceRefs: refs,
      });
      continue;
    }
    seen.set(key, { displayText: line, sourceRefs: refs });
    displayLines.push(line);
  }
  return { displayLines, hiddenExactDuplicates };
}

function queryEchoScore(queryAnalysis: QueryCompileResult, text: string): number {
  return Math.max(
    semanticTextSimilarity(queryAnalysis.queryText, text),
    semanticTextSimilarity(queryAnalysis.focusedQuery || queryAnalysis.queryText, text),
  );
}

function numericBreakdown(entry: PromptEvidenceCandidate, key: string): number {
  const value = entry.scoreBreakdown?.[key];
  return typeof value === "number" ? clamp01(value) : 0;
}

function slotCoverageScore(entry: PromptEvidenceCandidate): number {
  const explicitCoverage =
    entry.coverage &&
    (entry.coverage.requiredHits.length > 0 || entry.coverage.missingRequired.length > 0)
      ? entry.coverage.coverageScore * (entry.coverage.missingRequired.length > 0 ? 0.45 : 1)
      : 0;
  return Math.max(
    explicitCoverage,
    ...(entry.slotCoverage ?? []).map(
      (coverage) => coverage.coverageScore * (coverage.missingRequired.length > 0 ? 0.45 : 1),
    ),
  );
}

function bridgeSupportScore(entry: PromptEvidenceCandidate): number {
  return Math.max(0, ...(entry.bridgeMatches ?? []).map((match) => match.score));
}

function bridgePositiveSignalScore(entry: PromptEvidenceCandidate): number {
  return Math.max(0, ...(entry.bridgeMatches ?? []).map((match) => match.positiveSignalScore));
}

function bridgeNegativeSignalScore(entry: PromptEvidenceCandidate): number {
  return Math.max(
    0,
    ...(entry.bridgeMatches ?? [])
      .filter(
        (match) =>
          match.role !== "answer_value" &&
          match.role !== "answer_event" &&
          match.role !== "user_resource" &&
          match.role !== "prior_advice",
      )
      .map((match) => match.negativeSignalScore),
  );
}

function bridgeAnswerRoleScore(entry: PromptEvidenceCandidate): number {
  return Math.max(
    0,
    ...(entry.bridgeMatches ?? [])
      .filter(
        (match) =>
          match.role === "answer_value" ||
          match.role === "answer_event" ||
          match.role === "user_resource" ||
          match.role === "prior_advice",
      )
      .map((match) => match.score),
  );
}

function bridgeRoleScore(
  entry: PromptEvidenceCandidate,
  roles: Array<NonNullable<PromptEvidenceCandidate["slotEvidenceRole"]>>,
): number {
  return Math.max(
    0,
    ...(entry.bridgeMatches ?? [])
      .filter((match) => roles.includes(match.role))
      .map((match) => match.score),
  );
}

function bridgePositiveRoleScore(
  entry: PromptEvidenceCandidate,
  roles: Array<NonNullable<PromptEvidenceCandidate["slotEvidenceRole"]>>,
): number {
  return Math.max(
    0,
    ...(entry.bridgeMatches ?? [])
      .filter((match) => roles.includes(match.role))
      .map((match) => match.positiveSignalScore),
  );
}

type EvidenceShape = QuerySemanticBridge["evidenceShape"];

function bridgeShapeScore(entry: PromptEvidenceCandidate, shapes: EvidenceShape[]): number {
  return Math.max(
    0,
    ...(entry.bridgeMatches ?? [])
      .filter((match) => shapes.includes(match.evidenceShape))
      .map((match) => match.score),
  );
}

function bridgeShapePositiveScore(entry: PromptEvidenceCandidate, shapes: EvidenceShape[]): number {
  return Math.max(
    0,
    ...(entry.bridgeMatches ?? [])
      .filter((match) => shapes.includes(match.evidenceShape))
      .map((match) => match.positiveSignalScore),
  );
}

function semanticBridgesForShapes(
  queryAnalysis: QueryCompileResult,
  shapes: EvidenceShape[],
): QuerySemanticBridge[] {
  return (queryAnalysis.semanticBridges ?? []).filter((bridge) =>
    shapes.includes(bridge.evidenceShape),
  );
}

function queryUsesEvidenceShapes(queryAnalysis: QueryCompileResult, shapes: EvidenceShape[]): boolean {
  return semanticBridgesForShapes(queryAnalysis, shapes).length > 0;
}

function semanticBridgePlanTexts(bridge: QuerySemanticBridge): string[] {
  return [
    bridge.sourceConcept,
    bridge.evidenceShape,
    ...bridge.retrievalQueries,
    ...bridge.positiveSignals,
  ]
    .map((text) => text.trim())
    .filter(Boolean);
}

function evidenceShapeFitScore(
  queryAnalysis: QueryCompileResult,
  entry: PromptEvidenceCandidate,
  shapes: EvidenceShape[],
): number {
  const bridges = semanticBridgesForShapes(queryAnalysis, shapes);
  if (bridges.length === 0) {
    return 0;
  }
  const candidateText = `${entry.text} ${entry.scoringText ?? ""}`;
  const bridgeMatch = Math.max(
    bridgeShapeScore(entry, shapes),
    bridgeShapePositiveScore(entry, shapes),
  );
  const bridgeMatchRequired = shapes.some(
    (shape) =>
      shape === "validation_evidence" ||
      shape === "status_answer" ||
      shape === "decision_value" ||
      shape === "availability_statement",
  );
  if (bridgeMatchRequired) {
    return clamp01(bridgeMatch);
  }
  const planMatch = Math.max(
    0,
    ...bridges
      .flatMap((bridge) => semanticBridgePlanTexts(bridge))
      .map((text) => planTextMatchScore(text, candidateText)),
  );
  return clamp01(Math.max(bridgeMatch, planMatch));
}

function evidenceShapeFitScoreForTexts(
  queryAnalysis: QueryCompileResult,
  texts: string[],
  shapes: EvidenceShape[],
): number {
  const bridges = semanticBridgesForShapes(queryAnalysis, shapes);
  if (bridges.length === 0 || texts.length === 0) {
    return 0;
  }
  const planTexts = bridges.flatMap((bridge) => semanticBridgePlanTexts(bridge));
  return Math.max(
    0,
    ...planTexts.flatMap((planText) => texts.map((text) => planTextMatchScore(planText, text))),
  );
}

function strongestBridgeRole(
  entry: PromptEvidenceCandidate,
): PromptEvidenceCandidate["slotEvidenceRole"] | undefined {
  const match = [...(entry.bridgeMatches ?? [])].sort((left, right) => right.score - left.score)[0];
  return match?.score && match.score >= 0.34 ? match.role : undefined;
}

function filledSlotIds(entry: PromptEvidenceCandidate): string[] {
  return entry.filledSlotIds && entry.filledSlotIds.length > 0
    ? entry.filledSlotIds
    : (entry.slotCoverage ?? [])
        .filter((coverage) => coverage.filled)
        .map((coverage) => coverage.slotId);
}

function matchedSlotIds(entry: PromptEvidenceCandidate): string[] {
  return [
    ...new Set(
      [
        ...filledSlotIds(entry),
        ...(entry.slotCoverage ?? [])
          .filter((coverage) => coverage.coverageScore >= 0.24)
          .map((coverage) => coverage.slotId),
      ].filter(Boolean),
    ),
  ];
}

function normalizedHitSet(entries: PromptEvidenceCandidate[]): Set<string> {
  const hits = new Set<string>();
  for (const entry of entries) {
    for (const hit of entry.coverage?.requiredHits ?? []) {
      const normalized = normalizeText(hit);
      if (normalized) {
        hits.add(normalized);
      }
    }
    for (const coverage of entry.slotCoverage ?? []) {
      for (const hit of coverage.requiredHits) {
        const normalized = normalizeText(hit);
        if (normalized) {
          hits.add(normalized);
        }
      }
    }
  }
  return hits;
}

function missingRequiredAfterContext(
  entry: PromptEvidenceCandidate,
  contextCandidates: PromptEvidenceCandidate[],
): string[] {
  const entries = [entry, ...contextCandidates];
  const hits = normalizedHitSet(entries);
  const isCovered = (value: string): boolean => {
    const normalized = normalizeText(value);
    if (!normalized) {
      return true;
    }
    if (GENERIC_REQUIRED_FIELDS.has(normalized.replace(/\s+/gu, "_"))) {
      return true;
    }
    if (hits.has(normalized)) {
      return true;
    }
    return entries.some((candidate) => {
      const text = normalizeText(`${candidate.text} ${candidate.scoringText ?? ""}`);
      const valueTokens = normalized.split(/\s+/gu).filter((token) => token.length >= 4);
      const tokenCoverage =
        valueTokens.length > 0
          ? valueTokens.filter((token) => text.includes(token)).length /
            Math.min(valueTokens.length, 4)
          : 0;
      return (
        text.includes(normalized) ||
        tokenCoverage >= 0.5 ||
        semanticTextSimilarity(value, candidate.text) >= 0.5
      );
    });
  };
  const missing = new Set<string>();
  for (const candidate of entries) {
    for (const value of candidate.coverage?.missingRequired ?? []) {
      if (!isCovered(value)) {
        missing.add(value);
      }
    }
    for (const coverage of candidate.slotCoverage ?? []) {
      for (const value of coverage.missingRequired) {
        if (!isCovered(value)) {
          missing.add(value);
        }
      }
    }
  }
  return [...missing];
}

function inferredSlotRole(
  queryAnalysis: QueryCompileResult,
  entry: PromptEvidenceCandidate,
): PromptEvidenceCandidate["slotEvidenceRole"] | undefined {
  if (entry.slotEvidenceRole) {
    return entry.slotEvidenceRole;
  }
  const slots = queryAnalysis.evidencePlan?.slots ?? [];
  const slotById = new Map(slots.map((slot) => [slot.id, slot]));
  const answerSlotIds = new Set(answerPlanSlots(queryAnalysis).map((slot) => slot.id));
  const coverageRole = (entry.slotCoverage ?? [])
    .map((coverage) => {
      const slot = slotById.get(coverage.slotId);
      const explicitRole = evidenceSlotRequiredRole(slot);
      const inferredAnswerRole =
        !explicitRole && slot && answerSlotIds.has(slot.id)
          ? operationType(queryAnalysis) === "aggregate" ||
            queryAnalysis.answerMode === "count_aggregate"
            ? "answer_event"
            : operationType(queryAnalysis) === "tailor_advice"
              ? "user_resource"
              : "answer_value"
          : undefined;
      return {
        role: explicitRole ?? inferredAnswerRole,
        score: coverage.filled ? coverage.coverageScore + 0.18 : coverage.coverageScore,
      };
    })
    .filter((item) => item.role)
    .sort((left, right) => right.score - left.score)[0]?.role;
  const bridgeRole = strongestBridgeRole(entry);
  if (
    bridgeRole &&
    (bridgeRole === "answer_value" ||
      bridgeRole === "answer_event" ||
      bridgeRole === "user_resource" ||
      bridgeRole === "prior_advice")
  ) {
    const coverageIsContext =
      coverageRole === "query_context" || coverageRole === "time_constraint";
    if (!coverageRole || coverageIsContext || bridgeRoleScore(entry, [bridgeRole]) >= 0.52) {
      return bridgeRole;
    }
  }
  return coverageRole ?? bridgeRole;
}

function metadataStringArray(metadata: Record<string, unknown> | undefined, key: string): string[] {
  const value = metadata?.[key];
  if (!Array.isArray(value)) {
    return [];
  }
  return value.filter((item): item is string => typeof item === "string" && item.trim().length > 0);
}

function hasStructuredResourceSignal(entry: PromptEvidenceCandidate): boolean {
  return Boolean(
    entry.slotEvidenceRole === "user_resource" ||
      bridgeShapeScore(entry, ["resource_affordance"]) > 0 ||
      metadataStringArray(entry.metadata, "domains").length > 0 ||
      metadataStringArray(entry.metadata, "affordances").length > 0 ||
      (typeof entry.metadata?.resourceType === "string" &&
        entry.metadata.resourceType.trim().length > 0) ||
      entry.metadata?.signalKind === "resourceAssertion",
  );
}

function hasResourceAffordanceSignal(entry: PromptEvidenceCandidate): boolean {
  return Boolean(
    bridgeShapeScore(entry, ["resource_affordance"]) > 0 ||
      metadataStringArray(entry.metadata, "domains").length > 0 ||
      metadataStringArray(entry.metadata, "affordances").length > 0 ||
      (typeof entry.metadata?.resourceType === "string" &&
        entry.metadata.resourceType.trim().length > 0) ||
      entry.metadata?.signalKind === "resourceAssertion",
  );
}

function isAssistantAcknowledgement(entry: PromptEvidenceCandidate): boolean {
  if (entryAuthorRole(entry) !== "assistant") {
    return false;
  }
  return (
    entry.metadata?.semanticRole === "assistant_acknowledgement" ||
    entry.metadata?.memoryClass === "assistant_acknowledgement"
  );
}

function candidateRetrievalScore(entry: PromptEvidenceCandidate): number {
  return clamp01(
    Math.max(
      entry.injectionScore ?? 0,
      entry.priority,
      entry.goalScore,
      entry.semanticScore ?? 0,
      numericBreakdown(entry, "retrievalScore"),
    ),
  );
}

function isStateEvidence(entry: PromptEvidenceCandidate): boolean {
  return (
    entry.metadata?.memxDocType === "state" ||
    typeof entry.metadata?.stateLifecycleKind === "string" ||
    entry.lineage?.sourceKind === "state" ||
    entry.lineage?.canonicalKind === "state"
  );
}

function stateHardExclusions(entry: PromptEvidenceCandidate, now: string): string[] {
  if (!isStateEvidence(entry)) {
    return [];
  }
  const fromMetadata = stringArray(entry.metadata?.stateCurrentnessHardExclusions);
  const currentness = stateCurrentnessFromVectorMetadata(entry.metadata, now);
  const lifecycleKind =
    typeof entry.metadata?.stateLifecycleKind === "string"
      ? entry.metadata.stateLifecycleKind
      : undefined;
  const rawSupportRefs = stringArray(entry.metadata?.stateSupportRefs).filter(
    (ref) => !ref.startsWith("abstraction_candidate:"),
  );
  const supportOnlyBlockers =
    lifecycleKind === "derived_maintenance" && rawSupportRefs.length === 0
      ? ["maintenance-state-missing-raw-support"]
      : [];
  return [
    ...new Set([...fromMetadata, ...(currentness?.hardExclusions ?? []), ...supportOnlyBlockers]),
  ];
}

function stateSoftPenalties(
  entry: PromptEvidenceCandidate,
): Array<{ reason: string; weight: number }> {
  if (!isStateEvidence(entry)) {
    return [];
  }
  const penalties = stringArray(entry.metadata?.stateCurrentnessSoftPenalties).map((reason) => ({
    reason,
    weight: reason === "maintenance-derived-without-raw-support" ? 0.22 : 0.1,
  }));
  const score =
    typeof entry.metadata?.stateCurrentnessScore === "number"
      ? clamp01(entry.metadata.stateCurrentnessScore)
      : 0.5;
  if (score < 0.42) {
    penalties.push({ reason: "low-state-currentness", weight: 0.16 });
  }
  if (entry.metadata?.stateAnswerEligibleByDefault === false) {
    penalties.push({ reason: "state-support-only-by-default", weight: 0.12 });
  }
  return penalties;
}

function authorityScore(entry: PromptEvidenceCandidate): number {
  const stateLifecycleKind =
    typeof entry.metadata?.stateLifecycleKind === "string"
      ? entry.metadata.stateLifecycleKind
      : undefined;
  if (isStateEvidence(entry)) {
    const stateScore =
      stateLifecycleKind === "derived_maintenance"
        ? 0.42
        : stateLifecycleKind === "durable_profile"
          ? 0.66
          : 0.55;
    const sourceTrace = entry.sourceRef || (entry.mergedSourceRefs?.length ?? 0) > 0 ? 0.08 : 0;
    return clamp01(stateScore + sourceTrace);
  }
  const author = entryAuthorRole(entry);
  const surfaceScore =
    entry.surface === "fact"
      ? 0.86
      : entry.surface === "event"
        ? 0.78
        : entry.surface === "chunk"
          ? author === "assistant"
            ? 0.66
            : 0.7
          : 0.58;
  const sourceTrace = entry.sourceRef || (entry.mergedSourceRefs?.length ?? 0) > 0 ? 0.08 : 0;
  const sourceScore =
    entry.source === "selected"
      ? 0.05
      : entry.source === "candidate"
        ? 0.04
        : entry.source === "support_ref"
          ? 0.03
          : 0;
  const resourceScore = hasStructuredResourceSignal(entry) ? 0.08 : 0;
  return clamp01(surfaceScore + sourceTrace + sourceScore + resourceScore);
}

function stringArray(value: unknown): string[] {
  if (!Array.isArray(value)) return [];
  return value.filter((item): item is string => typeof item === "string" && item.trim().length > 0);
}

function requestedAttributeSlotsForQuery(queryAnalysis: QueryCompileResult): string[] {
  const slotHints = (queryAnalysis.evidencePlan?.slots ?? []).flatMap((slot) => [
    ...(slot.requestedAttributeSlots ?? []),
    slot.description,
    ...slot.requiredFields,
    ...(slot.relationHints ?? []),
  ]);
  return [
    ...new Set(
      [
        ...requestedAttributeSlotsFromText(
          queryAnalysis.queryText,
          queryAnalysis.focusedQuery,
          ...slotHints,
        ),
        ...slotHints.map(canonicalAttributeSlot).filter((slot): slot is string => Boolean(slot)),
      ],
    ),
  ];
}

function attributeSlotsForEntry(entry: PromptEvidenceCandidate): string[] {
  const metadata = entry.metadata ?? {};
  const directPredicate =
    typeof metadata.predicate === "string"
      ? metadata.predicate
      : typeof metadata.relationType === "string"
        ? metadata.relationType
        : undefined;
  const relationSlot =
    typeof metadata.relationSlot === "string"
      ? metadata.relationSlot
      : typeof metadata.componentRole === "string"
        ? metadata.componentRole
        : undefined;
  return [
    ...new Set(
      [
        attributeSlotFromPredicate(directPredicate),
        canonicalAttributeSlot(relationSlot),
        ...requestedAttributeSlotsFromText(
          entry.text,
          entry.rawText,
          entry.scoringText,
          directPredicate,
          relationSlot,
          typeof metadata.canonicalObject === "string" ? metadata.canonicalObject : undefined,
        ),
      ].filter((slot): slot is string => Boolean(slot)),
    ),
  ];
}

function entryMatchesRequestedAttributeSlot(
  queryAnalysis: QueryCompileResult,
  entry: PromptEvidenceCandidate,
): boolean {
  const requestedSlots = requestedAttributeSlotsForQuery(queryAnalysis);
  if (requestedSlots.length === 0 || entry.surface !== "fact") {
    return true;
  }
  const predicate = typeof entry.metadata?.predicate === "string" ? entry.metadata.predicate : undefined;
  if (predicate && predicateMatchesAttributeSlots(predicate, requestedSlots)) {
    return true;
  }
  return attributeSlotsForEntry(entry).some((slot) => requestedSlots.includes(slot));
}

function sameSemanticChainSupport(
  entry: PromptEvidenceCandidate,
  contextCandidates: PromptEvidenceCandidate[],
): number {
  const refs = bindingSourceRefsForEntry(entry);
  const families = new Set(refs.map((sourceRef) => sourceFamilyRef(sourceRef)));
  let best = 0;
  for (const candidate of contextCandidates) {
    const candidateRefs = bindingSourceRefsForEntry(candidate);
    if (candidateRefs.length === 0 || refs.length === 0) {
      continue;
    }
    const sharedFamily = candidateRefs.some((sourceRef) =>
      families.has(sourceFamilyRef(sourceRef)),
    );
    if (sharedFamily) {
      best = Math.max(best, 0.74);
    }
    if (sourceRefsAdjacent(refs, candidateRefs, 2)) {
      best = Math.max(best, 0.86);
    }
  }
  return best;
}

function contextBindingScore(params: {
  queryAnalysis: QueryCompileResult;
  entry: PromptEvidenceCandidate;
  slotRole: PromptEvidenceCandidate["slotEvidenceRole"] | undefined;
  contextCandidates: PromptEvidenceCandidate[];
}): number {
  const operation = operationType(params.queryAnalysis);
  if (params.slotRole === "query_context") {
    return 0.22;
  }
  if (params.slotRole === "time_constraint") {
    return 0.18;
  }
  const filledRoles = new Set(
    [params.entry, ...params.contextCandidates]
      .map((candidate) => inferredSlotRole(params.queryAnalysis, candidate))
      .filter(Boolean),
  );
  const entryRoles = new Set([
    inferredSlotRole(params.queryAnalysis, params.entry),
    ...(params.entry.bridgeMatches ?? [])
      .filter((match) => match.score >= 0.44)
      .map((match) => match.role),
  ]);
  const hasContext =
    filledRoles.has("query_context") ||
    entryRoles.has("query_context") ||
    queryContextSupportScore(params.queryAnalysis, params.entry) >= 0.24 ||
    (params.entry.slotCoverage ?? []).some((coverage) => {
      const slot = (params.queryAnalysis.evidencePlan?.slots ?? []).find(
        (candidate) => candidate.id === coverage.slotId,
      );
      return evidenceSlotRequiredRole(slot) === "query_context" && coverage.filled;
    });
  const contextSemanticSupport = Math.max(
    queryContextSupportScore(params.queryAnalysis, params.entry),
    0,
    ...params.contextCandidates.map((candidate) =>
      queryContextSupportScore(params.queryAnalysis, candidate),
    ),
  );
  const samePacketSupport = params.contextCandidates.some(
    (candidate) =>
      (inferredSlotRole(params.queryAnalysis, candidate) === "query_context" ||
        semanticTextSimilarity(candidate.text, params.entry.text) >= 0.34) &&
      queryContextSupportScore(params.queryAnalysis, candidate) >= 0.24,
  );
  const sameChainSupport = sameSemanticChainSupport(params.entry, params.contextCandidates);
  const contextSupport = Math.max(
    contextSemanticSupport >= 0.24 ? sameChainSupport : 0,
    samePacketSupport && params.contextCandidates.length > 0 ? 0.42 : 0,
    contextSemanticSupport,
  );
  if (operation === "tailor_advice" && params.slotRole === "user_resource") {
    return contextSupport >= 0.42 || hasContext ? 0.72 : 0.58;
  }
  if (operation === "tailor_advice" && params.slotRole === "prior_advice") {
    const topicSimilarity = Math.max(
      semanticTextSimilarity(params.queryAnalysis.queryText, params.entry.text),
      semanticTextSimilarity(params.queryAnalysis.focusedQuery ?? "", params.entry.text),
    );
    if (topicSimilarity < 0.2 && contextSupport < 0.42) {
      return 0.24;
    }
    return contextSupport >= 0.42 ? 0.68 : 0.48;
  }
  const lookupMode =
    operation === "return_value" || params.queryAnalysis.answerMode === "attribute_lookup";
  if (
    lookupMode &&
    params.slotRole === "answer_value" &&
    queryAsksSensitiveValue(params.queryAnalysis) &&
    statesUnavailableSensitiveValue(params.queryAnalysis, params.entry)
  ) {
    return Math.max(contextSupport, 0.68);
  }
  if (lookupMode) {
    const sameEntryContextAndAnswer =
      (entryRoles.has("answer_value") || entryRoles.has("answer_event")) &&
      entryRoles.has("query_context");
    if (sameEntryContextAndAnswer) {
      return 0.86;
    }
    if (hasContext && contextSupport >= 0.74) {
      return 0.86;
    }
    if (hasContext && contextSupport >= 0.42) {
      return 0.68;
    }
    if (hasContext) {
      return contextSemanticSupport >= 0.24 ? 0.46 : 0.28;
    }
    return contextSemanticSupport >= 0.24 ? 0.28 : 0.12;
  }
  if (hasContext && contextSupport >= 0.74) {
    return 0.82;
  }
  if (hasContext && contextSupport >= 0.42) {
    return 0.62;
  }
  if (hasContext) {
    return contextSemanticSupport >= 0.24 ? 0.62 : 0.36;
  }
  return 0.42;
}

function answerPlanSlots(queryAnalysis: QueryCompileResult): QueryEvidenceSlot[] {
  const operation = operationType(queryAnalysis);
  const slots = queryAnalysis.evidencePlan?.slots ?? [];
  const answerSlots = slots.filter((slot) => {
    const role = evidenceSlotRequiredRole(slot);
    if (role === "answer_value" || role === "answer_event") {
      return true;
    }
    if (operation === "tailor_advice" && (role === "user_resource" || role === "prior_advice")) {
      return true;
    }
    return slot.role === "answer_evidence";
  });
  if (answerSlots.length > 0) {
    return answerSlots;
  }
  return slots.filter((slot) => {
    const role = evidenceSlotRequiredRole(slot);
    return role !== "query_context" && role !== "time_constraint";
  });
}

function answerRelationFitScore(
  queryAnalysis: QueryCompileResult,
  entry: PromptEvidenceCandidate,
): number {
  const relationTexts = [
    ...answerPlanSlots(queryAnalysis).flatMap((slot) => [
      ...(slot.relationHints ?? []),
      ...(slot.capabilityQueries ?? []),
    ]),
    ...(queryAnalysis.semanticBridges ?? [])
      .filter((bridge) => bridge.role === "answer_value" || bridge.role === "answer_event")
      .flatMap((bridge) => bridge.positiveSignals ?? []),
  ]
    .map((text) => text.trim())
    .filter(Boolean);
  if (relationTexts.length === 0) {
    return 0.5;
  }
  const candidateText = `${entry.text} ${entry.scoringText ?? ""}`;
  return Math.max(0, ...relationTexts.map((text) => planTextMatchScore(text, candidateText)));
}

function hasStrongAnswerPlanningMatch(
  queryAnalysis: QueryCompileResult,
  entry: PromptEvidenceCandidate,
): boolean {
  return (
    slotCoverageScore(entry) >= 0.62 ||
    bridgeAnswerRoleScore(entry) >= 0.5 ||
    bridgePositiveRoleScore(entry, ["answer_value", "answer_event"]) >= 0.5 ||
    evidenceGoalDomainScore(queryAnalysis, entry) >= 0.48
  );
}

function querySeeksCausalAnswer(queryAnalysis: QueryCompileResult): boolean {
  return queryUsesEvidenceShapes(queryAnalysis, ["causal_explanation"]);
}

function causalExplanationScore(
  queryAnalysis: QueryCompileResult,
  entry: PromptEvidenceCandidate,
): number {
  return evidenceShapeFitScore(queryAnalysis, entry, ["causal_explanation"]);
}

function directCausalExplanationScore(
  queryAnalysis: QueryCompileResult,
  entry: PromptEvidenceCandidate,
): number {
  return evidenceShapeFitScore(queryAnalysis, entry, ["causal_explanation"]);
}

function validationEvidenceScore(
  queryAnalysis: QueryCompileResult,
  entry: PromptEvidenceCandidate,
): number {
  return evidenceShapeFitScore(queryAnalysis, entry, ["validation_evidence"]);
}

function statusSummaryEvidenceScore(
  queryAnalysis: QueryCompileResult,
  entry: PromptEvidenceCandidate,
): number {
  return evidenceShapeFitScore(queryAnalysis, entry, ["status_answer"]);
}

function queryAsksCurrentStatus(queryAnalysis: QueryCompileResult): boolean {
  return queryUsesEvidenceShapes(queryAnalysis, ["status_answer"]);
}

function currentStatusAnswerScore(
  queryAnalysis: QueryCompileResult,
  entry: PromptEvidenceCandidate,
): number {
  return evidenceShapeFitScore(queryAnalysis, entry, ["status_answer"]);
}

function staleOnlyStatusEvidence(
  queryAnalysis: QueryCompileResult,
  entry: PromptEvidenceCandidate,
): boolean {
  return (
    queryUsesEvidenceShapes(queryAnalysis, ["status_answer"]) &&
    evidenceShapeFitScore(queryAnalysis, entry, ["time_constraint"]) >
      currentStatusAnswerScore(queryAnalysis, entry) + 0.24
  );
}

function queryAsksBooleanDecision(queryAnalysis: QueryCompileResult): boolean {
  return queryUsesEvidenceShapes(queryAnalysis, ["decision_value"]);
}

function booleanDecisionFitScore(
  queryAnalysis: QueryCompileResult,
  entry: PromptEvidenceCandidate,
): number {
  if (!queryAsksBooleanDecision(queryAnalysis)) {
    return 0.5;
  }
  const relationFit = answerRelationFitScore(queryAnalysis, entry);
  const decisionFit = evidenceShapeFitScore(queryAnalysis, entry, ["decision_value"]);
  if (decisionFit >= 0.34) {
    return Math.max(decisionFit, relationFit);
  }
  return Math.min(relationFit, 0.32);
}

function queryAsksSensitiveValue(queryAnalysis: QueryCompileResult): boolean {
  return queryUsesEvidenceShapes(queryAnalysis, ["availability_statement"]);
}

function statesUnavailableSensitiveValue(
  queryAnalysis: QueryCompileResult,
  entry: PromptEvidenceCandidate,
): boolean {
  return evidenceShapeFitScore(queryAnalysis, entry, ["availability_statement"]) >= 0.42;
}

function slotPlanText(slot: QueryEvidenceSlot): string {
  const fields = slot.requiredFields.filter((field) => {
    const normalized = normalizeText(field).replace(/\s+/gu, "_");
    return !GENERIC_REQUIRED_FIELDS.has(normalized) && !TEMPORAL_REQUIRED_FIELDS.has(normalized);
  });
  return [
    slot.description,
    ...slot.subjectHints,
    ...(slot.relationHints ?? []),
    ...(slot.capabilityQueries ?? []),
    ...fields,
  ]
    .map((entry) => entry.trim())
    .filter(Boolean)
    .join(" ");
}

function queryContextPlanTexts(queryAnalysis: QueryCompileResult): string[] {
  return (queryAnalysis.evidencePlan?.slots ?? [])
    .filter((slot) => evidenceSlotRequiredRole(slot) === "query_context")
    .map((slot) => slotPlanText(slot))
    .filter(Boolean);
}

function planTextMatchScore(planText: string, candidateText: string): number {
  const semantic = semanticTextSimilarity(planText, candidateText);
  const planTokens = normalizeText(planText)
    .split(/\s+/gu)
    .filter((token) => token.length >= 4);
  if (planTokens.length === 0) {
    return semantic;
  }
  const normalizedCandidate = normalizeText(candidateText);
  const matched = planTokens.filter((token) => normalizedCandidate.includes(token)).length;
  const overlap = matched / Math.min(planTokens.length, 6);
  return Math.max(semantic, clamp01(overlap * 0.68));
}

function queryContextSupportScore(
  queryAnalysis: QueryCompileResult,
  entry: PromptEvidenceCandidate,
): number {
  const planTexts = queryContextPlanTexts(queryAnalysis);
  if (planTexts.length === 0) {
    return 0;
  }
  const candidateText = `${entry.text} ${entry.scoringText ?? ""}`;
  const planSemantic = Math.max(
    0,
    ...planTexts.map((text) => planTextMatchScore(text, candidateText)),
  );
  const contextCoverage = Math.max(
    0,
    ...(entry.slotCoverage ?? [])
      .filter((coverage) => {
        const slot = (queryAnalysis.evidencePlan?.slots ?? []).find(
          (candidate) => candidate.id === coverage.slotId,
        );
        return evidenceSlotRequiredRole(slot) === "query_context";
      })
      .map((coverage) => coverage.coverageScore * (coverage.missingRequired.length > 0 ? 0.45 : 1)),
  );
  const bridgeContext = bridgeRoleScore(entry, ["query_context"]);
  const bridgePositive = bridgePositiveRoleScore(entry, ["query_context"]);
  return clamp01(
    planSemantic * 0.5 + contextCoverage * 0.28 + bridgeContext * 0.14 + bridgePositive * 0.08,
  );
}

function evidenceGoalDomainScore(
  queryAnalysis: QueryCompileResult,
  entry: PromptEvidenceCandidate,
): number {
  const goalTexts = (queryAnalysis.evidenceGoals ?? [])
    .flatMap((goal) => [goal.goal, ...goal.positiveQueries, ...goal.focusAnchors])
    .map((text) => text.trim())
    .filter(Boolean);
  if (goalTexts.length === 0) {
    return 0;
  }
  const candidateText = `${entry.text} ${entry.scoringText ?? ""}`;
  return Math.max(0, ...goalTexts.map((text) => planTextMatchScore(text, candidateText)));
}

function contrastNegativeHints(queryAnalysis: QueryCompileResult): string[] {
  const rawHints = [
    ...(queryAnalysis.evidenceGoals ?? []).flatMap((goal) => goal.negativeHints ?? []),
    ...(queryAnalysis.evidencePlan?.slots ?? []).flatMap((slot) => slot.negativeHints ?? []),
    ...(queryAnalysis.semanticBridges ?? []).flatMap((bridge) => bridge.negativeSignals ?? []),
  ]
    .map((hint) => hint.trim())
    .filter(Boolean);
  const queryText = queryAnalysis.queryText || queryAnalysis.focusedQuery || "";
  return [
    ...new Set(
      rawHints.filter((hint) => {
        const normalized = normalizeText(hint);
        if (!normalized) {
          return false;
        }
        // Hints that are effectively restating the question are often polarity terms
        // for the answer itself. Keep only contrast concepts that are not the query.
        return semanticTextSimilarity(hint, queryText) < 0.42;
      }),
    ),
  ];
}

function negativeContrastScore(
  queryAnalysis: QueryCompileResult,
  entry: PromptEvidenceCandidate,
): number {
  const hints = contrastNegativeHints(queryAnalysis);
  if (hints.length === 0) {
    return 0;
  }
  const candidateText = `${entry.text} ${entry.scoringText ?? ""}`;
  const normalizedCandidate = normalizeText(candidateText);
  const overlapScore = (hint: string): number => {
    const normalizedHint = normalizeText(hint);
    if (!normalizedHint) {
      return 0;
    }
    if (normalizedCandidate.includes(normalizedHint)) {
      return 0.72;
    }
    const hintTokens = normalizedHint.split(/\s+/gu).filter((token) => token.length >= 4);
    if (hintTokens.length === 0) {
      return 0;
    }
    const matched = hintTokens.filter((token) => normalizedCandidate.includes(token)).length;
    return (matched / hintTokens.length) * 0.42;
  };
  return Math.max(
    0,
    ...hints.map((hint) => Math.max(semanticTextSimilarity(hint, candidateText), overlapScore(hint))),
  );
}

function tailorAdviceNeedFit(
  queryAnalysis: QueryCompileResult,
  entry: PromptEvidenceCandidate,
): number {
  const texts = [
    queryAnalysis.queryText,
    queryAnalysis.focusedQuery,
    ...(queryAnalysis.evidenceGoals ?? []).flatMap((goal) => [
      goal.goal,
      ...goal.positiveQueries,
      ...goal.focusAnchors,
    ]),
    ...queryContextPlanTexts(queryAnalysis),
  ]
    .map((text) => text?.trim())
    .filter((text): text is string => Boolean(text));
  if (texts.length === 0) {
    return 0;
  }
  const candidateText = `${entry.text} ${entry.scoringText ?? ""}`;
  return Math.max(0, ...texts.map((text) => semanticTextSimilarity(text, candidateText)));
}

function answerDomainScore(
  queryAnalysis: QueryCompileResult,
  entry: PromptEvidenceCandidate,
): number {
  const candidateText = `${entry.text} ${entry.scoringText ?? ""}`;
  const slotTexts = answerPlanSlots(queryAnalysis)
    .map((slot) => slotPlanText(slot))
    .filter(Boolean);
  const slotSemantic = Math.max(
    0,
    ...slotTexts.map((text) => planTextMatchScore(text, candidateText)),
  );
  const goalSemantic = evidenceGoalDomainScore(queryAnalysis, entry);
  const answerCoverage = Math.max(
    0,
    ...(entry.slotCoverage ?? [])
      .filter((coverage) => {
        const slot = (queryAnalysis.evidencePlan?.slots ?? []).find(
          (candidate) => candidate.id === coverage.slotId,
        );
        const role = evidenceSlotRequiredRole(slot);
        return (
          role === "answer_value" ||
          role === "answer_event" ||
          (operationType(queryAnalysis) === "tailor_advice" &&
            (role === "user_resource" || role === "prior_advice"))
        );
      })
      .map((coverage) => coverage.coverageScore * (coverage.missingRequired.length > 0 ? 0.45 : 1)),
  );
  const bridgeAnswer = bridgeAnswerRoleScore(entry);
  const bridgePositive = bridgePositiveRoleScore(entry, [
    "answer_value",
    "answer_event",
    "user_resource",
    "prior_advice",
  ]);
  if (operationType(queryAnalysis) === "tailor_advice") {
    const needFit = tailorAdviceNeedFit(queryAnalysis, entry);
    const blended =
      answerCoverage * 0.22 +
      bridgeAnswer * 0.12 +
      bridgePositive * 0.18 +
      slotSemantic * 0.12 +
      needFit * 0.28 +
      goalSemantic * 0.08;
    const corroborationBoost = Math.min(
      0.14,
      [
        answerCoverage >= 0.28,
        bridgePositive >= 0.28,
        slotSemantic >= 0.28,
        needFit >= 0.28,
      ].filter(Boolean).length * 0.035,
    );
    return clamp01(blended + corroborationBoost);
  }
  const positiveSourceCount = [
    answerCoverage >= 0.28,
    bridgeAnswer >= 0.28,
    bridgePositive >= 0.28,
    slotSemantic >= 0.28,
    goalSemantic >= 0.28,
  ].filter(Boolean).length;
  const blended =
    answerCoverage * 0.36 +
    bridgeAnswer * 0.22 +
    bridgePositive * 0.14 +
    slotSemantic * 0.1 +
    goalSemantic * 0.18;
  const corroborationBoost = Math.min(0.16, Math.max(0, positiveSourceCount - 1) * 0.05);
  return clamp01(blended + corroborationBoost);
}

function hasTemporalPlanning(queryAnalysis: QueryCompileResult): boolean {
  return (
    queryAnalysis.queryShape.timeframe === "historical" ||
    queryAnalysis.answerMode === "count_aggregate" ||
    operationType(queryAnalysis) === "aggregate" ||
    (queryAnalysis.evidencePlan?.slots ?? []).some(
      (slot) =>
        evidenceSlotRequiredRole(slot) === "time_constraint" ||
        slotNeedsTemporalField(queryAnalysis, slot),
    )
  );
}

function temporalFitScore(params: {
  queryAnalysis: QueryCompileResult;
  entry: PromptEvidenceCandidate;
  slotRole: PromptEvidenceCandidate["slotEvidenceRole"] | undefined;
  contextCandidates: PromptEvidenceCandidate[];
}): number {
  if (!hasTemporalPlanning(params.queryAnalysis)) {
    return 0.62;
  }
  if (
    params.entry.observedAt ||
    params.contextCandidates.some((entry) => Boolean(entry.observedAt))
  ) {
    return 0.78;
  }
  const timeBridgeScore = Math.max(
    bridgeRoleScore(params.entry, ["time_constraint"]),
    bridgeShapeScore(params.entry, ["time_constraint"]),
    ...params.contextCandidates.map((entry) =>
      Math.max(bridgeRoleScore(entry, ["time_constraint"]), bridgeShapeScore(entry, ["time_constraint"])),
    ),
  );
  if (timeBridgeScore >= 0.32) {
    return 0.62;
  }
  if (params.slotRole === "time_constraint") {
    return 0.54;
  }
  return 0.42;
}

function hardExclusionReasons(params: {
  queryAnalysis: QueryCompileResult;
  entry: PromptEvidenceCandidate;
  answer: number;
  now: string;
}): string[] {
  const text = `${params.entry.text} ${params.entry.scoringText ?? ""}`.trim();
  const sourceRefs = sourceRefsForEntry(params.entry);
  const blockers: string[] = [];
  if (!normalizeText(text)) {
    blockers.push("empty-evidence");
  }
  if (HARD_EXCLUSION_PATTERNS.some((pattern) => pattern.test(text))) {
    blockers.push("bootstrap-or-debug-memory");
  }
  if (
    sourceRefs.length === 0 &&
    params.entry.surface !== "chunk" &&
    params.answer >= 0.42 &&
    params.entry.source === "projected"
  ) {
    blockers.push("untraceable-summary-answer");
  }
  if (sourceRefs.length === 0) {
    blockers.push("untraceable-answer-evidence");
  }
  if (
    sourceRefs.length === 0 &&
    queryEchoScore(params.queryAnalysis, params.entry.text) >= 0.9 &&
    params.answer < 0.5
  ) {
    blockers.push("query-echo-without-history-source");
  }
  if (looksLikeBareMemoryUseInstruction(text)) {
    blockers.push("memory-use-instruction-not-answer");
  }
  if (isAssistantAcknowledgement(params.entry)) {
    blockers.push("assistant-acknowledgement-not-evidence");
  }
  if (
    entryAuthorRole(params.entry) === "assistant" &&
    !queryAllowsAssistantAuthoredAnswer(params.queryAnalysis)
  ) {
    const role = inferredSlotRole(params.queryAnalysis, params.entry);
    if (role === "answer_value" || role === "answer_event" || params.answer >= 0.42) {
      blockers.push("assistant-authored-answer-not-authoritative");
    }
  }
  if (!entryMatchesRequestedAttributeSlot(params.queryAnalysis, params.entry)) {
    blockers.push("attribute-slot-mismatch");
  }
  blockers.push(...stateHardExclusions(params.entry, params.now));
  return [...new Set(blockers)];
}

function softPenaltyReasons(params: {
  queryAnalysis: QueryCompileResult;
  entry: PromptEvidenceCandidate;
  slotRole: PromptEvidenceCandidate["slotEvidenceRole"] | undefined;
  contextCandidates: PromptEvidenceCandidate[];
  answer: number;
  answerDomain: number;
  contextBinding: number;
  temporalFit: number;
  remainingMissing: string[];
}): Array<{ reason: string; weight: number }> {
  const penalties: Array<{ reason: string; weight: number }> = [];
  const operation = operationType(params.queryAnalysis);
  const author = entryAuthorRole(params.entry);
  const filledRoles = new Set(
    [params.entry, ...params.contextCandidates]
      .map((candidate) => inferredSlotRole(params.queryAnalysis, candidate))
      .filter(Boolean),
  );
  const hasAnswerRole =
    params.slotRole === "answer_value" ||
    params.slotRole === "answer_event" ||
    params.slotRole === "user_resource" ||
    params.slotRole === "prior_advice" ||
    filledRoles.has("answer_value") ||
    filledRoles.has("answer_event") ||
    (operation === "tailor_advice" &&
      (filledRoles.has("user_resource") || filledRoles.has("prior_advice")));
  if (params.remainingMissing.length > 0) {
    penalties.push({
      reason: `missing-context:${params.remainingMissing.join("|")}`,
      weight: Math.min(0.34, params.remainingMissing.length * 0.09),
    });
  }
  if (
    operation === "return_value" &&
    queryContextPlanTexts(params.queryAnalysis).length > 0 &&
    !(queryAsksSensitiveValue(params.queryAnalysis) &&
      statesUnavailableSensitiveValue(params.queryAnalysis, params.entry)) &&
    Math.max(
      queryContextSupportScore(params.queryAnalysis, params.entry),
      0,
      ...params.contextCandidates.map((candidate) =>
        queryContextSupportScore(params.queryAnalysis, candidate),
      ),
    ) < 0.24
  ) {
    penalties.push({ reason: "weak-query-context-binding", weight: 0.16 });
  }
  if (params.slotRole === "query_context") {
    penalties.push({ reason: "query-context-only", weight: 0.12 });
  }
  if (params.slotRole === "time_constraint") {
    penalties.push({ reason: "time-constraint-only", weight: 0.11 });
  }
  if (
    params.entry.surface === "chunk" &&
    (author === "user" || entryIsQueryLikeEvidence(params.queryAnalysis, params.entry)) &&
    queryEchoScore(params.queryAnalysis, params.entry.text) >= 0.82 &&
    params.answer < 0.7
  ) {
    penalties.push({ reason: "query-echo-like", weight: 0.36 });
  } else if (entryIsQueryLikeEvidence(params.queryAnalysis, params.entry)) {
    penalties.push({ reason: "query-like-answer-candidate", weight: 0.28 });
  }
  if (
    hasTemporalPlanning(params.queryAnalysis) &&
    params.temporalFit < 0.58 &&
    params.contextCandidates.length === 0
  ) {
    penalties.push({ reason: "weak-temporal-fit", weight: 0.08 });
  }
  if (
    params.queryAnalysis.queryShape.timeframe === "historical" &&
    filledRoles.has("time_constraint") &&
    !hasAnswerRole &&
    params.answerDomain < 0.42
  ) {
    penalties.push({ reason: "temporal-without-answer-domain", weight: 0.22 });
  }
  if (
    params.queryAnalysis.queryShape.timeframe === "historical" &&
    params.slotRole !== "time_constraint" &&
    params.answerDomain < 0.5
  ) {
    penalties.push({ reason: "weak-historical-answer-domain", weight: 0.14 });
  }
  if (
    (params.queryAnalysis.answerMode === "count_aggregate" || operation === "aggregate") &&
    params.answerDomain < 0.46
  ) {
    penalties.push({ reason: "aggregate-without-answer-event", weight: 0.16 });
  }
  if (
    (params.queryAnalysis.answerMode === "count_aggregate" || operation === "aggregate") &&
    params.slotRole === "answer_event" &&
    params.answerDomain < 0.56
  ) {
    penalties.push({ reason: "weak-aggregate-event-domain", weight: 0.22 });
  }
  if (
    (operation === "return_value" || params.queryAnalysis.answerMode === "attribute_lookup") &&
    (params.slotRole === "answer_value" || filledRoles.has("answer_value")) &&
    !(queryAsksSensitiveValue(params.queryAnalysis) &&
      statesUnavailableSensitiveValue(params.queryAnalysis, params.entry)) &&
    params.contextBinding < 0.46
  ) {
    penalties.push({ reason: "answer-without-bound-context", weight: 0.18 });
  }
  if (
    operation === "return_value" &&
    (params.slotRole === "answer_value" || filledRoles.has("answer_value")) &&
    answerRelationFitScore(params.queryAnalysis, params.entry) < 0.28 &&
    !hasStrongAnswerPlanningMatch(params.queryAnalysis, params.entry)
  ) {
    penalties.push({ reason: "weak-answer-relation-fit", weight: 0.28 });
  }
  if (
    querySeeksCausalAnswer(params.queryAnalysis) &&
    (params.slotRole === "answer_value" || filledRoles.has("answer_value")) &&
    causalExplanationScore(params.queryAnalysis, params.entry) < 0.32
  ) {
    penalties.push({ reason: "weak-causal-explanation", weight: 0.34 });
  }
  if (
    querySeeksCausalAnswer(params.queryAnalysis) &&
    (params.slotRole === "answer_value" || filledRoles.has("answer_value")) &&
    validationEvidenceScore(params.queryAnalysis, params.entry) >= 0.72 &&
    directCausalExplanationScore(params.queryAnalysis, params.entry) < 0.5
  ) {
    penalties.push({ reason: "validation-evidence-not-cause", weight: 0.34 });
  }
  if (
    querySeeksCausalAnswer(params.queryAnalysis) &&
    (params.slotRole === "answer_value" || filledRoles.has("answer_value")) &&
    statusSummaryEvidenceScore(params.queryAnalysis, params.entry) >= 0.7 &&
    directCausalExplanationScore(params.queryAnalysis, params.entry) < 0.5
  ) {
    penalties.push({ reason: "status-summary-not-cause", weight: 0.3 });
  }
  if (
    queryAsksCurrentStatus(params.queryAnalysis) &&
    (params.slotRole === "answer_value" || filledRoles.has("answer_value")) &&
    currentStatusAnswerScore(params.queryAnalysis, params.entry) < 0.32
  ) {
    penalties.push({ reason: "weak-current-status-answer", weight: 0.3 });
  }
  if (
    queryAsksCurrentStatus(params.queryAnalysis) &&
    (params.slotRole === "answer_value" || filledRoles.has("answer_value")) &&
    staleOnlyStatusEvidence(params.queryAnalysis, params.entry)
  ) {
    penalties.push({ reason: "stale-status-only", weight: 0.42 });
  }
  if (
    queryAsksBooleanDecision(params.queryAnalysis) &&
    (params.slotRole === "answer_value" || filledRoles.has("answer_value")) &&
    booleanDecisionFitScore(params.queryAnalysis, params.entry) < 0.34
  ) {
    penalties.push({ reason: "weak-boolean-decision-fit", weight: 0.4 });
  }
  if (
    queryAsksSensitiveValue(params.queryAnalysis) &&
    (params.slotRole === "answer_value" || filledRoles.has("answer_value")) &&
    !statesUnavailableSensitiveValue(params.queryAnalysis, params.entry)
  ) {
    penalties.push({ reason: "sensitive-value-without-availability-evidence", weight: 0.46 });
  }
  const contrastScore = Math.max(
    negativeContrastScore(params.queryAnalysis, params.entry),
    0,
    ...params.contextCandidates.map((candidate) =>
      negativeContrastScore(params.queryAnalysis, candidate),
    ),
  );
  const contrastContext = Math.max(
    queryContextSupportScore(params.queryAnalysis, params.entry),
    0,
    ...params.contextCandidates.map((candidate) =>
      queryContextSupportScore(params.queryAnalysis, candidate),
    ),
  );
  if (
    operation === "return_value" &&
    contrastScore >= 0.28 &&
    contrastContext < 0.52
  ) {
    penalties.push({ reason: "negative-contrast-without-query-context", weight: 0.42 });
  }
  if (
    operation !== "tailor_advice" &&
    params.slotRole === "user_resource" &&
    params.answerDomain < 0.42
  ) {
    penalties.push({ reason: "resource-outside-advice-domain", weight: 0.22 });
  }
  if (operation === "tailor_advice" && params.slotRole === "user_resource") {
    const resourceRelevance = Math.max(
      tailorAdviceNeedFit(params.queryAnalysis, params.entry),
      semanticTextSimilarity(params.queryAnalysis.queryText, params.entry.text),
      semanticTextSimilarity(params.queryAnalysis.focusedQuery ?? "", params.entry.text),
      bridgePositiveRoleScore(params.entry, ["user_resource"]),
      numericBreakdown(params.entry, "capabilitySupport"),
    );
    if (resourceRelevance < 0.24 && !hasResourceAffordanceSignal(params.entry)) {
      penalties.push({ reason: "weak-user-resource-capability-match", weight: 0.3 });
    } else if (resourceRelevance < 0.24) {
      penalties.push({ reason: "weak-user-resource-capability-match", weight: 0.16 });
    }
  }
  if (sourceRefsForEntry(params.entry).length === 0) {
    penalties.push({ reason: "missing-source-ref", weight: 0.1 });
  }
  for (const penalty of stateSoftPenalties(params.entry)) {
    penalties.push(penalty);
  }
  if (isAssistantAcknowledgement(params.entry)) {
    penalties.push({ reason: "assistant-acknowledgement", weight: 0.36 });
  }
  if (
    operation === "tailor_advice" &&
    params.slotRole === "prior_advice" &&
    params.answerDomain < 0.42
  ) {
    penalties.push({ reason: "prior-advice-domain-mismatch", weight: 0.2 });
  }
  if (!hasAnswerRole && params.answer < 0.42 && params.answerDomain < 0.34) {
    penalties.push({ reason: "weak-answer-role", weight: 0.1 });
  }
  return penalties;
}

function answerScore(params: {
  queryAnalysis: QueryCompileResult;
  entry: PromptEvidenceCandidate;
  slotRole: PromptEvidenceCandidate["slotEvidenceRole"] | undefined;
}): number {
  const semantic = clamp01(Math.max(params.entry.semanticScore ?? 0, params.entry.goalScore));
  const coverage = slotCoverageScore(params.entry);
  const capability = numericBreakdown(params.entry, "capabilitySupport");
  const domain = answerDomainScore(params.queryAnalysis, params.entry);
  const bridgeSupport = bridgeSupportScore(params.entry);
  const bridgeAnswer = bridgeAnswerRoleScore(params.entry);
  const retrieval = candidateRetrievalScore(params.entry);
  const causalFit = querySeeksCausalAnswer(params.queryAnalysis)
    ? Math.max(
        directCausalExplanationScore(params.queryAnalysis, params.entry),
        causalExplanationScore(params.queryAnalysis, params.entry) * 0.55,
      )
    : 0;
  if (params.slotRole === "query_context") {
    return clamp01(domain * 0.22 + semantic * 0.08 + retrieval * 0.04);
  }
  if (params.slotRole === "user_resource") {
    const operation = operationType(params.queryAnalysis);
    const roleFloor = operation === "tailor_advice" ? 0.18 : 0.04;
    return clamp01(
      domain * 0.34 +
        bridgeSupport * 0.2 +
        semantic * 0.16 +
        coverage * 0.12 +
        capability * 0.12 +
        (hasResourceAffordanceSignal(params.entry) ? 0.06 : roleFloor),
    );
  }
  if (params.slotRole === "answer_value" || params.slotRole === "answer_event") {
    const base = clamp01(
      domain * (causalFit > 0 ? 0.34 : 0.42) +
        bridgeAnswer * 0.18 +
        semantic * 0.12 +
        coverage * 0.12 +
        causalFit * 0.18 +
        params.entry.priority * 0.06 +
        retrieval * 0.04,
    );
    return queryAsksSensitiveValue(params.queryAnalysis) &&
      statesUnavailableSensitiveValue(params.queryAnalysis, params.entry)
      ? Math.max(base, 0.84)
      : base;
  }
  if (params.slotRole === "time_constraint") {
    return clamp01(domain * 0.18 + semantic * 0.06);
  }
  if (params.slotRole === "prior_advice") {
    return clamp01(
      domain * 0.34 +
        bridgeSupport * 0.2 +
        semantic * 0.22 +
        coverage * 0.1 +
        params.entry.priority * 0.06,
    );
  }
  const evidenceCoverageScore =
    params.entry.coverage && params.entry.coverage.requiredHits.length > 0
      ? Math.max(0.5, params.entry.coverage.coverageScore)
      : 0;
  const surfaceAnswer =
    params.entry.surface === "fact" || params.entry.surface === "event"
      ? 0.46
      : params.entry.surface === "chunk"
        ? 0.34
        : 0.28;
  return clamp01(
    surfaceAnswer * 0.2 +
      semantic * 0.18 +
      domain * 0.26 +
      bridgeAnswer * 0.14 +
      causalFit * 0.12 +
      coverage * 0.12 +
      evidenceCoverageScore * 0.06 +
      params.entry.priority * 0.04,
  );
}

function gradeCandidate(params: {
  queryAnalysis: QueryCompileResult;
  entry: PromptEvidenceCandidate;
  contextCandidates: PromptEvidenceCandidate[];
  now: string;
}): {
  eligibility: EvidenceEligibility;
  grade: EvidenceGrade;
  blockers: string[];
  softPenalties: string[];
} {
  const slotRole = inferredSlotRole(params.queryAnalysis, params.entry);
  const retrievalScore = candidateRetrievalScore(params.entry);
  const answer = answerScore({ ...params, slotRole });
  const contextBinding = contextBindingScore({ ...params, slotRole });
  const slotScore = slotCoverageScore(params.entry);
  const authority = authorityScore(params.entry);
  const bridgeNegative = bridgeNegativeSignalScore(params.entry);
  const operation = operationType(params.queryAnalysis);
  const remainingMissing = missingRequiredAfterContext(params.entry, params.contextCandidates);
  const filledRoles = new Set(
    [params.entry, ...params.contextCandidates]
      .map((candidate) => inferredSlotRole(params.queryAnalysis, candidate))
      .filter(Boolean),
  );
  const hasAnswerRole =
    slotRole === "answer_value" ||
    slotRole === "answer_event" ||
    (operation === "tailor_advice" &&
      (slotRole === "user_resource" || slotRole === "prior_advice")) ||
    filledRoles.has("answer_value") ||
    filledRoles.has("answer_event") ||
    (operation === "tailor_advice" &&
      (filledRoles.has("user_resource") || filledRoles.has("prior_advice")));
  const answerDomain = answerDomainScore(params.queryAnalysis, params.entry);
  const temporalFit = temporalFitScore({ ...params, slotRole });
  const blockers = hardExclusionReasons({
    queryAnalysis: params.queryAnalysis,
    entry: params.entry,
    answer,
    now: params.now,
  });
  const softPenaltyEntries = softPenaltyReasons({
    queryAnalysis: params.queryAnalysis,
    entry: params.entry,
    slotRole,
    contextCandidates: params.contextCandidates,
    answer,
    answerDomain,
    contextBinding,
    temporalFit,
    remainingMissing,
  });
  const softPenaltyScore = clamp01(
    Math.min(
      0.48,
      softPenaltyEntries.reduce((sum, penalty) => sum + penalty.weight, 0),
    ),
  );

  const eligibilityRole: EvidenceEligibility["role"] =
    slotRole === "user_resource"
      ? "resource"
      : slotRole === "query_context" || slotRole === "time_constraint"
        ? "context"
        : hasAnswerRole || answer >= 0.42
          ? "answer"
          : "support";
  const eligible = blockers.length === 0;
  const sourceTrace = sourceRefsForEntry(params.entry).length > 0 ? 1 : 0;
  const roleFit =
    slotRole === "answer_value" || slotRole === "answer_event"
      ? 1
      : operation === "tailor_advice" &&
          (slotRole === "user_resource" || slotRole === "prior_advice")
        ? 0.95
        : slotRole === "query_context" || slotRole === "time_constraint"
          ? 0.32
          : answer >= 0.42
            ? 0.62
            : 0.4;
  const semanticScore = Math.max(params.entry.semanticScore ?? 0, params.entry.goalScore);
  const contextMultiplier =
    operation === "return_value" || params.queryAnalysis.answerMode === "attribute_lookup"
      ? 0.45 + 0.55 * contextBinding
      : 1;
  const baseScore =
    operation === "aggregate" || params.queryAnalysis.answerMode === "count_aggregate"
      ? answerDomain * 0.45 +
        retrievalScore * 0.2 +
        sourceTrace * 0.15 +
        temporalFit * 0.1 +
        authority * 0.1 +
        roleFit * 0.08
      : operation === "tailor_advice"
        ? answer * 0.35 +
          semanticScore * 0.25 +
          sourceTrace * 0.15 +
          authority * 0.1 +
          retrievalScore * 0.15 +
          roleFit * 0.08
        : hasTemporalPlanning(params.queryAnalysis) &&
            params.queryAnalysis.queryShape.timeframe === "historical"
          ? answerDomain * 0.45 +
            temporalFit * 0.2 +
            retrievalScore * 0.15 +
            sourceTrace * 0.1 +
            authority * 0.1 +
            roleFit * 0.08
          : (answer * 0.36 +
              answerDomain * 0.22 +
              retrievalScore * 0.16 +
              authority * 0.1 +
              sourceTrace * 0.08 +
              slotScore * 0.08 +
              roleFit * 0.08) *
            contextMultiplier;
  const finalScore = clamp01(
    baseScore - softPenaltyScore - bridgeNegative * 0.06 - Math.min(0.5, blockers.length * 0.28),
  );
  return {
    blockers,
    softPenalties: softPenaltyEntries.map((penalty) => penalty.reason),
    eligibility: {
      eligible,
      role: eligibilityRole,
      blockers,
    },
    grade: {
      retrievalScore,
      answerScore: answer,
      contextBindingScore: contextBinding,
      temporalFitScore: temporalFit,
      slotCoverageScore: slotScore,
      authorityScore: authority,
      softPenaltyScore,
      finalScore,
    },
  };
}

function groupEntries(entries: PromptEvidenceCandidate[]): Map<string, PromptEvidenceCandidate[]> {
  const groups = new Map<string, PromptEvidenceCandidate[]>();
  for (const entry of entries) {
    if (entry.dropReason) {
      continue;
    }
    const sourceRefs = bindingSourceRefsForEntry(entry);
    const key =
      sourceRefs.length > 0
        ? sourceKey(sourceRefs, entry.text)
        : `text:${stableHash([normalizeText(entry.text).slice(0, 240)])}`;
    const group = groups.get(key) ?? [];
    group.push(entry);
    groups.set(key, group);
  }
  return groups;
}

function contextCandidatesForGroup(params: {
  queryAnalysis: QueryCompileResult;
  entries: PromptEvidenceCandidate[];
  allEntries: PromptEvidenceCandidate[];
  answerEntry: PromptEvidenceCandidate;
}): PromptEvidenceCandidate[] {
  const bindingAnswerRefs = bindingSourceRefsForEntry(params.answerEntry);
  const operation = operationType(params.queryAnalysis);
  const answerRole = inferredSlotRole(params.queryAnalysis, params.answerEntry);
  const userQuestionBridge =
    entryAuthorRole(params.answerEntry) === "user" &&
    (entryIsQuestionLike(params.answerEntry) ||
      entryIsQueryLikeEvidence(params.queryAnalysis, params.answerEntry));
  const assistantAnswerBridge =
    entryAuthorRole(params.answerEntry) === "assistant" &&
    operation !== "aggregate" &&
    params.queryAnalysis.answerMode !== "count_aggregate" &&
    (answerRole === "answer_value" ||
      answerRole === "answer_event" ||
      params.answerEntry.metadata?.sourceExpansion === true);
  const bindAdjacentNonContext =
    operation !== "aggregate" &&
    params.queryAnalysis.answerMode !== "count_aggregate" &&
    (userQuestionBridge || assistantAnswerBridge);
  const sourceFamilies = new Set(bindingAnswerRefs.map((sourceRef) => sourceFamilyRef(sourceRef)));
  const answerSourceAdjacent = (entry: PromptEvidenceCandidate): boolean =>
    sourceRefsAdjacent(bindingAnswerRefs, bindingSourceRefsForEntry(entry), 2);
  const multiEvidenceCanonicalSiblingRank = (entry: PromptEvidenceCandidate): number => {
    if (params.queryAnalysis.answerMode !== "multi_evidence" || entry.surface !== "fact") {
      return 0;
    }
    if (!answerSourceAdjacent(entry)) {
      return 0;
    }
    return inferredSlotRole(params.queryAnalysis, entry) === "answer_value" ? 3 : 2;
  };
  const local = params.entries.filter((entry) => entryKey(entry) !== entryKey(params.answerEntry));
  const related = params.allEntries.filter((entry) => {
    if (entryKey(entry) === entryKey(params.answerEntry) || entry.dropReason) {
      return false;
    }
    const role = inferredSlotRole(params.queryAnalysis, entry);
    const entryRefs = bindingSourceRefsForEntry(entry);
    const entryFamilies = entryRefs.map((sourceRef) => sourceFamilyRef(sourceRef));
    if (sourceFamilies.size > 0 && entryFamilies.length > 0) {
      if (entryFamilies.some((family) => sourceFamilies.has(family))) {
        if (role === "query_context") {
          return (
            sourceRefsAdjacent(bindingAnswerRefs, entryRefs, 2) &&
            queryContextSupportScore(params.queryAnalysis, entry) >= 0.18
          );
        }
        return (
          bindAdjacentNonContext &&
          (role !== answerRole ||
            (userQuestionBridge && entryAuthorRole(entry) === "assistant") ||
            (assistantAnswerBridge && entryAuthorRole(entry) === "user")) &&
          sourceRefsAdjacent(bindingAnswerRefs, entryRefs, 2)
        );
      }
    }
    if (
      bindAdjacentNonContext &&
      (role !== answerRole ||
        (userQuestionBridge && entryAuthorRole(entry) === "assistant") ||
        (assistantAnswerBridge && entryAuthorRole(entry) === "user")) &&
      sourceRefsAdjacent(bindingAnswerRefs, entryRefs, 2)
    ) {
      return true;
    }
    return (
      role === "query_context" &&
      (sourceFamilies.size === 0 || entryFamilies.length === 0) &&
      semanticTextSimilarity(entry.text, params.answerEntry.text) >= 0.34
    );
  });
  return [...new Map([...local, ...related].map((entry) => [entryKey(entry), entry])).values()]
    .sort((left, right) => {
      const leftCanonicalSibling = multiEvidenceCanonicalSiblingRank(left);
      const rightCanonicalSibling = multiEvidenceCanonicalSiblingRank(right);
      if (leftCanonicalSibling !== rightCanonicalSibling) {
        return rightCanonicalSibling - leftCanonicalSibling;
      }
      const leftAdjacent = sourceRefsAdjacent(bindingAnswerRefs, bindingSourceRefsForEntry(left), 2)
        ? 1
        : 0;
      const rightAdjacent = sourceRefsAdjacent(
        bindingAnswerRefs,
        bindingSourceRefsForEntry(right),
        2,
      )
        ? 1
        : 0;
      if (leftAdjacent !== rightAdjacent) {
        return rightAdjacent - leftAdjacent;
      }
      const leftContext = inferredSlotRole(params.queryAnalysis, left) === "query_context" ? 1 : 0;
      const rightContext =
        inferredSlotRole(params.queryAnalysis, right) === "query_context" ? 1 : 0;
      if (leftContext !== rightContext) {
        return rightContext - leftContext;
      }
      return candidateRetrievalScore(right) - candidateRetrievalScore(left);
    })
    .slice(0, params.queryAnalysis.answerMode === "multi_evidence" ? 8 : 3);
}

function packetFromGroup(params: {
  queryAnalysis: QueryCompileResult;
  entries: PromptEvidenceCandidate[];
  allEntries: PromptEvidenceCandidate[];
  now: string;
}): EvidencePacket | null {
  const candidates = params.entries.map((entry) => {
    const contextCandidates = contextCandidatesForGroup({
      queryAnalysis: params.queryAnalysis,
      entries: params.entries,
      allEntries: params.allEntries,
      answerEntry: entry,
    });
    return {
      entry,
      contextCandidates,
      graded: gradeCandidate({
        queryAnalysis: params.queryAnalysis,
        entry,
        contextCandidates,
        now: params.now,
      }),
    };
  });
  const directFactAnswerMode =
    operationType(params.queryAnalysis) === "return_value" ||
    params.queryAnalysis.answerMode === "single_fact" ||
    params.queryAnalysis.answerMode === "attribute_lookup";
  const canonicalAnswerRank = (candidate: (typeof candidates)[number]): number => {
    if (!directFactAnswerMode) {
      return 0;
    }
    const contextSatisfied =
      candidate.entry.surface === "fact" &&
      sourceRefsForEntry(candidate.entry).length > 0 &&
      missingRequiredAfterContext(candidate.entry, candidate.contextCandidates).length === 0;
    if (contextSatisfied && candidate.graded.blockers.length === 0) {
      return 6;
    }
    if (!candidate.graded.eligibility.eligible) {
      return 0;
    }
    const role = inferredSlotRole(params.queryAnalysis, candidate.entry);
    const canAnswerDirectly =
      role === "answer_value" ||
      role === "answer_event" ||
      candidate.graded.eligibility.role === "answer" ||
      candidate.graded.grade.answerScore >= 0.42;
    if (!canAnswerDirectly || candidate.graded.grade.slotCoverageScore < 0.45) {
      return 0;
    }
    if (candidate.entry.surface === "fact") {
      return 3;
    }
    if (candidate.entry.surface === "event") {
      return 2;
    }
    if (candidate.entry.surface === "chunk" || candidate.entry.surface === "snippet") {
      return 1;
    }
    return 0;
  };
  const canonicalAdjustedScore = (candidate: (typeof candidates)[number]): number =>
    candidate.graded.grade.finalScore + canonicalAnswerRank(candidate) * 0.12;
  const rankedSelected = candidates.sort((left, right) => {
    if (left.graded.eligibility.eligible !== right.graded.eligibility.eligible) {
      return left.graded.eligibility.eligible ? -1 : 1;
    }
    const leftAdjusted = canonicalAdjustedScore(left);
    const rightAdjusted = canonicalAdjustedScore(right);
    if (leftAdjusted !== rightAdjusted) {
      return rightAdjusted - leftAdjusted;
    }
    return right.graded.grade.finalScore - left.graded.grade.finalScore;
  })[0];
  if (!rankedSelected) {
    return null;
  }
  const canonicalSubstituteEntry =
    directFactAnswerMode && rankedSelected.entry.surface !== "fact"
      ? params.allEntries.find(
          (entry) =>
            entry.surface === "fact" &&
            !entry.dropReason &&
            entriesShareBindingSourceIdentity(entry, rankedSelected.entry),
        )
      : undefined;
  const selected =
    canonicalSubstituteEntry && entryKey(canonicalSubstituteEntry) !== entryKey(rankedSelected.entry)
      ? {
          entry: canonicalSubstituteEntry,
          contextCandidates: [
            ...new Map(
              [rankedSelected.entry, ...rankedSelected.contextCandidates]
                .filter((entry) => entryKey(entry) !== entryKey(canonicalSubstituteEntry))
                .map((entry) => [entryKey(entry), entry]),
            ).values(),
          ],
          graded: gradeCandidate({
            queryAnalysis: params.queryAnalysis,
            entry: canonicalSubstituteEntry,
            contextCandidates: [
              ...new Map(
                [rankedSelected.entry, ...rankedSelected.contextCandidates]
                  .filter((entry) => entryKey(entry) !== entryKey(canonicalSubstituteEntry))
                  .map((entry) => [entryKey(entry), entry]),
              ).values(),
            ],
            now: params.now,
          }),
        }
      : rankedSelected;
  const canonicalFactContextSatisfied =
    directFactAnswerMode &&
    selected.entry.surface === "fact" &&
    selected.graded.blockers.length === 0 &&
    missingRequiredAfterContext(selected.entry, selected.contextCandidates).length === 0;
  const directFactContextSatisfied =
    params.queryAnalysis.answerMode !== "multi_evidence" && canonicalFactContextSatisfied;
  const hideCanonicalFactSupport =
    selected.entry.surface === "fact" &&
    canonicalFactContextSatisfied &&
    (directFactContextSatisfied || params.queryAnalysis.answerMode === "multi_evidence");
  const visibleCanonicalFactContextCandidates =
    params.queryAnalysis.answerMode === "multi_evidence"
      ? selected.contextCandidates.filter((entry) => entry.surface === "fact" && !entry.dropReason)
      : [];
  const selectedContextCandidates = directFactContextSatisfied
    ? []
    : hideCanonicalFactSupport
      ? visibleCanonicalFactContextCandidates
      : selected.contextCandidates;
  const coverageContextCandidates = hideCanonicalFactSupport
    ? selected.contextCandidates
    : selectedContextCandidates;
  const unitGroups = classifyPacketUnits({
    queryAnalysis: params.queryAnalysis,
    selectedEntry: selected.entry,
    contextCandidates: selectedContextCandidates,
  });
  const { displayLines, hiddenExactDuplicates } = packetDisplayLines({
    queryAnalysis: params.queryAnalysis,
    ...unitGroups,
  });
  const multiEvidenceContextSatisfied =
    params.queryAnalysis.answerMode === "multi_evidence" &&
    canonicalFactContextSatisfied &&
    unitGroups.answerUnits.some((unit) => unit.origin === "canonical_fact") &&
    (unitGroups.answerUnits.length > 1 ||
      selected.contextCandidates.some((entry) => {
        const author = entryAuthorRole(entry);
        return (
          (entry.surface === "chunk" || entry.surface === "event" || entry.surface === "snippet") &&
          author !== "assistant" &&
          !entryIsQueryLikeEvidence(params.queryAnalysis, entry) &&
          sourceRefsAdjacent(
            bindingSourceRefsForEntry(selected.entry),
            bindingSourceRefsForEntry(entry),
            2,
          )
        );
      }));
  const op = operationType(params.queryAnalysis);
  const returnValueMode = op === "return_value" || params.queryAnalysis.answerMode === "attribute_lookup";
  const hasAnswerDisplayLine = displayLines.some(
    (line) =>
      line.startsWith("[answer]") ||
      line.startsWith("[resource]") ||
      ((op === "aggregate" || params.queryAnalysis.answerMode === "count_aggregate") &&
        line.startsWith("[event]")),
  );
  const displayOnlyContext =
    returnValueMode && (!hasAnswerDisplayLine || unitGroups.answerUnits.length === 0);
  const packetDirectCauseScore = querySeeksCausalAnswer(params.queryAnalysis)
    ? Math.max(
        evidenceShapeFitScore(params.queryAnalysis, selected.entry, ["causal_explanation"]),
        evidenceShapeFitScoreForTexts(
          params.queryAnalysis,
          [
            ...displayLines,
            ...unitGroups.answerUnits.map((unit) => `${unit.displayText} ${unit.rawText}`),
          ],
          ["causal_explanation"],
        ),
      )
    : 0;
  const packetSuppliesDirectCause =
    packetDirectCauseScore >= 0.72 &&
    directCausalExplanationScore(params.queryAnalysis, selected.entry) < 0.5;
  const contextSupportedCanonicalFact = directFactContextSatisfied || multiEvidenceContextSatisfied;
  const directFactHiddenSupportPenalties = contextSupportedCanonicalFact
    ? selected.graded.softPenalties.filter(
        (reason) =>
          reason !== "answer-without-bound-context" &&
          reason !== "weak-query-context-binding" &&
          !reason.startsWith("missing-context:"),
      )
    : selected.graded.softPenalties;
  const baseSoftPenalties = packetSuppliesDirectCause
    ? directFactHiddenSupportPenalties.filter(
        (reason) => reason !== "weak-causal-explanation" && reason !== "query-like-answer-candidate",
      )
    : directFactHiddenSupportPenalties;
  const adjustedSoftPenalties = displayOnlyContext
    ? [...selected.graded.softPenalties, "no-answer-display-line"]
    : baseSoftPenalties;
  const packetCausalGrade: EvidenceGrade = packetSuppliesDirectCause
    ? {
        ...selected.graded.grade,
        answerScore: Math.max(selected.graded.grade.answerScore, packetDirectCauseScore),
        softPenaltyScore: Math.max(0, (selected.graded.grade.softPenaltyScore ?? 0) - 0.42),
        finalScore: Math.max(
          selected.graded.grade.finalScore,
          clamp01(
            0.46 +
              (selected.graded.grade.contextBindingScore ?? 0) * 0.1 +
              selected.graded.grade.authorityScore * 0.05 +
              selected.graded.grade.retrievalScore * 0.06,
          ),
        ),
      }
    : selected.graded.grade;
  const contextSupportedFactGrade: EvidenceGrade = contextSupportedCanonicalFact
    ? {
        ...packetCausalGrade,
        answerScore: Math.max(packetCausalGrade.answerScore, 0.46),
        contextBindingScore: Math.max(packetCausalGrade.contextBindingScore, 0.72),
        slotCoverageScore: Math.max(packetCausalGrade.slotCoverageScore, 0.58),
        softPenaltyScore: Math.max(0, (packetCausalGrade.softPenaltyScore ?? 0) - 0.24),
        finalScore: Math.max(packetCausalGrade.finalScore, 0.72),
      }
    : packetCausalGrade;
  const adjustedGrade: EvidenceGrade = displayOnlyContext
    ? {
        ...contextSupportedFactGrade,
        softPenaltyScore: clamp01((contextSupportedFactGrade.softPenaltyScore ?? 0) + 0.42),
        finalScore: clamp01(contextSupportedFactGrade.finalScore - 0.42),
      }
    : contextSupportedFactGrade;
  const adjustedEligibility: EvidenceEligibility = displayOnlyContext
    ? {
        ...selected.graded.eligibility,
        role: "support",
      }
    : selected.graded.eligibility;
  const allUnits = [
    ...unitGroups.answerUnits,
    ...unitGroups.contextUnits,
    ...unitGroups.supportUnits,
  ];
  const allSourceRefs = [...new Set(allUnits.flatMap((unit) => refsForUnit(unit)))];
  const sourceRefs = [
    ...new Set([
      ...sourceRefsForEntry(selected.entry),
      ...coverageContextCandidates.flatMap((entry) => sourceRefsForEntry(entry)),
      ...allSourceRefs,
    ]),
  ];
  const supportSourceRefs = [
    ...new Set([
      ...coverageContextCandidates.flatMap((entry) => sourceRefsForEntry(entry)),
      ...allUnits.flatMap((unit) => unit.supportRefs ?? []),
      ...unitGroups.contextUnits.flatMap((unit) => refsForUnit(unit)),
      ...unitGroups.supportUnits.flatMap((unit) => refsForUnit(unit)),
    ]),
  ];
  const slotIds = [
    ...new Set(
      [selected.entry, ...coverageContextCandidates].flatMap((entry) => matchedSlotIds(entry)),
    ),
  ];
  const slotId = slotIds[0] ?? "unplanned";
  const slotById = new Map(
    (params.queryAnalysis.evidencePlan?.slots ?? []).map((slot) => [slot.id, slot]),
  );
  const missing = [
    ...new Set(
      [
        ...missingRequiredAfterContext(selected.entry, coverageContextCandidates),
        ...slotIds
          .map((slotId) => slotById.get(slotId))
          .filter((slot): slot is QueryEvidenceSlot => Boolean(slot))
          .filter((slot) => slotNeedsTemporalField(params.queryAnalysis, slot))
          .filter(
            () =>
              !selected.entry.observedAt &&
              !coverageContextCandidates.some((candidate) => candidate.observedAt),
          )
          .map(() => "observedAt"),
      ].filter(Boolean),
    ),
  ];
  const role: EvidencePacket["role"] =
    adjustedEligibility.role === "answer" ||
    adjustedEligibility.role === "resource" ||
    (!displayOnlyContext && adjustedGrade.answerScore >= 0.28)
      ? "partial"
      : "support";
  const packetId = `packet:${stableHash([
    selected.entry.id,
    slotIds.join("|"),
    ...sourceRefs,
    normalizeText(selected.entry.text).slice(0, 120),
  ])}`;
  return {
    packetId,
    slotId,
    slotIds,
    operationType: op,
    role,
    protected: false,
    answerCandidate: selected.entry,
    contextCandidates: selectedContextCandidates,
    answerUnits: unitGroups.answerUnits,
    contextUnits: unitGroups.contextUnits,
    supportUnits: unitGroups.supportUnits,
    layers: [...new Set([selected.entry.surface as EvidencePlanLayer])],
    primaryText: displayLines[0] ?? truncateText(selected.entry.text, 560),
    supportingTexts:
      displayLines.length > 1
        ? displayLines.slice(1)
        : selectedContextCandidates
            .map((entry) => truncateText(entry.text, 260))
            .filter((text) => semanticTextSimilarity(text, selected.entry.text) < 0.82),
    sourceRefs,
    supportSourceRefs,
    allSourceRefs,
    normalizedSourceRefs: normalizeSourceRefs(sourceRefs),
    normalizedSupportSourceRefs: normalizeSourceRefs(supportSourceRefs),
    normalizedAllSourceRefs: normalizeSourceRefs(allSourceRefs),
    score: adjustedGrade.finalScore,
    scoreBreakdown: {
      retrievalScore: adjustedGrade.retrievalScore,
      answerScore: adjustedGrade.answerScore,
      contextBindingScore: adjustedGrade.contextBindingScore,
      temporalFitScore: adjustedGrade.temporalFitScore ?? 0,
      slotCoverageScore: adjustedGrade.slotCoverageScore,
      authorityScore: adjustedGrade.authorityScore,
      softPenaltyScore: adjustedGrade.softPenaltyScore ?? 0,
      bridgeSupportScore: bridgeSupportScore(selected.entry),
      bridgePositiveSignalScore: bridgePositiveSignalScore(selected.entry),
      bridgeNegativeSignalScore: bridgeNegativeSignalScore(selected.entry),
      finalScore: adjustedGrade.finalScore,
      answerUnitCount: unitGroups.answerUnits.length,
      contextUnitCount: unitGroups.contextUnits.length,
      supportUnitCount: unitGroups.supportUnits.length,
    },
    displayLines,
    hiddenExactDuplicates,
    observedAt:
      selected.entry.observedAt ??
      selectedContextCandidates.find((entry) => entry.observedAt)?.observedAt,
    resolvedDate:
      selected.entry.observedAt?.slice(0, 10) ??
      selectedContextCandidates.find((entry) => entry.observedAt)?.observedAt?.slice(0, 10),
    dedupeKey: sourceKey(sourceRefs, selected.entry.text),
    authorRoles: [...new Set(allUnits.map((unit) => unit.authorRole).filter(Boolean))].filter(
      (role): role is "user" | "assistant" | "tool" | "memory" =>
        Boolean(role) && role !== "unknown",
    ),
    coverage: {
      filled: adjustedEligibility.eligible && missing.length === 0,
      missing,
      confidence: adjustedGrade.finalScore,
    },
    eligibility: adjustedEligibility,
    grade: adjustedGrade,
    selectionReason:
      selected.graded.blockers.length > 0
        ? `excluded:${selected.graded.blockers.join(",")}`
        : `ranked:${adjustedEligibility.role}:${adjustedGrade.finalScore.toFixed(3)}${
            adjustedSoftPenalties.length > 0
              ? ` penalties=${adjustedSoftPenalties.join("|")}`
              : ""
          }`,
    blockedBy: selected.graded.blockers,
    softPenalties: adjustedSoftPenalties,
    hardExclusions: selected.graded.blockers,
    dropReason:
      selected.graded.blockers.length > 0 ? selected.graded.blockers.join(",") : undefined,
  };
}

function highLevelSupportPackets(params: {
  queryAnalysis: QueryCompileResult;
  candidateGenerationResult?: CandidateGenerationResult;
}): EvidencePacket[] {
  const packets: EvidencePacket[] = [];
  for (const candidate of params.candidateGenerationResult?.layerCandidates ?? []) {
    const sourceRefs = [
      ...new Set(
        [candidate.sourceRef, ...(candidate.sourceRefs ?? [])].filter((value): value is string =>
          Boolean(value),
        ),
      ),
    ];
    packets.push({
      packetId: `packet:${stableHash([candidate.id, ...sourceRefs])}`,
      slotId: candidate.slotMatches[0]?.slotId ?? "control",
      slotIds: [...new Set(candidate.slotMatches.map((match) => match.slotId))],
      operationType: operationType(params.queryAnalysis),
      role: HIGH_LEVEL_LAYERS.has(candidate.layer) ? "support" : "partial",
      protected: false,
      layers: [candidate.layer],
      primaryText: truncateText(candidate.text, 420),
      supportingTexts: [],
      sourceRefs,
      observedAt: candidate.observedAt,
      resolvedDate: candidate.observedAt?.slice(0, 10),
      dedupeKey: sourceKey(sourceRefs, candidate.text),
      authorRoles: ["memory"],
      coverage: {
        filled: false,
        missing: sourceRefs.length === 0 ? ["sourceRef"] : ["source_evidence"],
        confidence: clamp01(candidate.score),
      },
      eligibility: {
        eligible: false,
        role: "support",
        blockers:
          sourceRefs.length === 0
            ? ["high-level-memory-missing-source-ref"]
            : ["source-evidence-not-selected"],
      },
      grade: {
        retrievalScore: clamp01(candidate.score),
        answerScore: 0.1,
        contextBindingScore: 0.1,
        temporalFitScore: 0.42,
        slotCoverageScore: clamp01(candidate.score),
        authorityScore: 0.42,
        softPenaltyScore: 0.18,
        finalScore: clamp01(candidate.score * 0.28),
      },
      selectionReason: "high-level-support-only",
      blockedBy:
        sourceRefs.length === 0
          ? ["high-level-memory-missing-source-ref"]
          : ["source-evidence-not-selected"],
      dropReason:
        sourceRefs.length === 0
          ? "high-level-memory-missing-source-ref"
          : "source-evidence-not-selected",
      hardExclusions:
        sourceRefs.length === 0
          ? ["high-level-memory-missing-source-ref"]
          : ["source-evidence-not-selected"],
    });
  }
  return packets;
}

function packetSortValue(packet: EvidencePacket): number {
  return packet.grade?.finalScore ?? packet.coverage.confidence;
}

function packetInjectionBudget(queryAnalysis: QueryCompileResult): number {
  if (
    queryAnalysis.answerMode === "count_aggregate" ||
    operationType(queryAnalysis) === "aggregate"
  ) {
    return 5;
  }
  if (
    queryAnalysis.answerMode === "multi_evidence" ||
    operationType(queryAnalysis) === "tailor_advice" ||
    operationType(queryAnalysis) === "derive" ||
    operationType(queryAnalysis) === "compare"
  ) {
    return 4;
  }
  return 3;
}

function packetScoreFloor(queryAnalysis: QueryCompileResult): number {
  if (operationType(queryAnalysis) === "tailor_advice") {
    return 0.28;
  }
  if (
    queryAnalysis.answerMode === "count_aggregate" ||
    operationType(queryAnalysis) === "aggregate"
  ) {
    return 0.24;
  }
  return 0.28;
}

function packetCurveGap(queryAnalysis: QueryCompileResult): number {
  if (
    queryAnalysis.answerMode === "count_aggregate" ||
    operationType(queryAnalysis) === "aggregate"
  ) {
    return 0.34;
  }
  if (
    queryAnalysis.answerMode === "multi_evidence" ||
    operationType(queryAnalysis) === "tailor_advice" ||
    operationType(queryAnalysis) === "derive" ||
    operationType(queryAnalysis) === "compare"
  ) {
    return 0.16;
  }
  return 0.42;
}

function packetDistinctKey(packet: EvidencePacket): string {
  return (
    packet.sourceRefs[0] ??
    packet.allSourceRefs?.[0] ??
    packet.answerCandidate?.id ??
    packet.packetId
  );
}

function packetAnswerDisplayKey(packet: EvidencePacket): string {
  const primaryLine = packet.displayLines?.[0] ?? packet.primaryText;
  return exactDisplayKey(primaryLine.replace(/\s+\|\s+\[context\].*$/u, ""));
}

function packetHasSoftPenalty(packet: EvidencePacket, reason: string): boolean {
  return (packet.softPenalties ?? []).some((penalty) => penalty === reason);
}

function packetHasSoftPenaltyPrefix(packet: EvidencePacket, prefix: string): boolean {
  return (packet.softPenalties ?? []).some((penalty) => penalty.startsWith(prefix));
}

function packetCoverageSatisfied(packet: EvidencePacket): boolean {
  return packet.coverage.filled && packet.coverage.missing.length === 0;
}

function slotActsAsAnswerValue(slot: QueryEvidenceSlot): boolean {
  const requiredRole = evidenceSlotRequiredRole(slot);
  return requiredRole === "answer_value" || slot.role === "answer_evidence";
}

function answerSlotIds(queryAnalysis: QueryCompileResult): Set<string> {
  return new Set(
    (queryAnalysis.evidencePlan?.slots ?? [])
      .filter(slotActsAsAnswerValue)
      .map((slot) => slot.id),
  );
}

function packetFilledSlotIds(packet: EvidencePacket): Set<string> {
  return new Set(
    packet.answerCandidate?.filledSlotIds ??
      (packet.answerCandidate?.slotCoverage ?? [])
        .filter((coverage) => coverage.filled)
        .map((coverage) => coverage.slotId),
  );
}

function packetFillsAnswerSlot(queryAnalysis: QueryCompileResult, packet: EvidencePacket): boolean {
  const answerSlots = answerSlotIds(queryAnalysis);
  if (answerSlots.size === 0) {
    return false;
  }
  const filled = packetFilledSlotIds(packet);
  for (const slotId of filled) {
    if (answerSlots.has(slotId)) {
      return true;
    }
  }
  return false;
}

function packetFillsSlot(packet: EvidencePacket, slotId: string): boolean {
  return packetFilledSlotIds(packet).has(slotId);
}

function packetCoverageSatisfiedForPrompt(
  queryAnalysis: QueryCompileResult,
  packet: EvidencePacket,
): boolean {
  return (
    packetCoverageSatisfied(packet) ||
    (queryAnalysis.answerMode === "multi_evidence" && packetFillsAnswerSlot(queryAnalysis, packet))
  );
}

function packetHasBlockingPromptPenalty(
  queryAnalysis: QueryCompileResult,
  packet: EvidencePacket,
): boolean {
  if (
    packetHasSoftPenaltyPrefix(packet, "missing-context:") &&
    !(queryAnalysis.answerMode === "multi_evidence" && packetFillsAnswerSlot(queryAnalysis, packet))
  ) {
    return true;
  }
  if (packetHasSoftPenalty(packet, "answer-without-bound-context")) {
    return true;
  }
  return (
    operationType(queryAnalysis) === "return_value" &&
    packetHasSoftPenalty(packet, "weak-query-context-binding")
  );
}

function nonInjectedDropReason(
  queryAnalysis: QueryCompileResult,
  packet: EvidencePacket,
): string | undefined {
  if (packet.dropReason) {
    return packet.dropReason;
  }
  if (packetHasBlockingPromptPenalty(queryAnalysis, packet)) {
    return "unbound-query-context";
  }
  return undefined;
}

function packetPromptInjectionFloor(
  queryAnalysis: QueryCompileResult,
  packet: EvidencePacket,
  floor: number,
): number {
  if (operationType(queryAnalysis) === "tailor_advice") {
    return Math.max(0.46, floor + 0.12);
  }
  if (queryAnalysis.answerMode === "count_aggregate" || operationType(queryAnalysis) === "aggregate") {
    return Math.max(0.5, floor + 0.16);
  }
  return packetCoverageSatisfiedForPrompt(queryAnalysis, packet)
    ? Math.max(0.52, floor + 0.2)
    : Math.max(0.62, floor + 0.12);
}

function packetEligibleForPromptInjection(
  queryAnalysis: QueryCompileResult,
  packet: EvidencePacket,
  floor: number,
): boolean {
  if (packetHasBlockingPromptPenalty(queryAnalysis, packet)) {
    return false;
  }
  if (
    !packetHasAnswerDisplayForQuery(queryAnalysis, packet) &&
    operationType(queryAnalysis) !== "derive" &&
    operationType(queryAnalysis) !== "compare"
  ) {
    return false;
  }
  return (
    (packet.grade?.finalScore ?? 0) >= packetPromptInjectionFloor(queryAnalysis, packet, floor) &&
    (packet.grade?.slotCoverageScore ?? 0) >= 0.45
  );
}

function packetHasAnswerDisplayForQuery(
  queryAnalysis: QueryCompileResult,
  packet: EvidencePacket,
): boolean {
  const aggregateMode =
    queryAnalysis.answerMode === "count_aggregate" || operationType(queryAnalysis) === "aggregate";
  return (packet.displayLines ?? []).some(
    (line) =>
      line.startsWith("[answer]") ||
      line.startsWith("[resource]") ||
      (aggregateMode && line.startsWith("[event]")),
  );
}

function packetIsCanonicalFactAnswer(queryAnalysis: QueryCompileResult, packet: EvidencePacket): boolean {
  return (
    packet.answerCandidate?.surface === "fact" &&
    packetCoverageSatisfied(packet) &&
    packetHasAnswerDisplayForQuery(queryAnalysis, packet)
  );
}

function packetIsCanonicalSlotAnswer(
  queryAnalysis: QueryCompileResult,
  packet: EvidencePacket,
  slotId: string,
): boolean {
  return (
    packet.answerCandidate?.surface === "fact" &&
    packetFillsSlot(packet, slotId) &&
    packetHasAnswerDisplayForQuery(queryAnalysis, packet)
  );
}

function packetIsAssistantChunkAnswer(packet: EvidencePacket): boolean {
  const candidate = packet.answerCandidate;
  const metadataRole = candidate?.metadata?.role;
  return (
    (candidate?.surface === "chunk" || candidate?.surface === "snippet") &&
    (metadataRole === "assistant" || (packet.authorRoles ?? []).includes("assistant"))
  );
}

function normalizedPacketRefKeys(packet: EvidencePacket): Set<string> {
  const refs = [
    ...normalizeSourceRefs(packet.sourceRefs),
    ...normalizeSourceRefs(packet.supportSourceRefs ?? []),
    ...normalizeSourceRefs(packet.allSourceRefs ?? []),
  ];
  return new Set(refs.flatMap((ref) => [ref.raw, `${ref.kind}:${ref.id}`, ...ref.parentRefs]));
}

function packetsShareLineage(left: EvidencePacket, right: EvidencePacket): boolean {
  const leftRefs = normalizedPacketRefKeys(left);
  const rightRefs = normalizedPacketRefKeys(right);
  for (const ref of leftRefs) {
    if (rightRefs.has(ref)) {
      return true;
    }
  }
  return false;
}

function canonicalFactAnchorDominatesText(packet: EvidencePacket, text: string): boolean {
  const normalizedText = normalizeText(text);
  if (!normalizedText) {
    return false;
  }
  const candidate = packet.answerCandidate;
  const metadata = candidate?.metadata ?? {};
  const factText = candidate?.text ?? packet.primaryText ?? "";
  const subject =
    typeof metadata.canonicalSubject === "string"
      ? metadata.canonicalSubject
      : /^(?:\[answer\]\s*)?(.+?)\s+--/iu.exec(factText)?.[1]?.trim() ??
        /^(?:\[answer\]\s*)?(.+?)\s+has\s+/iu.exec(factText)?.[1]?.trim();
  const object =
    typeof metadata.canonicalObject === "string"
      ? metadata.canonicalObject
      : typeof metadata.value === "string"
        ? metadata.value
        : undefined;
  const parsedObject =
    !object && candidate?.text
      ? /--[^>\n]+-->\s*([^\n|]+)/u.exec(candidate.text)?.[1]?.trim()
      : undefined;
  const normalizedSubject = subject ? normalizeText(subject) : "";
  const normalizedObject = normalizeText(object ?? parsedObject ?? "");
  if (!normalizedSubject || !normalizedObject) {
    return false;
  }
  return normalizedText.includes(normalizedSubject) && normalizedText.includes(normalizedObject);
}

function canonicalFactDominatesAssistantRestatement(params: {
  queryAnalysis: QueryCompileResult;
  factPacket: EvidencePacket;
  assistantPacket: EvidencePacket;
}): boolean {
  if (!packetIsCanonicalFactAnswer(params.queryAnalysis, params.factPacket)) {
    return false;
  }
  if (!packetIsAssistantChunkAnswer(params.assistantPacket)) {
    return false;
  }
  if (packetsShareLineage(params.factPacket, params.assistantPacket)) {
    return true;
  }
  const factText = (params.factPacket.displayLines ?? [params.factPacket.primaryText]).join("\n");
  const assistantText = (
    params.assistantPacket.displayLines ?? [params.assistantPacket.primaryText]
  ).join("\n");
  if (canonicalFactAnchorDominatesText(params.factPacket, assistantText)) {
    return true;
  }
  return semanticTextSimilarity(factText, assistantText) >= 0.72;
}

function suppressDominatedInjectedPackets(
  queryAnalysis: QueryCompileResult,
  packets: EvidencePacket[],
  injectedPacketIds: Set<string>,
): Set<string> {
  const selected = new Set(injectedPacketIds);
  const canonicalFactPackets = packets.filter(
    (packet) => selected.has(packet.packetId) && packetIsCanonicalFactAnswer(queryAnalysis, packet),
  );
  if (canonicalFactPackets.length === 0) {
    return selected;
  }
  for (const packet of packets) {
    if (!selected.has(packet.packetId)) {
      continue;
    }
    if (!packetIsAssistantChunkAnswer(packet)) {
      continue;
    }
    if (
      canonicalFactPackets.some((factPacket) =>
        canonicalFactDominatesAssistantRestatement({
          queryAnalysis,
          factPacket,
          assistantPacket: packet,
        }),
      )
    ) {
      selected.delete(packet.packetId);
    }
  }
  return selected;
}

function selectInjectedPackets(
  queryAnalysis: QueryCompileResult,
  packets: EvidencePacket[],
): Set<string> {
  const limit = packetInjectionBudget(queryAnalysis);
  const floor = packetScoreFloor(queryAnalysis);
  const selected = new Set<string>();
  const selectedDistinct = new Set<string>();
  const selectedAnswerDisplays = new Set<string>();
  const ranked = packets
    .filter((packet) => !packet.dropReason && packet.eligibility?.eligible !== false)
    .filter((packet) => packetEligibleForPromptInjection(queryAnalysis, packet, floor))
    .sort((left, right) => packetSortValue(right) - packetSortValue(left));
  const nonContrastAnswerAvailable = ranked.some(
    (packet) =>
      !packetHasSoftPenalty(packet, "negative-contrast-without-query-context") &&
      ((packet.grade?.answerScore ?? 0) > 0.08 ||
        (packet.displayLines ?? []).some((line) => line.startsWith("[answer]"))),
  );
  const rankedForSelection = nonContrastAnswerAvailable
    ? ranked.filter(
        (packet) => !packetHasSoftPenalty(packet, "negative-contrast-without-query-context"),
      )
    : ranked;
  const positiveScoreAvailable = rankedForSelection.some((packet) => packetSortValue(packet) > 0);
  const scoredRankedForSelection = positiveScoreAvailable
    ? rankedForSelection.filter((packet) => packetSortValue(packet) > 0)
    : rankedForSelection;
  const aggregateMode =
    queryAnalysis.answerMode === "count_aggregate" || operationType(queryAnalysis) === "aggregate";
  const answerDisplayAvailable = scoredRankedForSelection.some((packet) =>
    packetHasAnswerDisplayForQuery(queryAnalysis, packet),
  );
  const answerDisplayRankedForSelection = answerDisplayAvailable
    ? scoredRankedForSelection.filter((packet) =>
        packetHasAnswerDisplayForQuery(queryAnalysis, packet),
      )
    : scoredRankedForSelection;
  const selectablePackets = answerDisplayRankedForSelection;
  const topScore = selectablePackets.length > 0 ? packetSortValue(selectablePackets[0]!) : 0;
  const allowedGap = packetCurveGap(queryAnalysis);
  const effectiveAllowedGap = allowedGap;
  const directCanonicalAnswer = (
    operationType(queryAnalysis) === "return_value" ||
    queryAnalysis.answerMode === "single_fact" ||
    queryAnalysis.answerMode === "attribute_lookup"
  ) && queryAnalysis.answerMode !== "multi_evidence"
    ? selectablePackets.find(
        (packet) =>
          packet.answerCandidate?.surface === "fact" &&
          packetCoverageSatisfied(packet) &&
          packetHasAnswerDisplayForQuery(queryAnalysis, packet),
      )
    : undefined;
  if (directCanonicalAnswer) {
    return new Set([directCanonicalAnswer.packetId]);
  }
  const slotCoverageMode =
    queryAnalysis.answerMode === "multi_evidence" ||
    operationType(queryAnalysis) === "derive" ||
    operationType(queryAnalysis) === "compare";
  if (slotCoverageMode) {
    const answerSlotsForSelection = answerSlotIds(queryAnalysis);
    const selectedAnswerSlotIds = new Set<string>();
    const bySlot = new Map<string, EvidencePacket[]>();
    for (const packet of selectablePackets) {
      const packetSlotIds =
        packet.slotIds && packet.slotIds.length > 0 ? packet.slotIds : [packet.slotId];
      for (const slotId of packetSlotIds) {
        const slotPackets = bySlot.get(slotId) ?? [];
        slotPackets.push(packet);
        bySlot.set(slotId, slotPackets);
      }
    }
    const plannedSlots =
      queryAnalysis.answerMode === "multi_evidence" && answerSlotsForSelection.size > 1
        ? (queryAnalysis.evidencePlan?.slots ?? []).filter((slot) =>
            answerSlotsForSelection.has(slot.id),
          )
        : (queryAnalysis.evidencePlan?.slots ?? []);
    for (const slot of plannedSlots) {
      if (selected.size >= limit) {
        break;
      }
      const slotPackets = bySlot.get(slot.id) ?? [];
      const packet =
        slotPackets.find(
          (candidate) =>
            packetIsCanonicalSlotAnswer(queryAnalysis, candidate, slot.id) &&
            (candidate.grade?.finalScore ?? 0) >= floor,
        ) ?? slotPackets.find((candidate) => (candidate.grade?.finalScore ?? 0) >= floor);
      if (packet) {
        const displayKey = packetAnswerDisplayKey(packet);
        if (!aggregateMode && displayKey && selectedAnswerDisplays.has(displayKey)) {
          continue;
        }
        selected.add(packet.packetId);
        if (answerSlotsForSelection.has(slot.id)) {
          selectedAnswerSlotIds.add(slot.id);
        }
        if (!aggregateMode && displayKey) {
          selectedAnswerDisplays.add(displayKey);
        }
      }
    }
    if (
      queryAnalysis.answerMode === "multi_evidence" &&
      answerSlotsForSelection.size > 1 &&
      [...answerSlotsForSelection].every((slotId) => selectedAnswerSlotIds.has(slotId))
    ) {
      return selected;
    }
  }
  for (const packet of selectablePackets) {
    if (selected.size >= limit) {
      break;
    }
    const score = packet.grade?.finalScore ?? 0;
    if (score < floor) {
      continue;
    }
    if (!aggregateMode && selected.size > 0 && topScore - score > effectiveAllowedGap) {
      continue;
    }
    const distinctKey = packetDistinctKey(packet);
    if (aggregateMode && selectedDistinct.has(distinctKey)) {
      continue;
    }
    if (
      !aggregateMode &&
      packetHasSoftPenaltyPrefix(packet, "missing-context:") &&
      [...selected].filter((packetId) => {
        const selectedPacket = packets.find((candidate) => candidate.packetId === packetId);
        return selectedPacket && !packetHasSoftPenaltyPrefix(selectedPacket, "missing-context:");
      }).length >= 2
    ) {
      continue;
    }
    const displayKey = packetAnswerDisplayKey(packet);
    if (!aggregateMode && displayKey && selectedAnswerDisplays.has(displayKey)) {
      continue;
    }
    selected.add(packet.packetId);
    selectedDistinct.add(distinctKey);
    if (!aggregateMode && displayKey) {
      selectedAnswerDisplays.add(displayKey);
    }
  }
  return selected;
}

function markPromptEvidenceFromPackets(params: {
  promptEvidence: PromptEvidenceCandidate[];
  packets: EvidencePacket[];
  injectedPacketIds: Set<string>;
}): PromptEvidenceCandidate[] {
  const packetByEntryKey = new Map<string, EvidencePacket>();
  for (const packet of params.packets) {
    if (packet.answerCandidate) {
      packetByEntryKey.set(entryKey(packet.answerCandidate), packet);
    }
  }
  const emittedInjectedPackets = new Set<string>();
  const mapped = params.promptEvidence.map((entry) => {
    const packet = packetByEntryKey.get(entryKey(entry));
    if (!packet) {
      return entry.role === "protected" ? { ...entry, role: "support" as const } : entry;
    }
    const injectedEntry =
      params.injectedPacketIds.has(packet.packetId) && !emittedInjectedPackets.has(packet.packetId);
    if (injectedEntry) {
      emittedInjectedPackets.add(packet.packetId);
    }
    return {
      ...entry,
      packetId: packet.packetId,
      text:
        injectedEntry && packet.displayLines && packet.displayLines.length > 0
          ? packet.displayLines.join("\n")
          : entry.text,
      mergedSourceRefs:
        injectedEntry && (packet.allSourceRefs?.length ?? 0) > 0
          ? packet.allSourceRefs
          : entry.mergedSourceRefs,
      role: injectedEntry ? ("protected" as const) : ("support" as const),
      injected: injectedEntry,
      eligibility: packet.eligibility,
      grade: packet.grade,
      blockedBy: packet.blockedBy,
      softPenalties: packet.softPenalties,
      hardExclusions: packet.hardExclusions,
      injectionScore: packet.grade?.finalScore ?? entry.injectionScore,
      scoreBreakdown: {
        ...(entry.scoreBreakdown ?? {}),
        packetFinalScore: packet.grade?.finalScore ?? 0,
        packetAnswerScore: packet.grade?.answerScore ?? 0,
        packetContextBindingScore: packet.grade?.contextBindingScore ?? 0,
        packetTemporalFitScore: packet.grade?.temporalFitScore ?? 0,
        packetSlotCoverageScore: packet.grade?.slotCoverageScore ?? 0,
        packetAuthorityScore: packet.grade?.authorityScore ?? 0,
        packetRetrievalScore: packet.grade?.retrievalScore ?? 0,
        packetSoftPenaltyScore: packet.grade?.softPenaltyScore ?? 0,
        packetEligible: packet.eligibility?.eligible ?? false,
        packetEligibilityRole: packet.eligibility?.role ?? "support",
        packetDisplayLineCount: packet.displayLines?.length ?? 0,
        packetHiddenExactDuplicateCount: packet.hiddenExactDuplicates?.length ?? 0,
        packetBridgeSupportScore: bridgeSupportScore(entry),
        packetBridgePositiveSignalScore: bridgePositiveSignalScore(entry),
        packetBridgeNegativeSignalScore: bridgeNegativeSignalScore(entry),
      },
      selectionReason: injectedEntry
        ? `packet:${packet.selectionReason ?? "selected"}`
        : (packet.selectionReason ?? entry.selectionReason),
      protectionReason: injectedEntry
        ? `packet:${packet.selectionReason ?? "selected"}`
        : (packet.dropReason ?? entry.protectionReason),
    };
  });
  return mapped.sort((left, right) => {
    const leftProtected = left.role === "protected" ? 1 : 0;
    const rightProtected = right.role === "protected" ? 1 : 0;
    if (leftProtected !== rightProtected) {
      return rightProtected - leftProtected;
    }
    return (right.injectionScore ?? 0) - (left.injectionScore ?? 0);
  });
}

function layerCountsForSlot(
  slot: QueryEvidenceSlot,
  candidateGenerationResult?: CandidateGenerationResult,
): Partial<Record<EvidencePlanLayer, number>> {
  const counts: Partial<Record<EvidencePlanLayer, number>> = {};
  for (const layer of slotLayers(slot)) {
    const count =
      candidateGenerationResult?.slotLayerStats
        .filter((stat) => stat.slotId === slot.id && stat.layer === layer)
        .reduce((sum, stat) => sum + stat.rawCount, 0) ?? 0;
    if (count > 0) {
      counts[layer] = count;
    }
  }
  return counts;
}

function auditForSlot(params: {
  slot: QueryEvidenceSlot;
  packets: EvidencePacket[];
  injectedPacketIds: Set<string>;
  candidateGenerationResult?: CandidateGenerationResult;
}): EvidencePacketSlotAudit {
  const slotPackets = params.packets.filter(
    (packet) => packet.slotId === params.slot.id || (packet.slotIds ?? []).includes(params.slot.id),
  );
  const injectedForSlot = slotPackets.filter((packet) =>
    params.injectedPacketIds.has(packet.packetId),
  );
  const missing =
    injectedForSlot.length >= Math.max(1, params.slot.minEvidence)
      ? []
      : [`minEvidence:${Math.max(1, params.slot.minEvidence)}`];
  return {
    slotId: params.slot.id,
    queriedLayers: slotLayers(params.slot),
    layerCandidateCounts: layerCountsForSlot(params.slot, params.candidateGenerationResult),
    packets: slotPackets.map((packet) => ({
      packetId: packet.packetId,
      role: packet.role,
      layers: packet.layers,
      sourceRefs: packet.sourceRefs,
      supportSourceRefs: packet.supportSourceRefs ?? [],
      allSourceRefs: packet.allSourceRefs ?? packet.sourceRefs,
      normalizedSourceRefs: packet.normalizedSourceRefs,
      normalizedSupportSourceRefs: packet.normalizedSupportSourceRefs,
      normalizedAllSourceRefs: packet.normalizedAllSourceRefs,
      coverage: packet.coverage,
      eligibility: packet.eligibility,
      grade: packet.grade,
      score: packet.score,
      scoreBreakdown: packet.scoreBreakdown,
      displayLines: packet.displayLines,
      hiddenExactDuplicates: packet.hiddenExactDuplicates,
      answerUnits: packet.answerUnits,
      contextUnits: packet.contextUnits,
      supportUnits: packet.supportUnits,
      selectionReason: packet.selectionReason,
      softPenalties: packet.softPenalties ?? [],
      hardExclusions: packet.hardExclusions ?? [],
      protected: params.injectedPacketIds.has(packet.packetId),
      injected: params.injectedPacketIds.has(packet.packetId),
      compilerSignalSeen: packet.sourceRefs.length > 0,
      materializedLayers: packet.layers,
      candidateSeen: true,
      packetAssembled: true,
      dropReason: packet.dropReason ?? null,
    })),
    missing,
  };
}

function allPacketUnits(packets: EvidencePacket[]): EvidenceUnit[] {
  const units = packets.flatMap((packet) => [
    ...(packet.answerUnits ?? []),
    ...(packet.contextUnits ?? []),
    ...(packet.supportUnits ?? []),
  ]);
  return [...new Map(units.map((unit) => [unit.unitId, unit])).values()];
}

function evidencePacketAudit(params: {
  operation: NonNullable<QueryCompileResult["evidencePlan"]>["operation"];
  slots: QueryEvidenceSlot[];
  packets: EvidencePacket[];
  promptEvidence: PromptEvidenceCandidate[];
  injectedPacketIds: Set<string>;
  candidateGenerationResult?: CandidateGenerationResult;
}): EvidencePacketAudit {
  const rankedPackets = [...params.packets]
    .filter((packet) => !packet.dropReason)
    .sort((left, right) => (right.grade?.finalScore ?? 0) - (left.grade?.finalScore ?? 0));
  const droppedPackets = params.packets.filter((packet) => packet.dropReason);
  const summarizePacket = (packet: EvidencePacket) => ({
    packetId: packet.packetId,
    injected: packet.injected ?? params.injectedPacketIds.has(packet.packetId),
    score: packet.score,
    finalScore: packet.grade?.finalScore,
    sourceRefs: packet.sourceRefs,
    allSourceRefs: packet.allSourceRefs ?? packet.sourceRefs,
    normalizedSourceRefs: packet.normalizedSourceRefs,
    normalizedAllSourceRefs: packet.normalizedAllSourceRefs,
    primaryText: truncateText(packet.primaryText, 720),
    displayLines: packet.displayLines,
    selectionReason: packet.selectionReason,
  });
  const units = allPacketUnits(params.packets);
  return {
    operation: params.operation,
    slots: params.slots.map((slot) =>
      auditForSlot({
        slot,
        packets: params.packets,
        injectedPacketIds: params.injectedPacketIds,
        candidateGenerationResult: params.candidateGenerationResult,
      }),
    ),
    candidatePool: params.promptEvidence.map((entry) => ({
      id: entry.id,
      surface: entry.surface,
      sourceRef: entry.sourceRef,
      mergedSourceRefs: entry.mergedSourceRefs,
      normalizedSourceRefs: normalizeSourceRefs([
        entry.sourceRef,
        ...(entry.mergedSourceRefs ?? []),
      ]),
      role: entry.role,
      injected: entry.injected,
      packetId: entry.packetId,
      injectionScore: entry.injectionScore,
      slotEvidenceRole: entry.slotEvidenceRole,
      text: truncateText(entry.text, 480),
    })),
    evidenceUnits: units,
    sourceExpansion: units
      .filter(
        (unit) =>
          (unit.supportRefs?.length ?? 0) > 0 ||
          (unit.derivedFromRefs?.length ?? 0) > 0 ||
          (unit.neighborRefs?.length ?? 0) > 0,
      )
      .map((unit) => ({
        unitId: unit.unitId,
        sourceRefs: unit.sourceRefs,
        normalizedSourceRefs: unit.normalizedSourceRefs,
        supportRefs: unit.supportRefs ?? [],
        normalizedSupportRefs: unit.normalizedSupportRefs,
        derivedFromRefs: unit.derivedFromRefs ?? [],
        normalizedDerivedFromRefs: unit.normalizedDerivedFromRefs,
        neighborRefs: unit.neighborRefs ?? [],
        normalizedNeighborRefs: unit.normalizedNeighborRefs,
        origin: unit.origin,
        roles: unit.roles,
      })),
    rankedPackets: rankedPackets.map((packet) => ({
      packetId: packet.packetId,
      injected: packet.injected ?? params.injectedPacketIds.has(packet.packetId),
      score: packet.score,
      finalScore: packet.grade?.finalScore,
      sourceRefs: packet.sourceRefs,
      allSourceRefs: packet.allSourceRefs ?? packet.sourceRefs,
      normalizedSourceRefs: packet.normalizedSourceRefs,
      normalizedAllSourceRefs: packet.normalizedAllSourceRefs,
      selectionReason: packet.selectionReason,
    })),
    eligibleEvidencePackets: rankedPackets.map(summarizePacket),
    droppedEvidencePackets: droppedPackets.map((packet) => ({
      ...summarizePacket(packet),
      dropReason: packet.dropReason ?? "dropped",
      softPenalties: packet.softPenalties ?? [],
      hardExclusions: packet.hardExclusions ?? [],
    })),
    injectedPackets: [...params.injectedPacketIds],
    renderedPromptLines: params.packets
      .filter((packet) => params.injectedPacketIds.has(packet.packetId))
      .flatMap((packet) => {
        const lines = (packet.displayLines ?? [packet.primaryText]).filter(Boolean);
        const evidenceUnitIds = [
          ...(packet.answerUnits ?? []),
          ...(packet.contextUnits ?? []),
          ...(packet.supportUnits ?? []),
        ].map((unit) => unit.unitId);
        return lines.map((line, index) => ({
          lineId: `prompt_line:${packet.packetId}:${index + 1}`,
          packetId: packet.packetId,
          role: promptLineRole(line),
          line,
          sourceRefs: packet.allSourceRefs ?? packet.sourceRefs,
          normalizedSourceRefs: normalizeSourceRefs(packet.allSourceRefs ?? packet.sourceRefs),
          evidenceUnitIds,
        }));
      }),
    hardExclusions: params.packets
      .filter((packet) => (packet.hardExclusions?.length ?? 0) > 0)
      .map((packet) => ({ packetId: packet.packetId, reasons: packet.hardExclusions ?? [] })),
    softPenalties: params.packets
      .filter((packet) => (packet.softPenalties?.length ?? 0) > 0)
      .map((packet) => ({ packetId: packet.packetId, reasons: packet.softPenalties ?? [] })),
    scoreCurve: rankedPackets.slice(0, 16).map((packet, index) => ({
      rank: index + 1,
      packetId: packet.packetId,
      injected: packet.injected ?? params.injectedPacketIds.has(packet.packetId),
      finalScore: packet.grade?.finalScore,
      score: packet.score,
    })),
    hiddenExactDuplicates: params.packets.flatMap((packet) => packet.hiddenExactDuplicates ?? []),
  };
}

export function assembleEvidencePackets(input: EvidenceAssemblerInput): EvidenceAssemblerResult {
  const plan = input.queryAnalysis.evidencePlan;
  const now = input.now ?? new Date().toISOString();

  const activeEntries = input.promptEvidence.filter((entry) => !entry.dropReason);
  const grouped = groupEntries(activeEntries);
  const groupedPackets = [...grouped.values()]
    .map((entries) =>
      packetFromGroup({
        queryAnalysis: input.queryAnalysis,
        entries,
        allEntries: activeEntries,
        now,
      }),
    )
    .filter((packet): packet is EvidencePacket => Boolean(packet));
  const allPackets = [
    ...groupedPackets,
    ...highLevelSupportPackets({
      queryAnalysis: input.queryAnalysis,
      candidateGenerationResult: input.candidateGenerationResult,
    }),
  ];
  const deduped = allPackets;
  const injectedPacketIds = suppressDominatedInjectedPackets(
    input.queryAnalysis,
    deduped,
    selectInjectedPackets(input.queryAnalysis, deduped),
  );
  const packets = deduped.map((packet) => {
    const injected = injectedPacketIds.has(packet.packetId);
    const dropReason = injected ? undefined : nonInjectedDropReason(input.queryAnalysis, packet);
    return {
      ...packet,
      role: injected ? ("answer" as const) : packet.role,
      protected: injected,
      injected,
      protectionReason: injected
        ? (packet.selectionReason ?? "packet-final-score")
        : packet.protectionReason,
      dropReason,
      coverage: packet.coverage,
    };
  });
  const markedPromptEvidence = markPromptEvidenceFromPackets({
    promptEvidence: input.promptEvidence,
    packets,
    injectedPacketIds,
  });
  return {
    packets,
    promptEvidence: markedPromptEvidence,
    audit: plan
      ? evidencePacketAudit({
          operation: plan.operation,
          slots: plan.slots,
          packets,
          promptEvidence: markedPromptEvidence,
          injectedPacketIds,
          candidateGenerationResult: input.candidateGenerationResult,
        })
      : undefined,
  };
}
