import { clamp01, isValidEntityName, normalizeText, stableHash, truncateText } from "../support.js";
import type {
  AbstractionCandidateRecord,
  AbstractionCandidateStage,
  ConversationChunk,
  ConversationTaskPhase,
  MemoryAction,
  MemoryCandidate,
  MemoryGuidanceFacet,
  MemoryAdviceSignalHint,
  MemoryCandidateDecisionHint,
  MemoryCandidatePreferenceHint,
  MemoryCandidateRelationHint,
  MemoryCandidateWorkflowHint,
  MemoryGuidanceType,
  MemoryPluginConfig,
  MemoryPrimaryRouteType,
  MemoryRecallPlan,
  MemoryResourceAssertionHint,
  MemoryRouteType,
  MemoryWorkflowStateKind,
  QueryCompileResult,
  RouteDecision,
  RouteEvidenceCandidate,
  RouteEvidenceDecision,
  RoutePriorDecision,
  SearchHit,
  SynthesizedTaskEvent,
  TaskAssignmentDecision,
  TaskAssignmentSnapshot,
  TurnCaptureMessage,
  TurnSemanticAssertionFamilyHint,
  TurnSemanticFrame,
  TurnSemanticRelationDraft,
  TurnSemanticTaskProposal,
  TurnSemanticTimeframeHint,
  MemxLogger,
  MemoryCallProvenance,
  MemoryLlmBudgetAudit,
  MemoryLlmCallStage,
} from "../types.js";
import { sanitizeTaskMetadata } from "./authority.js";
import { canonicalAttributeSlot } from "./attributeSlots.js";
import {
  loadJudgeModelConfig,
  type JudgeModelConfig,
  type SupportedJudgeProvider,
} from "./judgeModelConfig.js";
import { recordMemoryLlmBudgetCall } from "./llmBudgetAudit.js";
import { buildQueryCompilerPromptInput } from "./queryCompiler.js";
import { canonicalStateKey } from "./semantic/heuristics.js";
import { basicSemanticSimilarity } from "./semantic/textSimilarity.js";
import { canonicalizePreferencePredicate, normalizeGraphRelationType } from "./semantics.js";
import { assessAssistantChunk, renderTaskPromptChunk } from "./sourceWeighting.js";
import type { TaskSummaryEvidenceSet } from "./taskSummary.js";
import {
  buildTurnSemanticCompilerInput,
  type LongTurnSemanticScanInput,
} from "./turnSemanticCompiler.js";

export type ReasonerExecutionMode = "llm" | "degraded" | "disabled";

export type ReasonerTraceEntry = {
  label: string;
  mode: ReasonerExecutionMode;
  provenance: MemoryCallProvenance;
  detail: string;
  stage?: MemoryLlmCallStage;
  provider?: SupportedJudgeProvider;
  model?: string;
  at: string;
};

type ReasonerCallAuditOptions = {
  stage?: MemoryLlmCallStage;
  audit?: MemoryLlmBudgetAudit;
  maxTokens?: number;
  jsonMode?: boolean;
  temperature?: number;
  signal?: AbortSignal;
};

function shouldRetrySemanticCompile(options: ReasonerCallAuditOptions): boolean {
  return (
    (options.stage === "post_answer_writeback" || options.stage === "maintenance_async") &&
    !options.signal?.aborted
  );
}

export type DedupDecision = {
  action: "NEW" | "DUPLICATE" | "UPDATE";
  targetIndex?: number;
  mergedSummary?: string;
  reason: string;
};

export type OutcomePromotionDecision = {
  shouldPromote: boolean;
  promotionScore: number;
  reason: string;
};

type RelevantDecision = {
  relevant: number[];
  sufficient: boolean;
  reason: string;
};

export type TaskSummary = {
  title: string;
  summary: string;
  metadataJson: Record<string, unknown>;
  synthesizedEvent?: Omit<SynthesizedTaskEvent, "evidenceChunkIds" | "outcomeKey"> & {
    evidenceChunkIndexes: number[];
  };
};

type TaskAssignmentJudgeResult = {
  decision?: string;
  targetTaskId?: string;
  confidence?: number;
  reason?: string;
};

type RecallPlanDecision = {
  shouldRecall: boolean;
  focusedQuery: string;
  reason: string;
  routeHint?: string;
};

type TaskSummaryJudgeResult = {
  title?: string;
  summary?: string;
  project?: string;
  currentTask?: string;
  nextAction?: string;
  blocker?: string;
  taskPhase?: string;
  candidateResolution?: string;
  closureScore?: number;
  verificationScore?: number;
  contradictionRisk?: number;
  eventType?: string;
  eventSummary?: string;
  evidenceChunkIndexes?: number[];
};

type TaskSummaryBatchJudgeResult = {
  tasks?: Array<
    TaskSummaryJudgeResult & {
      taskId?: string;
    }
  >;
};

type OutcomePromotionJudgeResult = {
  shouldPromote?: boolean;
  promotionScore?: number;
  reason?: string;
};

export type AbstractionCandidateJudgeResult = {
  summary?: string;
  displayName?: string | null;
  stage?: AbstractionCandidateStage;
  reason?: string;
};

export type CandidatePolicyJudgeResult = {
  action?: MemoryAction;
  salienceScore?: number;
  expectedFutureUtility?: number;
  stabilityScore?: number;
  preference?: MemoryCandidatePreferenceHint | null;
  workflow?: MemoryCandidateWorkflowHint | null;
  workflows?: MemoryCandidateWorkflowHint[] | null;
  relation?: MemoryCandidateRelationHint | null;
  relations?: MemoryCandidateRelationHint[] | null;
  decision?: MemoryCandidateDecisionHint | null;
  reason?: string;
};

const PRIMARY_ROUTE_TYPES: MemoryPrimaryRouteType[] = [
  "workflow",
  "factual",
  "temporal",
  "explanatory",
] as const;

function isPrimaryRouteType(value: unknown): value is MemoryPrimaryRouteType {
  return (
    value === "workflow" || value === "factual" || value === "temporal" || value === "explanatory"
  );
}

function isRouteType(value: unknown): value is MemoryRouteType {
  return isPrimaryRouteType(value) || value === "mixed" || value === "unknown";
}

function isTaskPhase(value: unknown): value is ConversationTaskPhase {
  return (
    value === "investigating" ||
    value === "proposed" ||
    value === "attempting" ||
    value === "validated" ||
    value === "resolved" ||
    value === "reopened"
  );
}

function isMemoryAction(value: unknown): value is MemoryAction {
  return (
    value === "ignore" ||
    value === "session_state" ||
    value === "durable_state" ||
    value === "stable_fact" ||
    value === "episodic_event" ||
    value === "graph_relation"
  );
}

function isAbstractionCandidateJudgeStage(value: unknown): value is AbstractionCandidateStage {
  return value === "candidate" || value === "probationary" || value === "quarantined";
}

function normalizePreferenceHint(value: unknown): MemoryCandidatePreferenceHint | undefined {
  if (!value || typeof value !== "object") {
    return undefined;
  }
  const record = value as Record<string, unknown>;
  if (typeof record.predicate !== "string" || typeof record.object !== "string") {
    return undefined;
  }
  const predicate = canonicalizePreferencePredicate(record.predicate);
  if (!predicate) {
    return undefined;
  }
  return {
    predicate,
    object: record.object.trim(),
    guidance: normalizeGuidanceFacet(record.guidance),
    confidence:
      typeof record.confidence === "number" && Number.isFinite(record.confidence)
        ? clamp01(record.confidence)
        : undefined,
    reason: typeof record.reason === "string" ? record.reason.trim() : undefined,
  };
}

function isGuidanceType(value: unknown): value is MemoryGuidanceType {
  return (
    value === "language" ||
    value === "style" ||
    value === "charset" ||
    value === "output_order" ||
    value === "generic_preference"
  );
}

function normalizeGuidanceFacet(value: unknown): MemoryGuidanceFacet | undefined {
  if (!value || typeof value !== "object") {
    return undefined;
  }
  const record = value as Record<string, unknown>;
  if (!isGuidanceType(record.guidanceType) || typeof record.guidanceText !== "string") {
    return undefined;
  }
  const guidanceText = record.guidanceText.trim();
  if (!guidanceText) {
    return undefined;
  }
  return {
    guidanceType: record.guidanceType,
    guidanceText,
    confidence:
      typeof record.confidence === "number" && Number.isFinite(record.confidence)
        ? clamp01(record.confidence)
        : undefined,
    reason: typeof record.reason === "string" ? record.reason.trim() : undefined,
  };
}

function normalizeWorkflowHint(value: unknown): MemoryCandidateWorkflowHint | undefined {
  if (!value || typeof value !== "object") {
    return undefined;
  }
  const record = value as Record<string, unknown>;
  if (
    typeof record.key !== "string" ||
    !record.value ||
    typeof record.value !== "object" ||
    Array.isArray(record.value)
  ) {
    return undefined;
  }
  return {
    key: canonicalStateKey(record.key.trim()),
    value: record.value as Record<string, unknown>,
    stateKind:
      record.stateKind === "session" || record.stateKind === "durable"
        ? (record.stateKind as MemoryWorkflowStateKind)
        : undefined,
    confidence:
      typeof record.confidence === "number" && Number.isFinite(record.confidence)
        ? clamp01(record.confidence)
        : undefined,
    reason: typeof record.reason === "string" ? record.reason.trim() : undefined,
  };
}

function normalizeWorkflowHints(value: unknown): MemoryCandidateWorkflowHint[] {
  if (!Array.isArray(value)) {
    return [];
  }
  const seen = new Set<string>();
  const normalized: MemoryCandidateWorkflowHint[] = [];
  for (const entry of value) {
    const workflow = normalizeWorkflowHint(entry);
    if (!workflow) {
      continue;
    }
    const dedupKey = `${workflow.key}:${JSON.stringify(workflow.value)}`;
    if (seen.has(dedupKey)) {
      continue;
    }
    seen.add(dedupKey);
    normalized.push(workflow);
  }
  return normalized;
}

function normalizeRelationHint(value: unknown): MemoryCandidateRelationHint | undefined {
  if (!value || typeof value !== "object") {
    return undefined;
  }
  const record = value as Record<string, unknown>;
  if (
    typeof record.subject !== "string" ||
    typeof record.predicate !== "string" ||
    typeof record.object !== "string"
  ) {
    return undefined;
  }
  const normalizedPredicate = normalizeGraphRelationType(record.predicate);
  if (!normalizedPredicate) {
    return undefined;
  }
  const predicateLabel = normalizeText(record.predicate).replace(/[^\p{L}\p{N}]+/gu, "_");
  const inverseStructuredPredicate = new Set(["owned_by", "blocked_by"]).has(predicateLabel);
  const subject = inverseStructuredPredicate ? record.object : record.subject;
  const object = inverseStructuredPredicate ? record.subject : record.object;
  const relationSlot =
    typeof record.slot === "string" && record.slot.trim()
      ? normalizeText(record.slot)
          .replace(/[^\p{L}\p{N}]+/gu, "_")
          .replace(/^_+|_+$/g, "")
      : typeof record.relationSlot === "string" && record.relationSlot.trim()
        ? normalizeText(record.relationSlot)
            .replace(/[^\p{L}\p{N}]+/gu, "_")
            .replace(/^_+|_+$/g, "")
        : undefined;
  return {
    subject: subject.trim(),
    predicate: normalizedPredicate.relationType,
    object: object.trim(),
    sourceRef: typeof record.sourceRef === "string" ? record.sourceRef.trim() : undefined,
    polarity: record.polarity === "negated" || record.negated === true ? "negated" : "affirmed",
    rawPredicate:
      typeof record.rawPredicate === "string" && record.rawPredicate.trim()
        ? record.rawPredicate.trim()
        : normalizedPredicate.rawPredicate,
    ...(relationSlot ? { relationSlot } : {}),
    confidence:
      typeof record.confidence === "number" && Number.isFinite(record.confidence)
        ? clamp01(record.confidence)
        : undefined,
    reason: typeof record.reason === "string" ? record.reason.trim() : undefined,
  };
}

function normalizeRelationHints(value: unknown): MemoryCandidateRelationHint[] {
  if (!Array.isArray(value)) {
    return [];
  }
  const seen = new Set<string>();
  const normalized: MemoryCandidateRelationHint[] = [];
  for (const entry of value) {
    const relation = normalizeRelationHint(entry);
    if (!relation) {
      continue;
    }
    const dedupKey = `${normalizeText(relation.subject)}:${relation.predicate}:${relation.polarity ?? "affirmed"}:${normalizeText(relation.object)}`;
    if (seen.has(dedupKey)) {
      continue;
    }
    seen.add(dedupKey);
    normalized.push(relation);
  }
  return normalized;
}

function normalizeDecisionHint(value: unknown): MemoryCandidateDecisionHint | undefined {
  if (!value || typeof value !== "object") {
    return undefined;
  }
  const record = value as Record<string, unknown>;
  if (typeof record.summary !== "string") {
    return undefined;
  }
  return {
    summary: record.summary.trim(),
    confidence:
      typeof record.confidence === "number" && Number.isFinite(record.confidence)
        ? clamp01(record.confidence)
        : undefined,
    reason: typeof record.reason === "string" ? record.reason.trim() : undefined,
  };
}

export type ReasonerProbeReport = {
  enabled: boolean;
  resolvedConfigPath: string | null;
  provider: SupportedJudgeProvider | null;
  model: string | null;
  traces: ReasonerTraceEntry[];
  outputs: {
    chunkSummary: string;
    taskTitle: string;
    taskSummary: string;
    topicDecision: boolean | null;
    dedupAction: string | null;
    relevant: number[];
    recallPlan: string;
  };
};

function normalizeOpenAiEndpoint(baseUrl: string): string {
  const trimmed = baseUrl.replace(/\/+$/, "");
  if (trimmed.endsWith("/chat/completions")) {
    return trimmed;
  }
  if (trimmed.endsWith("/responses")) {
    return trimmed;
  }
  return `${trimmed}/chat/completions`;
}

function normalizeAnthropicEndpoint(baseUrl: string): string {
  const trimmed = baseUrl.replace(/\/+$/, "");
  return trimmed.endsWith("/messages") ? trimmed : `${trimmed}/messages`;
}

function normalizeGoogleEndpoint(baseUrl: string, model: string, apiKey: string): string {
  const trimmed = baseUrl.replace(/\/+$/, "");
  if (trimmed.includes(":generateContent")) {
    return trimmed.includes("?key=") ? trimmed : `${trimmed}?key=${encodeURIComponent(apiKey)}`;
  }
  return `${trimmed}/models/${encodeURIComponent(model)}:generateContent?key=${encodeURIComponent(apiKey)}`;
}

function normalizeOllamaEndpoint(baseUrl: string): string {
  const trimmed = baseUrl.replace(/\/+$/, "");
  return trimmed.endsWith("/api/chat") ? trimmed : `${trimmed}/api/chat`;
}

function extractTextContent(payload: unknown): string {
  if (typeof payload === "string") {
    return payload.trim();
  }
  if (Array.isArray(payload)) {
    return payload
      .map((entry) => extractTextContent(entry))
      .filter(Boolean)
      .join("\n")
      .trim();
  }
  if (!payload || typeof payload !== "object") {
    return "";
  }
  const record = payload as Record<string, unknown>;
  if (typeof record.text === "string") {
    return record.text.trim();
  }
  if (typeof record.output_text === "string") {
    return record.output_text.trim();
  }
  if (record.type === "text" && typeof record.text === "string") {
    return record.text.trim();
  }
  if (typeof record.content === "string") {
    return record.content.trim();
  }
  if (Array.isArray(record.content)) {
    return extractTextContent(record.content);
  }
  return "";
}

function parseJsonResponse<T>(value: string): T | null {
  const direct = value.trim();
  if (!direct) {
    return null;
  }
  const balancedObject = extractBalancedJsonObject(direct);
  const candidates = [
    direct,
    direct
      .replace(/^```(?:json)?\s*/i, "")
      .replace(/```$/i, "")
      .trim(),
  ];
  if (balancedObject) {
    candidates.push(balancedObject);
  }
  for (const candidate of candidates) {
    if (!candidate) {
      continue;
    }
    try {
      return JSON.parse(candidate) as T;
    } catch {
      continue;
    }
  }
  return null;
}

function extractBalancedJsonObject(value: string): string | null {
  let depth = 0;
  let start = -1;
  let inString = false;
  let escaped = false;
  for (let index = 0; index < value.length; index += 1) {
    const char = value[index];
    if (escaped) {
      escaped = false;
      continue;
    }
    if (char === "\\") {
      escaped = true;
      continue;
    }
    if (char === '"') {
      inString = !inString;
      continue;
    }
    if (inString) {
      continue;
    }
    if (char === "{") {
      if (depth === 0) {
        start = index;
      }
      depth += 1;
      continue;
    }
    if (char === "}") {
      if (depth === 0) {
        continue;
      }
      depth -= 1;
      if (depth === 0 && start >= 0) {
        return value.slice(start, index + 1);
      }
    }
  }
  return null;
}

function summarizeParseFailure(raw: string): string {
  const trimmed = raw.trim();
  const balancedObject = extractBalancedJsonObject(trimmed);
  return [
    `len=${trimmed.length}`,
    `startsWithBrace=${trimmed.startsWith("{")}`,
    `endsWithBrace=${trimmed.endsWith("}")}`,
    `balancedObject=${Boolean(balancedObject)}`,
  ].join(" ");
}

function estimateTokenCount(text: string): number {
  const trimmed = text.trim();
  if (!trimmed) {
    return 0;
  }
  // Maintenance cost only needs an order-of-magnitude estimate. Char-count
  // keeps the audit provider-agnostic while remaining comparable run to run.
  return Math.max(1, Math.ceil(trimmed.length / 4));
}

function localChunkPreview(content: string): string {
  const flattened = content
    .replace(/```[\s\S]*?```/g, " code block ")
    .replace(/\s+/g, " ")
    .trim();
  if (!flattened) {
    return "";
  }
  const sentenceCut = flattened.search(/[。.!?！？]/u);
  if (sentenceCut > 16) {
    return truncateText(flattened.slice(0, sentenceCut + 1).trim(), 180);
  }
  return truncateText(flattened, 180);
}

export function sanitizeChunkSummaryResult(
  text: string,
  role: ConversationChunk["role"] = "user",
  result?: ChunkSummaryJudgeResult | null,
): string {
  const fallback = localChunkPreview(text);
  const summary = result?.summary?.trim();
  if (!summary) {
    return fallback;
  }
  if (role === "user" && result?.summaryKind === "answer" && result.answerSupported !== true) {
    return fallback;
  }
  return truncateText(summary, 180);
}

function conservativeDedupDecision(
  newSummary: string,
  candidates?: Array<{ index: number; summary: string; text?: string }>,
): DedupDecision {
  // Simple word-overlap (Jaccard) check to merge near-identical chunks
  if (candidates && candidates.length > 0) {
    const newTokens = new Set(
      newSummary
        .toLowerCase()
        .split(/[\s,.:;!?，。！？；：]+/u)
        .filter((t) => t.length > 1),
    );
    if (newTokens.size > 0) {
      for (const candidate of candidates) {
        const candidateTokens = new Set(
          candidate.summary
            .toLowerCase()
            .split(/[\s,.:;!?，。！？；：]+/u)
            .filter((t) => t.length > 1),
        );
        if (candidateTokens.size === 0) continue;
        let intersection = 0;
        for (const token of newTokens) {
          if (candidateTokens.has(token)) intersection++;
        }
        const union = new Set([...newTokens, ...candidateTokens]).size;
        const jaccard = union > 0 ? intersection / union : 0;
        if (jaccard >= 0.7) {
          return {
            action: "UPDATE",
            targetIndex: candidate.index,
            mergedSummary: truncateText(newSummary.trim(), 220) || undefined,
            reason: `local dedup: Jaccard=${jaccard.toFixed(2)} merge with existing chunk`,
          };
        }
      }
    }
  }
  return {
    action: "NEW",
    mergedSummary: truncateText(newSummary.trim(), 220) || undefined,
    reason: "degraded dedup: keep the new memory separate until an LLM judgment is available",
  };
}

function conservativeRelevantDecision(): RelevantDecision {
  return {
    relevant: [],
    sufficient: false,
    reason: "degraded relevance filter: do not select evidence without an LLM judgment",
  };
}

function conservativeRecallPlan(
  query: string,
  judgmentMode: "degraded" | "disabled",
): MemoryRecallPlan {
  const focusedQuery = truncateText(query.trim(), 160);
  const reason = focusedQuery
    ? judgmentMode === "disabled"
      ? "recall planning unavailable: no LLM recall plan is configured"
      : "degraded recall planning: no LLM recall plan is available"
    : "empty query";
  return {
    shouldRecall: Boolean(focusedQuery),
    focusedQuery,
    reason,
    judgmentMode,
  };
}

function conservativeRoutePrior(
  query: string,
  judgmentMode: "degraded" | "disabled",
): RoutePriorDecision {
  const focusedQuery = truncateText(query.trim(), 180);
  const focusedQueries = focusedQuery
    ? (Object.fromEntries(
        PRIMARY_ROUTE_TYPES.map((route) => [route, focusedQuery]),
      ) as RoutePriorDecision["focusedQueries"])
    : {};
  return {
    primaryRoute: "unknown",
    secondaryRoutes: [],
    confidence: 0.05,
    focusedQueries,
    reason: focusedQuery
      ? judgmentMode === "disabled"
        ? "route prior unavailable: no LLM route prior is configured"
        : "degraded route prior: no LLM route prior is available"
      : "degraded route prior: empty query",
    judgmentMode,
  };
}

function conservativeRouteEvidenceDecision(
  routeType: MemoryPrimaryRouteType,
  judgmentMode: "degraded" | "disabled",
): RouteEvidenceDecision {
  return {
    relevant: [],
    sufficient: false,
    support: 0,
    reason:
      judgmentMode === "disabled"
        ? `route evidence unavailable for ${routeType}: defer to score-driven evidence support until a judge model is configured`
        : `degraded ${routeType} evidence: defer to score-driven evidence support until LLM judgment recovers`,
    judgmentMode,
  };
}

function conservativeOutcomePromotionDecision(): OutcomePromotionDecision {
  return {
    shouldPromote: false,
    promotionScore: 0,
    reason: "degraded promotion: keep the outcome provisional until an LLM judgment is available",
  };
}

function taskSummaryEvidenceSupport(
  text: string,
  chunks: ConversationChunk[],
): {
  supported: boolean;
  bestScore: number;
  groundedAssistantScore: number;
  userOrToolScore: number;
} {
  const query = text.trim();
  if (!query) {
    return {
      supported: false,
      bestScore: 0,
      groundedAssistantScore: 0,
      userOrToolScore: 0,
    };
  }
  let userOrToolScore = 0;
  let groundedAssistantScore = 0;
  for (const chunk of chunks) {
    const contentScore = basicSemanticSimilarity(query, chunk.content);
    const summaryScore = basicSemanticSimilarity(query, chunk.summary || chunk.content);
    const score = Math.max(contentScore, summaryScore);
    if (chunk.role === "assistant") {
      const assessment = assessAssistantChunk(chunk, chunks);
      groundedAssistantScore = Math.max(
        groundedAssistantScore,
        score * Math.max(0, assessment.weight),
      );
      continue;
    }
    userOrToolScore = Math.max(userOrToolScore, score);
  }
  const bestScore = Math.max(userOrToolScore, groundedAssistantScore);
  return {
    supported: userOrToolScore >= 0.42 || groundedAssistantScore >= 0.6,
    bestScore,
    groundedAssistantScore,
    userOrToolScore,
  };
}

function sanitizeResolvedTaskPhase(
  phase: ConversationTaskPhase,
  verificationScore: number,
  resolutionSupport:
    | {
        supported: boolean;
        bestScore: number;
      }
    | undefined,
  fallbackPhase: ConversationTaskPhase,
  chunks: ConversationChunk[],
): ConversationTaskPhase {
  if (phase !== "validated" && phase !== "resolved") {
    return phase;
  }
  const toolCount = chunks.filter((chunk) => chunk.role === "tool").length;
  if (resolutionSupport?.supported) {
    return phase;
  }
  if (verificationScore >= 0.72 && toolCount > 0) {
    return phase;
  }
  if (fallbackPhase === "proposed") {
    return toolCount > 0 ? "attempting" : "investigating";
  }
  if (fallbackPhase === "validated" || fallbackPhase === "resolved") {
    return toolCount > 0 ? "attempting" : "investigating";
  }
  return fallbackPhase;
}

type ChunkSummaryJudgeResult = {
  summary?: string;
  summaryKind?: "question" | "instruction" | "statement" | "answer" | "mixed";
  answerSupported?: boolean;
};

function buildChunkPrompt(
  text: string,
  role: ConversationChunk["role"] = "user",
): { system: string; user: string } {
  return {
    system:
      '你负责为记忆存储总结一段对话片段。请只返回严格 JSON：{"summary": string, "summaryKind": "question"|"instruction"|"statement"|"answer"|"mixed", "answerSupported": boolean}。summary 必须简短、客观、可检索，且不超过 180 个字符。若片段是用户提问，只能总结“用户问了什么”，不能替用户回答；除非答案文本已在同一片段中明确出现，否则 summaryKind 不得为 "answer"，answerSupported 必须为 false。若片段来自 assistant，优先概括可验证结论、决策或结果，不要复述长教程、示例代码或文件操作细节。',
    user: `片段角色：${role}\n\n对话片段：\n${truncateText(text, 5000)}`,
  };
}

function buildTaskPrompt(chunks: ConversationChunk[]): { system: string; user: string } {
  const transcript = chunks
    .map((chunk, index) => `${index + 1}. [${chunk.role}] ${renderTaskPromptChunk(chunk, chunks)}`)
    .join("\n");
  return {
    system:
      '你负责为记忆调度总结一个对话任务。请只返回严格 JSON：{"title": string, "summary": string, "project"?: string, "currentTask"?: string, "nextAction"?: string, "blocker"?: string, "taskPhase"?: "investigating"|"proposed"|"attempting"|"validated"|"resolved"|"reopened", "candidateResolution"?: string, "closureScore"?: number, "verificationScore"?: number, "contradictionRisk"?: number, "eventType"?: string, "eventSummary"?: string, "evidenceChunkIndexes"?: number[]}。taskPhase 描述当前任务所处阶段。只有当转录已经形成比较稳定的结果总结时，才填写 candidateResolution 或 eventSummary。只有在 eventSummary 直接得到 transcript 支撑时，才填写 evidenceChunkIndexes。优先依据 user 与 tool 证据；assistant 的长解释、教程、草稿和文件操作只算辅助上下文，除非它们被 user 或 tool 明确支撑，否则不要让它们主导 title、summary、currentTask、nextAction 或 candidateResolution。不要臆测问题已经解决。',
    user: `对话转录：\n${transcript}`,
  };
}

function buildTaskSummaryEvidencePrompt(evidence: TaskSummaryEvidenceSet): {
  system: string;
  user: string;
} {
  const transcript =
    evidence.chunks.length > 0
      ? evidence.chunks
          .slice(-16)
          .map(
            (chunk, index) =>
              `${index + 1}. [${chunk.role}] ${renderTaskPromptChunk(chunk, evidence.chunks)}`,
          )
          .join("\n")
      : "无";
  const eventBlock =
    evidence.linkedEvents.length > 0
      ? evidence.linkedEvents
          .map(
            (event, index) =>
              `${index + 1}. [${event.eventType}] ${truncateText(event.summary, 220)} @ ${event.observedAt}`,
          )
          .join("\n")
      : "无";
  const structuredContext = {
    ...(evidence.project ? { project: evidence.project } : {}),
    ...(evidence.currentTask ? { currentTask: evidence.currentTask } : {}),
    ...(evidence.nextAction ? { nextAction: evidence.nextAction } : {}),
    ...(evidence.blocker ? { blocker: evidence.blocker } : {}),
    ...(evidence.candidateResolution ? { candidateResolution: evidence.candidateResolution } : {}),
    ...(evidence.candidateResolutionPhase
      ? { candidateResolutionPhase: evidence.candidateResolutionPhase }
      : {}),
    ...(evidence.candidateResolutionEvidenceChunkIds.length > 0
      ? { candidateResolutionEvidenceChunkIds: evidence.candidateResolutionEvidenceChunkIds }
      : {}),
    ...(evidence.lastEmittedOutcomeKey
      ? { lastEmittedOutcomeKey: evidence.lastEmittedOutcomeKey }
      : {}),
    ...(evidence.compilerTaskSummary
      ? {
          compilerTaskSummary: evidence.compilerTaskSummary.summary,
          ...(typeof evidence.compilerTaskSummary.confidence === "number"
            ? { compilerTaskSummaryConfidence: evidence.compilerTaskSummary.confidence }
            : {}),
        }
      : {}),
    supportRefs: evidence.supportRefs,
  };
  return {
    system: [
      "你负责基于任务证据集重算一个稳定摘要（stable task summary）。",
      "你只能返回严格 JSON，不得输出任何解释文字，不得输出 markdown。",
      '返回格式必须是 {"title": string, "summary": string, "project"?: string, "currentTask"?: string, "nextAction"?: string, "blocker"?: string, "taskPhase"?: "investigating"|"proposed"|"attempting"|"validated"|"resolved"|"reopened", "candidateResolution"?: string, "closureScore"?: number, "verificationScore"?: number, "contradictionRisk"?: number, "eventType"?: string, "eventSummary"?: string, "evidenceChunkIndexes"?: number[]}。',
      "这是 evidence-backed recompute，不是润色旧 local summary。",
      "优先依据 transcript、tool 结果、structured task metadata、candidateResolution、linked events；不要把旧 task.summary 当主输入。",
      "不要发明新的事实、项目名、时间点、关系、解决结果或 outcome。",
      "只有当 transcript 已经支撑比较稳定的结果总结时，才填写 candidateResolution 或 eventSummary。",
      "只有在 eventSummary 直接得到 transcript 支撑时，才填写 evidenceChunkIndexes。",
      "summary 必须客观、稳定、便于 UI/检索；不要写成提问、计划口号或泛化策略。",
      "只输出 JSON。",
    ].join("\n"),
    user: [
      `taskId: ${evidence.taskId}`,
      `structuredContext: ${truncateText(JSON.stringify(structuredContext), 1800)}`,
      `linkedEvents:\n${eventBlock}`,
      `evidenceTranscript:\n${transcript}`,
    ].join("\n\n"),
  };
}

function buildTaskSummaryEvidenceBatchPrompt(evidenceSets: TaskSummaryEvidenceSet[]): {
  system: string;
  user: string;
} {
  const taskBlocks = evidenceSets
    .map((evidence, index) => {
      const transcript =
        evidence.chunks.length > 0
          ? evidence.chunks
              .slice(-8)
              .map(
                (chunk, chunkIndex) =>
                  `${chunkIndex + 1}. [${chunk.role}] ${truncateText(renderTaskPromptChunk(chunk, evidence.chunks), 220)}`,
              )
              .join("\n")
          : "无";
      const linkedEvents =
        evidence.linkedEvents.length > 0
          ? evidence.linkedEvents
              .slice(0, 6)
              .map(
                (event) =>
                  `- [${event.eventType}] ${truncateText(event.summary, 160)} @ ${event.observedAt}`,
              )
              .join("\n")
          : "无";
      return [
        `任务 ${index + 1} | taskId=${evidence.taskId}`,
        evidence.compilerTaskSummary
          ? `Compiler working summary: ${truncateText(evidence.compilerTaskSummary.summary, 160)}`
          : "",
        evidence.candidateResolution
          ? `Candidate resolution: ${truncateText(evidence.candidateResolution, 180)}`
          : "",
        evidence.project ? `Project: ${truncateText(evidence.project, 120)}` : "",
        evidence.currentTask ? `Current task: ${truncateText(evidence.currentTask, 160)}` : "",
        evidence.nextAction ? `Next action: ${truncateText(evidence.nextAction, 160)}` : "",
        evidence.blocker ? `Blocker: ${truncateText(evidence.blocker, 160)}` : "",
        evidence.lastEmittedOutcomeKey
          ? `Last emitted outcome: ${evidence.lastEmittedOutcomeKey}`
          : "",
        `Linked events:\n${linkedEvents}`,
        `Recent chunks:\n${transcript}`,
      ]
        .filter(Boolean)
        .join("\n");
    })
    .join("\n\n");
  return {
    system:
      '你负责基于任务证据集批量生成 stable task summary。请只返回严格 JSON：{"tasks":[{"taskId":string,"title":string,"summary":string,"project"?:string,"currentTask"?:string,"nextAction"?:string,"blocker"?:string,"taskPhase"?:string,"candidateResolution"?:string,"closureScore"?:number,"verificationScore"?:number,"contradictionRisk"?:number,"eventType"?:string,"eventSummary"?:string,"evidenceChunkIndexes"?:number[]}]}。要求：1) 每个任务都必须基于给定证据重新生成摘要，而不是润色旧摘要；2) 不要编造未被证据支持的 candidateResolution 或 eventSummary；3) 只在 evidence 明确支持时填写 evidenceChunkIndexes。',
    user: taskBlocks || "无任务",
  };
}

function buildTopicPrompt(
  currentContext: string,
  newMessage: string,
): { system: string; user: string } {
  return {
    system:
      '你负责判断新的用户消息是否开启了一个新的对话任务。请只返回严格 JSON：{"isNewTopic": boolean, "reason": string}。只有在主题或任务明显切换时才返回 true。',
    user: `当前任务上下文：\n${truncateText(currentContext, 4000)}\n\n新的用户消息：\n${truncateText(newMessage, 1000)}`,
  };
}

function buildTaskAssignmentPrompt(
  currentTask: TaskAssignmentSnapshot | null,
  candidates: TaskAssignmentSnapshot[],
  incomingTurn: string,
): { system: string; user: string } {
  const renderTask = (task: TaskAssignmentSnapshot) =>
    [
      `taskId: ${task.taskId}`,
      `status: ${task.status}`,
      task.isActive ? "isActive: true" : "",
      `title: ${truncateText(task.title, 160)}`,
      `summary: ${truncateText(task.summary, 260)}`,
      `metadata: ${truncateText(JSON.stringify(task.metadataJson), 320)}`,
      task.recentContext ? `recentContext: ${truncateText(task.recentContext, 360)}` : "",
      `updatedAt: ${task.updatedAt}`,
    ]
      .filter(Boolean)
      .join("\n");

  const currentBlock = currentTask ? renderTask(currentTask) : "无";
  const candidateBlock = candidates.length > 0 ? candidates.map(renderTask).join("\n\n") : "无";
  return {
    system:
      '你负责做任务归属判断。请只返回严格 JSON：{"decision":"continue"|"resume"|"new","targetTaskId"?:string,"confidence":number,"reason":string}。continue 表示继续当前 active task；resume 表示恢复某个候选历史任务；new 表示创建新任务。只有在新一轮消息与某个已有任务明显延续时，才返回 continue 或 resume。若 decision=resume，targetTaskId 必须来自候选列表。',
    user: `当前 active task：\n${currentBlock}\n\n最近任务候选：\n${candidateBlock}\n\n新一轮消息：\n${truncateText(incomingTurn, 1600)}`,
  };
}

function buildAbstractionCandidatePrompt(candidate: AbstractionCandidateRecord): {
  system: string;
  user: string;
} {
  return {
    system:
      '你负责为已经由确定性信号支持的抽象候选做可选 refinement。请只返回严格 JSON：{"summary"?: string, "displayName"?: string, "stage"?: "candidate"|"probationary"|"quarantined", "reason": string}。规则：1) 不能凭空创造新的抽象、证据或关系；2) 只能在当前候选的基础上微调命名、总结或边界阶段；3) 除非证据明显冲突或抽象明显过度，不要返回 quarantined；4) 不要返回 active、decaying 或 superseded；5) displayName 只在 concept_candidate、graph_hypothesis 或 outcome_hypothesis 时填写，控制在 2 到 6 个词；6) summary 必须客观、可检索，并且少于 220 个字符；7) 如果不需要调整，也要返回 reason，但可以省略其他字段。',
    user: [
      `abstractionType: ${candidate.abstractionType}`,
      `semanticKey: ${candidate.semanticKey}`,
      `currentStage: ${candidate.stage}`,
      `confidence: ${candidate.confidence.toFixed(2)}`,
      `usefulnessScore: ${candidate.usefulnessScore.toFixed(2)}`,
      `stabilityScore: ${candidate.stabilityScore.toFixed(2)}`,
      `contradictionScore: ${candidate.contradictionScore.toFixed(2)}`,
      `currentSummary: ${truncateText(candidate.summary, 260)}`,
      `supportContentRefs: ${truncateText(JSON.stringify(candidate.supportContentRefs.slice(0, 8)), 500)}`,
      `supportBeliefIds: ${truncateText(JSON.stringify(candidate.supportBeliefIds.slice(0, 8)), 500)}`,
      `metadata: ${truncateText(JSON.stringify(candidate.metadataJson), 1400)}`,
    ].join("\n"),
  };
}

function buildOutcomePromotionPrompt(params: {
  task: TaskAssignmentSnapshot;
  proposal: SynthesizedTaskEvent;
  evidenceChunks: ConversationChunk[];
}): { system: string; user: string } {
  const evidenceBlock =
    params.evidenceChunks.length > 0
      ? params.evidenceChunks
          .map(
            (chunk, index) => `${index + 1}. [${chunk.role}] ${truncateText(chunk.content, 420)}`,
          )
          .join("\n")
      : "无";
  return {
    system:
      '你负责判断一个任务结果总结是否已经足够稳定，可以提升为正式事件记忆。请只返回严格 JSON：{"shouldPromote": boolean, "promotionScore": number, "reason": string}。只有当结果已经形成较稳定的闭环，并且有 transcript 证据支撑，而不是单纯的建议、计划、礼貌回应或未经验证的推测时，才返回 shouldPromote=true。',
    user:
      `任务：\n` +
      `taskId: ${params.task.taskId}\n` +
      `title: ${truncateText(params.task.title, 160)}\n` +
      `summary: ${truncateText(params.task.summary, 240)}\n` +
      `metadata: ${truncateText(JSON.stringify(params.task.metadataJson), 320)}\n\n` +
      `候选结果事件：\n` +
      `eventType: ${params.proposal.eventType}\n` +
      `phase: ${params.proposal.phase}\n` +
      `summary: ${truncateText(params.proposal.summary, 260)}\n` +
      `closureScore: ${params.proposal.closureScore.toFixed(2)}\n` +
      `verificationScore: ${params.proposal.verificationScore.toFixed(2)}\n` +
      `contradictionRisk: ${params.proposal.contradictionRisk.toFixed(2)}\n` +
      `confidence: ${params.proposal.confidence.toFixed(2)}\n\n` +
      `相关 transcript 证据：\n${evidenceBlock}`,
  };
}

function buildCandidatePolicyPrompt(
  candidate: MemoryCandidate,
  config: Pick<
    MemoryPluginConfig,
    "minSalienceDurable" | "minSalienceSession" | "minUtilityForGraph"
  >,
): { system: string; user: string } {
  const hintBlock = JSON.stringify(
    {
      sourceKind: candidate.source.kind,
      eventType: candidate.eventType ?? null,
      toolName: candidate.source.toolName ?? null,
      entities: candidate.structuredHints?.entities ?? [],
      timeHints: candidate.structuredHints?.timeHints ?? [],
      metadata: candidate.metadata ?? {},
    },
    null,
    2,
  );
  const thresholdBlock = JSON.stringify(
    {
      minSalienceDurable: Number(config.minSalienceDurable.toFixed(2)),
      minSalienceSession: Number(config.minSalienceSession.toFixed(2)),
      minUtilityForGraph: Number(config.minUtilityForGraph.toFixed(2)),
      episodicEventMinSalience: Number(Math.min(config.minSalienceSession, 0.35).toFixed(2)),
    },
    null,
    2,
  );
  return {
    system:
      "你负责为单条记忆候选做语义裁决。请只返回严格 JSON：" +
      '{"action":"ignore"|"session_state"|"durable_state"|"stable_fact"|"episodic_event"|"graph_relation",' +
      '"salienceScore":number,"expectedFutureUtility":number,"stabilityScore":number,' +
      '"preference"?:{"predicate":"prefers_"|"uses_"|"has_"|"depends_" + topic,"object":string,"guidance"?:{"guidanceType":"language"|"style"|"charset"|"output_order"|"generic_preference","guidanceText":string},"confidence":number,"reason":string}|null,' +
      '"workflows"?:[{"key":string,"value":object,"stateKind"?:"session"|"durable","confidence":number,"reason":string}],' +
      '"relations"?:[{"subject":string,"predicate":string,"rawPredicate"?:string,"slot"?:string,"object":string,"confidence":number,"reason":string}],' +
      '"relation"?:{"subject":string,"predicate":string,"rawPredicate"?:string,"slot"?:string,"object":string,"confidence":number,"reason":string}|null,' +
      '"decision"?:{"summary":string,"confidence":number,"reason":string}|null,' +
      '"reason":string}。\n\n' +
      "判断规则：\n" +
      "1. ignore：提问、闲聊、无复用价值、缺乏可验证语义\n" +
      "2. session_state/durable_state：工作状态（项目、任务、卡点、下一步）；durable 仅限跨会话值得保留\n" +
      "3. stable_fact：稳定偏好、用户资料、长期约束；如影响回答方式/语言/风格，同时返回 guidance\n" +
      "4. graph_relation：命名实体间明确关系，必须返回 relation 或 relations；一段文本若包含多条明确关系，优先使用 relations。若关系表示项目画像中的组件角色，可附带 slot，例如 broker、primary_db、cache\n" +
      "4.1. 若文本省略了项目主语，但 metadata.currentProject 或 metadata.currentProjectProfile 已明确给出当前项目，可将该项目作为关系或 project.<code> workflow 的主语；若用户表达了替换/迁移/移除组件，优先输出替换后的 components 或 uses[slot] 关系，不要把 action/target/replacement 之类自由文本塞进项目档案\n" +
      "5. episodic_event：一次性结果、执行输出、历史经过\n" +
      "6. 根据整体语义判断，不依赖表面关键词；禁止臆测；workflows.value 排除格式约束和礼貌语\n" +
      "7. preference.predicate 必须以 prefers_/uses_/has_/depends_ 开头，后接下划线连接的名词 topic，例如 prefers_code_style、uses_spring_boot、has_senior_background\n\n" +
      "评分区间：\n" +
      "salienceScore: 0-0.24 忽略 | 0.45-0.71 session | 0.72-1.0 durable\n" +
      "expectedFutureUtility: 0-0.29 极低 | 0.55-0.74 常用 | 0.75-1.0 高复用\n" +
      "stabilityScore: 0-0.34 易变 | 0.60-0.79 稳定 | 0.80-1.0 很稳定\n\n" +
      "边界不清时优先 ignore 或 session_state，不要虚高打分。",
    user:
      `示例（仅供参考格式）：\n\n` +
      `输入："之后默认给我中英双语回答"\n` +
      `输出：{"action":"stable_fact","salienceScore":0.82,"expectedFutureUtility":0.88,"stabilityScore":0.85,"preference":{"predicate":"prefers_language","object":"bilingual responses","guidance":{"guidanceType":"language","guidanceText":"用户偏好中英双语回答"},"confidence":0.88,"reason":"明确语言偏好"},"reason":"长期回答语言偏好"}\n\n` +
      `输入："我用 Palantir Java 风格，4 空格缩进"\n` +
      `输出：{"action":"stable_fact","salienceScore":0.80,"expectedFutureUtility":0.85,"stabilityScore":0.88,"preference":{"predicate":"prefers_code_style","object":"palantir java style 4-space indent","guidance":{"guidanceType":"generic_preference","guidanceText":"用户使用 Palantir Java 风格，4 空格缩进"},"confidence":0.86,"reason":"代码风格偏好"},"reason":"长期代码风格偏好"}\n\n` +
      `输入："我上个月在 charity event 认识了 Tom，这周想再约他喝咖啡。你觉得合适吗？"\n` +
      `输出：{"action":"graph_relation","salienceScore":0.74,"expectedFutureUtility":0.72,"stabilityScore":0.46,"relations":[{"subject":"user","predicate":"related_to","rawPredicate":"met","object":"Tom","confidence":0.84,"reason":"自然对话里明确提到认识了某个联系人"}],"reason":"同一句里既有提问也有可复用的人际关系；提问部分不应抹掉关系信号"}\n\n` +
      `输入："当前任务是做 retrieval routing，卡点是 graph expansion"\n` +
      `输出：{"action":"session_state","salienceScore":0.62,"expectedFutureUtility":0.55,"stabilityScore":0.38,"workflows":[{"key":"workflow.current_task","value":{"task":"retrieval routing"},"stateKind":"session","confidence":0.78,"reason":"当前任务"},{"key":"workflow.blocker","value":{"blocker":"graph expansion","status":"blocked"},"stateKind":"session","confidence":0.76,"reason":"卡点"}],"reason":"workflow 状态"}\n\n` +
      `输入："帮我记住以下信息——我的项目代号是 Orion，版本号 3.2.1，上线日期是 2025 年 3 月 15 日。"\n` +
      `输出：{"action":"durable_state","salienceScore":0.88,"expectedFutureUtility":0.86,"stabilityScore":0.9,"workflows":[{"key":"project.Orion","value":{"projectCode":"Orion","version":"3.2.1","launchDate":"2025-03-15"},"stateKind":"durable","confidence":0.9,"reason":"项目画像主档案"}],"reason":"长期项目画像信息"}\n\n` +
      `输入："Orion 项目用的数据库是 MongoDB，消息队列是 RabbitMQ，缓存是 Redis。"\n` +
      `输出：{"action":"graph_relation","salienceScore":0.86,"expectedFutureUtility":0.84,"stabilityScore":0.88,"relations":[{"subject":"Orion","predicate":"uses","slot":"primary_db","object":"MongoDB","confidence":0.9,"reason":"项目数据库"},{"subject":"Orion","predicate":"uses","slot":"broker","object":"RabbitMQ","confidence":0.9,"reason":"项目消息队列"},{"subject":"Orion","predicate":"uses","slot":"cache","object":"Redis","confidence":0.9,"reason":"项目缓存"}],"reason":"项目组件画像"}\n\n` +
      `输入："我们已经把 RabbitMQ 换成 Kafka 了。"\n` +
      `上下文元数据：{"currentProject":"Orion","currentProjectProfile":{"projectCode":"Orion","components":{"broker":"RabbitMQ","primary_db":"MongoDB","cache":"Redis"}}}\n` +
      `输出：{"action":"durable_state","salienceScore":0.84,"expectedFutureUtility":0.83,"stabilityScore":0.82,"workflows":[{"key":"project.Orion","value":{"projectCode":"Orion","components":{"broker":"Kafka"}},"stateKind":"durable","confidence":0.88,"reason":"当前项目的 broker 已替换"}],"relations":[{"subject":"Orion","predicate":"uses","slot":"broker","object":"Kafka","confidence":0.87,"reason":"当前项目消息队列已更新"}],"reason":"项目组件替换更新"}\n\n` +
      `输入："api gateway 依赖 db router，而 db router 又依赖 primary db"\n` +
      `输出：{"action":"graph_relation","salienceScore":0.78,"expectedFutureUtility":0.79,"stabilityScore":0.74,"relations":[{"subject":"api gateway","predicate":"depends_on","object":"db router","confidence":0.84,"reason":"服务依赖关系"},{"subject":"db router","predicate":"depends_on","object":"primary db","confidence":0.83,"reason":"服务依赖关系"}],"reason":"同一句中包含两条明确依赖关系"}\n\n` +
      `输入："你好，今天天气怎么样？"\n` +
      `输出：{"action":"ignore","salienceScore":0.08,"expectedFutureUtility":0.05,"stabilityScore":0.10,"reason":"闲聊无记忆价值"}\n\n` +
      `---\n\n` +
      `候选文本：\n${truncateText(candidate.rawText, 2000)}\n\n` +
      `当前策略阈值：\n${thresholdBlock}\n\n` +
      `上下文元数据：\n${truncateText(hintBlock, 1800)}`,
  };
}
function buildDedupPrompt(
  newSummary: string,
  sourceText: string,
  candidates: Array<{ index: number; summary: string; text?: string }>,
): { system: string; user: string } {
  const candidateBlock = candidates
    .map((candidate) =>
      [
        `Candidate ${candidate.index}:`,
        `summary: ${truncateText(candidate.summary, 240)}`,
        candidate.text ? `source: ${truncateText(candidate.text, 300)}` : "",
      ]
        .filter(Boolean)
        .join("\n"),
    )
    .join("\n\n");
  return {
    system:
      '你负责判断记忆去重。请只返回严格 JSON：{"action": "NEW"|"DUPLICATE"|"UPDATE", "targetIndex"?: number, "mergedSummary"?: string, "reason": string}。只有当新片段是在细化已有记忆、并且应该用更好的合并摘要替换旧内容时，才使用 UPDATE。',
    user: `新片段摘要：\n${truncateText(newSummary, 240)}\n\n新片段原文：\n${truncateText(sourceText, 500)}\n\n候选记忆：\n${candidateBlock}`,
  };
}

function buildRelevancePrompt(
  query: string,
  candidates: Array<{ index: number; summary: string; role: string }>,
): { system: string; user: string } {
  const candidateBlock = candidates
    .map(
      (candidate) =>
        `${candidate.index}. [${candidate.role}] ${truncateText(candidate.summary, 260)}`,
    )
    .join("\n");
  return {
    system:
      '你负责过滤召回的记忆候选。请只返回严格 JSON：{"relevant": number[], "sufficient": boolean, "reason": string}。只保留那些对回答当前问题直接有帮助的候选。',
    user: `当前问题：\n${truncateText(query, 500)}\n\n候选记忆：\n${candidateBlock}`,
  };
}

function buildRecallPlanPrompt(query: string): { system: string; user: string } {
  return {
    system:
      '你负责把用户最新请求改写成会话记忆检索计划。请只返回严格 JSON：{"focusedQuery": string, "reason": string, "routeHint"?: "workflow"|"factual"|"temporal"|"explanatory"|"mixed"}。不要判断是否应该召回，也不要输出任何召回开关或跳过判断字段；是否有可用记忆由检索结果和后续过滤决定。',
    user: `用户最新请求：\n${truncateText(query, 1500)}`,
  };
}

function buildRoutePriorPrompt(query: string): { system: string; user: string } {
  return {
    system:
      '你负责为记忆检索选择路由先验。请只返回严格 JSON：{"primaryRoute":"workflow"|"factual"|"temporal"|"explanatory"|"mixed"|"unknown","secondaryRoutes"?: ("workflow"|"factual"|"temporal"|"explanatory")[],"confidence": number,"focusedQueries": {"workflow"?: string,"factual"?: string,"temporal"?: string,"explanatory"?: string},"reason": string}。不要依赖固定关键词模板，而要根据用户真实意图给出每条路由更适合的聚焦检索语句。',
    user: `用户问题：\n${truncateText(query, 1200)}`,
  };
}

function buildRecallPlanWithRoutePrompt(query: string): { system: string; user: string } {
  return {
    system:
      '你负责把用户最新请求改写成会话记忆检索计划，并选择路由先验。请只返回严格 JSON：{"focusedQuery": string, "reason": string, "routeHint"?: "workflow"|"factual"|"temporal"|"explanatory"|"mixed", "primaryRoute": "workflow"|"factual"|"temporal"|"explanatory"|"mixed"|"unknown", "secondaryRoutes"?: ("workflow"|"factual"|"temporal"|"explanatory")[], "routeConfidence": number, "focusedQueries": {"workflow"?: string, "factual"?: string, "temporal"?: string, "explanatory"?: string}, "routeReason"?: string}。不要判断是否应该召回，也不要输出任何召回开关或跳过判断字段；是否有可用记忆由检索结果和后续过滤决定。',
    user: `用户最新请求：\n${truncateText(query, 1500)}`,
  };
}

function buildQueryCompilePrompt(
  query: string,
  fallback: QueryCompileResult,
): { system: string; user: string } {
  const promptInput = buildQueryCompilerPromptInput(query, fallback);
  const promptPayload = {
    visibleQuery: promptInput.envelope.windows.map((window) => ({
      kind: window.kind,
      text: truncateText(window.text, 700),
    })),
    omittedChars: promptInput.envelope.omittedChars,
    scaffold: {
      focusedQuery: truncateText(promptInput.scaffold.focusedQuery, 160),
      primaryRoute: promptInput.scaffold.primaryRoute,
      queryShape: promptInput.scaffold.queryShape,
      queryEntities: promptInput.scaffold.queryEntities.slice(0, 8),
    },
  };
  return {
    system: [
      "把用户最新请求编译成轻量记忆检索计划；只输出严格 JSON。",
      "顶层字段只能是：focusedQuery、queryEntities、suppressedEntities、contextExclusions、queryShape、primaryRoute。",
      '返回格式：{"focusedQuery": string, "queryEntities": [{"name": string, "type"?: "person"|"project"|"tool"|"service"|"language"|"framework"|"concept"|"organization"|"unknown", "role"?: "subject"|"object"|"context"|"resource"}], "suppressedEntities"?: [{"name": string, "type"?: "person"|"project"|"tool"|"service"|"language"|"framework"|"concept"|"organization"|"unknown", "reason"?: string}], "contextExclusions"?: [{"kind": "prior_project"|"prior_topic"|"prior_context"|"host_native_memory", "label": string, "reason"?: string}], "queryShape": {"timeframe": "current"|"historical"|"compare"|"timeless", "granularity": "summary"|"exact_detail", "referentialMode": "anchored"|"deictic", "evidenceNeed": "workflow_context"|"canonical_state"|"factual_history"|"event_history"|"relation"|"chunk"}, "primaryRoute": "workflow"|"factual"|"temporal"|"explanatory"}。',
      "focusedQuery 最长 160 字，只压缩当前请求，不加入不存在的事实。",
      "queryEntities 只输出稳定具名实体，如项目、工具、服务、框架、语言、人、组织；不要输出代词、泛指词、数学变量、完整句子或学科名。",
      "suppressedEntities 只在用户明确说不谈、不考虑、排除某个具名实体或主题时输出；它表示这些实体不应触发 native context 注入，不是 recall 开关。",
      "contextExclusions 只在用户明确要求不要使用或不要提及旧项目、旧话题、历史上下文、宿主原生记忆时输出；它是下游过滤约束，不是 recall 开关。",
      "primaryRoute：workflow=任务/状态，factual=稳定事实/配置，temporal=过去事件，explanatory=原因/关系。",
      "是否有可用记忆由检索和过滤决定；不要输出 no-recall/skip/shouldRecall/use/avoid 等字段。",
      "只根据 visibleQuery 生成计划；不要根据 omittedChars 臆测。",
    ].join("\n"),
    user: JSON.stringify(promptPayload),
  };
}

function buildTurnSemanticCompilePrompt(
  messages: TurnCaptureMessage[],
  fallback: TurnSemanticFrame,
): { system: string; user: string } {
  const compilerInput = buildTurnSemanticCompilerInput(messages, fallback.referenceContext);
  const compactFallback = summarizeTurnSemanticFallback(fallback);
  return {
    system: [
      "你负责把一个对话轮次编译成语义草案（TurnSemanticFrame）。",
      "你只能返回严格 JSON，不得输出任何解释文字，不得输出 markdown。",
      "顶层字段只能是：sourceRefs、chunkDrafts、taskProposal、assertionDrafts、correctionDrafts、relationDrafts、resourceAssertions、adviceSignals、supportSpans、compilerProvenance。",
      "这是对当前轮次的语义编译结果。scaffold 只用于提供 sourceRef、chunk/task 容器和审计参考，不是语义 truth。",
      "semantic 字段必须由你显式输出：assertionDrafts、correctionDrafts、relationDrafts、resourceAssertions、adviceSignals 如果没有高价值内容就省略或返回空数组；运行时不会把本地 regex 语义自动补回来。",
      "chunkDrafts、taskProposal、supportSpans 可以省略；省略时运行时只保留非语义 scaffold。",
      "不要把输入文本、长句子或大段摘要复制进输出；输出应该尽可能短。",
      "这只是 semantic draft，不是最终写库动作。",
      "输入里的 turnMessageEnvelope 是当前轮消息的机械窗口视图；rawHash 只用于审计，omittedChars 表示没有展示给你的原文。",
      "不要根据 omitted 内容臆测事实；如果可见窗口不足以证明高价值语义信号，就不要输出对应 semantic 字段。",
      "输入里的 recentReferenceContext 是只读的最近对话焦点，只能用于解析“这个/那个/它/that/it”等当前轮指代；它不是可写事实来源。",
      "所有 sourceRefs、supportSpans、resourceAssertions.sourceRef、adviceSignals.sourceRefs 和 relationDrafts.sourceRef 都只能引用当前 turnMessageEnvelope.messages 里的 sourceRef，不能引用 recentReferenceContext 里的 sourceRef。",
      "如果用户当前轮用“这个/那个/它”表达保留、排除、纠偏、停止考虑等意图，你可以用当前 assistant 或 recentReferenceContext 解析所指实体，但必须把可写 signal 归因到表达该意图的当前 user/tool sourceRef。",
      "如果当前轮把“这个/那个/它”的某个属性改成新值，必须输出一个已解析实体的 assertionDraft：familyHint=\"fact_like\"、timeframeHint=\"current\"、entityHints 包含实体和新值实体、slotHints 包含稳定英文 snake_case 槽名、valueHint 是该属性的新短值；同时输出 correctionDraft，canonicalKey 用 \"实体名.slot\"，predicate 用 \"has_<slot>\" 或 \"uses_<slot>\"，priorValue/nextValue 只写旧值/新值。",
      "当用户明确确认、设定、改动或固定某个项目/服务/工具的配置、默认值、通道、provider、格式、数据库、队列、缓存、负责人、约束时，这就是高价值 durable fact_like；如果 assistant 在同一轮复述确认，可用 assistant 文本辅助解析 valueHint，但 sourceRef 仍归因到当前 user sourceRef。",
      "对于实体属性事实，slotHints 必须写短的稳定 snake_case 槽名，例如 default_database、export_format、alert_channel、queue、cache、provider；valueHint 必须只写该槽的短值，例如 PostgreSQL、Arrow IPC、Slack。一个属性对应一个 assertionDraft；不要把多个不同属性塞进同一个 assertionDraft。",
      "你不能输出 final action、classification、owner、supersede、slotReplacement。",
      "你不能直接决定 durable state/fact/event/graph relation，也不能改写 canonical truth。",
      "sourceRefs 必须是数组，且只能引用输入里已有的 sourceRef，不能发明新的 sourceRef。",
      "chunkDrafts 只在 scaffold 明显缺少必要 sourceRef 绑定信息时才允许返回；每个 chunkDraft 只允许 sourceRef 和一个不超过 24 个字符的 summary。",
      'taskProposal 只能表达任务连续性的草案；如果返回 taskProposal，必须把整个对象一起返回。decision 只能是 "continue"|"resume"|"new"|"none"，允许字段 targetTaskId、confidence、summary、summaryConfidence、reason；summary 只是一条 working summary 草案，不是事实 owner，也不是最终 assignment。',
      'assertionDrafts 只能表达 semantic hint，不能表达 final canonical class；familyHint 只能是 "workflow"|"preference"|"fact_like"|"event_like"|"relation_like"|"strategy_like"，timeframeHint 只能是 "current"|"historical"|"compare"|"timeless"。strategy_like 只用于步骤顺序、排查原则、程序化 guidance，不代表新增 durable schema。',
      'entityHints 是唯一实体抽取入口；如果当前 source 有稳定、可复现的具名实体，请在相关 assertionDrafts 上返回 entityHints，格式为 {"name":"...","type":"person|project|tool|service|language|framework|concept|organization|unknown"}。不要输出代词、泛指词、完整句子或只在本句中临时成立的描述。',
      "如果同一实体只作为 relationDrafts 的 subject/object 或 resourceAssertions 的 resource 出现，可以不重复放进 entityHints；否则需要通过 entityHints 显式给出。supportSpans 默认不要返回，除非 scaffold 把 sourceRef 绑定错了。",
      "不要把 assertionDrafts 写成 final state/fact/event/relation action。",
      "relationDrafts 用于明确实体关系，每项必须包含 sourceRef、relation、confidence。relation 只能包含 subject、predicate、object、polarity、rawPredicate、slot/relationSlot、confidence、reason；polarity 必须是 affirmed 或 negated。只有 affirmed 表示可写入正向 graph edge；negated 表示“该关系不成立/负向边界”，只能作为证据支持，不能写成正向 edge。subject/object 必须来自当前轮文本或已给上下文元数据，不得臆测；predicate 必须表达真实关系，例如 uses、depends_on、blocks、owner_of、part_of、supersedes、contradicts、resolved_by、reads、related_to。关系不是 final action，但它是 graph 写入的结构化输入。",
      "relationDrafts 的 subject/object 必须是可稳定复现的具名项目、模块、文件、工具、人、组织、资源或地点；不要把“这/它/that/it/this/上面的问题/这个偏好”等代词、泛指词、句子片段或纯概念标签当 entity。若只能看到代词但无法从本轮或上下文元数据解析到稳定实体，请省略 relationDraft。",
      "关系方向必须按 predicate contract 输出：owner_of 的 subject 永远是负责人/拥有者/owner，object 是被负责的模块、资源、问题域或职责；如果原文是 “X is owned by Y / X 归 Y / X 由 Y 负责”，必须输出 subject=Y, predicate=owner_of, object=X。blocks 的 subject 是 blocker，object 是被阻塞对象。depends_on 的 subject 是依赖方，object 是被依赖方。不确定方向时不要用强关系，改用 related_to 或不输出 relationDraft。",
      '显式 rename / alias / formerly / old-name / 曾用名 / 原名 / 改名 / current name replacement 关系必须用 supersedes，不要降成 related_to。方向固定为当前/新/canonical 名称是 subject，旧/历史/former 名称是 object；例如 “LuoShu, formerly Pine” 输出 subject="LuoShu", predicate="supersedes", object="Pine"。只有文本只是松散关联、没有身份或改名含义时，才使用 related_to。',
      'resourceAssertions 用于用户明确提到自己拥有、刚获得、正在使用、正在考虑的具体资源、工具、账号、服务、能力或约束；每项必须包含 owner、resource、ownershipStatus("owned"|"recently_acquired"|"uses"|"considering")、sourceRef、supportText、confidence、semanticStatus("observed"|"inferred_affordance")，可选 resourceType、domains、affordances。sourceRef 必须来自输入，supportText 必须是原文短摘且少于 160 字符。domains/affordances 是短检索提示，用来说明这个资源通常解决什么问题或提供什么能力；如果这层推断并非原文直说，semanticStatus 用 "inferred_affordance"，且不要把它写成用户事实。不要把纯推荐列表、assistant 建议、第三方反应、效果指标、抽象主题或泛化产品知识当成用户资源。',
      'adviceSignals 用于同一轮里有用户问题上下文、用户资源、assistant 建议或可复用建议上下文的情况；每项只能包含 problemContext、userResources、assistantRecommendation、domains、sourceRefs、supportText、confidence、semanticStatus("observed"|"assistant_suggested")。sourceRefs 必须全部来自输入，必须至少有一个 sourceRef；不要凭空生成 strategy。',
      "一次输出最多 8 个高价值 signals：relationDrafts 最多 4 条，resourceAssertions 最多 3 条，adviceSignals 最多 2 条。优先输出未来 advice/preference/recall 明显有用、且有 sourceRef 的结构化内容。",
      "correctionDrafts 只表达纠偏草案，不做最终物化；每项只需要 sourceRef、correction、confidence。correction 本身只应在文本里明确出现 current/historical change cue 时输出；priorValue 和 nextValue 只写简短短语，不要复制整句；运行时会自动补 lineage。",
      "supportSpans 默认不要返回；只有 scaffold 把 sourceRef 绑定错了时才允许返回，而且 text 必须是很短的精确子串。",
      'compilerProvenance 可以省略；如果返回它，只允许 {"source":"llm"}；其余 provenance 由运行时覆盖。',
      "保守策略：纯提问、假设、条件句、引用第三方内容，不要输出高置信 assertionDraft。",
      "如果一句话是否构成 assertion 不确定，宁可降低 confidence，或不输出对应 assertionDraft。",
      "一条 source 可以产出多个 assertionDraft，但不要为了看起来完整而补造草案；整个输出里 assertionDraft 最多 8 条，correctionDraft 最多 3 条。",
      "如果本轮没有值得写入的语义信号，可以返回 {}；这表示没有 compiler-confirmed semantic signals，不表示接受任何本地语义 hints。",
      "只输出 JSON。",
    ].join("\n"),
    user: `当前轮次 turnMessageEnvelope：\n${JSON.stringify(compilerInput, null, 2)}\n\n当前非语义 scaffold 摘要：\n${JSON.stringify(compactFallback, null, 2)}`,
  };
}

function compactTurnMessagePayload(
  messages: TurnCaptureMessage[],
  fallback: TurnSemanticFrame,
): Record<string, unknown> {
  const compilerInput = buildTurnSemanticCompilerInput(messages, fallback.referenceContext);
  return {
    messages: compilerInput.messages.map((message) => ({
      role: message.role,
      sourceRef: message.sourceRef,
      text: message.windows.map((window) => window.text).join("\n...\n"),
      omittedChars: message.omittedChars,
    })),
    recentReferenceContext: compilerInput.recentReferenceContext,
    sourceRefs: fallback.sourceRefs,
  };
}

function buildCompactTurnSemanticPrompt(
  messages: TurnCaptureMessage[],
  fallback: TurnSemanticFrame,
): { system: string; user: string } {
  return {
    system: [
      "Extract reusable memory signals from one current conversation turn. Return only valid JSON.",
      "Do not explain. Do not use markdown. Do not copy long text.",
      "Use only sourceRef values from the input messages. Prefer the current user sourceRef for facts confirmed by the assistant.",
      "recentReferenceContext is only for resolving pronouns such as this/that/it/这个/那个; it is not a writable source.",
      "Output schema:",
      '{"assertions":[{"sourceRef":string,"subject":string,"subjectType"?:string,"slot":string,"value":string,"family"?:string,"timeframe"?:string,"confidence"?:number,"supportText"?:string}],"corrections":[{"sourceRef":string,"subject":string,"slot":string,"priorValue"?:string,"nextValue":string,"predicate"?:string,"canonicalKey"?:string,"confidence"?:number,"supportText"?:string}],"relations":[{"sourceRef":string,"subject":string,"predicate":string,"object":string,"polarity"?:string,"slot"?:string,"confidence"?:number,"reason"?:string,"supportText"?:string}],"resourceAssertions":[],"adviceSignals":[],"taskProposal"?:object}',
      "assertions are durable semantic facts/preferences/workflow states. Use family=fact_like and timeframe=current for project/service/tool defaults, provider, database, queue, cache, channel, format, owner, and constraints.",
      "For entity attributes, make one assertion per property. Examples: subject=AuroraAccept slot=default_database value=PostgreSQL; subject=AuroraAccept slot=alert_channel value=Slack; subject=AuroraAccept slot=export_format value=Arrow IPC.",
      "slot must be a short stable snake_case English slot such as default_database, alert_channel, export_format, queue, cache, provider, owner, constraint.",
      "If the turn changes a previous value, output both an assertion for the new current value and a correction with subject, slot, priorValue when visible, and nextValue.",
      "Do not output assertions for pure questions, hypotheticals, quoted third-party claims, or unsupported assistant guesses.",
      "Maximum: assertions 8, corrections 3, relations 4, resourceAssertions 3, adviceSignals 2.",
      "If there is no reusable memory signal, return {}.",
    ].join("\n"),
    user: JSON.stringify(compactTurnMessagePayload(messages, fallback)),
  };
}

function buildCompactLongTurnSemanticPrompt(
  input: LongTurnSemanticScanInput,
  fallback: TurnSemanticFrame,
): { system: string; user: string } {
  return {
    system: [
      "Extract reusable memory signals from persisted source segments. Return only valid JSON.",
      "Use the same compact schema as the turn compiler: assertions, corrections, relations, resourceAssertions, adviceSignals, taskProposal.",
      "Use only sourceRef values from longTurnSegmentScan.messages. recentReferenceContext is read-only and only resolves pronouns.",
      "For project/service/tool defaults, provider, database, queue, cache, channel, format, owner, and constraints, output one fact_like current assertion per property with subject, slot, and value.",
      "If a segment changes a previous value, output both the new assertion and a correction.",
      "Do not invent facts from omitted text. Do not summarize a user question as an answer.",
      "Maximum: assertions 8, corrections 3, relations 4, resourceAssertions 3, adviceSignals 2.",
      "If there is no reusable memory signal, return {}.",
    ].join("\n"),
    user: JSON.stringify({
      messages: input.messages.map((message) => ({
        role: message.role,
        sourceRef: message.sourceRef,
        turnId: message.turnId,
        segments: message.segments.map((segment) => ({
          index: segment.index,
          text: segment.text,
        })),
      })),
      recentReferenceContext: input.recentReferenceContext,
      sourceRefs: fallback.sourceRefs,
    }),
  };
}

function compactSlot(value: unknown): string | undefined {
  if (typeof value !== "string") {
    return undefined;
  }
  const canonical = canonicalAttributeSlot(value);
  if (canonical) {
    return canonical;
  }
  const slot = normalizeText(value.replace(/([a-z0-9])([A-Z])/g, "$1_$2"))
    .replace(/[^\p{L}\p{N}]+/gu, "_")
    .replace(/^_+|_+$/g, "");
  return slot || undefined;
}

function compactConfidence(value: unknown, fallback: number): number {
  return typeof value === "number" && Number.isFinite(value) ? clamp01(value) : fallback;
}

function compactEntityType(value: unknown): string | undefined {
  return typeof value === "string" && TURN_ENTITY_HINT_TYPES.has(value) ? value : undefined;
}

function compactFamily(value: unknown): TurnSemanticAssertionFamilyHint {
  return normalizeTurnSemanticFamilyHint(value) ?? "fact_like";
}

function compactTimeframe(value: unknown): TurnSemanticTimeframeHint {
  return normalizeTurnSemanticTimeframeHint(value) ?? "current";
}

function compactSupportSpan(sourceRef: string, text: unknown): Array<{ text: string }> | undefined {
  if (typeof text !== "string" || !text.trim()) {
    return undefined;
  }
  return [{ text: truncateText(text.trim(), 180) }];
}

function compactSemanticResultToPatch(result: CompactTurnSemanticResult): Partial<TurnSemanticFrame> {
  const assertionDrafts = Array.isArray(result.assertions)
    ? result.assertions
        .map((entry, index) => {
          const subject = typeof entry?.subject === "string" ? entry.subject.trim() : "";
          const value = typeof entry?.value === "string" ? entry.value.trim() : "";
          const slot = compactSlot(entry?.slot);
          if (!subject || !value || !slot) {
            return null;
          }
          const entityHints = [
            {
              name: truncateText(subject, 120),
              ...(compactEntityType(entry.subjectType) ? { type: compactEntityType(entry.subjectType) } : {}),
            },
            { name: truncateText(value, 120), type: "concept" },
          ];
          return {
            draftId: `${entry.sourceRef ?? "source"}:${slot}:${index}`,
            sourceRef: entry.sourceRef ?? "",
            familyHint: compactFamily(entry.family),
            timeframeHint: compactTimeframe(entry.timeframe),
            entityHints,
            slotHints: [slot],
            valueHint: truncateText(value, 160),
            supportSpans: compactSupportSpan(entry.sourceRef ?? "", entry.supportText),
            confidence: compactConfidence(entry.confidence, 0.82),
            lineage: {
              sourceKind: "chunk" as const,
              sourceId: entry.sourceRef ?? "compact-semantic",
              sourceRef: entry.sourceRef,
            },
          };
        })
        .filter((entry): entry is NonNullable<typeof entry> => entry !== null)
    : undefined;
  const correctionDrafts = Array.isArray(result.corrections)
    ? result.corrections
        .map((entry) => {
          const subject = typeof entry?.subject === "string" ? entry.subject.trim() : "";
          const slot = compactSlot(entry?.slot);
          const nextValue = typeof entry?.nextValue === "string" ? entry.nextValue.trim() : "";
          if (!subject || !slot || !nextValue) {
            return null;
          }
          return {
            sourceRef: entry.sourceRef ?? "",
            correction: {
              timeframe: "current" as const,
              targetKind: "fact" as const,
              canonicalKey:
                typeof entry.canonicalKey === "string" && entry.canonicalKey.trim()
                  ? entry.canonicalKey.trim()
                  : `${subject}.${slot}`,
              predicate:
                typeof entry.predicate === "string" && entry.predicate.trim()
                  ? entry.predicate.trim()
                  : `has_${slot}`,
              ...(typeof entry.priorValue === "string" && entry.priorValue.trim()
                ? { priorValue: truncateText(entry.priorValue.trim(), 160) }
                : {}),
              nextValue: truncateText(nextValue, 160),
              confidence: compactConfidence(entry.confidence, 0.82),
              ...(typeof entry.supportText === "string" && entry.supportText.trim()
                ? { reason: truncateText(entry.supportText.trim(), 180) }
                : {}),
            },
            supportSpans: compactSupportSpan(entry.sourceRef ?? "", entry.supportText),
            confidence: compactConfidence(entry.confidence, 0.82),
            lineage: {
              sourceKind: "chunk" as const,
              sourceId: entry.sourceRef ?? "compact-semantic",
              sourceRef: entry.sourceRef,
            },
          };
        })
        .filter((entry): entry is NonNullable<typeof entry> => entry !== null)
    : undefined;
  const relationDrafts = Array.isArray(result.relations)
    ? result.relations
        .map((entry) => {
          const subject = typeof entry?.subject === "string" ? entry.subject.trim() : "";
          const object = typeof entry?.object === "string" ? entry.object.trim() : "";
          const predicate = typeof entry?.predicate === "string" ? entry.predicate.trim() : "";
          if (!subject || !object || !predicate) {
            return null;
          }
          const normalizedPredicate = normalizeGraphRelationType(predicate);
          return {
            sourceRef: entry.sourceRef ?? "",
            relation: {
              subject,
              predicate: normalizedPredicate?.relationType ?? "related_to",
              object,
              polarity: entry.polarity === "negated" ? ("negated" as const) : ("affirmed" as const),
              rawPredicate: normalizedPredicate?.rawPredicate ?? predicate,
              ...(typeof entry.slot === "string" && entry.slot.trim()
                ? { relationSlot: entry.slot.trim() }
                : {}),
              confidence: compactConfidence(entry.confidence, 0.72),
              ...(typeof entry.reason === "string" && entry.reason.trim()
                ? { reason: truncateText(entry.reason.trim(), 180) }
                : {}),
            },
            supportSpans: compactSupportSpan(entry.sourceRef ?? "", entry.supportText),
            confidence: compactConfidence(entry.confidence, 0.72),
            lineage: {
              sourceKind: "chunk" as const,
              sourceId: entry.sourceRef ?? "compact-semantic",
              sourceRef: entry.sourceRef,
            },
          };
        })
        .filter((entry): entry is NonNullable<typeof entry> => entry !== null)
    : undefined;
  return {
    ...(assertionDrafts ? { assertionDrafts } : {}),
    ...(correctionDrafts ? { correctionDrafts } : {}),
    ...(relationDrafts ? { relationDrafts } : {}),
    ...(Array.isArray(result.resourceAssertions)
      ? { resourceAssertions: result.resourceAssertions }
      : {}),
    ...(Array.isArray(result.adviceSignals) ? { adviceSignals: result.adviceSignals } : {}),
    ...(result.taskProposal ? { taskProposal: result.taskProposal as TurnSemanticTaskProposal } : {}),
  };
}

function hasRecognizedCompactSemanticResult(value: CompactTurnSemanticResult): boolean {
  const keys = Object.keys(value as Record<string, unknown>);
  if (keys.length === 0) {
    return true;
  }
  const allowed = new Set([
    "assertions",
    "corrections",
    "relations",
    "resourceAssertions",
    "adviceSignals",
    "taskProposal",
  ]);
  return keys.some((key) => allowed.has(key));
}

function buildLongTurnSemanticScanPrompt(
  input: LongTurnSemanticScanInput,
  fallback: TurnSemanticFrame,
): { system: string; user: string } {
  const compactFallback = summarizeTurnSemanticFallback(fallback);
  return {
    system: [
      "你负责在维护链路扫描对话轮次的机械分段视图，输出 LLM-only 语义草案。",
      "你只能返回严格 JSON，不得输出任何解释文字，不得输出 markdown。",
      "顶层字段只能是：sourceRefs、chunkDrafts、taskProposal、assertionDrafts、correctionDrafts、relationDrafts、resourceAssertions、adviceSignals、supportSpans、compilerProvenance。",
      "这些 segments 是待维护的源消息机械切片，不是旧记忆。你不能输出 final action、classification、owner、supersede、slotReplacement。",
      "semantic 字段必须由你显式输出；运行时不会把本地 regex 语义自动补回来。",
      'entityHints 是唯一实体抽取入口；如果当前 segment 有稳定、可复现的具名实体，请在相关 assertionDrafts 上返回 entityHints，格式为 {"name":"...","type":"person|project|tool|service|language|framework|concept|organization|unknown"}。',
      "用户明确确认、设定、改动或固定项目/服务/工具配置、默认值、通道、provider、格式、数据库、队列、缓存、负责人、约束时，是高价值 durable fact_like；不要因句子短而省略。",
      "实体属性类 assertionDraft 必须带 slotHints 和 valueHint：slotHints 是稳定英文 snake_case 槽名，valueHint 是该槽短值；不要把整句当 valueHint。",
      "不要把代词、泛指词、完整句子、临时变量、数学符号、代码局部变量或只在本句临时成立的描述当 entity。",
      "relationDrafts 只用于明确实体关系；subject/object 必须是可稳定复现的具名实体，predicate 必须表达真实关系。",
      "resourceAssertions 只用于用户明确拥有、使用、刚获得、正在考虑的具体资源、工具、账号、服务、能力或约束。",
      "adviceSignals 只用于同一轮中可复用的用户问题上下文、用户资源、assistant 建议或建议上下文。",
      "sourceRefs 必须引用输入里已有的 sourceRef，不能发明新的 sourceRef。",
      "输入里的 recentReferenceContext 是只读的最近对话焦点，只能用于解析“这个/那个/它/that/it”等指代；它不是可写事实来源。",
      "所有 sourceRefs、resourceAssertions.sourceRef、adviceSignals.sourceRefs 和 relationDrafts.sourceRef 都只能引用 longTurnSegmentScan.messages 里的 sourceRef，不能引用 recentReferenceContext 里的 sourceRef。",
      "不要复制大段 segment 文本；supportText/supportSpans 只允许很短的原文依据。",
      "如果 segments 只是长材料、日志、代码、题目或 assistant 长答案，且没有值得长期使用的用户事实/偏好/关系/资源/任务状态，可以返回 {}。",
      "保守策略：不确定是否值得写入时，不输出对应 semantic 字段。",
      "只输出 JSON。",
    ].join("\n"),
    user: `长轮次 longTurnSegmentScan：\n${JSON.stringify(input, null, 2)}\n\n当前非语义 scaffold 摘要：\n${JSON.stringify(compactFallback, null, 2)}`,
  };
}

function hasRecognizedTurnSemanticPatch(value: Partial<TurnSemanticFrame>): boolean {
  const keys = Object.keys(value as Record<string, unknown>);
  if (keys.length === 0) {
    return true;
  }
  const allowed = new Set([
    "sourceRefs",
    "chunkDrafts",
    "taskProposal",
    "assertionDrafts",
    "correctionDrafts",
    "relationDrafts",
    "resourceAssertions",
    "adviceSignals",
    "supportSpans",
    "compilerProvenance",
  ]);
  return keys.some((key) => allowed.has(key));
}

function summarizeTurnSemanticFallback(fallback: TurnSemanticFrame): Record<string, unknown> {
  const draftsBySource = new Map<
    string,
    {
      sourceRef: string;
      families: string[];
      timeframes: string[];
      slots: string[];
      hasCorrection: boolean;
      correctionTimeframe?: string;
    }
  >();
  for (const draft of fallback.assertionDrafts) {
    const current = draftsBySource.get(draft.sourceRef) ?? {
      sourceRef: draft.sourceRef,
      families: [],
      timeframes: [],
      slots: [],
      hasCorrection: false,
    };
    if (!current.families.includes(draft.familyHint)) {
      current.families.push(draft.familyHint);
    }
    if (!current.timeframes.includes(draft.timeframeHint)) {
      current.timeframes.push(draft.timeframeHint);
    }
    for (const slot of draft.slotHints ?? []) {
      if (!current.slots.includes(slot)) {
        current.slots.push(slot);
      }
    }
    draftsBySource.set(draft.sourceRef, current);
  }
  for (const correction of fallback.correctionDrafts) {
    const current = draftsBySource.get(correction.sourceRef) ?? {
      sourceRef: correction.sourceRef,
      families: [],
      timeframes: [],
      slots: [],
      hasCorrection: false,
    };
    current.hasCorrection = true;
    current.correctionTimeframe = correction.correction.timeframe;
    draftsBySource.set(correction.sourceRef, current);
  }
  for (const relation of fallback.relationDrafts ?? []) {
    const current = draftsBySource.get(relation.sourceRef) ?? {
      sourceRef: relation.sourceRef,
      families: [],
      timeframes: [],
      slots: [],
      hasCorrection: false,
    };
    if (!current.families.includes("relation_like")) {
      current.families.push("relation_like");
    }
    const slot = relation.relation.relationSlot ?? relation.relation.predicate;
    if (slot && !current.slots.includes(slot)) {
      current.slots.push(slot);
    }
    draftsBySource.set(relation.sourceRef, current);
  }
  return {
    sourceRefs: fallback.sourceRefs,
    taskProposal: fallback.taskProposal
      ? {
          decision: fallback.taskProposal.decision,
          targetTaskId: fallback.taskProposal.targetTaskId,
          confidence: fallback.taskProposal.confidence,
          summary: fallback.taskProposal.summary,
          summaryConfidence: fallback.taskProposal.summaryConfidence,
        }
      : undefined,
    semanticHints: [...draftsBySource.values()],
  };
}

function sourceRefForReasonerMessage(message: TurnCaptureMessage): string {
  return message.sourceRef || `${message.role}:${message.turnId}`;
}

function normalizeTurnSemanticDecision(
  value: unknown,
): "continue" | "resume" | "new" | "none" | undefined {
  return value === "continue" || value === "resume" || value === "new" || value === "none"
    ? value
    : undefined;
}

function normalizeTurnSemanticFamilyHint(
  value: unknown,
):
  | "workflow"
  | "preference"
  | "fact_like"
  | "event_like"
  | "relation_like"
  | "strategy_like"
  | undefined {
  return value === "workflow" ||
    value === "preference" ||
    value === "fact_like" ||
    value === "event_like" ||
    value === "relation_like" ||
    value === "strategy_like"
    ? value
    : undefined;
}

function normalizeTurnSemanticTimeframeHint(
  value: unknown,
): "current" | "historical" | "compare" | "timeless" | undefined {
  return value === "current" ||
    value === "historical" ||
    value === "compare" ||
    value === "timeless"
    ? value
    : undefined;
}

const TURN_ENTITY_HINT_TYPES = new Set([
  "person",
  "project",
  "tool",
  "service",
  "language",
  "framework",
  "concept",
  "organization",
  "unknown",
]);

const DISALLOWED_ENTITY_HINTS = new Set([
  "it",
  "this",
  "that",
  "they",
  "them",
  "这些",
  "这个",
  "这",
  "那个",
  "那",
  "它",
  "他们",
  "她们",
  "它们",
]);

function normalizeTurnSemanticEntityHints(
  value: unknown,
): Array<{ name: string; type?: string }> | undefined {
  if (!Array.isArray(value)) {
    return undefined;
  }
  const seen = new Set<string>();
  const normalized: Array<{ name: string; type?: string }> = [];
  for (const entry of value) {
    if (!entry || typeof entry !== "object" || Array.isArray(entry)) {
      continue;
    }
    const record = entry as Record<string, unknown>;
    const name = typeof record.name === "string" ? record.name.trim() : "";
    if (!name || !isValidEntityName(name) || DISALLOWED_ENTITY_HINTS.has(normalizeText(name))) {
      continue;
    }
    const key = normalizeText(name);
    if (seen.has(key)) {
      continue;
    }
    seen.add(key);
    const rawType = typeof record.type === "string" ? record.type.trim() : "";
    const type = TURN_ENTITY_HINT_TYPES.has(rawType) ? rawType : undefined;
    normalized.push({
      name,
      ...(type ? { type } : {}),
    });
    if (normalized.length >= 8) {
      break;
    }
  }
  return normalized.length > 0 ? normalized : undefined;
}

function normalizedStringArray(value: unknown, limit: number): string[] {
  if (!Array.isArray(value)) {
    return [];
  }
  const seen = new Set<string>();
  const normalized: string[] = [];
  for (const entry of value) {
    if (typeof entry !== "string") {
      continue;
    }
    const text = truncateText(entry.trim(), 80);
    if (!text) {
      continue;
    }
    const key = normalizeText(text);
    if (seen.has(key)) {
      continue;
    }
    seen.add(key);
    normalized.push(text);
    if (normalized.length >= limit) {
      break;
    }
  }
  return normalized;
}

function normalizeResourceOwnershipStatus(
  value: unknown,
): MemoryResourceAssertionHint["ownershipStatus"] | undefined {
  return value === "owned" ||
    value === "recently_acquired" ||
    value === "uses" ||
    value === "considering"
    ? value
    : undefined;
}

function normalizeResourceSemanticStatus(
  value: unknown,
): MemoryResourceAssertionHint["semanticStatus"] {
  return value === "inferred_affordance" ? "inferred_affordance" : "observed";
}

function normalizeAdviceSemanticStatus(value: unknown): MemoryAdviceSignalHint["semanticStatus"] {
  return value === "assistant_suggested" ? "assistant_suggested" : "observed";
}

function normalizeResourceAssertions(
  value: unknown,
  messageIndex: Map<string, TurnCaptureMessage>,
): MemoryResourceAssertionHint[] | undefined {
  if (!Array.isArray(value)) {
    return undefined;
  }
  const normalized: MemoryResourceAssertionHint[] = [];
  const seen = new Set<string>();
  for (const entry of value) {
    if (!entry || typeof entry !== "object") {
      continue;
    }
    const record = entry as Record<string, unknown>;
    const sourceRef = typeof record.sourceRef === "string" ? record.sourceRef.trim() : "";
    const owner =
      typeof record.owner === "string" && record.owner.trim() ? record.owner.trim() : "user";
    const resource = typeof record.resource === "string" ? record.resource.trim() : "";
    const ownershipStatus = normalizeResourceOwnershipStatus(record.ownershipStatus);
    const supportText =
      typeof record.supportText === "string" ? truncateText(record.supportText.trim(), 160) : "";
    if (
      !sourceRef ||
      !messageIndex.has(sourceRef) ||
      !resource ||
      !ownershipStatus ||
      !supportText
    ) {
      continue;
    }
    const key = `${sourceRef}:${normalizeText(owner)}:${normalizeText(resource)}:${ownershipStatus}`;
    if (seen.has(key)) {
      continue;
    }
    seen.add(key);
    normalized.push({
      owner,
      resource: truncateText(resource, 120),
      ownershipStatus,
      ...(typeof record.resourceType === "string" && record.resourceType.trim()
        ? { resourceType: truncateText(record.resourceType.trim(), 80) }
        : {}),
      domains: normalizedStringArray(record.domains, 5),
      affordances: normalizedStringArray(record.affordances, 5),
      sourceRef,
      supportText,
      confidence:
        typeof record.confidence === "number" && Number.isFinite(record.confidence)
          ? clamp01(record.confidence)
          : 0.65,
      semanticStatus: normalizeResourceSemanticStatus(record.semanticStatus),
    });
    if (normalized.length >= 3) {
      break;
    }
  }
  return normalized.length > 0 ? normalized : undefined;
}

function normalizeAdviceSignals(
  value: unknown,
  messageIndex: Map<string, TurnCaptureMessage>,
): MemoryAdviceSignalHint[] | undefined {
  if (!Array.isArray(value)) {
    return undefined;
  }
  const normalized: MemoryAdviceSignalHint[] = [];
  const seen = new Set<string>();
  for (const entry of value) {
    if (!entry || typeof entry !== "object") {
      continue;
    }
    const record = entry as Record<string, unknown>;
    const sourceRefs = normalizedStringArray(record.sourceRefs, 4).filter((sourceRef) =>
      messageIndex.has(sourceRef),
    );
    if (sourceRefs.length === 0) {
      continue;
    }
    const problemContext =
      typeof record.problemContext === "string" && record.problemContext.trim()
        ? truncateText(record.problemContext.trim(), 160)
        : undefined;
    const assistantRecommendation =
      typeof record.assistantRecommendation === "string" && record.assistantRecommendation.trim()
        ? truncateText(record.assistantRecommendation.trim(), 180)
        : undefined;
    const supportText =
      typeof record.supportText === "string" && record.supportText.trim()
        ? truncateText(record.supportText.trim(), 160)
        : undefined;
    const userResources = normalizedStringArray(record.userResources, 5);
    const domains = normalizedStringArray(record.domains, 5);
    if (!problemContext && !assistantRecommendation && userResources.length === 0 && !supportText) {
      continue;
    }
    const key = `${sourceRefs.join("|")}:${normalizeText(
      [problemContext, assistantRecommendation, ...userResources].filter(Boolean).join(" "),
    )}`;
    if (seen.has(key)) {
      continue;
    }
    seen.add(key);
    normalized.push({
      ...(problemContext ? { problemContext } : {}),
      ...(userResources.length > 0 ? { userResources } : {}),
      ...(assistantRecommendation ? { assistantRecommendation } : {}),
      ...(domains.length > 0 ? { domains } : {}),
      sourceRefs,
      ...(supportText ? { supportText } : {}),
      confidence:
        typeof record.confidence === "number" && Number.isFinite(record.confidence)
          ? clamp01(record.confidence)
          : 0.6,
      semanticStatus: normalizeAdviceSemanticStatus(record.semanticStatus),
    });
    if (normalized.length >= 2) {
      break;
    }
  }
  return normalized.length > 0 ? normalized : undefined;
}

function normalizeCompiledTurnSemanticPatch(
  messages: TurnCaptureMessage[],
  patch: Partial<TurnSemanticFrame>,
): Partial<TurnSemanticFrame> {
  const messageIndex = new Map(
    messages.map((message) => [sourceRefForReasonerMessage(message), message]),
  );
  const writableSourceRefs = messages
    .filter((message) => message.role === "user" || message.role === "tool")
    .map((message) => sourceRefForReasonerMessage(message))
    .filter(Boolean);
  const defaultWritableSourceRef =
    writableSourceRefs.length === 1 ? writableSourceRefs[0] : undefined;
  const resolvePatchSourceRef = (value: unknown): string | undefined => {
    if (typeof value === "string" && messageIndex.has(value)) {
      return value;
    }
    return defaultWritableSourceRef;
  };
  const withLineage = (sourceRef: string) => {
    const message = messageIndex.get(sourceRef);
    if (!message) {
      return undefined;
    }
    return {
      sourceKind: "chunk" as const,
      sourceId: message.turnId,
      sourceRef,
    };
  };
  const sourceRefs = Array.isArray(patch.sourceRefs)
    ? patch.sourceRefs.filter(
        (value): value is string => typeof value === "string" && messageIndex.has(value),
      )
    : undefined;
  const chunkDrafts = Array.isArray(patch.chunkDrafts)
    ? patch.chunkDrafts
        .map((entry) => {
          const sourceRef = resolvePatchSourceRef(entry?.sourceRef);
          const summary = typeof entry?.summary === "string" ? entry.summary.trim() : "";
          const lineage = sourceRef ? withLineage(sourceRef) : undefined;
          if (!sourceRef || !summary || !lineage) {
            return null;
          }
          return {
            sourceRef,
            summary: truncateText(summary, 180),
            lineage,
          };
        })
        .filter((entry): entry is NonNullable<typeof entry> => entry !== null)
    : undefined;
  const rawTaskProposal = patch.taskProposal as Record<string, unknown> | undefined;
  const taskDecision = normalizeTurnSemanticDecision(
    rawTaskProposal?.decision ?? rawTaskProposal?.action,
  );
  const taskProposal =
    rawTaskProposal && taskDecision
      ? {
          decision: taskDecision,
          ...(typeof rawTaskProposal.targetTaskId === "string" &&
          rawTaskProposal.targetTaskId.trim()
            ? { targetTaskId: rawTaskProposal.targetTaskId.trim() }
            : {}),
          confidence:
            typeof rawTaskProposal.confidence === "number" &&
            Number.isFinite(rawTaskProposal.confidence)
              ? clamp01(rawTaskProposal.confidence)
              : 0.6,
          ...(typeof rawTaskProposal.summary === "string" && rawTaskProposal.summary.trim()
            ? { summary: truncateText(rawTaskProposal.summary.trim(), 320) }
            : {}),
          ...(typeof rawTaskProposal.summaryConfidence === "number" &&
          Number.isFinite(rawTaskProposal.summaryConfidence)
            ? { summaryConfidence: clamp01(rawTaskProposal.summaryConfidence) }
            : {}),
          ...(typeof rawTaskProposal.reason === "string" && rawTaskProposal.reason.trim()
            ? { reason: truncateText(rawTaskProposal.reason.trim(), 180) }
            : {}),
        }
      : undefined;
  const normalizeCorrectionTimeframe = (value: unknown): "current" | "historical" | "compare" =>
    value === "historical" || value === "compare" || value === "current" ? value : "current";
  const normalizeCorrectionTargetKind = (
    value: unknown,
    correction: Record<string, unknown>,
  ): "state" | "fact" | "relation" | "project_profile" | "unknown" => {
    if (
      value === "state" ||
      value === "fact" ||
      value === "relation" ||
      value === "project_profile" ||
      value === "unknown"
    ) {
      return value;
    }
    if (
      typeof correction.predicate === "string" ||
      typeof correction.canonicalKey === "string" ||
      typeof correction.nextValue === "string"
    ) {
      return "fact";
    }
    return "unknown";
  };
  const assertionDrafts = Array.isArray(patch.assertionDrafts)
    ? patch.assertionDrafts
        .map((entry, index) => {
          const sourceRef = resolvePatchSourceRef(entry?.sourceRef);
          const familyHint = normalizeTurnSemanticFamilyHint(entry?.familyHint);
          const timeframeHint = normalizeTurnSemanticTimeframeHint(entry?.timeframeHint);
          const entityHints = normalizeTurnSemanticEntityHints(entry?.entityHints);
          const lineage = sourceRef ? withLineage(sourceRef) : undefined;
          if (!sourceRef || !familyHint || !timeframeHint || !lineage) {
            return null;
          }
          return {
            draftId:
              typeof entry?.draftId === "string" && entry.draftId.trim()
                ? entry.draftId.trim()
                : `${sourceRef}:${familyHint}:${index}`,
            sourceRef,
            familyHint,
            timeframeHint,
            ...(entityHints ? { entityHints } : {}),
            ...(Array.isArray(entry?.slotHints)
              ? {
                  slotHints: entry.slotHints.filter(
                    (slot): slot is string => typeof slot === "string" && slot.trim().length > 0,
                  ),
                }
              : {}),
            ...(typeof entry?.valueHint === "string" && entry.valueHint.trim()
              ? { valueHint: truncateText(entry.valueHint.trim(), 160) }
              : {}),
            ...(Array.isArray(entry?.supportSpans) ? { supportSpans: entry.supportSpans } : {}),
            ...(typeof entry?.confidence === "number" && Number.isFinite(entry.confidence)
              ? { confidence: clamp01(entry.confidence) }
              : {}),
            lineage,
          };
        })
        .filter((entry): entry is NonNullable<typeof entry> => entry !== null)
    : undefined;
  const correctionDrafts = Array.isArray(patch.correctionDrafts)
    ? patch.correctionDrafts
        .map((entry) => {
          const sourceRef = resolvePatchSourceRef(entry?.sourceRef);
          const correction =
            entry?.correction && typeof entry.correction === "object"
              ? entry.correction
              : undefined;
          const lineage = sourceRef ? withLineage(sourceRef) : undefined;
          if (!sourceRef || !correction || !lineage) {
            return null;
          }
          const correctionRecord = correction as Record<string, unknown>;
          return {
            sourceRef,
            correction: {
              ...correctionRecord,
              timeframe: normalizeCorrectionTimeframe(correctionRecord.timeframe),
              targetKind: normalizeCorrectionTargetKind(
                correctionRecord.targetKind,
                correctionRecord,
              ),
            },
            ...(Array.isArray(entry?.supportSpans) ? { supportSpans: entry.supportSpans } : {}),
            ...(typeof entry?.confidence === "number" && Number.isFinite(entry.confidence)
              ? { confidence: clamp01(entry.confidence) }
              : {}),
            lineage,
          };
        })
        .filter((entry): entry is NonNullable<typeof entry> => entry !== null)
    : undefined;
  const relationDrafts = Array.isArray(
    (patch as Partial<TurnSemanticFrame> & { relationDrafts?: unknown }).relationDrafts,
  )
    ? ((patch as Partial<TurnSemanticFrame> & { relationDrafts?: unknown[] }).relationDrafts ?? [])
        .map((entry): TurnSemanticRelationDraft | null => {
          if (!entry || typeof entry !== "object") {
            return null;
          }
          const record = entry as Record<string, unknown>;
          const sourceRef = resolvePatchSourceRef(record.sourceRef);
          const relation = normalizeRelationHint(record.relation);
          const lineage = sourceRef ? withLineage(sourceRef) : undefined;
          if (!sourceRef || !relation || !lineage) {
            return null;
          }
          return {
            sourceRef,
            relation: {
              ...relation,
              sourceRef,
            },
            ...(Array.isArray(record.supportSpans)
              ? {
                  supportSpans: record.supportSpans as Array<{
                    text: string;
                    start?: number;
                    end?: number;
                  }>,
                }
              : {}),
            confidence:
              typeof record.confidence === "number" && Number.isFinite(record.confidence)
                ? clamp01(record.confidence)
                : (relation.confidence ?? 0.72),
            lineage,
          };
        })
        .filter((entry): entry is TurnSemanticRelationDraft => entry !== null)
    : undefined;
  const resourceAssertions = normalizeResourceAssertions(patch.resourceAssertions, messageIndex);
  const adviceSignals = normalizeAdviceSignals(patch.adviceSignals, messageIndex);
  const supportSpans = Array.isArray(patch.supportSpans)
    ? patch.supportSpans
        .map((entry) => {
          const sourceRef = typeof entry?.sourceRef === "string" ? entry.sourceRef : undefined;
          const text = typeof entry?.text === "string" ? entry.text.trim() : "";
          if (!sourceRef || !text || !messageIndex.has(sourceRef)) {
            return null;
          }
          return {
            sourceRef,
            text: truncateText(text, 240),
            ...(typeof entry?.start === "number" ? { start: entry.start } : {}),
            ...(typeof entry?.end === "number" ? { end: entry.end } : {}),
          };
        })
        .filter((entry): entry is NonNullable<typeof entry> => entry !== null)
    : undefined;
  return {
    ...(sourceRefs ? { sourceRefs } : {}),
    ...(chunkDrafts ? { chunkDrafts } : {}),
    ...(taskProposal ? { taskProposal } : {}),
    ...(assertionDrafts ? { assertionDrafts } : {}),
    ...(correctionDrafts ? { correctionDrafts } : {}),
    ...(relationDrafts ? { relationDrafts } : {}),
    ...(resourceAssertions ? { resourceAssertions } : {}),
    ...(adviceSignals ? { adviceSignals } : {}),
    ...(supportSpans ? { supportSpans } : {}),
    ...(patch.compilerProvenance ? { compilerProvenance: patch.compilerProvenance } : {}),
  };
}

function buildRouteEvidencePrompt(
  query: string,
  routeType: MemoryPrimaryRouteType,
  candidates: RouteEvidenceCandidate[],
): { system: string; user: string } {
  const candidateBlock = candidates
    .map((candidate) => {
      const score = Number.isFinite(candidate.score) ? candidate.score.toFixed(2) : "0.00";
      const confidence =
        typeof candidate.confidence === "number"
          ? ` confidence=${candidate.confidence.toFixed(2)}`
          : "";
      const observedAt = candidate.observedAt ? ` observedAt=${candidate.observedAt}` : "";
      return `${candidate.index}. [${candidate.role}] score=${score}${confidence}${observedAt} ${truncateText(candidate.summary, 260)}`;
    })
    .join("\n");
  return {
    system:
      '你负责做记忆路由证据校验。请只返回严格 JSON：{"relevant": number[], "sufficient": boolean, "support": number, "reason": string}。support 取 0 到 1。判断标准不是句子像不像模板，而是这些证据是否真的支持该路由并足以帮助回答问题。',
    user: `用户问题：\n${truncateText(query, 800)}\n\n候选路由：${routeType}\n\n证据候选：\n${candidateBlock || "无候选"}`,
  };
}

// -- Consolidation confirmation prompts ---

type ConsolidationConfirmResult = {
  confirmed?: boolean;
  reason?: string;
};

type ConsolidationBatchConfirmJudgeResult = {
  items?: Array<{
    id?: string;
    decision?: "confirm" | "defer" | "reject";
    reason?: string;
  }>;
};

type ConsolidationBatchConfirmItem = {
  id: string;
  kind: "fact" | "relation";
  predicate: string;
  object: string;
  subject?: string;
  supportCount: number;
  latestObservedAt?: string;
  structuredSummaries: string[];
  sourceRefs: string[];
};

function buildConsolidationFactConfirmPrompt(
  text: string,
  preference: { predicate: string; object: string },
): { system: string; user: string } {
  return {
    system:
      '你负责确认一段反复出现的文本是否真的表达了一个稳定的用户偏好或事实。请只返回严格 JSON：{"confirmed": boolean, "reason": string}。只有当文本确实表达了持久的用户偏好、约束或个人资料时，才返回 confirmed=true。如果文本只是临时指令、闲聊、提问或不包含可存储的偏好语义，返回 confirmed=false。',
    user: `原始文本：\n${truncateText(text, 800)}\n\n启发式提取结果：\npredicate: ${preference.predicate}\nobject: ${preference.object}`,
  };
}

function buildConsolidationRelationConfirmPrompt(
  text: string,
  relation: { subject: string; predicate: string; object: string },
): { system: string; user: string } {
  return {
    system:
      '你负责确认一段反复出现的文本是否真的表达了命名实体之间的明确关系。请只返回严格 JSON：{"confirmed": boolean, "reason": string}。只有当文本确实描述了两个具体实体之间的结构化关系时，才返回 confirmed=true。如果关系提取有误、实体不具体、或原文语义实际上不支持该关系，返回 confirmed=false。',
    user: `原始文本：\n${truncateText(text, 800)}\n\n启发式提取结果：\nsubject: ${relation.subject}\npredicate: ${relation.predicate}\nobject: ${relation.object}`,
  };
}

function buildConsolidationBatchConfirmPrompt(items: ConsolidationBatchConfirmItem[]): {
  system: string;
  user: string;
} {
  const block = items
    .map((item, index) => {
      const details =
        item.kind === "fact"
          ? [`predicate: ${item.predicate}`, `object: ${item.object}`]
          : [
              `subject: ${item.subject ?? ""}`,
              `predicate: ${item.predicate}`,
              `object: ${item.object}`,
            ];
      return [
        `${index + 1}. id=${item.id} kind=${item.kind}`,
        ...details,
        `supportCount: ${item.supportCount}`,
        item.latestObservedAt ? `latestObservedAt: ${item.latestObservedAt}` : "",
        item.structuredSummaries.length > 0
          ? `structuredEvidence:\n${item.structuredSummaries.map((summary) => `- ${summary}`).join("\n")}`
          : "",
        item.sourceRefs.length > 0
          ? `sourceRefs:\n${item.sourceRefs.map((sourceRef) => `- ${sourceRef}`).join("\n")}`
          : "",
      ]
        .filter(Boolean)
        .join("\n");
    })
    .join("\n\n");
  return {
    system:
      '你负责批量确认结构化 consolidation 候选。请只返回严格 JSON：{"items":[{"id":string,"decision":"confirm"|"defer"|"reject","reason":string}]}。只能依据给定的结构化证据来判断：confirm 表示证据足够支持晋升；defer 表示证据暂时不足，应等待后续批次；reject 表示结构化候选与证据不符。不要重新解释原始对话文本，不要引入外部常识。',
    user: block || "无候选",
  };
}

function llmUnavailableTaskSummary(): TaskSummary {
  return {
    title: "Conversation task",
    summary: "",
    metadataJson: {
      taskPhase: "investigating",
      closureScore: 0,
      verificationScore: 0,
      contradictionRisk: 0.5,
      summarySource: "llm_unavailable",
      summaryQuality: "working",
    },
  };
}

function materializeTaskSummaryJudgeResult(
  evidence: TaskSummaryEvidenceSet,
  result: TaskSummaryJudgeResult,
): TaskSummary {
  const fallback = llmUnavailableTaskSummary();
  const fallbackPhase = "investigating";
  const rawTaskPhase = isTaskPhase(result.taskPhase) ? result.taskPhase : fallbackPhase;
  const closureScore =
    typeof result.closureScore === "number" && Number.isFinite(result.closureScore)
      ? clamp01(result.closureScore)
      : 0;
  const verificationScore =
    typeof result.verificationScore === "number" && Number.isFinite(result.verificationScore)
      ? clamp01(result.verificationScore)
      : 0;
  const contradictionRisk =
    typeof result.contradictionRisk === "number" && Number.isFinite(result.contradictionRisk)
      ? clamp01(result.contradictionRisk)
      : 0.5;
  const candidateResolution = result.candidateResolution?.trim() || undefined;
  const eventSummary = result.eventSummary?.trim() || undefined;
  const eventType = result.eventType?.trim() || "task_outcome";
  const evidenceChunkIndexes = Array.isArray(result.evidenceChunkIndexes)
    ? result.evidenceChunkIndexes
        .map((value) =>
          typeof value === "number" && Number.isFinite(value) ? Math.trunc(value) : NaN,
        )
        .filter((value) => value >= 1 && value <= evidence.chunks.length)
    : [];
  const resolutionSupport = candidateResolution
    ? taskSummaryEvidenceSupport(candidateResolution, evidence.chunks)
    : undefined;
  const eventSupport = eventSummary
    ? taskSummaryEvidenceSupport(eventSummary, evidence.chunks)
    : undefined;
  const taskPhase = sanitizeResolvedTaskPhase(
    rawTaskPhase,
    verificationScore,
    resolutionSupport,
    fallbackPhase,
    evidence.chunks,
  );
  const sanitizedCandidateResolution =
    candidateResolution && resolutionSupport?.supported ? candidateResolution : undefined;
  const sanitizedEventSummary = eventSummary && eventSupport?.supported ? eventSummary : undefined;
  const confidence = clamp01(
    0.42 * closureScore + 0.38 * verificationScore + 0.2 * (1 - contradictionRisk),
  );
  const canonicalTaskMetadata = sanitizeTaskMetadata({
    ...(result.project?.trim() ? { project: result.project.trim() } : {}),
    ...(result.currentTask?.trim() ? { currentTask: result.currentTask.trim() } : {}),
    ...(result.nextAction?.trim() ? { nextAction: result.nextAction.trim() } : {}),
    ...(result.blocker?.trim() ? { blocker: result.blocker.trim() } : {}),
  });
  const metadataJson = {
    ...canonicalTaskMetadata,
    taskPhase,
    closureScore,
    verificationScore,
    contradictionRisk,
    ...(sanitizedCandidateResolution ? { candidateResolution: sanitizedCandidateResolution } : {}),
  };
  return {
    title: truncateText(result.title?.trim() || fallback.title, 120),
    summary: truncateText(result.summary?.trim() || fallback.summary, 320),
    metadataJson,
    synthesizedEvent: sanitizedEventSummary
      ? {
          eventType,
          summary: truncateText(sanitizedEventSummary, 320),
          phase: taskPhase,
          closureScore,
          verificationScore,
          contradictionRisk,
          confidence,
          promotionScore: confidence,
          evidenceChunkIndexes,
        }
      : undefined,
  };
}

// -- Query route LLM prompt ---

function buildQueryRoutePrompt(query: string): { system: string; user: string } {
  return {
    system:
      '你负责为用户查询选择记忆检索路由类型。请只返回严格 JSON：{"routeType": "workflow"|"factual"|"temporal"|"explanatory"|"mixed"|"unknown", "routeConfidence": number, "reasons": string[]}。workflow 用于问"我在做什么、当前任务、卡点"；factual 用于问"我的偏好、约束、资料"；temporal 用于问"之前发生了什么、历史事件"；explanatory 用于问"为什么、怎么连起来的、依赖关系"。不要依赖固定关键词模板，要理解用户真实意图。',
    user: `用户查询：\n${truncateText(query, 1200)}`,
  };
}

// -- Strategy cluster validation prompt ---

type ClusterValidationResult = {
  valid?: boolean;
  reason?: string;
};

const QUERY_COMPILE_PROMPT_VERSION = "query-compile/v6";
const TURN_SEMANTIC_COMPILE_PROMPT_VERSION = "turn-semantic-compile/v3";
const TURN_SEMANTIC_COMPACT_PROMPT_VERSION = "turn-semantic-compact/v1";

type CompactSemanticAssertion = {
  sourceRef?: string;
  subject?: string;
  subjectType?: string;
  slot?: string;
  value?: string;
  family?: TurnSemanticAssertionFamilyHint;
  timeframe?: TurnSemanticTimeframeHint;
  confidence?: number;
  supportText?: string;
};

type CompactSemanticCorrection = {
  sourceRef?: string;
  subject?: string;
  slot?: string;
  priorValue?: string;
  nextValue?: string;
  predicate?: string;
  canonicalKey?: string;
  confidence?: number;
  supportText?: string;
};

type CompactSemanticRelation = {
  sourceRef?: string;
  subject?: string;
  predicate?: string;
  object?: string;
  polarity?: "affirmed" | "negated";
  slot?: string;
  confidence?: number;
  reason?: string;
  supportText?: string;
};

type CompactTurnSemanticResult = {
  assertions?: CompactSemanticAssertion[];
  corrections?: CompactSemanticCorrection[];
  relations?: CompactSemanticRelation[];
  resourceAssertions?: MemoryResourceAssertionHint[];
  adviceSignals?: MemoryAdviceSignalHint[];
  taskProposal?: Partial<TurnSemanticTaskProposal>;
};

function buildStrategyClusterValidationPrompt(
  entries: Array<{ domain: string; resolution: string }>,
): { system: string; user: string } {
  const block = entries
    .map(
      (entry, index) =>
        `${index + 1}. 领域: ${truncateText(entry.domain, 120)}\n   解法: ${truncateText(entry.resolution, 200)}`,
    )
    .join("\n");
  return {
    system:
      '你负责验证一组任务结果候选是否确实描述了相似的问题解决模式。请只返回严格 JSON：{"valid": boolean, "reason": string}。只有当这些候选描述的问题领域和解决方式有明显共同模式时，才返回 valid=true。如果候选之间差异过大或聚类存在误判，返回 valid=false。',
    user: `候选聚类：\n${block}`,
  };
}

async function callJudgeModel(
  cfg: JudgeModelConfig,
  prompt: { system: string; user: string },
  options: {
    maxTokens?: number;
    jsonMode?: boolean;
    temperature?: number;
    signal?: AbortSignal;
  } = {},
): Promise<string> {
  const maxTokens = Math.max(128, Math.min(1600, options.maxTokens ?? 800));
  const temperature = typeof options.temperature === "number" ? options.temperature : 0.1;
  const requestSignal = options.signal
    ? AbortSignal.any([options.signal, AbortSignal.timeout(30_000)])
    : AbortSignal.timeout(30_000);
  if (cfg.provider === "anthropic") {
    const response = await fetch(normalizeAnthropicEndpoint(cfg.baseUrl), {
      method: "POST",
      headers: {
        "content-type": "application/json",
        "x-api-key": cfg.apiKey ?? "",
        "anthropic-version": "2023-06-01",
        ...(cfg.headers ?? {}),
      },
      body: JSON.stringify({
        model: cfg.model,
        max_tokens: maxTokens,
        temperature,
        system: prompt.system,
        messages: [{ role: "user", content: prompt.user }],
      }),
      signal: requestSignal,
    });
    if (!response.ok) {
      throw new Error(`anthropic ${response.status}: ${await response.text()}`);
    }
    const json = (await response.json()) as { content?: unknown[] };
    return extractTextContent(json.content);
  }

  if (cfg.provider === "google") {
    const response = await fetch(
      normalizeGoogleEndpoint(cfg.baseUrl, cfg.model, cfg.apiKey ?? ""),
      {
        method: "POST",
        headers: {
          "content-type": "application/json",
          ...(cfg.headers ?? {}),
        },
        body: JSON.stringify({
          contents: [
            {
              role: "user",
              parts: [{ text: `${prompt.system}\n\n${prompt.user}` }],
            },
          ],
          generationConfig: {
            temperature,
            maxOutputTokens: maxTokens,
          },
        }),
        signal: requestSignal,
      },
    );
    if (!response.ok) {
      throw new Error(`google ${response.status}: ${await response.text()}`);
    }
    const json = (await response.json()) as {
      candidates?: Array<{ content?: { parts?: Array<{ text?: string }> } }>;
    };
    return extractTextContent(json.candidates?.[0]?.content?.parts ?? []);
  }

  if (cfg.provider === "ollama") {
    const response = await fetch(normalizeOllamaEndpoint(cfg.baseUrl), {
      method: "POST",
      headers: {
        "content-type": "application/json",
        ...(cfg.headers ?? {}),
      },
      body: JSON.stringify({
        model: cfg.model,
        stream: false,
        options: { temperature, num_predict: maxTokens },
        messages: [
          { role: "system", content: prompt.system },
          { role: "user", content: prompt.user },
        ],
      }),
      signal: requestSignal,
    });
    if (!response.ok) {
      throw new Error(`ollama ${response.status}: ${await response.text()}`);
    }
    const json = (await response.json()) as { message?: { content?: string } };
    return json.message?.content?.trim() ?? "";
  }

  const endpoint = normalizeOpenAiEndpoint(cfg.baseUrl);
  const headers = {
    "content-type": "application/json",
    ...(cfg.apiKey ? { authorization: `Bearer ${cfg.apiKey}` } : {}),
    ...(cfg.headers ?? {}),
  };
  const baseBody = {
    model: cfg.model,
    temperature,
    max_tokens: maxTokens,
    messages: [
      { role: "system", content: prompt.system },
      { role: "user", content: prompt.user },
    ],
  };
  const parseOpenAiCompatibleResponse = async (response: Response): Promise<string> => {
    if (!response.ok) {
      throw new Error(`openai-compatible ${response.status}: ${await response.text()}`);
    }
    const json = (await response.json()) as {
      choices?: Array<{ message?: { content?: string | unknown[] } }>;
      output_text?: string;
    };
    return extractTextContent(json.output_text || json.choices?.[0]?.message?.content);
  };
  if (options.jsonMode) {
    const response = await fetch(endpoint, {
      method: "POST",
      headers,
      body: JSON.stringify({
        ...baseBody,
        response_format: { type: "json_object" },
      }),
      signal: requestSignal,
    });
    if (response.ok) {
      const jsonModeText = await parseOpenAiCompatibleResponse(response);
      if (jsonModeText.trim()) {
        return jsonModeText;
      }
      // Some OpenAI-compatible providers accept response_format=json_object
      // but can still return an empty content field. Retry without JSON mode.
    } else {
      const detail = await response.text();
      // Some OpenAI-compatible providers reject response_format=json_object; fall back
      // to the plain request instead of treating that capability mismatch as fatal.
      if (
        response.status !== 400 &&
        response.status !== 404 &&
        response.status !== 422 &&
        response.status !== 500
      ) {
        throw new Error(`openai-compatible ${response.status}: ${detail}`);
      }
    }
  }
  const response = await fetch(endpoint, {
    method: "POST",
    headers,
    body: JSON.stringify(baseBody),
    signal: requestSignal,
  });
  return parseOpenAiCompatibleResponse(response);
}

function isAbortLike(error: unknown): boolean {
  if (error instanceof DOMException && error.name === "AbortError") {
    return true;
  }
  if (error instanceof Error) {
    return error.name === "AbortError" || /aborted|aborterror/iu.test(error.message);
  }
  return false;
}

export class MemxReasoner {
  private readonly judgeModel: JudgeModelConfig | null;
  private readonly warned = new Set<string>();
  private traces: ReasonerTraceEntry[] = [];

  constructor(
    private readonly config: MemoryPluginConfig,
    private readonly logger: MemxLogger,
  ) {
    this.judgeModel = loadJudgeModelConfig(config, logger);
  }

  async summarizeChunk(
    text: string,
    role: ConversationChunk["role"] = "user",
    options: ReasonerCallAuditOptions & { allowLlm?: boolean } = {},
  ): Promise<string> {
    const fallback = localChunkPreview(text);
    if (options.allowLlm === false) {
      this.recordTrace({
        label: "chunk-summary",
        mode: "degraded",
        provenance: "deterministic",
        detail: "Chunk summary LLM was explicitly disabled; stored only a local preview.",
        stage: options.stage,
      });
      recordMemoryLlmBudgetCall(options.audit, {
        label: "chunk-summary",
        stage: options.stage ?? "write_hot_path",
        provenance: "deterministic",
        mode: "degraded",
        detail: "chunk-summary LLM explicitly disabled; local preview is not semantic extraction",
      });
      return fallback;
    }
    const result = await this.callJson<ChunkSummaryJudgeResult>(
      "chunk-summary",
      buildChunkPrompt(text, role),
      "degraded",
      options,
    );
    return sanitizeChunkSummaryResult(text, role, result);
  }

  async summarizeTask(
    chunks: ConversationChunk[],
    options: ReasonerCallAuditOptions & { allowLlm?: boolean } = {},
  ): Promise<TaskSummary> {
    const fallback = llmUnavailableTaskSummary();
    if (options.allowLlm === false) {
      this.recordTrace({
        label: "task-summary",
        mode: "degraded",
        provenance: "deterministic",
        detail: "Task summary LLM was explicitly disabled; semantic summary is unavailable.",
        stage: options.stage,
      });
      recordMemoryLlmBudgetCall(options.audit, {
        label: "task-summary",
        stage: options.stage ?? "write_hot_path",
        provenance: "deterministic",
        mode: "degraded",
        detail: "task-summary LLM explicitly disabled; semantic task summary unavailable",
      });
      return fallback;
    }
    const promptChunks = chunks.slice(-16);
    const result = await this.callJson<TaskSummaryJudgeResult>(
      "task-summary",
      buildTaskPrompt(promptChunks),
      "degraded",
      options,
    );
    if (!result) {
      return fallback;
    }
    const fallbackPhase = "investigating";
    const rawTaskPhase = isTaskPhase(result.taskPhase) ? result.taskPhase : fallbackPhase;
    const closureScore =
      typeof result.closureScore === "number" && Number.isFinite(result.closureScore)
        ? clamp01(result.closureScore)
        : 0;
    const verificationScore =
      typeof result.verificationScore === "number" && Number.isFinite(result.verificationScore)
        ? clamp01(result.verificationScore)
        : 0;
    const contradictionRisk =
      typeof result.contradictionRisk === "number" && Number.isFinite(result.contradictionRisk)
        ? clamp01(result.contradictionRisk)
        : 0.5;
    const candidateResolution = result.candidateResolution?.trim() || undefined;
    const eventSummary = result.eventSummary?.trim() || undefined;
    const eventType = result.eventType?.trim() || "task_outcome";
    const evidenceChunkIndexes = Array.isArray(result.evidenceChunkIndexes)
      ? result.evidenceChunkIndexes
          .map((value) =>
            typeof value === "number" && Number.isFinite(value) ? Math.trunc(value) : NaN,
          )
          .filter((value) => value >= 1 && value <= promptChunks.length)
      : [];
    const resolutionSupport = candidateResolution
      ? taskSummaryEvidenceSupport(candidateResolution, promptChunks)
      : undefined;
    const eventSupport = eventSummary
      ? taskSummaryEvidenceSupport(eventSummary, promptChunks)
      : undefined;
    const taskPhase = sanitizeResolvedTaskPhase(
      rawTaskPhase,
      verificationScore,
      resolutionSupport,
      fallbackPhase,
      promptChunks,
    );
    const sanitizedCandidateResolution =
      candidateResolution && resolutionSupport?.supported ? candidateResolution : undefined;
    const sanitizedEventSummary =
      eventSummary && eventSupport?.supported ? eventSummary : undefined;
    const confidence = clamp01(
      0.42 * closureScore + 0.38 * verificationScore + 0.2 * (1 - contradictionRisk),
    );
    const canonicalTaskMetadata = sanitizeTaskMetadata({
      ...(result.project?.trim() ? { project: result.project.trim() } : {}),
      ...(result.currentTask?.trim() ? { currentTask: result.currentTask.trim() } : {}),
      ...(result.nextAction?.trim() ? { nextAction: result.nextAction.trim() } : {}),
      ...(result.blocker?.trim() ? { blocker: result.blocker.trim() } : {}),
    });
    const metadataJson = {
      ...canonicalTaskMetadata,
      taskPhase,
      closureScore,
      verificationScore,
      contradictionRisk,
      ...(sanitizedCandidateResolution
        ? { candidateResolution: sanitizedCandidateResolution }
        : {}),
    };
    return {
      title: truncateText(result.title?.trim() || fallback.title, 120),
      summary: truncateText(result.summary?.trim() || fallback.summary, 320),
      metadataJson,
      synthesizedEvent: sanitizedEventSummary
        ? {
            eventType,
            summary: truncateText(sanitizedEventSummary, 320),
            phase: taskPhase,
            closureScore,
            verificationScore,
            contradictionRisk,
            confidence,
            promotionScore: confidence,
            evidenceChunkIndexes,
          }
        : undefined,
    };
  }

  async summarizeTaskFromEvidence(
    evidence: TaskSummaryEvidenceSet,
    options: ReasonerCallAuditOptions = {},
  ): Promise<TaskSummary | null> {
    if (evidence.chunks.length === 0) {
      return null;
    }
    const result = await this.callJson<TaskSummaryJudgeResult>(
      "task-summary",
      buildTaskSummaryEvidencePrompt(evidence),
      "degraded",
      {
        ...options,
        maxTokens: Math.min(options.maxTokens ?? 1200, 1200),
        jsonMode: true,
        temperature: 0,
      },
    );
    if (!result) {
      return null;
    }
    return materializeTaskSummaryJudgeResult(evidence, result);
  }

  async summarizeTaskEvidenceBatch(
    evidenceSets: TaskSummaryEvidenceSet[],
    options: ReasonerCallAuditOptions = {},
  ): Promise<Map<string, TaskSummary> | null> {
    const selected = evidenceSets.filter((entry) => entry.chunks.length > 0);
    if (selected.length === 0) {
      return null;
    }
    const result = await this.callJson<TaskSummaryBatchJudgeResult>(
      "task-summary-batch",
      buildTaskSummaryEvidenceBatchPrompt(selected),
      "degraded",
      {
        ...options,
        maxTokens: Math.min(options.maxTokens ?? 1200, 1200),
        jsonMode: true,
        temperature: 0,
      },
    );
    if (!result?.tasks || result.tasks.length === 0) {
      return null;
    }
    const evidenceByTaskId = new Map(selected.map((entry) => [entry.taskId, entry]));
    const summaries = new Map<string, TaskSummary>();
    for (const entry of result.tasks) {
      const taskId = typeof entry?.taskId === "string" ? entry.taskId : undefined;
      if (!taskId) {
        continue;
      }
      const evidence = evidenceByTaskId.get(taskId);
      if (!evidence) {
        continue;
      }
      summaries.set(taskId, materializeTaskSummaryJudgeResult(evidence, entry));
    }
    return summaries.size > 0 ? summaries : null;
  }

  async confirmConsolidationBatch(
    items: ConsolidationBatchConfirmItem[],
    options: ReasonerCallAuditOptions = {},
  ): Promise<Map<string, { decision: "confirm" | "defer" | "reject"; reason?: string }> | null> {
    if (items.length === 0) {
      return null;
    }
    const result = await this.callJson<ConsolidationBatchConfirmJudgeResult>(
      "consolidation-confirm-batch",
      buildConsolidationBatchConfirmPrompt(items),
      "degraded",
      {
        ...options,
        stage: options.stage ?? "maintenance_async",
      },
    );
    if (!result?.items || result.items.length === 0) {
      return null;
    }
    const decisions = new Map<
      string,
      { decision: "confirm" | "defer" | "reject"; reason?: string }
    >();
    for (const item of result.items) {
      const id = typeof item?.id === "string" ? item.id : undefined;
      if (!id) {
        continue;
      }
      const decision =
        item.decision === "confirm" || item.decision === "defer" || item.decision === "reject"
          ? item.decision
          : undefined;
      if (!decision) {
        continue;
      }
      decisions.set(id, {
        decision,
        ...(typeof item.reason === "string" && item.reason.trim()
          ? { reason: item.reason.trim() }
          : {}),
      });
    }
    return decisions.size > 0 ? decisions : null;
  }

  async judgeTaskAssignment(
    currentTask: TaskAssignmentSnapshot | null,
    candidates: TaskAssignmentSnapshot[],
    incomingTurn: string,
  ): Promise<TaskAssignmentDecision | null> {
    if (!incomingTurn.trim()) {
      return null;
    }
    const result = await this.callJson<TaskAssignmentJudgeResult>(
      "task-assignment",
      buildTaskAssignmentPrompt(currentTask, candidates, incomingTurn),
      "degraded",
    );
    if (!result) {
      return null;
    }
    const decision =
      result.decision === "continue" || result.decision === "resume" || result.decision === "new"
        ? result.decision
        : undefined;
    if (!decision) {
      return null;
    }
    const allowedTaskIds = new Set(candidates.map((candidate) => candidate.taskId));
    const normalizedTarget =
      typeof result.targetTaskId === "string" && allowedTaskIds.has(result.targetTaskId)
        ? result.targetTaskId
        : undefined;
    if (decision === "resume" && !normalizedTarget) {
      return null;
    }
    if (decision === "continue" && !currentTask) {
      return null;
    }
    return {
      decision,
      targetTaskId: normalizedTarget,
      confidence:
        typeof result.confidence === "number" && Number.isFinite(result.confidence)
          ? clamp01(result.confidence)
          : 0.6,
      reason: result.reason?.trim() || "llm task assignment",
    };
  }

  async judgeNewTopic(currentContext: string, newMessage: string): Promise<boolean | null> {
    const result = await this.callJson<{ isNewTopic?: boolean }>(
      "topic-judge",
      buildTopicPrompt(currentContext, newMessage),
      "degraded",
    );
    if (typeof result?.isNewTopic === "boolean") {
      return result.isNewTopic;
    }
    return null;
  }

  async judgeDedup(
    newSummary: string,
    sourceText: string,
    candidates: Array<{ index: number; summary: string; text?: string }>,
  ): Promise<DedupDecision | null> {
    const result = await this.callJson<DedupDecision>(
      "dedup-judge",
      buildDedupPrompt(newSummary, sourceText, candidates),
      "degraded",
    );
    if (!result?.action) {
      return conservativeDedupDecision(newSummary, candidates);
    }
    return {
      action: result.action,
      targetIndex: typeof result.targetIndex === "number" ? result.targetIndex : undefined,
      mergedSummary: result.mergedSummary?.trim(),
      reason: result.reason?.trim() || "llm dedup",
    };
  }

  async filterRelevant(
    query: string,
    candidates: Array<{ index: number; summary: string; role: string }>,
  ): Promise<RelevantDecision | null> {
    const result = await this.callJson<RelevantDecision>(
      "relevance-filter",
      buildRelevancePrompt(query, candidates),
      "degraded",
    );
    if (!result || !Array.isArray(result.relevant)) {
      return conservativeRelevantDecision();
    }
    return {
      relevant: result.relevant.filter((value) => Number.isInteger(value)),
      sufficient: Boolean(result.sufficient),
      reason: result.reason?.trim() || "llm relevance filter",
    };
  }

  async planRecall(query: string): Promise<MemoryRecallPlan> {
    const fallback = conservativeRecallPlan(query, this.isEnabled() ? "degraded" : "disabled");
    const result = await this.callJson<RecallPlanDecision>(
      "recall-plan",
      buildRecallPlanPrompt(query),
      "degraded",
    );
    if (!result) {
      return fallback;
    }
    const focusedQuery = truncateText(result.focusedQuery?.trim() || fallback.focusedQuery, 160);
    return {
      shouldRecall: Boolean(focusedQuery || fallback.focusedQuery),
      focusedQuery: focusedQuery || fallback.focusedQuery,
      reason: result.reason?.trim() || fallback.reason,
      routeHint:
        result.routeHint === "workflow" ||
        result.routeHint === "factual" ||
        result.routeHint === "temporal" ||
        result.routeHint === "explanatory" ||
        result.routeHint === "mixed"
          ? result.routeHint
          : fallback.routeHint,
      judgmentMode: "llm",
    };
  }

  async judgeRoutePrior(query: string): Promise<RoutePriorDecision> {
    const fallback = conservativeRoutePrior(query, this.isEnabled() ? "degraded" : "disabled");
    const result = await this.callJson<RoutePriorDecision>(
      "route-prior",
      buildRoutePriorPrompt(query),
      "degraded",
    );
    if (!result) {
      // Hard failure (unparseable/empty LLM response) — stay conservative so retrieve falls back
      // to score-driven evaluation rather than guessing a route from shallow keywords.
      return fallback;
    }

    const primaryRoute = isRouteType(result.primaryRoute)
      ? result.primaryRoute
      : fallback.primaryRoute;
    const secondaryRoutes = Array.isArray(result.secondaryRoutes)
      ? result.secondaryRoutes.filter((route): route is MemoryPrimaryRouteType =>
          isPrimaryRouteType(route),
        )
      : fallback.secondaryRoutes;
    const focusedQueries: RoutePriorDecision["focusedQueries"] = {};
    for (const route of PRIMARY_ROUTE_TYPES) {
      const value = result.focusedQueries?.[route];
      const next =
        typeof value === "string" && value.trim()
          ? truncateText(value.trim(), 180)
          : fallback.focusedQueries[route];
      if (next) {
        focusedQueries[route] = next;
      }
    }

    return {
      primaryRoute,
      secondaryRoutes,
      confidence:
        typeof result.confidence === "number" && Number.isFinite(result.confidence)
          ? clamp01(result.confidence)
          : fallback.confidence,
      focusedQueries,
      reason: result.reason?.trim() || fallback.reason,
      judgmentMode: "llm",
    };
  }

  /**
   * Combined single-LLM-call version of planRecall + judgeRoutePrior.
   * Saves one sequential LLM round-trip (~3.7s) by asking a single prompt
   * that covers both recall planning and route prior selection.
   */
  async planRecallWithRoute(
    query: string,
  ): Promise<{ plan: MemoryRecallPlan; routePrior: RoutePriorDecision }> {
    const planFallback = conservativeRecallPlan(query, this.isEnabled() ? "degraded" : "disabled");
    const routeFallback = conservativeRoutePrior(query, this.isEnabled() ? "degraded" : "disabled");
    const result = await this.callJson<{
      focusedQuery?: string;
      reason?: string;
      routeHint?: string;
      primaryRoute?: string;
      secondaryRoutes?: string[];
      routeConfidence?: number;
      focusedQueries?: Record<string, string>;
      routeReason?: string;
    }>("recall-plan-with-route", buildRecallPlanWithRoutePrompt(query), "degraded");

    if (!result) {
      return { plan: planFallback, routePrior: routeFallback };
    }

    // Parse plan portion
    const focusedQuery = truncateText(
      result.focusedQuery?.trim() || planFallback.focusedQuery,
      160,
    );
    const plan: MemoryRecallPlan = {
      shouldRecall: Boolean(focusedQuery || planFallback.focusedQuery),
      focusedQuery: focusedQuery || planFallback.focusedQuery,
      reason: result.reason?.trim() || planFallback.reason,
      routeHint:
        result.routeHint === "workflow" ||
        result.routeHint === "factual" ||
        result.routeHint === "temporal" ||
        result.routeHint === "explanatory" ||
        result.routeHint === "mixed"
          ? result.routeHint
          : planFallback.routeHint,
      judgmentMode: "llm",
    };

    const primaryRoute = isRouteType(result.primaryRoute)
      ? (result.primaryRoute as MemoryRouteType)
      : routeFallback.primaryRoute;
    const secondaryRoutes = Array.isArray(result.secondaryRoutes)
      ? result.secondaryRoutes.filter((route): route is MemoryPrimaryRouteType =>
          isPrimaryRouteType(route),
        )
      : routeFallback.secondaryRoutes;
    const routeFocusedQueries: RoutePriorDecision["focusedQueries"] = {};
    for (const route of PRIMARY_ROUTE_TYPES) {
      const value = result.focusedQueries?.[route];
      const next =
        typeof value === "string" && value.trim()
          ? truncateText(value.trim(), 180)
          : routeFallback.focusedQueries[route];
      if (next) {
        routeFocusedQueries[route] = next;
      }
    }
    const routePrior: RoutePriorDecision = {
      primaryRoute,
      secondaryRoutes,
      confidence:
        typeof result.routeConfidence === "number" && Number.isFinite(result.routeConfidence)
          ? clamp01(result.routeConfidence)
          : routeFallback.confidence,
      focusedQueries: routeFocusedQueries,
      reason:
        result.routeReason?.trim() ||
        result.reason?.trim() ||
        routeFallback.reason,
      judgmentMode: "llm",
    };

    return { plan, routePrior };
  }

  async compileQuerySemantics(
    query: string,
    fallback: QueryCompileResult,
    options: ReasonerCallAuditOptions = {},
  ): Promise<Partial<QueryCompileResult> | null> {
    let result = await this.callJson<Partial<QueryCompileResult>>(
      "query-compile",
      buildQueryCompilePrompt(query, fallback),
      "degraded",
      {
        ...options,
        maxTokens: Math.min(options.maxTokens ?? 560, 700),
        jsonMode: true,
        temperature: 0,
      },
    );
    if (!result && !options.signal?.aborted) {
      result = await this.callJson<Partial<QueryCompileResult>>(
        "query-compile-retry",
        buildQueryCompilePrompt(query, fallback),
        "degraded",
        {
          ...options,
          maxTokens: Math.min(options.maxTokens ?? 560, 700),
          jsonMode: false,
          temperature: 0,
        },
      );
    }
    if (!result) {
      return null;
    }
    return {
      ...result,
      compilerProvenance: {
        source: "llm",
        mode: "llm",
        promptVersion: QUERY_COMPILE_PROMPT_VERSION,
        model: this.judgeModel?.model,
      },
    };
  }

  async compileTurnSemantics(
    messages: TurnCaptureMessage[],
    fallback: TurnSemanticFrame,
    options: ReasonerCallAuditOptions = {},
  ): Promise<Partial<TurnSemanticFrame> | null> {
    let compactResult = await this.callJson<CompactTurnSemanticResult>(
      "turn-semantic-compact",
      buildCompactTurnSemanticPrompt(messages, fallback),
      "degraded",
      {
        ...options,
        maxTokens: Math.min(options.maxTokens ?? 900, 900),
        jsonMode: true,
        temperature: 0,
      },
    );
    if (!compactResult && shouldRetrySemanticCompile(options)) {
      compactResult = await this.callJson<CompactTurnSemanticResult>(
        "turn-semantic-compact-retry",
        buildCompactTurnSemanticPrompt(messages, fallback),
        "degraded",
        {
          ...options,
          maxTokens: Math.min(options.maxTokens ?? 900, 900),
          jsonMode: false,
          temperature: 0,
        },
      );
    }
    if (compactResult && hasRecognizedCompactSemanticResult(compactResult)) {
      const normalizedCompact = normalizeCompiledTurnSemanticPatch(
        messages,
        compactSemanticResultToPatch(compactResult),
      );
      return {
        ...normalizedCompact,
        compilerProvenance: {
          source: "llm",
          mode: "llm",
          promptVersion: TURN_SEMANTIC_COMPACT_PROMPT_VERSION,
          model: this.judgeModel?.model,
        },
      };
    }

    let result = await this.callJson<Partial<TurnSemanticFrame>>(
      "turn-semantic-compile",
      buildTurnSemanticCompilePrompt(messages, fallback),
      "degraded",
      {
        ...options,
        maxTokens: Math.min(options.maxTokens ?? 1200, 1200),
        jsonMode: true,
        temperature: 0,
      },
    );
    if (!result && shouldRetrySemanticCompile(options)) {
      result = await this.callJson<Partial<TurnSemanticFrame>>(
        "turn-semantic-compile-retry",
        buildTurnSemanticCompilePrompt(messages, fallback),
        "degraded",
        {
          ...options,
          maxTokens: Math.min(options.maxTokens ?? 1200, 1200),
          jsonMode: false,
          temperature: 0,
        },
      );
    }
    if (!result || !hasRecognizedTurnSemanticPatch(result)) {
      return null;
    }
    const normalized = normalizeCompiledTurnSemanticPatch(messages, result);
    return {
      ...normalized,
      compilerProvenance: {
        source: "llm",
        mode: "llm",
        promptVersion: TURN_SEMANTIC_COMPILE_PROMPT_VERSION,
        model: this.judgeModel?.model,
      },
    };
  }

  async compileLongTurnSemantics(
    input: LongTurnSemanticScanInput,
    fallback: TurnSemanticFrame,
    options: ReasonerCallAuditOptions = {},
  ): Promise<Partial<TurnSemanticFrame> | null> {
    const messagesBySourceRef = new Map<string, TurnCaptureMessage>(
      fallback.sourceRefs.map((sourceRef) => [
        sourceRef,
        {
          role: "user" as const,
          content: "",
          scope: "",
          sessionKey: "",
          agentId: "long-turn-semantic-scan",
          turnId: sourceRef,
          sourceRef,
          observedAt: "",
        },
      ]),
    );
    const syntheticMessages: TurnCaptureMessage[] = input.messages.map((message) => ({
      role: message.role,
      content: "",
      scope: "",
      sessionKey: "",
      agentId: "long-turn-semantic-scan",
      turnId: message.turnId,
      sourceRef: message.sourceRef,
      observedAt: "",
    }));
    for (const message of syntheticMessages) {
      messagesBySourceRef.set(message.sourceRef, message);
    }
    const syntheticFrameMessages = [...messagesBySourceRef.values()];
    let compactResult = await this.callJson<CompactTurnSemanticResult>(
      "turn-semantic-long-scan-compact",
      buildCompactLongTurnSemanticPrompt(input, fallback),
      "degraded",
      {
        ...options,
        maxTokens: Math.min(options.maxTokens ?? 1000, 1000),
        jsonMode: true,
        temperature: 0,
      },
    );
    if (!compactResult && shouldRetrySemanticCompile(options)) {
      compactResult = await this.callJson<CompactTurnSemanticResult>(
        "turn-semantic-long-scan-compact-retry",
        buildCompactLongTurnSemanticPrompt(input, fallback),
        "degraded",
        {
          ...options,
          maxTokens: Math.min(options.maxTokens ?? 1000, 1000),
          jsonMode: false,
          temperature: 0,
        },
      );
    }
    if (compactResult && hasRecognizedCompactSemanticResult(compactResult)) {
      const normalizedCompact = normalizeCompiledTurnSemanticPatch(
        syntheticFrameMessages,
        compactSemanticResultToPatch(compactResult),
      );
      return {
        ...normalizedCompact,
        compilerProvenance: {
          source: "llm",
          mode: "llm",
          promptVersion: `${TURN_SEMANTIC_COMPACT_PROMPT_VERSION}:long-turn-scan`,
          model: this.judgeModel?.model,
          reasons: ["long-turn-semantic-scan"],
        },
      };
    }

    let result = await this.callJson<Partial<TurnSemanticFrame>>(
      "turn-semantic-long-scan",
      buildLongTurnSemanticScanPrompt(input, fallback),
      "degraded",
      {
        ...options,
        maxTokens: Math.min(options.maxTokens ?? 1400, 1400),
        jsonMode: true,
        temperature: 0,
      },
    );
    if (!result && shouldRetrySemanticCompile(options)) {
      result = await this.callJson<Partial<TurnSemanticFrame>>(
        "turn-semantic-long-scan-retry",
        buildLongTurnSemanticScanPrompt(input, fallback),
        "degraded",
        {
          ...options,
          maxTokens: Math.min(options.maxTokens ?? 1400, 1400),
          jsonMode: false,
          temperature: 0,
        },
      );
    }
    if (!result || !hasRecognizedTurnSemanticPatch(result)) {
      return null;
    }
    const normalized = normalizeCompiledTurnSemanticPatch(
      syntheticFrameMessages,
      result,
    );
    return {
      ...normalized,
      compilerProvenance: {
        source: "llm",
        mode: "llm",
        promptVersion: `${TURN_SEMANTIC_COMPILE_PROMPT_VERSION}:long-turn-scan`,
        model: this.judgeModel?.model,
        reasons: ["long-turn-semantic-scan"],
      },
    };
  }

  async judgeRouteEvidence(
    query: string,
    routeType: MemoryPrimaryRouteType,
    candidates: RouteEvidenceCandidate[],
  ): Promise<RouteEvidenceDecision> {
    const fallback = conservativeRouteEvidenceDecision(
      routeType,
      this.isEnabled() ? "degraded" : "disabled",
    );
    const result = await this.callJson<RouteEvidenceDecision>(
      `route-evidence:${routeType}`,
      buildRouteEvidencePrompt(query, routeType, candidates),
      "degraded",
    );
    if (!result) {
      return fallback;
    }

    const validSupport = typeof result.support === "number" && Number.isFinite(result.support);
    return {
      relevant: Array.isArray(result.relevant)
        ? result.relevant.filter((value) => Number.isInteger(value))
        : fallback.relevant,
      sufficient: typeof result.sufficient === "boolean" ? result.sufficient : fallback.sufficient,
      support: validSupport ? clamp01(result.support as number) : fallback.support,
      reason: result.reason?.trim() || fallback.reason,
      // When support is missing, signal the caller to use score-driven evaluation instead
      // of trusting a structurally invalid LLM response.
      judgmentMode: validSupport ? "llm" : "degraded",
    };
  }

  async judgeOutcomePromotion(
    task: TaskAssignmentSnapshot,
    proposal: SynthesizedTaskEvent,
    evidenceChunks: ConversationChunk[],
  ): Promise<OutcomePromotionDecision> {
    const fallback = conservativeOutcomePromotionDecision();
    const result = await this.callJson<OutcomePromotionJudgeResult>(
      "outcome-promotion",
      buildOutcomePromotionPrompt({ task, proposal, evidenceChunks }),
      "degraded",
    );
    if (!result) {
      return fallback;
    }
    return {
      shouldPromote:
        typeof result.shouldPromote === "boolean" ? result.shouldPromote : fallback.shouldPromote,
      promotionScore:
        typeof result.promotionScore === "number" && Number.isFinite(result.promotionScore)
          ? clamp01(result.promotionScore)
          : fallback.promotionScore,
      reason: result.reason?.trim() || fallback.reason,
    };
  }

  async judgeAbstractionCandidate(
    candidate: AbstractionCandidateRecord,
    options: ReasonerCallAuditOptions = {},
  ): Promise<AbstractionCandidateJudgeResult | null> {
    const result = await this.callJson<AbstractionCandidateJudgeResult>(
      "abstraction-refinement",
      buildAbstractionCandidatePrompt(candidate),
      "degraded",
      {
        ...options,
        stage: options.stage ?? "maintenance_async",
      },
    );
    if (!result) {
      return null;
    }
    const summary =
      typeof result.summary === "string" ? truncateText(result.summary.trim(), 220) : undefined;
    const displayName =
      typeof result.displayName === "string" && result.displayName.trim()
        ? truncateText(result.displayName.trim(), 80)
        : undefined;
    const stage = isAbstractionCandidateJudgeStage(result.stage) ? result.stage : undefined;
    const reason = result.reason?.trim() || "llm abstraction refinement";
    if (!summary && !displayName && !stage) {
      return { reason };
    }
    return {
      ...(summary ? { summary } : {}),
      ...(displayName ? { displayName } : {}),
      ...(stage ? { stage } : {}),
      reason,
    };
  }

  async judgeCandidatePolicy(
    candidate: MemoryCandidate,
    options: ReasonerCallAuditOptions = {},
  ): Promise<CandidatePolicyJudgeResult | null> {
    const result = await this.callJson<CandidatePolicyJudgeResult>(
      "candidate-policy",
      buildCandidatePolicyPrompt(candidate, this.config),
      "degraded",
      options,
    );
    if (!result || !isMemoryAction(result.action)) {
      return null;
    }
    const workflows = normalizeWorkflowHints((result as Record<string, unknown>).workflows);
    const workflow = normalizeWorkflowHint(result.workflow);
    const normalizedWorkflows =
      workflows.length > 0 ? workflows : workflow ? [workflow] : undefined;
    const relations = normalizeRelationHints((result as Record<string, unknown>).relations);
    const relation = normalizeRelationHint(result.relation);
    const normalizedRelations =
      relations.length > 0 ? relations : relation ? [relation] : undefined;
    return {
      action: result.action,
      salienceScore:
        typeof result.salienceScore === "number" && Number.isFinite(result.salienceScore)
          ? clamp01(result.salienceScore)
          : undefined,
      expectedFutureUtility:
        typeof result.expectedFutureUtility === "number" &&
        Number.isFinite(result.expectedFutureUtility)
          ? clamp01(result.expectedFutureUtility)
          : undefined,
      stabilityScore:
        typeof result.stabilityScore === "number" && Number.isFinite(result.stabilityScore)
          ? clamp01(result.stabilityScore)
          : undefined,
      preference: normalizePreferenceHint(result.preference),
      workflow: normalizedWorkflows?.[0],
      workflows: normalizedWorkflows,
      relation: normalizedRelations?.[0],
      relations: normalizedRelations,
      decision: normalizeDecisionHint(result.decision),
      reason: result.reason?.trim() || "llm candidate policy",
    };
  }

  async confirmConsolidationFact(
    text: string,
    preference: { predicate: string; object: string },
    options: ReasonerCallAuditOptions = {},
  ): Promise<boolean> {
    const result = await this.callJson<ConsolidationConfirmResult>(
      "consolidation-fact-confirm",
      buildConsolidationFactConfirmPrompt(text, preference),
      "degraded",
      {
        ...options,
        stage: options.stage ?? "maintenance_async",
      },
    );
    return result?.confirmed === true;
  }

  async confirmConsolidationRelation(
    text: string,
    relation: { subject: string; predicate: string; object: string },
    options: ReasonerCallAuditOptions = {},
  ): Promise<boolean> {
    const result = await this.callJson<ConsolidationConfirmResult>(
      "consolidation-relation-confirm",
      buildConsolidationRelationConfirmPrompt(text, relation),
      "degraded",
      {
        ...options,
        stage: options.stage ?? "maintenance_async",
      },
    );
    return result?.confirmed === true;
  }

  async judgeQueryRouteWithLlm(query: string): Promise<RouteDecision | null> {
    const result = await this.callJson<RouteDecision>(
      "query-route",
      buildQueryRoutePrompt(query),
      "degraded",
    );
    if (
      !result ||
      typeof result.routeType !== "string" ||
      typeof result.routeConfidence !== "number"
    ) {
      return null;
    }
    return {
      routeType: result.routeType,
      routeConfidence: clamp01(result.routeConfidence),
      reasons: Array.isArray(result.reasons)
        ? result.reasons.filter((r): r is string => typeof r === "string")
        : ["llm-route"],
    };
  }

  async validateStrategyCluster(
    entries: Array<{ domain: string; resolution: string }>,
  ): Promise<boolean> {
    const result = await this.callJson<ClusterValidationResult>(
      "strategy-cluster-validation",
      buildStrategyClusterValidationPrompt(entries),
      "degraded",
    );
    return result?.valid === true;
  }

  getResolvedJudgeModel(): { provider: SupportedJudgeProvider; model: string } | null {
    return this.judgeModel
      ? {
          provider: this.judgeModel.provider,
          model: this.judgeModel.model,
        }
      : null;
  }

  getResolvedJudgeConfigPath(): string | null {
    return this.judgeModel?.configPath ?? null;
  }

  getTrace(): ReasonerTraceEntry[] {
    return [...this.traces];
  }

  clearTrace(): void {
    this.traces = [];
  }

  async runProbeSuite(): Promise<ReasonerProbeReport> {
    this.clearTrace();
    const chunkSummary = await this.summarizeChunk(
      "I am working on retrieval routing, graph expansion is still unclear, and the next step is to add replay tests.",
    );
    const task = await this.summarizeTask([
      {
        chunkId: "probe-chunk-1",
        agentId: "probe",
        scope: "agent:probe",
        sessionKey: "probe",
        turnId: "probe-turn-1",
        seq: 0,
        role: "user",
        chunkKind: "message",
        content: "I am working on retrieval routing for the memx plugin.",
        summary: "retrieval routing for memx plugin",
        contentHash: "probe-hash-1",
        dedupStatus: "active",
        mergeCount: 0,
        sourceRef: "user:probe-turn-1",
        createdAt: "2026-03-18T00:00:00.000Z",
        updatedAt: "2026-03-18T00:00:00.000Z",
      },
      {
        chunkId: "probe-chunk-2",
        agentId: "probe",
        scope: "agent:probe",
        sessionKey: "probe",
        turnId: "probe-turn-2",
        seq: 0,
        role: "user",
        chunkKind: "message",
        content: "The blocker is graph expansion and the next action is to add replay tests.",
        summary: "graph expansion blocker and replay tests next",
        contentHash: "probe-hash-2",
        dedupStatus: "active",
        mergeCount: 0,
        sourceRef: "user:probe-turn-2",
        createdAt: "2026-03-18T00:05:00.000Z",
        updatedAt: "2026-03-18T00:05:00.000Z",
      },
    ]);
    const topicDecision = await this.judgeNewTopic(
      "[User] I am working on retrieval routing for memx.\n[User] The blocker is graph expansion.",
      "Now let's switch to provider onboarding.",
    );
    const dedup = await this.judgeDedup(
      "retrieval routing for memx plugin",
      "I am working on retrieval routing for memx plugin",
      [
        {
          index: 1,
          summary: "retrieval routing for memx plugin",
          text: "same topic and same project",
        },
        { index: 2, summary: "provider onboarding flow", text: "different topic" },
      ],
    );
    const relevant = await this.filterRelevant("what am I doing and what is blocked", [
      { index: 1, summary: "workflow.current_task: retrieval routing", role: "state" },
      { index: 2, summary: "workflow.blocker: graph expansion", role: "state" },
      { index: 3, summary: "user prefers bilingual responses", role: "fact" },
    ]);
    const recallPlan = await this.planRecall("What was I doing and what is blocked?");
    const model = this.getResolvedJudgeModel();

    return {
      enabled: this.isEnabled(),
      resolvedConfigPath: this.getResolvedJudgeConfigPath(),
      provider: model?.provider ?? null,
      model: model?.model ?? null,
      traces: this.getTrace(),
      outputs: {
        chunkSummary,
        taskTitle: task.title,
        taskSummary: task.summary,
        topicDecision,
        dedupAction: dedup?.action ?? null,
        relevant: relevant?.relevant ?? [],
        recallPlan: recallPlan.focusedQuery,
      },
    };
  }

  private async callJson<T>(
    label: string,
    prompt: { system: string; user: string },
    failureMode: "degraded" = "degraded",
    options: ReasonerCallAuditOptions = {},
  ): Promise<T | null> {
    if (!this.judgeModel) {
      this.recordTrace({
        label,
        mode: "disabled",
        provenance: "deterministic",
        detail: "LLM classifier is disabled or no judge model could be resolved.",
        stage: options.stage,
      });
      recordMemoryLlmBudgetCall(options.audit, {
        label,
        stage: options.stage ?? "write_hot_path",
        provenance: "deterministic",
        mode: "disabled",
        detail: "LLM classifier unavailable; caller must use a degraded non-semantic outcome.",
      });
      return null;
    }
    const promptChars = prompt.system.length + prompt.user.length;
    const estimatedPromptTokens = estimateTokenCount(`${prompt.system}\n${prompt.user}`);
    try {
      const _tLlm0 = performance.now();
      const raw = await callJudgeModel(this.judgeModel, prompt, {
        maxTokens: options.maxTokens,
        jsonMode: options.jsonMode,
        temperature: options.temperature,
        signal: options.signal,
      });
      const _tLlm1 = performance.now();
      const elapsedMs = Math.round(_tLlm1 - _tLlm0);
      const responseChars = raw.length;
      const estimatedCompletionTokens = estimateTokenCount(raw);
      this.logger.info?.(
        `memx: TIMING llm label=${label} stage=${options.stage ?? "unspecified"} elapsed=${elapsedMs.toFixed(0)}ms provider=${this.judgeModel.provider} model=${this.judgeModel.model}`,
      );
      const parsed = parseJsonResponse<T>(raw);
      if (parsed) {
        this.recordTrace({
          label,
          mode: "llm",
          provenance: "llm",
          detail: "LLM request succeeded and returned parseable JSON.",
          stage: options.stage,
          provider: this.judgeModel.provider,
          model: this.judgeModel.model,
        });
        recordMemoryLlmBudgetCall(options.audit, {
          label,
          stage: options.stage ?? "write_hot_path",
          provenance: "llm",
          mode: "llm",
          provider: this.judgeModel.provider,
          model: this.judgeModel.model,
          detail: "LLM request succeeded.",
          promptChars,
          responseChars,
          estimatedPromptTokens,
          estimatedCompletionTokens,
          estimatedTotalTokens: estimatedPromptTokens + estimatedCompletionTokens,
          elapsedMs,
        });
        return parsed;
      }
      this.recordTrace({
        label,
        mode: failureMode,
        provenance: "hybrid",
        detail: `LLM request returned unparsable output; the caller degraded conservatively instead of rebuilding semantics locally (${summarizeParseFailure(raw)}).`,
        stage: options.stage,
        provider: this.judgeModel.provider,
        model: this.judgeModel.model,
      });
      recordMemoryLlmBudgetCall(options.audit, {
        label,
        stage: options.stage ?? "write_hot_path",
        provenance: "hybrid",
        mode: failureMode,
        provider: this.judgeModel.provider,
        model: this.judgeModel.model,
        detail: `LLM output was unparsable; caller degraded conservatively (${summarizeParseFailure(raw)}).`,
        promptChars,
        responseChars,
        estimatedPromptTokens,
        estimatedCompletionTokens,
        estimatedTotalTokens: estimatedPromptTokens + estimatedCompletionTokens,
        elapsedMs,
      });
      this.logger.info?.(
        `memx: PROBE llm-fallback label=${label} stage=${options.stage ?? "unspecified"} reason=unparsable mode=${failureMode} ${summarizeParseFailure(raw)}`,
      );
      return null;
    } catch (error) {
      const elapsedMs = undefined;
      if (options.signal?.aborted || isAbortLike(error)) {
        this.logger.info?.(
          `memx: PROBE llm-fallback label=${label} stage=${options.stage ?? "unspecified"} reason=aborted mode=${failureMode}`,
        );
        this.recordTrace({
          label,
          mode: failureMode,
          provenance: "hybrid",
          detail:
            "LLM request was canceled by the caller budget; the caller degraded conservatively.",
          stage: options.stage,
          provider: this.judgeModel.provider,
          model: this.judgeModel.model,
        });
        recordMemoryLlmBudgetCall(options.audit, {
          label,
          stage: options.stage ?? "write_hot_path",
          provenance: "hybrid",
          mode: failureMode,
          provider: this.judgeModel.provider,
          model: this.judgeModel.model,
          detail: "LLM request canceled by caller budget.",
          promptChars,
          responseChars: 0,
          estimatedPromptTokens,
          estimatedCompletionTokens: 0,
          estimatedTotalTokens: estimatedPromptTokens,
          elapsedMs,
        });
        return null;
      }
      const key = `${label}:${this.judgeModel.provider}:${this.judgeModel.model}`;
      if (!this.warned.has(key)) {
        this.warned.add(key);
        this.logger.warn(
          `memx: ${label} degraded conservatively after LLM failure (${String(error)})`,
        );
      }
      this.logger.info?.(
        `memx: PROBE llm-fallback label=${label} stage=${options.stage ?? "unspecified"} reason=error mode=${failureMode} err=${String(error).slice(0, 120)}`,
      );
      this.recordTrace({
        label,
        mode: failureMode,
        provenance: "hybrid",
        detail: `LLM request failed; the caller degraded conservatively instead of rebuilding semantics locally: ${String(error)}`,
        stage: options.stage,
        provider: this.judgeModel.provider,
        model: this.judgeModel.model,
      });
      recordMemoryLlmBudgetCall(options.audit, {
        label,
        stage: options.stage ?? "write_hot_path",
        provenance: "hybrid",
        mode: failureMode,
        provider: this.judgeModel.provider,
        model: this.judgeModel.model,
        detail: `LLM request failed: ${String(error)}`,
        promptChars,
        responseChars: 0,
        estimatedPromptTokens,
        estimatedCompletionTokens: 0,
        estimatedTotalTokens: estimatedPromptTokens,
        elapsedMs,
      });
      return null;
    }
  }

  private recordTrace(entry: Omit<ReasonerTraceEntry, "at">): void {
    this.traces.push({
      ...entry,
      at: new Date().toISOString(),
    });
    if (this.traces.length > 50) {
      this.traces = this.traces.slice(-50);
    }
  }

  isEnabled(): boolean {
    return Boolean(this.judgeModel);
  }
}

export function roleLabelForHit(hit: SearchHit): string {
  if (typeof hit.metadata.memxDocType === "string") {
    return hit.metadata.memxDocType;
  }
  if (typeof hit.metadata.role === "string") {
    return hit.metadata.role;
  }
  return hit.backend;
}
