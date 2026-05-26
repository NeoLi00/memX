import type { MemxStoreBundle } from "../runtime.js";
import { containsLikelySecret, looksLikePromptInjection } from "../security/injection.js";
import { containsSensitiveValue, sensitivityScore } from "../security/pii.js";
import {
  clamp01,
  normalizeName,
  normalizeText,
  normalizedTerms,
  objectRecord,
  randomId,
  stableHash,
  truncateText,
} from "../support.js";
import type {
  AbstractionCandidateRecord,
  ClassifiedCandidate,
  ConversationChunk,
  ConversationTask,
  MemoryOperationContext,
  MemoryLlmCallStage,
  MemoryCandidateStructuredHints,
  SynthesizedTaskEvent,
  TurnCaptureMessage,
  TurnSemanticFrame,
} from "../types.js";
import type { MemxLogger } from "../types.js";
import { eligibleForLlmRefinement } from "./abstractionRefinement.js";
import { sanitizeTaskMetadata } from "./authority.js";
import { classifyAction } from "./classify.js";
import {
  CHUNK_VECTOR_CONFIDENCE,
  SCHEDULER_DEDUP_PROBE_FLOOR,
  SCHEDULER_DEDUP_PROBE_SCALE,
  SCHEDULER_RECENT_ACTIVE_LIMIT,
  SCHEDULER_RECENT_TASKS_LIMIT,
  SCHEDULER_SUMMARY_SCORE_BOOST,
  SCHEDULER_TRUNCATE_VECTOR_TEXT,
  TASK_VECTOR_CONFIDENCE,
} from "./constants.js";
import { buildCandidate } from "./extract.js";
import {
  inferWriteLlmStage,
  recordMemoryLlmBudgetCall,
  snapshotMemoryLlmBudgetAudit,
} from "./llmBudgetAudit.js";
import { computeConfidence } from "./normalize.js";
import {
  buildOutcomeHypothesisCandidate,
  isAuthoritativeOutcomeResolutionMetadata,
} from "./outcomeHypotheses.js";
import { evaluatePolicy } from "./policy.js";
import {
  isProjectProfileStateKey,
  projectCodeFromStateKey,
  projectIdentityKey,
  projectNamesMatch,
  resolveProjectReference,
} from "./projectIdentity.js";
import type { OutcomePromotionDecision } from "./reasoner.js";
import { semanticTextSimilarity } from "./semantic/textSimilarity.js";
import { isQuestionLike } from "./semantics.js";
import { emitAssistantOutcomeLearningSignals, emitOutcomeFeedbackSignal } from "./signalLedger.js";
import { buildSourceSegmentsForChunk, buildSourceSegmentVectorDocs } from "./sourceSegments.js";
import {
  assessAssistantChunk,
  assistantVectorSummary,
  assistantVectorText,
} from "./sourceWeighting.js";
import { decideTaskAssignment } from "./taskJudge.js";
import { resolveWorkingTaskSummary, semanticTaskSummaryText } from "./taskSummary.js";
import { compileTurnSemantics, frameHintsForSourceRef } from "./turnSemanticCompiler.js";
import { buildVectorDocMetadata } from "./vectorDocMetadata.js";
import { policyAuditSourceMetadata, writeCandidate } from "./write.js";

function buildOutcomeKey(
  taskId: string,
  proposal: Omit<SynthesizedTaskEvent, "evidenceChunkIds" | "outcomeKey">,
): string {
  return stableHash([taskId, proposal.eventType, normalizeText(proposal.summary)]);
}

function shouldSuppressMessageFromSemanticMemory(
  ctx: MemoryOperationContext,
  message: TurnCaptureMessage,
): boolean {
  const content = message.content.trim();
  if (!content) {
    return false;
  }
  const secretLike = containsLikelySecret(content);
  const promptInjection = looksLikePromptInjection(content) && !secretLike;
  return (
    promptInjection ||
    secretLike ||
    containsSensitiveValue(content) ||
    sensitivityScore(content) > ctx.config.maxSensitivityAllowed
  );
}

function semanticDraftHasWriteSignals(hints: Partial<MemoryCandidateStructuredHints> | undefined): boolean {
  const draft = hints?.semanticDraft;
  return Boolean(
    (draft?.assertionDrafts.length ?? 0) > 0 ||
      (draft?.correctionDrafts.length ?? 0) > 0 ||
      (draft?.relationDrafts?.length ?? 0) > 0 ||
      (draft?.resourceAssertions?.length ?? 0) > 0 ||
      (draft?.adviceSignals?.length ?? 0) > 0 ||
      (hints?.resourceAssertions?.length ?? 0) > 0 ||
      (hints?.adviceSignals?.length ?? 0) > 0,
  );
}

function semanticOnlyCandidateText(
  hints: Partial<MemoryCandidateStructuredHints> | undefined,
): string | undefined {
  if (!semanticDraftHasWriteSignals(hints)) {
    return undefined;
  }
  const draft = hints?.semanticDraft;
  const lines: string[] = [];
  for (const assertion of draft?.assertionDrafts ?? []) {
    const entities = (assertion.entityHints ?? []).map((entry) => entry.name).filter(Boolean);
    const slots = assertion.slotHints?.filter(Boolean) ?? [];
    const value = assertion.valueHint?.trim();
    const support = assertion.supportSpans?.find((span) => span.text.trim())?.text.trim();
    lines.push(
      [
        assertion.familyHint,
        assertion.timeframeHint,
        entities.join(" / "),
        slots.join(" / "),
        value,
        support,
      ]
        .filter(Boolean)
        .join(" | "),
    );
  }
  for (const correction of draft?.correctionDrafts ?? []) {
    const entry = correction.correction;
    lines.push(
      [
        "correction",
        entry.targetKind,
        entry.canonicalKey,
        entry.predicate,
        entry.priorValue ? `from ${entry.priorValue}` : undefined,
        entry.nextValue ? `to ${entry.nextValue}` : undefined,
        entry.reason,
      ]
        .filter(Boolean)
        .join(" | "),
    );
  }
  for (const relation of draft?.relationDrafts ?? []) {
    const entry = relation.relation;
    if (entry.polarity === "negated") {
      continue;
    }
    lines.push(
      [
        "relation",
        entry.subject,
        entry.predicate,
        entry.relationSlot,
        entry.object,
        entry.reason,
      ]
        .filter(Boolean)
        .join(" | "),
    );
  }
  for (const resource of [
    ...(draft?.resourceAssertions ?? []),
    ...(hints?.resourceAssertions ?? []),
  ]) {
    lines.push(
      [
        "resource",
        resource.owner,
        resource.ownershipStatus,
        resource.resource,
        resource.resourceType,
        resource.supportText,
      ]
        .filter(Boolean)
        .join(" | "),
    );
  }
  for (const advice of [...(draft?.adviceSignals ?? []), ...(hints?.adviceSignals ?? [])]) {
    lines.push(
      [
        "advice",
        advice.problemContext,
        advice.userResources?.join(" / "),
        advice.assistantRecommendation,
        advice.supportText,
      ]
        .filter(Boolean)
        .join(" | "),
    );
  }
  const text = lines.filter((line) => line.trim()).join("\n").trim();
  return text ? truncateText(text, 1200) : undefined;
}

function mapEvidenceChunkIds(chunks: ConversationChunk[], indexes: number[]): string[] {
  const unique = new Set<string>();
  for (const index of indexes) {
    const chunk = chunks[index - 1];
    if (chunk) {
      unique.add(chunk.chunkId);
    }
  }
  return [...unique];
}

async function refineOutcomeHypothesisCandidate(
  candidate: AbstractionCandidateRecord,
  stage: MemoryLlmCallStage,
  audit: MemoryOperationContext["llmBudgetAudit"],
): Promise<AbstractionCandidateRecord> {
  if (!eligibleForLlmRefinement(candidate)) {
    recordMemoryLlmBudgetCall(audit, {
      label: "abstraction-refinement",
      stage,
      provenance: "deterministic",
      mode: "deterministic",
      detail: "abstraction refinement skipped because the candidate is not refinement-eligible",
    });
    return candidate;
  }
  recordMemoryLlmBudgetCall(audit, {
    label: "abstraction-refinement",
    stage,
    provenance: "deterministic",
    mode: "deferred",
    detail:
      "abstraction refinement is deferred to maintenance and does not block the write hot path",
  });
  return candidate;
}

function buildPromotedOutcome(
  task: ConversationTask,
  chunks: ConversationChunk[],
  proposal: Omit<SynthesizedTaskEvent, "evidenceChunkIds" | "outcomeKey"> & {
    evidenceChunkIndexes?: number[];
  },
): SynthesizedTaskEvent {
  const evidenceChunkIds = mapEvidenceChunkIds(chunks, proposal.evidenceChunkIndexes ?? []);
  const outcomeKey = buildOutcomeKey(task.taskId, proposal);
  return {
    ...proposal,
    evidenceChunkIds,
    outcomeKey,
  };
}

function deterministicOutcomePromotionDecision(
  outcome: SynthesizedTaskEvent,
  evidenceChunks: ConversationChunk[],
): OutcomePromotionDecision {
  const assistantCount = evidenceChunks.filter((chunk) => chunk.role === "assistant").length;
  const userCount = evidenceChunks.filter((chunk) => chunk.role === "user").length;
  const toolCount = evidenceChunks.filter((chunk) => chunk.role === "tool").length;
  const hasUserEvidence = userCount > 0;
  const hasToolEvidence = toolCount > 0;
  const hasNonAssistantGrounding = hasUserEvidence || hasToolEvidence;
  const strongPhase = outcome.phase === "validated" || outcome.phase === "resolved";
  const promotionScore = clamp01(
    outcome.confidence * 0.22 +
      outcome.verificationScore * 0.3 +
      outcome.closureScore * 0.2 +
      (hasToolEvidence ? 0.16 : hasUserEvidence ? 0.1 : 0) +
      (strongPhase ? 0.1 : 0.02) -
      outcome.contradictionRisk * 0.32,
  );
  const shouldPromote =
    hasNonAssistantGrounding &&
    strongPhase &&
    outcome.verificationScore >= 0.72 &&
    outcome.closureScore >= 0.68 &&
    outcome.contradictionRisk <= 0.28 &&
    promotionScore >= 0.72;
  return {
    shouldPromote,
    promotionScore,
    reason: shouldPromote
      ? "deterministic promotion gate: grounded, validated, and low-contradiction outcome"
      : hasNonAssistantGrounding
        ? "deterministic promotion gate: keep the outcome provisional until it is more strongly validated"
        : "deterministic promotion gate: no non-assistant grounding is available",
  };
}

function mergeTaskMetadata(
  base: Record<string, unknown>,
  summaryMetadata: Record<string, unknown>,
  proposal: SynthesizedTaskEvent | undefined,
  observedAt: string,
  emitted: boolean,
  alreadyEmitted: boolean,
  outcomeHypothesis?: AbstractionCandidateRecord,
): Record<string, unknown> {
  const next: Record<string, unknown> = {
    ...base,
    ...summaryMetadata,
  };
  if (!proposal) {
    delete next.candidateResolution;
    delete next.candidateOutcomeKey;
    delete next.candidateResolutionPhase;
    delete next.candidateResolutionEvidenceChunkIds;
    delete next.candidateResolutionPromotionScore;
    delete next.candidateOutcomeHypothesisId;
    delete next.candidateOutcomeHypothesisStage;
    delete next.candidateOutcomeHypothesisUpdatedAt;
    delete next.candidateOutcomeJudgeShouldPromote;
    delete next.candidateOutcomeJudgeReason;
    delete next.candidateOutcomeHasUserEvidence;
    delete next.candidateOutcomeHasToolEvidence;
    delete next.candidateOutcomeHasNonAssistantGrounding;
    return next;
  }
  next.candidateOutcomeKey = proposal.outcomeKey;
  if (outcomeHypothesis) {
    next.candidateOutcomeHypothesisId = outcomeHypothesis.candidateId;
    next.candidateOutcomeHypothesisStage = outcomeHypothesis.stage;
    next.candidateOutcomeHypothesisUpdatedAt = outcomeHypothesis.updatedAt;
    next.candidateOutcomeJudgeShouldPromote = Boolean(
      outcomeHypothesis.metadataJson.judgeShouldPromote,
    );
    next.candidateOutcomeHasUserEvidence = Boolean(outcomeHypothesis.metadataJson.hasUserEvidence);
    next.candidateOutcomeHasToolEvidence = Boolean(outcomeHypothesis.metadataJson.hasToolEvidence);
    next.candidateOutcomeHasNonAssistantGrounding = Boolean(
      outcomeHypothesis.metadataJson.hasNonAssistantGrounding,
    );
    if (typeof outcomeHypothesis.metadataJson.judgeReason === "string") {
      next.candidateOutcomeJudgeReason = outcomeHypothesis.metadataJson.judgeReason;
    }
  }
  if (isAuthoritativeOutcomeResolutionMetadata(next)) {
    next.candidateResolution = proposal.summary;
    next.candidateResolutionPhase = proposal.phase;
    next.candidateResolutionEvidenceChunkIds = proposal.evidenceChunkIds;
    next.candidateResolutionPromotionScore = proposal.promotionScore;
  } else {
    delete next.candidateResolution;
    delete next.candidateResolutionPhase;
    delete next.candidateResolutionEvidenceChunkIds;
    delete next.candidateResolutionPromotionScore;
  }
  if (!emitted && !alreadyEmitted) {
    delete next.lastEmittedOutcomeKey;
    delete next.lastEmittedOutcomeAt;
    delete next.lastPromotedOutcomeHypothesisId;
  }
  if (emitted) {
    next.lastEmittedOutcomeKey = proposal.outcomeKey;
    next.lastEmittedOutcomeAt = observedAt;
    if (outcomeHypothesis) {
      next.lastPromotedOutcomeHypothesisId = outcomeHypothesis.candidateId;
    }
  } else if (alreadyEmitted) {
    next.lastEmittedOutcomeKey = proposal.outcomeKey;
  }
  return next;
}

function buildChunkVectorDoc(chunk: ConversationChunk, taskChunks?: ConversationChunk[]) {
  const summary = assistantVectorSummary(chunk, taskChunks);
  const assistantAssessment =
    chunk.role === "assistant" && taskChunks ? assessAssistantChunk(chunk, taskChunks) : undefined;
  return {
    docId: `event:chunk:${chunk.chunkId}`,
    docKind: "event" as const,
    sourceId: `chunk:${chunk.chunkId}`,
    scope: chunk.scope,
    agentId: chunk.agentId,
    text: `${chunk.role}: ${summary}\n${assistantVectorText(
      {
        ...chunk,
        content: truncateText(chunk.content, SCHEDULER_TRUNCATE_VECTOR_TEXT),
      },
      taskChunks,
    )}`.trim(),
    metadataJson: buildVectorDocMetadata({
      docType: "chunk",
      confidence: CHUNK_VECTOR_CONFIDENCE,
      observedAt: chunk.createdAt,
      lineage: {
        sourceKind: "chunk",
        sourceId: chunk.chunkId,
        sourceRef: chunk.sourceRef,
      },
      extra: {
        chunkId: chunk.chunkId,
        scope: chunk.scope,
        role: chunk.role,
        sessionKey: chunk.sessionKey,
        taskId: chunk.taskId,
        dedupStatus: chunk.dedupStatus,
        ...(assistantAssessment
          ? {
              assistantWeight: Number(assistantAssessment.weight.toFixed(3)),
              assistantGrounding: Number(assistantAssessment.grounding.toFixed(3)),
              assistantComplexity: Number(assistantAssessment.complexity.toFixed(3)),
              assistantSummaryOnly: assistantAssessment.useSummaryOnly,
              semanticRole: assistantAssessment.semanticRole,
              memoryClass: assistantAssessment.memoryClass,
              recallVisibility: assistantAssessment.recallVisibility,
            }
          : {}),
      },
    }),
    createdAt: chunk.createdAt,
    updatedAt: chunk.updatedAt,
  };
}

function taskSupportRefs(chunks: ConversationChunk[] | undefined): string[] {
  if (!chunks || chunks.length === 0) {
    return [];
  }
  return [
    ...new Set(
      chunks
        .map((chunk) => chunk.sourceRef)
        .filter((sourceRef): sourceRef is string => Boolean(sourceRef?.trim())),
    ),
  ];
}

function taskVectorDocId(task: ConversationTask): string {
  return `state:task:${task.taskId}`;
}

function recallableTaskTitle(title: string): string | undefined {
  const trimmed = title.trim();
  if (!trimmed) {
    return undefined;
  }
  return normalizeText(trimmed) === "active task" ? undefined : trimmed;
}

function buildTaskVectorDoc(task: ConversationTask, chunks?: ConversationChunk[]) {
  const metadata = sanitizeTaskMetadata(task.metadataJson);
  const rawMetadata = objectRecord(task.metadataJson);
  const semanticSummary = semanticTaskSummaryText(task);
  const supportRefs = taskSupportRefs(chunks);
  const taskText = [
    recallableTaskTitle(task.title),
    typeof rawMetadata?.candidateResolution === "string" && rawMetadata.candidateResolution.trim()
      ? rawMetadata.candidateResolution.trim()
      : undefined,
    metadata.currentTask,
    metadata.project,
    metadata.nextAction,
    metadata.blocker,
    semanticSummary,
  ]
    .filter((value): value is string => Boolean(value))
    .join("\n")
    .trim();
  if (!taskText) {
    return null;
  }
  return {
    docId: taskVectorDocId(task),
    docKind: "state" as const,
    sourceId: `task:${task.taskId}`,
    scope: task.scope,
    agentId: task.agentId,
    text: taskText,
    metadataJson: buildVectorDocMetadata({
      docType: "task",
      confidence: TASK_VECTOR_CONFIDENCE,
      observedAt: task.updatedAt,
      lineage: {
        sourceKind: "task",
        sourceId: task.taskId,
        sourceRef: `task:${task.taskId}`,
      },
      extra: {
        taskId: task.taskId,
        scope: task.scope,
        sessionKey: task.sessionKey,
        status: task.status,
        sourceRef: `task:${task.taskId}`,
        supportRefs,
      },
    }),
    createdAt: task.startedAt,
    updatedAt: task.updatedAt,
  };
}

function upsertTaskVectorDoc(
  store: MemxStoreBundle,
  task: ConversationTask,
  chunks?: ConversationChunk[],
): void {
  const doc = buildTaskVectorDoc(task, chunks);
  if (!doc) {
    store.retrievalBackend.deleteDocs([taskVectorDocId(task)]);
    return;
  }
  store.retrievalBackend.upsertDocs([doc]);
}

function resolveCurrentProjectContext(params: {
  store: MemxStoreBundle;
  ctx: MemoryOperationContext;
  scope: string;
  observedAt: string;
  activeTask: ConversationTask;
  taskChunks?: ConversationChunk[];
  message: TurnCaptureMessage;
}): { currentProject?: string; currentProjectProfile?: Record<string, unknown> } {
  const canonicalTaskMetadata = sanitizeTaskMetadata(params.activeTask.metadataJson);
  const metadataProject = canonicalTaskMetadata.project;
  const activeProjectState = params.store.stateRepo.get({
    agentId: params.ctx.agentId,
    scopes: [params.scope],
    key: "project.active_project",
    includeExpired: true,
    now: params.observedAt,
  })[0];
  const stateProject =
    typeof activeProjectState?.valueJson.project === "string" &&
    activeProjectState.valueJson.project.trim()
      ? activeProjectState.valueJson.project.trim()
      : undefined;
  const projectStates = params.store.stateRepo
    .get({
      agentId: params.ctx.agentId,
      scopes: [params.scope],
      includeExpired: true,
      now: params.observedAt,
    })
    .filter((state) => isProjectProfileStateKey(state.key));
  const messageProjectKey = projectIdentityKey(params.message.content);
  const entityMatchedProject = projectStates.find((state) => {
    const projectCode = projectCodeFromStateKey(state.key);
    return projectCode ? messageProjectKey.includes(projectIdentityKey(projectCode)) : false;
  });
  const knownProjects = projectStates
    .map((state) => projectCodeFromStateKey(state.key))
    .filter((entry): entry is string => Boolean(entry));
  const singletonProject =
    !metadataProject && !stateProject && projectStates.length === 1
      ? projectCodeFromStateKey(projectStates[0]!.key)
      : undefined;
  const currentProject = resolveProjectReference(
      metadataProject ??
      stateProject ??
      (entityMatchedProject ? projectCodeFromStateKey(entityMatchedProject.key) : undefined) ??
      singletonProject ??
      "",
    {
      currentProject: metadataProject ?? stateProject,
      knownProjects,
      allowDescriptorAlias: true,
    },
  );
  if (!currentProject) {
    return {};
  }
  const directProfile =
    params.store.stateRepo.get({
      agentId: params.ctx.agentId,
      scopes: [params.scope],
      key: `project.${currentProject}`,
      includeExpired: true,
      now: params.observedAt,
    })[0] ??
    projectStates.find((state) => {
      const projectCode = projectCodeFromStateKey(state.key);
      return projectCode ? projectNamesMatch(projectCode, currentProject) : false;
    });
  const currentProjectProfile = objectRecord(directProfile?.valueJson);
  return currentProjectProfile ? { currentProject, currentProjectProfile } : { currentProject };
}

function turnSourceRefs(messages: TurnCaptureMessage[]): string[] {
  return [
    ...new Set(
      messages
        .map((message) => message.sourceRef)
        .filter((entry): entry is string => Boolean(entry?.trim())),
    ),
  ];
}

function turnSemanticInputHash(messages: TurnCaptureMessage[]): string {
  return stableHash(
    messages.map((message) => [
      message.role,
      message.sourceRef,
      normalizeText(message.content),
    ].join("\u0001")),
  );
}

function semanticFrameDraftCounts(frame: TurnSemanticFrame | undefined): Record<string, unknown> {
  return {
    assertionDrafts: frame?.assertionDrafts?.length ?? 0,
    correctionDrafts: frame?.correctionDrafts?.length ?? 0,
    relationDrafts: frame?.relationDrafts?.length ?? 0,
    resourceAssertions: frame?.resourceAssertions?.length ?? 0,
    adviceSignals: frame?.adviceSignals?.length ?? 0,
    supportSpans: frame?.supportSpans?.length ?? 0,
  };
}

function semanticFrameResolvedByLlm(frame: TurnSemanticFrame | undefined): boolean {
  const provenance = objectRecord(frame?.compilerProvenance);
  return provenance?.source === "llm" && provenance.mode !== "fallback";
}

export class MemxTurnScheduler {
  private chain = Promise.resolve();

  constructor(
    private readonly store: MemxStoreBundle,
    private readonly logger: MemxLogger,
  ) {}

  enqueue(ctx: MemoryOperationContext, messages: TurnCaptureMessage[]): Promise<void> {
    if (messages.length === 0) {
      return this.chain;
    }
    this.chain = this.chain
      .then(() => this.processTurn(ctx, messages))
      .catch((error) => {
        const turnId = messages[0]?.turnId ?? "unknown";
        this.logger.warn(
          `memx: turn scheduler failed for turn ${turnId} (${messages.length} messages): ${String(error)}`,
        );
        // Re-throw so callers can detect the failure; the chain stays resolved
        // for subsequent turns since the .catch itself succeeds.
      });
    return this.chain;
  }

  flush(): Promise<void> {
    return this.chain;
  }

  async stageRecallableTurn(
    ctx: MemoryOperationContext,
    messages: TurnCaptureMessage[],
  ): Promise<boolean> {
    const safeMessages = this.safeMessagesForTurn(ctx, messages);
    const hasUserMessage = messages.some((message) => message.role === "user");
    const hasSafeUserMessage = safeMessages.some((message) => message.role === "user");
    if (safeMessages.length === 0 || (hasUserMessage && !hasSafeUserMessage)) {
      return false;
    }
    const scope = safeMessages[0]?.scope ?? ctx.scopes[0] ?? `agent:${ctx.agentId}`;
    const activeTask = this.getOrCreateActiveTask(ctx, safeMessages, scope);
    await this.persistRecallableChunks(ctx, activeTask, safeMessages);
    return true;
  }

  private safeMessagesForTurn(
    ctx: MemoryOperationContext,
    messages: TurnCaptureMessage[],
  ): TurnCaptureMessage[] {
    return messages.filter((message) => !shouldSuppressMessageFromSemanticMemory(ctx, message));
  }

  private getOrCreateActiveTask(
    ctx: MemoryOperationContext,
    safeMessages: TurnCaptureMessage[],
    scope: string,
  ): ConversationTask {
    let activeTask = this.store.taskRepo.getActive({
      agentId: ctx.agentId,
      scope,
      sessionKey: safeMessages[0]!.sessionKey,
    });
    if (!activeTask) {
      activeTask = {
        taskId: randomId("task"),
        agentId: ctx.agentId,
        scope,
        sessionKey: safeMessages[0]!.sessionKey,
        title: "Active task",
        summary: "",
        status: "active",
        startedAt: safeMessages[0]!.observedAt,
        updatedAt: safeMessages[0]!.observedAt,
        metadataJson: {},
      };
      this.store.taskRepo.create(activeTask);
    }
    return activeTask;
  }

  private async maybePromoteTaskOutcome(
    ctx: MemoryOperationContext,
    task: ConversationTask,
    taskChunks: ConversationChunk[],
    observedAt: string,
    stage: "write_hot_path" | "post_answer_writeback",
    proposal:
      | (Omit<SynthesizedTaskEvent, "evidenceChunkIds" | "outcomeKey"> & {
          evidenceChunkIndexes?: number[];
        })
      | undefined,
  ): Promise<{
    promotedOutcome?: SynthesizedTaskEvent;
    emitted: boolean;
    alreadyEmitted: boolean;
    outcomeHypothesis?: AbstractionCandidateRecord;
  }> {
    const promotedOutcome = proposal ? buildPromotedOutcome(task, taskChunks, proposal) : undefined;
    let emitted = false;
    const lastEmittedOutcomeKey =
      typeof task.metadataJson.lastEmittedOutcomeKey === "string"
        ? task.metadataJson.lastEmittedOutcomeKey
        : undefined;
    if (promotedOutcome && lastEmittedOutcomeKey === promotedOutcome.outcomeKey) {
      return { promotedOutcome, emitted: false, alreadyEmitted: true };
    }
    if (promotedOutcome && lastEmittedOutcomeKey !== promotedOutcome.outcomeKey) {
      const evidenceChunks = taskChunks.filter((chunk) =>
        promotedOutcome.evidenceChunkIds.includes(chunk.chunkId),
      );
      const decision = deterministicOutcomePromotionDecision(promotedOutcome, evidenceChunks);
      recordMemoryLlmBudgetCall(ctx.llmBudgetAudit, {
        label: "outcome-promotion",
        stage,
        provenance: "deterministic",
        mode: "deterministic",
        detail:
          "outcome promotion is resolved by the deterministic gate and does not use an extra hot-path LLM",
      });
      promotedOutcome.promotionScore = decision.promotionScore;
      const outcomeHypothesis = await refineOutcomeHypothesisCandidate(
        buildOutcomeHypothesisCandidate({
          agentId: ctx.agentId,
          task,
          outcome: promotedOutcome,
          observedAt,
          evidenceChunks,
          decision,
        }),
        stage,
        ctx.llmBudgetAudit,
      );
      const sourceEpoch = ctx.readEpoch ?? this.store.client.currentMemoryEpoch(ctx.agentId);
      outcomeHypothesis.derivedFromMinEpoch = outcomeHypothesis.derivedFromMinEpoch ?? sourceEpoch;
      outcomeHypothesis.derivedFromMaxEpoch = outcomeHypothesis.derivedFromMaxEpoch ?? sourceEpoch;
      outcomeHypothesis.materializedEpoch = this.store.client.nextMemoryEpoch(ctx.agentId, ctx.now);
      outcomeHypothesis.derivedFromKind = outcomeHypothesis.derivedFromKind ?? "task_outcome";
      outcomeHypothesis.derivedFromIds = outcomeHypothesis.derivedFromIds ?? [
        ...outcomeHypothesis.supportContentRefs,
        ...outcomeHypothesis.supportBeliefIds,
      ];
      outcomeHypothesis.derivedAtEpoch = outcomeHypothesis.derivedAtEpoch ?? sourceEpoch;
      outcomeHypothesis.derivationPolicyVersion =
        outcomeHypothesis.derivationPolicyVersion ?? "memx-authority-v3";
      this.store.abstractionRepo.upsert(outcomeHypothesis);
      emitAssistantOutcomeLearningSignals(this.store, ctx, {
        task,
        outcome: promotedOutcome,
        evidenceChunks,
        shouldPromote: decision.shouldPromote,
        emitted: false,
        reason: decision.reason,
      });
      emitOutcomeFeedbackSignal(this.store, ctx, {
        task,
        outcome: promotedOutcome,
        emitted: false,
      });
      return { promotedOutcome, emitted: false, alreadyEmitted: false, outcomeHypothesis };
    }
    return { promotedOutcome, emitted, alreadyEmitted: false };
  }

  private async closeTask(
    ctx: MemoryOperationContext,
    task: ConversationTask,
    observedAt: string,
    status: ConversationTask["status"],
    stage: "write_hot_path" | "post_answer_writeback" = "write_hot_path",
  ): Promise<ConversationTask> {
    const closedChunks = this.store.chunkRepo.listByTask(task.taskId);
    const workingSummary = resolveWorkingTaskSummary({
      task,
      chunks: closedChunks,
      observedAt,
    });
    const outcomeSummary = await this.store.reasoner.summarizeTask(closedChunks, {
      stage,
      audit: ctx.llmBudgetAudit,
    });
    const outcome = await this.maybePromoteTaskOutcome(
      ctx,
      task,
      closedChunks,
      observedAt,
      stage,
      outcomeSummary.synthesizedEvent,
    );
    const closedTask: ConversationTask = {
      ...task,
      title: workingSummary.title,
      summary: workingSummary.summary,
      metadataJson: mergeTaskMetadata(
        task.metadataJson,
        workingSummary.metadataJson,
        outcome.promotedOutcome,
        observedAt,
        outcome.emitted,
        outcome.alreadyEmitted,
        outcome.outcomeHypothesis,
      ),
      status,
      endedAt: observedAt,
      updatedAt: observedAt,
    };
    this.store.taskRepo.update(task.taskId, closedTask);
    upsertTaskVectorDoc(this.store, closedTask, closedChunks);
    return closedTask;
  }

  private reopenTask(task: ConversationTask, observedAt: string): ConversationTask {
    const reopenedTask: ConversationTask = {
      ...task,
      status: "active",
      endedAt: undefined,
      updatedAt: observedAt,
    };
    this.store.taskRepo.update(task.taskId, reopenedTask);
    upsertTaskVectorDoc(this.store, reopenedTask, this.store.chunkRepo.listByTask(reopenedTask.taskId));
    return reopenedTask;
  }

  private async persistRecallableChunks(
    ctx: MemoryOperationContext,
    activeTask: ConversationTask,
    messages: TurnCaptureMessage[],
  ): Promise<void> {
    for (const [index, message] of messages.entries()) {
      const contentHash = stableHash([
        ctx.agentId,
        message.scope,
        message.role,
        normalizeText(message.content),
      ]);
      const exact = this.store.chunkRepo.findActiveByHash({
        agentId: ctx.agentId,
        scope: message.scope,
        role: message.role,
        contentHash,
      });
      if (exact) {
        continue;
      }
      const summary = await this.store.reasoner.summarizeChunk(message.content, message.role, {
        stage: inferWriteLlmStage(message.role),
        audit: ctx.llmBudgetAudit,
        allowLlm: false,
      });
      const chunk: ConversationChunk = {
        chunkId: randomId("chunk"),
        agentId: ctx.agentId,
        scope: message.scope,
        sessionKey: message.sessionKey,
        turnId: message.turnId,
        seq: index,
        role: message.role,
        toolName: message.toolName,
        chunkKind: message.role === "tool" ? "tool_result" : "message",
        content: message.content,
        summary,
        contentHash,
        taskId: activeTask.taskId,
        dedupStatus: "active",
        mergeCount: 0,
        sourceRef: message.sourceRef,
        createdAt: message.observedAt,
        updatedAt: message.observedAt,
      };
      this.store.chunkRepo.insert(chunk);
      const sourceSegments = buildSourceSegmentsForChunk(chunk);
      this.store.sourceSegmentRepo.insertMany(sourceSegments);
      const taskChunks = this.store.chunkRepo.listByTask(activeTask.taskId);
      this.store.retrievalBackend.upsertDocs([
        buildChunkVectorDoc(chunk, taskChunks),
        ...buildSourceSegmentVectorDocs({ chunk, segments: sourceSegments }),
      ]);
    }
  }

  private async processTurn(
    ctx: MemoryOperationContext,
    messages: TurnCaptureMessage[],
  ): Promise<void> {
    const safeMessages = this.safeMessagesForTurn(ctx, messages);
    const hasUserMessage = messages.some((message) => message.role === "user");
    const hasSafeUserMessage = safeMessages.some((message) => message.role === "user");
    if (safeMessages.length === 0 || (hasUserMessage && !hasSafeUserMessage)) {
      return;
    }

    const scope = safeMessages[0]?.scope ?? ctx.scopes[0] ?? `agent:${ctx.agentId}`;
    let activeTask = this.getOrCreateActiveTask(ctx, safeMessages, scope);
    await this.persistRecallableChunks(ctx, activeTask, safeMessages);
    const activeChunks = activeTask ? this.store.chunkRepo.listByTask(activeTask.taskId) : [];
    const recentTasks = this.store.taskRepo.listRecent({
      agentId: ctx.agentId,
      scopes: [scope],
      sessionKey: safeMessages[0]!.sessionKey,
      limit: SCHEDULER_RECENT_TASKS_LIMIT,
      includeSkipped: false,
    });
    const recentChunksByTask = Object.fromEntries(
      recentTasks.map((task) => [task.taskId, this.store.chunkRepo.listByTask(task.taskId)]),
    );
    const semanticJobSourceRefs = turnSourceRefs(safeMessages);
    const semanticJobInputHash = turnSemanticInputHash(safeMessages);
    const semanticJobTurnId = safeMessages[0]!.turnId;
    const semanticJobScope = safeMessages[0]!.scope ?? scope;
    if (ctx.config.advanced.enableTurnSemanticCompiler) {
      this.store.auditRepo.recordSemanticWriteAttemptStart({
        agentId: ctx.agentId,
        sessionKey: safeMessages[0]!.sessionKey,
        scope: semanticJobScope,
        turnId: semanticJobTurnId,
        sourceRefs: semanticJobSourceRefs,
        inputHash: semanticJobInputHash,
        startedAt: ctx.now,
      });
    }
    let turnSemanticFrame: TurnSemanticFrame | undefined;
    try {
      turnSemanticFrame = await compileTurnSemantics({
        messages: safeMessages,
        ctx,
        activeTask,
        activeChunks,
        recentTasks,
        recentChunksByTask,
        reasoner: this.store.reasoner,
      });
    } catch (error) {
      this.store.auditRepo.finishSemanticWriteAttempt({
        agentId: ctx.agentId,
        sessionKey: safeMessages[0]!.sessionKey,
        turnId: semanticJobTurnId,
        status: "retrying",
        error: `turn semantic compiler failed: ${error instanceof Error ? error.message : String(error)}`,
        resultJson: {
          stage: "write_hot_path",
          sourceRefs: semanticJobSourceRefs,
          llmBudget: snapshotMemoryLlmBudgetAudit(ctx.llmBudgetAudit),
        },
        completedAt: ctx.now,
      });
      this.logger.warn(
        `memx: turn semantic compiler failed for turn ${semanticJobTurnId}: ${String(error)}`,
      );
    }
    if (ctx.config.advanced.enableTurnSemanticCompiler && turnSemanticFrame) {
      const resolvedByLlm = semanticFrameResolvedByLlm(turnSemanticFrame);
      this.store.auditRepo.finishSemanticWriteAttempt({
        agentId: ctx.agentId,
        sessionKey: safeMessages[0]!.sessionKey,
        turnId: semanticJobTurnId,
        status: resolvedByLlm ? "succeeded" : "retrying",
        error: resolvedByLlm
          ? undefined
          : "turn semantic compiler returned no recognized LLM semantic frame",
        resultJson: {
          stage: "write_hot_path",
          sourceRefs: semanticJobSourceRefs,
          compilerProvenance: turnSemanticFrame.compilerProvenance,
          ...semanticFrameDraftCounts(turnSemanticFrame),
          llmBudget: snapshotMemoryLlmBudgetAudit(ctx.llmBudgetAudit),
        },
        completedAt: ctx.now,
      });
    } else if (ctx.config.advanced.enableTurnSemanticCompiler) {
      this.store.auditRepo.finishSemanticWriteAttempt({
        agentId: ctx.agentId,
        sessionKey: safeMessages[0]!.sessionKey,
        turnId: semanticJobTurnId,
        status: "retrying",
        error: "turn semantic compiler returned no recognized LLM semantic frame",
        resultJson: {
          stage: "write_hot_path",
          sourceRefs: semanticJobSourceRefs,
          llmBudget: snapshotMemoryLlmBudgetAudit(ctx.llmBudgetAudit),
        },
        completedAt: ctx.now,
      });
    }
    const taskProposal: TurnSemanticFrame["taskProposal"] | undefined =
      turnSemanticFrame?.taskProposal;
    const assignment = await decideTaskAssignment({
      activeTask,
      activeChunks,
      recentTasks,
      recentChunksByTask,
      newMessages: safeMessages,
      ctx,
      taskProposal,
    });

    if (assignment.decision === "new") {
      if (activeTask) {
        const closedChunks = this.store.chunkRepo.listByTask(activeTask.taskId);
        activeTask = await this.closeTask(
          ctx,
          activeTask,
          safeMessages[0]!.observedAt,
          closedChunks.length >= 2 ? "completed" : "skipped",
          "write_hot_path",
        );
      }
      activeTask = {
        taskId: randomId("task"),
        agentId: ctx.agentId,
        scope,
        sessionKey: safeMessages[0]!.sessionKey,
        title: "Active task",
        summary: "",
        status: "active",
        startedAt: safeMessages[0]!.observedAt,
        updatedAt: safeMessages[0]!.observedAt,
        metadataJson: {},
      };
      this.store.taskRepo.create(activeTask);
    } else if (
      assignment.decision === "resume" &&
      assignment.targetTaskId &&
      assignment.targetTaskId !== activeTask?.taskId
    ) {
      if (activeTask) {
        const closedChunks = this.store.chunkRepo.listByTask(activeTask.taskId);
        await this.closeTask(
          ctx,
          activeTask,
          safeMessages[0]!.observedAt,
          closedChunks.length >= 2 ? "completed" : "skipped",
          "write_hot_path",
        );
      }
      const resumedTask = recentTasks.find((task) => task.taskId === assignment.targetTaskId);
      activeTask = resumedTask
        ? this.reopenTask(resumedTask, safeMessages[0]!.observedAt)
        : {
            taskId: randomId("task"),
            agentId: ctx.agentId,
            scope,
            sessionKey: safeMessages[0]!.sessionKey,
            title: "Active task",
            summary: "",
            status: "active",
            startedAt: safeMessages[0]!.observedAt,
            updatedAt: safeMessages[0]!.observedAt,
            metadataJson: {},
          };
      if (!resumedTask) {
        this.store.taskRepo.create(activeTask);
      }
    }
    if (!activeTask) {
      return;
    }

    const createdChunks: ConversationChunk[] = [];

    // Opt 4: process messages in parallel — each message operates on
    // independent content with isolated error handling.
    const results = await Promise.allSettled(
      safeMessages.map((message, index) =>
        this.processMessage(ctx, activeTask, message, index, createdChunks, turnSemanticFrame),
      ),
    );
    for (const [index, result] of results.entries()) {
      if (result.status === "rejected") {
        this.logger.warn(
          `memx: turn scheduler skipped message ${index} for task ${activeTask.taskId}: ${String(result.reason)}`,
        );
      }
    }

    // Task summarization is isolated so already-written chunks are not lost
    // if the summarizer or outcome promotion fails.
    try {
      await this.summarizeAndUpdateTask(ctx, activeTask, safeMessages, turnSemanticFrame);
    } catch (error) {
      this.logger.warn(
        `memx: turn scheduler failed task summarization for ${activeTask.taskId}: ${String(error)}`,
      );
    }
  }

  /** Process a single message: chunk creation, dedup, candidate extraction, policy eval, write. */
  private async processMessage(
    ctx: MemoryOperationContext,
    activeTask: ConversationTask,
    message: TurnCaptureMessage,
    index: number,
    createdChunks: ConversationChunk[],
    turnSemanticFrame?: TurnSemanticFrame,
  ): Promise<void> {
    const contentHash = stableHash([
      ctx.agentId,
      message.scope,
      message.role,
      normalizeText(message.content),
    ]);
    const exact = this.store.chunkRepo.findActiveByHash({
      agentId: ctx.agentId,
      scope: message.scope,
      role: message.role,
      contentHash,
    });
    const summary =
      exact?.summary ??
      (await this.store.reasoner.summarizeChunk(message.content, message.role, {
        stage: inferWriteLlmStage(message.role),
        audit: ctx.llmBudgetAudit,
        allowLlm: false,
      }));

    let chunk: ConversationChunk = exact ?? {
      chunkId: randomId("chunk"),
      agentId: ctx.agentId,
      scope: message.scope,
      sessionKey: message.sessionKey,
      turnId: message.turnId,
      seq: index,
      role: message.role,
      toolName: message.toolName,
      chunkKind: message.role === "tool" ? "tool_result" : "message",
      content: message.content,
      summary,
      contentHash,
      taskId: activeTask.taskId,
      dedupStatus: "active",
      mergeCount: 0,
      sourceRef: message.sourceRef,
      createdAt: message.observedAt,
      updatedAt: message.observedAt,
    };
    let sourceSegments = exact ? this.store.sourceSegmentRepo.listByChunk(exact.chunkId) : [];

    if (!exact) {
      const recent = this.store.chunkRepo.listRecentActive({
        agentId: ctx.agentId,
        scopes: [message.scope],
        sessionKey: message.sessionKey,
        limit: SCHEDULER_RECENT_ACTIVE_LIMIT,
      });
      const dedupThreshold = ctx.config.advanced.chunkDedupThreshold;
      const dedupProbeThreshold = Math.max(
        SCHEDULER_DEDUP_PROBE_FLOOR,
        dedupThreshold * SCHEDULER_DEDUP_PROBE_SCALE,
      );
      const scoredRecent = recent
        .filter((candidate) => candidate.role === message.role)
        .map((candidate) => {
          const contentScore = semanticTextSimilarity(candidate.content, message.content);
          const summaryScore = semanticTextSimilarity(
            candidate.summary || candidate.content,
            summary || message.content,
          );
          return {
            candidate,
            score: Math.max(contentScore, summaryScore * SCHEDULER_SUMMARY_SCORE_BOOST),
          };
        })
        .filter((entry) => entry.score >= dedupProbeThreshold)
        .sort((left, right) => right.score - left.score);
      const similar = scoredRecent[0];
      const nearDuplicateThreshold = Math.max(0.96, dedupThreshold + 0.08);
      if (similar && similar.score >= nearDuplicateThreshold) {
        // Conservative duplicate gate: only collapse near-identical content.
        chunk = {
          ...chunk,
          dedupStatus: "duplicate",
          dedupTarget: similar.candidate.chunkId,
          dedupReason: `conservative similarity duplicate=${similar.score.toFixed(3)}`,
        };
      }
    }

    if (!exact) {
      this.store.chunkRepo.insert(chunk);
      sourceSegments = buildSourceSegmentsForChunk(chunk);
      this.store.sourceSegmentRepo.insertMany(sourceSegments);
    }
    const taskChunks = this.store.chunkRepo.listByTask(activeTask.taskId);
    if (!exact && chunk.dedupStatus === "active") {
      this.store.retrievalBackend.upsertDocs([
        buildChunkVectorDoc(chunk, taskChunks),
        ...buildSourceSegmentVectorDocs({ chunk, segments: sourceSegments }),
      ]);
      createdChunks.push(chunk);
    }

    const projectContext = resolveCurrentProjectContext({
      store: this.store,
      ctx,
      scope: message.scope,
      observedAt: message.observedAt,
      activeTask,
      taskChunks,
      message,
    });
    const sourceRef = message.sourceRef || `${message.role}:${message.turnId}`;
    const compilerHints = frameHintsForSourceRef(turnSemanticFrame, sourceRef);
    const assistantSemanticText =
      message.role === "assistant" ? semanticOnlyCandidateText(compilerHints) : undefined;

    const candidate = buildCandidate({
      sourceKind:
        message.role === "tool" ? "tool" : message.role === "assistant" ? "assistant" : "user",
      rawText: message.role === "assistant" ? assistantSemanticText ?? "" : message.content,
      observedAt: message.observedAt,
      config: ctx.config,
      source: {
        sessionKey: message.sessionKey,
        toolName: message.toolName,
        messageId: message.turnId,
      },
      eventType: message.role === "tool" ? "tool_result" : "conversation_turn",
      metadata: {
        chunkId: chunk.chunkId,
        taskId: activeTask.taskId,
        chunkSummary: chunk.summary,
        sourceRef,
        ...(message.role === "assistant" ? { assistantSemanticOnly: Boolean(assistantSemanticText) } : {}),
        sourceGroupId: sourceSegments[0]?.sourceGroupId,
        segmentRefs: sourceSegments.map((segment) => segment.segmentId),
        segmentCount: sourceSegments.length,
        rawContentLength: message.content.length,
        semanticTextTruncated: message.content.trim().length > ctx.config.captureMaxChars,
        ...projectContext,
      },
    });
    if (!candidate) {
      return;
    }
    if (message.role === "assistant" && !assistantSemanticText) {
      return;
    }
    const candidateForPolicy = compilerHints
      ? {
          ...candidate,
          structuredHints: {
            ...(candidate.structuredHints ?? {}),
            ...compilerHints,
          },
          metadata: {
            ...(candidate.metadata ?? {}),
            turnSemanticCompiler: turnSemanticFrame?.compilerProvenance,
            turnSemanticFrame,
          },
        }
      : candidate;
    const policyResult = await evaluatePolicy(candidateForPolicy, ctx, {
      reasoner: this.store.reasoner,
    });
    const classification = classifyAction(policyResult.decision.action);
    if (classification === "ignore") {
      if (ctx.config.advanced.enableTelemetryAudit) {
        this.store.auditRepo.recordPolicyDecision({
          agentId: ctx.agentId,
          sessionKey: ctx.sessionKey,
          sourceRef: `${candidate.source.kind}:rejected:${candidate.candidateId}`,
          candidateText: candidate.rawText,
          decision: policyResult.decision,
          createdAt: message.observedAt,
          metadataJson: {
            ...policyAuditSourceMetadata(policyResult.candidate),
            turnSemanticCompile: turnSemanticFrame?.compilerProvenance,
            turnSemanticFrame,
            semanticDraftConsumed: policyResult.candidate.structuredHints?.semanticDraft
              ? {
                  sourceRef: policyResult.candidate.structuredHints.semanticDraft.sourceRef,
                  families: [
                    ...new Set(
                      policyResult.candidate.structuredHints.semanticDraft.assertionDrafts.map(
                        (entry) => entry.familyHint,
                      ),
                    ),
                  ],
                  timeframes: [
                    ...new Set(
                      policyResult.candidate.structuredHints.semanticDraft.assertionDrafts.map(
                        (entry) => entry.timeframeHint,
                      ),
                    ),
                  ],
                }
              : undefined,
            materializationHint: policyResult.candidate.structuredHints?.materializationHint,
            llmBudget: snapshotMemoryLlmBudgetAudit(ctx.llmBudgetAudit),
          },
        });
      }
      return;
    }
    const classified: ClassifiedCandidate = {
      ...policyResult.candidate,
      normalizedText: normalizeText(policyResult.candidate.rawText),
      scope: message.scope,
      policy: policyResult.decision,
      classification,
      confidence: 0,
    };
    classified.confidence = computeConfidence(classified);
    writeCandidate(this.store, ctx, classified);
  }

  /** Summarize the task and promote outcome after all messages are processed. */
  private async summarizeAndUpdateTask(
    ctx: MemoryOperationContext,
    activeTask: ConversationTask,
    messages: TurnCaptureMessage[],
    turnSemanticFrame?: TurnSemanticFrame,
  ): Promise<void> {
    const taskChunks = this.store.chunkRepo.listByTask(activeTask.taskId);
    const taskSummaryStage = messages.some(
      (message) => message.role === "user" || message.role === "tool",
    )
      ? "write_hot_path"
      : "post_answer_writeback";
    recordMemoryLlmBudgetCall(ctx.llmBudgetAudit, {
      label: "task-summary",
      stage: taskSummaryStage,
      provenance: "deterministic",
      mode: "deterministic",
      detail: "task-summary hot path resolved via compiler-first working summary selection",
    });
    const workingSummary = resolveWorkingTaskSummary({
      task: activeTask,
      chunks: taskChunks,
      taskProposal: turnSemanticFrame?.taskProposal,
      observedAt: messages.at(-1)?.observedAt ?? ctx.now,
    });
    const currentProjectContext = resolveCurrentProjectContext({
      store: this.store,
      ctx,
      scope: activeTask.scope,
      observedAt: messages.at(-1)?.observedAt ?? ctx.now,
      activeTask,
      taskChunks,
      message:
        messages.find((message) => message.role === "user") ?? messages[messages.length - 1]!,
    });
    const outcome = await this.maybePromoteTaskOutcome(
      ctx,
      activeTask,
      taskChunks,
      messages.at(-1)?.observedAt ?? ctx.now,
      taskSummaryStage,
      undefined,
    );
    const updatedTask: ConversationTask = {
      ...activeTask,
      title: workingSummary.title,
      summary: workingSummary.summary,
      metadataJson: mergeTaskMetadata(
        {
          ...activeTask.metadataJson,
          ...(currentProjectContext.currentProject
            ? { project: currentProjectContext.currentProject }
            : {}),
        },
        workingSummary.metadataJson,
        outcome.promotedOutcome,
        messages.at(-1)?.observedAt ?? ctx.now,
        outcome.emitted,
        outcome.alreadyEmitted,
        outcome.outcomeHypothesis,
      ),
      updatedAt: messages.at(-1)?.observedAt ?? ctx.now,
    };
    this.store.taskRepo.update(activeTask.taskId, updatedTask);
    upsertTaskVectorDoc(this.store, updatedTask, taskChunks);
  }
}
