import { clamp01, normalizeText, objectRecord, randomId, stableHash, truncateText } from "../support.mjs";
import { eligibleForLlmRefinement } from "./abstractionRefinement.mjs";
import { isProjectProfileStateKey, projectCodeFromStateKey, projectIdentityKey, projectNamesMatch, resolveProjectReference } from "./projectIdentity.mjs";
import { semanticTextSimilarity } from "./semantic/textSimilarity.mjs";
import { inferWriteLlmStage, recordMemoryLlmBudgetCall, snapshotMemoryLlmBudgetAudit } from "./llmBudgetAudit.mjs";
import { containsLikelySecret, looksLikePromptInjection } from "../security/injection.mjs";
import { assessAssistantChunk, assistantVectorSummary, assistantVectorText } from "./sourceWeighting.mjs";
import { buildOutcomeHypothesisCandidate, isAuthoritativeOutcomeResolutionMetadata } from "./outcomeHypotheses.mjs";
import { sanitizeTaskMetadata } from "./authority.mjs";
import { containsSensitiveValue, sensitivityScore } from "../security/pii.mjs";
import { buildVectorDocMetadata } from "./vectorDocMetadata.mjs";
import { computeConfidence } from "./normalize.mjs";
import { emitAssistantOutcomeLearningSignals, emitOutcomeFeedbackSignal } from "./signalLedger.mjs";
import { policyAuditSourceMetadata, writeCandidate } from "./write.mjs";
import { CHUNK_VECTOR_CONFIDENCE, SCHEDULER_DEDUP_PROBE_FLOOR, SCHEDULER_DEDUP_PROBE_SCALE, SCHEDULER_SUMMARY_SCORE_BOOST, TASK_VECTOR_CONFIDENCE } from "./constants.mjs";
import { resolveWorkingTaskSummary, semanticTaskSummaryText } from "./taskSummary.mjs";
import { compileTurnSemantics, frameHintsForSourceRef } from "./turnSemanticCompiler.mjs";
import { classifyAction } from "./classify.mjs";
import { evaluatePolicy } from "./policy.mjs";
import { buildCandidate } from "./extract.mjs";
import { buildSourceSegmentVectorDocs, buildSourceSegmentsForChunk } from "./sourceSegments.mjs";
import { decideTaskAssignment } from "./taskJudge.mjs";
//#region src/pipeline/turnScheduler.ts
function buildOutcomeKey(taskId, proposal) {
	return stableHash([
		taskId,
		proposal.eventType,
		normalizeText(proposal.summary)
	]);
}
function shouldSuppressMessageFromSemanticMemory(ctx, message) {
	const content = message.content.trim();
	if (!content) return false;
	const secretLike = containsLikelySecret(content);
	return looksLikePromptInjection(content) && !secretLike || secretLike || containsSensitiveValue(content) || sensitivityScore(content) > ctx.config.maxSensitivityAllowed;
}
function semanticDraftHasWriteSignals(hints) {
	const draft = hints?.semanticDraft;
	return Boolean((draft?.assertionDrafts.length ?? 0) > 0 || (draft?.correctionDrafts.length ?? 0) > 0 || (draft?.relationDrafts?.length ?? 0) > 0 || (draft?.resourceAssertions?.length ?? 0) > 0 || (draft?.adviceSignals?.length ?? 0) > 0 || (hints?.resourceAssertions?.length ?? 0) > 0 || (hints?.adviceSignals?.length ?? 0) > 0);
}
function semanticOnlyCandidateText(hints) {
	if (!semanticDraftHasWriteSignals(hints)) return;
	const draft = hints?.semanticDraft;
	const lines = [];
	for (const assertion of draft?.assertionDrafts ?? []) {
		const entities = (assertion.entityHints ?? []).map((entry) => entry.name).filter(Boolean);
		const slots = assertion.slotHints?.filter(Boolean) ?? [];
		const value = assertion.valueHint?.trim();
		const support = assertion.supportSpans?.find((span) => span.text.trim())?.text.trim();
		lines.push([
			assertion.familyHint,
			assertion.timeframeHint,
			entities.join(" / "),
			slots.join(" / "),
			value,
			support
		].filter(Boolean).join(" | "));
	}
	for (const correction of draft?.correctionDrafts ?? []) {
		const entry = correction.correction;
		lines.push([
			"correction",
			entry.targetKind,
			entry.canonicalKey,
			entry.predicate,
			entry.priorValue ? `from ${entry.priorValue}` : void 0,
			entry.nextValue ? `to ${entry.nextValue}` : void 0,
			entry.reason
		].filter(Boolean).join(" | "));
	}
	for (const relation of draft?.relationDrafts ?? []) {
		const entry = relation.relation;
		if (entry.polarity === "negated") continue;
		lines.push([
			"relation",
			entry.subject,
			entry.predicate,
			entry.relationSlot,
			entry.object,
			entry.reason
		].filter(Boolean).join(" | "));
	}
	for (const resource of [...draft?.resourceAssertions ?? [], ...hints?.resourceAssertions ?? []]) lines.push([
		"resource",
		resource.owner,
		resource.ownershipStatus,
		resource.resource,
		resource.resourceType,
		resource.supportText
	].filter(Boolean).join(" | "));
	for (const advice of [...draft?.adviceSignals ?? [], ...hints?.adviceSignals ?? []]) lines.push([
		"advice",
		advice.problemContext,
		advice.userResources?.join(" / "),
		advice.assistantRecommendation,
		advice.supportText
	].filter(Boolean).join(" | "));
	const text = lines.filter((line) => line.trim()).join("\n").trim();
	return text ? truncateText(text, 1200) : void 0;
}
function mapEvidenceChunkIds(chunks, indexes) {
	const unique = /* @__PURE__ */ new Set();
	for (const index of indexes) {
		const chunk = chunks[index - 1];
		if (chunk) unique.add(chunk.chunkId);
	}
	return [...unique];
}
async function refineOutcomeHypothesisCandidate(candidate, stage, audit) {
	if (!eligibleForLlmRefinement(candidate)) {
		recordMemoryLlmBudgetCall(audit, {
			label: "abstraction-refinement",
			stage,
			provenance: "deterministic",
			mode: "deterministic",
			detail: "abstraction refinement skipped because the candidate is not refinement-eligible"
		});
		return candidate;
	}
	recordMemoryLlmBudgetCall(audit, {
		label: "abstraction-refinement",
		stage,
		provenance: "deterministic",
		mode: "deferred",
		detail: "abstraction refinement is deferred to maintenance and does not block the write hot path"
	});
	return candidate;
}
function buildPromotedOutcome(task, chunks, proposal) {
	const evidenceChunkIds = mapEvidenceChunkIds(chunks, proposal.evidenceChunkIndexes ?? []);
	const outcomeKey = buildOutcomeKey(task.taskId, proposal);
	return {
		...proposal,
		evidenceChunkIds,
		outcomeKey
	};
}
function deterministicOutcomePromotionDecision(outcome, evidenceChunks) {
	evidenceChunks.filter((chunk) => chunk.role === "assistant").length;
	const userCount = evidenceChunks.filter((chunk) => chunk.role === "user").length;
	const toolCount = evidenceChunks.filter((chunk) => chunk.role === "tool").length;
	const hasUserEvidence = userCount > 0;
	const hasToolEvidence = toolCount > 0;
	const hasNonAssistantGrounding = hasUserEvidence || hasToolEvidence;
	const strongPhase = outcome.phase === "validated" || outcome.phase === "resolved";
	const promotionScore = clamp01(outcome.confidence * .22 + outcome.verificationScore * .3 + outcome.closureScore * .2 + (hasToolEvidence ? .16 : hasUserEvidence ? .1 : 0) + (strongPhase ? .1 : .02) - outcome.contradictionRisk * .32);
	const shouldPromote = hasNonAssistantGrounding && strongPhase && outcome.verificationScore >= .72 && outcome.closureScore >= .68 && outcome.contradictionRisk <= .28 && promotionScore >= .72;
	return {
		shouldPromote,
		promotionScore,
		reason: shouldPromote ? "deterministic promotion gate: grounded, validated, and low-contradiction outcome" : hasNonAssistantGrounding ? "deterministic promotion gate: keep the outcome provisional until it is more strongly validated" : "deterministic promotion gate: no non-assistant grounding is available"
	};
}
function mergeTaskMetadata(base, summaryMetadata, proposal, observedAt, emitted, alreadyEmitted, outcomeHypothesis) {
	const next = {
		...base,
		...summaryMetadata
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
		next.candidateOutcomeJudgeShouldPromote = Boolean(outcomeHypothesis.metadataJson.judgeShouldPromote);
		next.candidateOutcomeHasUserEvidence = Boolean(outcomeHypothesis.metadataJson.hasUserEvidence);
		next.candidateOutcomeHasToolEvidence = Boolean(outcomeHypothesis.metadataJson.hasToolEvidence);
		next.candidateOutcomeHasNonAssistantGrounding = Boolean(outcomeHypothesis.metadataJson.hasNonAssistantGrounding);
		if (typeof outcomeHypothesis.metadataJson.judgeReason === "string") next.candidateOutcomeJudgeReason = outcomeHypothesis.metadataJson.judgeReason;
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
		if (outcomeHypothesis) next.lastPromotedOutcomeHypothesisId = outcomeHypothesis.candidateId;
	} else if (alreadyEmitted) next.lastEmittedOutcomeKey = proposal.outcomeKey;
	return next;
}
function buildChunkVectorDoc(chunk, taskChunks) {
	const summary = assistantVectorSummary(chunk, taskChunks);
	const assistantAssessment = chunk.role === "assistant" && taskChunks ? assessAssistantChunk(chunk, taskChunks) : void 0;
	return {
		docId: `event:chunk:${chunk.chunkId}`,
		docKind: "event",
		sourceId: `chunk:${chunk.chunkId}`,
		scope: chunk.scope,
		agentId: chunk.agentId,
		text: `${chunk.role}: ${summary}\n${assistantVectorText({
			...chunk,
			content: truncateText(chunk.content, 500)
		}, taskChunks)}`.trim(),
		metadataJson: buildVectorDocMetadata({
			docType: "chunk",
			confidence: CHUNK_VECTOR_CONFIDENCE,
			observedAt: chunk.createdAt,
			lineage: {
				sourceKind: "chunk",
				sourceId: chunk.chunkId,
				sourceRef: chunk.sourceRef
			},
			extra: {
				chunkId: chunk.chunkId,
				scope: chunk.scope,
				role: chunk.role,
				sessionKey: chunk.sessionKey,
				taskId: chunk.taskId,
				dedupStatus: chunk.dedupStatus,
				...assistantAssessment ? {
					assistantWeight: Number(assistantAssessment.weight.toFixed(3)),
					assistantGrounding: Number(assistantAssessment.grounding.toFixed(3)),
					assistantComplexity: Number(assistantAssessment.complexity.toFixed(3)),
					assistantSummaryOnly: assistantAssessment.useSummaryOnly,
					semanticRole: assistantAssessment.semanticRole,
					memoryClass: assistantAssessment.memoryClass,
					recallVisibility: assistantAssessment.recallVisibility
				} : {}
			}
		}),
		createdAt: chunk.createdAt,
		updatedAt: chunk.updatedAt
	};
}
function taskSupportRefs(chunks) {
	if (!chunks || chunks.length === 0) return [];
	return [...new Set(chunks.map((chunk) => chunk.sourceRef).filter((sourceRef) => Boolean(sourceRef?.trim())))];
}
function taskVectorDocId(task) {
	return `state:task:${task.taskId}`;
}
function recallableTaskTitle(title) {
	const trimmed = title.trim();
	if (!trimmed) return;
	return normalizeText(trimmed) === "active task" ? void 0 : trimmed;
}
function buildTaskVectorDoc(task, chunks) {
	const metadata = sanitizeTaskMetadata(task.metadataJson);
	const rawMetadata = objectRecord(task.metadataJson);
	const semanticSummary = semanticTaskSummaryText(task);
	const supportRefs = taskSupportRefs(chunks);
	const taskText = [
		recallableTaskTitle(task.title),
		typeof rawMetadata?.candidateResolution === "string" && rawMetadata.candidateResolution.trim() ? rawMetadata.candidateResolution.trim() : void 0,
		metadata.currentTask,
		metadata.project,
		metadata.nextAction,
		metadata.blocker,
		semanticSummary
	].filter((value) => Boolean(value)).join("\n").trim();
	if (!taskText) return null;
	return {
		docId: taskVectorDocId(task),
		docKind: "state",
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
				sourceRef: `task:${task.taskId}`
			},
			extra: {
				taskId: task.taskId,
				scope: task.scope,
				sessionKey: task.sessionKey,
				status: task.status,
				sourceRef: `task:${task.taskId}`,
				supportRefs
			}
		}),
		createdAt: task.startedAt,
		updatedAt: task.updatedAt
	};
}
function upsertTaskVectorDoc(store, task, chunks) {
	const doc = buildTaskVectorDoc(task, chunks);
	if (!doc) {
		store.retrievalBackend.deleteDocs([taskVectorDocId(task)]);
		return;
	}
	store.retrievalBackend.upsertDocs([doc]);
}
function resolveCurrentProjectContext(params) {
	const metadataProject = sanitizeTaskMetadata(params.activeTask.metadataJson).project;
	const activeProjectState = params.store.stateRepo.get({
		agentId: params.ctx.agentId,
		scopes: [params.scope],
		key: "project.active_project",
		includeExpired: true,
		now: params.observedAt
	})[0];
	const stateProject = typeof activeProjectState?.valueJson.project === "string" && activeProjectState.valueJson.project.trim() ? activeProjectState.valueJson.project.trim() : void 0;
	const projectStates = params.store.stateRepo.get({
		agentId: params.ctx.agentId,
		scopes: [params.scope],
		includeExpired: true,
		now: params.observedAt
	}).filter((state) => isProjectProfileStateKey(state.key));
	const messageProjectKey = projectIdentityKey(params.message.content);
	const entityMatchedProject = projectStates.find((state) => {
		const projectCode = projectCodeFromStateKey(state.key);
		return projectCode ? messageProjectKey.includes(projectIdentityKey(projectCode)) : false;
	});
	const knownProjects = projectStates.map((state) => projectCodeFromStateKey(state.key)).filter((entry) => Boolean(entry));
	const singletonProject = !metadataProject && !stateProject && projectStates.length === 1 ? projectCodeFromStateKey(projectStates[0].key) : void 0;
	const currentProject = resolveProjectReference(metadataProject ?? stateProject ?? (entityMatchedProject ? projectCodeFromStateKey(entityMatchedProject.key) : void 0) ?? singletonProject ?? "", {
		currentProject: metadataProject ?? stateProject,
		knownProjects,
		allowDescriptorAlias: true
	});
	if (!currentProject) return {};
	const currentProjectProfile = objectRecord((params.store.stateRepo.get({
		agentId: params.ctx.agentId,
		scopes: [params.scope],
		key: `project.${currentProject}`,
		includeExpired: true,
		now: params.observedAt
	})[0] ?? projectStates.find((state) => {
		const projectCode = projectCodeFromStateKey(state.key);
		return projectCode ? projectNamesMatch(projectCode, currentProject) : false;
	}))?.valueJson);
	return currentProjectProfile ? {
		currentProject,
		currentProjectProfile
	} : { currentProject };
}
function turnSourceRefs(messages) {
	return [...new Set(messages.map((message) => message.sourceRef).filter((entry) => Boolean(entry?.trim())))];
}
function turnSemanticInputHash(messages) {
	return stableHash(messages.map((message) => [
		message.role,
		message.sourceRef,
		normalizeText(message.content)
	].join("")));
}
function semanticFrameDraftCounts(frame) {
	return {
		assertionDrafts: frame?.assertionDrafts?.length ?? 0,
		correctionDrafts: frame?.correctionDrafts?.length ?? 0,
		relationDrafts: frame?.relationDrafts?.length ?? 0,
		resourceAssertions: frame?.resourceAssertions?.length ?? 0,
		adviceSignals: frame?.adviceSignals?.length ?? 0,
		supportSpans: frame?.supportSpans?.length ?? 0
	};
}
function semanticFrameResolvedByLlm(frame) {
	const provenance = objectRecord(frame?.compilerProvenance);
	return provenance?.source === "llm" && provenance.mode !== "fallback";
}
var MemxTurnScheduler = class {
	store;
	logger;
	chain = Promise.resolve();
	constructor(store, logger) {
		this.store = store;
		this.logger = logger;
	}
	enqueue(ctx, messages) {
		if (messages.length === 0) return this.chain;
		this.chain = this.chain.then(() => this.processTurn(ctx, messages)).catch((error) => {
			const turnId = messages[0]?.turnId ?? "unknown";
			this.logger.warn(`memx: turn scheduler failed for turn ${turnId} (${messages.length} messages): ${String(error)}`);
		});
		return this.chain;
	}
	flush() {
		return this.chain;
	}
	async stageRecallableTurn(ctx, messages) {
		const safeMessages = this.safeMessagesForTurn(ctx, messages);
		const hasUserMessage = messages.some((message) => message.role === "user");
		const hasSafeUserMessage = safeMessages.some((message) => message.role === "user");
		if (safeMessages.length === 0 || hasUserMessage && !hasSafeUserMessage) return false;
		const scope = safeMessages[0]?.scope ?? ctx.scopes[0] ?? `agent:${ctx.agentId}`;
		const activeTask = this.getOrCreateActiveTask(ctx, safeMessages, scope);
		await this.persistRecallableChunks(ctx, activeTask, safeMessages);
		return true;
	}
	safeMessagesForTurn(ctx, messages) {
		return messages.filter((message) => !shouldSuppressMessageFromSemanticMemory(ctx, message));
	}
	getOrCreateActiveTask(ctx, safeMessages, scope) {
		let activeTask = this.store.taskRepo.getActive({
			agentId: ctx.agentId,
			scope,
			sessionKey: safeMessages[0].sessionKey
		});
		if (!activeTask) {
			activeTask = {
				taskId: randomId("task"),
				agentId: ctx.agentId,
				scope,
				sessionKey: safeMessages[0].sessionKey,
				title: "Active task",
				summary: "",
				status: "active",
				startedAt: safeMessages[0].observedAt,
				updatedAt: safeMessages[0].observedAt,
				metadataJson: {}
			};
			this.store.taskRepo.create(activeTask);
		}
		return activeTask;
	}
	async maybePromoteTaskOutcome(ctx, task, taskChunks, observedAt, stage, proposal) {
		const promotedOutcome = proposal ? buildPromotedOutcome(task, taskChunks, proposal) : void 0;
		let emitted = false;
		const lastEmittedOutcomeKey = typeof task.metadataJson.lastEmittedOutcomeKey === "string" ? task.metadataJson.lastEmittedOutcomeKey : void 0;
		if (promotedOutcome && lastEmittedOutcomeKey === promotedOutcome.outcomeKey) return {
			promotedOutcome,
			emitted: false,
			alreadyEmitted: true
		};
		if (promotedOutcome && lastEmittedOutcomeKey !== promotedOutcome.outcomeKey) {
			const evidenceChunks = taskChunks.filter((chunk) => promotedOutcome.evidenceChunkIds.includes(chunk.chunkId));
			const decision = deterministicOutcomePromotionDecision(promotedOutcome, evidenceChunks);
			recordMemoryLlmBudgetCall(ctx.llmBudgetAudit, {
				label: "outcome-promotion",
				stage,
				provenance: "deterministic",
				mode: "deterministic",
				detail: "outcome promotion is resolved by the deterministic gate and does not use an extra hot-path LLM"
			});
			promotedOutcome.promotionScore = decision.promotionScore;
			const outcomeHypothesis = await refineOutcomeHypothesisCandidate(buildOutcomeHypothesisCandidate({
				agentId: ctx.agentId,
				task,
				outcome: promotedOutcome,
				observedAt,
				evidenceChunks,
				decision
			}), stage, ctx.llmBudgetAudit);
			const sourceEpoch = ctx.readEpoch ?? this.store.client.currentMemoryEpoch(ctx.agentId);
			outcomeHypothesis.derivedFromMinEpoch = outcomeHypothesis.derivedFromMinEpoch ?? sourceEpoch;
			outcomeHypothesis.derivedFromMaxEpoch = outcomeHypothesis.derivedFromMaxEpoch ?? sourceEpoch;
			outcomeHypothesis.materializedEpoch = this.store.client.nextMemoryEpoch(ctx.agentId, ctx.now);
			outcomeHypothesis.derivedFromKind = outcomeHypothesis.derivedFromKind ?? "task_outcome";
			outcomeHypothesis.derivedFromIds = outcomeHypothesis.derivedFromIds ?? [...outcomeHypothesis.supportContentRefs, ...outcomeHypothesis.supportBeliefIds];
			outcomeHypothesis.derivedAtEpoch = outcomeHypothesis.derivedAtEpoch ?? sourceEpoch;
			outcomeHypothesis.derivationPolicyVersion = outcomeHypothesis.derivationPolicyVersion ?? "memx-authority-v3";
			this.store.abstractionRepo.upsert(outcomeHypothesis);
			emitAssistantOutcomeLearningSignals(this.store, ctx, {
				task,
				outcome: promotedOutcome,
				evidenceChunks,
				shouldPromote: decision.shouldPromote,
				emitted: false,
				reason: decision.reason
			});
			emitOutcomeFeedbackSignal(this.store, ctx, {
				task,
				outcome: promotedOutcome,
				emitted: false
			});
			return {
				promotedOutcome,
				emitted: false,
				alreadyEmitted: false,
				outcomeHypothesis
			};
		}
		return {
			promotedOutcome,
			emitted,
			alreadyEmitted: false
		};
	}
	async closeTask(ctx, task, observedAt, status, stage = "write_hot_path") {
		const closedChunks = this.store.chunkRepo.listByTask(task.taskId);
		const workingSummary = resolveWorkingTaskSummary({
			task,
			chunks: closedChunks,
			observedAt
		});
		const outcomeSummary = await this.store.reasoner.summarizeTask(closedChunks, {
			stage,
			audit: ctx.llmBudgetAudit
		});
		const outcome = await this.maybePromoteTaskOutcome(ctx, task, closedChunks, observedAt, stage, outcomeSummary.synthesizedEvent);
		const closedTask = {
			...task,
			title: workingSummary.title,
			summary: workingSummary.summary,
			metadataJson: mergeTaskMetadata(task.metadataJson, workingSummary.metadataJson, outcome.promotedOutcome, observedAt, outcome.emitted, outcome.alreadyEmitted, outcome.outcomeHypothesis),
			status,
			endedAt: observedAt,
			updatedAt: observedAt
		};
		this.store.taskRepo.update(task.taskId, closedTask);
		upsertTaskVectorDoc(this.store, closedTask, closedChunks);
		return closedTask;
	}
	reopenTask(task, observedAt) {
		const reopenedTask = {
			...task,
			status: "active",
			endedAt: void 0,
			updatedAt: observedAt
		};
		this.store.taskRepo.update(task.taskId, reopenedTask);
		upsertTaskVectorDoc(this.store, reopenedTask, this.store.chunkRepo.listByTask(reopenedTask.taskId));
		return reopenedTask;
	}
	async persistRecallableChunks(ctx, activeTask, messages) {
		for (const [index, message] of messages.entries()) {
			const contentHash = stableHash([
				ctx.agentId,
				message.scope,
				message.role,
				normalizeText(message.content)
			]);
			if (this.store.chunkRepo.findActiveByHash({
				agentId: ctx.agentId,
				scope: message.scope,
				role: message.role,
				contentHash
			})) continue;
			const summary = await this.store.reasoner.summarizeChunk(message.content, message.role, {
				stage: inferWriteLlmStage(message.role),
				audit: ctx.llmBudgetAudit,
				allowLlm: false
			});
			const chunk = {
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
				updatedAt: message.observedAt
			};
			this.store.chunkRepo.insert(chunk);
			const sourceSegments = buildSourceSegmentsForChunk(chunk);
			this.store.sourceSegmentRepo.insertMany(sourceSegments);
			const taskChunks = this.store.chunkRepo.listByTask(activeTask.taskId);
			this.store.retrievalBackend.upsertDocs([buildChunkVectorDoc(chunk, taskChunks), ...buildSourceSegmentVectorDocs({
				chunk,
				segments: sourceSegments
			})]);
		}
	}
	async processTurn(ctx, messages) {
		const safeMessages = this.safeMessagesForTurn(ctx, messages);
		const hasUserMessage = messages.some((message) => message.role === "user");
		const hasSafeUserMessage = safeMessages.some((message) => message.role === "user");
		if (safeMessages.length === 0 || hasUserMessage && !hasSafeUserMessage) return;
		const scope = safeMessages[0]?.scope ?? ctx.scopes[0] ?? `agent:${ctx.agentId}`;
		let activeTask = this.getOrCreateActiveTask(ctx, safeMessages, scope);
		await this.persistRecallableChunks(ctx, activeTask, safeMessages);
		const activeChunks = activeTask ? this.store.chunkRepo.listByTask(activeTask.taskId) : [];
		const recentTasks = this.store.taskRepo.listRecent({
			agentId: ctx.agentId,
			scopes: [scope],
			sessionKey: safeMessages[0].sessionKey,
			limit: 6,
			includeSkipped: false
		});
		const recentChunksByTask = Object.fromEntries(recentTasks.map((task) => [task.taskId, this.store.chunkRepo.listByTask(task.taskId)]));
		const semanticJobSourceRefs = turnSourceRefs(safeMessages);
		const semanticJobInputHash = turnSemanticInputHash(safeMessages);
		const semanticJobTurnId = safeMessages[0].turnId;
		const semanticJobScope = safeMessages[0].scope ?? scope;
		if (ctx.config.advanced.enableTurnSemanticCompiler) this.store.auditRepo.recordSemanticWriteAttemptStart({
			agentId: ctx.agentId,
			sessionKey: safeMessages[0].sessionKey,
			scope: semanticJobScope,
			turnId: semanticJobTurnId,
			sourceRefs: semanticJobSourceRefs,
			inputHash: semanticJobInputHash,
			startedAt: ctx.now
		});
		let turnSemanticFrame;
		try {
			turnSemanticFrame = await compileTurnSemantics({
				messages: safeMessages,
				ctx,
				activeTask,
				activeChunks,
				recentTasks,
				recentChunksByTask,
				reasoner: this.store.reasoner
			});
		} catch (error) {
			this.store.auditRepo.finishSemanticWriteAttempt({
				agentId: ctx.agentId,
				sessionKey: safeMessages[0].sessionKey,
				turnId: semanticJobTurnId,
				status: "retrying",
				error: `turn semantic compiler failed: ${error instanceof Error ? error.message : String(error)}`,
				resultJson: {
					stage: "write_hot_path",
					sourceRefs: semanticJobSourceRefs,
					llmBudget: snapshotMemoryLlmBudgetAudit(ctx.llmBudgetAudit)
				},
				completedAt: ctx.now
			});
			this.logger.warn(`memx: turn semantic compiler failed for turn ${semanticJobTurnId}: ${String(error)}`);
		}
		if (ctx.config.advanced.enableTurnSemanticCompiler && turnSemanticFrame) {
			const resolvedByLlm = semanticFrameResolvedByLlm(turnSemanticFrame);
			this.store.auditRepo.finishSemanticWriteAttempt({
				agentId: ctx.agentId,
				sessionKey: safeMessages[0].sessionKey,
				turnId: semanticJobTurnId,
				status: resolvedByLlm ? "succeeded" : "retrying",
				error: resolvedByLlm ? void 0 : "turn semantic compiler returned no recognized LLM semantic frame",
				resultJson: {
					stage: "write_hot_path",
					sourceRefs: semanticJobSourceRefs,
					compilerProvenance: turnSemanticFrame.compilerProvenance,
					...semanticFrameDraftCounts(turnSemanticFrame),
					llmBudget: snapshotMemoryLlmBudgetAudit(ctx.llmBudgetAudit)
				},
				completedAt: ctx.now
			});
		} else if (ctx.config.advanced.enableTurnSemanticCompiler) this.store.auditRepo.finishSemanticWriteAttempt({
			agentId: ctx.agentId,
			sessionKey: safeMessages[0].sessionKey,
			turnId: semanticJobTurnId,
			status: "retrying",
			error: "turn semantic compiler returned no recognized LLM semantic frame",
			resultJson: {
				stage: "write_hot_path",
				sourceRefs: semanticJobSourceRefs,
				llmBudget: snapshotMemoryLlmBudgetAudit(ctx.llmBudgetAudit)
			},
			completedAt: ctx.now
		});
		const taskProposal = turnSemanticFrame?.taskProposal;
		const assignment = await decideTaskAssignment({
			activeTask,
			activeChunks,
			recentTasks,
			recentChunksByTask,
			newMessages: safeMessages,
			ctx,
			taskProposal
		});
		if (assignment.decision === "new") {
			if (activeTask) {
				const closedChunks = this.store.chunkRepo.listByTask(activeTask.taskId);
				activeTask = await this.closeTask(ctx, activeTask, safeMessages[0].observedAt, closedChunks.length >= 2 ? "completed" : "skipped", "write_hot_path");
			}
			activeTask = {
				taskId: randomId("task"),
				agentId: ctx.agentId,
				scope,
				sessionKey: safeMessages[0].sessionKey,
				title: "Active task",
				summary: "",
				status: "active",
				startedAt: safeMessages[0].observedAt,
				updatedAt: safeMessages[0].observedAt,
				metadataJson: {}
			};
			this.store.taskRepo.create(activeTask);
		} else if (assignment.decision === "resume" && assignment.targetTaskId && assignment.targetTaskId !== activeTask?.taskId) {
			if (activeTask) {
				const closedChunks = this.store.chunkRepo.listByTask(activeTask.taskId);
				await this.closeTask(ctx, activeTask, safeMessages[0].observedAt, closedChunks.length >= 2 ? "completed" : "skipped", "write_hot_path");
			}
			const resumedTask = recentTasks.find((task) => task.taskId === assignment.targetTaskId);
			activeTask = resumedTask ? this.reopenTask(resumedTask, safeMessages[0].observedAt) : {
				taskId: randomId("task"),
				agentId: ctx.agentId,
				scope,
				sessionKey: safeMessages[0].sessionKey,
				title: "Active task",
				summary: "",
				status: "active",
				startedAt: safeMessages[0].observedAt,
				updatedAt: safeMessages[0].observedAt,
				metadataJson: {}
			};
			if (!resumedTask) this.store.taskRepo.create(activeTask);
		}
		if (!activeTask) return;
		const createdChunks = [];
		const results = await Promise.allSettled(safeMessages.map((message, index) => this.processMessage(ctx, activeTask, message, index, createdChunks, turnSemanticFrame)));
		for (const [index, result] of results.entries()) if (result.status === "rejected") this.logger.warn(`memx: turn scheduler skipped message ${index} for task ${activeTask.taskId}: ${String(result.reason)}`);
		try {
			await this.summarizeAndUpdateTask(ctx, activeTask, safeMessages, turnSemanticFrame);
		} catch (error) {
			this.logger.warn(`memx: turn scheduler failed task summarization for ${activeTask.taskId}: ${String(error)}`);
		}
	}
	/** Process a single message: chunk creation, dedup, candidate extraction, policy eval, write. */
	async processMessage(ctx, activeTask, message, index, createdChunks, turnSemanticFrame) {
		const contentHash = stableHash([
			ctx.agentId,
			message.scope,
			message.role,
			normalizeText(message.content)
		]);
		const exact = this.store.chunkRepo.findActiveByHash({
			agentId: ctx.agentId,
			scope: message.scope,
			role: message.role,
			contentHash
		});
		const summary = exact?.summary ?? await this.store.reasoner.summarizeChunk(message.content, message.role, {
			stage: inferWriteLlmStage(message.role),
			audit: ctx.llmBudgetAudit,
			allowLlm: false
		});
		let chunk = exact ?? {
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
			updatedAt: message.observedAt
		};
		let sourceSegments = exact ? this.store.sourceSegmentRepo.listByChunk(exact.chunkId) : [];
		if (!exact) {
			const recent = this.store.chunkRepo.listRecentActive({
				agentId: ctx.agentId,
				scopes: [message.scope],
				sessionKey: message.sessionKey,
				limit: 12
			});
			const dedupThreshold = ctx.config.advanced.chunkDedupThreshold;
			const dedupProbeThreshold = Math.max(SCHEDULER_DEDUP_PROBE_FLOOR, dedupThreshold * SCHEDULER_DEDUP_PROBE_SCALE);
			const similar = recent.filter((candidate) => candidate.role === message.role).map((candidate) => {
				const contentScore = semanticTextSimilarity(candidate.content, message.content);
				const summaryScore = semanticTextSimilarity(candidate.summary || candidate.content, summary || message.content);
				return {
					candidate,
					score: Math.max(contentScore, summaryScore * SCHEDULER_SUMMARY_SCORE_BOOST)
				};
			}).filter((entry) => entry.score >= dedupProbeThreshold).sort((left, right) => right.score - left.score)[0];
			const nearDuplicateThreshold = Math.max(.96, dedupThreshold + .08);
			if (similar && similar.score >= nearDuplicateThreshold) chunk = {
				...chunk,
				dedupStatus: "duplicate",
				dedupTarget: similar.candidate.chunkId,
				dedupReason: `conservative similarity duplicate=${similar.score.toFixed(3)}`
			};
		}
		if (!exact) {
			this.store.chunkRepo.insert(chunk);
			sourceSegments = buildSourceSegmentsForChunk(chunk);
			this.store.sourceSegmentRepo.insertMany(sourceSegments);
		}
		const taskChunks = this.store.chunkRepo.listByTask(activeTask.taskId);
		if (!exact && chunk.dedupStatus === "active") {
			this.store.retrievalBackend.upsertDocs([buildChunkVectorDoc(chunk, taskChunks), ...buildSourceSegmentVectorDocs({
				chunk,
				segments: sourceSegments
			})]);
			createdChunks.push(chunk);
		}
		const projectContext = resolveCurrentProjectContext({
			store: this.store,
			ctx,
			scope: message.scope,
			observedAt: message.observedAt,
			activeTask,
			taskChunks,
			message
		});
		const sourceRef = message.sourceRef || `${message.role}:${message.turnId}`;
		const compilerHints = frameHintsForSourceRef(turnSemanticFrame, sourceRef);
		const assistantSemanticText = message.role === "assistant" ? semanticOnlyCandidateText(compilerHints) : void 0;
		const candidate = buildCandidate({
			sourceKind: message.role === "tool" ? "tool" : message.role === "assistant" ? "assistant" : "user",
			rawText: message.role === "assistant" ? assistantSemanticText ?? "" : message.content,
			observedAt: message.observedAt,
			config: ctx.config,
			source: {
				sessionKey: message.sessionKey,
				toolName: message.toolName,
				messageId: message.turnId
			},
			eventType: message.role === "tool" ? "tool_result" : "conversation_turn",
			metadata: {
				chunkId: chunk.chunkId,
				taskId: activeTask.taskId,
				chunkSummary: chunk.summary,
				sourceRef,
				...message.role === "assistant" ? { assistantSemanticOnly: Boolean(assistantSemanticText) } : {},
				sourceGroupId: sourceSegments[0]?.sourceGroupId,
				segmentRefs: sourceSegments.map((segment) => segment.segmentId),
				segmentCount: sourceSegments.length,
				rawContentLength: message.content.length,
				semanticTextTruncated: message.content.trim().length > ctx.config.captureMaxChars,
				...projectContext
			}
		});
		if (!candidate) return;
		if (message.role === "assistant" && !assistantSemanticText) return;
		const policyResult = await evaluatePolicy(compilerHints ? {
			...candidate,
			structuredHints: {
				...candidate.structuredHints ?? {},
				...compilerHints
			},
			metadata: {
				...candidate.metadata ?? {},
				turnSemanticCompiler: turnSemanticFrame?.compilerProvenance,
				turnSemanticFrame
			}
		} : candidate, ctx, { reasoner: this.store.reasoner });
		const classification = classifyAction(policyResult.decision.action);
		if (classification === "ignore") {
			if (ctx.config.advanced.enableTelemetryAudit) this.store.auditRepo.recordPolicyDecision({
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
					semanticDraftConsumed: policyResult.candidate.structuredHints?.semanticDraft ? {
						sourceRef: policyResult.candidate.structuredHints.semanticDraft.sourceRef,
						families: [...new Set(policyResult.candidate.structuredHints.semanticDraft.assertionDrafts.map((entry) => entry.familyHint))],
						timeframes: [...new Set(policyResult.candidate.structuredHints.semanticDraft.assertionDrafts.map((entry) => entry.timeframeHint))]
					} : void 0,
					materializationHint: policyResult.candidate.structuredHints?.materializationHint,
					llmBudget: snapshotMemoryLlmBudgetAudit(ctx.llmBudgetAudit)
				}
			});
			return;
		}
		const classified = {
			...policyResult.candidate,
			normalizedText: normalizeText(policyResult.candidate.rawText),
			scope: message.scope,
			policy: policyResult.decision,
			classification,
			confidence: 0
		};
		classified.confidence = computeConfidence(classified);
		writeCandidate(this.store, ctx, classified);
	}
	/** Summarize the task and promote outcome after all messages are processed. */
	async summarizeAndUpdateTask(ctx, activeTask, messages, turnSemanticFrame) {
		const taskChunks = this.store.chunkRepo.listByTask(activeTask.taskId);
		const taskSummaryStage = messages.some((message) => message.role === "user" || message.role === "tool") ? "write_hot_path" : "post_answer_writeback";
		recordMemoryLlmBudgetCall(ctx.llmBudgetAudit, {
			label: "task-summary",
			stage: taskSummaryStage,
			provenance: "deterministic",
			mode: "deterministic",
			detail: "task-summary hot path resolved via compiler-first working summary selection"
		});
		const workingSummary = resolveWorkingTaskSummary({
			task: activeTask,
			chunks: taskChunks,
			taskProposal: turnSemanticFrame?.taskProposal,
			observedAt: messages.at(-1)?.observedAt ?? ctx.now
		});
		const currentProjectContext = resolveCurrentProjectContext({
			store: this.store,
			ctx,
			scope: activeTask.scope,
			observedAt: messages.at(-1)?.observedAt ?? ctx.now,
			activeTask,
			taskChunks,
			message: messages.find((message) => message.role === "user") ?? messages[messages.length - 1]
		});
		const outcome = await this.maybePromoteTaskOutcome(ctx, activeTask, taskChunks, messages.at(-1)?.observedAt ?? ctx.now, taskSummaryStage, void 0);
		const updatedTask = {
			...activeTask,
			title: workingSummary.title,
			summary: workingSummary.summary,
			metadataJson: mergeTaskMetadata({
				...activeTask.metadataJson,
				...currentProjectContext.currentProject ? { project: currentProjectContext.currentProject } : {}
			}, workingSummary.metadataJson, outcome.promotedOutcome, messages.at(-1)?.observedAt ?? ctx.now, outcome.emitted, outcome.alreadyEmitted, outcome.outcomeHypothesis),
			updatedAt: messages.at(-1)?.observedAt ?? ctx.now
		};
		this.store.taskRepo.update(activeTask.taskId, updatedTask);
		upsertTaskVectorDoc(this.store, updatedTask, taskChunks);
	}
};
//#endregion
export { MemxTurnScheduler };
