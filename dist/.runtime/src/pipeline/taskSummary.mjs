import { objectRecord, stableHash } from "../support.mjs";
import { isQuestionLike } from "./semantic/heuristics.mjs";
import "./semantics.mjs";
import { sanitizeTaskMetadata } from "./authority.mjs";
//#region src/pipeline/taskSummary.ts
function metadataString(value) {
	return typeof value === "string" && value.trim() ? value.trim() : void 0;
}
function stringSet(value) {
	if (!Array.isArray(value)) return [];
	const seen = /* @__PURE__ */ new Set();
	const ordered = [];
	for (const entry of value) {
		if (typeof entry !== "string" || !entry.trim()) continue;
		const trimmed = entry.trim();
		if (seen.has(trimmed)) continue;
		seen.add(trimmed);
		ordered.push(trimmed);
	}
	return ordered;
}
function normalizeTaskSummarySourceValue(value) {
	if (value === "compiler" || value === "llm_unavailable" || value === "heuristic_fallback" || value === "maintenance_llm") return value;
	if (value === "heuristic") return "heuristic_fallback";
	if (value === "llm") return "maintenance_llm";
}
function normalizeTaskSummaryQualityValue(value) {
	return value === "working" || value === "stable" ? value : void 0;
}
function taskSummarySource(metadataJson) {
	return normalizeTaskSummarySourceValue(objectRecord(metadataJson)?.summarySource);
}
function taskSummaryQuality(metadataJson) {
	return normalizeTaskSummaryQualityValue(objectRecord(metadataJson)?.summaryQuality);
}
function taskSummarySupportsSemanticConsumers(metadataJson) {
	const source = taskSummarySource(metadataJson);
	return source === "compiler" || source === "maintenance_llm";
}
function semanticTaskSummaryText(task) {
	return taskSummarySupportsSemanticConsumers(task.metadataJson) ? metadataString(task.summary) : void 0;
}
function taskSummaryMetadataFields(params) {
	return {
		summarySource: params.summarySource,
		summaryQuality: params.summaryQuality,
		summaryBasisFingerprint: params.summaryBasisFingerprint,
		summaryUpdatedAt: params.observedAt,
		...params.compilerTaskSummary ? { compilerTaskSummary: params.compilerTaskSummary } : {},
		...typeof params.compilerTaskSummaryConfidence === "number" ? { compilerTaskSummaryConfidence: params.compilerTaskSummaryConfidence } : {}
	};
}
function compilerSummaryFromProposal(taskProposal, existingMetadata) {
	const proposalSummary = metadataString(taskProposal?.summary);
	if (proposalSummary) return {
		summary: proposalSummary,
		...typeof taskProposal?.summaryConfidence === "number" ? { confidence: taskProposal.summaryConfidence } : {}
	};
	const metadata = objectRecord(existingMetadata);
	const persistedSummary = metadataString(metadata?.compilerTaskSummary);
	if (!persistedSummary) return;
	const confidence = typeof metadata?.compilerTaskSummaryConfidence === "number" ? metadata.compilerTaskSummaryConfidence : void 0;
	return {
		summary: persistedSummary,
		...typeof confidence === "number" ? { confidence } : {}
	};
}
function isAcceptableCompilerSummary(summary, confidence) {
	const normalized = summary.trim();
	if (!normalized || normalized.length < 12) return false;
	if (isQuestionLike(normalized)) return false;
	if (/^(?:active task|conversation task|当前任务|对话任务)$/iu.test(normalized)) return false;
	if (typeof confidence === "number" && confidence < .42) return false;
	return true;
}
function recentSinceIso(now, days) {
	const date = new Date(now);
	date.setUTCDate(date.getUTCDate() - days);
	return date.toISOString();
}
function structuredEventSummary(event) {
	const metadata = objectRecord(event.metadataJson);
	return metadataString(objectRecord(metadata?.memxTemporalFacet)?.summary) ?? metadataString(metadata?.memxStructuredSummary) ?? event.text;
}
function computeTaskSummaryBasisFingerprint(params) {
	return stableHash([
		params.taskId,
		...params.chunkIds,
		params.candidateResolution ?? "",
		params.candidateResolutionPhase ?? "",
		...params.candidateResolutionEvidenceChunkIds ?? [],
		params.compilerTaskSummary ?? "",
		typeof params.compilerTaskSummaryConfidence === "number" ? params.compilerTaskSummaryConfidence.toFixed(3) : "",
		params.project ?? "",
		params.currentTask ?? "",
		params.nextAction ?? "",
		params.blocker ?? "",
		params.lastEmittedOutcomeKey ?? "",
		...params.linkedEventIds ?? []
	]);
}
function buildTaskSummaryEvidenceSet(params) {
	const rawMetadata = objectRecord(params.task.metadataJson);
	const canonicalMetadata = sanitizeTaskMetadata(params.task.metadataJson);
	const compilerTaskSummary = compilerSummaryFromProposal(void 0, rawMetadata);
	const candidateResolution = metadataString(rawMetadata?.candidateResolution);
	const candidateResolutionPhase = metadataString(rawMetadata?.candidateResolutionPhase);
	const candidateResolutionEvidenceChunkIds = stringSet(rawMetadata?.candidateResolutionEvidenceChunkIds);
	const lastEmittedOutcomeKey = metadataString(rawMetadata?.lastEmittedOutcomeKey);
	const chunkIds = params.chunks.map((chunk) => chunk.chunkId);
	const chunkIdSet = new Set(chunkIds);
	const linkedEvents = params.eventRepo.search({
		agentId: params.task.agentId,
		scopes: [params.task.scope],
		limit: 48,
		since: recentSinceIso(params.now, 21),
		...typeof params.readEpoch === "number" ? { readEpoch: params.readEpoch } : {}
	}).filter((event) => {
		const metadata = objectRecord(event.metadataJson);
		if (metadataString(metadata?.taskId) === params.task.taskId) return true;
		if (stringSet(metadata?.evidenceChunkIds).some((chunkId) => chunkIdSet.has(chunkId))) return true;
		return event.sourceRef.includes(params.task.taskId);
	}).slice(0, 6).map((event) => ({
		eventId: event.eventId,
		eventType: event.eventType,
		summary: structuredEventSummary(event),
		observedAt: event.observedAt,
		sourceRef: event.sourceRef
	}));
	const supportRefs = [
		`task:${params.task.taskId}`,
		...chunkIds.map((chunkId) => `chunk:${chunkId}`),
		...linkedEvents.map((event) => `event:${event.eventId}`)
	];
	const fingerprint = computeTaskSummaryBasisFingerprint({
		taskId: params.task.taskId,
		chunkIds,
		candidateResolution,
		candidateResolutionPhase,
		candidateResolutionEvidenceChunkIds,
		compilerTaskSummary: compilerTaskSummary?.summary,
		compilerTaskSummaryConfidence: compilerTaskSummary?.confidence,
		project: canonicalMetadata.project,
		currentTask: canonicalMetadata.currentTask,
		nextAction: canonicalMetadata.nextAction,
		blocker: canonicalMetadata.blocker,
		lastEmittedOutcomeKey,
		linkedEventIds: linkedEvents.map((event) => event.eventId)
	});
	return {
		taskId: params.task.taskId,
		chunks: params.chunks,
		...compilerTaskSummary ? { compilerTaskSummary } : {},
		...candidateResolution ? { candidateResolution } : {},
		...candidateResolutionPhase ? { candidateResolutionPhase } : {},
		candidateResolutionEvidenceChunkIds,
		...canonicalMetadata.project ? { project: canonicalMetadata.project } : {},
		...canonicalMetadata.currentTask ? { currentTask: canonicalMetadata.currentTask } : {},
		...canonicalMetadata.nextAction ? { nextAction: canonicalMetadata.nextAction } : {},
		...canonicalMetadata.blocker ? { blocker: canonicalMetadata.blocker } : {},
		...lastEmittedOutcomeKey ? { lastEmittedOutcomeKey } : {},
		linkedEvents,
		supportRefs,
		fingerprint
	};
}
function resolveWorkingTaskSummary(params) {
	const compilerTaskSummary = compilerSummaryFromProposal(params.taskProposal, params.task.metadataJson);
	const compilerSummary = compilerTaskSummary && isAcceptableCompilerSummary(compilerTaskSummary.summary, compilerTaskSummary.confidence) ? compilerTaskSummary.summary : void 0;
	const compilerSummaryConfidence = compilerSummary ? compilerTaskSummary?.confidence : void 0;
	const summarySource = compilerSummary ? "compiler" : "llm_unavailable";
	const summaryBasisFingerprint = computeTaskSummaryBasisFingerprint({
		taskId: params.task.taskId,
		chunkIds: params.chunks.map((chunk) => chunk.chunkId),
		candidateResolution: typeof params.task.metadataJson?.candidateResolution === "string" ? params.task.metadataJson.candidateResolution : void 0,
		candidateResolutionPhase: typeof params.task.metadataJson?.candidateResolutionPhase === "string" ? params.task.metadataJson.candidateResolutionPhase : void 0,
		candidateResolutionEvidenceChunkIds: stringSet(objectRecord(params.task.metadataJson)?.candidateResolutionEvidenceChunkIds),
		compilerTaskSummary: compilerSummary ?? compilerTaskSummary?.summary,
		compilerTaskSummaryConfidence: compilerSummaryConfidence ?? compilerTaskSummary?.confidence,
		lastEmittedOutcomeKey: typeof params.task.metadataJson?.lastEmittedOutcomeKey === "string" ? params.task.metadataJson.lastEmittedOutcomeKey : void 0
	});
	return {
		title: params.task.title || "Active task",
		summary: compilerSummary ?? params.task.summary ?? "",
		metadataJson: { ...taskSummaryMetadataFields({
			summarySource,
			summaryQuality: "working",
			summaryBasisFingerprint,
			observedAt: params.observedAt,
			...compilerTaskSummary?.summary ? {
				compilerTaskSummary: compilerTaskSummary.summary,
				compilerTaskSummaryConfidence: compilerTaskSummary.confidence
			} : {}
		}) },
		summarySource,
		summaryQuality: "working",
		summaryBasisFingerprint,
		...compilerTaskSummary?.summary ? { compilerTaskSummary: compilerTaskSummary.summary } : {},
		...typeof compilerTaskSummary?.confidence === "number" ? { compilerTaskSummaryConfidence: compilerTaskSummary.confidence } : {}
	};
}
function taskSummaryNeedsUpgrade(params) {
	if (params.evidence.chunks.length === 0) return false;
	const metadata = objectRecord(params.task.metadataJson);
	const source = taskSummarySource(metadata);
	const quality = taskSummaryQuality(metadata);
	const currentFingerprint = metadataString(metadata?.summaryBasisFingerprint);
	const summaryUpdatedAt = metadataString(metadata?.summaryUpdatedAt) ?? params.task.updatedAt;
	const currentMs = Date.parse(params.now);
	const updatedMs = Date.parse(summaryUpdatedAt);
	const hoursSinceSummaryUpdate = Number.isFinite(currentMs) && Number.isFinite(updatedMs) ? Math.max(0, (currentMs - updatedMs) / (3600 * 1e3)) : 0;
	const hasChangedEvidence = currentFingerprint !== params.evidence.fingerprint;
	if (!(Boolean(params.evidence.candidateResolution) || params.evidence.candidateResolutionEvidenceChunkIds.length > 0 || Boolean(params.evidence.lastEmittedOutcomeKey) || params.evidence.linkedEvents.length > 0 || params.evidence.chunks.some((chunk) => chunk.role === "tool") || params.evidence.chunks.length >= 5)) return false;
	if (source === "heuristic_fallback" || source === "llm_unavailable" || !source) return true;
	if (quality !== "stable") return true;
	if (hasChangedEvidence && hoursSinceSummaryUpdate >= 1) return true;
	if (hoursSinceSummaryUpdate >= 24 && params.evidence.linkedEvents.length > 0) return true;
	return false;
}
function taskSummaryUpgradePriority(params) {
	const source = taskSummarySource(objectRecord(params.task.metadataJson));
	const toolCount = params.evidence.chunks.filter((chunk) => chunk.role === "tool").length;
	return (source === "heuristic_fallback" || source === "llm_unavailable" || !source ? 3 : 0) + (params.evidence.candidateResolution ? 3 : 0) + (params.evidence.lastEmittedOutcomeKey ? 2 : 0) + (params.evidence.linkedEvents.length > 0 ? 2 : 0) + (params.evidence.compilerTaskSummary ? 1.5 : 0) + (toolCount > 0 ? 1.5 : 0) + Math.min(params.evidence.chunks.length, 8) * .08;
}
//#endregion
export { buildTaskSummaryEvidenceSet, resolveWorkingTaskSummary, semanticTaskSummaryText, taskSummaryMetadataFields, taskSummaryNeedsUpgrade, taskSummarySource, taskSummaryUpgradePriority };
