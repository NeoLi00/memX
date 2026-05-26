import { normalizeText, nowIso, randomId, stableHash, truncateText } from "../support.mjs";
import { computeConfidence } from "./normalize.mjs";
import { writeCandidate } from "./write.mjs";
import { buildLongTurnSemanticScanInputFromSegments, frameHintsForSourceRef } from "./turnSemanticCompiler.mjs";
import { classifyAction } from "./classify.mjs";
import { evaluatePolicy } from "./policy.mjs";
//#region src/pipeline/sourceSegmentSemanticExtraction.ts
function emptyStats() {
	return {
		sourceGroupsConsidered: 0,
		sourceGroupsScanned: 0,
		candidatesWritten: 0,
		skippedReasons: []
	};
}
function sourceRefsForSegments(segments) {
	return [...new Set(segments.map((segment) => segment.parentSourceRef).filter(Boolean))];
}
const MAINTENANCE_REFERENCE_CONTEXT_MAX_TURNS = 4;
const MAINTENANCE_REFERENCE_CONTEXT_MAX_MESSAGES_PER_TURN = 4;
const MAINTENANCE_REFERENCE_SUMMARY_CHARS = 160;
const MAINTENANCE_REFERENCE_TEXT_CHARS = 360;
const SEMANTIC_WRITE_RETRY_MAX_ATTEMPTS = 6;
const SEMANTIC_WRITE_RETRY_BACKOFF_MS = [
	5 * 6e4,
	15 * 6e4,
	60 * 6e4,
	360 * 6e4
];
function fallbackTurnFrame(sourceRefs, referenceContext) {
	return {
		sourceRefs,
		...referenceContext ? { referenceContext } : {},
		chunkDrafts: [],
		assertionDrafts: [],
		correctionDrafts: [],
		relationDrafts: [],
		resourceAssertions: [],
		adviceSignals: [],
		supportSpans: [],
		compilerProvenance: {
			source: "deterministic",
			mode: "fallback",
			reasons: ["maintenance-source-segment-scaffold"]
		}
	};
}
function referenceMessageForChunk(chunk) {
	const text = (chunk.summary || chunk.content).trim();
	if (!text) return null;
	return {
		role: chunk.role,
		turnId: chunk.turnId,
		sourceRef: chunk.sourceRef,
		summary: chunk.summary ? truncateText(chunk.summary, MAINTENANCE_REFERENCE_SUMMARY_CHARS) : void 0,
		textExcerpt: truncateText(text, MAINTENANCE_REFERENCE_TEXT_CHARS)
	};
}
function buildMaintenanceReferenceContext(store, ctx, sessionKey) {
	const chunks = store.chunkRepo.listRecentActive({
		agentId: ctx.agentId,
		scopes: ctx.scopes,
		sessionKey,
		limit: MAINTENANCE_REFERENCE_CONTEXT_MAX_TURNS * MAINTENANCE_REFERENCE_CONTEXT_MAX_MESSAGES_PER_TURN * 2
	}).sort((left, right) => {
		const timeDelta = Date.parse(left.createdAt) - Date.parse(right.createdAt);
		return timeDelta !== 0 ? timeDelta : left.seq - right.seq;
	});
	const byTurn = /* @__PURE__ */ new Map();
	for (const chunk of chunks) {
		const group = byTurn.get(chunk.turnId) ?? [];
		group.push(chunk);
		byTurn.set(chunk.turnId, group);
	}
	const turns = [...byTurn.entries()].slice(-MAINTENANCE_REFERENCE_CONTEXT_MAX_TURNS).map(([turnId, turnChunks]) => ({
		turnId,
		messages: turnChunks.sort((left, right) => left.seq - right.seq).slice(0, MAINTENANCE_REFERENCE_CONTEXT_MAX_MESSAGES_PER_TURN).map((chunk) => referenceMessageForChunk(chunk)).filter((entry) => Boolean(entry))
	})).filter((turn) => turn.messages.length > 0);
	if (turns.length === 0) return;
	return {
		purpose: "deictic_reference_resolution",
		maxTurns: MAINTENANCE_REFERENCE_CONTEXT_MAX_TURNS,
		turns
	};
}
function mergeMaintenanceFrame(fallback, patch) {
	return {
		...fallback,
		...patch,
		sourceRefs: patch.sourceRefs && patch.sourceRefs.length > 0 ? patch.sourceRefs : fallback.sourceRefs,
		chunkDrafts: patch.chunkDrafts ?? [],
		taskProposal: patch.taskProposal,
		assertionDrafts: patch.assertionDrafts ?? [],
		correctionDrafts: patch.correctionDrafts ?? [],
		relationDrafts: patch.relationDrafts ?? [],
		resourceAssertions: patch.resourceAssertions ?? [],
		adviceSignals: patch.adviceSignals ?? [],
		supportSpans: patch.supportSpans ?? [],
		compilerProvenance: patch.compilerProvenance ?? {
			source: "llm",
			mode: "llm",
			reasons: ["maintenance-source-segment-semantic-extraction"]
		}
	};
}
function segmentsBySourceRef(segments) {
	const grouped = /* @__PURE__ */ new Map();
	for (const segment of segments) {
		const current = grouped.get(segment.parentSourceRef) ?? [];
		current.push(segment);
		grouped.set(segment.parentSourceRef, current);
	}
	return grouped;
}
function segmentsByTurnId(segments) {
	const grouped = /* @__PURE__ */ new Map();
	for (const segment of segments) {
		const current = grouped.get(segment.turnId) ?? [];
		current.push(segment);
		grouped.set(segment.turnId, current);
	}
	return grouped;
}
function semanticRepairInputHash(turnId, segments) {
	return stableHash([turnId, ...segments.map((segment) => [
		segment.parentSourceRef,
		String(segment.segmentIndex),
		segment.contentHash,
		normalizeText(segment.text)
	].join(""))]);
}
function uniqueTexts(texts) {
	const seen = /* @__PURE__ */ new Set();
	const result = [];
	for (const text of texts.map((entry) => entry.trim()).filter(Boolean)) {
		const key = normalizeText(text);
		if (!key || seen.has(key)) continue;
		seen.add(key);
		result.push(text);
	}
	return result;
}
function addMsToIso(baseIso, deltaMs) {
	const baseMs = Date.parse(baseIso);
	return new Date((Number.isFinite(baseMs) ? baseMs : Date.now()) + deltaMs).toISOString();
}
function semanticWriteFailureDisposition(attemptCount, failedAt) {
	if (attemptCount >= SEMANTIC_WRITE_RETRY_MAX_ATTEMPTS) return {
		status: "failed",
		maxAttempts: SEMANTIC_WRITE_RETRY_MAX_ATTEMPTS
	};
	return {
		status: "retrying",
		nextAttemptAt: addMsToIso(failedAt, SEMANTIC_WRITE_RETRY_BACKOFF_MS[Math.max(0, Math.min(attemptCount - 2, SEMANTIC_WRITE_RETRY_BACKOFF_MS.length - 1))] ?? SEMANTIC_WRITE_RETRY_BACKOFF_MS.at(-1)),
		maxAttempts: SEMANTIC_WRITE_RETRY_MAX_ATTEMPTS
	};
}
function candidateTextFromHints(hints) {
	const draft = hints.semanticDraft;
	return truncateText(uniqueTexts([
		...(draft?.supportSpans ?? []).map((entry) => entry.text),
		...(draft?.assertionDrafts ?? []).map((entry) => [
			entry.familyHint,
			entry.timeframeHint,
			...(entry.entityHints ?? []).map((entity) => entity.name),
			...entry.slotHints ?? []
		].filter(Boolean).join(" ")),
		...(draft?.correctionDrafts ?? []).map((entry) => [
			"correction",
			entry.correction.targetKind,
			entry.correction.priorValue,
			entry.correction.nextValue,
			entry.correction.reason
		].filter(Boolean).join(" ")),
		...(draft?.relationDrafts ?? []).map((entry) => [
			entry.relation.subject,
			entry.relation.predicate,
			entry.relation.object,
			entry.relation.reason
		].filter(Boolean).join(" ")),
		...(hints.resourceAssertions ?? []).map((entry) => [
			entry.owner,
			entry.ownershipStatus,
			entry.resource,
			entry.supportText,
			...entry.domains ?? [],
			...entry.affordances ?? []
		].filter(Boolean).join(" ")),
		...(hints.adviceSignals ?? []).map((entry) => [
			entry.problemContext,
			...entry.userResources ?? [],
			entry.assistantRecommendation,
			...entry.domains ?? [],
			entry.supportText
		].filter(Boolean).join(" "))
	]).join(" | "), 900);
}
function candidateForSourceRef(params) {
	const first = params.segments[0];
	if (!first) return null;
	const rawText = candidateTextFromHints(params.hints);
	if (!rawText) return null;
	return {
		candidateId: randomId("candidate"),
		source: {
			kind: first.role,
			messageId: first.turnId,
			toolName: first.toolName,
			sessionKey: first.sessionKey,
			runId: params.ctx.runId
		},
		observedAt: first.createdAt,
		rawText,
		eventType: "maintenance_source_segment_semantics",
		structuredHints: params.hints,
		metadata: {
			sourceRef: params.sourceRef,
			generatedFrom: "maintenance_source_segment_semantic_extraction",
			sourceGroupId: first.sourceGroupId,
			segmentRefs: params.segments.map((segment) => segment.segmentId),
			segmentCount: params.segments.length,
			rawContentLength: Math.max(...params.segments.map((segment) => segment.charEnd)),
			turnId: first.turnId,
			sessionKey: first.sessionKey,
			turnSemanticCompiler: params.frame.compilerProvenance,
			turnSemanticFrame: params.frame
		}
	};
}
async function runSourceSegmentSemanticExtraction(store, ctx, params) {
	const stats = emptyStats();
	if (params.turnIds.length === 0) return stats;
	const segments = store.sourceSegmentRepo.listByTurnIds({
		agentId: ctx.agentId,
		scopes: ctx.scopes,
		sessionKey: params.sessionKey,
		turnIds: params.turnIds,
		limit: 256
	});
	const grouped = segmentsBySourceRef(segments);
	const turnSegments = segmentsByTurnId(segments);
	stats.sourceGroupsConsidered = grouped.size;
	const repairStartedAt = ctx.now || nowIso();
	const attemptsByTurn = /* @__PURE__ */ new Map();
	for (const [turnId, entries] of turnSegments) {
		const attempt = store.auditRepo.recordSemanticWriteAttemptStart({
			agentId: ctx.agentId,
			sessionKey: params.sessionKey,
			scope: entries[0]?.scope ?? ctx.scopes[0] ?? `agent:${ctx.agentId}`,
			turnId,
			sourceRefs: sourceRefsForSegments(entries),
			inputHash: semanticRepairInputHash(turnId, entries),
			startedAt: repairStartedAt,
			retryOnly: true
		});
		if (attempt.claimed) attemptsByTurn.set(turnId, { attemptCount: attempt.attemptCount });
	}
	const compileLongTurnSemantics = store.reasoner.compileLongTurnSemantics?.bind(store.reasoner);
	if (!store.reasoner.isEnabled?.() || !compileLongTurnSemantics) {
		stats.skippedReasons.push("llm-unavailable");
		for (const turnId of attemptsByTurn.keys()) {
			const completedAt = ctx.now || nowIso();
			const attemptCount = attemptsByTurn.get(turnId)?.attemptCount ?? 1;
			const disposition = semanticWriteFailureDisposition(attemptCount, completedAt);
			store.auditRepo.finishSemanticWriteAttempt({
				agentId: ctx.agentId,
				sessionKey: params.sessionKey,
				turnId,
				status: disposition.status,
				error: disposition.status === "failed" ? `maintenance semantic scanner unavailable: llm-unavailable after ${attemptCount} attempts` : "maintenance semantic scanner unavailable: llm-unavailable",
				resultJson: {
					stage: "maintenance_async",
					skippedReasons: stats.skippedReasons,
					retry: {
						attemptCount,
						maxAttempts: disposition.maxAttempts,
						nextAttemptAt: disposition.nextAttemptAt
					}
				},
				nextAttemptAt: disposition.nextAttemptAt,
				completedAt
			});
		}
		return stats;
	}
	const referenceContext = buildMaintenanceReferenceContext(store, ctx, params.sessionKey);
	const scanInput = buildLongTurnSemanticScanInputFromSegments(segments, referenceContext);
	stats.sourceGroupsScanned = scanInput.messages.length;
	if (scanInput.messages.length === 0) return stats;
	const fallback = fallbackTurnFrame(sourceRefsForSegments(segments), referenceContext);
	const patch = await compileLongTurnSemantics(scanInput, fallback, {
		stage: "maintenance_async",
		audit: ctx.llmBudgetAudit
	});
	if (!patch) {
		stats.skippedReasons.push("llm-empty");
		for (const turnId of attemptsByTurn.keys()) {
			const completedAt = ctx.now || nowIso();
			const attemptCount = attemptsByTurn.get(turnId)?.attemptCount ?? 1;
			const disposition = semanticWriteFailureDisposition(attemptCount, completedAt);
			store.auditRepo.finishSemanticWriteAttempt({
				agentId: ctx.agentId,
				sessionKey: params.sessionKey,
				turnId,
				status: disposition.status,
				error: disposition.status === "failed" ? `maintenance semantic scanner returned no recognized LLM semantic frame after ${attemptCount} attempts` : "maintenance semantic scanner returned no recognized LLM semantic frame",
				resultJson: {
					stage: "maintenance_async",
					skippedReasons: stats.skippedReasons,
					retry: {
						attemptCount,
						maxAttempts: disposition.maxAttempts,
						nextAttemptAt: disposition.nextAttemptAt
					}
				},
				nextAttemptAt: disposition.nextAttemptAt,
				completedAt
			});
		}
		return stats;
	}
	const frame = mergeMaintenanceFrame(fallback, patch);
	for (const sourceRef of frame.sourceRefs) {
		const hints = frameHintsForSourceRef(frame, sourceRef);
		const sourceSegments = grouped.get(sourceRef);
		if (!hints || !sourceSegments || sourceSegments.length === 0) continue;
		const candidate = candidateForSourceRef({
			sourceRef,
			segments: sourceSegments,
			hints,
			frame,
			ctx
		});
		if (!candidate) {
			stats.skippedReasons.push(`empty-candidate:${sourceRef}`);
			continue;
		}
		const policyResult = await evaluatePolicy(candidate, ctx, { reasoner: store.reasoner });
		const classification = classifyAction(policyResult.decision.action);
		if (classification === "ignore") {
			stats.skippedReasons.push(`policy-ignore:${sourceRef}`);
			continue;
		}
		const classified = {
			...policyResult.candidate,
			normalizedText: normalizeText(policyResult.candidate.rawText),
			scope: sourceSegments[0]?.scope ?? ctx.scopes[0] ?? `agent:${ctx.agentId}`,
			policy: policyResult.decision,
			classification,
			confidence: 0
		};
		classified.confidence = computeConfidence(classified);
		writeCandidate(store, ctx, classified);
		stats.candidatesWritten += 1;
	}
	for (const turnId of attemptsByTurn.keys()) store.auditRepo.finishSemanticWriteAttempt({
		agentId: ctx.agentId,
		sessionKey: params.sessionKey,
		turnId,
		status: "succeeded",
		resultJson: {
			stage: "maintenance_async",
			compilerProvenance: frame.compilerProvenance,
			sourceGroupsScanned: stats.sourceGroupsScanned,
			candidatesWritten: stats.candidatesWritten,
			skippedReasons: stats.skippedReasons
		},
		completedAt: nowIso()
	});
	return stats;
}
//#endregion
export { runSourceSegmentSemanticExtraction };
