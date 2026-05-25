import { clamp01, normalizeText, objectRecord, randomId, stableHash, truncateText } from "../support.mjs";
import { snapshotMemoryLlmBudgetAudit } from "./llmBudgetAudit.mjs";
import { compileQuery, compileQueryWithoutSemanticFallback } from "./queryCompiler.mjs";
import { isQuestionLike, queryAnchorSupport } from "./semantic/heuristics.mjs";
import { projectNamesMatch } from "./projectIdentity.mjs";
import { isSnapshotFactualStateKey } from "./authority.mjs";
import { filterBootstrapRows } from "./bootstrapFilter.mjs";
import { semanticTextSimilarity } from "./semantic/textSimilarity.mjs";
import { capScoreByEvidenceCoverage, evidenceCoverageForText } from "./evidenceCoverage.mjs";
import { sourceRefsFromMaintenanceMetadata, uniqueMaintenanceRefs } from "./maintenanceContract.mjs";
import "./semantics.mjs";
import { dedupeEvidenceRows, formatFactLine, lineageFromMetadata, normalizeSearchText, rowsFromSearchHits, shouldSuppressRecallText, splitLabelValue, toEvidenceRow } from "./memoryObjectsHelpers.mjs";
import { isAnswerPromptLineRole, normalizeSourceRefs, promptLineRole } from "./sourceRefs.mjs";
import { candidateGenerationAuditPayload, generateCandidates } from "./candidateGeneration.mjs";
import { assembleEvidencePackets } from "./evidenceAssembler.mjs";
import { createMemorySelectionObjective, projectScheduledMemoryObjects } from "./memoryObjectsProjection.mjs";
import { semanticTaskSummaryText } from "./taskSummary.mjs";
import { buildBackgroundRecallBundle as buildBackgroundRecallBundle$1, collectAndScheduleMemoryObjectsWithBudget, collectBehavioralGuidance, collectMemoryObjects, queryMemoryFacts, scheduleMemoryObjects, toRouteEvidenceCandidatesFromObjects } from "./memoryObjects.mjs";
import { buildRecallAuditPayload, compareEvidenceRowsChronologically, sanitizeFocusedRecallQuery } from "./retrieveTracing.mjs";
import { emitContradictionSignals, emitFullRetrievalSignals } from "./signalLedger.mjs";
import "./sourceSegments.mjs";
//#region src/pipeline/retrieve.ts
const PRIMARY_ROUTE_TYPES = [
	"workflow",
	"factual",
	"temporal",
	"explanatory"
];
function analyzeRecallQuery(query) {
	return compileQueryWithoutSemanticFallback(query, "llm-only-recall-analysis-unavailable");
}
function buildBackgroundRecallBundle(store, ctx) {
	return buildBackgroundRecallBundle$1(store, ctx);
}
function hasBackgroundRecallMaterial(bundle) {
	return bundle.projectionBlocks.length > 0 || bundle.behavioralGuidance.length > 0;
}
function taskMetadataValue(task, key) {
	const value = task.metadataJson?.[key];
	return typeof value === "string" && value.trim() ? value.trim() : "";
}
function taskProject(task) {
	return taskMetadataValue(task, "project");
}
function uniqueNonEmpty(values, limit = Number.POSITIVE_INFINITY) {
	const seen = /* @__PURE__ */ new Set();
	const ordered = [];
	for (const value of values) {
		const trimmed = value.trim();
		if (!trimmed) continue;
		const normalized = normalizeText(trimmed);
		if (!normalized || seen.has(normalized)) continue;
		seen.add(normalized);
		ordered.push(trimmed);
		if (ordered.length >= limit) break;
	}
	return ordered;
}
function shouldUseExactSnippetSupport(queryAnalysis) {
	return queryAnalysis.evidenceFidelity === "high" || queryAnalysis.queryShape.timeframe === "compare" || queryAnalysis.queryShape.granularity === "exact_detail" || queryAnalysis.queryShape.evidenceNeed !== "workflow_context" && queryAnalysis.candidateSurfaces.includes("chunk");
}
function factStatusFromMetadata(metadata) {
	return metadata?.status === "active" || metadata?.status === "superseded" || metadata?.status === "deleted" || metadata?.status === "uncertain" ? metadata.status : void 0;
}
function isHistoricalFactQuery(queryAnalysis) {
	return queryAnalysis.queryShape.timeframe === "historical" || queryAnalysis.queryShape.timeframe === "compare";
}
function factCandidateVisibleForQuery(queryAnalysis, metadata) {
	const status = factStatusFromMetadata(metadata);
	if (!status) return true;
	if (status === "deleted") return false;
	if (status === "superseded" && !isHistoricalFactQuery(queryAnalysis)) return false;
	return true;
}
function evidenceRowFromCandidateHit(hit, store) {
	const metadata = store ? hydrateFactCandidateMetadata(store, hit) : hit.metadata;
	const observedAt = typeof metadata?.observedAt === "string" ? metadata.observedAt : void 0;
	const status = factStatusFromMetadata(metadata);
	const text = hit.surface === "fact" && typeof metadata?.canonicalSubject === "string" && typeof metadata?.predicate === "string" ? formatFactLine({
		subject: metadata.canonicalSubject,
		predicate: metadata.predicate,
		object: typeof metadata.canonicalObject === "string" ? metadata.canonicalObject : void 0,
		status: status === "superseded" || hit.supersededHint ? "superseded" : void 0
	}) : normalizeSearchText(hit.text);
	const sourceRef = typeof metadata?.sourceRef === "string" && metadata.sourceRef.trim() ? metadata.sourceRef.trim() : hit.lineage.sourceRef;
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
			sourceRef
		}
	});
}
function factIdFromCandidateHit(hit) {
	if (hit.surface !== "fact") return;
	const candidates = [
		hit.lineage.canonicalId,
		hit.lineage.sourceId,
		hit.docId,
		hit.candidateId
	];
	for (const candidate of candidates) {
		if (!candidate) continue;
		return candidate.startsWith("fact:") ? candidate.slice(5) : candidate;
	}
}
function semanticAssertionMetadata(fact) {
	return objectRecord(fact.objectValueJson?.semanticAssertion);
}
function factSupportSourceRefs(fact) {
	const assertion = semanticAssertionMetadata(fact);
	return uniqueMaintenanceRefs([
		fact.sourceRef,
		...sourceRefsFromMaintenanceMetadata(fact.objectValueJson),
		typeof assertion?.sourceRef === "string" ? assertion.sourceRef : void 0
	]);
}
function factSupportText(fact) {
	const assertion = semanticAssertionMetadata(fact);
	const supportText = typeof assertion?.supportText === "string" ? assertion.supportText.trim() : "";
	if (supportText) return supportText;
	return fact.provenanceText?.trim() || void 0;
}
function hydrateFactCandidateMetadata(store, hit) {
	if (hit.surface !== "fact") return hit.metadata;
	const factId = factIdFromCandidateHit(hit);
	const fact = factId ? store.factRepo.get(factId) : null;
	if (!fact) return hit.metadata;
	const sourceRefs = factSupportSourceRefs(fact);
	const supportText = factSupportText(fact);
	return {
		...hit.metadata ?? {},
		canonicalSubject: fact.canonicalSubject,
		predicate: fact.predicate,
		...fact.canonicalObject ? { canonicalObject: fact.canonicalObject } : {},
		status: fact.status,
		activeHint: fact.status === "active" || fact.status === "uncertain",
		supersededHint: fact.status === "superseded",
		currentnessHint: fact.status === "superseded" ? "historical" : "current",
		observedAt: fact.updatedAt,
		confidence: fact.confidence,
		...sourceRefs[0] ? { sourceRef: sourceRefs[0] } : {},
		...sourceRefs.length > 0 ? {
			sourceRefs,
			supportRefs: sourceRefs,
			supportContentRefs: sourceRefs
		} : {},
		...supportText ? { supportText } : {}
	};
}
function candidateDisplayText(hit, metadata) {
	const status = factStatusFromMetadata(metadata);
	if (hit.surface === "fact" && typeof metadata?.canonicalSubject === "string" && typeof metadata?.predicate === "string") return formatFactLine({
		subject: metadata.canonicalSubject,
		predicate: metadata.predicate,
		object: typeof metadata.canonicalObject === "string" ? metadata.canonicalObject : void 0,
		status: status === "superseded" || hit.supersededHint ? "superseded" : void 0
	});
	return hit.text;
}
function evidenceRowFromFactId(store, factId) {
	const fact = store.factRepo.get(factId);
	if (!fact) return null;
	const sourceRef = factSupportSourceRefs(fact)[0] ?? fact.sourceRef;
	return toEvidenceRow({
		id: fact.factId,
		text: formatFactLine({
			subject: fact.canonicalSubject,
			predicate: fact.predicate,
			object: fact.canonicalObject ?? void 0,
			objectValueJson: fact.objectValueJson,
			status: fact.status
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
			materializedEpoch: fact.materializedEpoch
		}
	});
}
function primaryCandidateRowsForSurface(result, surface, options) {
	return result.candidates.filter((candidate) => candidate.surface === surface && candidate.tier === "primary").map((candidate) => ({
		candidate,
		metadata: options?.store ? hydrateFactCandidateMetadata(options.store, candidate) : candidate.metadata
	})).filter(({ candidate, metadata }) => candidate.surface !== "fact" || !options || factCandidateVisibleForQuery(options.queryAnalysis, metadata)).map(({ candidate }) => evidenceRowFromCandidateHit(candidate, options?.store));
}
function candidateSurfaceAnchorScore(row, queryAnchors) {
	if (queryAnchors.length === 0) return 0;
	return queryAnchorSupport(row.text, queryAnchors);
}
function prioritizeCandidateRowsForMainSurface(rows, queryAnchors, limit) {
	return dedupeEvidenceRows(rows.slice().sort((left, right) => {
		const rightAnchor = candidateSurfaceAnchorScore(right, queryAnchors);
		const leftAnchor = candidateSurfaceAnchorScore(left, queryAnchors);
		if (rightAnchor !== leftAnchor) return rightAnchor - leftAnchor;
		return (right.score ?? 0) - (left.score ?? 0);
	}), limit);
}
function selectComplementaryFactRowsForMainSurface(params) {
	if (!(params.queryAnalysis.answerGranularity === "detail" || params.queryAnalysis.evidenceFidelity === "high") || params.queryAnalysis.queryShape.timeframe !== "current") return [];
	const selectedIds = new Set(params.selectedFacts.map((entry) => entry.id));
	const seed = params.selectedFacts.map((row) => {
		const fact = params.store.factRepo.get(row.id);
		if (!fact || fact.predicate !== "reported_detail") return null;
		if (fact.objectValueJson?.answerBearing !== true || fact.objectValueJson?.currentnessHint !== "current") return null;
		return {
			row,
			fact,
			anchorSupport: candidateSurfaceAnchorScore(row, params.queryAnchors)
		};
	}).filter((entry) => Boolean(entry)).sort((left, right) => {
		return right.anchorSupport * .7 + (right.row.score ?? 0) * .3 - (left.anchorSupport * .7 + (left.row.score ?? 0) * .3);
	})[0];
	if (!seed) return [];
	const reserveFactRows = (params.reserveFactTrace ?? []).map((entry) => {
		const factId = entry.objectId.startsWith("fact:") ? entry.objectId.slice(5) : entry.objectId;
		return evidenceRowFromFactId(params.store, factId);
	}).filter((entry) => Boolean(entry));
	return dedupeEvidenceRows([...params.fallbackFacts, ...reserveFactRows], Math.max(6, params.limit * 6)).filter((row) => !selectedIds.has(row.id)).map((row) => {
		const fact = params.store.factRepo.get(row.id);
		if (!fact || fact.predicate !== seed.fact.predicate) return null;
		if (fact.canonicalSubject !== seed.fact.canonicalSubject) return null;
		if (fact.objectValueJson?.answerBearing !== true || fact.objectValueJson?.currentnessHint !== "current") return null;
		const similarity = semanticTextSimilarity(seed.row.text, row.text);
		const anchorSupport = candidateSurfaceAnchorScore(row, params.queryAnchors);
		if (similarity < .42) return null;
		return {
			row,
			score: similarity * .75 + anchorSupport * .25
		};
	}).filter((entry) => Boolean(entry)).sort((left, right) => right.score - left.score).slice(0, params.limit).map((entry) => entry.row);
}
function preferredProjectedExactSnippets(params) {
	return params.projectedIds.map((snippetId, index) => ({
		snippetId,
		text: params.projectedTexts[index] ?? "",
		lineage: {
			sourceKind: "vector_doc",
			sourceId: snippetId
		},
		source: "projected",
		anchorSupport: params.queryAnchors.length > 0 ? queryAnchorSupport(params.projectedTexts[index] ?? "", params.queryAnchors) : 1
	})).filter((entry) => entry.text.trim().length > 0).filter((entry) => params.queryAnchors.length > 0 ? entry.anchorSupport >= .32 : true).slice(0, params.limit);
}
function buildSourceRefFallbackSnippets(params) {
	const chunks = params.store.chunkRepo.listBySourceRefs({
		agentId: params.ctx.agentId,
		scopes: params.ctx.scopes,
		sourceRefs: params.sourceRefs,
		limit: Math.max(params.limit * 2, 6)
	});
	const seen = /* @__PURE__ */ new Set();
	return chunks.map((chunk) => ({
		chunk,
		anchorScore: params.queryAnchors && params.queryAnchors.length > 0 ? queryAnchorSupport(chunk.content, params.queryAnchors) : 1
	})).filter((entry) => {
		if (seen.has(entry.chunk.chunkId)) return false;
		seen.add(entry.chunk.chunkId);
		return params.queryAnchors && params.queryAnchors.length > 0 ? entry.anchorScore >= .22 : true;
	}).sort((left, right) => {
		if (right.anchorScore !== left.anchorScore) return right.anchorScore - left.anchorScore;
		return right.chunk.createdAt.localeCompare(left.chunk.createdAt);
	}).slice(0, params.limit).map(({ chunk }) => ({
		snippetId: `event:chunk:${chunk.chunkId}`,
		text: truncateText(chunk.content.trim() || chunk.summary.trim(), 280),
		sourceRef: chunk.sourceRef,
		lineage: {
			sourceKind: "chunk",
			sourceId: chunk.chunkId,
			sourceRef: chunk.sourceRef
		},
		source: "fallback"
	}));
}
function rawScoreTextAgainstEvidenceGoals(params) {
	const goals = params.queryAnalysis.evidenceGoals ?? [];
	const anchorScore = params.queryAnalysis.anchors.length > 0 ? informativePromptAnchorSupport(params.text, params.queryAnalysis.anchors) : 0;
	const goalScore = goals.reduce((best, goal) => {
		const positiveScore = Math.max(semanticTextSimilarity(goal.goal, params.text), ...goal.positiveQueries.map((query) => semanticTextSimilarity(query, params.text)));
		const focusScore = goal.focusAnchors.length > 0 ? informativePromptAnchorSupport(params.text, goal.focusAnchors) : 0;
		const surfaceBoost = params.surface && goal.preferredSurfaces.includes(params.surface) ? .08 : 0;
		const negativePenalty = Math.max(0, ...(goal.negativeHints ?? []).map((hint) => semanticTextSimilarity(hint, params.text) * .12));
		return Math.max(best, clamp01(positiveScore * .68 + focusScore * .18 + surfaceBoost - negativePenalty));
	}, 0);
	const explicitGoalScore = params.explicitGoalScore ?? 0;
	const explicitSupport = Math.max(evidenceGoalQuerySemanticSupport(params.queryAnalysis, params.text), evidenceGoalFocusSupport(params.queryAnalysis, params.text), anchorScore);
	const tailorAdvice = params.queryAnalysis.evidencePlan?.operation.type === "tailor_advice";
	const adjustedExplicitGoalScore = tailorAdvice ? Math.min(explicitGoalScore, explicitSupport + .18) : explicitGoalScore > 0 && explicitSupport < .12 ? Math.min(explicitGoalScore, .42) : explicitGoalScore;
	const adjustedGoalScore = explicitSupport < .12 ? Math.min(goalScore, .42) : tailorAdvice ? Math.min(goalScore, explicitSupport + .18) : goalScore;
	return clamp01(Math.max(adjustedExplicitGoalScore, adjustedGoalScore, anchorScore * .72));
}
function scoreTextAgainstEvidenceGoals(params) {
	return capScoreByEvidenceCoverage(rawScoreTextAgainstEvidenceGoals(params), evidenceCoverageForText(params.queryAnalysis, params.text));
}
function topCandidateGoalScore(hit) {
	return Math.max(0, ...(hit.goalMatches ?? []).map((match) => match.score), ...(hit.bridgeMatches ?? []).map((match) => match.score));
}
function rankExactSnippetCandidates(candidates, queryAnalysis) {
	const byId = /* @__PURE__ */ new Map();
	for (const candidate of candidates) {
		const goalScore = scoreTextAgainstEvidenceGoals({
			text: candidate.text,
			queryAnalysis,
			surface: "snippet"
		});
		const scored = {
			...candidate,
			goalScore
		};
		const existing = byId.get(candidate.snippetId);
		if (!existing || (scored.goalScore ?? 0) > (existing.goalScore ?? 0)) byId.set(candidate.snippetId, scored);
	}
	return [...byId.values()].sort((left, right) => {
		const scoreDelta = (right.goalScore ?? 0) - (left.goalScore ?? 0);
		if (scoreDelta !== 0) return scoreDelta;
		const sourceRank = {
			projected: 3,
			support_ref: 2,
			fallback: 1
		};
		return sourceRank[right.source] - sourceRank[left.source];
	});
}
function candidateSurfaceToPromptSurface(surface) {
	if (surface === "fact" || surface === "event" || surface === "chunk") return surface;
	if (surface === "state" || surface === "graph") return "fact";
	if (surface === "task") return "snippet";
	return null;
}
function preferredSurfaceSupport(queryAnalysis, surface) {
	const mappedSurface = surface === "snippet" ? "chunk" : surface;
	return (queryAnalysis.evidenceGoals ?? []).some((goal) => goal.preferredSurfaces.includes(mappedSurface)) ? .08 : 0;
}
function evidenceGoalFocusSupport(queryAnalysis, text) {
	const anchors = uniqueNonEmpty((queryAnalysis.evidenceGoals ?? []).flatMap((goal) => goal.focusAnchors));
	if (anchors.length === 0) return queryAnalysis.anchors.length > 0 ? informativePromptAnchorSupport(text, queryAnalysis.anchors) : 0;
	return informativePromptAnchorSupport(text, anchors);
}
function informativePromptAnchors(anchors) {
	return uniqueNonEmpty(anchors.filter((anchor) => normalizeText(anchor).length > 0), 12);
}
function informativePromptAnchorSupport(text, anchors) {
	const usefulAnchors = informativePromptAnchors(anchors);
	if (usefulAnchors.length === 0) return 0;
	const normalizedText = normalizeText(text);
	let best = 0;
	for (const anchor of usefulAnchors) {
		const normalizedAnchor = normalizeText(anchor);
		if (!normalizedAnchor) continue;
		const containment = normalizedAnchor.length >= 4 && normalizedText.includes(normalizedAnchor) ? .94 : 0;
		best = Math.max(best, containment, semanticTextSimilarity(anchor, text));
	}
	return clamp01(best);
}
function evidenceGoalQuerySemanticSupport(queryAnalysis, text) {
	let best = 0;
	for (const goal of queryAnalysis.evidenceGoals ?? []) for (const query of [goal.goal, ...goal.positiveQueries]) {
		if (!normalizeText(query)) continue;
		best = Math.max(best, semanticTextSimilarity(query, text));
	}
	return clamp01(best);
}
function promptEvidenceExcerptAnchors(queryAnalysis) {
	return informativePromptAnchors([
		...queryAnalysis.anchors,
		...(queryAnalysis.evidenceGoals ?? []).flatMap((goal) => goal.focusAnchors),
		...(queryAnalysis.evidencePlan?.slots ?? []).flatMap((slot) => slot.subjectHints),
		...(queryAnalysis.evidencePlan?.slots ?? []).flatMap((slot) => slot.relationHints ?? []),
		...(queryAnalysis.evidencePlan?.slots ?? []).flatMap((slot) => slot.capabilityQueries ?? [])
	]);
}
function evidencePlanLayersForSlot(slot) {
	return [...new Set([...slot.preferredLayers, ...slot.fallbackLayers])];
}
function evidenceSlotRequiredRole(slot) {
	if (!slot) return;
	if (slot.requiredRole) return slot.requiredRole;
	if (slot.role === "query_context" || slot.role === "user_resource" || slot.role === "prior_advice" || slot.role === "answer_value" || slot.role === "answer_event" || slot.role === "time_constraint") return slot.role;
	if (slot.role === "answer_evidence") return "answer_value";
}
function effectiveSlotHints(hints) {
	return uniqueNonEmpty(hints.filter((hint) => normalizeText(hint).length > 0), 8);
}
function promptMetadataStringArray(metadata, key) {
	const value = metadata?.[key];
	return Array.isArray(value) ? value.filter((entry) => typeof entry === "string" && entry.trim().length > 0) : [];
}
function promptEvidenceScoringTextFromMetadata(text, metadata) {
	if (!metadata) return text;
	return uniqueNonEmpty([
		text,
		typeof metadata.canonicalSubject === "string" ? metadata.canonicalSubject : void 0,
		typeof metadata.predicate === "string" ? metadata.predicate : void 0,
		typeof metadata.stateKey === "string" ? metadata.stateKey : void 0,
		typeof metadata.resource === "string" ? metadata.resource : void 0,
		typeof metadata.canonicalObject === "string" ? metadata.canonicalObject : void 0,
		typeof metadata.resourceType === "string" ? `resource type: ${metadata.resourceType}` : void 0,
		promptMetadataStringArray(metadata, "domains").length > 0 ? `domains: ${promptMetadataStringArray(metadata, "domains").join(", ")}` : void 0,
		promptMetadataStringArray(metadata, "affordances").length > 0 ? `affordances: ${promptMetadataStringArray(metadata, "affordances").join(", ")}` : void 0,
		typeof metadata.supportText === "string" ? `evidence: ${metadata.supportText}` : void 0,
		metadata.signalKind === "resourceAssertion" ? "resource assertion" : void 0
	].filter((part) => Boolean(part)), 16).join(" | ");
}
function promptEvidenceScoringText(candidate) {
	return uniqueNonEmpty([
		candidate.text,
		candidate.scoringText,
		candidate.rawText
	].filter((part) => typeof part === "string" && part.trim().length > 0), 6).join(" | ");
}
function slotSubjectFillThreshold(queryAnalysis) {
	const operationType = queryAnalysis.evidencePlan?.operation.type;
	if (operationType === "tailor_advice") return .52;
	return (queryAnalysis.evidencePlan?.slots.length ?? 0) > 1 || operationType === "derive" || operationType === "aggregate" || operationType === "relate" ? .52 : .42;
}
function semanticSlotFillThreshold(queryAnalysis) {
	const operationType = queryAnalysis.evidencePlan?.operation.type;
	if (operationType === "derive" || operationType === "compare") return .66;
	if (operationType === "aggregate" || queryAnalysis.answerMode === "count_aggregate") return .6;
	if (queryAnalysis.answerMode === "attribute_lookup") return .58;
	return queryAnalysis.evidenceFidelity === "high" || queryAnalysis.answerGranularity === "detail" ? .56 : .52;
}
function slotFieldSupport(text, fields) {
	if (fields.length === 0) return {
		hits: [],
		missing: [],
		score: 1
	};
	const normalized = normalizeText(text);
	const hits = [];
	const missing = [];
	let concreteFieldCount = 0;
	for (const field of fields) {
		const key = normalizeText(field);
		if (isGenericRequiredField(key)) continue;
		concreteFieldCount += 1;
		const textKey = key.replace(/_/gu, " ");
		if (textKey.length >= 3 && normalizedTextContainsExactPhrase(normalized, textKey)) hits.push(field);
		else missing.push(field);
	}
	return {
		hits,
		missing,
		score: concreteFieldCount > 0 ? clamp01(hits.length / concreteFieldCount) : fields.length > 0 ? .15 : 1
	};
}
function normalizedTextContainsExactPhrase(normalizedText, normalizedPhrase) {
	const escaped = normalizedPhrase.split(/\s+/u).filter(Boolean).map((part) => part.replace(/[.*+?^${}()|[\]\\]/gu, "\\$&")).join("\\s+");
	if (!escaped) return false;
	return new RegExp(`(^|[^\\p{L}\\p{N}])${escaped}($|[^\\p{L}\\p{N}])`, "u").test(normalizedText);
}
function promptEvidenceNonSubjectSlotHitCount(queryAnalysis, candidate) {
	const slotById = new Map((queryAnalysis.evidencePlan?.slots ?? []).map((slot) => [slot.id, slot]));
	const queryAnchors = queryAnalysis.anchors ?? [];
	const focusAnchors = new Set([...queryAnchors, ...(queryAnalysis.evidenceGoals ?? []).flatMap((goal) => goal.focusAnchors)].map((anchor) => normalizeText(anchor)).filter(Boolean));
	const hits = /* @__PURE__ */ new Set();
	const slotCoverage = candidate.slotCoverage ?? slotCoverageForText(queryAnalysis, promptEvidenceScoringText(candidate));
	for (const coverage of slotCoverage) {
		if (!coverage.filled) continue;
		const slot = slotById.get(coverage.slotId);
		const subjectHints = new Set([...slot?.subjectHints ?? [], ...focusAnchors].map((hint) => normalizeText(hint)).filter(Boolean));
		for (const hit of coverage.requiredHits) {
			const normalizedHit = normalizeText(hit);
			if (!normalizedHit || isGenericRequiredField(normalizedHit)) continue;
			if (!subjectHints.has(normalizedHit)) hits.add(normalizedHit);
		}
	}
	return hits.size;
}
function promptEvidenceRequiresAnswerSlotSupport(queryAnalysis) {
	const operationType = queryAnalysis.evidencePlan?.operation.type;
	const answerMode = queryAnalysis.answerMode ?? "single_fact";
	return operationType === "return_value" || answerMode === "single_fact" || answerMode === "attribute_lookup";
}
function isGenericRequiredField(key) {
	const normalizedKey = normalizeText(key).replace(/[\s-]+/g, "_");
	return [
		"answer_value",
		"attribute_value",
		"value",
		"countable_item",
		"temporal_marker",
		"preference_or_prior_action",
		"source_evidence",
		"query_context"
	].includes(normalizedKey);
}
function slotCoverageForText(queryAnalysis, text) {
	return (queryAnalysis.evidencePlan?.slots ?? []).map((slot) => {
		const isTailorAdvice = queryAnalysis.evidencePlan?.operation.type === "tailor_advice";
		const slotRole = evidenceSlotRequiredRole(slot);
		const isTailorAdviceMemory = isTailorAdvice && (slotRole ? slotRole !== "query_context" : slot.id !== "current_need");
		const subjectHints = effectiveSlotHints(slot.subjectHints);
		const relationHints = effectiveSlotHints([...slot.relationHints ?? [], ...slot.capabilityQueries ?? []]);
		const subjectScore = subjectHints.length > 0 ? informativePromptAnchorSupport(text, subjectHints) : slot.subjectHints.length > 0 ? 0 : .5;
		const relationScore = relationHints.length > 0 ? informativePromptAnchorSupport(text, relationHints) : (slot.relationHints ?? []).length > 0 ? 0 : .35;
		const field = slotFieldSupport(text, slot.requiredFields);
		const querySimilarity = semanticTextSimilarity(queryAnalysis.focusedQuery || queryAnalysis.queryText, text);
		const semanticSupport = Math.max(querySimilarity, rawScoreTextAgainstEvidenceGoals({
			text,
			queryAnalysis
		}));
		const subjectThreshold = slotSubjectFillThreshold(queryAnalysis);
		const subjectHits = subjectHints.filter((hint) => informativePromptAnchorSupport(text, [hint]) >= subjectThreshold);
		const relationHits = relationHints.filter((hint) => informativePromptAnchorSupport(text, [hint]) >= .52);
		const requiredHits = [
			...subjectHits,
			...relationHits,
			...field.hits
		];
		const fieldHitKeys = new Set(field.hits.map((hit) => normalizeText(hit)));
		const structuralHits = requiredHits.filter((hit) => !fieldHitKeys.has(normalizeText(hit)));
		const coverageScore = isTailorAdviceMemory ? clamp01(subjectScore * .22 + relationScore * .46 + field.score * .16 + semanticSupport * .16) : clamp01(subjectScore * .58 + relationScore * .18 + field.score * .24);
		const semanticStructuralSupport = subjectScore >= Math.max(.24, subjectThreshold - .18) || relationScore >= .32 || structuralHits.length > 0 || coverageScore >= .34;
		const semanticFilled = semanticSupport >= semanticSlotFillThreshold(queryAnalysis) && semanticStructuralSupport;
		const adviceFilled = isTailorAdvice && field.missing.length === 0 && structuralHits.length > 0 && (subjectScore >= .34 || relationScore >= .42 || querySimilarity >= .38 && (subjectScore >= .24 || relationScore >= .28));
		const filled = isTailorAdviceMemory && field.missing.length === 0 && (relationScore >= .36 || semanticSupport >= .4 || coverageScore >= .34 || relationHits.length > 0 && Math.max(relationScore, semanticSupport) >= .24) || adviceFilled || semanticFilled || coverageScore >= .56 && (subjectHints.length === 0 || subjectScore >= subjectThreshold) && field.missing.length === 0;
		const missingRequired = filled ? [] : [...isTailorAdviceMemory ? [] : subjectHints.filter((hint) => informativePromptAnchorSupport(text, [hint]) < subjectThreshold), ...field.missing];
		return {
			slotId: slot.id,
			requiredHits: uniqueNonEmpty(requiredHits),
			missingRequired: uniqueNonEmpty(missingRequired),
			coverageScore,
			filled
		};
	});
}
function filledSlotIdsForText(queryAnalysis, text) {
	return slotCoverageForText(queryAnalysis, text).filter((coverage) => coverage.filled).map((coverage) => coverage.slotId);
}
function isTailorAdviceMemorySlot(queryAnalysis, slotId) {
	const requiredRole = evidenceSlotRequiredRole((queryAnalysis.evidencePlan?.slots ?? []).find((entry) => entry.id === slotId));
	return queryAnalysis.evidencePlan?.operation.type === "tailor_advice" && (requiredRole ? requiredRole !== "query_context" : slotId !== "current_need");
}
function isTailorAdviceResourceSlot(queryAnalysis, slotId) {
	const requiredRole = evidenceSlotRequiredRole((queryAnalysis.evidencePlan?.slots ?? []).find((entry) => entry.id === slotId));
	return queryAnalysis.evidencePlan?.operation.type === "tailor_advice" && (requiredRole === "user_resource" || slotId === "relevant_user_resources");
}
function tailorAdviceQueryContextText(queryAnalysis) {
	const currentNeedSlot = (queryAnalysis.evidencePlan?.slots ?? []).find((slot) => slot.id === "current_need");
	return normalizeText([
		queryAnalysis.queryText,
		queryAnalysis.focusedQuery,
		...queryAnalysis.anchors,
		...currentNeedSlot?.subjectHints ?? [],
		...currentNeedSlot?.relationHints ?? []
	].join(" "));
}
function isQueryContextHit(queryAnalysis, hit) {
	const normalizedHit = normalizeText(hit);
	if (!normalizedHit || isGenericRequiredField(normalizedHit)) return true;
	const queryContext = tailorAdviceQueryContextText(queryAnalysis);
	return normalizedTextContainsExactPhrase(queryContext, normalizedHit) || semanticTextSimilarity(normalizedHit, queryContext) >= .82;
}
function slotSemanticSupport(queryAnalysis, slotId, text) {
	const slot = (queryAnalysis.evidencePlan?.slots ?? []).find((entry) => entry.id === slotId);
	if (!slot) return 0;
	const slotQueries = uniqueNonEmpty([
		slot.description,
		...slot.relationHints ?? [],
		...slot.capabilityQueries ?? [],
		...slot.requiredFields.filter((field) => !isGenericRequiredField(normalizeText(field)))
	], 16);
	return clamp01(Math.max(0, ...slotQueries.map((query) => semanticTextSimilarity(query, text))));
}
function pruneTailorAdviceMemorySlotCoverage(params) {
	if (!isTailorAdviceMemorySlot(params.queryAnalysis, params.coverage.slotId)) return params.coverage;
	const roleSpecificHits = params.coverage.requiredHits.filter((hit) => !isQueryContextHit(params.queryAnalysis, hit));
	if (roleSpecificHits.length > 0) return params.coverage;
	const semanticSupport = evidenceGoalQuerySemanticSupport(params.queryAnalysis, params.text);
	const slotSupport = slotSemanticSupport(params.queryAnalysis, params.coverage.slotId, params.text);
	const requiredSemanticSupport = isTailorAdviceResourceSlot(params.queryAnalysis, params.coverage.slotId) ? .34 : .42;
	const roleSupport = Math.max(semanticSupport, slotSupport, params.coverage.coverageScore);
	if (roleSupport >= requiredSemanticSupport) return {
		...params.coverage,
		missingRequired: [],
		coverageScore: Math.max(params.coverage.coverageScore, roleSupport),
		filled: true
	};
	return {
		...params.coverage,
		requiredHits: roleSpecificHits,
		missingRequired: uniqueNonEmpty([...params.coverage.missingRequired, ...params.coverage.requiredHits.length > 0 ? params.coverage.requiredHits : ["role-specific memory evidence"]]),
		coverageScore: Math.min(params.coverage.coverageScore, .28),
		filled: false
	};
}
function slotCoverageWithCandidateMatches(queryAnalysis, text, hit) {
	const baseCoverage = slotCoverageForText(queryAnalysis, text).map((coverage) => pruneTailorAdviceMemorySlotCoverage({
		queryAnalysis,
		text,
		coverage
	}));
	const slotMatches = hit.slotMatches ?? [];
	if (slotMatches.length === 0) return baseCoverage;
	return baseCoverage.map((coverage) => {
		const bestMatch = slotMatches.filter((match) => match.slotId === coverage.slotId).sort((left, right) => right.score - left.score)[0];
		const resourceSlot = isTailorAdviceResourceSlot(queryAnalysis, coverage.slotId);
		const threshold = queryAnalysis.evidencePlan?.operation.type === "tailor_advice" ? resourceSlot ? .16 : .34 : .52;
		if (!bestMatch || bestMatch.score < threshold) return coverage;
		const memorySlot = isTailorAdviceMemorySlot(queryAnalysis, coverage.slotId);
		const querySemanticSupport = evidenceGoalQuerySemanticSupport(queryAnalysis, text);
		const specificSupport = memorySlot ? querySemanticSupport : Math.max(querySemanticSupport, evidenceGoalFocusSupport(queryAnalysis, text));
		const matchedQuerySupport = semanticTextSimilarity(bestMatch.matchedQuery, text);
		const hasConcreteBaseHit = coverage.requiredHits.some((entry) => !isGenericRequiredField(normalizeText(entry)));
		const highConfidenceSemanticMatch = bestMatch.score >= .86 && hit.retrievalBackend === "embedding";
		const semanticSlotMatch = specificSupport >= .18 || matchedQuerySupport >= .42 || bestMatch.score >= .82 && matchedQuerySupport >= .32;
		const structuredResourceMatch = resourceSlot && candidateHitHasStructuredResourceSignal(hit) && bestMatch.score >= .16;
		if (!(memorySlot ? hasConcreteBaseHit && specificSupport >= .18 || semanticSlotMatch || structuredResourceMatch : hasConcreteBaseHit && specificSupport >= .08 || semanticSlotMatch || highConfidenceSemanticMatch && specificSupport >= .12)) return {
			...coverage,
			requiredHits: uniqueNonEmpty([...coverage.requiredHits, bestMatch.matchedQuery]),
			coverageScore: Math.max(coverage.coverageScore, clamp01(bestMatch.score * .62))
		};
		return {
			...coverage,
			requiredHits: uniqueNonEmpty([...coverage.requiredHits, bestMatch.matchedQuery]),
			missingRequired: [],
			coverageScore: Math.max(coverage.coverageScore, clamp01(bestMatch.score)),
			filled: true
		};
	});
}
function filledSlotIdsFromCoverage(slotCoverage) {
	return slotCoverage.filter((coverage) => coverage.filled).map((coverage) => coverage.slotId);
}
function stripPromptEvidenceRolePrefix(text) {
	return text.trim().replace(/^(?:\[(?:user|assistant|tool|memory)\]|(?:user|assistant|tool|memory))\s*:\s*/iu, "").replace(/^assistant\s+explanatory\s+response\s*[:.-]?\s*/iu, "").trim();
}
function collapseRepeatedLeadingSentence(text) {
	const trimmed = text.trim();
	const firstSentence = trimmed.split(/(?<=[.!?])\s+/u)[0] ?? "";
	if (firstSentence.length < 24 || firstSentence.length >= trimmed.length) return trimmed;
	const remainder = trimmed.slice(firstSentence.length).trimStart();
	if (remainder.toLowerCase().startsWith(firstSentence.toLowerCase())) return `${firstSentence} ${remainder.slice(firstSentence.length).trimStart()}`.trim();
	return trimmed;
}
function truncatedPrefixStem(text) {
	const trimmed = text.trim();
	if (!/(?:…|\.\.\.)\s*$/u.test(trimmed)) return "";
	const withoutEllipsis = trimmed.replace(/(?:…|\.\.\.)+\s*$/u, "").trim();
	const withoutPartialToken = withoutEllipsis.replace(/\s+\S+$/u, "").trim();
	return withoutPartialToken.length >= 16 ? withoutPartialToken : withoutEllipsis;
}
function cleanPromptEvidenceText(text) {
	const trimmed = text.trim();
	if (!trimmed) return trimmed;
	const lines = trimmed.split(/\r?\n/u);
	const firstLine = stripPromptEvidenceRolePrefix(lines[0] ?? "");
	const rest = lines.slice(1).join("\n").trim();
	if (rest) {
		const cleanedRest = stripPromptEvidenceRolePrefix(rest);
		const firstNormalized = normalizeText(firstLine);
		const restNormalized = normalizeText(cleanedRest);
		const truncatedStemNormalized = normalizeText(truncatedPrefixStem(firstLine));
		const firstRestSentence = cleanedRest.split(/(?<=[.!?])\s+/u)[0] ?? cleanedRest;
		const restStartsWithSummary = firstNormalized.length >= 16 && restNormalized.startsWith(firstNormalized);
		const restStartsWithTruncatedSummary = truncatedStemNormalized.length >= 16 && restNormalized.startsWith(truncatedStemNormalized);
		const restRepeatsSummary = firstNormalized.length >= 24 && semanticTextSimilarity(firstLine, firstRestSentence.slice(0, 260)) >= .82;
		const uninformativeSummary = /^assistant\s+explanatory\s+response$/iu.test(firstLine) || firstNormalized === "assistant explanatory response";
		if (restStartsWithSummary || restStartsWithTruncatedSummary || restRepeatsSummary || uninformativeSummary) return collapseRepeatedLeadingSentence(cleanedRest);
		if (firstLine !== (lines[0] ?? "").trim()) return collapseRepeatedLeadingSentence(`${firstLine}\n${cleanedRest}`.trim());
		return collapseRepeatedLeadingSentence(`${lines[0] ?? ""}\n${cleanedRest}`.trim());
	}
	return collapseRepeatedLeadingSentence(stripPromptEvidenceRolePrefix(trimmed));
}
function rawAssistantExplanatorySummary(text) {
	return normalizeText((text.trim().split(/\r?\n/u)[0] ?? "").trim().replace(/^(?:\[(?:user|assistant|tool|memory)\]|(?:user|assistant|tool|memory))\s*:\s*/iu, "")) === "assistant explanatory response";
}
function promptEvidenceMetadataForRawText(text, metadata) {
	if (!rawAssistantExplanatorySummary(text)) return metadata;
	return {
		...metadata ?? {},
		semanticRole: "assistant_acknowledgement"
	};
}
function promptEvidenceRole(candidate) {
	const text = (candidate.rawText ?? candidate.text).trim().toLowerCase();
	if (/^(?:\[assistant\]|assistant)\s*:/iu.test(text) || text.startsWith("assistant explanatory response")) return "assistant";
	if (/^(?:\[user\]|user)\s*:/iu.test(text)) return "user";
	if (/^(?:\[tool\]|tool)\s*:/iu.test(text)) return "tool";
	return "unknown";
}
function queryAllowsAssistantAuthoredEvidence(queryAnalysis) {
	if ((queryAnalysis.evidencePlan?.slots ?? []).some((slot) => slot.role === "prior_advice" || slot.requiredRole === "prior_advice")) return true;
	return (queryAnalysis.semanticBridges ?? []).some((bridge) => bridge.role === "prior_advice");
}
function promptEvidenceHasLiteralFocusAnchor(queryAnalysis, text) {
	const anchors = informativePromptAnchors([...queryAnalysis.anchors, ...(queryAnalysis.evidenceGoals ?? []).flatMap((goal) => goal.focusAnchors)]);
	const normalizedText = normalizeText(text);
	return anchors.some((anchor) => {
		const normalizedAnchor = normalizeText(anchor);
		return normalizedAnchor.length >= 4 && normalizedText.includes(normalizedAnchor);
	});
}
function promptEvidenceAnswerSpecificity(queryAnalysis, candidate) {
	const scoringText = promptEvidenceScoringText(candidate);
	const coverage = candidate.coverage ?? evidenceCoverageForText(queryAnalysis, scoringText);
	const answerSlotSupport = promptEvidenceAnswerSlotStrength(queryAnalysis, candidate);
	const querySemanticSupport = evidenceGoalQuerySemanticSupport(queryAnalysis, scoringText);
	const focusSupport = evidenceGoalFocusSupport(queryAnalysis, scoringText);
	const literalFocus = promptEvidenceHasLiteralFocusAnchor(queryAnalysis, scoringText) ? .6 : 0;
	const coverageSupport = coverage.requiredHits.length > 0 ? .7 : 0;
	const highSemanticSupport = (candidate.semanticScore ?? candidate.goalScore) >= .82 || candidate.goalScore >= .82 ? .42 : 0;
	return clamp01(Math.max(querySemanticSupport, focusSupport, literalFocus, answerSlotSupport, coverageSupport, highSemanticSupport));
}
function promptEvidenceAnswerSlotStrength(queryAnalysis, candidate) {
	if ((queryAnalysis.evidencePlan?.slots ?? []).length === 0) return 0;
	const nonSubjectHitCount = promptEvidenceNonSubjectSlotHitCount(queryAnalysis, candidate);
	if (nonSubjectHitCount === 0) return 0;
	const slotCoverage = candidate.slotCoverage ?? slotCoverageForText(queryAnalysis, candidate.text);
	const bestSlotCoverage = Math.max(0, ...slotCoverage.map((entry) => entry.coverageScore));
	return clamp01(Math.max(.52, Math.min(.86, bestSlotCoverage)) + Math.min(.18, nonSubjectHitCount * .055));
}
function promptEvidenceSameSourceQuality(queryAnalysis, candidate) {
	const scoringText = promptEvidenceScoringText(candidate);
	const coverage = candidate.coverage ?? evidenceCoverageForText(queryAnalysis, scoringText);
	const coverageSupport = Math.min(.22, coverage.requiredHits.length * .11);
	const detailSupport = Math.min(.12, normalizeText(candidate.text).length / 900);
	return clamp01(promptEvidenceAnswerSpecificity(queryAnalysis, candidate) * .42 + promptEvidenceAnswerSlotStrength(queryAnalysis, candidate) * .24 + coverageSupport + promptEvidenceAuthorityRank(candidate) * .035 + detailSupport - evidenceTextCompletenessPenalty(candidate.text));
}
function promptEvidenceAuthorityRank(candidate) {
	const role = promptEvidenceRole(candidate);
	const sourceRef = candidate.sourceRef ?? "";
	if (role === "assistant") return .6;
	if (role === "user" && (candidate.surface === "chunk" || candidate.surface === "snippet") && sourceRef) return 5;
	if (candidate.surface === "chunk" || candidate.surface === "snippet") return sourceRef ? 4.4 : 3.8;
	if (role === "user") return 3.8;
	if (candidate.surface === "fact") return 3.2;
	if (candidate.surface === "event") return 2.8;
	return 2;
}
function evidenceTextCompletenessPenalty(text) {
	const trimmed = text.trim();
	if (!trimmed) return .18;
	let penalty = 0;
	if (isTruncatedEvidenceText(trimmed)) penalty += .24;
	return Math.min(.32, penalty);
}
function isTruncatedEvidenceText(text) {
	const trimmed = text.trim();
	return trimmed.includes("…") || trimmed.endsWith("...");
}
function shouldReplacePromptEvidence(existing, candidate, queryAnalysis) {
	if (queryAnalysis && existing.sourceRef && existing.sourceRef === candidate.sourceRef) {
		const candidateQuality = promptEvidenceSameSourceQuality(queryAnalysis, candidate);
		const existingQuality = promptEvidenceSameSourceQuality(queryAnalysis, existing);
		if (candidateQuality >= existingQuality + .05) return true;
		if (existingQuality >= candidateQuality + .08) return false;
	}
	const existingTruncated = isTruncatedEvidenceText(existing.text);
	const candidateTruncated = isTruncatedEvidenceText(candidate.text);
	if (existingTruncated !== candidateTruncated) return candidateTruncated ? false : candidate.priority >= existing.priority - .18;
	if (candidate.goalScore !== existing.goalScore) {
		if (promptEvidenceAuthorityRank(candidate) - promptEvidenceAuthorityRank(existing) >= 1 && candidate.goalScore >= existing.goalScore - .08) return true;
		return candidate.goalScore > existing.goalScore;
	}
	if (promptEvidenceAuthorityRank(candidate) - promptEvidenceAuthorityRank(existing) >= 1 && candidate.priority >= existing.priority - .14) return true;
	return candidate.priority > existing.priority;
}
function mergePromptEvidenceSourceRefs(winner, loser) {
	const mergedSourceRefs = uniqueNonEmpty([
		winner.sourceRef,
		...winner.mergedSourceRefs ?? [],
		loser.sourceRef,
		...loser.mergedSourceRefs ?? []
	].filter((entry) => typeof entry === "string"));
	return {
		...winner,
		mergedSourceRefs: mergedSourceRefs.length > 0 ? mergedSourceRefs : void 0
	};
}
function promptEvidencePriority(params) {
	const semanticScore = rawScoreTextAgainstEvidenceGoals({
		text: params.text,
		queryAnalysis: params.queryAnalysis,
		surface: params.surface,
		explicitGoalScore: params.explicitGoalScore
	});
	const goalScore = params.queryAnalysis.evidencePlan?.operation.type === "tailor_advice" ? semanticScore : capScoreByEvidenceCoverage(semanticScore, evidenceCoverageForText(params.queryAnalysis, params.text));
	const sourceBoost = {
		candidate: .1,
		selected: .08,
		support_ref: .07,
		projected: .06,
		fallback: .03
	};
	const surfaceBoost = params.surface === "chunk" || params.surface === "snippet" ? params.queryAnalysis.evidenceFidelity === "high" || params.queryAnalysis.answerGranularity === "detail" ? .06 : .03 : params.surface === "event" ? .04 : .02;
	const tierBoost = params.tier === "primary" ? .05 : params.tier === "alternate" ? -.06 : 0;
	const focusSupport = evidenceGoalFocusSupport(params.queryAnalysis, params.text);
	const querySemanticSupport = evidenceGoalQuerySemanticSupport(params.queryAnalysis, params.text);
	const focusPenalty = (params.queryAnalysis.evidenceFidelity === "high" || params.queryAnalysis.answerGranularity === "detail") && (params.queryAnalysis.evidenceGoals ?? []).some((goal) => goal.focusAnchors.length > 0) && semanticScore < .48 && focusSupport < .22 ? .22 : semanticScore < .4 && focusSupport < .38 ? .08 : 0;
	const weakUncoveredGoalPenalty = params.queryAnalysis.evidencePlan?.operation.type === "tailor_advice" && params.source === "candidate" && params.score && params.score >= .8 && goalScore >= .6 && evidenceCoverageForText(params.queryAnalysis, params.text).requiredHits.length === 0 && querySemanticSupport < .14 && focusSupport < .22 ? .08 : 0;
	const completenessPenalty = evidenceTextCompletenessPenalty(params.text);
	const slotCoverage = slotCoverageForText(params.queryAnalysis, params.text);
	const topSlotCoverage = Math.max(0, ...slotCoverage.map((coverage) => coverage.coverageScore));
	const filledSlotBoost = slotCoverage.some((coverage) => coverage.filled) ? .12 : 0;
	return {
		priority: clamp01(goalScore * .52 + Math.max(0, params.score ?? 0) * .08 + sourceBoost[params.source] + surfaceBoost + preferredSurfaceSupport(params.queryAnalysis, params.surface) + tierBoost + focusSupport * .08 + querySemanticSupport * .22 + Math.max(0, topSlotCoverage - .45) * .08 + filledSlotBoost * .55 - focusPenalty - weakUncoveredGoalPenalty - completenessPenalty),
		goalScore,
		semanticScore
	};
}
function promptEvidenceSlotRoleRank(role) {
	if (role === "user_resource") return 4;
	if (role === "answer_value" || role === "answer_event") return 3;
	if (role === "prior_advice") return 2;
	if (role === "query_context" || role === "time_constraint") return 1;
	return 0;
}
function promptEvidenceSlotRoleForCandidate(queryAnalysis, candidate) {
	const slotById = new Map((queryAnalysis.evidencePlan?.slots ?? []).map((slot) => [slot.id, slot]));
	const scoringText = promptEvidenceScoringText(candidate);
	const best = (candidate.slotCoverage ?? slotCoverageForText(queryAnalysis, scoringText)).map((entry) => {
		const role = evidenceSlotRequiredRole(slotById.get(entry.slotId));
		return {
			role,
			filled: entry.filled,
			coverageScore: entry.coverageScore,
			rank: promptEvidenceSlotRoleRank(role)
		};
	}).filter((entry) => entry.role).sort((left, right) => {
		if (left.filled !== right.filled) return left.filled ? -1 : 1;
		if (right.rank !== left.rank) return right.rank - left.rank;
		return right.coverageScore - left.coverageScore;
	})[0];
	if (!best || !best.filled && best.coverageScore < .36) return;
	return best.role;
}
function metadataHasStructuredResourceSignal(metadata) {
	if (!metadata) return false;
	return metadata.signalKind === "resourceAssertion" || typeof metadata.resourceType === "string" || Array.isArray(metadata.affordances) && metadata.affordances.length > 0 || Array.isArray(metadata.domains) && metadata.domains.length > 0;
}
function textHasStructuredResourceMarker(text) {
	const normalized = normalizeText(text);
	return normalized.includes("user.resource") || normalized.includes("has_resource") || normalized.includes("resource_observation") || normalized.includes("affordances") || normalized.includes("resource type") || normalized.includes("signalkind resourceassertion");
}
function promptEvidenceHasStructuredResourceSignal(candidate) {
	return metadataHasStructuredResourceSignal(candidate.metadata) || textHasStructuredResourceMarker(promptEvidenceScoringText(candidate));
}
function candidateHitHasStructuredResourceSignal(hit) {
	return metadataHasStructuredResourceSignal(hit.metadata) || textHasStructuredResourceMarker(promptEvidenceScoringTextFromMetadata(hit.text, hit.metadata));
}
function promptEvidenceCapabilitySupport(queryAnalysis, candidate) {
	const slotById = new Map((queryAnalysis.evidencePlan?.slots ?? []).map((slot) => [slot.id, slot]));
	const role = candidate.slotEvidenceRole ?? promptEvidenceSlotRoleForCandidate(queryAnalysis, candidate);
	const capabilityHints = uniqueNonEmpty((candidate.slotCoverage ?? slotCoverageForText(queryAnalysis, promptEvidenceScoringText(candidate))).filter((coverage) => coverage.filled || coverage.coverageScore >= .36).map((coverage) => slotById.get(coverage.slotId)).filter((slot) => Boolean(slot)).filter((slot) => !role || evidenceSlotRequiredRole(slot) === role).flatMap((slot) => slot.capabilityQueries && slot.capabilityQueries.length > 0 ? slot.capabilityQueries : slot.relationHints ?? []), 12);
	if (capabilityHints.length === 0) return 0;
	const scoringText = promptEvidenceScoringText(candidate);
	const anchorSupport = informativePromptAnchorSupport(scoringText, capabilityHints);
	const semanticSupport = Math.max(0, ...capabilityHints.map((hint) => semanticTextSimilarity(hint, scoringText)));
	return Math.max(anchorSupport, semanticSupport);
}
function promptEvidenceInjectionLimit(queryAnalysis, ranked) {
	if (ranked.length === 0) return 0;
	const planSlots = queryAnalysis.evidencePlan?.slots ?? [];
	if (queryAnalysis.evidencePlan?.operation.type && queryAnalysis.evidencePlan.operation.type !== "tailor_advice" && planSlots.length > 0) {
		const minEvidence = planSlots.reduce((sum, slot) => sum + slot.minEvidence, 0);
		return Math.max(1, Math.min(3, minEvidence, ranked.length));
	}
	if (queryAnalysis.evidencePlan?.operation.type === "tailor_advice") return Math.min(3, ranked.length);
	if (queryAnalysis.answerMode === "count_aggregate") return Math.max(1, Math.min(3, queryAnalysis.evidenceCoverage?.minProtectedItems ?? 2, ranked.length));
	if (promptEvidenceNeedsCompanions(queryAnalysis)) return Math.min(3, ranked.length);
	return 1;
}
function scorePromptEvidenceForInjection(queryAnalysis, candidate) {
	const slotEvidenceRole = candidate.slotEvidenceRole ?? promptEvidenceSlotRoleForCandidate(queryAnalysis, candidate);
	const semanticScore = candidate.semanticScore ?? candidate.goalScore;
	const scoringText = promptEvidenceScoringText(candidate);
	const slotCoverage = candidate.slotCoverage ?? slotCoverageForText(queryAnalysis, scoringText);
	const bestSlotCoverage = Math.max(0, ...slotCoverage.map((entry) => entry.coverageScore));
	const filledSlotCount = slotCoverage.filter((entry) => entry.filled).length;
	const focusSupport = evidenceGoalFocusSupport(queryAnalysis, scoringText);
	const goalQuerySupport = evidenceGoalQuerySemanticSupport(queryAnalysis, scoringText);
	const bridgeSupport = Math.max(0, ...(candidate.bridgeMatches ?? []).map((match) => match.score));
	const bridgePositiveSupport = Math.max(0, ...(candidate.bridgeMatches ?? []).map((match) => match.positiveSignalScore));
	const bridgeNegativeSupport = Math.max(0, ...(candidate.bridgeMatches ?? []).map((match) => match.negativeSignalScore));
	const capabilitySupport = promptEvidenceCapabilitySupport(queryAnalysis, candidate);
	const authority = promptEvidenceAuthorityRank(candidate) / 5;
	const operationType = queryAnalysis.evidencePlan?.operation.type;
	const roleWeight = operationType === "tailor_advice" ? slotEvidenceRole === "user_resource" ? .22 : slotEvidenceRole === "prior_advice" ? .14 : slotEvidenceRole === "answer_value" || slotEvidenceRole === "answer_event" ? .16 : slotEvidenceRole === "query_context" || slotEvidenceRole === "time_constraint" ? -.08 : 0 : slotEvidenceRole === "answer_value" || slotEvidenceRole === "answer_event" ? .12 : slotEvidenceRole === "user_resource" ? .08 : slotEvidenceRole === "time_constraint" ? -.04 : 0;
	const sourceTrace = candidate.sourceRef || (candidate.mergedSourceRefs?.length ?? 0) > 0 ? .04 : 0;
	const structuredResource = slotEvidenceRole === "user_resource" && promptEvidenceHasStructuredResourceSignal(candidate) && (capabilitySupport >= .2 || goalQuerySupport >= .24 || semanticScore >= .32) ? .1 : 0;
	const resourceCapabilityPenalty = slotEvidenceRole === "user_resource" && promptEvidenceHasStructuredResourceSignal(candidate) && capabilitySupport < .14 && goalQuerySupport < .2 && semanticScore < .3 ? .24 : 0;
	const assistantPenalty = promptEvidenceRole(candidate) === "assistant" && !queryAllowsAssistantAuthoredEvidence(queryAnalysis) ? .16 : 0;
	const queryContextOnlyPenalty = operationType === "tailor_advice" && (slotEvidenceRole === "query_context" || slotEvidenceRole === "time_constraint" || !slotEvidenceRole) && filledSlotCount <= 1 && bestSlotCoverage < .52 ? .18 : 0;
	const incompletePenalty = evidenceTextCompletenessPenalty(candidate.text);
	const injectionScore = clamp01(candidate.priority * .34 + semanticScore * .2 + bestSlotCoverage * .16 + Math.min(.12, filledSlotCount * .055) + focusSupport * .08 + goalQuerySupport * .12 + bridgeSupport * .14 + bridgePositiveSupport * .08 + capabilitySupport * .16 + authority * .07 + sourceTrace + roleWeight + structuredResource - resourceCapabilityPenalty - assistantPenalty - queryContextOnlyPenalty - bridgeNegativeSupport * .12 - incompletePenalty);
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
			incompletePenalty
		}
	};
}
function promptEvidenceNeedsCompanions(queryAnalysis) {
	const surfaces = new Set(queryAnalysis.candidateSurfaces);
	const planSlots = queryAnalysis.evidencePlan?.slots ?? [];
	return planSlots.length > 1 || planSlots.some((slot) => slot.minEvidence > 1) || queryAnalysis.answerMode === "count_aggregate" || queryAnalysis.answerMode === "multi_evidence" || queryAnalysis.queryShape.timeframe === "compare" || queryAnalysis.queryShape.evidenceNeed === "relation" || (queryAnalysis.evidenceGoals ?? []).length > 1 || surfaces.has("fact") && surfaces.has("event") && !surfaces.has("state") && !surfaces.has("chunk");
}
function promptEvidenceFromRow(params) {
	const text = cleanPromptEvidenceText(params.row.text);
	const supportText = params.row.provenance?.trim();
	const scoringText = supportText ? uniqueNonEmpty([text, supportText], 6).join(" | ") : text;
	const scored = promptEvidencePriority({
		text: scoringText,
		score: params.row.score,
		surface: params.surface,
		source: params.source,
		queryAnalysis: params.queryAnalysis
	});
	return {
		id: params.row.id,
		surface: params.surface,
		text,
		rawText: supportText ? `${params.row.text}\n${supportText}` : params.row.text,
		scoringText: scoringText !== text ? scoringText : void 0,
		sourceRef: params.row.sourceRef,
		mergedSourceRefs: params.row.lineage?.sourceRef ? uniqueNonEmpty([params.row.sourceRef ?? "", params.row.lineage.sourceRef]) : void 0,
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
		role: "support"
	};
}
function metadataStringArray(metadata, key) {
	const value = metadata?.[key];
	if (!Array.isArray(value)) return [];
	return value.filter((entry) => typeof entry === "string" && entry.trim().length > 0);
}
function sourceRefsFromCandidateMetadata(metadata) {
	const refs = [
		...typeof metadata?.sourceRef === "string" ? [metadata.sourceRef] : [],
		...metadataStringArray(metadata, "sourceRefs"),
		...metadataStringArray(metadata, "supportRefs"),
		...metadataStringArray(metadata, "supportContentRefs"),
		...metadataStringArray(metadata, "sourceRefsForExpansion")
	];
	const maintenanceContract = metadata?.maintenanceContract;
	if (maintenanceContract && typeof maintenanceContract === "object") refs.push(...metadataStringArray(maintenanceContract, "sourceRefsForExpansion"));
	const frame = metadata?.turnSemanticFrame;
	if (frame && typeof frame === "object") {
		const record = frame;
		refs.push(...metadataStringArray(record, "sourceRefs"));
		for (const key of [
			"chunkDrafts",
			"assertionDrafts",
			"correctionDrafts"
		]) {
			const entries = record[key];
			if (!Array.isArray(entries)) continue;
			for (const entry of entries) {
				if (!entry || typeof entry !== "object") continue;
				const draft = entry;
				if (typeof draft.sourceRef === "string") refs.push(draft.sourceRef);
				const lineage = draft.lineage;
				if (lineage && typeof lineage === "object") {
					const sourceRef = lineage.sourceRef;
					if (typeof sourceRef === "string") refs.push(sourceRef);
				}
				refs.push(...metadataStringArray(draft, "supportRefs"));
				const supportSpans = draft.supportSpans;
				if (Array.isArray(supportSpans)) {
					for (const span of supportSpans) if (span && typeof span === "object") {
						const sourceRef = span.sourceRef;
						if (typeof sourceRef === "string") refs.push(sourceRef);
					}
				}
			}
		}
	}
	return uniqueNonEmpty(refs);
}
function numericSuffix(value) {
	const match = /(\d+)$/u.exec(value);
	if (!match?.[1]) return;
	const parsed = Number.parseInt(match[1], 10);
	return Number.isFinite(parsed) ? parsed : void 0;
}
function turnPartLike(original, value) {
	const match = /^([^\d]*)(\d+)$/u.exec(original ?? "");
	return match?.[1] ? `${match[1]}${value}` : String(value);
}
function sourceExpansionNeighborRefs(sourceRef, window = 2) {
	const parts = sourceRef.split(":").filter(Boolean);
	const refs = [sourceRef];
	const pushTurnWindow = (prefix, turn, suffixes = [], userPrefix, turnTemplate) => {
		for (let next = Math.max(1, turn - window); next <= turn + window; next += 1) {
			const turnPart = turnPartLike(turnTemplate, next);
			refs.push([...prefix, turnPart].join(":"));
			for (const suffix of suffixes) refs.push([
				...prefix,
				turnPart,
				suffix
			].join(":"));
			if (userPrefix) {
				refs.push([...userPrefix, turnPart].join(":"));
				for (const suffix of suffixes) refs.push([
					...userPrefix,
					turnPart,
					suffix
				].join(":"));
			}
		}
	};
	if (parts[0] === "lme" && parts.length >= 4) {
		const turnPart = parts[3] ?? "";
		const turn = numericSuffix(turnPart);
		if (turn !== void 0) pushTurnWindow(parts.slice(0, 3), turn, ["user", "assistant"], void 0, turnPart);
	}
	if (parts[0] === "user" && parts[1] === "lme" && parts.length >= 5) {
		const turnPart = parts[4] ?? "";
		const turn = numericSuffix(turnPart);
		if (turn !== void 0) pushTurnWindow([
			"lme",
			parts[2],
			parts[3]
		], turn, ["user", "assistant"], void 0, turnPart);
	}
	if (parts[0] === "agentmem" && parts.length >= 4) {
		const turnPart = parts[3] ?? "";
		const turn = numericSuffix(turnPart);
		if (turn !== void 0) pushTurnWindow([
			"agentmem",
			parts[1],
			parts[2]
		], turn, ["user", "assistant"], [
			"user",
			"agentmem",
			parts[1],
			parts[2]
		], turnPart);
	}
	if (parts[0] === "user" && parts[1] === "agentmem" && parts.length >= 5) {
		const turnPart = parts[4] ?? "";
		const turn = numericSuffix(turnPart);
		if (turn !== void 0) pushTurnWindow([
			"agentmem",
			parts[2],
			parts[3]
		], turn, ["user", "assistant"], [
			"user",
			"agentmem",
			parts[2],
			parts[3]
		], turnPart);
	}
	if (parts[0] === "user" && parts[1] === "lme" && parts.length >= 6 && parts[4] === "batch") {
		const batch = numericSuffix(parts[5] ?? "");
		if (batch !== void 0) {
			const prefix = [
				"lme",
				parts[2],
				parts[3]
			];
			const firstTurn = Math.max(1, batch * 2 - 1);
			const lastTurn = Math.max(firstTurn, batch * 2);
			for (let next = Math.max(1, firstTurn - window); next <= lastTurn + window; next += 1) refs.push([...prefix, String(next)].join(":"));
		}
	}
	return uniqueNonEmpty(refs);
}
function maintenanceObjectExpansionRefs(store, sourceRef, seen = /* @__PURE__ */ new Set()) {
	if (!sourceRef || seen.has(sourceRef)) return [];
	seen.add(sourceRef);
	if (sourceRef.startsWith("abstraction_candidate:")) {
		const candidate = store.abstractionRepo.getById(sourceRef.slice(22));
		if (!candidate) return [];
		const refs = uniqueNonEmpty([...sourceRefsFromMaintenanceMetadata(candidate.metadataJson), ...candidate.supportContentRefs]);
		return uniqueNonEmpty([...refs, ...refs.flatMap((ref) => maintenanceObjectExpansionRefs(store, ref, seen))]);
	}
	if (sourceRef.startsWith("fact:")) {
		const fact = store.factRepo.get(sourceRef.slice(5));
		if (!fact) return [];
		const refs = factSupportSourceRefs(fact);
		return uniqueNonEmpty([...refs, ...refs.flatMap((ref) => maintenanceObjectExpansionRefs(store, ref, seen))]);
	}
	if (sourceRef.startsWith("event:")) {
		const event = store.eventRepo.get(sourceRef.slice(6));
		return event ? uniqueNonEmpty([event.sourceRef]) : [];
	}
	return [];
}
function sourceExpansionSeedScore(candidate) {
	const roleBonus = candidate.slotEvidenceRole === "answer_value" || candidate.slotEvidenceRole === "answer_event" ? .18 : candidate.slotEvidenceRole === "query_context" ? .12 : candidate.slotEvidenceRole === "user_resource" || candidate.slotEvidenceRole === "prior_advice" ? .1 : 0;
	const sourceTraceBonus = candidate.sourceRef || (candidate.mergedSourceRefs?.length ?? 0) > 0 ? .06 : 0;
	return clamp01(Math.max(candidate.priority, candidate.goalScore, candidate.semanticScore ?? 0) + roleBonus + sourceTraceBonus + Math.min(.12, (candidate.filledSlotIds?.length ?? 0) * .04));
}
function sourceExpansionBaseRefs(candidate) {
	return uniqueNonEmpty([
		candidate.sourceRef,
		candidate.lineage?.sourceRef,
		...candidate.mergedSourceRefs ?? [],
		...sourceRefsFromCandidateMetadata(candidate.metadata)
	].filter((value) => typeof value === "string"));
}
function sourceExpansionRefsForCandidates(store, candidates) {
	const refs = [];
	const seen = /* @__PURE__ */ new Set();
	const pushRef = (sourceRef) => {
		if (!sourceRef || seen.has(sourceRef)) return;
		seen.add(sourceRef);
		refs.push(sourceRef);
	};
	const sortedSeeds = [...candidates].filter((candidate) => sourceExpansionBaseRefs(candidate).length > 0).sort((left, right) => sourceExpansionSeedScore(right) - sourceExpansionSeedScore(left)).slice(0, 40);
	for (const candidate of sortedSeeds) {
		for (const sourceRef of sourceExpansionBaseRefs(candidate)) {
			for (const neighborRef of sourceExpansionNeighborRefs(sourceRef)) pushRef(neighborRef);
			for (const expandedRef of maintenanceObjectExpansionRefs(store, sourceRef)) for (const neighborRef of sourceExpansionNeighborRefs(expandedRef)) pushRef(neighborRef);
		}
		if (refs.length >= 360) break;
	}
	return refs.slice(0, 360);
}
function sourceExpansionChunksForRefs(params) {
	const chunkById = /* @__PURE__ */ new Map();
	const directSourceRefs = [];
	const pushChunk = (chunk) => {
		if (!chunk) return;
		if (chunk.agentId !== params.ctx.agentId || !params.ctx.scopes.includes(chunk.scope)) return;
		if (chunk.dedupStatus !== "active") return;
		chunkById.set(chunk.chunkId, chunk);
		directSourceRefs.push(chunk.sourceRef);
	};
	for (const sourceRef of params.sourceRefs) {
		if (sourceRef.startsWith("chunk:")) {
			pushChunk(params.store.chunkRepo.get(sourceRef.slice(6)));
			continue;
		}
		if (sourceRef.startsWith("task:")) {
			const taskId = sourceRef.slice(5).split(":")[0];
			if (taskId) for (const chunk of params.store.chunkRepo.listByTask(taskId)) pushChunk(chunk);
			continue;
		}
		directSourceRefs.push(sourceRef);
	}
	const sourceRefs = uniqueNonEmpty([...params.sourceRefs, ...directSourceRefs]);
	for (const chunk of params.store.chunkRepo.listBySourceRefs({
		agentId: params.ctx.agentId,
		scopes: params.ctx.scopes,
		sourceRefs,
		limit: Math.max(32, sourceRefs.length)
	})) pushChunk(chunk);
	return [...chunkById.values()].sort((left, right) => Date.parse(right.createdAt) - Date.parse(left.createdAt) || right.seq - left.seq);
}
function promptEvidenceFromChunkExpansion(params) {
	const text = cleanPromptEvidenceText(`${params.chunk.role}: ${params.chunk.content}`);
	const scored = promptEvidencePriority({
		text,
		score: .64,
		surface: "chunk",
		source: "support_ref",
		queryAnalysis: params.queryAnalysis
	});
	const mergedSourceRefs = uniqueNonEmpty([params.chunk.sourceRef]);
	const candidate = {
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
			sourceRef: params.chunk.sourceRef
		},
		metadata: {
			sourceExpansion: true,
			authorRole: params.chunk.role,
			neighborOf: [params.chunk.sourceRef],
			turnId: params.chunk.turnId,
			sessionKey: params.chunk.sessionKey
		},
		priority: scored.priority,
		goalScore: scored.goalScore,
		semanticScore: scored.semanticScore,
		coverage: evidenceCoverageForText(params.queryAnalysis, text),
		slotCoverage: slotCoverageForText(params.queryAnalysis, text),
		filledSlotIds: filledSlotIdsForText(params.queryAnalysis, text),
		source: "support_ref",
		role: "support",
		selectionReason: "source-expansion:neighbor-chunk"
	};
	return {
		...candidate,
		slotEvidenceRole: promptEvidenceSlotRoleForCandidate(params.queryAnalysis, candidate)
	};
}
function promptEvidenceFromSegmentExpansion(params) {
	const text = cleanPromptEvidenceText(`${params.segment.role}: ${params.segment.text}`);
	const scored = promptEvidencePriority({
		text,
		score: .64,
		surface: "chunk",
		source: "support_ref",
		queryAnalysis: params.queryAnalysis
	});
	const candidate = {
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
			sourceRef: params.segment.parentSourceRef
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
			charEnd: params.segment.charEnd
		},
		priority: scored.priority,
		goalScore: scored.goalScore,
		semanticScore: scored.semanticScore,
		coverage: evidenceCoverageForText(params.queryAnalysis, text),
		slotCoverage: slotCoverageForText(params.queryAnalysis, text),
		filledSlotIds: filledSlotIdsForText(params.queryAnalysis, text),
		source: "support_ref",
		role: "support",
		selectionReason: "source-expansion:source-segment"
	};
	return {
		...candidate,
		slotEvidenceRole: promptEvidenceSlotRoleForCandidate(params.queryAnalysis, candidate)
	};
}
function sourceExpansionPromptEvidenceCandidates(params) {
	const sourceRefs = sourceExpansionRefsForCandidates(params.store, params.candidates);
	if (sourceRefs.length === 0) return [];
	const chunks = sourceExpansionChunksForRefs({
		store: params.store,
		ctx: params.ctx,
		sourceRefs
	});
	const candidates = [];
	for (const chunk of chunks) {
		const segments = chunk.content.length > 1800 ? params.store.sourceSegmentRepo.listByChunk(chunk.chunkId) : [];
		if (segments.length > 1) {
			candidates.push(...segments.map((segment) => promptEvidenceFromSegmentExpansion({
				segment,
				queryAnalysis: params.queryAnalysis,
				seedSourceRefs: sourceRefs
			})));
			continue;
		}
		candidates.push(promptEvidenceFromChunkExpansion({
			chunk,
			queryAnalysis: params.queryAnalysis,
			seedSourceRefs: sourceRefs
		}));
	}
	return candidates;
}
function promptEvidenceSourceRefs(candidate) {
	return uniqueNonEmpty([
		candidate.sourceRef,
		candidate.lineage?.sourceRef,
		...candidate.mergedSourceRefs ?? [],
		...sourceRefsFromCandidateMetadata(candidate.metadata)
	].filter((value) => typeof value === "string" && value.length > 0));
}
function sourceRefFamily(sourceRef) {
	const parts = sourceRef.split(":").filter(Boolean);
	const normalizeSessionPart = (part) => /^answer_[a-z0-9]+_\d+$/iu.test(part) ? part.replace(/_\d+$/u, "") : part;
	if (parts[0] === "user" && parts[1] === "agentmem" && parts.length >= 4) return `agentmem:${parts[2]}:${normalizeSessionPart(parts[3] ?? "")}`;
	if (parts[0] === "agentmem" && parts.length >= 3) return `agentmem:${parts[1]}:${normalizeSessionPart(parts[2] ?? "")}`;
	if (parts[0] === "user" && parts[1] === "lme" && parts.length >= 4) return `lme:${parts[2]}:${normalizeSessionPart(parts[3] ?? "")}`;
	if (parts[0] === "lme" && parts.length >= 3) return `lme:${parts[1]}:${normalizeSessionPart(parts[2] ?? "")}`;
	return parts.slice(0, Math.min(3, parts.length)).join(":") || sourceRef;
}
function sourceRefTurnIndex(sourceRef) {
	const parts = sourceRef.split(":").filter(Boolean);
	for (let index = parts.length - 1; index >= 0; index -= 1) {
		const value = numericSuffix(parts[index] ?? "");
		if (value !== void 0) return value;
	}
}
function promptEvidenceRefsAdjacent(left, right, maxDistance = 2) {
	for (const leftRef of promptEvidenceSourceRefs(left)) {
		const leftFamily = sourceRefFamily(leftRef);
		const leftTurn = sourceRefTurnIndex(leftRef);
		for (const rightRef of promptEvidenceSourceRefs(right)) {
			if (leftFamily !== sourceRefFamily(rightRef)) continue;
			const rightTurn = sourceRefTurnIndex(rightRef);
			if (leftTurn === void 0 || rightTurn === void 0) {
				if (leftRef === rightRef) return true;
				continue;
			}
			if (Math.abs(leftTurn - rightTurn) <= maxDistance) return true;
		}
	}
	return false;
}
function shouldRetainSourceExpansionAnswerCandidate(queryAnalysis, candidate) {
	if (candidate.metadata?.sourceExpansion !== true) return false;
	if (promptEvidenceRole(candidate) !== "assistant") return false;
	const slotRole = candidate.slotEvidenceRole ?? promptEvidenceSlotRoleForCandidate(queryAnalysis, candidate);
	if (slotRole === "answer_value" || slotRole === "answer_event") return true;
	return Math.max(candidate.injectionScore ?? 0, candidate.priority, candidate.goalScore, candidate.semanticScore ?? 0, promptEvidenceAnswerSlotStrength(queryAnalysis, candidate), promptEvidenceAnswerSpecificity(queryAnalysis, candidate)) >= .42;
}
function keepRetainedPromptEvidenceCandidates(params) {
	const keyFor = (candidate) => [
		candidate.surface,
		candidate.id,
		candidate.sourceRef ?? "",
		normalizeText(candidate.text).slice(0, 120)
	].join("|");
	const result = params.candidates.slice(0, params.cap);
	const seen = new Set(result.map(keyFor));
	const retainedAnswers = params.candidates.filter((candidate) => shouldRetainSourceExpansionAnswerCandidate(params.queryAnalysis, candidate)).slice(0, 8);
	const retainedContext = params.candidates.filter((candidate) => retainedAnswers.some((answer) => {
		if (keyFor(answer) === keyFor(candidate)) return false;
		if (candidate.metadata?.sourceExpansion !== true) return false;
		const role = candidate.slotEvidenceRole ?? promptEvidenceSlotRoleForCandidate(params.queryAnalysis, candidate);
		return promptEvidenceRefsAdjacent(answer, candidate, 2) && (role === "query_context" || promptEvidenceRole(candidate) === "user");
	})).slice(0, 12);
	const reserve = [...retainedAnswers, ...retainedContext];
	for (const candidate of reserve) {
		const key = keyFor(candidate);
		if (seen.has(key)) continue;
		const replaceIndex = [...result].map((entry, index) => ({
			entry,
			index
		})).reverse().find(({ entry }) => !shouldRetainSourceExpansionAnswerCandidate(params.queryAnalysis, entry))?.index;
		if (replaceIndex === void 0) result.push(candidate);
		else {
			seen.delete(keyFor(result[replaceIndex]));
			result[replaceIndex] = candidate;
		}
		seen.add(key);
	}
	return result.slice(0, Math.max(params.cap, reserve.length));
}
function controlPromptEvidenceCandidates(params) {
	if (!(params.queryAnalysis.evidencePlan?.slots ?? []).some((slot) => evidencePlanLayersForSlot(slot).some((layer) => layer === "control" || layer === "strategy" || layer === "abstraction" || layer === "belief"))) return [];
	const strategies = params.store.strategyRepo.listByAgent({
		agentId: params.ctx.agentId,
		scopes: params.ctx.scopes,
		stages: ["active", "candidate"],
		limit: 24,
		readEpoch: params.ctx.readEpoch
	});
	const abstractions = params.store.abstractionRepo.listByAgent({
		agentId: params.ctx.agentId,
		scopes: params.ctx.scopes,
		stages: [
			"active",
			"candidate",
			"probationary"
		],
		limit: 24,
		readEpoch: params.ctx.readEpoch
	});
	const beliefs = params.store.beliefRepo.listByAgent({
		agentId: params.ctx.agentId,
		limit: 48,
		readEpoch: params.ctx.readEpoch
	}).filter((belief) => params.ctx.scopes.includes(belief.scope));
	return [
		...strategies.map((entry) => {
			const sourceRefs = uniqueMaintenanceRefs([...sourceRefsFromMaintenanceMetadata(entry.metadataJson), ...entry.supportTaskIds.map((taskId) => `task:${taskId}`)]);
			return {
				id: `strategy:${entry.strategyId}`,
				text: `strategy: ${entry.summary}`,
				sourceRef: sourceRefs[0],
				sourceRefs,
				layer: "strategy",
				score: entry.confidence * .6 + entry.usefulnessScore * .4
			};
		}),
		...abstractions.map((entry) => {
			const sourceRefs = uniqueMaintenanceRefs([...sourceRefsFromMaintenanceMetadata(entry.metadataJson), ...entry.supportContentRefs]);
			return {
				id: `abstraction:${entry.candidateId}`,
				text: `${entry.abstractionType}: ${entry.summary}`,
				sourceRef: sourceRefs[0],
				sourceRefs,
				layer: "abstraction",
				score: entry.confidence * .6 + entry.usefulnessScore * .4
			};
		}),
		...beliefs.map((entry) => ({
			id: `belief:${entry.beliefId}`,
			text: `belief: ${entry.semanticKey}`,
			sourceRef: entry.contentRef,
			layer: "belief",
			score: entry.posteriorConfidence * .7 + entry.usefulnessScore * .3
		}))
	].map((row) => {
		const text = cleanPromptEvidenceText(row.text);
		const slotCoverage = slotCoverageForText(params.queryAnalysis, text);
		const filledSlotIds = slotCoverage.filter((coverage) => coverage.filled).map((coverage) => coverage.slotId);
		const scored = promptEvidencePriority({
			text,
			score: row.score,
			surface: "snippet",
			source: "selected",
			queryAnalysis: params.queryAnalysis
		});
		return {
			id: row.id,
			surface: "snippet",
			text,
			rawText: row.text,
			sourceRef: row.sourceRef,
			mergedSourceRefs: "sourceRefs" in row ? row.sourceRefs : void 0,
			observedAt: void 0,
			excerptAnchors: promptEvidenceExcerptAnchors(params.queryAnalysis),
			lineage: {
				sourceKind: "alternate",
				sourceId: row.id,
				sourceRef: row.sourceRef
			},
			priority: scored.priority,
			goalScore: scored.goalScore,
			semanticScore: scored.semanticScore,
			coverage: evidenceCoverageForText(params.queryAnalysis, text),
			slotCoverage,
			filledSlotIds,
			source: "selected",
			role: "support"
		};
	}).filter((entry) => (entry.filledSlotIds?.length ?? 0) > 0 || entry.priority >= .48).sort((left, right) => right.priority - left.priority).slice(0, 4);
}
function buildPromptEvidenceCandidates(params) {
	const candidates = [];
	const excerptAnchors = promptEvidenceExcerptAnchors(params.queryAnalysis);
	const candidateHits = [
		...params.candidateGenerationResult?.candidates ?? [],
		...params.candidateGenerationResult?.slotCandidates ?? [],
		...params.candidateGenerationResult?.bridgeCandidates ?? []
	];
	for (const hit of candidateHits) {
		if (uniqueNonEmpty([
			hit.docId,
			hit.candidateId,
			hit.lineage.canonicalId,
			hit.lineage.sourceId
		].filter((id) => typeof id === "string" && id.trim().length > 0)).some((id) => params.suppressedCandidateIds?.has(id))) continue;
		const surface = candidateSurfaceToPromptSurface(hit.surface);
		if (!surface) continue;
		const metadata = hydrateFactCandidateMetadata(params.store, hit);
		const observedAt = typeof metadata?.observedAt === "string" ? Date.parse(metadata.observedAt) : NaN;
		if (typeof params.suppressCandidateEvidenceBeforeMs === "number" && Number.isFinite(observedAt) && observedAt < params.suppressCandidateEvidenceBeforeMs) continue;
		if (hit.surface === "fact" && !factCandidateVisibleForQuery(params.queryAnalysis, metadata)) continue;
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
			explicitGoalScore: topCandidateGoalScore(hit)
		});
		const matchedSlotBoost = slotCoverage.some((coverage) => coverage.filled) ? .12 : 0;
		const filledSlotIds = filledSlotIdsFromCoverage(slotCoverage);
		const priority = clamp01(scored.priority + matchedSlotBoost);
		const promptCandidate = {
			id: hit.docId ?? hit.candidateId,
			surface,
			text,
			rawText: hit.text,
			scoringText,
			sourceRef: typeof metadata?.sourceRef === "string" && metadata.sourceRef.trim() ? metadata.sourceRef.trim() : hit.lineage.sourceRef,
			mergedSourceRefs: uniqueNonEmpty([...sourceRefsFromCandidateMetadata(metadata), hit.lineage.sourceRef]),
			observedAt: typeof metadata?.observedAt === "string" ? metadata.observedAt : void 0,
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
			role: "support"
		};
		candidates.push({
			...promptCandidate,
			slotEvidenceRole: promptEvidenceSlotRoleForCandidate(params.queryAnalysis, promptCandidate)
		});
	}
	candidates.push(...params.controlEvidence ?? []);
	for (const fact of params.facts) candidates.push(promptEvidenceFromRow({
		row: fact,
		surface: "fact",
		source: "selected",
		queryAnalysis: params.queryAnalysis
	}));
	for (const event of params.events) candidates.push(promptEvidenceFromRow({
		row: event,
		surface: isChunkEvidenceRow(event) ? "chunk" : "event",
		source: event.lineage?.sourceKind === "chunk" ? "support_ref" : "selected",
		queryAnalysis: params.queryAnalysis
	}));
	for (const snippet of params.selectedExactSnippets ?? []) {
		const text = cleanPromptEvidenceText(snippet.text);
		const scored = promptEvidencePriority({
			text,
			score: snippet.goalScore,
			surface: "snippet",
			source: snippet.source,
			queryAnalysis: params.queryAnalysis,
			explicitGoalScore: snippet.goalScore
		});
		candidates.push({
			id: snippet.snippetId,
			surface: "snippet",
			text,
			rawText: snippet.text,
			sourceRef: snippet.sourceRef,
			observedAt: void 0,
			excerptAnchors,
			lineage: snippet.lineage,
			priority: scored.priority,
			goalScore: scored.goalScore,
			semanticScore: scored.semanticScore,
			coverage: evidenceCoverageForText(params.queryAnalysis, text),
			slotCoverage: slotCoverageForText(params.queryAnalysis, text),
			filledSlotIds: filledSlotIdsForText(params.queryAnalysis, text),
			source: snippet.source,
			role: "support"
		});
	}
	candidates.push(...sourceExpansionPromptEvidenceCandidates({
		store: params.store,
		ctx: params.ctx,
		queryAnalysis: params.queryAnalysis,
		candidates
	}));
	const ordered = candidates.sort((left, right) => {
		const priorityDelta = right.priority - left.priority;
		const nonSubjectHitDelta = promptEvidenceNonSubjectSlotHitCount(params.queryAnalysis, right) - promptEvidenceNonSubjectSlotHitCount(params.queryAnalysis, left);
		if (nonSubjectHitDelta !== 0 && promptEvidenceRequiresAnswerSlotSupport(params.queryAnalysis) && Math.abs(priorityDelta) <= .32) return nonSubjectHitDelta;
		const answerSlotDelta = promptEvidenceAnswerSlotStrength(params.queryAnalysis, right) - promptEvidenceAnswerSlotStrength(params.queryAnalysis, left);
		if (Math.abs(answerSlotDelta) >= .08 && Math.abs(priorityDelta) <= .3) return answerSlotDelta;
		const specificityDelta = promptEvidenceAnswerSpecificity(params.queryAnalysis, right) - promptEvidenceAnswerSpecificity(params.queryAnalysis, left);
		if (Math.abs(specificityDelta) >= .08 && Math.abs(priorityDelta) <= .22) return specificityDelta;
		if (Math.abs(priorityDelta) > .03) return priorityDelta;
		if (specificityDelta !== 0) return specificityDelta;
		if ((right.semanticScore ?? 0) !== (left.semanticScore ?? 0)) return (right.semanticScore ?? 0) - (left.semanticScore ?? 0);
		if (priorityDelta !== 0) return priorityDelta;
		if (right.goalScore !== left.goalScore) return right.goalScore - left.goalScore;
		const filledSlotDelta = (right.filledSlotIds?.length ?? 0) - (left.filledSlotIds?.length ?? 0);
		if (filledSlotDelta !== 0) return filledSlotDelta;
		const authorityDelta = promptEvidenceAuthorityRank(right) - promptEvidenceAuthorityRank(left);
		if (authorityDelta !== 0) return authorityDelta;
		const observedDelta = (right.observedAt ?? "").localeCompare(left.observedAt ?? "");
		if (observedDelta !== 0) return observedDelta;
		if (right.priority !== left.priority) return right.priority - left.priority;
		return right.goalScore - left.goalScore;
	});
	const selected = [];
	const dropped = [];
	const seen = /* @__PURE__ */ new Set();
	for (const candidate of ordered) {
		const key = [
			candidate.sourceRef ?? "",
			candidate.id,
			normalizeText(candidate.text).slice(0, 240)
		].join("|");
		if (seen.has(key)) {
			const sameSourceIndex = selected.findIndex((entry) => {
				return (entry.sourceRef ?? "") === (candidate.sourceRef ?? "") && entry.id === candidate.id && normalizeText(entry.text).slice(0, 240) === normalizeText(candidate.text).slice(0, 240);
			});
			if (sameSourceIndex >= 0) {
				const existing = selected[sameSourceIndex];
				if (shouldReplacePromptEvidence(existing, candidate, params.queryAnalysis)) {
					selected[sameSourceIndex] = mergePromptEvidenceSourceRefs(candidate, existing);
					dropped.push({
						...existing,
						role: "alternate",
						dropReason: "duplicate-source-or-text"
					});
					continue;
				}
				selected[sameSourceIndex] = mergePromptEvidenceSourceRefs(existing, candidate);
			}
			dropped.push({
				...candidate,
				role: "alternate",
				dropReason: "duplicate-source-or-text"
			});
			continue;
		}
		seen.add(key);
		selected.push(candidate);
	}
	const ranked = selected.map((candidate) => scorePromptEvidenceForInjection(params.queryAnalysis, candidate)).sort((left, right) => {
		const scoreDelta = (right.injectionScore ?? 0) - (left.injectionScore ?? 0);
		if (Math.abs(scoreDelta) > 1e-4) return scoreDelta;
		if (right.priority !== left.priority) return right.priority - left.priority;
		return right.goalScore - left.goalScore;
	});
	const injectionLimit = promptEvidenceInjectionLimit(params.queryAnalysis, ranked);
	const selectedWithRoles = ranked.map((candidate, index) => {
		const role = index < Math.max(8, injectionLimit) ? "support" : "alternate";
		const candidateMissingRequired = candidate.coverage?.missingRequired;
		const missingRequired = candidateMissingRequired && candidateMissingRequired.length > 0 ? candidateMissingRequired : evidenceCoverageForText(params.queryAnalysis, candidate.text).missingRequired;
		const selectionReason = missingRequired.length > 0 ? `candidate-stage-missing-required:${missingRequired.join(",")}` : `candidate-stage-ranked#${index + 1}`;
		return {
			...candidate,
			role,
			selectionReason,
			protectionReason: selectionReason
		};
	});
	const scoredDropped = dropped.map((candidate) => ({
		...scorePromptEvidenceForInjection(params.queryAnalysis, candidate),
		role: "alternate",
		selectionReason: candidate.dropReason ?? "dropped-before-ranking",
		protectionReason: candidate.dropReason ?? "dropped-before-ranking"
	}));
	return keepRetainedPromptEvidenceCandidates({
		queryAnalysis: params.queryAnalysis,
		candidates: [...selectedWithRoles, ...scoredDropped],
		cap: 64
	});
}
function buildEvidencePlanAudit(params) {
	const plan = params.queryAnalysis.evidencePlan;
	if (!plan || plan.slots.length === 0) return;
	const statsBySlotLayer = /* @__PURE__ */ new Map();
	for (const stat of params.candidateGenerationResult?.slotLayerStats ?? []) {
		const key = `${stat.slotId}:${stat.layer}`;
		const current = statsBySlotLayer.get(key) ?? {
			raw: 0,
			selected: 0,
			alternate: 0
		};
		current.raw += stat.rawCount;
		current.selected += stat.selectedCount;
		current.alternate += stat.alternateCount;
		statsBySlotLayer.set(key, current);
	}
	return {
		operation: plan.operation,
		slots: plan.slots.map((slot) => {
			const queriedLayers = evidencePlanLayersForSlot(slot);
			const layerCandidateCounts = {};
			for (const layer of queriedLayers) {
				const stats = statsBySlotLayer.get(`${slot.id}:${layer}`);
				if (stats) layerCandidateCounts[layer] = stats.raw;
			}
			const selectedEvidence = params.promptEvidence.filter((entry) => (entry.slotCoverage ?? []).some((coverage) => coverage.slotId === slot.id && (coverage.filled || coverage.coverageScore >= .3)));
			const filledEvidence = selectedEvidence.filter((entry) => (entry.filledSlotIds ?? []).includes(slot.id));
			const protectedEvidence = selectedEvidence.filter((entry) => entry.role === "protected" && !entry.dropReason);
			const injectedEvidence = params.injectedEvidence.filter((entry) => (entry.filledSlotIds ?? []).includes(slot.id));
			const bestCoverage = Math.max(0, ...selectedEvidence.flatMap((entry) => (entry.slotCoverage ?? []).filter((coverage) => coverage.slotId === slot.id).map((coverage) => coverage.coverageScore)));
			const missingFields = uniqueNonEmpty(selectedEvidence.flatMap((entry) => (entry.slotCoverage ?? []).filter((coverage) => coverage.slotId === slot.id).flatMap((coverage) => coverage.missingRequired)));
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
				sourceRefs: uniqueNonEmpty(selectedEvidence.flatMap((entry) => [entry.sourceRef, ...entry.mergedSourceRefs ?? []]).filter((ref) => typeof ref === "string" && ref.length > 0))
			};
		})
	};
}
function normalizedPromptEvidenceSourceRefs(entry) {
	return normalizeSourceRefs([entry.sourceRef, ...entry.mergedSourceRefs ?? []]);
}
function renderedPromptLineAudit(bundle) {
	return bundle.evidencePackets.filter((packet) => packet.injected && !packet.dropReason).sort((left, right) => (right.grade?.finalScore ?? right.coverage.confidence) - (left.grade?.finalScore ?? left.coverage.confidence) || right.coverage.confidence - left.coverage.confidence).slice(0, 6).flatMap((packet) => {
		const evidenceUnitIds = [
			...packet.answerUnits ?? [],
			...packet.contextUnits ?? [],
			...packet.supportUnits ?? []
		].map((unit) => unit.unitId);
		const refs = packet.allSourceRefs ?? packet.sourceRefs;
		return (packet.displayLines ?? [packet.primaryText]).filter((line) => line.trim().length > 0).map((line, index) => {
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
				evidenceUnitIds
			};
		});
	});
}
function isChunkEvidenceRow(row) {
	return row.lineage?.sourceKind === "chunk";
}
function supportRefsFromFactObjectValueJson(value) {
	const refs = value?.supportRefs;
	return Array.isArray(refs) ? uniqueNonEmpty(refs.filter((entry) => typeof entry === "string")) : [];
}
function supportRefsFromSelectedFacts(store, facts, queryAnchors) {
	const rankedFacts = facts.map((factRow) => ({
		factRow,
		anchorScore: candidateSurfaceAnchorScore(factRow, queryAnchors),
		score: factRow.score ?? 0
	})).sort((left, right) => {
		if (right.anchorScore !== left.anchorScore) return right.anchorScore - left.anchorScore;
		return right.score - left.score;
	});
	const bestAnchorScore = rankedFacts[0]?.anchorScore ?? 0;
	const selectedFacts = rankedFacts.filter((entry, index) => entry.anchorScore >= Math.max(.38, bestAnchorScore - .08) || index === 0 && bestAnchorScore < .38).slice(0, 3).map((entry) => entry.factRow);
	const refs = [];
	for (const factRow of selectedFacts) {
		const factId = factRow.lineage?.canonicalKind === "fact" ? factRow.lineage.canonicalId ?? factRow.lineage.sourceId : factRow.lineage?.sourceKind === "fact" ? factRow.lineage.sourceId : void 0;
		if (!factId) continue;
		const fact = store.factRepo.get(factId);
		refs.push(...supportRefsFromFactObjectValueJson(fact?.objectValueJson));
	}
	return uniqueNonEmpty(refs);
}
function chunkEvidenceRowFromConversationChunk(chunk, queryAnchors) {
	const text = normalizeSearchText(chunk.content.trim() || chunk.summary.trim());
	const anchorScore = queryAnchors.length > 0 ? queryAnchorSupport(text, queryAnchors) : 0;
	return toEvidenceRow({
		id: chunk.chunkId,
		text,
		score: Math.max(.36, .72 + anchorScore * .22),
		scope: chunk.scope,
		confidence: .82,
		observedAt: chunk.createdAt,
		sourceRef: chunk.sourceRef,
		provenance: chunk.role,
		lineage: {
			sourceKind: "chunk",
			sourceId: chunk.chunkId,
			sourceRef: chunk.sourceRef
		}
	});
}
function buildSupportRefChunkRows(params) {
	return dedupeEvidenceRows(params.store.chunkRepo.listBySourceRefs({
		agentId: params.ctx.agentId,
		scopes: params.ctx.scopes,
		sourceRefs: params.sourceRefs,
		limit: Math.max(params.limit * 3, 8)
	}).map((chunk) => ({
		row: chunkEvidenceRowFromConversationChunk(chunk, params.queryAnchors),
		anchorScore: params.queryAnchors.length > 0 ? queryAnchorSupport(chunk.content, params.queryAnchors) : 1
	})).filter((entry) => params.queryAnchors.length > 0 ? entry.anchorScore >= .22 : true).sort((left, right) => {
		if (right.anchorScore !== left.anchorScore) return right.anchorScore - left.anchorScore;
		return (right.row.score ?? 0) - (left.row.score ?? 0);
	}).map((entry) => entry.row), params.limit);
}
function queryEntityAnchorTerms(queryAnalysis) {
	return uniqueNonEmpty((queryAnalysis.queryEntities ?? []).map((entity) => entity.name).filter((value) => typeof value === "string" && value.trim().length > 0));
}
function queryAttributeAnchorTerms(queryAnalysis) {
	const entityTerms = new Set(queryEntityAnchorTerms(queryAnalysis).map((entry) => normalizeText(entry)));
	const slotTerms = (queryAnalysis.evidencePlan?.slots ?? []).flatMap((slot) => [
		slot.description,
		...slot.relationHints ?? [],
		...slot.capabilityQueries ?? [],
		...slot.requiredFields
	]);
	const goalTerms = (queryAnalysis.evidenceGoals ?? []).flatMap((goal) => [
		goal.goal,
		...goal.positiveQueries ?? [],
		...goal.focusAnchors ?? []
	]);
	return uniqueNonEmpty([
		...queryAnalysis.anchors ?? [],
		...slotTerms,
		...goalTerms
	], 24).filter((term) => {
		const normalized = normalizeText(term);
		return normalized && !entityTerms.has(normalized) && !isGenericRequiredField(normalized) && isMeaningfulRecallAnchor(term);
	});
}
function rowMentionsQueryEntity(row, entityTerms) {
	if (entityTerms.length === 0) return true;
	const normalizedText = normalizeText(row.text);
	return entityTerms.some((term) => {
		const normalized = normalizeText(term);
		return normalized.length > 0 && (normalizedText.includes(normalized) || projectNamesMatch(row.text, term) || semanticTextSimilarity(row.text, term) >= .46);
	});
}
function rowSupportsQueryAttribute(row, attributeTerms) {
	if (attributeTerms.length === 0) return true;
	return Math.max(queryAnchorSupport(row.text, attributeTerms), ...attributeTerms.map((term) => semanticTextSimilarity(row.text, term))) >= .24;
}
function observedAtMs(row) {
	const parsed = row.observedAt ? Date.parse(row.observedAt) : NaN;
	return Number.isFinite(parsed) ? parsed : 0;
}
function shouldApplyNewerSourceFactGuard(queryAnalysis) {
	const operation = queryAnalysis.evidencePlan?.operation.type;
	return queryAnalysis.queryShape.timeframe === "current" && (queryAnalysis.queryShape.granularity === "exact_detail" || queryAnalysis.answerGranularity === "detail") && (operation === "return_value" || queryAnalysis.answerMode === "attribute_lookup" || queryAnalysis.answerMode === "single_fact");
}
function newerSourceRowsForCurrentFactGuard(params) {
	if (!shouldApplyNewerSourceFactGuard(params.queryAnalysis) || params.facts.length === 0) return [];
	const newestFactObservedAt = Math.max(0, ...params.facts.map((row) => observedAtMs(row)));
	if (newestFactObservedAt <= 0) return [];
	const entityTerms = queryEntityAnchorTerms(params.queryAnalysis);
	const attributeTerms = queryAttributeAnchorTerms(params.queryAnalysis);
	return dedupeEvidenceRows(params.sourceRows.filter((row) => observedAtMs(row) > newestFactObservedAt).filter((row) => !isQuestionLike(row.text)).filter((row) => rowMentionsQueryEntity(row, entityTerms)).filter((row) => rowSupportsQueryAttribute(row, attributeTerms)).sort((left, right) => observedAtMs(right) - observedAtMs(left) || (right.score ?? 0) - (left.score ?? 0)), 3);
}
function mergeRecalledChunkSupport(params) {
	const merged = [];
	const seen = /* @__PURE__ */ new Set();
	for (const row of params.supportRows) {
		const sourceId = row.lineage?.sourceId;
		if (!sourceId || seen.has(sourceId)) continue;
		seen.add(sourceId);
		merged.push({
			id: sourceId,
			text: truncateText(row.text, 280)
		});
		if (merged.length >= params.limit) break;
	}
	if (merged.length < params.limit) for (let index = 0; index < params.projectedIds.length; index += 1) {
		const id = params.projectedIds[index];
		if (!id || seen.has(id)) continue;
		seen.add(id);
		merged.push({
			id,
			text: params.projectedTexts[index] ?? ""
		});
		if (merged.length >= params.limit) break;
	}
	return {
		ids: merged.map((entry) => entry.id),
		texts: merged.map((entry) => entry.text)
	};
}
function shouldUseWorkflowMainSurface(queryAnalysis) {
	return queryAnalysis.queryShape.evidenceNeed === "workflow_context";
}
function shouldRetainTasksInMainSurface(queryAnalysis) {
	if (shouldUseWorkflowMainSurface(queryAnalysis)) return true;
	return queryAnalysis.candidateSurfaces.includes("task");
}
function shouldRetainStatesInMainSurface(queryAnalysis) {
	return shouldUseWorkflowMainSurface(queryAnalysis) || queryAnalysis.candidateSurfaces.includes("state");
}
function shouldApplyCandidateAuthorityToMainSurface(route, queryAnalysis) {
	if (route.routeType === "factual") return true;
	if (route.routeType !== "mixed") return false;
	if (shouldUseWorkflowMainSurface(queryAnalysis)) return false;
	if (!queryAnalysis.candidateSurfaces.includes("fact") && !queryAnalysis.candidateSurfaces.includes("state")) return false;
	return (queryAnalysis.routeWeights.factual ?? 0) >= .24 && (queryAnalysis.answerGranularity === "detail" || queryAnalysis.evidenceFidelity === "high");
}
function isQuestionLikeTask(task) {
	const candidateResolution = typeof task.metadataJson?.candidateResolution === "string" && task.metadataJson.candidateResolution.trim() ? task.metadataJson.candidateResolution.trim() : "";
	const stableSummary = semanticTaskSummaryText(task) ?? "";
	return isQuestionLike(task.title) || isQuestionLike(candidateResolution) || isQuestionLike(stableSummary) || isQuestionLike(taskMetadataValue(task, "currentTask"));
}
function resolveChunkMentionedProject(chunk, knownProjects) {
	for (const candidate of knownProjects) {
		if (!candidate) continue;
		if (chunk.content.includes(candidate) || chunk.summary.includes(candidate)) return candidate;
	}
}
function recentProjectMentionSequence(params) {
	const mentions = [];
	for (const chunk of params.chunks) {
		const project = resolveChunkMentionedProject(chunk, params.knownProjects);
		if (!project) continue;
		if (mentions.length > 0 && projectNamesMatch(mentions.at(-1), project)) continue;
		mentions.push(project);
	}
	return uniqueNonEmpty(mentions);
}
function isMeaningfulRecallAnchor(anchor) {
	const trimmed = anchor.trim();
	const normalized = normalizeText(trimmed);
	if (!normalized) return false;
	if ([...trimmed].some((char) => /\p{Script=Han}/u.test(char))) return trimmed.length >= 2;
	return normalized.length >= 4;
}
function appendAnchorsToQuery(query, anchors) {
	const missing = anchors.filter((anchor) => !normalizeText(query).includes(normalizeText(anchor)));
	return missing.length > 0 ? `${query} ${missing.join(" ")}`.trim() : query;
}
function resolveReferentialQueryAnchors(store, ctx, query) {
	const reasons = [];
	const recentTasks = uniqueTasksById([...store.taskRepo.listRecent({
		agentId: ctx.agentId,
		scopes: ctx.scopes,
		limit: 10,
		sessionKey: ctx.sessionKey
	}), ...store.taskRepo.listRecent({
		agentId: ctx.agentId,
		scopes: ctx.scopes,
		limit: 10
	})].sort((left, right) => right.updatedAt.localeCompare(left.updatedAt))).filter((task) => !isQuestionLikeTask(task));
	const recentChunks = store.chunkRepo.listRecentActive({
		agentId: ctx.agentId,
		scopes: ctx.scopes,
		limit: 24,
		sessionKey: ctx.sessionKey
	});
	const activeProjectState = store.stateRepo.get({
		agentId: ctx.agentId,
		scopes: ctx.scopes,
		key: "project.active_project",
		now: ctx.now,
		readEpoch: ctx.readEpoch
	})[0];
	const authoritativeActiveProject = typeof activeProjectState?.valueJson.project === "string" && activeProjectState.valueJson.project.trim() ? activeProjectState.valueJson.project.trim() : "";
	const projectCandidates = uniqueNonEmpty([authoritativeActiveProject, ...recentTasks.map((task) => taskProject(task))]);
	const projectMentionSequence = recentProjectMentionSequence({
		chunks: recentChunks.filter((chunk) => chunk.role === "user"),
		knownProjects: projectCandidates
	});
	const activeTasks = recentTasks.filter((task) => task.status === "active");
	const activeTask = activeTasks[0] ?? recentTasks[0];
	const activeProject = authoritativeActiveProject || (activeTask ? taskProject(activeTask) : "");
	const anchors = [];
	if (/(?:前一个项目|上一个项目|previous project|last project)/iu.test(query)) {
		const previousProject = projectMentionSequence[1] ?? projectCandidates.find((candidate) => normalizeText(candidate) !== normalizeText(activeProject)) ?? projectCandidates[1] ?? "";
		if (previousProject) {
			anchors.push(previousProject);
			reasons.push(`resolved-previous-project:${previousProject}`);
		}
	}
	const sideHints = [...query.matchAll(/([\p{Script=Han}\p{L}\p{N}_-]{1,12})那边/gu)].map((match) => match[1]?.trim() ?? "").filter((hint) => hint && hint !== "这" && hint !== "那" && hint !== "那个" && hint !== "这个" && hint !== "项目" && hint !== "前一个");
	for (const hint of sideHints) {
		const ranked = recentTasks.map((task) => {
			const references = uniqueNonEmpty([
				taskProject(task),
				taskMetadataValue(task, "currentTask"),
				task.title,
				typeof task.metadataJson?.candidateResolution === "string" ? task.metadataJson.candidateResolution : "",
				semanticTaskSummaryText(task) ?? ""
			]);
			return {
				task,
				references,
				score: references.reduce((best, reference) => Math.max(best, reference.includes(hint) || hint.includes(reference) ? .96 : queryAnchorSupport(reference, [hint])), 0)
			};
		}).filter((entry) => entry.score >= .2).sort((left, right) => right.score - left.score);
		const best = ranked[0];
		const second = ranked[1];
		let selectedTask = best?.task;
		let selectedReason = best ? `resolved-side-reference:${hint}` : "";
		if (!best || best.score < .3 && second && best.score - second.score < .08 || best.score < .2) {
			selectedTask = activeTasks.length === 1 ? activeTasks[0] : void 0;
			selectedReason = `resolved-side-reference:fallback-active:${hint}`;
		}
		if (!selectedTask) continue;
		const selected = uniqueNonEmpty([
			taskProject(selectedTask),
			taskMetadataValue(selectedTask, "currentTask"),
			selectedTask.title,
			selectedTask.summary
		])[0];
		if (!selected) continue;
		anchors.push(selected);
		reasons.push(`${selectedReason}->${selected}`);
	}
	return {
		anchors: uniqueNonEmpty(anchors),
		reasons
	};
}
function uniqueTasksById(tasks) {
	const seen = /* @__PURE__ */ new Set();
	const ordered = [];
	for (const task of tasks) {
		if (seen.has(task.taskId)) continue;
		seen.add(task.taskId);
		ordered.push(task);
	}
	return ordered;
}
function buildFocusedQueries(queryAnalysis, originalQuery, searchQuery) {
	const normalizedFallback = sanitizeFocusedRecallQuery(originalQuery, queryAnalysis.focusedQuery || searchQuery);
	return {
		workflow: normalizedFallback,
		factual: normalizedFallback,
		temporal: normalizedFallback,
		explanatory: normalizedFallback
	};
}
function isSnapshotFactualCandidateSummary(summary) {
	const { label } = splitLabelValue(summary);
	return Boolean(label && isSnapshotFactualStateKey(label));
}
function routeRoleFit(routeType, role, summary) {
	switch (routeType) {
		case "workflow": return role === "state" || role === "task" ? 1 : role === "event" ? .72 : .42;
		case "factual": return role === "fact" ? 1 : role === "state" && isSnapshotFactualCandidateSummary(summary) ? .94 : role === "chunk" ? .66 : role === "graph" ? .6 : .4;
		case "temporal": return role === "event" || role === "chunk" ? 1 : role === "state" ? .64 : .4;
		case "explanatory": return role === "graph" ? 1 : role === "fact" ? .84 : role === "chunk" ? .68 : .42;
	}
}
function buildScoreDrivenRouteEvidenceDecision(routeType, candidates, judgmentMode) {
	if (candidates.length === 0) return {
		relevant: [],
		sufficient: false,
		support: 0,
		reason: `surface ${routeType} evidence: no candidates`,
		judgmentMode
	};
	const ordered = [...candidates.map((candidate) => {
		const fit = routeRoleFit(routeType, candidate.role, candidate.summary);
		const confidence = typeof candidate.confidence === "number" ? clamp01(candidate.confidence) : .6;
		return {
			candidate,
			fit,
			weightedScore: clamp01(candidate.score * (.72 + fit * .28) * (.88 + confidence * .12)),
			confidence
		};
	})].sort((left, right) => right.weightedScore - left.weightedScore);
	const top = ordered[0]?.weightedScore ?? 0;
	const topThree = ordered.slice(0, 3);
	const averageTop = topThree.reduce((sum, entry) => sum + entry.weightedScore, 0) / Math.max(topThree.length, 1);
	const roleFit = topThree.reduce((sum, entry) => sum + entry.fit, 0) / Math.max(topThree.length, 1);
	const confidence = topThree.reduce((sum, entry) => sum + entry.confidence, 0) / Math.max(topThree.length, 1);
	const coverage = Math.min(1, ordered.length / 3);
	const aggregateSupport = clamp01(top * .54 + averageTop * .2 + roleFit * .12 + confidence * .08 + coverage * .06);
	const support = clamp01(Math.max(aggregateSupport, top * (routeType === "temporal" || routeType === "explanatory" ? .94 : routeType === "workflow" ? .9 : .88)));
	const relevantFloor = Math.max(.42, top * .72);
	const relevant = ordered.filter((entry) => entry.weightedScore >= relevantFloor).slice(0, 3).map((entry) => entry.candidate.index);
	return {
		relevant,
		sufficient: support >= .46 && (top >= .44 || relevant.length >= 2),
		support,
		reason: `surface ${routeType} evidence: top=${top.toFixed(2)} avg=${averageTop.toFixed(2)} count=${ordered.length}`,
		judgmentMode
	};
}
/**
* Score-based relevance filter that replaces the LLM `filterRelevant` call.
* Uses the hybrid search score combined with semantic text similarity to
* re-rank and trim low-relevance hits.
*/
function scoreBasedRelevanceFilter(hits, _query, maxResults) {
	const scored = hits.slice(0, maxResults).map((hit) => {
		return {
			hit,
			combined: hit.score
		};
	});
	scored.sort((left, right) => right.combined - left.combined);
	const filtered = scored.filter((entry) => entry.combined >= .25);
	if (filtered.length === 0 && scored.length > 0) return scored.slice(0, 3).map((entry) => entry.hit);
	return filtered.map((entry) => entry.hit);
}
function candidateRoleShare(candidates, role) {
	const topCandidates = candidates.slice(0, 3);
	const totalScore = topCandidates.reduce((sum, candidate) => sum + Math.max(candidate.score, 0), 0);
	if (totalScore <= 0) return 0;
	return clamp01(topCandidates.reduce((sum, candidate) => sum + (candidate.role === role ? Math.max(candidate.score, 0) : 0), 0) / totalScore);
}
function buildPriorWeights(queryAnalysis) {
	const weights = {
		workflow: .25,
		factual: .25,
		temporal: .25,
		explanatory: .25
	};
	for (const route of PRIMARY_ROUTE_TYPES) if (typeof queryAnalysis.routeWeights[route] === "number") weights[route] = Math.max(weights[route], clamp01(queryAnalysis.routeWeights[route] ?? 0));
	return weights;
}
function allocateBudgetByWeight(entries, total) {
	const allocations = {
		workflow: 0,
		factual: 0,
		temporal: 0,
		explanatory: 0
	};
	const active = entries.filter((entry) => entry.activated);
	const budget = Math.max(0, Math.floor(total));
	if (budget <= 0 || active.length === 0) return allocations;
	const totalWeight = active.reduce((sum, entry) => sum + Math.max(entry.weight, .001), 0);
	if (totalWeight <= 0) {
		const evenShare = Math.floor(budget / active.length);
		let remainder = budget - evenShare * active.length;
		for (const entry of active) {
			allocations[entry.routeType] = evenShare + (remainder > 0 ? 1 : 0);
			if (remainder > 0) remainder -= 1;
		}
		return allocations;
	}
	const minBudgetSum = active.reduce((sum, entry) => sum + Math.max(0, entry.minBudget), 0);
	if (minBudgetSum > budget) {
		const provisional = active.map((entry) => {
			const weighted = budget * Math.max(entry.weight, .001) / totalWeight;
			const floor = Math.floor(weighted);
			return {
				routeType: entry.routeType,
				remainder: weighted - floor,
				budget: floor
			};
		});
		let remainder = budget - provisional.reduce((sum, entry) => sum + entry.budget, 0);
		provisional.sort((left, right) => right.remainder - left.remainder).forEach((entry, index) => {
			allocations[entry.routeType] = entry.budget + (index < remainder ? 1 : 0);
		});
		return allocations;
	}
	for (const entry of active) allocations[entry.routeType] = Math.max(0, entry.minBudget);
	const remainderBudget = budget - minBudgetSum;
	if (remainderBudget <= 0) return allocations;
	const proportional = active.map((entry) => {
		const weighted = remainderBudget * Math.max(entry.weight, .001) / totalWeight;
		const floor = Math.floor(weighted);
		allocations[entry.routeType] += floor;
		return {
			routeType: entry.routeType,
			remainder: weighted - floor
		};
	});
	let leftover = budget - active.reduce((sum, entry) => sum + allocations[entry.routeType], 0);
	proportional.sort((left, right) => right.remainder - left.remainder).forEach((entry, index) => {
		if (index < leftover) allocations[entry.routeType] += 1;
	});
	return allocations;
}
function buildRouteDecision(evaluations, reasons, intent) {
	const ordered = [...evaluations].sort((left, right) => right.finalScore - left.finalScore);
	const top = ordered[0];
	const second = ordered[1];
	if (!top) return {
		routeType: "unknown",
		routeConfidence: .25,
		reasons: [...reasons, "route:unknown:insufficient-evidence"]
	};
	if (top.finalScore < .38 && !(intent?.preferChunkEvidence && top.evidenceSupport >= .34 && top.candidateCount >= 1)) return {
		routeType: "unknown",
		routeConfidence: Math.max(.25, top.finalScore),
		reasons: [...reasons, "route:unknown:insufficient-evidence"]
	};
	if (intent?.preferProjectSnapshot) {
		const factual = ordered.find((entry) => entry.routeType === "factual");
		if (factual && factual.evidenceSupport >= .42 && factual.finalScore >= .42 && (top.routeType === "factual" || top.finalScore - factual.finalScore < .22)) return {
			routeType: "factual",
			routeConfidence: Math.max(.5, factual.finalScore),
			reasons: [...reasons, "route:factual:project-profile-snapshot-authoritative"]
		};
	}
	if (intent?.preferHistoricalLookup) {
		const temporal = ordered.find((entry) => entry.routeType === "temporal");
		if (temporal && temporal.evidenceSupport >= .42 && temporal.finalScore >= .42 && (top.routeType === "temporal" || top.finalScore - temporal.finalScore < .18)) return {
			routeType: "temporal",
			routeConfidence: Math.max(.5, temporal.finalScore),
			reasons: [...reasons, "route:temporal:historical-query-authoritative"]
		};
	}
	if (intent?.preferRelationalLookup) {
		const explanatory = ordered.find((entry) => entry.routeType === "explanatory");
		if (explanatory && explanatory.evidenceSupport >= .38 && explanatory.finalScore >= .38 && (top.routeType === "explanatory" || top.finalScore - explanatory.finalScore < .18)) return {
			routeType: "explanatory",
			routeConfidence: Math.max(.5, explanatory.finalScore),
			reasons: [...reasons, "route:explanatory:relation-query-authoritative"]
		};
	}
	const explanatory = ordered.find((entry) => entry.routeType === "explanatory");
	if (explanatory && explanatory.graphRoleShare >= .28 && explanatory.evidenceSupport >= .36 && top.routeType === "workflow" && top.finalScore - explanatory.finalScore < .12) return {
		routeType: "explanatory",
		routeConfidence: Math.max(.5, explanatory.finalScore),
		reasons: [...reasons, "route:explanatory:graph-supported-close-rival"]
	};
	if (intent?.preferWorkflowDeictic) {
		const workflow = ordered.find((entry) => entry.routeType === "workflow");
		if (workflow && workflow.evidenceSupport >= .48 && workflow.finalScore >= .48 && (top.routeType === "workflow" || top.finalScore - workflow.finalScore < .16)) return {
			routeType: "workflow",
			routeConfidence: Math.max(.5, workflow.finalScore),
			reasons: [...reasons, "route:workflow:deictic-reference-authoritative"]
		};
	}
	if (second && top.evidenceSufficient && second.evidenceSufficient && top.evidenceSupport >= .62 && second.evidenceSupport >= .58 && second.evidenceSupport / Math.max(top.evidenceSupport, .01) >= .8) {
		const collapsedPrimaryReason = collapseMixedRouteIntoPrimary(top, second);
		if (collapsedPrimaryReason) return {
			routeType: top.routeType,
			routeConfidence: Math.max(.5, top.finalScore),
			reasons: [...reasons, collapsedPrimaryReason]
		};
		return {
			routeType: "mixed",
			routeConfidence: clamp01((top.evidenceSupport + second.evidenceSupport) / 2),
			reasons: [...reasons, `route:mixed:${top.routeType}+${second.routeType}`]
		};
	}
	return {
		routeType: top.routeType,
		routeConfidence: Math.max(.5, top.finalScore),
		reasons: [...reasons, `route:${top.routeType}:${top.reason}`]
	};
}
function buildQueryDrivenRouteDecision(evaluations, queryAnalysis, reasons) {
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
			reason: "query-driven route fallback"
		}
	})).sort((left, right) => {
		if (right.queryWeight !== left.queryWeight) return right.queryWeight - left.queryWeight;
		return right.evaluation.finalScore - left.evaluation.finalScore;
	});
	const top = ranked[0];
	const second = ranked[1];
	if (!top) return {
		routeType: "unknown",
		routeConfidence: .25,
		reasons: [...reasons, "route:unknown:missing-query-weights"]
	};
	const lowContextGenericQuery = queryAnalysis.turnMode === "mixed" && queryAnalysis.queryShape.timeframe === "timeless" && queryAnalysis.queryShape.evidenceNeed === "canonical_state" && queryAnalysis.evidenceFidelity === "low" && queryAnalysis.answerGranularity === "summary" && queryAnalysis.anchors.length <= 1;
	const confidence = clamp01(top.queryWeight * .76 + top.evaluation.evidenceSupport * .16 + top.evaluation.finalScore * .08);
	if (lowContextGenericQuery ? top.evaluation.evidenceSupport < .28 : top.queryWeight < .22 && top.evaluation.evidenceSupport < .28) return {
		routeType: "unknown",
		routeConfidence: Math.max(.25, confidence),
		reasons: [...reasons, lowContextGenericQuery ? "route:unknown:generic-query-with-weak-support" : "route:unknown:query-weights-insufficient"]
	};
	if (second && second.queryWeight >= .28 && top.queryWeight - second.queryWeight < .08 && top.evaluation.finalScore - second.evaluation.finalScore < .12) return {
		routeType: "mixed",
		routeConfidence: clamp01((confidence + second.queryWeight) / 2),
		reasons: [...reasons, `route:mixed:query-compiler-close-rival:${top.routeType}+${second.routeType}`]
	};
	return {
		routeType: top.routeType,
		routeConfidence: Math.max(.5, confidence),
		reasons: [...reasons, `route:${top.routeType}:query-compiler-authority:q=${top.queryWeight.toFixed(2)} support=${top.evaluation.evidenceSupport.toFixed(2)}`]
	};
}
function collapseMixedRouteIntoPrimary(top, second) {
	if (top.routeType === "explanatory" && second.routeType === "factual") {
		const supportGap = top.evidenceSupport - second.evidenceSupport;
		const finalGap = top.finalScore - second.finalScore;
		const graphLead = top.graphRoleShare - second.graphRoleShare;
		if (top.graphRoleShare >= .28 && graphLead >= .08 && (supportGap >= .04 || finalGap >= .07)) return "route:explanatory:structural-primary-over-factual";
	}
	if (top.routeType === "explanatory" && second.routeType === "temporal") {
		const supportGap = top.evidenceSupport - second.evidenceSupport;
		const finalGap = top.finalScore - second.finalScore;
		const graphLead = top.graphRoleShare - second.graphRoleShare;
		const structuralPrimary = top.graphRoleShare >= .32;
		if ((second.topRole === "event" || second.topRole === "chunk") && structuralPrimary && graphLead >= .16 && (supportGap >= .03 || finalGap >= .1)) return "route:explanatory:structural-primary-over-temporal";
	}
	if (top.routeType === "explanatory" && second.routeType === "workflow") {
		const supportGap = top.evidenceSupport - second.evidenceSupport;
		const finalGap = top.finalScore - second.finalScore;
		const graphLead = top.graphRoleShare - second.graphRoleShare;
		const structuralPrimary = top.graphRoleShare >= .3;
		if ((second.topRole === "task" || second.topRole === "state") && structuralPrimary && graphLead >= .16 && (supportGap >= .05 || finalGap >= .1)) return "route:explanatory:structural-primary-over-workflow";
	}
	return null;
}
function reservedObjectiveRoutes(evaluations, routeDecision) {
	const ordered = [...evaluations].sort((left, right) => right.finalScore - left.finalScore);
	const top = ordered[0];
	const second = ordered[1];
	if (routeDecision.routeType === "unknown") return new Set(PRIMARY_ROUTE_TYPES);
	if (routeDecision.routeType === "mixed") return new Set([top?.routeType, second?.routeType].filter(Boolean));
	return new Set(routeDecision.routeType === "workflow" || routeDecision.routeType === "factual" || routeDecision.routeType === "temporal" || routeDecision.routeType === "explanatory" ? [routeDecision.routeType] : []);
}
function buildObjectiveBudgets(ctx, evaluations, routeDecision) {
	const reservedRoutes = reservedObjectiveRoutes(evaluations, routeDecision);
	const minWeight = clamp01(ctx.config.advanced.recallObjectiveMinWeight);
	const totalObjectBudget = Math.max(4, Math.floor(ctx.config.advanced.recallTotalObjectBudget));
	const promptFloor = Math.max(0, Math.min(ctx.config.maxInjectedChars, Math.floor(ctx.config.advanced.recallPromptBudgetFloor)));
	const reservedBackgroundChars = Math.max(0, Math.min(Math.max(0, ctx.config.maxInjectedChars - promptFloor), Math.floor(ctx.config.advanced.recallBackgroundCharReserve)));
	const totalPromptChars = Math.max(0, ctx.config.maxInjectedChars - reservedBackgroundChars);
	const effectivePromptFloor = Math.max(0, Math.min(totalPromptChars, Math.floor(ctx.config.advanced.recallPromptBudgetFloor)));
	const seeded = evaluations.map((entry) => {
		const reserved = reservedRoutes.has(entry.routeType);
		const activated = routeDecision.routeType === "unknown" ? true : reserved || entry.finalScore >= minWeight;
		const rawScore = activated ? Math.max(entry.finalScore, reserved ? minWeight : 0) : 0;
		return {
			...entry,
			reserved,
			activated,
			rawScore
		};
	});
	const normalizedSeed = seeded.filter((entry) => entry.activated).reduce((sum, entry) => sum + entry.rawScore, 0) > 0 ? seeded : seeded.map((entry) => ({
		...entry,
		activated: true,
		rawScore: 1
	}));
	const weightTotal = normalizedSeed.reduce((sum, entry) => sum + entry.rawScore, 0);
	const weighted = normalizedSeed.map((entry) => ({
		...entry,
		weight: weightTotal > 0 ? entry.rawScore / weightTotal : 0
	}));
	const overflowReserve = Math.min(totalObjectBudget, Math.floor(totalObjectBudget * clamp01(ctx.config.advanced.recallObjectiveOverflowRatio)));
	const objectAllocations = allocateBudgetByWeight(weighted.map((entry) => ({
		routeType: entry.routeType,
		activated: entry.activated,
		weight: entry.weight,
		minBudget: entry.reserved ? 1 : 0
	})), Math.max(0, totalObjectBudget - overflowReserve));
	const promptAllocations = allocateBudgetByWeight(weighted.map((entry) => ({
		routeType: entry.routeType,
		activated: entry.activated,
		weight: entry.weight,
		minBudget: entry.activated ? effectivePromptFloor : 0
	})), totalPromptChars);
	return {
		objectiveBudgets: Object.fromEntries(weighted.map((entry) => [entry.routeType, {
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
				`support=${entry.evidenceSupport.toFixed(2)}`
			]
		}])),
		totalObjectBudget,
		totalPromptChars,
		reservedBackgroundChars,
		globalOverflowObjects: totalObjectBudget - Object.values(objectAllocations).reduce((sum, budget) => sum + budget, 0)
	};
}
function selectionObjectiveForRoute(routeType, query, now, currentSessionKey) {
	return createMemorySelectionObjective(routeType, query, now, currentSessionKey);
}
function buildRouteEvidencePacks(store, ctx, focusedQueries, hybridHits) {
	const supersetObjective = selectionObjectiveForRoute("temporal", focusedQueries.temporal, ctx.now, ctx.sessionKey);
	supersetObjective.includeHistorical = true;
	const allObjects = collectMemoryObjects(store, ctx, supersetObjective, hybridHits);
	return Object.fromEntries(PRIMARY_ROUTE_TYPES.map((route) => {
		const objective = selectionObjectiveForRoute(route, focusedQueries[route], ctx.now, ctx.sessionKey);
		return [route, {
			routeType: route,
			query: focusedQueries[route],
			candidates: toRouteEvidenceCandidatesFromObjects(scheduleMemoryObjects(allObjects, objective), 10)
		}];
	}));
}
async function planRecallAllocation(store, ctx, query, searchQuery, hybridHits, queryAnalysis) {
	const resolvedQueryAnalysis = queryAnalysis ?? analyzeRecallQuery(query);
	const focusedQueries = buildFocusedQueries(resolvedQueryAnalysis, query, searchQuery);
	const packs = buildRouteEvidencePacks(store, ctx, focusedQueries, hybridHits);
	const priorWeights = buildPriorWeights(resolvedQueryAnalysis);
	const queryDrivenAuthority = ctx.config.advanced.enableQueryCompiler;
	const queryBlendWeight = queryDrivenAuthority ? .72 : resolvedQueryAnalysis.queryShape.timeframe === "timeless" && resolvedQueryAnalysis.queryShape.granularity === "summary" ? .2 : .32;
	const priorBlendWeight = queryDrivenAuthority ? .06 : .1;
	const evidenceBlendWeight = 1 - priorBlendWeight - queryBlendWeight;
	const evaluations = PRIMARY_ROUTE_TYPES.map((routeType) => {
		const pack = packs[routeType];
		const evidence = buildScoreDrivenRouteEvidenceDecision(routeType, pack.candidates, "score-driven");
		const queryWeight = resolvedQueryAnalysis.routeWeights[routeType] ?? 0;
		const finalScore = clamp01(priorWeights[routeType] * priorBlendWeight + evidence.support * evidenceBlendWeight + queryWeight * queryBlendWeight);
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
			reason: `${evidence.reason};query=${queryWeight.toFixed(2)}`
		};
	});
	const ordered = [...evaluations].sort((left, right) => right.finalScore - left.finalScore);
	const reasons = [...PRIMARY_ROUTE_TYPES.map((routeType) => `queryWeight:${routeType}:${(resolvedQueryAnalysis.routeWeights[routeType] ?? 0).toFixed(2)}`), ...ordered.map((entry) => `evidence:${entry.routeType}:final=${entry.finalScore.toFixed(2)} support=${entry.evidenceSupport.toFixed(2)} candidates=${entry.candidateCount}`)];
	const routeDecision = queryDrivenAuthority ? buildQueryDrivenRouteDecision(evaluations, resolvedQueryAnalysis, reasons) : buildRouteDecision(evaluations, reasons, {
		preferWorkflowDeictic: resolvedQueryAnalysis.queryShape.evidenceNeed === "workflow_context",
		preferProjectSnapshot: resolvedQueryAnalysis.queryShape.timeframe === "current",
		preferHistoricalLookup: resolvedQueryAnalysis.queryShape.timeframe === "historical" || resolvedQueryAnalysis.queryShape.timeframe === "compare",
		preferRelationalLookup: resolvedQueryAnalysis.queryShape.evidenceNeed === "relation",
		preferChunkEvidence: resolvedQueryAnalysis.queryShape.evidenceNeed === "chunk" || resolvedQueryAnalysis.evidenceFidelity === "high"
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
		globalOverflowObjects: budgets.globalOverflowObjects
	};
}
function renderSection(title, lines) {
	if (lines.length === 0) return "";
	return `## ${title}\n${lines.join("\n")}`;
}
function truncateLinesToBudget(lines, maxChars) {
	if (maxChars <= 0 || lines.length === 0) return [];
	const selected = [];
	let used = 0;
	for (const line of lines) {
		const normalized = line.trim();
		if (!normalized) continue;
		const addition = (selected.length > 0 ? 1 : 0) + normalized.length;
		if (used + addition <= maxChars) {
			selected.push(normalized);
			used += addition;
			continue;
		}
		if (selected.length === 0) selected.push(truncateText(normalized, Math.max(24, maxChars)));
		break;
	}
	return selected;
}
function renderBudgetedSection(title, lines, maxChars) {
	if (maxChars <= 0 || lines.length === 0) return "";
	const titleCost = title.length + 5;
	const rendered = renderSection(title, truncateLinesToBudget(lines, Math.max(24, maxChars - titleCost)));
	return rendered.length <= maxChars ? rendered : truncateText(rendered, maxChars);
}
function splitBudget(total, firstWeight) {
	if (total <= 0) return [0, 0];
	const first = Math.max(0, Math.min(total, Math.floor(total * firstWeight)));
	return [first, total - first];
}
function deriveProjectionOptions(ctx, plan, query, route, queryAnalysis) {
	const queryShape = queryAnalysis?.queryShape;
	const allowHistoricalFacts = queryShape?.timeframe === "historical" || queryShape?.timeframe === "compare";
	const workflowBudget = plan.objectiveBudgets.workflow.objectBudget;
	const factualBudget = plan.objectiveBudgets.factual.objectBudget;
	const temporalBudget = plan.objectiveBudgets.temporal.objectBudget;
	const explanatoryBudget = plan.objectiveBudgets.explanatory.objectBudget;
	let [stateLimit, taskLimit] = workflowBudget > 0 ? splitBudget(Math.max(2, workflowBudget), .5) : [0, 0];
	if (route.routeType === "workflow" && workflowBudget > 0) {
		stateLimit = Math.max(stateLimit, 3);
		taskLimit = Math.max(taskLimit, 1);
	}
	const alternateLimit = Math.max(0, Math.min(6, plan.globalOverflowObjects + (plan.routeDecision.routeType === "mixed" || plan.routeDecision.routeType === "unknown" ? 2 : 1)));
	const eventLimit = temporalBudget > 0 ? Math.max(1, temporalBudget) : 0;
	return {
		routeType: route.routeType,
		routeConfidence: route.routeConfidence,
		allowHistoricalFacts,
		preferTemporalEvents: route.routeType === "temporal" || queryShape?.timeframe === "historical" || queryShape?.timeframe === "compare",
		stateLimit,
		taskLimit,
		factLimit: factualBudget > 0 || explanatoryBudget > 0 ? Math.max(factualBudget, explanatoryBudget > 0 ? 2 : 0, factualBudget > 0 ? 1 : 0, allowHistoricalFacts ? 2 : 0) : 0,
		eventLimit,
		graphLimit: explanatoryBudget > 0 ? Math.max(1, explanatoryBudget) : 0,
		alternateLimit,
		recallChunkBudget: eventLimit > 0 ? Math.min(ctx.config.advanced.recallChunkBudget, eventLimit, queryAnalysis?.evidenceFidelity === "high" || queryShape?.timeframe === "compare" || queryShape?.granularity === "exact_detail" ? 3 : eventLimit) : 0
	};
}
async function retrieveEvidence(store, ctx, query, searchQuery = query, auditOptions = {}) {
	const behavioralGuidance = collectBehavioralGuidance(store, ctx);
	const queryAnalysis = auditOptions.queryAnalysis ?? (ctx.config.advanced.enableQueryCompiler ? await compileQuery({
		query,
		ctx,
		reasoner: store.reasoner
	}) : analyzeRecallQuery(query));
	const referentResolution = resolveReferentialQueryAnchors(store, ctx, query);
	const retrievalQuery = appendAnchorsToQuery(query, referentResolution.anchors);
	const retrievalSearchQuery = appendAnchorsToQuery(searchQuery, referentResolution.anchors);
	const snapshotFocus = queryAnalysis.queryShape.timeframe === "current";
	const candidateGenerationResult = await generateCandidates(store, ctx, {
		...queryAnalysis,
		focusedQuery: retrievalSearchQuery,
		anchors: uniqueNonEmpty([...queryAnalysis.anchors, ...referentResolution.anchors])
	});
	const rawHybridHits = candidateGenerationResult ? candidateGenerationResult.searchHits : await store.retrievalBackend.hybridSearch({
		agentId: ctx.agentId,
		scopes: ctx.scopes,
		query: retrievalSearchQuery,
		limit: 16,
		readEpoch: ctx.readEpoch
	});
	let hybridHits = rawHybridHits.filter((hit) => !shouldSuppressRecallText(hit.text));
	if (hybridHits.length !== rawHybridHits.length) {}
	const allocationPlan = await planRecallAllocation(store, ctx, retrievalQuery, retrievalSearchQuery, hybridHits, queryAnalysis);
	if (referentResolution.anchors.length > 0) for (const routeType of PRIMARY_ROUTE_TYPES) allocationPlan.focusedQueries[routeType] = appendAnchorsToQuery(allocationPlan.focusedQueries[routeType] || retrievalQuery, referentResolution.anchors);
	const route = allocationPlan.routeDecision;
	const diagnostics = [...route.reasons, ...referentResolution.reasons];
	if (hybridHits.length !== rawHybridHits.length) diagnostics.push("filtered-bootstrap-memory-noise");
	if (hybridHits.length > 6 && (route.routeType === "mixed" || route.routeConfidence < .72)) hybridHits = scoreBasedRelevanceFilter(hybridHits, retrievalQuery, 12);
	const budgetedSelection = collectAndScheduleMemoryObjectsWithBudget(store, ctx, allocationPlan, hybridHits);
	const budgetedScheduled = budgetedSelection.scheduled;
	const projected = projectScheduledMemoryObjects(budgetedScheduled, deriveProjectionOptions(ctx, allocationPlan, retrievalQuery, route, queryAnalysis));
	const allowHistoricalFacts = queryAnalysis.queryShape.timeframe === "historical" || queryAnalysis.queryShape.timeframe === "compare";
	const queryAnchors = uniqueNonEmpty(referentResolution.anchors).filter((anchor) => isMeaningfulRecallAnchor(anchor));
	const resolvedReferentAnchors = uniqueNonEmpty(referentResolution.anchors).filter((anchor) => isMeaningfulRecallAnchor(anchor));
	let states = projected.states;
	let tasks = projected.tasks;
	let facts = projected.facts;
	let events = projected.events;
	let graph = projected.graph;
	let alternates = projected.alternates;
	const candidateDrivenAuthority = true;
	const candidateAlternateRows = candidateGenerationResult ? filterBootstrapRows(rowsFromSearchHits(candidateGenerationResult.alternateSearchHits)) : [];
	if (events.length > 1) events = [...events].sort(compareEvidenceRowsChronologically);
	if (allocationPlan.objectiveBudgets.explanatory.activated && graph.nodes.length === 0) diagnostics.push("graph-no-seeds");
	if (alternates.length === 0) alternates = dedupeEvidenceRows([...candidateAlternateRows, ...filterBootstrapRows(rowsFromSearchHits(hybridHits))].filter((entry) => !shouldSuppressRecallText(entry.text)), 6);
	else if (candidateAlternateRows.length > 0) alternates = dedupeEvidenceRows([...alternates, ...candidateAlternateRows], 6);
	if (allowHistoricalFacts) {
		const conservativeHistoricalAlternates = queryMemoryFacts({
			store,
			ctx,
			text: retrievalQuery,
			limit: 2,
			includeHistorical: true
		}).filter((fact) => fact.status === "superseded").map((fact) => {
			const sourceRef = factSupportSourceRefs(fact)[0] ?? fact.sourceRef;
			return toEvidenceRow({
				id: fact.factId,
				text: formatFactLine({
					subject: fact.canonicalSubject,
					predicate: fact.predicate,
					object: fact.canonicalObject ?? void 0,
					objectValueJson: fact.objectValueJson,
					status: fact.status
				}),
				score: .34,
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
					materializedEpoch: fact.materializedEpoch
				}
			});
		});
		if (conservativeHistoricalAlternates.length > 0) {
			alternates = dedupeEvidenceRows([...alternates, ...conservativeHistoricalAlternates].sort((left, right) => (right.score ?? 0) - (left.score ?? 0)), 6);
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
	if (route.routeType === "workflow" && states.some((entry) => entry.id.endsWith("workflow.current_task")) && (states.some((entry) => entry.id.endsWith("workflow.blocker")) || states.some((entry) => entry.id.endsWith("workflow.next_action")))) {
		tasks = [];
		diagnostics.push("workflow-state-authority");
	}
	states = dedupeEvidenceRows(filterBootstrapRows(states), 4);
	tasks = dedupeEvidenceRows(filterBootstrapRows(tasks), 4);
	const factualMainLimit = shouldUseExactSnippetSupport(queryAnalysis) ? 8 : 6;
	facts = dedupeEvidenceRows(filterBootstrapRows(facts), factualMainLimit);
	events = dedupeEvidenceRows(filterBootstrapRows(events), Math.max(4, ctx.config.advanced.recallChunkBudget));
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
		if (factualStates.length !== states.length) diagnostics.push("main-surface:workflow-state-suppressed");
		states = factualStates;
	}
	if (!shouldUseWorkflowMainSurface(queryAnalysis)) {
		const factualFacts = facts.filter((entry) => {
			return store.factRepo.get(entry.id)?.predicate !== "has_workflow_guidance";
		});
		if (factualFacts.length !== facts.length) diagnostics.push("main-surface:workflow-guidance-suppressed");
		facts = factualFacts;
	}
	const scheduledFactRows = facts.slice();
	let currentFactGuardSourceRows = [];
	const suppressedPromptEvidenceCandidateIds = /* @__PURE__ */ new Set();
	let suppressPromptEvidenceBeforeMs;
	if (candidateGenerationResult && shouldApplyCandidateAuthorityToMainSurface(route, queryAnalysis)) {
		const candidateFactLimit = shouldUseExactSnippetSupport(queryAnalysis) ? 8 : 6;
		const candidateStateRows = prioritizeCandidateRowsForMainSurface(primaryCandidateRowsForSurface(candidateGenerationResult, "state"), queryAnchors, 4);
		const candidateFactRows = prioritizeCandidateRowsForMainSurface(primaryCandidateRowsForSurface(candidateGenerationResult, "fact", {
			store,
			queryAnalysis
		}), queryAnchors, candidateFactLimit);
		const candidateEventRows = prioritizeCandidateRowsForMainSurface(primaryCandidateRowsForSurface(candidateGenerationResult, "event"), queryAnchors, Math.max(4, ctx.config.advanced.recallChunkBudget));
		const candidateChunkRows = prioritizeCandidateRowsForMainSurface(primaryCandidateRowsForSurface(candidateGenerationResult, "chunk").filter((entry) => entry.provenance !== "assistant"), queryAnchors, Math.max(3, Math.min(6, ctx.config.advanced.recallChunkBudget)));
		currentFactGuardSourceRows = [...candidateEventRows, ...candidateChunkRows];
		if (candidateStateRows.length > 0) {
			states = candidateStateRows;
			diagnostics.push("candidate-authority:state-main");
		}
		if (candidateFactRows.length > 0) {
			facts = candidateFactRows;
			diagnostics.push("candidate-authority:fact-main");
		}
		if (route.routeType === "factual" && (candidateEventRows.length > 0 || candidateChunkRows.length > 0)) {
			events = dedupeEvidenceRows([...candidateEventRows, ...candidateChunkRows], Math.max(4, ctx.config.advanced.recallChunkBudget));
			diagnostics.push("candidate-authority:event-main");
		}
	}
	const newerCurrentSourceRows = newerSourceRowsForCurrentFactGuard({
		facts,
		sourceRows: currentFactGuardSourceRows,
		queryAnalysis
	});
	if (newerCurrentSourceRows.length > 0) {
		for (const fact of facts) for (const id of uniqueNonEmpty([
			fact.id,
			fact.id.startsWith("fact:") ? fact.id.slice(5) : `fact:${fact.id}`,
			fact.lineage?.canonicalId,
			fact.lineage?.sourceId
		])) suppressedPromptEvidenceCandidateIds.add(id);
		suppressPromptEvidenceBeforeMs = Math.min(...newerCurrentSourceRows.map((row) => observedAtMs(row)));
		facts = [];
		events = dedupeEvidenceRows([...newerCurrentSourceRows, ...events.filter((row) => observedAtMs(row) >= (suppressPromptEvidenceBeforeMs ?? 0))], Math.max(4, ctx.config.advanced.recallChunkBudget));
		diagnostics.push("newer-source-evidence-suppressed-stale-current-fact");
	}
	const complementaryFactRows = selectComplementaryFactRowsForMainSurface({
		store,
		queryAnalysis,
		queryAnchors,
		selectedFacts: facts,
		fallbackFacts: scheduledFactRows,
		reserveFactTrace: budgetedSelection.trace.reserveSelections.factual,
		limit: 1
	});
	if (complementaryFactRows.length > 0) {
		facts = dedupeEvidenceRows([...facts, ...complementaryFactRows], factualMainLimit);
		diagnostics.push("candidate-authority:fact-companion");
	}
	const workflowGateAnchors = resolvedReferentAnchors.length > 0 ? resolvedReferentAnchors : queryAnchors;
	if (route.routeType === "workflow" && queryAnalysis.queryShape.evidenceNeed === "workflow_context" && workflowGateAnchors.length > 0 && tasks.some((entry) => queryAnchorSupport(entry.text, workflowGateAnchors) >= .58)) {
		tasks = tasks.filter((entry) => queryAnchorSupport(entry.text, workflowGateAnchors) >= .58);
		facts = [];
		graph = {
			nodes: [],
			edges: [],
			paths: [],
			pathCandidates: []
		};
		if (!diagnostics.includes("workflow-anchored-task-authority")) diagnostics.push("workflow-anchored-task-authority");
	}
	const supportLinkedSourceRefs = shouldUseExactSnippetSupport(queryAnalysis) ? supportRefsFromSelectedFacts(store, facts, queryAnchors) : [];
	const supportLinkedChunkRows = shouldUseExactSnippetSupport(queryAnalysis) ? buildSupportRefChunkRows({
		store,
		ctx,
		sourceRefs: supportLinkedSourceRefs,
		queryAnchors,
		limit: Math.max(3, Math.min(6, ctx.config.advanced.recallChunkBudget))
	}) : [];
	if (supportLinkedChunkRows.length > 0) {
		events = dedupeEvidenceRows([...supportLinkedChunkRows, ...events.filter((entry) => !isChunkEvidenceRow(entry))], Math.max(4, ctx.config.advanced.recallChunkBudget));
		diagnostics.push("source-linked-chunk-rescue:fact-support");
	}
	const exactSnippetFallbacks = shouldUseExactSnippetSupport(queryAnalysis) ? buildSourceRefFallbackSnippets({
		store,
		ctx,
		sourceRefs: uniqueNonEmpty([
			...supportLinkedSourceRefs,
			...facts.map((entry) => entry.sourceRef ?? ""),
			...events.map((entry) => entry.sourceRef ?? "")
		]),
		limit: 3,
		queryAnchors
	}) : [];
	const projectedSnippetCandidates = shouldUseExactSnippetSupport(queryAnalysis) ? preferredProjectedExactSnippets({
		projectedIds: projected.recalledChunkIds,
		projectedTexts: projected.recalledChunkTexts,
		queryAnchors,
		limit: Math.min(3, projected.recalledChunkIds.length)
	}) : [];
	const supportLinkedSnippetCandidates = shouldUseExactSnippetSupport(queryAnalysis) ? buildSourceRefFallbackSnippets({
		store,
		ctx,
		sourceRefs: supportLinkedSourceRefs,
		limit: 3,
		queryAnchors
	}).map((entry) => ({
		...entry,
		source: "support_ref"
	})) : [];
	const rankedExactSnippetCandidates = shouldUseExactSnippetSupport(queryAnalysis) ? rankExactSnippetCandidates([
		...supportLinkedSnippetCandidates,
		...projectedSnippetCandidates,
		...exactSnippetFallbacks
	], queryAnalysis).slice(0, 3) : [];
	const exactSnippetIds = shouldUseExactSnippetSupport(queryAnalysis) ? rankedExactSnippetCandidates.map((entry) => entry.snippetId) : void 0;
	const exactSnippetTexts = shouldUseExactSnippetSupport(queryAnalysis) ? rankedExactSnippetCandidates.map((entry) => entry.text) : void 0;
	const selectedExactSnippets = shouldUseExactSnippetSupport(queryAnalysis) ? rankedExactSnippetCandidates.length > 0 ? rankedExactSnippetCandidates.map((entry) => {
		if (entry.source !== "projected") return entry;
		const snippetId = entry.snippetId;
		const matchedHit = candidateGenerationResult?.searchHits.find((hit) => hit.docId === snippetId) ?? rawHybridHits.find((hit) => hit.docId === snippetId);
		return {
			snippetId,
			text: entry.text,
			sourceRef: typeof matchedHit?.metadata.sourceRef === "string" ? matchedHit.metadata.sourceRef : void 0,
			lineage: lineageFromMetadata(matchedHit?.metadata, {
				sourceKind: "vector_doc",
				sourceId: snippetId
			}) ?? {
				sourceKind: "vector_doc",
				sourceId: snippetId
			},
			source: "projected",
			goalScore: entry.goalScore
		};
	}) : void 0 : void 0;
	const recalledChunkSupport = mergeRecalledChunkSupport({
		supportRows: supportLinkedChunkRows,
		projectedIds: projected.recalledChunkIds,
		projectedTexts: projected.recalledChunkTexts,
		limit: Math.max(3, Math.min(6, ctx.config.advanced.recallChunkBudget))
	});
	if (shouldUseExactSnippetSupport(queryAnalysis) && projected.recalledChunkIds.length === 0 && exactSnippetFallbacks.length > 0) diagnostics.push("exact-snippet:source-ref-fallback");
	if (shouldUseExactSnippetSupport(queryAnalysis) && supportLinkedSnippetCandidates.length > 0) diagnostics.push("exact-snippet:support-ref-rescue");
	const controlEvidence = controlPromptEvidenceCandidates({
		store,
		ctx,
		queryAnalysis
	});
	const evidenceAssembly = assembleEvidencePackets({
		queryAnalysis,
		candidateGenerationResult,
		promptEvidence: buildPromptEvidenceCandidates({
			store,
			ctx,
			queryAnalysis,
			candidateGenerationResult,
			facts,
			events,
			controlEvidence,
			selectedExactSnippets,
			suppressedCandidateIds: suppressedPromptEvidenceCandidateIds,
			suppressCandidateEvidenceBeforeMs: suppressPromptEvidenceBeforeMs
		}),
		now: ctx.now
	});
	const promptEvidence = evidenceAssembly.promptEvidence;
	const evidencePlanAudit = buildEvidencePlanAudit({
		queryAnalysis,
		promptEvidence,
		candidateGenerationResult,
		injectedEvidence: promptEvidence.filter((entry) => entry.role === "protected" && !entry.dropReason)
	});
	const bundle = {
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
		renderedBlock: ""
	};
	bundle.renderedBlock = renderEvidenceBundle(bundle, ctx.config.maxInjectedChars);
	const auditId = randomId("audit");
	emitFullRetrievalSignals(store, ctx, {
		auditId,
		bundle,
		scheduled: budgetedSelection.scheduled
	});
	emitContradictionSignals(store, ctx, {
		query,
		routeType: route.routeType === "mixed" || route.routeType === "unknown" ? "explanatory" : route.routeType,
		graphEdges: bundle.graph.edges
	});
	if (ctx.config.advanced.enableTelemetryAudit) {
		const injectedPromptEvidence = bundle.promptEvidence.filter((entry) => !entry.dropReason && entry.role !== "alternate").filter((entry) => entry.injected);
		const renderedPromptLines = renderedPromptLineAudit(bundle);
		const evidencePlanAuditForTelemetry = buildEvidencePlanAudit({
			queryAnalysis,
			promptEvidence: bundle.promptEvidence,
			candidateGenerationResult,
			injectedEvidence: injectedPromptEvidence
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
				rawText: entry.rawText && entry.rawText !== entry.text ? truncateText(entry.rawText, 720) : void 0,
				lineage: entry.lineage
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
				lineage: hit.lineage
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
				lineage: hit.lineage
			})),
			candidatePool: bundle.promptEvidence.filter((entry) => !entry.dropReason).map((entry) => ({
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
				text: truncateText(entry.text, 360)
			})),
			rankedPromptEvidence: bundle.promptEvidence.filter((entry) => !entry.dropReason).sort((left, right) => (right.injectionScore ?? right.priority) - (left.injectionScore ?? left.priority)).map((entry) => ({
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
				text: truncateText(entry.text, 360)
			})),
			evidencePackets: bundle.evidencePackets.map((packet) => ({
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
				answerCandidate: packet.answerCandidate ? {
					id: packet.answerCandidate.id,
					surface: packet.answerCandidate.surface,
					sourceRef: packet.answerCandidate.sourceRef,
					normalizedSourceRefs: normalizedPromptEvidenceSourceRefs(packet.answerCandidate),
					slotEvidenceRole: packet.answerCandidate.slotEvidenceRole,
					text: truncateText(packet.answerCandidate.text, 360)
				} : void 0,
				contextCandidates: (packet.contextCandidates ?? []).map((entry) => ({
					id: entry.id,
					surface: entry.surface,
					sourceRef: entry.sourceRef,
					normalizedSourceRefs: normalizedPromptEvidenceSourceRefs(entry),
					slotEvidenceRole: entry.slotEvidenceRole,
					text: truncateText(entry.text, 240)
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
				supportingTexts: packet.supportingTexts.map((text) => truncateText(text, 360))
			})),
			rankedInjectedPackets: bundle.evidencePackets.filter((packet) => packet.injected || packet.protected).sort((left, right) => (right.score ?? 0) - (left.score ?? 0)).map((packet) => ({
				packetId: packet.packetId,
				slotId: packet.slotId,
				score: packet.score,
				scoreBreakdown: packet.scoreBreakdown,
				sourceRefs: packet.sourceRefs,
				allSourceRefs: packet.allSourceRefs,
				normalizedSourceRefs: packet.normalizedSourceRefs,
				normalizedAllSourceRefs: packet.normalizedAllSourceRefs,
				displayLines: packet.displayLines,
				selectionReason: packet.selectionReason
			})),
			scoreCurve: bundle.evidencePackets.filter((packet) => !packet.dropReason).sort((left, right) => (right.score ?? 0) - (left.score ?? 0)).slice(0, 12).map((packet, index) => ({
				rank: index + 1,
				packetId: packet.packetId,
				injected: packet.injected ?? false,
				score: packet.score,
				finalScore: packet.grade?.finalScore,
				slotId: packet.slotId,
				sourceRefs: packet.sourceRefs,
				normalizedSourceRefs: packet.normalizedSourceRefs
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
				rawText: entry.rawText && entry.rawText !== entry.text ? truncateText(entry.rawText, 720) : void 0
			})),
			droppedEvidence: bundle.promptEvidence.filter((entry) => entry.dropReason).map((entry) => ({
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
				rawText: entry.rawText && entry.rawText !== entry.text ? truncateText(entry.rawText, 240) : void 0
			})),
			sourceExpansion: bundle.promptEvidence.filter((entry) => entry.metadata?.sourceExpansion === true).map((entry) => ({
				id: entry.id,
				surface: entry.surface,
				sourceRef: entry.sourceRef,
				mergedSourceRefs: entry.mergedSourceRefs,
				normalizedSourceRefs: normalizedPromptEvidenceSourceRefs(entry),
				neighborOf: Array.isArray(entry.metadata?.neighborOf) ? entry.metadata.neighborOf : [],
				role: entry.role,
				packetId: entry.packetId,
				injectionScore: entry.injectionScore,
				text: truncateText(entry.text, 360)
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
				compilerProvenance: queryAnalysis.compilerProvenance
			},
			candidateGeneration: candidateGenerationResult ? candidateGenerationAuditPayload(candidateGenerationResult) : void 0,
			evidencePlanAudit: evidencePlanAuditForTelemetry,
			evidencePacketAudit: bundle.evidencePacketAudit,
			selectionAuthority: {
				candidateDrivenAuthority,
				historicalBackfillMode: diagnostics.includes("historical-backfill:primary") ? "primary" : diagnostics.includes("historical-backfill:alternate-only") ? "alternate-only" : "none",
				alternatePromotionEnabled: false
			},
			mainVsAlternate: {
				stateIds: bundle.states.map((entry) => entry.id),
				taskIds: bundle.tasks.map((entry) => entry.id),
				factIds: bundle.facts.map((entry) => entry.id),
				eventIds: bundle.events.map((entry) => entry.id),
				alternateIds: bundle.alternates.map((entry) => entry.id)
			},
			llmBudget: snapshotMemoryLlmBudgetAudit(ctx.llmBudgetAudit),
			recall: buildRecallAuditPayload({
				...auditOptions,
				recallMode: auditOptions.recallMode ?? "full",
				budgetPlan: allocationPlan,
				selectionTrace: budgetedSelection.trace,
				queryAnalysis
			})
		};
		store.auditRepo.recordRetrieval({
			auditId,
			agentId: ctx.agentId,
			scope: ctx.scopes.join(","),
			routeType: bundle.routeType,
			queryText: query,
			queryHash: stableHash([query]),
			selectedItemsJson: selected,
			injectedChars: bundle.renderedBlock.length,
			createdAt: ctx.now
		});
	}
	return bundle;
}
function sentenceWindowForPromptEvidence(text, anchors, maxChars) {
	const cleaned = text.trim();
	if (cleaned.length <= maxChars) return cleaned;
	const sentences = cleaned.split(/(?<=[.!?])\s+|\n+/u).map((entry) => entry.trim()).filter(Boolean);
	const usefulAnchors = informativePromptAnchors(anchors);
	const selected = (sentences.length > 0 ? sentences : [cleaned]).map((sentence) => ({
		sentence,
		score: usefulAnchors.length > 0 ? informativePromptAnchorSupport(sentence, usefulAnchors) : semanticTextSimilarity(sentence, cleaned)
	})).sort((left, right) => right.score - left.score)[0]?.sentence ?? cleaned;
	if (selected.length <= maxChars) return selected;
	const normalizedSelected = normalizeText(selected);
	const anchor = usefulAnchors.find((entry) => normalizedSelected.includes(normalizeText(entry)));
	const anchorIndex = anchor ? selected.toLowerCase().indexOf(anchor.toLowerCase()) : -1;
	if (anchorIndex >= 0) {
		const start = Math.max(0, anchorIndex - Math.floor(maxChars * .45));
		return truncateText(selected.slice(start).trim(), maxChars);
	}
	return truncateText(selected, maxChars);
}
function promptEvidenceLineText(entry, maxChars) {
	return sentenceWindowForPromptEvidence(entry.text, entry.excerptAnchors ?? [], maxChars);
}
function packetPromptEvidenceEntry(bundle, packet) {
	const sourceRefs = new Set(packet.sourceRefs);
	return bundle.promptEvidence.find((entry) => {
		if (entry.dropReason) return false;
		if (entry.filledSlotIds && entry.filledSlotIds.length > 0 && !entry.filledSlotIds.includes(packet.slotId)) return false;
		return [entry.sourceRef, ...entry.mergedSourceRefs ?? []].some((sourceRef) => sourceRef && sourceRefs.has(sourceRef));
	});
}
function evidencePacketRenderLine(packet, bundle) {
	const slot = `[slot:${packet.slotId}]`;
	const date = packet.resolvedDate ?? packet.observedAt?.slice(0, 10);
	const datePart = date ? ` [date:${date}]` : "";
	const quantity = typeof packet.quantityHint === "number" ? ` [quantity:${packet.quantityHint}${packet.unitHint ? ` ${packet.unitHint}` : ""}]` : "";
	const matchingEntry = packetPromptEvidenceEntry(bundle, packet);
	return `- ${slot}${datePart}${quantity} ${matchingEntry ? promptEvidenceLineText(matchingEntry, 360) : sentenceWindowForPromptEvidence(packet.primaryText, bundle.queryAnchors ?? [], 360)}`;
}
function renderEvidenceBundle(bundle, maxChars) {
	const effectiveBudget = Math.max(220, Math.min(maxChars, bundle.budgetPlan?.totalPromptChars ?? maxChars));
	const promptEvidenceBudget = Math.max(360, Math.floor(effectiveBudget * .86));
	const sections = [renderBudgetedSection("Priority Evidence", bundle.evidencePackets.filter((packet) => packet.injected && !packet.dropReason).sort((left, right) => (right.grade?.finalScore ?? right.coverage.confidence) - (left.grade?.finalScore ?? left.coverage.confidence)).slice(0, 6).flatMap((packet) => {
		const displayLines = (packet.displayLines ?? []).filter((line) => line.trim().length > 0);
		if (displayLines.length > 0) return displayLines.map((line) => line.trim().startsWith("-") ? line.trim() : `- ${line.trim()}`);
		return [evidencePacketRenderLine(packet, bundle)];
	}), promptEvidenceBudget)].filter(Boolean);
	if (sections.length === 0) return "";
	const rendered = ["## Memory Context", ...sections].join("\n\n").trim();
	return rendered.length <= effectiveBudget ? rendered : truncateText(rendered, effectiveBudget);
}
//#endregion
export { buildBackgroundRecallBundle, hasBackgroundRecallMaterial, renderEvidenceBundle, retrieveEvidence };
