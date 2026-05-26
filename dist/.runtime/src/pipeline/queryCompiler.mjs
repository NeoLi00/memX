import { clamp01, isValidEntityName, normalizeName, normalizeText, normalizedTerms, stableHash, truncateText } from "../support.mjs";
import { recordMemoryLlmBudgetCall } from "./llmBudgetAudit.mjs";
import { attributeSlotAliasesForSlots, attributeSlotContractHints, queryAsksForAttributeValue, requestedAttributeSlotsFromText } from "./attributeSlots.mjs";
import { MEMX_NATIVE_HOOK_TIMEOUT_MS } from "../timeouts.mjs";
import { entityNameAliases } from "./entityAliases.mjs";
//#region src/pipeline/queryCompiler.ts
const QUERY_ENVELOPE_LONG_THRESHOLD_CHARS = 2400;
const QUERY_ENVELOPE_HEAD_CHARS = 700;
const QUERY_ENVELOPE_TAIL_CHARS = 1100;
const QUERY_ENVELOPE_LATEST_INSTRUCTION_CHARS = 900;
const QUERY_COMPACT_SCAFFOLD_QUERY_CHARS = 240;
const QUERY_COMPACT_SCAFFOLD_FOCUS_CHARS = 180;
const QUERY_FOCUSED_TASK_CHARS = 360;
const PRIMARY_ROUTE_TYPES = [
	"workflow",
	"factual",
	"temporal",
	"explanatory"
];
const VALID_CANDIDATE_SURFACES = [
	"state",
	"fact",
	"event",
	"task",
	"chunk",
	"graph",
	"entity_alias"
];
const VALID_EVIDENCE_PLAN_LAYERS = [
	...VALID_CANDIDATE_SURFACES,
	"control",
	"strategy",
	"abstraction",
	"belief",
	"snippet"
];
const VALID_EVIDENCE_SLOT_ROLES = [
	"query_context",
	"answer_evidence",
	"answer_value",
	"answer_event",
	"time_constraint",
	"user_resource",
	"prior_advice",
	"supporting_context"
];
const VALID_QUERY_ENTITY_TYPES = [
	"person",
	"project",
	"tool",
	"service",
	"language",
	"framework",
	"concept",
	"organization",
	"unknown"
];
const VALID_QUERY_ENTITY_ROLES = [
	"subject",
	"object",
	"context",
	"resource"
];
function clampWindow(query, start, end) {
	const safeStart = Math.max(0, Math.min(query.length, start));
	const safeEnd = Math.max(safeStart, Math.min(query.length, end));
	const text = query.slice(safeStart, safeEnd).trim();
	if (!text) return null;
	return {
		kind: "full",
		start: safeStart,
		end: safeEnd,
		text
	};
}
function latestInstructionWindow(query) {
	const trimmedEnd = query.trimEnd().length;
	if (trimmedEnd <= 0) return null;
	const prefix = query.slice(0, trimmedEnd);
	const lastParagraphBreak = Math.max(prefix.lastIndexOf("\n\n"), prefix.lastIndexOf("\r\n\r\n"));
	const paragraphStart = lastParagraphBreak >= 0 ? lastParagraphBreak + 2 : 0;
	const window = clampWindow(query, Math.max(paragraphStart, trimmedEnd - QUERY_ENVELOPE_LATEST_INSTRUCTION_CHARS), trimmedEnd);
	return window ? {
		...window,
		kind: "latest_instruction"
	} : null;
}
function queryParagraphs(query) {
	return query.split(/\n\s*\n/gu).map((paragraph) => paragraph.replace(/\s+/gu, " ").trim()).filter(Boolean);
}
function looksLikeGenericTaskInstruction(text) {
	const normalized = normalizeText(text);
	if (!normalized) return true;
	return /^(give|provide|write|produce|return|answer|solve|continue)\b.*\b(solution|answer|proof|response|reply)\b/iu.test(normalized) || /^if you cannot finish\b/iu.test(normalized) || /^do not use\b/iu.test(normalized) || /^you are solving\b/iu.test(normalized) || /^只回复\b/u.test(normalized);
}
function taskParagraphScore(paragraph) {
	const normalized = normalizeText(paragraph);
	let score = Math.min(1, paragraph.length / 180);
	if (/\b(problem|task|issue|error|exception|requirements?)\b/iu.test(normalized)) score += .75;
	if (/\b(let|given|determine|find|prove|show|calculate|suppose)\b/iu.test(normalized)) score += .55;
	if (/[。！？.!?]/u.test(paragraph) && paragraph.length >= 80) score += .25;
	if (looksLikeGenericTaskInstruction(paragraph)) score -= 1.25;
	return score;
}
function buildTaskBearingFocusedQuery(query) {
	const paragraphs = queryParagraphs(query);
	if (paragraphs.length === 0) return query.trim();
	const best = paragraphs.map((paragraph, index) => ({
		paragraph,
		index,
		score: taskParagraphScore(paragraph)
	})).sort((left, right) => {
		if (right.score !== left.score) return right.score - left.score;
		return right.paragraph.length - left.paragraph.length;
	})[0];
	if (!best || best.score <= 0) return query.trim();
	return truncateText(best.paragraph, QUERY_FOCUSED_TASK_CHARS);
}
function hasMeaningfulTermOverlap(left, right) {
	const leftTerms = new Set(normalizedTerms(left, { minLength: 3 }));
	if (leftTerms.size === 0) return false;
	let overlap = 0;
	for (const term of normalizedTerms(right, { minLength: 3 })) if (leftTerms.has(term)) {
		overlap++;
		if (overlap >= 2) return true;
	}
	return false;
}
function protectFocusedQuery(query, focusedQuery) {
	const raw = query.trim();
	const focused = focusedQuery.trim();
	if (!raw || !focused) return focused || buildTaskBearingFocusedQuery(raw);
	if (raw.length <= focused.length * 2) return focused;
	const taskBearing = buildTaskBearingFocusedQuery(raw);
	if (!taskBearing || taskBearing.length <= focused.length * 1.5) return focused;
	if (looksLikeGenericTaskInstruction(focused)) return taskBearing;
	if (focused.length < 80 && !hasMeaningfulTermOverlap(focused, taskBearing)) return taskBearing;
	return focused;
}
function mergedVisibleChars(windows) {
	const intervals = windows.map((window) => ({
		start: window.start,
		end: window.end
	})).filter((interval) => interval.end > interval.start).sort((left, right) => left.start - right.start);
	let visible = 0;
	let cursorStart;
	let cursorEnd;
	for (const interval of intervals) {
		if (cursorStart === void 0 || cursorEnd === void 0) {
			cursorStart = interval.start;
			cursorEnd = interval.end;
			continue;
		}
		if (interval.start <= cursorEnd) {
			cursorEnd = Math.max(cursorEnd, interval.end);
			continue;
		}
		visible += cursorEnd - cursorStart;
		cursorStart = interval.start;
		cursorEnd = interval.end;
	}
	if (cursorStart !== void 0 && cursorEnd !== void 0) visible += cursorEnd - cursorStart;
	return visible;
}
function uniqueWindows(windows) {
	const seen = /* @__PURE__ */ new Set();
	const unique = [];
	for (const window of windows) {
		const key = `${window.kind}:${window.start}:${window.end}`;
		if (seen.has(key)) continue;
		seen.add(key);
		unique.push(window);
	}
	return unique;
}
function buildQueryEnvelope(query) {
	const rawLength = query.length;
	const rawHash = stableHash(["query-envelope", query]);
	if (rawLength <= QUERY_ENVELOPE_LONG_THRESHOLD_CHARS) {
		const full = clampWindow(query, 0, rawLength);
		const windows = full ? [{
			...full,
			kind: "full"
		}] : [];
		const visibleChars = mergedVisibleChars(windows);
		return {
			rawLength,
			rawHash,
			truncated: false,
			visibleChars,
			omittedChars: Math.max(0, rawLength - visibleChars),
			windows
		};
	}
	const head = clampWindow(query, 0, QUERY_ENVELOPE_HEAD_CHARS);
	const tail = clampWindow(query, rawLength - QUERY_ENVELOPE_TAIL_CHARS, rawLength);
	const latest = latestInstructionWindow(query);
	const windows = uniqueWindows([
		head ? {
			...head,
			kind: "head"
		} : null,
		tail ? {
			...tail,
			kind: "tail"
		} : null,
		latest
	].filter((window) => Boolean(window)));
	const visibleChars = mergedVisibleChars(windows);
	return {
		rawLength,
		rawHash,
		truncated: true,
		visibleChars,
		omittedChars: Math.max(0, rawLength - visibleChars),
		windows
	};
}
function buildCompactQueryCompilerScaffold(fallback) {
	return {
		queryText: truncateText(fallback.queryText, QUERY_COMPACT_SCAFFOLD_QUERY_CHARS),
		focusedQuery: truncateText(fallback.focusedQuery, QUERY_COMPACT_SCAFFOLD_FOCUS_CHARS),
		queryEntities: fallback.queryEntities,
		queryShape: fallback.queryShape,
		primaryRoute: fallback.primaryRoute,
		answerGranularity: fallback.answerGranularity,
		evidenceFidelity: fallback.evidenceFidelity,
		routeWeights: fallback.routeWeights,
		candidateSurfaces: fallback.candidateSurfaces,
		detailNeedScore: fallback.detailNeedScore,
		supportNeed: fallback.supportNeed,
		ambiguityLevel: fallback.ambiguityLevel,
		turnMode: fallback.turnMode
	};
}
function buildQueryCompilerPromptInput(query, fallback) {
	return {
		envelope: buildQueryEnvelope(query),
		scaffold: buildCompactQueryCompilerScaffold(fallback)
	};
}
function normalizeRouteWeights(weights) {
	const total = PRIMARY_ROUTE_TYPES.reduce((sum, routeType) => sum + Math.max(0, weights[routeType] ?? 0), 0);
	if (total <= 0) return {};
	return Object.fromEntries(PRIMARY_ROUTE_TYPES.map((routeType) => [routeType, clamp01((weights[routeType] ?? 0) / total)]));
}
function sanitizePrimaryRoute(value) {
	return PRIMARY_ROUTE_TYPES.includes(value) ? value : void 0;
}
function primaryRouteFromWeights(routeWeights) {
	let best;
	let bestScore = 0;
	for (const routeType of PRIMARY_ROUTE_TYPES) {
		const score = routeWeights[routeType] ?? 0;
		if (score > bestScore) {
			best = routeType;
			bestScore = score;
		}
	}
	return bestScore > 0 ? best : void 0;
}
function routeWeightsFromPrimaryRoute(primaryRoute, queryShape) {
	if (!primaryRoute) return {};
	const weights = deriveRouteWeights(queryShape);
	weights[primaryRoute] = Math.max(weights[primaryRoute] ?? 0, .72);
	return normalizeRouteWeights(weights);
}
function sanitizeQueryEntityType(value) {
	return VALID_QUERY_ENTITY_TYPES.includes(value) ? value : void 0;
}
function sanitizeQueryEntityRole(value) {
	return VALID_QUERY_ENTITY_ROLES.includes(value) ? value : void 0;
}
function looksLikeLocalSymbolEntity(name, type) {
	const compact = normalizeName(name).replace(/[^\p{L}\p{N}]+/gu, "");
	if (!compact) return true;
	if (/^[a-z]$/iu.test(compact)) return true;
	if (type === "concept" && /^[a-z][0-9]?$/iu.test(compact)) return true;
	return false;
}
function looksLikeDeicticEntity(name) {
	const normalized = normalizeName(name);
	return new Set([
		"this",
		"that",
		"it",
		"these",
		"those",
		"thing",
		"topic",
		"这个",
		"那个",
		"它",
		"这",
		"那",
		"上面",
		"前面",
		"刚才",
		"这件事",
		"那个问题"
	]).has(normalized);
}
function sanitizeQueryEntities(value, limit = 8) {
	if (!Array.isArray(value)) return [];
	const seen = /* @__PURE__ */ new Set();
	const entities = [];
	for (const entry of value) {
		if (!entry || typeof entry !== "object" || Array.isArray(entry)) continue;
		const record = entry;
		const name = typeof record.name === "string" ? record.name.trim() : "";
		const type = sanitizeQueryEntityType(record.type);
		if (!name || !isValidEntityName(name) || looksLikeLocalSymbolEntity(name, type) || looksLikeDeicticEntity(name)) continue;
		const key = `${type ?? "unknown"}:${normalizeName(name)}`;
		if (seen.has(key)) continue;
		seen.add(key);
		const role = sanitizeQueryEntityRole(record.role);
		entities.push({
			name,
			...type ? { type } : {},
			...role ? { role } : {}
		});
		if (entities.length >= limit) break;
	}
	return entities;
}
function sanitizeSuppressedEntities(value, limit = 8) {
	if (!Array.isArray(value)) return [];
	const seen = /* @__PURE__ */ new Set();
	const entities = [];
	for (const entry of value) {
		if (!entry || typeof entry !== "object" || Array.isArray(entry)) continue;
		const record = entry;
		const name = typeof record.name === "string" ? record.name.trim() : "";
		const type = sanitizeQueryEntityType(record.type);
		if (!name || !isValidEntityName(name) || looksLikeLocalSymbolEntity(name, type) || looksLikeDeicticEntity(name)) continue;
		const key = `${type ?? "unknown"}:${normalizeName(name)}`;
		if (seen.has(key)) continue;
		seen.add(key);
		const reason = typeof record.reason === "string" ? truncateText(record.reason.trim(), 160) : "";
		entities.push({
			name,
			...type ? { type } : {},
			...reason ? { reason } : {}
		});
		if (entities.length >= limit) break;
	}
	return entities;
}
const VALID_CONTEXT_EXCLUSION_KINDS = new Set([
	"prior_project",
	"prior_topic",
	"prior_context",
	"host_native_memory"
]);
function sanitizeContextExclusionKind(value) {
	if (typeof value !== "string") return;
	const normalized = value.trim();
	return VALID_CONTEXT_EXCLUSION_KINDS.has(normalized) ? normalized : void 0;
}
function sanitizeContextExclusions(value, limit = 8) {
	if (!Array.isArray(value)) return [];
	const seen = /* @__PURE__ */ new Set();
	const exclusions = [];
	for (const entry of value) {
		if (!entry || typeof entry !== "object" || Array.isArray(entry)) continue;
		const record = entry;
		const kind = sanitizeContextExclusionKind(record.kind);
		const label = typeof record.label === "string" ? truncateText(record.label.trim(), 160) : "";
		if (!kind || !label) continue;
		const key = `${kind}:${normalizeName(label)}`;
		if (seen.has(key)) continue;
		seen.add(key);
		const reason = typeof record.reason === "string" ? truncateText(record.reason.trim(), 160) : "";
		exclusions.push({
			kind,
			label,
			...reason ? { reason } : {}
		});
		if (exclusions.length >= limit) break;
	}
	return exclusions;
}
function deriveAnswerGranularity(queryShape) {
	if (queryShape.granularity === "exact_detail") return "detail";
	return "summary";
}
function deriveEvidenceFidelity(queryShape, anchors) {
	if (queryShape.timeframe === "compare" || queryShape.granularity === "exact_detail") return "high";
	if (queryShape.timeframe === "historical" || anchors.length === 0 || queryShape.referentialMode === "deictic") return "medium";
	return "low";
}
function deriveRouteWeights(queryShape) {
	const rawWeights = {
		workflow: .08,
		factual: .08,
		temporal: .08,
		explanatory: .08
	};
	switch (queryShape.evidenceNeed) {
		case "workflow_context":
			rawWeights.workflow += .62;
			break;
		case "canonical_state":
			rawWeights.factual += .56;
			break;
		case "factual_history":
			rawWeights.factual += .44;
			rawWeights.temporal += .22;
			break;
		case "event_history":
			rawWeights.temporal += .56;
			rawWeights.factual += .16;
			break;
		case "relation":
			rawWeights.explanatory += .48;
			rawWeights.factual += .16;
			break;
		case "chunk":
			rawWeights.temporal += .28;
			rawWeights.factual += .28;
			break;
	}
	if (queryShape.timeframe === "current") rawWeights.factual += .32;
	else if (queryShape.timeframe === "historical") rawWeights.temporal += .32;
	else if (queryShape.timeframe === "compare") {
		rawWeights.temporal += .3;
		rawWeights.factual += .3;
	}
	if (queryShape.referentialMode === "deictic") rawWeights.workflow += .14;
	if (queryShape.granularity === "exact_detail") {
		rawWeights.temporal += .12;
		rawWeights.factual += .12;
	}
	return normalizeRouteWeights(rawWeights);
}
function deriveCandidateSurfaces(queryShape, answerGranularity, evidenceFidelity) {
	const surfaces = /* @__PURE__ */ new Set();
	switch (queryShape.evidenceNeed) {
		case "workflow_context":
			surfaces.add("task");
			surfaces.add("state");
			surfaces.add("chunk");
			break;
		case "canonical_state":
			surfaces.add("state");
			surfaces.add("fact");
			break;
		case "factual_history":
			surfaces.add("fact");
			surfaces.add("event");
			surfaces.add("chunk");
			break;
		case "event_history":
			surfaces.add("event");
			surfaces.add("chunk");
			break;
		case "relation":
			surfaces.add("graph");
			surfaces.add("entity_alias");
			surfaces.add("fact");
			break;
		case "chunk":
			surfaces.add("chunk");
			surfaces.add("event");
			break;
	}
	if (queryShape.timeframe === "current") {
		surfaces.add("state");
		surfaces.add("fact");
	}
	if (queryShape.timeframe === "historical" || queryShape.timeframe === "compare") {
		surfaces.add("event");
		surfaces.add("fact");
	}
	if (answerGranularity === "detail" || evidenceFidelity === "high") surfaces.add("chunk");
	if (queryShape.referentialMode === "deictic") surfaces.add("task");
	if (queryShape.referentialMode === "anchored") {
		surfaces.add("event");
		surfaces.add("chunk");
	}
	if (queryShape.evidenceNeed === "workflow_context") {
		surfaces.add("fact");
		surfaces.add("event");
	}
	return [...surfaces];
}
function sanitizeCandidateSurfaces(value, fallback) {
	if (!Array.isArray(value)) return fallback;
	const valid = value.filter((surface) => VALID_CANDIDATE_SURFACES.includes(surface));
	return valid.length > 0 ? [...new Set(valid)] : fallback;
}
function uniqueNonEmpty(values, limit = 8) {
	const seen = /* @__PURE__ */ new Set();
	const result = [];
	for (const value of values) {
		const trimmed = value.trim();
		if (!trimmed) continue;
		const key = normalizeText(trimmed);
		if (!key || seen.has(key)) continue;
		seen.add(key);
		result.push(trimmed);
		if (result.length >= limit) break;
	}
	return result;
}
function queryEntitySubjectHints(queryEntities, limit = 8) {
	return uniqueNonEmpty(queryEntities.flatMap((entity) => entityNameAliases(entity.name)), limit);
}
function normalizeCompilerHint(value) {
	const cleaned = value.replace(/\s+/gu, " ").trim();
	if (!normalizeText(cleaned)) return "";
	return cleaned;
}
function usefulQueryHint(value) {
	const normalized = normalizeText(value);
	if (!normalized || /^question$/iu.test(normalized)) return false;
	if (/^question\s*:/iu.test(value.trim())) return false;
	return normalizedTerms(normalized, { minLength: 2 }).length > 0;
}
function usefulQueryHints(values, limit = 8) {
	return uniqueNonEmpty(values.map(normalizeCompilerHint).filter((hint) => hint && usefulQueryHint(hint)), limit);
}
function meaningfulQueryTerms(query) {
	return uniqueNonEmpty([normalizeCompilerHint(query)].filter(usefulQueryHint), 8);
}
function querySuggestsMultipleAnswerValues(query) {
	const normalized = query.normalize("NFKC").toLowerCase();
	const delimiterMatches = normalized.match(/(?:、|，|,|\/|\b(?:and|plus)\b|和|与|以及)/giu) ?? [];
	const explicitlyDistributed = /(?:分别|各自|分别是|各是什么|respectively|each|both|all of|list|列出)/iu.test(normalized);
	const asksForValue = /(?:什么|哪些|哪几个|是什么|what|which|where|when|how many|value|values)/iu.test(normalized);
	return explicitlyDistributed && delimiterMatches.length >= 1 || asksForValue && delimiterMatches.length >= 2;
}
function deriveAnswerMode(query, queryShape) {
	const requestedSlots = requestedAttributeSlotsFromText(query);
	if (queryAsksForAttributeValue(query) && requestedSlots.length > 1) return "multi_evidence";
	if (querySuggestsMultipleAnswerValues(query)) return "multi_evidence";
	if (queryShape.timeframe === "compare" || queryShape.evidenceNeed === "relation") return "multi_evidence";
	if (queryAsksForAttributeValue(query) && requestedSlots.length === 1) return "attribute_lookup";
	return "single_fact";
}
function sanitizeAnswerMode(value, query, queryShape) {
	if (querySuggestsMultipleAnswerValues(query) && (value === "single_fact" || value === "attribute_lookup" || value === void 0)) return "multi_evidence";
	return value === "single_fact" || value === "attribute_lookup" || value === "count_aggregate" || value === "multi_evidence" ? value : deriveAnswerMode(query, queryShape);
}
function deriveTopicAnchors(query, mode, anchors) {
	return uniqueNonEmpty([...usefulQueryHints(anchors), ...meaningfulQueryTerms(query)], 4);
}
function deriveEvidenceCoverage(params) {
	const topicAnchors = deriveTopicAnchors(params.query, params.answerMode, params.anchors);
	const optionalAnchors = uniqueNonEmpty([...meaningfulQueryTerms(params.query), ...topicAnchors], 6);
	const [minProtectedItems, maxProtectedItems] = params.answerMode === "count_aggregate" ? [2, 4] : params.answerMode === "multi_evidence" ? [2, 3] : [1, 1];
	return {
		requiredAnchors: [],
		optionalAnchors,
		minProtectedItems,
		maxProtectedItems
	};
}
function sanitizeEvidenceCoverage(query, compiled) {
	const answerMode = compiled.answerMode ?? deriveAnswerMode(query, compiled.queryShape);
	const fallback = deriveEvidenceCoverage({
		query,
		queryShape: compiled.queryShape,
		anchors: compiled.anchors,
		answerMode
	});
	const input = compiled.evidenceCoverage;
	if (!input) return fallback;
	const rawRequiredAnchors = uniqueNonEmpty((Array.isArray(input.requiredAnchors) ? input.requiredAnchors : []).map(normalizeCompilerHint).filter(usefulQueryHint), 6);
	const optionalAnchors = uniqueNonEmpty((Array.isArray(input.optionalAnchors) ? input.optionalAnchors : []).map(normalizeCompilerHint).filter(usefulQueryHint), 6);
	return {
		requiredAnchors: rawRequiredAnchors.length > 0 ? rawRequiredAnchors : fallback.requiredAnchors,
		optionalAnchors: optionalAnchors.length > 0 ? optionalAnchors : fallback.optionalAnchors,
		minProtectedItems: typeof input.minProtectedItems === "number" && Number.isFinite(input.minProtectedItems) ? Math.max(0, Math.min(4, Math.floor(input.minProtectedItems))) : fallback.minProtectedItems,
		maxProtectedItems: typeof input.maxProtectedItems === "number" && Number.isFinite(input.maxProtectedItems) ? Math.max(1, Math.min(4, Math.floor(input.maxProtectedItems))) : fallback.maxProtectedItems
	};
}
function buildDefaultEvidenceGoals(params) {
	const focusedQuery = params.focusedQuery.trim();
	const anchorQuery = uniqueNonEmpty([...params.evidenceCoverage?.optionalAnchors ?? [], ...params.anchors.map(normalizeCompilerHint)].filter(usefulQueryHint), 6).join(" ").trim();
	const positiveQueries = uniqueNonEmpty([
		focusedQuery,
		anchorQuery && focusedQuery ? `${anchorQuery} ${focusedQuery}` : "",
		anchorQuery
	], 4);
	return [{
		goal: `Find remembered evidence that answers: ${focusedQuery || anchorQuery || "the current query"}`,
		positiveQueries: positiveQueries.length > 0 ? positiveQueries : [focusedQuery || anchorQuery || "memory evidence"],
		focusAnchors: [],
		preferredSurfaces: params.candidateSurfaces,
		fidelity: params.evidenceFidelity
	}];
}
function normalizePlanLayers(layers, fallbackLayers, limit = 12) {
	if (!Array.isArray(layers)) return fallbackLayers;
	const valid = layers.filter((layer) => VALID_EVIDENCE_PLAN_LAYERS.includes(layer));
	return valid.length > 0 ? [...new Set(valid)].slice(0, limit) : fallbackLayers;
}
function normalizeBridgePreferredLayers(layers, fallbackLayers, operation) {
	const withRawEvidence = [...normalizePlanLayers(layers, fallbackLayers)];
	const add = (layer) => {
		if (!withRawEvidence.includes(layer)) withRawEvidence.push(layer);
	};
	add("chunk");
	if (operation === "aggregate") add("event");
	return withRawEvidence.slice(0, 12);
}
function defaultPreferredLayers(params) {
	const layers = [];
	const add = (layer) => {
		if (!layers.includes(layer)) layers.push(layer);
	};
	if (params.queryShape.evidenceNeed === "relation") {
		add("graph");
		add("entity_alias");
	}
	if (params.queryShape.timeframe === "current") add("state");
	for (const surface of params.candidateSurfaces) add(surface);
	if (!layers.includes("event")) add("event");
	if (!layers.includes("chunk")) add("chunk");
	return layers.slice(0, 12);
}
function defaultFallbackLayers(preferredLayers) {
	const fallback = new Set(preferredLayers);
	fallback.add("fact");
	fallback.add("event");
	fallback.add("chunk");
	fallback.add("snippet");
	return [...fallback].slice(0, 12);
}
function requiredFieldsForPlan(params) {
	params.query;
	if (params.answerMode === "attribute_lookup") return ["attribute_value"];
	if (params.answerMode === "count_aggregate") return ["countable_item"];
	if (params.queryShape.timeframe === "compare") return ["temporal_marker"];
	if (params.queryShape.granularity === "exact_detail") return ["answer_value"];
	return [];
}
const BRIDGE_TEMPLATE_NOISE = new Set([
	"requested topic or domain",
	"query context",
	"matching event or item",
	"count target",
	"event evidence",
	"time constraint from query",
	"temporal marker",
	"source evidence",
	"direct answer value",
	"requested attribute value",
	"topic or domain that the historical answer must be about"
]);
function meaningfulContractHint(value, _params) {
	const normalized = normalizeText(value);
	if (!retrievalHintAllowed(normalized)) return false;
	if (BRIDGE_TEMPLATE_NOISE.has(normalized)) return false;
	return true;
}
function compactEvidenceContractHints(values, params) {
	return uniqueNonEmpty(values.map(normalizeCompilerHint).filter((value) => meaningfulContractHint(value, params)), params.limit);
}
function operationTypeForPlan(params) {
	params.query;
	if (params.answerMode === "count_aggregate") return {
		type: "aggregate",
		description: "Count or aggregate only evidence that fills the requested slots."
	};
	if (params.queryShape.timeframe === "compare") return {
		type: "derive",
		description: "Compare or derive the answer only after the relevant slots are filled."
	};
	if (params.queryShape.evidenceNeed === "relation") return {
		type: "relate",
		description: "Use relation evidence and source evidence together; graph-only support is not final proof."
	};
	return {
		type: "return_value",
		description: "Return the value directly supported by filled evidence slots."
	};
}
const VALID_EVIDENCE_OPERATION_TYPES = [
	"return_value",
	"aggregate",
	"derive",
	"compare",
	"relate",
	"tailor_advice"
];
function sanitizeEvidenceOperationType(value, fallback) {
	if (typeof value !== "string") return fallback;
	const normalized = normalizeText(value).replace(/\s+/gu, "_");
	return VALID_EVIDENCE_OPERATION_TYPES.includes(normalized) ? normalized : fallback;
}
function splitComparisonAnchors(query, anchors) {
	const cleaned = anchors.map(normalizeCompilerHint).filter(Boolean);
	return cleaned.length >= 2 ? cleaned.slice(0, 4) : [];
}
function slotFromHints(params) {
	return {
		id: params.id,
		...params.role ? { role: params.role } : {},
		description: params.description,
		subjectHints: uniqueNonEmpty(params.subjectHints.map(normalizeCompilerHint).filter(Boolean), 6),
		relationHints: uniqueNonEmpty(params.relationHints.map(normalizeCompilerHint).filter(Boolean), 6),
		requestedAttributeSlots: uniqueNonEmpty(params.requestedAttributeSlots ?? [], 6),
		capabilityQueries: uniqueNonEmpty((params.capabilityQueries ?? []).map(normalizeCompilerHint).filter(Boolean), 8),
		negativeHints: uniqueNonEmpty((params.negativeHints ?? []).map(normalizeCompilerHint).filter(Boolean), 6),
		...params.requiredRole ? { requiredRole: params.requiredRole } : {},
		requiredFields: uniqueNonEmpty(params.requiredFields, 4),
		preferredLayers: params.preferredLayers,
		fallbackLayers: params.fallbackLayers,
		minEvidence: Math.max(1, Math.min(4, Math.trunc(params.minEvidence)))
	};
}
const ATTRIBUTE_QUERY_ENGLISH_STOP_WORDS = new Set([
	"a",
	"an",
	"and",
	"are",
	"current",
	"default",
	"for",
	"is",
	"of",
	"the",
	"what",
	"which",
	"with"
]);
const ATTRIBUTE_QUERY_ENGLISH_LEADING_RE = /^(?:what|which|who|where|when|how|is|are|was|were|does|do|did|can|could|please|tell\s+me|show\s+me)\b[\s,.:;!?-]*/iu;
const ATTRIBUTE_QUERY_CJK_NOISE_RE = /(?:现在|当前|目前|请问|帮我|告诉我|是什么|是哪一个|哪一个|哪种|哪些|什么|用什么|使用什么|默认|主要|主|的|是|为|请|一下)+$/u;
function cleanAttributeSubjectCandidate(value) {
	let cleaned = value.normalize("NFKC").replace(/[“”"'\u2018\u2019`]/gu, "").replace(/\b([A-Za-z0-9_.-]+)'s\b/giu, "$1").replace(/[?？!！,，、;；:：()[\]{}<>]/gu, " ").replace(/\s+/gu, " ").trim();
	while (ATTRIBUTE_QUERY_ENGLISH_LEADING_RE.test(cleaned)) cleaned = cleaned.replace(ATTRIBUTE_QUERY_ENGLISH_LEADING_RE, "").trim();
	cleaned = cleaned.replace(ATTRIBUTE_QUERY_CJK_NOISE_RE, "").trim();
	cleaned = cleaned.split(/\s+/u).filter(Boolean).filter((token) => {
		const normalized = normalizeText(token);
		return normalized && !ATTRIBUTE_QUERY_ENGLISH_STOP_WORDS.has(normalized);
	}).join(" ").trim() || cleaned;
	return cleaned.replace(/\s+(?:and|or)$/iu, "").replace(/(?:和|与|以及)$/u, "").trim();
}
function codeLikeQueryEntityHints(query) {
	const stopWords = new Set([
		"A",
		"An",
		"And",
		"Are",
		"Current",
		"Default",
		"For",
		"Is",
		"Of",
		"The",
		"What",
		"Which"
	]);
	return uniqueNonEmpty([...query.matchAll(/[`"'“”‘’]([^`"'“”‘’]{2,80})[`"'“”‘’]/gu), ...query.matchAll(/\b[A-Z][A-Za-z0-9]*(?:[._/-][A-Za-z0-9]+)*(?:[A-Z][A-Za-z0-9]*)*\b/gu)].map((match) => (match[1] ?? match[0]).replace(/'s$/iu, "").trim()).filter((entry) => entry.length >= 2 && !stopWords.has(entry)), 6);
}
function attributeQuerySubjectHints(query, requestedSlots) {
	if (requestedSlots.length === 0) return [];
	const aliases = attributeSlotAliasesForSlots(requestedSlots);
	const lowerQuery = query.normalize("NFKC").toLowerCase();
	let earliest = Number.POSITIVE_INFINITY;
	for (const alias of aliases) {
		const normalizedAlias = alias.normalize("NFKC").toLowerCase();
		const index = lowerQuery.indexOf(normalizedAlias);
		if (index >= 0) earliest = Math.min(earliest, index);
	}
	const cleanedPrefix = cleanAttributeSubjectCandidate(Number.isFinite(earliest) && earliest > 0 ? query.slice(0, earliest) : query);
	return uniqueNonEmpty([...codeLikeQueryEntityHints(query), cleanedPrefix].filter((hint) => {
		const normalized = normalizeText(hint);
		if (!normalized || requestedAttributeSlotsFromText(hint).length > 0) return false;
		return normalizedTerms(normalized, { minLength: 2 }).length > 0 || /[\p{Script=Han}]/u.test(hint);
	}), 6);
}
function buildDefaultEvidencePlan(params) {
	const preferredLayers = defaultPreferredLayers({
		query: params.query,
		queryShape: params.queryShape,
		candidateSurfaces: params.candidateSurfaces
	});
	const fallbackLayers = defaultFallbackLayers(preferredLayers);
	const requiredFields = requiredFieldsForPlan({
		query: params.query,
		queryShape: params.queryShape,
		answerMode: params.answerMode
	});
	const requestedAttributeSlots = requestedAttributeSlotsFromText(params.query, params.focusedQuery);
	const requestedAttributeHints = attributeSlotContractHints(requestedAttributeSlots);
	const attributeSubjectHints = attributeQuerySubjectHints(params.query, requestedAttributeSlots);
	const relationHints = [];
	const comparisonAnchors = params.queryShape.timeframe === "compare" ? splitComparisonAnchors(params.query, params.anchors) : [];
	const topicHints = uniqueNonEmpty(attributeSubjectHints.length > 0 ? [...attributeSubjectHints, ...params.anchors] : [
		...params.anchors,
		...meaningfulQueryTerms(params.focusedQuery || params.query),
		...meaningfulQueryTerms(params.query)
	], 6);
	return {
		slots: comparisonAnchors.length >= 2 ? comparisonAnchors.map((anchor, index) => slotFromHints({
			id: `slot_${index + 1}`,
			role: "answer_evidence",
			description: `Evidence for ${anchor}`,
			subjectHints: [anchor],
			relationHints,
			requiredFields,
			preferredLayers,
			fallbackLayers,
			minEvidence: 1
		})) : params.answerMode === "count_aggregate" ? [slotFromHints({
			id: "answer_event",
			role: "answer_event",
			requiredRole: "answer_event",
			description: `Distinct remembered events/items that should be counted for: ${params.focusedQuery || params.query}`,
			subjectHints: topicHints,
			relationHints: ["distinct countable event"],
			negativeHints: ["event outside the requested count target"],
			requiredFields: ["countable_item", "source_evidence"],
			preferredLayers: [
				"event",
				"chunk",
				"fact"
			],
			fallbackLayers: [
				"event",
				"chunk",
				"fact",
				"snippet"
			],
			minEvidence: 2
		}), slotFromHints({
			id: "time_constraint",
			role: "time_constraint",
			requiredRole: "time_constraint",
			description: "Temporal window or time constraint from the query.",
			subjectHints: topicHints,
			relationHints: ["temporal constraint"],
			requiredFields: ["temporal_marker"],
			preferredLayers: [
				"event",
				"chunk",
				"fact"
			],
			fallbackLayers: [
				"event",
				"chunk",
				"snippet"
			],
			minEvidence: 1
		})] : requestedAttributeSlots.length > 1 ? [slotFromHints({
			id: "query_context",
			role: "query_context",
			requiredRole: "query_context",
			description: "Subject or situation the requested attributes belong to.",
			subjectHints: topicHints,
			relationHints: ["query subject", "lookup context"],
			requiredFields: ["query_context"],
			preferredLayers,
			fallbackLayers,
			minEvidence: 1
		}), ...requestedAttributeSlots.map((slot) => slotFromHints({
			id: `answer_${slot}`.slice(0, 48),
			role: "answer_value",
			requiredRole: "answer_value",
			description: `The ${slot.replace(/_/g, " ")} value that directly answers: ${params.focusedQuery || params.query}`,
			subjectHints: topicHints,
			relationHints: [
				"requested attribute value",
				"direct answer value",
				...attributeSlotContractHints([slot])
			],
			requestedAttributeSlots: [slot],
			requiredFields: ["attribute_value", slot],
			preferredLayers,
			fallbackLayers,
			minEvidence: 1
		}))] : params.answerMode === "attribute_lookup" ? [slotFromHints({
			id: "query_context",
			role: "query_context",
			requiredRole: "query_context",
			description: "Subject or situation the requested attribute belongs to.",
			subjectHints: topicHints,
			relationHints: ["query subject", "lookup context"],
			requiredFields: ["query_context"],
			preferredLayers,
			fallbackLayers,
			minEvidence: 1
		}), slotFromHints({
			id: "answer_value",
			role: "answer_value",
			requiredRole: "answer_value",
			description: `The attribute value that directly answers: ${params.focusedQuery || params.query}`,
			subjectHints: topicHints,
			relationHints: [
				"requested attribute value",
				"direct answer value",
				...requestedAttributeHints
			],
			requestedAttributeSlots,
			requiredFields: ["attribute_value", ...requestedAttributeSlots],
			preferredLayers,
			fallbackLayers,
			minEvidence: 1
		})] : params.queryShape.timeframe === "historical" ? [
			slotFromHints({
				id: "query_context",
				role: "query_context",
				requiredRole: "query_context",
				description: "Topic or domain that the historical answer must be about.",
				subjectHints: topicHints,
				relationHints: ["domain context"],
				requiredFields: ["query_context"],
				preferredLayers,
				fallbackLayers,
				minEvidence: 1
			}),
			slotFromHints({
				id: "time_constraint",
				role: "time_constraint",
				requiredRole: "time_constraint",
				description: "Historical time constraint from the query.",
				subjectHints: topicHints,
				relationHints: ["temporal constraint"],
				requiredFields: ["temporal_marker"],
				preferredLayers: [
					"event",
					"chunk",
					"fact"
				],
				fallbackLayers: [
					"event",
					"chunk",
					"snippet"
				],
				minEvidence: 1
			}),
			slotFromHints({
				id: "answer_value",
				role: "answer_value",
				requiredRole: "answer_value",
				description: `The remembered value or event detail that answers: ${params.focusedQuery || params.query}`,
				subjectHints: topicHints,
				relationHints: ["direct answer value", "answer evidence"],
				requiredFields: requiredFields.length > 0 ? requiredFields : ["answer_value"],
				preferredLayers,
				fallbackLayers,
				minEvidence: 1
			})
		] : [slotFromHints({
			id: "query_context",
			role: "query_context",
			requiredRole: "query_context",
			description: "Subject or situation the answer must be bound to.",
			subjectHints: topicHints,
			relationHints: ["query context", "requested subject"],
			requiredFields: ["query_context"],
			preferredLayers,
			fallbackLayers,
			minEvidence: 1
		}), slotFromHints({
			id: "answer_value",
			role: "answer_value",
			requiredRole: "answer_value",
			description: `Evidence that can directly answer: ${params.focusedQuery || params.query}`,
			subjectHints: topicHints,
			relationHints: [...relationHints, ...requestedAttributeHints],
			requestedAttributeSlots,
			requiredFields: requiredFields.length > 0 || requestedAttributeSlots.length > 0 ? [...requiredFields, ...requestedAttributeSlots] : ["answer_value"],
			preferredLayers,
			fallbackLayers,
			minEvidence: 1
		})],
		operation: operationTypeForPlan({
			query: params.query,
			queryShape: params.queryShape,
			answerMode: params.answerMode
		})
	};
}
function isAllowedGoalAnchor(anchor, fallback) {
	const normalized = normalizeText(anchor);
	if (!normalized || !usefulQueryHint(anchor)) return false;
	const focused = normalizeText(fallback.focusedQuery);
	const query = normalizeText(fallback.queryText);
	if (focused.includes(normalized) || query.includes(normalized)) return true;
	return fallback.anchors.some((candidate) => normalizeText(candidate) === normalized);
}
function normalizeGoalSurfaces(surfaces, fallbackSurfaces) {
	if (!Array.isArray(surfaces)) return fallbackSurfaces;
	const valid = surfaces.filter((surface) => VALID_CANDIDATE_SURFACES.includes(surface));
	return valid.length > 0 ? [...new Set(valid)] : fallbackSurfaces;
}
function sanitizeEvidenceGoals(fallback, compiledGoals, candidateSurfaces, evidenceFidelity) {
	const finish = (goals) => uniqueGoalsByText(goals).slice(0, fallback.evidencePlan?.operation.type === "tailor_advice" ? 4 : 3);
	if (!Array.isArray(compiledGoals)) return finish(buildDefaultEvidenceGoals({
		query: fallback.queryText,
		focusedQuery: fallback.focusedQuery,
		anchors: fallback.anchors,
		candidateSurfaces,
		evidenceFidelity,
		evidenceCoverage: fallback.evidenceCoverage
	}));
	const sanitized = compiledGoals.map((goal) => {
		if (!goal || typeof goal !== "object") return null;
		const entry = goal;
		const goalText = typeof entry.goal === "string" ? entry.goal.trim() : "";
		const rawPositiveQueries = Array.isArray(entry.positiveQueries) ? entry.positiveQueries.filter((query) => typeof query === "string") : [];
		const focusAnchors = uniqueNonEmpty([...fallback.evidenceCoverage?.requiredAnchors ?? [], ...Array.isArray(entry.focusAnchors) ? entry.focusAnchors.filter((anchor) => typeof anchor === "string") : []].map(normalizeCompilerHint).filter(Boolean).filter((anchor) => isAllowedGoalAnchor(anchor, fallback)), 6);
		const positiveQueries = uniqueNonEmpty([...rawPositiveQueries.filter((query) => retrievalHintAllowed(query)), fallback.focusedQuery], 7);
		const negativeHints = uniqueNonEmpty(Array.isArray(entry.negativeHints) ? entry.negativeHints.filter((hint) => typeof hint === "string") : [], 4);
		const fidelity = entry.fidelity === "low" || entry.fidelity === "medium" || entry.fidelity === "high" ? entry.fidelity : evidenceFidelity;
		const preferredSurfaceSet = new Set(normalizeGoalSurfaces(entry.preferredSurfaces, candidateSurfaces));
		if (fidelity === "high" || fallback.answerGranularity === "detail") {
			for (const surface of candidateSurfaces) if (surface === "event" || surface === "chunk") preferredSurfaceSet.add(surface);
		}
		const preferredSurfaces = [...preferredSurfaceSet];
		if (!goalText && positiveQueries.length === 0) return null;
		return {
			goal: goalText || `Find remembered evidence that answers: ${fallback.focusedQuery}`,
			positiveQueries: positiveQueries.length > 0 ? positiveQueries : [fallback.focusedQuery],
			negativeHints: negativeHints.length > 0 ? negativeHints : void 0,
			focusAnchors,
			preferredSurfaces,
			fidelity
		};
	}).filter((goal) => Boolean(goal)).slice(0, fallback.evidencePlan?.operation.type === "tailor_advice" ? 5 : 3);
	return finish(sanitized.length > 0 ? sanitized : buildDefaultEvidenceGoals({
		query: fallback.queryText,
		focusedQuery: fallback.focusedQuery,
		anchors: fallback.anchors,
		candidateSurfaces,
		evidenceFidelity,
		evidenceCoverage: fallback.evidenceCoverage
	}));
}
function uniqueGoalsByText(goals) {
	const byGoal = /* @__PURE__ */ new Map();
	for (const goal of goals) {
		const key = normalizeText(goal.goal);
		if (!key || byGoal.has(key)) continue;
		byGoal.set(key, goal);
	}
	return [...byGoal.values()];
}
function retrievalHintAllowed(query) {
	const normalized = normalizeText(query);
	if (!normalized || normalized.length > 140) return false;
	if (normalized.replace(/\s+/gu, "").length < 2) return false;
	return true;
}
function sanitizeSlotTextArray(values, fallback, limit) {
	if (!Array.isArray(values)) return fallback;
	const cleaned = uniqueNonEmpty(values.filter((value) => typeof value === "string").map(normalizeCompilerHint).filter(Boolean), limit);
	return cleaned.length > 0 ? cleaned : fallback;
}
function sanitizeRequiredFields(values, fallback) {
	if (!Array.isArray(values)) return fallback;
	const cleaned = uniqueNonEmpty(values.filter((value) => typeof value === "string").map((value) => normalizeText(value).replace(/[^a-z0-9_ -]/giu, " ").trim()).filter((value) => value.length > 0).map((value) => value.replace(/\s+/gu, "_")), 4);
	return cleaned.length > 0 ? cleaned : fallback;
}
function sanitizeSlotRole(value, fallback) {
	if (typeof value !== "string") return fallback;
	const normalized = normalizeText(value).replace(/\s+/gu, "_");
	return VALID_EVIDENCE_SLOT_ROLES.includes(normalized) ? normalized : fallback;
}
function sanitizeRequiredRole(value, fallback) {
	if (typeof value !== "string") return fallback;
	const normalized = normalizeText(value).replace(/\s+/gu, "_");
	return normalized === "query_context" || normalized === "user_resource" || normalized === "prior_advice" || normalized === "answer_value" || normalized === "answer_event" || normalized === "time_constraint" ? normalized : fallback;
}
function ensureTailorAdviceSlots(plan, fallback) {
	if (plan.operation.type !== "tailor_advice") return plan;
	const existingHints = uniqueNonEmpty(plan.slots.flatMap((slot) => [
		...slot.subjectHints,
		...slot.relationHints ?? [],
		slot.description
	]), 12);
	const queryHints = uniqueNonEmpty(usefulQueryHints([
		...fallback.anchors,
		fallback.focusedQuery,
		fallback.queryText
	]), 6);
	const makeSlot = (slot) => {
		const existing = plan.slots.find((entry) => entry.id === slot.id);
		if (!existing) return slot;
		return {
			...slot,
			role: existing.role ?? slot.role,
			requiredRole: existing.requiredRole ?? slot.requiredRole,
			description: existing.description || slot.description,
			subjectHints: uniqueNonEmpty([...existing.subjectHints, ...slot.subjectHints], 8),
			relationHints: uniqueNonEmpty([...existing.relationHints ?? [], ...slot.relationHints ?? []], 10),
			requestedAttributeSlots: uniqueNonEmpty([...existing.requestedAttributeSlots ?? [], ...slot.requestedAttributeSlots ?? []], 8),
			capabilityQueries: uniqueNonEmpty([...existing.capabilityQueries ?? [], ...slot.capabilityQueries ?? []], 10),
			negativeHints: uniqueNonEmpty([...existing.negativeHints ?? [], ...slot.negativeHints ?? []], 8),
			requiredFields: uniqueNonEmpty([...existing.requiredFields, ...slot.requiredFields], 6),
			preferredLayers: [...new Set([...slot.preferredLayers, ...existing.preferredLayers])].slice(0, 12),
			fallbackLayers: [...new Set([...slot.fallbackLayers, ...existing.fallbackLayers])].slice(0, 12),
			minEvidence: Math.max(slot.minEvidence, existing.minEvidence)
		};
	};
	return {
		...plan,
		slots: [
			makeSlot(slotFromHints({
				id: "current_need",
				role: "query_context",
				requiredRole: "query_context",
				description: "Current user need or problem to tailor advice for.",
				subjectHints: queryHints,
				relationHints: uniqueNonEmpty(["current user need", ...existingHints], 6),
				capabilityQueries: [],
				requiredFields: ["query_context"],
				preferredLayers: ["chunk", "snippet"],
				fallbackLayers: [
					"task",
					"event",
					"chunk",
					"snippet"
				],
				minEvidence: 1
			})),
			makeSlot(slotFromHints({
				id: "relevant_user_resources",
				role: "user_resource",
				requiredRole: "user_resource",
				description: "Remembered user resources or constraints that could change the advice.",
				subjectHints: queryHints,
				relationHints: uniqueNonEmpty(["resource affordance for current need", ...existingHints], 6),
				capabilityQueries: uniqueNonEmpty([`user resources tools or constraints that can help with ${fallback.focusedQuery || fallback.queryText}`, `owned or available resources with affordances relevant to ${fallback.focusedQuery || fallback.queryText}`], 4),
				negativeHints: ["unrelated advice topics", "resources unrelated to the current need"],
				requiredFields: ["source_evidence"],
				preferredLayers: [
					"state",
					"fact",
					"graph",
					"event",
					"chunk"
				],
				fallbackLayers: [
					"entity_alias",
					"fact",
					"event",
					"chunk",
					"snippet"
				],
				minEvidence: 1
			})),
			makeSlot(slotFromHints({
				id: "prior_advice_or_strategy",
				role: "prior_advice",
				requiredRole: "prior_advice",
				description: "Remembered prior advice, strategy, or abstraction for this problem.",
				subjectHints: queryHints,
				relationHints: uniqueNonEmpty(["prior advice for current need", ...existingHints], 6),
				capabilityQueries: uniqueNonEmpty([`prior advice or strategy relevant to ${fallback.focusedQuery || fallback.queryText}`], 4),
				negativeHints: ["unrelated advice topics"],
				requiredFields: ["source_evidence"],
				preferredLayers: [
					"strategy",
					"abstraction",
					"belief",
					"fact",
					"chunk"
				],
				fallbackLayers: [
					"event",
					"chunk",
					"snippet"
				],
				minEvidence: 1
			}))
		]
	};
}
function effectiveSlotRequiredRole(slot) {
	if (slot.requiredRole) return slot.requiredRole;
	if (slot.role === "answer_evidence") return "answer_value";
	if (slot.role === "query_context" || slot.role === "user_resource" || slot.role === "prior_advice" || slot.role === "answer_value" || slot.role === "answer_event" || slot.role === "time_constraint") return slot.role;
	if (slot.role === "supporting_context") return "query_context";
}
function ensureCoreEvidenceSlots(plan, fallbackPlan) {
	if (plan.operation.type === "tailor_advice") return plan;
	const requiredRoles = plan.operation.type === "aggregate" ? ["answer_event", "time_constraint"] : plan.operation.type === "derive" || plan.operation.type === "compare" ? ["answer_value", "time_constraint"] : ["query_context", "answer_value"];
	const slots = [...plan.slots];
	for (const role of requiredRoles) {
		if (slots.some((slot) => effectiveSlotRequiredRole(slot) === role)) continue;
		const fallbackSlot = fallbackPlan.slots.find((slot) => effectiveSlotRequiredRole(slot) === role) ?? fallbackPlan.slots.find((slot) => slot.id === role);
		if (fallbackSlot) slots.push(fallbackSlot);
	}
	return {
		...plan,
		slots: slots.slice(0, 6)
	};
}
function sanitizeEvidencePlan(fallback, compiledPlan) {
	const fallbackPlan = buildDefaultEvidencePlan({
		query: fallback.queryText,
		focusedQuery: fallback.focusedQuery,
		queryShape: fallback.queryShape,
		anchors: fallback.anchors,
		candidateSurfaces: fallback.candidateSurfaces,
		answerMode: fallback.answerMode ?? deriveAnswerMode(fallback.queryText, fallback.queryShape)
	});
	if (!compiledPlan || typeof compiledPlan !== "object") return fallbackPlan;
	const plan = compiledPlan;
	const slots = (Array.isArray(plan.slots) ? plan.slots : []).map((slot, index) => {
		if (!slot || typeof slot !== "object") return null;
		const entry = slot;
		const fallbackSlot = fallbackPlan.slots[Math.min(index, fallbackPlan.slots.length - 1)] ?? fallbackPlan.slots[0];
		if (!fallbackSlot) return null;
		const id = (typeof entry.id === "string" && entry.id.trim() ? entry.id.trim() : `slot_${index + 1}`).replace(/[^a-z0-9_-]/giu, "_").slice(0, 48) || `slot_${index + 1}`;
		const description = typeof entry.description === "string" && entry.description.trim() ? entry.description.trim().slice(0, 220) : fallbackSlot.description;
		const role = sanitizeSlotRole(entry.role, fallbackSlot.role);
		const requiredRole = sanitizeRequiredRole(entry.requiredRole, fallbackSlot.requiredRole);
		const subjectHints = sanitizeSlotTextArray(entry.subjectHints, fallbackSlot.subjectHints, 6).filter((hint) => isAllowedGoalAnchor(hint, fallback));
		const relationHintLimit = 8;
		const relationHints = uniqueNonEmpty([...sanitizeSlotTextArray(entry.relationHints, fallbackSlot.relationHints ?? [], 8), ...fallbackSlot.relationHints ?? []], relationHintLimit);
		const requestedAttributeSlots = uniqueNonEmpty([...requestedAttributeSlotsFromText(description, ...relationHints, ...sanitizeSlotTextArray(entry.requiredFields, fallbackSlot.requiredFields, 8)), ...fallbackSlot.requestedAttributeSlots ?? []], 8);
		const capabilityQueries = uniqueNonEmpty([...sanitizeSlotTextArray(entry.capabilityQueries, fallbackSlot.capabilityQueries ?? [], 8), ...fallbackSlot.capabilityQueries ?? []].filter(retrievalHintAllowed), 10);
		const negativeHints = uniqueNonEmpty([...sanitizeSlotTextArray(entry.negativeHints, fallbackSlot.negativeHints ?? [], 6), ...fallbackSlot.negativeHints ?? []], 8);
		const requiredFields = sanitizeRequiredFields(entry.requiredFields, fallbackSlot.requiredFields);
		const preferredLayers = normalizePlanLayers([...fallbackSlot.preferredLayers, ...Array.isArray(entry.preferredLayers) ? entry.preferredLayers : []], fallbackSlot.preferredLayers);
		const fallbackLayers = normalizePlanLayers([...fallbackSlot.fallbackLayers, ...Array.isArray(entry.fallbackLayers) ? entry.fallbackLayers : []], fallbackSlot.fallbackLayers);
		const minEvidence = typeof entry.minEvidence === "number" && Number.isFinite(entry.minEvidence) ? Math.max(1, Math.min(4, Math.trunc(entry.minEvidence))) : fallbackSlot.minEvidence;
		return {
			id,
			...role ? { role } : {},
			...requiredRole ? { requiredRole } : {},
			description,
			subjectHints: subjectHints.length > 0 ? subjectHints : fallbackSlot.subjectHints,
			relationHints,
			requestedAttributeSlots,
			capabilityQueries,
			negativeHints,
			requiredFields,
			preferredLayers,
			fallbackLayers,
			minEvidence
		};
	}).filter((slot) => Boolean(slot)).slice(0, 4);
	const operationInput = plan.operation && typeof plan.operation === "object" ? plan.operation : {};
	const operationType = sanitizeEvidenceOperationType(operationInput.type, fallbackPlan.operation.type);
	const operationDescription = typeof operationInput.description === "string" && operationInput.description.trim() ? operationInput.description.trim().slice(0, 260) : fallbackPlan.operation.description;
	return ensureTailorAdviceSlots(ensureCoreEvidenceSlots({
		slots: slots.length > 0 ? slots : fallbackPlan.slots,
		operation: {
			type: operationType,
			description: operationDescription
		}
	}, fallbackPlan), fallback);
}
function applyQueryEntitySubjectHints(plan, queryEntities, queryText) {
	const entityHints = queryEntitySubjectHints(queryEntities, 8);
	if (entityHints.length === 0) return plan;
	const normalizedQuery = normalizeText(queryText);
	return {
		...plan,
		slots: plan.slots.map((slot) => {
			const existing = (slot.subjectHints ?? []).filter((hint) => normalizeText(hint) !== normalizedQuery);
			return {
				...slot,
				subjectHints: uniqueNonEmpty([...entityHints, ...existing], 8)
			};
		})
	};
}
function semanticBridgeShapeForRole(role, operation) {
	if (role === "user_resource") return "resource_affordance";
	if (role === "answer_event" || operation === "aggregate") return operation === "aggregate" ? "aggregate_item" : "event";
	if (role === "time_constraint") return "time_constraint";
	if (role === "query_context") return "query_context";
	return "attribute_value";
}
function defaultSemanticBridgeQueries(slot, fallback) {
	const role = effectiveSlotRequiredRole(slot);
	const operation = fallback.evidencePlan?.operation.type;
	const base = compactEvidenceContractHints([
		slot.description,
		...slot.subjectHints,
		...slot.relationHints ?? [],
		...slot.capabilityQueries ?? [],
		...slot.requiredFields,
		fallback.focusedQuery
	], {
		operation,
		limit: role === "user_resource" ? 7 : 5
	});
	return base.length > 0 ? base : [fallback.focusedQuery || fallback.queryText];
}
function buildDefaultSemanticBridges(fallback) {
	const operation = fallback.evidencePlan?.operation.type ?? "return_value";
	return (fallback.evidencePlan?.slots ?? []).map((slot) => {
		const role = effectiveSlotRequiredRole(slot);
		if (!role) return null;
		const retrievalQueries = defaultSemanticBridgeQueries(slot, fallback);
		return {
			bridgeId: `bridge_${slot.id}`.replace(/[^a-z0-9_-]/giu, "_").slice(0, 64),
			sourceConcept: slot.description || slot.id,
			role,
			evidenceShape: semanticBridgeShapeForRole(role, operation),
			retrievalQueries,
			positiveSignals: compactEvidenceContractHints([
				...retrievalQueries,
				...slot.subjectHints,
				...slot.relationHints ?? [],
				...slot.capabilityQueries ?? []
			], {
				operation,
				limit: 8
			}),
			negativeSignals: uniqueNonEmpty(slot.negativeHints ?? [], 6),
			preferredLayers: normalizePlanLayers([...slot.preferredLayers, ...slot.fallbackLayers], fallback.candidateSurfaces),
			hypothesisOnly: true
		};
	}).filter((bridge) => Boolean(bridge)).slice(0, 6);
}
function sanitizeBridgeRole(value, fallback) {
	return sanitizeRequiredRole(value, fallback) ?? fallback;
}
function sanitizeBridgeShape(value, fallback) {
	if (typeof value !== "string") return fallback;
	const normalized = normalizeText(value).replace(/\s+/gu, "_");
	return normalized === "event" || normalized === "attribute_value" || normalized === "resource_affordance" || normalized === "query_context" || normalized === "time_constraint" || normalized === "aggregate_item" || normalized === "causal_explanation" || normalized === "validation_evidence" || normalized === "status_answer" || normalized === "decision_value" || normalized === "availability_statement" ? normalized : fallback;
}
function sanitizeSemanticBridges(fallback, compiledBridges) {
	const defaults = buildDefaultSemanticBridges(fallback);
	if (!Array.isArray(compiledBridges)) return defaults;
	const defaultByIndex = (index) => defaults[Math.min(index, Math.max(0, defaults.length - 1))];
	const bridges = compiledBridges.map((bridge, index) => {
		if (!bridge || typeof bridge !== "object") return null;
		const entry = bridge;
		const fallbackBridge = defaultByIndex(index);
		if (!fallbackBridge) return null;
		const rawId = typeof entry.bridgeId === "string" && entry.bridgeId.trim() ? entry.bridgeId.trim() : fallbackBridge.bridgeId;
		const role = sanitizeBridgeRole(entry.role, fallbackBridge.role);
		const operation = fallback.evidencePlan?.operation.type ?? "return_value";
		const retrievalQueries = compactEvidenceContractHints([...Array.isArray(entry.retrievalQueries) ? entry.retrievalQueries.filter((query) => typeof query === "string") : [], ...compiledBridges.length === 0 ? fallbackBridge.retrievalQueries : []], {
			operation,
			limit: 8
		});
		const positiveSignals = compactEvidenceContractHints([...Array.isArray(entry.positiveSignals) ? entry.positiveSignals.filter((signal) => typeof signal === "string") : [], ...compiledBridges.length === 0 ? fallbackBridge.positiveSignals : []], {
			operation,
			limit: 10
		});
		const negativeSignals = role === "answer_value" || role === "answer_event" || role === "user_resource" || role === "prior_advice" ? [] : compactEvidenceContractHints([...Array.isArray(entry.negativeSignals) ? entry.negativeSignals.filter((signal) => typeof signal === "string") : [], ...fallbackBridge.negativeSignals ?? []], {
			operation,
			limit: 8
		});
		if (retrievalQueries.length === 0 && positiveSignals.length === 0) return null;
		return {
			bridgeId: rawId.replace(/[^a-z0-9_-]/giu, "_").slice(0, 64) || `bridge_${index + 1}`,
			sourceConcept: typeof entry.sourceConcept === "string" && entry.sourceConcept.trim() ? entry.sourceConcept.trim().slice(0, 160) : fallbackBridge.sourceConcept,
			role,
			evidenceShape: sanitizeBridgeShape(entry.evidenceShape, semanticBridgeShapeForRole(role, fallback.evidencePlan?.operation.type ?? "return_value")),
			retrievalQueries: retrievalQueries.length > 0 ? retrievalQueries : positiveSignals,
			positiveSignals,
			negativeSignals: negativeSignals.length > 0 ? negativeSignals : void 0,
			preferredLayers: normalizeBridgePreferredLayers(entry.preferredLayers, fallbackBridge.preferredLayers, operation),
			hypothesisOnly: true
		};
	}).filter((bridge) => Boolean(bridge));
	const merged = bridges.length > 0 ? bridges : defaults;
	const byId = /* @__PURE__ */ new Map();
	for (const bridge of merged) if (!byId.has(bridge.bridgeId)) byId.set(bridge.bridgeId, bridge);
	return [...byId.values()].slice(0, 8);
}
function shouldAddEpisodicRecoverySurface(query, compiled) {
	return compiled.queryShape.evidenceNeed === "canonical_state" && compiled.queryShape.timeframe === "timeless" && compiled.queryShape.referentialMode === "anchored" && compiled.anchors.length > 0 && !looksShortAndContextDependent(query) && (compiled.turnMode === "memory_qa" || compiled.turnMode === "mixed") && (compiled.answerGranularity === "detail" || compiled.anchors.length > 1 || compiled.supportNeed >= .56);
}
function applyQueryCompileGuards(query, compiled) {
	const guarded = { ...compiled };
	const llmSemantic = guarded.compilerProvenance.source === "llm";
	guarded.shouldRecall = true;
	guarded.queryEntities = sanitizeQueryEntities(compiled.queryEntities);
	guarded.suppressedEntities = sanitizeSuppressedEntities(compiled.suppressedEntities);
	guarded.contextExclusions = sanitizeContextExclusions(compiled.contextExclusions);
	guarded.primaryRoute = sanitizePrimaryRoute(compiled.primaryRoute) ?? primaryRouteFromWeights(compiled.routeWeights);
	guarded.answerMode = compiled.answerMode ? sanitizeAnswerMode(compiled.answerMode, query, guarded.queryShape) : void 0;
	guarded.anchors = uniqueNonEmpty([...guarded.anchors.map(normalizeCompilerHint).filter(usefulQueryHint)], 8);
	guarded.evidenceCoverage = compiled.evidenceCoverage ? sanitizeEvidenceCoverage(query, guarded) : void 0;
	if (!llmSemantic && shouldAddEpisodicRecoverySurface(query, guarded)) {
		guarded.candidateSurfaces = [...new Set([
			...guarded.candidateSurfaces,
			"event",
			"chunk"
		])];
		guarded.answerGranularity = "detail";
		guarded.detailNeedScore = Math.max(guarded.detailNeedScore, .92);
		guarded.evidenceFidelity = "high";
		guarded.supportNeed = clamp01(Math.max(guarded.supportNeed, .72));
		guarded.compilerProvenance = {
			...guarded.compilerProvenance,
			source: guarded.compilerProvenance.source === "llm" ? "hybrid" : guarded.compilerProvenance.source,
			reasons: [
				...guarded.compilerProvenance.reasons ?? [],
				"episodic-recovery-surface:event",
				"episodic-recovery-surface:chunk",
				"episodic-recovery-granularity:detail",
				"episodic-recovery-fidelity:high"
			]
		};
	}
	if (!llmSemantic && guarded.turnMode === "memory_qa" && guarded.queryShape.evidenceNeed === "canonical_state" && guarded.answerGranularity === "detail" && !guarded.candidateSurfaces.includes("chunk")) {
		guarded.candidateSurfaces = [...new Set([...guarded.candidateSurfaces, "chunk"])];
		guarded.compilerProvenance = {
			...guarded.compilerProvenance,
			source: guarded.compilerProvenance.source === "llm" ? "hybrid" : guarded.compilerProvenance.source,
			reasons: [...guarded.compilerProvenance.reasons ?? [], "detail-surface-recovery:chunk"]
		};
	}
	const sanitizerInput = {
		...guarded,
		queryText: query,
		focusedQuery: guarded.focusedQuery || query,
		anchors: guarded.anchors,
		candidateSurfaces: guarded.candidateSurfaces,
		evidenceFidelity: guarded.evidenceFidelity,
		evidenceCoverage: guarded.evidenceCoverage
	};
	guarded.evidenceGoals = guarded.evidenceGoals.length > 0 ? sanitizeEvidenceGoals(sanitizerInput, guarded.evidenceGoals, guarded.candidateSurfaces, guarded.evidenceFidelity) : [];
	guarded.evidencePlan = guarded.evidencePlan ? sanitizeEvidencePlan(sanitizerInput, guarded.evidencePlan) : void 0;
	guarded.semanticBridges = guarded.semanticBridges && guarded.semanticBridges.length > 0 ? sanitizeSemanticBridges({
		...sanitizerInput,
		evidencePlan: guarded.evidencePlan
	}, guarded.semanticBridges) : void 0;
	return guarded;
}
function looksShortAndContextDependent(query) {
	const normalized = query.trim();
	const tokenCount = normalized.split(/\s+/u).filter(Boolean).length;
	return normalized.length <= 28 || tokenCount <= 4;
}
function compileQueryWithoutSemanticFallback(query, reason = "llm-only-query-compiler-unavailable") {
	const focusedQuery = buildTaskBearingFocusedQuery(query);
	const requestedAttributeSlots = requestedAttributeSlotsFromText(query);
	const attributeLookupFallback = queryAsksForAttributeValue(query) && requestedAttributeSlots.length > 0;
	const queryShape = {
		timeframe: attributeLookupFallback ? "current" : "timeless",
		granularity: attributeLookupFallback ? "exact_detail" : "summary",
		referentialMode: "anchored",
		evidenceNeed: attributeLookupFallback ? "canonical_state" : "chunk"
	};
	const answerGranularity = deriveAnswerGranularity(queryShape);
	const evidenceFidelity = deriveEvidenceFidelity(queryShape, []);
	const routeWeights = deriveRouteWeights(queryShape);
	const answerMode = deriveAnswerMode(query, queryShape);
	const candidateSurfaces = deriveCandidateSurfaces(queryShape, answerGranularity, evidenceFidelity);
	const evidencePlan = buildDefaultEvidencePlan({
		query,
		focusedQuery,
		queryShape,
		anchors: [],
		candidateSurfaces,
		answerMode
	});
	return {
		queryText: query,
		shouldRecall: true,
		focusedQuery,
		queryEntities: [],
		suppressedEntities: [],
		contextExclusions: [],
		queryShape,
		primaryRoute: primaryRouteFromWeights(routeWeights),
		answerGranularity,
		evidenceFidelity,
		routeWeights,
		anchors: [],
		candidateSurfaces,
		evidenceGoals: [],
		evidencePlan,
		answerMode,
		detailNeedScore: 0,
		supportNeed: 0,
		ambiguityLevel: 0,
		turnMode: "mixed",
		compilerProvenance: {
			source: "deterministic",
			mode: "fallback",
			reasons: [reason]
		}
	};
}
function mergeCompiledQuery(fallback, compiled) {
	const { memoryUseIntent: _discardedMemoryUseIntent, shouldRecall: _discardedShouldRecall, ...compiledFields } = compiled;
	const safeCompiled = compiledFields;
	const shouldRecall = true;
	const queryShape = safeCompiled.queryShape ?? fallback.queryShape;
	const compilerProvenance = safeCompiled.compilerProvenance ?? fallback.compilerProvenance;
	const hasLlmSemanticContract = compilerProvenance.source === "llm" || compilerProvenance.source === "hybrid" || compilerProvenance.mode === "llm";
	const focusedQuery = typeof safeCompiled.focusedQuery === "string" && safeCompiled.focusedQuery.trim() ? protectFocusedQuery(fallback.queryText, safeCompiled.focusedQuery) : fallback.focusedQuery;
	const rawAnchors = safeCompiled.anchors ? safeCompiled.anchors : fallback.anchors;
	const queryEntities = sanitizeQueryEntities(safeCompiled.queryEntities);
	const anchors = uniqueNonEmpty([...queryEntitySubjectHints(queryEntities, 8), ...rawAnchors], 8);
	const suppressedEntities = sanitizeSuppressedEntities(safeCompiled.suppressedEntities);
	const contextExclusions = sanitizeContextExclusions(safeCompiled.contextExclusions);
	const primaryRoute = sanitizePrimaryRoute(safeCompiled.primaryRoute) ?? primaryRouteFromWeights(safeCompiled.routeWeights ?? {}) ?? primaryRouteFromWeights(deriveRouteWeights(queryShape));
	const answerGranularity = safeCompiled.answerGranularity ?? deriveAnswerGranularity(queryShape);
	const evidenceFidelity = safeCompiled.evidenceFidelity ?? deriveEvidenceFidelity(queryShape, anchors);
	const routeWeights = safeCompiled.routeWeights ? normalizeRouteWeights(safeCompiled.routeWeights) : routeWeightsFromPrimaryRoute(primaryRoute, queryShape);
	const derivedSurfaces = deriveCandidateSurfaces(queryShape, answerGranularity, evidenceFidelity);
	if (queryEntities.length > 0) derivedSurfaces.push("entity_alias", "graph");
	const sanitizedCandidateSurfaces = sanitizeCandidateSurfaces(safeCompiled.candidateSurfaces, [...new Set(derivedSurfaces)]);
	const candidateSurfaces = [...new Set([...sanitizedCandidateSurfaces, ...derivedSurfaces])];
	const sanitizerFallback = {
		...fallback,
		...safeCompiled,
		queryText: fallback.queryText,
		shouldRecall,
		focusedQuery,
		queryEntities,
		suppressedEntities,
		contextExclusions,
		queryShape,
		primaryRoute,
		answerGranularity,
		anchors,
		candidateSurfaces,
		evidenceFidelity,
		routeWeights,
		compilerProvenance
	};
	const rawSanitizedPlan = safeCompiled.evidencePlan ? sanitizeEvidencePlan(sanitizerFallback, safeCompiled.evidencePlan) : hasLlmSemanticContract ? sanitizeEvidencePlan(sanitizerFallback, void 0) : void 0;
	const sanitizedPlan = rawSanitizedPlan ? applyQueryEntitySubjectHints(rawSanitizedPlan, queryEntities, fallback.queryText) : void 0;
	const bridgeFallback = {
		...sanitizerFallback,
		evidencePlan: sanitizedPlan
	};
	const evidenceGoals = safeCompiled.evidenceGoals ? sanitizeEvidenceGoals(sanitizerFallback, safeCompiled.evidenceGoals, candidateSurfaces, evidenceFidelity) : hasLlmSemanticContract ? sanitizeEvidenceGoals(sanitizerFallback, void 0, candidateSurfaces, evidenceFidelity) : [];
	const semanticBridges = safeCompiled.semanticBridges ? sanitizeSemanticBridges(bridgeFallback, safeCompiled.semanticBridges) : hasLlmSemanticContract && sanitizedPlan ? sanitizeSemanticBridges(bridgeFallback, void 0) : void 0;
	return {
		...fallback,
		...safeCompiled,
		queryText: fallback.queryText,
		shouldRecall,
		focusedQuery,
		queryEntities,
		suppressedEntities,
		contextExclusions,
		queryShape,
		primaryRoute,
		answerGranularity,
		evidenceFidelity,
		routeWeights,
		anchors,
		candidateSurfaces,
		evidenceGoals,
		evidencePlan: sanitizedPlan,
		semanticBridges,
		detailNeedScore: typeof safeCompiled.detailNeedScore === "number" && Number.isFinite(safeCompiled.detailNeedScore) ? clamp01(safeCompiled.detailNeedScore) : fallback.detailNeedScore,
		supportNeed: typeof safeCompiled.supportNeed === "number" && Number.isFinite(safeCompiled.supportNeed) ? clamp01(safeCompiled.supportNeed) : fallback.supportNeed,
		ambiguityLevel: typeof safeCompiled.ambiguityLevel === "number" && Number.isFinite(safeCompiled.ambiguityLevel) ? clamp01(safeCompiled.ambiguityLevel) : fallback.ambiguityLevel,
		turnMode: safeCompiled.turnMode ?? fallback.turnMode,
		compilerProvenance
	};
}
function queryCompilerTimeoutMs(params) {
	const configured = params.ctx.config.advanced.queryCompilerHotPathTimeoutMs;
	const configuredTimeout = Number.isFinite(configured) && configured > 0 ? Math.max(250, Math.min(15e3, configured)) : MEMX_NATIVE_HOOK_TIMEOUT_MS;
	const requested = params.hotPathTimeoutMs;
	if (Number.isFinite(requested) && requested !== void 0 && requested > 0) return Math.max(250, Math.min(configuredTimeout, requested));
	return configuredTimeout;
}
function timedQueryCompile(start, timeoutMs) {
	const controller = new AbortController();
	const promise = start(controller.signal);
	promise.catch(() => {});
	return new Promise((resolve) => {
		let settled = false;
		const timer = setTimeout(() => {
			if (settled) return;
			settled = true;
			controller.abort();
			resolve({ status: "timeout" });
		}, timeoutMs);
		promise.then((value) => {
			if (settled) return;
			settled = true;
			clearTimeout(timer);
			resolve({
				status: "fulfilled",
				value
			});
		}, (reason) => {
			if (settled) return;
			settled = true;
			clearTimeout(timer);
			resolve(controller.signal.aborted ? { status: "timeout" } : {
				status: "rejected",
				reason
			});
		});
	});
}
async function compileQuery(params) {
	const fallback = compileQueryWithoutSemanticFallback(params.query);
	if (!params.ctx.config.advanced.enableQueryCompiler || !params.reasoner?.isEnabled?.() || !params.reasoner.compileQuerySemantics) {
		recordMemoryLlmBudgetCall(params.ctx.llmBudgetAudit, {
			label: "query-compile",
			stage: "query_hot_path",
			provenance: "deterministic",
			mode: "fallback",
			detail: "queryCompiler requires LLM semantics and failed closed"
		});
		return fallback;
	}
	const compiledResult = await timedQueryCompile((signal) => params.reasoner.compileQuerySemantics(params.query, fallback, {
		stage: "query_hot_path",
		audit: params.ctx.llmBudgetAudit,
		signal
	}), queryCompilerTimeoutMs(params));
	if (compiledResult.status === "timeout") {
		recordMemoryLlmBudgetCall(params.ctx.llmBudgetAudit, {
			label: "query-compile",
			stage: "query_hot_path",
			provenance: "deterministic",
			mode: "fallback",
			detail: "query compiler LLM exceeded hot-path timeout"
		});
		return compileQueryWithoutSemanticFallback(params.query, "query-compile-llm-timeout");
	}
	if (compiledResult.status === "rejected") {
		recordMemoryLlmBudgetCall(params.ctx.llmBudgetAudit, {
			label: "query-compile",
			stage: "query_hot_path",
			provenance: "deterministic",
			mode: "fallback",
			detail: "query compiler LLM rejected on hot path"
		});
		return compileQueryWithoutSemanticFallback(params.query, "query-compile-llm-error");
	}
	const compiled = compiledResult.value;
	if (!compiled) return compileQueryWithoutSemanticFallback(params.query, "query-compile-llm-empty");
	return applyQueryCompileGuards(params.query, mergeCompiledQuery(fallback, {
		compilerProvenance: {
			source: "llm",
			mode: "llm"
		},
		...compiled
	}));
}
//#endregion
export { buildQueryCompilerPromptInput, compileQuery, compileQueryWithoutSemanticFallback };
