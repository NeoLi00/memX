import { clamp01, normalizeText, stableHash, truncateText } from "../support.mjs";
import { looksLikeBareMemoryUseInstruction } from "./semantic/heuristics.mjs";
import { semanticTextSimilarity } from "./semantic/textSimilarity.mjs";
import { normalizeSourceRefs, promptLineRole } from "./sourceRefs.mjs";
import { stateCurrentnessFromVectorMetadata } from "./stateLifecycle.mjs";
//#region src/pipeline/evidenceAssembler.ts
const HIGH_LEVEL_LAYERS = new Set([
	"control",
	"strategy",
	"abstraction",
	"belief",
	"graph",
	"entity_alias"
]);
const TEMPORAL_REQUIRED_FIELDS = new Set([
	"temporal_marker",
	"date",
	"time",
	"observed_at"
]);
const GENERIC_REQUIRED_FIELDS = new Set([
	"answer_value",
	"attribute_value",
	"countable_item",
	"source_evidence",
	"query_context",
	"temporal_marker",
	"date",
	"time",
	"observed_at"
]);
const HARD_EXCLUSION_PATTERNS = [/\b(?:BOOTSTRAP|IDENTITY|MEMORY\.md|USER\.md|debug|trace|stack trace)\b/iu, /\b(?:memory search|database inspection|plugin debugging|workspace file inspection)\b/iu];
function slotLayers(slot) {
	return [...new Set([...slot.preferredLayers, ...slot.fallbackLayers])];
}
function operationType(queryAnalysis) {
	return queryAnalysis.evidencePlan?.operation.type ?? "return_value";
}
function evidenceSlotRequiredRole(slot) {
	if (!slot) return;
	if (slot.requiredRole) return slot.requiredRole;
	if (slot.id === "current_need") return "query_context";
	if (slot.id === "relevant_user_resources") return "user_resource";
	if (slot.id === "prior_advice_or_strategy") return "prior_advice";
	if (slot.role === "answer_value" || slot.role === "answer_evidence") return "answer_value";
	if (slot.role === "answer_event" || slot.role === "time_constraint") return slot.role;
	if (slot.role === "query_context" || slot.role === "user_resource" || slot.role === "prior_advice") return slot.role;
}
function slotNeedsTemporalField(queryAnalysis, slot) {
	const op = operationType(queryAnalysis);
	if (op === "derive" || op === "compare") return true;
	return slot.requiredFields.some((field) => TEMPORAL_REQUIRED_FIELDS.has(field));
}
function sourceRefsForEntry(entry) {
	const metadataSourceRef = typeof entry.metadata?.sourceRef === "string" ? entry.metadata.sourceRef : void 0;
	const metadataSupportRefs = Array.isArray(entry.metadata?.supportRefs) ? entry.metadata.supportRefs.filter((value) => typeof value === "string" && value.trim().length > 0) : [];
	const metadataLineage = entry.metadata?.lineage && typeof entry.metadata.lineage === "object" ? entry.metadata.lineage : void 0;
	const metadataLineageSourceRef = typeof metadataLineage?.sourceRef === "string" ? metadataLineage.sourceRef : void 0;
	return [...new Set([
		entry.sourceRef,
		entry.lineage?.sourceRef,
		metadataSourceRef,
		metadataLineageSourceRef,
		...entry.mergedSourceRefs ?? [],
		...metadataSupportRefs
	].filter((value) => Boolean(value)))];
}
function supportRefsForEntry(entry) {
	const metadataSupportRefs = Array.isArray(entry.metadata?.supportRefs) ? entry.metadata.supportRefs.filter((value) => typeof value === "string" && value.trim().length > 0) : [];
	return [...new Set([...entry.mergedSourceRefs ?? [], ...metadataSupportRefs])];
}
function bindingSourceRefsForEntry(entry) {
	if (entry.sourceRef) return [entry.sourceRef];
	if (entry.lineage?.sourceRef) return [entry.lineage.sourceRef];
	if (typeof entry.metadata?.sourceRef === "string") return [entry.metadata.sourceRef];
	const metadataLineage = entry.metadata?.lineage && typeof entry.metadata.lineage === "object" ? entry.metadata.lineage : void 0;
	if (typeof metadataLineage?.sourceRef === "string") return [metadataLineage.sourceRef];
	return [...new Set((entry.mergedSourceRefs ?? []).filter(Boolean))];
}
function sourceFamilyRef(sourceRef) {
	const parts = sourceRef.split(":").filter(Boolean);
	const normalizeSessionPart = (part) => /^answer_[a-z0-9]+_\d+$/iu.test(part) ? part.replace(/_\d+$/u, "") : part;
	if (parts[0] === "user" && parts[1] === "agentmem" && parts.length >= 4) return `agentmem:${parts[2]}:${normalizeSessionPart(parts[3] ?? "")}`;
	if (parts[0] === "agentmem" && parts.length >= 3) return `agentmem:${parts[1]}:${normalizeSessionPart(parts[2] ?? "")}`;
	if (parts[0] === "user" && parts[1] === "lme" && parts.length >= 4) return `user:lme:${parts[2]}:${normalizeSessionPart(parts[3] ?? "")}`;
	if (parts[0] === "lme" && parts.length >= 3) return `lme:${parts[1]}:${normalizeSessionPart(parts[2] ?? "")}`;
	if (parts.length >= 3) return parts.slice(0, 3).join(":");
	return sourceRef.replace(/(?:[:/_-](?:turn|message|chunk|answer)?[:/_-]?\d+)$/iu, "");
}
function sourceTurnIndex(sourceRef) {
	const match = /(?:^|[:/_-])(?:turn|message|chunk|answer)?[:/_-]?(\d+)(?=$|[:/_-](?:user|assistant|tool|memory)$)/iu.exec(sourceRef);
	if (!match?.[1]) return;
	const value = Number.parseInt(match[1], 10);
	return Number.isFinite(value) ? value : void 0;
}
function sourceSessionId(sourceRef) {
	const parts = sourceRef.split(":").filter(Boolean);
	if (parts[0] === "user" && parts[1] === "agentmem" && parts.length >= 4) return parts[3];
	if (parts[0] === "agentmem" && parts.length >= 3) return parts[2];
	if (parts[0] === "user" && parts[1] === "lme" && parts.length >= 4) return parts[3];
	if (parts[0] === "lme" && parts.length >= 3) return parts[2];
}
function sourceRefsAdjacent(leftRefs, rightRefs, maxDistance = 2) {
	for (const left of leftRefs) {
		const leftFamily = sourceFamilyRef(left);
		const leftTurn = sourceTurnIndex(left);
		for (const right of rightRefs) {
			if (leftFamily !== sourceFamilyRef(right)) continue;
			const rightTurn = sourceTurnIndex(right);
			if (leftTurn === void 0 || rightTurn === void 0) {
				if (left === right) return true;
				continue;
			}
			if (Math.abs(leftTurn - rightTurn) <= maxDistance) return true;
		}
	}
	return false;
}
function sourceKey(sourceRefs, fallbackText) {
	return sourceRefs.length > 0 ? `source:${sourceRefs.sort().join("|")}` : `text:${normalizeText(fallbackText).slice(0, 180)}`;
}
function entryKey(entry) {
	return [
		entry.surface,
		entry.id,
		entry.sourceRef ?? "",
		normalizeText(entry.text).slice(0, 80)
	].filter(Boolean).join("|");
}
function entryAuthorRole(entry) {
	const text = (entry.rawText ?? entry.text).trim().toLowerCase();
	if (/^(?:\[assistant\]|assistant)\s*:/iu.test(text) || text.startsWith("assistant ")) return "assistant";
	if (/^(?:\[user\]|user)\s*:/iu.test(text)) return "user";
	if (/^(?:\[tool\]|tool)\s*:/iu.test(text)) return "tool";
	const roleFromRefs = (sourceRefs) => {
		const hasUser = sourceRefs.some((ref) => /(?:^|:)user$/iu.test(ref));
		const hasAssistant = sourceRefs.some((ref) => /(?:^|:)assistant$/iu.test(ref));
		const hasTool = sourceRefs.some((ref) => /(?:^|:)tool$/iu.test(ref));
		if (hasUser && !hasAssistant && !hasTool) return "user";
		if (hasAssistant && !hasUser && !hasTool) return "assistant";
		if (hasTool && !hasUser && !hasAssistant) return "tool";
	};
	const primaryRole = roleFromRefs(bindingSourceRefsForEntry(entry));
	if (primaryRole) return primaryRole;
	const sourceRole = roleFromRefs(sourceRefsForEntry(entry));
	if (sourceRole) return sourceRole;
	if (sourceRefsForEntry(entry).some((ref) => /(?:^|:)user$/iu.test(ref))) return "user";
	if (sourceRefsForEntry(entry).some((ref) => /(?:^|:)tool$/iu.test(ref))) return "tool";
	if (entry.surface === "fact" || entry.surface === "event") return "memory";
	return "unknown";
}
function evidenceUnitOrigin(entry) {
	const sourceId = entry.lineage?.sourceId ?? entry.id;
	if (sourceId.startsWith("belief:")) return "belief";
	if (sourceId.startsWith("strategy:")) return "strategy";
	if (entry.metadata?.recallLayer === "belief") return "belief";
	if (entry.metadata?.recallLayer === "strategy") return "strategy";
	if (entry.surface === "chunk") return entry.source === "support_ref" || entry.lineage?.sourceKind === "chunk" ? "raw_chunk" : "derived_summary";
	if (entry.surface === "fact") return "canonical_fact";
	if (entry.surface === "event") return entry.source === "support_ref" ? "raw_chunk" : "event";
	return "snippet";
}
function evidenceUnitRoles(queryAnalysis, entry) {
	const role = inferredSlotRole(queryAnalysis, entry);
	const roles = [];
	if (role) roles.push(role);
	if (entry.source === "support_ref" && !roles.includes("support")) roles.push("support");
	if (roles.length === 0) roles.push("support");
	return [...new Set(roles)];
}
function evidenceUnitFromEntry(queryAnalysis, entry) {
	const sourceRefs = sourceRefsForEntry(entry);
	const supportRefs = supportRefsForEntry(entry);
	const primarySourceRef = sourceRefs[0];
	return {
		unitId: `unit:${stableHash([
			entry.surface,
			entry.id,
			...sourceRefs,
			normalizeText(entry.text).slice(0, 160)
		])}`,
		surfaceRefs: [entry.id],
		sourceRefs,
		normalizedSourceRefs: normalizeSourceRefs(sourceRefs),
		supportRefs,
		normalizedSupportRefs: normalizeSourceRefs(supportRefs),
		derivedFromRefs: entry.lineage?.sourceRef ? [entry.lineage.sourceRef] : [],
		normalizedDerivedFromRefs: normalizeSourceRefs(entry.lineage?.sourceRef ? [entry.lineage.sourceRef] : []),
		neighborRefs: entry.metadata?.sourceExpansion === true && Array.isArray(entry.metadata.neighborOf) ? entry.metadata.neighborOf.filter((value) => typeof value === "string" && value.trim().length > 0) : [],
		normalizedNeighborRefs: normalizeSourceRefs(entry.metadata?.sourceExpansion === true && Array.isArray(entry.metadata.neighborOf) ? entry.metadata.neighborOf : []),
		sessionId: primarySourceRef ? sourceSessionId(primarySourceRef) : void 0,
		turnIndex: primarySourceRef ? sourceTurnIndex(primarySourceRef) : void 0,
		authorRole: entryAuthorRole(entry),
		observedAt: entry.observedAt,
		rawText: entry.rawText ?? entry.text,
		displayText: truncateText(entry.text, 560),
		roles: evidenceUnitRoles(queryAnalysis, entry),
		origin: evidenceUnitOrigin(entry)
	};
}
const reportedQuestionPrefixPattern = /^\s*(?:(?:\[user\]|user)\s*:?\s*)?(?:they\s+also\s+ask|user\s+asks?|(?:a|the)\s+reviewer\s+asks?|security\s+review\s+asks?|(?:security\s+)?reviewer\s+asks?|question\s*:)/iu;
function entryIsQuestionLike(entry) {
	const raw = entry.rawText ?? entry.text;
	const trimmed = raw.trim();
	return /\?\s*$/u.test(trimmed) || reportedQuestionPrefixPattern.test(raw);
}
function entryIsQueryLikeEvidence(queryAnalysis, entry) {
	const author = entryAuthorRole(entry);
	if (author === "assistant" || author === "tool") return false;
	const text = entry.rawText ?? entry.text;
	if (reportedQuestionPrefixPattern.test(text)) return true;
	const echo = queryEchoScore(queryAnalysis, text);
	return echo >= .86 || entryIsQuestionLike(entry) && echo >= .52;
}
function unitIsQuestionLike(queryAnalysis, unit) {
	if (unit.authorRole === "assistant" || unit.authorRole === "tool") return false;
	const text = unit.rawText || unit.displayText;
	return /\?\s*$/u.test(text.trim()) || reportedQuestionPrefixPattern.test(text) || queryEchoScore(queryAnalysis, text) >= .86;
}
function unitHasAnswerRole(unit) {
	return unit.roles.some((role) => role === "answer_value" || role === "answer_event" || role === "user_resource" || role === "prior_advice");
}
function unitLooksLikeAnswer(queryAnalysis, unit) {
	if (unitIsQuestionLike(queryAnalysis, unit) && !unit.roles.includes("user_resource")) return false;
	if (unitHasAnswerRole(unit)) return true;
	if (unit.authorRole === "assistant") return true;
	return unit.origin === "canonical_fact" || unit.origin === "event" || unit.origin === "raw_chunk" || unit.origin === "snippet";
}
function refsForUnit(unit) {
	return [...new Set([
		...unit.sourceRefs,
		...unit.supportRefs ?? [],
		...unit.derivedFromRefs ?? [],
		...unit.neighborRefs ?? []
	])];
}
function exactDisplayKey(text) {
	return normalizeText(text).replace(/\s+/gu, " ").trim();
}
function compactResourceDisplay(text) {
	const resource = /(?:^|\|\s*)resource\s*=\s*([^|]+)/iu.exec(text)?.[1]?.trim() ?? /(?:has_resource|user\.resource\.)([^|:(]+)/iu.exec(text)?.[1]?.replace(/[_-]+/gu, " ").trim();
	if (!resource) return;
	const affordances = /affordances\s*=\s*([^|]+)/iu.exec(text)?.[1]?.trim();
	const domains = /domains\s*=\s*([^|]+)/iu.exec(text)?.[1]?.trim();
	const support = /supportText\s*=\s*([^|]+)/iu.exec(text)?.[1]?.trim() ?? /evidence:\s*([^|]+)/iu.exec(text)?.[1]?.trim();
	return truncateText([
		`The user has ${resource}`,
		affordances ? `useful for ${affordances}` : void 0,
		domains ? `domain: ${domains}` : void 0,
		support ? `evidence: ${support}` : void 0
	].filter(Boolean).join(" | "), 360);
}
function displayLabelForUnit(unit) {
	if (unit.roles.includes("user_resource")) return "resource";
	if (unit.roles.includes("answer_event") || unit.origin === "event") return "event";
	if (unit.roles.includes("query_context") || unit.roles.includes("time_constraint")) return "context";
	return "answer";
}
function displayLabelForAnswerUnit(unit) {
	if (unit.roles.includes("user_resource")) return "resource";
	if (unit.roles.includes("answer_value")) return "answer";
	if (unit.roles.includes("answer_event") || unit.origin === "event") return "event";
	if (unit.roles.includes("query_context") && unit.authorRole !== "assistant" && !unitHasAnswerRole(unit)) return "context";
	return "answer";
}
function unitDisplayText(unit, maxLength = 360) {
	const raw = unit.displayText || unit.rawText;
	return truncateText((unit.roles.includes("user_resource") ? compactResourceDisplay(raw) : void 0) ?? raw, maxLength);
}
function classifyPacketUnits(params) {
	const selectedUnit = evidenceUnitFromEntry(params.queryAnalysis, params.selectedEntry);
	const contextUnits = params.contextCandidates.map((entry) => evidenceUnitFromEntry(params.queryAnalysis, entry));
	const selectedRole = inferredSlotRole(params.queryAnalysis, params.selectedEntry);
	entryAuthorRole(params.selectedEntry);
	const selectedIsQuestion = !(selectedRole === "user_resource" || selectedRole === "prior_advice") && entryIsQueryLikeEvidence(params.queryAnalysis, params.selectedEntry);
	const contextAnswerUnits = contextUnits.filter((unit) => unitLooksLikeAnswer(params.queryAnalysis, unit));
	const selectedAssistantAnswer = selectedUnit.authorRole === "assistant" && !selectedIsQuestion && unitLooksLikeAnswer(params.queryAnalysis, selectedUnit);
	if (selectedIsQuestion || (selectedRole === "query_context" || selectedRole === "time_constraint") && !selectedAssistantAnswer) {
		const answerUnits = contextAnswerUnits.filter((unit) => (unit.authorRole === "assistant" || unitHasAnswerRole(unit)) && !unitIsQuestionLike(params.queryAnalysis, unit));
		if (answerUnits.length > 0) return {
			answerUnits,
			contextUnits: [selectedUnit, ...contextUnits.filter((unit) => !answerUnits.includes(unit))],
			supportUnits: []
		};
	}
	if (unitLooksLikeAnswer(params.queryAnalysis, selectedUnit) && !selectedIsQuestion) return {
		answerUnits: [selectedUnit],
		contextUnits: contextUnits.filter((unit) => unit.roles.includes("query_context") || unit.roles.includes("time_constraint")),
		supportUnits: contextUnits.filter((unit) => !unit.roles.includes("query_context") && !unit.roles.includes("time_constraint"))
	};
	return {
		answerUnits: contextAnswerUnits,
		contextUnits: [selectedUnit, ...contextUnits.filter((unit) => !contextAnswerUnits.includes(unit))],
		supportUnits: []
	};
}
function packetDisplayLines(params) {
	const displayLines = [];
	const hiddenExactDuplicates = [];
	const seen = /* @__PURE__ */ new Map();
	const contextText = params.contextUnits.map((unit) => unitDisplayText(unit, 220)).find((text) => text.length > 0);
	const units = params.answerUnits.length > 0 ? params.answerUnits : [...params.contextUnits, ...params.supportUnits].slice(0, 1);
	for (const unit of units) {
		const answerText = unitDisplayText(unit, 420);
		if (!answerText) continue;
		const answerLabel = params.answerUnits.includes(unit) ? displayLabelForAnswerUnit(unit) : unitIsQuestionLike(params.queryAnalysis, unit) && !unit.roles.includes("user_resource") ? "context" : displayLabelForUnit(unit);
		const line = contextText && semanticTextSimilarity(answerText, contextText) < .78 ? `[${answerLabel}] ${answerText} | [context] ${contextText}` : `[${answerLabel}] ${answerText}`;
		const key = exactDisplayKey(line);
		const refs = refsForUnit(unit);
		const existing = seen.get(key);
		if (existing) {
			hiddenExactDuplicates.push({
				displayText: existing.displayText,
				sourceRefs: existing.sourceRefs,
				hiddenSourceRefs: refs
			});
			continue;
		}
		seen.set(key, {
			displayText: line,
			sourceRefs: refs
		});
		displayLines.push(line);
	}
	return {
		displayLines,
		hiddenExactDuplicates
	};
}
function queryEchoScore(queryAnalysis, text) {
	return Math.max(semanticTextSimilarity(queryAnalysis.queryText, text), semanticTextSimilarity(queryAnalysis.focusedQuery || queryAnalysis.queryText, text));
}
function numericBreakdown(entry, key) {
	const value = entry.scoreBreakdown?.[key];
	return typeof value === "number" ? clamp01(value) : 0;
}
function slotCoverageScore(entry) {
	const explicitCoverage = entry.coverage && (entry.coverage.requiredHits.length > 0 || entry.coverage.missingRequired.length > 0) ? entry.coverage.coverageScore * (entry.coverage.missingRequired.length > 0 ? .45 : 1) : 0;
	return Math.max(explicitCoverage, ...(entry.slotCoverage ?? []).map((coverage) => coverage.coverageScore * (coverage.missingRequired.length > 0 ? .45 : 1)));
}
function bridgeSupportScore(entry) {
	return Math.max(0, ...(entry.bridgeMatches ?? []).map((match) => match.score));
}
function bridgePositiveSignalScore(entry) {
	return Math.max(0, ...(entry.bridgeMatches ?? []).map((match) => match.positiveSignalScore));
}
function bridgeNegativeSignalScore(entry) {
	return Math.max(0, ...(entry.bridgeMatches ?? []).filter((match) => match.role !== "answer_value" && match.role !== "answer_event" && match.role !== "user_resource" && match.role !== "prior_advice").map((match) => match.negativeSignalScore));
}
function bridgeAnswerRoleScore(entry) {
	return Math.max(0, ...(entry.bridgeMatches ?? []).filter((match) => match.role === "answer_value" || match.role === "answer_event" || match.role === "user_resource" || match.role === "prior_advice").map((match) => match.score));
}
function bridgeRoleScore(entry, roles) {
	return Math.max(0, ...(entry.bridgeMatches ?? []).filter((match) => roles.includes(match.role)).map((match) => match.score));
}
function bridgePositiveRoleScore(entry, roles) {
	return Math.max(0, ...(entry.bridgeMatches ?? []).filter((match) => roles.includes(match.role)).map((match) => match.positiveSignalScore));
}
function bridgeShapeScore(entry, shapes) {
	return Math.max(0, ...(entry.bridgeMatches ?? []).filter((match) => shapes.includes(match.evidenceShape)).map((match) => match.score));
}
function bridgeShapePositiveScore(entry, shapes) {
	return Math.max(0, ...(entry.bridgeMatches ?? []).filter((match) => shapes.includes(match.evidenceShape)).map((match) => match.positiveSignalScore));
}
function semanticBridgesForShapes(queryAnalysis, shapes) {
	return (queryAnalysis.semanticBridges ?? []).filter((bridge) => shapes.includes(bridge.evidenceShape));
}
function queryUsesEvidenceShapes(queryAnalysis, shapes) {
	return semanticBridgesForShapes(queryAnalysis, shapes).length > 0;
}
function semanticBridgePlanTexts(bridge) {
	return [
		bridge.sourceConcept,
		bridge.evidenceShape,
		...bridge.retrievalQueries,
		...bridge.positiveSignals
	].map((text) => text.trim()).filter(Boolean);
}
function evidenceShapeFitScore(queryAnalysis, entry, shapes) {
	const bridges = semanticBridgesForShapes(queryAnalysis, shapes);
	if (bridges.length === 0) return 0;
	const candidateText = `${entry.text} ${entry.scoringText ?? ""}`;
	const bridgeMatch = Math.max(bridgeShapeScore(entry, shapes), bridgeShapePositiveScore(entry, shapes));
	if (shapes.some((shape) => shape === "validation_evidence" || shape === "status_answer" || shape === "decision_value" || shape === "availability_statement")) return clamp01(bridgeMatch);
	const planMatch = Math.max(0, ...bridges.flatMap((bridge) => semanticBridgePlanTexts(bridge)).map((text) => planTextMatchScore(text, candidateText)));
	return clamp01(Math.max(bridgeMatch, planMatch));
}
function evidenceShapeFitScoreForTexts(queryAnalysis, texts, shapes) {
	const bridges = semanticBridgesForShapes(queryAnalysis, shapes);
	if (bridges.length === 0 || texts.length === 0) return 0;
	const planTexts = bridges.flatMap((bridge) => semanticBridgePlanTexts(bridge));
	return Math.max(0, ...planTexts.flatMap((planText) => texts.map((text) => planTextMatchScore(planText, text))));
}
function strongestBridgeRole(entry) {
	const match = [...entry.bridgeMatches ?? []].sort((left, right) => right.score - left.score)[0];
	return match?.score && match.score >= .34 ? match.role : void 0;
}
function filledSlotIds(entry) {
	return entry.filledSlotIds && entry.filledSlotIds.length > 0 ? entry.filledSlotIds : (entry.slotCoverage ?? []).filter((coverage) => coverage.filled).map((coverage) => coverage.slotId);
}
function matchedSlotIds(entry) {
	return [...new Set([...filledSlotIds(entry), ...(entry.slotCoverage ?? []).filter((coverage) => coverage.coverageScore >= .24).map((coverage) => coverage.slotId)].filter(Boolean))];
}
function normalizedHitSet(entries) {
	const hits = /* @__PURE__ */ new Set();
	for (const entry of entries) {
		for (const hit of entry.coverage?.requiredHits ?? []) {
			const normalized = normalizeText(hit);
			if (normalized) hits.add(normalized);
		}
		for (const coverage of entry.slotCoverage ?? []) for (const hit of coverage.requiredHits) {
			const normalized = normalizeText(hit);
			if (normalized) hits.add(normalized);
		}
	}
	return hits;
}
function missingRequiredAfterContext(entry, contextCandidates) {
	const entries = [entry, ...contextCandidates];
	const hits = normalizedHitSet(entries);
	const isCovered = (value) => {
		const normalized = normalizeText(value);
		if (!normalized) return true;
		if (GENERIC_REQUIRED_FIELDS.has(normalized.replace(/\s+/gu, "_"))) return true;
		if (hits.has(normalized)) return true;
		return entries.some((candidate) => {
			const text = normalizeText(`${candidate.text} ${candidate.scoringText ?? ""}`);
			const valueTokens = normalized.split(/\s+/gu).filter((token) => token.length >= 4);
			const tokenCoverage = valueTokens.length > 0 ? valueTokens.filter((token) => text.includes(token)).length / Math.min(valueTokens.length, 4) : 0;
			return text.includes(normalized) || tokenCoverage >= .5 || semanticTextSimilarity(value, candidate.text) >= .5;
		});
	};
	const missing = /* @__PURE__ */ new Set();
	for (const candidate of entries) {
		for (const value of candidate.coverage?.missingRequired ?? []) if (!isCovered(value)) missing.add(value);
		for (const coverage of candidate.slotCoverage ?? []) for (const value of coverage.missingRequired) if (!isCovered(value)) missing.add(value);
	}
	return [...missing];
}
function inferredSlotRole(queryAnalysis, entry) {
	if (entry.slotEvidenceRole) return entry.slotEvidenceRole;
	const slots = queryAnalysis.evidencePlan?.slots ?? [];
	const slotById = new Map(slots.map((slot) => [slot.id, slot]));
	const answerSlotIds = new Set(answerPlanSlots(queryAnalysis).map((slot) => slot.id));
	const coverageRole = (entry.slotCoverage ?? []).map((coverage) => {
		const slot = slotById.get(coverage.slotId);
		const explicitRole = evidenceSlotRequiredRole(slot);
		const inferredAnswerRole = !explicitRole && slot && answerSlotIds.has(slot.id) ? operationType(queryAnalysis) === "aggregate" || queryAnalysis.answerMode === "count_aggregate" ? "answer_event" : operationType(queryAnalysis) === "tailor_advice" ? "user_resource" : "answer_value" : void 0;
		return {
			role: explicitRole ?? inferredAnswerRole,
			score: coverage.filled ? coverage.coverageScore + .18 : coverage.coverageScore
		};
	}).filter((item) => item.role).sort((left, right) => right.score - left.score)[0]?.role;
	const bridgeRole = strongestBridgeRole(entry);
	if (bridgeRole && (bridgeRole === "answer_value" || bridgeRole === "answer_event" || bridgeRole === "user_resource" || bridgeRole === "prior_advice")) {
		if (!coverageRole || coverageRole === "query_context" || coverageRole === "time_constraint" || bridgeRoleScore(entry, [bridgeRole]) >= .52) return bridgeRole;
	}
	return coverageRole ?? bridgeRole;
}
function metadataStringArray(metadata, key) {
	const value = metadata?.[key];
	if (!Array.isArray(value)) return [];
	return value.filter((item) => typeof item === "string" && item.trim().length > 0);
}
function hasStructuredResourceSignal(entry) {
	return Boolean(entry.slotEvidenceRole === "user_resource" || bridgeShapeScore(entry, ["resource_affordance"]) > 0 || metadataStringArray(entry.metadata, "domains").length > 0 || metadataStringArray(entry.metadata, "affordances").length > 0 || typeof entry.metadata?.resourceType === "string" && entry.metadata.resourceType.trim().length > 0 || entry.metadata?.signalKind === "resourceAssertion");
}
function hasResourceAffordanceSignal(entry) {
	return Boolean(bridgeShapeScore(entry, ["resource_affordance"]) > 0 || metadataStringArray(entry.metadata, "domains").length > 0 || metadataStringArray(entry.metadata, "affordances").length > 0 || typeof entry.metadata?.resourceType === "string" && entry.metadata.resourceType.trim().length > 0 || entry.metadata?.signalKind === "resourceAssertion");
}
function isAssistantAcknowledgement(entry) {
	if (entryAuthorRole(entry) !== "assistant") return false;
	return entry.metadata?.semanticRole === "assistant_acknowledgement" || entry.metadata?.memoryClass === "assistant_acknowledgement";
}
function candidateRetrievalScore(entry) {
	return clamp01(Math.max(entry.injectionScore ?? 0, entry.priority, entry.goalScore, entry.semanticScore ?? 0, numericBreakdown(entry, "retrievalScore")));
}
function isStateEvidence(entry) {
	return entry.metadata?.memxDocType === "state" || typeof entry.metadata?.stateLifecycleKind === "string" || entry.lineage?.sourceKind === "state" || entry.lineage?.canonicalKind === "state";
}
function stateHardExclusions(entry, now) {
	if (!isStateEvidence(entry)) return [];
	const fromMetadata = stringArray(entry.metadata?.stateCurrentnessHardExclusions);
	const currentness = stateCurrentnessFromVectorMetadata(entry.metadata, now);
	const lifecycleKind = typeof entry.metadata?.stateLifecycleKind === "string" ? entry.metadata.stateLifecycleKind : void 0;
	const rawSupportRefs = stringArray(entry.metadata?.stateSupportRefs).filter((ref) => !ref.startsWith("abstraction_candidate:"));
	const supportOnlyBlockers = lifecycleKind === "derived_maintenance" && rawSupportRefs.length === 0 ? ["maintenance-state-missing-raw-support"] : [];
	return [...new Set([
		...fromMetadata,
		...currentness?.hardExclusions ?? [],
		...supportOnlyBlockers
	])];
}
function stateSoftPenalties(entry) {
	if (!isStateEvidence(entry)) return [];
	const penalties = stringArray(entry.metadata?.stateCurrentnessSoftPenalties).map((reason) => ({
		reason,
		weight: reason === "maintenance-derived-without-raw-support" ? .22 : .1
	}));
	if ((typeof entry.metadata?.stateCurrentnessScore === "number" ? clamp01(entry.metadata.stateCurrentnessScore) : .5) < .42) penalties.push({
		reason: "low-state-currentness",
		weight: .16
	});
	if (entry.metadata?.stateAnswerEligibleByDefault === false) penalties.push({
		reason: "state-support-only-by-default",
		weight: .12
	});
	return penalties;
}
function authorityScore(entry) {
	const stateLifecycleKind = typeof entry.metadata?.stateLifecycleKind === "string" ? entry.metadata.stateLifecycleKind : void 0;
	if (isStateEvidence(entry)) return clamp01((stateLifecycleKind === "derived_maintenance" ? .42 : stateLifecycleKind === "durable_profile" ? .66 : .55) + (entry.sourceRef || (entry.mergedSourceRefs?.length ?? 0) > 0 ? .08 : 0));
	const author = entryAuthorRole(entry);
	const surfaceScore = entry.surface === "fact" ? .86 : entry.surface === "event" ? .78 : entry.surface === "chunk" ? author === "assistant" ? .66 : .7 : .58;
	const sourceTrace = entry.sourceRef || (entry.mergedSourceRefs?.length ?? 0) > 0 ? .08 : 0;
	const sourceScore = entry.source === "selected" ? .05 : entry.source === "candidate" ? .04 : entry.source === "support_ref" ? .03 : 0;
	const resourceScore = hasStructuredResourceSignal(entry) ? .08 : 0;
	return clamp01(surfaceScore + sourceTrace + sourceScore + resourceScore);
}
function stringArray(value) {
	if (!Array.isArray(value)) return [];
	return value.filter((item) => typeof item === "string" && item.trim().length > 0);
}
function sameSemanticChainSupport(entry, contextCandidates) {
	const refs = bindingSourceRefsForEntry(entry);
	const families = new Set(refs.map((sourceRef) => sourceFamilyRef(sourceRef)));
	let best = 0;
	for (const candidate of contextCandidates) {
		const candidateRefs = bindingSourceRefsForEntry(candidate);
		if (candidateRefs.length === 0 || refs.length === 0) continue;
		if (candidateRefs.some((sourceRef) => families.has(sourceFamilyRef(sourceRef)))) best = Math.max(best, .74);
		if (sourceRefsAdjacent(refs, candidateRefs, 2)) best = Math.max(best, .86);
	}
	return best;
}
function contextBindingScore(params) {
	const operation = operationType(params.queryAnalysis);
	if (params.slotRole === "query_context") return .22;
	if (params.slotRole === "time_constraint") return .18;
	const filledRoles = new Set([params.entry, ...params.contextCandidates].map((candidate) => inferredSlotRole(params.queryAnalysis, candidate)).filter(Boolean));
	const entryRoles = new Set([inferredSlotRole(params.queryAnalysis, params.entry), ...(params.entry.bridgeMatches ?? []).filter((match) => match.score >= .44).map((match) => match.role)]);
	const hasContext = filledRoles.has("query_context") || entryRoles.has("query_context") || queryContextSupportScore(params.queryAnalysis, params.entry) >= .24 || (params.entry.slotCoverage ?? []).some((coverage) => {
		return evidenceSlotRequiredRole((params.queryAnalysis.evidencePlan?.slots ?? []).find((candidate) => candidate.id === coverage.slotId)) === "query_context" && coverage.filled;
	});
	const contextSemanticSupport = Math.max(queryContextSupportScore(params.queryAnalysis, params.entry), 0, ...params.contextCandidates.map((candidate) => queryContextSupportScore(params.queryAnalysis, candidate)));
	const samePacketSupport = params.contextCandidates.some((candidate) => (inferredSlotRole(params.queryAnalysis, candidate) === "query_context" || semanticTextSimilarity(candidate.text, params.entry.text) >= .34) && queryContextSupportScore(params.queryAnalysis, candidate) >= .24);
	const sameChainSupport = sameSemanticChainSupport(params.entry, params.contextCandidates);
	const contextSupport = Math.max(contextSemanticSupport >= .24 ? sameChainSupport : 0, samePacketSupport && params.contextCandidates.length > 0 ? .42 : 0, contextSemanticSupport);
	if (operation === "tailor_advice" && params.slotRole === "user_resource") return contextSupport >= .42 || hasContext ? .72 : .58;
	if (operation === "tailor_advice" && params.slotRole === "prior_advice") {
		if (Math.max(semanticTextSimilarity(params.queryAnalysis.queryText, params.entry.text), semanticTextSimilarity(params.queryAnalysis.focusedQuery ?? "", params.entry.text)) < .2 && contextSupport < .42) return .24;
		return contextSupport >= .42 ? .68 : .48;
	}
	const lookupMode = operation === "return_value" || params.queryAnalysis.answerMode === "attribute_lookup";
	if (lookupMode && params.slotRole === "answer_value" && queryAsksSensitiveValue(params.queryAnalysis) && statesUnavailableSensitiveValue(params.queryAnalysis, params.entry)) return Math.max(contextSupport, .68);
	if (lookupMode) {
		if ((entryRoles.has("answer_value") || entryRoles.has("answer_event")) && entryRoles.has("query_context")) return .86;
		if (hasContext && contextSupport >= .74) return .86;
		if (hasContext && contextSupport >= .42) return .68;
		if (hasContext) return contextSemanticSupport >= .24 ? .46 : .28;
		return contextSemanticSupport >= .24 ? .28 : .12;
	}
	if (hasContext && contextSupport >= .74) return .82;
	if (hasContext && contextSupport >= .42) return .62;
	if (hasContext) return contextSemanticSupport >= .24 ? .62 : .36;
	return .42;
}
function answerPlanSlots(queryAnalysis) {
	const operation = operationType(queryAnalysis);
	const slots = queryAnalysis.evidencePlan?.slots ?? [];
	const answerSlots = slots.filter((slot) => {
		const role = evidenceSlotRequiredRole(slot);
		if (role === "answer_value" || role === "answer_event") return true;
		if (operation === "tailor_advice" && (role === "user_resource" || role === "prior_advice")) return true;
		return slot.role === "answer_evidence";
	});
	if (answerSlots.length > 0) return answerSlots;
	return slots.filter((slot) => {
		const role = evidenceSlotRequiredRole(slot);
		return role !== "query_context" && role !== "time_constraint";
	});
}
function answerRelationFitScore(queryAnalysis, entry) {
	const relationTexts = [...answerPlanSlots(queryAnalysis).flatMap((slot) => [...slot.relationHints ?? [], ...slot.capabilityQueries ?? []]), ...(queryAnalysis.semanticBridges ?? []).filter((bridge) => bridge.role === "answer_value" || bridge.role === "answer_event").flatMap((bridge) => bridge.positiveSignals ?? [])].map((text) => text.trim()).filter(Boolean);
	if (relationTexts.length === 0) return .5;
	const candidateText = `${entry.text} ${entry.scoringText ?? ""}`;
	return Math.max(0, ...relationTexts.map((text) => planTextMatchScore(text, candidateText)));
}
function hasStrongAnswerPlanningMatch(queryAnalysis, entry) {
	return slotCoverageScore(entry) >= .62 || bridgeAnswerRoleScore(entry) >= .5 || bridgePositiveRoleScore(entry, ["answer_value", "answer_event"]) >= .5 || evidenceGoalDomainScore(queryAnalysis, entry) >= .48;
}
function querySeeksCausalAnswer(queryAnalysis) {
	return queryUsesEvidenceShapes(queryAnalysis, ["causal_explanation"]);
}
function causalExplanationScore(queryAnalysis, entry) {
	return evidenceShapeFitScore(queryAnalysis, entry, ["causal_explanation"]);
}
function directCausalExplanationScore(queryAnalysis, entry) {
	return evidenceShapeFitScore(queryAnalysis, entry, ["causal_explanation"]);
}
function validationEvidenceScore(queryAnalysis, entry) {
	return evidenceShapeFitScore(queryAnalysis, entry, ["validation_evidence"]);
}
function statusSummaryEvidenceScore(queryAnalysis, entry) {
	return evidenceShapeFitScore(queryAnalysis, entry, ["status_answer"]);
}
function queryAsksCurrentStatus(queryAnalysis) {
	return queryUsesEvidenceShapes(queryAnalysis, ["status_answer"]);
}
function currentStatusAnswerScore(queryAnalysis, entry) {
	return evidenceShapeFitScore(queryAnalysis, entry, ["status_answer"]);
}
function staleOnlyStatusEvidence(queryAnalysis, entry) {
	return queryUsesEvidenceShapes(queryAnalysis, ["status_answer"]) && evidenceShapeFitScore(queryAnalysis, entry, ["time_constraint"]) > currentStatusAnswerScore(queryAnalysis, entry) + .24;
}
function queryAsksBooleanDecision(queryAnalysis) {
	return queryUsesEvidenceShapes(queryAnalysis, ["decision_value"]);
}
function booleanDecisionFitScore(queryAnalysis, entry) {
	if (!queryAsksBooleanDecision(queryAnalysis)) return .5;
	const relationFit = answerRelationFitScore(queryAnalysis, entry);
	const decisionFit = evidenceShapeFitScore(queryAnalysis, entry, ["decision_value"]);
	if (decisionFit >= .34) return Math.max(decisionFit, relationFit);
	return Math.min(relationFit, .32);
}
function queryAsksSensitiveValue(queryAnalysis) {
	return queryUsesEvidenceShapes(queryAnalysis, ["availability_statement"]);
}
function statesUnavailableSensitiveValue(queryAnalysis, entry) {
	return evidenceShapeFitScore(queryAnalysis, entry, ["availability_statement"]) >= .42;
}
function slotPlanText(slot) {
	const fields = slot.requiredFields.filter((field) => {
		const normalized = normalizeText(field).replace(/\s+/gu, "_");
		return !GENERIC_REQUIRED_FIELDS.has(normalized) && !TEMPORAL_REQUIRED_FIELDS.has(normalized);
	});
	return [
		slot.description,
		...slot.subjectHints,
		...slot.relationHints ?? [],
		...slot.capabilityQueries ?? [],
		...fields
	].map((entry) => entry.trim()).filter(Boolean).join(" ");
}
function queryContextPlanTexts(queryAnalysis) {
	return (queryAnalysis.evidencePlan?.slots ?? []).filter((slot) => evidenceSlotRequiredRole(slot) === "query_context").map((slot) => slotPlanText(slot)).filter(Boolean);
}
function planTextMatchScore(planText, candidateText) {
	const semantic = semanticTextSimilarity(planText, candidateText);
	const planTokens = normalizeText(planText).split(/\s+/gu).filter((token) => token.length >= 4);
	if (planTokens.length === 0) return semantic;
	const normalizedCandidate = normalizeText(candidateText);
	const overlap = planTokens.filter((token) => normalizedCandidate.includes(token)).length / Math.min(planTokens.length, 6);
	return Math.max(semantic, clamp01(overlap * .68));
}
function queryContextSupportScore(queryAnalysis, entry) {
	const planTexts = queryContextPlanTexts(queryAnalysis);
	if (planTexts.length === 0) return 0;
	const candidateText = `${entry.text} ${entry.scoringText ?? ""}`;
	const planSemantic = Math.max(0, ...planTexts.map((text) => planTextMatchScore(text, candidateText)));
	const contextCoverage = Math.max(0, ...(entry.slotCoverage ?? []).filter((coverage) => {
		return evidenceSlotRequiredRole((queryAnalysis.evidencePlan?.slots ?? []).find((candidate) => candidate.id === coverage.slotId)) === "query_context";
	}).map((coverage) => coverage.coverageScore * (coverage.missingRequired.length > 0 ? .45 : 1)));
	const bridgeContext = bridgeRoleScore(entry, ["query_context"]);
	const bridgePositive = bridgePositiveRoleScore(entry, ["query_context"]);
	return clamp01(planSemantic * .5 + contextCoverage * .28 + bridgeContext * .14 + bridgePositive * .08);
}
function evidenceGoalDomainScore(queryAnalysis, entry) {
	const goalTexts = (queryAnalysis.evidenceGoals ?? []).flatMap((goal) => [
		goal.goal,
		...goal.positiveQueries,
		...goal.focusAnchors
	]).map((text) => text.trim()).filter(Boolean);
	if (goalTexts.length === 0) return 0;
	const candidateText = `${entry.text} ${entry.scoringText ?? ""}`;
	return Math.max(0, ...goalTexts.map((text) => planTextMatchScore(text, candidateText)));
}
function contrastNegativeHints(queryAnalysis) {
	const rawHints = [
		...(queryAnalysis.evidenceGoals ?? []).flatMap((goal) => goal.negativeHints ?? []),
		...(queryAnalysis.evidencePlan?.slots ?? []).flatMap((slot) => slot.negativeHints ?? []),
		...(queryAnalysis.semanticBridges ?? []).flatMap((bridge) => bridge.negativeSignals ?? [])
	].map((hint) => hint.trim()).filter(Boolean);
	const queryText = queryAnalysis.queryText || queryAnalysis.focusedQuery || "";
	return [...new Set(rawHints.filter((hint) => {
		if (!normalizeText(hint)) return false;
		return semanticTextSimilarity(hint, queryText) < .42;
	}))];
}
function negativeContrastScore(queryAnalysis, entry) {
	const hints = contrastNegativeHints(queryAnalysis);
	if (hints.length === 0) return 0;
	const candidateText = `${entry.text} ${entry.scoringText ?? ""}`;
	const normalizedCandidate = normalizeText(candidateText);
	const overlapScore = (hint) => {
		const normalizedHint = normalizeText(hint);
		if (!normalizedHint) return 0;
		if (normalizedCandidate.includes(normalizedHint)) return .72;
		const hintTokens = normalizedHint.split(/\s+/gu).filter((token) => token.length >= 4);
		if (hintTokens.length === 0) return 0;
		return hintTokens.filter((token) => normalizedCandidate.includes(token)).length / hintTokens.length * .42;
	};
	return Math.max(0, ...hints.map((hint) => Math.max(semanticTextSimilarity(hint, candidateText), overlapScore(hint))));
}
function tailorAdviceNeedFit(queryAnalysis, entry) {
	const texts = [
		queryAnalysis.queryText,
		queryAnalysis.focusedQuery,
		...(queryAnalysis.evidenceGoals ?? []).flatMap((goal) => [
			goal.goal,
			...goal.positiveQueries,
			...goal.focusAnchors
		]),
		...queryContextPlanTexts(queryAnalysis)
	].map((text) => text?.trim()).filter((text) => Boolean(text));
	if (texts.length === 0) return 0;
	const candidateText = `${entry.text} ${entry.scoringText ?? ""}`;
	return Math.max(0, ...texts.map((text) => semanticTextSimilarity(text, candidateText)));
}
function answerDomainScore(queryAnalysis, entry) {
	const candidateText = `${entry.text} ${entry.scoringText ?? ""}`;
	const slotTexts = answerPlanSlots(queryAnalysis).map((slot) => slotPlanText(slot)).filter(Boolean);
	const slotSemantic = Math.max(0, ...slotTexts.map((text) => planTextMatchScore(text, candidateText)));
	const goalSemantic = evidenceGoalDomainScore(queryAnalysis, entry);
	const answerCoverage = Math.max(0, ...(entry.slotCoverage ?? []).filter((coverage) => {
		const role = evidenceSlotRequiredRole((queryAnalysis.evidencePlan?.slots ?? []).find((candidate) => candidate.id === coverage.slotId));
		return role === "answer_value" || role === "answer_event" || operationType(queryAnalysis) === "tailor_advice" && (role === "user_resource" || role === "prior_advice");
	}).map((coverage) => coverage.coverageScore * (coverage.missingRequired.length > 0 ? .45 : 1)));
	const bridgeAnswer = bridgeAnswerRoleScore(entry);
	const bridgePositive = bridgePositiveRoleScore(entry, [
		"answer_value",
		"answer_event",
		"user_resource",
		"prior_advice"
	]);
	if (operationType(queryAnalysis) === "tailor_advice") {
		const needFit = tailorAdviceNeedFit(queryAnalysis, entry);
		return clamp01(answerCoverage * .22 + bridgeAnswer * .12 + bridgePositive * .18 + slotSemantic * .12 + needFit * .28 + goalSemantic * .08 + Math.min(.14, [
			answerCoverage >= .28,
			bridgePositive >= .28,
			slotSemantic >= .28,
			needFit >= .28
		].filter(Boolean).length * .035));
	}
	const positiveSourceCount = [
		answerCoverage >= .28,
		bridgeAnswer >= .28,
		bridgePositive >= .28,
		slotSemantic >= .28,
		goalSemantic >= .28
	].filter(Boolean).length;
	return clamp01(answerCoverage * .36 + bridgeAnswer * .22 + bridgePositive * .14 + slotSemantic * .1 + goalSemantic * .18 + Math.min(.16, Math.max(0, positiveSourceCount - 1) * .05));
}
function hasTemporalPlanning(queryAnalysis) {
	return queryAnalysis.queryShape.timeframe === "historical" || queryAnalysis.answerMode === "count_aggregate" || operationType(queryAnalysis) === "aggregate" || (queryAnalysis.evidencePlan?.slots ?? []).some((slot) => evidenceSlotRequiredRole(slot) === "time_constraint" || slotNeedsTemporalField(queryAnalysis, slot));
}
function temporalFitScore(params) {
	if (!hasTemporalPlanning(params.queryAnalysis)) return .62;
	if (params.entry.observedAt || params.contextCandidates.some((entry) => Boolean(entry.observedAt))) return .78;
	if (Math.max(bridgeRoleScore(params.entry, ["time_constraint"]), bridgeShapeScore(params.entry, ["time_constraint"]), ...params.contextCandidates.map((entry) => Math.max(bridgeRoleScore(entry, ["time_constraint"]), bridgeShapeScore(entry, ["time_constraint"])))) >= .32) return .62;
	if (params.slotRole === "time_constraint") return .54;
	return .42;
}
function hardExclusionReasons(params) {
	const text = `${params.entry.text} ${params.entry.scoringText ?? ""}`.trim();
	const sourceRefs = sourceRefsForEntry(params.entry);
	const blockers = [];
	if (!normalizeText(text)) blockers.push("empty-evidence");
	if (HARD_EXCLUSION_PATTERNS.some((pattern) => pattern.test(text))) blockers.push("bootstrap-or-debug-memory");
	if (sourceRefs.length === 0 && params.entry.surface !== "chunk" && params.answer >= .42 && params.entry.source === "projected") blockers.push("untraceable-summary-answer");
	if (sourceRefs.length === 0) blockers.push("untraceable-answer-evidence");
	if (sourceRefs.length === 0 && queryEchoScore(params.queryAnalysis, params.entry.text) >= .9 && params.answer < .5) blockers.push("query-echo-without-history-source");
	if (looksLikeBareMemoryUseInstruction(text)) blockers.push("memory-use-instruction-not-answer");
	if (isAssistantAcknowledgement(params.entry)) blockers.push("assistant-acknowledgement-not-evidence");
	blockers.push(...stateHardExclusions(params.entry, params.now));
	return [...new Set(blockers)];
}
function softPenaltyReasons(params) {
	const penalties = [];
	const operation = operationType(params.queryAnalysis);
	const author = entryAuthorRole(params.entry);
	const filledRoles = new Set([params.entry, ...params.contextCandidates].map((candidate) => inferredSlotRole(params.queryAnalysis, candidate)).filter(Boolean));
	const hasAnswerRole = params.slotRole === "answer_value" || params.slotRole === "answer_event" || params.slotRole === "user_resource" || params.slotRole === "prior_advice" || filledRoles.has("answer_value") || filledRoles.has("answer_event") || operation === "tailor_advice" && (filledRoles.has("user_resource") || filledRoles.has("prior_advice"));
	if (params.remainingMissing.length > 0) penalties.push({
		reason: `missing-context:${params.remainingMissing.join("|")}`,
		weight: Math.min(.34, params.remainingMissing.length * .09)
	});
	if (operation === "return_value" && queryContextPlanTexts(params.queryAnalysis).length > 0 && !(queryAsksSensitiveValue(params.queryAnalysis) && statesUnavailableSensitiveValue(params.queryAnalysis, params.entry)) && Math.max(queryContextSupportScore(params.queryAnalysis, params.entry), 0, ...params.contextCandidates.map((candidate) => queryContextSupportScore(params.queryAnalysis, candidate))) < .24) penalties.push({
		reason: "weak-query-context-binding",
		weight: .16
	});
	if (params.slotRole === "query_context") penalties.push({
		reason: "query-context-only",
		weight: .12
	});
	if (params.slotRole === "time_constraint") penalties.push({
		reason: "time-constraint-only",
		weight: .11
	});
	if (params.entry.surface === "chunk" && (author === "user" || entryIsQueryLikeEvidence(params.queryAnalysis, params.entry)) && queryEchoScore(params.queryAnalysis, params.entry.text) >= .82 && params.answer < .7) penalties.push({
		reason: "query-echo-like",
		weight: .36
	});
	else if (entryIsQueryLikeEvidence(params.queryAnalysis, params.entry)) penalties.push({
		reason: "query-like-answer-candidate",
		weight: .28
	});
	if (hasTemporalPlanning(params.queryAnalysis) && params.temporalFit < .58 && params.contextCandidates.length === 0) penalties.push({
		reason: "weak-temporal-fit",
		weight: .08
	});
	if (params.queryAnalysis.queryShape.timeframe === "historical" && filledRoles.has("time_constraint") && !hasAnswerRole && params.answerDomain < .42) penalties.push({
		reason: "temporal-without-answer-domain",
		weight: .22
	});
	if (params.queryAnalysis.queryShape.timeframe === "historical" && params.slotRole !== "time_constraint" && params.answerDomain < .5) penalties.push({
		reason: "weak-historical-answer-domain",
		weight: .14
	});
	if ((params.queryAnalysis.answerMode === "count_aggregate" || operation === "aggregate") && params.answerDomain < .46) penalties.push({
		reason: "aggregate-without-answer-event",
		weight: .16
	});
	if ((params.queryAnalysis.answerMode === "count_aggregate" || operation === "aggregate") && params.slotRole === "answer_event" && params.answerDomain < .56) penalties.push({
		reason: "weak-aggregate-event-domain",
		weight: .22
	});
	if ((operation === "return_value" || params.queryAnalysis.answerMode === "attribute_lookup") && (params.slotRole === "answer_value" || filledRoles.has("answer_value")) && !(queryAsksSensitiveValue(params.queryAnalysis) && statesUnavailableSensitiveValue(params.queryAnalysis, params.entry)) && params.contextBinding < .46) penalties.push({
		reason: "answer-without-bound-context",
		weight: .18
	});
	if (operation === "return_value" && (params.slotRole === "answer_value" || filledRoles.has("answer_value")) && answerRelationFitScore(params.queryAnalysis, params.entry) < .28 && !hasStrongAnswerPlanningMatch(params.queryAnalysis, params.entry)) penalties.push({
		reason: "weak-answer-relation-fit",
		weight: .28
	});
	if (querySeeksCausalAnswer(params.queryAnalysis) && (params.slotRole === "answer_value" || filledRoles.has("answer_value")) && causalExplanationScore(params.queryAnalysis, params.entry) < .32) penalties.push({
		reason: "weak-causal-explanation",
		weight: .34
	});
	if (querySeeksCausalAnswer(params.queryAnalysis) && (params.slotRole === "answer_value" || filledRoles.has("answer_value")) && validationEvidenceScore(params.queryAnalysis, params.entry) >= .72 && directCausalExplanationScore(params.queryAnalysis, params.entry) < .5) penalties.push({
		reason: "validation-evidence-not-cause",
		weight: .34
	});
	if (querySeeksCausalAnswer(params.queryAnalysis) && (params.slotRole === "answer_value" || filledRoles.has("answer_value")) && statusSummaryEvidenceScore(params.queryAnalysis, params.entry) >= .7 && directCausalExplanationScore(params.queryAnalysis, params.entry) < .5) penalties.push({
		reason: "status-summary-not-cause",
		weight: .3
	});
	if (queryAsksCurrentStatus(params.queryAnalysis) && (params.slotRole === "answer_value" || filledRoles.has("answer_value")) && currentStatusAnswerScore(params.queryAnalysis, params.entry) < .32) penalties.push({
		reason: "weak-current-status-answer",
		weight: .3
	});
	if (queryAsksCurrentStatus(params.queryAnalysis) && (params.slotRole === "answer_value" || filledRoles.has("answer_value")) && staleOnlyStatusEvidence(params.queryAnalysis, params.entry)) penalties.push({
		reason: "stale-status-only",
		weight: .42
	});
	if (queryAsksBooleanDecision(params.queryAnalysis) && (params.slotRole === "answer_value" || filledRoles.has("answer_value")) && booleanDecisionFitScore(params.queryAnalysis, params.entry) < .34) penalties.push({
		reason: "weak-boolean-decision-fit",
		weight: .4
	});
	if (queryAsksSensitiveValue(params.queryAnalysis) && (params.slotRole === "answer_value" || filledRoles.has("answer_value")) && !statesUnavailableSensitiveValue(params.queryAnalysis, params.entry)) penalties.push({
		reason: "sensitive-value-without-availability-evidence",
		weight: .46
	});
	const contrastScore = Math.max(negativeContrastScore(params.queryAnalysis, params.entry), 0, ...params.contextCandidates.map((candidate) => negativeContrastScore(params.queryAnalysis, candidate)));
	const contrastContext = Math.max(queryContextSupportScore(params.queryAnalysis, params.entry), 0, ...params.contextCandidates.map((candidate) => queryContextSupportScore(params.queryAnalysis, candidate)));
	if (operation === "return_value" && contrastScore >= .28 && contrastContext < .52) penalties.push({
		reason: "negative-contrast-without-query-context",
		weight: .42
	});
	if (operation !== "tailor_advice" && params.slotRole === "user_resource" && params.answerDomain < .42) penalties.push({
		reason: "resource-outside-advice-domain",
		weight: .22
	});
	if (operation === "tailor_advice" && params.slotRole === "user_resource") {
		const resourceRelevance = Math.max(tailorAdviceNeedFit(params.queryAnalysis, params.entry), semanticTextSimilarity(params.queryAnalysis.queryText, params.entry.text), semanticTextSimilarity(params.queryAnalysis.focusedQuery ?? "", params.entry.text), bridgePositiveRoleScore(params.entry, ["user_resource"]), numericBreakdown(params.entry, "capabilitySupport"));
		if (resourceRelevance < .24 && !hasResourceAffordanceSignal(params.entry)) penalties.push({
			reason: "weak-user-resource-capability-match",
			weight: .3
		});
		else if (resourceRelevance < .24) penalties.push({
			reason: "weak-user-resource-capability-match",
			weight: .16
		});
	}
	if (sourceRefsForEntry(params.entry).length === 0) penalties.push({
		reason: "missing-source-ref",
		weight: .1
	});
	for (const penalty of stateSoftPenalties(params.entry)) penalties.push(penalty);
	if (isAssistantAcknowledgement(params.entry)) penalties.push({
		reason: "assistant-acknowledgement",
		weight: .36
	});
	if (operation === "tailor_advice" && params.slotRole === "prior_advice" && params.answerDomain < .42) penalties.push({
		reason: "prior-advice-domain-mismatch",
		weight: .2
	});
	if (!hasAnswerRole && params.answer < .42 && params.answerDomain < .34) penalties.push({
		reason: "weak-answer-role",
		weight: .1
	});
	return penalties;
}
function answerScore(params) {
	const semantic = clamp01(Math.max(params.entry.semanticScore ?? 0, params.entry.goalScore));
	const coverage = slotCoverageScore(params.entry);
	const capability = numericBreakdown(params.entry, "capabilitySupport");
	const domain = answerDomainScore(params.queryAnalysis, params.entry);
	const bridgeSupport = bridgeSupportScore(params.entry);
	const bridgeAnswer = bridgeAnswerRoleScore(params.entry);
	const retrieval = candidateRetrievalScore(params.entry);
	const causalFit = querySeeksCausalAnswer(params.queryAnalysis) ? Math.max(directCausalExplanationScore(params.queryAnalysis, params.entry), causalExplanationScore(params.queryAnalysis, params.entry) * .55) : 0;
	if (params.slotRole === "query_context") return clamp01(domain * .22 + semantic * .08 + retrieval * .04);
	if (params.slotRole === "user_resource") {
		const roleFloor = operationType(params.queryAnalysis) === "tailor_advice" ? .18 : .04;
		return clamp01(domain * .34 + bridgeSupport * .2 + semantic * .16 + coverage * .12 + capability * .12 + (hasResourceAffordanceSignal(params.entry) ? .06 : roleFloor));
	}
	if (params.slotRole === "answer_value" || params.slotRole === "answer_event") {
		const base = clamp01(domain * (causalFit > 0 ? .34 : .42) + bridgeAnswer * .18 + semantic * .12 + coverage * .12 + causalFit * .18 + params.entry.priority * .06 + retrieval * .04);
		return queryAsksSensitiveValue(params.queryAnalysis) && statesUnavailableSensitiveValue(params.queryAnalysis, params.entry) ? Math.max(base, .84) : base;
	}
	if (params.slotRole === "time_constraint") return clamp01(domain * .18 + semantic * .06);
	if (params.slotRole === "prior_advice") return clamp01(domain * .34 + bridgeSupport * .2 + semantic * .22 + coverage * .1 + params.entry.priority * .06);
	const evidenceCoverageScore = params.entry.coverage && params.entry.coverage.requiredHits.length > 0 ? Math.max(.5, params.entry.coverage.coverageScore) : 0;
	return clamp01((params.entry.surface === "fact" || params.entry.surface === "event" ? .46 : params.entry.surface === "chunk" ? .34 : .28) * .2 + semantic * .18 + domain * .26 + bridgeAnswer * .14 + causalFit * .12 + coverage * .12 + evidenceCoverageScore * .06 + params.entry.priority * .04);
}
function gradeCandidate(params) {
	const slotRole = inferredSlotRole(params.queryAnalysis, params.entry);
	const retrievalScore = candidateRetrievalScore(params.entry);
	const answer = answerScore({
		...params,
		slotRole
	});
	const contextBinding = contextBindingScore({
		...params,
		slotRole
	});
	const slotScore = slotCoverageScore(params.entry);
	const authority = authorityScore(params.entry);
	const bridgeNegative = bridgeNegativeSignalScore(params.entry);
	const operation = operationType(params.queryAnalysis);
	const remainingMissing = missingRequiredAfterContext(params.entry, params.contextCandidates);
	const filledRoles = new Set([params.entry, ...params.contextCandidates].map((candidate) => inferredSlotRole(params.queryAnalysis, candidate)).filter(Boolean));
	const hasAnswerRole = slotRole === "answer_value" || slotRole === "answer_event" || operation === "tailor_advice" && (slotRole === "user_resource" || slotRole === "prior_advice") || filledRoles.has("answer_value") || filledRoles.has("answer_event") || operation === "tailor_advice" && (filledRoles.has("user_resource") || filledRoles.has("prior_advice"));
	const answerDomain = answerDomainScore(params.queryAnalysis, params.entry);
	const temporalFit = temporalFitScore({
		...params,
		slotRole
	});
	const blockers = hardExclusionReasons({
		queryAnalysis: params.queryAnalysis,
		entry: params.entry,
		answer,
		now: params.now
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
		remainingMissing
	});
	const softPenaltyScore = clamp01(Math.min(.48, softPenaltyEntries.reduce((sum, penalty) => sum + penalty.weight, 0)));
	const eligibilityRole = slotRole === "user_resource" ? "resource" : slotRole === "query_context" || slotRole === "time_constraint" ? "context" : hasAnswerRole || answer >= .42 ? "answer" : "support";
	const eligible = blockers.length === 0;
	const sourceTrace = sourceRefsForEntry(params.entry).length > 0 ? 1 : 0;
	const roleFit = slotRole === "answer_value" || slotRole === "answer_event" ? 1 : operation === "tailor_advice" && (slotRole === "user_resource" || slotRole === "prior_advice") ? .95 : slotRole === "query_context" || slotRole === "time_constraint" ? .32 : answer >= .42 ? .62 : .4;
	const semanticScore = Math.max(params.entry.semanticScore ?? 0, params.entry.goalScore);
	const contextMultiplier = operation === "return_value" || params.queryAnalysis.answerMode === "attribute_lookup" ? .45 + .55 * contextBinding : 1;
	const finalScore = clamp01((operation === "aggregate" || params.queryAnalysis.answerMode === "count_aggregate" ? answerDomain * .45 + retrievalScore * .2 + sourceTrace * .15 + temporalFit * .1 + authority * .1 + roleFit * .08 : operation === "tailor_advice" ? answer * .35 + semanticScore * .25 + sourceTrace * .15 + authority * .1 + retrievalScore * .15 + roleFit * .08 : hasTemporalPlanning(params.queryAnalysis) && params.queryAnalysis.queryShape.timeframe === "historical" ? answerDomain * .45 + temporalFit * .2 + retrievalScore * .15 + sourceTrace * .1 + authority * .1 + roleFit * .08 : (answer * .36 + answerDomain * .22 + retrievalScore * .16 + authority * .1 + sourceTrace * .08 + slotScore * .08 + roleFit * .08) * contextMultiplier) - softPenaltyScore - bridgeNegative * .06 - Math.min(.5, blockers.length * .28));
	return {
		blockers,
		softPenalties: softPenaltyEntries.map((penalty) => penalty.reason),
		eligibility: {
			eligible,
			role: eligibilityRole,
			blockers
		},
		grade: {
			retrievalScore,
			answerScore: answer,
			contextBindingScore: contextBinding,
			temporalFitScore: temporalFit,
			slotCoverageScore: slotScore,
			authorityScore: authority,
			softPenaltyScore,
			finalScore
		}
	};
}
function groupEntries(entries) {
	const groups = /* @__PURE__ */ new Map();
	for (const entry of entries) {
		if (entry.dropReason) continue;
		const sourceRefs = bindingSourceRefsForEntry(entry);
		const key = sourceRefs.length > 0 ? sourceKey(sourceRefs, entry.text) : `text:${stableHash([normalizeText(entry.text).slice(0, 240)])}`;
		const group = groups.get(key) ?? [];
		group.push(entry);
		groups.set(key, group);
	}
	return groups;
}
function contextCandidatesForGroup(params) {
	const bindingAnswerRefs = bindingSourceRefsForEntry(params.answerEntry);
	const operation = operationType(params.queryAnalysis);
	const answerRole = inferredSlotRole(params.queryAnalysis, params.answerEntry);
	const userQuestionBridge = entryAuthorRole(params.answerEntry) === "user" && (entryIsQuestionLike(params.answerEntry) || entryIsQueryLikeEvidence(params.queryAnalysis, params.answerEntry));
	const assistantAnswerBridge = entryAuthorRole(params.answerEntry) === "assistant" && operation !== "aggregate" && params.queryAnalysis.answerMode !== "count_aggregate" && (answerRole === "answer_value" || answerRole === "answer_event" || params.answerEntry.metadata?.sourceExpansion === true);
	const bindAdjacentNonContext = operation !== "aggregate" && params.queryAnalysis.answerMode !== "count_aggregate" && (userQuestionBridge || assistantAnswerBridge);
	const sourceFamilies = new Set(bindingAnswerRefs.map((sourceRef) => sourceFamilyRef(sourceRef)));
	const local = params.entries.filter((entry) => entryKey(entry) !== entryKey(params.answerEntry));
	const related = params.allEntries.filter((entry) => {
		if (entryKey(entry) === entryKey(params.answerEntry) || entry.dropReason) return false;
		const role = inferredSlotRole(params.queryAnalysis, entry);
		const entryRefs = bindingSourceRefsForEntry(entry);
		const entryFamilies = entryRefs.map((sourceRef) => sourceFamilyRef(sourceRef));
		if (sourceFamilies.size > 0 && entryFamilies.length > 0) {
			if (entryFamilies.some((family) => sourceFamilies.has(family))) {
				if (role === "query_context") return sourceRefsAdjacent(bindingAnswerRefs, entryRefs, 2) && queryContextSupportScore(params.queryAnalysis, entry) >= .18;
				return bindAdjacentNonContext && (role !== answerRole || userQuestionBridge && entryAuthorRole(entry) === "assistant" || assistantAnswerBridge && entryAuthorRole(entry) === "user") && sourceRefsAdjacent(bindingAnswerRefs, entryRefs, 2);
			}
		}
		if (bindAdjacentNonContext && (role !== answerRole || userQuestionBridge && entryAuthorRole(entry) === "assistant" || assistantAnswerBridge && entryAuthorRole(entry) === "user") && sourceRefsAdjacent(bindingAnswerRefs, entryRefs, 2)) return true;
		return role === "query_context" && (sourceFamilies.size === 0 || entryFamilies.length === 0) && semanticTextSimilarity(entry.text, params.answerEntry.text) >= .34;
	});
	return [...new Map([...local, ...related].map((entry) => [entryKey(entry), entry])).values()].sort((left, right) => {
		const leftAdjacent = sourceRefsAdjacent(bindingAnswerRefs, bindingSourceRefsForEntry(left), 2) ? 1 : 0;
		const rightAdjacent = sourceRefsAdjacent(bindingAnswerRefs, bindingSourceRefsForEntry(right), 2) ? 1 : 0;
		if (leftAdjacent !== rightAdjacent) return rightAdjacent - leftAdjacent;
		const leftContext = inferredSlotRole(params.queryAnalysis, left) === "query_context" ? 1 : 0;
		const rightContext = inferredSlotRole(params.queryAnalysis, right) === "query_context" ? 1 : 0;
		if (leftContext !== rightContext) return rightContext - leftContext;
		return candidateRetrievalScore(right) - candidateRetrievalScore(left);
	}).slice(0, 3);
}
function packetFromGroup(params) {
	const candidates = params.entries.map((entry) => {
		const contextCandidates = contextCandidatesForGroup({
			queryAnalysis: params.queryAnalysis,
			entries: params.entries,
			allEntries: params.allEntries,
			answerEntry: entry
		});
		return {
			entry,
			contextCandidates,
			graded: gradeCandidate({
				queryAnalysis: params.queryAnalysis,
				entry,
				contextCandidates,
				now: params.now
			})
		};
	});
	const directFactAnswerMode = operationType(params.queryAnalysis) === "return_value" || params.queryAnalysis.answerMode === "single_fact" || params.queryAnalysis.answerMode === "attribute_lookup";
	const canonicalAnswerRank = (candidate) => {
		if (!directFactAnswerMode) return 0;
		if (candidate.entry.surface === "fact" && sourceRefsForEntry(candidate.entry).length > 0 && missingRequiredAfterContext(candidate.entry, candidate.contextCandidates).length === 0 && candidate.graded.blockers.length === 0) return 6;
		if (!candidate.graded.eligibility.eligible) return 0;
		const role = inferredSlotRole(params.queryAnalysis, candidate.entry);
		if (!(role === "answer_value" || role === "answer_event" || candidate.graded.eligibility.role === "answer" || candidate.graded.grade.answerScore >= .42) || candidate.graded.grade.slotCoverageScore < .45) return 0;
		if (candidate.entry.surface === "fact") return 3;
		if (candidate.entry.surface === "event") return 2;
		if (candidate.entry.surface === "chunk" || candidate.entry.surface === "snippet") return 1;
		return 0;
	};
	const canonicalAdjustedScore = (candidate) => candidate.graded.grade.finalScore + canonicalAnswerRank(candidate) * .12;
	const rankedSelected = candidates.sort((left, right) => {
		if (left.graded.eligibility.eligible !== right.graded.eligibility.eligible) return left.graded.eligibility.eligible ? -1 : 1;
		const leftAdjusted = canonicalAdjustedScore(left);
		const rightAdjusted = canonicalAdjustedScore(right);
		if (leftAdjusted !== rightAdjusted) return rightAdjusted - leftAdjusted;
		return right.graded.grade.finalScore - left.graded.grade.finalScore;
	})[0];
	if (!rankedSelected) return null;
	const rankedSourceRefs = sourceRefsForEntry(rankedSelected.entry);
	const canonicalSubstituteEntry = directFactAnswerMode && rankedSelected.entry.surface !== "fact" ? params.allEntries.find((entry) => entry.surface === "fact" && !entry.dropReason && sourceRefsForEntry(entry).some((sourceRef) => rankedSourceRefs.includes(sourceRef))) : void 0;
	const selected = canonicalSubstituteEntry && entryKey(canonicalSubstituteEntry) !== entryKey(rankedSelected.entry) ? {
		entry: canonicalSubstituteEntry,
		contextCandidates: [...new Map([rankedSelected.entry, ...rankedSelected.contextCandidates].filter((entry) => entryKey(entry) !== entryKey(canonicalSubstituteEntry)).map((entry) => [entryKey(entry), entry])).values()],
		graded: gradeCandidate({
			queryAnalysis: params.queryAnalysis,
			entry: canonicalSubstituteEntry,
			contextCandidates: [...new Map([rankedSelected.entry, ...rankedSelected.contextCandidates].filter((entry) => entryKey(entry) !== entryKey(canonicalSubstituteEntry)).map((entry) => [entryKey(entry), entry])).values()],
			now: params.now
		})
	} : rankedSelected;
	const directFactContextSatisfied = directFactAnswerMode && selected.entry.surface === "fact" && selected.graded.blockers.length === 0 && missingRequiredAfterContext(selected.entry, selected.contextCandidates).length === 0;
	const selectedContextCandidates = directFactContextSatisfied ? [] : selected.contextCandidates;
	const coverageContextCandidates = directFactContextSatisfied ? selected.contextCandidates : selectedContextCandidates;
	const unitGroups = classifyPacketUnits({
		queryAnalysis: params.queryAnalysis,
		selectedEntry: selected.entry,
		contextCandidates: selectedContextCandidates
	});
	const { displayLines, hiddenExactDuplicates } = packetDisplayLines({
		queryAnalysis: params.queryAnalysis,
		...unitGroups
	});
	const op = operationType(params.queryAnalysis);
	const returnValueMode = op === "return_value" || params.queryAnalysis.answerMode === "attribute_lookup";
	const hasAnswerDisplayLine = displayLines.some((line) => line.startsWith("[answer]") || line.startsWith("[resource]") || (op === "aggregate" || params.queryAnalysis.answerMode === "count_aggregate") && line.startsWith("[event]"));
	const displayOnlyContext = returnValueMode && (!hasAnswerDisplayLine || unitGroups.answerUnits.length === 0);
	const packetDirectCauseScore = querySeeksCausalAnswer(params.queryAnalysis) ? Math.max(evidenceShapeFitScore(params.queryAnalysis, selected.entry, ["causal_explanation"]), evidenceShapeFitScoreForTexts(params.queryAnalysis, [...displayLines, ...unitGroups.answerUnits.map((unit) => `${unit.displayText} ${unit.rawText}`)], ["causal_explanation"])) : 0;
	const packetSuppliesDirectCause = packetDirectCauseScore >= .72 && directCausalExplanationScore(params.queryAnalysis, selected.entry) < .5;
	const directFactHiddenSupportPenalties = directFactContextSatisfied ? selected.graded.softPenalties.filter((reason) => reason !== "answer-without-bound-context" && reason !== "weak-query-context-binding" && !reason.startsWith("missing-context:")) : selected.graded.softPenalties;
	const baseSoftPenalties = packetSuppliesDirectCause ? directFactHiddenSupportPenalties.filter((reason) => reason !== "weak-causal-explanation" && reason !== "query-like-answer-candidate") : directFactHiddenSupportPenalties;
	const adjustedSoftPenalties = displayOnlyContext ? [...selected.graded.softPenalties, "no-answer-display-line"] : baseSoftPenalties;
	const packetCausalGrade = packetSuppliesDirectCause ? {
		...selected.graded.grade,
		answerScore: Math.max(selected.graded.grade.answerScore, packetDirectCauseScore),
		softPenaltyScore: Math.max(0, (selected.graded.grade.softPenaltyScore ?? 0) - .42),
		finalScore: Math.max(selected.graded.grade.finalScore, clamp01(.46 + (selected.graded.grade.contextBindingScore ?? 0) * .1 + selected.graded.grade.authorityScore * .05 + selected.graded.grade.retrievalScore * .06))
	} : selected.graded.grade;
	const contextSupportedFactGrade = directFactContextSatisfied ? {
		...packetCausalGrade,
		answerScore: Math.max(packetCausalGrade.answerScore, .46),
		contextBindingScore: Math.max(packetCausalGrade.contextBindingScore, .72),
		slotCoverageScore: Math.max(packetCausalGrade.slotCoverageScore, .58),
		softPenaltyScore: Math.max(0, (packetCausalGrade.softPenaltyScore ?? 0) - .24),
		finalScore: Math.max(packetCausalGrade.finalScore, .72)
	} : packetCausalGrade;
	const adjustedGrade = displayOnlyContext ? {
		...contextSupportedFactGrade,
		softPenaltyScore: clamp01((contextSupportedFactGrade.softPenaltyScore ?? 0) + .42),
		finalScore: clamp01(contextSupportedFactGrade.finalScore - .42)
	} : contextSupportedFactGrade;
	const adjustedEligibility = displayOnlyContext ? {
		...selected.graded.eligibility,
		role: "support"
	} : selected.graded.eligibility;
	const allUnits = [
		...unitGroups.answerUnits,
		...unitGroups.contextUnits,
		...unitGroups.supportUnits
	];
	const allSourceRefs = [...new Set(allUnits.flatMap((unit) => refsForUnit(unit)))];
	const sourceRefs = [...new Set([
		...sourceRefsForEntry(selected.entry),
		...coverageContextCandidates.flatMap((entry) => sourceRefsForEntry(entry)),
		...allSourceRefs
	])];
	const supportSourceRefs = [...new Set([
		...coverageContextCandidates.flatMap((entry) => sourceRefsForEntry(entry)),
		...allUnits.flatMap((unit) => unit.supportRefs ?? []),
		...unitGroups.contextUnits.flatMap((unit) => refsForUnit(unit)),
		...unitGroups.supportUnits.flatMap((unit) => refsForUnit(unit))
	])];
	const slotIds = [...new Set([selected.entry, ...coverageContextCandidates].flatMap((entry) => matchedSlotIds(entry)))];
	const slotId = slotIds[0] ?? "unplanned";
	const slotById = new Map((params.queryAnalysis.evidencePlan?.slots ?? []).map((slot) => [slot.id, slot]));
	const missing = [...new Set([...missingRequiredAfterContext(selected.entry, coverageContextCandidates), ...slotIds.map((slotId) => slotById.get(slotId)).filter((slot) => Boolean(slot)).filter((slot) => slotNeedsTemporalField(params.queryAnalysis, slot)).filter(() => !selected.entry.observedAt && !coverageContextCandidates.some((candidate) => candidate.observedAt)).map(() => "observedAt")].filter(Boolean))];
	const role = adjustedEligibility.role === "answer" || adjustedEligibility.role === "resource" || !displayOnlyContext && adjustedGrade.answerScore >= .28 ? "partial" : "support";
	return {
		packetId: `packet:${stableHash([
			selected.entry.id,
			slotIds.join("|"),
			...sourceRefs,
			normalizeText(selected.entry.text).slice(0, 120)
		])}`,
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
		layers: [...new Set([selected.entry.surface])],
		primaryText: displayLines[0] ?? truncateText(selected.entry.text, 560),
		supportingTexts: displayLines.length > 1 ? displayLines.slice(1) : selectedContextCandidates.map((entry) => truncateText(entry.text, 260)).filter((text) => semanticTextSimilarity(text, selected.entry.text) < .82),
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
			supportUnitCount: unitGroups.supportUnits.length
		},
		displayLines,
		hiddenExactDuplicates,
		observedAt: selected.entry.observedAt ?? selectedContextCandidates.find((entry) => entry.observedAt)?.observedAt,
		resolvedDate: selected.entry.observedAt?.slice(0, 10) ?? selectedContextCandidates.find((entry) => entry.observedAt)?.observedAt?.slice(0, 10),
		dedupeKey: sourceKey(sourceRefs, selected.entry.text),
		authorRoles: [...new Set(allUnits.map((unit) => unit.authorRole).filter(Boolean))].filter((role) => Boolean(role) && role !== "unknown"),
		coverage: {
			filled: adjustedEligibility.eligible && missing.length === 0,
			missing,
			confidence: adjustedGrade.finalScore
		},
		eligibility: adjustedEligibility,
		grade: adjustedGrade,
		selectionReason: selected.graded.blockers.length > 0 ? `excluded:${selected.graded.blockers.join(",")}` : `ranked:${adjustedEligibility.role}:${adjustedGrade.finalScore.toFixed(3)}${adjustedSoftPenalties.length > 0 ? ` penalties=${adjustedSoftPenalties.join("|")}` : ""}`,
		blockedBy: selected.graded.blockers,
		softPenalties: adjustedSoftPenalties,
		hardExclusions: selected.graded.blockers,
		dropReason: selected.graded.blockers.length > 0 ? selected.graded.blockers.join(",") : void 0
	};
}
function highLevelSupportPackets(params) {
	const packets = [];
	for (const candidate of params.candidateGenerationResult?.layerCandidates ?? []) {
		const sourceRefs = [...new Set([candidate.sourceRef, ...candidate.sourceRefs ?? []].filter((value) => Boolean(value)))];
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
				confidence: clamp01(candidate.score)
			},
			eligibility: {
				eligible: false,
				role: "support",
				blockers: sourceRefs.length === 0 ? ["high-level-memory-missing-source-ref"] : ["source-evidence-not-selected"]
			},
			grade: {
				retrievalScore: clamp01(candidate.score),
				answerScore: .1,
				contextBindingScore: .1,
				temporalFitScore: .42,
				slotCoverageScore: clamp01(candidate.score),
				authorityScore: .42,
				softPenaltyScore: .18,
				finalScore: clamp01(candidate.score * .28)
			},
			selectionReason: "high-level-support-only",
			blockedBy: sourceRefs.length === 0 ? ["high-level-memory-missing-source-ref"] : ["source-evidence-not-selected"],
			dropReason: sourceRefs.length === 0 ? "high-level-memory-missing-source-ref" : "source-evidence-not-selected",
			hardExclusions: sourceRefs.length === 0 ? ["high-level-memory-missing-source-ref"] : ["source-evidence-not-selected"]
		});
	}
	return packets;
}
function packetSortValue(packet) {
	return packet.grade?.finalScore ?? packet.coverage.confidence;
}
function packetInjectionBudget(queryAnalysis) {
	if (queryAnalysis.answerMode === "count_aggregate" || operationType(queryAnalysis) === "aggregate") return 5;
	if (queryAnalysis.answerMode === "multi_evidence" || operationType(queryAnalysis) === "tailor_advice" || operationType(queryAnalysis) === "derive" || operationType(queryAnalysis) === "compare") return 4;
	return 3;
}
function packetScoreFloor(queryAnalysis) {
	if (operationType(queryAnalysis) === "tailor_advice") return .28;
	if (queryAnalysis.answerMode === "count_aggregate" || operationType(queryAnalysis) === "aggregate") return .24;
	return .28;
}
function packetCurveGap(queryAnalysis) {
	if (queryAnalysis.answerMode === "count_aggregate" || operationType(queryAnalysis) === "aggregate") return .34;
	if (queryAnalysis.answerMode === "multi_evidence" || operationType(queryAnalysis) === "tailor_advice" || operationType(queryAnalysis) === "derive" || operationType(queryAnalysis) === "compare") return .16;
	return .42;
}
function packetDistinctKey(packet) {
	return packet.sourceRefs[0] ?? packet.allSourceRefs?.[0] ?? packet.answerCandidate?.id ?? packet.packetId;
}
function packetAnswerDisplayKey(packet) {
	return exactDisplayKey((packet.displayLines?.[0] ?? packet.primaryText).replace(/\s+\|\s+\[context\].*$/u, ""));
}
function packetHasSoftPenalty(packet, reason) {
	return (packet.softPenalties ?? []).some((penalty) => penalty === reason);
}
function packetHasSoftPenaltyPrefix(packet, prefix) {
	return (packet.softPenalties ?? []).some((penalty) => penalty.startsWith(prefix));
}
function packetCoverageSatisfied(packet) {
	return packet.coverage.filled && packet.coverage.missing.length === 0;
}
function packetHasBlockingPromptPenalty(queryAnalysis, packet) {
	if (packetHasSoftPenaltyPrefix(packet, "missing-context:")) return true;
	if (packetHasSoftPenalty(packet, "answer-without-bound-context")) return true;
	return operationType(queryAnalysis) === "return_value" && packetHasSoftPenalty(packet, "weak-query-context-binding");
}
function packetPromptInjectionFloor(queryAnalysis, packet, floor) {
	if (operationType(queryAnalysis) === "tailor_advice") return Math.max(.46, floor + .12);
	if (queryAnalysis.answerMode === "count_aggregate" || operationType(queryAnalysis) === "aggregate") return Math.max(.5, floor + .16);
	return packetCoverageSatisfied(packet) ? Math.max(.52, floor + .2) : Math.max(.62, floor + .12);
}
function packetEligibleForPromptInjection(queryAnalysis, packet, floor) {
	if (packetHasBlockingPromptPenalty(queryAnalysis, packet)) return false;
	if (!packetHasAnswerDisplayForQuery(queryAnalysis, packet) && operationType(queryAnalysis) !== "derive" && operationType(queryAnalysis) !== "compare") return false;
	return (packet.grade?.finalScore ?? 0) >= packetPromptInjectionFloor(queryAnalysis, packet, floor) && (packet.grade?.slotCoverageScore ?? 0) >= .45;
}
function packetHasAnswerDisplayForQuery(queryAnalysis, packet) {
	const aggregateMode = queryAnalysis.answerMode === "count_aggregate" || operationType(queryAnalysis) === "aggregate";
	return (packet.displayLines ?? []).some((line) => line.startsWith("[answer]") || line.startsWith("[resource]") || aggregateMode && line.startsWith("[event]"));
}
function selectInjectedPackets(queryAnalysis, packets) {
	const limit = packetInjectionBudget(queryAnalysis);
	const floor = packetScoreFloor(queryAnalysis);
	const selected = /* @__PURE__ */ new Set();
	const selectedDistinct = /* @__PURE__ */ new Set();
	const selectedAnswerDisplays = /* @__PURE__ */ new Set();
	const ranked = packets.filter((packet) => !packet.dropReason && packet.eligibility?.eligible !== false).filter((packet) => packetEligibleForPromptInjection(queryAnalysis, packet, floor)).sort((left, right) => packetSortValue(right) - packetSortValue(left));
	const rankedForSelection = ranked.some((packet) => !packetHasSoftPenalty(packet, "negative-contrast-without-query-context") && ((packet.grade?.answerScore ?? 0) > .08 || (packet.displayLines ?? []).some((line) => line.startsWith("[answer]")))) ? ranked.filter((packet) => !packetHasSoftPenalty(packet, "negative-contrast-without-query-context")) : ranked;
	const scoredRankedForSelection = rankedForSelection.some((packet) => packetSortValue(packet) > 0) ? rankedForSelection.filter((packet) => packetSortValue(packet) > 0) : rankedForSelection;
	const aggregateMode = queryAnalysis.answerMode === "count_aggregate" || operationType(queryAnalysis) === "aggregate";
	const selectablePackets = scoredRankedForSelection.some((packet) => packetHasAnswerDisplayForQuery(queryAnalysis, packet)) ? scoredRankedForSelection.filter((packet) => packetHasAnswerDisplayForQuery(queryAnalysis, packet)) : scoredRankedForSelection;
	const topScore = selectablePackets.length > 0 ? packetSortValue(selectablePackets[0]) : 0;
	const effectiveAllowedGap = packetCurveGap(queryAnalysis);
	const directCanonicalAnswer = operationType(queryAnalysis) === "return_value" || queryAnalysis.answerMode === "single_fact" || queryAnalysis.answerMode === "attribute_lookup" ? selectablePackets.find((packet) => packet.answerCandidate?.surface === "fact" && packetCoverageSatisfied(packet) && packetHasAnswerDisplayForQuery(queryAnalysis, packet)) : void 0;
	if (directCanonicalAnswer) return new Set([directCanonicalAnswer.packetId]);
	if (operationType(queryAnalysis) === "derive" || operationType(queryAnalysis) === "compare") {
		const bySlot = /* @__PURE__ */ new Map();
		for (const packet of selectablePackets) {
			const packetSlotIds = packet.slotIds && packet.slotIds.length > 0 ? packet.slotIds : [packet.slotId];
			for (const slotId of packetSlotIds) {
				const slotPackets = bySlot.get(slotId) ?? [];
				slotPackets.push(packet);
				bySlot.set(slotId, slotPackets);
			}
		}
		for (const slot of queryAnalysis.evidencePlan?.slots ?? []) {
			if (selected.size >= limit) break;
			const packet = bySlot.get(slot.id)?.find((candidate) => (candidate.grade?.finalScore ?? 0) >= floor);
			if (packet) {
				const displayKey = packetAnswerDisplayKey(packet);
				if (!aggregateMode && displayKey && selectedAnswerDisplays.has(displayKey)) continue;
				selected.add(packet.packetId);
				if (!aggregateMode && displayKey) selectedAnswerDisplays.add(displayKey);
			}
		}
	}
	for (const packet of selectablePackets) {
		if (selected.size >= limit) break;
		const score = packet.grade?.finalScore ?? 0;
		if (score < floor) continue;
		if (!aggregateMode && selected.size > 0 && topScore - score > effectiveAllowedGap) continue;
		const distinctKey = packetDistinctKey(packet);
		if (aggregateMode && selectedDistinct.has(distinctKey)) continue;
		if (!aggregateMode && packetHasSoftPenaltyPrefix(packet, "missing-context:") && [...selected].filter((packetId) => {
			const selectedPacket = packets.find((candidate) => candidate.packetId === packetId);
			return selectedPacket && !packetHasSoftPenaltyPrefix(selectedPacket, "missing-context:");
		}).length >= 2) continue;
		const displayKey = packetAnswerDisplayKey(packet);
		if (!aggregateMode && displayKey && selectedAnswerDisplays.has(displayKey)) continue;
		selected.add(packet.packetId);
		selectedDistinct.add(distinctKey);
		if (!aggregateMode && displayKey) selectedAnswerDisplays.add(displayKey);
	}
	return selected;
}
function markPromptEvidenceFromPackets(params) {
	const packetByEntryKey = /* @__PURE__ */ new Map();
	for (const packet of params.packets) if (packet.answerCandidate) packetByEntryKey.set(entryKey(packet.answerCandidate), packet);
	const emittedInjectedPackets = /* @__PURE__ */ new Set();
	return params.promptEvidence.map((entry) => {
		const packet = packetByEntryKey.get(entryKey(entry));
		if (!packet) return entry.role === "protected" ? {
			...entry,
			role: "support"
		} : entry;
		const injectedEntry = params.injectedPacketIds.has(packet.packetId) && !emittedInjectedPackets.has(packet.packetId);
		if (injectedEntry) emittedInjectedPackets.add(packet.packetId);
		return {
			...entry,
			packetId: packet.packetId,
			text: injectedEntry && packet.displayLines && packet.displayLines.length > 0 ? packet.displayLines.join("\n") : entry.text,
			mergedSourceRefs: injectedEntry && (packet.allSourceRefs?.length ?? 0) > 0 ? packet.allSourceRefs : entry.mergedSourceRefs,
			role: injectedEntry ? "protected" : "support",
			injected: injectedEntry,
			eligibility: packet.eligibility,
			grade: packet.grade,
			blockedBy: packet.blockedBy,
			softPenalties: packet.softPenalties,
			hardExclusions: packet.hardExclusions,
			injectionScore: packet.grade?.finalScore ?? entry.injectionScore,
			scoreBreakdown: {
				...entry.scoreBreakdown ?? {},
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
				packetBridgeNegativeSignalScore: bridgeNegativeSignalScore(entry)
			},
			selectionReason: injectedEntry ? `packet:${packet.selectionReason ?? "selected"}` : packet.selectionReason ?? entry.selectionReason,
			protectionReason: injectedEntry ? `packet:${packet.selectionReason ?? "selected"}` : packet.dropReason ?? entry.protectionReason
		};
	}).sort((left, right) => {
		const leftProtected = left.role === "protected" ? 1 : 0;
		const rightProtected = right.role === "protected" ? 1 : 0;
		if (leftProtected !== rightProtected) return rightProtected - leftProtected;
		return (right.injectionScore ?? 0) - (left.injectionScore ?? 0);
	});
}
function layerCountsForSlot(slot, candidateGenerationResult) {
	const counts = {};
	for (const layer of slotLayers(slot)) {
		const count = candidateGenerationResult?.slotLayerStats.filter((stat) => stat.slotId === slot.id && stat.layer === layer).reduce((sum, stat) => sum + stat.rawCount, 0) ?? 0;
		if (count > 0) counts[layer] = count;
	}
	return counts;
}
function auditForSlot(params) {
	const slotPackets = params.packets.filter((packet) => packet.slotId === params.slot.id || (packet.slotIds ?? []).includes(params.slot.id));
	const missing = slotPackets.filter((packet) => params.injectedPacketIds.has(packet.packetId)).length >= Math.max(1, params.slot.minEvidence) ? [] : [`minEvidence:${Math.max(1, params.slot.minEvidence)}`];
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
			dropReason: packet.dropReason ?? null
		})),
		missing
	};
}
function allPacketUnits(packets) {
	const units = packets.flatMap((packet) => [
		...packet.answerUnits ?? [],
		...packet.contextUnits ?? [],
		...packet.supportUnits ?? []
	]);
	return [...new Map(units.map((unit) => [unit.unitId, unit])).values()];
}
function evidencePacketAudit(params) {
	const rankedPackets = [...params.packets].filter((packet) => !packet.dropReason).sort((left, right) => (right.grade?.finalScore ?? 0) - (left.grade?.finalScore ?? 0));
	const units = allPacketUnits(params.packets);
	return {
		operation: params.operation,
		slots: params.slots.map((slot) => auditForSlot({
			slot,
			packets: params.packets,
			injectedPacketIds: params.injectedPacketIds,
			candidateGenerationResult: params.candidateGenerationResult
		})),
		candidatePool: params.promptEvidence.map((entry) => ({
			id: entry.id,
			surface: entry.surface,
			sourceRef: entry.sourceRef,
			mergedSourceRefs: entry.mergedSourceRefs,
			normalizedSourceRefs: normalizeSourceRefs([entry.sourceRef, ...entry.mergedSourceRefs ?? []]),
			role: entry.role,
			injected: entry.injected,
			packetId: entry.packetId,
			injectionScore: entry.injectionScore,
			slotEvidenceRole: entry.slotEvidenceRole,
			text: truncateText(entry.text, 480)
		})),
		evidenceUnits: units,
		sourceExpansion: units.filter((unit) => (unit.supportRefs?.length ?? 0) > 0 || (unit.derivedFromRefs?.length ?? 0) > 0 || (unit.neighborRefs?.length ?? 0) > 0).map((unit) => ({
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
			roles: unit.roles
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
			selectionReason: packet.selectionReason
		})),
		injectedPackets: [...params.injectedPacketIds],
		renderedPromptLines: params.packets.filter((packet) => params.injectedPacketIds.has(packet.packetId)).flatMap((packet) => {
			const lines = (packet.displayLines ?? [packet.primaryText]).filter(Boolean);
			const evidenceUnitIds = [
				...packet.answerUnits ?? [],
				...packet.contextUnits ?? [],
				...packet.supportUnits ?? []
			].map((unit) => unit.unitId);
			return lines.map((line, index) => ({
				lineId: `prompt_line:${packet.packetId}:${index + 1}`,
				packetId: packet.packetId,
				role: promptLineRole(line),
				line,
				sourceRefs: packet.allSourceRefs ?? packet.sourceRefs,
				normalizedSourceRefs: normalizeSourceRefs(packet.allSourceRefs ?? packet.sourceRefs),
				evidenceUnitIds
			}));
		}),
		hardExclusions: params.packets.filter((packet) => (packet.hardExclusions?.length ?? 0) > 0).map((packet) => ({
			packetId: packet.packetId,
			reasons: packet.hardExclusions ?? []
		})),
		softPenalties: params.packets.filter((packet) => (packet.softPenalties?.length ?? 0) > 0).map((packet) => ({
			packetId: packet.packetId,
			reasons: packet.softPenalties ?? []
		})),
		scoreCurve: rankedPackets.slice(0, 16).map((packet, index) => ({
			rank: index + 1,
			packetId: packet.packetId,
			injected: packet.injected ?? params.injectedPacketIds.has(packet.packetId),
			finalScore: packet.grade?.finalScore,
			score: packet.score
		})),
		hiddenExactDuplicates: params.packets.flatMap((packet) => packet.hiddenExactDuplicates ?? [])
	};
}
function assembleEvidencePackets(input) {
	const plan = input.queryAnalysis.evidencePlan;
	const now = input.now ?? (/* @__PURE__ */ new Date()).toISOString();
	const activeEntries = input.promptEvidence.filter((entry) => !entry.dropReason);
	const deduped = [...[...groupEntries(activeEntries).values()].map((entries) => packetFromGroup({
		queryAnalysis: input.queryAnalysis,
		entries,
		allEntries: activeEntries,
		now
	})).filter((packet) => Boolean(packet)), ...highLevelSupportPackets({
		queryAnalysis: input.queryAnalysis,
		candidateGenerationResult: input.candidateGenerationResult
	})];
	const injectedPacketIds = selectInjectedPackets(input.queryAnalysis, deduped);
	const packets = deduped.map((packet) => {
		const injected = injectedPacketIds.has(packet.packetId);
		return {
			...packet,
			role: injected ? "answer" : packet.role,
			protected: injected,
			injected,
			protectionReason: injected ? packet.selectionReason ?? "packet-final-score" : packet.protectionReason,
			dropReason: injected ? void 0 : packet.dropReason,
			coverage: packet.coverage
		};
	});
	const markedPromptEvidence = markPromptEvidenceFromPackets({
		promptEvidence: input.promptEvidence,
		packets,
		injectedPacketIds
	});
	return {
		packets,
		promptEvidence: markedPromptEvidence,
		audit: plan ? evidencePacketAudit({
			operation: plan.operation,
			slots: plan.slots,
			packets,
			promptEvidence: markedPromptEvidence,
			injectedPacketIds,
			candidateGenerationResult: input.candidateGenerationResult
		}) : void 0
	};
}
//#endregion
export { assembleEvidencePackets };
