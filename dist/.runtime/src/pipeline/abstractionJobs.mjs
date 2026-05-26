import { clamp01, normalizeText, nowIso, objectRecord, stableHash, truncateText } from "../support.mjs";
import { applyAbstractionRefinement, eligibleForLlmRefinement } from "./abstractionRefinement.mjs";
import { canonicalStateKey, tokenizeSearchTerms } from "./semantic/heuristics.mjs";
import { buildEntityMention, resolveEntityMention } from "./entityResolver.mjs";
import { buildGraphPathCandidates } from "./graphPathEngine.mjs";
import { snapshotMemoryLlmBudgetAudit } from "./llmBudgetAudit.mjs";
import { buildMaintenanceContractMetadata, summarizeMaintenanceContractDiagnostics } from "./maintenanceContract.mjs";
import "./semantics.mjs";
import { describeStateValue } from "./memoryObjectsHelpers.mjs";
import { contentStructuralComplexity } from "./sourceWeighting.mjs";
import { deriveWorkflowPatternSummaries } from "./strategyHypotheses.mjs";
//#region src/pipeline/abstractionJobs.ts
const ABSTRACTION_CANDIDATE_CAP = 12;
const ABSTRACTION_CANDIDATE_BUDGETS = {
	derived_state: 4,
	workflow_pattern: 4,
	graph_hypothesis: 3,
	concept_candidate: 1,
	outcome_hypothesis: 0
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
	"workflow.blocker"
]);
const CONCEPT_GENERIC_NAMES = new Set([
	"assistant",
	"system",
	"user"
]);
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
	"总结"
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
	"reads"
]);
function relationalFactGraphRelation(fact) {
	const explicit = stringValue(objectRecord(fact.objectValueJson?.graph)?.relationType);
	if (explicit && (CONCEPT_RELATIONAL_FACT_PREDICATES.has(explicit) || explicit === "supported_by")) return explicit;
	if (CONCEPT_RELATIONAL_FACT_PREDICATES.has(fact.predicate)) return fact.predicate;
	if (fact.predicate.startsWith("uses_")) return "uses";
	return null;
}
function olderThanDays(days, now = (/* @__PURE__ */ new Date()).toISOString()) {
	const date = new Date(now);
	if (!Number.isFinite(date.getTime())) {
		const fallback = /* @__PURE__ */ new Date();
		fallback.setUTCDate(fallback.getUTCDate() - days);
		return fallback.toISOString();
	}
	date.setUTCDate(date.getUTCDate() - days);
	return date.toISOString();
}
function compareAbstractionCandidates(left, right) {
	if (right.confidence !== left.confidence) return right.confidence - left.confidence;
	if (right.usefulnessScore !== left.usefulnessScore) return right.usefulnessScore - left.usefulnessScore;
	return right.stabilityScore - left.stabilityScore;
}
function selectAbstractionCandidatesByType(candidatesByType) {
	const selectedIds = /* @__PURE__ */ new Set();
	const selected = [];
	const stats = {
		cap: ABSTRACTION_CANDIDATE_CAP,
		byType: {}
	};
	for (const [type, budget] of Object.entries(ABSTRACTION_CANDIDATE_BUDGETS)) {
		const ranked = [...candidatesByType[type] ?? []].sort(compareAbstractionCandidates);
		const picked = ranked.slice(0, Math.max(0, budget));
		for (const candidate of picked) {
			selectedIds.add(candidate.candidateId);
			selected.push(candidate);
		}
		stats.byType[type] = {
			budget,
			available: ranked.length,
			selected: picked.length,
			deferred: Math.max(0, ranked.length - picked.length)
		};
	}
	const leftovers = Object.values(candidatesByType).flat().filter((candidate) => Boolean(candidate)).filter((candidate) => !selectedIds.has(candidate.candidateId)).sort(compareAbstractionCandidates);
	for (const candidate of leftovers) {
		if (selected.length >= ABSTRACTION_CANDIDATE_CAP) break;
		selectedIds.add(candidate.candidateId);
		selected.push(candidate);
		const typeStats = stats.byType[candidate.abstractionType] ?? (stats.byType[candidate.abstractionType] = {
			budget: ABSTRACTION_CANDIDATE_BUDGETS[candidate.abstractionType] ?? 0,
			available: 0,
			selected: 0,
			deferred: 0
		});
		typeStats.selected += 1;
		typeStats.deferred = Math.max(0, typeStats.available - typeStats.selected);
	}
	return {
		selected: selected.sort(compareAbstractionCandidates),
		stats
	};
}
function candidateStageCounts(store, agentId) {
	return {
		active: store.abstractionRepo.countByAgent({
			agentId,
			stages: ["active"]
		}),
		candidate: store.abstractionRepo.countByAgent({
			agentId,
			stages: ["candidate"]
		}),
		decaying: store.abstractionRepo.countByAgent({
			agentId,
			stages: ["decaying"]
		}),
		probationary: store.abstractionRepo.countByAgent({
			agentId,
			stages: ["probationary"]
		}),
		quarantined: store.abstractionRepo.countByAgent({
			agentId,
			stages: ["quarantined"]
		}),
		superseded: store.abstractionRepo.countByAgent({
			agentId,
			stages: ["superseded"]
		})
	};
}
function average(values) {
	if (values.length === 0) return 0;
	return values.reduce((sum, value) => sum + value, 0) / values.length;
}
function daysBetween(earlierIso, laterIso) {
	const deltaMs = Date.parse(laterIso) - Date.parse(earlierIso);
	return Number.isFinite(deltaMs) && deltaMs > 0 ? deltaMs / 864e5 : 0;
}
function buildBeliefMap(beliefs) {
	const entries = beliefs.filter((belief) => belief.contentRef).map((belief) => [`${belief.memoryKind}:${belief.contentRef}`, belief]);
	return new Map(entries);
}
function stringValue(value) {
	return typeof value === "string" && value.trim() ? value.trim() : void 0;
}
function eventStructuredHints(event) {
	const structured = objectRecord(objectRecord(event.metadataJson)?.memxStructuredHints);
	if (!structured) return null;
	return structured;
}
function eventWorkflowHints(event) {
	const hints = eventStructuredHints(event);
	if (!hints) return [];
	if (Array.isArray(hints.workflows) && hints.workflows.length > 0) return hints.workflows;
	if (hints.workflow) return [hints.workflow];
	return [];
}
function eventRelationHints(event) {
	const hints = eventStructuredHints(event);
	if (!hints) return [];
	if (Array.isArray(hints.relations) && hints.relations.length > 0) return hints.relations;
	return hints.relation ? [hints.relation] : [];
}
function graphHypothesisNodeId(name, type) {
	return `graph_hypothesis_node:${stableHash([normalizeText(name), type])}`;
}
function graphRelationPhrase(relationType) {
	switch (relationType) {
		case "depends_on": return "depends on";
		case "blocks": return "blocks";
		case "caused_by": return "is caused by";
		case "uses": return "uses";
		case "part_of": return "is part of";
		case "owner_of": return "owns";
		case "supersedes": return "supersedes";
		case "contradicts": return "contradicts";
		case "resolved_by": return "is resolved by";
		case "related_to": return "relates to";
		case "reads": return "reads";
		case "supported_by": return "is supported by";
		case "derived_from": return "is derived from";
		case "updates": return "updates";
		case "targets": return "targets";
	}
}
function semanticValueKey(key, valueJson) {
	return normalizeText(`${canonicalStateKey(key)}:${describeStateValue(key, valueJson)}`);
}
function derivedStateSummary(key, valueText) {
	switch (canonicalStateKey(key)) {
		case "project.active_project": return `The active project appears to be ${valueText}.`;
		case "workflow.current_task": return `The current working focus appears to be ${valueText}.`;
		case "workflow.next_action": return `The next likely action appears to be ${valueText}.`;
		case "workflow.blocker": return `The current blocker appears to be ${valueText}.`;
		default: return `The current state appears to be ${valueText}.`;
	}
}
function buildEventSupport(beliefMap, event) {
	const workflow = eventWorkflowHints(event)[0];
	if (!workflow) return null;
	const stateKey = canonicalStateKey(workflow.key);
	if (!DERIVED_STATE_SUPPORTED_KEYS.has(stateKey)) return null;
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
		confidence: clamp01(average([workflow.confidence ?? .6, belief?.posteriorConfidence ?? .58])),
		usefulnessScore: belief?.usefulnessScore ?? .42,
		stabilityScore: belief?.stabilityScore ?? .48,
		contradictionScore: belief?.contradictionScore ?? .08,
		beliefId: belief?.beliefId
	};
}
function taskMetadataStateEntries(task) {
	const metadata = task.metadataJson ?? {};
	const entries = [];
	if (typeof metadata.project === "string" && metadata.project.trim()) entries.push({
		key: "project.active_project",
		valueJson: {
			project: metadata.project.trim(),
			status: "active"
		}
	});
	if (typeof metadata.currentTask === "string" && metadata.currentTask.trim()) entries.push({
		key: "workflow.current_task",
		valueJson: { task: metadata.currentTask.trim() }
	});
	if (typeof metadata.nextAction === "string" && metadata.nextAction.trim()) entries.push({
		key: "workflow.next_action",
		valueJson: { step: metadata.nextAction.trim() }
	});
	if (typeof metadata.blocker === "string" && metadata.blocker.trim()) entries.push({
		key: "workflow.blocker",
		valueJson: {
			blocker: metadata.blocker.trim(),
			status: "blocked"
		}
	});
	return entries;
}
function buildTaskSupports(beliefMap, task) {
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
		confidence: clamp01(average([belief?.posteriorConfidence ?? .7, .84])),
		usefulnessScore: belief?.usefulnessScore ?? .6,
		stabilityScore: belief?.stabilityScore ?? .64,
		contradictionScore: belief?.contradictionScore ?? .06,
		beliefId: belief?.beliefId
	}));
}
function supportDiversityScore(entries) {
	const supportKinds = new Set(entries.map((entry) => entry.supportKind));
	const sessions = new Set(entries.map((entry) => entry.sessionKey).filter(Boolean));
	return clamp01(Math.min(.45, (entries.length - 1) * .12) + Math.min(.25, Math.max(0, supportKinds.size - 1) * .25) + Math.min(.3, Math.max(0, sessions.size - 1) * .18));
}
function temporalPersistenceScore(entries) {
	if (entries.length === 0) return 0;
	const observedAt = entries.map((entry) => entry.observedAt).sort();
	const spreadDays = daysBetween(observedAt[0], observedAt.at(-1));
	return clamp01(Math.min(.5, spreadDays / 7) + Math.min(.22, Math.max(0, entries.length - 1) * .08) + (spreadDays >= 1 ? .16 : 0));
}
function contradictionPressure(entries, siblingGroups) {
	const totalConfidence = entries.reduce((sum, entry) => sum + entry.confidence, 0);
	const competingConfidence = siblingGroups.filter((group) => group !== entries).reduce((sum, group) => sum + group.reduce((groupSum, entry) => groupSum + entry.confidence, 0), 0);
	return clamp01((totalConfidence + competingConfidence > 0 ? competingConfidence / (totalConfidence + competingConfidence) : 0) * .75 + average(entries.map((entry) => entry.contradictionScore)) * .25);
}
function conceptSupportDiversity(entries) {
	const supportKinds = new Set(entries.map((entry) => entry.supportKind));
	const relationFamilies = new Set(entries.map((entry) => entry.relationLabel));
	return clamp01(Math.min(.32, Math.max(0, entries.length - 1) * .08) + Math.min(.28, Math.max(0, supportKinds.size - 1) * .28) + Math.min(.4, Math.max(0, relationFamilies.size - 1) * .12));
}
function conceptTemporalPersistence(entries) {
	if (entries.length === 0) return 0;
	const observedAt = entries.map((entry) => entry.observedAt).sort();
	const spreadDays = daysBetween(observedAt[0], observedAt.at(-1));
	return clamp01(Math.min(.54, spreadDays / 30) + Math.min(.18, Math.max(0, entries.length - 1) * .06) + (spreadDays >= 3 ? .12 : 0));
}
function topLabels(entries, limit = 4) {
	const counts = /* @__PURE__ */ new Map();
	for (const entry of entries) counts.set(entry.relationLabel, (counts.get(entry.relationLabel) ?? 0) + 1);
	return [...counts.entries()].sort((left, right) => right[1] - left[1] || left[0].localeCompare(right[0])).map(([label]) => label).slice(0, limit);
}
function conceptSummary(entityName, relationLabels) {
	return truncateText(`A stable concept cluster appears around ${entityName}, linking ${relationLabels.length > 0 ? relationLabels.join(", ") : "stable supporting relations"}.`, 220);
}
function conceptCandidateStage(params) {
	if (params.contradictionScore >= .46) return "quarantined";
	if (params.confidence >= .82 && params.usefulnessScore >= .64 && params.stabilityScore >= .66 && params.structuralStrength >= .52) return "probationary";
	return "candidate";
}
function trimGraphHypothesisName(value) {
	return truncateText(value.trim(), 140).trim();
}
function graphHypothesisSupportDiversity(entries) {
	const supportKinds = new Set(entries.map((entry) => entry.supportKind));
	const sessions = new Set(entries.map((entry) => entry.sessionKey).filter(Boolean));
	return clamp01(Math.min(.4, Math.max(0, entries.length - 1) * .1) + Math.min(.3, Math.max(0, supportKinds.size - 1) * .18) + Math.min(.3, Math.max(0, sessions.size - 1) * .16));
}
function graphHypothesisTemporalPersistence(entries) {
	if (entries.length === 0) return 0;
	const observedAt = entries.map((entry) => entry.observedAt).sort();
	const spreadDays = daysBetween(observedAt[0], observedAt.at(-1));
	return clamp01(Math.min(.5, spreadDays / 10) + Math.min(.18, Math.max(0, entries.length - 1) * .08) + (spreadDays >= 1 ? .14 : 0));
}
function graphHypothesisContradictionPressure(entries, siblingGroups) {
	const totalConfidence = entries.reduce((sum, entry) => sum + entry.confidence, 0);
	const competingConfidence = siblingGroups.filter((group) => group !== entries).reduce((sum, group) => sum + group.reduce((groupSum, entry) => groupSum + entry.confidence, 0), 0);
	return clamp01((totalConfidence + competingConfidence > 0 ? competingConfidence / (totalConfidence + competingConfidence) : 0) * .72 + average(entries.map((entry) => entry.contradictionScore)) * .28);
}
function graphHypothesisSummary(relationClass, relationType, sourceName, targetName) {
	const clause = `${sourceName} ${graphRelationPhrase(relationType)} ${targetName}`;
	if (relationClass === "observed") return truncateText(clause, 220);
	return truncateText(`A recurring structure suggests ${clause}.`, 220);
}
function graphHypothesisStage(params) {
	if (params.contradictionScore >= .5) return "quarantined";
	if (params.relationClass === "observed" && params.confidence >= .74 && params.usefulnessScore >= .5 && params.stabilityScore >= .5) return "probationary";
	if (params.confidence >= .68 && params.supportCount >= 2 && (params.supportDiversity >= .28 || params.temporalPersistence >= .18)) return "probationary";
	return "candidate";
}
function buildEventGraphSupports(beliefMap, event) {
	const relations = eventRelationHints(event);
	if (relations.length === 0) return [];
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
		confidence: clamp01(average([event.confidence, belief?.posteriorConfidence ?? .68])),
		usefulnessScore: belief?.usefulnessScore ?? .46,
		stabilityScore: belief?.stabilityScore ?? .54,
		contradictionScore: belief?.contradictionScore ?? .08,
		beliefId: belief?.beliefId
	}));
}
function buildFactGraphSupport(beliefMap, fact) {
	const relationType = relationalFactGraphRelation(fact);
	if (!fact.canonicalObject || !relationType) return null;
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
		confidence: clamp01(average([fact.confidence, belief?.posteriorConfidence ?? .72])),
		usefulnessScore: belief?.usefulnessScore ?? .5,
		stabilityScore: belief?.stabilityScore ?? .58,
		contradictionScore: belief?.contradictionScore ?? .08,
		beliefId: belief?.beliefId
	};
}
function buildGraphHypothesisCandidate(params) {
	const sourceName = trimGraphHypothesisName(params.sourceName);
	const targetName = trimGraphHypothesisName(params.targetName);
	if (!sourceName || !targetName || sourceName === targetName) return null;
	const supportDiversity = graphHypothesisSupportDiversity(params.supports);
	const temporalPersistence = graphHypothesisTemporalPersistence(params.supports);
	const contradictionScore = graphHypothesisContradictionPressure(params.supports, params.siblingGroups);
	const usefulnessScore = clamp01(average(params.supports.map((entry) => entry.usefulnessScore)));
	const stabilityScore = clamp01(average(params.supports.map((entry) => entry.stabilityScore)) * .34 + temporalPersistence * .26 + supportDiversity * .16 + (params.relationClass === "observed" ? .12 : .06) + (1 - contradictionScore) * .12);
	const confidence = clamp01(average(params.supports.map((entry) => entry.confidence)) * .42 + supportDiversity * .18 + temporalPersistence * .12 + usefulnessScore * .1 + (params.relationClass === "observed" ? .12 : .06) + (1 - contradictionScore) * .06);
	if (confidence < .52) return null;
	const stage = graphHypothesisStage({
		relationClass: params.relationClass,
		confidence,
		usefulnessScore,
		stabilityScore,
		contradictionScore,
		supportCount: params.supports.length,
		supportDiversity,
		temporalPersistence
	});
	const sourceNormalized = normalizeText(sourceName);
	const targetNormalized = normalizeText(targetName);
	const semanticKey = `graph_hypothesis:${params.scope}:${sourceNormalized}:${params.relationType}:${params.relationSlot ?? ""}:${targetNormalized}`;
	return {
		candidateId: stableHash([
			params.agentId,
			params.scope,
			semanticKey
		]),
		agentId: params.agentId,
		scope: params.scope,
		abstractionType: "graph_hypothesis",
		semanticKey,
		summary: graphHypothesisSummary(params.relationClass, params.relationType, sourceName, targetName),
		supportContentRefs: [...new Set(params.supports.map((entry) => entry.contentRef))],
		supportBeliefIds: [...new Set(params.supports.map((entry) => entry.beliefId).filter((beliefId) => Boolean(beliefId)))],
		confidence,
		usefulnessScore,
		stabilityScore,
		contradictionScore,
		stage,
		metadataJson: {
			relationType: params.relationType,
			...params.relationSlot ? { relationSlot: params.relationSlot } : {},
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
			firstSeenAt: params.supports.map((entry) => entry.observedAt).sort().at(0),
			lastSeenAt: params.supports.map((entry) => entry.observedAt).sort().at(-1),
			generatedFrom: [...new Set(params.supports.map((entry) => entry.supportKind))],
			semanticSource: "upstream_structured",
			semanticSources: ["upstream_structured", "deterministic_lifecycle"],
			frameworkRound: 4
		},
		createdAt: params.now,
		updatedAt: params.now
	};
}
function buildGraphHypothesisCandidates(store, ctx, beliefMap, recentEvents, _activeTasks) {
	const recentFacts = store.factRepo.query({
		agentId: ctx.agentId,
		scopes: ctx.scopes,
		limit: GRAPH_HYPOTHESIS_FACT_LIMIT,
		includeHistorical: true
	});
	store.stateRepo.get({
		agentId: ctx.agentId,
		scopes: ctx.scopes,
		now: ctx.now
	});
	const supports = [...recentEvents.flatMap((event) => buildEventGraphSupports(beliefMap, event)), ...recentFacts.map((fact) => buildFactGraphSupport(beliefMap, fact)).filter((entry) => entry !== null)];
	const grouped = /* @__PURE__ */ new Map();
	for (const entry of supports) {
		const key = `${entry.scope}:${normalizeText(entry.sourceName)}:${entry.relationType}:${entry.relationSlot ?? ""}:${normalizeText(entry.targetName)}`;
		const bucket = grouped.get(key) ?? [];
		bucket.push(entry);
		grouped.set(key, bucket);
	}
	const siblingGroups = /* @__PURE__ */ new Map();
	for (const group of grouped.values()) {
		const familyKey = `${group[0].scope}:${normalizeText(group[0].sourceName)}:${group[0].relationType}:${group[0].relationSlot ?? ""}`;
		const bucket = siblingGroups.get(familyKey) ?? [];
		bucket.push(group);
		siblingGroups.set(familyKey, bucket);
	}
	return [...grouped.values()].map((group) => buildGraphHypothesisCandidate({
		agentId: ctx.agentId,
		scope: group[0].scope,
		relationType: group[0].relationType,
		relationSlot: group[0].relationSlot,
		relationClass: group[0].relationClass,
		sourceName: group[0].sourceName,
		targetName: group[0].targetName,
		sourceType: group[0].sourceType,
		targetType: group[0].targetType,
		supports: group,
		siblingGroups: siblingGroups.get(`${group[0].scope}:${normalizeText(group[0].sourceName)}:${group[0].relationType}:${group[0].relationSlot ?? ""}`) ?? [group],
		now: ctx.now
	})).filter((candidate) => candidate !== null);
}
function candidateStage(params) {
	if (params.confidence >= .72 && params.contradiction <= .28 && params.supportCount >= 3 && (params.supportDiversity >= .4 || params.temporalPersistence >= .32)) return "probationary";
	return "candidate";
}
function workflowCandidateStage(params) {
	if (params.contradictionScore >= .52) return "quarantined";
	if (params.confidence >= .78 && params.usefulnessScore >= .62 && params.stabilityScore >= .62) return "probationary";
	const roles = new Set(params.groundedByRoles ?? []);
	const resolvedPhase = params.taskPhase === "resolved" || params.taskPhase === "validated";
	if ((params.explicitInstruction === true || params.groundedResolution === true || resolvedPhase && roles.has("user") && (roles.has("tool") || roles.has("assistant"))) && params.contradictionScore <= .24 && params.confidence >= .64 && params.usefulnessScore >= .54 && params.stabilityScore >= .54) return "probationary";
	return "candidate";
}
function explicitWorkflowCandidateStage(params) {
	if (params.contradictionScore >= .52) return "quarantined";
	if (params.explicitInstruction && params.confidence >= .78 && params.stabilityScore >= .56 && params.contradictionScore <= .18) return "probationary";
	if (params.explicitInstruction && params.confidence >= .68 && params.usefulnessScore >= .52 && params.stabilityScore >= .52) return "probationary";
	return workflowCandidateStage(params);
}
function explicitStrategyDomainLabel(summary) {
	const concise = conciseWorkflowDomainPart(summary);
	if (concise) return concise;
	const tokens = tokenizeSearchTerms(summary, EXPLICIT_STRATEGY_STOPWORDS).slice(0, 4);
	if (tokens.length > 0) return tokens.join(" ");
	return truncateText(summary, 48);
}
function conciseWorkflowDomainPart(value) {
	const trimmed = value.replace(/\s+/g, " ").replace(/[。.!！?？]+$/u, "").trim();
	if (!trimmed) return;
	if (/[^\x00-\x7F]/u.test(trimmed)) {
		const phrase = trimmed.split(/[，,;；。.!！?？]/u).map((part) => part.trim()).find((part) => part.length >= 4 && part.length <= 24);
		if (phrase) return truncateText(phrase, 48);
		if (trimmed.length <= 24) return truncateText(trimmed, 48);
	}
	const wordCount = trimmed.split(/\s+/u).filter(Boolean).length;
	if (wordCount >= 2 && wordCount <= 6 && trimmed.length <= 48) return trimmed;
}
function workflowDomainLabelFromParts(parts, fallback) {
	for (const part of parts) {
		const concise = conciseWorkflowDomainPart(part);
		if (concise) return concise;
	}
	return explicitStrategyDomainLabel(fallback);
}
function explicitStrategyDomainKey(summary) {
	return `explicit.${normalizeText(explicitStrategyDomainLabel(summary)).replace(/[^\p{L}\p{N}]+/gu, ".").replace(/^\.+|\.+$/g, "") || "explicit"}.${stableHash([normalizeText(summary)]).slice(0, 8)}`;
}
function explicitStrategySummary(summary, domainLabel) {
	const cleaned = summary.replace(/[。.!！?？]+$/u, "").trim();
	if (!cleaned) return "";
	if (/^when handling\b/i.test(cleaned) || /^prefer:/i.test(cleaned)) return truncateText(cleaned, 240);
	return truncateText(`When handling ${domainLabel || "similar tasks"}, prefer: ${cleaned}.`, 240);
}
function factGuidanceText(fact) {
	const guidanceText = stringValue(objectRecord(fact.objectValueJson?.guidance)?.guidanceText);
	if (guidanceText) return guidanceText;
	if (fact.predicate !== ADVICE_SIGNAL_PREDICATE) return;
	const problemContext = stringValue(fact.objectValueJson?.problemContext);
	const assistantRecommendation = stringValue(fact.objectValueJson?.assistantRecommendation);
	const resources = Array.isArray(fact.objectValueJson?.userResources) ? fact.objectValueJson.userResources.filter((entry) => typeof entry === "string" && entry.trim().length > 0).slice(0, 4) : [];
	const parts = [
		problemContext ? `problem: ${problemContext}` : void 0,
		resources.length > 0 ? `resources: ${resources.join(", ")}` : void 0,
		assistantRecommendation ? `recommendation: ${assistantRecommendation}` : void 0
	].filter(Boolean);
	return parts.length > 0 ? truncateText(parts.join(" | "), 240) : void 0;
}
function structuredWorkflowGuidanceSemanticSource(fact) {
	const guidance = objectRecord(fact.objectValueJson?.guidance);
	const semanticFamily = stringValue(fact.objectValueJson?.semanticFamily);
	if (fact.predicate === WORKFLOW_GUIDANCE_PREDICATE || semanticFamily === "strategy_like" || semanticFamily === "workflow") return "upstream_structured";
	if (stringValue(guidance?.guidanceText) && stringValue(guidance?.reason)?.includes("semantic draft")) return "upstream_structured";
	return null;
}
function isWorkflowGuidanceFact(fact) {
	if (fact.canonicalSubject !== "user" || fact.status !== "active") return false;
	return Boolean(factGuidanceText(fact) && structuredWorkflowGuidanceSemanticSource(fact));
}
function explicitWorkflowSupportChunks(store, fact) {
	const supportRefs = stringSet(objectRecord(objectRecord(fact.objectValueJson?.guidance))?.supportContentRefs ?? fact.objectValueJson?.supportContentRefs);
	if (supportRefs.size === 0) return [];
	return uniqueChunks([...supportRefs].filter((ref) => ref.startsWith("chunk:")).map((ref) => store.chunkRepo.get(ref.slice(6))).filter((chunk) => Boolean(chunk)));
}
function stringSet(value) {
	if (!Array.isArray(value)) return /* @__PURE__ */ new Set();
	return new Set(value.filter((entry) => typeof entry === "string" && entry.trim().length > 0));
}
function uniqueChunks(chunks) {
	const seen = /* @__PURE__ */ new Set();
	const ordered = [];
	for (const chunk of chunks) {
		if (seen.has(chunk.chunkId)) continue;
		seen.add(chunk.chunkId);
		ordered.push(chunk);
	}
	return ordered;
}
function maintenanceSemanticSourceRank(candidate) {
	switch (stringValue(candidate.metadataJson.semanticSource)) {
		case "upstream_structured": return 4;
		case "embedding_clustered": return 3;
		case "deterministic_lifecycle": return 2;
		case "llm_upgrade": return 1;
		case "lexical_fallback": return 0;
		default: return 1;
	}
}
function resolutionGroundingSignal(params) {
	const metadata = objectRecord(params.task.metadataJson) ?? {};
	const explicitEvidenceCount = uniqueChunks(params.evidenceChunks).length;
	const hasUserEvidence = params.evidenceChunks.some((chunk) => chunk.role === "user");
	const hasToolEvidence = params.evidenceChunks.some((chunk) => chunk.role === "tool");
	const hasAssistantEvidence = params.evidenceChunks.some((chunk) => chunk.role === "assistant");
	const promotionScore = typeof metadata.candidateResolutionPromotionScore === "number" && Number.isFinite(metadata.candidateResolutionPromotionScore) ? metadata.candidateResolutionPromotionScore : 0;
	const hasOutcomeKey = Boolean(stringValue(metadata.lastEmittedOutcomeKey));
	const complexity = contentStructuralComplexity(params.resolution);
	const score = clamp01((explicitEvidenceCount > 0 ? .36 : 0) + (hasToolEvidence ? .28 : 0) + (hasUserEvidence ? .12 : 0) + (hasAssistantEvidence ? .08 : 0) + (hasOutcomeKey ? .1 : 0) + promotionScore * .24);
	return {
		grounded: (explicitEvidenceCount > 0 || hasToolEvidence || promotionScore >= .74 || hasOutcomeKey) && score >= .28 && complexity <= .42 && !params.resolution.includes("```"),
		score,
		complexity,
		semanticSource: "upstream_structured"
	};
}
function buildExplicitWorkflowPatternCandidates(store, ctx, beliefMap) {
	const workflowGuidanceFacts = store.factRepo.query({
		agentId: ctx.agentId,
		scopes: ctx.scopes,
		limit: 48,
		includeHistorical: false
	}).filter(isWorkflowGuidanceFact);
	const explicitFacts = [...new Map(workflowGuidanceFacts.map((fact) => [fact.factId, fact])).values()];
	const candidates = [];
	for (const fact of explicitFacts) {
		const objectSummary = stringValue(fact.canonicalObject)?.trim();
		const guidanceText = factGuidanceText(fact);
		const semanticSource = structuredWorkflowGuidanceSemanticSource(fact);
		if (!semanticSource) continue;
		const supportChunks = explicitWorkflowSupportChunks(store, fact);
		const summarySource = (supportChunks.length > 0 ? truncateText(supportChunks.map((chunk) => chunk.content.trim()).filter(Boolean).join("；"), 240) : void 0) || objectSummary || guidanceText;
		if (!summarySource) continue;
		const belief = beliefMap.get(`fact:${fact.factId}`);
		const explicitInstruction = fact.canonicalSubject === "user" && fact.status === "active" && fact.predicate !== ADVICE_SIGNAL_PREDICATE && Boolean(guidanceText);
		const groundedByRoles = supportChunks.length > 0 ? ["user"] : [];
		const contradictionScore = clamp01(belief?.contradictionScore ?? .08);
		const usefulnessScore = clamp01((belief?.usefulnessScore ?? .58) * .64 + fact.confidence * .18 + (belief?.stage === "active" ? .18 : .08) + (explicitInstruction ? .08 : 0) + (supportChunks.length > 0 ? .08 : 0));
		const stabilityScore = clamp01((belief?.stabilityScore ?? .62) * .58 + (belief?.stage === "active" ? .18 : .08) + (1 - contradictionScore) * .24 + (explicitInstruction ? .06 : 0) + (supportChunks.length > 0 ? .06 : 0));
		const confidence = clamp01(average([fact.confidence, belief?.posteriorConfidence ?? fact.confidence]) * .72 + usefulnessScore * .14 + (1 - contradictionScore) * .14 + (explicitInstruction ? .1 : 0) + (supportChunks.length > 0 ? .08 : 0));
		const stage = explicitWorkflowCandidateStage({
			confidence,
			usefulnessScore,
			stabilityScore,
			contradictionScore,
			explicitInstruction
		});
		const domainLabel = explicitStrategyDomainLabel(summarySource);
		const domainKey = explicitStrategyDomainKey(summarySource);
		const normalizedSummary = explicitStrategySummary(summarySource, domainLabel);
		if (!normalizedSummary) continue;
		candidates.push({
			candidateId: stableHash([
				ctx.agentId,
				fact.scope,
				"explicit_workflow_pattern",
				fact.factId
			]),
			agentId: ctx.agentId,
			scope: fact.scope,
			abstractionType: "workflow_pattern",
			semanticKey: `workflow_pattern:${fact.scope}:${domainKey}`,
			summary: normalizedSummary,
			supportContentRefs: [`fact:${fact.factId}`, ...supportChunks.map((chunk) => `chunk:${chunk.chunkId}`)],
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
				frameworkRound: 3
			},
			createdAt: ctx.now,
			updatedAt: ctx.now
		});
	}
	return candidates;
}
function buildGroundedWorkflowPatternCandidates(store, ctx, beliefMap) {
	const recentTasks = store.taskRepo.listRecent({
		agentId: ctx.agentId,
		scopes: ctx.scopes,
		limit: 24
	});
	const candidates = [];
	for (const task of recentTasks) {
		const metadata = objectRecord(task.metadataJson) ?? {};
		const resolution = stringValue(metadata.candidateResolution);
		const phase = stringValue(metadata.candidateResolutionPhase) ?? stringValue(metadata.taskPhase) ?? "";
		if (!resolution || task.status !== "completed" && phase !== "validated" && phase !== "resolved") continue;
		const belief = beliefMap.get(`task:${task.taskId}`);
		if (belief && (belief.stage === "quarantined" || belief.stage === "superseded")) continue;
		if (tokenizeSearchTerms(resolution, EXPLICIT_STRATEGY_STOPWORDS).length < 3) continue;
		const chunks = store.chunkRepo.listByTask(task.taskId);
		const evidenceChunkIds = stringSet(metadata.candidateResolutionEvidenceChunkIds);
		const explicitEvidenceChunks = chunks.filter((chunk) => evidenceChunkIds.has(chunk.chunkId));
		const toolEvidenceChunks = chunks.filter((chunk) => chunk.role === "tool");
		const supportChunks = uniqueChunks([...explicitEvidenceChunks, ...toolEvidenceChunks]);
		const supportChunkIds = [...supportChunks.filter((chunk) => chunk.role === "user" || chunk.role === "assistant" || chunk.role === "tool").map((chunk) => `chunk:${chunk.chunkId}`)];
		const hasUserEvidence = supportChunks.some((chunk) => chunk.role === "user");
		const hasToolEvidence = supportChunks.some((chunk) => chunk.role === "tool");
		const hasAssistantEvidence = supportChunks.some((chunk) => chunk.role === "assistant");
		const resolutionGrounding = resolutionGroundingSignal({
			task,
			resolution,
			evidenceChunks: supportChunks
		});
		const hasGroundedResolution = resolutionGrounding.grounded;
		if (!(hasToolEvidence || hasAssistantEvidence && hasUserEvidence || hasGroundedResolution || (belief?.outcomeSupportScore ?? 0) >= .72)) continue;
		const contradictionScore = clamp01(belief?.contradictionScore ?? .1);
		const usefulnessScore = clamp01((belief?.usefulnessScore ?? .58) * .44 + (hasToolEvidence ? .2 : .1) + (hasUserEvidence ? .12 : .04) + (phase === "resolved" ? .12 : .08) + (hasGroundedResolution ? .08 : 0) + (1 - contradictionScore) * .12);
		const stabilityScore = clamp01((belief?.stabilityScore ?? .6) * .42 + (hasToolEvidence ? .18 : .06) + (hasAssistantEvidence ? .08 : 0) + (phase === "resolved" ? .12 : .08) + (hasGroundedResolution ? .08 : 0) + (1 - contradictionScore) * .2);
		const confidence = clamp01(average([belief?.posteriorConfidence ?? .68, task.status === "completed" ? .76 : .7]) * .56 + usefulnessScore * .18 + stabilityScore * .18 + (hasToolEvidence ? .08 : hasAssistantEvidence ? .04 : 0) + (hasGroundedResolution ? .04 : 0));
		const groundedByRoles = [
			...hasUserEvidence ? ["user"] : [],
			...hasAssistantEvidence ? ["assistant"] : [],
			...hasToolEvidence ? ["tool"] : []
		];
		const stage = workflowCandidateStage({
			confidence,
			usefulnessScore,
			stabilityScore,
			contradictionScore,
			groundedByRoles,
			taskPhase: phase,
			groundedResolution: hasGroundedResolution
		});
		const domainParts = [
			stringValue(metadata.currentTask),
			task.title.trim(),
			stringValue(metadata.project)
		].filter((value) => Boolean(value));
		const domainSource = domainParts.join(" ").trim();
		const domainLabel = workflowDomainLabelFromParts(domainParts, domainSource || resolution);
		const domainKey = explicitStrategyDomainKey(domainSource || resolution);
		const summary = explicitStrategySummary(resolution, domainLabel);
		if (!summary) continue;
		candidates.push({
			candidateId: stableHash([
				ctx.agentId,
				task.scope,
				"grounded_workflow_pattern",
				task.taskId
			]),
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
				taskPhase: phase || void 0,
				groundedByRoles,
				groundedResolution: hasGroundedResolution || void 0,
				groundedResolutionScore: Number(resolutionGrounding.score.toFixed(3)),
				groundedResolutionComplexity: Number(resolutionGrounding.complexity.toFixed(3)),
				generatedFrom: "grounded_task_resolution",
				semanticSource: resolutionGrounding.semanticSource,
				semanticSources: [resolutionGrounding.semanticSource],
				frameworkRound: 6
			},
			createdAt: ctx.now,
			updatedAt: ctx.now
		});
	}
	return candidates;
}
function conceptSeedNames(facts) {
	const names = /* @__PURE__ */ new Map();
	for (const fact of facts) for (const name of [fact.canonicalSubject, fact.canonicalObject]) {
		if (!name) continue;
		const normalized = normalizeText(name);
		if (!normalized || CONCEPT_GENERIC_NAMES.has(normalized)) continue;
		if (!names.has(normalized)) names.set(normalized, name);
	}
	return [...names.values()];
}
function factMatchesConceptSeed(fact, seedName) {
	const normalizedSeed = normalizeText(seedName);
	return normalizeText(fact.canonicalSubject) === normalizedSeed || normalizeText(fact.canonicalObject ?? "") === normalizedSeed;
}
function buildConceptSupportEntries(params) {
	const factSupports = params.facts.map((fact) => {
		const belief = params.beliefMap.get(`fact:${fact.factId}`);
		return {
			supportKind: "fact",
			contentRef: `fact:${fact.factId}`,
			relationLabel: fact.predicate,
			observedAt: fact.updatedAt,
			confidence: clamp01(average([fact.confidence, belief?.posteriorConfidence ?? .66])),
			usefulnessScore: belief?.usefulnessScore ?? .46,
			stabilityScore: belief?.stabilityScore ?? .58,
			contradictionScore: belief?.contradictionScore ?? .08,
			beliefId: belief?.beliefId
		};
	});
	const edgeSupports = params.graph.edges.map((edge) => {
		const belief = params.beliefMap.get(`graph_edge:${edge.edgeId}`);
		const isSeedIncident = edge.srcEntityId === params.seedEntityId || edge.dstEntityId === params.seedEntityId;
		return {
			supportKind: "graph_edge",
			contentRef: edge.sourceKind === "stored" ? `graph_edge:${edge.edgeId}` : edge.evidenceRef || `graph_edge:${edge.edgeId}`,
			relationLabel: edge.relType,
			observedAt: edge.updatedAt ?? "",
			confidence: clamp01(average([edge.confidence, belief?.posteriorConfidence ?? (isSeedIncident ? .7 : .62)])),
			usefulnessScore: belief?.usefulnessScore ?? (isSeedIncident ? .54 : .44),
			stabilityScore: belief?.stabilityScore ?? (isSeedIncident ? .62 : .52),
			contradictionScore: belief?.contradictionScore ?? (edge.relType === "contradicts" ? .42 : .08),
			beliefId: belief?.beliefId
		};
	});
	return {
		factSupports,
		edgeSupports,
		supportBeliefIds: [...new Set([...factSupports, ...edgeSupports].map((entry) => entry.beliefId).filter((beliefId) => Boolean(beliefId)))],
		supportContentRefs: [...new Set([...factSupports, ...edgeSupports].map((entry) => entry.contentRef))]
	};
}
function resolveConceptGraphEntity(params) {
	const result = resolveEntityMention(params.store, params.ctx, buildEntityMention({
		ctx: params.ctx,
		scope: params.scope,
		rawText: params.name,
		semanticRole: "support",
		sourceRef: params.sourceRef,
		supportText: params.supportText,
		observedAt: params.observedAt,
		metadataJson: { generatedFrom: "abstraction-concept-graph-resolution" }
	}), {
		createIfMissing: false,
		persist: false
	});
	if (result.method !== "uncertain") return result.entity;
	return {
		entityId: graphHypothesisNodeId(params.name, "unknown"),
		canonicalName: params.name.trim(),
		entityType: "unknown",
		normalizedName: normalizeText(params.name),
		confidence: .58
	};
}
function ensureConceptGraphNode(params) {
	const entity = resolveConceptGraphEntity(params);
	const existing = params.nodes.get(entity.entityId);
	if (existing) return existing;
	const node = {
		nodeId: entity.entityId,
		nodeKind: "entity",
		entityId: entity.entityId,
		name: entity.canonicalName,
		type: entity.entityType,
		confidence: entity.confidence,
		observedAt: params.observedAt
	};
	params.nodes.set(node.nodeId, node);
	return node;
}
function conceptGraphEdgeKey(edge) {
	return `${edge.srcNodeId}:${edge.relType}:${edge.dstNodeId}`;
}
function upsertConceptGraphEdge(edges, edge) {
	const key = conceptGraphEdgeKey(edge);
	const existing = edges.get(key);
	if (!existing) {
		edges.set(key, edge);
		return;
	}
	const preferred = existing.sourceKind === "stored" || edge.sourceKind !== "stored" ? existing : edge;
	const fallback = preferred === existing ? edge : existing;
	edges.set(key, {
		...preferred,
		confidence: Math.max(existing.confidence, edge.confidence),
		updatedAt: Date.parse(existing.updatedAt ?? "") >= Date.parse(edge.updatedAt ?? "") ? existing.updatedAt : edge.updatedAt,
		evidenceRef: preferred.evidenceRef ?? fallback.evidenceRef
	});
}
function mergeConceptStoredGraph(params) {
	for (const node of params.graph.nodes) if (!params.nodes.has(node.nodeId)) params.nodes.set(node.nodeId, node);
	for (const edge of params.graph.edges) upsertConceptGraphEdge(params.edges, edge);
}
function buildConceptGraphEvidence(params) {
	const seedEntity = resolveConceptGraphEntity({
		store: params.store,
		ctx: params.ctx,
		scope: params.scope,
		name: params.seedName,
		observedAt: params.ctx.now,
		sourceRef: `maintenance:concept:${stableHash([params.scope, params.seedName])}`,
		supportText: params.seedName
	});
	const nodes = /* @__PURE__ */ new Map();
	const edges = /* @__PURE__ */ new Map();
	ensureConceptGraphNode({
		store: params.store,
		ctx: params.ctx,
		scope: params.scope,
		nodes,
		name: seedEntity.canonicalName,
		observedAt: params.ctx.now,
		sourceRef: `maintenance:concept:${stableHash([
			params.scope,
			params.seedName,
			"seed"
		])}`,
		supportText: params.seedName
	});
	for (const fact of params.facts) {
		const relType = relationalFactGraphRelation(fact);
		if (!fact.canonicalObject || !relType) continue;
		const src = ensureConceptGraphNode({
			store: params.store,
			ctx: params.ctx,
			scope: params.scope,
			nodes,
			name: fact.canonicalSubject,
			observedAt: fact.updatedAt,
			sourceRef: `fact:${fact.factId}`,
			supportText: fact.provenanceText
		});
		const dst = ensureConceptGraphNode({
			store: params.store,
			ctx: params.ctx,
			scope: params.scope,
			nodes,
			name: fact.canonicalObject,
			observedAt: fact.updatedAt,
			sourceRef: `fact:${fact.factId}`,
			supportText: fact.provenanceText
		});
		upsertConceptGraphEdge(edges, {
			edgeId: stableHash([
				"concept-fact-edge",
				fact.factId,
				src.nodeId,
				relType,
				dst.nodeId
			]),
			srcNodeId: src.nodeId,
			srcEntityId: src.nodeId,
			relType,
			dstNodeId: dst.nodeId,
			dstEntityId: dst.nodeId,
			confidence: clamp01(fact.confidence),
			evidenceRef: `fact:${fact.factId}`,
			updatedAt: fact.updatedAt,
			sourceKind: "synthesized"
		});
	}
	if (!seedEntity.entityId.startsWith("graph_hypothesis_node:")) mergeConceptStoredGraph({
		nodes,
		edges,
		graph: params.store.graphRepo.expandNeighborhood({
			agentId: params.ctx.agentId,
			scopes: [params.scope],
			seedEntityIds: [seedEntity.entityId],
			maxHops: Math.min(params.ctx.config.graphMaxHops, 2),
			maxEdges: Math.min(params.ctx.config.maxGraphEdges, CONCEPT_GRAPH_EDGE_BUDGET),
			maxNodes: Math.min(params.ctx.config.maxGraphNodes, CONCEPT_GRAPH_NODE_BUDGET),
			now: params.ctx.now
		})
	});
	const edgeList = [...edges.values()].sort((left, right) => right.confidence - left.confidence || (right.updatedAt ? Date.parse(right.updatedAt) : 0) - (left.updatedAt ? Date.parse(left.updatedAt) : 0));
	const pathCandidates = buildGraphPathCandidates({
		seedNodeIds: [seedEntity.entityId],
		nodes,
		edges: edgeList,
		now: params.ctx.now,
		maxPaths: Math.max(6, Math.min(CONCEPT_GRAPH_EDGE_BUDGET + 2, 12)),
		maxHops: 2
	});
	return {
		seedEntityId: seedEntity.entityId,
		entityName: seedEntity.canonicalName,
		graph: {
			nodes: [...nodes.values()],
			edges: edgeList,
			pathCandidates,
			paths: pathCandidates.map((path) => path.summary)
		}
	};
}
function buildConceptCandidate(params) {
	if (params.facts.length < 2) return null;
	const relationalFactCount = params.facts.filter((fact) => Boolean(relationalFactGraphRelation(fact))).length;
	if (params.graph.edges.length === 0 && relationalFactCount < 2) return null;
	const { factSupports, edgeSupports, supportBeliefIds, supportContentRefs } = buildConceptSupportEntries({
		seedEntityId: params.entityId,
		facts: params.facts,
		graph: params.graph,
		beliefMap: params.beliefMap
	});
	const supports = [...factSupports, ...edgeSupports];
	const relationLabels = topLabels(supports);
	const relationFamilies = new Set(supports.map((entry) => entry.relationLabel));
	const supportDiversity = conceptSupportDiversity(supports);
	const temporalPersistence = conceptTemporalPersistence(supports);
	const contradictionScore = clamp01(average(supports.map((entry) => entry.contradictionScore)));
	const pathScoreAverage = average(params.graph.pathCandidates.map((path) => path.score));
	const relationDensity = clamp01((relationFamilies.size + params.graph.edges.length + relationalFactCount) / 10);
	const graphCoverage = clamp01((params.graph.edges.length + params.graph.nodes.length + params.graph.pathCandidates.length) / 18);
	const structuralStrength = clamp01(relationDensity * .34 + pathScoreAverage * .34 + graphCoverage * .2 + supportDiversity * .12);
	if (relationFamilies.size < 2 || structuralStrength < .38) return null;
	const usefulnessScore = clamp01(average(supports.map((entry) => entry.usefulnessScore)) * .58 + pathScoreAverage * .18 + relationDensity * .14 + supportDiversity * .1);
	const stabilityScore = clamp01(average(supports.map((entry) => entry.stabilityScore)) * .42 + temporalPersistence * .18 + structuralStrength * .22 + supportDiversity * .08 + (1 - contradictionScore) * .1);
	const confidence = clamp01(average(supports.map((entry) => entry.confidence)) * .34 + structuralStrength * .28 + supportDiversity * .16 + usefulnessScore * .12 + temporalPersistence * .06 + (1 - contradictionScore) * .04);
	if (confidence < .56) return null;
	const stage = conceptCandidateStage({
		confidence,
		usefulnessScore,
		stabilityScore,
		contradictionScore,
		structuralStrength
	});
	const semanticKey = `concept_candidate:${params.scope}:${normalizeText(params.entityName)}`;
	return {
		candidateId: stableHash([
			params.agentId,
			params.scope,
			semanticKey
		]),
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
			frameworkRound: 5
		},
		createdAt: params.now,
		updatedAt: params.now
	};
}
function buildConceptCandidates(store, ctx, beliefMap) {
	const recentFacts = store.factRepo.query({
		agentId: ctx.agentId,
		scopes: ctx.scopes,
		limit: CONCEPT_FACT_LIMIT,
		includeHistorical: false
	});
	const candidates = [];
	for (const seedName of conceptSeedNames(recentFacts)) {
		const scopeFacts = recentFacts.filter((fact) => ctx.scopes.includes(fact.scope) && factMatchesConceptSeed(fact, seedName));
		if (scopeFacts.length < 2) continue;
		const scopeFactGroups = /* @__PURE__ */ new Map();
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
				facts
			});
			const candidate = buildConceptCandidate({
				agentId: ctx.agentId,
				scope,
				entityId: seedEntityId,
				entityName,
				facts,
				graph,
				beliefMap,
				now: ctx.now
			});
			if (candidate) candidates.push(candidate);
		}
	}
	return candidates;
}
function buildDerivedStateCandidate(params) {
	const valueText = describeStateValue(params.stateKey, params.valueJson).trim();
	if (!valueText) return null;
	const supportDiversity = supportDiversityScore(params.supports);
	const temporalPersistence = temporalPersistenceScore(params.supports);
	const contradiction = contradictionPressure(params.supports, params.siblingGroups);
	const usefulnessScore = clamp01(average(params.supports.map((entry) => entry.usefulnessScore)));
	const stabilityScore = clamp01(average(params.supports.map((entry) => entry.stabilityScore)) * .35 + temporalPersistence * .4 + supportDiversity * .15 + (1 - contradiction) * .1);
	const confidence = clamp01(average(params.supports.map((entry) => entry.confidence)) * .42 + supportDiversity * .2 + temporalPersistence * .16 + usefulnessScore * .14 + (1 - contradiction) * .08);
	if (params.supports.length < 2 || confidence < .52) return null;
	const stage = candidateStage({
		confidence,
		contradiction,
		supportCount: params.supports.length,
		supportDiversity,
		temporalPersistence
	});
	const semanticKey = `derived_state:${params.stateKey}:${semanticValueKey(params.stateKey, params.valueJson)}`;
	const contentRefs = [...new Set(params.supports.map((entry) => entry.contentRef))];
	const beliefIds = [...new Set(params.supports.map((entry) => entry.beliefId).filter((beliefId) => Boolean(beliefId)))];
	return {
		candidateId: stableHash([
			params.agentId,
			params.scope,
			semanticKey
		]),
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
			firstSeenAt: params.supports.map((entry) => entry.observedAt).sort().at(0),
			lastSeenAt: params.supports.map((entry) => entry.observedAt).sort().at(-1),
			supportKinds: [...new Set(params.supports.map((entry) => entry.supportKind))],
			semanticSource: "upstream_structured",
			semanticSources: ["upstream_structured", "deterministic_lifecycle"],
			frameworkRound: 2
		},
		createdAt: params.now,
		updatedAt: params.now
	};
}
async function buildWorkflowPatternCandidates(store, ctx, beliefMap) {
	const repeatedCandidates = (await deriveWorkflowPatternSummaries(store, ctx)).map((pattern) => {
		const stage = workflowCandidateStage({
			confidence: pattern.confidence,
			usefulnessScore: pattern.usefulnessScore,
			stabilityScore: pattern.stabilityScore,
			contradictionScore: pattern.contradictionScore,
			groundedByRoles: Array.isArray(pattern.metadataJson.groundedByRoles) ? pattern.metadataJson.groundedByRoles.filter((role) => typeof role === "string" && role.trim().length > 0) : void 0,
			taskPhase: stringValue(pattern.metadataJson.taskPhase),
			explicitInstruction: Boolean(pattern.metadataJson.explicitInstruction),
			groundedResolution: Boolean(pattern.metadataJson.groundedResolution)
		});
		const semanticKey = `workflow_pattern:${pattern.scope}:${pattern.domainKey}`;
		return {
			candidateId: stableHash([
				ctx.agentId,
				pattern.scope,
				semanticKey
			]),
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
				frameworkRound: 3
			},
			createdAt: ctx.now,
			updatedAt: ctx.now
		};
	});
	const explicitCandidates = buildExplicitWorkflowPatternCandidates(store, ctx, beliefMap);
	const groundedCandidates = buildGroundedWorkflowPatternCandidates(store, ctx, beliefMap);
	const merged = /* @__PURE__ */ new Map();
	for (const candidate of [
		...repeatedCandidates,
		...explicitCandidates,
		...groundedCandidates
	]) {
		const existing = merged.get(candidate.semanticKey);
		const candidateRank = maintenanceSemanticSourceRank(candidate);
		const existingRank = existing ? maintenanceSemanticSourceRank(existing) : -1;
		if (!existing || candidateRank > existingRank || candidateRank === existingRank && (candidate.confidence > existing.confidence || candidate.usefulnessScore > existing.usefulnessScore)) merged.set(candidate.semanticKey, candidate);
	}
	return [...merged.values()];
}
async function refineCandidatesWithLlm(store, candidates, now, ctx) {
	if (!store.reasoner.isEnabled()) return {
		candidates,
		considered: 0,
		refined: 0
	};
	const refinedCandidates = [...candidates];
	const resolvedModel = store.reasoner.getResolvedJudgeModel();
	let considered = 0;
	let refined = 0;
	for (const [index, candidate] of refinedCandidates.entries()) {
		if (!eligibleForLlmRefinement(candidate) || considered >= ABSTRACTION_LLM_REFINEMENT_LIMIT) continue;
		considered += 1;
		const result = await store.reasoner.judgeAbstractionCandidate(candidate, {
			stage: "maintenance_async",
			audit: ctx.llmBudgetAudit
		});
		if (!result) continue;
		const refinedCandidate = applyAbstractionRefinement({
			candidate,
			result,
			now,
			resolvedModel
		});
		if (!refinedCandidate) continue;
		refined += 1;
		refinedCandidates[index] = refinedCandidate;
	}
	return {
		candidates: refinedCandidates,
		considered,
		refined
	};
}
async function runAbstractionJobs(store, ctx, options = { refineWithLlm: false }) {
	const runStartedAt = nowIso();
	const runId = store.auditRepo.startMaintenance({
		agentId: ctx.agentId,
		sessionKey: options.batch?.sessionKey ?? ctx.sessionKey,
		jobType: "abstraction-jobs",
		stats: {},
		startedAt: runStartedAt
	});
	try {
		if (options.batch && options.deltaTriggered === false) {
			const skippedStats = {
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
				supersededCandidates: 0
			};
			store.auditRepo.finishMaintenance({
				runId,
				agentId: ctx.agentId,
				jobType: "abstraction-jobs",
				statsJson: {
					...options.batch ? { batch: options.batch } : {},
					...skippedStats,
					llmBudget: snapshotMemoryLlmBudgetAudit(ctx.llmBudgetAudit)
				},
				startedAt: runStartedAt,
				completedAt: nowIso(),
				status: "completed"
			});
			return skippedStats;
		}
		const recentEvents = store.eventRepo.search({
			agentId: ctx.agentId,
			scopes: ctx.scopes,
			...options.batch?.sessionKey ? { sessionKey: options.batch.sessionKey } : {},
			limit: 64,
			since: olderThanDays(Math.max(DERIVED_STATE_EVENT_WINDOW_DAYS, ctx.config.episodicDedupWindowDays * 2), ctx.now)
		});
		const recentBeliefs = store.beliefRepo.listByAgent({
			agentId: ctx.agentId,
			limit: 192
		});
		const taskBeliefs = recentBeliefs.filter((belief) => belief.memoryKind === "task").slice(0, 24);
		const beliefMap = buildBeliefMap(recentBeliefs);
		const activeTasks = store.taskRepo.listActive({
			agentId: ctx.agentId,
			scopes: ctx.scopes,
			...options.batch?.sessionKey ? { sessionKey: options.batch.sessionKey } : {},
			limit: 24
		});
		const factFamilies = Number(store.client.prepare(`SELECT COUNT(*) AS count
               FROM (
                 SELECT canonical_subject, predicate
                   FROM facts
                  WHERE agent_id = ?
                    AND scope IN (${ctx.scopes.map(() => "?").join(", ")})
                    AND status IN ('active', 'uncertain')
                  GROUP BY canonical_subject, predicate
                  LIMIT 24
               )`).get(ctx.agentId, ...ctx.scopes)?.count ?? 0);
		const supports = [...recentEvents.map((event) => buildEventSupport(beliefMap, event)).filter((entry) => entry !== null), ...activeTasks.flatMap((task) => buildTaskSupports(beliefMap, task))];
		const groupedByKey = /* @__PURE__ */ new Map();
		for (const entry of supports) {
			const familyKey = `${entry.scope}:${entry.stateKey}:${semanticValueKey(entry.stateKey, entry.valueJson)}`;
			const bucket = groupedByKey.get(familyKey) ?? [];
			bucket.push(entry);
			groupedByKey.set(familyKey, bucket);
		}
		const siblingGroupsByState = /* @__PURE__ */ new Map();
		for (const group of groupedByKey.values()) {
			const stateKey = `${group[0].scope}:${group[0].stateKey}`;
			const bucket = siblingGroupsByState.get(stateKey) ?? [];
			bucket.push(group);
			siblingGroupsByState.set(stateKey, bucket);
		}
		const derivedStateCandidates = [...groupedByKey.values()].map((group) => buildDerivedStateCandidate({
			agentId: ctx.agentId,
			scope: group[0].scope,
			stateKey: group[0].stateKey,
			valueJson: group[0].valueJson,
			supports: group,
			siblingGroups: siblingGroupsByState.get(`${group[0].scope}:${group[0].stateKey}`) ?? [group],
			now: ctx.now
		})).filter((candidate) => candidate !== null).sort((left, right) => {
			if (right.confidence !== left.confidence) return right.confidence - left.confidence;
			return right.usefulnessScore - left.usefulnessScore;
		});
		const workflowPatternCandidates = await buildWorkflowPatternCandidates(store, ctx, beliefMap);
		const graphHypothesisCandidates = buildGraphHypothesisCandidates(store, ctx, beliefMap, recentEvents, activeTasks);
		const conceptCandidates = buildConceptCandidates(store, ctx, beliefMap);
		const candidates = [
			...derivedStateCandidates,
			...workflowPatternCandidates,
			...graphHypothesisCandidates,
			...conceptCandidates
		].sort(compareAbstractionCandidates);
		const candidateSelection = selectAbstractionCandidatesByType({
			derived_state: derivedStateCandidates,
			workflow_pattern: workflowPatternCandidates,
			graph_hypothesis: graphHypothesisCandidates,
			concept_candidate: conceptCandidates
		});
		const selected = candidateSelection.selected;
		const refinement = options.refineWithLlm === false ? {
			candidates: selected,
			considered: 0,
			refined: 0
		} : await refineCandidatesWithLlm(store, selected, ctx.now, ctx);
		const sourceEpoch = ctx.readEpoch ?? store.client.currentMemoryEpoch(ctx.agentId);
		for (const candidate of refinement.candidates) {
			candidate.derivedFromMinEpoch = candidate.derivedFromMinEpoch ?? sourceEpoch;
			candidate.derivedFromMaxEpoch = candidate.derivedFromMaxEpoch ?? sourceEpoch;
			candidate.materializedEpoch = store.client.nextMemoryEpoch(ctx.agentId, ctx.now);
			candidate.derivedFromKind = candidate.derivedFromKind ?? "abstraction_job";
			candidate.derivedFromIds = candidate.derivedFromIds ?? [...candidate.supportContentRefs, ...candidate.supportBeliefIds];
			candidate.derivedAtEpoch = candidate.derivedAtEpoch ?? sourceEpoch;
			candidate.derivationPolicyVersion = candidate.derivationPolicyVersion ?? "memx-authority-v3";
			candidate.metadataJson = buildMaintenanceContractMetadata({
				existing: {
					...candidate.metadataJson,
					...candidate.abstractionType === "concept_candidate" ? { promotionBlocker: "not_promotable_yet" } : {}
				},
				sourceRef: `abstraction_candidate:${candidate.candidateId}`,
				supportContentRefs: candidate.supportContentRefs,
				supportBeliefIds: candidate.supportBeliefIds,
				derivedFromIds: candidate.derivedFromIds,
				semanticSource: stringValue(candidate.metadataJson.semanticSource) ?? "upstream_structured",
				semanticSources: Array.isArray(candidate.metadataJson.semanticSources) ? candidate.metadataJson.semanticSources.filter((source) => typeof source === "string" && source.trim().length > 0) : void 0,
				authoritySource: options.refineWithLlm === true ? "llm_upgrade" : "deterministic_aggregated",
				generatedFrom: typeof candidate.metadataJson.generatedFrom === "string" || Array.isArray(candidate.metadataJson.generatedFrom) ? candidate.metadataJson.generatedFrom : "abstraction_job",
				recallLayer: "abstraction",
				answerEligibleByDefault: false,
				materializedEpoch: candidate.materializedEpoch,
				derivationPolicyVersion: candidate.derivationPolicyVersion
			});
			store.abstractionRepo.upsert(candidate);
		}
		const materializedCandidateIds = refinement.candidates.map((candidate) => candidate.candidateId);
		const stages = candidateStageCounts(store, ctx.agentId);
		const semanticSources = [...new Set(refinement.candidates.flatMap((candidate) => {
			const primary = stringValue(candidate.metadataJson.semanticSource);
			const secondary = Array.isArray(candidate.metadataJson.semanticSources) ? candidate.metadataJson.semanticSources.filter((source) => typeof source === "string" && source.trim().length > 0) : [];
			return [...primary ? [primary] : [], ...secondary];
		}))];
		const maintenanceContractDiagnostics = summarizeMaintenanceContractDiagnostics(refinement.candidates.map((candidate) => candidate.metadataJson));
		const stats = {
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
			deltaTriggered: options.batch ? options.deltaTriggered !== false : void 0,
			skippedNoRelevantDelta: false,
			authoritySources: options.refineWithLlm === true ? ["deterministic_aggregated", "llm_upgrade"] : ["deterministic_aggregated"],
			semanticSources: options.refineWithLlm === true ? [...new Set([...semanticSources, "llm_upgrade"])] : semanticSources,
			maintenanceContractDiagnostics,
			recallFacingDiagnostics: {
				recallVisible: maintenanceContractDiagnostics.recallVisibleCount > 0,
				answerEligibleByDefault: maintenanceContractDiagnostics.answerEligibleByDefaultCount > 0,
				sourceRefsForExpansion: maintenanceContractDiagnostics.sourceRefsForExpansion,
				recallLayers: maintenanceContractDiagnostics.recallLayers
			},
			activeCandidates: stages.active,
			probationaryCandidates: stages.probationary,
			candidateCandidates: stages.candidate,
			decayingCandidates: stages.decaying,
			quarantinedCandidates: stages.quarantined,
			supersededCandidates: stages.superseded
		};
		store.auditRepo.finishMaintenance({
			runId,
			agentId: ctx.agentId,
			jobType: "abstraction-jobs",
			statsJson: {
				...options.batch ? { batch: options.batch } : {},
				...stats,
				llmBudget: snapshotMemoryLlmBudgetAudit(ctx.llmBudgetAudit)
			},
			startedAt: runStartedAt,
			completedAt: nowIso(),
			status: "completed"
		});
		return stats;
	} catch (error) {
		store.auditRepo.finishMaintenance({
			runId,
			agentId: ctx.agentId,
			jobType: "abstraction-jobs",
			statsJson: {
				...options.batch ? { batch: options.batch } : {},
				error: String(error),
				llmBudget: snapshotMemoryLlmBudgetAudit(ctx.llmBudgetAudit)
			},
			startedAt: runStartedAt,
			completedAt: nowIso(),
			status: "failed"
		});
		throw error;
	}
}
//#endregion
export { runAbstractionJobs };
