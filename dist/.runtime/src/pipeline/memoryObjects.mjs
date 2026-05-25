import { clamp01, normalizeText, objectRecord, stableHash, truncateText } from "../support.mjs";
import { canonicalStateKey } from "./semantic/heuristics.mjs";
import { TASK_METADATA_WORKFLOW_SNAPSHOT_DESCRIPTORS, isSnapshotFactualStateKey, sanitizeTaskMetadata } from "./authority.mjs";
import { filterBootstrapRows, isBootstrapMemoryContamination } from "./bootstrapFilter.mjs";
import { buildEntityMention, resolveEntityMention } from "./entityResolver.mjs";
import { containsLikelySecret, looksLikePromptInjection } from "../security/injection.mjs";
import "./semantics.mjs";
import { dedupeEvidenceRows, describeStateValue, formatFactLine, lineageFromMetadata, normalizeSearchText, rowsFromSearchHits, shouldSuppressRecallText, splitLabelValue, toEvidenceRow } from "./memoryObjectsHelpers.mjs";
import { evaluateStateCurrentness } from "./stateLifecycle.mjs";
import { STRATEGY_MAX_CONTRADICTION } from "./constants.mjs";
import { buildGraphPathCandidates } from "./graphPathEngine.mjs";
import { buildWorkingProjectionBlocks, createMemorySelectionObjective } from "./memoryObjectsProjection.mjs";
import { semanticTaskSummaryText } from "./taskSummary.mjs";
//#region src/pipeline/memoryObjects.ts
const RELATIONAL_FACT_PREDICATES = new Set([
	"depends_on",
	"blocks",
	"uses",
	"reads",
	"caused_by",
	"related_to",
	"part_of",
	"owner_of",
	"supersedes",
	"contradicts",
	"resolved_by"
]);
const GRAPH_STRUCTURAL_RELATION_TYPES = new Set([
	"resolved_by",
	"supported_by",
	"derived_from",
	"updates",
	"contradicts"
]);
function metadataString(value) {
	return typeof value === "string" && value.trim() ? value.trim() : void 0;
}
function eventStructuredSummary(event) {
	const detail = metadataString(event.metadataJson.memxRetrievalDetailExcerpt);
	const summary = metadataString(objectRecord(event.metadataJson.memxTemporalFacet)?.summary) ?? metadataString(event.metadataJson.memxStructuredSummary) ?? event.text;
	if (detail && !normalizeText(summary).includes(normalizeText(detail))) return `${summary} | Detail: ${detail}`;
	return summary;
}
function eventStructuredHints(event) {
	return objectRecord(event.metadataJson.memxStructuredHints);
}
function eventTemporalFacet(event) {
	return objectRecord(event.metadataJson.memxTemporalFacet);
}
function eventTemporalRole(event) {
	return metadataString(eventTemporalFacet(event)?.role);
}
function eventTemporalStructured(event) {
	const facet = eventTemporalFacet(event);
	if (facet && typeof facet.structured === "boolean") return facet.structured;
	return Boolean(metadataString(event.metadataJson.memxStructuredSummary));
}
function eventStructuredRelationCount(event) {
	const hints = eventStructuredHints(event);
	const relations = hints?.relations;
	if (Array.isArray(relations) && relations.length > 0) return relations.length;
	return hints?.relation ? 1 : 0;
}
function eventStructuredEntityCount(event) {
	const hints = eventStructuredHints(event);
	return Array.isArray(hints?.entities) ? hints.entities.length : 0;
}
function taskSemanticReferences(task) {
	const metadata = sanitizeTaskMetadata(task.metadataJson);
	const candidateResolution = metadataString(objectRecord(task.metadataJson)?.candidateResolution);
	const stableSummary = semanticTaskSummaryText(task);
	return [
		metadataString(task.title),
		candidateResolution,
		metadata.currentTask,
		metadata.project,
		metadata.nextAction,
		metadata.blocker,
		stableSummary
	].filter((value) => Boolean(value));
}
function eventRows(events) {
	return events.map((event, index) => toEvidenceRow({
		id: event.eventId,
		text: eventStructuredSummary(event),
		score: clamp01(Math.max(.2, 1 - index * .08) + (event.sourceKind === "tool" ? .06 : 0) + (event.sourceKind === "user" ? .03 : 0) + (event.sourceKind === "assistant" && !eventTemporalStructured(event) ? -.12 : 0) + (eventStructuredRelationCount(event) > 0 ? .06 : 0)),
		scope: event.scope,
		confidence: event.confidence,
		sourceRef: event.sourceRef,
		observedAt: event.observedAt,
		provenance: event.sourceKind,
		lineage: {
			canonicalKind: "event",
			canonicalId: event.eventId,
			sourceKind: "event",
			sourceId: event.eventId,
			sourceRef: event.sourceRef,
			materializedEpoch: event.materializedEpoch
		}
	}));
}
function taskRows(tasks) {
	return tasks.map((task, index) => {
		const metadata = sanitizeTaskMetadata(task.metadataJson);
		const contextBits = [
			metadata.project ? `project=${metadata.project}` : "",
			metadata.currentTask ? `current_task=${metadata.currentTask}` : "",
			metadata.blocker ? `blocker=${metadata.blocker}` : ""
		].filter(Boolean);
		const contextSuffix = contextBits.length > 0 ? ` | ${contextBits.join(" | ")}` : "";
		return toEvidenceRow({
			id: task.taskId,
			text: `${taskSemanticReferences(task).join(" | ")}${contextSuffix}`,
			score: Math.max(.28, .94 - index * .12),
			scope: task.scope,
			confidence: .9,
			sourceRef: `task:${task.taskId}`,
			observedAt: task.updatedAt,
			lineage: {
				canonicalKind: "task",
				canonicalId: task.taskId,
				sourceKind: "task",
				sourceId: task.taskId,
				sourceRef: `task:${task.taskId}`
			}
		});
	});
}
function workflowActiveTasks(store, ctx, limit = 4) {
	const scopedSessionTasks = ctx.sessionKey ? ctx.scopes.map((scope) => store.taskRepo.getActive({
		agentId: ctx.agentId,
		scope,
		sessionKey: ctx.sessionKey
	})).filter((task) => Boolean(task)).sort((left, right) => right.updatedAt.localeCompare(left.updatedAt)) : [];
	if (scopedSessionTasks.length > 0) return scopedSessionTasks.slice(0, limit);
	return store.taskRepo.listActive({
		agentId: ctx.agentId,
		scopes: ctx.scopes,
		limit
	});
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
function stateRowsFromTaskMetadata(tasks) {
	const rows = [];
	for (const [index, task] of tasks.entries()) {
		const metadata = sanitizeTaskMetadata(task.metadataJson);
		const observedAt = task.updatedAt;
		for (const descriptor of TASK_METADATA_WORKFLOW_SNAPSHOT_DESCRIPTORS) {
			const rawValue = metadata[descriptor.metadataKey];
			if (!rawValue) continue;
			rows.push(toEvidenceRow({
				id: `${task.taskId}:${descriptor.stateKey}`,
				text: `${descriptor.stateKey}: ${rawValue}`,
				score: Math.max(.16, descriptor.baseScore * .58 - index * .08),
				scope: task.scope,
				confidence: descriptor.confidence,
				sourceRef: `task:${task.taskId}`,
				observedAt,
				lineage: {
					sourceKind: "task",
					sourceId: task.taskId,
					sourceRef: `task:${task.taskId}`
				}
			}));
		}
	}
	return rows;
}
function filterTaskMetadataBackgroundStates(baseStates, derivedStates) {
	const canonicalBaseKeys = new Set(baseStates.map((row) => splitLabelValue(row.text).label.trim()).filter(Boolean));
	return derivedStates.filter((row) => {
		const key = splitLabelValue(row.text).label.trim();
		return Boolean(key) && !canonicalBaseKeys.has(key);
	});
}
function chunkRows(hits) {
	return hits.map((hit, index) => {
		const chunkId = typeof hit.metadata.chunkId === "string" ? hit.metadata.chunkId : hit.docId.replace(/^event:chunk:/, "");
		const role = typeof hit.metadata.role === "string" ? hit.metadata.role : "memory";
		const assistantWeight = role === "assistant" && typeof hit.metadata.assistantWeight === "number" ? hit.metadata.assistantWeight : void 0;
		const text = normalizeSearchText(hit.text);
		const newline = text.indexOf("\n");
		return toEvidenceRow({
			id: chunkId,
			text: `[${role}] ${newline >= 0 && newline < text.length - 1 ? text.slice(newline + 1).trim() || text : text}`,
			score: Math.max(.12, (assistantWeight != null ? hit.score * Math.max(.35, assistantWeight) : hit.score) - index * .04),
			scope: typeof hit.metadata.scope === "string" ? hit.metadata.scope : "unknown",
			confidence: typeof hit.metadata.confidence === "number" ? hit.metadata.confidence : void 0,
			observedAt: typeof hit.metadata.observedAt === "string" ? hit.metadata.observedAt : void 0,
			provenance: role,
			lineage: lineageFromMetadata(hit.metadata, {
				sourceKind: "chunk",
				sourceId: chunkId
			})
		});
	});
}
function guidanceTextFromFact(fact) {
	if (fact.canonicalSubject !== "user") return null;
	if (fact.predicate === "has_workflow_guidance") return null;
	const guidance = fact.objectValueJson?.guidance;
	if (guidance && typeof guidance === "object") {
		const guidanceText = guidance.guidanceText;
		if (typeof guidanceText === "string" && guidanceText.trim()) return guidanceText.trim();
	}
	return null;
}
function collectBehavioralGuidance(store, ctx) {
	const beliefIndex = buildBeliefIndex(store, ctx.agentId, ctx.readEpoch);
	const facts = store.factRepo.query({
		agentId: ctx.agentId,
		scopes: ctx.scopes,
		limit: 24,
		readEpoch: ctx.readEpoch
	}).sort((left, right) => {
		const leftBelief = beliefIndex.byContentRef.get(left.factId);
		const priorityDiff = backgroundBeliefPriority(beliefIndex.byContentRef.get(right.factId)) - backgroundBeliefPriority(leftBelief);
		if (Math.abs(priorityDiff) < .05) return left.updatedAt > right.updatedAt ? -1 : left.updatedAt < right.updatedAt ? 1 : 0;
		return priorityDiff;
	});
	const guidance = /* @__PURE__ */ new Set();
	for (const fact of facts) {
		if (!includeBackgroundBelief(beliefIndex.byContentRef.get(fact.factId))) continue;
		const line = guidanceTextFromFact(fact);
		if (line && !isBootstrapMemoryContamination(line)) guidance.add(line);
	}
	return [...guidance].slice(0, 6);
}
function collectStrategyGuidance(store, ctx) {
	return store.strategyRepo.listByAgent({
		agentId: ctx.agentId,
		scopes: ctx.scopes,
		stages: ["active"],
		limit: 4,
		readEpoch: ctx.readEpoch
	}).filter((strategy) => strategy.contradictionScore < STRATEGY_MAX_CONTRADICTION).sort((left, right) => {
		const leftGrounded = left.metadataJson.groundedResolution === true ? 1 : 0;
		const rightGrounded = right.metadataJson.groundedResolution === true ? 1 : 0;
		const leftExplicitOnly = left.metadataJson.explicitInstruction === true && left.metadataJson.groundedResolution !== true ? 1 : 0;
		const rightExplicitOnly = right.metadataJson.explicitInstruction === true && right.metadataJson.groundedResolution !== true ? 1 : 0;
		const leftPriority = leftGrounded * .18 + left.usefulnessScore * .34 + left.stabilityScore * .24 + left.confidence * .18 - leftExplicitOnly * .08;
		return rightGrounded * .18 + right.usefulnessScore * .34 + right.stabilityScore * .24 + right.confidence * .18 - rightExplicitOnly * .08 - leftPriority || right.updatedAt.localeCompare(left.updatedAt);
	}).map((strategy) => strategy.summary.trim()).filter(Boolean);
}
function baseStateRows(store, ctx) {
	const dedupedStates = /* @__PURE__ */ new Map();
	for (const state of store.stateRepo.get({
		agentId: ctx.agentId,
		scopes: ctx.scopes,
		now: ctx.now,
		readEpoch: ctx.readEpoch
	})) {
		const key = canonicalStateKey(state.key);
		if (!dedupedStates.has(key)) dedupedStates.set(key, state);
	}
	return [...dedupedStates.values()].flatMap((state, index) => {
		const currentness = evaluateStateCurrentness({
			key: state.key,
			stateKind: state.stateKind,
			valueJson: state.valueJson,
			sourceRef: state.sourceRef,
			updatedAt: state.updatedAt,
			observedAt: state.updatedAt,
			expiresAt: state.expiresAt,
			scope: state.scope,
			now: ctx.now
		});
		if (currentness.hardExclusions.length > 0) return [];
		const stateRef = `${state.scope}:${state.key}`;
		return [toEvidenceRow({
			id: stateRef,
			text: `${canonicalStateKey(state.key)}: ${describeStateValue(state.key, state.valueJson)}`,
			score: Math.max(.18, currentness.currentnessScore * .76 + state.confidence * .16 - index * .04),
			scope: state.scope,
			confidence: state.confidence,
			sourceRef: state.sourceRef,
			observedAt: state.updatedAt,
			lineage: {
				canonicalKind: "state",
				canonicalId: stateRef,
				sourceKind: "state",
				sourceId: stateRef,
				sourceRef: state.sourceRef,
				materializedEpoch: state.materializedEpoch
			}
		})];
	});
}
const BACKGROUND_STATE_KEYS = new Set([
	"workflow.current_task",
	"workflow.next_action",
	"workflow.current_consideration"
]);
function includeMinimalBackgroundState(row) {
	const { label } = splitLabelValue(row.text);
	return BACKGROUND_STATE_KEYS.has(label.trim());
}
function buildBackgroundRecallBundle(store, ctx) {
	const beliefIndex = buildBeliefIndex(store, ctx.agentId, ctx.readEpoch);
	const behavioralGuidance = collectBehavioralGuidance(store, ctx).slice(0, 5);
	const strategyGuidance = collectStrategyGuidance(store, ctx).slice(0, 3);
	const activeTasks = workflowActiveTasks(store, ctx, 3);
	const tasks = dedupeEvidenceRows(filterBootstrapRows(taskRows(activeTasks)).filter((entry) => !looksLikePromptInjection(entry.text) && !containsLikelySecret(entry.text)).filter((entry) => includeBackgroundBelief(beliefIndex.byContentRef.get(entry.id))).sort((left, right) => backgroundBeliefPriority(beliefIndex.byContentRef.get(right.id)) - backgroundBeliefPriority(beliefIndex.byContentRef.get(left.id))), 3);
	const canonicalStates = filterBootstrapRows(baseStateRows(store, ctx));
	const derivedTaskMetadataStates = filterTaskMetadataBackgroundStates(canonicalStates, stateRowsFromTaskMetadata(activeTasks));
	const states = dedupeEvidenceRows([...canonicalStates, ...filterBootstrapRows(derivedTaskMetadataStates)].filter((entry) => includeMinimalBackgroundState(entry)).filter((entry) => !looksLikePromptInjection(entry.text) && !containsLikelySecret(entry.text)).filter((entry) => includeBackgroundBelief(beliefIndex.byContentRef.get(entry.id))).sort((left, right) => backgroundBeliefPriority(beliefIndex.byContentRef.get(right.id)) - backgroundBeliefPriority(beliefIndex.byContentRef.get(left.id))), 5);
	return {
		behavioralGuidance,
		strategyGuidance,
		states,
		tasks,
		projectionBlocks: buildWorkingProjectionBlocks({
			behavioralGuidance,
			states
		})
	};
}
function isRelationalFactPredicate(predicate) {
	return RELATIONAL_FACT_PREDICATES.has(predicate) || predicate.startsWith("uses_");
}
function queryMemoryFacts(params) {
	return params.store.factRepo.query({
		agentId: params.ctx.agentId,
		scopes: params.ctx.scopes,
		text: params.text,
		limit: params.limit,
		includeHistorical: params.includeHistorical,
		readEpoch: params.ctx.readEpoch
	});
}
function recencyScore(observedAt, now) {
	if (!observedAt) return .42;
	const nowMs = Date.parse(now);
	const observedMs = Date.parse(observedAt);
	if (!Number.isFinite(nowMs) || !Number.isFinite(observedMs)) return .42;
	const ageDays = Math.max(0, (nowMs - observedMs) / 864e5);
	const longRange = Math.exp(-ageDays / 21);
	const shortRange = Math.exp(-ageDays / .167);
	return clamp01(longRange * .65 + shortRange * .35);
}
function stabilityScoreForObject(kind, attrs) {
	switch (kind) {
		case "fact": return attrs.guidance ? .94 : attrs.relational ? .76 : .88;
		case "state": return attrs.syntheticWorkflow ? .78 : .82;
		case "task": return attrs.activeTask ? .8 : .72;
		case "graph_path": return .7;
		case "event": return .46;
		case "chunk": return .34;
		case "alternate": return .28;
		default: return .5;
	}
}
function semanticProfileForObject(params) {
	const similarity = clamp01(params.baseScore);
	const recency = recencyScore(params.observedAt, params.now);
	const stability = stabilityScoreForObject(params.kind, params.attrs);
	const entityCount = typeof params.attrs.entityCount === "number" ? params.attrs.entityCount : 0;
	const graphNodeCount = params.attrs.graphNodeCount ?? 0;
	const graphEdgeCount = params.attrs.graphEdgeCount ?? 0;
	const entityFactor = clamp01(entityCount / 3);
	const nodeFactor = clamp01(graphNodeCount / 5);
	const edgeFactor = clamp01(graphEdgeCount / 4);
	const relationSignal = clamp01((params.attrs.relational ? .78 : 0) + (params.kind === "graph_path" ? .16 : 0) + edgeFactor * .34 + nodeFactor * .08);
	const relationDensity = clamp01((params.kind === "graph_path" ? .42 : .08) + relationSignal * .34 + entityFactor * .16 + edgeFactor * .18);
	const connectivity = clamp01((params.kind === "graph_path" ? .38 : .06) + edgeFactor * .28 + nodeFactor * .22 + relationSignal * .12);
	const sessionMatch = params.currentSessionKey && params.attrs.sessionKey && params.currentSessionKey === params.attrs.sessionKey;
	const continuityScore = clamp01((params.attrs.activeTask ? .34 : 0) + (params.kind === "task" ? .18 : params.kind === "state" ? .16 : 0) + (params.attrs.syntheticWorkflow ? .08 : 0) + similarity * .18 + recency * .06 + (sessionMatch ? .08 : 0));
	const guidanceScore = params.attrs.guidance ? clamp01(.72 + stability * .18) : 0;
	return {
		workflowScore: clamp01(similarity * .34 + (params.kind === "state" ? .32 : params.kind === "task" ? .28 : 0) + (params.attrs.syntheticWorkflow ? .1 : 0) + (params.attrs.activeTask ? .12 : 0)),
		factualScore: clamp01(similarity * .34 + (params.kind === "fact" ? params.attrs.relational ? .32 : .44 : 0) + guidanceScore * .16 + stability * .1),
		temporalScore: clamp01(similarity * .24 + recency * .32 + (params.kind === "event" ? .32 : params.kind === "chunk" ? .28 : 0) + (params.kind === "event" && params.attrs.relational ? .1 : 0) + (params.kind === "event" && params.attrs.temporalStructured ? .14 : 0) + (params.kind === "event" && params.attrs.temporalRole === "resolution" ? .08 : 0) + (params.kind === "event" && params.attrs.temporalRole === "cause" ? .08 : 0) + (params.kind === "event" && params.attrs.sourceKind === "tool" ? .08 : 0) + (params.kind === "event" && params.attrs.sourceKind === "user" ? .04 : 0) + (params.kind === "event" && params.attrs.sourceKind === "assistant" ? -.08 : 0) + (params.broadTemporal ? .08 : 0) + (sessionMatch && (params.kind === "event" || params.kind === "chunk") ? .1 : 0)),
		explanationScore: clamp01(similarity * .2 + relationDensity * .34 + connectivity * .28 + (params.kind === "graph_path" ? .12 : 0) + (params.kind === "fact" && params.attrs.relational ? .08 : 0) + (params.kind === "event" && params.attrs.relational ? .12 : 0) + (params.kind === "event" && params.attrs.temporalRole === "cause" ? .08 : 0)),
		relationDensity,
		connectivity,
		continuityScore,
		recencyScore: recency,
		stabilityScore: stability,
		guidanceScore
	};
}
function routeRoleForKind(kind) {
	switch (kind) {
		case "state": return "state";
		case "task": return "task";
		case "fact": return "fact";
		case "event": return "event";
		case "graph_path": return "graph";
		case "chunk": return "chunk";
		case "alternate": return "alternate";
	}
}
function beliefStagePriority(stage) {
	switch (stage) {
		case "active": return 5;
		case "probationary": return 4;
		case "candidate": return 3;
		case "decaying": return 2;
		case "superseded": return 1;
		case "quarantined": return 0;
	}
}
function beliefKindForObjectKind(kind) {
	switch (kind) {
		case "state":
		case "task":
		case "fact":
		case "event":
		case "chunk": return kind;
		default: return null;
	}
}
function semanticKeyForObject(object) {
	switch (object.kind) {
		case "state": return `state:${normalizeText(object.row.id)}`;
		case "task": return `task:${normalizeText(object.row.id)}`;
		case "fact": {
			const subject = object.attributes.factSubject?.trim();
			const predicate = object.attributes.factPredicate?.trim();
			return subject && predicate ? `fact:${normalizeText(subject)}:${predicate}` : `fact:${normalizeText(object.row.id)}`;
		}
		case "event": return `event:${normalizeText(object.row.id)}`;
		case "chunk": return `chunk:${normalizeText(object.row.id)}`;
		default: return;
	}
}
function toMemoryObjectBelief(record) {
	return {
		beliefId: record.beliefId,
		stage: record.stage,
		posteriorConfidence: record.posteriorConfidence,
		usefulnessScore: record.usefulnessScore,
		stabilityScore: record.stabilityScore,
		contradictionScore: record.contradictionScore,
		outcomeSupportScore: record.outcomeSupportScore,
		lastUsedAt: record.lastUsedAt,
		useCount: record.useCount
	};
}
function bestBeliefRecord(records) {
	return records?.slice().sort((left, right) => {
		const stageDelta = beliefStagePriority(right.stage) - beliefStagePriority(left.stage);
		if (stageDelta !== 0) return stageDelta;
		if (right.posteriorConfidence !== left.posteriorConfidence) return right.posteriorConfidence - left.posteriorConfidence;
		if (right.usefulnessScore !== left.usefulnessScore) return right.usefulnessScore - left.usefulnessScore;
		return right.updatedAt.localeCompare(left.updatedAt);
	})[0];
}
function buildBeliefIndex(store, agentId, readEpoch) {
	const beliefs = store.beliefRepo.listByAgent({
		agentId,
		readEpoch
	});
	const byContentRef = /* @__PURE__ */ new Map();
	const bySemanticKey = /* @__PURE__ */ new Map();
	for (const belief of beliefs) {
		if (belief.contentRef) {
			const existing = byContentRef.get(belief.contentRef);
			if (!existing || beliefStagePriority(belief.stage) > beliefStagePriority(existing.stage) || belief.posteriorConfidence > existing.posteriorConfidence) byContentRef.set(belief.contentRef, belief);
		}
		const bucket = bySemanticKey.get(belief.semanticKey) ?? [];
		bucket.push(belief);
		bySemanticKey.set(belief.semanticKey, bucket);
	}
	return {
		byContentRef,
		bySemanticKey
	};
}
function attachBeliefsToObjects(store, ctx, objects) {
	const index = buildBeliefIndex(store, ctx.agentId, ctx.readEpoch);
	return objects.map((object) => {
		const beliefKind = beliefKindForObjectKind(object.kind);
		if (!beliefKind) return object;
		const direct = index.byContentRef.get(object.row.id);
		const semanticKey = semanticKeyForObject(object);
		const semantic = semanticKey ? bestBeliefRecord(index.bySemanticKey.get(semanticKey)) : void 0;
		const record = direct ?? semantic;
		if (!record || record.memoryKind !== beliefKind) return object;
		return {
			...object,
			belief: toMemoryObjectBelief(record)
		};
	});
}
function routeAlignedBeliefStability(routeType, belief) {
	switch (routeType) {
		case "workflow": return clamp01(belief.stabilityScore * .72 + belief.usefulnessScore * .18);
		case "factual": return clamp01(belief.stabilityScore * .82 + (1 - belief.contradictionScore) * .12);
		case "temporal": return clamp01(belief.stabilityScore * .35 + belief.usefulnessScore * .25);
		case "explanatory": return clamp01(belief.stabilityScore * .64 + (1 - belief.contradictionScore) * .14 + belief.outcomeSupportScore * .12);
	}
}
function applyBeliefAdjustment(object, objective, score, reasons) {
	const belief = object.belief;
	if (!belief) return {
		score,
		reasons
	};
	const supportScore = clamp01(belief.posteriorConfidence * .5 + belief.usefulnessScore * .22 + routeAlignedBeliefStability(objective.routeType, belief) * .18 + belief.outcomeSupportScore * .1);
	const contradictionPenalty = belief.contradictionScore * (objective.routeType === "explanatory" ? .12 : .24);
	let adjusted = score;
	switch (belief.stage) {
		case "active":
			adjusted = clamp01(adjusted + supportScore * .22 + .06 - contradictionPenalty);
			break;
		case "probationary":
			adjusted = adjusted >= .48 ? clamp01(adjusted + supportScore * .12 - contradictionPenalty * .8) : clamp01(adjusted - .08 - contradictionPenalty);
			break;
		case "candidate":
			adjusted = clamp01(adjusted + supportScore * .03 - contradictionPenalty);
			break;
		case "decaying":
			adjusted = clamp01(adjusted + supportScore * .04 - .12 - contradictionPenalty);
			break;
		case "superseded":
			adjusted = clamp01(adjusted + supportScore * .02 - .2 - contradictionPenalty);
			break;
		case "quarantined":
			adjusted = 0;
			break;
	}
	return {
		score: adjusted,
		reasons: [
			...reasons,
			`beliefStage=${belief.stage}`,
			`beliefPosterior=${belief.posteriorConfidence.toFixed(2)}`,
			`beliefUsefulness=${belief.usefulnessScore.toFixed(2)}`,
			`beliefContradiction=${belief.contradictionScore.toFixed(2)}`
		]
	};
}
function backgroundBeliefPriority(belief) {
	if (!belief) return .4;
	return clamp01(belief.posteriorConfidence * .48 + belief.usefulnessScore * .18 + belief.stabilityScore * .12 + beliefStagePriority(belief.stage) * .04 - belief.contradictionScore * .22);
}
function includeBackgroundBelief(belief) {
	if (!belief) return true;
	return belief.stage !== "quarantined" && belief.stage !== "superseded";
}
function buildMemoryObject(params) {
	const attributes = params.attrs ?? {};
	return {
		objectId: `${params.kind}:${params.row.id}`,
		kind: params.kind,
		routeRole: routeRoleForKind(params.kind),
		row: params.row,
		lineage: params.row.lineage,
		baseScore: clamp01(params.row.score),
		attributes,
		profile: semanticProfileForObject({
			kind: params.kind,
			text: params.row.text,
			query: params.objective.query,
			attrs: attributes,
			observedAt: params.row.observedAt,
			now: params.objective.now,
			baseScore: params.row.score,
			broadTemporal: params.objective.broadTemporal,
			currentSessionKey: params.objective.currentSessionKey
		}),
		graphNodes: params.graphNodes,
		graphEdges: params.graphEdges,
		graphPathCandidate: params.graphPathCandidate
	};
}
function stateObjects(store, ctx, objective) {
	const activeTasks = workflowActiveTasks(store, ctx, 4);
	const states = filterBootstrapRows(baseStateRows(store, ctx));
	const explicitStateKeys = new Set(states.map((row) => splitLabelValue(row.text).label).filter(Boolean));
	const synthetic = filterBootstrapRows(stateRowsFromTaskMetadata(activeTasks)).filter((row) => {
		const { label } = splitLabelValue(row.text);
		return !explicitStateKeys.has(label);
	});
	return [...states.map((row) => {
		const { label } = splitLabelValue(row.text);
		return buildMemoryObject({
			kind: "state",
			row,
			objective,
			attrs: { stateKey: label }
		});
	}), ...synthetic.map((row) => {
		const { label } = splitLabelValue(row.text);
		const separator = row.id.indexOf(":");
		return buildMemoryObject({
			kind: "state",
			row,
			objective,
			attrs: {
				syntheticWorkflow: true,
				activeTask: true,
				parentTaskId: separator >= 0 ? row.id.slice(0, separator) : void 0,
				stateKey: label
			}
		});
	})];
}
function taskObjects(store, ctx, objective, hybridHits) {
	const activeTasks = workflowActiveTasks(store, ctx, 4);
	const normalizedObjectiveQuery = normalizeText(objective.query);
	const selectedTasks = uniqueTasksById([...store.taskRepo.listRecent({
		agentId: ctx.agentId,
		scopes: ctx.scopes,
		limit: 8,
		sessionKey: ctx.sessionKey
	}).filter((task) => {
		if (taskSemanticReferences(task).some((reference) => normalizeText(reference) && normalizedObjectiveQuery.includes(normalizeText(reference)))) return true;
		return false;
	}), ...activeTasks]).slice(0, 4);
	return filterBootstrapRows(selectedTasks.length > 0 ? taskRows(selectedTasks) : rowsFromSearchHits(hybridHits.filter((hit) => hit.metadata.memxDocType === "task").slice(0, 4))).map((row, index) => {
		const task = selectedTasks.length > 0 ? selectedTasks[index] : void 0;
		return buildMemoryObject({
			kind: "task",
			row,
			objective,
			attrs: {
				activeTask: Boolean(task && activeTasks.some((entry) => entry.taskId === task.taskId)),
				taskId: task?.taskId
			}
		});
	});
}
function factObjects(store, ctx, objective) {
	const facts = queryMemoryFacts({
		store,
		ctx,
		text: objective.query,
		limit: 10,
		includeHistorical: objective.includeHistorical ?? false
	});
	return filterBootstrapRows(facts.map((fact, index) => toEvidenceRow({
		id: fact.factId,
		text: formatFactLine({
			subject: fact.canonicalSubject,
			predicate: fact.predicate,
			object: fact.canonicalObject,
			objectValueJson: fact.objectValueJson,
			status: fact.status
		}),
		score: Math.max(.22, .96 - index * .08),
		scope: fact.scope,
		confidence: fact.confidence,
		observedAt: fact.updatedAt,
		sourceRef: fact.sourceRef,
		lineage: {
			canonicalKind: "fact",
			canonicalId: fact.factId,
			sourceKind: "fact",
			sourceId: fact.factId,
			sourceRef: fact.sourceRef,
			materializedEpoch: fact.materializedEpoch
		}
	}))).map((row, index) => {
		const fact = facts[index];
		return buildMemoryObject({
			kind: "fact",
			row,
			objective,
			attrs: {
				relational: isRelationalFactPredicate(fact.predicate),
				guidance: Boolean(guidanceTextFromFact(fact)),
				entityCount: [fact.canonicalSubject, fact.canonicalObject].filter(Boolean).length,
				factSubject: fact.canonicalSubject,
				factPredicate: fact.predicate,
				factObject: fact.canonicalObject
			}
		});
	});
}
function eventObjects(store, ctx, objective) {
	let events = store.eventRepo.search({
		agentId: ctx.agentId,
		scopes: ctx.scopes,
		text: objective.broadTemporal ? void 0 : objective.query,
		limit: objective.routeType === "temporal" ? 8 : 6,
		since: objective.since,
		readEpoch: ctx.readEpoch
	});
	if (events.length === 0 && objective.routeType === "temporal") events = store.eventRepo.search({
		agentId: ctx.agentId,
		scopes: ctx.scopes,
		limit: 8,
		since: objective.since,
		readEpoch: ctx.readEpoch
	});
	return filterBootstrapRows(eventRows(events)).map((row, index) => buildMemoryObject({
		kind: "event",
		row,
		objective,
		attrs: {
			entityCount: eventStructuredEntityCount(events[index]),
			eventType: events[index]?.eventType,
			sourceKind: events[index]?.sourceKind,
			sessionKey: events[index]?.sessionKey,
			relational: eventStructuredRelationCount(events[index]) > 0,
			temporalStructured: eventTemporalStructured(events[index]),
			temporalRole: eventTemporalRole(events[index]),
			graphNodeCount: eventStructuredEntityCount(events[index]),
			graphEdgeCount: eventStructuredRelationCount(events[index])
		}
	}));
}
function chunkObjects(objective, hybridHits) {
	return filterBootstrapRows(chunkRows(hybridHits.filter((hit) => hit.metadata.memxDocType === "chunk" && normalizeSearchText(hit.text).length > 5).slice(0, 8))).map((row) => buildMemoryObject({
		kind: "chunk",
		row,
		objective,
		attrs: { entityCount: 0 }
	}));
}
function alternateObjects(objective, hybridHits) {
	return filterBootstrapRows(rowsFromSearchHits(hybridHits).slice(0, 10)).map((row) => buildMemoryObject({
		kind: "alternate",
		row,
		objective,
		attrs: { docType: row.id.split(":")[0] }
	}));
}
function stringValue(value) {
	return typeof value === "string" && value.trim() ? value.trim() : void 0;
}
function parseGraphHypothesisCandidate(candidate) {
	if (candidate.abstractionType !== "graph_hypothesis") return null;
	const relationType = stringValue(candidate.metadataJson.relationType);
	const relationClass = stringValue(candidate.metadataJson.relationClass);
	const relationSlot = stringValue(candidate.metadataJson.relationSlot);
	const sourceName = stringValue(candidate.metadataJson.sourceName);
	const targetName = stringValue(candidate.metadataJson.targetName);
	const sourceType = stringValue(candidate.metadataJson.sourceType);
	const targetType = stringValue(candidate.metadataJson.targetType);
	const sourceNodeId = stringValue(candidate.metadataJson.sourceNodeId);
	const targetNodeId = stringValue(candidate.metadataJson.targetNodeId);
	if (!relationType || !relationClass || !sourceName || !targetName || !sourceType || !targetType || !sourceNodeId || !targetNodeId) return null;
	return {
		relationType,
		...relationSlot ? { relationSlot } : {},
		relationClass: relationClass === "observed" ? "observed" : "inferred",
		sourceName,
		targetName,
		sourceType,
		targetType,
		sourceNodeId,
		targetNodeId
	};
}
function graphSeedSpecificity(name) {
	const normalized = name.trim().toLowerCase();
	if (!normalized) return 0;
	const tokenCount = normalized.split(/[^a-z0-9]+/iu).filter(Boolean).length;
	const hasDelimiter = /[-_/.:]/u.test(name);
	const hasDigit = /\d/u.test(name);
	return clamp01(.24 + Math.min(.22, normalized.length / 36) + (tokenCount >= 2 ? .22 : 0) + (hasDelimiter ? .16 : 0) + (hasDigit ? .08 : 0));
}
function filterSpecificGraphEntityMatches(matches) {
	return matches.filter((candidate) => {
		return !isShadowedGraphEntity(candidate, matches);
	});
}
function isShadowedGraphEntity(candidate, matches) {
	const candidateSpecificity = graphSeedSpecificity(candidate.canonicalName);
	return matches.some((other) => {
		if (other.entityId === candidate.entityId) return false;
		const otherSpecificity = graphSeedSpecificity(other.canonicalName);
		return other.normalizedName.includes(candidate.normalizedName) && otherSpecificity >= candidateSpecificity + .18;
	});
}
function resolveGraphSeedMatches(params) {
	const result = resolveEntityMention(params.store, params.ctx, buildEntityMention({
		ctx: params.ctx,
		scope: params.ctx.scopes[0] ?? "agent:unknown",
		rawText: params.name,
		semanticRole: "query",
		sourceRef: `query:${stableHash([params.objective.query, params.name])}`,
		supportText: params.objective.query,
		observedAt: params.objective.now,
		sessionKey: params.objective.currentSessionKey,
		metadataJson: { generatedFrom: "graph-seed-resolution" }
	}), {
		createIfMissing: false,
		persist: false
	});
	if (result.method === "uncertain") return {
		exact: null,
		expanded: result.candidateEntityIds.map((entityId) => params.store.graphRepo.getEntityById(entityId)).filter((entity) => Boolean(entity)).slice(0, params.limit)
	};
	return {
		exact: result.entity,
		expanded: []
	};
}
function rememberGraphSeed(seeds, entityId, score, reason, options) {
	const clampedScore = clamp01(score);
	if (clampedScore <= 0) return;
	const existing = seeds.get(entityId);
	if (existing) {
		existing.score = Math.max(existing.score, clampedScore);
		if (!existing.reasons.includes(reason)) existing.reasons.push(reason);
		existing.queryLexicalOnly = existing.queryLexicalOnly && (options?.queryLexicalOnly ?? false);
		existing.queryDerived = existing.queryDerived || (options?.queryDerived ?? false);
		existing.fromActiveContext = existing.fromActiveContext || (options?.fromActiveContext ?? false);
		existing.specificity = Math.max(existing.specificity, options?.specificity ?? 0);
		existing.activeContextSupportCount += options?.fromActiveContext ? 1 : 0;
		existing.querySupportCount += options?.queryDerived ? 1 : 0;
		return;
	}
	seeds.set(entityId, {
		entityId,
		score: clampedScore,
		reasons: [reason],
		queryLexicalOnly: options?.queryLexicalOnly ?? false,
		queryDerived: options?.queryDerived ?? false,
		fromActiveContext: options?.fromActiveContext ?? false,
		specificity: options?.specificity ?? 0,
		activeContextSupportCount: options?.fromActiveContext ? 1 : 0,
		querySupportCount: options?.queryDerived ? 1 : 0
	});
}
function contextSeedBaseScore(object) {
	switch (object.kind) {
		case "task": return object.attributes.activeTask ? .96 : .84;
		case "state": return object.attributes.activeTask || object.attributes.syntheticWorkflow ? .88 : .74;
		case "fact": return object.attributes.relational ? .82 : object.attributes.guidance ? .68 : .6;
		case "event": return .56;
		default: return .48;
	}
}
function memoryObjectContentRefs(object) {
	switch (object.kind) {
		case "state": return [`state:${object.row.id}`];
		case "task": return [`task:${object.row.id}`];
		case "fact": return [`fact:${object.row.id}`];
		case "event": return [`event:${object.row.id}`];
		default: return [];
	}
}
function isWorkflowControlObject(object) {
	if (object.kind === "task") return true;
	if (object.kind !== "state") return false;
	const stateKey = object.attributes.stateKey ?? "";
	return object.attributes.syntheticWorkflow === true || stateKey.startsWith("workflow.");
}
function shouldSuppressTargetBridge(object, hasDurableGraph) {
	if (isWorkflowControlObject(object)) return true;
	if (object.kind === "event" && object.attributes.relational) return true;
	if (object.kind === "fact" && object.attributes.factSubject && object.attributes.factPredicate && object.attributes.factObject) return true;
	if (!hasDurableGraph) return false;
	if (object.kind === "task") return true;
	if (object.kind === "state") return (object.attributes.stateKey ?? "").startsWith("project.");
	return false;
}
function collectGraphHypothesisSeedNodeIds(params) {
	const supportRefs = new Set(params.supportObjects.flatMap((object) => memoryObjectContentRefs(object)));
	const names = /* @__PURE__ */ new Set();
	for (const object of params.supportObjects) {
		if (object.attributes.factSubject) names.add(normalizeText(object.attributes.factSubject));
		if (object.attributes.factObject) names.add(normalizeText(object.attributes.factObject));
	}
	const seeded = /* @__PURE__ */ new Set();
	for (const candidate of params.graphHypotheses) {
		const metadata = parseGraphHypothesisCandidate(candidate);
		if (!metadata) continue;
		if (candidate.supportContentRefs.some((ref) => supportRefs.has(ref))) {
			seeded.add(metadata.sourceNodeId);
			seeded.add(metadata.targetNodeId);
			continue;
		}
		if (names.has(normalizeText(metadata.sourceName))) seeded.add(metadata.sourceNodeId);
		if (names.has(normalizeText(metadata.targetName))) seeded.add(metadata.targetNodeId);
	}
	return [...seeded];
}
function collectGraphSeedEntityIds(params) {
	const seeds = /* @__PURE__ */ new Map();
	for (const object of params.supportObjects) {
		const baseScore = contextSeedBaseScore(object);
		if (baseScore <= 0) continue;
		const workflowSurface = object.kind === "task" || isWorkflowControlObject(object);
		const names = [object.attributes.factSubject, object.attributes.factObject].filter((name) => Boolean(name)).slice(0, 4);
		const objectMatches = /* @__PURE__ */ new Map();
		for (const name of names) {
			const resolved = resolveGraphSeedMatches({
				store: params.store,
				ctx: params.ctx,
				objective: params.objective,
				name,
				limit: 6
			});
			if (resolved.exact) {
				objectMatches.set(resolved.exact.entityId, {
					entity: resolved.exact,
					score: baseScore,
					reason: `context:${object.kind}:${name}`
				});
				continue;
			}
			for (const fuzzy of resolved.expanded) {
				const specificity = graphSeedSpecificity(fuzzy.canonicalName);
				if (workflowSurface && specificity < .58) continue;
				objectMatches.set(fuzzy.entityId, {
					entity: fuzzy,
					score: baseScore * (workflowSurface ? .9 : .82),
					reason: `context-expanded:${object.kind}:${name}`
				});
			}
		}
		const filteredMatches = filterSpecificGraphEntityMatches([...objectMatches.values()].map((entry) => entry.entity));
		const allowedIds = new Set(filteredMatches.map((entry) => entry.entityId));
		for (const { entity, score, reason } of objectMatches.values()) {
			if (!allowedIds.has(entity.entityId)) continue;
			rememberGraphSeed(seeds, entity.entityId, score, reason, {
				fromActiveContext: true,
				specificity: graphSeedSpecificity(entity.canonicalName)
			});
		}
	}
	const hasStrongContextSeeds = [...seeds.values()].some((entry) => entry.fromActiveContext && entry.score >= .78);
	const strongestContextSpecificity = [...seeds.values()].filter((entry) => entry.fromActiveContext && entry.score >= .78).reduce((max, entry) => Math.max(max, entry.specificity), 0);
	const contextSpecificityFloor = Math.max(.58, strongestContextSpecificity - .18);
	return [...seeds.values()].filter((entry) => {
		if (!hasStrongContextSeeds || !entry.queryDerived || entry.fromActiveContext) {
			if (!hasStrongContextSeeds || !entry.fromActiveContext) return true;
			return entry.specificity >= contextSpecificityFloor || entry.activeContextSupportCount >= 2;
		}
		if (entry.queryLexicalOnly && entry.score < .66) return false;
		return entry.specificity >= contextSpecificityFloor;
	}).sort((left, right) => right.score - left.score).slice(0, params.maxSeeds).map((entry) => entry.entityId);
}
function collectQueryGraphSeedEntityIds(params) {
	return [];
}
function expandGraphSupportObjects(params) {
	const expanded = new Map(params.supportObjects.map((object) => [object.objectId, object]));
	for (const object of params.supportObjects) {
		if (object.kind !== "fact" || !object.attributes.relational || !object.attributes.factSubject || !object.attributes.factPredicate) continue;
		const peers = params.store.factRepo.findBySemanticKey({
			agentId: params.ctx.agentId,
			scope: object.row.scope,
			canonicalSubject: object.attributes.factSubject,
			predicate: object.attributes.factPredicate,
			includeHistorical: true
		});
		for (const fact of peers) {
			if (expanded.has(fact.factId)) continue;
			const text = formatFactLine({
				subject: fact.canonicalSubject,
				predicate: fact.predicate,
				object: fact.canonicalObject,
				objectValueJson: fact.objectValueJson,
				status: fact.status
			});
			expanded.set(fact.factId, buildMemoryObject({
				kind: "fact",
				row: toEvidenceRow({
					id: fact.factId,
					text,
					score: .72,
					scope: fact.scope,
					confidence: fact.confidence,
					observedAt: fact.updatedAt,
					sourceRef: fact.sourceRef,
					lineage: {
						canonicalKind: "fact",
						canonicalId: fact.factId,
						sourceKind: "fact",
						sourceId: fact.factId,
						sourceRef: fact.sourceRef,
						materializedEpoch: fact.materializedEpoch
					}
				}),
				objective: params.objective,
				attrs: {
					relational: isRelationalFactPredicate(fact.predicate),
					guidance: Boolean(guidanceTextFromFact(fact)),
					entityCount: [fact.canonicalSubject, fact.canonicalObject].filter(Boolean).length,
					factSubject: fact.canonicalSubject,
					factPredicate: fact.predicate,
					factObject: fact.canonicalObject
				}
			}));
		}
	}
	if (![...expanded.values()].some((object) => object.kind === "fact" && object.attributes.relational) && params.objective.routeType === "explanatory") {
		const subjectNames = /* @__PURE__ */ new Set();
		for (const object of params.supportObjects) {
			if (object.attributes.factSubject) subjectNames.add(object.attributes.factSubject);
			if (object.attributes.factObject) subjectNames.add(object.attributes.factObject);
		}
		for (const name of [...subjectNames].filter(Boolean).slice(0, 4)) {
			const resolution = resolveEntityMention(params.store, params.ctx, buildEntityMention({
				ctx: params.ctx,
				scope: params.ctx.scopes[0] ?? "agent:unknown",
				rawText: name,
				semanticRole: "query",
				sourceRef: `query:${stableHash([
					params.objective.query,
					name,
					"relational-fact-expansion"
				])}`,
				supportText: params.objective.query,
				observedAt: params.objective.now,
				sessionKey: params.objective.currentSessionKey,
				metadataJson: { generatedFrom: "relational-fact-expansion" }
			}), {
				createIfMissing: false,
				persist: false
			});
			const canonicalSubject = resolution.method === "uncertain" ? "" : resolution.entity.canonicalName;
			if (!canonicalSubject) continue;
			for (const scope of params.ctx.scopes) for (const fact of params.store.factRepo.findActiveBySubject({
				agentId: params.ctx.agentId,
				scope,
				canonicalSubject
			})) {
				if (!isRelationalFactPredicate(fact.predicate) || expanded.has(fact.factId)) continue;
				const text = formatFactLine({
					subject: fact.canonicalSubject,
					predicate: fact.predicate,
					object: fact.canonicalObject,
					objectValueJson: fact.objectValueJson,
					status: fact.status
				});
				expanded.set(fact.factId, buildMemoryObject({
					kind: "fact",
					row: toEvidenceRow({
						id: fact.factId,
						text,
						score: .68,
						scope: fact.scope,
						confidence: fact.confidence,
						observedAt: fact.updatedAt,
						sourceRef: fact.sourceRef,
						lineage: {
							canonicalKind: "fact",
							canonicalId: fact.factId,
							sourceKind: "fact",
							sourceId: fact.factId,
							sourceRef: fact.sourceRef,
							materializedEpoch: fact.materializedEpoch
						}
					}),
					objective: params.objective,
					attrs: {
						relational: true,
						guidance: Boolean(guidanceTextFromFact(fact)),
						entityCount: [fact.canonicalSubject, fact.canonicalObject].filter(Boolean).length,
						factSubject: fact.canonicalSubject,
						factPredicate: fact.predicate,
						factObject: fact.canonicalObject
					}
				}));
			}
		}
	}
	return [...expanded.values()];
}
function ensureGraphNode(nodes, node) {
	const existing = nodes.get(node.nodeId);
	if (existing) return existing;
	nodes.set(node.nodeId, node);
	return node;
}
function ensureGraphEdge(edges, edge) {
	const existing = edges.get(edge.edgeId);
	if (existing) return existing;
	edges.set(edge.edgeId, edge);
	return edge;
}
function collectGraphLinkedEntities(params) {
	const linked = /* @__PURE__ */ new Map();
	const names = [...new Set([...params.explicitNames ?? []])].filter(Boolean).slice(0, 6);
	for (const name of names) {
		const resolved = resolveGraphSeedMatches({
			store: params.store,
			ctx: params.ctx,
			objective: params.objective,
			name,
			limit: 3
		});
		if (resolved.exact) {
			linked.set(resolved.exact.entityId, {
				nodeId: resolved.exact.entityId,
				nodeKind: "entity",
				entityId: resolved.exact.entityId,
				name: resolved.exact.canonicalName,
				type: resolved.exact.entityType,
				confidence: resolved.exact.confidence
			});
			continue;
		}
		for (const fuzzy of resolved.expanded) linked.set(fuzzy.entityId, {
			nodeId: fuzzy.entityId,
			nodeKind: "entity",
			entityId: fuzzy.entityId,
			name: fuzzy.canonicalName,
			type: fuzzy.entityType,
			confidence: fuzzy.confidence
		});
	}
	for (const node of linked.values()) ensureGraphNode(params.nodes, node);
	return [...linked.values()];
}
function memoryGraphNodeFromObject(object) {
	switch (object.kind) {
		case "task": return {
			nodeId: object.objectId,
			nodeKind: "task",
			name: truncateText(object.row.text, 160),
			type: object.attributes.activeTask ? "active_task" : "task",
			sourceObjectId: object.objectId,
			observedAt: object.row.observedAt,
			confidence: object.row.confidence
		};
		case "state": return {
			nodeId: object.objectId,
			nodeKind: "state",
			name: truncateText(object.row.text, 160),
			type: object.attributes.stateKey ?? "state",
			sourceObjectId: object.objectId,
			observedAt: object.row.observedAt,
			confidence: object.row.confidence
		};
		case "fact": return {
			nodeId: object.objectId,
			nodeKind: "fact",
			name: truncateText(object.row.text, 160),
			type: object.attributes.factPredicate ?? "fact",
			sourceObjectId: object.objectId,
			observedAt: object.row.observedAt,
			confidence: object.row.confidence
		};
		case "event": return {
			nodeId: object.objectId,
			nodeKind: "event",
			name: truncateText(object.row.text, 160),
			type: object.attributes.eventType ?? "event",
			sourceObjectId: object.objectId,
			observedAt: object.row.observedAt,
			confidence: object.row.confidence
		};
		default: return null;
	}
}
function augmentGraphWithSupportObjects(params) {
	const hasDurableGraph = params.graph.edges.length > 0 || params.graph.pathCandidates.length > 0;
	const nodes = new Map(params.graph.nodes.map((node) => [node.nodeId, node]));
	const edges = new Map(params.graph.edges.map((edge) => [edge.edgeId, edge]));
	const objectNodes = /* @__PURE__ */ new Map();
	const linkedEntityIdsByObject = /* @__PURE__ */ new Map();
	const taskNodesByTaskId = /* @__PURE__ */ new Map();
	for (const object of params.supportObjects) {
		const node = memoryGraphNodeFromObject(object);
		if (!node) continue;
		ensureGraphNode(nodes, node);
		objectNodes.set(object.objectId, node);
		if (object.kind === "task" && object.attributes.taskId) taskNodesByTaskId.set(object.attributes.taskId, node);
	}
	for (const candidate of params.graphHypotheses) {
		const metadata = parseGraphHypothesisCandidate(candidate);
		if (!metadata) continue;
		if ([...edges.values()].some((edge) => edge.sourceKind === "stored" && edge.relType === metadata.relationType && edge.relationSlot === metadata.relationSlot && edge.srcNodeId === metadata.sourceNodeId && edge.dstNodeId === metadata.targetNodeId)) continue;
		ensureGraphNode(nodes, {
			nodeId: metadata.sourceNodeId,
			nodeKind: "entity",
			name: metadata.sourceName,
			type: metadata.sourceType,
			sourceObjectId: `abstraction_candidate:${candidate.candidateId}`,
			observedAt: candidate.updatedAt,
			confidence: candidate.confidence
		});
		ensureGraphNode(nodes, {
			nodeId: metadata.targetNodeId,
			nodeKind: "entity",
			name: metadata.targetName,
			type: metadata.targetType,
			sourceObjectId: `abstraction_candidate:${candidate.candidateId}`,
			observedAt: candidate.updatedAt,
			confidence: candidate.confidence
		});
		const confidence = clamp01(candidate.confidence * (metadata.relationClass === "observed" ? .96 : .88) + (candidate.stage === "active" ? .04 : 0));
		ensureGraphEdge(edges, {
			edgeId: `graph_hypothesis:${candidate.candidateId}`,
			srcNodeId: metadata.sourceNodeId,
			dstNodeId: metadata.targetNodeId,
			srcEntityId: metadata.sourceNodeId,
			dstEntityId: metadata.targetNodeId,
			relType: metadata.relationType,
			relationSlot: metadata.relationSlot,
			confidence,
			evidenceRef: `abstraction_candidate:${candidate.candidateId}`,
			updatedAt: candidate.updatedAt,
			sourceKind: "synthesized"
		});
	}
	for (const object of params.supportObjects) {
		if (!objectNodes.get(object.objectId)) continue;
		if (shouldSuppressTargetBridge(object, hasDurableGraph)) continue;
		const entityNames = new Set([object.attributes.factSubject, object.attributes.factObject].filter((entry) => Boolean(entry)));
		if (object.attributes.factSubject) entityNames.add(object.attributes.factSubject);
		if (object.attributes.factObject) entityNames.add(object.attributes.factObject);
		const linked = collectGraphLinkedEntities({
			store: params.store,
			ctx: params.ctx,
			objective: params.objective,
			nodes,
			text: [...entityNames].join(" "),
			explicitNames: [...entityNames]
		});
		if (linked.length === 0) continue;
		const linkedIds = linkedEntityIdsByObject.get(object.objectId) ?? /* @__PURE__ */ new Set();
		for (const entity of linked) {
			linkedIds.add(entity.nodeId);
			ensureGraphEdge(edges, {
				edgeId: stableHash([
					"support-edge",
					object.objectId,
					"targets",
					entity.nodeId
				]),
				srcNodeId: object.objectId,
				dstNodeId: entity.nodeId,
				srcEntityId: object.objectId,
				dstEntityId: entity.entityId ?? entity.nodeId,
				relType: "targets",
				confidence: clamp01(.68 + Math.min(object.baseScore, .2)),
				evidenceRef: object.row.id,
				updatedAt: object.row.observedAt,
				sourceKind: "synthesized"
			});
		}
		linkedEntityIdsByObject.set(object.objectId, linkedIds);
	}
	for (const object of params.supportObjects) {
		if (object.kind !== "state") continue;
		const stateNode = objectNodes.get(object.objectId);
		if (!stateNode || !object.attributes.parentTaskId) continue;
		const taskNode = taskNodesByTaskId.get(object.attributes.parentTaskId);
		if (!taskNode) continue;
		ensureGraphEdge(edges, {
			edgeId: stableHash([
				"support-edge",
				taskNode.nodeId,
				"updates",
				stateNode.nodeId
			]),
			srcNodeId: taskNode.nodeId,
			dstNodeId: stateNode.nodeId,
			srcEntityId: taskNode.nodeId,
			dstEntityId: stateNode.nodeId,
			relType: "updates",
			confidence: .84,
			evidenceRef: object.row.id,
			updatedAt: object.row.observedAt,
			sourceKind: "synthesized"
		});
	}
	const factObjects = params.supportObjects.filter((object) => object.kind === "fact" && object.attributes.factSubject && object.attributes.factPredicate);
	for (let index = 0; index < factObjects.length; index += 1) {
		const left = factObjects[index];
		for (let offset = index + 1; offset < factObjects.length; offset += 1) {
			const right = factObjects[offset];
			if (left.attributes.factSubject !== right.attributes.factSubject || left.attributes.factPredicate !== right.attributes.factPredicate || !left.attributes.factObject || !right.attributes.factObject || left.attributes.factObject === right.attributes.factObject) continue;
			const [srcNodeId, dstNodeId] = [left.objectId, right.objectId].sort();
			ensureGraphEdge(edges, {
				edgeId: stableHash([
					"support-edge",
					srcNodeId,
					"contradicts",
					dstNodeId
				]),
				srcNodeId,
				dstNodeId,
				srcEntityId: srcNodeId,
				dstEntityId: dstNodeId,
				relType: "contradicts",
				confidence: Math.min(left.row.confidence ?? .78, right.row.confidence ?? .78),
				evidenceRef: `${left.row.id}|${right.row.id}`,
				updatedAt: left.row.observedAt ?? right.row.observedAt,
				sourceKind: "synthesized"
			});
		}
	}
	const edgeList = [...edges.values()].sort((left, right) => right.confidence - left.confidence || (right.updatedAt ? Date.parse(right.updatedAt) : 0) - (left.updatedAt ? Date.parse(left.updatedAt) : 0));
	const pathCandidates = buildGraphPathCandidates({
		seedNodeIds: params.seedEntityIds,
		prioritySeedNodeIds: params.prioritySeedEntityIds,
		nodes,
		edges: edgeList,
		now: params.objective.now,
		maxPaths: Math.max(8, Math.min(params.ctx.config.maxGraphEdges + 4, 16)),
		maxHops: Math.min(3, Math.max(2, params.ctx.config.graphMaxHops + 1))
	});
	return {
		nodes: [...nodes.values()],
		edges: edgeList,
		pathCandidates,
		paths: pathCandidates.map((path) => path.summary)
	};
}
function graphObjects(store, ctx, objective, supportObjects) {
	const querySeeds = collectQueryGraphSeedEntityIds({
		store,
		ctx,
		objective,
		maxSeeds: Math.max(2, Math.min(ctx.config.maxGraphNodes, 6))
	});
	const graphSupportObjects = expandGraphSupportObjects({
		store,
		ctx,
		objective,
		supportObjects
	});
	const storedSeeds = collectGraphSeedEntityIds({
		store,
		ctx,
		objective,
		supportObjects: graphSupportObjects,
		maxSeeds: Math.max(4, Math.min(ctx.config.maxGraphNodes, 8))
	});
	const graph = store.graphRepo.expandNeighborhood({
		agentId: ctx.agentId,
		scopes: ctx.scopes,
		seedEntityIds: storedSeeds,
		maxHops: Math.min(2, ctx.config.graphMaxHops),
		maxEdges: Math.min(8, ctx.config.maxGraphEdges),
		maxNodes: Math.min(8, ctx.config.maxGraphNodes),
		now: objective.now,
		readEpoch: ctx.readEpoch
	});
	const includeCandidateHypotheses = graphSupportObjects.some((object) => (object.kind === "event" || object.kind === "fact") && object.attributes.relational === true) || graph.edges.length === 0 && graph.nodes.length === 0 && graph.pathCandidates.length === 0;
	const graphHypotheses = store.abstractionRepo.listByAgent({
		agentId: ctx.agentId,
		scopes: ctx.scopes,
		abstractionTypes: ["graph_hypothesis"],
		stages: includeCandidateHypotheses ? [
			"active",
			"probationary",
			"candidate"
		] : ["active", "probationary"],
		limit: 12,
		readEpoch: ctx.readEpoch
	});
	const hypothesisSeeds = collectGraphHypothesisSeedNodeIds({
		objective,
		supportObjects: graphSupportObjects,
		graphHypotheses
	});
	const seedNodeIds = [...new Set([
		...storedSeeds,
		...querySeeds,
		...hypothesisSeeds
	])];
	const heterogeneousGraph = augmentGraphWithSupportObjects({
		store,
		ctx,
		objective,
		graph,
		supportObjects: graphSupportObjects,
		graphHypotheses,
		seedEntityIds: seedNodeIds,
		prioritySeedEntityIds: querySeeds.length > 0 ? querySeeds : seedNodeIds
	});
	return heterogeneousGraph.pathCandidates.map((pathCandidate) => buildMemoryObject({
		kind: "graph_path",
		row: toEvidenceRow({
			id: pathCandidate.pathId,
			text: pathCandidate.summary,
			score: pathCandidate.score,
			scope: ctx.scopes[0] ?? "unknown",
			confidence: pathCandidate.features.edgeConfidence,
			provenance: pathCandidate.reasons.join("; "),
			lineage: {
				sourceKind: "alternate",
				sourceId: pathCandidate.pathId
			}
		}),
		objective,
		attrs: {
			relational: true,
			entityCount: pathCandidate.nodeIds.length,
			graphNodeCount: pathCandidate.nodeIds.length,
			graphEdgeCount: pathCandidate.edgeIds.length
		},
		graphNodes: heterogeneousGraph.nodes.filter((node) => pathCandidate.nodeIds.includes(node.nodeId)),
		graphEdges: heterogeneousGraph.edges.filter((edge) => pathCandidate.edgeIds.includes(edge.edgeId)),
		graphPathCandidate: pathCandidate
	}));
}
function dedupeObjects(objects) {
	const seen = /* @__PURE__ */ new Set();
	const deduped = [];
	for (const object of objects) {
		const key = `${object.kind}:${object.row.id}:${object.row.text.trim().toLowerCase()}`;
		if (seen.has(key)) continue;
		seen.add(key);
		deduped.push(object);
	}
	return deduped;
}
function collectMemoryObjects(store, ctx, objective, hybridHits) {
	const filteredHits = hybridHits.filter((hit) => !shouldSuppressRecallText(hit.text));
	const states = stateObjects(store, ctx, objective);
	const tasks = taskObjects(store, ctx, objective, filteredHits);
	const facts = factObjects(store, ctx, objective);
	const events = eventObjects(store, ctx, objective);
	const graph = graphObjects(store, ctx, objective, [
		...states,
		...tasks,
		...facts,
		...events
	]);
	const chunks = chunkObjects(objective, filteredHits);
	const alternates = alternateObjects(objective, filteredHits);
	return attachBeliefsToObjects(store, ctx, dedupeObjects([
		...states,
		...tasks,
		...facts,
		...events,
		...graph,
		...chunks,
		...alternates
	]));
}
function objectiveScoreForObject(object, objective) {
	const profile = object.profile;
	let score = object.baseScore;
	const reasons = [];
	if (objective.routeType === "workflow") {
		score = clamp01(profile.workflowScore * .44 + profile.continuityScore * .24 + profile.stabilityScore * .14 + object.baseScore * .18);
		reasons.push(`workflow=${profile.workflowScore.toFixed(2)}`, `continuity=${profile.continuityScore.toFixed(2)}`);
	} else if (objective.routeType === "factual") {
		score = clamp01(profile.factualScore * .48 + profile.guidanceScore * .14 + profile.stabilityScore * .18 + object.baseScore * .2);
		reasons.push(`factual=${profile.factualScore.toFixed(2)}`, `guidance=${profile.guidanceScore.toFixed(2)}`);
	} else if (objective.routeType === "temporal") {
		score = clamp01(profile.temporalScore * .46 + profile.recencyScore * .24 + object.baseScore * .2 + profile.continuityScore * .1);
		reasons.push(`temporal=${profile.temporalScore.toFixed(2)}`, `recency=${profile.recencyScore.toFixed(2)}`);
	} else if (objective.routeType === "explanatory") {
		const relationalFactBridge = object.kind === "fact" && object.attributes.relational ? .18 : 0;
		score = clamp01(profile.explanationScore * .28 + profile.relationDensity * .24 + profile.connectivity * .16 + profile.factualScore * .22 + relationalFactBridge + object.baseScore * .1);
		reasons.push(`explanation=${profile.explanationScore.toFixed(2)}`, `relation=${profile.relationDensity.toFixed(2)}`, `connectivity=${profile.connectivity.toFixed(2)}`, `bridge=${relationalFactBridge.toFixed(2)}`);
	}
	return applyBeliefAdjustment(object, objective, score, reasons);
}
function graphRouteFitForPath(pathObject, routeType) {
	const candidate = pathObject.graphPathCandidate;
	const nodes = pathObject.graphNodes ?? [];
	const edges = pathObject.graphEdges ?? [];
	if (!candidate || nodes.length === 0 || edges.length === 0) return 0;
	const nodeKinds = new Set(nodes.map((node) => node.nodeKind));
	const relTypes = new Set(edges.map((edge) => edge.relType));
	const structuralRatio = edges.filter((edge) => GRAPH_STRUCTURAL_RELATION_TYPES.has(edge.relType)).length / Math.max(1, edges.length);
	const relationFit = candidate.features.relationFit;
	const heterogeneous = candidate.features.heterogeneousSupport;
	const recency = candidate.features.recency;
	switch (routeType) {
		case "workflow": return clamp01(heterogeneous * .22 + relationFit * .12 + structuralRatio * .14 + (nodeKinds.has("task") ? .2 : 0) + (nodeKinds.has("state") ? .18 : 0) + (nodeKinds.has("outcome") ? .14 : 0) + (relTypes.has("updates") ? .18 : 0) + (relTypes.has("resolved_by") || relTypes.has("derived_from") ? .14 : 0) + (relTypes.has("blocks") || relTypes.has("depends_on") ? .12 : 0));
		case "factual": return clamp01(heterogeneous * .14 + relationFit * .18 + structuralRatio * .14 + (nodeKinds.has("fact") ? .26 : 0) + (relTypes.has("contradicts") || relTypes.has("supersedes") ? .24 : 0) + (relTypes.has("depends_on") || relTypes.has("uses") || relTypes.has("part_of") || relTypes.has("owner_of") ? .18 : 0));
		case "temporal": return clamp01(recency * .22 + heterogeneous * .12 + structuralRatio * .12 + (nodeKinds.has("event") ? .3 : 0) + (relTypes.has("supported_by") ? .22 : 0) + (relTypes.has("resolved_by") || relTypes.has("updates") ? .1 : 0));
		case "explanatory": return clamp01(heterogeneous * .28 + relationFit * .22 + structuralRatio * .18 + (nodeKinds.has("fact") || nodeKinds.has("task") || nodeKinds.has("outcome") ? .12 : 0) + (relTypes.has("contradicts") || relTypes.has("resolved_by") || relTypes.has("caused_by") || relTypes.has("depends_on") ? .2 : 0));
	}
}
function buildGraphGuidance(candidates, objective) {
	const directSupportByObjectId = /* @__PURE__ */ new Map();
	const paths = [];
	let strongestSupport = 0;
	for (const entry of candidates) {
		if (entry.object.kind !== "graph_path") continue;
		const candidate = entry.object.graphPathCandidate;
		if (!candidate) continue;
		const routeFit = graphRouteFitForPath(entry.object, objective.routeType);
		const pathWeight = clamp01(entry.objectiveScore * .58 + routeFit * .26 + candidate.score * .16);
		if (pathWeight < .16) continue;
		strongestSupport = Math.max(strongestSupport, pathWeight);
		const directObjectIds = new Set((entry.object.graphNodes ?? []).map((node) => node.sourceObjectId).filter((nodeId) => typeof nodeId === "string" && Boolean(nodeId)));
		for (const objectId of directObjectIds) directSupportByObjectId.set(objectId, Math.max(directSupportByObjectId.get(objectId) ?? 0, pathWeight));
		paths.push({
			summary: entry.object.row.text,
			weight: pathWeight,
			directObjectIds,
			nodeNames: [...new Set((entry.object.graphNodes ?? []).map((node) => node.name.toLowerCase()))]
		});
	}
	paths.sort((left, right) => right.weight - left.weight);
	return {
		strongestSupport,
		directSupportByObjectId,
		paths: paths.slice(0, 6)
	};
}
function graphContextSupportForObject(object, guidance) {
	if (object.kind === "graph_path") return {
		direct: 0,
		contextual: 0
	};
	const direct = guidance.directSupportByObjectId.get(object.objectId) ?? 0;
	const objectText = object.row.text.toLowerCase();
	const objectEntities = /* @__PURE__ */ new Set();
	let contextual = 0;
	for (const path of guidance.paths) {
		if (path.directObjectIds.has(object.objectId)) continue;
		const matchedNodeCount = path.nodeNames.filter((name) => objectEntities.has(name) || objectText.includes(name)).length;
		const candidate = (matchedNodeCount === 0 ? 0 : clamp01(matchedNodeCount / Math.max(1, path.nodeNames.length))) * path.weight;
		contextual = Math.max(contextual, candidate);
	}
	return {
		direct,
		contextual
	};
}
function graphBoostWeight(routeType, kind) {
	switch (routeType) {
		case "workflow": return kind === "task" || kind === "state" ? .22 : kind === "fact" || kind === "event" ? .14 : kind === "chunk" ? .1 : .06;
		case "factual": return kind === "fact" ? .2 : kind === "event" || kind === "chunk" ? .1 : .08;
		case "temporal": return kind === "event" ? .22 : kind === "chunk" ? .18 : kind === "fact" || kind === "task" || kind === "state" ? .1 : .06;
		case "explanatory": return kind === "fact" ? .18 : kind === "task" || kind === "state" || kind === "event" ? .14 : kind === "chunk" ? .12 : .08;
	}
}
function graphPenaltyWeight(routeType, kind) {
	switch (routeType) {
		case "workflow": return kind === "task" || kind === "state" ? .02 : .05;
		case "factual": return kind === "fact" ? .03 : .04;
		case "temporal": return kind === "event" || kind === "chunk" ? .06 : .03;
		case "explanatory": return kind === "fact" || kind === "chunk" ? .05 : .03;
	}
}
function applyGraphGuidanceToCandidate(entry, objective, guidance) {
	if (entry.object.kind === "graph_path" || guidance.paths.length === 0) return entry;
	const { direct, contextual } = graphContextSupportForObject(entry.object, guidance);
	const totalSupport = clamp01(direct * .8 + contextual * .35);
	const boost = totalSupport * graphBoostWeight(objective.routeType, entry.object.kind);
	const penalty = guidance.strongestSupport >= .72 && totalSupport < .12 && entry.object.baseScore < .92 && entry.object.kind !== "alternate" ? graphPenaltyWeight(objective.routeType, entry.object.kind) : 0;
	if (boost <= 0 && penalty <= 0) return entry;
	return {
		...entry,
		objectiveScore: clamp01(entry.objectiveScore + boost - penalty),
		graphSupport: totalSupport,
		graphPenalty: penalty,
		reasons: [
			...entry.reasons,
			`graphDirect=${direct.toFixed(2)}`,
			`graphContext=${contextual.toFixed(2)}`,
			`graphBoost=${boost.toFixed(2)}`,
			`graphPenalty=${penalty.toFixed(2)}`
		]
	};
}
function scheduleMemoryObjects(objects, objective) {
	const candidates = objects.map((object) => {
		const scheduled = objectiveScoreForObject(object, objective);
		return {
			object,
			objectiveScore: scheduled.score,
			reasons: scheduled.reasons
		};
	});
	const guidance = buildGraphGuidance(candidates, objective);
	return candidates.map((entry) => applyGraphGuidanceToCandidate(entry, objective, guidance)).filter((entry) => entry.objectiveScore > .08).sort((left, right) => right.objectiveScore - left.objectiveScore);
}
function collectAndScheduleMemoryObjects(store, ctx, objective, hybridHits) {
	return scheduleMemoryObjects(collectMemoryObjects(store, ctx, objective, hybridHits), objective);
}
function selectionObjectiveForRoute(routeType, query, now, currentSessionKey) {
	return createMemorySelectionObjective(routeType, query, now, currentSessionKey);
}
function activeObjectiveBudgets(plan) {
	return Object.entries(plan.objectiveBudgets).filter(([, budget]) => budget.activated).sort((left, right) => {
		if (right[1].weight !== left[1].weight) return right[1].weight - left[1].weight;
		return right[1].rawScore - left[1].rawScore;
	});
}
function beliefBudgetPriority(entry) {
	const belief = entry.object.belief;
	if (!belief) return entry.objectiveScore;
	const stageBonus = belief.stage === "active" ? .12 : belief.stage === "probationary" && entry.objectiveScore >= .52 ? .06 : belief.stage === "candidate" ? .01 : belief.stage === "decaying" ? -.08 : belief.stage === "superseded" ? -.16 : -.3;
	return clamp01(entry.objectiveScore + stageBonus + belief.posteriorConfidence * .04 + belief.usefulnessScore * .03 - belief.contradictionScore * .06);
}
function buildBudgetAggregate(plan, scheduledByRoute) {
	const aggregates = /* @__PURE__ */ new Map();
	for (const [routeType, scheduled] of Object.entries(scheduledByRoute)) for (const entry of scheduled) {
		const existing = aggregates.get(entry.object.objectId);
		if (!existing) {
			aggregates.set(entry.object.objectId, {
				object: entry.object,
				routeScores: { [routeType]: entry.objectiveScore },
				routeReasons: { [routeType]: entry.reasons },
				weightedScore: 0,
				maxRouteScore: entry.objectiveScore
			});
			continue;
		}
		existing.routeScores[routeType] = entry.objectiveScore;
		existing.routeReasons[routeType] = entry.reasons;
		if (entry.objectiveScore > existing.maxRouteScore) {
			existing.object = entry.object;
			existing.maxRouteScore = entry.objectiveScore;
		}
	}
	for (const aggregate of aggregates.values()) {
		let weightedScore = 0;
		for (const [routeType, budget] of Object.entries(plan.objectiveBudgets)) {
			if (!budget.activated) continue;
			weightedScore += (aggregate.routeScores[routeType] ?? 0) * budget.weight;
		}
		aggregate.weightedScore = clamp01(weightedScore);
	}
	return aggregates;
}
function toBudgetedScheduledEntry(aggregate, selectionReason) {
	const routeReasonParts = Object.entries(aggregate.routeScores).sort((left, right) => (right[1] ?? 0) - (left[1] ?? 0)).map(([routeType, score]) => `${routeType}=${(score ?? 0).toFixed(2)}`);
	return {
		object: aggregate.object,
		objectiveScore: aggregate.weightedScore,
		reasons: [
			selectionReason,
			...aggregate.object.kind === "graph_path" && aggregate.object.row.provenance ? [aggregate.object.row.provenance] : [],
			...routeReasonParts
		]
	};
}
function strongestRouteForAggregate(aggregate) {
	return Object.entries(aggregate.routeScores).sort((left, right) => (right[1] ?? 0) - (left[1] ?? 0))[0]?.[0];
}
function toSelectionTraceEntry(aggregate, selectionReason) {
	return {
		objectId: aggregate.object.objectId,
		kind: aggregate.object.kind,
		text: aggregate.object.row.text,
		weightedScore: aggregate.weightedScore,
		maxRouteScore: aggregate.maxRouteScore,
		selectionReason: aggregate.object.kind === "graph_path" && aggregate.object.row.provenance ? `${selectionReason}; ${aggregate.object.row.provenance}` : selectionReason,
		strongestRoute: strongestRouteForAggregate(aggregate),
		routeScores: aggregate.routeScores
	};
}
function reserveEligibleForRoute(aggregate, routeType) {
	if (routeType !== "factual") return true;
	const object = aggregate.object;
	if (object.kind === "task") return false;
	if (object.kind === "state" && (!object.attributes.stateKey || !isSnapshotFactualStateKey(object.attributes.stateKey))) return false;
	if (object.kind === "fact" && object.attributes.guidance && !object.attributes.relational) return false;
	return true;
}
function collectAndScheduleMemoryObjectsWithBudget(store, ctx, plan, hybridHits) {
	const objectiveEntries = activeObjectiveBudgets(plan);
	if (objectiveEntries.length === 0) return {
		scheduled: [],
		trace: {
			candidateCountsByRoute: {
				workflow: 0,
				factual: 0,
				temporal: 0,
				explanatory: 0
			},
			reserveSelections: {
				workflow: [],
				factual: [],
				temporal: [],
				explanatory: []
			},
			overflowSelections: [],
			droppedHighScore: []
		}
	};
	const scheduledByRoute = {
		workflow: [],
		factual: [],
		temporal: [],
		explanatory: []
	};
	const candidateCountsByRoute = {
		workflow: 0,
		factual: 0,
		temporal: 0,
		explanatory: 0
	};
	const reserveSelections = {
		workflow: [],
		factual: [],
		temporal: [],
		explanatory: []
	};
	const overflowSelections = [];
	for (const [routeType, budget] of objectiveEntries) {
		const scheduled = collectAndScheduleMemoryObjects(store, ctx, selectionObjectiveForRoute(routeType, plan.focusedQueries[routeType], ctx.now, ctx.sessionKey), hybridHits);
		const scanLimit = Math.max(8, budget.objectBudget + plan.globalOverflowObjects + 4);
		scheduledByRoute[routeType] = scheduled.slice().sort((left, right) => {
			const rightPriority = beliefBudgetPriority(right);
			const leftPriority = beliefBudgetPriority(left);
			if (rightPriority !== leftPriority) return rightPriority - leftPriority;
			return right.objectiveScore - left.objectiveScore;
		}).slice(0, scanLimit);
		candidateCountsByRoute[routeType] = scheduledByRoute[routeType].length;
	}
	const aggregates = buildBudgetAggregate(plan, scheduledByRoute);
	const selectedIds = /* @__PURE__ */ new Set();
	const selected = [];
	for (const [routeType, budget] of objectiveEntries) {
		let remaining = budget.objectBudget;
		if (remaining <= 0) continue;
		for (const entry of scheduledByRoute[routeType]) {
			if (remaining <= 0) break;
			const objectId = entry.object.objectId;
			if (selectedIds.has(objectId)) continue;
			const aggregate = aggregates.get(objectId);
			if (!aggregate) continue;
			if (!reserveEligibleForRoute(aggregate, routeType)) continue;
			selectedIds.add(objectId);
			selected.push(toBudgetedScheduledEntry(aggregate, `reserve:${routeType}`));
			reserveSelections[routeType].push(toSelectionTraceEntry(aggregate, `reserve:${routeType}`));
			remaining -= 1;
		}
	}
	const overflowBudget = Math.max(0, plan.totalObjectBudget - selected.length);
	let overflowCandidates = [];
	if (overflowBudget > 0) {
		overflowCandidates = [...aggregates.values()].filter((aggregate) => !selectedIds.has(aggregate.object.objectId)).filter((aggregate) => plan.routeDecision.routeType === "factual" ? reserveEligibleForRoute(aggregate, "factual") : true).sort((left, right) => {
			const rightPriority = beliefBudgetPriority({
				object: right.object,
				objectiveScore: right.weightedScore
			});
			const leftPriority = beliefBudgetPriority({
				object: left.object,
				objectiveScore: left.weightedScore
			});
			if (rightPriority !== leftPriority) return rightPriority - leftPriority;
			if (right.weightedScore !== left.weightedScore) return right.weightedScore - left.weightedScore;
			return right.maxRouteScore - left.maxRouteScore;
		});
		for (const aggregate of overflowCandidates.slice(0, overflowBudget)) {
			selectedIds.add(aggregate.object.objectId);
			selected.push(toBudgetedScheduledEntry(aggregate, "overflow"));
			overflowSelections.push(toSelectionTraceEntry(aggregate, "overflow"));
		}
	}
	const selectedCount = new Set(selected.map((entry) => entry.object.objectId));
	const droppedHighScore = [...aggregates.values()].filter((aggregate) => !selectedCount.has(aggregate.object.objectId)).sort((left, right) => {
		const rightPriority = beliefBudgetPriority({
			object: right.object,
			objectiveScore: right.weightedScore
		});
		const leftPriority = beliefBudgetPriority({
			object: left.object,
			objectiveScore: left.weightedScore
		});
		if (rightPriority !== leftPriority) return rightPriority - leftPriority;
		if (right.weightedScore !== left.weightedScore) return right.weightedScore - left.weightedScore;
		return right.maxRouteScore - left.maxRouteScore;
	}).slice(0, Math.max(4, Math.min(8, overflowCandidates.length || 4))).map((aggregate) => toSelectionTraceEntry(aggregate, "dropped:budget-limit"));
	return {
		scheduled: selected.sort((left, right) => right.objectiveScore - left.objectiveScore),
		trace: {
			candidateCountsByRoute,
			reserveSelections,
			overflowSelections,
			droppedHighScore
		}
	};
}
function topScheduledMemoryScore(scheduled, predicate) {
	let best = 0;
	for (const entry of scheduled) {
		if (predicate && !predicate(entry)) continue;
		best = Math.max(best, entry.objectiveScore);
	}
	return best;
}
function probeSupportForEntry(entry, routeType) {
	const profile = entry.object.profile;
	const graphSupport = entry.graphSupport ?? 0;
	const graphPenalty = entry.graphPenalty ?? 0;
	switch (routeType) {
		case "workflow": {
			const kindBias = entry.object.kind === "task" ? .1 : entry.object.kind === "state" ? .08 : .02;
			return clamp01(entry.objectiveScore * .58 + graphSupport * .12 - graphPenalty * .08 + profile.workflowScore * .14 + profile.continuityScore * .14 + profile.stabilityScore * .06 + kindBias);
		}
		case "factual": {
			const kindBias = entry.object.kind === "fact" ? .08 : .02;
			return clamp01(entry.objectiveScore * .62 + graphSupport * .12 - graphPenalty * .08 + profile.factualScore * .18 + profile.guidanceScore * .06 + profile.stabilityScore * .06 + kindBias);
		}
		case "temporal": {
			const kindBias = entry.object.kind === "event" ? .1 : entry.object.kind === "chunk" ? .08 : .02;
			return clamp01(entry.objectiveScore * .62 + graphSupport * .16 - graphPenalty * .08 + profile.temporalScore * .16 + profile.recencyScore * .1 + profile.continuityScore * .04 + kindBias);
		}
		case "explanatory": {
			const relationBridge = entry.object.kind === "graph_path" ? .12 : entry.object.kind === "fact" && entry.object.attributes.relational ? .1 : .02;
			return clamp01(entry.objectiveScore * .44 + graphSupport * .14 - graphPenalty * .08 + profile.explanationScore * .18 + profile.relationDensity * .16 + profile.connectivity * .12 + profile.factualScore * .08 + relationBridge);
		}
	}
}
function topProbeSupportForRoute(scheduled, routeType) {
	let topEntry;
	let support = 0;
	for (const entry of scheduled) {
		const candidateSupport = probeSupportForEntry(entry, routeType);
		if (candidateSupport > support || candidateSupport === support && topEntry && entry.objectiveScore > topEntry.objectiveScore) {
			topEntry = entry;
			support = candidateSupport;
		}
	}
	return {
		support,
		topEntry
	};
}
function routeHintFromMemoryObjectKind(kind) {
	switch (kind) {
		case "state":
		case "task": return "workflow";
		case "fact": return "factual";
		case "event":
		case "chunk": return "temporal";
		case "graph_path": return "explanatory";
		default: return;
	}
}
function toRouteEvidenceCandidatesFromObjects(scheduled, limit) {
	const selected = /* @__PURE__ */ new Set();
	const result = [];
	for (const entry of scheduled) {
		const key = `${entry.object.kind}:${entry.object.row.text.trim().toLowerCase()}`;
		if (selected.has(key)) continue;
		selected.add(key);
		result.push({
			index: result.length + 1,
			summary: entry.object.row.text,
			role: entry.object.routeRole,
			score: clamp01(entry.objectiveScore),
			confidence: entry.object.row.confidence,
			observedAt: entry.object.row.observedAt
		});
		if (result.length >= limit) break;
	}
	return result;
}
//#endregion
export { buildBackgroundRecallBundle, collectAndScheduleMemoryObjectsWithBudget, collectBehavioralGuidance, collectMemoryObjects, queryMemoryFacts, routeHintFromMemoryObjectKind, scheduleMemoryObjects, toRouteEvidenceCandidatesFromObjects, topProbeSupportForRoute, topScheduledMemoryScore };
