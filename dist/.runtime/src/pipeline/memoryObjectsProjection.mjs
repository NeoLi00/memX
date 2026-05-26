import { stableHash, truncateText } from "../support.mjs";
import { filterBootstrapRows } from "./bootstrapFilter.mjs";
import { dedupeEvidenceRows, splitLabelValue, toEvidenceRow } from "./memoryObjectsHelpers.mjs";
import { shouldProjectActiveProjectAlias } from "./authority.mjs";
//#region src/pipeline/memoryObjectsProjection.ts
function stateValueFromRow(row) {
	return splitLabelValue(row.text).value || row.text;
}
function projectionBlock(role, title, lines, sourceIds) {
	const deduped = dedupeEvidenceRows(lines.map((line, index) => toEvidenceRow({
		id: `${role}:${index}`,
		text: line,
		score: 1,
		scope: "projection"
	})), 4).map((row) => row.text);
	if (deduped.length === 0) return null;
	return {
		blockId: stableHash([
			role,
			...sourceIds,
			...deduped
		]),
		role,
		title,
		lines: deduped,
		sourceIds
	};
}
function projectUserStyleBlock(behavioralGuidance) {
	return projectionBlock("user_style", "Reply Guidance", behavioralGuidance.slice(0, 3), behavioralGuidance.slice(0, 3));
}
function projectActiveTaskBlock(params) {
	const lines = [];
	const sourceIds = [];
	const currentTask = params.states.filter((row) => row.id.endsWith("workflow.current_task"));
	const nextAction = params.states.filter((row) => row.id.endsWith("workflow.next_action"));
	const activeProject = params.states.filter((row) => row.id.endsWith("project.active_project"));
	const currentConsideration = params.states.filter((row) => row.id.endsWith("workflow.current_consideration"));
	for (const row of currentTask.slice(0, 1)) {
		lines.push(`Active task: ${truncateText(stateValueFromRow(row), 180)}`);
		sourceIds.push(row.id);
	}
	if (shouldProjectActiveProjectAlias()) for (const row of activeProject.slice(0, 1)) {
		lines.push(`Active project: ${truncateText(stateValueFromRow(row), 180)}`);
		sourceIds.push(row.id);
	}
	for (const row of nextAction.slice(0, 1)) {
		lines.push(`Next action: ${truncateText(stateValueFromRow(row), 180)}`);
		sourceIds.push(row.id);
	}
	for (const row of currentConsideration.slice(0, 1)) {
		lines.push(`Current consideration: ${truncateText(stateValueFromRow(row), 180)}`);
		sourceIds.push(row.id);
	}
	return projectionBlock("active_task", "Current Working Context", lines, sourceIds);
}
function buildWorkingProjectionBlocks(params) {
	return [projectUserStyleBlock(params.behavioralGuidance), projectActiveTaskBlock({ states: params.states })].filter((block) => Boolean(block));
}
function shouldProjectBeliefBoundEntry(entry, routeType, routeConfidence, allowHistoricalFacts) {
	const belief = entry.object.belief;
	if (!belief) return true;
	switch (belief.stage) {
		case "active": return true;
		case "probationary":
			if (routeType === "temporal" && entry.object.kind === "event" && entry.objectiveScore >= .56) return true;
			return routeConfidence >= .62 || entry.objectiveScore >= .66;
		case "candidate":
			if (routeType === "temporal" && entry.object.kind === "event" && entry.objectiveScore >= .64) return true;
			return entry.objectiveScore >= .78;
		case "decaying":
			if (routeType === "temporal" && entry.object.kind === "event" && entry.objectiveScore >= .68) return true;
			return routeConfidence >= .78 && entry.objectiveScore >= .74;
		case "superseded": return allowHistoricalFacts && routeType === "factual" && entry.object.kind === "fact" && entry.objectiveScore >= .58;
		case "quarantined": return false;
	}
}
function temporalEventPriority(row) {
	const sourceRef = row.sourceRef ?? "";
	const text = row.text.toLowerCase();
	let score = 0;
	if (sourceRef.startsWith("user:") || sourceRef.startsWith("tool:")) score += 3;
	else if (sourceRef.startsWith("assistant:")) score -= 1;
	if (/(?:失败|成功|修复|解决|超时|error|failed|success|fixed|resolved|timeout|deployed|rollback)/u.test(text)) score += 2;
	if (/\bgeneric llm response\b/i.test(text)) score -= 3;
	return score;
}
function prioritizeTemporalEventRows(rows) {
	return [...rows].sort((left, right) => {
		const priorityDelta = temporalEventPriority(right) - temporalEventPriority(left);
		if (priorityDelta !== 0) return priorityDelta;
		return (left.observedAt ?? "").localeCompare(right.observedAt ?? "");
	});
}
function workflowStatePriority(row) {
	const { label } = splitLabelValue(row.text);
	switch (label.trim().toLowerCase()) {
		case "workflow.current_task": return 6;
		case "workflow.blocker": return 5;
		case "workflow.current_consideration": return 4.5;
		case "workflow.next_action": return 4;
		case "project.active_project": return shouldProjectActiveProjectAlias() ? 3 : 0;
		case "workflow.task_phase": return 2;
		case "workflow.candidate_resolution": return 1;
		default: return 0;
	}
}
function prioritizeWorkflowStateRows(rows) {
	return [...rows].sort((left, right) => {
		const priorityDelta = workflowStatePriority(right) - workflowStatePriority(left);
		if (priorityDelta !== 0) return priorityDelta;
		if (right.score !== left.score) return right.score - left.score;
		return (right.observedAt ?? "").localeCompare(left.observedAt ?? "");
	});
}
function projectScheduledMemoryObjects(scheduled, options) {
	const states = [];
	const tasks = [];
	const facts = [];
	const events = [];
	const alternates = [];
	const graphNodes = /* @__PURE__ */ new Map();
	const graphEdges = /* @__PURE__ */ new Map();
	const graphPathCandidates = /* @__PURE__ */ new Map();
	const graphPaths = [];
	const recalledChunkIds = [];
	const recalledChunkTexts = [];
	const seenRows = /* @__PURE__ */ new Set();
	for (const entry of scheduled) {
		if (!shouldProjectBeliefBoundEntry(entry, options.routeType, options.routeConfidence, options.allowHistoricalFacts)) continue;
		const row = entry.object.row;
		const rowKey = `${entry.object.kind}:${row.text.trim().toLowerCase()}`;
		if (seenRows.has(rowKey)) continue;
		if (entry.object.kind === "state" && states.length < options.stateLimit) {
			seenRows.add(rowKey);
			states.push(row);
			continue;
		}
		if (entry.object.kind === "task" && tasks.length < options.taskLimit) {
			seenRows.add(rowKey);
			tasks.push(row);
			continue;
		}
		if (entry.object.kind === "fact" && facts.length < options.factLimit) {
			seenRows.add(rowKey);
			facts.push(row);
			continue;
		}
		if ((entry.object.kind === "event" || entry.object.kind === "chunk") && events.length < options.eventLimit) {
			if (entry.object.kind === "chunk" && row.provenance === "assistant") continue;
			seenRows.add(rowKey);
			events.push(row);
			if (entry.object.kind === "chunk" && recalledChunkIds.length < options.recallChunkBudget && !recalledChunkIds.includes(row.id)) {
				recalledChunkIds.push(row.id);
				recalledChunkTexts.push(row.text);
			}
			continue;
		}
		if (entry.object.kind === "graph_path" && graphPaths.length < options.graphLimit) {
			seenRows.add(rowKey);
			graphPaths.push(row.text);
			if (entry.object.graphPathCandidate) graphPathCandidates.set(entry.object.graphPathCandidate.pathId, entry.object.graphPathCandidate);
			for (const node of entry.object.graphNodes ?? []) graphNodes.set(node.nodeId, node);
			for (const edge of entry.object.graphEdges ?? []) graphEdges.set(edge.edgeId, edge);
			continue;
		}
		if (entry.object.kind === "alternate" && alternates.length < options.alternateLimit) {
			seenRows.add(rowKey);
			alternates.push(row);
		}
	}
	return {
		states: dedupeEvidenceRows(filterBootstrapRows(options.routeType === "workflow" ? prioritizeWorkflowStateRows(states) : states), options.stateLimit),
		tasks: dedupeEvidenceRows(filterBootstrapRows(tasks), options.taskLimit),
		facts: dedupeEvidenceRows(filterBootstrapRows(facts), options.factLimit),
		events: dedupeEvidenceRows(filterBootstrapRows(options.preferTemporalEvents ? prioritizeTemporalEventRows(events) : events), options.eventLimit),
		graph: {
			nodes: [...graphNodes.values()].slice(0, options.graphLimit),
			edges: [...graphEdges.values()].slice(0, options.graphLimit),
			pathCandidates: [...graphPathCandidates.values()].slice(0, options.graphLimit),
			paths: graphPaths.slice(0, options.graphLimit)
		},
		alternates: dedupeEvidenceRows(filterBootstrapRows(alternates), options.alternateLimit),
		recalledChunkIds,
		recalledChunkTexts
	};
}
function createMemorySelectionObjective(routeType, query, now, currentSessionKey) {
	return {
		routeType,
		query,
		now,
		includeHistorical: false,
		broadTemporal: false,
		currentSessionKey
	};
}
//#endregion
export { buildWorkingProjectionBlocks, createMemorySelectionObjective, projectScheduledMemoryObjects };
