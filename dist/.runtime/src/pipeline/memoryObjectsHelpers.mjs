import { canonicalStateKey } from "./semantic/heuristics.mjs";
import { containsLikelySecret, looksLikePromptInjection } from "../security/injection.mjs";
import { isBootstrapMemoryContamination } from "./bootstrapFilter.mjs";
import "./semantics.mjs";
//#region src/pipeline/memoryObjectsHelpers.ts
function splitLabelValue(text) {
	const separator = text.indexOf(":");
	if (separator < 0) return {
		label: text.trim(),
		value: ""
	};
	return {
		label: text.slice(0, separator).trim(),
		value: text.slice(separator + 1).trim()
	};
}
function toEvidenceRow(params) {
	return {
		id: params.id,
		text: params.text,
		score: params.score,
		scope: params.scope,
		confidence: params.confidence,
		sourceRef: params.sourceRef,
		observedAt: params.observedAt,
		provenance: params.provenance,
		lineage: params.lineage
	};
}
function asLineageSourceKind(value) {
	switch (value) {
		case "chunk":
		case "task":
		case "state":
		case "fact":
		case "event":
		case "entity_alias":
		case "graph_edge":
		case "vector_doc":
		case "alternate": return value;
		default: return;
	}
}
function asCanonicalKind(value) {
	switch (value) {
		case "state":
		case "fact":
		case "event":
		case "entity":
		case "graph_edge":
		case "task":
		case "chunk": return value;
		default: return;
	}
}
function normalizeLineageRef(value) {
	if (!value || typeof value !== "object" || Array.isArray(value)) return;
	const record = value;
	const sourceKind = asLineageSourceKind(record.sourceKind);
	const sourceId = typeof record.sourceId === "string" && record.sourceId.trim() ? record.sourceId.trim() : void 0;
	if (!sourceKind || !sourceId) return;
	return {
		sourceKind,
		sourceId,
		...typeof record.sourceRef === "string" && record.sourceRef.trim() ? { sourceRef: record.sourceRef.trim() } : {},
		...asCanonicalKind(record.canonicalKind) && typeof record.canonicalId === "string" && record.canonicalId.trim() ? {
			canonicalKind: asCanonicalKind(record.canonicalKind),
			canonicalId: record.canonicalId.trim()
		} : {},
		...typeof record.materializedEpoch === "number" && Number.isFinite(record.materializedEpoch) ? { materializedEpoch: Math.trunc(record.materializedEpoch) } : {}
	};
}
function lineageFromMetadata(metadata, fallback) {
	const direct = normalizeLineageRef(metadata?.lineage);
	if (direct) return direct;
	const sourceKind = asLineageSourceKind(metadata?.sourceKind) ?? fallback?.sourceKind;
	const sourceId = (typeof metadata?.sourceId === "string" && metadata.sourceId.trim() ? metadata.sourceId.trim() : void 0) ?? fallback?.sourceId;
	if (!sourceKind || !sourceId) return;
	return {
		sourceKind,
		sourceId,
		...typeof metadata?.sourceRef === "string" && metadata.sourceRef.trim() ? { sourceRef: metadata.sourceRef.trim() } : fallback?.sourceRef ? { sourceRef: fallback.sourceRef } : {},
		...asCanonicalKind(metadata?.canonicalKind) && typeof metadata?.canonicalId === "string" && metadata.canonicalId.trim() ? {
			canonicalKind: asCanonicalKind(metadata?.canonicalKind),
			canonicalId: metadata.canonicalId.trim()
		} : fallback?.canonicalKind && fallback?.canonicalId ? {
			canonicalKind: fallback.canonicalKind,
			canonicalId: fallback.canonicalId
		} : {},
		...typeof metadata?.materializedEpoch === "number" && Number.isFinite(metadata.materializedEpoch) ? { materializedEpoch: Math.trunc(metadata.materializedEpoch) } : typeof fallback?.materializedEpoch === "number" ? { materializedEpoch: fallback.materializedEpoch } : {}
	};
}
function lineageMetadata(lineage) {
	return {
		sourceKind: lineage.sourceKind,
		sourceId: lineage.sourceId,
		...lineage.sourceRef ? { sourceRef: lineage.sourceRef } : {},
		...lineage.canonicalKind && lineage.canonicalId ? {
			canonicalKind: lineage.canonicalKind,
			canonicalId: lineage.canonicalId
		} : {},
		...typeof lineage.materializedEpoch === "number" ? { materializedEpoch: lineage.materializedEpoch } : {},
		lineage
	};
}
function dedupeEvidenceRows(rows, limit) {
	const seen = /* @__PURE__ */ new Set();
	const result = [];
	for (const row of rows) {
		const key = row.text.trim().toLowerCase() || row.id;
		if (seen.has(key)) continue;
		seen.add(key);
		result.push(row);
		if (result.length >= limit) break;
	}
	return result;
}
function normalizeSearchText(text) {
	return text.replace(/^(user|assistant|tool):\s*/i, "").trim();
}
function shouldSuppressRecallText(text) {
	const normalized = normalizeSearchText(text);
	return isBootstrapMemoryContamination(normalized) || looksLikePromptInjection(normalized) || containsLikelySecret(normalized);
}
function rowsFromSearchHits(hits) {
	return hits.map((hit) => toEvidenceRow({
		id: hit.docId,
		text: normalizeSearchText(hit.text),
		score: hit.score,
		scope: typeof hit.metadata.scope === "string" ? hit.metadata.scope : "unknown",
		confidence: typeof hit.metadata.confidence === "number" ? hit.metadata.confidence : void 0,
		observedAt: typeof hit.metadata.observedAt === "string" ? hit.metadata.observedAt : void 0,
		provenance: typeof hit.metadata.role === "string" ? hit.metadata.role : typeof hit.metadata.sourceKind === "string" ? hit.metadata.sourceKind : void 0,
		lineage: lineageFromMetadata(hit.metadata, {
			sourceKind: "vector_doc",
			sourceId: hit.docId
		})
	}));
}
function humanizePredicate(predicate) {
	if (predicate.startsWith("prefers_")) return "prefers";
	return predicate.replaceAll("_", " ");
}
function humanizeFactObject(predicate, object) {
	if (!object) return "";
	if (predicate === "prefers_output_order") return object.replace(/\b(zh|en) first(?:,)? (zh|en) second\b/i, "$1 first, $2 second");
	return object;
}
function describeStateValue(key, valueJson) {
	const detailValue = (value) => {
		if (typeof value === "string") return value.trim();
		if (typeof value === "number" || typeof value === "boolean") return String(value);
		if (Array.isArray(value)) return value.map((entry) => detailValue(entry)).filter(Boolean).join(", ");
		if (value && typeof value === "object") return JSON.stringify(value);
		return "";
	};
	const formatStateDetailEntries = (omittedKeys, preferredKeys, extrasOnly = false) => {
		const omitted = new Set(omittedKeys);
		const seen = /* @__PURE__ */ new Set();
		const details = [];
		const pushKey = (entryKey) => {
			if (seen.has(entryKey) || omitted.has(entryKey)) return;
			seen.add(entryKey);
			const rendered = detailValue(valueJson[entryKey]);
			if (!rendered) return;
			details.push(`${entryKey}=${rendered}`);
		};
		for (const entryKey of preferredKeys) pushKey(entryKey);
		for (const entryKey of Object.keys(valueJson).sort()) pushKey(entryKey);
		if (!extrasOnly && details.length === 0) {
			const fallback = detailValue(valueJson);
			return fallback ? [fallback] : [];
		}
		return details;
	};
	if (key.startsWith("project.") && canonicalStateKey(key) !== "project.active_project") {
		const projectCode = typeof valueJson.projectCode === "string" && valueJson.projectCode.trim() ? valueJson.projectCode.trim() : key.slice(8).trim();
		const version = typeof valueJson.version === "string" ? valueJson.version.trim() : "";
		const launchDate = typeof valueJson.launchDate === "string" ? valueJson.launchDate.trim() : "";
		const internalTrialDate = typeof valueJson.internalTrialDate === "string" ? valueJson.internalTrialDate.trim() : "";
		const historicalAliases = Array.isArray(valueJson.historicalAliases) ? valueJson.historicalAliases.filter((entry) => typeof entry === "string" && entry.trim().length > 0).join(", ") : typeof valueJson.historicalAliases === "string" ? valueJson.historicalAliases.trim() : "";
		const components = valueJson.components && typeof valueJson.components === "object" && !Array.isArray(valueJson.components) ? Object.entries(valueJson.components).map(([slot, value]) => typeof value === "string" && value.trim() ? `${slot}=${value.trim()}` : "").filter(Boolean).join(", ") : "";
		return [
			projectCode,
			version ? `v${version}` : "",
			launchDate ? `launch ${launchDate}` : "",
			internalTrialDate ? `internal ${internalTrialDate}` : "",
			historicalAliases ? `historical aliases=${historicalAliases}` : "",
			components,
			...formatStateDetailEntries([
				"projectCode",
				"version",
				"launchDate",
				"internalTrialDate",
				"historicalAliases",
				"components"
			], [
				"status",
				"note",
				"decision"
			], true)
		].filter(Boolean).join(" | ");
	}
	const project = typeof valueJson.project === "string" ? valueJson.project : void 0;
	const task = typeof valueJson.task === "string" ? valueJson.task : void 0;
	const step = typeof valueJson.step === "string" ? valueJson.step : void 0;
	const blocker = typeof valueJson.blocker === "string" ? valueJson.blocker : void 0;
	const genericValue = typeof valueJson.value === "string" ? valueJson.value : void 0;
	switch (canonicalStateKey(key)) {
		case "project.active_project": return [project ?? genericValue ?? "", ...formatStateDetailEntries(["project", "value"], ["status", "note"], true)].filter(Boolean).join(" | ");
		case "workflow.current_task": return [task ?? genericValue ?? "", ...formatStateDetailEntries(["task", "value"], [
			"status",
			"note",
			"decision",
			"topic",
			"option"
		], true)].filter(Boolean).join(" | ");
		case "workflow.next_action": return [step ?? genericValue ?? "", ...formatStateDetailEntries(["step", "value"], [
			"status",
			"note",
			"reason"
		], true)].filter(Boolean).join(" | ");
		case "workflow.blocker": return [blocker ?? genericValue ?? "", ...formatStateDetailEntries(["blocker", "value"], [
			"status",
			"note",
			"resolution"
		], true)].filter(Boolean).join(" | ");
		default: return formatStateDetailEntries(["value"], [
			"topic",
			"option",
			"decision",
			"status",
			"note"
		]).join(" | ");
	}
}
function formatFactLine(params) {
	const value = params.object ? humanizeFactObject(params.predicate, params.object) : JSON.stringify(params.objectValueJson ?? {});
	return `${params.status === "superseded" ? "[previous] " : params.status === "uncertain" ? "[uncertain] " : ""}${params.subject} ${humanizePredicate(params.predicate)} ${value}`.trim();
}
//#endregion
export { dedupeEvidenceRows, describeStateValue, formatFactLine, lineageFromMetadata, lineageMetadata, normalizeSearchText, rowsFromSearchHits, shouldSuppressRecallText, splitLabelValue, toEvidenceRow };
