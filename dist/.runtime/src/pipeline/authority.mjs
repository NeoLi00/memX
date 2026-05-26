import { isProjectProfileStateKey, looksLikeProjectDescriptor, resolveProjectReference } from "./projectIdentity.mjs";
import { canonicalStateKey } from "./semantic/heuristics.mjs";
//#region src/pipeline/authority.ts
const TASK_METADATA_WORKFLOW_SNAPSHOT_DESCRIPTORS = [{
	metadataKey: "currentTask",
	stateKey: "workflow.current_task",
	baseScore: .9,
	confidence: .86
}, {
	metadataKey: "nextAction",
	stateKey: "workflow.next_action",
	baseScore: .88,
	confidence: .84
}];
const SNAPSHOT_ALLOWED_STATE_KEYS = new Set([
	"project.active_project",
	"workflow.current_consideration",
	"workflow.decision"
]);
const SNAPSHOT_SUPPRESSED_WORKFLOW_STATE_KEYS = new Set([
	"workflow.current_task",
	"workflow.next_action",
	"workflow.blocker",
	"workflow.current_strategy",
	"workflow.debugging_strategy",
	"workflow.context",
	"workflow.guideline",
	"workflow.task_phase",
	"workflow.candidate_resolution"
]);
const CANONICAL_WORKFLOW_STATE_KEYS = new Set([
	"workflow.current_task",
	"workflow.next_action",
	"workflow.blocker",
	"workflow.current_consideration",
	"project.active_project",
	"workflow.decision"
]);
const TASK_METADATA_DEICTIC_RE = /^(?:it|this|that|this one|that one|that line|this line|which one|哪条线|哪边|那条线|这条线|这边|那边|这里|那里|前一个项目|后一个项目|搜索那边)$/iu;
const TASK_METADATA_AMBIGUOUS_RE = /(?:哪条线|哪边|搜索那边|前一个项目|后一个项目|这条线|那条线|这边|那边|应该知道我说的是)/u;
const TASK_METADATA_LIST_RE = /(?:,|，|、|\band\b|\bor\b|\/|\|)/iu;
const TASK_METADATA_SENTENCE_RE = /[。！？!?]/u;
const TASK_METADATA_WRAPPER_RE = /^[\s"'“”‘’`《》〈〉「」『』\[\]()（）【】]+|[\s"'“”‘’`《》〈〉「」『』\[\]()（）【】]+$/gu;
const TASK_METADATA_LEADING_PUNCT_RE = /^[,，。:：;；、-]+/u;
function normalizeTaskMetadataText(value) {
	return value.replace(TASK_METADATA_WRAPPER_RE, "").replace(/\s+/g, " ").trim();
}
function sanitizeAtomicTaskMetadataValue(key, value) {
	const normalized = normalizeTaskMetadataText(value);
	if (!normalized) return;
	if (normalized.length < 2 || normalized.length > 96 || TASK_METADATA_LEADING_PUNCT_RE.test(normalized) || TASK_METADATA_DEICTIC_RE.test(normalized) || TASK_METADATA_AMBIGUOUS_RE.test(normalized) || TASK_METADATA_LIST_RE.test(normalized) || TASK_METADATA_SENTENCE_RE.test(normalized)) return;
	if (key !== "nextAction" && looksLikeProjectDescriptor(normalized)) return;
	return normalized;
}
function sanitizeProjectMetadataValue(value, params) {
	const normalized = normalizeTaskMetadataText(value);
	if (!normalized) return;
	if (normalized.length < 2 || normalized.length > 64 || TASK_METADATA_LEADING_PUNCT_RE.test(normalized) || TASK_METADATA_DEICTIC_RE.test(normalized) || TASK_METADATA_AMBIGUOUS_RE.test(normalized) || TASK_METADATA_LIST_RE.test(normalized) || looksLikeProjectDescriptor(normalized)) return;
	const candidate = normalizeTaskMetadataText(resolveProjectReference(normalized, {
		currentProject: params?.currentProject,
		knownProjects: params?.knownProjects,
		allowDescriptorAlias: Boolean(params?.currentProject || (params?.knownProjects?.length ?? 0) > 0)
	}) || normalized);
	if (!candidate || TASK_METADATA_LIST_RE.test(candidate)) return;
	return candidate;
}
function sanitizeTaskMetadataValue(key, value, params) {
	if (typeof value !== "string" || !value.trim()) return;
	if (key === "project") return sanitizeProjectMetadataValue(value, params);
	return sanitizeAtomicTaskMetadataValue(key, value);
}
function sanitizeTaskMetadata(metadata, params) {
	if (!metadata) return {};
	const project = sanitizeTaskMetadataValue("project", metadata.project, params);
	const currentTask = sanitizeTaskMetadataValue("currentTask", metadata.currentTask, params);
	const nextAction = sanitizeTaskMetadataValue("nextAction", metadata.nextAction, params);
	const blocker = sanitizeTaskMetadataValue("blocker", metadata.blocker, params);
	return {
		...project ? { project } : {},
		...currentTask ? { currentTask } : {},
		...nextAction ? { nextAction } : {},
		...blocker ? { blocker } : {}
	};
}
function isSnapshotFactualStateKey(key) {
	if (SNAPSHOT_ALLOWED_STATE_KEYS.has(key)) return true;
	if (key.startsWith("project.")) return true;
	if (SNAPSHOT_SUPPRESSED_WORKFLOW_STATE_KEYS.has(key)) return false;
	return !key.startsWith("workflow.");
}
function isCanonicalWorkflowStateKey(key) {
	return isProjectProfileStateKey(key) || CANONICAL_WORKFLOW_STATE_KEYS.has(key);
}
function canonicalOwnerForRelationHint(relation) {
	return relation.relationSlot ? "state" : "graph";
}
function sanitizeWorkflowHint(workflow) {
	const key = canonicalStateKey(workflow.key);
	if (!isCanonicalWorkflowStateKey(key)) return null;
	return {
		...workflow,
		key
	};
}
function shouldDeriveProjectProfileArtifacts(state) {
	return state.stateKind === "durable" && isProjectProfileStateKey(state.key);
}
function shouldDeriveRelationFact(relation) {
	return canonicalOwnerForRelationHint(relation) === "fact";
}
function shouldMaterializePreferenceFact(preference) {
	return Boolean(preference.predicate.trim() && preference.object.trim());
}
function shouldProjectActiveProjectAlias() {
	return false;
}
//#endregion
export { TASK_METADATA_WORKFLOW_SNAPSHOT_DESCRIPTORS, isSnapshotFactualStateKey, sanitizeTaskMetadata, sanitizeWorkflowHint, shouldDeriveProjectProfileArtifacts, shouldDeriveRelationFact, shouldMaterializePreferenceFact, shouldProjectActiveProjectAlias };
