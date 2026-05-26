import { lineageMetadata } from "./memoryObjectsHelpers.mjs";
//#region src/pipeline/vectorDocMetadata.ts
const ALLOWED_VECTOR_DOC_METADATA_KEYS = new Set([
	"memxDocType",
	"chunkId",
	"taskId",
	"scope",
	"role",
	"sessionKey",
	"status",
	"observedAt",
	"confidence",
	"dedupStatus",
	"assistantWeight",
	"assistantGrounding",
	"assistantComplexity",
	"assistantSummaryOnly",
	"semanticRole",
	"memoryClass",
	"recallVisibility",
	"stateKey",
	"stateKind",
	"expiresAt",
	"canonicalSubject",
	"predicate",
	"canonicalObject",
	"eventType",
	"sourceKind",
	"relationType",
	"rawPredicate",
	"relationSlot",
	"entityId",
	"activeHint",
	"supersededHint",
	"currentnessHint",
	"validFrom",
	"validTo",
	"resource",
	"resourceType",
	"owner",
	"ownershipStatus",
	"semanticStatus",
	"signalKind",
	"supportText",
	"supportRefs",
	"sourceGroupId",
	"parentSourceRef",
	"segmentId",
	"segmentIndex",
	"segmentCount",
	"charStart",
	"charEnd",
	"rawTextLength",
	"rawContentLength",
	"semanticTextTruncated",
	"segmentRefs",
	"domains",
	"affordances"
]);
function sanitizeVectorDocMetadataValue(value) {
	if (value === null) return value;
	if (typeof value === "string" || typeof value === "number" || typeof value === "boolean") return value;
	if (Array.isArray(value)) return value.map((entry) => sanitizeVectorDocMetadataValue(entry)).filter((entry) => entry !== void 0);
	if (typeof value === "object") {
		const next = {};
		for (const [key, entry] of Object.entries(value)) {
			const sanitized = sanitizeVectorDocMetadataValue(entry);
			if (sanitized !== void 0) next[key] = sanitized;
		}
		return next;
	}
}
function buildVectorDocMetadata(params) {
	const metadata = {
		memxDocType: params.docType,
		confidence: params.confidence,
		observedAt: params.observedAt
	};
	for (const [key, value] of Object.entries(params.extra ?? {})) {
		if (!ALLOWED_VECTOR_DOC_METADATA_KEYS.has(key)) continue;
		const sanitized = sanitizeVectorDocMetadataValue(value);
		if (sanitized !== void 0) metadata[key] = sanitized;
	}
	return {
		...metadata,
		...lineageMetadata(params.lineage)
	};
}
//#endregion
export { buildVectorDocMetadata };
