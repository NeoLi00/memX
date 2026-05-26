import type { LineageRef } from "../types.js";
import { lineageMetadata } from "./memoryObjectsHelpers.js";

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
  "affordances",
]);

function sanitizeVectorDocMetadataValue(value: unknown): unknown {
  if (value === null) {
    return value;
  }
  if (typeof value === "string" || typeof value === "number" || typeof value === "boolean") {
    return value;
  }
  if (Array.isArray(value)) {
    return value
      .map((entry) => sanitizeVectorDocMetadataValue(entry))
      .filter((entry) => entry !== undefined);
  }
  if (typeof value === "object") {
    const next: Record<string, unknown> = {};
    for (const [key, entry] of Object.entries(value as Record<string, unknown>)) {
      const sanitized = sanitizeVectorDocMetadataValue(entry);
      if (sanitized !== undefined) {
        next[key] = sanitized;
      }
    }
    return next;
  }
  return undefined;
}

export function buildVectorDocMetadata(params: {
  docType: string;
  confidence: number;
  observedAt: string;
  lineage: LineageRef;
  extra?: Record<string, unknown>;
}): Record<string, unknown> {
  const metadata: Record<string, unknown> = {
    memxDocType: params.docType,
    confidence: params.confidence,
    observedAt: params.observedAt,
  };
  for (const [key, value] of Object.entries(params.extra ?? {})) {
    if (!ALLOWED_VECTOR_DOC_METADATA_KEYS.has(key)) {
      continue;
    }
    const sanitized = sanitizeVectorDocMetadataValue(value);
    if (sanitized !== undefined) {
      metadata[key] = sanitized;
    }
  }
  return {
    ...metadata,
    ...lineageMetadata(params.lineage),
  };
}
