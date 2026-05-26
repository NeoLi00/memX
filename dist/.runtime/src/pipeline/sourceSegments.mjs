import { stableHash } from "../support.mjs";
import { buildVectorDocMetadata } from "./vectorDocMetadata.mjs";
import { CHUNK_VECTOR_CONFIDENCE } from "./constants.mjs";
//#region src/pipeline/sourceSegments.ts
const SOURCE_SEGMENT_TARGET_CHARS = 1800;
function segmentBoundary(text, start, idealEnd, targetChars) {
	if (idealEnd >= text.length) return text.length;
	const minEnd = Math.min(text.length, start + Math.floor(targetChars * .6));
	let boundary = text.lastIndexOf("\n\n", idealEnd);
	if (boundary >= minEnd) return boundary + 2;
	boundary = text.lastIndexOf("\n", idealEnd);
	if (boundary >= minEnd) return boundary + 1;
	boundary = text.lastIndexOf(" ", idealEnd);
	if (boundary >= minEnd) return boundary + 1;
	return idealEnd;
}
function splitSourceText(text, params) {
	const targetChars = Math.max(400, params?.targetChars ?? 1800);
	const overlapChars = Math.min(Math.max(0, params?.overlapChars ?? 220), targetChars - 1);
	if (!text.trim()) return [];
	if (text.length <= targetChars) return [{
		index: 0,
		start: 0,
		end: text.length,
		text
	}];
	const segments = [];
	let start = 0;
	while (start < text.length) {
		const idealEnd = Math.min(text.length, start + targetChars);
		const end = segmentBoundary(text, start, idealEnd, targetChars);
		const segmentText = text.slice(start, end);
		if (segmentText.trim()) segments.push({
			index: segments.length,
			start,
			end,
			text: segmentText
		});
		if (end >= text.length) break;
		const nextStart = Math.max(0, end - overlapChars);
		start = nextStart > start ? nextStart : end;
	}
	return segments;
}
function sourceGroupIdForChunk(chunk) {
	return `source_group:${stableHash([
		chunk.agentId,
		chunk.scope,
		chunk.sourceRef
	])}`;
}
function buildSourceSegmentsForChunk(chunk) {
	const sourceGroupId = sourceGroupIdForChunk(chunk);
	const parts = splitSourceText(chunk.content);
	return parts.map((part) => {
		return {
			segmentId: `segment:${stableHash([
				chunk.chunkId,
				chunk.sourceRef,
				String(part.index),
				String(part.start),
				String(part.end)
			])}`,
			sourceGroupId,
			parentSourceRef: chunk.sourceRef,
			chunkId: chunk.chunkId,
			agentId: chunk.agentId,
			scope: chunk.scope,
			sessionKey: chunk.sessionKey,
			turnId: chunk.turnId,
			seq: chunk.seq,
			role: chunk.role,
			toolName: chunk.toolName,
			segmentIndex: part.index,
			charStart: part.start,
			charEnd: part.end,
			text: part.text,
			contentHash: stableHash([
				chunk.contentHash,
				String(part.index),
				part.text
			]),
			createdAt: chunk.createdAt,
			updatedAt: chunk.updatedAt,
			metadataJson: {
				sourceGroupId,
				parentSourceRef: chunk.sourceRef,
				chunkId: chunk.chunkId,
				role: chunk.role,
				sessionKey: chunk.sessionKey,
				segmentIndex: part.index,
				charStart: part.start,
				charEnd: part.end,
				segmentCount: parts.length
			}
		};
	});
}
function buildSourceSegmentVectorDocs(params) {
	if (params.segments.length <= 1) return [];
	return params.segments.map((segment) => ({
		docId: `event:source-segment:${segment.segmentId}`,
		docKind: "event",
		sourceId: segment.segmentId,
		scope: segment.scope,
		agentId: segment.agentId,
		text: `${segment.role}: ${segment.text}`,
		metadataJson: buildVectorDocMetadata({
			docType: "source_segment",
			confidence: CHUNK_VECTOR_CONFIDENCE,
			observedAt: segment.createdAt,
			lineage: {
				sourceKind: "chunk",
				sourceId: segment.chunkId,
				sourceRef: segment.parentSourceRef
			},
			extra: {
				chunkId: segment.chunkId,
				role: segment.role,
				sessionKey: segment.sessionKey,
				sourceGroupId: segment.sourceGroupId,
				parentSourceRef: segment.parentSourceRef,
				segmentId: segment.segmentId,
				segmentIndex: segment.segmentIndex,
				segmentCount: params.segments.length,
				charStart: segment.charStart,
				charEnd: segment.charEnd
			}
		}),
		createdAt: segment.createdAt,
		updatedAt: segment.updatedAt
	}));
}
//#endregion
export { SOURCE_SEGMENT_TARGET_CHARS, buildSourceSegmentVectorDocs, buildSourceSegmentsForChunk };
