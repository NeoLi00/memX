import { clamp01, truncateText } from "../support.mjs";
import { semanticTextSimilarity } from "./semantic/textSimilarity.mjs";
//#region src/pipeline/sourceWeighting.ts
function contentStructuralComplexity(text) {
	const trimmed = text.trim();
	if (!trimmed) return 0;
	const lineCount = trimmed.split(/\r?\n/u).length;
	const bulletCount = (trimmed.match(/^\s*(?:[-*•]|\d+\.)\s+/gmu) ?? []).length;
	const fileMentionCount = (trimmed.match(/\b[\p{L}\p{N}_./-]+\.(?:ts|js|json|md|sh|sql|txt|yml|yaml|html|css)\b/gu) ?? []).length;
	const hasCodeFence = trimmed.includes("```");
	const hasCommandDensity = /(?:^|\n)\s*(?:pnpm|npm|bun|node|git|sqlite3|openclaw)\b/u.test(trimmed);
	return clamp01(Math.min(trimmed.length, 2400) / 2400 * .34 + Math.min(lineCount, 40) / 40 * .16 + Math.min(bulletCount, 12) / 12 * .12 + Math.min(fileMentionCount, 8) / 8 * .08 + (hasCodeFence ? .18 : 0) + (hasCommandDensity ? .12 : 0));
}
function surroundingSupportText(chunks, index) {
	return chunks.filter((_, chunkIndex) => Math.abs(chunkIndex - index) <= 2 && chunkIndex !== index).filter((chunk) => chunk.role !== "assistant").map((chunk) => chunk.content).join("\n").trim();
}
function nearestToolDistance(chunks, index) {
	let bestDistance = null;
	for (let chunkIndex = 0; chunkIndex < chunks.length; chunkIndex += 1) {
		if (chunks[chunkIndex]?.role !== "tool") continue;
		const distance = Math.abs(chunkIndex - index);
		if (bestDistance === null || distance < bestDistance) bestDistance = distance;
	}
	return bestDistance;
}
const ASSISTANT_ACK_CUE_RE = /(?:\b(?:acknowledged|got it|noted|recorded|saved|remembered|understood)\b|\b(?:i(?:'ll| will| have|’ll) (?:remember|record|save|keep|use|treat)|going forward|from now on)\b|(?:已(?:记住|记下|记录|保存)|(?:记住|记下|记录|保存)(?:了)?|收到|好的|明白|了解|后续(?:我)?会|以后(?:我)?会|我会(?:记得|按|用|照)))/iu;
function looksLikeAssistantAcknowledgement(chunk, taskChunks, assessment) {
	const trimmed = chunk.content.trim();
	if (!trimmed || chunk.role !== "assistant") return false;
	const lineCount = trimmed.split(/\r?\n/u).length;
	if (trimmed.length > 560 || lineCount > 6 || trimmed.includes("```")) return false;
	if (!ASSISTANT_ACK_CUE_RE.test(trimmed)) return false;
	const index = taskChunks.findIndex((entry) => entry.chunkId === chunk.chunkId);
	const supportText = index >= 0 ? surroundingSupportText(taskChunks, index) : "";
	const echoScore = supportText ? Math.max(semanticTextSimilarity(trimmed, supportText), semanticTextSimilarity(chunk.summary || trimmed, supportText)) : assessment.grounding;
	return /\b(?:remember|record|save|noted|saved|recorded|remembered)\b|(?:记住|记下|记录|保存)/iu.test(trimmed) ? echoScore >= .18 || supportText.length === 0 : echoScore >= .42;
}
function assessAssistantChunk(chunk, taskChunks) {
	if (chunk.role !== "assistant") return {
		weight: 1,
		grounding: 1,
		complexity: contentStructuralComplexity(chunk.content),
		useSummaryOnly: false
	};
	const index = taskChunks.findIndex((entry) => entry.chunkId === chunk.chunkId);
	const supportText = index >= 0 ? surroundingSupportText(taskChunks, index) : "";
	const grounding = supportText ? Math.max(semanticTextSimilarity(chunk.summary || chunk.content, supportText), semanticTextSimilarity(chunk.content, supportText)) : 0;
	const complexity = contentStructuralComplexity(chunk.content);
	const trimmed = chunk.content.trim();
	const lineCount = trimmed ? trimmed.split(/\r?\n/u).length : 0;
	const toolDistance = index >= 0 ? nearestToolDistance(taskChunks, index) : null;
	const toolSupport = toolDistance === null ? 0 : toolDistance <= 1 ? .16 : toolDistance <= 2 ? .08 : 0;
	const longTutorialPenalty = toolDistance === null && trimmed.length > 900 && lineCount > 10 && complexity > .56 ? .18 : 0;
	const conciseAssistantBonus = trimmed.length > 0 && trimmed.length <= 360 && lineCount <= 5 && complexity <= .38 ? .24 : 0;
	const weight = clamp01(.28 + grounding * .46 + toolSupport + conciseAssistantBonus - complexity * .28 - longTutorialPenalty);
	const acknowledgement = looksLikeAssistantAcknowledgement(chunk, taskChunks, {
		grounding,
		complexity
	});
	return {
		weight: acknowledgement ? Math.min(weight, .34) : weight,
		grounding,
		complexity,
		useSummaryOnly: acknowledgement || weight < .58 || complexity > .68,
		...acknowledgement ? {
			semanticRole: "assistant_acknowledgement",
			memoryClass: "assistant_acknowledgement",
			recallVisibility: "support_only"
		} : {}
	};
}
function renderTaskPromptChunk(chunk, taskChunks) {
	if (chunk.role !== "assistant") return truncateText(chunk.content, 500);
	const assessment = assessAssistantChunk(chunk, taskChunks);
	if (assessment.weight < .38) return "assistant explanatory response (low grounding; use only if corroborated by user or tool evidence)";
	return truncateText(assessment.useSummaryOnly ? chunk.summary || chunk.content : chunk.content, assessment.useSummaryOnly ? 220 : 380);
}
function assistantVectorText(chunk, taskChunks) {
	if (chunk.role !== "assistant") return truncateText(chunk.content, 500);
	if (taskChunks) {
		const assessment = assessAssistantChunk(chunk, taskChunks);
		if (assessment.weight < .38) return truncateText(chunk.summary || "assistant explanatory response", 140);
		if (assessment.useSummaryOnly) return truncateText(chunk.summary || chunk.content, 180);
	}
	if (contentStructuralComplexity(chunk.content) >= .64) return truncateText(chunk.summary || chunk.content, 180);
	return truncateText(chunk.content, 320);
}
function assistantVectorSummary(chunk, taskChunks) {
	if (chunk.role !== "assistant") return truncateText(chunk.summary || chunk.content, 180);
	if (taskChunks) {
		const assessment = assessAssistantChunk(chunk, taskChunks);
		if (assessment.weight < .38) return "assistant explanatory response";
		if (assessment.useSummaryOnly) return truncateText(chunk.summary || "assistant response", 140);
	}
	return truncateText(chunk.summary || chunk.content, 180);
}
//#endregion
export { assessAssistantChunk, assistantVectorSummary, assistantVectorText, contentStructuralComplexity, renderTaskPromptChunk };
