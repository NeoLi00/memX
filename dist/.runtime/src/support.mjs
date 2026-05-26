import { expandCjkFamilySubwords, wordLikeSegments } from "./multilingualLexicon.mjs";
import fs from "node:fs/promises";
import { homedir } from "node:os";
import path from "node:path";
import { createHash, randomUUID } from "node:crypto";
//#region src/support.ts
function nowIso(date = /* @__PURE__ */ new Date()) {
	return date.toISOString();
}
function randomId(prefix) {
	return `${prefix}_${randomUUID()}`;
}
function stableHash(parts) {
	const hash = createHash("sha256");
	for (const part of parts) {
		hash.update(part ?? "");
		hash.update("\0");
	}
	return hash.digest("hex");
}
const ZERO_WIDTH_CHARS_RE = /[\u200B-\u200D\u2060\uFEFF]/gu;
const UNICODE_DASH_RE = /[‐‑‒–—―]/gu;
const OPEN_QUOTE_RE = /[“”„‟«»]/gu;
const APOSTROPHE_RE = /[‘’‚‛`´]/gu;
const TOKEN_EDGE_RE = /^[^\p{L}\p{N}]+|[^\p{L}\p{N}]+$/gu;
function normalizeText(text) {
	return text.normalize("NFKC").replace(ZERO_WIDTH_CHARS_RE, "").replace(UNICODE_DASH_RE, "-").replace(OPEN_QUOTE_RE, "\"").replace(APOSTROPHE_RE, "'").toLowerCase().replace(/\s+/g, " ").trim();
}
function trimTokenEdges(value) {
	return value.replace(TOKEN_EDGE_RE, "").trim();
}
function cjkSubwordTerms(token) {
	return expandCjkFamilySubwords(token);
}
function normalizedTerms(text, params = {}) {
	const normalized = normalizeText(text);
	if (!normalized) return [];
	const stopwords = params.stopwords ?? /* @__PURE__ */ new Set();
	const minLength = Math.max(1, params.minLength ?? 2);
	const terms = /* @__PURE__ */ new Set();
	const pushTerm = (value) => {
		const token = trimTokenEdges(value);
		if (!token || token.length < minLength || stopwords.has(token)) return;
		terms.add(token);
	};
	for (const segment of wordLikeSegments(normalized)) {
		const token = trimTokenEdges(segment);
		if (!token) continue;
		const expanded = params.includeCjkSubwords === false ? [token] : cjkSubwordTerms(token);
		for (const entry of expanded) pushTerm(entry);
	}
	return [...terms];
}
function truncateText(text, maxChars) {
	if (text.length <= maxChars) return text;
	return `${text.slice(0, Math.max(0, maxChars - 1)).trimEnd()}…`;
}
async function ensureParentDir(filePath) {
	await fs.mkdir(path.dirname(filePath), { recursive: true });
}
function resolveUserPath(input) {
	if (input.startsWith("~/")) return path.join(homedir(), input.slice(2));
	if (input === "~") return homedir();
	return path.resolve(input);
}
function safeJsonParse(value, fallback) {
	if (!value) return fallback;
	try {
		return JSON.parse(value);
	} catch {
		return fallback;
	}
}
function clamp01(value) {
	if (!Number.isFinite(value)) return 0;
	if (value <= 0) return 0;
	if (value >= 1) return 1;
	return value;
}
function normalizeName(value) {
	return normalizeText(value).replace(/[^\p{L}\p{N}:/._ +#-]+/gu, "");
}
function addHours(iso, hours) {
	const date = new Date(iso);
	date.setTime(date.getTime() + hours * 60 * 60 * 1e3);
	return date.toISOString();
}
function cosineSimilarity(left, right) {
	if (left.length === 0 || right.length === 0 || left.length !== right.length) return 0;
	let dot = 0;
	let leftMag = 0;
	let rightMag = 0;
	for (let index = 0; index < left.length; index += 1) {
		const leftValue = left[index] ?? 0;
		const rightValue = right[index] ?? 0;
		dot += leftValue * rightValue;
		leftMag += leftValue * leftValue;
		rightMag += rightValue * rightValue;
	}
	if (leftMag === 0 || rightMag === 0) return 0;
	return dot / Math.sqrt(leftMag * rightMag);
}
function orderByScore(items) {
	return [...items].sort((left, right) => right.score - left.score);
}
const SENTENCE_START_PATTERNS = /^(?:the |a |an |when |because |which |that |if |but |and |or |so |this |these |those |it |its )/i;
const CJK_RANGE = /[\p{Script=Han}]/u;
const CJK_ONLY = /^[\p{Script=Han}]+$/u;
const CJK_PUNCTUATION = /[，。！？；：、「」『』【】（）《》〈〉""''…—]/;
const ENTITY_SENTENCE_MARKERS = new Set([
	"the",
	"a",
	"an",
	"when",
	"because",
	"which",
	"that",
	"if",
	"but",
	"and",
	"or",
	"so",
	"then",
	"what",
	"why",
	"how",
	"where",
	"who",
	"with",
	"from",
	"into",
	"for",
	"在",
	"把",
	"被",
	"和",
	"跟",
	"如果",
	"因为",
	"所以",
	"但是",
	"然后",
	"什么",
	"怎么",
	"为什么",
	"哪里",
	"谁"
]);
/**
* Validates that a string is a reasonable entity name rather than a sentence fragment.
* Rejects overly long strings, sentence-like structures, and strings lacking
* any alphabetic or CJK content.
*/
function isValidEntityName(name) {
	const trimmed = name.trim();
	if (trimmed.length === 0 || trimmed.length > 80) return false;
	if (CJK_ONLY.test(trimmed) && trimmed.length > 20) return false;
	if (CJK_RANGE.test(trimmed) && CJK_PUNCTUATION.test(trimmed)) return false;
	if (!/[\p{L}]/u.test(trimmed)) return false;
	const lexicalTerms = normalizedTerms(trimmed, {
		minLength: 1,
		includeCjkSubwords: false
	}).filter((term) => /[\p{L}]/u.test(term));
	if (lexicalTerms.length === 0 || lexicalTerms.length > 8) return false;
	const sentenceMarkerCount = lexicalTerms.filter((term) => ENTITY_SENTENCE_MARKERS.has(term)).length;
	if (sentenceMarkerCount >= 2) return false;
	if (lexicalTerms.length >= 4 && sentenceMarkerCount / lexicalTerms.length >= .34) return false;
	if (SENTENCE_START_PATTERNS.test(trimmed) && lexicalTerms.length > 1) return false;
	if (normalizeName(trimmed).replace(/[^\p{L}\p{N}]+/gu, "").length > 48 && lexicalTerms.length >= 5) return false;
	return true;
}
function objectRecord(value) {
	return value && typeof value === "object" && !Array.isArray(value) ? value : void 0;
}
//#endregion
export { addHours, clamp01, cosineSimilarity, ensureParentDir, isValidEntityName, normalizeName, normalizeText, normalizedTerms, nowIso, objectRecord, orderByScore, randomId, resolveUserPath, safeJsonParse, stableHash, truncateText };
