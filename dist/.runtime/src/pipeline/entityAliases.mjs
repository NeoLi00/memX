import { isValidEntityName, normalizeName } from "../support.mjs";
//#region src/pipeline/entityAliases.ts
const ENTITY_ALIAS_SPLIT_RE = /[\/\\|,;，、；:：()[\]{}（）【】「」『』《》]+/u;
const CODE_LIKE_ENTITY_TOKEN_RE = /[A-Za-z][A-Za-z0-9_.:-]{2,}/gu;
const CJK_ENTITY_RUN_RE = /[\p{Script=Han}\p{Script=Hiragana}\p{Script=Katakana}\p{Script=Hangul}]{2,}/gu;
function usefulAliasTerm(term) {
	if (!term) return false;
	if (/[\p{Script=Han}\p{Script=Hiragana}\p{Script=Katakana}\p{Script=Hangul}]/u.test(term)) return term.length >= 2;
	return term.length >= 3;
}
function entityNameAliases(name, limit = 8) {
	const trimmed = name.trim();
	if (!trimmed || !isValidEntityName(trimmed)) return [];
	const aliases = [];
	const push = (value) => {
		const candidate = value?.trim();
		if (!candidate) return;
		if (!usefulAliasTerm(normalizeName(candidate))) return;
		aliases.push(candidate);
	};
	push(trimmed);
	for (const part of trimmed.split(ENTITY_ALIAS_SPLIT_RE)) push(part);
	for (const match of trimmed.matchAll(CODE_LIKE_ENTITY_TOKEN_RE)) push(match[0]);
	for (const match of trimmed.matchAll(CJK_ENTITY_RUN_RE)) push(match[0]);
	const seen = /* @__PURE__ */ new Set();
	const unique = [];
	for (const alias of aliases) {
		const key = normalizeName(alias);
		if (!key || seen.has(key)) continue;
		seen.add(key);
		unique.push(alias);
		if (unique.length >= limit) break;
	}
	return unique;
}
function entityNameAliasTerms(name, limit = 8) {
	return entityNameAliases(name, limit).map((alias) => normalizeName(alias)).filter(usefulAliasTerm);
}
//#endregion
export { entityNameAliasTerms, entityNameAliases };
