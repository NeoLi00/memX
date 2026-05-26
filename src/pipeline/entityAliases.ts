import { isValidEntityName, normalizeName } from "../support.js";

const ENTITY_ALIAS_SPLIT_RE = /[\/\\|,;，、；:：()[\]{}（）【】「」『』《》]+/u;
const CODE_LIKE_ENTITY_TOKEN_RE = /[A-Za-z][A-Za-z0-9_.:-]{2,}/gu;
const CJK_ENTITY_RUN_RE = /[\p{Script=Han}\p{Script=Hiragana}\p{Script=Katakana}\p{Script=Hangul}]{2,}/gu;

function usefulAliasTerm(term: string): boolean {
  if (!term) {
    return false;
  }
  if (/[\p{Script=Han}\p{Script=Hiragana}\p{Script=Katakana}\p{Script=Hangul}]/u.test(term)) {
    return term.length >= 2;
  }
  return term.length >= 3;
}

export function entityNameAliases(name: string, limit = 8): string[] {
  const trimmed = name.trim();
  if (!trimmed || !isValidEntityName(trimmed)) {
    return [];
  }
  const aliases: string[] = [];
  const push = (value: string | undefined): void => {
    const candidate = value?.trim();
    if (!candidate) {
      return;
    }
    const normalized = normalizeName(candidate);
    if (!usefulAliasTerm(normalized)) {
      return;
    }
    aliases.push(candidate);
  };

  push(trimmed);
  for (const part of trimmed.split(ENTITY_ALIAS_SPLIT_RE)) {
    push(part);
  }
  for (const match of trimmed.matchAll(CODE_LIKE_ENTITY_TOKEN_RE)) {
    push(match[0]);
  }
  for (const match of trimmed.matchAll(CJK_ENTITY_RUN_RE)) {
    push(match[0]);
  }

  const seen = new Set<string>();
  const unique: string[] = [];
  for (const alias of aliases) {
    const key = normalizeName(alias);
    if (!key || seen.has(key)) {
      continue;
    }
    seen.add(key);
    unique.push(alias);
    if (unique.length >= limit) {
      break;
    }
  }
  return unique;
}

export function entityNameAliasTerms(name: string, limit = 8): string[] {
  return entityNameAliases(name, limit)
    .map((alias) => normalizeName(alias))
    .filter(usefulAliasTerm);
}
