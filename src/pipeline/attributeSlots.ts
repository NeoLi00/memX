const SEMANTIC_FACT_VERB_PREFIXES = new Set(["has", "uses", "prefers", "depends"]);

const CANONICAL_ATTRIBUTE_SLOT_ALIASES: Record<string, string> = {
  alertchannel: "alert_channel",
  alert_channel: "alert_channel",
  alertmethod: "alert_channel",
  alert_method: "alert_channel",
  alarmchannel: "alert_channel",
  alarm_channel: "alert_channel",
  notificationchannel: "alert_channel",
  notification_channel: "alert_channel",
  notificationmethod: "alert_channel",
  notification_method: "alert_channel",
  告警渠道: "alert_channel",
  告警通道: "alert_channel",
  告警方式: "alert_channel",
  报警渠道: "alert_channel",
  报警通道: "alert_channel",
  报警方式: "alert_channel",
  通知渠道: "alert_channel",
  通知通道: "alert_channel",
  通知方式: "alert_channel",
  database: "default_database",
  数据库: "default_database",
  defaultdatabase: "default_database",
  default_database: "default_database",
  defaultdb: "default_database",
  default_db: "default_database",
  primarydatabase: "default_database",
  primary_database: "default_database",
  exportformat: "export_format",
  export_format: "export_format",
  outputformat: "export_format",
  output_format: "export_format",
  archiveformat: "archive_format",
  archive_format: "archive_format",
  archivalformat: "archive_format",
  archival_format: "archive_format",
  storageformat: "archive_format",
  storage_format: "archive_format",
  归档格式: "archive_format",
  存档格式: "archive_format",
  档案格式: "archive_format",
  defaultmessagequeue: "default_message_queue",
  default_message_queue: "default_message_queue",
  messagequeue: "default_message_queue",
  message_queue: "default_message_queue",
  defaultqueue: "default_message_queue",
  default_queue: "default_message_queue",
  defaulttaskqueue: "default_message_queue",
  default_task_queue: "default_message_queue",
  taskqueue: "default_message_queue",
  task_queue: "default_message_queue",
  jobqueue: "default_message_queue",
  job_queue: "default_message_queue",
  msgqueue: "default_message_queue",
  msg_queue: "default_message_queue",
  queue: "default_message_queue",
  消息队列: "default_message_queue",
  任务队列: "default_message_queue",
  队列: "default_message_queue",
  defaultcache: "default_cache",
  default_cache: "default_cache",
  cache: "default_cache",
  缓存: "default_cache",
  owner: "owner",
  owner_user: "owner",
 负责人: "owner",
  provider: "provider",
  服务商: "provider",
  retrypolicy: "retry_strategy",
  retry_policy: "retry_strategy",
  retrystrategy: "retry_strategy",
  retry_strategy: "retry_strategy",
  failureretrypolicy: "retry_strategy",
  failure_retry_policy: "retry_strategy",
  failureretrystrategy: "retry_strategy",
  failure_retry_strategy: "retry_strategy",
  重试策略: "retry_strategy",
  失败重试策略: "retry_strategy",
  constraint: "constraint",
  约束: "constraint",
};

const ATTRIBUTE_SLOT_MODIFIER_PREFIXES = new Set([
  "default",
  "primary",
  "main",
  "current",
  "selected",
]);
const CJK_ATTRIBUTE_SLOT_MODIFIER_PREFIXES = ["默认", "主要", "主", "当前"];

const ATTRIBUTE_VALUE_QUERY_RE =
  /(?:什么|哪些|哪一个|哪种|是哪|是什么|用什么|使用什么|what|which|who|where|when|value|default|current)/iu;

function normalizeSlotText(value: string): string {
  return value
    .replace(/([a-z0-9])([A-Z])/g, "$1_$2")
    .normalize("NFKC")
    .toLowerCase()
    .replace(/[^\p{L}\p{N}]+/gu, "_")
    .replace(/_+/g, "_")
    .replace(/^_+|_+$/g, "");
}

function attributeSlotAlias(slot: string): string | undefined {
  return CANONICAL_ATTRIBUTE_SLOT_ALIASES[slot] ?? CANONICAL_ATTRIBUTE_SLOT_ALIASES[slot.replace(/_/g, "")];
}

function stripAttributeSlotModifierPrefix(slot: string): string {
  const parts = slot.split("_").filter(Boolean);
  while (parts.length > 1 && ATTRIBUTE_SLOT_MODIFIER_PREFIXES.has(parts[0] ?? "")) {
    parts.shift();
  }
  let stripped = parts.join("_") || slot;
  for (const prefix of CJK_ATTRIBUTE_SLOT_MODIFIER_PREFIXES) {
    if (stripped.startsWith(prefix) && stripped.length > prefix.length) {
      stripped = stripped.slice(prefix.length);
      break;
    }
  }
  return stripped;
}

export function canonicalAttributeSlot(value: string | undefined): string | undefined {
  const raw = value?.trim();
  if (!raw) {
    return undefined;
  }
  const scopedTail =
    raw
      .split(/[.:/#|]+/u)
      .map((entry) => entry.trim())
      .filter(Boolean)
      .at(-1) ?? raw;
  const slot = normalizeSlotText(scopedTail);
  if (!slot) {
    return undefined;
  }
  const parts = slot.split("_");
  const tail = SEMANTIC_FACT_VERB_PREFIXES.has(parts[0] ?? "") ? parts.slice(1).join("_") : slot;
  return attributeSlotAlias(tail) ?? attributeSlotAlias(stripAttributeSlotModifierPrefix(tail));
}

export function normalizeSemanticFactPredicate(value: string | undefined): string | undefined {
  const raw = value?.trim();
  if (!raw) {
    return undefined;
  }
  const slot = normalizeSlotText(raw);
  if (!slot) {
    return undefined;
  }
  const attributeSlot = canonicalAttributeSlot(raw);
  if (attributeSlot) {
    return `has_${attributeSlot}`;
  }
  const verb = slot.split("_")[0];
  return SEMANTIC_FACT_VERB_PREFIXES.has(verb) ? slot : `has_${slot}`;
}

export function semanticFactPredicatesForAttributeSlot(slot: string): string[] {
  return [`has_${slot}`, `uses_${slot}`];
}

export function attributeSlotFromPredicate(predicate: string | undefined): string | undefined {
  return canonicalAttributeSlot(predicate);
}

export function predicateMatchesAttributeSlots(
  predicate: string | undefined,
  requestedSlots: string[],
): boolean {
  const slot = attributeSlotFromPredicate(predicate);
  return Boolean(slot && requestedSlots.includes(slot));
}

function aliasMatchesNormalizedText(normalizedText: string, alias: string): boolean {
  const normalizedAlias = normalizeSlotText(alias);
  if (!normalizedAlias) {
    return false;
  }
  const compactText = normalizedText.replace(/_/g, "");
  const compactAlias = normalizedAlias.replace(/_/g, "");
  if (/[\p{Script=Han}]/u.test(normalizedAlias)) {
    return compactText.includes(compactAlias);
  }
  const tokens = normalizedText.split("_").filter(Boolean);
  const aliasTokens = normalizedAlias.split("_").filter(Boolean);
  if (aliasTokens.length === 0 || tokens.length < aliasTokens.length) {
    return false;
  }
  for (let index = 0; index <= tokens.length - aliasTokens.length; index += 1) {
    if (aliasTokens.every((token, offset) => tokens[index + offset] === token)) {
      return true;
    }
  }
  return false;
}

export function requestedAttributeSlotsFromText(...values: Array<string | undefined>): string[] {
  const slots = new Set<string>();
  for (const value of values) {
    const text = value?.trim();
    if (!text) {
      continue;
    }
    const normalizedText = normalizeSlotText(text);
    if (!normalizedText) {
      continue;
    }
    for (const slot of Object.values(CANONICAL_ATTRIBUTE_SLOT_ALIASES)) {
      if (aliasMatchesNormalizedText(normalizedText, slot)) {
        slots.add(slot);
      }
    }
    const aliases = Object.keys(CANONICAL_ATTRIBUTE_SLOT_ALIASES).sort((left, right) => right.length - left.length);
    for (const alias of aliases) {
      if (aliasMatchesNormalizedText(normalizedText, alias)) {
        slots.add(CANONICAL_ATTRIBUTE_SLOT_ALIASES[alias]!);
      }
    }
  }
  return [...slots];
}

export function queryAsksForAttributeValue(query: string): boolean {
  return ATTRIBUTE_VALUE_QUERY_RE.test(query.normalize("NFKC"));
}

export function attributeSlotContractHints(slots: string[]): string[] {
  const hints = new Set<string>();
  for (const slot of slots) {
    hints.add(slot);
    hints.add(slot.replace(/_/g, " "));
    for (const predicate of semanticFactPredicatesForAttributeSlot(slot)) {
      hints.add(predicate);
      hints.add(predicate.replace(/_/g, " "));
    }
    for (const [alias, canonical] of Object.entries(CANONICAL_ATTRIBUTE_SLOT_ALIASES)) {
      if (canonical === slot) {
        hints.add(alias);
        hints.add(alias.replace(/_/g, " "));
      }
    }
  }
  return [...hints].filter(Boolean);
}

export function attributeSlotAliasesForSlots(slots: string[]): string[] {
  const requested = new Set(slots);
  const aliases = new Set<string>();
  for (const [alias, canonical] of Object.entries(CANONICAL_ATTRIBUTE_SLOT_ALIASES)) {
    if (!requested.has(canonical)) {
      continue;
    }
    aliases.add(alias);
    aliases.add(alias.replace(/_/g, " "));
  }
  for (const slot of requested) {
    aliases.add(slot);
    aliases.add(slot.replace(/_/g, " "));
  }
  return [...aliases].filter(Boolean).sort((left, right) => right.length - left.length);
}
