import type { MemxHostId } from "./hookPayload.js";

type NativeMemoryBypassDecision = {
  reason: string;
};

const TOOL_EVENTS = new Set(["PreToolUse", "PermissionRequest"]);
const FILE_TARGET_TOOLS = new Set([
  "bash",
  "shell",
  "exec",
  "exec_command",
  "write",
  "edit",
  "multiedit",
  "read",
  "grep",
  "glob",
  "apply_patch",
]);

const HOST_MEMORY_PATH_PATTERNS = [
  /(?:^|[\/\\])\.codex(?:[\/\\](?:memories?|sessions?|history)|[\/\\][^\s"'`]*?(?:memory|memories|session|history))/iu,
  /(?:^|[\/\\])\.claude(?:[\/\\](?:memories?|projects?|history)|[\/\\][^\s"'`]*?(?:memory|memories|session|history))/iu,
  /(?:^|[\/\\])\.openclaw(?:[\/\\].*?[\/\\])?memory(?:[\/\\]|$)/iu,
  /(?:^|[\/\\])user_facts\.txt\b/iu,
];

const MEMORY_MARKDOWN_RE = /(?:^|[\/\\])(?:memory|memories|MEMORY|MEMORIES)\.md\b/u;
const MEMORY_INTENT_RE =
  /\b(?:remember|memory|memories|recall|history|transcript|user_facts)\b|(?:记忆|记住|记下|回忆|历史记录|对话记录)/iu;

function isRecord(value: unknown): value is Record<string, unknown> {
  return Boolean(value) && typeof value === "object" && !Array.isArray(value);
}

function readString(record: Record<string, unknown>, keys: string[]): string | undefined {
  for (const key of keys) {
    const value = record[key];
    if (typeof value === "string" && value.trim()) {
      return value.trim();
    }
  }
  return undefined;
}

function collectStrings(value: unknown, output: string[] = [], depth = 0): string[] {
  if (depth > 6 || output.join("\n").length > 40_000) {
    return output;
  }
  if (typeof value === "string") {
    if (value.trim()) {
      output.push(value);
    }
    return output;
  }
  if (Array.isArray(value)) {
    for (const entry of value) {
      collectStrings(entry, output, depth + 1);
    }
    return output;
  }
  if (isRecord(value)) {
    for (const entry of Object.values(value)) {
      collectStrings(entry, output, depth + 1);
    }
  }
  return output;
}

function normalizedToolName(payload: Record<string, unknown>): string {
  return (
    readString(payload, ["tool_name", "toolName", "name", "matcher"]) ?? ""
  )
    .toLowerCase()
    .replace(/[^a-z0-9_-]+/gu, "_");
}

function toolInputText(payload: Record<string, unknown>): string {
  return collectStrings(payload.tool_input ?? payload.toolInput ?? payload.input ?? payload).join("\n");
}

function mentionsHostMemoryStore(text: string): boolean {
  if (HOST_MEMORY_PATH_PATTERNS.some((pattern) => pattern.test(text))) {
    return true;
  }
  return MEMORY_MARKDOWN_RE.test(text) && MEMORY_INTENT_RE.test(text);
}

export function detectNativeMemoryBypass(
  host: MemxHostId,
  eventName: string,
  payload: Record<string, unknown>,
): NativeMemoryBypassDecision | null {
  if ((host !== "codex" && host !== "claude-code") || !TOOL_EVENTS.has(eventName)) {
    return null;
  }
  const toolName = normalizedToolName(payload);
  if (toolName && ![...FILE_TARGET_TOOLS].some((tool) => toolName.includes(tool))) {
    return null;
  }
  const text = toolInputText(payload);
  if (!mentionsHostMemoryStore(text)) {
    return null;
  }
  return {
    reason:
      "memX lifecycle memory is already active. Do not read or write host-native memory stores, transcript archives, or memory.md files for recall; continue using the current turn normally.",
  };
}
