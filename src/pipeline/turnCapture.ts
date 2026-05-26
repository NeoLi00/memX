import { stripInjectedHistoricalBlock } from "../security/escaping.js";
import { normalizeText, normalizedTerms, stableHash } from "../support.js";
import type { TurnCaptureMessage, TurnCaptureRole } from "../types.js";
import { shouldSuppressCapturedMessage } from "./bootstrapFilter.js";
import { isHeartbeatControlText } from "./heartbeatFilter.js";
import { readMessageText, stripInboundMetadata } from "./messageText.js";

const SYSTEM_BOILERPLATE_RE = /^A new session was started via \/new or \/reset\b/;
const SELF_TOOL_RE = /^memory[_-]/i;
const ASSISTANT_MEMORY_CONFIRMATION_RE =
  /(?:\b(?:acknowledged|got it|noted|recorded|saved|remembered|understood)\b|\b(?:i(?:'ve| have)? (?:noted|saved|remembered|recorded)|i(?:'ll| will) (?:remember|record|save|keep (?:that )?in mind)|going forward|from now on|noted for future use|saved for later)\b|(?:已(?:记住|记下|记录|保存)|(?:记住|记下|记录|保存)(?:了)?|我(?:已经)?(?:记住|记下|记录|保存)(?:了)?|收到|好的|明白|了解|我会记得|后续(?:我)?会|以后(?:我)?会|我会(?:按|用|照)|记忆更新确认|供后续使用|后面会用|之后会用|未来使用))/iu;

function tokenOverlap(left: string, right: string): number {
  const tokenize = (value: string) => new Set(normalizedTerms(value, { minLength: 2 }));
  const leftTokens = tokenize(left);
  const rightTokens = tokenize(right);
  if (leftTokens.size === 0 || rightTokens.size === 0) {
    return 0;
  }
  let overlap = 0;
  for (const token of leftTokens) {
    if (rightTokens.has(token)) {
      overlap += 1;
    }
  }
  return overlap / Math.max(leftTokens.size, 1);
}

function looksLikeRecallEcho(text: string, recalledTexts: string[]): boolean {
  const lowered = normalizeText(text);
  if (lowered.length < 20) {
    return false;
  }
  for (const recalled of recalledTexts) {
    const normalized = normalizeText(recalled);
    if (normalized.length < 12) {
      continue;
    }
    if (lowered.includes(normalized) || normalized.includes(lowered)) {
      return true;
    }
    if (tokenOverlap(lowered, normalized) >= 0.55) {
      return true;
    }
  }
  return false;
}

function looksLikeAssistantMemoryConfirmation(text: string): boolean {
  const trimmed = text.trim();
  if (!trimmed) {
    return false;
  }
  const lineCount = trimmed.split(/\r?\n/u).length;
  if (trimmed.length > 560 || lineCount > 6 || trimmed.includes("```")) {
    return false;
  }
  if (!ASSISTANT_MEMORY_CONFIRMATION_RE.test(trimmed)) {
    return false;
  }
  return /(?:记住|记下|记录|保存|remember|noted|saved|future use|later|后续|未来)/iu.test(trimmed);
}

function offsetObservedAt(baseObservedAt: string, index: number): string {
  const baseMs = Date.parse(baseObservedAt);
  if (!Number.isFinite(baseMs)) {
    return baseObservedAt;
  }
  return new Date(baseMs + index).toISOString();
}

type CaptureParams = {
  agentId: string;
  scope: string;
  sessionKey: string;
  turnId: string;
  observedAt: string;
  messages: unknown[];
  recalledTexts?: string[];
};

export function captureAgentEndTurn(params: CaptureParams): TurnCaptureMessage[] {
  const raw: Array<{ role: TurnCaptureRole; content: string; toolName?: string }> = [];

  for (const message of params.messages) {
    if (!message || typeof message !== "object") {
      continue;
    }
    const entry = message as Record<string, unknown>;
    const role = entry.role as TurnCaptureRole;
    if (role !== "user" && role !== "assistant" && role !== "tool") {
      continue;
    }
    const toolName =
      role === "tool"
        ? typeof entry.name === "string"
          ? entry.name
          : typeof entry.toolName === "string"
            ? entry.toolName
            : undefined
        : undefined;
    if (role === "tool" && toolName && SELF_TOOL_RE.test(toolName)) {
      continue;
    }
    let text = readMessageText(entry.content).trim();
    if (!text) {
      continue;
    }
    if (role === "user") {
      text = stripInboundMetadata(stripInjectedHistoricalBlock(text));
      if (SYSTEM_BOILERPLATE_RE.test(text)) {
        continue;
      }
    } else {
      text = stripInjectedHistoricalBlock(text);
    }
    text = text.trim();
    if (!text) {
      continue;
    }
    if (isHeartbeatControlText(text)) {
      continue;
    }
    if (shouldSuppressCapturedMessage({ role, content: text, toolName })) {
      continue;
    }
    raw.push({ role, content: text, toolName });
  }

  const merged: Array<{ role: TurnCaptureRole; content: string; toolName?: string }> = [];
  for (let index = 0; index < raw.length; index += 1) {
    const current = raw[index]!;
    if (current.role !== "assistant") {
      merged.push(current);
      continue;
    }
    let combined = current.content;
    while (index + 1 < raw.length && raw[index + 1]?.role === "assistant") {
      index += 1;
      combined = `${combined}\n\n${raw[index]!.content}`.trim();
    }
    if (
      !looksLikeRecallEcho(combined, params.recalledTexts ?? []) &&
      !looksLikeAssistantMemoryConfirmation(combined)
    ) {
      merged.push({ role: "assistant", content: combined });
    }
  }

  return merged.map((entry, index) => ({
    role: entry.role,
    content: entry.content,
    toolName: entry.toolName,
    observedAt: offsetObservedAt(params.observedAt, index),
    turnId: params.turnId,
    sessionKey: params.sessionKey,
    agentId: params.agentId,
    scope: params.scope,
    sourceRef: `${entry.role}:${stableHash([params.agentId, params.sessionKey, params.turnId, String(index)])}`,
  }));
}
