import { open, readdir, stat } from "node:fs/promises";
import { homedir } from "node:os";
import { basename, join } from "node:path";
import { truncateText } from "../support.js";
import {
  MEMX_TRANSCRIPT_CAPTURE_INTERVAL_MS,
  MEMX_TRANSCRIPT_CAPTURE_TIMEOUT_MS,
} from "../timeouts.js";
import type { MemxHostId, MemxHostMessage, MemxTurnEnvelope } from "./hookPayload.js";

type TranscriptCandidate = {
  role: "user" | "assistant";
  content: string;
};

export type TranscriptAssistantCapture = {
  hostId: MemxHostId;
  path: string;
  message: MemxHostMessage;
};

const MAX_TRANSCRIPT_BYTES = 2_000_000;
const MAX_DISCOVERY_DEPTH = 7;
const MAX_DISCOVERY_FILES = 80;

function isRecord(value: unknown): value is Record<string, unknown> {
  return Boolean(value) && typeof value === "object" && !Array.isArray(value);
}

function parsePositiveInt(value: string | undefined, fallback: number): number {
  const parsed = Number(value);
  return Number.isFinite(parsed) && parsed >= 0 ? parsed : fallback;
}

async function sleep(ms: number): Promise<void> {
  await new Promise<void>((resolve) => {
    setTimeout(resolve, Math.max(0, ms));
  });
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

function textFromContentParts(value: unknown, textTypes: string[]): string {
  if (typeof value === "string") {
    return value.trim();
  }
  if (!Array.isArray(value)) {
    return "";
  }
  return value
    .map((part) => {
      if (!isRecord(part)) {
        return "";
      }
      const type = typeof part.type === "string" ? part.type : "";
      if (!textTypes.includes(type)) {
        return "";
      }
      return readString(part, ["text", "message", "content"]) ?? "";
    })
    .filter(Boolean)
    .join("\n")
    .trim();
}

function candidateFromCodexRecord(record: Record<string, unknown>): TranscriptCandidate | null {
  const payload = isRecord(record.payload) ? record.payload : undefined;
  if (record.type === "response_item" && payload?.type === "message") {
    const role = payload.role === "user" || payload.role === "assistant" ? payload.role : undefined;
    if (!role) {
      return null;
    }
    const content = textFromContentParts(
      payload.content,
      role === "assistant" ? ["output_text"] : ["input_text"],
    );
    return content ? { role, content } : null;
  }
  if (record.type === "event_msg" && payload?.type === "user_message") {
    const content = readString(payload, ["message"]);
    return content ? { role: "user", content } : null;
  }
  if (record.type === "event_msg" && payload?.type === "agent_message") {
    const content = readString(payload, ["message"]);
    return content ? { role: "assistant", content } : null;
  }
  if (record.type === "event_msg" && payload?.type === "task_complete") {
    const content = readString(payload, ["last_agent_message"]);
    return content ? { role: "assistant", content } : null;
  }
  return null;
}

function candidateFromClaudeRecord(record: Record<string, unknown>): TranscriptCandidate | null {
  if (record.type === "user" && isRecord(record.message)) {
    const content = textFromContentParts(record.message.content, ["text"]);
    return content ? { role: "user", content } : null;
  }
  if (record.type === "assistant" && isRecord(record.message)) {
    const content = textFromContentParts(record.message.content, ["text"]);
    return content ? { role: "assistant", content } : null;
  }
  return null;
}

function candidateFromGenericRecord(
  hostId: MemxHostId,
  record: Record<string, unknown>,
): TranscriptCandidate | null {
  if (hostId === "codex") {
    return candidateFromCodexRecord(record);
  }
  if (hostId === "claude-code") {
    return candidateFromClaudeRecord(record);
  }
  const role = record.role === "user" || record.role === "assistant" ? record.role : undefined;
  const content = role ? textFromContentParts(record.content ?? record.message, ["text"]) : "";
  return role && content ? { role, content } : null;
}

async function readTranscriptTail(path: string): Promise<string> {
  const handle = await open(path, "r");
  try {
    const info = await handle.stat();
    const length = Math.min(info.size, MAX_TRANSCRIPT_BYTES);
    const offset = Math.max(0, info.size - length);
    const buffer = Buffer.alloc(length);
    await handle.read(buffer, 0, length, offset);
    const text = buffer.toString("utf8");
    return offset > 0 ? text.slice(text.indexOf("\n") + 1) : text;
  } finally {
    await handle.close();
  }
}

function compactForMatch(text: string): string {
  return text.replace(/\s+/g, " ").trim();
}

function matchesExpectedUser(candidate: TranscriptCandidate, expectedUserText: string | undefined): boolean {
  if (candidate.role !== "user" || !expectedUserText?.trim()) {
    return false;
  }
  const candidateText = compactForMatch(candidate.content);
  const expectedText = compactForMatch(expectedUserText);
  if (!candidateText || !expectedText) {
    return false;
  }
  return candidateText.includes(expectedText) || expectedText.includes(candidateText);
}

function latestAssistantAfterUser(
  candidates: TranscriptCandidate[],
  expectedUserText: string | undefined,
): TranscriptCandidate | null {
  let anchorIndex = -1;
  for (let index = candidates.length - 1; index >= 0; index -= 1) {
    if (matchesExpectedUser(candidates[index]!, expectedUserText)) {
      anchorIndex = index;
      break;
    }
  }
  const searchStart = anchorIndex >= 0 ? anchorIndex + 1 : 0;
  for (let index = candidates.length - 1; index >= searchStart; index -= 1) {
    const candidate = candidates[index]!;
    if (candidate.role === "assistant" && candidate.content.trim()) {
      return candidate;
    }
  }
  return null;
}

async function discoverJsonlFiles(root: string, depth = 0): Promise<Array<{ path: string; mtimeMs: number }>> {
  if (depth > MAX_DISCOVERY_DEPTH) {
    return [];
  }
  let entries: Array<{
    name: string;
    isDirectory(): boolean;
    isFile(): boolean;
  }>;
  try {
    entries = await readdir(root, { withFileTypes: true });
  } catch {
    return [];
  }
  const files: Array<{ path: string; mtimeMs: number }> = [];
  for (const entry of entries) {
    if (entry.name === "cache" || entry.name === "plugins" || entry.name === ".tmp") {
      continue;
    }
    const path = join(root, entry.name);
    if (entry.isDirectory()) {
      files.push(...(await discoverJsonlFiles(path, depth + 1)));
      continue;
    }
    if (!entry.isFile() || !entry.name.endsWith(".jsonl")) {
      continue;
    }
    try {
      const info = await stat(path);
      files.push({ path, mtimeMs: info.mtimeMs });
    } catch {
      // Ignore files that disappear while the host is writing transcripts.
    }
  }
  return files;
}

async function discoverTranscriptPath(hostId: MemxHostId, sessionId: string): Promise<string | undefined> {
  if (process.env["MEMX_TRANSCRIPT_DISCOVERY"] === "0") {
    return undefined;
  }
  const roots =
    hostId === "codex"
      ? [join(process.env["CODEX_HOME"] || join(homedir(), ".codex"), "sessions")]
      : hostId === "claude-code"
        ? [
            join(process.env["CLAUDE_CONFIG_DIR"] || join(homedir(), ".claude"), "projects"),
            join(process.env["CLAUDE_HOME"] || join(homedir(), ".claude"), "projects"),
          ]
        : [];
  const files = (
    await Promise.all(roots.map((root) => discoverJsonlFiles(root)))
  )
    .flat()
    .sort((left, right) => right.mtimeMs - left.mtimeMs)
    .slice(0, MAX_DISCOVERY_FILES);
  const exactSessionMatch = files.find((entry) => basename(entry.path).includes(sessionId))?.path;
  if (exactSessionMatch) {
    return exactSessionMatch;
  }
  if (process.env["MEMX_TRANSCRIPT_DISCOVERY_FALLBACK"] === "1") {
    return files[0]?.path;
  }
  return undefined;
}

export async function extractAssistantFromTranscript(params: {
  hostId: MemxHostId;
  transcriptPath?: string;
  sessionId: string;
  expectedUserText?: string;
}): Promise<TranscriptAssistantCapture | null> {
  const path = params.transcriptPath ?? (await discoverTranscriptPath(params.hostId, params.sessionId));
  if (!path) {
    return null;
  }
  let text: string;
  try {
    text = await readTranscriptTail(path);
  } catch {
    return null;
  }
  const candidates: TranscriptCandidate[] = [];
  for (const line of text.split(/\r?\n/u)) {
    if (!line.trim()) {
      continue;
    }
    let record: unknown;
    try {
      record = JSON.parse(line);
    } catch {
      continue;
    }
    if (!isRecord(record)) {
      continue;
    }
    const candidate = candidateFromGenericRecord(params.hostId, record);
    if (candidate) {
      candidates.push(candidate);
    }
  }
  const assistant = latestAssistantAfterUser(candidates, params.expectedUserText);
  if (!assistant) {
    return null;
  }
  return {
    hostId: params.hostId,
    path,
    message: {
      role: "assistant",
      content: truncateText(assistant.content, 24_000),
    },
  };
}

export async function completeEnvelopeFromTranscript(
  envelope: MemxTurnEnvelope,
  pending?: MemxTurnEnvelope | null,
  options: { timeoutMs?: number; intervalMs?: number } = {},
): Promise<MemxTurnEnvelope> {
  if (envelope.messages.some((message) => message.role === "assistant" && message.content.trim())) {
    return envelope;
  }
  const expectedUserText = [...(pending?.messages ?? [])]
    .reverse()
    .find((message) => message.role === "user" && message.content.trim())?.content;
  const transcriptPath =
    envelope.metadata && typeof envelope.metadata.transcriptPath === "string"
      ? envelope.metadata.transcriptPath
      : pending?.metadata && typeof pending.metadata.transcriptPath === "string"
        ? pending.metadata.transcriptPath
      : undefined;
  const timeoutMs = parsePositiveInt(
    options.timeoutMs !== undefined
      ? String(options.timeoutMs)
      : process.env["MEMX_TRANSCRIPT_CAPTURE_TIMEOUT_MS"],
    MEMX_TRANSCRIPT_CAPTURE_TIMEOUT_MS,
  );
  const intervalMs = parsePositiveInt(
    options.intervalMs !== undefined
      ? String(options.intervalMs)
      : process.env["MEMX_TRANSCRIPT_CAPTURE_INTERVAL_MS"],
    MEMX_TRANSCRIPT_CAPTURE_INTERVAL_MS,
  );
  const startedAt = Date.now();
  let capture: TranscriptAssistantCapture | null = null;
  do {
    capture = await extractAssistantFromTranscript({
      hostId: envelope.hostId,
      transcriptPath,
      sessionId: envelope.sessionId,
      expectedUserText,
    });
    if (capture || timeoutMs === 0) {
      break;
    }
    const remainingMs = timeoutMs - (Date.now() - startedAt);
    if (remainingMs <= 0) {
      break;
    }
    await sleep(Math.min(intervalMs, remainingMs));
  } while (Date.now() - startedAt <= timeoutMs);
  if (!capture) {
    return {
      ...envelope,
      metadata: {
        ...(envelope.metadata ?? {}),
        transcriptAssistantCapture: "missing",
      },
    };
  }
  return {
    ...envelope,
    messages: [...envelope.messages, capture.message],
    metadata: {
      ...(envelope.metadata ?? {}),
      transcriptPath: capture.path,
      transcriptAssistantCapture: capture.hostId,
    },
  };
}
