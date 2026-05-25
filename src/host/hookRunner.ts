import { createHash } from "node:crypto";
import { mkdir, readFile, rename, rm, writeFile } from "node:fs/promises";
import { homedir } from "node:os";
import { join } from "node:path";
import {
  deriveNativeHookHttpTimeoutMs,
  deriveNativeHookQueryCompilerTimeoutMs,
  MEMX_NATIVE_HOOK_TIMEOUT_MS,
} from "../timeouts.js";
import { normalizeHookPayload, type MemxHostId, type MemxTurnEnvelope } from "./hookPayload.js";
import { completeEnvelopeFromTranscript } from "./transcript.js";

const DEFAULT_URL = "http://127.0.0.1:3878";
const CONTEXT_INJECTION_EVENTS = new Set(["UserPromptSubmit"]);

type HookRuntimeConfig = {
  memxUrl?: string;
  memxSecret?: string;
  pendingDir?: string;
  hookTimeoutMs?: number;
  hookContextTimeoutMs?: number;
  hookObserveTimeoutMs?: number;
};

function parseHookArgs(argv: string[]): {
  host: MemxHostId;
  eventName: string;
  hookConfigPath?: string;
} {
  const positional: string[] = [];
  let hookConfigPath = process.env["MEMX_HOOK_CONFIG"]?.trim() || undefined;
  for (let index = 0; index < argv.length; index += 1) {
    const entry = argv[index];
    if (entry === "--hook-config") {
      const next = argv[index + 1]?.trim();
      if (next) {
        hookConfigPath = next;
      }
      index += 1;
      continue;
    }
    positional.push(entry);
  }
  return {
    host: (positional[0] || process.env["MEMX_HOOK_HOST"] || "generic") as MemxHostId,
    eventName: positional[1] || process.env["MEMX_HOOK_EVENT"] || "observe",
    hookConfigPath,
  };
}

function stringSetting(value: unknown): string | undefined {
  return typeof value === "string" && value.trim() ? value.trim() : undefined;
}

function positiveNumberSetting(value: unknown): number | undefined {
  const parsed = typeof value === "number" ? value : typeof value === "string" ? Number(value) : Number.NaN;
  return Number.isFinite(parsed) && parsed > 0 ? parsed : undefined;
}

async function readHookRuntimeConfig(path: string | undefined): Promise<HookRuntimeConfig> {
  if (!path) {
    return {};
  }
  try {
    const parsed = JSON.parse(await readFile(path, "utf8")) as unknown;
    if (!isRecord(parsed)) {
      return {};
    }
    return {
      memxUrl: stringSetting(parsed.memxUrl),
      memxSecret: stringSetting(parsed.memxSecret),
      pendingDir: stringSetting(parsed.pendingDir),
      hookTimeoutMs: positiveNumberSetting(parsed.hookTimeoutMs),
      hookContextTimeoutMs: positiveNumberSetting(parsed.hookContextTimeoutMs),
      hookObserveTimeoutMs: positiveNumberSetting(parsed.hookObserveTimeoutMs),
    };
  } catch (error) {
    debug(`memx hook config ignored: ${error instanceof Error ? error.message : String(error)}`);
    return {};
  }
}

async function readStdinJson(): Promise<Record<string, unknown>> {
  let input = "";
  for await (const chunk of process.stdin) {
    input += chunk;
  }
  if (!input.trim()) {
    return {};
  }
  return JSON.parse(input) as Record<string, unknown>;
}

function authHeaders(config: HookRuntimeConfig): Record<string, string> {
  const secret = process.env["MEMX_SECRET"] || config.memxSecret;
  return secret ? { authorization: `Bearer ${secret}` } : {};
}

async function post(
  path: string,
  body: unknown,
  timeoutMs: number,
  config: HookRuntimeConfig,
): Promise<unknown> {
  const url = (process.env["MEMX_URL"] || config.memxUrl || DEFAULT_URL).replace(/\/+$/u, "");
  const response = await fetch(`${url}${path}`, {
    method: "POST",
    headers: {
      "content-type": "application/json",
      ...authHeaders(config),
    },
    body: JSON.stringify(body),
    signal: AbortSignal.timeout(timeoutMs),
  });
  if (!response.ok) {
    throw new Error(`${path} -> ${response.status} ${response.statusText}`);
  }
  const text = await response.text();
  return text ? JSON.parse(text) : null;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return Boolean(value) && typeof value === "object" && !Array.isArray(value);
}

function hookCanInjectContext(host: MemxHostId, eventName: string): boolean {
  return (host === "codex" || host === "claude-code") && CONTEXT_INJECTION_EVENTS.has(eventName);
}

function hookShouldStorePending(eventName: string): boolean {
  return eventName === "UserPromptSubmit";
}

function hookShouldFlushPending(eventName: string): boolean {
  return eventName === "Stop" || eventName === "SessionEnd";
}

function userQueryFromEnvelope(envelope: ReturnType<typeof normalizeHookPayload>): string | null {
  const userMessage = envelope.messages.find((message) => message.role === "user" && message.content.trim());
  return userMessage?.content.trim() || null;
}

function contextRequestFromEnvelope(
  envelope: ReturnType<typeof normalizeHookPayload>,
  hotPathTimeoutMs: number,
): Record<string, unknown> | null {
  const query = userQueryFromEnvelope(envelope);
  if (!query) {
    return null;
  }
  return {
    query,
    hostId: envelope.hostId,
    actorId: envelope.actorId,
    sessionId: envelope.sessionId,
    workspaceDir: envelope.workspaceDir,
    project: envelope.project,
    limit: 6,
    hotPathTimeoutMs,
  };
}

function recalledContext(response: unknown): string | null {
  if (!isRecord(response)) {
    return null;
  }
  const value = response.prependContext ?? response.context;
  return typeof value === "string" && value.trim() ? value.trim() : null;
}

function writeAdditionalContext(eventName: string, additionalContext: string): void {
  process.stdout.write(
    `${JSON.stringify({
      hookSpecificOutput: {
        hookEventName: eventName,
        additionalContext,
      },
    })}\n`,
  );
}

function parsePositiveInt(value: string | undefined, fallback: number): number {
  const parsed = Number(value);
  return Number.isFinite(parsed) && parsed > 0 ? parsed : fallback;
}

function runtimePositiveInt(
  envValue: string | undefined,
  configValue: number | undefined,
  fallback: number,
): number {
  const parsed = parsePositiveInt(envValue, Number.NaN);
  return Number.isFinite(parsed) && parsed > 0 ? parsed : configValue ?? fallback;
}

function pendingRoot(config: HookRuntimeConfig): string {
  return process.env["MEMX_PENDING_DIR"]?.trim() || config.pendingDir || join(homedir(), ".memx", "pending-hooks");
}

function pendingKey(envelope: Pick<MemxTurnEnvelope, "hostId" | "actorId" | "sessionId" | "workspaceDir">): string {
  return createHash("sha256")
    .update(JSON.stringify([envelope.hostId, envelope.actorId, envelope.sessionId, envelope.workspaceDir ?? ""]))
    .digest("hex");
}

function pendingPath(
  envelope: Pick<MemxTurnEnvelope, "hostId" | "actorId" | "sessionId" | "workspaceDir">,
  config: HookRuntimeConfig,
): string {
  return join(pendingRoot(config), `${pendingKey(envelope)}.json`);
}

async function writePendingTurn(envelope: MemxTurnEnvelope, config: HookRuntimeConfig): Promise<void> {
  if (envelope.messages.length === 0) {
    return;
  }
  const path = pendingPath(envelope, config);
  await mkdir(pendingRoot(config), { recursive: true });
  const tmp = `${path}.tmp-${process.pid}-${Date.now()}`;
  await writeFile(tmp, `${JSON.stringify(envelope)}\n`, "utf8");
  await rename(tmp, path);
}

async function readPendingTurn(
  envelope: MemxTurnEnvelope,
  config: HookRuntimeConfig,
): Promise<MemxTurnEnvelope | null> {
  try {
    const parsed = JSON.parse(await readFile(pendingPath(envelope, config), "utf8")) as unknown;
    if (!parsed || typeof parsed !== "object" || !Array.isArray((parsed as MemxTurnEnvelope).messages)) {
      return null;
    }
    return parsed as MemxTurnEnvelope;
  } catch {
    return null;
  }
}

async function clearPendingTurn(envelope: MemxTurnEnvelope, config: HookRuntimeConfig): Promise<void> {
  await rm(pendingPath(envelope, config), { force: true });
}

function mergePendingTurn(current: MemxTurnEnvelope, pending: MemxTurnEnvelope | null): MemxTurnEnvelope {
  const pendingMessages = pending?.messages ?? [];
  return {
    ...current,
    eventName: pending ? "turn" : current.eventName,
    observedAt: current.observedAt,
    messages: [...pendingMessages, ...current.messages],
    metadata: {
      ...(pending?.metadata ?? {}),
      ...(current.metadata ?? {}),
      pendingHookEvent: pending?.eventName,
      rawHookEvent: current.eventName,
    },
  };
}

function hasAssistantMessage(envelope: MemxTurnEnvelope): boolean {
  return envelope.messages.some(
    (message) => message.role === "assistant" && message.content.trim().length > 0,
  );
}

function debug(message: string): void {
  if (process.env["MEMX_HOOK_DEBUG"] === "1") {
    console.error(message);
  }
}

export async function runMemxHook(argv = process.argv.slice(2)): Promise<void> {
  const { host, eventName, hookConfigPath } = parseHookArgs(argv);
  const runtimeConfig = await readHookRuntimeConfig(hookConfigPath);
  const payload = await readStdinJson();
  const timeoutMs = runtimePositiveInt(
    process.env["MEMX_HOOK_TIMEOUT_MS"],
    runtimeConfig.hookTimeoutMs,
    MEMX_NATIVE_HOOK_TIMEOUT_MS,
  );
  const contextTimeoutMs = runtimePositiveInt(
    process.env["MEMX_HOOK_CONTEXT_TIMEOUT_MS"],
    runtimeConfig.hookContextTimeoutMs,
    deriveNativeHookHttpTimeoutMs(timeoutMs),
  );
  const observeTimeoutMs = runtimePositiveInt(
    process.env["MEMX_HOOK_OBSERVE_TIMEOUT_MS"],
    runtimeConfig.hookObserveTimeoutMs,
    deriveNativeHookHttpTimeoutMs(timeoutMs),
  );
  const queryCompilerTimeoutMs = deriveNativeHookQueryCompilerTimeoutMs(contextTimeoutMs);
  try {
    const envelope = normalizeHookPayload(host, eventName, payload);
    if (hookShouldStorePending(eventName)) {
      await writePendingTurn(envelope, runtimeConfig);
    }
    const contextRequest = hookCanInjectContext(envelope.hostId, eventName)
      ? contextRequestFromEnvelope(envelope, queryCompilerTimeoutMs)
      : null;
    if (contextRequest) {
      // Recall must read the previous memory epoch. The current prompt is only staged locally
      // and is not committed until the host emits a completed assistant turn.
      const contextResult = await Promise.resolve()
        .then(() => post("/v1/context", contextRequest, contextTimeoutMs, runtimeConfig))
        .then(
          (value) => ({ status: "fulfilled" as const, value }),
          (reason) => ({ status: "rejected" as const, reason }),
        );
      if (contextResult.status === "fulfilled") {
        const context = recalledContext(contextResult.value);
        if (context) {
          writeAdditionalContext(eventName, context);
        }
      } else {
        debug(
          `memx hook recall failed: ${
            contextResult.reason instanceof Error ? contextResult.reason.message : String(contextResult.reason)
          }`,
        );
      }
    }

    if (hookShouldStorePending(eventName)) {
      return;
    }

    const shouldFlushPending = hookShouldFlushPending(eventName);
    const pending = shouldFlushPending ? await readPendingTurn(envelope, runtimeConfig) : null;
    if (eventName === "SessionEnd" && !pending) {
      debug("memx hook observe skipped: SessionEnd has no pending user turn");
      return;
    }
    const completedEnvelope = shouldFlushPending
      ? await completeEnvelopeFromTranscript(envelope, pending)
      : envelope;
    const observeEnvelope = shouldFlushPending
      ? mergePendingTurn(completedEnvelope, pending)
      : completedEnvelope;
    if (shouldFlushPending && pending && !hasAssistantMessage(observeEnvelope)) {
      debug("memx hook observe deferred: assistant output is not available yet");
      return;
    }
    if (observeEnvelope.messages.length === 0) {
      return;
    }
    const observeResult = await Promise.resolve()
      .then(() => post("/v1/observe", observeEnvelope, observeTimeoutMs, runtimeConfig))
      .then(
        () => ({ status: "fulfilled" as const }),
        (reason) => ({ status: "rejected" as const, reason }),
      );
    if (observeResult.status === "fulfilled" && shouldFlushPending) {
      await clearPendingTurn(envelope, runtimeConfig);
    }
    if (observeResult.status === "rejected") {
      debug(
        `memx hook observe failed: ${
          observeResult.reason instanceof Error ? observeResult.reason.message : String(observeResult.reason)
        }`,
      );
    }
  } catch (error) {
    debug(`memx hook failed: ${error instanceof Error ? error.message : String(error)}`);
  }
}
