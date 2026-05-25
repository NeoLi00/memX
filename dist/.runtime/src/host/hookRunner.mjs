import { normalizeHookPayload } from "./hookPayload.mjs";
import { MEMX_NATIVE_HOOK_TIMEOUT_MS, deriveNativeHookHttpTimeoutMs, deriveNativeHookQueryCompilerTimeoutMs } from "../timeouts.mjs";
import { completeEnvelopeFromTranscript } from "./transcript.mjs";
import { homedir } from "node:os";
import { join } from "node:path";
import { createHash } from "node:crypto";
import { mkdir, readFile, rename, rm, writeFile } from "node:fs/promises";
//#region src/host/hookRunner.ts
const DEFAULT_URL = "http://127.0.0.1:3878";
const CONTEXT_INJECTION_EVENTS = new Set(["UserPromptSubmit"]);
function parseHookArgs(argv) {
	const positional = [];
	let hookConfigPath = process.env["MEMX_HOOK_CONFIG"]?.trim() || void 0;
	for (let index = 0; index < argv.length; index += 1) {
		const entry = argv[index];
		if (entry === "--hook-config") {
			const next = argv[index + 1]?.trim();
			if (next) hookConfigPath = next;
			index += 1;
			continue;
		}
		positional.push(entry);
	}
	return {
		host: positional[0] || process.env["MEMX_HOOK_HOST"] || "generic",
		eventName: positional[1] || process.env["MEMX_HOOK_EVENT"] || "observe",
		hookConfigPath
	};
}
function stringSetting(value) {
	return typeof value === "string" && value.trim() ? value.trim() : void 0;
}
function positiveNumberSetting(value) {
	const parsed = typeof value === "number" ? value : typeof value === "string" ? Number(value) : NaN;
	return Number.isFinite(parsed) && parsed > 0 ? parsed : void 0;
}
async function readHookRuntimeConfig(path) {
	if (!path) return {};
	try {
		const parsed = JSON.parse(await readFile(path, "utf8"));
		if (!isRecord(parsed)) return {};
		return {
			memxUrl: stringSetting(parsed.memxUrl),
			memxSecret: stringSetting(parsed.memxSecret),
			pendingDir: stringSetting(parsed.pendingDir),
			hookTimeoutMs: positiveNumberSetting(parsed.hookTimeoutMs),
			hookContextTimeoutMs: positiveNumberSetting(parsed.hookContextTimeoutMs),
			hookObserveTimeoutMs: positiveNumberSetting(parsed.hookObserveTimeoutMs)
		};
	} catch (error) {
		debug(`memx hook config ignored: ${error instanceof Error ? error.message : String(error)}`);
		return {};
	}
}
async function readStdinJson() {
	let input = "";
	for await (const chunk of process.stdin) input += chunk;
	if (!input.trim()) return {};
	return JSON.parse(input);
}
function authHeaders(config) {
	const secret = process.env["MEMX_SECRET"] || config.memxSecret;
	return secret ? { authorization: `Bearer ${secret}` } : {};
}
async function post(path, body, timeoutMs, config) {
	const url = (process.env["MEMX_URL"] || config.memxUrl || DEFAULT_URL).replace(/\/+$/u, "");
	const response = await fetch(`${url}${path}`, {
		method: "POST",
		headers: {
			"content-type": "application/json",
			...authHeaders(config)
		},
		body: JSON.stringify(body),
		signal: AbortSignal.timeout(timeoutMs)
	});
	if (!response.ok) throw new Error(`${path} -> ${response.status} ${response.statusText}`);
	const text = await response.text();
	return text ? JSON.parse(text) : null;
}
function isRecord(value) {
	return Boolean(value) && typeof value === "object" && !Array.isArray(value);
}
function hookCanInjectContext(host, eventName) {
	return (host === "codex" || host === "claude-code") && CONTEXT_INJECTION_EVENTS.has(eventName);
}
function hookShouldStorePending(eventName) {
	return eventName === "UserPromptSubmit";
}
function hookShouldFlushPending(eventName) {
	return eventName === "Stop" || eventName === "SessionEnd";
}
function userQueryFromEnvelope(envelope) {
	return envelope.messages.find((message) => message.role === "user" && message.content.trim())?.content.trim() || null;
}
function contextRequestFromEnvelope(envelope, hotPathTimeoutMs) {
	const query = userQueryFromEnvelope(envelope);
	if (!query) return null;
	return {
		query,
		hostId: envelope.hostId,
		actorId: envelope.actorId,
		sessionId: envelope.sessionId,
		workspaceDir: envelope.workspaceDir,
		project: envelope.project,
		limit: 6,
		hotPathTimeoutMs
	};
}
function recalledContext(response) {
	if (!isRecord(response)) return null;
	const value = response.prependContext ?? response.context;
	return typeof value === "string" && value.trim() ? value.trim() : null;
}
function writeAdditionalContext(eventName, additionalContext) {
	process.stdout.write(`${JSON.stringify({ hookSpecificOutput: {
		hookEventName: eventName,
		additionalContext
	} })}\n`);
}
function parsePositiveInt(value, fallback) {
	const parsed = Number(value);
	return Number.isFinite(parsed) && parsed > 0 ? parsed : fallback;
}
function runtimePositiveInt(envValue, configValue, fallback) {
	const parsed = parsePositiveInt(envValue, NaN);
	return Number.isFinite(parsed) && parsed > 0 ? parsed : configValue ?? fallback;
}
function pendingRoot(config) {
	return process.env["MEMX_PENDING_DIR"]?.trim() || config.pendingDir || join(homedir(), ".memx", "pending-hooks");
}
function pendingKey(envelope) {
	return createHash("sha256").update(JSON.stringify([
		envelope.hostId,
		envelope.actorId,
		envelope.sessionId,
		envelope.workspaceDir ?? ""
	])).digest("hex");
}
function pendingPath(envelope, config) {
	return join(pendingRoot(config), `${pendingKey(envelope)}.json`);
}
async function writePendingTurn(envelope, config) {
	if (envelope.messages.length === 0) return;
	const path = pendingPath(envelope, config);
	await mkdir(pendingRoot(config), { recursive: true });
	const tmp = `${path}.tmp-${process.pid}-${Date.now()}`;
	await writeFile(tmp, `${JSON.stringify(envelope)}\n`, "utf8");
	await rename(tmp, path);
}
async function readPendingTurn(envelope, config) {
	try {
		const parsed = JSON.parse(await readFile(pendingPath(envelope, config), "utf8"));
		if (!parsed || typeof parsed !== "object" || !Array.isArray(parsed.messages)) return null;
		return parsed;
	} catch {
		return null;
	}
}
async function clearPendingTurn(envelope, config) {
	await rm(pendingPath(envelope, config), { force: true });
}
function mergePendingTurn(current, pending) {
	const pendingMessages = pending?.messages ?? [];
	return {
		...current,
		eventName: pending ? "turn" : current.eventName,
		observedAt: current.observedAt,
		messages: [...pendingMessages, ...current.messages],
		metadata: {
			...pending?.metadata ?? {},
			...current.metadata ?? {},
			pendingHookEvent: pending?.eventName,
			rawHookEvent: current.eventName
		}
	};
}
function hasAssistantMessage(envelope) {
	return envelope.messages.some((message) => message.role === "assistant" && message.content.trim().length > 0);
}
function debug(message) {
	if (process.env["MEMX_HOOK_DEBUG"] === "1") console.error(message);
}
async function runMemxHook(argv = process.argv.slice(2)) {
	const { host, eventName, hookConfigPath } = parseHookArgs(argv);
	const runtimeConfig = await readHookRuntimeConfig(hookConfigPath);
	const payload = await readStdinJson();
	const timeoutMs = runtimePositiveInt(process.env["MEMX_HOOK_TIMEOUT_MS"], runtimeConfig.hookTimeoutMs, MEMX_NATIVE_HOOK_TIMEOUT_MS);
	const contextTimeoutMs = runtimePositiveInt(process.env["MEMX_HOOK_CONTEXT_TIMEOUT_MS"], runtimeConfig.hookContextTimeoutMs, deriveNativeHookHttpTimeoutMs(timeoutMs));
	const observeTimeoutMs = runtimePositiveInt(process.env["MEMX_HOOK_OBSERVE_TIMEOUT_MS"], runtimeConfig.hookObserveTimeoutMs, deriveNativeHookHttpTimeoutMs(timeoutMs));
	const queryCompilerTimeoutMs = deriveNativeHookQueryCompilerTimeoutMs(contextTimeoutMs);
	try {
		const envelope = normalizeHookPayload(host, eventName, payload);
		if (hookShouldStorePending(eventName)) await writePendingTurn(envelope, runtimeConfig);
		const contextRequest = hookCanInjectContext(envelope.hostId, eventName) ? contextRequestFromEnvelope(envelope, queryCompilerTimeoutMs) : null;
		if (contextRequest) {
			const contextResult = await Promise.resolve().then(() => post("/v1/context", contextRequest, contextTimeoutMs, runtimeConfig)).then((value) => ({
				status: "fulfilled",
				value
			}), (reason) => ({
				status: "rejected",
				reason
			}));
			if (contextResult.status === "fulfilled") {
				const context = recalledContext(contextResult.value);
				if (context) writeAdditionalContext(eventName, context);
			} else debug(`memx hook recall failed: ${contextResult.reason instanceof Error ? contextResult.reason.message : String(contextResult.reason)}`);
		}
		if (hookShouldStorePending(eventName)) return;
		const shouldFlushPending = hookShouldFlushPending(eventName);
		const pending = shouldFlushPending ? await readPendingTurn(envelope, runtimeConfig) : null;
		if (eventName === "SessionEnd" && !pending) {
			debug("memx hook observe skipped: SessionEnd has no pending user turn");
			return;
		}
		const completedEnvelope = shouldFlushPending ? await completeEnvelopeFromTranscript(envelope, pending) : envelope;
		const observeEnvelope = shouldFlushPending ? mergePendingTurn(completedEnvelope, pending) : completedEnvelope;
		if (shouldFlushPending && pending && !hasAssistantMessage(observeEnvelope)) {
			debug("memx hook observe deferred: assistant output is not available yet");
			return;
		}
		if (observeEnvelope.messages.length === 0) return;
		const observeResult = await Promise.resolve().then(() => post("/v1/observe", observeEnvelope, observeTimeoutMs, runtimeConfig)).then(() => ({ status: "fulfilled" }), (reason) => ({
			status: "rejected",
			reason
		}));
		if (observeResult.status === "fulfilled" && shouldFlushPending) await clearPendingTurn(envelope, runtimeConfig);
		if (observeResult.status === "rejected") debug(`memx hook observe failed: ${observeResult.reason instanceof Error ? observeResult.reason.message : String(observeResult.reason)}`);
	} catch (error) {
		debug(`memx hook failed: ${error instanceof Error ? error.message : String(error)}`);
	}
}
//#endregion
export { runMemxHook };
