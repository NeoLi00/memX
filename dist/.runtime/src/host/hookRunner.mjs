import { MEMX_NATIVE_HOOK_TIMEOUT_MS, deriveNativeHookBudget, deriveNativeHookQueryCompilerTimeoutMs } from "../timeouts.mjs";
import { normalizeHookPayload } from "./hookPayload.mjs";
import { detectNativeMemoryBypass } from "./nativeMemoryBypass.mjs";
import { completeEnvelopeFromTranscript } from "./transcript.mjs";
import { mkdir, readFile, rename, rm, writeFile } from "node:fs/promises";
import { homedir } from "node:os";
import { join } from "node:path";
import { createHash } from "node:crypto";
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
			hookObserveTimeoutMs: positiveNumberSetting(parsed.hookObserveTimeoutMs),
			hookQueryCompilerTimeoutMs: positiveNumberSetting(parsed.hookQueryCompilerTimeoutMs)
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
function recallEchoTextsFromContext(context) {
	const lines = context.split(/\r?\n/u).map((line) => line.trim()).filter(Boolean).filter((line) => !/^#{1,6}\s+/u.test(line)).map((line) => line.replace(/^\s*(?:[-*]|\d+[.)])\s+/u, "").trim()).filter((line) => line.length >= 12);
	return [...new Set([context.trim(), ...lines])];
}
function injectedRecallMetadata(query, response) {
	const context = recalledContext(response);
	if (!context) return null;
	return { memxRecall: {
		query,
		injectedTexts: recallEchoTextsFromContext(context)
	} };
}
function writeAdditionalContext(eventName, additionalContext) {
	process.stdout.write(`${JSON.stringify({ hookSpecificOutput: {
		hookEventName: eventName,
		additionalContext
	} })}\n`);
}
function writePreToolUseDeny(eventName, reason) {
	process.stdout.write(`${JSON.stringify({ hookSpecificOutput: {
		hookEventName: eventName,
		permissionDecision: "deny",
		permissionDecisionReason: reason
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
	const nextTurns = [...(await readPendingTurns(envelope, config)).filter((pending) => pendingTurnFingerprint(pending) !== pendingTurnFingerprint(envelope)), envelope].slice(-16);
	const latest = nextTurns.at(-1) ?? envelope;
	const tmp = `${path}.tmp-${process.pid}-${Date.now()}`;
	await writeFile(tmp, `${JSON.stringify({
		...latest,
		pendingTurns: nextTurns
	})}\n`, "utf8");
	await rename(tmp, path);
}
function pendingTurnFingerprint(envelope) {
	const userText = envelope.messages.filter((message) => message.role === "user").map((message) => message.content.trim()).join("\n");
	return createHash("sha256").update(JSON.stringify([
		envelope.hostId,
		envelope.actorId,
		envelope.sessionId,
		envelope.workspaceDir ?? "",
		envelope.runId ?? "",
		envelope.observedAt,
		userText
	])).digest("hex");
}
function validPendingEnvelope(value) {
	return Boolean(value && typeof value === "object" && Array.isArray(value.messages));
}
function writePendingTurns(envelope, turns, config) {
	const latest = turns.at(-1);
	if (!latest) return rm(pendingPath(envelope, config), { force: true });
	const path = pendingPath(envelope, config);
	const tmp = `${path}.tmp-${process.pid}-${Date.now()}`;
	return writeFile(tmp, `${JSON.stringify({
		...latest,
		pendingTurns: turns
	})}\n`, "utf8").then(() => rename(tmp, path));
}
async function readPendingTurns(envelope, config) {
	try {
		const parsed = JSON.parse(await readFile(pendingPath(envelope, config), "utf8"));
		if (parsed && typeof parsed === "object" && Array.isArray(parsed.pendingTurns)) return parsed.pendingTurns.filter(validPendingEnvelope);
		if (validPendingEnvelope(parsed)) return [parsed];
		return [];
	} catch {
		return [];
	}
}
async function clearPendingTurn(envelope, completedPending, config) {
	const completed = pendingTurnFingerprint(completedPending);
	await writePendingTurns(envelope, (await readPendingTurns(envelope, config)).filter((pending) => pendingTurnFingerprint(pending) !== completed), config);
}
async function annotatePendingTurn(envelope, config, metadata) {
	const pendingTurns = await readPendingTurns(envelope, config);
	if (pendingTurns.length === 0) return;
	const fingerprint = pendingTurnFingerprint(envelope);
	await writePendingTurns(envelope, pendingTurns.map((pending) => pendingTurnFingerprint(pending) === fingerprint ? {
		...pending,
		metadata: {
			...pending.metadata ?? {},
			...metadata
		}
	} : pending), config);
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
function pendingCompletionProbe(current, pending) {
	return {
		...pending,
		eventName: "deferred_flush",
		observedAt: pending.observedAt,
		messages: [],
		metadata: {
			...pending.metadata ?? {},
			deferredFlushHookEvent: current.eventName
		}
	};
}
async function flushCompletedPendingTurns(envelope, config, observeTimeoutMs, transcriptTimeoutMs) {
	const pendingTurns = await readPendingTurns(envelope, config);
	for (const pending of pendingTurns) {
		const observeEnvelope = mergePendingTurn(await completeEnvelopeFromTranscript(pendingCompletionProbe(envelope, pending), pending, { timeoutMs: transcriptTimeoutMs }), pending);
		if (!hasAssistantMessage(observeEnvelope)) continue;
		const observeResult = await Promise.resolve().then(() => post("/v1/observe", observeEnvelope, observeTimeoutMs, config)).then(() => ({ status: "fulfilled" }), (reason) => ({
			status: "rejected",
			reason
		}));
		if (observeResult.status === "fulfilled") await clearPendingTurn(envelope, pending, config);
		else debug(`memx hook deferred observe failed: ${observeResult.reason instanceof Error ? observeResult.reason.message : String(observeResult.reason)}`);
	}
}
function preservePendingTranscriptMiss(pending, completedEnvelope) {
	const transcriptPath = completedEnvelope.metadata && typeof completedEnvelope.metadata.transcriptPath === "string" ? completedEnvelope.metadata.transcriptPath : pending.metadata && typeof pending.metadata.transcriptPath === "string" ? pending.metadata.transcriptPath : void 0;
	return {
		...pending,
		metadata: {
			...pending.metadata ?? {},
			...transcriptPath ? { transcriptPath } : {},
			transcriptAssistantCapture: "missing"
		}
	};
}
function debug(message) {
	if (process.env["MEMX_HOOK_DEBUG"] === "1") console.error(message);
}
async function runMemxHook(argv = process.argv.slice(2)) {
	const { host, eventName, hookConfigPath } = parseHookArgs(argv);
	const runtimeConfig = await readHookRuntimeConfig(hookConfigPath);
	const payload = await readStdinJson();
	const budget = deriveNativeHookBudget(runtimePositiveInt(process.env["MEMX_HOOK_TIMEOUT_MS"], runtimeConfig.hookTimeoutMs, MEMX_NATIVE_HOOK_TIMEOUT_MS));
	const contextTimeoutMs = runtimePositiveInt(process.env["MEMX_HOOK_CONTEXT_TIMEOUT_MS"], runtimeConfig.hookContextTimeoutMs, budget.contextTimeoutMs);
	const observeTimeoutMs = runtimePositiveInt(process.env["MEMX_HOOK_OBSERVE_TIMEOUT_MS"], runtimeConfig.hookObserveTimeoutMs, budget.observeTimeoutMs);
	const queryCompilerTimeoutMs = runtimePositiveInt(process.env["MEMX_HOOK_QUERY_COMPILER_TIMEOUT_MS"] ?? process.env["MEMX_QUERY_COMPILER_TIMEOUT_MS"], runtimeConfig.hookQueryCompilerTimeoutMs, contextTimeoutMs === budget.contextTimeoutMs ? budget.queryCompilerTimeoutMs : deriveNativeHookQueryCompilerTimeoutMs(contextTimeoutMs));
	try {
		const envelope = normalizeHookPayload(host, eventName, payload);
		const bypassDecision = detectNativeMemoryBypass(envelope.hostId, eventName, payload);
		if (bypassDecision) {
			writePreToolUseDeny(eventName, bypassDecision.reason);
			return;
		}
		if (eventName === "SessionStart" || hookShouldStorePending(eventName)) await flushCompletedPendingTurns(envelope, runtimeConfig, observeTimeoutMs, 0);
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
				if (context) {
					const metadata = injectedRecallMetadata(String(contextRequest.query ?? ""), contextResult.value);
					if (metadata) await annotatePendingTurn(envelope, runtimeConfig, metadata);
					writeAdditionalContext(eventName, context);
				}
			} else debug(`memx hook recall failed: ${contextResult.reason instanceof Error ? contextResult.reason.message : String(contextResult.reason)}`);
		}
		if (hookShouldStorePending(eventName)) return;
		const shouldFlushPending = hookShouldFlushPending(eventName);
		const pending = (shouldFlushPending ? await readPendingTurns(envelope, runtimeConfig) : []).at(-1) ?? null;
		if (eventName === "SessionEnd" && !pending) {
			debug("memx hook observe skipped: SessionEnd has no pending user turn");
			return;
		}
		const completedEnvelope = shouldFlushPending ? await completeEnvelopeFromTranscript(envelope, pending) : envelope;
		const observeEnvelope = shouldFlushPending ? mergePendingTurn(completedEnvelope, pending) : completedEnvelope;
		if (shouldFlushPending && pending && !hasAssistantMessage(observeEnvelope)) {
			await writePendingTurn(preservePendingTranscriptMiss(pending, completedEnvelope), runtimeConfig);
			debug("memx hook observe deferred: assistant output is not available yet");
			return;
		}
		if (observeEnvelope.messages.length === 0) return;
		const observeResult = await Promise.resolve().then(() => post("/v1/observe", observeEnvelope, observeTimeoutMs, runtimeConfig)).then(() => ({ status: "fulfilled" }), (reason) => ({
			status: "rejected",
			reason
		}));
		if (observeResult.status === "fulfilled" && shouldFlushPending && pending) await clearPendingTurn(envelope, pending, runtimeConfig);
		if (observeResult.status === "rejected") debug(`memx hook observe failed: ${observeResult.reason instanceof Error ? observeResult.reason.message : String(observeResult.reason)}`);
	} catch (error) {
		debug(`memx hook failed: ${error instanceof Error ? error.message : String(error)}`);
	}
}
//#endregion
export { runMemxHook };
