import { cosineSimilarity, orderByScore } from "../../support.mjs";
import { SqliteFtsBackend } from "./ftsBackend.mjs";
import { mergeHybridHits } from "./hybrid.mjs";
import { mkdir, readFile, readdir, rm, stat, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { createHash, randomUUID } from "node:crypto";
import { spawn } from "node:child_process";
import { fileURLToPath } from "node:url";
//#region src/search/backends/embeddingBackend.ts
var LocalEmbeddingTimeoutError = class extends Error {
	constructor(message) {
		super(message);
		this.name = "LocalEmbeddingTimeoutError";
	}
};
var LocalEmbeddingHardFailureError = class extends Error {
	constructor(message) {
		super(message);
		this.name = "LocalEmbeddingHardFailureError";
	}
};
const DEFAULT_LOCAL_MODEL = "intfloat/multilingual-e5-small";
const DEFAULT_LOCAL_PYTHON_BIN = "python3";
const DEFAULT_LOCAL_DEVICE = "auto";
const LOCAL_EMBEDDING_REQUEST_TIMEOUT_MS = 12e4;
const LOCAL_EMBEDDING_COLD_START_TIMEOUT_MS = 3e5;
const LOCAL_EMBEDDING_PREWARM_TEXT = "memx local embedding warmup";
const QUERY_EMBEDDING_CACHE_LIMIT = 128;
const LOCAL_EMBEDDING_REGISTRY_DIR = join(tmpdir(), "memx-embedder-registry");
const LEGACY_EMBEDDING_STATE_MAX_AGE_MS = 600 * 1e3;
const LOCAL_WORKER_PATH = fileURLToPath(new URL("../../../sentence_transformers_embedder.py", import.meta.url));
async function runCommandWithTimeout(commandAndArgs, options) {
	const [command, ...args] = commandAndArgs;
	if (!command) throw new Error("missing command");
	return new Promise((resolve, reject) => {
		const child = spawn(command, args, {
			shell: false,
			stdio: [
				"ignore",
				"pipe",
				"pipe"
			]
		});
		let stdout = "";
		let stderr = "";
		let settled = false;
		let timeout;
		let noOutputTimeout;
		const clearTimers = () => {
			if (timeout) clearTimeout(timeout);
			if (noOutputTimeout) clearTimeout(noOutputTimeout);
		};
		const finish = (result) => {
			if (settled) return;
			settled = true;
			clearTimers();
			resolve(result);
		};
		const resetNoOutputTimeout = () => {
			if (!options.noOutputTimeoutMs) return;
			if (noOutputTimeout) clearTimeout(noOutputTimeout);
			noOutputTimeout = setTimeout(() => {
				child.kill("SIGTERM");
				finish({
					code: null,
					stdout,
					stderr,
					termination: "no-output-timeout"
				});
			}, options.noOutputTimeoutMs);
			noOutputTimeout.unref();
		};
		child.stdout?.on("data", (chunk) => {
			stdout += chunk.toString("utf8");
			resetNoOutputTimeout();
		});
		child.stderr?.on("data", (chunk) => {
			stderr += chunk.toString("utf8");
			resetNoOutputTimeout();
		});
		child.once("error", (error) => {
			if (settled) return;
			clearTimers();
			reject(error);
		});
		child.once("close", (code) => finish({
			code,
			stdout,
			stderr
		}));
		timeout = setTimeout(() => {
			child.kill("SIGTERM");
			finish({
				code: null,
				stdout,
				stderr,
				termination: "timeout"
			});
		}, options.timeoutMs);
		timeout.unref();
		resetNoOutputTimeout();
	});
}
async function fetchJson(url, init) {
	const response = await fetch(url, init);
	if (!response.ok) throw new Error(`HTTP ${response.status} ${response.statusText}`);
	return response.json();
}
function errorCode(error) {
	return typeof error === "object" && error !== null && "code" in error ? String(error.code) : void 0;
}
function isProcessAlive(pid) {
	if (!Number.isInteger(pid) || (pid ?? 0) <= 0) return false;
	try {
		process.kill(pid, 0);
		return true;
	} catch (error) {
		return errorCode(error) === "EPERM";
	}
}
async function sleep(ms) {
	await new Promise((resolve) => setTimeout(resolve, ms));
}
async function waitForProcessExit(pid, timeoutMs) {
	const deadline = Date.now() + timeoutMs;
	while (Date.now() < deadline) {
		if (!isProcessAlive(pid)) return true;
		await sleep(50);
	}
	return !isProcessAlive(pid);
}
function killProcessBestEffort(pid, signal) {
	try {
		process.kill(pid, signal);
	} catch (error) {
		if (errorCode(error) !== "ESRCH") {}
	}
}
function resolveLocalModel(config) {
	return config.model?.trim() || DEFAULT_LOCAL_MODEL;
}
function resolveLocalPythonBin(config) {
	return config.localPythonBin?.trim() || DEFAULT_LOCAL_PYTHON_BIN;
}
function resolveLocalDevice(config) {
	return config.localDevice?.trim() || DEFAULT_LOCAL_DEVICE;
}
function localEmbeddingWorkerRegistryDir() {
	return process.env["MEMX_EMBEDDING_WORKER_REGISTRY_DIR"]?.trim() || LOCAL_EMBEDDING_REGISTRY_DIR;
}
function shortHash(value) {
	return createHash("sha256").update(value).digest("hex").slice(0, 20);
}
function localEmbeddingWorkerStatePath(config, registryDir = localEmbeddingWorkerRegistryDir(), ownerPid = process.pid) {
	return join(registryDir, `memx-embedder-${shortHash(localWorkerPoolKey(config))}-${ownerPid}.json`);
}
function parseWorkerState(value) {
	try {
		const parsed = JSON.parse(value);
		if (!parsed || typeof parsed !== "object") return null;
		const record = parsed;
		const url = typeof record.url === "string" ? record.url.trim() : "";
		const token = typeof record.token === "string" ? record.token.trim() : "";
		const pid = typeof record.pid === "number" && Number.isInteger(record.pid) ? record.pid : void 0;
		const ownerPid = typeof record.ownerPid === "number" && Number.isInteger(record.ownerPid) ? record.ownerPid : typeof record.parentPid === "number" && Number.isInteger(record.parentPid) ? record.parentPid : void 0;
		if (!url || !token) return null;
		return {
			url: url.replace(/\/$/, ""),
			token,
			pid,
			ownerPid,
			workerKey: typeof record.workerKey === "string" ? record.workerKey : void 0
		};
	} catch {
		return null;
	}
}
async function readWorkerStateFile(path) {
	try {
		const state = parseWorkerState(await readFile(path, "utf8"));
		return state ? {
			...state,
			stateFile: path
		} : null;
	} catch {
		return null;
	}
}
async function requestWorkerShutdown(server) {
	if (!server.url || !server.token) return false;
	const controller = new AbortController();
	const timer = setTimeout(() => controller.abort(), 1500);
	try {
		await fetch(`${server.url}/shutdown`, {
			method: "POST",
			headers: { "x-memx-token": server.token },
			signal: controller.signal
		});
		return true;
	} catch {
		return false;
	} finally {
		clearTimeout(timer);
	}
}
async function removeWorkerStateFile(path) {
	try {
		await rm(path, { force: true });
		return true;
	} catch {
		return false;
	}
}
async function listWorkerStateFiles(dir) {
	try {
		return (await readdir(dir)).filter((entry) => entry.startsWith("memx-embedder-") && entry.endsWith(".json")).map((entry) => join(dir, entry));
	} catch {
		return [];
	}
}
async function stateFileAgeMs(path) {
	try {
		const info = await stat(path);
		return Math.max(0, Date.now() - info.mtimeMs);
	} catch {
		return Number.POSITIVE_INFINITY;
	}
}
function legacyWorkerStateDirs(registryDir, explicitDirs) {
	const candidates = explicitDirs ?? [tmpdir()];
	return [...new Set(candidates.filter((dir) => dir && dir !== registryDir))];
}
async function cleanupWorkerStateFile(params) {
	params.stats.checkedStateFiles += 1;
	const state = await readWorkerStateFile(params.path);
	if (!state) {
		if (await removeWorkerStateFile(params.path)) params.stats.removedStateFiles += 1;
		return;
	}
	const ownerPid = state.ownerPid;
	const ownerAlive = ownerPid ? isProcessAlive(ownerPid) : false;
	const workerAlive = isProcessAlive(state.pid);
	const legacyFreshEnough = params.mode === "legacy" && !ownerPid && await stateFileAgeMs(params.path) < params.legacyStateMaxAgeMs;
	if (params.mode === "legacy" && !state.pid && !ownerPid) {
		if (legacyFreshEnough) return;
		if (await removeWorkerStateFile(params.path)) params.stats.removedStateFiles += 1;
		return;
	}
	if (!workerAlive) {
		if (await removeWorkerStateFile(params.path)) params.stats.removedStateFiles += 1;
		return;
	}
	if (ownerPid === params.currentPid || ownerAlive || legacyFreshEnough) return;
	const shutdownRequested = await requestWorkerShutdown(state);
	if (state.pid && !await waitForProcessExit(state.pid, 1500) && isProcessAlive(state.pid)) {
		killProcessBestEffort(state.pid, "SIGTERM");
		if (!await waitForProcessExit(state.pid, 1e3) && isProcessAlive(state.pid)) killProcessBestEffort(state.pid, "SIGKILL");
	}
	if (shutdownRequested) params.stats.stoppedWorkers += 1;
	if (await removeWorkerStateFile(params.path)) params.stats.removedStateFiles += 1;
}
async function cleanupStaleLocalEmbeddingWorkers(params = {}) {
	const registryDir = params.registryDir ?? localEmbeddingWorkerRegistryDir();
	const currentPid = params.currentPid ?? process.pid;
	const legacyStateMaxAgeMs = Math.max(0, params.legacyStateMaxAgeMs ?? LEGACY_EMBEDDING_STATE_MAX_AGE_MS);
	const stats = {
		checkedStateFiles: 0,
		removedStateFiles: 0,
		stoppedWorkers: 0
	};
	const registryFiles = await listWorkerStateFiles(registryDir);
	const legacyFiles = (await Promise.all(legacyWorkerStateDirs(registryDir, params.legacyStateDirs).map((dir) => listWorkerStateFiles(dir)))).flat();
	const fileModes = /* @__PURE__ */ new Map();
	for (const path of legacyFiles) fileModes.set(path, "legacy");
	for (const path of registryFiles) fileModes.set(path, "registry");
	await Promise.all([...fileModes.entries()].map(async ([path, mode]) => {
		await cleanupWorkerStateFile({
			path,
			mode,
			currentPid,
			legacyStateMaxAgeMs,
			stats
		});
	}));
	if (stats.removedStateFiles > 0 || stats.stoppedWorkers > 0) params.logger?.debug?.(`memx: cleaned stale local embedding workers stateFiles=${stats.removedStateFiles} stopped=${stats.stoppedWorkers}`);
	return stats;
}
var LocalSentenceTransformerWorker = class {
	config;
	logger;
	hasCompletedRequest = false;
	closed = false;
	server = null;
	starting = null;
	constructor(config, logger) {
		this.config = config;
		this.logger = logger;
	}
	async embed(texts, mode) {
		if (texts.length === 0) return [];
		if (this.closed) throw new Error("local sentence-transformers worker is closed");
		const timeoutMs = this.hasCompletedRequest ? LOCAL_EMBEDDING_REQUEST_TIMEOUT_MS : LOCAL_EMBEDDING_COLD_START_TIMEOUT_MS;
		const server = await this.ensureServer(timeoutMs);
		const embeddings = await this.requestEmbedding(server, texts, mode, timeoutMs);
		this.hasCompletedRequest = true;
		return embeddings;
	}
	isWarm() {
		return Boolean(this.server) && this.hasCompletedRequest && !this.closed;
	}
	isClosed() {
		return this.closed;
	}
	async close() {
		this.closed = true;
		await this.stopServer();
	}
	async ensureServer(timeoutMs) {
		if (this.closed) throw new Error("local sentence-transformers worker is closed");
		if (this.server) return this.server;
		this.starting ??= this.launchServer(timeoutMs).finally(() => {
			this.starting = null;
		});
		this.server = await this.starting;
		return this.server;
	}
	async launchServer(timeoutMs) {
		const token = randomUUID();
		const registryDir = localEmbeddingWorkerRegistryDir();
		await cleanupStaleLocalEmbeddingWorkers({
			registryDir,
			logger: this.logger
		});
		const stateFile = localEmbeddingWorkerStatePath(this.config, registryDir);
		await mkdir(dirname(stateFile), { recursive: true });
		await rm(stateFile, { force: true });
		const workerKey = localWorkerPoolKey(this.config);
		const args = [
			LOCAL_WORKER_PATH,
			"--launch-server",
			"--model",
			resolveLocalModel(this.config),
			"--device",
			resolveLocalDevice(this.config),
			"--token",
			token,
			"--state-file",
			stateFile,
			"--parent-pid",
			String(process.pid)
		];
		if (this.config.localCacheDir?.trim()) args.push("--cache-dir", this.config.localCacheDir.trim());
		const result = await runCommandWithTimeout([resolveLocalPythonBin(this.config), ...args], {
			timeoutMs,
			noOutputTimeoutMs: timeoutMs
		});
		if (result.termination === "timeout" || result.termination === "no-output-timeout") throw new LocalEmbeddingTimeoutError(`local sentence-transformers server startup timed out (${timeoutMs}ms)`);
		if (result.code !== 0) {
			const suffix = result.stderr.trim() ? `; stderr: ${result.stderr.trim()}` : "";
			throw new LocalEmbeddingHardFailureError(`local sentence-transformers server launcher exited with code ${result.code ?? -1}${suffix}`);
		}
		const line = result.stdout.split(/\r?\n/).map((entry) => entry.trim()).filter(Boolean).at(-1);
		if (!line) throw new LocalEmbeddingHardFailureError("local sentence-transformers server launcher returned no output");
		let response;
		try {
			response = JSON.parse(line);
		} catch (error) {
			throw new LocalEmbeddingHardFailureError(`local sentence-transformers server launcher returned invalid JSON (${String(error)})`);
		}
		if (response.error) throw new LocalEmbeddingHardFailureError(response.error);
		if (!response.url || response.token !== token) throw new LocalEmbeddingHardFailureError("local sentence-transformers server launcher returned invalid metadata");
		const server = {
			url: response.url.replace(/\/$/, ""),
			token: response.token,
			pid: response.pid,
			ownerPid: process.pid,
			stateFile,
			workerKey
		};
		await writeFile(stateFile, `${JSON.stringify({
			...server,
			updatedAt: (/* @__PURE__ */ new Date()).toISOString()
		})}\n`, "utf8");
		return server;
	}
	async requestEmbedding(server, texts, mode, timeoutMs) {
		const controller = new AbortController();
		const timer = setTimeout(() => controller.abort(), timeoutMs);
		try {
			const response = await fetch(`${server.url}/embed`, {
				method: "POST",
				headers: {
					"content-type": "application/json",
					"x-memx-token": server.token
				},
				body: JSON.stringify({
					mode,
					texts
				}),
				signal: controller.signal
			});
			if (!response.ok) throw new LocalEmbeddingHardFailureError(`local sentence-transformers server returned HTTP ${response.status}`);
			const json = await response.json();
			if (json.error) throw new LocalEmbeddingHardFailureError(json.error);
			return json.embeddings ?? [];
		} catch (error) {
			if (error instanceof LocalEmbeddingHardFailureError) throw error;
			if (error instanceof Error && error.name === "AbortError") {
				if (this.server === server) this.server = null;
				this.stopServerInstance(server).catch((shutdownError) => {
					this.logger.debug?.(`memx: failed to stop timed-out local embedding server (${String(shutdownError)})`);
				});
				throw new LocalEmbeddingTimeoutError(`local sentence-transformers request timed out (${mode}, ${timeoutMs}ms)`);
			}
			if (this.server === server) this.server = null;
			this.stopServerInstance(server).catch((shutdownError) => {
				this.logger.debug?.(`memx: failed to stop failed local embedding server (${String(shutdownError)})`);
			});
			throw new LocalEmbeddingHardFailureError(String(error));
		} finally {
			clearTimeout(timer);
		}
	}
	async stopServer() {
		const server = this.server;
		this.server = null;
		if (!server) return;
		await this.stopServerInstance(server);
	}
	async stopServerInstance(server) {
		const controller = new AbortController();
		const timer = setTimeout(() => controller.abort(), 2e3);
		try {
			await fetch(`${server.url}/shutdown`, {
				method: "POST",
				headers: { "x-memx-token": server.token },
				signal: controller.signal
			});
		} catch {} finally {
			clearTimeout(timer);
		}
		if (!server.pid || !await waitForProcessExit(server.pid, 1500)) {
			if (server.pid && isProcessAlive(server.pid)) {
				killProcessBestEffort(server.pid, "SIGTERM");
				if (!await waitForProcessExit(server.pid, 1e3) && isProcessAlive(server.pid)) killProcessBestEffort(server.pid, "SIGKILL");
			}
		}
		if (server.stateFile) await removeWorkerStateFile(server.stateFile);
	}
};
const LOCAL_WORKER_POOL = /* @__PURE__ */ new Map();
function localWorkerPoolKey(config) {
	return JSON.stringify({
		pythonBin: resolveLocalPythonBin(config),
		model: resolveLocalModel(config),
		device: resolveLocalDevice(config),
		cacheDir: config.localCacheDir?.trim() || ""
	});
}
function acquireLocalEmbeddingWorker(config, logger) {
	const key = localWorkerPoolKey(config);
	let entry = LOCAL_WORKER_POOL.get(key);
	if (!entry || entry.worker.isClosed()) {
		entry = {
			worker: new LocalSentenceTransformerWorker(config, logger),
			refs: 0
		};
		LOCAL_WORKER_POOL.set(key, entry);
	}
	entry.refs++;
	let released = false;
	const release = async (force = false) => {
		if (released && !force) return;
		released = true;
		const current = LOCAL_WORKER_POOL.get(key);
		if (!current || current.worker !== entry.worker) return;
		if (force) {
			LOCAL_WORKER_POOL.delete(key);
			current.refs = 0;
			await current.worker.close();
			return;
		}
		current.refs = Math.max(0, current.refs - 1);
		if (current.refs === 0) {
			LOCAL_WORKER_POOL.delete(key);
			await current.worker.close();
		}
	};
	return {
		worker: entry.worker,
		release: () => release(false),
		invalidate: () => release(true)
	};
}
async function embedTextsRemote(config, texts) {
	if (config.provider === "openai-compatible") {
		const url = `${config.baseURL?.replace(/\/$/, "") ?? ""}/embeddings`;
		const body = {
			model: config.model,
			input: texts,
			...config.dimensions ? { dimensions: config.dimensions } : {}
		};
		const headers = {
			"content-type": "application/json",
			...config.headers ?? {}
		};
		if (config.apiKey) headers.authorization = `Bearer ${config.apiKey}`;
		return ((await fetchJson(url, {
			method: "POST",
			headers,
			body: JSON.stringify(body)
		})).data ?? []).map((entry) => entry.embedding ?? []);
	}
	return await Promise.all(texts.map(async (text) => {
		return (await fetchJson(`${config.ollamaBaseURL?.replace(/\/$/, "") ?? "http://127.0.0.1:11434"}/api/embeddings`, {
			method: "POST",
			headers: { "content-type": "application/json" },
			body: JSON.stringify({
				model: config.ollamaModel ?? config.model,
				prompt: text
			})
		})).embedding ?? [];
	}));
}
var OptionalEmbeddingBackend = class {
	repo;
	embedding;
	logger;
	lexical;
	localWorkerLease;
	localWorker;
	queryEmbeddingCache = /* @__PURE__ */ new Map();
	warnedUnavailable = false;
	acceptingUpserts = true;
	localUnavailableForProcess = false;
	closed = false;
	upsertQueue = Promise.resolve();
	localPrewarm = null;
	constructor(repo, embedding, logger) {
		this.repo = repo;
		this.embedding = embedding;
		this.logger = logger;
		this.lexical = new SqliteFtsBackend(repo);
		this.localWorkerLease = embedding.provider === "sentence-transformers-local" ? acquireLocalEmbeddingWorker(embedding, logger) : null;
		this.localWorker = this.localWorkerLease?.worker ?? null;
	}
	upsertDocs(docs) {
		this.repo.upsertDocs(docs);
		if (this.embedding.provider === "off" || !this.acceptingUpserts || this.closed || this.isEmbeddingDisabledForProcess()) return;
		const eligibleDocs = docs.filter((doc) => doc.text.trim().length > 0);
		if (eligibleDocs.length === 0) return;
		this.upsertQueue = this.upsertQueue.then(async () => {
			const vectors = await this.embedTexts(eligibleDocs.map((doc) => doc.text), "passage", "vector-upsert");
			if (this.closed) return;
			for (const [index, doc] of eligibleDocs.entries()) {
				const vector = vectors[index] ?? [];
				if (vector.length === 0) continue;
				this.repo.upsertEmbedding({
					docId: doc.docId,
					agentId: doc.agentId,
					scope: doc.scope,
					embedding: vector,
					updatedAt: doc.updatedAt
				});
			}
		}).catch((error) => {
			this.handleEmbeddingFailure(error, `memx: embeddings unavailable, using lexical retrieval only (${String(error)})`);
		});
	}
	async flushPendingUpserts() {
		let pending = this.upsertQueue;
		while (true) {
			await pending;
			if (pending === this.upsertQueue) return;
			pending = this.upsertQueue;
		}
	}
	async close() {
		if (this.closed) return;
		this.acceptingUpserts = false;
		this.queryEmbeddingCache.clear();
		await this.flushPendingUpserts();
		await this.localPrewarm?.catch(() => {});
		this.closed = true;
		await this.localWorkerLease?.release();
	}
	deleteDocs(docIds) {
		this.repo.deleteDocs(docIds);
	}
	keywordSearch(params) {
		return this.lexical.keywordSearch(params);
	}
	async similaritySearch(params) {
		if (this.embedding.provider === "off" || this.isEmbeddingDisabledForProcess()) return [];
		if (this.embedding.provider === "sentence-transformers-local" && !this.localWorker?.isWarm()) {
			this.logger.debug?.("memx: local embeddings are warming; skipping prompt hot-path similarity search");
			return [];
		}
		try {
			const queryEmbedding = await this.getCachedQueryEmbedding(params.query);
			if (!queryEmbedding || queryEmbedding.length === 0) return [];
			const rows = this.repo.listEmbeddings({
				agentId: params.agentId,
				scopes: params.scopes,
				limit: Math.max(params.limit * 16, 128),
				readEpoch: params.readEpoch,
				docKinds: params.docKinds,
				docTypes: params.docTypes
			});
			const hits = [];
			for (const row of rows) {
				const score = cosineSimilarity(queryEmbedding, row.embedding);
				if (score <= 0) continue;
				const doc = this.repo.getDoc(row.docId, params.readEpoch);
				if (!doc) continue;
				hits.push({
					docId: doc.docId,
					text: doc.text,
					metadata: doc.metadataJson,
					score,
					backend: "embedding"
				});
			}
			return orderByScore(hits).slice(0, params.limit);
		} catch (error) {
			this.handleEmbeddingFailure(error, `memx: similarity search failed, falling back to FTS (${String(error)})`);
			return [];
		}
	}
	async hybridSearch(params) {
		const keywordHits = this.keywordSearch(params);
		const similarityHits = await this.similaritySearch(params);
		return mergeHybridHits(params.query, keywordHits, similarityHits, params.limit);
	}
	async embedTextsBatch(texts, mode = "passage") {
		if (this.isEmbeddingDisabledForProcess()) return [];
		try {
			return await this.embedTexts(texts, mode, "batch");
		} catch (error) {
			this.handleEmbeddingFailure(error, `memx: embedding batch unavailable, using lexical fallback (${String(error)})`);
			return [];
		}
	}
	async prewarmLocalEmbeddings() {
		if (this.embedding.provider !== "sentence-transformers-local" || !this.localWorker || this.closed || this.isEmbeddingDisabledForProcess()) return;
		if (this.localWorker.isWarm()) return;
		this.localPrewarm ??= this.localWorker.embed([LOCAL_EMBEDDING_PREWARM_TEXT], "query").then(() => {}).catch((error) => {
			this.handleEmbeddingFailure(error, `memx: local embedding prewarm failed; using lexical retrieval until available (${String(error)})`);
		}).finally(() => {
			this.localPrewarm = null;
		});
		await this.localPrewarm;
	}
	isEmbeddingDisabledForProcess() {
		return this.embedding.provider === "sentence-transformers-local" && this.localUnavailableForProcess;
	}
	handleEmbeddingFailure(error, message) {
		this.warnOnce(message);
		this.queryEmbeddingCache.clear();
		if (this.embedding.provider !== "sentence-transformers-local" || error instanceof LocalEmbeddingTimeoutError) return;
		this.localUnavailableForProcess = true;
		this.acceptingUpserts = false;
		this.localWorkerLease?.invalidate().catch((error) => {
			this.logger.debug?.(`memx: failed to close unavailable local embedding worker (${String(error)})`);
		});
	}
	async getCachedQueryEmbedding(query) {
		const cacheKey = query.normalize("NFKC").replace(/\s+/gu, " ").trim();
		if (!cacheKey) return [];
		const existing = this.queryEmbeddingCache.get(cacheKey);
		if (existing) return existing;
		const pending = this.embedTexts([cacheKey], "query", "similarity-search").then(([embedding]) => embedding ?? []).catch((error) => {
			this.queryEmbeddingCache.delete(cacheKey);
			throw error;
		});
		this.queryEmbeddingCache.set(cacheKey, pending);
		if (this.queryEmbeddingCache.size > QUERY_EMBEDDING_CACHE_LIMIT) {
			const oldest = this.queryEmbeddingCache.keys().next().value;
			if (oldest && oldest !== cacheKey) this.queryEmbeddingCache.delete(oldest);
		}
		return pending;
	}
	warnOnce(message) {
		if (this.warnedUnavailable) return;
		this.warnedUnavailable = true;
		this.logger.warn(message);
	}
	async embedTexts(texts, mode, label) {
		if (this.embedding.provider === "off") throw new Error("embeddings disabled");
		if (this.isEmbeddingDisabledForProcess()) throw new Error("local embeddings unavailable for this process");
		const started = Date.now();
		try {
			if (this.embedding.provider === "sentence-transformers-local") {
				if (!this.localWorker) throw new Error("local sentence-transformers worker unavailable");
				return await this.localWorker.embed(texts, mode);
			}
			return await embedTextsRemote(this.embedding, texts);
		} finally {
			const elapsedMs = Math.max(0, Date.now() - started);
			this.logger.info?.(`memx: TIMING embedding label=${label} mode=${mode} batch=${texts.length} provider=${this.embedding.provider} elapsed=${elapsedMs}ms`);
		}
	}
};
//#endregion
export { OptionalEmbeddingBackend, cleanupStaleLocalEmbeddingWorkers, localEmbeddingWorkerRegistryDir, localEmbeddingWorkerStatePath };
