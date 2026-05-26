import { mkdir, open, rename, rm, writeFile } from "node:fs/promises";
import { dirname, join } from "node:path";
import { existsSync, readFileSync } from "node:fs";
import { spawn } from "node:child_process";
//#region src/host/serviceManager.ts
const DEFAULT_MEMX_URL = "http://127.0.0.1:3878";
const DEFAULT_HEALTH_TIMEOUT_MS = 1500;
const DEFAULT_START_TIMEOUT_MS = 1e4;
const DEFAULT_SERVICE_RECORD = "service.json";
const DEFAULT_SERVICE_LOG = "memx-server.log";
function serviceStateDir(homeDir) {
	return join(homeDir, ".memx");
}
function serviceRecordPath(homeDir) {
	return join(serviceStateDir(homeDir), DEFAULT_SERVICE_RECORD);
}
function serviceLogPath(homeDir) {
	return join(serviceStateDir(homeDir), DEFAULT_SERVICE_LOG);
}
function normalizeUrl(url) {
	return (url?.trim() || DEFAULT_MEMX_URL).replace(/\/+$/u, "");
}
function parsePort(url) {
	try {
		const parsed = new URL(url);
		return parsed.port || (parsed.protocol === "https:" ? "443" : "80");
	} catch {
		return;
	}
}
function parseHost(url) {
	try {
		return new URL(url).hostname;
	} catch {
		return;
	}
}
function errorMessage(error) {
	return error instanceof Error ? error.message : String(error);
}
function isProcessAlive(pid) {
	if (!Number.isInteger(pid) || (pid ?? 0) <= 0) return false;
	try {
		process.kill(pid, 0);
		return true;
	} catch {
		return false;
	}
}
async function writeAtomic(path, text) {
	await mkdir(dirname(path), { recursive: true });
	const tmp = `${path}.tmp-${process.pid}-${Date.now()}`;
	await writeFile(tmp, text, "utf8");
	await rename(tmp, path);
}
function readServiceRecord(path) {
	if (!existsSync(path)) return null;
	try {
		const record = JSON.parse(readFileSync(path, "utf8"));
		if (typeof record.pid !== "number" || typeof record.url !== "string") return null;
		return {
			pid: record.pid,
			url: record.url,
			configPath: String(record.configPath ?? ""),
			startedAt: String(record.startedAt ?? ""),
			runtimeDir: String(record.runtimeDir ?? "")
		};
	} catch {
		return null;
	}
}
async function serviceHealth(url, secret, timeoutMs) {
	try {
		return (await fetch(`${url}/v1/health`, {
			method: "GET",
			headers: secret ? { authorization: `Bearer ${secret}` } : void 0,
			signal: AbortSignal.timeout(timeoutMs)
		})).ok;
	} catch {
		return false;
	}
}
async function waitForHealth(url, secret, timeoutMs, healthTimeoutMs) {
	const deadline = Date.now() + timeoutMs;
	while (Date.now() < deadline) {
		if (await serviceHealth(url, secret, healthTimeoutMs)) return true;
		await new Promise((resolve) => setTimeout(resolve, 150));
	}
	return serviceHealth(url, secret, healthTimeoutMs);
}
async function waitForProcessExit(pid, timeoutMs = 2500) {
	const deadline = Date.now() + timeoutMs;
	while (Date.now() < deadline && isProcessAlive(pid)) await new Promise((resolve) => setTimeout(resolve, 100));
	return !isProcessAlive(pid);
}
async function stopRecordedProcess(pidPath, pid, timeoutMs = 2500) {
	if (isProcessAlive(pid)) try {
		process.kill(pid, "SIGTERM");
		if (!await waitForProcessExit(pid, timeoutMs) && isProcessAlive(pid)) {
			process.kill(pid, "SIGKILL");
			await waitForProcessExit(pid, 1e3);
		}
	} catch {}
	await rm(pidPath, { force: true });
}
async function ensureMemxService(options) {
	const url = normalizeUrl(options.url);
	const pidPath = serviceRecordPath(options.homeDir);
	const logPath = serviceLogPath(options.homeDir);
	const healthTimeoutMs = Math.max(250, options.healthTimeoutMs ?? DEFAULT_HEALTH_TIMEOUT_MS);
	const startTimeoutMs = Math.max(1e3, options.startTimeoutMs ?? DEFAULT_START_TIMEOUT_MS);
	const stopTimeoutMs = Math.max(250, options.stopTimeoutMs ?? 2500);
	const existing = readServiceRecord(pidPath);
	let stoppedManagedService = false;
	if (existing?.pid && isProcessAlive(existing.pid)) {
		await stopRecordedProcess(pidPath, existing.pid, stopTimeoutMs);
		stoppedManagedService = true;
	} else if (existing?.pid && !isProcessAlive(existing.pid)) await rm(pidPath, { force: true });
	if (await serviceHealth(url, options.secret, healthTimeoutMs)) return {
		ok: false,
		alreadyRunning: true,
		url,
		pidPath,
		logPath,
		error: stoppedManagedService ? `previous memx service at ${url} did not stop cleanly` : `unmanaged memx-compatible service is already listening at ${url}; pass --memx-url with a free local port or stop the existing service`
	};
	await mkdir(dirname(logPath), { recursive: true });
	const serverEntry = join(options.runtimeDir, "src", "bin", "memx-server.mjs");
	if (!existsSync(serverEntry)) return {
		ok: false,
		alreadyRunning: false,
		url,
		pidPath,
		logPath,
		error: `memx-server runtime entry not found: ${serverEntry}`
	};
	const logHandle = await open(logPath, "a");
	try {
		const child = spawn(options.nodeBin?.trim() || process.execPath, [serverEntry], {
			detached: true,
			stdio: [
				"ignore",
				logHandle.fd,
				logHandle.fd
			],
			env: {
				...process.env,
				MEMX_CONFIG_PATH: options.configPath,
				MEMX_URL: url,
				...options.secret ? { MEMX_SECRET: options.secret } : {},
				...parseHost(url) ? { MEMX_HOST: parseHost(url) } : {},
				...parsePort(url) ? { MEMX_PORT: parsePort(url) } : {}
			}
		});
		child.unref();
		const record = {
			pid: child.pid ?? 0,
			url,
			configPath: options.configPath,
			startedAt: (/* @__PURE__ */ new Date()).toISOString(),
			runtimeDir: options.runtimeDir
		};
		await writeAtomic(pidPath, `${JSON.stringify(record, null, 2)}\n`);
		if (!await waitForHealth(url, options.secret, startTimeoutMs, healthTimeoutMs)) return {
			ok: false,
			alreadyRunning: false,
			url,
			pid: child.pid,
			pidPath,
			logPath,
			error: `memx service did not become healthy within ${startTimeoutMs}ms`
		};
		return {
			ok: true,
			alreadyRunning: false,
			url,
			pid: child.pid,
			pidPath,
			logPath
		};
	} finally {
		await logHandle.close();
	}
}
async function readMemxServiceStatus(options) {
	const pidPath = serviceRecordPath(options.homeDir);
	const logPath = serviceLogPath(options.homeDir);
	const record = readServiceRecord(pidPath);
	const url = normalizeUrl(options.url ?? record?.url);
	const ok = await serviceHealth(url, options.secret, options.healthTimeoutMs ?? DEFAULT_HEALTH_TIMEOUT_MS);
	return {
		ok,
		alreadyRunning: ok,
		url,
		pid: record?.pid,
		pidPath,
		logPath,
		...ok ? {} : { error: record?.pid && isProcessAlive(record.pid) ? "health check failed" : "not running" }
	};
}
async function stopMemxService(options) {
	const pidPath = serviceRecordPath(options.homeDir);
	const logPath = serviceLogPath(options.homeDir);
	const record = readServiceRecord(pidPath);
	const url = normalizeUrl(options.url ?? record?.url);
	const stopTimeoutMs = Math.max(250, options.stopTimeoutMs ?? 2500);
	if (record?.pid && isProcessAlive(record.pid)) try {
		process.kill(record.pid, "SIGTERM");
		if (!await waitForProcessExit(record.pid, stopTimeoutMs) && isProcessAlive(record.pid)) {
			process.kill(record.pid, "SIGKILL");
			await waitForProcessExit(record.pid, 1e3);
		}
	} catch (error) {
		return {
			ok: false,
			alreadyRunning: false,
			url,
			pid: record.pid,
			pidPath,
			logPath,
			error: errorMessage(error)
		};
	}
	await rm(pidPath, { force: true });
	return {
		ok: !await serviceHealth(url, options.secret, options.healthTimeoutMs ?? DEFAULT_HEALTH_TIMEOUT_MS),
		alreadyRunning: false,
		url,
		pid: record?.pid,
		pidPath,
		logPath
	};
}
//#endregion
export { ensureMemxService, readMemxServiceStatus, stopMemxService };
