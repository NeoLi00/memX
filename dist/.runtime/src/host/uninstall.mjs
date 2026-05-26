import { LEGACY_MEMX_PLUGIN_ID, MEMX_PLUGIN_ID, withoutLegacyPluginIds } from "../identity.mjs";
import { applyClaudeJsonDisconnect, applyCodexTomlDisconnect } from "./connect.mjs";
import { stopMemxService } from "./serviceManager.mjs";
import { copyFile, mkdir, readFile, readdir, rename, rm, writeFile } from "node:fs/promises";
import { homedir } from "node:os";
import { dirname, join } from "node:path";
import { existsSync } from "node:fs";
import { spawn } from "node:child_process";
//#region src/host/uninstall.ts
const DEFAULT_OPENCLAW_CONFIG_PATH = join(homedir(), ".openclaw", "openclaw.json");
const DEFAULT_CODEX_CONFIG_PATH = join(homedir(), ".codex", "config.toml");
const DEFAULT_CLAUDE_CONFIG_PATH = join(homedir(), ".claude.json");
const DEFAULT_CODEX_MARKETPLACE_DIRNAME = "codex-marketplace";
const DEFAULT_CLAUDE_MARKETPLACE_DIRNAME = "claude-marketplace";
const CLAUDE_SETTINGS_BACKUP = "claude-settings-backup.json";
function asOpenClawConfig(input) {
	return input && typeof input === "object" && !Array.isArray(input) ? structuredClone(input) : {};
}
function trimOrUndefined(value) {
	const trimmed = value?.trim();
	return trimmed ? trimmed : void 0;
}
function isMemxId(value) {
	return value === "memx" || value === "memory-memx";
}
function localCodexMarketplaceDir(homeDir) {
	return join(homeDir, ".memx", DEFAULT_CODEX_MARKETPLACE_DIRNAME);
}
function localClaudeMarketplaceDir(homeDir) {
	return join(homeDir, ".memx", DEFAULT_CLAUDE_MARKETPLACE_DIRNAME);
}
function claudeSettingsPath(homeDir) {
	return join(homeDir, ".claude", "settings.json");
}
function claudeSettingsBackupPath(homeDir) {
	return join(homeDir, ".memx", CLAUDE_SETTINGS_BACKUP);
}
async function removeCachedMemxPlugin(cacheRoot) {
	let marketplaceEntries;
	try {
		marketplaceEntries = await readdir(cacheRoot);
	} catch {
		return;
	}
	await Promise.all(marketplaceEntries.map((marketplace) => rm(join(cacheRoot, marketplace, "memx"), {
		recursive: true,
		force: true
	})));
}
function applyOpenClawUninstallConfig(input) {
	const next = asOpenClawConfig(input);
	const plugins = { ...next.plugins ?? {} };
	const slots = { ...plugins.slots ?? {} };
	if (isMemxId(slots.memory)) delete slots.memory;
	const entries = { ...plugins.entries ?? {} };
	delete entries[MEMX_PLUGIN_ID];
	delete entries[LEGACY_MEMX_PLUGIN_ID];
	return {
		...next,
		plugins: {
			...plugins,
			slots,
			allow: withoutLegacyPluginIds(plugins.allow).filter((value) => value !== MEMX_PLUGIN_ID),
			entries
		}
	};
}
async function writeAtomic(path, text) {
	await mkdir(dirname(path), { recursive: true });
	const tmp = `${path}.tmp-${process.pid}-${Date.now()}`;
	await writeFile(tmp, text, "utf8");
	await rename(tmp, path);
}
async function backupIfExists(path, now) {
	if (!existsSync(path)) return null;
	const backupPath = `${path}.bak.${now()}`;
	await copyFile(path, backupPath);
	return backupPath;
}
async function defaultRunCommand(command, args) {
	return new Promise((resolve) => {
		const child = spawn(command, args, {
			shell: false,
			stdio: "pipe"
		});
		let stdout = "";
		let stderr = "";
		child.stdout?.on("data", (chunk) => {
			stdout += String(chunk);
		});
		child.stderr?.on("data", (chunk) => {
			stderr += String(chunk);
		});
		child.once("error", (error) => {
			resolve({
				code: 1,
				stderr: error.message
			});
		});
		child.once("close", (code) => resolve({
			code: code ?? 1,
			stdout,
			stderr
		}));
	});
}
function warningForFailedUninstall(result) {
	const detail = (result.stderr || result.stdout || "").trim();
	return `plugin uninstall exited ${result.code}${detail ? `: ${detail}` : ""}`;
}
function isExpectedMissingCleanup(result) {
	const detail = `${result.stderr ?? ""}\n${result.stdout ?? ""}`;
	return /not found|not installed|not configured/iu.test(detail);
}
function stripTomlSection(toml, header) {
	const escaped = header.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
	return toml.replace(new RegExp(`\\n?${escaped}\\n[\\s\\S]*?(?=\\n\\[|$)`, "u"), "");
}
function applyCodexPluginDisconnect(toml) {
	return stripTomlSection(stripTomlSection(toml, "[plugins.\"memx@memx\"]"), "[marketplaces.memx]").trim();
}
function restoreSnapshotValue(target, key, snapshot) {
	if (!snapshot || typeof snapshot !== "object" || Array.isArray(snapshot)) return;
	const record = snapshot;
	if (record.present === true) {
		target[key] = record.value;
		return;
	}
	if (record.present === false) delete target[key];
}
async function restoreClaudeNativeSettings(homeDir) {
	const backupPath = claudeSettingsBackupPath(homeDir);
	if (!existsSync(backupPath)) return null;
	const settingsPath = claudeSettingsPath(homeDir);
	const current = existsSync(settingsPath) ? JSON.parse(await readFile(settingsPath, "utf8")) : {};
	const backup = JSON.parse(await readFile(backupPath, "utf8"));
	restoreSnapshotValue(current, "autoMemoryEnabled", backup.autoMemoryEnabled);
	const currentEnv = current.env && typeof current.env === "object" && !Array.isArray(current.env) ? { ...current.env } : {};
	restoreSnapshotValue(currentEnv, "CLAUDE_CODE_DISABLE_AUTO_MEMORY", (backup.env && typeof backup.env === "object" && !Array.isArray(backup.env) ? backup.env : {}).CLAUDE_CODE_DISABLE_AUTO_MEMORY);
	if (Object.keys(currentEnv).length > 0) current.env = currentEnv;
	else delete current.env;
	await writeAtomic(settingsPath, `${JSON.stringify(current, null, 2)}\n`);
	await rm(backupPath, { force: true });
	return settingsPath;
}
async function runOpenClawUninstall(rawOptions = {}, deps = {}) {
	const configPath = trimOrUndefined(rawOptions.configPath) ?? DEFAULT_OPENCLAW_CONFIG_PATH;
	const openclawBin = trimOrUndefined(rawOptions.openclawBin) ?? "openclaw";
	const now = deps.now ?? Date.now;
	const dryRun = Boolean(rawOptions.dryRun);
	const current = existsSync(configPath) ? JSON.parse(await readFile(configPath, "utf8")) : {};
	const next = applyOpenClawUninstallConfig(current);
	const warnings = [];
	let backupPath = null;
	if (!dryRun) {
		backupPath = await backupIfExists(configPath, now);
		await writeAtomic(configPath, `${JSON.stringify(next, null, 2)}\n`);
		if (!rawOptions.skipPluginUninstall) {
			const runCommand = deps.runCommand ?? defaultRunCommand;
			for (const pluginId of [MEMX_PLUGIN_ID, LEGACY_MEMX_PLUGIN_ID]) {
				const result = await runCommand(openclawBin, [
					"plugins",
					"uninstall",
					pluginId,
					"--force"
				]);
				if (result.code !== 0 && !isExpectedMissingCleanup(result)) warnings.push(warningForFailedUninstall(result));
			}
		}
	}
	return {
		ok: true,
		target: "openclaw",
		dryRun,
		configPath,
		backupPath,
		warnings,
		removed: {
			slot: current?.plugins?.slots?.memory === MEMX_PLUGIN_ID,
			legacySlot: current?.plugins?.slots?.memory === LEGACY_MEMX_PLUGIN_ID,
			entry: Boolean(current?.plugins?.entries?.[MEMX_PLUGIN_ID]),
			legacyEntry: Boolean(current?.plugins?.entries?.[LEGACY_MEMX_PLUGIN_ID]),
			allow: Array.isArray(current?.plugins?.allow) ? current.plugins.allow.filter(isMemxId).length : 0
		}
	};
}
async function runCodexUninstall(rawOptions = {}, deps = {}) {
	const configPath = trimOrUndefined(rawOptions.configPath) ?? DEFAULT_CODEX_CONFIG_PATH;
	const codexBin = trimOrUndefined(rawOptions.codexBin) ?? "codex";
	const homeDir = trimOrUndefined(rawOptions.homeDir) ?? homedir();
	const marketplaceDir = trimOrUndefined(rawOptions.codexMarketplaceDir) ?? localCodexMarketplaceDir(homeDir);
	const now = deps.now ?? Date.now;
	const dryRun = Boolean(rawOptions.dryRun);
	const current = existsSync(configPath) ? await readFile(configPath, "utf8") : "";
	const next = applyCodexPluginDisconnect(applyCodexTomlDisconnect(current));
	let backupPath = null;
	const warnings = [];
	let service = null;
	if (!dryRun) {
		service = await (deps.stopService ?? stopMemxService)({
			homeDir,
			url: trimOrUndefined(rawOptions.memxUrl),
			secret: trimOrUndefined(rawOptions.memxSecret)
		});
		backupPath = await backupIfExists(configPath, now);
		await writeAtomic(configPath, next ? `${next}\n` : "");
		const runCommand = deps.runCommand ?? defaultRunCommand;
		for (const args of [[
			"plugin",
			"remove",
			"memx@memx"
		], [
			"plugin",
			"marketplace",
			"remove",
			"memx"
		]]) {
			const result = await runCommand(codexBin, args);
			if (result.code !== 0 && !isExpectedMissingCleanup(result)) warnings.push(warningForFailedUninstall(result));
		}
		await rm(marketplaceDir, {
			recursive: true,
			force: true
		});
		await removeCachedMemxPlugin(join(homeDir, ".codex", "plugins", "cache"));
	}
	return {
		ok: true,
		target: "codex",
		dryRun,
		configPath,
		marketplaceDir,
		backupPath,
		warnings,
		service,
		removed: current !== next
	};
}
async function runClaudeCodeUninstall(rawOptions = {}, deps = {}) {
	const configPath = trimOrUndefined(rawOptions.configPath) ?? DEFAULT_CLAUDE_CONFIG_PATH;
	const claudeBin = trimOrUndefined(rawOptions.claudeBin) ?? "claude";
	const homeDir = trimOrUndefined(rawOptions.homeDir) ?? homedir();
	const marketplaceDir = trimOrUndefined(rawOptions.claudeMarketplaceDir) ?? localClaudeMarketplaceDir(homeDir);
	const now = deps.now ?? Date.now;
	const dryRun = Boolean(rawOptions.dryRun);
	const current = existsSync(configPath) ? JSON.parse(await readFile(configPath, "utf8")) : {};
	const next = applyClaudeJsonDisconnect(current);
	let backupPath = null;
	let settingsPath = null;
	const warnings = [];
	let service = null;
	if (!dryRun) {
		service = await (deps.stopService ?? stopMemxService)({
			homeDir,
			url: trimOrUndefined(rawOptions.memxUrl),
			secret: trimOrUndefined(rawOptions.memxSecret)
		});
		backupPath = await backupIfExists(configPath, now);
		await writeAtomic(configPath, `${JSON.stringify(next, null, 2)}\n`);
		settingsPath = await restoreClaudeNativeSettings(homeDir);
		const runCommand = deps.runCommand ?? defaultRunCommand;
		for (const args of [
			[
				"plugin",
				"uninstall",
				"memx@memx"
			],
			[
				"plugin",
				"uninstall",
				"memx"
			],
			[
				"plugin",
				"marketplace",
				"remove",
				"memx"
			]
		]) {
			const result = await runCommand(claudeBin, args);
			if (result.code !== 0 && !isExpectedMissingCleanup(result)) warnings.push(warningForFailedUninstall(result));
		}
		await rm(marketplaceDir, {
			recursive: true,
			force: true
		});
		await removeCachedMemxPlugin(join(homeDir, ".claude", "plugins", "cache"));
	}
	return {
		ok: true,
		target: "claude-code",
		dryRun,
		configPath,
		marketplaceDir,
		backupPath,
		settingsPath,
		warnings,
		service,
		removed: Boolean(current?.mcpServers && typeof current.mcpServers === "object" && !Array.isArray(current.mcpServers) && current.mcpServers.memx)
	};
}
//#endregion
export { applyOpenClawUninstallConfig, runClaudeCodeUninstall, runCodexUninstall, runOpenClawUninstall };
