import { homedir } from "node:os";
import path from "node:path";
import fs from "node:fs";
//#region src/pipeline/judgeModelConfig.ts
function resolveEnvTemplate(value) {
	const match = /^\$\{([A-Z][A-Z0-9_]*)\}$/.exec(value.trim());
	if (!match) return value.trim() || void 0;
	return process.env[match[1]]?.trim() || void 0;
}
function resolveSecretLikeString(value) {
	if (typeof value === "string") return resolveEnvTemplate(value);
	if (!value || typeof value !== "object") return;
	const record = value;
	if (record.source === "env" && typeof record.id === "string") return process.env[record.id]?.trim() || void 0;
}
function guessProvider(baseUrl, api, providerKey) {
	if (api === "anthropic-messages" || /anthropic/i.test(baseUrl) || /anthropic/i.test(providerKey ?? "")) return "anthropic";
	if (api === "google-generative-ai" || /generativelanguage|google/i.test(baseUrl) || /google/i.test(providerKey ?? "")) return "google";
	if (api === "ollama" || /11434|ollama/i.test(baseUrl) || /ollama/i.test(providerKey ?? "")) return "ollama";
	return "openai-compatible";
}
function loadJudgeModelConfig(config, logger) {
	if (!config.advanced.llmClassifierEnabled) return null;
	const directBaseUrl = config.advanced.llmBaseURL?.trim();
	const directModel = config.advanced.llmClassifierModel?.trim();
	if (directBaseUrl && directModel) {
		const model = directModel.includes("/") ? directModel.split("/").at(-1) || directModel : directModel;
		return {
			configPath: process.env.MEMX_CONFIG_PATH?.trim() || "memx-config",
			provider: config.advanced.llmProvider ?? guessProvider(directBaseUrl, void 0, config.advanced.llmProvider),
			baseUrl: directBaseUrl,
			model,
			apiKey: resolveSecretLikeString(config.advanced.llmApiKey),
			headers: config.advanced.llmHeaders
		};
	}
	const cfgPath = process.env.OPENCLAW_CONFIG_PATH?.trim() || path.join(homedir(), ".openclaw", "openclaw.json");
	if (!fs.existsSync(cfgPath)) {
		logger.debug?.(`memx: no config available for LLM reasoner at ${cfgPath}`);
		return null;
	}
	try {
		const raw = JSON.parse(fs.readFileSync(cfgPath, "utf-8"));
		const modelEntry = ((raw.agents?.defaults)?.model)?.primary;
		const requested = config.advanced.llmClassifierModel?.trim() || (typeof modelEntry === "string" ? modelEntry : "");
		if (!requested) return null;
		const providers = raw.models?.providers ?? {};
		let providerKey;
		let model = requested;
		if (requested.includes("/")) [providerKey, model] = requested.split("/", 2);
		else if (typeof modelEntry === "string" && modelEntry.includes("/")) [providerKey] = modelEntry.split("/", 2);
		if (!providerKey) providerKey = Object.keys(providers)[0];
		if (!providerKey) return null;
		const providerCfg = providers[providerKey];
		if (!providerCfg) return null;
		const baseUrl = typeof providerCfg.baseUrl === "string" ? providerCfg.baseUrl.trim() : "";
		if (!baseUrl) return null;
		const headers = providerCfg.headers && typeof providerCfg.headers === "object" ? Object.fromEntries(Object.entries(providerCfg.headers).map(([key, value]) => [key, resolveSecretLikeString(value)]).filter((entry) => Boolean(entry[1]))) : void 0;
		return {
			configPath: cfgPath,
			provider: guessProvider(baseUrl, typeof providerCfg.api === "string" ? providerCfg.api : void 0, providerKey),
			baseUrl,
			model,
			apiKey: resolveSecretLikeString(providerCfg.apiKey),
			headers
		};
	} catch (error) {
		logger.warn(`memx: failed to load LLM reasoner config (${String(error)})`);
		return null;
	}
}
//#endregion
export { loadJudgeModelConfig };
