import { LEGACY_MEMX_PLUGIN_ID, MEMX_BRAND_NAME, MEMX_PLUGIN_ID, withoutLegacyPluginIds } from "../identity.mjs";
import { nowIso, resolveUserPath } from "../support.mjs";
import { runAbstractionJobs } from "../pipeline/abstractionJobs.mjs";
import { runAbstractionPromotion } from "../pipeline/abstractionPromotion.mjs";
import { runConsolidation } from "../pipeline/consolidate.mjs";
import { MemxReasoner } from "../pipeline/reasoner.mjs";
import { OptionalEmbeddingBackend } from "../search/backends/embeddingBackend.mjs";
import { buildOperationContext } from "../runtime.mjs";
import { mkdir, readFile, writeFile } from "node:fs/promises";
import { homedir } from "node:os";
import path from "node:path";
//#region src/cli/registerCli.ts
const DEFAULT_USER_CONFIG_PATH = path.join(homedir(), ".openclaw", "openclaw.json");
const PLUGIN_ID = MEMX_PLUGIN_ID;
function defaultSetupEntry(config, options) {
	const llmClassifierEnabled = options.enableLlmJudge === true ? true : config.advanced.llmClassifierEnabled;
	const llmClassifierModel = options.llmModel?.trim() || config.advanced.llmClassifierModel?.trim() || void 0;
	const embeddingProvider = options.embeddingProvider ?? config.embedding.provider;
	const embeddingModel = options.embeddingModel?.trim() || config.embedding.model?.trim() || void 0;
	const embeddingPythonBin = options.embeddingPythonBin?.trim() || config.embedding.localPythonBin?.trim() || void 0;
	const embeddingCacheDir = options.embeddingCacheDir?.trim() || config.embedding.localCacheDir?.trim() || void 0;
	const embeddingDevice = options.embeddingDevice ?? config.embedding.localDevice;
	return {
		enabled: true,
		config: {
			dbPath: config.dbPath,
			autoCapture: config.autoCapture,
			autoRecall: config.autoRecall,
			reflectionEnabled: config.reflectionEnabled,
			consentMode: config.consentMode,
			piiMode: config.piiMode,
			maxInjectedChars: config.maxInjectedChars,
			captureMaxChars: config.captureMaxChars,
			reflectionMaxChars: config.reflectionMaxChars,
			reflectionMaxItems: config.reflectionMaxItems,
			defaultScope: config.defaultScope,
			allowedScopes: [...config.allowedScopes],
			embedding: {
				...config.embedding,
				provider: embeddingProvider,
				...embeddingModel ? { model: embeddingModel } : {},
				...embeddingPythonBin ? { localPythonBin: embeddingPythonBin } : {},
				...embeddingCacheDir ? { localCacheDir: embeddingCacheDir } : {},
				...embeddingDevice ? { localDevice: embeddingDevice } : {}
			},
			advanced: {
				...config.advanced,
				enableTurnScheduler: true,
				enableCompatibilityMemoryTools: false,
				llmClassifierEnabled,
				...llmClassifierModel ? { llmClassifierModel } : {}
			}
		}
	};
}
function normalizeConfig(input) {
	if (!input || typeof input !== "object") return {};
	return input;
}
async function readUserConfig(configPath) {
	try {
		const raw = await readFile(configPath, "utf8");
		return normalizeConfig(JSON.parse(raw));
	} catch (error) {
		if (String(error).includes("ENOENT")) return {};
		throw error;
	}
}
function applyMemxSetupToConfig(appConfig, pluginConfig, options = {}) {
	const next = normalizeConfig(structuredClone(appConfig));
	const currentAllow = new Set(withoutLegacyPluginIds(next.plugins?.allow));
	currentAllow.add(PLUGIN_ID);
	const existingEntries = { ...next.plugins?.entries ?? {} };
	const existingEntry = existingEntries[PLUGIN_ID] ?? existingEntries["memory-memx"] ?? {};
	delete existingEntries[LEGACY_MEMX_PLUGIN_ID];
	const existingHooks = existingEntry.hooks && typeof existingEntry.hooks === "object" ? existingEntry.hooks : {};
	const existingConfig = existingEntry.config && typeof existingEntry.config === "object" ? existingEntry.config : {};
	const setupConfig = defaultSetupEntry(pluginConfig, options).config ?? {};
	const setupEmbedding = setupConfig.embedding && typeof setupConfig.embedding === "object" ? setupConfig.embedding : {};
	const existingEmbedding = existingConfig.embedding && typeof existingConfig.embedding === "object" ? existingConfig.embedding : {};
	const setupAdvanced = setupConfig.advanced && typeof setupConfig.advanced === "object" ? setupConfig.advanced : {};
	const existingAdvanced = existingConfig.advanced && typeof existingConfig.advanced === "object" ? existingConfig.advanced : {};
	const setupEntry = {
		...defaultSetupEntry(pluginConfig, options),
		...existingEntry,
		hooks: {
			...typeof existingHooks.allowPromptInjection === "boolean" ? { allowPromptInjection: existingHooks.allowPromptInjection } : {},
			allowPromptInjection: true
		},
		config: {
			...setupConfig,
			...existingConfig,
			embedding: {
				...setupEmbedding,
				...existingEmbedding,
				...options.embeddingProvider ? { provider: options.embeddingProvider } : {},
				...options.embeddingModel?.trim() ? { model: options.embeddingModel.trim() } : {},
				...options.embeddingPythonBin?.trim() ? { localPythonBin: options.embeddingPythonBin.trim() } : {},
				...options.embeddingCacheDir?.trim() ? { localCacheDir: options.embeddingCacheDir.trim() } : {},
				...options.embeddingDevice ? { localDevice: options.embeddingDevice } : {}
			},
			advanced: {
				...setupAdvanced,
				...existingAdvanced,
				enableTurnScheduler: true,
				enableCompatibilityMemoryTools: false,
				...options.enableLlmJudge === true ? { llmClassifierEnabled: true } : {},
				...options.llmModel?.trim() ? { llmClassifierModel: options.llmModel.trim() } : {}
			}
		},
		enabled: true
	};
	next.plugins = {
		...next.plugins ?? {},
		allow: [...currentAllow],
		slots: {
			...next.plugins?.slots ?? {},
			memory: PLUGIN_ID
		},
		entries: {
			...existingEntries,
			[PLUGIN_ID]: setupEntry
		}
	};
	return next;
}
function buildMemxDoctorReport(params) {
	const appConfig = normalizeConfig(params.appConfig);
	const memorySlot = appConfig.plugins?.slots?.memory ?? null;
	const allow = appConfig.plugins?.allow ?? [];
	const entry = appConfig.plugins?.entries?.[PLUGIN_ID] ?? {};
	const entryConfig = entry.config && typeof entry.config === "object" ? entry.config : {};
	const legacyTopLevelConfigKeys = [
		"dbPath",
		"autoCapture",
		"autoRecall",
		"reflectionEnabled",
		"consentMode",
		"piiMode",
		"maxInjectedChars",
		"captureMaxChars",
		"reflectionMaxChars",
		"reflectionMaxItems",
		"defaultScope",
		"allowedScopes",
		"embedding",
		"advanced"
	].filter((key) => Object.prototype.hasOwnProperty.call(entry, key));
	const entryAdvanced = entryConfig.advanced && typeof entryConfig.advanced === "object" ? entryConfig.advanced : {};
	const checks = [
		{
			key: "plugin_loaded",
			ok: true,
			detail: `${MEMX_BRAND_NAME} CLI is available, so the plugin is currently loaded.`
		},
		{
			key: "allowed",
			ok: allow.includes(PLUGIN_ID),
			detail: allow.includes(PLUGIN_ID) ? `plugins.allow includes ${PLUGIN_ID}.` : `plugins.allow does not include ${PLUGIN_ID}.`
		},
		{
			key: "memory_slot",
			ok: memorySlot === PLUGIN_ID,
			detail: memorySlot === PLUGIN_ID ? `plugins.slots.memory points to ${PLUGIN_ID}.` : `plugins.slots.memory points to ${memorySlot ?? "nothing"}.`
		},
		{
			key: "entry_enabled",
			ok: entry.enabled !== false,
			detail: entry.enabled !== false ? `plugins.entries.${PLUGIN_ID} is enabled.` : `plugins.entries.${PLUGIN_ID} is disabled.`
		},
		{
			key: "config_nesting",
			ok: legacyTopLevelConfigKeys.length === 0,
			detail: legacyTopLevelConfigKeys.length === 0 ? `plugin config keys are nested under plugins.entries.${PLUGIN_ID}.config.` : `legacy top-level plugin keys detected: ${legacyTopLevelConfigKeys.join(", ")}.`
		},
		{
			key: "turn_scheduler",
			ok: entryAdvanced.enableTurnScheduler !== false,
			detail: entryAdvanced.enableTurnScheduler !== false ? "turn scheduler is enabled." : "turn scheduler is disabled."
		},
		{
			key: "llm_classifier",
			ok: Boolean(entryAdvanced.llmClassifierEnabled ?? params.pluginConfig.advanced.llmClassifierEnabled),
			detail: entryAdvanced.llmClassifierEnabled ?? params.pluginConfig.advanced.llmClassifierEnabled ? `LLM classifier is enabled${typeof entryAdvanced.llmClassifierModel === "string" ? ` with ${entryAdvanced.llmClassifierModel}` : params.pluginConfig.advanced.llmClassifierModel ? ` with ${params.pluginConfig.advanced.llmClassifierModel}` : ""}.` : "LLM classifier is disabled; semantic extraction will fail closed."
		}
	];
	const recommendedFixes = checks.filter((check) => !check.ok).map((check) => {
		switch (check.key) {
			case "allowed":
			case "memory_slot":
			case "entry_enabled":
			case "config_nesting":
			case "turn_scheduler": return "Run `openclaw memx setup` to write the recommended memX config.";
			case "llm_classifier": return `Run \`openclaw memx setup\` or set plugins.entries.${PLUGIN_ID}.config.advanced.llmClassifierEnabled=true.`;
			default: return "Review memX configuration.";
		}
	});
	return {
		ok: checks.every((check) => check.ok),
		configPath: params.configPath,
		reasonerConfigPath: null,
		pluginLoaded: true,
		checks,
		recommendedFixes: [...new Set(recommendedFixes)],
		configSummary: {
			allowed: allow.includes(PLUGIN_ID),
			memorySlot,
			pluginEnabled: entry.enabled !== false,
			turnSchedulerEnabled: entryAdvanced.enableTurnScheduler !== false,
			llmClassifierEnabled: Boolean(entryAdvanced.llmClassifierEnabled ?? params.pluginConfig.advanced.llmClassifierEnabled),
			llmClassifierModel: (typeof entryAdvanced.llmClassifierModel === "string" ? entryAdvanced.llmClassifierModel : params.pluginConfig.advanced.llmClassifierModel) ?? null,
			dbPath: (typeof entryConfig.dbPath === "string" ? entryConfig.dbPath : params.pluginConfig.dbPath) ?? null,
			embeddingProvider: (typeof entryConfig.embedding?.provider === "string" ? entryConfig.embedding.provider : params.pluginConfig.embedding.provider) ?? null,
			embeddingModel: (typeof entryConfig.embedding?.model === "string" ? entryConfig.embedding.model : params.pluginConfig.embedding.model) ?? null
		}
	};
}
async function runEmbeddingProbe(config) {
	const startedAt = Date.now();
	const provider = config.embedding.provider;
	const model = config.embedding.model?.trim() || null;
	if (provider === "off") return {
		enabled: false,
		ok: true,
		provider,
		model,
		dimension: null,
		durationMs: 0,
		detail: "embedding provider is off."
	};
	const backend = new OptionalEmbeddingBackend({}, config.embedding, {
		warn() {},
		info() {},
		debug() {},
		error() {}
	});
	try {
		await backend.prewarmLocalEmbeddings();
		const dimension = (await backend.embedTextsBatch(["memx embedding probe"], "query"))[0]?.length ?? 0;
		return {
			enabled: true,
			ok: dimension > 0,
			provider,
			model,
			dimension: dimension > 0 ? dimension : null,
			durationMs: Date.now() - startedAt,
			detail: dimension > 0 ? "embedding request succeeded." : "embedding request returned no vector; retrieval will use lexical fallback."
		};
	} catch (error) {
		return {
			enabled: true,
			ok: false,
			provider,
			model,
			dimension: null,
			durationMs: Date.now() - startedAt,
			detail: `embedding request failed: ${String(error)}`
		};
	} finally {
		await backend.close();
	}
}
async function writeUserConfig(configPath, config) {
	await mkdir(path.dirname(configPath), { recursive: true });
	await writeFile(configPath, `${JSON.stringify(config, null, 2)}\n`, "utf8");
}
function resolveConfigPath(input) {
	return resolveUserPath(input?.trim() || DEFAULT_USER_CONFIG_PATH);
}
function resolveAgentId(config, agent) {
	if (agent?.trim()) return agent.trim();
	return config.agents?.list?.find((entry) => entry.id)?.id ?? "main";
}
function resolveEffectivePluginConfig(appConfig, fallback) {
	const entry = appConfig.plugins?.entries?.[PLUGIN_ID] ?? appConfig.plugins?.entries?.["memory-memx"] ?? {};
	const nested = entry.config && typeof entry.config === "object" ? entry.config : {};
	const advanced = nested.advanced && typeof nested.advanced === "object" ? nested.advanced : {};
	const embedding = nested.embedding && typeof nested.embedding === "object" ? nested.embedding : {};
	return {
		...fallback,
		...nested,
		embedding: {
			...fallback.embedding,
			...embedding
		},
		advanced: {
			...fallback.advanced,
			...advanced
		}
	};
}
async function withStore(params) {
	const agentId = resolveAgentId(params.appConfig, params.agent);
	const ctx = buildOperationContext(params.config, {
		agentId,
		sessionKey: params.sessionKey,
		workspaceDir: params.workspaceDir
	});
	if (!ctx) throw new Error("agent context unavailable");
	const store = await params.manager.getStore(ctx);
	await params.run(ctx, store);
}
async function printStats(ctx, store) {
	const count = (table) => Number(store.client.prepare(`SELECT COUNT(*) AS count FROM ${table}`).get().count ?? 0);
	console.log(JSON.stringify({
		agentId: ctx.agentId,
		dbPath: ctx.dbPath,
		scopes: ctx.scopes,
		taskCount: count("conversation_tasks"),
		chunkCount: count("conversation_chunks"),
		stateCount: count("state_kv"),
		factCount: count("facts"),
		eventCount: count("episodic_events"),
		entityCount: count("entities"),
		edgeCount: count("graph_edges"),
		vectorDocCount: count("vector_docs"),
		policyDecisionCount: count("policy_decisions"),
		memorySignalCount: count("memory_signal_events"),
		beliefCount: count("memory_beliefs"),
		abstractionCandidateCount: count("abstraction_candidates"),
		strategyCount: count("strategy_hypotheses")
	}, null, 2));
}
function wipeAgentMemory(ctx, store) {
	const deleted = {};
	let orphanEntitiesDeleted = 0;
	let orphanAliasesDeleted = 0;
	const countByAgent = (table, column = "agent_id") => Number(store.client.prepare(`SELECT COUNT(*) AS count FROM ${table} WHERE ${column} = ?`).get(ctx.agentId).count ?? 0);
	deleted.conversationChunks = countByAgent("conversation_chunks");
	deleted.conversationTasks = countByAgent("conversation_tasks");
	deleted.states = countByAgent("state_kv");
	deleted.facts = countByAgent("facts");
	deleted.factVersions = Number(store.client.prepare(`SELECT COUNT(*) AS count
             FROM fact_versions
            WHERE fact_id IN (SELECT fact_id FROM facts WHERE agent_id = ?)`).get(ctx.agentId).count ?? 0);
	deleted.events = countByAgent("episodic_events");
	deleted.edges = countByAgent("graph_edges");
	deleted.vectorDocs = countByAgent("vector_docs");
	deleted.vectorEmbeddings = countByAgent("vector_embeddings");
	deleted.retrievalAudit = countByAgent("retrieval_audit");
	deleted.policyDecisions = countByAgent("policy_decisions");
	deleted.maintenanceRuns = countByAgent("maintenance_runs");
	deleted.memorySignals = countByAgent("memory_signal_events");
	deleted.beliefs = countByAgent("memory_beliefs");
	deleted.abstractionCandidates = countByAgent("abstraction_candidates");
	deleted.strategies = countByAgent("strategy_hypotheses");
	store.client.withTransaction(() => {
		store.client.prepare(`DELETE FROM fact_versions
          WHERE fact_id IN (SELECT fact_id FROM facts WHERE agent_id = ?)`).run(ctx.agentId);
		store.client.prepare("DELETE FROM vector_embeddings WHERE agent_id = ?").run(ctx.agentId);
		store.client.prepare("DELETE FROM vector_docs_fts WHERE agent_id = ?").run(ctx.agentId);
		store.client.prepare("DELETE FROM vector_docs WHERE agent_id = ?").run(ctx.agentId);
		store.client.prepare("DELETE FROM state_kv WHERE agent_id = ?").run(ctx.agentId);
		store.client.prepare("DELETE FROM episodic_events WHERE agent_id = ?").run(ctx.agentId);
		store.client.prepare("DELETE FROM graph_edges WHERE agent_id = ?").run(ctx.agentId);
		store.client.prepare("DELETE FROM facts WHERE agent_id = ?").run(ctx.agentId);
		store.client.prepare("DELETE FROM retrieval_audit WHERE agent_id = ?").run(ctx.agentId);
		store.client.prepare("DELETE FROM policy_decisions WHERE agent_id = ?").run(ctx.agentId);
		store.client.prepare("DELETE FROM maintenance_runs WHERE agent_id = ?").run(ctx.agentId);
		store.client.prepare("DELETE FROM memory_signal_events WHERE agent_id = ?").run(ctx.agentId);
		store.client.prepare("DELETE FROM memory_beliefs WHERE agent_id = ?").run(ctx.agentId);
		store.client.prepare("DELETE FROM abstraction_candidates WHERE agent_id = ?").run(ctx.agentId);
		store.client.prepare("DELETE FROM strategy_hypotheses WHERE agent_id = ?").run(ctx.agentId);
		store.client.prepare("DELETE FROM conversation_chunks WHERE agent_id = ?").run(ctx.agentId);
		store.client.prepare("DELETE FROM conversation_tasks WHERE agent_id = ?").run(ctx.agentId);
		orphanAliasesDeleted = Number(store.client.prepare(`SELECT COUNT(*) AS count
             FROM entity_aliases
            WHERE entity_id NOT IN (SELECT src_entity_id FROM graph_edges)
              AND entity_id NOT IN (SELECT dst_entity_id FROM graph_edges)`).get().count ?? 0);
		store.client.prepare(`DELETE FROM entity_aliases
        WHERE entity_id NOT IN (SELECT src_entity_id FROM graph_edges)
          AND entity_id NOT IN (SELECT dst_entity_id FROM graph_edges)`).run();
		orphanEntitiesDeleted = Number(store.client.prepare(`SELECT COUNT(*) AS count
             FROM entities
            WHERE entity_id NOT IN (SELECT src_entity_id FROM graph_edges)
              AND entity_id NOT IN (SELECT dst_entity_id FROM graph_edges)`).get().count ?? 0);
		store.client.prepare(`DELETE FROM entities
        WHERE entity_id NOT IN (SELECT src_entity_id FROM graph_edges)
          AND entity_id NOT IN (SELECT dst_entity_id FROM graph_edges)`).run();
	});
	return {
		agentId: ctx.agentId,
		dbPath: ctx.dbPath,
		deleted,
		orphanEntitiesDeleted,
		orphanAliasesDeleted
	};
}
function wipeDatabase(ctx, store) {
	const deleted = {};
	const count = (table) => Number(store.client.prepare(`SELECT COUNT(*) AS count FROM ${table}`).get().count ?? 0);
	deleted.conversationChunks = count("conversation_chunks");
	deleted.conversationTasks = count("conversation_tasks");
	deleted.states = count("state_kv");
	deleted.facts = count("facts");
	deleted.factVersions = count("fact_versions");
	deleted.events = count("episodic_events");
	deleted.edges = count("graph_edges");
	deleted.entities = count("entities");
	deleted.entityAliases = count("entity_aliases");
	deleted.vectorDocs = count("vector_docs");
	deleted.vectorEmbeddings = count("vector_embeddings");
	deleted.vectorDocsFts = count("vector_docs_fts");
	deleted.retrievalAudit = count("retrieval_audit");
	deleted.policyDecisions = count("policy_decisions");
	deleted.maintenanceRuns = count("maintenance_runs");
	deleted.memorySignals = count("memory_signal_events");
	deleted.beliefs = count("memory_beliefs");
	deleted.abstractionCandidates = count("abstraction_candidates");
	deleted.strategies = count("strategy_hypotheses");
	store.client.withTransaction(() => {
		store.client.exec("DELETE FROM fact_versions;");
		store.client.exec("DELETE FROM vector_embeddings;");
		store.client.exec("DELETE FROM vector_docs_fts;");
		store.client.exec("DELETE FROM vector_docs;");
		store.client.exec("DELETE FROM retrieval_audit;");
		store.client.exec("DELETE FROM policy_decisions;");
		store.client.exec("DELETE FROM maintenance_runs;");
		store.client.exec("DELETE FROM memory_signal_events;");
		store.client.exec("DELETE FROM memory_beliefs;");
		store.client.exec("DELETE FROM abstraction_candidates;");
		store.client.exec("DELETE FROM strategy_hypotheses;");
		store.client.exec("DELETE FROM episodic_events;");
		store.client.exec("DELETE FROM graph_edges;");
		store.client.exec("DELETE FROM entity_aliases;");
		store.client.exec("DELETE FROM entities;");
		store.client.exec("DELETE FROM facts;");
		store.client.exec("DELETE FROM state_kv;");
		store.client.exec("DELETE FROM conversation_chunks;");
		store.client.exec("DELETE FROM conversation_tasks;");
	});
	return {
		dbPath: ctx.dbPath,
		deleted
	};
}
function registerMemxCli(params) {
	const command = params.program.command("memx").description("Manage memX databases");
	command.command("setup").description("Write the recommended memX plugin config into openclaw.json").option("--config <file>", "Path to openclaw.json").option("--llm-judge", "Enable the MemOS-style LLM classifier/reasoner").option("--llm-model <provider/model>", "Optional LLM classifier model override").option("--embedding-provider <provider>", "Embedding provider: off, openai-compatible, ollama, or sentence-transformers-local").option("--embedding-model <model>", "Embedding model id").option("--embedding-python <bin>", "Python binary for sentence-transformers-local").option("--embedding-cache-dir <dir>", "Model cache dir for sentence-transformers-local").option("--embedding-device <device>", "Embedding device: auto, cpu, mps, or cuda").option("--local-embedding", "Shortcut for --embedding-provider sentence-transformers-local --embedding-model intfloat/multilingual-e5-small").action(async (options) => {
		const requestedProvider = options.localEmbedding ? "sentence-transformers-local" : options.embeddingProvider?.trim();
		if (requestedProvider && ![
			"off",
			"openai-compatible",
			"ollama",
			"sentence-transformers-local"
		].includes(requestedProvider)) throw new Error(`unsupported embedding provider: ${requestedProvider}. Expected one of off, openai-compatible, ollama, sentence-transformers-local`);
		const requestedDevice = options.embeddingDevice?.trim();
		if (requestedDevice && ![
			"auto",
			"cpu",
			"mps",
			"cuda"
		].includes(requestedDevice)) throw new Error(`unsupported embedding device: ${requestedDevice}. Expected one of auto, cpu, mps, cuda`);
		const configPath = resolveConfigPath(options.config);
		const next = applyMemxSetupToConfig(await readUserConfig(configPath), params.pluginConfig, {
			enableLlmJudge: Boolean(options.llmJudge),
			llmModel: options.llmModel,
			embeddingProvider: requestedProvider,
			embeddingModel: options.embeddingModel?.trim() || (options.localEmbedding ? "intfloat/multilingual-e5-small" : void 0),
			embeddingPythonBin: options.embeddingPython,
			embeddingCacheDir: options.embeddingCacheDir,
			embeddingDevice: requestedDevice
		});
		await writeUserConfig(configPath, next);
		console.log(JSON.stringify({
			ok: true,
			configPath,
			configuredPlugin: PLUGIN_ID,
			memorySlot: PLUGIN_ID,
			llmClassifierEnabled: ((next.plugins?.entries?.[PLUGIN_ID]?.config)?.advanced)?.llmClassifierEnabled ?? false,
			llmClassifierModel: ((next.plugins?.entries?.[PLUGIN_ID]?.config)?.advanced)?.llmClassifierModel ?? null,
			embeddingProvider: ((next.plugins?.entries?.[PLUGIN_ID]?.config)?.embedding)?.provider ?? null,
			embeddingModel: ((next.plugins?.entries?.[PLUGIN_ID]?.config)?.embedding)?.model ?? null,
			nextStep: "Restart OpenClaw so the updated memX config is applied."
		}, null, 2));
	});
	command.command("doctor").description("Check whether OpenClaw is configured to use memX correctly").option("--config <file>", "Path to openclaw.json").option("--deep", "Run live reasoner probes and report whether LLM semantic extraction is available").action(async (options) => {
		const configPath = resolveConfigPath(options.config);
		const current = await readUserConfig(configPath);
		const report = buildMemxDoctorReport({
			configPath,
			appConfig: current,
			pluginConfig: params.pluginConfig
		});
		if (options.deep) {
			const effectiveConfig = resolveEffectivePluginConfig(current, params.pluginConfig);
			const reasoner = new MemxReasoner(effectiveConfig, {
				warn() {},
				info() {},
				debug() {},
				error() {}
			});
			report.reasonerConfigPath = reasoner.getResolvedJudgeConfigPath();
			const [reasonerProbe, embeddingProbe] = await Promise.all([reasoner.runProbeSuite(), runEmbeddingProbe(effectiveConfig)]);
			report.reasonerProbe = reasonerProbe;
			report.embeddingProbe = embeddingProbe;
		}
		console.log(JSON.stringify(report, null, 2));
	});
	command.command("stats").option("--agent <id>", "Agent id").action(async (options) => {
		await withStore({
			config: params.pluginConfig,
			appConfig: params.appConfig,
			manager: params.manager,
			agent: options.agent,
			run: async (ctx, store) => {
				await printStats(ctx, store);
			}
		});
	});
	command.command("vacuum").option("--agent <id>", "Agent id").action(async (options) => {
		await withStore({
			config: params.pluginConfig,
			appConfig: params.appConfig,
			manager: params.manager,
			agent: options.agent,
			run: async (_ctx, store) => {
				store.client.exec("VACUUM;");
				console.log(JSON.stringify({ ok: true }, null, 2));
			}
		});
	});
	command.command("inspect").requiredOption("--id <id>", "Document id or record id").option("--agent <id>", "Agent id").action(async (options) => {
		await withStore({
			config: params.pluginConfig,
			appConfig: params.appConfig,
			manager: params.manager,
			agent: options.agent,
			run: async (_ctx, store) => {
				const doc = store.vectorRepo.getDoc(options.id);
				console.log(JSON.stringify(doc ?? {
					error: "not found",
					id: options.id
				}, null, 2));
			}
		});
	});
	command.command("export").description("Export stored memory as JSONL").option("--agent <id>", "Agent id").option("--output <file>", "Output file").action(async (options) => {
		await withStore({
			config: params.pluginConfig,
			appConfig: params.appConfig,
			manager: params.manager,
			agent: options.agent,
			run: async (_ctx, store) => {
				const payload = [...store.vectorRepo.listDocs({
					agentId: _ctx.agentId,
					scopes: _ctx.scopes,
					limit: 5e3
				})].map((entry) => JSON.stringify(entry)).join("\n");
				if (options.output) {
					await writeFile(options.output, `${payload}\n`, "utf8");
					console.log(JSON.stringify({
						ok: true,
						output: options.output
					}, null, 2));
					return;
				}
				console.log(payload);
			}
		});
	});
	command.command("export-jsonl").description("Alias for export").option("--agent <id>", "Agent id").option("--output <file>", "Output file").action(async (options) => {
		await withStore({
			config: params.pluginConfig,
			appConfig: params.appConfig,
			manager: params.manager,
			agent: options.agent,
			run: async (ctx, store) => {
				const payload = store.vectorRepo.listDocs({
					agentId: ctx.agentId,
					scopes: ctx.scopes,
					limit: 5e3
				}).map((entry) => JSON.stringify(entry)).join("\n");
				if (options.output) {
					await writeFile(options.output, `${payload}\n`, "utf8");
					console.log(JSON.stringify({
						ok: true,
						output: options.output
					}, null, 2));
					return;
				}
				console.log(payload);
			}
		});
	});
	command.command("forget").requiredOption("--id <id>", "Document id to delete").option("--agent <id>", "Agent id").action(async (options) => {
		await withStore({
			config: params.pluginConfig,
			appConfig: params.appConfig,
			manager: params.manager,
			agent: options.agent,
			run: async (_ctx, store) => {
				store.vectorRepo.deleteDocs([options.id]);
				console.log(JSON.stringify({
					ok: true,
					deleted: options.id
				}, null, 2));
			}
		});
	});
	command.command("wipe").description("Delete all memX data for one agent from the current memx database").requiredOption("--yes", "Confirm the destructive wipe").option("--agent <id>", "Agent id").action(async (options) => {
		if (!options.yes) throw new Error("refusing to wipe memory without --yes");
		await withStore({
			config: params.pluginConfig,
			appConfig: params.appConfig,
			manager: params.manager,
			agent: options.agent,
			run: async (ctx, store) => {
				const stats = wipeAgentMemory(ctx, store);
				console.log(JSON.stringify({
					ok: true,
					...stats
				}, null, 2));
			}
		});
	});
	command.command("wipe-db").description("Delete all data from the current memX database file while keeping the schema").requiredOption("--yes", "Confirm the destructive database wipe").option("--agent <id>", "Agent id used to resolve the target dbPath").action(async (options) => {
		if (!options.yes) throw new Error("refusing to wipe database without --yes");
		await withStore({
			config: params.pluginConfig,
			appConfig: params.appConfig,
			manager: params.manager,
			agent: options.agent,
			run: async (ctx, store) => {
				const stats = wipeDatabase(ctx, store);
				console.log(JSON.stringify({
					ok: true,
					...stats
				}, null, 2));
			}
		});
	});
	command.command("prune").option("--agent <id>", "Agent id").action(async (options) => {
		await withStore({
			config: params.pluginConfig,
			appConfig: params.appConfig,
			manager: params.manager,
			agent: options.agent,
			run: async (ctx, store) => {
				const stats = await runConsolidation(store, {
					...ctx,
					now: nowIso()
				});
				console.log(JSON.stringify(stats, null, 2));
			}
		});
	});
	command.command("consolidate").option("--agent <id>", "Agent id").action(async (options) => {
		await withStore({
			config: params.pluginConfig,
			appConfig: params.appConfig,
			manager: params.manager,
			agent: options.agent,
			run: async (ctx, store) => {
				const stats = await runConsolidation(store, {
					...ctx,
					now: nowIso()
				});
				console.log(JSON.stringify(stats, null, 2));
			}
		});
	});
	command.command("abstraction-jobs").description("Run the standalone abstraction candidate pass without changing canonical memory").option("--agent <id>", "Agent id").option("--refine-llm", "Enable maintenance LLM refinement for abstraction candidates").action(async (options) => {
		await withStore({
			config: params.pluginConfig,
			appConfig: params.appConfig,
			manager: params.manager,
			agent: options.agent,
			run: async (ctx, store) => {
				const stats = await runAbstractionJobs(store, {
					...ctx,
					now: nowIso()
				}, { refineWithLlm: options.refineLlm === true });
				console.log(JSON.stringify(stats, null, 2));
			}
		});
	});
	command.command("abstraction-promote").description("Run the standalone abstraction promotion pass").option("--agent <id>", "Agent id").action(async (options) => {
		await withStore({
			config: params.pluginConfig,
			appConfig: params.appConfig,
			manager: params.manager,
			agent: options.agent,
			run: async (ctx, store) => {
				const stats = runAbstractionPromotion(store, {
					...ctx,
					now: nowIso()
				});
				console.log(JSON.stringify(stats, null, 2));
			}
		});
	});
	command.command("rebuild-fts").option("--agent <id>", "Agent id").action(async (options) => {
		await withStore({
			config: params.pluginConfig,
			appConfig: params.appConfig,
			manager: params.manager,
			agent: options.agent,
			run: async (ctx, store) => {
				const docs = store.vectorRepo.listDocs({
					agentId: ctx.agentId,
					scopes: ctx.scopes,
					limit: 5e3
				});
				store.client.exec("DELETE FROM vector_docs_fts;");
				store.retrievalBackend.upsertDocs(docs);
				console.log(JSON.stringify({
					ok: true,
					rebuilt: docs.length
				}, null, 2));
			}
		});
	});
	command.command("reindex").option("--agent <id>", "Agent id").action(async (options) => {
		await withStore({
			config: params.pluginConfig,
			appConfig: params.appConfig,
			manager: params.manager,
			agent: options.agent,
			run: async (ctx, store) => {
				const docs = store.vectorRepo.listDocs({
					agentId: ctx.agentId,
					scopes: ctx.scopes,
					limit: 5e3
				});
				store.retrievalBackend.upsertDocs(docs);
				console.log(JSON.stringify({
					ok: true,
					reindexed: docs.length
				}, null, 2));
			}
		});
	});
}
//#endregion
export { registerMemxCli };
