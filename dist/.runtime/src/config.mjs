import { MEMX_NATIVE_HOOK_TIMEOUT_MS, deriveNativeHookBudget } from "./timeouts.mjs";
import { MEMORY_CONSENT_MODES, MEMORY_EMBEDDING_PROVIDERS, MEMORY_LLM_PROVIDERS, MEMORY_PII_MODES, MEMORY_SCOPE_TEMPLATES } from "./types.mjs";
import { homedir } from "node:os";
import { join } from "node:path";
//#region src/config.ts
const DEFAULT_DB_PATH = join(homedir(), ".openclaw", "agents", "{agentId}", "memory", "memx.sqlite");
const DEFAULT_EMBEDDING = {
	provider: "sentence-transformers-local",
	model: "intfloat/multilingual-e5-small",
	localPythonBin: "python3",
	localDevice: "auto"
};
const DEFAULT_ADVANCED = {
	llmClassifierEnabled: true,
	enableMaintenanceJobs: true,
	maintenanceTriggerMode: "batched",
	maintenanceBatchTurns: 3,
	maintenanceIdleFlushMinutes: 10,
	enableGraphPromotion: true,
	enableFactPromotion: true,
	enableTelemetryAudit: true,
	enableExplicitRecallTool: true,
	suggestExplicitRecallTool: false,
	enableCompatibilityMemoryTools: false,
	enableTurnScheduler: true,
	chunkDedupThreshold: .84,
	taskIdleTimeoutMinutes: 120,
	recallChunkBudget: 8,
	recallTotalObjectBudget: 24,
	recallBackgroundCharReserve: 700,
	recallPromptBudgetFloor: 220,
	recallObjectiveMinWeight: .12,
	recallObjectiveOverflowRatio: .2,
	recallProbeWorkflowStrongThreshold: .72,
	recallProbeWorkflowContinuationThreshold: .58,
	recallProbeFactualStrongThreshold: .76,
	recallProbeFactualShortQueryThreshold: .62,
	recallProbeHybridStrongThreshold: .84,
	recallProbeHybridModerateThreshold: .74,
	recallProbeEscalateThreshold: .72,
	recallProbeContinuationEscalateThreshold: .68,
	enableTurnSemanticCompiler: true,
	enableQueryCompiler: true,
	queryCompilerHotPathTimeoutMs: deriveNativeHookBudget(MEMX_NATIVE_HOOK_TIMEOUT_MS).queryCompilerTimeoutMs,
	enableEmbeddingCandidates: true,
	enableEmbeddingClustering: true,
	enableHotPathChunkSummaryLlm: false,
	enableHotPathTaskSummaryLlm: false,
	candidateSurfaceBudgets: {
		state: 4,
		fact: 6,
		event: 6,
		task: 4,
		chunk: 6,
		graph: 4,
		entityAlias: 3
	}
};
const DEFAULT_MEMORY_CONFIG = {
	enabled: true,
	dbPath: DEFAULT_DB_PATH,
	autoCapture: true,
	autoRecall: true,
	reflectionEnabled: true,
	consentMode: "implicit",
	maxInjectedChars: 3500,
	captureMaxChars: 800,
	reflectionMaxChars: 1e3,
	reflectionMaxItems: 8,
	piiMode: "redact",
	defaultScope: "agent:{agentId}",
	allowedScopes: [...MEMORY_SCOPE_TEMPLATES],
	minSalienceDurable: .72,
	minSalienceSession: .45,
	minUtilityForGraph: .7,
	maxSensitivityAllowed: 1,
	stateTtlHours: 168,
	episodicDedupWindowDays: 3,
	graphMaxHops: 2,
	maxGraphNodes: 12,
	maxGraphEdges: 16,
	embedding: DEFAULT_EMBEDDING,
	advanced: DEFAULT_ADVANCED
};
const uiHints = {
	enabled: { label: "Enabled" },
	dbPath: {
		label: "Database Path",
		placeholder: DEFAULT_DB_PATH,
		advanced: true
	},
	autoCapture: {
		label: "Auto Capture",
		help: "Capture conversation turns by default and derive memory from them with the MemOS-style turn scheduler."
	},
	autoRecall: {
		label: "Auto Recall",
		help: "Inject recalled memory context and proactive reply guidance before each run."
	},
	reflectionEnabled: {
		label: "Reflection Enabled",
		help: "Allow bounded end-of-run reflection over turn memory and tiered memory."
	},
	consentMode: {
		label: "Consent Mode",
		help: "implicit matches the MemOS-style automatic memory pipeline; off disables background capture."
	},
	maxInjectedChars: {
		label: "Max Injected Chars",
		advanced: true
	},
	captureMaxChars: {
		label: "Capture Max Chars",
		advanced: true
	},
	reflectionMaxChars: {
		label: "Reflection Max Chars",
		advanced: true
	},
	reflectionMaxItems: {
		label: "Reflection Max Items",
		advanced: true
	},
	piiMode: {
		label: "PII Mode",
		help: "allow stores raw memory text; redact/off remain available when stricter governance is needed."
	},
	defaultScope: {
		label: "Default Scope",
		help: "Default capture scope template. Supports {agentId}, {sessionKey}, {project}, and hashed {workspace}."
	},
	allowedScopes: {
		label: "Allowed Scopes",
		help: "Allowed scope templates for tools and automatic capture."
	},
	minSalienceDurable: {
		label: "Min Durable Salience",
		advanced: true
	},
	minSalienceSession: {
		label: "Min Session Salience",
		advanced: true
	},
	minUtilityForGraph: {
		label: "Min Graph Utility",
		advanced: true
	},
	maxSensitivityAllowed: {
		label: "Max Sensitivity Allowed",
		advanced: true
	},
	stateTtlHours: {
		label: "State TTL Hours",
		advanced: true
	},
	episodicDedupWindowDays: {
		label: "Event Dedup Window Days",
		advanced: true
	},
	graphMaxHops: {
		label: "Graph Max Hops",
		advanced: true
	},
	maxGraphNodes: {
		label: "Max Graph Nodes",
		advanced: true
	},
	maxGraphEdges: {
		label: "Max Graph Edges",
		advanced: true
	},
	"embedding.provider": { label: "Embedding Provider" },
	"embedding.baseURL": {
		label: "Embedding Base URL",
		advanced: true
	},
	"embedding.apiKey": {
		label: "Embedding API Key",
		sensitive: true,
		advanced: true
	},
	"embedding.model": {
		label: "Embedding Model",
		advanced: true
	},
	"embedding.dimensions": {
		label: "Embedding Dimensions",
		advanced: true
	},
	"embedding.ollamaBaseURL": {
		label: "Ollama Base URL",
		advanced: true
	},
	"embedding.ollamaModel": {
		label: "Ollama Model",
		advanced: true
	},
	"embedding.localPythonBin": {
		label: "Local Python Bin",
		advanced: true
	},
	"embedding.localCacheDir": {
		label: "Local Model Cache Dir",
		advanced: true
	},
	"embedding.localDevice": {
		label: "Local Embedding Device",
		advanced: true
	},
	"advanced.llmClassifierEnabled": {
		label: "LLM Classifier Enabled",
		help: "Enable maintenance-only LLM upgrades and compiler/model-assisted semantic calls. This no longer re-enables legacy hot-path judges or recall planners.",
		advanced: true
	},
	"advanced.llmClassifierModel": {
		label: "LLM Classifier Model",
		help: "Optional provider/model override in provider/model form. Falls back to the default OpenClaw agent model when omitted.",
		advanced: true
	},
	"advanced.llmProvider": {
		label: "LLM Provider",
		help: "Standalone memX LLM provider used by non-OpenClaw hosts.",
		advanced: true
	},
	"advanced.llmBaseURL": {
		label: "LLM Base URL",
		help: "Standalone memX LLM endpoint base URL.",
		advanced: true
	},
	"advanced.llmApiKey": {
		label: "LLM API Key",
		sensitive: true,
		advanced: true
	},
	"advanced.enableMaintenanceJobs": {
		label: "Maintenance Jobs",
		advanced: true
	},
	"advanced.maintenanceTriggerMode": {
		label: "Maintenance Trigger Mode",
		help: "batched aggregates automatic maintenance by session before flushing; per_turn restores the older fire-and-forget behavior.",
		advanced: true
	},
	"advanced.maintenanceBatchTurns": {
		label: "Maintenance Batch Turns",
		help: "How many turns to accumulate per session before automatic maintenance flushes a batch.",
		advanced: true
	},
	"advanced.maintenanceIdleFlushMinutes": {
		label: "Maintenance Idle Flush Minutes",
		help: "Flush a pending maintenance batch after this many idle minutes even if the turn threshold was not reached.",
		advanced: true
	},
	"advanced.enableGraphPromotion": {
		label: "Graph Promotion",
		advanced: true
	},
	"advanced.enableFactPromotion": {
		label: "Fact Promotion",
		advanced: true
	},
	"advanced.enableTelemetryAudit": {
		label: "Telemetry Audit",
		advanced: true
	},
	"advanced.enableExplicitRecallTool": {
		label: "Explicit Recall Tool",
		help: "Expose the high-level memory_recall tool to the agent. Disable this to prevent explicit recall tool calls entirely.",
		advanced: true
	},
	"advanced.suggestExplicitRecallTool": {
		label: "Suggest Explicit Recall Tool",
		help: "When enabled, memory prompts may explicitly suggest calling memory_recall. Leave off to keep prompts conversational and reduce tool-chasing on weak-memory turns.",
		advanced: true
	},
	"advanced.enableCompatibilityMemoryTools": {
		label: "Legacy Memory Tools",
		help: "Opt in to legacy memory_search/memory_get compatibility tools. Leave off to avoid MEMORY.md-style recall prompts.",
		advanced: true
	},
	"advanced.enableTurnScheduler": {
		label: "Turn Scheduler",
		help: "Use the MemOS-style turn capture -> chunk -> task -> derived-memory pipeline.",
		advanced: true
	},
	"advanced.enableTurnSemanticCompiler": {
		label: "Turn Semantic Compiler",
		help: "Compile a turn into a single LLM semantic draft before storage materialization.",
		advanced: true
	},
	"advanced.enableQueryCompiler": {
		label: "Query Compiler",
		help: "Compile recall queries into structured query-shape and surface budgets before retrieval.",
		advanced: true
	},
	"advanced.enableEmbeddingCandidates": {
		label: "Embedding Candidates",
		help: "Enable per-surface hybrid candidate generation with embedding-backed candidates and lexical fallback.",
		advanced: true
	},
	"advanced.enableEmbeddingClustering": {
		label: "Embedding Clustering",
		help: "Enable embedding-assisted clustering for maintenance and control-plane abstractions.",
		advanced: true
	},
	"advanced.enableHotPathChunkSummaryLlm": {
		label: "Hot Path Chunk Summary LLM",
		help: "Deprecated no-op. Semantic extraction is LLM-only; hot-path chunk summaries remain local summaries.",
		advanced: true
	},
	"advanced.enableHotPathTaskSummaryLlm": {
		label: "Hot Path Task Summary LLM",
		help: "Deprecated no-op. Semantic extraction is LLM-only; task summary upgrades happen in maintenance.",
		advanced: true
	},
	"advanced.candidateSurfaceBudgets": {
		label: "Candidate Surface Budgets",
		help: "Per-surface retrieval maxima used by candidate generation before deterministic selection.",
		advanced: true
	},
	"advanced.chunkDedupThreshold": {
		label: "Chunk Dedup Threshold",
		advanced: true
	},
	"advanced.taskIdleTimeoutMinutes": {
		label: "Task Idle Timeout Minutes",
		advanced: true
	},
	"advanced.recallChunkBudget": {
		label: "Recall Chunk Budget",
		advanced: true
	},
	"advanced.recallTotalObjectBudget": {
		label: "Recall Total Object Budget",
		advanced: true
	},
	"advanced.recallBackgroundCharReserve": {
		label: "Recall Background Char Reserve",
		advanced: true
	},
	"advanced.recallPromptBudgetFloor": {
		label: "Recall Prompt Budget Floor",
		advanced: true
	},
	"advanced.recallObjectiveMinWeight": {
		label: "Recall Objective Min Weight",
		advanced: true
	},
	"advanced.recallObjectiveOverflowRatio": {
		label: "Recall Objective Overflow Ratio",
		advanced: true
	},
	"advanced.recallProbeWorkflowStrongThreshold": {
		label: "Recall Probe Workflow Strong",
		advanced: true
	},
	"advanced.recallProbeWorkflowContinuationThreshold": {
		label: "Recall Probe Workflow Continuation",
		advanced: true
	},
	"advanced.recallProbeFactualStrongThreshold": {
		label: "Recall Probe Factual Strong",
		advanced: true
	},
	"advanced.recallProbeFactualShortQueryThreshold": {
		label: "Recall Probe Factual Short Query",
		advanced: true
	},
	"advanced.recallProbeHybridStrongThreshold": {
		label: "Recall Probe Hybrid Strong",
		advanced: true
	},
	"advanced.recallProbeHybridModerateThreshold": {
		label: "Recall Probe Hybrid Moderate",
		advanced: true
	},
	"advanced.recallProbeEscalateThreshold": {
		label: "Recall Probe Escalate",
		advanced: true
	},
	"advanced.recallProbeContinuationEscalateThreshold": {
		label: "Recall Probe Continuation Escalate",
		advanced: true
	}
};
const envSecretRefJsonSchema = {
	type: "object",
	additionalProperties: false,
	properties: {
		source: {
			type: "string",
			enum: ["env"]
		},
		provider: { type: "string" },
		id: { type: "string" }
	},
	required: ["source", "id"]
};
const secretLikeStringJsonSchema = {
	type: ["string", "object"],
	additionalProperties: false,
	properties: envSecretRefJsonSchema.properties,
	required: envSecretRefJsonSchema.required
};
const jsonSchema = {
	type: "object",
	additionalProperties: false,
	properties: {
		enabled: { type: "boolean" },
		dbPath: { type: "string" },
		autoCapture: { type: "boolean" },
		autoRecall: { type: "boolean" },
		reflectionEnabled: { type: "boolean" },
		consentMode: {
			type: "string",
			enum: [...MEMORY_CONSENT_MODES]
		},
		maxInjectedChars: {
			type: "number",
			minimum: 200,
			maximum: 12e3
		},
		captureMaxChars: {
			type: "number",
			minimum: 100,
			maximum: 1e4
		},
		reflectionMaxChars: {
			type: "number",
			minimum: 100,
			maximum: 4e3
		},
		reflectionMaxItems: {
			type: "number",
			minimum: 1,
			maximum: 32
		},
		piiMode: {
			type: "string",
			enum: [...MEMORY_PII_MODES]
		},
		defaultScope: { type: "string" },
		allowedScopes: {
			type: "array",
			items: { type: "string" }
		},
		minSalienceDurable: {
			type: "number",
			minimum: 0,
			maximum: 1
		},
		minSalienceSession: {
			type: "number",
			minimum: 0,
			maximum: 1
		},
		minUtilityForGraph: {
			type: "number",
			minimum: 0,
			maximum: 1
		},
		maxSensitivityAllowed: {
			type: "number",
			minimum: 0,
			maximum: 1
		},
		stateTtlHours: {
			type: "number",
			minimum: 1,
			maximum: 24 * 365
		},
		episodicDedupWindowDays: {
			type: "number",
			minimum: 1,
			maximum: 365
		},
		graphMaxHops: {
			type: "number",
			minimum: 1,
			maximum: 4
		},
		maxGraphNodes: {
			type: "number",
			minimum: 1,
			maximum: 64
		},
		maxGraphEdges: {
			type: "number",
			minimum: 1,
			maximum: 128
		},
		embedding: {
			type: "object",
			additionalProperties: false,
			properties: {
				provider: {
					type: "string",
					enum: [...MEMORY_EMBEDDING_PROVIDERS]
				},
				baseURL: { type: "string" },
				apiKey: { type: "string" },
				model: { type: "string" },
				dimensions: {
					type: "number",
					minimum: 1,
					maximum: 8192
				},
				headers: {
					type: "object",
					additionalProperties: { type: "string" }
				},
				ollamaBaseURL: { type: "string" },
				ollamaModel: { type: "string" },
				localPythonBin: { type: "string" },
				localCacheDir: { type: "string" },
				localDevice: {
					type: "string",
					enum: [
						"auto",
						"cpu",
						"mps",
						"cuda"
					]
				}
			}
		},
		advanced: {
			type: "object",
			additionalProperties: false,
			properties: {
				llmClassifierEnabled: { type: "boolean" },
				llmClassifierModel: { type: "string" },
				llmProvider: {
					type: "string",
					enum: [...MEMORY_LLM_PROVIDERS]
				},
				llmBaseURL: { type: "string" },
				llmApiKey: secretLikeStringJsonSchema,
				llmHeaders: {
					type: "object",
					additionalProperties: { type: "string" }
				},
				enableMaintenanceJobs: { type: "boolean" },
				maintenanceTriggerMode: {
					type: "string",
					enum: ["batched", "per_turn"]
				},
				maintenanceBatchTurns: {
					type: "number",
					minimum: 1,
					maximum: 32
				},
				maintenanceIdleFlushMinutes: {
					type: "number",
					minimum: 0,
					maximum: 1440
				},
				enableGraphPromotion: { type: "boolean" },
				enableFactPromotion: { type: "boolean" },
				enableTelemetryAudit: { type: "boolean" },
				enableExplicitRecallTool: { type: "boolean" },
				suggestExplicitRecallTool: { type: "boolean" },
				enableCompatibilityMemoryTools: { type: "boolean" },
				enableTurnScheduler: { type: "boolean" },
				enableTurnSemanticCompiler: { type: "boolean" },
				enableQueryCompiler: { type: "boolean" },
				queryCompilerHotPathTimeoutMs: {
					type: "number",
					minimum: 250,
					maximum: 15e3
				},
				enableEmbeddingCandidates: { type: "boolean" },
				enableEmbeddingClustering: { type: "boolean" },
				enableHotPathChunkSummaryLlm: { type: "boolean" },
				enableHotPathTaskSummaryLlm: { type: "boolean" },
				candidateSurfaceBudgets: {
					type: "object",
					additionalProperties: false,
					properties: {
						state: {
							type: "number",
							minimum: 0,
							maximum: 16
						},
						fact: {
							type: "number",
							minimum: 0,
							maximum: 24
						},
						event: {
							type: "number",
							minimum: 0,
							maximum: 24
						},
						task: {
							type: "number",
							minimum: 0,
							maximum: 16
						},
						chunk: {
							type: "number",
							minimum: 0,
							maximum: 24
						},
						graph: {
							type: "number",
							minimum: 0,
							maximum: 16
						},
						entityAlias: {
							type: "number",
							minimum: 0,
							maximum: 16
						}
					}
				},
				chunkDedupThreshold: {
					type: "number",
					minimum: 0,
					maximum: 1
				},
				taskIdleTimeoutMinutes: {
					type: "number",
					minimum: 1,
					maximum: 1440
				},
				recallChunkBudget: {
					type: "number",
					minimum: 1,
					maximum: 32
				},
				recallTotalObjectBudget: {
					type: "number",
					minimum: 4,
					maximum: 128
				},
				recallBackgroundCharReserve: {
					type: "number",
					minimum: 0,
					maximum: 6e3
				},
				recallPromptBudgetFloor: {
					type: "number",
					minimum: 0,
					maximum: 4e3
				},
				recallObjectiveMinWeight: {
					type: "number",
					minimum: 0,
					maximum: 1
				},
				recallObjectiveOverflowRatio: {
					type: "number",
					minimum: 0,
					maximum: 1
				},
				recallProbeWorkflowStrongThreshold: {
					type: "number",
					minimum: 0,
					maximum: 1
				},
				recallProbeWorkflowContinuationThreshold: {
					type: "number",
					minimum: 0,
					maximum: 1
				},
				recallProbeFactualStrongThreshold: {
					type: "number",
					minimum: 0,
					maximum: 1
				},
				recallProbeFactualShortQueryThreshold: {
					type: "number",
					minimum: 0,
					maximum: 1
				},
				recallProbeHybridStrongThreshold: {
					type: "number",
					minimum: 0,
					maximum: 1
				},
				recallProbeHybridModerateThreshold: {
					type: "number",
					minimum: 0,
					maximum: 1
				},
				recallProbeEscalateThreshold: {
					type: "number",
					minimum: 0,
					maximum: 1
				},
				recallProbeContinuationEscalateThreshold: {
					type: "number",
					minimum: 0,
					maximum: 1
				}
			}
		}
	}
};
function resolveEnvString(value) {
	return value.replace(/\$\{([^}]+)\}/g, (_match, name) => {
		const resolved = process.env[name];
		if (resolved === void 0) throw new Error(`Environment variable ${name} is not set`);
		return resolved;
	});
}
function envBooleanOverride(name, current) {
	const raw = process.env[name];
	if (raw === void 0) return current;
	if (/^(?:1|true|yes|on)$/iu.test(raw.trim())) return true;
	if (/^(?:0|false|no|off)$/iu.test(raw.trim())) return false;
	return current;
}
function envNumberOverride(name, current) {
	const raw = process.env[name];
	if (raw === void 0) return current;
	const parsed = Number(raw);
	return Number.isFinite(parsed) ? Math.max(250, Math.min(15e3, parsed)) : current;
}
function resolveEnvObject(value) {
	if (typeof value === "string") return resolveEnvString(value);
	if (Array.isArray(value)) return value.map((entry) => resolveEnvObject(entry));
	if (value && typeof value === "object") {
		const output = {};
		for (const [key, entry] of Object.entries(value)) output[key] = resolveEnvObject(entry);
		return output;
	}
	return value;
}
function isRecord(value) {
	return Boolean(value) && typeof value === "object" && !Array.isArray(value);
}
function asNumber(raw, fallback, issues, path, opts = {}) {
	const value = typeof raw === "number" && Number.isFinite(raw) ? raw : fallback;
	if (opts.min !== void 0 && value < opts.min) issues.push({
		path: [path],
		message: `must be >= ${opts.min}`
	});
	if (opts.max !== void 0 && value > opts.max) issues.push({
		path: [path],
		message: `must be <= ${opts.max}`
	});
	return value;
}
function asBoolean(raw, fallback) {
	return typeof raw === "boolean" ? raw : fallback;
}
function asString(raw, fallback) {
	return typeof raw === "string" && raw.trim() ? raw.trim() : fallback;
}
function asEnum(raw, allowed, fallback, issues, path) {
	if (typeof raw !== "string" || !allowed.includes(raw)) {
		if (raw !== void 0) issues.push({
			path: [path],
			message: `must be one of: ${allowed.join(", ")}`
		});
		return fallback;
	}
	return raw;
}
function asStringArray(raw, fallback) {
	if (!Array.isArray(raw)) return fallback;
	const values = raw.filter((entry) => typeof entry === "string").map((entry) => entry.trim()).filter(Boolean);
	return values.length > 0 ? values : fallback;
}
function asSecretLikeString(raw) {
	if (typeof raw === "string") return raw.trim() || void 0;
	if (!isRecord(raw) || raw.source !== "env" || typeof raw.id !== "string") return;
	const id = raw.id.trim();
	if (!id) return;
	return {
		source: "env",
		...typeof raw.provider === "string" && raw.provider.trim() ? { provider: raw.provider.trim() } : {},
		id
	};
}
function parseConfigInternal(input) {
	const rawRoot = isRecord(input) ? resolveEnvObject(input) : {};
	const issues = [];
	const rawEmbedding = isRecord(rawRoot.embedding) ? rawRoot.embedding : {};
	const rawAdvanced = isRecord(rawRoot.advanced) ? rawRoot.advanced : {};
	const embedding = {
		provider: asEnum(rawEmbedding.provider, MEMORY_EMBEDDING_PROVIDERS, DEFAULT_MEMORY_CONFIG.embedding.provider, issues, "embedding.provider"),
		baseURL: typeof rawEmbedding.baseURL === "string" ? rawEmbedding.baseURL.trim() : void 0,
		apiKey: typeof rawEmbedding.apiKey === "string" ? rawEmbedding.apiKey.trim() : void 0,
		model: typeof rawEmbedding.model === "string" ? rawEmbedding.model.trim() : void 0,
		dimensions: typeof rawEmbedding.dimensions === "number" && Number.isFinite(rawEmbedding.dimensions) ? Math.trunc(rawEmbedding.dimensions) : void 0,
		headers: isRecord(rawEmbedding.headers) ? Object.fromEntries(Object.entries(rawEmbedding.headers).filter(([, value]) => typeof value === "string").map(([key, value]) => [key, String(value)])) : void 0,
		ollamaBaseURL: typeof rawEmbedding.ollamaBaseURL === "string" ? rawEmbedding.ollamaBaseURL.trim() : void 0,
		ollamaModel: typeof rawEmbedding.ollamaModel === "string" ? rawEmbedding.ollamaModel.trim() : void 0,
		localPythonBin: typeof rawEmbedding.localPythonBin === "string" ? rawEmbedding.localPythonBin.trim() : void 0,
		localCacheDir: typeof rawEmbedding.localCacheDir === "string" ? rawEmbedding.localCacheDir.trim() : void 0,
		localDevice: rawEmbedding.localDevice === "auto" || rawEmbedding.localDevice === "cpu" || rawEmbedding.localDevice === "mps" || rawEmbedding.localDevice === "cuda" ? rawEmbedding.localDevice : void 0
	};
	if (embedding.provider === "openai-compatible") {
		if (!embedding.baseURL) issues.push({
			path: ["embedding.baseURL"],
			message: "required for openai-compatible"
		});
		if (!embedding.apiKey) issues.push({
			path: ["embedding.apiKey"],
			message: "required for openai-compatible"
		});
		if (!embedding.model) issues.push({
			path: ["embedding.model"],
			message: "required for openai-compatible"
		});
	}
	if (embedding.provider === "ollama" && !embedding.ollamaModel && !embedding.model) issues.push({
		path: ["embedding.ollamaModel"],
		message: "required for ollama"
	});
	if (embedding.provider === "sentence-transformers-local") {
		embedding.model = embedding.model?.trim() ?? "intfloat/multilingual-e5-small";
		if (embedding.localDevice && ![
			"auto",
			"cpu",
			"mps",
			"cuda"
		].includes(embedding.localDevice)) issues.push({
			path: ["embedding.localDevice"],
			message: "must be auto, cpu, mps, or cuda"
		});
	}
	const advanced = {
		llmClassifierEnabled: asBoolean(rawAdvanced.llmClassifierEnabled, DEFAULT_MEMORY_CONFIG.advanced.llmClassifierEnabled),
		llmClassifierModel: typeof rawAdvanced.llmClassifierModel === "string" ? rawAdvanced.llmClassifierModel.trim() : void 0,
		llmProvider: rawAdvanced.llmProvider === "openai-compatible" || rawAdvanced.llmProvider === "anthropic" || rawAdvanced.llmProvider === "google" || rawAdvanced.llmProvider === "ollama" ? rawAdvanced.llmProvider : void 0,
		llmBaseURL: typeof rawAdvanced.llmBaseURL === "string" ? rawAdvanced.llmBaseURL.trim() : void 0,
		llmApiKey: asSecretLikeString(rawAdvanced.llmApiKey),
		llmHeaders: isRecord(rawAdvanced.llmHeaders) ? Object.fromEntries(Object.entries(rawAdvanced.llmHeaders).filter(([, value]) => typeof value === "string").map(([key, value]) => [key, String(value)])) : void 0,
		enableMaintenanceJobs: asBoolean(rawAdvanced.enableMaintenanceJobs, DEFAULT_MEMORY_CONFIG.advanced.enableMaintenanceJobs),
		maintenanceTriggerMode: rawAdvanced.maintenanceTriggerMode === "batched" || rawAdvanced.maintenanceTriggerMode === "per_turn" ? rawAdvanced.maintenanceTriggerMode : DEFAULT_MEMORY_CONFIG.advanced.maintenanceTriggerMode,
		maintenanceBatchTurns: asNumber(rawAdvanced.maintenanceBatchTurns, DEFAULT_MEMORY_CONFIG.advanced.maintenanceBatchTurns, issues, "advanced.maintenanceBatchTurns", {
			min: 1,
			max: 32
		}),
		maintenanceIdleFlushMinutes: asNumber(rawAdvanced.maintenanceIdleFlushMinutes, DEFAULT_MEMORY_CONFIG.advanced.maintenanceIdleFlushMinutes, issues, "advanced.maintenanceIdleFlushMinutes", {
			min: 0,
			max: 1440
		}),
		enableGraphPromotion: asBoolean(rawAdvanced.enableGraphPromotion, DEFAULT_MEMORY_CONFIG.advanced.enableGraphPromotion),
		enableFactPromotion: asBoolean(rawAdvanced.enableFactPromotion, DEFAULT_MEMORY_CONFIG.advanced.enableFactPromotion),
		enableTelemetryAudit: asBoolean(rawAdvanced.enableTelemetryAudit, DEFAULT_MEMORY_CONFIG.advanced.enableTelemetryAudit),
		enableExplicitRecallTool: asBoolean(rawAdvanced.enableExplicitRecallTool, DEFAULT_MEMORY_CONFIG.advanced.enableExplicitRecallTool),
		suggestExplicitRecallTool: asBoolean(rawAdvanced.suggestExplicitRecallTool, DEFAULT_MEMORY_CONFIG.advanced.suggestExplicitRecallTool),
		enableCompatibilityMemoryTools: asBoolean(rawAdvanced.enableCompatibilityMemoryTools, DEFAULT_MEMORY_CONFIG.advanced.enableCompatibilityMemoryTools),
		enableTurnScheduler: asBoolean(rawAdvanced.enableTurnScheduler, DEFAULT_MEMORY_CONFIG.advanced.enableTurnScheduler),
		enableTurnSemanticCompiler: asBoolean(rawAdvanced.enableTurnSemanticCompiler, DEFAULT_MEMORY_CONFIG.advanced.enableTurnSemanticCompiler),
		enableQueryCompiler: asBoolean(rawAdvanced.enableQueryCompiler, DEFAULT_MEMORY_CONFIG.advanced.enableQueryCompiler),
		queryCompilerHotPathTimeoutMs: asNumber(rawAdvanced.queryCompilerHotPathTimeoutMs, DEFAULT_MEMORY_CONFIG.advanced.queryCompilerHotPathTimeoutMs, issues, "advanced.queryCompilerHotPathTimeoutMs", {
			min: 250,
			max: 15e3
		}),
		enableEmbeddingCandidates: asBoolean(rawAdvanced.enableEmbeddingCandidates, DEFAULT_MEMORY_CONFIG.advanced.enableEmbeddingCandidates),
		enableEmbeddingClustering: asBoolean(rawAdvanced.enableEmbeddingClustering, DEFAULT_MEMORY_CONFIG.advanced.enableEmbeddingClustering),
		enableHotPathChunkSummaryLlm: asBoolean(rawAdvanced.enableHotPathChunkSummaryLlm, DEFAULT_MEMORY_CONFIG.advanced.enableHotPathChunkSummaryLlm),
		enableHotPathTaskSummaryLlm: asBoolean(rawAdvanced.enableHotPathTaskSummaryLlm, DEFAULT_MEMORY_CONFIG.advanced.enableHotPathTaskSummaryLlm),
		candidateSurfaceBudgets: {
			state: asNumber(rawAdvanced.candidateSurfaceBudgets?.state, DEFAULT_MEMORY_CONFIG.advanced.candidateSurfaceBudgets.state, issues, "advanced.candidateSurfaceBudgets.state", {
				min: 0,
				max: 16
			}),
			fact: asNumber(rawAdvanced.candidateSurfaceBudgets?.fact, DEFAULT_MEMORY_CONFIG.advanced.candidateSurfaceBudgets.fact, issues, "advanced.candidateSurfaceBudgets.fact", {
				min: 0,
				max: 24
			}),
			event: asNumber(rawAdvanced.candidateSurfaceBudgets?.event, DEFAULT_MEMORY_CONFIG.advanced.candidateSurfaceBudgets.event, issues, "advanced.candidateSurfaceBudgets.event", {
				min: 0,
				max: 24
			}),
			task: asNumber(rawAdvanced.candidateSurfaceBudgets?.task, DEFAULT_MEMORY_CONFIG.advanced.candidateSurfaceBudgets.task, issues, "advanced.candidateSurfaceBudgets.task", {
				min: 0,
				max: 16
			}),
			chunk: asNumber(rawAdvanced.candidateSurfaceBudgets?.chunk, DEFAULT_MEMORY_CONFIG.advanced.candidateSurfaceBudgets.chunk, issues, "advanced.candidateSurfaceBudgets.chunk", {
				min: 0,
				max: 24
			}),
			graph: asNumber(rawAdvanced.candidateSurfaceBudgets?.graph, DEFAULT_MEMORY_CONFIG.advanced.candidateSurfaceBudgets.graph, issues, "advanced.candidateSurfaceBudgets.graph", {
				min: 0,
				max: 16
			}),
			entityAlias: asNumber(rawAdvanced.candidateSurfaceBudgets?.entityAlias, DEFAULT_MEMORY_CONFIG.advanced.candidateSurfaceBudgets.entityAlias, issues, "advanced.candidateSurfaceBudgets.entityAlias", {
				min: 0,
				max: 16
			})
		},
		chunkDedupThreshold: asNumber(rawAdvanced.chunkDedupThreshold, DEFAULT_MEMORY_CONFIG.advanced.chunkDedupThreshold, issues, "advanced.chunkDedupThreshold", {
			min: 0,
			max: 1
		}),
		taskIdleTimeoutMinutes: asNumber(rawAdvanced.taskIdleTimeoutMinutes, DEFAULT_MEMORY_CONFIG.advanced.taskIdleTimeoutMinutes, issues, "advanced.taskIdleTimeoutMinutes", {
			min: 1,
			max: 1440
		}),
		recallChunkBudget: asNumber(rawAdvanced.recallChunkBudget, DEFAULT_MEMORY_CONFIG.advanced.recallChunkBudget, issues, "advanced.recallChunkBudget", {
			min: 1,
			max: 32
		}),
		recallTotalObjectBudget: asNumber(rawAdvanced.recallTotalObjectBudget, DEFAULT_MEMORY_CONFIG.advanced.recallTotalObjectBudget, issues, "advanced.recallTotalObjectBudget", {
			min: 4,
			max: 128
		}),
		recallBackgroundCharReserve: asNumber(rawAdvanced.recallBackgroundCharReserve, DEFAULT_MEMORY_CONFIG.advanced.recallBackgroundCharReserve, issues, "advanced.recallBackgroundCharReserve", {
			min: 0,
			max: 6e3
		}),
		recallPromptBudgetFloor: asNumber(rawAdvanced.recallPromptBudgetFloor, DEFAULT_MEMORY_CONFIG.advanced.recallPromptBudgetFloor, issues, "advanced.recallPromptBudgetFloor", {
			min: 0,
			max: 4e3
		}),
		recallObjectiveMinWeight: asNumber(rawAdvanced.recallObjectiveMinWeight, DEFAULT_MEMORY_CONFIG.advanced.recallObjectiveMinWeight, issues, "advanced.recallObjectiveMinWeight", {
			min: 0,
			max: 1
		}),
		recallObjectiveOverflowRatio: asNumber(rawAdvanced.recallObjectiveOverflowRatio, DEFAULT_MEMORY_CONFIG.advanced.recallObjectiveOverflowRatio, issues, "advanced.recallObjectiveOverflowRatio", {
			min: 0,
			max: 1
		}),
		recallProbeWorkflowStrongThreshold: asNumber(rawAdvanced.recallProbeWorkflowStrongThreshold, DEFAULT_MEMORY_CONFIG.advanced.recallProbeWorkflowStrongThreshold, issues, "advanced.recallProbeWorkflowStrongThreshold", {
			min: 0,
			max: 1
		}),
		recallProbeWorkflowContinuationThreshold: asNumber(rawAdvanced.recallProbeWorkflowContinuationThreshold, DEFAULT_MEMORY_CONFIG.advanced.recallProbeWorkflowContinuationThreshold, issues, "advanced.recallProbeWorkflowContinuationThreshold", {
			min: 0,
			max: 1
		}),
		recallProbeFactualStrongThreshold: asNumber(rawAdvanced.recallProbeFactualStrongThreshold, DEFAULT_MEMORY_CONFIG.advanced.recallProbeFactualStrongThreshold, issues, "advanced.recallProbeFactualStrongThreshold", {
			min: 0,
			max: 1
		}),
		recallProbeFactualShortQueryThreshold: asNumber(rawAdvanced.recallProbeFactualShortQueryThreshold, DEFAULT_MEMORY_CONFIG.advanced.recallProbeFactualShortQueryThreshold, issues, "advanced.recallProbeFactualShortQueryThreshold", {
			min: 0,
			max: 1
		}),
		recallProbeHybridStrongThreshold: asNumber(rawAdvanced.recallProbeHybridStrongThreshold, DEFAULT_MEMORY_CONFIG.advanced.recallProbeHybridStrongThreshold, issues, "advanced.recallProbeHybridStrongThreshold", {
			min: 0,
			max: 1
		}),
		recallProbeHybridModerateThreshold: asNumber(rawAdvanced.recallProbeHybridModerateThreshold, DEFAULT_MEMORY_CONFIG.advanced.recallProbeHybridModerateThreshold, issues, "advanced.recallProbeHybridModerateThreshold", {
			min: 0,
			max: 1
		}),
		recallProbeEscalateThreshold: asNumber(rawAdvanced.recallProbeEscalateThreshold, DEFAULT_MEMORY_CONFIG.advanced.recallProbeEscalateThreshold, issues, "advanced.recallProbeEscalateThreshold", {
			min: 0,
			max: 1
		}),
		recallProbeContinuationEscalateThreshold: asNumber(rawAdvanced.recallProbeContinuationEscalateThreshold, DEFAULT_MEMORY_CONFIG.advanced.recallProbeContinuationEscalateThreshold, issues, "advanced.recallProbeContinuationEscalateThreshold", {
			min: 0,
			max: 1
		})
	};
	advanced.enableTurnSemanticCompiler = envBooleanOverride("MEMX_ENABLE_TURN_SEMANTIC_COMPILER", advanced.enableTurnSemanticCompiler);
	advanced.enableQueryCompiler = envBooleanOverride("MEMX_ENABLE_QUERY_COMPILER", advanced.enableQueryCompiler);
	advanced.queryCompilerHotPathTimeoutMs = envNumberOverride("MEMX_QUERY_COMPILER_TIMEOUT_MS", advanced.queryCompilerHotPathTimeoutMs);
	advanced.enableEmbeddingCandidates = envBooleanOverride("MEMX_ENABLE_EMBEDDING_CANDIDATES", advanced.enableEmbeddingCandidates);
	advanced.enableEmbeddingClustering = envBooleanOverride("MEMX_ENABLE_EMBEDDING_CLUSTERING", advanced.enableEmbeddingClustering);
	advanced.enableHotPathChunkSummaryLlm = envBooleanOverride("MEMX_ENABLE_HOT_PATH_CHUNK_SUMMARY_LLM", advanced.enableHotPathChunkSummaryLlm);
	advanced.enableHotPathTaskSummaryLlm = envBooleanOverride("MEMX_ENABLE_HOT_PATH_TASK_SUMMARY_LLM", advanced.enableHotPathTaskSummaryLlm);
	const value = {
		enabled: asBoolean(rawRoot.enabled, DEFAULT_MEMORY_CONFIG.enabled),
		dbPath: asString(rawRoot.dbPath, DEFAULT_MEMORY_CONFIG.dbPath),
		autoCapture: asBoolean(rawRoot.autoCapture, DEFAULT_MEMORY_CONFIG.autoCapture),
		autoRecall: asBoolean(rawRoot.autoRecall, DEFAULT_MEMORY_CONFIG.autoRecall),
		reflectionEnabled: asBoolean(rawRoot.reflectionEnabled, DEFAULT_MEMORY_CONFIG.reflectionEnabled),
		consentMode: asEnum(rawRoot.consentMode, MEMORY_CONSENT_MODES, DEFAULT_MEMORY_CONFIG.consentMode, issues, "consentMode"),
		maxInjectedChars: asNumber(rawRoot.maxInjectedChars, DEFAULT_MEMORY_CONFIG.maxInjectedChars, issues, "maxInjectedChars", {
			min: 200,
			max: 12e3
		}),
		captureMaxChars: asNumber(rawRoot.captureMaxChars, DEFAULT_MEMORY_CONFIG.captureMaxChars, issues, "captureMaxChars", {
			min: 100,
			max: 1e4
		}),
		reflectionMaxChars: asNumber(rawRoot.reflectionMaxChars, DEFAULT_MEMORY_CONFIG.reflectionMaxChars, issues, "reflectionMaxChars", {
			min: 100,
			max: 4e3
		}),
		reflectionMaxItems: asNumber(rawRoot.reflectionMaxItems, DEFAULT_MEMORY_CONFIG.reflectionMaxItems, issues, "reflectionMaxItems", {
			min: 1,
			max: 32
		}),
		piiMode: asEnum(rawRoot.piiMode, MEMORY_PII_MODES, DEFAULT_MEMORY_CONFIG.piiMode, issues, "piiMode"),
		defaultScope: asString(rawRoot.defaultScope, DEFAULT_MEMORY_CONFIG.defaultScope),
		allowedScopes: asStringArray(rawRoot.allowedScopes, DEFAULT_MEMORY_CONFIG.allowedScopes),
		minSalienceDurable: asNumber(rawRoot.minSalienceDurable, DEFAULT_MEMORY_CONFIG.minSalienceDurable, issues, "minSalienceDurable", {
			min: 0,
			max: 1
		}),
		minSalienceSession: asNumber(rawRoot.minSalienceSession, DEFAULT_MEMORY_CONFIG.minSalienceSession, issues, "minSalienceSession", {
			min: 0,
			max: 1
		}),
		minUtilityForGraph: asNumber(rawRoot.minUtilityForGraph, DEFAULT_MEMORY_CONFIG.minUtilityForGraph, issues, "minUtilityForGraph", {
			min: 0,
			max: 1
		}),
		maxSensitivityAllowed: asNumber(rawRoot.maxSensitivityAllowed, DEFAULT_MEMORY_CONFIG.maxSensitivityAllowed, issues, "maxSensitivityAllowed", {
			min: 0,
			max: 1
		}),
		stateTtlHours: asNumber(rawRoot.stateTtlHours, DEFAULT_MEMORY_CONFIG.stateTtlHours, issues, "stateTtlHours", {
			min: 1,
			max: 24 * 365
		}),
		episodicDedupWindowDays: asNumber(rawRoot.episodicDedupWindowDays, DEFAULT_MEMORY_CONFIG.episodicDedupWindowDays, issues, "episodicDedupWindowDays", {
			min: 1,
			max: 365
		}),
		graphMaxHops: asNumber(rawRoot.graphMaxHops, DEFAULT_MEMORY_CONFIG.graphMaxHops, issues, "graphMaxHops", {
			min: 1,
			max: 4
		}),
		maxGraphNodes: asNumber(rawRoot.maxGraphNodes, DEFAULT_MEMORY_CONFIG.maxGraphNodes, issues, "maxGraphNodes", {
			min: 1,
			max: 64
		}),
		maxGraphEdges: asNumber(rawRoot.maxGraphEdges, DEFAULT_MEMORY_CONFIG.maxGraphEdges, issues, "maxGraphEdges", {
			min: 1,
			max: 128
		}),
		embedding,
		advanced
	};
	return {
		value: issues.length === 0 ? value : void 0,
		issues
	};
}
const memxConfigSchema = {
	parse(input) {
		const result = parseConfigInternal(input);
		if (result.value) return result.value;
		const message = result.issues.map((issue) => `${issue.path.join(".")}: ${issue.message}`).join("; ");
		throw new Error(message || "invalid memx config");
	},
	safeParse(input) {
		const result = parseConfigInternal(input);
		if (result.value) return {
			success: true,
			data: result.value
		};
		return {
			success: false,
			error: { issues: result.issues }
		};
	},
	validate(input) {
		const result = parseConfigInternal(input);
		return result.value ? {
			ok: true,
			value: result.value
		} : {
			ok: false,
			errors: result.issues.map((issue) => `${issue.path.join(".")}: ${issue.message}`)
		};
	},
	uiHints,
	jsonSchema
};
//#endregion
export { DEFAULT_MEMORY_CONFIG, memxConfigSchema };
