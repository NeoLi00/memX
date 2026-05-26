import { normalizeName, nowIso, randomId, stableHash, truncateText } from "../support.mjs";
import { attributeSlotContractHints, requestedAttributeSlotsFromText } from "../pipeline/attributeSlots.mjs";
import { entityNameAliasTerms } from "../pipeline/entityAliases.mjs";
import { compileQuery } from "../pipeline/queryCompiler.mjs";
import { lexicalSearchTerms } from "../search/lexical.mjs";
import { resolveDefaultScope, scopeVarsForContext } from "../security/scopes.mjs";
import { MemxRuntimeManager, buildOperationContext } from "../runtime.mjs";
import { DEFAULT_MEMORY_CONFIG, memxConfigSchema } from "../config.mjs";
import { renderEvidenceBundle, retrieveEvidence } from "../pipeline/retrieve.mjs";
import { captureAgentEndTurn } from "../pipeline/turnCapture.mjs";
import { normalizeObservePayload } from "./hookPayload.mjs";
import { STANDALONE_ALLOWED_SCOPES, STANDALONE_DEFAULT_SCOPE, normalizeStandaloneScopeDefaults } from "./standaloneConfig.mjs";
import { homedir } from "node:os";
import { join } from "node:path";
import { existsSync, readFileSync } from "node:fs";
//#region src/host/service.ts
const DEFAULT_SERVER_DB_PATH = join(homedir(), ".memx", "{agentId}", "memx.sqlite");
const DEFAULT_SERVICE_CONFIG_PATH = join(homedir(), ".memx", "config.json");
function isRecord(value) {
	return Boolean(value) && typeof value === "object" && !Array.isArray(value);
}
function stringArray(value) {
	if (!Array.isArray(value)) return [];
	return value.filter((entry) => typeof entry === "string" && entry.trim().length > 0);
}
function recallEchoTextsFromContext(context) {
	const lines = context.split(/\r?\n/u).map((line) => line.trim()).filter(Boolean).filter((line) => !/^#{1,6}\s+/u.test(line)).map((line) => line.replace(/^\s*(?:[-*]|\d+[.)])\s+/u, "").trim()).filter((line) => line.length >= 12);
	return [...new Set([context.trim(), ...lines])];
}
function recalledTextsFromMetadata(metadata) {
	if (!metadata) return [];
	const direct = stringArray(metadata.recalledTexts);
	const recall = isRecord(metadata.memxRecall) ? metadata.memxRecall : void 0;
	const rawTexts = [
		...direct,
		...stringArray(recall?.injectedTexts),
		...typeof recall?.prependContext === "string" && recall.prependContext.trim() ? [recall.prependContext.trim()] : []
	];
	return [...new Set(rawTexts.flatMap(recallEchoTextsFromContext))];
}
function deepMerge(base, override) {
	if (!isRecord(base) || !isRecord(override)) return override === void 0 ? base : override;
	const output = { ...base };
	for (const [key, value] of Object.entries(override)) output[key] = key in output ? deepMerge(output[key], value) : value;
	return output;
}
function serviceDefaultConfig() {
	const config = structuredClone(DEFAULT_MEMORY_CONFIG);
	config.dbPath = DEFAULT_SERVER_DB_PATH;
	config.defaultScope = STANDALONE_DEFAULT_SCOPE;
	config.allowedScopes = [...STANDALONE_ALLOWED_SCOPES];
	return config;
}
function readServiceConfigFile(path) {
	if (!existsSync(path)) return {};
	return JSON.parse(readFileSync(path, "utf8"));
}
function applyServiceEnvOverrides(config, env) {
	const next = structuredClone(config);
	next.dbPath = env["MEMX_DB_PATH"]?.trim() || next.dbPath;
	next.defaultScope = env["MEMX_DEFAULT_SCOPE"]?.trim() || next.defaultScope;
	if (env["MEMX_LLM_PROVIDER"] === "openai-compatible" || env["MEMX_LLM_PROVIDER"] === "anthropic" || env["MEMX_LLM_PROVIDER"] === "google" || env["MEMX_LLM_PROVIDER"] === "ollama") next.advanced.llmProvider = env["MEMX_LLM_PROVIDER"];
	if (env["MEMX_LLM_BASE_URL"]) next.advanced.llmBaseURL = env["MEMX_LLM_BASE_URL"];
	if (env["MEMX_LLM_API_KEY"]) next.advanced.llmApiKey = env["MEMX_LLM_API_KEY"];
	if (env["MEMX_LLM_MODEL"]) next.advanced.llmClassifierModel = env["MEMX_LLM_MODEL"];
	if (env["MEMX_EMBEDDING_PROVIDER"]) {
		const provider = env["MEMX_EMBEDDING_PROVIDER"];
		if (provider === "off" || provider === "openai-compatible" || provider === "ollama" || provider === "sentence-transformers-local") next.embedding.provider = provider;
	}
	if (env["MEMX_EMBEDDING_MODEL"]) next.embedding.model = env["MEMX_EMBEDDING_MODEL"];
	if (env["MEMX_EMBEDDING_BASE_URL"]) next.embedding.baseURL = env["MEMX_EMBEDDING_BASE_URL"];
	if (env["MEMX_EMBEDDING_API_KEY"]) next.embedding.apiKey = env["MEMX_EMBEDDING_API_KEY"];
	if (env["MEMX_EMBEDDING_OLLAMA_BASE_URL"]) next.embedding.ollamaBaseURL = env["MEMX_EMBEDDING_OLLAMA_BASE_URL"];
	if (env["MEMX_EMBEDDING_PYTHON"]) next.embedding.localPythonBin = env["MEMX_EMBEDDING_PYTHON"];
	if (env["MEMX_EMBEDDING_CACHE_DIR"]) next.embedding.localCacheDir = env["MEMX_EMBEDDING_CACHE_DIR"];
	if (env["MEMX_EMBEDDING_DEVICE"] === "auto" || env["MEMX_EMBEDDING_DEVICE"] === "cpu" || env["MEMX_EMBEDDING_DEVICE"] === "mps" || env["MEMX_EMBEDDING_DEVICE"] === "cuda") next.embedding.localDevice = env["MEMX_EMBEDDING_DEVICE"];
	return memxConfigSchema.parse(next);
}
function loggerOrConsole(logger) {
	return logger ?? {
		warn: (message) => console.warn(message),
		info: (message) => console.error(message),
		debug: () => {},
		error: (message) => console.error(message)
	};
}
function createServiceConfigFromEnv(env = process.env) {
	const configPath = env["MEMX_CONFIG_PATH"]?.trim() || DEFAULT_SERVICE_CONFIG_PATH;
	return applyServiceEnvOverrides(normalizeStandaloneScopeDefaults(memxConfigSchema.parse(deepMerge(serviceDefaultConfig(), readServiceConfigFile(configPath)))), env);
}
function hostSessionKey(envelope) {
	return `${envelope.hostId}:${envelope.sessionId || "default"}`;
}
function safeAgentPart(value, fallback) {
	return (value?.trim() || fallback).replace(/[^A-Za-z0-9._-]+/g, "-").replace(/^-+|-+$/g, "") || fallback;
}
function hostScopedAgentId(envelope) {
	const actor = safeAgentPart(envelope.actorId, "memx-shared");
	if (envelope.hostId === "generic") return actor;
	const hostPrefix = `${envelope.hostId}--`;
	return actor.startsWith(hostPrefix) ? actor : `${hostPrefix}${actor}`;
}
function asEnvelopeContext(config, envelope) {
	const ctx = buildOperationContext(config, {
		agentId: hostScopedAgentId(envelope),
		sessionKey: hostSessionKey(envelope),
		workspaceDir: envelope.workspaceDir,
		project: envelope.project,
		runId: envelope.runId
	});
	if (!ctx) throw new Error("unable to build memX operation context");
	return ctx;
}
function countTable(store, table) {
	const row = store.client.prepare(`SELECT COUNT(*) AS count FROM ${table}`).get();
	return Number(row?.count ?? 0);
}
function graphPathText(path) {
	if (typeof path === "string") return path;
	if (isRecord(path) && typeof path.summary === "string") return path.summary;
	return "";
}
function formatNativeRecallContext(bundle, maxChars) {
	return renderEvidenceBundle(bundle, maxChars);
}
function injectedPackets(bundle) {
	return bundle.evidencePackets.filter((packet) => packet.injected && !packet.dropReason);
}
function finalInjectedPacketAudit(bundle) {
	return injectedPackets(bundle).map((packet) => ({
		packetId: packet.packetId,
		slotId: packet.slotId,
		role: packet.role,
		sourceRefs: packet.sourceRefs,
		score: packet.grade?.finalScore ?? packet.score ?? packet.coverage.confidence,
		selectionReason: packet.selectionReason,
		primaryText: truncateText(packet.primaryText, 720),
		displayLines: (packet.displayLines ?? []).map((line) => truncateText(line, 360))
	}));
}
function bestInjectedPacketScore(packets) {
	return packets.reduce((best, packet) => Math.max(best, packet.grade?.finalScore ?? packet.score ?? packet.coverage.confidence ?? 0), 0);
}
function packetHasNativeAnswerSurface(packet) {
	if (packet.role === "answer" || packet.eligibility?.role === "answer") return true;
	return (packet.displayLines && packet.displayLines.length > 0 ? packet.displayLines : [packet.primaryText]).some((line) => line.startsWith("[answer]") || line.startsWith("[resource]") || packet.operationType === "aggregate" && line.startsWith("[event]"));
}
function packetIsStrongNativeContextEvidence(packet) {
	const score = packet.grade?.finalScore ?? packet.score ?? packet.coverage.confidence ?? 0;
	const slotCoverage = packet.grade?.slotCoverageScore ?? (packet.coverage.filled ? packet.coverage.confidence : 0);
	const contextBinding = packet.grade?.contextBindingScore ?? (packet.coverage.filled ? packet.coverage.confidence : 0);
	return packetHasSourceGroundedEvidence(packet) && packetHasNativeAnswerSurface(packet) && packet.coverage.missing.length === 0 && score >= .62 && slotCoverage >= .45 && contextBinding >= .42;
}
function appendStagedPendingEvidence(bundle, stagedTurns, ctx, query, queryAnalysis) {
	if (stagedTurns.length === 0) return bundle;
	const stagedRows = stagedTurns.map((turn) => ({
		id: `pending-staged:${turn.turnId}`,
		text: turn.text,
		score: .62,
		scope: ctx.scopes[0] ?? `agent:${ctx.agentId}`,
		confidence: .62,
		observedAt: turn.observedAt,
		sourceRef: `pending-staged:${turn.turnId}`,
		lineage: {
			sourceKind: "chunk",
			sourceId: turn.turnId,
			sourceRef: `pending-staged:${turn.turnId}`
		}
	})).filter((row) => stagedPendingTurnCanInject(query, queryAnalysis, row.text));
	if (stagedRows.length === 0) return {
		...bundle,
		diagnostics: [...bundle.diagnostics, "pending-staged-turn-withheld"]
	};
	const stagedPackets = stagedRows.map((row) => {
		const sourceRef = row.sourceRef ?? row.id;
		const score = row.score ?? .62;
		const confidence = row.confidence ?? score;
		return {
			packetId: row.id,
			slotId: "pending-staged-turn",
			operationType: "return_value",
			role: "answer",
			protected: true,
			injected: true,
			layers: ["chunk"],
			primaryText: row.text,
			supportingTexts: [],
			sourceRefs: [sourceRef],
			allSourceRefs: [sourceRef],
			score,
			scoreBreakdown: {
				stagedPendingTurn: true,
				retrievalScore: score
			},
			displayLines: [`[answer] ${truncateText(row.text, 360)}`],
			observedAt: row.observedAt,
			authorRoles: ["user", "assistant"],
			coverage: {
				filled: true,
				missing: [],
				confidence
			},
			eligibility: {
				eligible: true,
				role: "answer",
				blockers: []
			},
			grade: {
				retrievalScore: score,
				answerScore: score,
				contextBindingScore: score,
				slotCoverageScore: score,
				authorityScore: .72,
				finalScore: score
			},
			selectionReason: "pending staged turn evidence while semantic write is still queued"
		};
	});
	return {
		...bundle,
		events: [...stagedRows, ...bundle.events],
		evidencePackets: [...stagedPackets, ...bundle.evidencePackets],
		recalledChunkTexts: [...stagedRows.map((row) => row.text), ...bundle.recalledChunkTexts],
		diagnostics: [...bundle.diagnostics, "pending-staged-turn-evidence"]
	};
}
function packetHasSourceGroundedEvidence(packet) {
	const hasSource = packet.sourceRefs.length > 0 || (packet.allSourceRefs?.length ?? 0) > 0 || (packet.answerUnits ?? []).some((unit) => unit.sourceRefs.length > 0) || (packet.contextUnits ?? []).some((unit) => unit.sourceRefs.length > 0) || (packet.supportUnits ?? []).some((unit) => unit.sourceRefs.length > 0);
	const hasRenderableEvidence = packet.primaryText.trim().length > 0 || packet.supportingTexts.some((text) => text.trim().length > 0) || (packet.displayLines ?? []).some((line) => line.trim().length > 0);
	return hasSource && hasRenderableEvidence;
}
function packetTextForSuppression(packet) {
	return [
		packet.primaryText,
		...packet.supportingTexts,
		...packet.displayLines ?? [],
		...packet.entityAliases ?? [],
		packet.answerCandidate?.text,
		...(packet.contextCandidates ?? []).map((candidate) => candidate.text)
	].filter((value) => typeof value === "string" && value.trim().length > 0).join("\n");
}
function packetMentionsSuppressedEntity(packet, queryAnalysis) {
	const suppressed = queryAnalysis.suppressedEntities ?? [];
	if (suppressed.length === 0) return false;
	const normalizedPacketText = normalizeName(packetTextForSuppression(packet));
	if (!normalizedPacketText) return false;
	return suppressed.some((entity) => {
		return entityNameAliasTerms(entity.name).some((term) => normalizedPacketText.includes(term));
	});
}
function entityFocusTerms(queryAnalysis) {
	const terms = /* @__PURE__ */ new Set();
	for (const entity of queryAnalysis.queryEntities ?? []) for (const term of entityNameAliasTerms(entity.name)) terms.add(term);
	return [...terms];
}
function textMentionsFocusEntity(text, terms) {
	if (!text || terms.length === 0) return false;
	const normalized = normalizeName(text);
	return terms.some((term) => normalized.includes(term));
}
function packetMentionsFocusEntity(packet, terms) {
	return textMentionsFocusEntity(packetTextForSuppression(packet), terms);
}
const QUERY_CONTROL_ANCHOR_STOPWORDS = new Set([
	"answer",
	"confirm",
	"context",
	"default",
	"memory",
	"name",
	"please",
	"project",
	"question",
	"remember",
	"reply",
	"test",
	"today",
	"中文",
	"今天",
	"先只",
	"不用",
	"名字",
	"只",
	"回答",
	"确认",
	"查看",
	"简短",
	"什么",
	"文件",
	"现在",
	"虚构",
	"记住",
	"项目"
]);
const CODE_LIKE_ANCHOR_RE = /[A-Za-z][A-Za-z0-9_.:-]{2,}|[A-Za-z0-9_.:-]*\d[A-Za-z0-9_.:-]*/g;
const QUOTED_ANCHOR_RE = /["'`“”‘’「」『』《》]([^"'`“”‘’「」『』《》]{2,80})["'`“”‘’「」『』《》]/gu;
function hasCjkAnchorText(value) {
	return /[\p{Script=Han}\p{Script=Hiragana}\p{Script=Katakana}\p{Script=Hangul}]/u.test(value);
}
function isDistinctiveAnchorTerm(term) {
	const normalized = normalizeName(term);
	if (!normalized || QUERY_CONTROL_ANCHOR_STOPWORDS.has(normalized)) return false;
	if (hasCjkAnchorText(normalized)) return normalized.length >= 2;
	if (/[0-9_.:-]/u.test(normalized)) return normalized.length >= 3;
	return normalized.length >= 4 && !QUERY_CONTROL_ANCHOR_STOPWORDS.has(normalized);
}
function addQuotedCjkAnchorNgrams(value, terms) {
	for (const match of value.matchAll(/[\p{Script=Han}\p{Script=Hiragana}\p{Script=Katakana}\p{Script=Hangul}]{3,}/gu)) {
		const run = normalizeName(match[0]);
		const width = Math.min(4, run.length);
		for (let index = 0; index <= run.length - width; index += 1) {
			const gram = run.slice(index, index + width);
			if (isDistinctiveAnchorTerm(gram)) terms.add(gram);
		}
	}
}
function queryAnchorTerms(query) {
	const terms = /* @__PURE__ */ new Set();
	const add = (value) => {
		const normalized = normalizeName(value ?? "");
		if (isDistinctiveAnchorTerm(normalized)) terms.add(normalized);
	};
	for (const match of query.matchAll(CODE_LIKE_ANCHOR_RE)) add(match[0]);
	for (const match of query.matchAll(QUOTED_ANCHOR_RE)) {
		add(match[1]);
		addQuotedCjkAnchorNgrams(match[1], terms);
	}
	for (const term of lexicalSearchTerms(query, 96)) add(term);
	return [...terms].slice(0, 24);
}
function queryHardAnchorTerms(query) {
	const terms = /* @__PURE__ */ new Set();
	for (const match of query.matchAll(CODE_LIKE_ANCHOR_RE)) {
		const raw = match[0];
		if (!(/[0-9_.:-]/u.test(raw) || /[a-z][A-Z]/u.test(raw))) continue;
		const normalized = normalizeName(raw);
		if (isDistinctiveAnchorTerm(normalized)) terms.add(normalized);
	}
	return [...terms].slice(0, 8);
}
function requestedAttributeAnchorTerms(query, queryAnalysis) {
	const slots = /* @__PURE__ */ new Set();
	for (const requested of requestedAttributeSlotsFromText(query)) slots.add(requested);
	for (const slot of queryAnalysis.evidencePlan?.slots ?? []) {
		for (const requested of slot.requestedAttributeSlots ?? []) slots.add(requested);
		for (const requested of requestedAttributeSlotsFromText(query, slot.description, ...slot.relationHints ?? [], ...slot.requiredFields ?? [])) slots.add(requested);
	}
	return attributeSlotContractHints([...slots]).map(normalizeName).filter(isDistinctiveAnchorTerm);
}
function queryCompilerIsDegraded(queryAnalysis) {
	const provenance = queryAnalysis.compilerProvenance;
	return provenance?.mode === "fallback" || provenance?.source === "deterministic" || (provenance?.reasons ?? []).some((reason) => /timeout|fallback|unavailable|unparsable/iu.test(reason));
}
function packetQueryAnchorCoverage(packet, terms) {
	const text = packetTextForSuppression(packet);
	let coverage = 0;
	for (const term of terms) if (textMentionsFocusEntity(text, [term])) coverage += 1;
	return coverage;
}
function packetSatisfiesHardAnchor(packet, terms) {
	return terms.length === 0 || packetQueryAnchorCoverage(packet, terms) > 0;
}
function packetSatisfiesDegradedQueryAnchors(packet, terms, hardTerms = []) {
	if (!packetSatisfiesHardAnchor(packet, hardTerms)) return false;
	const requiredCoverage = terms.length >= 2 ? 2 : 1;
	return packetQueryAnchorCoverage(packet, terms) >= requiredCoverage;
}
function compilerAnchorTerms(queryAnalysis) {
	return [...queryAnalysis.anchors ?? [], ...queryAnalysis.evidenceCoverage?.requiredAnchors ?? []].map(normalizeName).filter(isDistinctiveAnchorTerm);
}
function textCoversAnchorTerms(text, terms) {
	if (terms.length === 0) return false;
	const requiredCoverage = terms.length >= 2 ? 2 : 1;
	let coverage = 0;
	for (const term of terms) if (textMentionsFocusEntity(text, [term])) coverage += 1;
	return coverage >= requiredCoverage;
}
function textCoversAnyAnchorTerm(text, terms) {
	if (terms.length === 0) return true;
	return terms.some((term) => textMentionsFocusEntity(text, [term]));
}
function stagedPendingTurnCanInject(query, queryAnalysis, text) {
	const focusTerms = entityFocusTerms(queryAnalysis);
	if (focusTerms.length > 0) return textMentionsFocusEntity(text, focusTerms);
	if (textCoversAnchorTerms(text, compilerAnchorTerms(queryAnalysis))) return true;
	if (queryCompilerIsDegraded(queryAnalysis)) {
		if (!textCoversAnyAnchorTerm(text, queryHardAnchorTerms(query))) return false;
		if (queryAnalysis.queryShape?.referentialMode === "deictic" || queryAnalysis.turnMode === "memory_qa" || queryAnalysis.queryShape?.evidenceNeed === "canonical_state") return true;
		return textCoversAnchorTerms(text, queryAnchorTerms(query));
	}
	if (queryAnalysis.queryShape?.referentialMode === "deictic" || queryAnalysis.turnMode === "memory_qa" || queryAnalysis.queryShape?.evidenceNeed === "canonical_state") return true;
	return false;
}
function queryHasContextExclusion(queryAnalysis) {
	return (queryAnalysis.contextExclusions ?? []).some((exclusion) => exclusion.kind === "prior_project" || exclusion.kind === "prior_topic" || exclusion.kind === "prior_context" || exclusion.kind === "host_native_memory");
}
function hasUsableCompilerAnchors(queryAnalysis) {
	return compilerAnchorTerms(queryAnalysis).length > 0;
}
function queryHasPositiveMemoryBinding(queryAnalysis) {
	if (entityFocusTerms(queryAnalysis).length > 0 || hasUsableCompilerAnchors(queryAnalysis)) return true;
	if (queryAnalysis.queryShape?.referentialMode === "deictic") return true;
	if (queryAnalysis.turnMode === "memory_qa") return true;
	return queryAnalysis.queryShape?.evidenceNeed === "canonical_state";
}
function focusEvidenceRows(rows, terms) {
	return rows.filter((row) => textMentionsFocusEntity(row.text, terms));
}
function hasFocusedEvidence(bundle) {
	return bundle.states.length > 0 || bundle.tasks.length > 0 || bundle.facts.length > 0 || bundle.events.length > 0 || bundle.graph.paths.length > 0 || bundle.behavioralGuidance.length > 0 || bundle.promptEvidence.length > 0 || bundle.evidencePackets.length > 0;
}
function focusRecallBundleForQueryEntities(queryAnalysis, bundle) {
	const terms = entityFocusTerms(queryAnalysis);
	if (terms.length === 0) return bundle;
	const focusedGraphNodes = bundle.graph.nodes.filter((node) => textMentionsFocusEntity(`${node.name} ${node.type}`, terms));
	const focusedNodeIds = new Set(focusedGraphNodes.map((node) => node.nodeId));
	const focused = {
		...bundle,
		states: focusEvidenceRows(bundle.states, terms),
		tasks: focusEvidenceRows(bundle.tasks, terms),
		facts: focusEvidenceRows(bundle.facts, terms),
		events: focusEvidenceRows(bundle.events, terms),
		alternates: focusEvidenceRows(bundle.alternates, terms),
		graph: {
			...bundle.graph,
			nodes: focusedGraphNodes,
			edges: focusedNodeIds.size > 0 ? bundle.graph.edges.filter((edge) => focusedNodeIds.has(edge.srcNodeId) || focusedNodeIds.has(edge.dstNodeId)) : [],
			pathCandidates: bundle.graph.pathCandidates.filter((candidate) => textMentionsFocusEntity(JSON.stringify(candidate), terms)),
			paths: bundle.graph.paths.filter((path) => textMentionsFocusEntity(graphPathText(path), terms))
		},
		behavioralGuidance: bundle.behavioralGuidance.filter((text) => textMentionsFocusEntity(text, terms)),
		recalledChunkTexts: bundle.recalledChunkTexts.filter((text) => textMentionsFocusEntity(text, terms)),
		promptEvidence: bundle.promptEvidence.filter((candidate) => textMentionsFocusEntity([
			candidate.text,
			candidate.rawText,
			candidate.scoringText
		].filter(Boolean).join("\n"), terms)),
		evidencePackets: bundle.evidencePackets.filter((packet) => packetMentionsFocusEntity(packet, terms))
	};
	return hasFocusedEvidence(focused) ? focused : {
		...focused,
		diagnostics: [...bundle.diagnostics, "target-entity-no-focused-evidence"]
	};
}
function focusRecallBundleForDegradedQueryAnchors(query, queryAnalysis, bundle) {
	if (!queryCompilerIsDegraded(queryAnalysis) || entityFocusTerms(queryAnalysis).length > 0) return bundle;
	const terms = queryHardAnchorTerms(query);
	if (terms.length === 0) return bundle;
	const focused = {
		...bundle,
		states: focusEvidenceRows(bundle.states, terms),
		tasks: focusEvidenceRows(bundle.tasks, terms),
		facts: focusEvidenceRows(bundle.facts, terms),
		events: focusEvidenceRows(bundle.events, terms),
		alternates: focusEvidenceRows(bundle.alternates, terms),
		graph: {
			...bundle.graph,
			nodes: bundle.graph.nodes.filter((node) => textMentionsFocusEntity(`${node.name} ${node.type}`, terms)),
			edges: bundle.graph.edges.filter((edge) => textMentionsFocusEntity(JSON.stringify(edge), terms)),
			pathCandidates: bundle.graph.pathCandidates.filter((candidate) => textMentionsFocusEntity(JSON.stringify(candidate), terms)),
			paths: bundle.graph.paths.filter((path) => textMentionsFocusEntity(graphPathText(path), terms))
		},
		behavioralGuidance: bundle.behavioralGuidance.filter((text) => textMentionsFocusEntity(text, terms)),
		recalledChunkTexts: bundle.recalledChunkTexts.filter((text) => textMentionsFocusEntity(text, terms)),
		promptEvidence: bundle.promptEvidence.filter((candidate) => textMentionsFocusEntity([
			candidate.text,
			candidate.rawText,
			candidate.scoringText
		].filter(Boolean).join("\n"), terms)),
		evidencePackets: bundle.evidencePackets.filter((packet) => packetSatisfiesHardAnchor(packet, terms)),
		diagnostics: [...bundle.diagnostics, "degraded-hard-anchor-focused"]
	};
	return hasFocusedEvidence(focused) ? focused : {
		...focused,
		diagnostics: [...focused.diagnostics, "degraded-hard-anchor-no-focused-evidence"]
	};
}
function assessNativeContextEligibility(query, queryAnalysis, bundle) {
	const packets = injectedPackets(bundle);
	if (packets.length === 0) return {
		eligible: false,
		reason: "no-injected-packets",
		bestScore: 0
	};
	const bestScore = bestInjectedPacketScore(packets);
	if (packets.some((packet) => packetMentionsSuppressedEntity(packet, queryAnalysis))) return {
		eligible: false,
		reason: "suppressed-entity-anchor",
		bestScore
	};
	const focusTerms = entityFocusTerms(queryAnalysis);
	if (focusTerms.length > 0 && !packets.some((packet) => packetMentionsFocusEntity(packet, focusTerms))) return {
		eligible: false,
		reason: "target-entity-mismatch",
		bestScore
	};
	const positiveMemoryBinding = queryHasPositiveMemoryBinding(queryAnalysis);
	if (queryHasContextExclusion(queryAnalysis) && !positiveMemoryBinding) return {
		eligible: false,
		reason: "excluded-context",
		bestScore
	};
	if (!packets.some(packetIsStrongNativeContextEvidence)) return {
		eligible: false,
		reason: "weak-evidence",
		bestScore
	};
	const degradedQueryCompiler = queryCompilerIsDegraded(queryAnalysis);
	if (!degradedQueryCompiler && !positiveMemoryBinding) return {
		eligible: false,
		reason: "unbound-context",
		bestScore
	};
	const rawAnchorTerms = focusTerms.length === 0 && degradedQueryCompiler ? [...new Set([...queryAnchorTerms(query), ...requestedAttributeAnchorTerms(query, queryAnalysis)])] : [];
	if (rawAnchorTerms.length > 0) {
		const hardAnchorTerms = queryHardAnchorTerms(query);
		const attributeAnchorTerms = requestedAttributeAnchorTerms(query, queryAnalysis);
		if (hardAnchorTerms.length > 0 && attributeAnchorTerms.length > 0 && packets.some((packet) => packetSatisfiesHardAnchor(packet, hardAnchorTerms) && packetQueryAnchorCoverage(packet, attributeAnchorTerms) > 0)) return {
			eligible: true,
			reason: queryAnalysis.queryEntities.length > 0 ? "llm-query-entities" : "strong-evidence",
			bestScore
		};
		if (Math.max(0, ...packets.map((packet) => packetQueryAnchorCoverage(packet, rawAnchorTerms))) === 0) return {
			eligible: false,
			reason: "query-anchor-mismatch",
			bestScore
		};
		if (!packets.some((packet) => packetSatisfiesDegradedQueryAnchors(packet, rawAnchorTerms, hardAnchorTerms))) return {
			eligible: false,
			reason: "degraded-query-semantic-mismatch",
			bestScore
		};
	}
	return {
		eligible: true,
		reason: queryAnalysis.queryEntities.length > 0 ? "llm-query-entities" : "strong-evidence",
		bestScore
	};
}
var MemxHostService = class {
	config;
	logger;
	manager;
	pendingWrites = /* @__PURE__ */ new Map();
	constructor(options = {}) {
		this.config = options.config ?? createServiceConfigFromEnv();
		this.logger = loggerOrConsole(options.logger);
		this.manager = new MemxRuntimeManager(this.logger);
	}
	async close() {
		await Promise.allSettled([...this.pendingWrites.values()]);
		await this.manager.closeAll();
	}
	pendingWriteKey(ctx) {
		const workspace = ctx.workspaceDir?.trim();
		const scope = workspace ? `workspace:${workspace}` : `session:${ctx.sessionKey ?? "default"}`;
		return `${ctx.agentId}\u0000${ctx.dbPath}\u0000${scope}`;
	}
	hasPendingWrite(ctx) {
		return this.pendingWrites.has(this.pendingWriteKey(ctx));
	}
	enqueuePendingWrite(ctx, work) {
		const key = this.pendingWriteKey(ctx);
		const tracked = (this.pendingWrites.get(key) ?? Promise.resolve()).catch(() => {}).then(work).catch((error) => {
			this.logger.warn(`memx: host observe flush failed (${String(error)})`);
		});
		this.pendingWrites.set(key, tracked);
		tracked.finally(() => {
			if (this.pendingWrites.get(key) === tracked) this.pendingWrites.delete(key);
		});
	}
	async waitForPendingWrites(ctx, hotPathTimeoutMs) {
		const pending = this.pendingWrites.get(this.pendingWriteKey(ctx));
		if (!pending) return 0;
		const startedAt = performance.now();
		const configuredBudget = Number.isFinite(hotPathTimeoutMs ?? NaN) ? Math.max(0, Number(hotPathTimeoutMs)) : 0;
		const timeoutMs = configuredBudget > 0 ? Math.max(0, Math.min(250, configuredBudget - 1500)) : 250;
		if (timeoutMs <= 0) return 0;
		let timeout;
		await Promise.race([pending, new Promise((resolve) => {
			timeout = setTimeout(resolve, timeoutMs);
		})]);
		if (timeout) clearTimeout(timeout);
		return Math.round(performance.now() - startedAt);
	}
	async observe(input) {
		const envelope = normalizeObservePayload(input);
		const ctx = asEnvelopeContext(this.config, envelope);
		const store = await this.manager.getStore(ctx);
		const scope = resolveDefaultScope(this.config, scopeVarsForContext(ctx));
		const turnId = randomId("turn");
		const captured = captureAgentEndTurn({
			agentId: ctx.agentId,
			scope,
			sessionKey: ctx.sessionKey ?? "default",
			turnId,
			observedAt: envelope.observedAt || nowIso(),
			messages: envelope.messages,
			recalledTexts: recalledTextsFromMetadata(envelope.metadata)
		});
		if (captured.length === 0) return {
			ok: true,
			accepted: false,
			reason: "no-capturable-messages"
		};
		try {
			if (await store.turnScheduler.stageRecallableTurn(ctx, captured)) this.manager.rememberStagedRecallableTurn(ctx, captured);
		} catch (error) {
			this.logger.warn?.(`memx: fast turn staging failed (${String(error)})`);
		}
		this.enqueuePendingWrite(ctx, async () => {
			await store.turnScheduler.enqueue(ctx, captured);
			await this.manager.recordMaintenanceTurn(ctx, {
				store,
				turnId,
				observedAt: captured.at(-1)?.observedAt ?? ctx.now
			});
		});
		return {
			ok: true,
			accepted: true,
			hostId: envelope.hostId,
			actorId: ctx.agentId,
			sessionKey: ctx.sessionKey,
			turnId,
			captured: captured.length
		};
	}
	async recall(request) {
		if (!request.query?.trim()) throw new Error("query required");
		const envelope = {
			hostId: request.hostId === "codex" || request.hostId === "claude-code" ? request.hostId : "generic",
			actorId: request.actorId || process.env["MEMX_ACTOR_ID"] || "memx-shared",
			sessionId: request.sessionId || "mcp",
			workspaceDir: request.workspaceDir || process.cwd(),
			project: request.project,
			eventName: "recall",
			observedAt: nowIso(),
			messages: [{
				role: "user",
				content: request.query
			}]
		};
		const ctx = asEnvelopeContext(this.config, envelope);
		const waitElapsedMs = await this.waitForPendingWrites(ctx, request.hotPathTimeoutMs);
		const store = await this.manager.getStore({
			...ctx,
			readEpoch: 0
		});
		const recallCtx = {
			...ctx,
			readEpoch: store.client.currentMemoryEpoch(ctx.agentId)
		};
		const remainingHotPathTimeoutMs = typeof request.hotPathTimeoutMs === "number" && Number.isFinite(request.hotPathTimeoutMs) ? Math.max(500, request.hotPathTimeoutMs - waitElapsedMs) : request.hotPathTimeoutMs;
		const compiled = await compileQuery({
			query: request.query,
			ctx: recallCtx,
			reasoner: store.reasoner,
			hotPathTimeoutMs: remainingHotPathTimeoutMs
		});
		const bundle = await retrieveEvidence(store, recallCtx, request.query, compiled.focusedQuery, { queryAnalysis: compiled });
		const focusedBundle = appendStagedPendingEvidence(focusRecallBundleForDegradedQueryAnchors(request.query, compiled, focusRecallBundleForQueryEntities(compiled, bundle)), this.hasPendingWrite(ctx) ? this.manager.recentStagedRecallableTurns(ctx, 4) : [], ctx, request.query, compiled);
		const limit = Math.max(1, Math.min(Math.trunc(request.limit ?? 6), 24));
		const contextEligibility = assessNativeContextEligibility(request.query, compiled, focusedBundle);
		const graphPaths = Array.isArray(focusedBundle.graph?.paths) ? focusedBundle.graph.paths : [];
		const graphEdges = Array.isArray(focusedBundle.graph?.edges) ? focusedBundle.graph.edges : [];
		const nativeContext = formatNativeRecallContext(focusedBundle, this.config.maxInjectedChars);
		return {
			ok: true,
			routeType: focusedBundle.routeType,
			routeConfidence: focusedBundle.routeConfidence,
			focusedQuery: compiled.focusedQuery,
			context: nativeContext,
			contextEligibility,
			states: focusedBundle.states.slice(0, limit),
			facts: focusedBundle.facts.slice(0, limit),
			events: focusedBundle.events.slice(0, limit),
			graph: {
				paths: graphPaths.slice(0, Math.min(limit, 6)),
				edges: graphEdges.slice(0, Math.min(limit, 12))
			},
			diagnostics: focusedBundle.diagnostics,
			audit: {
				finalInjectedPackets: finalInjectedPacketAudit(focusedBundle),
				finalDiagnostics: focusedBundle.diagnostics
			}
		};
	}
	async remember(request) {
		const content = typeof request.content === "string" ? request.content.trim() : "";
		if (!content) throw new Error("content required");
		return this.observe({
			hostId: request.hostId ?? "generic",
			actorId: request.actorId ?? process.env["MEMX_ACTOR_ID"] ?? "memx-shared",
			sessionId: request.sessionId ?? "manual",
			workspaceDir: request.workspaceDir ?? process.cwd(),
			eventName: "remember",
			observedAt: nowIso(),
			messages: [{
				role: "user",
				content
			}],
			metadata: {
				manual: true,
				memoryType: request.type
			}
		});
	}
	async forget(request) {
		const id = typeof request.id === "string" ? request.id.trim() : "";
		if (!id) throw new Error("id required");
		const ctx = asEnvelopeContext(this.config, {
			hostId: "generic",
			actorId: typeof request.actorId === "string" ? request.actorId : "memx-shared",
			sessionId: typeof request.sessionId === "string" ? request.sessionId : "manual"
		});
		const store = await this.manager.getStore(ctx);
		const kind = typeof request.kind === "string" ? request.kind : "doc";
		let deleted = 0;
		if (kind === "event") {
			deleted = store.eventRepo.delete({
				agentId: ctx.agentId,
				eventId: id
			});
			store.vectorRepo.deleteDocs([`event:${id}`]);
		} else if (kind === "fact") {
			deleted = store.factRepo.markDeleted({
				agentId: ctx.agentId,
				factId: id
			});
			store.vectorRepo.deleteDocs([`fact:${id}`]);
		} else if (kind === "state") {
			deleted = store.stateRepo.delete({
				agentId: ctx.agentId,
				key: id
			});
			store.vectorRepo.deleteDocs([`state:${id}`]);
		} else {
			store.vectorRepo.deleteDocs([id]);
			deleted = 1;
		}
		return {
			ok: true,
			deleted,
			kind,
			id
		};
	}
	async stats(request = {}) {
		const ctx = asEnvelopeContext(this.config, {
			hostId: request.hostId === "codex" || request.hostId === "claude-code" ? request.hostId : "generic",
			actorId: request.actorId || process.env["MEMX_ACTOR_ID"] || "memx-shared",
			sessionId: request.sessionId || "stats",
			workspaceDir: request.workspaceDir,
			project: request.project
		});
		const store = await this.manager.getStore(ctx);
		return {
			ok: true,
			agentId: ctx.agentId,
			dbPath: ctx.dbPath,
			scopes: ctx.scopes,
			taskCount: countTable(store, "conversation_tasks"),
			chunkCount: countTable(store, "conversation_chunks"),
			stateCount: countTable(store, "state_kv"),
			factCount: countTable(store, "facts"),
			eventCount: countTable(store, "episodic_events"),
			edgeCount: countTable(store, "graph_edges"),
			vectorDocCount: countTable(store, "vector_docs")
		};
	}
	async audit(limit = 50, request = {}) {
		const ctx = asEnvelopeContext(this.config, {
			hostId: request.hostId === "codex" || request.hostId === "claude-code" ? request.hostId : "generic",
			actorId: request.actorId || process.env["MEMX_ACTOR_ID"] || "memx-shared",
			sessionId: request.sessionId || "audit",
			workspaceDir: request.workspaceDir,
			project: request.project
		});
		const store = await this.manager.getStore(ctx);
		const boundedLimit = Math.max(1, Math.min(Math.trunc(limit), 200));
		return {
			ok: true,
			agentId: ctx.agentId,
			dbPath: ctx.dbPath,
			scopes: ctx.scopes,
			signals: store.auditRepo.listSignals({
				agentId: ctx.agentId,
				sessionKey: ctx.sessionKey,
				limit: boundedLimit
			}),
			retrievals: store.auditRepo.listRetrievals({
				agentId: ctx.agentId,
				sessionKey: ctx.sessionKey,
				limit: boundedLimit
			}),
			policyDecisions: store.auditRepo.listPolicyDecisions({
				agentId: ctx.agentId,
				sessionKey: ctx.sessionKey,
				limit: boundedLimit
			}),
			maintenanceRuns: store.auditRepo.listMaintenanceRuns({
				agentId: ctx.agentId,
				sessionKey: ctx.sessionKey,
				limit: boundedLimit
			}),
			semanticWriteJobs: store.auditRepo.listSemanticWriteJobs({
				agentId: ctx.agentId,
				sessionKey: ctx.sessionKey,
				limit: boundedLimit
			}),
			maintenanceSchedulerStates: store.maintenanceRepo.listPendingStates().filter((state) => state.agentId === ctx.agentId && state.sessionKey === ctx.sessionKey).slice(0, boundedLimit)
		};
	}
	async context(request) {
		const recalled = await this.recall(request);
		const eligibility = recalled.contextEligibility;
		const candidateContext = typeof recalled.context === "string" ? recalled.context : "";
		const prependContext = eligibility?.eligible === false ? "" : candidateContext;
		if (eligibility && !eligibility.eligible) this.logger.info?.(`memx: native context withheld reason=${eligibility.reason} best=${eligibility.bestScore.toFixed(2)} query="${request.query.slice(0, 80)}"`);
		try {
			const envelope = {
				hostId: request.hostId === "codex" || request.hostId === "claude-code" ? request.hostId : "generic",
				actorId: request.actorId || process.env["MEMX_ACTOR_ID"] || "memx-shared",
				sessionId: request.sessionId || "mcp",
				workspaceDir: request.workspaceDir || process.cwd(),
				project: request.project,
				eventName: "context",
				observedAt: nowIso(),
				messages: [{
					role: "user",
					content: request.query
				}]
			};
			const ctx = asEnvelopeContext(this.config, envelope);
			const store = await this.manager.getStore(ctx);
			const auditPayload = isRecord(recalled.audit) ? recalled.audit : {};
			const finalInjectedPackets = Array.isArray(auditPayload.finalInjectedPackets) ? auditPayload.finalInjectedPackets : [];
			const finalDiagnostics = Array.isArray(auditPayload.finalDiagnostics) ? auditPayload.finalDiagnostics.filter((entry) => typeof entry === "string") : void 0;
			store.auditRepo.annotateLatestRetrievalInjection({
				agentId: ctx.agentId,
				sessionKey: ctx.sessionKey,
				queryText: request.query,
				candidateChars: candidateContext.length,
				actualInjectedChars: prependContext.length,
				actualContextPreview: truncateText(prependContext, 1600),
				finalInjectedPackets: prependContext ? finalInjectedPackets : [],
				finalDiagnostics,
				eligible: eligibility?.eligible ?? true,
				reason: eligibility?.reason,
				finalizedAt: ctx.now
			});
			store.auditRepo.recordSignal({
				signalId: randomId("signal"),
				agentId: ctx.agentId,
				scope: ctx.scopes[0] ?? `agent:${ctx.agentId}`,
				sessionKey: ctx.sessionKey,
				signalType: "retrieval_support",
				memoryKind: "chunk",
				semanticKey: "native_context_injection",
				value: prependContext ? 1 : 0,
				sourceRef: `native_context:${stableHash([
					request.query,
					ctx.sessionKey,
					ctx.now
				])}`,
				metadataJson: {
					eligible: eligibility?.eligible ?? true,
					reason: eligibility?.reason,
					bestScore: eligibility?.bestScore,
					candidateChars: candidateContext.length,
					actualInjectedChars: prependContext.length
				},
				createdAt: ctx.now
			});
		} catch (error) {
			this.logger.debug?.(`memx: native context audit failed (${String(error)})`);
		}
		return {
			ok: true,
			prependContext,
			nativeContext: {
				eligible: eligibility?.eligible ?? true,
				reason: eligibility?.reason,
				candidateChars: candidateContext.length,
				actualInjectedChars: prependContext.length
			},
			recall: recalled
		};
	}
};
function stableHostTurnId(envelope) {
	return stableHash([
		envelope.hostId,
		envelope.actorId,
		envelope.sessionId,
		envelope.observedAt
	]);
}
//#endregion
export { MemxHostService, assessNativeContextEligibility, createServiceConfigFromEnv, focusRecallBundleForDegradedQueryAnchors, focusRecallBundleForQueryEntities, formatNativeRecallContext, stableHostTurnId };
