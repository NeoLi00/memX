import { existsSync, readFileSync } from "node:fs";
import { homedir } from "node:os";
import { join } from "node:path";
import { DEFAULT_MEMORY_CONFIG, memxConfigSchema } from "../config.js";
import { compileQuery } from "../pipeline/queryCompiler.js";
import { renderEvidenceBundle, retrieveEvidence } from "../pipeline/retrieve.js";
import { captureAgentEndTurn } from "../pipeline/turnCapture.js";
import { buildOperationContext, MemxRuntimeManager, type MemxStoreBundle } from "../runtime.js";
import { resolveDefaultScope } from "../security/scopes.js";
import { normalizeName, nowIso, randomId, stableHash, truncateText } from "../support.js";
import type {
  EvidenceBundle,
  EvidencePacket,
  MemoryOperationContext,
  MemoryPluginConfig,
  MemxLogger,
  QueryCompileResult,
} from "../types.js";
import { normalizeObservePayload, type MemxTurnEnvelope } from "./hookPayload.js";

const DEFAULT_SERVER_DB_PATH = join(homedir(), ".memx", "{agentId}", "memx.sqlite");
const DEFAULT_SERVICE_CONFIG_PATH = join(homedir(), ".memx", "config.json");

export type MemxServiceOptions = {
  config?: MemoryPluginConfig;
  logger?: MemxLogger;
};

export type MemxRecallRequest = {
  query: string;
  limit?: number;
  hostId?: string;
  actorId?: string;
  sessionId?: string;
  workspaceDir?: string;
  project?: string;
  hotPathTimeoutMs?: number;
};

export type MemxAgentRequest = {
  hostId?: string;
  actorId?: string;
  sessionId?: string;
  workspaceDir?: string;
  project?: string;
};

function isRecord(value: unknown): value is Record<string, unknown> {
  return Boolean(value) && typeof value === "object" && !Array.isArray(value);
}

function deepMerge<T>(base: T, override: unknown): T {
  if (!isRecord(base) || !isRecord(override)) {
    return (override === undefined ? base : override) as T;
  }
  const output: Record<string, unknown> = { ...base };
  for (const [key, value] of Object.entries(override)) {
    output[key] = key in output ? deepMerge(output[key], value) : value;
  }
  return output as T;
}

function serviceDefaultConfig(): MemoryPluginConfig {
  const config = structuredClone(DEFAULT_MEMORY_CONFIG);
  config.dbPath = DEFAULT_SERVER_DB_PATH;
  config.defaultScope = "agent:{agentId}";
  config.allowedScopes = ["global", "agent:{agentId}", "session:{sessionKey}", "project:{project}"];
  return config;
}

function readServiceConfigFile(path: string): Record<string, unknown> {
  if (!existsSync(path)) {
    return {};
  }
  return JSON.parse(readFileSync(path, "utf8")) as Record<string, unknown>;
}

function applyServiceEnvOverrides(config: MemoryPluginConfig, env: NodeJS.ProcessEnv): MemoryPluginConfig {
  const next = structuredClone(config);
  next.dbPath = env["MEMX_DB_PATH"]?.trim() || next.dbPath;
  next.defaultScope = env["MEMX_DEFAULT_SCOPE"]?.trim() || next.defaultScope;
  if (
    env["MEMX_LLM_PROVIDER"] === "openai-compatible" ||
    env["MEMX_LLM_PROVIDER"] === "anthropic" ||
    env["MEMX_LLM_PROVIDER"] === "google" ||
    env["MEMX_LLM_PROVIDER"] === "ollama"
  ) {
    next.advanced.llmProvider = env["MEMX_LLM_PROVIDER"];
  }
  if (env["MEMX_LLM_BASE_URL"]) {
    next.advanced.llmBaseURL = env["MEMX_LLM_BASE_URL"];
  }
  if (env["MEMX_LLM_API_KEY"]) {
    next.advanced.llmApiKey = env["MEMX_LLM_API_KEY"];
  }
  if (env["MEMX_LLM_MODEL"]) {
    next.advanced.llmClassifierModel = env["MEMX_LLM_MODEL"];
  }
  if (env["MEMX_EMBEDDING_PROVIDER"]) {
    const provider = env["MEMX_EMBEDDING_PROVIDER"];
    if (
      provider === "off" ||
      provider === "openai-compatible" ||
      provider === "ollama" ||
      provider === "sentence-transformers-local"
    ) {
      next.embedding.provider = provider;
    }
  }
  if (env["MEMX_EMBEDDING_MODEL"]) {
    next.embedding.model = env["MEMX_EMBEDDING_MODEL"];
  }
  if (env["MEMX_EMBEDDING_BASE_URL"]) {
    next.embedding.baseURL = env["MEMX_EMBEDDING_BASE_URL"];
  }
  if (env["MEMX_EMBEDDING_API_KEY"]) {
    next.embedding.apiKey = env["MEMX_EMBEDDING_API_KEY"];
  }
  if (env["MEMX_EMBEDDING_OLLAMA_BASE_URL"]) {
    next.embedding.ollamaBaseURL = env["MEMX_EMBEDDING_OLLAMA_BASE_URL"];
  }
  if (env["MEMX_EMBEDDING_PYTHON"]) {
    next.embedding.localPythonBin = env["MEMX_EMBEDDING_PYTHON"];
  }
  if (env["MEMX_EMBEDDING_CACHE_DIR"]) {
    next.embedding.localCacheDir = env["MEMX_EMBEDDING_CACHE_DIR"];
  }
  if (
    env["MEMX_EMBEDDING_DEVICE"] === "auto" ||
    env["MEMX_EMBEDDING_DEVICE"] === "cpu" ||
    env["MEMX_EMBEDDING_DEVICE"] === "mps" ||
    env["MEMX_EMBEDDING_DEVICE"] === "cuda"
  ) {
    next.embedding.localDevice = env["MEMX_EMBEDDING_DEVICE"];
  }
  return memxConfigSchema.parse!(next) as MemoryPluginConfig;
}

function loggerOrConsole(logger?: MemxLogger): MemxLogger {
  return (
    logger ?? {
      warn: (message) => console.warn(message),
      info: (message) => console.error(message),
      debug: () => {},
      error: (message) => console.error(message),
    }
  );
}

export function createServiceConfigFromEnv(env: NodeJS.ProcessEnv = process.env): MemoryPluginConfig {
  const configPath = env["MEMX_CONFIG_PATH"]?.trim() || DEFAULT_SERVICE_CONFIG_PATH;
  const raw = deepMerge(serviceDefaultConfig(), readServiceConfigFile(configPath));
  return applyServiceEnvOverrides(memxConfigSchema.parse!(raw) as MemoryPluginConfig, env);
}

function hostSessionKey(envelope: Pick<MemxTurnEnvelope, "hostId" | "sessionId">): string {
  return `${envelope.hostId}:${envelope.sessionId || "default"}`;
}

function safeAgentPart(value: string | undefined, fallback: string): string {
  const safe = (value?.trim() || fallback)
    .replace(/[^A-Za-z0-9._-]+/g, "-")
    .replace(/^-+|-+$/g, "");
  return safe || fallback;
}

function hostScopedAgentId(envelope: Pick<MemxTurnEnvelope, "hostId" | "actorId">): string {
  const actor = safeAgentPart(envelope.actorId, "memx-shared");
  if (envelope.hostId === "generic") {
    return actor;
  }
  const hostPrefix = `${envelope.hostId}--`;
  return actor.startsWith(hostPrefix) ? actor : `${hostPrefix}${actor}`;
}

function asEnvelopeContext(
  config: MemoryPluginConfig,
  envelope: Pick<MemxTurnEnvelope, "actorId" | "sessionId" | "hostId" | "workspaceDir" | "project" | "runId">,
): MemoryOperationContext {
  const ctx = buildOperationContext(config, {
    agentId: hostScopedAgentId(envelope),
    sessionKey: hostSessionKey(envelope),
    workspaceDir: envelope.workspaceDir,
    project: envelope.project,
    runId: envelope.runId,
  });
  if (!ctx) {
    throw new Error("unable to build memX operation context");
  }
  return ctx;
}

function countTable(store: MemxStoreBundle, table: string): number {
  const row = store.client.prepare(`SELECT COUNT(*) AS count FROM ${table}`).get() as
    | { count: number }
    | undefined;
  return Number(row?.count ?? 0);
}

function formatEvidenceRows(
  title: string,
  rows: Array<{ text?: string; observedAt?: string }>,
  limit: number,
): string[] {
  const usableRows = rows.filter((row) => typeof row.text === "string" && row.text.trim().length > 0);
  if (usableRows.length === 0) {
    return [];
  }
  return [
    `## ${title}`,
    ...usableRows.slice(0, limit).map((row) => {
      const date = row.observedAt ? ` [${row.observedAt.slice(0, 10)}]` : "";
      return `- ${truncateText(row.text ?? "", 360)}${date}`;
    }),
  ];
}

function graphPathText(path: unknown): string {
  if (typeof path === "string") {
    return path;
  }
  if (isRecord(path) && typeof path.summary === "string") {
    return path.summary;
  }
  return "";
}

function formatRecallContext(bundle: EvidenceBundle, limit: number): string {
  const graphPaths = Array.isArray(bundle.graph?.paths) ? bundle.graph.paths : [];
  const evidenceLines = [
    ...formatEvidenceRows("Guidance", bundle.behavioralGuidance.map((text) => ({ text })), Math.min(limit, 4)),
    ...formatEvidenceRows("State", bundle.states, limit),
    ...formatEvidenceRows("Facts", bundle.facts, limit),
    ...formatEvidenceRows("Events", bundle.events, limit),
    ...formatEvidenceRows(
      "Graph",
      graphPaths.map((path) => ({ text: graphPathText(path) })),
      Math.min(limit, 4),
    ),
  ].filter((line) => line.trim().length > 0);
  if (evidenceLines.length === 0) {
    return "";
  }
  return [
    "## memX Memory",
    "Use the following remembered context only when it directly helps the current request.",
    ...evidenceLines,
  ].join("\n");
}

export function formatNativeRecallContext(bundle: EvidenceBundle, maxChars: number): string {
  return renderEvidenceBundle(bundle, maxChars);
}

type NativeContextEligibility = {
  eligible: boolean;
  reason: string;
  bestScore: number;
};

function injectedPackets(bundle: EvidenceBundle): EvidencePacket[] {
  return bundle.evidencePackets.filter((packet) => packet.injected && !packet.dropReason);
}

function bestInjectedPacketScore(packets: EvidencePacket[]): number {
  return packets.reduce(
    (best, packet) =>
      Math.max(best, packet.grade?.finalScore ?? packet.score ?? packet.coverage.confidence ?? 0),
    0,
  );
}

function packetHasNativeAnswerSurface(packet: EvidencePacket): boolean {
  if (packet.role === "answer" || packet.eligibility?.role === "answer") {
    return true;
  }
  const lines = packet.displayLines && packet.displayLines.length > 0 ? packet.displayLines : [packet.primaryText];
  return lines.some(
    (line) =>
      line.startsWith("[answer]") ||
      line.startsWith("[resource]") ||
      (packet.operationType === "aggregate" && line.startsWith("[event]")),
  );
}

function packetIsStrongNativeContextEvidence(packet: EvidencePacket): boolean {
  const score = packet.grade?.finalScore ?? packet.score ?? packet.coverage.confidence ?? 0;
  const slotCoverage =
    packet.grade?.slotCoverageScore ?? (packet.coverage.filled ? packet.coverage.confidence : 0);
  const contextBinding =
    packet.grade?.contextBindingScore ?? (packet.coverage.filled ? packet.coverage.confidence : 0);
  return (
    packetHasSourceGroundedEvidence(packet) &&
    packetHasNativeAnswerSurface(packet) &&
    packet.coverage.missing.length === 0 &&
    score >= 0.62 &&
    slotCoverage >= 0.45 &&
    contextBinding >= 0.42
  );
}

function appendStagedPendingEvidence(
  bundle: EvidenceBundle,
  stagedTurns: Array<{ turnId: string; observedAt: string; text: string }>,
  ctx: Pick<MemoryOperationContext, "agentId" | "scopes">,
): EvidenceBundle {
  if (stagedTurns.length === 0) {
    return bundle;
  }
  const stagedRows: EvidenceBundle["events"] = stagedTurns.map((turn) => ({
    id: `pending-staged:${turn.turnId}`,
    text: turn.text,
    score: 0.62,
    scope: ctx.scopes[0] ?? `agent:${ctx.agentId}`,
    confidence: 0.62,
    observedAt: turn.observedAt,
    sourceRef: `pending-staged:${turn.turnId}`,
    lineage: {
      sourceKind: "chunk",
      sourceId: turn.turnId,
      sourceRef: `pending-staged:${turn.turnId}`,
    },
  }));
  const stagedPackets: EvidencePacket[] = stagedRows.map((row) => ({
    packetId: row.id,
    slotId: "pending-staged-turn",
    operationType: "return_value",
    role: "answer",
    protected: true,
    injected: true,
    layers: ["chunk"],
    primaryText: row.text,
    supportingTexts: [],
    sourceRefs: [row.sourceRef],
    allSourceRefs: [row.sourceRef],
    score: row.score,
    scoreBreakdown: {
      stagedPendingTurn: true,
      retrievalScore: row.score,
    },
    displayLines: [`[answer] ${truncateText(row.text, 360)}`],
    observedAt: row.observedAt,
    authorRoles: ["user", "assistant"],
    coverage: {
      filled: true,
      missing: [],
      confidence: row.confidence,
    },
    eligibility: {
      eligible: true,
      role: "answer",
      blockers: [],
    },
    grade: {
      retrievalScore: row.score,
      answerScore: row.score,
      contextBindingScore: row.score,
      slotCoverageScore: row.score,
      authorityScore: 0.72,
      finalScore: row.score,
    },
    selectionReason: "pending staged turn evidence while semantic write is still queued",
  }));
  return {
    ...bundle,
    events: [...stagedRows, ...bundle.events],
    evidencePackets: [...stagedPackets, ...bundle.evidencePackets],
    recalledChunkTexts: [...stagedRows.map((row) => row.text), ...bundle.recalledChunkTexts],
    diagnostics: [...bundle.diagnostics, "pending-staged-turn-evidence"],
  };
}

function packetHasSourceGroundedEvidence(packet: EvidencePacket): boolean {
  const hasSource =
    packet.sourceRefs.length > 0 ||
    (packet.allSourceRefs?.length ?? 0) > 0 ||
    (packet.answerUnits ?? []).some((unit) => unit.sourceRefs.length > 0) ||
    (packet.contextUnits ?? []).some((unit) => unit.sourceRefs.length > 0) ||
    (packet.supportUnits ?? []).some((unit) => unit.sourceRefs.length > 0);
  const hasRenderableEvidence =
    packet.primaryText.trim().length > 0 ||
    packet.supportingTexts.some((text) => text.trim().length > 0) ||
    (packet.displayLines ?? []).some((line) => line.trim().length > 0);
  return hasSource && hasRenderableEvidence;
}

function packetTextForSuppression(packet: EvidencePacket): string {
  return [
    packet.primaryText,
    ...packet.supportingTexts,
    ...(packet.displayLines ?? []),
    ...(packet.entityAliases ?? []),
    packet.answerCandidate?.text,
    ...(packet.contextCandidates ?? []).map((candidate) => candidate.text),
  ]
    .filter((value): value is string => typeof value === "string" && value.trim().length > 0)
    .join("\n");
}

function packetMentionsSuppressedEntity(packet: EvidencePacket, queryAnalysis: QueryCompileResult): boolean {
  const suppressed = queryAnalysis.suppressedEntities ?? [];
  if (suppressed.length === 0) {
    return false;
  }
  const normalizedPacketText = normalizeName(packetTextForSuppression(packet));
  if (!normalizedPacketText) {
    return false;
  }
  return suppressed.some((entity) => {
    const normalizedEntity = normalizeName(entity.name);
    return normalizedEntity.length >= 2 && normalizedPacketText.includes(normalizedEntity);
  });
}

function entityFocusTerms(queryAnalysis: Pick<QueryCompileResult, "queryEntities">): string[] {
  const terms = new Set<string>();
  for (const entity of queryAnalysis.queryEntities ?? []) {
    const normalized = normalizeName(entity.name);
    if (normalized.length >= 2) {
      terms.add(normalized);
    }
  }
  return [...terms];
}

function textMentionsFocusEntity(text: string | undefined, terms: string[]): boolean {
  if (!text || terms.length === 0) {
    return false;
  }
  const normalized = normalizeName(text);
  return terms.some((term) => normalized.includes(term));
}

function packetMentionsFocusEntity(packet: EvidencePacket, terms: string[]): boolean {
  return textMentionsFocusEntity(packetTextForSuppression(packet), terms);
}

function focusEvidenceRows<T extends { text?: string }>(rows: T[], terms: string[]): T[] {
  return rows.filter((row) => textMentionsFocusEntity(row.text, terms));
}

function hasFocusedEvidence(bundle: EvidenceBundle): boolean {
  return (
    bundle.states.length > 0 ||
    bundle.tasks.length > 0 ||
    bundle.facts.length > 0 ||
    bundle.events.length > 0 ||
    bundle.graph.paths.length > 0 ||
    bundle.behavioralGuidance.length > 0 ||
    bundle.promptEvidence.length > 0 ||
    bundle.evidencePackets.length > 0
  );
}

export function focusRecallBundleForQueryEntities(
  queryAnalysis: Pick<QueryCompileResult, "queryEntities">,
  bundle: EvidenceBundle,
): EvidenceBundle {
  const terms = entityFocusTerms(queryAnalysis);
  if (terms.length === 0) {
    return bundle;
  }
  const focusedGraphNodes = bundle.graph.nodes.filter((node) =>
    textMentionsFocusEntity(`${node.name} ${node.type}`, terms),
  );
  const focusedNodeIds = new Set(focusedGraphNodes.map((node) => node.nodeId));
  const focused: EvidenceBundle = {
    ...bundle,
    states: focusEvidenceRows(bundle.states, terms),
    tasks: focusEvidenceRows(bundle.tasks, terms),
    facts: focusEvidenceRows(bundle.facts, terms),
    events: focusEvidenceRows(bundle.events, terms),
    alternates: focusEvidenceRows(bundle.alternates, terms),
    graph: {
      ...bundle.graph,
      nodes: focusedGraphNodes,
      edges:
        focusedNodeIds.size > 0
          ? bundle.graph.edges.filter(
              (edge) => focusedNodeIds.has(edge.srcNodeId) || focusedNodeIds.has(edge.dstNodeId),
            )
          : [],
      pathCandidates: bundle.graph.pathCandidates.filter((candidate) =>
        textMentionsFocusEntity(JSON.stringify(candidate), terms),
      ),
      paths: bundle.graph.paths.filter((path) => textMentionsFocusEntity(graphPathText(path), terms)),
    },
    behavioralGuidance: bundle.behavioralGuidance.filter((text) =>
      textMentionsFocusEntity(text, terms),
    ),
    recalledChunkTexts: bundle.recalledChunkTexts.filter((text) =>
      textMentionsFocusEntity(text, terms),
    ),
    promptEvidence: bundle.promptEvidence.filter((candidate) =>
      textMentionsFocusEntity(
        [candidate.text, candidate.rawText, candidate.scoringText].filter(Boolean).join("\n"),
        terms,
      ),
    ),
    evidencePackets: bundle.evidencePackets.filter((packet) =>
      packetMentionsFocusEntity(packet, terms),
    ),
  };
  return hasFocusedEvidence(focused)
    ? focused
    : {
        ...focused,
        diagnostics: [...bundle.diagnostics, "target-entity-no-focused-evidence"],
      };
}

export function assessNativeContextEligibility(
  _query: string,
  queryAnalysis: QueryCompileResult,
  bundle: EvidenceBundle,
): NativeContextEligibility {
  const packets = injectedPackets(bundle);
  if (packets.length === 0) {
    return {
      eligible: false,
      reason: "no-injected-packets",
      bestScore: 0,
    };
  }
  const bestScore = bestInjectedPacketScore(packets);
  if (packets.some((packet) => packetMentionsSuppressedEntity(packet, queryAnalysis))) {
    return { eligible: false, reason: "suppressed-entity-anchor", bestScore };
  }
  const focusTerms = entityFocusTerms(queryAnalysis);
  if (focusTerms.length > 0 && !packets.some((packet) => packetMentionsFocusEntity(packet, focusTerms))) {
    return { eligible: false, reason: "target-entity-mismatch", bestScore };
  }
  const enoughEvidence = packets.some(packetIsStrongNativeContextEvidence);
  if (enoughEvidence) {
    return {
      eligible: true,
      reason: queryAnalysis.queryEntities.length > 0 ? "llm-query-entities" : "strong-evidence",
      bestScore,
    };
  }
  return { eligible: false, reason: "weak-evidence", bestScore };
}

export class MemxHostService {
  private readonly config: MemoryPluginConfig;
  private readonly logger: MemxLogger;
  private readonly manager: MemxRuntimeManager;
  private readonly pendingWrites = new Map<string, Promise<void>>();

  constructor(options: MemxServiceOptions = {}) {
    this.config = options.config ?? createServiceConfigFromEnv();
    this.logger = loggerOrConsole(options.logger);
    this.manager = new MemxRuntimeManager(this.logger);
  }

  async close(): Promise<void> {
    await Promise.allSettled([...this.pendingWrites.values()]);
    await this.manager.closeAll();
  }

  private pendingWriteKey(ctx: Pick<MemoryOperationContext, "agentId" | "sessionKey">): string {
    return `${ctx.agentId}\u0000${ctx.sessionKey ?? "default"}`;
  }

  private hasPendingWrite(ctx: Pick<MemoryOperationContext, "agentId" | "sessionKey">): boolean {
    return this.pendingWrites.has(this.pendingWriteKey(ctx));
  }

  private enqueuePendingWrite(ctx: MemoryOperationContext, work: () => Promise<void>): void {
    const key = this.pendingWriteKey(ctx);
    const previous = this.pendingWrites.get(key) ?? Promise.resolve();
    const tracked = previous
      .catch(() => {})
      .then(work)
      .catch((error) => {
        this.logger.warn(`memx: host observe flush failed (${String(error)})`);
      });
    this.pendingWrites.set(key, tracked);
    void tracked.finally(() => {
      if (this.pendingWrites.get(key) === tracked) {
        this.pendingWrites.delete(key);
      }
    });
  }

  private async waitForPendingWrites(
    ctx: Pick<MemoryOperationContext, "agentId" | "sessionKey">,
    hotPathTimeoutMs?: number,
  ): Promise<number> {
    const pending = this.pendingWrites.get(this.pendingWriteKey(ctx));
    if (!pending) {
      return 0;
    }
    const startedAt = performance.now();
    const configuredBudget = Number.isFinite(hotPathTimeoutMs ?? Number.NaN)
      ? Math.max(0, Number(hotPathTimeoutMs))
      : 0;
    const timeoutMs =
      configuredBudget > 0 ? Math.max(0, Math.min(5000, configuredBudget - 1500)) : 2500;
    if (timeoutMs <= 0) {
      return 0;
    }
    let timeout: NodeJS.Timeout | undefined;
    await Promise.race([
      pending,
      new Promise<void>((resolve) => {
        timeout = setTimeout(resolve, timeoutMs);
      }),
    ]);
    if (timeout) {
      clearTimeout(timeout);
    }
    return Math.round(performance.now() - startedAt);
  }

  async observe(input: unknown): Promise<Record<string, unknown>> {
    const envelope = normalizeObservePayload(input);
    const ctx = asEnvelopeContext(this.config, envelope);
    const store = await this.manager.getStore(ctx);
    const scope = resolveDefaultScope(this.config, {
      agentId: ctx.agentId,
      sessionKey: ctx.sessionKey,
      project: ctx.project,
      workspace: ctx.workspaceDir,
    });
    const turnId = randomId("turn");
    const captured = captureAgentEndTurn({
      agentId: ctx.agentId,
      scope,
      sessionKey: ctx.sessionKey ?? "default",
      turnId,
      observedAt: envelope.observedAt || nowIso(),
      messages: envelope.messages,
    });
    if (captured.length === 0) {
      return { ok: true, accepted: false, reason: "no-capturable-messages" };
    }
    try {
      const staged = await store.turnScheduler.stageRecallableTurn(ctx, captured);
      if (staged) {
        this.manager.rememberStagedRecallableTurn(ctx, captured);
      }
    } catch (error) {
      this.logger.warn?.(`memx: fast turn staging failed (${String(error)})`);
    }
    this.enqueuePendingWrite(ctx, async () => {
      await store.turnScheduler.enqueue(ctx, captured);
      await this.manager.recordMaintenanceTurn(ctx, {
          store,
          turnId,
          observedAt: captured.at(-1)?.observedAt ?? ctx.now,
      });
    });
    return {
      ok: true,
      accepted: true,
      hostId: envelope.hostId,
      actorId: ctx.agentId,
      sessionKey: ctx.sessionKey,
      turnId,
      captured: captured.length,
    };
  }

  async recall(request: MemxRecallRequest): Promise<Record<string, unknown>> {
    if (!request.query?.trim()) {
      throw new Error("query required");
    }
    const envelope: MemxTurnEnvelope = {
      hostId: request.hostId === "codex" || request.hostId === "claude-code" ? request.hostId : "generic",
      actorId: request.actorId || process.env["MEMX_ACTOR_ID"] || "memx-shared",
      sessionId: request.sessionId || "mcp",
      workspaceDir: request.workspaceDir || process.cwd(),
      project: request.project,
      eventName: "recall",
      observedAt: nowIso(),
      messages: [{ role: "user", content: request.query }],
    };
    const ctx = asEnvelopeContext(this.config, envelope);
    const waitElapsedMs = await this.waitForPendingWrites(ctx, request.hotPathTimeoutMs);
    const store = await this.manager.getStore({
      ...ctx,
      readEpoch: 0,
    });
    const recallCtx = {
      ...ctx,
      readEpoch: store.client.currentMemoryEpoch(ctx.agentId),
    };
    const remainingHotPathTimeoutMs =
      typeof request.hotPathTimeoutMs === "number" && Number.isFinite(request.hotPathTimeoutMs)
        ? Math.max(500, request.hotPathTimeoutMs - waitElapsedMs)
        : request.hotPathTimeoutMs;
    const compiled = await compileQuery({
      query: request.query,
      ctx: recallCtx,
      reasoner: store.reasoner,
      hotPathTimeoutMs: remainingHotPathTimeoutMs,
    });
    const bundle = await retrieveEvidence(store, recallCtx, request.query, compiled.focusedQuery, {
      queryAnalysis: compiled,
    });
    const focusedBundle = appendStagedPendingEvidence(
      focusRecallBundleForQueryEntities(compiled, bundle),
      this.hasPendingWrite(ctx) ? this.manager.recentStagedRecallableTurns(ctx, 4) : [],
      ctx,
    );
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
        edges: graphEdges.slice(0, Math.min(limit, 12)),
      },
      diagnostics: focusedBundle.diagnostics,
    };
  }

  async remember(request: Record<string, unknown>): Promise<Record<string, unknown>> {
    const content = typeof request.content === "string" ? request.content.trim() : "";
    if (!content) {
      throw new Error("content required");
    }
    return this.observe({
      hostId: request.hostId ?? "generic",
      actorId: request.actorId ?? process.env["MEMX_ACTOR_ID"] ?? "memx-shared",
      sessionId: request.sessionId ?? "manual",
      workspaceDir: request.workspaceDir ?? process.cwd(),
      eventName: "remember",
      observedAt: nowIso(),
      messages: [{ role: "user", content }],
      metadata: { manual: true, memoryType: request.type },
    });
  }

  async forget(request: Record<string, unknown>): Promise<Record<string, unknown>> {
    const id = typeof request.id === "string" ? request.id.trim() : "";
    if (!id) {
      throw new Error("id required");
    }
    const ctx = asEnvelopeContext(this.config, {
      hostId: "generic",
      actorId: typeof request.actorId === "string" ? request.actorId : "memx-shared",
      sessionId: typeof request.sessionId === "string" ? request.sessionId : "manual",
    });
    const store = await this.manager.getStore(ctx);
    const kind = typeof request.kind === "string" ? request.kind : "doc";
    let deleted = 0;
    if (kind === "event") {
      deleted = store.eventRepo.delete({ agentId: ctx.agentId, eventId: id });
      store.vectorRepo.deleteDocs([`event:${id}`]);
    } else if (kind === "fact") {
      deleted = store.factRepo.markDeleted({ agentId: ctx.agentId, factId: id });
      store.vectorRepo.deleteDocs([`fact:${id}`]);
    } else if (kind === "state") {
      deleted = store.stateRepo.delete({ agentId: ctx.agentId, key: id });
      store.vectorRepo.deleteDocs([`state:${id}`]);
    } else {
      store.vectorRepo.deleteDocs([id]);
      deleted = 1;
    }
    return { ok: true, deleted, kind, id };
  }

  async stats(request: MemxAgentRequest = {}): Promise<Record<string, unknown>> {
    const ctx = asEnvelopeContext(this.config, {
      hostId: request.hostId === "codex" || request.hostId === "claude-code" ? request.hostId : "generic",
      actorId: request.actorId || process.env["MEMX_ACTOR_ID"] || "memx-shared",
      sessionId: request.sessionId || "stats",
      workspaceDir: request.workspaceDir,
      project: request.project,
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
      vectorDocCount: countTable(store, "vector_docs"),
    };
  }

  async audit(limit = 50, request: MemxAgentRequest = {}): Promise<Record<string, unknown>> {
    const ctx = asEnvelopeContext(this.config, {
      hostId: request.hostId === "codex" || request.hostId === "claude-code" ? request.hostId : "generic",
      actorId: request.actorId || process.env["MEMX_ACTOR_ID"] || "memx-shared",
      sessionId: request.sessionId || "audit",
      workspaceDir: request.workspaceDir,
      project: request.project,
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
        limit: boundedLimit,
      }),
      retrievals: store.auditRepo.listRetrievals({
        agentId: ctx.agentId,
        limit: boundedLimit,
      }),
      policyDecisions: store.auditRepo.listPolicyDecisions({
        agentId: ctx.agentId,
        limit: boundedLimit,
      }),
      maintenanceRuns: store.auditRepo.listMaintenanceRuns({
        agentId: ctx.agentId,
        limit: boundedLimit,
      }),
    };
  }

  async context(request: MemxRecallRequest): Promise<Record<string, unknown>> {
    const recalled = await this.recall(request);
    const eligibility = recalled.contextEligibility as NativeContextEligibility | undefined;
    const candidateContext = typeof recalled.context === "string" ? recalled.context : "";
    const prependContext = eligibility?.eligible === false ? "" : candidateContext;
    if (eligibility && !eligibility.eligible) {
      this.logger.info?.(
        `memx: native context withheld reason=${eligibility.reason} best=${eligibility.bestScore.toFixed(2)} query="${request.query.slice(0, 80)}"`,
      );
    }
    try {
      const envelope: MemxTurnEnvelope = {
        hostId: request.hostId === "codex" || request.hostId === "claude-code" ? request.hostId : "generic",
        actorId: request.actorId || process.env["MEMX_ACTOR_ID"] || "memx-shared",
        sessionId: request.sessionId || "mcp",
        workspaceDir: request.workspaceDir || process.cwd(),
        project: request.project,
        eventName: "context",
        observedAt: nowIso(),
        messages: [{ role: "user", content: request.query }],
      };
      const ctx = asEnvelopeContext(this.config, envelope);
      const store = await this.manager.getStore(ctx);
      store.auditRepo.annotateLatestRetrievalInjection({
        agentId: ctx.agentId,
        queryText: request.query,
        candidateChars: candidateContext.length,
        actualInjectedChars: prependContext.length,
        eligible: eligibility?.eligible ?? true,
        reason: eligibility?.reason,
        finalizedAt: ctx.now,
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
        sourceRef: `native_context:${stableHash([request.query, ctx.sessionKey, ctx.now])}`,
        metadataJson: {
          eligible: eligibility?.eligible ?? true,
          reason: eligibility?.reason,
          bestScore: eligibility?.bestScore,
          candidateChars: candidateContext.length,
          actualInjectedChars: prependContext.length,
        },
        createdAt: ctx.now,
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
        actualInjectedChars: prependContext.length,
      },
      recall: recalled,
    };
  }
}

export function stableHostTurnId(envelope: MemxTurnEnvelope): string {
  return stableHash([envelope.hostId, envelope.actorId, envelope.sessionId, envelope.observedAt]);
}
