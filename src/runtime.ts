import { MemxDbClient } from "./db/client.js";
import { AbstractionRepo } from "./db/repositories/abstractionRepo.js";
import { AuditRepo } from "./db/repositories/auditRepo.js";
import { BeliefRepo } from "./db/repositories/beliefRepo.js";
import { ChunkRepo } from "./db/repositories/chunkRepo.js";
import { EventRepo } from "./db/repositories/eventRepo.js";
import { FactRepo } from "./db/repositories/factRepo.js";
import { GraphRepo } from "./db/repositories/graphRepo.js";
import { MaintenanceRepo } from "./db/repositories/maintenanceRepo.js";
import { SourceSegmentRepo } from "./db/repositories/sourceSegmentRepo.js";
import { StateRepo } from "./db/repositories/stateRepo.js";
import { StrategyRepo } from "./db/repositories/strategyRepo.js";
import { TaskRepo } from "./db/repositories/taskRepo.js";
import { VectorRepo } from "./db/repositories/vectorRepo.js";
import { createMemoryLlmBudgetAudit } from "./pipeline/llmBudgetAudit.js";
import { runAutomaticMaintenanceBatch } from "./pipeline/maintenanceBatch.js";
import { MemxReasoner } from "./pipeline/reasoner.js";
import { MemxTurnScheduler } from "./pipeline/turnScheduler.js";
import { OptionalEmbeddingBackend } from "./search/backends/embeddingBackend.js";
import { defaultRetrievalScopes, renderTemplate, scopeVarsForContext } from "./security/scopes.js";
import { nowIso, randomId, resolveUserPath } from "./support.js";
import type {
  MaintenanceBatchTriggerReason,
  MemoryOperationContext,
  MemoryPluginConfig,
  MemxLogger,
  PluginActorContext,
  TurnCaptureMessage,
} from "./types.js";

export type MemxStoreBundle = {
  client: MemxDbClient;
  stateRepo: StateRepo;
  taskRepo: TaskRepo;
  chunkRepo: ChunkRepo;
  factRepo: FactRepo;
  eventRepo: EventRepo;
  graphRepo: GraphRepo;
  sourceSegmentRepo: SourceSegmentRepo;
  vectorRepo: VectorRepo;
  auditRepo: AuditRepo;
  maintenanceRepo: MaintenanceRepo;
  beliefRepo: BeliefRepo;
  abstractionRepo: AbstractionRepo;
  strategyRepo: StrategyRepo;
  retrievalBackend: OptionalEmbeddingBackend;
  reasoner: MemxReasoner;
  turnScheduler: MemxTurnScheduler;
};

function storeKey(agentId: string, dbPath: string): string {
  return `${agentId}:${dbPath}`;
}

function maintenanceKey(agentId: string, dbPath: string, sessionKey: string): string {
  return `${agentId}:${dbPath}:${sessionKey}`;
}

function uniqueStrings(values: string[]): string[] {
  return [...new Set(values.filter((value) => value.trim().length > 0))];
}

type MaintenanceContextTemplate = MemoryOperationContext;
type StagedRecallableTurn = {
  turnId: string;
  observedAt: string;
  text: string;
};

function stagedRecallKey(
  ctx: Pick<MemoryOperationContext, "agentId" | "dbPath" | "sessionKey" | "workspaceDir">,
): string {
  const workspace = ctx.workspaceDir?.trim();
  const scope = workspace ? `workspace:${workspace}` : `session:${ctx.sessionKey ?? "default"}`;
  return `${ctx.agentId}:${ctx.dbPath}:${scope}`;
}

export function resolveDbPath(
  config: MemoryPluginConfig,
  actor: PluginActorContext,
): string | null {
  if (!actor.agentId) {
    return null;
  }
  const rendered = renderTemplate(config.dbPath, scopeVarsForContext(actor));
  return resolveUserPath(rendered);
}

export function buildOperationContext(
  config: MemoryPluginConfig,
  actor: PluginActorContext,
  overrides?: { now?: string },
): MemoryOperationContext | null {
  if (!actor.agentId) {
    return null;
  }
  const dbPath = resolveDbPath(config, actor);
  if (!dbPath) {
    return null;
  }
  return {
    agentId: actor.agentId,
    sessionKey: actor.sessionKey,
    workspaceDir: actor.workspaceDir,
    project: actor.project,
    runId: actor.runId,
    channelId: actor.channelId,
    config,
    dbPath,
    scopes: defaultRetrievalScopes(config, scopeVarsForContext(actor)),
    now: overrides?.now ?? nowIso(),
    llmBudgetAudit: createMemoryLlmBudgetAudit(),
  };
}

export class MemxRuntimeManager {
  private readonly stores = new Map<string, Promise<MemxStoreBundle>>();
  private readonly storeContexts = new Map<string, MaintenanceContextTemplate>();
  private readonly sessionCursors = new Map<string, number>();
  private readonly lastRecall = new Map<string, { chunkIds: string[]; texts: string[] }>();
  private readonly stagedRecallableTurns = new Map<string, StagedRecallableTurn[]>();
  private readonly maintenanceTimers = new Map<string, ReturnType<typeof setTimeout>>();
  private readonly maintenanceContexts = new Map<string, MaintenanceContextTemplate>();
  private readonly maintenanceLoops = new Map<string, Promise<void>>();
  private readonly maintenanceLeaseOwner = randomId("maintenance-runtime");

  constructor(private readonly logger: MemxLogger) {}

  async getStore(ctx: MemoryOperationContext): Promise<MemxStoreBundle> {
    const key = storeKey(ctx.agentId, ctx.dbPath);
    this.storeContexts.set(key, { ...ctx, sessionKey: ctx.sessionKey ?? "default" });
    const existing = this.stores.get(key);
    if (existing) {
      return existing;
    }
    const created = this.createStore(ctx);
    this.stores.set(key, created);
    // Evict the cached promise if creation fails so a subsequent call can retry.
    created.catch(() => {
      this.stores.delete(key);
    });
    return created;
  }

  async closeAll(): Promise<void> {
    await this.flushAll();
    await this.flushPendingMaintenance("shutdown");
    await this.flushRetryableSemanticWriteJobs("shutdown");
    for (const timer of this.maintenanceTimers.values()) {
      clearTimeout(timer);
    }
    this.maintenanceTimers.clear();
    for (const entry of this.stores.values()) {
      const store = await entry;
      try {
        await store.retrievalBackend.close?.();
      } catch (error) {
        this.logger.warn(`memx: retrieval backend close failed (${String(error)})`);
      } finally {
        store.client.close();
      }
    }
    this.stores.clear();
    this.storeContexts.clear();
    this.sessionCursors.clear();
    this.lastRecall.clear();
    this.stagedRecallableTurns.clear();
    this.maintenanceContexts.clear();
    this.maintenanceLoops.clear();
  }

  async flushAll(): Promise<void> {
    for (const entry of this.stores.values()) {
      const store = await entry;
      await store.turnScheduler.flush();
      await store.retrievalBackend.flushPendingUpserts?.();
    }
  }

  getSessionCursor(agentId: string, sessionKey: string): number | undefined {
    return this.sessionCursors.get(`${agentId}:${sessionKey}`);
  }

  setSessionCursor(agentId: string, sessionKey: string, cursor: number): void {
    this.sessionCursors.set(`${agentId}:${sessionKey}`, cursor);
  }

  rememberRecall(
    agentId: string,
    sessionKey: string | undefined,
    payload: { chunkIds: string[]; texts: string[] },
  ): void {
    if (!sessionKey) {
      return;
    }
    this.lastRecall.set(`${agentId}:${sessionKey}`, payload);
  }

  consumeRecall(
    agentId: string,
    sessionKey: string | undefined,
  ): { chunkIds: string[]; texts: string[] } {
    if (!sessionKey) {
      return { chunkIds: [], texts: [] };
    }
    const key = `${agentId}:${sessionKey}`;
    const payload = this.lastRecall.get(key) ?? { chunkIds: [], texts: [] };
    this.lastRecall.delete(key);
    return payload;
  }

  rememberStagedRecallableTurn(
    ctx: Pick<MemoryOperationContext, "agentId" | "dbPath" | "sessionKey" | "workspaceDir">,
    messages: TurnCaptureMessage[],
  ): void {
    const text = messages
      .map((message) => `[${message.role}] ${message.content.trim()}`)
      .filter((entry) => entry.trim().length > 0)
      .join("\n");
    if (!text.trim()) {
      return;
    }
    const key = stagedRecallKey(ctx);
    const turnId = messages[0]?.turnId ?? randomId("staged-turn");
    const observedAt = messages.at(-1)?.observedAt ?? nowIso();
    const existing = (this.stagedRecallableTurns.get(key) ?? []).filter(
      (entry) => entry.turnId !== turnId,
    );
    existing.push({ turnId, observedAt, text });
    this.stagedRecallableTurns.set(key, existing.slice(-6));
  }

  recentStagedRecallableTurns(
    ctx: Pick<MemoryOperationContext, "agentId" | "dbPath" | "sessionKey" | "workspaceDir">,
    limit = 4,
  ): StagedRecallableTurn[] {
    const key = stagedRecallKey(ctx);
    return [...(this.stagedRecallableTurns.get(key) ?? [])]
      .sort((left, right) => right.observedAt.localeCompare(left.observedAt))
      .slice(0, Math.max(1, Math.min(Math.trunc(limit), 8)));
  }

  async recordMaintenanceTurn(
    ctx: MemoryOperationContext,
    params: {
      turnId: string;
      observedAt: string;
      store?: MemxStoreBundle;
    },
  ): Promise<void> {
    if (!ctx.config.advanced.enableMaintenanceJobs) {
      return;
    }
    const sessionKey = ctx.sessionKey ?? "default";
    const store = params.store ?? (await this.getStore(ctx));
    const key = maintenanceKey(ctx.agentId, ctx.dbPath, sessionKey);
    const template = { ...ctx, sessionKey };
    this.maintenanceContexts.set(key, template);
    const state = store.maintenanceRepo.recordPendingTurn({
      agentId: ctx.agentId,
      sessionKey,
      turnId: params.turnId,
      observedAt: params.observedAt,
      updatedAt: ctx.now,
    });
    const threshold =
      ctx.config.advanced.maintenanceTriggerMode === "per_turn"
        ? 1
        : Math.max(1, ctx.config.advanced.maintenanceBatchTurns);
    if (state.pendingTurnCount >= threshold) {
      this.clearMaintenanceTimer(key);
      this.startMaintenanceLoop(store, template, "threshold");
      return;
    }
    if (
      ctx.config.advanced.maintenanceTriggerMode === "batched" &&
      ctx.config.advanced.maintenanceIdleFlushMinutes > 0
    ) {
      this.scheduleIdleFlush(store, template);
    }
  }

  private startMaintenanceLoop(
    store: MemxStoreBundle,
    template: MaintenanceContextTemplate,
    reason: MaintenanceBatchTriggerReason,
  ): void {
    const sessionKey = template.sessionKey ?? "default";
    const key = maintenanceKey(template.agentId, template.dbPath, sessionKey);
    const existing = this.maintenanceLoops.get(key);
    if (existing) {
      return;
    }
    const loop = this.runMaintenanceLoop(store, template, reason).finally(() => {
      this.maintenanceLoops.delete(key);
    });
    this.maintenanceLoops.set(key, loop);
    loop.catch((error) => {
      this.logger.warn?.(`memx: maintenance batch loop failed (${String(error)})`);
    });
  }

  private scheduleIdleFlush(store: MemxStoreBundle, template: MaintenanceContextTemplate): void {
    const sessionKey = template.sessionKey ?? "default";
    const key = maintenanceKey(template.agentId, template.dbPath, sessionKey);
    this.clearMaintenanceTimer(key);
    const delayMs = Math.max(0.001, template.config.advanced.maintenanceIdleFlushMinutes) * 60_000;
    const timer = setTimeout(() => {
      this.maintenanceTimers.delete(key);
      this.startMaintenanceLoop(store, template, "idle");
    }, delayMs);
    this.maintenanceTimers.set(key, timer);
  }

  private clearMaintenanceTimer(key: string): void {
    const existing = this.maintenanceTimers.get(key);
    if (!existing) {
      return;
    }
    clearTimeout(existing);
    this.maintenanceTimers.delete(key);
  }

  private buildMaintenanceContext(template: MaintenanceContextTemplate): MemoryOperationContext {
    return {
      ...template,
      now: nowIso(),
      llmBudgetAudit: createMemoryLlmBudgetAudit(),
    };
  }

  private async runMaintenanceLoop(
    store: MemxStoreBundle,
    template: MaintenanceContextTemplate,
    initialReason: MaintenanceBatchTriggerReason,
  ): Promise<void> {
    const sessionKey = template.sessionKey ?? "default";
    const key = maintenanceKey(template.agentId, template.dbPath, sessionKey);
    let reason: MaintenanceBatchTriggerReason = initialReason;
    const threshold =
      template.config.advanced.maintenanceTriggerMode === "per_turn"
        ? 1
        : Math.max(1, template.config.advanced.maintenanceBatchTurns);

    while (true) {
      const state = store.maintenanceRepo.getState(template.agentId, sessionKey);
      const pendingTurnCount = state?.pendingTurnCount ?? 0;
      const shouldClaim =
        reason === "shutdown"
          ? pendingTurnCount > 0
          : reason === "idle"
            ? pendingTurnCount > 0
            : pendingTurnCount >= threshold;
      if (!shouldClaim) {
        if (
          pendingTurnCount > 0 &&
          template.config.advanced.maintenanceTriggerMode === "batched" &&
          template.config.advanced.maintenanceIdleFlushMinutes > 0
        ) {
          this.scheduleIdleFlush(store, template);
        }
        return;
      }

      const claim = store.maintenanceRepo.claimBatch({
        agentId: template.agentId,
        sessionKey,
        reason,
        leaseOwner: this.maintenanceLeaseOwner,
        leaseTtlMs: 5 * 60_000,
        now: nowIso(),
      });
      if (!claim) {
        return;
      }

      this.clearMaintenanceTimer(key);
      const upperWatermarks = {
        event: store.eventRepo.latestObservedAt({
          agentId: template.agentId,
          scopes: template.scopes,
          sessionKey,
        }),
        signal: store.auditRepo.latestSignalCreatedAt({
          agentId: template.agentId,
          sessionKey,
        }),
        task: store.taskRepo.latestUpdatedAt({
          agentId: template.agentId,
          scopes: template.scopes,
          sessionKey,
        }),
      };

      let success = false;
      try {
        await runAutomaticMaintenanceBatch(store, this.buildMaintenanceContext(template), {
          sessionKey,
          turnIds: claim.turnIds,
          turnCount: claim.turnCount,
          reason: claim.reason,
          firstObservedAt: claim.firstObservedAt,
          lastObservedAt: claim.lastObservedAt,
          lowerWatermarks: claim.lowerWatermarks,
          upperWatermarks,
        });
        success = true;
      } catch (error) {
        this.logger.warn?.(`memx: maintenance batch failed (${String(error)})`);
      } finally {
        const nextState = store.maintenanceRepo.finishBatch({
          agentId: template.agentId,
          sessionKey,
          leaseOwner: this.maintenanceLeaseOwner,
          completedAt: nowIso(),
          success,
          ...(success ? { upperWatermarks } : {}),
        });
        if (!success) {
          if (
            nextState?.pendingTurnCount &&
            template.config.advanced.maintenanceTriggerMode === "batched" &&
            template.config.advanced.maintenanceIdleFlushMinutes > 0
          ) {
            this.scheduleIdleFlush(store, template);
          }
          return;
        }
      }

      const postState = store.maintenanceRepo.getState(template.agentId, sessionKey);
      const postPending = postState?.pendingTurnCount ?? 0;
      if (reason === "shutdown") {
        if (postPending > 0) {
          reason = "shutdown";
          continue;
        }
        return;
      }
      if (postPending >= threshold) {
        reason = "threshold";
        continue;
      }
      if (
        postPending > 0 &&
        template.config.advanced.maintenanceTriggerMode === "batched" &&
        template.config.advanced.maintenanceIdleFlushMinutes > 0
      ) {
        this.scheduleIdleFlush(store, template);
      }
      return;
    }
  }

  private async flushPendingMaintenance(reason: MaintenanceBatchTriggerReason): Promise<void> {
    const entries = await Promise.all(this.stores.values());
    for (const store of entries) {
      const pendingStates = store.maintenanceRepo.listPendingStates();
      for (const state of pendingStates) {
        const key = maintenanceKey(state.agentId, store.client.dbPath, state.sessionKey);
        const template = this.maintenanceContextForPendingState(store, state);
        if (!template) {
          continue;
        }
        this.maintenanceContexts.set(key, template);
        const existing = this.maintenanceLoops.get(key);
        if (existing) {
          await existing;
        }
        await this.runMaintenanceLoop(store, template, reason);
      }
    }
  }

  private storeContextTemplatesForStore(store: MemxStoreBundle): MaintenanceContextTemplate[] {
    const templates = [...this.storeContexts.values()].filter(
      (template) => template.dbPath === store.client.dbPath,
    );
    const seen = new Set<string>();
    const deduped: MaintenanceContextTemplate[] = [];
    for (const template of templates) {
      const key = `${template.agentId}:${template.sessionKey ?? "default"}`;
      if (seen.has(key)) {
        continue;
      }
      seen.add(key);
      deduped.push(template);
    }
    return deduped;
  }

  private async flushRetryableSemanticWriteJobs(reason: MaintenanceBatchTriggerReason): Promise<void> {
    const entries = await Promise.all(this.stores.values());
    for (const store of entries) {
      for (const baseTemplate of this.storeContextTemplatesForStore(store)) {
        const now = nowIso();
        const retryJobs = store.auditRepo.listRetryableSemanticWriteJobs({
          agentId: baseTemplate.agentId,
          now,
          limit: 200,
        });
        const sessionKeys = uniqueStrings(retryJobs.map((job) => job.sessionKey));
        for (const sessionKey of sessionKeys) {
          const sessionJobs = retryJobs.filter((job) => job.sessionKey === sessionKey);
          const template: MaintenanceContextTemplate = {
            ...baseTemplate,
            sessionKey,
            dbPath: store.client.dbPath,
            scopes: uniqueStrings([...baseTemplate.scopes, ...sessionJobs.map((job) => job.scope)]),
          };
          const key = maintenanceKey(template.agentId, template.dbPath, sessionKey);
          const existing = this.maintenanceLoops.get(key);
          if (existing) {
            await existing;
          }
          if (
            store.auditRepo.listRetryableSemanticWriteJobs({
              agentId: template.agentId,
              sessionKey,
              now: nowIso(),
              limit: 1,
            }).length === 0
          ) {
            continue;
          }
          const upperWatermarks = {
            event: store.eventRepo.latestObservedAt({
              agentId: template.agentId,
              scopes: template.scopes,
              sessionKey,
            }),
            signal: store.auditRepo.latestSignalCreatedAt({
              agentId: template.agentId,
              sessionKey,
            }),
            task: store.taskRepo.latestUpdatedAt({
              agentId: template.agentId,
              scopes: template.scopes,
              sessionKey,
            }),
          };
          await runAutomaticMaintenanceBatch(store, this.buildMaintenanceContext(template), {
            sessionKey,
            turnIds: [],
            turnCount: 0,
            reason,
            lowerWatermarks: {},
            upperWatermarks,
          });
        }
      }
    }
  }

  private maintenanceContextForPendingState(
    store: MemxStoreBundle,
    state: { agentId: string; sessionKey: string },
  ): MaintenanceContextTemplate | undefined {
    const sessionKey = state.sessionKey || "default";
    const exactKey = maintenanceKey(state.agentId, store.client.dbPath, sessionKey);
    const exact = this.maintenanceContexts.get(exactKey);
    if (exact) {
      return exact;
    }
    const base = this.storeContexts.get(storeKey(state.agentId, store.client.dbPath));
    if (!base) {
      return undefined;
    }
    return {
      ...base,
      agentId: state.agentId,
      sessionKey,
      dbPath: store.client.dbPath,
    };
  }

  private async createStore(ctx: MemoryOperationContext): Promise<MemxStoreBundle> {
    const client = await MemxDbClient.open(ctx.dbPath);
    const vectorRepo = new VectorRepo(client);
    const retrievalBackend = new OptionalEmbeddingBackend(vectorRepo, ctx.config.embedding, this.logger);
    void retrievalBackend.prewarmLocalEmbeddings();
    const auditRepo = new AuditRepo(client);
    auditRepo.markExpiredRunningMaintenanceRunsInterrupted({
      agentId: ctx.agentId,
      completedAt: nowIso(),
      startedBefore: new Date(Date.now() - 5 * 60_000).toISOString(),
      reason: "runtime-startup-expired",
    });
    const bundle = {
      client,
      stateRepo: new StateRepo(client),
      taskRepo: new TaskRepo(client),
      chunkRepo: new ChunkRepo(client),
      factRepo: new FactRepo(client),
      eventRepo: new EventRepo(client),
      graphRepo: new GraphRepo(client),
      sourceSegmentRepo: new SourceSegmentRepo(client),
      vectorRepo,
      auditRepo,
      maintenanceRepo: new MaintenanceRepo(client),
      beliefRepo: new BeliefRepo(client),
      abstractionRepo: new AbstractionRepo(client),
      strategyRepo: new StrategyRepo(client),
      retrievalBackend,
      reasoner: new MemxReasoner(ctx.config, this.logger),
      turnScheduler: undefined as unknown as MemxTurnScheduler,
    };
    bundle.turnScheduler = new MemxTurnScheduler(bundle, this.logger);
    return bundle;
  }
}
