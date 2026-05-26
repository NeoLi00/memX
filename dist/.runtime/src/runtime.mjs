import { nowIso, randomId, resolveUserPath } from "./support.mjs";
import { createMemoryLlmBudgetAudit } from "./pipeline/llmBudgetAudit.mjs";
import { MemxReasoner } from "./pipeline/reasoner.mjs";
import { MemxDbClient } from "./db/client.mjs";
import { AbstractionRepo } from "./db/repositories/abstractionRepo.mjs";
import { AuditRepo } from "./db/repositories/auditRepo.mjs";
import { BeliefRepo } from "./db/repositories/beliefRepo.mjs";
import { ChunkRepo } from "./db/repositories/chunkRepo.mjs";
import { EventRepo } from "./db/repositories/eventRepo.mjs";
import { FactRepo } from "./db/repositories/factRepo.mjs";
import { GraphRepo } from "./db/repositories/graphRepo.mjs";
import { MaintenanceRepo } from "./db/repositories/maintenanceRepo.mjs";
import { SourceSegmentRepo } from "./db/repositories/sourceSegmentRepo.mjs";
import { StateRepo } from "./db/repositories/stateRepo.mjs";
import { StrategyRepo } from "./db/repositories/strategyRepo.mjs";
import { TaskRepo } from "./db/repositories/taskRepo.mjs";
import { VectorRepo } from "./db/repositories/vectorRepo.mjs";
import { runAutomaticMaintenanceBatch } from "./pipeline/maintenanceBatch.mjs";
import { MemxTurnScheduler } from "./pipeline/turnScheduler.mjs";
import { OptionalEmbeddingBackend } from "./search/backends/embeddingBackend.mjs";
import { defaultRetrievalScopes, renderTemplate, scopeVarsForContext } from "./security/scopes.mjs";
//#region src/runtime.ts
function storeKey(agentId, dbPath) {
	return `${agentId}:${dbPath}`;
}
function maintenanceKey(agentId, dbPath, sessionKey) {
	return `${agentId}:${dbPath}:${sessionKey}`;
}
function uniqueStrings(values) {
	return [...new Set(values.filter((value) => value.trim().length > 0))];
}
function stagedRecallKey(ctx) {
	const workspace = ctx.workspaceDir?.trim();
	const scope = workspace ? `workspace:${workspace}` : `session:${ctx.sessionKey ?? "default"}`;
	return `${ctx.agentId}:${ctx.dbPath}:${scope}`;
}
function resolveDbPath(config, actor) {
	if (!actor.agentId) return null;
	return resolveUserPath(renderTemplate(config.dbPath, scopeVarsForContext(actor)));
}
function buildOperationContext(config, actor, overrides) {
	if (!actor.agentId) return null;
	const dbPath = resolveDbPath(config, actor);
	if (!dbPath) return null;
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
		llmBudgetAudit: createMemoryLlmBudgetAudit()
	};
}
var MemxRuntimeManager = class {
	logger;
	stores = /* @__PURE__ */ new Map();
	storeContexts = /* @__PURE__ */ new Map();
	sessionCursors = /* @__PURE__ */ new Map();
	lastRecall = /* @__PURE__ */ new Map();
	stagedRecallableTurns = /* @__PURE__ */ new Map();
	maintenanceTimers = /* @__PURE__ */ new Map();
	maintenanceContexts = /* @__PURE__ */ new Map();
	maintenanceLoops = /* @__PURE__ */ new Map();
	maintenanceLeaseOwner = randomId("maintenance-runtime");
	constructor(logger) {
		this.logger = logger;
	}
	async getStore(ctx) {
		const key = storeKey(ctx.agentId, ctx.dbPath);
		this.storeContexts.set(key, {
			...ctx,
			sessionKey: ctx.sessionKey ?? "default"
		});
		const existing = this.stores.get(key);
		if (existing) return existing;
		const created = this.createStore(ctx);
		this.stores.set(key, created);
		created.catch(() => {
			this.stores.delete(key);
		});
		return created;
	}
	async closeAll() {
		await this.flushAll();
		await this.flushPendingMaintenance("shutdown");
		await this.flushRetryableSemanticWriteJobs("shutdown");
		for (const timer of this.maintenanceTimers.values()) clearTimeout(timer);
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
	async flushAll() {
		for (const entry of this.stores.values()) {
			const store = await entry;
			await store.turnScheduler.flush();
			await store.retrievalBackend.flushPendingUpserts?.();
		}
	}
	getSessionCursor(agentId, sessionKey) {
		return this.sessionCursors.get(`${agentId}:${sessionKey}`);
	}
	setSessionCursor(agentId, sessionKey, cursor) {
		this.sessionCursors.set(`${agentId}:${sessionKey}`, cursor);
	}
	rememberRecall(agentId, sessionKey, payload) {
		if (!sessionKey) return;
		this.lastRecall.set(`${agentId}:${sessionKey}`, payload);
	}
	consumeRecall(agentId, sessionKey) {
		if (!sessionKey) return {
			chunkIds: [],
			texts: []
		};
		const key = `${agentId}:${sessionKey}`;
		const payload = this.lastRecall.get(key) ?? {
			chunkIds: [],
			texts: []
		};
		this.lastRecall.delete(key);
		return payload;
	}
	rememberStagedRecallableTurn(ctx, messages) {
		const text = messages.map((message) => `[${message.role}] ${message.content.trim()}`).filter((entry) => entry.trim().length > 0).join("\n");
		if (!text.trim()) return;
		const key = stagedRecallKey(ctx);
		const turnId = messages[0]?.turnId ?? randomId("staged-turn");
		const observedAt = messages.at(-1)?.observedAt ?? nowIso();
		const existing = (this.stagedRecallableTurns.get(key) ?? []).filter((entry) => entry.turnId !== turnId);
		existing.push({
			turnId,
			observedAt,
			text
		});
		this.stagedRecallableTurns.set(key, existing.slice(-6));
	}
	recentStagedRecallableTurns(ctx, limit = 4) {
		const key = stagedRecallKey(ctx);
		return [...this.stagedRecallableTurns.get(key) ?? []].sort((left, right) => right.observedAt.localeCompare(left.observedAt)).slice(0, Math.max(1, Math.min(Math.trunc(limit), 8)));
	}
	async recordMaintenanceTurn(ctx, params) {
		if (!ctx.config.advanced.enableMaintenanceJobs) return;
		const sessionKey = ctx.sessionKey ?? "default";
		const store = params.store ?? await this.getStore(ctx);
		const key = maintenanceKey(ctx.agentId, ctx.dbPath, sessionKey);
		const template = {
			...ctx,
			sessionKey
		};
		this.maintenanceContexts.set(key, template);
		const state = store.maintenanceRepo.recordPendingTurn({
			agentId: ctx.agentId,
			sessionKey,
			turnId: params.turnId,
			observedAt: params.observedAt,
			updatedAt: ctx.now
		});
		const threshold = ctx.config.advanced.maintenanceTriggerMode === "per_turn" ? 1 : Math.max(1, ctx.config.advanced.maintenanceBatchTurns);
		if (state.pendingTurnCount >= threshold) {
			this.clearMaintenanceTimer(key);
			this.startMaintenanceLoop(store, template, "threshold");
			return;
		}
		if (ctx.config.advanced.maintenanceTriggerMode === "batched" && ctx.config.advanced.maintenanceIdleFlushMinutes > 0) this.scheduleIdleFlush(store, template);
	}
	startMaintenanceLoop(store, template, reason) {
		const sessionKey = template.sessionKey ?? "default";
		const key = maintenanceKey(template.agentId, template.dbPath, sessionKey);
		if (this.maintenanceLoops.get(key)) return;
		const loop = this.runMaintenanceLoop(store, template, reason).finally(() => {
			this.maintenanceLoops.delete(key);
		});
		this.maintenanceLoops.set(key, loop);
		loop.catch((error) => {
			this.logger.warn?.(`memx: maintenance batch loop failed (${String(error)})`);
		});
	}
	scheduleIdleFlush(store, template) {
		const sessionKey = template.sessionKey ?? "default";
		const key = maintenanceKey(template.agentId, template.dbPath, sessionKey);
		this.clearMaintenanceTimer(key);
		const delayMs = Math.max(.001, template.config.advanced.maintenanceIdleFlushMinutes) * 6e4;
		const timer = setTimeout(() => {
			this.maintenanceTimers.delete(key);
			this.startMaintenanceLoop(store, template, "idle");
		}, delayMs);
		this.maintenanceTimers.set(key, timer);
	}
	clearMaintenanceTimer(key) {
		const existing = this.maintenanceTimers.get(key);
		if (!existing) return;
		clearTimeout(existing);
		this.maintenanceTimers.delete(key);
	}
	buildMaintenanceContext(template) {
		return {
			...template,
			now: nowIso(),
			llmBudgetAudit: createMemoryLlmBudgetAudit()
		};
	}
	async runMaintenanceLoop(store, template, initialReason) {
		const sessionKey = template.sessionKey ?? "default";
		const key = maintenanceKey(template.agentId, template.dbPath, sessionKey);
		let reason = initialReason;
		const threshold = template.config.advanced.maintenanceTriggerMode === "per_turn" ? 1 : Math.max(1, template.config.advanced.maintenanceBatchTurns);
		while (true) {
			const pendingTurnCount = store.maintenanceRepo.getState(template.agentId, sessionKey)?.pendingTurnCount ?? 0;
			if (!(reason === "shutdown" ? pendingTurnCount > 0 : reason === "idle" ? pendingTurnCount > 0 : pendingTurnCount >= threshold)) {
				if (pendingTurnCount > 0 && template.config.advanced.maintenanceTriggerMode === "batched" && template.config.advanced.maintenanceIdleFlushMinutes > 0) this.scheduleIdleFlush(store, template);
				return;
			}
			const claim = store.maintenanceRepo.claimBatch({
				agentId: template.agentId,
				sessionKey,
				reason,
				leaseOwner: this.maintenanceLeaseOwner,
				leaseTtlMs: 5 * 6e4,
				now: nowIso()
			});
			if (!claim) return;
			this.clearMaintenanceTimer(key);
			const upperWatermarks = {
				event: store.eventRepo.latestObservedAt({
					agentId: template.agentId,
					scopes: template.scopes,
					sessionKey
				}),
				signal: store.auditRepo.latestSignalCreatedAt({
					agentId: template.agentId,
					sessionKey
				}),
				task: store.taskRepo.latestUpdatedAt({
					agentId: template.agentId,
					scopes: template.scopes,
					sessionKey
				})
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
					upperWatermarks
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
					...success ? { upperWatermarks } : {}
				});
				if (!success) {
					if (nextState?.pendingTurnCount && template.config.advanced.maintenanceTriggerMode === "batched" && template.config.advanced.maintenanceIdleFlushMinutes > 0) this.scheduleIdleFlush(store, template);
					return;
				}
			}
			const postPending = store.maintenanceRepo.getState(template.agentId, sessionKey)?.pendingTurnCount ?? 0;
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
			if (postPending > 0 && template.config.advanced.maintenanceTriggerMode === "batched" && template.config.advanced.maintenanceIdleFlushMinutes > 0) this.scheduleIdleFlush(store, template);
			return;
		}
	}
	async flushPendingMaintenance(reason) {
		const entries = await Promise.all(this.stores.values());
		for (const store of entries) {
			const pendingStates = store.maintenanceRepo.listPendingStates();
			for (const state of pendingStates) {
				const key = maintenanceKey(state.agentId, store.client.dbPath, state.sessionKey);
				const template = this.maintenanceContextForPendingState(store, state);
				if (!template) continue;
				this.maintenanceContexts.set(key, template);
				const existing = this.maintenanceLoops.get(key);
				if (existing) await existing;
				await this.runMaintenanceLoop(store, template, reason);
			}
		}
	}
	storeContextTemplatesForStore(store) {
		const templates = [...this.storeContexts.values()].filter((template) => template.dbPath === store.client.dbPath);
		const seen = /* @__PURE__ */ new Set();
		const deduped = [];
		for (const template of templates) {
			const key = `${template.agentId}:${template.sessionKey ?? "default"}`;
			if (seen.has(key)) continue;
			seen.add(key);
			deduped.push(template);
		}
		return deduped;
	}
	async flushRetryableSemanticWriteJobs(reason) {
		const entries = await Promise.all(this.stores.values());
		for (const store of entries) for (const baseTemplate of this.storeContextTemplatesForStore(store)) {
			const now = nowIso();
			const retryJobs = store.auditRepo.listRetryableSemanticWriteJobs({
				agentId: baseTemplate.agentId,
				now,
				limit: 200
			});
			const sessionKeys = uniqueStrings(retryJobs.map((job) => job.sessionKey));
			for (const sessionKey of sessionKeys) {
				const sessionJobs = retryJobs.filter((job) => job.sessionKey === sessionKey);
				const template = {
					...baseTemplate,
					sessionKey,
					dbPath: store.client.dbPath,
					scopes: uniqueStrings([...baseTemplate.scopes, ...sessionJobs.map((job) => job.scope)])
				};
				const key = maintenanceKey(template.agentId, template.dbPath, sessionKey);
				const existing = this.maintenanceLoops.get(key);
				if (existing) await existing;
				if (store.auditRepo.listRetryableSemanticWriteJobs({
					agentId: template.agentId,
					sessionKey,
					now: nowIso(),
					limit: 1
				}).length === 0) continue;
				const upperWatermarks = {
					event: store.eventRepo.latestObservedAt({
						agentId: template.agentId,
						scopes: template.scopes,
						sessionKey
					}),
					signal: store.auditRepo.latestSignalCreatedAt({
						agentId: template.agentId,
						sessionKey
					}),
					task: store.taskRepo.latestUpdatedAt({
						agentId: template.agentId,
						scopes: template.scopes,
						sessionKey
					})
				};
				await runAutomaticMaintenanceBatch(store, this.buildMaintenanceContext(template), {
					sessionKey,
					turnIds: [],
					turnCount: 0,
					reason,
					lowerWatermarks: {},
					upperWatermarks
				});
			}
		}
	}
	maintenanceContextForPendingState(store, state) {
		const sessionKey = state.sessionKey || "default";
		const exactKey = maintenanceKey(state.agentId, store.client.dbPath, sessionKey);
		const exact = this.maintenanceContexts.get(exactKey);
		if (exact) return exact;
		const base = this.storeContexts.get(storeKey(state.agentId, store.client.dbPath));
		if (!base) return;
		return {
			...base,
			agentId: state.agentId,
			sessionKey,
			dbPath: store.client.dbPath
		};
	}
	async createStore(ctx) {
		const client = await MemxDbClient.open(ctx.dbPath);
		const vectorRepo = new VectorRepo(client);
		const retrievalBackend = new OptionalEmbeddingBackend(vectorRepo, ctx.config.embedding, this.logger);
		retrievalBackend.prewarmLocalEmbeddings();
		const auditRepo = new AuditRepo(client);
		auditRepo.markExpiredRunningMaintenanceRunsInterrupted({
			agentId: ctx.agentId,
			completedAt: nowIso(),
			startedBefore: (/* @__PURE__ */ new Date(Date.now() - 5 * 6e4)).toISOString(),
			reason: "runtime-startup-expired"
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
			turnScheduler: void 0
		};
		bundle.turnScheduler = new MemxTurnScheduler(bundle, this.logger);
		return bundle;
	}
};
//#endregion
export { MemxRuntimeManager, buildOperationContext };
