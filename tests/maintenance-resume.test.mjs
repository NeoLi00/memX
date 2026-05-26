import assert from "node:assert/strict";
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";
import { DEFAULT_MEMORY_CONFIG } from "../dist/.runtime/src/config.mjs";
import { MemxDbClient } from "../dist/.runtime/src/db/client.mjs";
import { MaintenanceRepo } from "../dist/.runtime/src/db/repositories/maintenanceRepo.mjs";
import { createServiceConfigFromEnv, MemxHostService } from "../dist/.runtime/src/host/service.mjs";
import { buildOperationContext, MemxRuntimeManager } from "../dist/.runtime/src/runtime.mjs";

const observedAt = "2026-05-13T00:00:00.000Z";

function logger() {
  return {
    debug() {},
    info() {},
    warn() {},
    error() {},
  };
}

function configFor(dbPath) {
  return {
    ...DEFAULT_MEMORY_CONFIG,
    dbPath,
    embedding: {
      ...DEFAULT_MEMORY_CONFIG.embedding,
      provider: "off",
    },
    advanced: {
      ...DEFAULT_MEMORY_CONFIG.advanced,
      enableMaintenanceJobs: true,
      maintenanceTriggerMode: "batched",
      maintenanceBatchTurns: 3,
      maintenanceIdleFlushMinutes: 10,
      enableEmbeddingCandidates: false,
    },
  };
}

function ctxFor(dbPath) {
  const config = configFor(dbPath);
  return {
    agentId: "main",
    sessionKey: "active-session",
    workspaceDir: "/tmp/memx-test-workspace",
    project: "memx-test",
    runId: "test-run",
    channelId: "test-channel",
    config,
    dbPath,
    scopes: ["agent:main"],
    now: observedAt,
  };
}

async function waitFor(predicate, timeoutMs = 1000) {
  const started = Date.now();
  while (Date.now() - started < timeoutMs) {
    if (predicate()) {
      return true;
    }
    await new Promise((resolve) => setTimeout(resolve, 20));
  }
  return predicate();
}

test("runtime threshold immediately starts an automatic maintenance batch", async () => {
  const tempDir = await mkdtemp(join(tmpdir(), "memx-maintenance-threshold-"));
  const dbPath = join(tempDir, "memx.sqlite");

  try {
    const manager = new MemxRuntimeManager(logger());
    const ctx = ctxFor(dbPath);
    const store = await manager.getStore(ctx);

    await manager.recordMaintenanceTurn(ctx, {
      store,
      turnId: "turn-1",
      observedAt,
    });
    await manager.recordMaintenanceTurn(ctx, {
      store,
      turnId: "turn-2",
      observedAt,
    });
    await manager.recordMaintenanceTurn(ctx, {
      store,
      turnId: "turn-3",
      observedAt,
    });

    const completed = await waitFor(() => {
      const state = store.maintenanceRepo.getState("main", "active-session");
      const runCount = store.client
        .prepare("SELECT COUNT(*) AS count FROM maintenance_runs")
        .get().count;
      return Number(runCount) > 0 && state?.pendingTurnCount === 0 && state.inflightTurnCount === 0;
    });

    assert.equal(completed, true);
    await manager.closeAll();
  } finally {
    await rm(tempDir, { recursive: true, force: true });
  }
});

test("standalone service config preserves maintenance scheduler defaults", async () => {
  const tempDir = await mkdtemp(join(tmpdir(), "memx-maintenance-config-"));

  try {
    const config = createServiceConfigFromEnv({
      MEMX_CONFIG_PATH: join(tempDir, "missing-config.json"),
    });

    assert.equal(config.advanced.maintenanceTriggerMode, "batched");
    assert.equal(config.advanced.maintenanceBatchTurns, 3);
    assert.equal(config.advanced.maintenanceIdleFlushMinutes, 10);
  } finally {
    await rm(tempDir, { recursive: true, force: true });
  }
});

test("standalone service scopes native memory by workspace without exposing raw paths", async () => {
  const tempDir = await mkdtemp(join(tmpdir(), "memx-workspace-scope-"));

  try {
    const config = createServiceConfigFromEnv({
      MEMX_CONFIG_PATH: join(tempDir, "missing-config.json"),
    });
    const left = buildOperationContext(config, {
      agentId: "codex--memx-shared",
      sessionKey: "codex:left",
      workspaceDir: "/tmp/acme/private/project-alpha",
    });
    const right = buildOperationContext(config, {
      agentId: "codex--memx-shared",
      sessionKey: "codex:right",
      workspaceDir: "/tmp/acme/private/project-beta",
    });

    const leftWorkspaceScope = left.scopes.find((scope) => scope.startsWith("workspace:"));
    const rightWorkspaceScope = right.scopes.find((scope) => scope.startsWith("workspace:"));
    assert.ok(leftWorkspaceScope, "expected a workspace scope for native standalone hosts");
    assert.ok(rightWorkspaceScope, "expected a workspace scope for native standalone hosts");
    assert.notEqual(leftWorkspaceScope, rightWorkspaceScope);
    assert.equal(left.scopes.includes("agent:codex--memx-shared"), false);
    assert.deepEqual(left.scopes, [leftWorkspaceScope, "session:codex:left"]);
    assert.equal(left.scopes.some((scope) => scope.includes("/tmp/acme/private")), false);
    assert.equal(config.defaultScope, "workspace:{workspace}");
  } finally {
    await rm(tempDir, { recursive: true, force: true });
  }
});

test("standalone native capture writes to hashed workspace scope instead of raw path or ambient agent scope", async () => {
  const tempDir = await mkdtemp(join(tmpdir(), "memx-native-workspace-scope-"));
  const dbPathTemplate = join(tempDir, "{agentId}", "memx.sqlite");

  try {
    const base = createServiceConfigFromEnv({
      MEMX_CONFIG_PATH: join(tempDir, "missing-config.json"),
    });
    const service = new MemxHostService({
      config: {
        ...base,
        dbPath: dbPathTemplate,
        embedding: { ...base.embedding, provider: "off" },
        advanced: {
          ...base.advanced,
          llmClassifierEnabled: false,
          enableEmbeddingCandidates: false,
        },
      },
      logger: logger(),
    });
    await service.observe({
      hostId: "codex",
      actorId: "memx-shared",
      sessionId: "codex:alpha",
      workspaceDir: "/tmp/acme/private/project-alpha",
      messages: [{ role: "user", content: "记住 AsterFlow 默认数据库是 SQLite。" }],
    });
    await service.close();

    const client = new MemxDbClient(join(tempDir, "codex--memx-shared", "memx.sqlite"));
    try {
      const rows = client
        .prepare("SELECT DISTINCT scope FROM conversation_chunks WHERE agent_id = ?")
        .all("codex--memx-shared");
      assert.equal(rows.length, 1);
      const scope = rows[0]?.scope;
      assert.match(scope, /^workspace:[0-9a-f]{16}$/);
      assert.equal(scope.includes("/tmp/acme/private"), false);
      assert.notEqual(scope, "agent:codex--memx-shared");
    } finally {
      client.close();
    }
  } finally {
    await rm(tempDir, { recursive: true, force: true });
  }
});

test("runtime shutdown flushes persisted maintenance rows without an in-memory session context", async () => {
  const tempDir = await mkdtemp(join(tmpdir(), "memx-maintenance-resume-"));
  const dbPath = join(tempDir, "memx.sqlite");

  try {
    const manager = new MemxRuntimeManager(logger());
    const store = await manager.getStore(ctxFor(dbPath));

    store.maintenanceRepo.recordPendingTurn({
      agentId: "main",
      sessionKey: "orphan-session",
      turnId: "turn-orphan",
      observedAt,
      updatedAt: observedAt,
    });

    await manager.closeAll();

    const client = await MemxDbClient.open(dbPath);
    try {
      const repo = new MaintenanceRepo(client);
      const state = repo.getState("main", "orphan-session");
      assert.ok(state);
      assert.equal(state.pendingTurnCount, 0);
      assert.equal(state.inflightTurnCount, 0);
      assert.equal(state.status, "idle");
      const runCount = client
        .prepare("SELECT COUNT(*) AS count FROM maintenance_runs")
        .get().count;
      assert.ok(Number(runCount) > 0);
    } finally {
      client.close();
    }
  } finally {
    await rm(tempDir, { recursive: true, force: true });
  }
});

test("runtime startup marks expired running maintenance audit rows as interrupted", async () => {
  const tempDir = await mkdtemp(join(tmpdir(), "memx-maintenance-stale-run-"));
  const dbPath = join(tempDir, "memx.sqlite");

  try {
    const client = await MemxDbClient.open(dbPath);
    try {
      client
        .prepare(
          `INSERT INTO maintenance_runs(run_id, agent_id, job_type, stats_json, started_at, completed_at, status)
           VALUES (?, ?, ?, ?, ?, NULL, 'running')`,
        )
        .run(
          "maintenance-stale",
          "main",
          "source-segment-semantic-extraction",
          JSON.stringify({ status: "started" }),
          "2026-05-13T00:00:00.000Z",
        );
    } finally {
      client.close();
    }

    const manager = new MemxRuntimeManager(logger());
    await manager.getStore(ctxFor(dbPath));
    await manager.closeAll();

    const checked = await MemxDbClient.open(dbPath);
    try {
      const row = checked
        .prepare("SELECT status, completed_at, stats_json FROM maintenance_runs WHERE run_id = ?")
        .get("maintenance-stale");
      assert.equal(row.status, "failed");
      assert.ok(row.completed_at);
      const stats = JSON.parse(row.stats_json);
      assert.equal(stats.recovery.reason, "runtime-startup-expired");
    } finally {
      checked.close();
    }
  } finally {
    await rm(tempDir, { recursive: true, force: true });
  }
});
