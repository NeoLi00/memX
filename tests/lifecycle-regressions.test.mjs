import assert from "node:assert/strict";
import { createServer } from "node:http";
import { mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { spawn } from "node:child_process";
import test from "node:test";
import { DEFAULT_MEMORY_CONFIG } from "../dist/.runtime/src/config.mjs";
import { buildOperationContext, MemxRuntimeManager } from "../dist/.runtime/src/runtime.mjs";
import { runAutomaticMaintenanceBatch } from "../dist/.runtime/src/pipeline/maintenanceBatch.mjs";
import { compileQueryWithoutSemanticFallback } from "../dist/.runtime/src/pipeline/queryCompiler.mjs";
import { retrieveEvidence } from "../dist/.runtime/src/pipeline/retrieve.mjs";
import {
  MEMX_NATIVE_HOOK_TIMEOUT_MS,
  deriveNativeHookHttpTimeoutMs,
  deriveNativeHookQueryCompilerTimeoutMs,
} from "../dist/.runtime/src/timeouts.mjs";

const observedAt = "2026-05-21T00:00:00.000Z";

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
      enableTurnSemanticCompiler: true,
      enableTelemetryAudit: true,
      enableMaintenanceJobs: true,
      enableEmbeddingCandidates: false,
      maintenanceTriggerMode: "batched",
      maintenanceBatchTurns: 3,
    },
  };
}

function ctxFor(dbPath) {
  const config = configFor(dbPath);
  return {
    agentId: "main",
    sessionKey: "s1",
    workspaceDir: "/tmp/memx-lifecycle-test",
    project: "memx-lifecycle",
    runId: "test-run",
    channelId: "test-channel",
    config,
    dbPath,
    scopes: ["agent:main"],
    now: observedAt,
    llmBudgetAudit: {
      calls: [],
      hotPathLlmCallCount: 0,
      writeHotPathLlmCallCount: 0,
      queryHotPathLlmCallCount: 0,
      postAnswerWritebackLlmCallCount: 0,
      maintenanceLlmCallCount: 0,
    },
  };
}

function directFactQueryAnalysis(query, subject, relation) {
  return {
    ...compileQueryWithoutSemanticFallback(query),
    queryEntities: [{ name: subject, type: "project", role: "subject" }],
    queryShape: {
      timeframe: "current",
      granularity: "exact_detail",
      referentialMode: "anchored",
      evidenceNeed: "canonical_state",
    },
    primaryRoute: "factual",
    answerGranularity: "detail",
    evidenceFidelity: "medium",
    routeWeights: { factual: 0.78, temporal: 0.12, workflow: 0.05, explanatory: 0.05 },
    anchors: [subject, relation],
    candidateSurfaces: ["fact", "event", "chunk", "state"],
    evidenceGoals: [
      {
        goal: `Return the current ${relation} value for ${subject}.`,
        positiveQueries: [query, `${subject} ${relation}`],
        negativeHints: [],
        focusAnchors: [subject, relation],
        preferredSurfaces: ["fact"],
        fidelity: "medium",
      },
    ],
    evidencePlan: {
      operation: {
        type: "return_value",
        description: "Return the single current attribute value directly supported by memory.",
      },
      slots: [
        {
          id: "query_context",
          role: "query_context",
          requiredRole: "query_context",
          description: "The entity whose attribute is being requested.",
          subjectHints: [subject],
          relationHints: [relation],
          capabilityQueries: [],
          negativeHints: [],
          requiredFields: [subject],
          preferredLayers: ["fact"],
          fallbackLayers: ["chunk"],
          minEvidence: 1,
        },
        {
          id: "answer_value",
          role: "answer_value",
          requiredRole: "answer_value",
          description: "The stored value for the requested attribute.",
          subjectHints: [subject],
          relationHints: [relation, relation.replace(/_/g, " ")],
          capabilityQueries: [],
          negativeHints: [],
          requiredFields: ["answer_value"],
          preferredLayers: ["fact"],
          fallbackLayers: ["chunk"],
          minEvidence: 1,
        },
      ],
    },
    semanticBridges: [],
    answerMode: "attribute_lookup",
    supportNeed: 0.42,
    detailNeedScore: 0.48,
    ambiguityLevel: 0.05,
    compilerProvenance: {
      source: "llm",
      mode: "semantic-compiler-authoritative",
      reasons: ["test-direct-fact-query"],
    },
  };
}

function relationPatch(sourceRef) {
  return {
    sourceRefs: [sourceRef],
    assertionDrafts: [
      {
        draftId: "draft-relation",
        sourceRef,
        familyHint: "relation_like",
        timeframeHint: "current",
        entityHints: [
          { name: "InvoicePilot", type: "project" },
          { name: "PostgreSQL", type: "service" },
        ],
        confidence: 0.92,
        lineage: { sourceKind: "chunk", sourceId: "chunk-test", sourceRef },
      },
    ],
    relationDrafts: [
      {
        sourceRef,
        relation: {
          subject: "InvoicePilot",
          predicate: "uses",
          relationSlot: "database",
          object: "PostgreSQL",
          sourceRef,
          confidence: 0.92,
        },
        confidence: 0.92,
        lineage: { sourceKind: "chunk", sourceId: "chunk-test", sourceRef },
      },
    ],
    supportSpans: [{ sourceRef, text: "InvoicePilot 默认数据库是 PostgreSQL" }],
    compilerProvenance: {
      source: "llm",
      mode: "semantic-compiler-authoritative",
      reasons: ["test-semantic-frame"],
    },
  };
}

test("native hook timeout budget keeps query compiler inside the unified 8 second hook limit", () => {
  const httpTimeoutMs = deriveNativeHookHttpTimeoutMs(MEMX_NATIVE_HOOK_TIMEOUT_MS);
  const queryTimeoutMs = deriveNativeHookQueryCompilerTimeoutMs(httpTimeoutMs);

  assert.equal(MEMX_NATIVE_HOOK_TIMEOUT_MS, 8000);
  assert.equal(httpTimeoutMs, 7500);
  assert.ok(
    queryTimeoutMs <= 4500,
    "the query compiler must leave enough of the 8s hook budget for retrieval and HTTP return",
  );
  assert.ok(queryTimeoutMs >= 3000);
  assert.ok(queryTimeoutMs < httpTimeoutMs);
});

test("anchored native recall with no target entity evidence withholds unrelated strong evidence", async () => {
  const { assessNativeContextEligibility, focusRecallBundleForQueryEntities } = await import(
    "../dist/.runtime/src/host/service.mjs"
  );
  const query = "TideLedger 导出流水线现在默认用什么队列？";
  const queryAnalysis = directFactQueryAnalysis(query, "TideLedger 导出流水线", "default message queue");
  const unrelatedPacket = {
    packetId: "packet-unrelated-qingshi",
    slotId: "answer_value",
    operationType: "return_value",
    role: "answer",
    protected: false,
    injected: true,
    layers: ["fact"],
    primaryText: "青石报表 has default database postgresql",
    supportingTexts: [],
    sourceRefs: ["fact:qingshi:default_database"],
    supportSourceRefs: [],
    allSourceRefs: ["fact:qingshi:default_database"],
    normalizedSourceRefs: ["fact:qingshi:default_database"],
    normalizedSupportSourceRefs: [],
    normalizedAllSourceRefs: ["fact:qingshi:default_database"],
    score: 0.91,
    scoreBreakdown: {
      retrievalScore: 0.91,
      answerScore: 0.91,
      contextBindingScore: 0.91,
      slotCoverageScore: 0.91,
      authorityScore: 0.91,
      finalScore: 0.91,
    },
    displayLines: ["[answer] 青石报表 has default database postgresql"],
    authorRoles: ["memory"],
    coverage: { filled: true, missing: [], confidence: 0.91 },
    eligibility: { eligible: true, role: "answer", blockers: [] },
    grade: {
      retrievalScore: 0.91,
      answerScore: 0.91,
      contextBindingScore: 0.91,
      slotCoverageScore: 0.91,
      authorityScore: 0.91,
      finalScore: 0.91,
    },
    selectionReason: "test-unrelated-strong-evidence",
  };
  const bundle = {
    routeType: "factual",
    routeConfidence: 0.91,
    queryText: query,
    queryAnchors: ["TideLedger 导出流水线", "default message queue"],
    states: [],
    tasks: [],
    facts: [
      {
        id: "fact:qingshi:default_database",
        text: "青石报表 has default database postgresql",
        score: 0.91,
        scope: "agent:main",
        confidence: 0.91,
        sourceRef: "fact:qingshi:default_database",
      },
    ],
    events: [],
    graph: { nodes: [], edges: [], paths: [], pathCandidates: [] },
    alternates: [],
    diagnostics: [],
    behavioralGuidance: [],
    recalledChunkIds: [],
    recalledChunkTexts: [],
    promptEvidence: [],
    evidencePackets: [unrelatedPacket],
    renderedBlock: "",
  };

  const focused = focusRecallBundleForQueryEntities(queryAnalysis, bundle);
  assert.equal(focused.evidencePackets.length, 0);
  assert.equal(focused.facts.length, 0);

  const eligibility = assessNativeContextEligibility(query, queryAnalysis, focused);
  assert.equal(eligibility.eligible, false);
  assert.equal(eligibility.reason, "no-injected-packets");
});

test("SessionEnd without a pending turn does not replay the latest transcript assistant", async () => {
  const tempDir = await mkdtemp(join(tmpdir(), "memx-hook-sessionend-"));
  const transcriptPath = join(tempDir, "claude-session.jsonl");
  const pendingDir = join(tempDir, "pending");
  const requests = [];
  const server = createServer((req, res) => {
    const chunks = [];
    req.on("data", (chunk) => chunks.push(chunk));
    req.on("end", () => {
      requests.push({
        url: req.url,
        body: Buffer.concat(chunks).toString("utf8"),
      });
      res.writeHead(200, { "content-type": "application/json" });
      res.end(JSON.stringify({ ok: true }));
    });
  });

  try {
    await writeFile(
      transcriptPath,
      `${JSON.stringify({
        type: "assistant",
        message: { content: [{ type: "text", text: "assistant output that was already captured" }] },
      })}\n`,
      "utf8",
    );
    await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
    const port = server.address().port;
    const child = spawn(process.execPath, ["dist/.runtime/src/bin/memx-hook.mjs", "claude-code", "SessionEnd"], {
      cwd: process.cwd(),
      env: {
        ...process.env,
        MEMX_URL: `http://127.0.0.1:${port}`,
        MEMX_PENDING_DIR: pendingDir,
        MEMX_TRANSCRIPT_CAPTURE_TIMEOUT_MS: "0",
      },
      stdio: ["pipe", "pipe", "pipe"],
    });
    child.stdin.end(
      JSON.stringify({
        session_id: "session-no-pending",
        cwd: "/tmp/memx-lifecycle-test",
        transcript_path: transcriptPath,
      }),
    );
    const code = await new Promise((resolve) => child.on("close", resolve));

    assert.equal(code, 0);
    assert.deepEqual(requests, []);
  } finally {
    server.close();
    await rm(tempDir, { recursive: true, force: true });
  }
});

test("turn scheduler sends the complete user plus assistant turn to the LLM semantic compiler", async () => {
  const tempDir = await mkdtemp(join(tmpdir(), "memx-turn-semantic-"));
  const dbPath = join(tempDir, "memx.sqlite");

  try {
    const manager = new MemxRuntimeManager(logger());
    const ctx = ctxFor(dbPath);
    const store = await manager.getStore(ctx);
    let capturedMessages = [];
    const chunkSummaryOptions = [];
    store.reasoner.isEnabled = () => true;
    store.reasoner.summarizeChunk = async (text, _role, options = {}) => {
      chunkSummaryOptions.push(options);
      return text.slice(0, 120);
    };
    store.reasoner.compileTurnSemantics = async (messages) => {
      capturedMessages = messages;
      return relationPatch("user:turn-semantic:0");
    };

    await store.turnScheduler.enqueue(ctx, [
      {
        role: "user",
        content: "记住：InvoicePilot 默认数据库是 PostgreSQL。",
        scope: "agent:main",
        sessionKey: "s1",
        turnId: "turn-semantic",
        sourceRef: "user:turn-semantic:0",
        observedAt,
      },
      {
        role: "assistant",
        content: "好的，后续我会按 PostgreSQL 处理 InvoicePilot 的默认数据库。",
        scope: "agent:main",
        sessionKey: "s1",
        turnId: "turn-semantic",
        sourceRef: "assistant:turn-semantic:1",
        observedAt,
      },
    ]);
    await store.turnScheduler.flush();

    assert.deepEqual(
      capturedMessages.map((message) => message.role),
      ["user", "assistant"],
    );
    assert.deepEqual(
      chunkSummaryOptions.map((options) => options.allowLlm),
      [false, false],
      "chunk summaries should be local previews; semantic extraction belongs to the turn compiler",
    );
    assert.equal(
      store.client.prepare("SELECT COUNT(*) AS count FROM conversation_chunks WHERE role = 'assistant'").get()
        .count,
      1,
    );
    assert.ok(
      Number(store.client.prepare("SELECT COUNT(*) AS count FROM graph_edges").get().count) > 0,
      "LLM relation drafts should be materialized into graph edges",
    );
    const ignored = store.client
      .prepare("SELECT reasons_json FROM policy_decisions WHERE reasons_json LIKE ?")
      .all("%no-compiler-family%");
    assert.deepEqual(ignored, []);
    await manager.closeAll();
  } finally {
    await rm(tempDir, { recursive: true, force: true });
  }
});

test("assistant-only LLM semantic drafts are materialized without storing the full assistant answer as a fact", async () => {
  const tempDir = await mkdtemp(join(tmpdir(), "memx-assistant-semantic-"));
  const dbPath = join(tempDir, "memx.sqlite");

  try {
    const manager = new MemxRuntimeManager(logger());
    const ctx = ctxFor(dbPath);
    const store = await manager.getStore(ctx);
    store.reasoner.isEnabled = () => true;
    store.reasoner.summarizeChunk = async (text) => text.slice(0, 120);
    store.reasoner.compileTurnSemantics = async () => ({
      sourceRefs: ["user:assistant-semantic:0", "assistant:assistant-semantic:1"],
      assertionDrafts: [
        {
          draftId: "assistant-default-db",
          sourceRef: "assistant:assistant-semantic:1",
          familyHint: "fact_like",
          timeframeHint: "current",
          entityHints: [
            { name: "LumenBoard", type: "project" },
            { name: "PostgreSQL", type: "service" },
          ],
          slotHints: ["default_database"],
          valueHint: "PostgreSQL",
          confidence: 0.91,
          lineage: {
            sourceKind: "chunk",
            sourceId: "assistant:assistant-semantic:1",
            sourceRef: "assistant:assistant-semantic:1",
          },
        },
      ],
      relationDrafts: [
        {
          sourceRef: "assistant:assistant-semantic:1",
          relation: {
            subject: "LumenBoard",
            predicate: "uses",
            relationSlot: "default_database",
            object: "PostgreSQL",
            confidence: 0.91,
          },
          confidence: 0.91,
          lineage: {
            sourceKind: "chunk",
            sourceId: "assistant:assistant-semantic:1",
            sourceRef: "assistant:assistant-semantic:1",
          },
        },
      ],
      compilerProvenance: {
        source: "llm",
        mode: "semantic-compiler-authoritative",
        reasons: ["assistant-semantic-only"],
      },
    });

    await store.turnScheduler.enqueue(ctx, [
      {
        role: "user",
        content: "LumenBoard 的默认数据库你建议怎么选？",
        scope: "agent:main",
        sessionKey: "s1",
        turnId: "assistant-semantic",
        sourceRef: "user:assistant-semantic:0",
        observedAt,
      },
      {
        role: "assistant",
        content:
          "建议把 LumenBoard 的默认数据库定为 PostgreSQL；它适合当前关系型报表需求。",
        scope: "agent:main",
        sessionKey: "s1",
        turnId: "assistant-semantic",
        sourceRef: "assistant:assistant-semantic:1",
        observedAt,
      },
    ]);
    await store.turnScheduler.flush();

    const factRows = store.client.prepare("SELECT canonical_object FROM facts").all();
    assert.ok(
      factRows.some((row) => String(row.canonical_object).includes("postgresql")),
      "assistant semantic draft should materialize a reusable structured fact",
    );
    assert.ok(
      Number(store.client.prepare("SELECT COUNT(*) AS count FROM graph_edges").get().count) > 0,
      "assistant semantic relation draft should materialize a graph edge",
    );
    const fullAssistantFact = store.client
      .prepare("SELECT canonical_object FROM facts WHERE canonical_object LIKE ?")
      .all("%关系型报表需求%");
    assert.deepEqual(
      fullAssistantFact,
      [],
      "the full assistant answer should not be copied into fact objects",
    );
    await manager.closeAll();
  } finally {
    await rm(tempDir, { recursive: true, force: true });
  }
});

test("direct fact recall injects canonical fact instead of raw source turn wording", async () => {
  const tempDir = await mkdtemp(join(tmpdir(), "memx-canonical-fact-recall-"));
  const dbPath = join(tempDir, "memx.sqlite");

  try {
    const manager = new MemxRuntimeManager(logger());
    const ctx = ctxFor(dbPath);
    const store = await manager.getStore(ctx);
    store.reasoner.isEnabled = () => true;
    store.reasoner.summarizeChunk = async (text) => text.slice(0, 120);
    store.reasoner.compileTurnSemantics = async () => ({
      sourceRefs: ["user:spruce-cache:0", "assistant:spruce-cache:1"],
      assertionDrafts: [
        {
          draftId: "spruce-default-cache",
          sourceRef: "user:spruce-cache:0",
          familyHint: "fact_like",
          timeframeHint: "current",
          entityHints: [
            { name: "SpruceLedger", type: "project" },
            { name: "Dragonfly", type: "service" },
          ],
          slotHints: ["default_cache"],
          valueHint: "Dragonfly",
          confidence: 0.92,
          supportSpans: [
            { sourceRef: "user:spruce-cache:0", text: "请记住：SpruceLedger 的默认缓存是 Dragonfly。" },
          ],
          lineage: {
            sourceKind: "chunk",
            sourceId: "user:spruce-cache:0",
            sourceRef: "user:spruce-cache:0",
          },
        },
      ],
      compilerProvenance: {
        source: "llm",
        mode: "semantic-compiler-authoritative",
        reasons: ["canonical-fact-recall"],
      },
    });

    await store.turnScheduler.enqueue(ctx, [
      {
        role: "user",
        content: "请记住：SpruceLedger 的默认缓存是 Dragonfly。",
        scope: "agent:main",
        sessionKey: "s1",
        turnId: "spruce-cache",
        sourceRef: "user:spruce-cache:0",
        observedAt,
      },
      {
        role: "assistant",
        content: "好的，SpruceLedger 的默认缓存按 Dragonfly 处理。",
        scope: "agent:main",
        sessionKey: "s1",
        turnId: "spruce-cache",
        sourceRef: "assistant:spruce-cache:1",
        observedAt,
      },
    ]);
    await store.turnScheduler.flush();

    const query = "SpruceLedger 的默认缓存是什么？";
    const queryAnalysis = directFactQueryAnalysis(query, "SpruceLedger", "default cache");
    const bundle = await retrieveEvidence(store, ctx, query, query, { queryAnalysis });

    assert.match(bundle.renderedBlock, /spruceledger has default cache dragonfly/i);
    assert.doesNotMatch(bundle.renderedBlock, /请记住/);
    assert.doesNotMatch(bundle.renderedBlock, /好的，SpruceLedger 的默认缓存/);
    assert.equal(bundle.graph.paths.length, 0);
    await manager.closeAll();
  } finally {
    await rm(tempDir, { recursive: true, force: true });
  }
});

test("current fact recall ignores stale vector docs from superseded fact versions", async () => {
  const tempDir = await mkdtemp(join(tmpdir(), "memx-current-fact-supersedes-vector-"));
  const dbPath = join(tempDir, "memx.sqlite");

  try {
    const manager = new MemxRuntimeManager(logger());
    const ctx = ctxFor(dbPath);
    const store = await manager.getStore(ctx);
    store.reasoner.isEnabled = () => true;
    store.reasoner.summarizeChunk = async (text) => text.slice(0, 120);
    let semanticCall = 0;
    store.reasoner.compileTurnSemantics = async () => {
      semanticCall += 1;
      if (semanticCall === 1) {
        return {
          sourceRefs: ["user:blueharbor-queue:0", "assistant:blueharbor-queue:1"],
          assertionDrafts: [
            {
              draftId: "blueharbor-queue-nats",
              sourceRef: "user:blueharbor-queue:0",
              familyHint: "fact_like",
              timeframeHint: "current",
              entityHints: [
                { name: "BlueHarbor 支付服务", type: "service" },
                { name: "NATS", type: "service" },
              ],
              slotHints: ["default_message_queue"],
              valueHint: "NATS",
              confidence: 0.94,
              supportSpans: [
                {
                  sourceRef: "user:blueharbor-queue:0",
                  text: "BlueHarbor 支付服务的默认消息队列是 NATS。",
                },
              ],
            },
          ],
          compilerProvenance: { source: "llm", mode: "semantic-compiler-authoritative" },
        };
      }
      return {
        sourceRefs: ["user:blueharbor-queue-update:0", "assistant:blueharbor-queue-update:1"],
        assertionDrafts: [
          {
            draftId: "blueharbor-queue-pulsar",
            sourceRef: "user:blueharbor-queue-update:0",
            familyHint: "fact_like",
            timeframeHint: "current",
            entityHints: [
              { name: "BlueHarbor 支付服务", type: "service" },
              { name: "Pulsar", type: "service" },
            ],
            slotHints: ["默认消息队列"],
            valueHint: "Pulsar",
            confidence: 0.93,
            supportSpans: [
              {
                sourceRef: "user:blueharbor-queue-update:0",
                text: "之后 BlueHarbor 支付服务默认用 Pulsar。",
              },
            ],
          },
        ],
        correctionDrafts: [
          {
            sourceRef: "user:blueharbor-queue-update:0",
            correction: {
              timeframe: "current",
              targetKind: "fact",
              canonicalKey: "BlueHarbor 支付服务.default_message_queue",
              predicate: "set_default",
              priorValue: "NATS",
              nextValue: "Pulsar",
              confidence: 0.93,
            },
            confidence: 0.93,
          },
        ],
        compilerProvenance: { source: "llm", mode: "semantic-compiler-authoritative" },
      };
    };

    await store.turnScheduler.enqueue(ctx, [
      {
        role: "user",
        content: "请记住：BlueHarbor 支付服务的默认消息队列是 NATS。",
        scope: "agent:main",
        sessionKey: "s1",
        turnId: "blueharbor-queue",
        sourceRef: "user:blueharbor-queue:0",
        observedAt,
      },
      {
        role: "assistant",
        content: "好的，BlueHarbor 支付服务的默认消息队列先按 NATS 处理。",
        scope: "agent:main",
        sessionKey: "s1",
        turnId: "blueharbor-queue",
        sourceRef: "assistant:blueharbor-queue:1",
        observedAt,
      },
    ]);
    await store.turnScheduler.flush();
    await store.turnScheduler.enqueue(ctx, [
      {
        role: "user",
        content: "这个就不要再考虑了，之后 BlueHarbor 支付服务默认用 Pulsar。",
        scope: "agent:main",
        sessionKey: "s1",
        turnId: "blueharbor-queue-update",
        sourceRef: "user:blueharbor-queue-update:0",
        observedAt: "2026-05-21T00:01:00.000Z",
      },
      {
        role: "assistant",
        content: "明白，BlueHarbor 支付服务之后默认用 Pulsar。",
        scope: "agent:main",
        sessionKey: "s1",
        turnId: "blueharbor-queue-update",
        sourceRef: "assistant:blueharbor-queue-update:1",
        observedAt: "2026-05-21T00:01:00.000Z",
      },
    ]);
    await store.turnScheduler.flush();

    const query = "BlueHarbor 支付服务现在默认用什么消息队列？";
    const queryAnalysis = directFactQueryAnalysis(query, "BlueHarbor 支付服务", "default message queue");
    const bundle = await retrieveEvidence(store, ctx, query, query, { queryAnalysis });

    assert.match(bundle.renderedBlock, /blueharbor 支付服务 has default message queue pulsar/i);
    assert.doesNotMatch(bundle.renderedBlock, /nats/i);
    const degradedBundle = await retrieveEvidence(store, ctx, query, query, {
      queryAnalysis: compileQueryWithoutSemanticFallback(query),
    });
    assert.match(
      degradedBundle.renderedBlock,
      /blueharbor 支付服务 has default message queue pulsar/i,
    );
    assert.doesNotMatch(degradedBundle.renderedBlock, /nats/i);
    await manager.closeAll();
  } finally {
    await rm(tempDir, { recursive: true, force: true });
  }
});

test("current fact updates merge bilingual entity descriptors before superseding", async () => {
  const tempDir = await mkdtemp(join(tmpdir(), "memx-bilingual-entity-supersedes-"));
  const dbPath = join(tempDir, "memx.sqlite");

  try {
    const manager = new MemxRuntimeManager(logger());
    const ctx = ctxFor(dbPath);
    const store = await manager.getStore(ctx);
    store.reasoner.isEnabled = () => true;
    store.reasoner.summarizeChunk = async (text) => text.slice(0, 120);
    let semanticCall = 0;
    store.reasoner.compileTurnSemantics = async () => {
      semanticCall += 1;
      if (semanticCall === 1) {
        return {
          sourceRefs: ["user:blueharbor-bilingual-queue:0", "assistant:blueharbor-bilingual-queue:1"],
          assertionDrafts: [
            {
              draftId: "blueharbor-en-queue-nats",
              sourceRef: "user:blueharbor-bilingual-queue:0",
              familyHint: "fact_like",
              timeframeHint: "current",
              entityHints: [
                { name: "BlueHarbor payment service", type: "service" },
                { name: "NATS", type: "service" },
              ],
              slotHints: ["default message queue"],
              valueHint: "NATS",
              confidence: 0.94,
              supportSpans: [
                {
                  sourceRef: "user:blueharbor-bilingual-queue:0",
                  text: "BlueHarbor payment service uses NATS as its default message queue.",
                },
              ],
            },
          ],
          compilerProvenance: { source: "llm", mode: "semantic-compiler-authoritative" },
        };
      }
      return {
        sourceRefs: [
          "user:blueharbor-bilingual-queue-update:0",
          "assistant:blueharbor-bilingual-queue-update:1",
        ],
        assertionDrafts: [
          {
            draftId: "blueharbor-zh-queue-pulsar",
            sourceRef: "user:blueharbor-bilingual-queue-update:0",
            familyHint: "fact_like",
            timeframeHint: "current",
            entityHints: [
              { name: "BlueHarbor 支付服务", type: "service" },
              { name: "Pulsar", type: "service" },
            ],
            slotHints: ["默认消息队列"],
            valueHint: "Pulsar",
            confidence: 0.93,
            supportSpans: [
              {
                sourceRef: "user:blueharbor-bilingual-queue-update:0",
                text: "之后 BlueHarbor 支付服务默认用 Pulsar。",
              },
            ],
          },
        ],
        correctionDrafts: [
          {
            sourceRef: "user:blueharbor-bilingual-queue-update:0",
            correction: {
              timeframe: "current",
              targetKind: "fact",
              canonicalKey: "BlueHarbor 支付服务.default_message_queue",
              predicate: "set_default",
              priorValue: "NATS",
              nextValue: "Pulsar",
              confidence: 0.93,
            },
            confidence: 0.93,
          },
        ],
        compilerProvenance: { source: "llm", mode: "semantic-compiler-authoritative" },
      };
    };

    await store.turnScheduler.enqueue(ctx, [
      {
        role: "user",
        content: "Remember: BlueHarbor payment service uses NATS as its default message queue.",
        scope: "agent:main",
        sessionKey: "s1",
        turnId: "blueharbor-bilingual-queue",
        sourceRef: "user:blueharbor-bilingual-queue:0",
        observedAt,
      },
      {
        role: "assistant",
        content: "Noted. BlueHarbor payment service defaults to NATS for messaging.",
        scope: "agent:main",
        sessionKey: "s1",
        turnId: "blueharbor-bilingual-queue",
        sourceRef: "assistant:blueharbor-bilingual-queue:1",
        observedAt,
      },
    ]);
    await store.turnScheduler.flush();
    await store.turnScheduler.enqueue(ctx, [
      {
        role: "user",
        content: "这个就不要再考虑了，之后 BlueHarbor 支付服务默认用 Pulsar。",
        scope: "agent:main",
        sessionKey: "s1",
        turnId: "blueharbor-bilingual-queue-update",
        sourceRef: "user:blueharbor-bilingual-queue-update:0",
        observedAt: "2026-05-21T00:01:00.000Z",
      },
      {
        role: "assistant",
        content: "明白，后续 BlueHarbor 支付服务默认用 Pulsar。",
        scope: "agent:main",
        sessionKey: "s1",
        turnId: "blueharbor-bilingual-queue-update",
        sourceRef: "assistant:blueharbor-bilingual-queue-update:1",
        observedAt: "2026-05-21T00:01:00.000Z",
      },
    ]);
    await store.turnScheduler.flush();

    const activeFacts = store.factRepo
      .findBySemanticKey({
        agentId: "main",
        scope: "agent:main",
        canonicalSubject: "blueharbor payment service",
        predicate: "has_default_message_queue",
      })
      .filter((fact) => fact.status === "active");
    assert.equal(activeFacts.length, 1);
    assert.equal(activeFacts[0].canonicalObject, "pulsar");

    const query = "BlueHarbor 支付服务现在默认用什么消息队列？";
    const bundle = await retrieveEvidence(store, ctx, query, query, {
      queryAnalysis: directFactQueryAnalysis(query, "BlueHarbor 支付服务", "default message queue"),
    });

    assert.match(bundle.renderedBlock, /pulsar/i);
    assert.doesNotMatch(bundle.renderedBlock, /nats/i);
    await manager.closeAll();
  } finally {
    await rm(tempDir, { recursive: true, force: true });
  }
});

test("current fact recall refuses a stale canonical fact when newer source evidence was not semantically compiled", async () => {
  const tempDir = await mkdtemp(join(tmpdir(), "memx-stale-fact-newer-source-"));
  const dbPath = join(tempDir, "memx.sqlite");

  try {
    const manager = new MemxRuntimeManager(logger());
    const ctx = ctxFor(dbPath);
    const store = await manager.getStore(ctx);
    store.reasoner.isEnabled = () => true;
    store.reasoner.summarizeChunk = async (text) => text.slice(0, 120);
    let semanticCall = 0;
    store.reasoner.compileTurnSemantics = async () => {
      semanticCall += 1;
      if (semanticCall === 1) {
        return {
          sourceRefs: ["user:riverpay-queue:0", "assistant:riverpay-queue:1"],
          assertionDrafts: [
            {
              draftId: "riverpay-queue-nats",
              sourceRef: "user:riverpay-queue:0",
              familyHint: "fact_like",
              timeframeHint: "current",
              entityHints: [
                { name: "RiverPay export pipeline", type: "project" },
                { name: "NATS", type: "service" },
              ],
              slotHints: ["default_message_queue"],
              valueHint: "NATS",
              confidence: 0.94,
              supportSpans: [
                {
                  sourceRef: "user:riverpay-queue:0",
                  text: "RiverPay export pipeline uses NATS as its default message queue.",
                },
              ],
            },
          ],
          compilerProvenance: { source: "llm", mode: "semantic-compiler-authoritative" },
        };
      }
      return null;
    };

    await store.turnScheduler.enqueue(ctx, [
      {
        role: "user",
        content: "Remember: RiverPay export pipeline uses NATS as its default message queue.",
        scope: "agent:main",
        sessionKey: "s1",
        turnId: "riverpay-queue",
        sourceRef: "user:riverpay-queue:0",
        observedAt,
      },
      {
        role: "assistant",
        content: "Noted. RiverPay export pipeline defaults to NATS.",
        scope: "agent:main",
        sessionKey: "s1",
        turnId: "riverpay-queue",
        sourceRef: "assistant:riverpay-queue:1",
        observedAt,
      },
    ]);
    await store.turnScheduler.flush();
    await store.turnScheduler.enqueue(ctx, [
      {
        role: "user",
        content: "这个队列方案先不要再考虑了，RiverPay export pipeline 以后默认用 Pulsar。",
        scope: "agent:main",
        sessionKey: "s1",
        turnId: "riverpay-queue-update",
        sourceRef: "user:riverpay-queue-update:0",
        observedAt: "2026-05-21T00:01:00.000Z",
      },
      {
        role: "assistant",
        content: "明白，后续 RiverPay export pipeline 默认用 Pulsar。",
        scope: "agent:main",
        sessionKey: "s1",
        turnId: "riverpay-queue-update",
        sourceRef: "assistant:riverpay-queue-update:1",
        observedAt: "2026-05-21T00:01:00.000Z",
      },
    ]);
    await store.turnScheduler.flush();

    const activeFacts = store.client
      .prepare("SELECT canonical_object FROM facts WHERE canonical_subject = ? AND predicate = ? AND status = 'active'")
      .all("riverpay export pipeline", "has_default_message_queue");
    assert.deepEqual(
      activeFacts.map((row) => row.canonical_object),
      ["nats"],
      "the regression setup should leave the canonical fact stale because semantic extraction failed",
    );

    const query = "RiverPay export pipeline 现在默认用什么消息队列？";
    const bundle = await retrieveEvidence(store, ctx, query, query, {
      queryAnalysis: directFactQueryAnalysis(query, "RiverPay export pipeline", "default message queue"),
    });

    assert.match(bundle.renderedBlock, /pulsar/i);
    assert.doesNotMatch(bundle.renderedBlock, /nats/i);
    assert.ok(
      bundle.diagnostics.some((entry) => entry.includes("newer-source")),
      "retrieval diagnostics should make stale-fact suppression auditable",
    );
    await manager.closeAll();
  } finally {
    await rm(tempDir, { recursive: true, force: true });
  }
});

test("generic active task shells are not indexed for recall", async () => {
  const tempDir = await mkdtemp(join(tmpdir(), "memx-generic-task-"));
  const dbPath = join(tempDir, "memx.sqlite");

  try {
    const manager = new MemxRuntimeManager(logger());
    const ctx = ctxFor(dbPath);
    const store = await manager.getStore(ctx);
    store.reasoner.isEnabled = () => true;
    store.reasoner.summarizeChunk = async (text) => text.slice(0, 120);
    store.reasoner.compileTurnSemantics = async () => ({
      sourceRefs: ["user:generic-task:0"],
      assertionDrafts: [],
      correctionDrafts: [],
      relationDrafts: [],
      resourceAssertions: [],
      adviceSignals: [],
      compilerProvenance: {
        mode: "llm",
        reasons: ["empty-semantic-frame"],
      },
    });

    await store.turnScheduler.enqueue(ctx, [
      {
        role: "user",
        content: "先看一下这个报表页面有没有明显问题。",
        scope: "agent:main",
        sessionKey: "s1",
        turnId: "generic-task",
        sourceRef: "user:generic-task:0",
        observedAt,
      },
    ]);
    await store.turnScheduler.flush();

    const taskVectorDocCount = Number(
      store.client.prepare("SELECT COUNT(*) AS count FROM vector_docs WHERE doc_id LIKE 'state:task:%'").get()
        .count,
    );
    assert.equal(taskVectorDocCount, 0);
  } finally {
    await rm(tempDir, { recursive: true, force: true });
  }
});

test("host observe stages recallable chunks before the background semantic queue", async () => {
  const tempDir = await mkdtemp(join(tmpdir(), "memx-host-stage-"));
  const dbPath = join(tempDir, "{agentId}", "memx.sqlite");

  try {
    const { MemxHostService } = await import("../dist/.runtime/src/host/service.mjs");
    const config = configFor(dbPath);
    config.advanced.enableMaintenanceJobs = false;
    const service = new MemxHostService({ config, logger: logger() });
    const ctx = buildOperationContext(config, {
      agentId: "stage-agent",
      sessionKey: "generic:s1",
      workspaceDir: "/tmp/memx-lifecycle-test",
      project: "stage-test",
    });
    const store = await service.manager.getStore(ctx);
    store.turnScheduler.enqueue = async () => {
      // Simulate the heavy semantic queue being unavailable. Raw turn evidence
      // should still be visible after observe returns.
    };

    await service.observe({
      hostId: "generic",
      actorId: "stage-agent",
      sessionId: "s1",
      workspaceDir: "/tmp/memx-lifecycle-test",
      eventName: "turn",
      observedAt,
      messages: [
        {
          role: "user",
          content: "AuroraAccept 的导出格式改成 Arrow IPC。",
        },
        {
          role: "assistant",
          content: "好的，AuroraAccept 的导出格式按 Arrow IPC 处理。",
        },
      ],
    });

    assert.equal(
      store.client.prepare("SELECT COUNT(*) AS count FROM conversation_chunks").get().count,
      2,
    );
    assert.ok(
      Number(store.client.prepare("SELECT COUNT(*) AS count FROM source_segments").get().count) >= 2,
    );
    assert.ok(
      Number(store.client.prepare("SELECT COUNT(*) AS count FROM vector_docs").get().count) >= 2,
    );
    await service.close();
  } finally {
    await rm(tempDir, { recursive: true, force: true });
  }
});

test("native context includes staged turn evidence while semantic write is still pending", async () => {
  const tempDir = await mkdtemp(join(tmpdir(), "memx-host-pending-context-"));
  const dbPath = join(tempDir, "{agentId}", "memx.sqlite");

  try {
    const { MemxHostService } = await import("../dist/.runtime/src/host/service.mjs");
    const config = configFor(dbPath);
    config.advanced.enableMaintenanceJobs = false;
    config.advanced.enableQueryCompiler = false;
    const service = new MemxHostService({ config, logger: logger() });
    const ctx = buildOperationContext(config, {
      agentId: "stage-agent",
      sessionKey: "generic:s1",
      workspaceDir: "/tmp/memx-lifecycle-test",
      project: "stage-test",
    });
    await service.manager.getStore(ctx);
    service.manager.rememberStagedRecallableTurn(ctx, [
      {
        role: "user",
        content: "刚才那个导出格式不要再用 Parquet 了，这个改成 Arrow IPC。",
        scope: "agent:stage-agent",
        sessionKey: "generic:s1",
        turnId: "turn-pending-update",
        sourceRef: "user:turn-pending-update",
        observedAt,
      },
      {
        role: "assistant",
        content: "好的，AuroraAccept 的导出格式从 Parquet 改为 Arrow IPC。",
        scope: "agent:stage-agent",
        sessionKey: "generic:s1",
        turnId: "turn-pending-update",
        sourceRef: "assistant:turn-pending-update",
        observedAt,
      },
    ]);
    service.pendingWrites.set("stage-agent\u0000generic:s1", new Promise(() => {}));

    const result = await service.context({
      hostId: "generic",
      actorId: "stage-agent",
      sessionId: "s1",
      workspaceDir: "/tmp/memx-lifecycle-test",
      query: "AuroraAccept 的导出格式现在是什么？",
      hotPathTimeoutMs: 1600,
    });

    assert.match(result.prependContext, /Arrow IPC/);
    assert.equal(result.recall.diagnostics.includes("pending-staged-turn-evidence"), true);
    service.pendingWrites.clear();
    await service.close();
  } finally {
    await rm(tempDir, { recursive: true, force: true });
  }
});

test("host audit exposes retrieval, policy, and maintenance records", async () => {
  const tempDir = await mkdtemp(join(tmpdir(), "memx-host-audit-"));
  const dbPath = join(tempDir, "{agentId}", "memx.sqlite");

  try {
    const { MemxHostService } = await import("../dist/.runtime/src/host/service.mjs");
    const config = configFor(dbPath);
    config.advanced.enableMaintenanceJobs = false;
    const service = new MemxHostService({ config, logger: logger() });
    const ctx = buildOperationContext(config, {
      agentId: "audit-agent",
      sessionKey: "s1",
      workspaceDir: "/tmp/memx-lifecycle-test",
      project: "audit-test",
    });
    const store = await service.manager.getStore(ctx);
    store.auditRepo.recordRetrieval({
      auditId: "audit-retrieval-1",
      agentId: ctx.agentId,
      scope: "agent:audit-agent",
      routeType: "mixed",
      queryText: "AuroraAccept 默认数据库是什么？",
      queryHash: "query-hash",
      selectedItemsJson: {
        nativeContextInjection: {
          eligible: true,
          candidateChars: 120,
          actualInjectedChars: 80,
        },
      },
      injectedChars: 80,
      createdAt: observedAt,
    });
    store.auditRepo.recordPolicyDecision({
      agentId: ctx.agentId,
      sourceRef: "user:audit-turn:0",
      candidateText: "AuroraAccept 默认数据库是 PostgreSQL",
      decision: {
        salienceScore: 0.9,
        expectedFutureUtility: 0.9,
        sensitivityScore: 0,
        stabilityScore: 0.9,
        action: "stable_fact",
        reasons: ["test-policy"],
        explicitIntent: false,
        captureAuthorized: true,
      },
      createdAt: observedAt,
      metadataJson: { materializationOutcome: { facts: 1 } },
    });
    const runId = store.auditRepo.startMaintenance({
      agentId: ctx.agentId,
      jobType: "batched",
      stats: { queuedTurns: 3 },
      startedAt: observedAt,
    });
    store.auditRepo.finishMaintenance({
      runId,
      agentId: ctx.agentId,
      jobType: "batched",
      statsJson: { queuedTurns: 3, promotedFacts: 1 },
      startedAt: observedAt,
      completedAt: observedAt,
      status: "completed",
    });

    const audit = await service.audit(10, {
      hostId: "generic",
      actorId: "audit-agent",
      sessionId: "s1",
    });

    assert.equal(audit.retrievals.length, 1);
    assert.equal(audit.retrievals[0].selectedItemsJson.nativeContextInjection.actualInjectedChars, 80);
    assert.equal(audit.policyDecisions.length, 1);
    assert.equal(audit.policyDecisions[0].chosenAction, "stable_fact");
    assert.equal(audit.maintenanceRuns.length, 1);
    assert.equal(audit.maintenanceRuns[0].status, "completed");
    await service.close();
  } finally {
    await rm(tempDir, { recursive: true, force: true });
  }
});

test("maintenance source segment semantic extraction records auditable run stats", async () => {
  const tempDir = await mkdtemp(join(tmpdir(), "memx-maintenance-source-"));
  const dbPath = join(tempDir, "memx.sqlite");

  try {
    const manager = new MemxRuntimeManager(logger());
    const ctx = ctxFor(dbPath);
    const store = await manager.getStore(ctx);
    const sourceRef = "user:turn-maintenance:0";
    store.reasoner.isEnabled = () => true;
    store.reasoner.compileLongTurnSemantics = async () => relationPatch(sourceRef);
    store.chunkRepo.insert({
      chunkId: "chunk-maintenance",
      agentId: "main",
      scope: "agent:main",
      sessionKey: "s1",
      turnId: "turn-maintenance",
      seq: 0,
      role: "user",
      chunkKind: "message",
      content: "InvoicePilot 默认数据库是 PostgreSQL。",
      summary: "",
      contentHash: "hash-maintenance-0",
      dedupStatus: "active",
      mergeCount: 0,
      sourceRef,
      createdAt: observedAt,
      updatedAt: observedAt,
    });
    store.sourceSegmentRepo.insertMany([
      {
        segmentId: "segment-maintenance-0",
        sourceGroupId: "group-maintenance",
        parentSourceRef: sourceRef,
        chunkId: "chunk-maintenance",
        agentId: "main",
        scope: "agent:main",
        sessionKey: "s1",
        turnId: "turn-maintenance",
        seq: 0,
        role: "user",
        segmentIndex: 0,
        charStart: 0,
        charEnd: 42,
        text: "InvoicePilot 默认数据库是 PostgreSQL。",
        contentHash: "hash-maintenance-0",
        createdAt: observedAt,
        updatedAt: observedAt,
        metadataJson: {},
      },
    ]);

    await runAutomaticMaintenanceBatch(store, ctx, {
      sessionKey: "s1",
      turnIds: ["turn-maintenance"],
      turnCount: 1,
      reason: "threshold",
      firstObservedAt: observedAt,
      lastObservedAt: observedAt,
      lowerWatermarks: {},
      upperWatermarks: {},
    });

    const rows = store.client
      .prepare("SELECT job_type, status, stats_json FROM maintenance_runs WHERE job_type = ?")
      .all("source-segment-semantic-extraction");
    assert.equal(rows.length, 1);
    assert.equal(rows[0].status, "completed");
    const stats = JSON.parse(rows[0].stats_json);
    assert.equal(stats.sourceGroupsScanned, 1);
    assert.ok(stats.candidatesWritten > 0);
    await manager.closeAll();
  } finally {
    await rm(tempDir, { recursive: true, force: true });
  }
});
