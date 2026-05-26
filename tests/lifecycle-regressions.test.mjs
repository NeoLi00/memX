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
  deriveNativeHookBudget,
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
  const budget = deriveNativeHookBudget(MEMX_NATIVE_HOOK_TIMEOUT_MS);

  assert.equal(MEMX_NATIVE_HOOK_TIMEOUT_MS, 8000);
  assert.deepEqual(budget, {
    hookTimeoutMs: 8000,
    contextTimeoutMs: 7750,
    observeTimeoutMs: 7750,
    queryCompilerTimeoutMs: 5000,
  });
  assert.ok(
    budget.queryCompilerTimeoutMs <= budget.contextTimeoutMs - 1000,
    "the query compiler must leave enough of the 8s hook budget for retrieval and HTTP return",
  );
  assert.ok(budget.queryCompilerTimeoutMs >= 4500);
  assert.ok(budget.queryCompilerTimeoutMs < budget.contextTimeoutMs);
});

test("query compiler carries LLM extracted entity aliases into evidence subject hints", async () => {
  const { compileQuery } = await import("../dist/.runtime/src/pipeline/queryCompiler.mjs");
  const tempDir = await mkdtemp(join(tmpdir(), "memx-query-entity-hints-"));
  const query = "FrostBridge/霜桥同步 的 default queue、API timeout 和导出格式分别是什么？";
  try {
    const ctx = ctxFor(join(tempDir, "memx.sqlite"));
    const compiled = await compileQuery({
      query,
      ctx,
      reasoner: {
        isEnabled: () => true,
        compileQuerySemantics: async () => ({
          focusedQuery: query,
          queryEntities: [{ name: "FrostBridge/霜桥同步", type: "project", role: "subject" }],
          queryShape: {
            timeframe: "timeless",
            granularity: "exact_detail",
            referentialMode: "anchored",
            evidenceNeed: "canonical_state",
          },
          primaryRoute: "factual",
        }),
      },
      hotPathTimeoutMs: 5500,
    });

    assert.ok(compiled.candidateSurfaces.includes("fact"));
    const subjectHints = [
      ...new Set(
        (compiled.evidencePlan?.slots ?? []).flatMap((slot) => slot.subjectHints ?? []),
      ),
    ];
    assert.ok(subjectHints.includes("FrostBridge/霜桥同步"));
    assert.ok(subjectHints.includes("FrostBridge"));
    assert.ok(subjectHints.includes("霜桥同步"));
    assert.equal(subjectHints.includes(query), false);
  } finally {
    await rm(tempDir, { recursive: true, force: true });
  }
});

test("fallback query compiler builds bilingual multi-slot attribute contracts", () => {
  const cases = [
    {
      query: "PineFlowMature 现在默认用什么消息队列？归档格式是什么？",
      subject: "PineFlowMature",
    },
    {
      query: "What are PineFlowMature's default message queue and archive format?",
      subject: "PineFlowMature",
    },
  ];

  for (const { query, subject } of cases) {
    const compiled = compileQueryWithoutSemanticFallback(query, "query-compile-llm-timeout");
    const answerSlots = (compiled.evidencePlan?.slots ?? []).filter(
      (slot) => slot.requiredRole === "answer_value",
    );
    const requestedSlots = [
      ...new Set(answerSlots.flatMap((slot) => slot.requestedAttributeSlots ?? [])),
    ].sort();
    const subjectHints = [
      ...new Set((compiled.evidencePlan?.slots ?? []).flatMap((slot) => slot.subjectHints ?? [])),
    ];

    assert.equal(compiled.queryShape.timeframe, "current");
    assert.equal(compiled.queryShape.evidenceNeed, "canonical_state");
    assert.equal(compiled.answerMode, "multi_evidence");
    assert.deepEqual(requestedSlots, ["archive_format", "default_message_queue"]);
    assert.ok(subjectHints.includes(subject));
    assert.equal(subjectHints.some((hint) => hint.includes("什么消息队列") || hint.includes("archive format?")), false);
  }
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

test("degraded native recall filters similarly prefixed code-like entity anchors", async () => {
  const {
    focusRecallBundleForDegradedQueryAnchors,
    formatNativeRecallContext,
  } = await import("../dist/.runtime/src/host/service.mjs");
  const query = "MatureProbeClean1779799273 默认告警通道是什么？";
  const queryAnalysis = compileQueryWithoutSemanticFallback(query);
  const packet = (packetId, primaryText) => ({
    packetId,
    slotId: "answer_value",
    operationType: "return_value",
    role: "answer",
    protected: true,
    injected: true,
    layers: ["fact"],
    primaryText,
    supportingTexts: [],
    sourceRefs: [`fact:${packetId}`],
    supportSourceRefs: [],
    allSourceRefs: [`fact:${packetId}`],
    normalizedSourceRefs: [`fact:${packetId}`],
    normalizedSupportSourceRefs: [],
    normalizedAllSourceRefs: [`fact:${packetId}`],
    score: 0.91,
    scoreBreakdown: { finalScore: 0.91 },
    displayLines: [`[answer] ${primaryText}`],
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
    selectionReason: "test-code-like-anchor",
  });
  const bundle = {
    routeType: "factual",
    routeConfidence: 0.91,
    queryText: query,
    queryAnchors: [],
    states: [],
    tasks: [],
    facts: [],
    events: [],
    graph: { nodes: [], edges: [], paths: [], pathCandidates: [] },
    alternates: [],
    diagnostics: [],
    behavioralGuidance: [],
    recalledChunkIds: [],
    recalledChunkTexts: [],
    promptEvidence: [],
    evidencePackets: [
      packet("old-prefixed", "matureprobe1779799145 has alert channel relaynine"),
      packet(
        "current-exact",
        "[user] 工程记录：MatureProbeClean1779799273 的默认告警通道是 RelayNine。",
      ),
    ],
    renderedBlock: "",
  };

  const focused = focusRecallBundleForDegradedQueryAnchors(query, queryAnalysis, bundle);
  const context = formatNativeRecallContext(focused, 4000);

  assert.doesNotMatch(context, /matureprobe1779799145/i);
  assert.match(context, /MatureProbeClean1779799273/);
  assert.equal(focused.diagnostics.includes("degraded-hard-anchor-focused"), true);
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
    const semanticPolicyMetadata = store.client
      .prepare("SELECT metadata_json FROM policy_decisions WHERE metadata_json LIKE ?")
      .all("%semanticDraft%")
      .map((row) => JSON.parse(row.metadata_json));
    assert.ok(semanticPolicyMetadata.length > 0);
    assert.ok(
      semanticPolicyMetadata.some((metadata) => metadata.semanticSource === "llm"),
      "policy audit should expose that the semantic draft came from the LLM compiler",
    );
    assert.equal(
      semanticPolicyMetadata.some((metadata) => metadata.decisionSource === "deterministic"),
      false,
      "LLM semantic draft decisions should not be labeled as deterministic",
    );
    await manager.closeAll();
  } finally {
    await rm(tempDir, { recursive: true, force: true });
  }
});

test("assistant-only LLM fact-like drafts are not materialized as user facts or graph edges", async () => {
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

    const factRows = store.client.prepare("SELECT canonical_subject, predicate, canonical_object FROM facts").all();
    assert.equal(
      factRows.some(
        (row) =>
          row.canonical_subject === "lumenboard" &&
          row.predicate === "has_default_database" &&
          String(row.canonical_object).includes("postgresql"),
      ),
      false,
      "assistant-only recommendations must not become canonical user facts",
    );
    assert.equal(
      Number(store.client.prepare("SELECT COUNT(*) AS count FROM graph_edges").get().count),
      0,
      "assistant-only fact-like relation drafts must not become graph edges",
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

test("assistant recall answers do not supersede grounded user facts when they conflict", async () => {
  const tempDir = await mkdtemp(join(tmpdir(), "memx-assistant-conflict-"));
  const dbPath = join(tempDir, "memx.sqlite");
  const later = "2026-05-21T00:01:00.000Z";

  try {
    const manager = new MemxRuntimeManager(logger());
    const ctx = ctxFor(dbPath);
    const store = await manager.getStore(ctx);
    store.reasoner.isEnabled = () => true;
    store.reasoner.summarizeChunk = async (text) => text.slice(0, 120);
    let compileCall = 0;
    store.reasoner.compileTurnSemantics = async () => {
      compileCall += 1;
      if (compileCall === 1) {
        return {
          sourceRefs: ["user:yunsan-queue:0", "assistant:yunsan-queue:1"],
          assertionDrafts: [
            {
              draftId: "user-yunsan-queue",
              sourceRef: "user:yunsan-queue:0",
              familyHint: "fact_like",
              timeframeHint: "current",
              entityHints: [
                { name: "云杉票据", type: "project" },
                { name: "Pulsar", type: "service" },
              ],
              slotHints: ["default_message_queue"],
              valueHint: "Pulsar",
              confidence: 1,
            },
          ],
          relationDrafts: [],
          compilerProvenance: { source: "llm", mode: "llm" },
        };
      }
      return {
        sourceRefs: ["user:yunsan-recall:0", "assistant:yunsan-recall:1"],
        assertionDrafts: [
          {
            draftId: "assistant-yunsan-queue-wrong",
            sourceRef: "assistant:yunsan-recall:1",
            familyHint: "fact_like",
            timeframeHint: "current",
            entityHints: [
              { name: "云杉票据", type: "project" },
              { name: "RabbitMQ", type: "service" },
            ],
            slotHints: ["default_message_queue"],
            valueHint: "RabbitMQ",
            confidence: 1,
          },
        ],
        relationDrafts: [],
        compilerProvenance: { source: "llm", mode: "llm" },
      };
    };

    await store.turnScheduler.enqueue(ctx, [
      {
        role: "user",
        content: "请记住：云杉票据的默认消息队列是 Pulsar。",
        scope: "agent:main",
        sessionKey: "s1",
        turnId: "yunsan-queue",
        sourceRef: "user:yunsan-queue:0",
        observedAt,
      },
      {
        role: "assistant",
        content: "已记录。",
        scope: "agent:main",
        sessionKey: "s1",
        turnId: "yunsan-queue",
        sourceRef: "assistant:yunsan-queue:1",
        observedAt,
      },
    ]);
    await store.turnScheduler.flush();

    await store.turnScheduler.enqueue(ctx, [
      {
        role: "user",
        content: "云杉票据的默认消息队列是什么？",
        scope: "agent:main",
        sessionKey: "s1",
        turnId: "yunsan-recall",
        sourceRef: "user:yunsan-recall:0",
        observedAt: later,
      },
      {
        role: "assistant",
        content: "云杉票据的默认消息队列是 RabbitMQ。",
        scope: "agent:main",
        sessionKey: "s1",
        turnId: "yunsan-recall",
        sourceRef: "assistant:yunsan-recall:1",
        observedAt: later,
      },
    ]);
    await store.turnScheduler.flush();

    const activeQueues = store.client
      .prepare(
        "SELECT canonical_object FROM facts WHERE canonical_subject = ? AND predicate = ? AND status = 'active'",
      )
      .all("云杉票据", "has_default_message_queue")
      .map((row) => row.canonical_object);

    assert.deepEqual(activeQueues, ["pulsar"]);
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
    assert.ok(
      bundle.graph.paths.every((path) => JSON.stringify(path).toLowerCase().includes("dragonfly")),
    );
    await manager.closeAll();
  } finally {
    await rm(tempDir, { recursive: true, force: true });
  }
});

test("fallback multi-attribute recall injects canonical facts instead of raw turn wording", async () => {
  const tempDir = await mkdtemp(join(tmpdir(), "memx-fallback-multi-attribute-recall-"));
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
          sourceRefs: ["user:pineflow-mature:0", "assistant:pineflow-mature:1"],
          assertionDrafts: [
            {
              draftId: "pineflow-mature-queue-nats",
              sourceRef: "user:pineflow-mature:0",
              familyHint: "fact_like",
              timeframeHint: "current",
              entityHints: [
                { name: "PineFlowMature", type: "project" },
                { name: "NATS", type: "service" },
              ],
              slotHints: ["default_message_queue"],
              valueHint: "NATS",
              confidence: 0.93,
            },
            {
              draftId: "pineflow-mature-archive",
              sourceRef: "user:pineflow-mature:0",
              familyHint: "fact_like",
              timeframeHint: "current",
              entityHints: [
                { name: "PineFlowMature", type: "project" },
                { name: "Parquet", type: "concept" },
              ],
              slotHints: ["archive_format"],
              valueHint: "Parquet",
              confidence: 0.93,
            },
          ],
          relationDrafts: [],
          compilerProvenance: { source: "llm", mode: "semantic-compiler-authoritative" },
        };
      }
      return {
        sourceRefs: ["user:pineflow-mature-update:0", "assistant:pineflow-mature-update:1"],
        assertionDrafts: [],
        correctionDrafts: [
          {
            sourceRef: "user:pineflow-mature-update:0",
            correction: {
              timeframe: "current",
              targetKind: "fact",
              canonicalKey: "PineFlowMature.default_message_queue",
              predicate: "has_default_message_queue",
              priorValue: "NATS",
              nextValue: "Pulsar",
              confidence: 0.94,
            },
            confidence: 0.94,
          },
        ],
        relationDrafts: [],
        compilerProvenance: { source: "llm", mode: "semantic-compiler-authoritative" },
      };
    };

    await store.turnScheduler.enqueue(ctx, [
      {
        role: "user",
        content: "请记住：PineFlowMature 默认消息队列是 NATS，归档格式是 Parquet。",
        scope: "agent:main",
        sessionKey: "s1",
        turnId: "pineflow-mature",
        sourceRef: "user:pineflow-mature:0",
        observedAt,
      },
      {
        role: "assistant",
        content: "已记录。",
        scope: "agent:main",
        sessionKey: "s1",
        turnId: "pineflow-mature",
        sourceRef: "assistant:pineflow-mature:1",
        observedAt,
      },
    ]);
    await store.turnScheduler.flush();
    await store.turnScheduler.enqueue(ctx, [
      {
        role: "user",
        content: "PineFlowMature 的队列选择不要再沿用，后面默认改成 Pulsar。归档格式保持不变。",
        scope: "agent:main",
        sessionKey: "s1",
        turnId: "pineflow-mature-update",
        sourceRef: "user:pineflow-mature-update:0",
        observedAt: "2026-05-21T00:01:00.000Z",
      },
      {
        role: "assistant",
        content: "明白，PineFlowMature 默认消息队列改成 Pulsar，归档格式仍然是 Parquet。",
        scope: "agent:main",
        sessionKey: "s1",
        turnId: "pineflow-mature-update",
        sourceRef: "assistant:pineflow-mature-update:1",
        observedAt: "2026-05-21T00:01:00.000Z",
      },
    ]);
    await store.turnScheduler.flush();

    const query = "PineFlowMature 现在默认用什么消息队列？归档格式是什么？";
    const queryAnalysis = compileQueryWithoutSemanticFallback(query, "query-compile-llm-timeout");
    const bundle = await retrieveEvidence(store, ctx, query, query, { queryAnalysis });

    assert.match(bundle.renderedBlock, /pineflowmature has default message queue pulsar/i);
    assert.match(bundle.renderedBlock, /pineflowmature has archive format parquet/i);
    assert.doesNotMatch(bundle.renderedBlock, /nats/i);
    assert.doesNotMatch(bundle.renderedBlock, /请记住/);
    assert.doesNotMatch(bundle.renderedBlock, /已记录/);
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

test("deictic attribute updates resolve generic project subjects through active project focus", async () => {
  const tempDir = await mkdtemp(join(tmpdir(), "memx-deictic-project-focus-"));
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
          sourceRefs: ["user:auroraflow-queue:0", "assistant:auroraflow-queue:1"],
          assertionDrafts: [
            {
              draftId: "auroraflow-queue-nats",
              sourceRef: "user:auroraflow-queue:0",
              familyHint: "fact_like",
              timeframeHint: "current",
              entityHints: [
                { name: "AuroraFlow", type: "project" },
                { name: "NATS", type: "service" },
              ],
              slotHints: ["default_message_queue"],
              valueHint: "NATS",
              confidence: 0.94,
            },
          ],
          relationDrafts: [
            {
              sourceRef: "user:auroraflow-queue:0",
              relation: {
                subject: "AuroraFlow",
                predicate: "uses",
                relationSlot: "default_message_queue",
                object: "NATS",
                confidence: 0.94,
              },
              confidence: 0.94,
            },
          ],
          compilerProvenance: { source: "llm", mode: "semantic-compiler-authoritative" },
        };
      }
      return {
        sourceRefs: ["user:auroraflow-queue-update:0", "assistant:auroraflow-queue-update:1"],
        assertionDrafts: [
          {
            draftId: "generic-project-queue-kafka",
            sourceRef: "user:auroraflow-queue-update:0",
            familyHint: "fact_like",
            timeframeHint: "current",
            entityHints: [
              { name: "project", type: "project" },
              { name: "Kafka", type: "service" },
            ],
            slotHints: ["default_message_queue"],
            valueHint: "Kafka",
            confidence: 0.92,
          },
        ],
        correctionDrafts: [
          {
            sourceRef: "user:auroraflow-queue-update:0",
            correction: {
              timeframe: "current",
              targetKind: "fact",
              canonicalKey: "project.default_message_queue",
              predicate: "has_default_message_queue",
              priorValue: "NATS",
              nextValue: "Kafka",
              confidence: 0.92,
            },
            confidence: 0.92,
          },
        ],
        compilerProvenance: { source: "llm", mode: "semantic-compiler-authoritative" },
      };
    };

    await store.turnScheduler.enqueue(ctx, [
      {
        role: "user",
        content: "请记住：AuroraFlow 的默认消息队列是 NATS。",
        scope: "agent:main",
        sessionKey: "s1",
        turnId: "auroraflow-queue",
        sourceRef: "user:auroraflow-queue:0",
        observedAt,
      },
      {
        role: "assistant",
        content: "已记录，AuroraFlow 的默认消息队列是 NATS。",
        scope: "agent:main",
        sessionKey: "s1",
        turnId: "auroraflow-queue",
        sourceRef: "assistant:auroraflow-queue:1",
        observedAt,
      },
    ]);
    await store.turnScheduler.flush();

    await store.turnScheduler.enqueue(ctx, [
      {
        role: "user",
        content: "这个就不要再考虑了，以后该项目默认用 Kafka。",
        scope: "agent:main",
        sessionKey: "s1",
        turnId: "auroraflow-queue-update",
        sourceRef: "user:auroraflow-queue-update:0",
        observedAt: "2026-05-21T00:01:00.000Z",
      },
      {
        role: "assistant",
        content: "明白，以后该项目默认用 Kafka。",
        scope: "agent:main",
        sessionKey: "s1",
        turnId: "auroraflow-queue-update",
        sourceRef: "assistant:auroraflow-queue-update:1",
        observedAt: "2026-05-21T00:01:00.000Z",
      },
    ]);
    await store.turnScheduler.flush();

    const activeQueues = store.client
      .prepare(
        "SELECT canonical_subject, canonical_object FROM facts WHERE predicate = ? AND status = 'active' ORDER BY canonical_subject",
      )
      .all("has_default_message_queue");
    assert.deepEqual(activeQueues.map((row) => ({ ...row })), [
      { canonical_subject: "auroraflow", canonical_object: "kafka" },
    ]);

    const query = "AuroraFlow 现在默认用什么消息队列？";
    const bundle = await retrieveEvidence(store, ctx, query, query, {
      queryAnalysis: directFactQueryAnalysis(query, "AuroraFlow", "default message queue"),
    });
    assert.match(bundle.renderedBlock, /auroraflow has default message queue kafka/i);
    assert.doesNotMatch(bundle.renderedBlock, /nats/i);
    await manager.closeAll();
  } finally {
    await rm(tempDir, { recursive: true, force: true });
  }
});

test("LLM assertion-only attribute updates project matching graph slot edges", async () => {
  const tempDir = await mkdtemp(join(tmpdir(), "memx-assertion-attribute-edge-"));
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
          sourceRefs: ["user:fastbridge-queue:0", "assistant:fastbridge-queue:1"],
          assertionDrafts: [
            {
              draftId: "fastbridge-queue-nats",
              sourceRef: "user:fastbridge-queue:0",
              familyHint: "fact_like",
              timeframeHint: "current",
              entityHints: [
                { name: "FastBridge", type: "project" },
                { name: "NATS", type: "service" },
              ],
              slotHints: ["default_message_queue"],
              valueHint: "NATS",
              confidence: 0.93,
            },
          ],
          relationDrafts: [],
          compilerProvenance: { source: "llm", mode: "semantic-compiler-authoritative" },
        };
      }
      return {
        sourceRefs: ["user:fastbridge-queue-update:0", "assistant:fastbridge-queue-update:1"],
        assertionDrafts: [
          {
            draftId: "fastbridge-queue-pulsar",
            sourceRef: "user:fastbridge-queue-update:0",
            familyHint: "fact_like",
            timeframeHint: "current",
            entityHints: [
              { name: "FastBridge", type: "project" },
              { name: "Pulsar", type: "service" },
            ],
            slotHints: ["default_message_queue"],
            valueHint: "Pulsar",
            confidence: 0.94,
          },
        ],
        correctionDrafts: [
          {
            sourceRef: "user:fastbridge-queue-update:0",
            correction: {
              timeframe: "current",
              targetKind: "fact",
              canonicalKey: "FastBridge.default_message_queue",
              predicate: "has_default_message_queue",
              priorValue: "NATS",
              nextValue: "Pulsar",
              confidence: 0.94,
            },
            confidence: 0.94,
          },
        ],
        relationDrafts: [],
        compilerProvenance: { source: "llm", mode: "semantic-compiler-authoritative" },
      };
    };

    await store.turnScheduler.enqueue(ctx, [
      {
        role: "user",
        content: "请记住：FastBridge 的默认消息队列是 NATS。",
        scope: "agent:main",
        sessionKey: "s1",
        turnId: "fastbridge-queue",
        sourceRef: "user:fastbridge-queue:0",
        observedAt,
      },
      {
        role: "assistant",
        content: "已记录。",
        scope: "agent:main",
        sessionKey: "s1",
        turnId: "fastbridge-queue",
        sourceRef: "assistant:fastbridge-queue:1",
        observedAt,
      },
    ]);
    await store.turnScheduler.flush();

    await store.turnScheduler.enqueue(ctx, [
      {
        role: "user",
        content: "FastBridge 的默认消息队列改成 Pulsar。",
        scope: "agent:main",
        sessionKey: "s1",
        turnId: "fastbridge-queue-update",
        sourceRef: "user:fastbridge-queue-update:0",
        observedAt: "2026-05-21T00:01:00.000Z",
      },
      {
        role: "assistant",
        content: "已记录，FastBridge 默认消息队列改成 Pulsar。",
        scope: "agent:main",
        sessionKey: "s1",
        turnId: "fastbridge-queue-update",
        sourceRef: "assistant:fastbridge-queue-update:1",
        observedAt: "2026-05-21T00:01:00.000Z",
      },
    ]);
    await store.turnScheduler.flush();

    const activeQueues = store.client
      .prepare(
        "SELECT canonical_subject, canonical_object FROM facts WHERE predicate = ? AND status = 'active' ORDER BY canonical_subject",
      )
      .all("has_default_message_queue");
    assert.deepEqual(activeQueues.map((row) => ({ ...row })), [
      { canonical_subject: "fastbridge", canonical_object: "pulsar" },
    ]);

    const queueEdges = store.client
      .prepare(
        `SELECT s.normalized_name AS src, e.rel_type, e.relation_slot, d.normalized_name AS dst, e.valid_to
           FROM graph_edges e
           JOIN entities s ON s.entity_id = e.src_entity_id
           JOIN entities d ON d.entity_id = e.dst_entity_id
          WHERE s.normalized_name = ?
            AND e.rel_type = 'uses'
            AND e.relation_slot = ?
          ORDER BY d.normalized_name`,
      )
      .all("fastbridge", "default_message_queue")
      .map((row) => ({ ...row }));
    assert.deepEqual(queueEdges, [
      {
        src: "fastbridge",
        rel_type: "uses",
        relation_slot: "default_message_queue",
        dst: "nats",
        valid_to: "2026-05-21T00:01:00.000Z",
      },
      {
        src: "fastbridge",
        rel_type: "uses",
        relation_slot: "default_message_queue",
        dst: "pulsar",
        valid_to: null,
      },
    ]);
    await manager.closeAll();
  } finally {
    await rm(tempDir, { recursive: true, force: true });
  }
});

test("LLM correction-only attribute updates project matching graph slot edges", async () => {
  const tempDir = await mkdtemp(join(tmpdir(), "memx-correction-attribute-edge-"));
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
          sourceRefs: ["user:pinestream-queue:0", "assistant:pinestream-queue:1"],
          assertionDrafts: [
            {
              draftId: "pinestream-queue-nats",
              sourceRef: "user:pinestream-queue:0",
              familyHint: "fact_like",
              timeframeHint: "current",
              entityHints: [
                { name: "PineStream", type: "project" },
                { name: "NATS", type: "service" },
              ],
              slotHints: ["default_message_queue"],
              valueHint: "NATS",
              confidence: 0.93,
            },
          ],
          relationDrafts: [],
          compilerProvenance: { source: "llm", mode: "semantic-compiler-authoritative" },
        };
      }
      return {
        sourceRefs: ["user:pinestream-queue-update:0", "assistant:pinestream-queue-update:1"],
        assertionDrafts: [],
        correctionDrafts: [
          {
            sourceRef: "user:pinestream-queue-update:0",
            correction: {
              timeframe: "current",
              targetKind: "fact",
              canonicalKey: "PineStream.default_message_queue",
              predicate: "has_default_message_queue",
              priorValue: "NATS",
              nextValue: "Pulsar",
              confidence: 0.94,
            },
            confidence: 0.94,
          },
        ],
        relationDrafts: [],
        compilerProvenance: { source: "llm", mode: "semantic-compiler-authoritative" },
      };
    };

    await store.turnScheduler.enqueue(ctx, [
      {
        role: "user",
        content: "请记住：PineStream 的默认消息队列是 NATS。",
        scope: "agent:main",
        sessionKey: "s1",
        turnId: "pinestream-queue",
        sourceRef: "user:pinestream-queue:0",
        observedAt,
      },
      {
        role: "assistant",
        content: "已记录。",
        scope: "agent:main",
        sessionKey: "s1",
        turnId: "pinestream-queue",
        sourceRef: "assistant:pinestream-queue:1",
        observedAt,
      },
    ]);
    await store.turnScheduler.flush();

    await store.turnScheduler.enqueue(ctx, [
      {
        role: "user",
        content: "PineStream 的队列选择不要再沿用，后面默认改成 Pulsar。",
        scope: "agent:main",
        sessionKey: "s1",
        turnId: "pinestream-queue-update",
        sourceRef: "user:pinestream-queue-update:0",
        observedAt: "2026-05-21T00:01:00.000Z",
      },
      {
        role: "assistant",
        content: "明白，PineStream 默认消息队列改成 Pulsar。",
        scope: "agent:main",
        sessionKey: "s1",
        turnId: "pinestream-queue-update",
        sourceRef: "assistant:pinestream-queue-update:1",
        observedAt: "2026-05-21T00:01:00.000Z",
      },
    ]);
    await store.turnScheduler.flush();

    const activeQueues = store.client
      .prepare(
        "SELECT canonical_subject, canonical_object FROM facts WHERE predicate = ? AND status = 'active' ORDER BY canonical_subject",
      )
      .all("has_default_message_queue");
    assert.deepEqual(activeQueues.map((row) => ({ ...row })), [
      { canonical_subject: "pinestream", canonical_object: "pulsar" },
    ]);

    const queueEdges = store.client
      .prepare(
        `SELECT s.normalized_name AS src, e.rel_type, e.relation_slot, d.normalized_name AS dst, e.valid_to
           FROM graph_edges e
           JOIN entities s ON s.entity_id = e.src_entity_id
           JOIN entities d ON d.entity_id = e.dst_entity_id
          WHERE s.normalized_name = ?
            AND e.rel_type = 'uses'
            AND e.relation_slot = ?
          ORDER BY d.normalized_name`,
      )
      .all("pinestream", "default_message_queue")
      .map((row) => ({ ...row }));
    assert.deepEqual(queueEdges, [
      {
        src: "pinestream",
        rel_type: "uses",
        relation_slot: "default_message_queue",
        dst: "nats",
        valid_to: "2026-05-21T00:01:00.000Z",
      },
      {
        src: "pinestream",
        rel_type: "uses",
        relation_slot: "default_message_queue",
        dst: "pulsar",
        valid_to: null,
      },
    ]);
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
    store.reasoner.compileLongTurnSemantics = async () => null;

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

test("maintenance semantic extraction repairs a stale current fact after hot-path semantic compilation fails", async () => {
  const tempDir = await mkdtemp(join(tmpdir(), "memx-stale-fact-maintenance-repair-"));
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
          sourceRefs: ["user:riverpay-maintenance:0"],
          assertionDrafts: [
            {
              draftId: "riverpay-maintenance-nats",
              sourceRef: "user:riverpay-maintenance:0",
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
                  sourceRef: "user:riverpay-maintenance:0",
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
        turnId: "riverpay-maintenance",
        sourceRef: "user:riverpay-maintenance:0",
        observedAt,
      },
      {
        role: "assistant",
        content: "Noted. RiverPay export pipeline defaults to NATS.",
        scope: "agent:main",
        sessionKey: "s1",
        turnId: "riverpay-maintenance",
        sourceRef: "assistant:riverpay-maintenance:1",
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
        turnId: "riverpay-maintenance-update",
        sourceRef: "user:riverpay-maintenance-update:0",
        observedAt: "2026-05-21T00:01:00.000Z",
      },
      {
        role: "assistant",
        content: "明白，后续 RiverPay export pipeline 默认用 Pulsar。",
        scope: "agent:main",
        sessionKey: "s1",
        turnId: "riverpay-maintenance-update",
        sourceRef: "assistant:riverpay-maintenance-update:1",
        observedAt: "2026-05-21T00:01:00.000Z",
      },
    ]);
    await store.turnScheduler.flush();

    store.reasoner.compileLongTurnSemantics = async () => ({
      sourceRefs: ["user:riverpay-maintenance-update:0"],
      assertionDrafts: [
        {
          draftId: "riverpay-maintenance-pulsar",
          sourceRef: "user:riverpay-maintenance-update:0",
          familyHint: "fact_like",
          timeframeHint: "current",
          entityHints: [
            { name: "RiverPay export pipeline", type: "project" },
            { name: "Pulsar", type: "service" },
          ],
          slotHints: ["default_message_queue"],
          valueHint: "Pulsar",
          confidence: 0.94,
          supportSpans: [
            {
              sourceRef: "user:riverpay-maintenance-update:0",
              text: "RiverPay export pipeline 以后默认用 Pulsar。",
            },
          ],
        },
      ],
      compilerProvenance: { source: "llm", mode: "maintenance-semantic-repair" },
    });

    await runAutomaticMaintenanceBatch(store, ctx, {
      sessionKey: "s1",
      turnIds: ["riverpay-maintenance-update"],
      turnCount: 1,
      reason: "threshold",
      firstObservedAt: "2026-05-21T00:01:00.000Z",
      lastObservedAt: "2026-05-21T00:01:00.000Z",
      lowerWatermarks: {},
      upperWatermarks: {},
    });

    const activeFacts = store.client
      .prepare("SELECT canonical_object FROM facts WHERE canonical_subject = ? AND predicate = ? AND status = 'active'")
      .all("riverpay export pipeline", "has_default_message_queue");
    assert.deepEqual(activeFacts.map((row) => row.canonical_object), ["pulsar"]);
    await manager.closeAll();
  } finally {
    await rm(tempDir, { recursive: true, force: true });
  }
});

test("semantic write ledger records hot-path LLM extraction failure and maintenance repair", async () => {
  const tempDir = await mkdtemp(join(tmpdir(), "memx-semantic-ledger-"));
  const dbPath = join(tempDir, "memx.sqlite");

  try {
    const manager = new MemxRuntimeManager(logger());
    const ctx = ctxFor(dbPath);
    const store = await manager.getStore(ctx);
    const sourceRef = "user:semantic-ledger:0";
    store.reasoner.isEnabled = () => true;
    store.reasoner.summarizeChunk = async (text) => text.slice(0, 120);
    store.reasoner.compileTurnSemantics = async () => null;

    await store.turnScheduler.enqueue(ctx, [
      {
        role: "user",
        content: "AtlasPipeline 默认数据库是 PostgreSQL，并且导出格式是 Parquet。",
        scope: "agent:main",
        sessionKey: "s1",
        turnId: "semantic-ledger",
        sourceRef,
        observedAt,
      },
      {
        role: "assistant",
        content: "明白，AtlasPipeline 使用 PostgreSQL，导出格式是 Parquet。",
        scope: "agent:main",
        sessionKey: "s1",
        turnId: "semantic-ledger",
        sourceRef: "assistant:semantic-ledger:1",
        observedAt,
      },
    ]);
    await store.turnScheduler.flush();

    let jobs = store.client
      .prepare(
        "SELECT turn_id, status, attempt_count, last_error, source_refs_json FROM semantic_write_jobs WHERE turn_id = ?",
      )
      .all("semantic-ledger");
    assert.equal(jobs.length, 1);
    assert.equal(jobs[0].status, "retrying");
    assert.equal(jobs[0].attempt_count, 1);
    assert.match(jobs[0].last_error, /turn semantic compiler returned no recognized LLM semantic frame/);
    assert.deepEqual(JSON.parse(jobs[0].source_refs_json), [sourceRef, "assistant:semantic-ledger:1"]);

    store.reasoner.compileLongTurnSemantics = async () => relationPatch(sourceRef);
    await runAutomaticMaintenanceBatch(store, ctx, {
      sessionKey: "s1",
      turnIds: ["semantic-ledger"],
      turnCount: 1,
      reason: "threshold",
      firstObservedAt: observedAt,
      lastObservedAt: observedAt,
      lowerWatermarks: {},
      upperWatermarks: {},
    });

    jobs = store.client
      .prepare(
        "SELECT turn_id, status, attempt_count, last_error FROM semantic_write_jobs WHERE turn_id = ?",
      )
      .all("semantic-ledger");
    assert.equal(jobs.length, 1);
    assert.equal(jobs[0].status, "succeeded");
    assert.equal(jobs[0].attempt_count, 2);
    assert.equal(jobs[0].last_error, null);

    const audit = await store.auditRepo.listSemanticWriteJobs({
      agentId: "main",
      sessionKey: "s1",
      limit: 10,
    });
    assert.equal(audit.length, 1);
    assert.equal(audit[0].turnId, "semantic-ledger");
    assert.equal(audit[0].status, "succeeded");
    await manager.closeAll();
  } finally {
    await rm(tempDir, { recursive: true, force: true });
  }
});

test("semantic write ledger does not re-claim succeeded jobs during maintenance repair", async () => {
  const tempDir = await mkdtemp(join(tmpdir(), "memx-semantic-ledger-succeeded-"));
  const dbPath = join(tempDir, "memx.sqlite");

  try {
    const manager = new MemxRuntimeManager(logger());
    const ctx = ctxFor(dbPath);
    const store = await manager.getStore(ctx);
    store.auditRepo.recordSemanticWriteAttemptStart({
      agentId: "main",
      sessionKey: "s1",
      scope: "agent:main",
      turnId: "semantic-succeeded",
      sourceRefs: ["user:semantic-succeeded:0"],
      inputHash: "input-v1",
      startedAt: observedAt,
    });
    store.auditRepo.finishSemanticWriteAttempt({
      agentId: "main",
      sessionKey: "s1",
      turnId: "semantic-succeeded",
      status: "succeeded",
      resultJson: { stage: "write_hot_path", assertionDrafts: 1 },
      completedAt: observedAt,
    });

    const attempt = store.auditRepo.recordSemanticWriteAttemptStart({
      agentId: "main",
      sessionKey: "s1",
      scope: "agent:main",
      turnId: "semantic-succeeded",
      sourceRefs: ["user:semantic-succeeded:0"],
      inputHash: "input-v1",
      startedAt: "2026-05-21T00:03:00.000Z",
      retryOnly: true,
    });
    assert.equal(attempt.claimed, false);

    const rows = store.client
      .prepare("SELECT status, attempt_count, last_error FROM semantic_write_jobs WHERE turn_id = ?")
      .all("semantic-succeeded");
    assert.equal(rows.length, 1);
    assert.equal(rows[0].status, "succeeded");
    assert.equal(rows[0].attempt_count, 1);
    assert.equal(rows[0].last_error, null);
    await manager.closeAll();
  } finally {
    await rm(tempDir, { recursive: true, force: true });
  }
});

test("maintenance repairs retrying semantic write jobs even without pending scheduler turns", async () => {
  const tempDir = await mkdtemp(join(tmpdir(), "memx-semantic-retry-queue-"));
  const dbPath = join(tempDir, "memx.sqlite");

  try {
    const manager = new MemxRuntimeManager(logger());
    const ctx = ctxFor(dbPath);
    const store = await manager.getStore(ctx);
    const sourceRef = "user:semantic-retry-queue:0";
    store.reasoner.isEnabled = () => true;
    store.reasoner.summarizeChunk = async (text) => text.slice(0, 120);
    store.reasoner.compileTurnSemantics = async () => null;

    await store.turnScheduler.enqueue(ctx, [
      {
        role: "user",
        content: "InvoicePilot 默认数据库是 PostgreSQL。",
        scope: "agent:main",
        sessionKey: "s1",
        turnId: "semantic-retry-queue",
        sourceRef,
        observedAt,
      },
      {
        role: "assistant",
        content: "收到，InvoicePilot 当前使用 PostgreSQL。",
        scope: "agent:main",
        sessionKey: "s1",
        turnId: "semantic-retry-queue",
        sourceRef: "assistant:semantic-retry-queue:1",
        observedAt,
      },
    ]);
    await store.turnScheduler.flush();

    let jobs = store.auditRepo.listSemanticWriteJobs({
      agentId: "main",
      sessionKey: "s1",
      limit: 10,
    });
    assert.equal(jobs.length, 1);
    assert.equal(jobs[0].status, "retrying");

    store.reasoner.compileLongTurnSemantics = async () => relationPatch(sourceRef);
    await runAutomaticMaintenanceBatch(store, ctx, {
      sessionKey: "s1",
      turnIds: [],
      turnCount: 0,
      reason: "idle",
      lowerWatermarks: {},
      upperWatermarks: {},
    });

    jobs = store.auditRepo.listSemanticWriteJobs({
      agentId: "main",
      sessionKey: "s1",
      limit: 10,
    });
    assert.equal(jobs.length, 1);
    assert.equal(jobs[0].status, "succeeded");
    assert.equal(jobs[0].attemptCount, 2);

    const rows = store.client
      .prepare("SELECT job_type, status, stats_json FROM maintenance_runs WHERE job_type = ?")
      .all("source-segment-semantic-extraction");
    assert.equal(rows.length, 1);
    const stats = JSON.parse(rows[0].stats_json);
    assert.deepEqual(stats.turnIds, []);
    assert.deepEqual(stats.repairTurnIds, ["semantic-retry-queue"]);
    assert.equal(stats.retryJobCount, 1);
    await manager.closeAll();
  } finally {
    await rm(tempDir, { recursive: true, force: true });
  }
});

test("maintenance retry queue records LLM-unavailable repair attempts", async () => {
  const tempDir = await mkdtemp(join(tmpdir(), "memx-semantic-retry-unavailable-"));
  const dbPath = join(tempDir, "memx.sqlite");

  try {
    const manager = new MemxRuntimeManager(logger());
    const ctx = ctxFor(dbPath);
    const store = await manager.getStore(ctx);
    store.reasoner.isEnabled = () => true;
    store.reasoner.summarizeChunk = async (text) => text.slice(0, 120);
    store.reasoner.compileTurnSemantics = async () => null;

    await store.turnScheduler.enqueue(ctx, [
      {
        role: "user",
        content: "LedgerPilot 默认导出格式是 Parquet。",
        scope: "agent:main",
        sessionKey: "s1",
        turnId: "semantic-retry-unavailable",
        sourceRef: "user:semantic-retry-unavailable:0",
        observedAt,
      },
      {
        role: "assistant",
        content: "收到，LedgerPilot 默认导出 Parquet。",
        scope: "agent:main",
        sessionKey: "s1",
        turnId: "semantic-retry-unavailable",
        sourceRef: "assistant:semantic-retry-unavailable:1",
        observedAt,
      },
    ]);
    await store.turnScheduler.flush();

    store.reasoner.isEnabled = () => false;
    await runAutomaticMaintenanceBatch(store, ctx, {
      sessionKey: "s1",
      turnIds: [],
      turnCount: 0,
      reason: "idle",
      lowerWatermarks: {},
      upperWatermarks: {},
    });

    const jobs = store.auditRepo.listSemanticWriteJobs({
      agentId: "main",
      sessionKey: "s1",
      limit: 10,
    });
    assert.equal(jobs.length, 1);
    assert.equal(jobs[0].status, "retrying");
    assert.equal(jobs[0].attemptCount, 2);
    assert.match(jobs[0].lastError, /llm-unavailable/);

    const rows = store.client
      .prepare("SELECT status, stats_json FROM maintenance_runs WHERE job_type = ?")
      .all("source-segment-semantic-extraction");
    assert.equal(rows.length, 1);
    const stats = JSON.parse(rows[0].stats_json);
    assert.deepEqual(stats.repairTurnIds, ["semantic-retry-unavailable"]);
    assert.deepEqual(stats.skippedReasons, ["llm-unavailable"]);
    await manager.closeAll();
  } finally {
    await rm(tempDir, { recursive: true, force: true });
  }
});

test("maintenance semantic retries back off and dead-letter repeated LLM failures", async () => {
  const tempDir = await mkdtemp(join(tmpdir(), "memx-semantic-retry-backoff-"));
  const dbPath = join(tempDir, "memx.sqlite");

  try {
    const manager = new MemxRuntimeManager(logger());
    const ctx = ctxFor(dbPath);
    const store = await manager.getStore(ctx);
    store.reasoner.isEnabled = () => true;
    store.reasoner.summarizeChunk = async (text) => text.slice(0, 120);
    store.reasoner.compileTurnSemantics = async () => null;

    await store.turnScheduler.enqueue(ctx, [
      {
        role: "user",
        content: "VectorDesk 默认索引后端是 SQLite FTS5。",
        scope: "agent:main",
        sessionKey: "s1",
        turnId: "semantic-retry-backoff",
        sourceRef: "user:semantic-retry-backoff:0",
        observedAt,
      },
      {
        role: "assistant",
        content: "收到，VectorDesk 默认索引后端是 SQLite FTS5。",
        scope: "agent:main",
        sessionKey: "s1",
        turnId: "semantic-retry-backoff",
        sourceRef: "assistant:semantic-retry-backoff:1",
        observedAt,
      },
    ]);
    await store.turnScheduler.flush();

    store.reasoner.isEnabled = () => false;
    await runAutomaticMaintenanceBatch(store, ctx, {
      sessionKey: "s1",
      turnIds: [],
      turnCount: 0,
      reason: "idle",
      lowerWatermarks: {},
      upperWatermarks: {},
    });

    let jobs = store.auditRepo.listSemanticWriteJobs({
      agentId: "main",
      sessionKey: "s1",
      limit: 10,
    });
    assert.equal(jobs.length, 1);
    assert.equal(jobs[0].status, "retrying");
    assert.equal(jobs[0].attemptCount, 2);
    assert.ok(jobs[0].nextAttemptAt, "failed maintenance repair should schedule a future retry");
    assert.ok(Date.parse(jobs[0].nextAttemptAt) > Date.parse(ctx.now));

    await runAutomaticMaintenanceBatch(store, ctx, {
      sessionKey: "s1",
      turnIds: [],
      turnCount: 0,
      reason: "idle",
      lowerWatermarks: {},
      upperWatermarks: {},
    });
    jobs = store.auditRepo.listSemanticWriteJobs({
      agentId: "main",
      sessionKey: "s1",
      limit: 10,
    });
    assert.equal(jobs[0].attemptCount, 2, "not-yet-due retry jobs should not be retried immediately");

    for (let expectedAttempt = 3; expectedAttempt <= 6; expectedAttempt += 1) {
      const dueAt = jobs[0].nextAttemptAt;
      assert.ok(dueAt, `attempt ${expectedAttempt} should have a due time`);
      await runAutomaticMaintenanceBatch(store, { ...ctx, now: dueAt }, {
        sessionKey: "s1",
        turnIds: [],
        turnCount: 0,
        reason: "idle",
        lowerWatermarks: {},
        upperWatermarks: {},
      });
      jobs = store.auditRepo.listSemanticWriteJobs({
        agentId: "main",
        sessionKey: "s1",
        limit: 10,
      });
      assert.equal(jobs[0].attemptCount, expectedAttempt);
    }

    assert.equal(jobs[0].status, "failed");
    assert.equal(jobs[0].nextAttemptAt, undefined);

    await runAutomaticMaintenanceBatch(store, { ...ctx, now: "2026-05-28T00:00:00.000Z" }, {
      sessionKey: "s1",
      turnIds: [],
      turnCount: 0,
      reason: "idle",
      lowerWatermarks: {},
      upperWatermarks: {},
    });
    jobs = store.auditRepo.listSemanticWriteJobs({
      agentId: "main",
      sessionKey: "s1",
      limit: 10,
    });
    assert.equal(jobs[0].attemptCount, 6, "dead-lettered semantic jobs should not be retried forever");
    await manager.closeAll();
  } finally {
    await rm(tempDir, { recursive: true, force: true });
  }
});

test("runtime shutdown flush repairs retrying semantic jobs without scheduler state", async () => {
  const tempDir = await mkdtemp(join(tmpdir(), "memx-shutdown-semantic-retry-"));
  const dbPath = join(tempDir, "memx.sqlite");

  try {
    const manager = new MemxRuntimeManager(logger());
    const ctx = ctxFor(dbPath);
    const store = await manager.getStore(ctx);
    const sourceRef = "user:shutdown-semantic-retry:0";
    store.reasoner.isEnabled = () => true;
    store.reasoner.summarizeChunk = async (text) => text.slice(0, 120);
    store.reasoner.compileTurnSemantics = async () => null;

    await store.turnScheduler.enqueue(ctx, [
      {
        role: "user",
        content: "InvoicePilot 默认数据库是 PostgreSQL。",
        scope: "agent:main",
        sessionKey: "s1",
        turnId: "shutdown-semantic-retry",
        sourceRef,
        observedAt,
      },
      {
        role: "assistant",
        content: "收到，InvoicePilot 使用 PostgreSQL。",
        scope: "agent:main",
        sessionKey: "s1",
        turnId: "shutdown-semantic-retry",
        sourceRef: "assistant:shutdown-semantic-retry:1",
        observedAt,
      },
    ]);
    await store.turnScheduler.flush();
    store.reasoner.compileLongTurnSemantics = async () => relationPatch(sourceRef);

    await manager.closeAll();

    const verifier = new MemxRuntimeManager(logger());
    const verifierStore = await verifier.getStore(ctx);
    const jobs = verifierStore.auditRepo.listSemanticWriteJobs({
      agentId: "main",
      sessionKey: "s1",
      limit: 10,
    });
    assert.equal(jobs.length, 1);
    assert.equal(jobs[0].status, "succeeded");
    assert.equal(jobs[0].attemptCount, 2);
    await verifier.closeAll();
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

test("assistant memory acknowledgements are indexed as non-answer support", async () => {
  const tempDir = await mkdtemp(join(tmpdir(), "memx-assistant-ack-"));
  const dbPath = join(tempDir, "memx.sqlite");

  try {
    const manager = new MemxRuntimeManager(logger());
    const ctx = ctxFor(dbPath);
    const store = await manager.getStore(ctx);
    store.reasoner.isEnabled = () => false;
    store.reasoner.summarizeChunk = async (text) => text.slice(0, 120);

    await store.turnScheduler.enqueue(ctx, [
      {
        role: "user",
        content: "青岚工单默认消息队列用 Pulsar。",
        scope: "agent:main",
        sessionKey: "s1",
        turnId: "turn-assistant-ack",
        sourceRef: "user:turn-assistant-ack:0",
        observedAt,
      },
      {
        role: "assistant",
        content: "已记住：青岚工单默认消息队列用 Pulsar。",
        scope: "agent:main",
        sessionKey: "s1",
        turnId: "turn-assistant-ack",
        sourceRef: "assistant:turn-assistant-ack:1",
        observedAt,
      },
    ]);
    await store.turnScheduler.flush();

    const rows = store.client
      .prepare("SELECT text, metadata_json FROM vector_docs WHERE metadata_json LIKE ?")
      .all("%turn-assistant-ack%");
    const assistantDoc = rows
      .map((row) => ({ ...row, metadata: JSON.parse(row.metadata_json) }))
      .find((row) => row.metadata.role === "assistant");

    assert.ok(assistantDoc, "assistant acknowledgement chunk should still be indexed for lineage");
    assert.equal(assistantDoc.metadata.semanticRole, "assistant_acknowledgement");
    assert.equal(assistantDoc.metadata.memoryClass, "assistant_acknowledgement");
    assert.equal(assistantDoc.metadata.recallVisibility, "support_only");
    await manager.closeAll();
  } finally {
    await rm(tempDir, { recursive: true, force: true });
  }
});

test("native recall does not wait for unresolved semantic write work before using staged evidence", async () => {
  const tempDir = await mkdtemp(join(tmpdir(), "memx-host-nonblocking-context-"));
  const dbPath = join(tempDir, "{agentId}", "memx.sqlite");
  const workspaceDir = "/tmp/memx-lifecycle-test";

  try {
    const { MemxHostService } = await import("../dist/.runtime/src/host/service.mjs");
    const config = configFor(dbPath);
    config.advanced.enableMaintenanceJobs = false;
    config.advanced.enableQueryCompiler = false;
    const service = new MemxHostService({ config, logger: logger() });
    const ctx = buildOperationContext(config, {
      agentId: "stage-agent",
      sessionKey: "generic:s1",
      workspaceDir,
      project: "stage-test",
    });
    await service.manager.getStore(ctx);
    service.manager.rememberStagedRecallableTurn(ctx, [
      {
        role: "user",
        content: "工程记录：FastBridge 的默认消息队列改成 Pulsar。",
        scope: "agent:stage-agent",
        sessionKey: "generic:s1",
        turnId: "turn-fast-pending",
        sourceRef: "user:turn-fast-pending",
        observedAt,
      },
    ]);
    service.pendingWrites.set(
      `${ctx.agentId}\u0000${ctx.dbPath}\u0000workspace:${ctx.workspaceDir}`,
      new Promise(() => {}),
    );

    const startedAt = Date.now();
    const result = await service.context({
      hostId: "generic",
      actorId: "stage-agent",
      sessionId: "s2",
      workspaceDir,
      query: "FastBridge 默认消息队列是什么？",
      hotPathTimeoutMs: 6500,
    });
    const elapsedMs = Date.now() - startedAt;

    assert.equal(elapsedMs < 1000, true, `context call should not wait on semantic work, took ${elapsedMs}ms`);
    assert.match(result.prependContext, /Pulsar/);
    assert.equal(result.recall.diagnostics.includes("pending-staged-turn-evidence"), true);
    service.pendingWrites.clear();
    await service.close();
  } finally {
    await rm(tempDir, { recursive: true, force: true });
  }
});

test("host observe uses injected native recall provenance to suppress assistant memory echoes", async () => {
  const tempDir = await mkdtemp(join(tmpdir(), "memx-host-recall-echo-"));
  const dbPath = join(tempDir, "{agentId}", "memx.sqlite");

  try {
    const { MemxHostService } = await import("../dist/.runtime/src/host/service.mjs");
    const config = configFor(dbPath);
    config.advanced.enableMaintenanceJobs = false;
    const service = new MemxHostService({ config, logger: logger() });
    const ctx = buildOperationContext(config, {
      agentId: "codex--echo-agent",
      sessionKey: "codex:s1",
      workspaceDir: "/tmp/memx-lifecycle-test",
      project: "echo-test",
    });
    const store = await service.manager.getStore(ctx);
    let capturedMessages = [];
    store.turnScheduler.stageRecallableTurn = async () => false;
    store.turnScheduler.enqueue = async (_ctx, messages) => {
      capturedMessages = messages;
    };

    await service.observe({
      hostId: "codex",
      actorId: "echo-agent",
      sessionId: "s1",
      workspaceDir: "/tmp/memx-lifecycle-test",
      eventName: "turn",
      observedAt,
      messages: [
        {
          role: "user",
          content: "LyraLedger 的默认队列是什么？",
        },
        {
          role: "assistant",
          content: "LyraLedger 默认队列是 Kafka。",
        },
      ],
      metadata: {
        memxRecall: {
          injectedTexts: ["## memX Memory\n- LyraLedger 默认队列是 Kafka。"],
        },
      },
    });
    await service.close();

    assert.deepEqual(
      capturedMessages.map((message) => message.role),
      ["user"],
      "assistant replies that only echo injected native context should not be written as new turn content",
    );
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
    service.pendingWrites.set(
      `${ctx.agentId}\u0000${ctx.dbPath}\u0000workspace:${ctx.workspaceDir}`,
      new Promise(() => {}),
    );

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

test("native context includes same-workspace staged evidence across sessions while semantic write is pending", async () => {
  const tempDir = await mkdtemp(join(tmpdir(), "memx-host-pending-context-cross-session-"));
  const dbPath = join(tempDir, "{agentId}", "memx.sqlite");
  const workspaceDir = "/tmp/memx-lifecycle-test";

  try {
    const { MemxHostService } = await import("../dist/.runtime/src/host/service.mjs");
    const config = configFor(dbPath);
    config.advanced.enableMaintenanceJobs = false;
    config.advanced.enableQueryCompiler = false;
    const service = new MemxHostService({ config, logger: logger() });
    const ctx = buildOperationContext(config, {
      agentId: "stage-agent",
      sessionKey: "generic:s1",
      workspaceDir,
      project: "stage-test",
    });
    const store = await service.manager.getStore(ctx);
    store.turnScheduler.stageRecallableTurn = async () => true;
    store.turnScheduler.enqueue = async () => new Promise(() => {});

    await service.observe({
      hostId: "generic",
      actorId: "stage-agent",
      sessionId: "s1",
      workspaceDir,
      eventName: "turn",
      observedAt,
      messages: [
        {
          role: "user",
          content: "工程记录：AuroraBridge 的默认告警通道是 RelayNine。",
        },
        {
          role: "assistant",
          content: "已记录，AuroraBridge 默认告警通道是 RelayNine。",
        },
      ],
    });

    const result = await service.context({
      hostId: "generic",
      actorId: "stage-agent",
      sessionId: "s2",
      workspaceDir,
      query: "AuroraBridge 默认告警通道是什么？",
      hotPathTimeoutMs: 1600,
    });

    assert.match(result.prependContext, /RelayNine/);
    assert.equal(result.recall.diagnostics.includes("pending-staged-turn-evidence"), true);
    service.pendingWrites.clear();
    await service.close();
  } finally {
    await rm(tempDir, { recursive: true, force: true });
  }
});

test("native context audit records the finalized injected context after staged evidence is added", async () => {
  const tempDir = await mkdtemp(join(tmpdir(), "memx-host-final-context-audit-"));
  const dbPath = join(tempDir, "{agentId}", "memx.sqlite");
  const workspaceDir = "/tmp/memx-lifecycle-test";

  try {
    const { MemxHostService } = await import("../dist/.runtime/src/host/service.mjs");
    const config = configFor(dbPath);
    config.advanced.enableMaintenanceJobs = false;
    config.advanced.enableQueryCompiler = false;
    const service = new MemxHostService({ config, logger: logger() });
    const ctx = buildOperationContext(config, {
      agentId: "stage-agent",
      sessionKey: "generic:s1",
      workspaceDir,
      project: "stage-test",
    });
    const store = await service.manager.getStore(ctx);
    store.turnScheduler.stageRecallableTurn = async () => true;
    store.turnScheduler.enqueue = async () => new Promise(() => {});

    await service.observe({
      hostId: "generic",
      actorId: "stage-agent",
      sessionId: "s1",
      workspaceDir,
      eventName: "turn",
      observedAt,
      messages: [
        {
          role: "user",
          content: "工程记录：AuditBridge 的默认告警通道是 RelayNine。",
        },
        {
          role: "assistant",
          content: "已记录，AuditBridge 默认告警通道是 RelayNine。",
        },
      ],
    });

    const result = await service.context({
      hostId: "generic",
      actorId: "stage-agent",
      sessionId: "s2",
      workspaceDir,
      query: "AuditBridge 默认告警通道是什么？",
      hotPathTimeoutMs: 1600,
    });
    assert.match(result.prependContext, /AuditBridge/);
    assert.match(result.prependContext, /RelayNine/);

    const audit = await service.audit(5, {
      hostId: "generic",
      actorId: "stage-agent",
      sessionId: "s2",
      workspaceDir,
    });
    const retrieval = audit.retrievals.find(
      (entry) => entry.queryText === "AuditBridge 默认告警通道是什么？",
    );
    assert.ok(retrieval, "expected retrieval audit row for final native context");
    const finalInjection = retrieval.selectedItemsJson.nativeContextInjection;
    assert.match(finalInjection.actualContextPreview, /AuditBridge/);
    assert.match(finalInjection.actualContextPreview, /RelayNine/);
    assert.equal(finalInjection.finalInjectedPackets[0].packetId.startsWith("pending-staged:"), true);
    assert.equal(finalInjection.finalInjectedPackets[0].primaryText.includes("已记录"), false);
    service.pendingWrites.clear();
    await service.close();
  } finally {
    await rm(tempDir, { recursive: true, force: true });
  }
});

test("degraded native context accepts canonical facts that match a full code-like query anchor", async () => {
  const tempDir = await mkdtemp(join(tmpdir(), "memx-host-degraded-code-anchor-fact-"));
  const dbPath = join(tempDir, "{agentId}", "memx.sqlite");
  const workspaceDir = "/tmp/memx-lifecycle-test";

  try {
    const { MemxHostService } = await import("../dist/.runtime/src/host/service.mjs");
    const config = configFor(dbPath);
    config.advanced.enableMaintenanceJobs = false;
    config.advanced.enableQueryCompiler = false;
    const service = new MemxHostService({ config, logger: logger() });
    const ctx = buildOperationContext(config, {
      agentId: "main",
      sessionKey: "generic:s1",
      workspaceDir,
      project: "stage-test",
    });
    const store = await service.manager.getStore(ctx);
    store.reasoner.isEnabled = () => true;
    store.reasoner.compileTurnSemantics = async () => ({
      sourceRefs: ["user:audit-fact:0", "assistant:audit-fact:1"],
      assertionDrafts: [
        {
          draftId: "audit-fact-alert",
          sourceRef: "user:audit-fact:0",
          familyHint: "fact_like",
          timeframeHint: "current",
          entityHints: [
            { name: "AuditFixProbe1779800108", type: "project" },
            { name: "RelayNine", type: "service" },
          ],
          slotHints: ["alert_channel"],
          valueHint: "RelayNine",
          confidence: 0.94,
          lineage: {
            sourceKind: "chunk",
            sourceId: "audit-fact",
            sourceRef: "user:audit-fact:0",
          },
        },
      ],
      compilerProvenance: { source: "llm", mode: "semantic-compiler-authoritative" },
    });
    await store.turnScheduler.enqueue(ctx, [
      {
        role: "user",
        content: "工程记录：AuditFixProbe1779800108 的默认告警通道是 RelayNine。",
        scope: ctx.scopes[0],
        sessionKey: "generic:s1",
        turnId: "audit-fact",
        sourceRef: "user:audit-fact:0",
        observedAt,
      },
      {
        role: "assistant",
        content: "已记录，AuditFixProbe1779800108 默认告警通道是 RelayNine。",
        scope: ctx.scopes[0],
        sessionKey: "generic:s1",
        turnId: "audit-fact",
        sourceRef: "assistant:audit-fact:1",
        observedAt,
      },
    ]);
    await store.turnScheduler.flush();

    const result = await service.context({
      hostId: "generic",
      actorId: "main",
      sessionId: "s2",
      workspaceDir,
      query: "AuditFixProbe1779800108 默认告警通道是什么？",
      hotPathTimeoutMs: 1600,
    });

    assert.match(result.prependContext, /auditfixprobe1779800108 has alert channel relaynine/i);
    assert.equal(result.nativeContext.eligible, true);
    await service.close();
  } finally {
    await rm(tempDir, { recursive: true, force: true });
  }
});

test("native context withholds unrelated staged turn evidence while semantic write is still pending", async () => {
  const tempDir = await mkdtemp(join(tmpdir(), "memx-host-pending-context-unrelated-"));
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
    service.pendingWrites.set(
      `${ctx.agentId}\u0000${ctx.dbPath}\u0000workspace:${ctx.workspaceDir}`,
      new Promise(() => {}),
    );

    const result = await service.context({
      hostId: "generic",
      actorId: "stage-agent",
      sessionId: "s1",
      workspaceDir: "/tmp/memx-lifecycle-test",
      query: "给 API 分页评审写一个英文 checklist。",
      hotPathTimeoutMs: 1600,
    });

    assert.equal(result.prependContext, "");
    assert.equal(result.recall.diagnostics.includes("pending-staged-turn-withheld"), true);
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
      sessionKey: "generic:s1",
      workspaceDir: "/tmp/memx-lifecycle-test",
      project: "audit-test",
    });
    const store = await service.manager.getStore(ctx);
    store.auditRepo.recordRetrieval({
      auditId: "audit-retrieval-1",
      agentId: ctx.agentId,
      sessionKey: ctx.sessionKey,
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
    store.auditRepo.recordRetrieval({
      auditId: "audit-retrieval-other-session",
      agentId: ctx.agentId,
      sessionKey: "generic:s2",
      scope: "agent:audit-agent",
      routeType: "mixed",
      queryText: "OtherSession 默认数据库是什么？",
      queryHash: "other-query-hash",
      selectedItemsJson: {},
      injectedChars: 10,
      createdAt: observedAt,
    });
    store.auditRepo.recordPolicyDecision({
      agentId: ctx.agentId,
      sessionKey: ctx.sessionKey,
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
    store.auditRepo.recordPolicyDecision({
      agentId: ctx.agentId,
      sessionKey: "generic:s2",
      sourceRef: "user:other-session:0",
      candidateText: "OtherSession 默认数据库是 SQLite",
      decision: {
        salienceScore: 0.8,
        expectedFutureUtility: 0.8,
        sensitivityScore: 0,
        stabilityScore: 0.8,
        action: "stable_fact",
        reasons: ["other-session-policy"],
        explicitIntent: false,
        captureAuthorized: true,
      },
      createdAt: observedAt,
      metadataJson: { materializationOutcome: { facts: 1 } },
    });
    const runId = store.auditRepo.startMaintenance({
      agentId: ctx.agentId,
      sessionKey: ctx.sessionKey,
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
    const otherRunId = store.auditRepo.startMaintenance({
      agentId: ctx.agentId,
      sessionKey: "generic:s2",
      jobType: "batched",
      stats: { queuedTurns: 1 },
      startedAt: observedAt,
    });
    store.auditRepo.finishMaintenance({
      runId: otherRunId,
      agentId: ctx.agentId,
      sessionKey: "generic:s2",
      jobType: "batched",
      statsJson: { queuedTurns: 1, promotedFacts: 1 },
      startedAt: observedAt,
      completedAt: observedAt,
      status: "completed",
    });
    store.maintenanceRepo.recordPendingTurn({
      agentId: ctx.agentId,
      sessionKey: "generic:s1",
      turnId: "audit-pending-turn",
      observedAt,
      updatedAt: observedAt,
    });
    store.maintenanceRepo.recordPendingTurn({
      agentId: ctx.agentId,
      sessionKey: "generic:s2",
      turnId: "audit-pending-other-session",
      observedAt,
      updatedAt: observedAt,
    });

    const audit = await service.audit(10, {
      hostId: "generic",
      actorId: "audit-agent",
      sessionId: "s1",
    });

    assert.equal(audit.retrievals.length, 1);
    assert.equal(audit.retrievals[0].queryText, "AuroraAccept 默认数据库是什么？");
    assert.equal(audit.retrievals[0].selectedItemsJson.nativeContextInjection.actualInjectedChars, 80);
    assert.equal(audit.policyDecisions.length, 1);
    assert.equal(audit.policyDecisions[0].chosenAction, "stable_fact");
    assert.equal(audit.policyDecisions[0].sourceRef, "user:audit-turn:0");
    assert.equal(audit.maintenanceRuns.length, 1);
    assert.equal(audit.maintenanceRuns[0].status, "completed");
    assert.deepEqual(audit.maintenanceRuns[0].statsJson, { queuedTurns: 3, promotedFacts: 1 });
    assert.equal(audit.maintenanceSchedulerStates.length, 1);
    assert.equal(audit.maintenanceSchedulerStates[0].pendingTurnCount, 1);
    assert.deepEqual(audit.maintenanceSchedulerStates[0].pendingTurnIds, ["audit-pending-turn"]);
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
