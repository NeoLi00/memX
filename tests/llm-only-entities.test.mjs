import assert from "node:assert/strict";
import test from "node:test";
import { buildCandidate } from "../dist/.runtime/src/pipeline/extract.mjs";
import {
  buildQueryCompilerPromptInput,
  compileQuery,
  compileQueryWithoutSemanticFallback,
} from "../dist/.runtime/src/pipeline/queryCompiler.mjs";
import { normalizeCandidate } from "../dist/.runtime/src/pipeline/normalize.mjs";
import { MemxReasoner, sanitizeChunkSummaryResult } from "../dist/.runtime/src/pipeline/reasoner.mjs";
import * as semantics from "../dist/.runtime/src/pipeline/semantics.mjs";
import {
  buildLongTurnSemanticScanInputFromSegments,
  buildTurnSemanticCompilerInput,
  compileTurnSemantics,
  frameHintsForSourceRef,
} from "../dist/.runtime/src/pipeline/turnSemanticCompiler.mjs";

const observedAt = "2026-05-12T00:00:00.000Z";

function minimalConfig() {
  return {
    captureMaxChars: 4000,
    advanced: {
      enableTurnSemanticCompiler: true,
      enableQueryCompiler: true,
    },
  };
}

function minimalCtx() {
  return {
    agentId: "main",
    scope: "agent:main",
    scopes: ["agent:main"],
    now: observedAt,
    config: minimalConfig(),
  };
}

test("candidate extraction does not create deterministic semantic hints", () => {
  const candidate = buildCandidate({
    sourceKind: "user",
    rawText: '记住：我喜欢中文。明天继续 project OpenClaw uses "DeepSeek" for e_p sanity check',
    observedAt,
    config: minimalConfig(),
    source: {
      sessionKey: "s1",
      messageId: "m1",
    },
    eventType: "conversation_turn",
  });

  assert.ok(candidate);
  assert.deepEqual(candidate.structuredHints?.entities, []);
  assert.deepEqual(candidate.structuredHints?.timeHints, []);
  assert.equal(candidate.structuredHints?.preference, undefined);
  assert.equal(candidate.structuredHints?.preferenceHint, undefined);
  assert.equal(candidate.structuredHints?.workflow, undefined);
  assert.equal(candidate.structuredHints?.workflows, undefined);
  assert.equal(candidate.structuredHints?.taskStateHint, undefined);
  assert.equal(candidate.structuredHints?.decision, undefined);
  assert.equal(candidate.structuredHints?.decisionHint, undefined);
  assert.equal(candidate.structuredHints?.correction, undefined);
  assert.equal(candidate.structuredHints?.correctionHint, undefined);
  assert.equal(candidate.structuredHints?.relation, undefined);
  assert.equal(candidate.structuredHints?.relations, undefined);
  assert.equal(candidate.structuredHints?.relationHint, undefined);
});

test("legacy deterministic semantic extractors are not exported", () => {
  for (const name of [
    "inferEntityNames",
    "extractQueryAnchors",
    "seedEntityNamesFromQuery",
    "analyzeSemanticHints",
    "parsePreferenceSignal",
    "parseWorkflowState",
    "analyzeRecallQueryShape",
    "analyzeCorrectionHint",
  ]) {
    assert.equal(name in semantics, false, `${name} should not remain on semantics surface`);
  }
});

test("query compiler honors request hot-path budget before the global config timeout", async () => {
  const ctx = minimalCtx();
  ctx.config.advanced.queryCompilerHotPathTimeoutMs = 8000;
  const startedAt = Date.now();

  const result = await compileQuery({
    query: "PendingFirstProbe 的默认格式是什么？",
    ctx,
    hotPathTimeoutMs: 25,
    reasoner: {
      isEnabled: () => true,
      compileQuerySemantics: async (_query, _fallback, options = {}) =>
        new Promise((resolve) => {
          options.signal?.addEventListener("abort", () => resolve(null), { once: true });
        }),
    },
  });

  assert.ok(Date.now() - startedAt < 1000);
  assert.deepEqual(result.compilerProvenance, {
    source: "deterministic",
    mode: "fallback",
    reasons: ["query-compile-llm-timeout"],
  });
});

test("query compiler retries one unparsable LLM JSON response before degrading", async (t) => {
  const originalFetch = globalThis.fetch;
  const responses = [
    "this is not json",
    JSON.stringify({
      focusedQuery: "ClaudeProbe default format",
      queryEntities: [{ name: "ClaudeProbe", type: "project", role: "subject" }],
      queryShape: {
        timeframe: "timeless",
        granularity: "detail",
        referentialMode: "anchored",
        evidenceNeed: "canonical_state",
      },
      primaryRoute: "factual",
      answerGranularity: "detail",
      evidenceFidelity: "high",
      routeWeights: { factual: 1 },
      anchors: ["ClaudeProbe", "default format"],
      candidateSurfaces: ["fact", "state", "graph"],
      evidenceGoals: [],
      detailNeedScore: 0.9,
      supportNeed: 0.7,
      ambiguityLevel: 0.1,
      turnMode: "memory_qa",
    }),
  ];
  let fetchCalls = 0;
  globalThis.fetch = async () => {
    const content = responses[fetchCalls] ?? responses.at(-1);
    fetchCalls += 1;
    return new Response(
      JSON.stringify({
        choices: [{ message: { content } }],
      }),
      { status: 200, headers: { "content-type": "application/json" } },
    );
  };
  t.after(() => {
    globalThis.fetch = originalFetch;
  });

  const config = minimalConfig();
  config.advanced.llmClassifierEnabled = true;
  config.advanced.llmProvider = "openai-compatible";
  config.advanced.llmBaseURL = "https://llm.example.com/v1";
  config.advanced.llmClassifierModel = "fast-memory-model";
  config.advanced.llmApiKey = "sk-test";
  const reasoner = new MemxReasoner(config, {
    debug() {},
    info() {},
    warn() {},
  });
  const fallback = compileQueryWithoutSemanticFallback(
    "ClaudeProbe 的默认格式是什么？",
    "test-fallback",
  );

  const result = await reasoner.compileQuerySemantics(
    "ClaudeProbe 的默认格式是什么？",
    fallback,
    { stage: "query_hot_path" },
  );

  assert.equal(fetchCalls, 2);
  assert.equal(result?.focusedQuery, "ClaudeProbe default format");
  assert.equal(result?.compilerProvenance?.source, "llm");
  assert.equal(
    reasoner.getTrace().some((entry) => entry.label === "query-compile-retry"),
    true,
  );
});

test("post-answer turn semantic compiler retries one unparsable LLM JSON response", async (t) => {
  const originalFetch = globalThis.fetch;
  const responses = [
    "not-json",
    JSON.stringify({
      assertions: [
        {
          sourceRef: "wrong-source-ref",
          subject: "InvoicePilot",
          subjectType: "project",
          slot: "export_format",
          value: "Arrow IPC",
          family: "fact_like",
          timeframe: "current",
          confidence: 0.94,
        },
      ],
      corrections: [
        {
          sourceRef: "wrong-source-ref",
          subject: "InvoicePilot",
          slot: "export_format",
          canonicalKey: "InvoicePilot.export_format",
          predicate: "has_export_format",
          priorValue: "Parquet",
          nextValue: "Arrow IPC",
          confidence: 0.94,
        },
      ],
    }),
  ];
  let fetchCalls = 0;
  globalThis.fetch = async () => {
    const content = responses[fetchCalls] ?? responses.at(-1);
    fetchCalls += 1;
    return new Response(
      JSON.stringify({
        choices: [{ message: { content } }],
      }),
      { status: 200, headers: { "content-type": "application/json" } },
    );
  };
  t.after(() => {
    globalThis.fetch = originalFetch;
  });

  const config = minimalConfig();
  config.advanced.llmClassifierEnabled = true;
  config.advanced.llmProvider = "openai-compatible";
  config.advanced.llmBaseURL = "https://llm.example.com/v1";
  config.advanced.llmClassifierModel = "fast-memory-model";
  config.advanced.llmApiKey = "sk-test";
  const reasoner = new MemxReasoner(config, {
    debug() {},
    info() {},
    warn() {},
  });

  const frame = await compileTurnSemantics({
    messages: [
      {
        role: "user",
        content: "这个导出格式以后改成 Arrow IPC，不要再用 Parquet。",
        scope: "agent:main",
        sessionKey: "s1",
        turnId: "turn-post-answer",
        sourceRef: "user:turn-post-answer",
        observedAt,
      },
      {
        role: "assistant",
        content: "好的，后续 InvoicePilot 的导出格式按 Arrow IPC 处理。",
        scope: "agent:main",
        sessionKey: "s1",
        turnId: "turn-post-answer",
        sourceRef: "assistant:turn-post-answer",
        observedAt,
      },
    ],
    ctx: minimalCtx(),
    reasoner,
  });

  assert.equal(fetchCalls, 2);
  assert.equal(frame?.assertionDrafts[0]?.sourceRef, "user:turn-post-answer");
  assert.equal(frame?.assertionDrafts[0]?.entityHints?.[0]?.name, "InvoicePilot");
  assert.equal(frame?.assertionDrafts[0]?.slotHints?.[0], "export_format");
  assert.equal(frame?.assertionDrafts[0]?.valueHint, "Arrow IPC");
  assert.equal(frame?.correctionDrafts[0]?.correction.targetKind, "fact");
  assert.equal(frame?.correctionDrafts[0]?.correction.timeframe, "current");
  assert.equal(
    reasoner.getTrace().some((entry) => entry.label === "turn-semantic-compact-retry"),
    true,
  );
});

test("turn semantic fallback does not synthesize deterministic entity hints or relation drafts", async () => {
  const frame = await compileTurnSemantics({
    messages: [
      {
        role: "user",
        content: 'project OpenClaw uses "DeepSeek" for e_p sanity check',
        scope: "agent:main",
        sessionKey: "s1",
        turnId: "turn1",
        sourceRef: "user:turn1",
        observedAt,
      },
    ],
    ctx: minimalCtx(),
    reasoner: {
      isEnabled: () => false,
    },
  });

  assert.ok(frame);
  assert.equal(frame.taskProposal, undefined);
  assert.equal(frame.assertionDrafts.length, 0);
  assert.equal(frame.correctionDrafts.length, 0);
  assert.equal(frame.relationDrafts?.length ?? 0, 0);
  assert.equal(
    frame.assertionDrafts.some((draft) => (draft.entityHints?.length ?? 0) > 0),
    false,
  );
});

test("compact LLM turn compiler can return multiple entity attribute facts", async (t) => {
  const originalFetch = globalThis.fetch;
  globalThis.fetch = async () =>
    new Response(
      JSON.stringify({
        choices: [
          {
            message: {
              content: JSON.stringify({
                assertions: [
                  {
                    sourceRef: "user:turn-multi",
                    subject: "AuroraAccept",
                    subjectType: "project",
                    slot: "default_database",
                    value: "PostgreSQL",
                    confidence: 0.93,
                  },
                  {
                    sourceRef: "user:turn-multi",
                    subject: "AuroraAccept",
                    subjectType: "project",
                    slot: "alert_channel",
                    value: "Slack",
                    confidence: 0.92,
                  },
                  {
                    sourceRef: "user:turn-multi",
                    subject: "AuroraAccept",
                    subjectType: "project",
                    slot: "export_format",
                    value: "Parquet",
                    confidence: 0.91,
                  },
                ],
              }),
            },
          },
        ],
      }),
      { status: 200, headers: { "content-type": "application/json" } },
    );
  t.after(() => {
    globalThis.fetch = originalFetch;
  });

  const config = minimalConfig();
  config.advanced.llmClassifierEnabled = true;
  config.advanced.llmProvider = "openai-compatible";
  config.advanced.llmBaseURL = "https://llm.example.com/v1";
  config.advanced.llmClassifierModel = "fast-memory-model";
  config.advanced.llmApiKey = "sk-test";
  const reasoner = new MemxReasoner(config, {
    debug() {},
    info() {},
    warn() {},
  });

  const frame = await compileTurnSemantics({
    messages: [
      {
        role: "user",
        content: "AuroraAccept 的默认数据库是 PostgreSQL，告警渠道是 Slack，导出格式先用 Parquet。",
        scope: "agent:main",
        sessionKey: "s1",
        turnId: "turn-multi",
        sourceRef: "user:turn-multi",
        observedAt,
      },
      {
        role: "assistant",
        content: "已确认这三个默认配置。",
        scope: "agent:main",
        sessionKey: "s1",
        turnId: "turn-multi",
        sourceRef: "assistant:turn-multi",
        observedAt,
      },
    ],
    ctx: minimalCtx(),
    reasoner,
  });

  assert.equal(frame?.compilerProvenance.promptVersion, "turn-semantic-compact/v1");
  assert.deepEqual(
    frame?.assertionDrafts.map((draft) => [draft.slotHints?.[0], draft.valueHint]),
    [
      ["default_database", "PostgreSQL"],
      ["alert_channel", "Slack"],
      ["export_format", "Parquet"],
    ],
  );
  assert.deepEqual(
    frame?.assertionDrafts.map((draft) => draft.entityHints?.[0]?.name),
    ["AuroraAccept", "AuroraAccept", "AuroraAccept"],
  );
});

test("LLM turn compiler entity hints remain the only entity hint source", async () => {
  const frame = await compileTurnSemantics({
    messages: [
      {
        role: "user",
        content: "DeepSeek should be remembered as the provider for this turn",
        scope: "agent:main",
        sessionKey: "s1",
        turnId: "turn2",
        sourceRef: "user:turn2",
        observedAt,
      },
    ],
    ctx: minimalCtx(),
    reasoner: {
      isEnabled: () => true,
      compileTurnSemantics: async () => ({
        assertionDrafts: [
          {
            sourceRef: "user:turn2",
            familyHint: "fact_like",
            timeframeHint: "timeless",
            entityHints: [{ name: "DeepSeek", type: "service" }],
            confidence: 0.9,
          },
        ],
      }),
    },
  });

  const hints = frameHintsForSourceRef(frame, "user:turn2");

  assert.deepEqual(hints?.entities, [{ name: "DeepSeek", type: "service" }]);
});

test("turn semantic compiler input compacts long messages without losing the tail", () => {
  const middleMarker = "MIDDLE_ONLY_TURN_MARKER_DO_NOT_LEAK";
  const tailMarker = "TAIL_TURN_RECALL_MARKER";
  const content = `${"head facts ".repeat(220)}\n${middleMarker}\n${"body filler ".repeat(220)}\n${tailMarker}`;

  const input = buildTurnSemanticCompilerInput([
    {
      role: "user",
      content,
      scope: "agent:main",
      sessionKey: "s1",
      turnId: "turn-long-input",
      sourceRef: "user:turn-long-input",
      observedAt,
    },
  ]);
  const serialized = JSON.stringify(input);

  assert.equal(input.messages[0].rawLength, content.length);
  assert.equal(input.messages[0].truncated, true);
  assert.ok(serialized.includes(tailMarker));
  assert.equal(serialized.includes(middleMarker), false);
  assert.equal(serialized.includes(content), false);
});

test("turn semantic compiler receives bounded recent reference context for deictic writes", async () => {
  const chunk = (turnId, seq, role, content, summary = "") => ({
    chunkId: `chunk_${turnId}_${seq}`,
    agentId: "main",
    scope: "agent:main",
    sessionKey: "s1",
    turnId,
    seq,
    role,
    chunkKind: "message",
    content,
    summary,
    contentHash: `hash_${turnId}_${seq}`,
    taskId: "task_1",
    dedupStatus: "active",
    mergeCount: 0,
    sourceRef: `${role}:${turnId}:${seq}`,
    createdAt: observedAt,
    updatedAt: observedAt,
  });

  let capturedReferenceContext;
  const frame = await compileTurnSemantics({
    messages: [
      {
        role: "user",
        content: "那这个就不要再考虑了。",
        scope: "agent:main",
        sessionKey: "s1",
        turnId: "turn-current",
        sourceRef: "user:turn-current",
        observedAt,
      },
      {
        role: "assistant",
        content: "好的，后续不再把 Kafka 作为 LuoShu 的 broker 候选。",
        scope: "agent:main",
        sessionKey: "s1",
        turnId: "turn-current",
        sourceRef: "assistant:turn-current",
        observedAt,
      },
    ],
    activeChunks: [
      chunk("turn-old", 0, "user", "最早讨论的是 NATS，OLD_CONTEXT_SHOULD_NOT_LEAK。"),
      chunk("turn-old", 1, "assistant", "NATS 是很早之前的候选。"),
      chunk("turn-prev-1", 0, "user", "broker 方案有哪些？"),
      chunk("turn-prev-1", 1, "assistant", "可以比较 RabbitMQ、Kafka、NATS。"),
      chunk("turn-prev-2", 0, "user", "Kafka 的维护成本是不是太高？"),
      chunk("turn-prev-2", 1, "assistant", "Kafka 对 LuoShu broker 来说维护成本偏高。"),
    ],
    ctx: minimalCtx(),
    reasoner: {
      isEnabled: () => true,
      compileTurnSemantics: async (_messages, fallback) => {
        capturedReferenceContext = fallback.referenceContext;
        return {};
      },
    },
  });

  assert.ok(capturedReferenceContext);
  assert.equal(capturedReferenceContext.purpose, "deictic_reference_resolution");
  assert.deepEqual(
    capturedReferenceContext.turns.map((turn) => turn.turnId),
    ["turn-prev-1", "turn-prev-2"],
  );
  const serializedContext = JSON.stringify(capturedReferenceContext);
  assert.ok(serializedContext.includes("Kafka"));
  assert.equal(serializedContext.includes("OLD_CONTEXT_SHOULD_NOT_LEAK"), false);
  assert.ok(serializedContext.length < 2400);

  const input = buildTurnSemanticCompilerInput(
    [
      {
        role: "user",
        content: "那这个就不要再考虑了。",
        scope: "agent:main",
        sessionKey: "s1",
        turnId: "turn-current",
        sourceRef: "user:turn-current",
        observedAt,
      },
    ],
    capturedReferenceContext,
  );
  assert.deepEqual(input.recentReferenceContext, capturedReferenceContext);
  assert.deepEqual(frame.referenceContext, capturedReferenceContext);
});

test("long turn semantic scan is deferred out of the hot path", async () => {
  const middleMarker = "MEMORY_MEMX_WRITE_MIDDLE_ENTITY_ANCHOR";
  const content = `${"alpha ".repeat(500)}\n${middleMarker}\n${"omega ".repeat(500)}`;

  const frame = await compileTurnSemantics({
    messages: [
      {
        role: "user",
        content,
        scope: "agent:main",
        sessionKey: "s1",
        turnId: "turn-long-write",
        sourceRef: "user:turn-long-write",
        observedAt,
      },
    ],
    ctx: minimalCtx(),
    reasoner: {
      isEnabled: () => true,
      compileTurnSemantics: async () => ({}),
      compileLongTurnSemantics: async () => {
        throw new Error("long turn semantic scan must run in maintenance, not write hot path");
      },
    },
  });

  const hints = frameHintsForSourceRef(frame, "user:turn-long-write");

  assert.equal(hints, undefined);
  assert.equal(frame?.compilerProvenance.source, "llm");
});

test("maintenance long turn scan input uses persisted source segments", () => {
  const middleMarker = "MEMORY_MEMX_WRITE_MIDDLE_ENTITY_ANCHOR";
  const sourceRef = "user:turn-long-write";
  const base = {
    sourceGroupId: "source_group:1",
    parentSourceRef: sourceRef,
    chunkId: "chunk_1",
    agentId: "main",
    scope: "agent:main",
    sessionKey: "s1",
    turnId: "turn-long-write",
    seq: 0,
    role: "user",
    createdAt: observedAt,
    updatedAt: observedAt,
    metadataJson: {},
  };
  const input = buildLongTurnSemanticScanInputFromSegments([
    {
      ...base,
      segmentId: "segment_0",
      segmentIndex: 0,
      charStart: 0,
      charEnd: 1800,
      text: "alpha ".repeat(300),
      contentHash: "hash_0",
    },
    {
      ...base,
      segmentId: "segment_1",
      segmentIndex: 1,
      charStart: 1620,
      charEnd: 3420,
      text: `${"middle ".repeat(50)} ${middleMarker} ${"middle ".repeat(50)}`,
      contentHash: "hash_1",
    },
    {
      ...base,
      segmentId: "segment_2",
      segmentIndex: 2,
      charStart: 3240,
      charEnd: 5000,
      text: "omega ".repeat(300),
      contentHash: "hash_2",
    },
  ]);

  assert.equal(input.messages.length, 1);
  assert.equal(input.messages[0].sourceRef, sourceRef);
  assert.equal(input.messages[0].segmentCount, 3);
  assert.ok(input.messages[0].segments.some((segment) => segment.text.includes(middleMarker)));
});

test("maintenance semantic scan includes short turns and recent reference context", () => {
  const sourceRef = "user:turn-short";
  const base = {
    sourceGroupId: "source_group:short",
    parentSourceRef: sourceRef,
    chunkId: "chunk_short",
    agentId: "main",
    scope: "agent:main",
    sessionKey: "s1",
    turnId: "turn-short",
    seq: 0,
    role: "user",
    createdAt: observedAt,
    updatedAt: observedAt,
    metadataJson: {},
  };
  const input = buildLongTurnSemanticScanInputFromSegments(
    [
      {
        ...base,
        segmentId: "segment_short_0",
        segmentIndex: 0,
        charStart: 0,
        charEnd: 26,
        text: "那这个就不要再考虑了，以后默认 Kafka。",
        contentHash: "hash_short_0",
      },
    ],
    {
      purpose: "deictic_reference_resolution",
      maxTurns: 2,
      turns: [
        {
          turnId: "turn-prev",
          messages: [
            {
              role: "user",
              turnId: "turn-prev",
              sourceRef: "user:turn-prev",
              textExcerpt: "RabbitMQ 的稳定性风险我不接受。",
            },
            {
              role: "assistant",
              turnId: "turn-prev",
              sourceRef: "assistant:turn-prev",
              textExcerpt: "确认：RabbitMQ 不再作为默认队列候选。",
            },
          ],
        },
      ],
    },
  );

  assert.equal(input.messages.length, 1);
  assert.equal(input.messages[0].sourceRef, sourceRef);
  assert.equal(input.messages[0].segmentCount, 1);
  assert.ok(input.messages[0].segments[0].text.includes("Kafka"));
  assert.equal(input.recentReferenceContext?.turns[0].turnId, "turn-prev");
});

test("chunk summary validation rejects unsupported answers to user questions", () => {
  const question = "不看当前对话历史，只根据你能自动获得的记忆回答：MingLedger 的默认队列是什么？";
  const sanitized = sanitizeChunkSummaryResult(question, "user", {
    summary: "MingLedger 默认队列是 Kafka。",
    summaryKind: "answer",
    answerSupported: false,
  });

  assert.match(sanitized, /默认队列是什么/);
  assert.doesNotMatch(sanitized, /Kafka/);
});

test("query compiler protects task-bearing prompt from generic focused query", async () => {
  const query = [
    "You are solving an olympiad problem.",
    "",
    "Problem C8. Let n be a positive integer. Given an n x n board, the unit cell in the top left corner is initially coloured black, and the other cells are coloured white. In each operation, choose a 2 x 2 square with exactly one black cell and colour the remaining three cells black. Determine all values of n such that the whole board can become black.",
    "",
    "Give a rigorous solution.",
  ].join("\n");

  const compiled = await compileQuery({
    query,
    ctx: minimalCtx(),
    reasoner: {
      isEnabled: () => true,
      compileQuerySemantics: async () => ({
        focusedQuery: "Give a rigorous solution.",
        queryEntities: [],
        queryShape: {
          timeframe: "timeless",
          granularity: "summary",
          referentialMode: "anchored",
          evidenceNeed: "chunk",
        },
        primaryRoute: "factual",
      }),
    },
  });

  assert.match(compiled.focusedQuery, /Problem C8/);
  assert.match(compiled.focusedQuery, /n x n board/);
  assert.match(compiled.focusedQuery, /2 x 2 square/);
});

test("query compiler falls back to a neutral recall plan without deterministic semantics", async () => {
  const result = await compileQuery({
    query: "召回 e_p 和 sanity check 的相关记忆",
    ctx: minimalCtx(),
    reasoner: {
      isEnabled: () => false,
    },
  });

  assert.equal(result.shouldRecall, true);
  assert.deepEqual(result.anchors, []);
  assert.ok(result.candidateSurfaces.includes("chunk"));
  assert.deepEqual(result.evidenceGoals, []);
  assert.equal(result.evidencePlan?.operation.type, "return_value");
  assert.deepEqual(
    result.evidencePlan?.slots.flatMap((slot) => slot.requestedAttributeSlots ?? []),
    [],
  );
  assert.deepEqual(result.semanticBridges, undefined);
  assert.equal(result.supportNeed, 0);
  assert.ok((result.routeWeights.factual ?? 0) > 0);
  assert.equal(result.compilerProvenance.mode, "fallback");
  assert.ok(result.compilerProvenance.reasons?.includes("llm-only-query-compiler-unavailable"));
});

test("query compiler uses LLM semantics when available", async () => {
  const result = await compileQuery({
    query: "召回 e_p 和 sanity check 的相关记忆",
    ctx: minimalCtx(),
    reasoner: {
      isEnabled: () => true,
      compileQuerySemantics: async () => ({
        focusedQuery: "e_p sanity check",
        queryShape: {
          timeframe: "historical",
          granularity: "exact_detail",
          referentialMode: "anchored",
          evidenceNeed: "workflow_context",
        },
        answerGranularity: "detail",
        evidenceFidelity: "high",
        routeWeights: { workflow: 1 },
        anchors: ["e_p", "sanity check"],
        candidateSurfaces: ["task", "chunk"],
        evidenceGoals: [
          {
            goal: "Find prior correction about e_p sanity check",
            positiveQueries: ["e_p sanity check"],
            focusAnchors: ["e_p"],
            preferredSurfaces: ["task", "chunk"],
            fidelity: "high",
          },
        ],
        detailNeedScore: 0.9,
        supportNeed: 0.8,
        ambiguityLevel: 0.2,
        turnMode: "memory_qa",
      }),
    },
  });

  assert.equal(result.compilerProvenance.source, "llm");
  assert.deepEqual(result.anchors, ["e_p", "sanity check"]);
  assert.deepEqual(result.candidateSurfaces.slice(0, 2), ["task", "chunk"]);
  assert.ok(result.candidateSurfaces.includes("state"));
  assert.ok(result.candidateSurfaces.includes("event"));
  assert.ok(result.candidateSurfaces.includes("fact"));
  assert.equal(result.supportNeed, 0.8);
});

test("query compiler drops legacy memory-intent fields from LLM output", async () => {
  const result = await compileQuery({
    query: "现在先不谈 MingLedger，给一个发票导入 API 起三个英文方法名。",
    ctx: minimalCtx(),
    reasoner: {
      isEnabled: () => true,
      compileQuerySemantics: async () => ({
        focusedQuery: "发票导入 API 英文方法名",
        memoryUseIntent: "avoid",
        queryEntities: [{ name: "MingLedger", type: "project", role: "context" }],
        queryShape: {
          timeframe: "timeless",
          granularity: "summary",
          referentialMode: "anchored",
          evidenceNeed: "chunk",
        },
        primaryRoute: "factual",
      }),
    },
  });

  assert.equal(result.shouldRecall, true);
  assert.equal(Object.prototype.hasOwnProperty.call(result, "memoryUseIntent"), false);
  assert.deepEqual(result.queryEntities, [
    { name: "MingLedger", type: "project", role: "context" },
  ]);
});

test("query compiler preserves LLM-only suppressed entity constraints for negative anchors", async () => {
  const result = await compileQuery({
    query: "现在先不谈 MingLedger，给一个发票导入 API 起三个英文方法名。",
    ctx: minimalCtx(),
    reasoner: {
      isEnabled: () => true,
      compileQuerySemantics: async () => ({
        focusedQuery: "发票导入 API 英文方法名",
        queryEntities: [],
        suppressedEntities: [
          { name: "MingLedger", type: "project", reason: "user explicitly excluded this context" },
          { name: "这个", type: "unknown", reason: "pronoun should be dropped" },
        ],
        queryShape: {
          timeframe: "timeless",
          granularity: "summary",
          referentialMode: "anchored",
          evidenceNeed: "chunk",
        },
        primaryRoute: "factual",
      }),
    },
  });

  assert.equal(result.shouldRecall, true);
  assert.deepEqual(result.queryEntities, []);
  assert.deepEqual(result.suppressedEntities, [
    { name: "MingLedger", type: "project", reason: "user explicitly excluded this context" },
  ]);
});

test("query compiler preserves LLM-only context exclusions for prior-context directives", async () => {
  const result = await compileQuery({
    query: "换个话题，别提上一个项目，给我一个轻量级周报模板。",
    ctx: minimalCtx(),
    reasoner: {
      isEnabled: () => true,
      compileQuerySemantics: async () => ({
        focusedQuery: "轻量级周报模板",
        queryEntities: [],
        suppressedEntities: [],
        contextExclusions: [
          {
            kind: "prior_project",
            label: "上一个项目",
            reason: "user explicitly excluded previous project context",
          },
          {
            kind: "random_value",
            label: "ignored",
          },
        ],
        queryShape: {
          timeframe: "timeless",
          granularity: "summary",
          referentialMode: "anchored",
          evidenceNeed: "chunk",
        },
        primaryRoute: "workflow",
      }),
    },
  });

  assert.deepEqual(result.contextExclusions, [
    {
      kind: "prior_project",
      label: "上一个项目",
      reason: "user explicitly excluded previous project context",
    },
  ]);
});

test("query compiler falls back to raw-query retrieval when LLM exceeds hot-path deadline", async () => {
  const started = Date.now();
  let aborted = false;
  const compiled = await compileQuery({
    query: "BlueWhaleLedger 的校验命令和幂等字段是什么？",
    ctx: {
      ...minimalCtx(),
      config: {
        ...minimalConfig(),
        advanced: {
          ...minimalConfig().advanced,
          queryCompilerHotPathTimeoutMs: 250,
        },
      },
    },
    reasoner: {
      isEnabled: () => true,
      compileQuerySemantics: (_query, _fallback, options) =>
        new Promise((resolve) => {
          const timer = setTimeout(
            () => resolve({ focusedQuery: "SHOULD_NOT_BLOCK_HOOK" }),
            1000,
          );
          options?.signal?.addEventListener(
            "abort",
            () => {
              aborted = true;
              clearTimeout(timer);
            },
            { once: true },
          );
        }),
    },
  });

  assert.ok(Date.now() - started < 600);
  assert.equal(aborted, true);
  assert.equal(compiled.focusedQuery, "BlueWhaleLedger 的校验命令和幂等字段是什么？");
  assert.deepEqual(compiled.queryEntities, []);
  assert.ok(compiled.candidateSurfaces.includes("chunk"));
  assert.ok(compiled.compilerProvenance.reasons.includes("query-compile-llm-timeout"));
});

test("query compiler accepts lightweight query intent and derives entity recall surfaces", async () => {
  const result = await compileQuery({
    query: "之前 memx 的 node_modules 安装问题是什么",
    ctx: minimalCtx(),
    reasoner: {
      isEnabled: () => true,
      compileQuerySemantics: async () => ({
        shouldRecall: true,
        focusedQuery: "memx node_modules install blocker",
        queryEntities: [
          { name: "memx", type: "project", role: "subject" },
          { name: "n", type: "concept", role: "context" },
          { name: "Q", type: "concept", role: "context" },
        ],
        queryShape: {
          timeframe: "historical",
          granularity: "summary",
          referentialMode: "anchored",
          evidenceNeed: "factual_history",
        },
        primaryRoute: "factual",
      }),
    },
  });

  assert.equal(result.shouldRecall, true);
  assert.equal(Object.prototype.hasOwnProperty.call(result, "memoryUseIntent"), false);
  assert.deepEqual(result.queryEntities, [
    { name: "memx", type: "project", role: "subject" },
  ]);
  assert.equal(result.primaryRoute, "factual");
  assert.ok((result.routeWeights.factual ?? 0) > 0.5);
  assert.ok(result.candidateSurfaces.includes("entity_alias"));
  assert.ok(result.candidateSurfaces.includes("fact"));
  assert.ok(result.evidencePlan);
  assert.ok(result.evidencePlan.slots.length > 0);
  assert.ok(result.semanticBridges?.length > 0);
  assert.ok(result.evidenceGoals.length > 0);
});

test("LLM query compiler output includes a downstream retrieval contract", async () => {
  const result = await compileQuery({
    query: "继续上次的 notecheck 工程任务，找回之前的约束和进度",
    ctx: minimalCtx(),
    reasoner: {
      isEnabled: () => true,
      compileQuerySemantics: async () => ({
        focusedQuery: "notecheck 工程任务 约束 进度",
        queryEntities: [{ name: "notecheck", type: "project", role: "subject" }],
        queryShape: {
          timeframe: "current",
          granularity: "detail",
          referentialMode: "anchored",
          evidenceNeed: "workflow_context",
        },
        primaryRoute: "workflow",
      }),
    },
  });

  assert.equal(result.compilerProvenance.source, "llm");
  assert.ok(result.evidenceGoals.length > 0);
  assert.ok(result.evidencePlan);
  assert.ok(result.evidencePlan.slots.some((slot) => slot.requiredFields.length > 0));
  assert.ok(result.semanticBridges?.some((bridge) => bridge.retrievalQueries.length > 0));
});

test("query compiler ignores no-recall output and only filters invalid entity hints", async () => {
  const result = await compileQuery({
    query:
      "Determine all integers n such that every polynomial with integer coefficients is n-good.",
    ctx: minimalCtx(),
    reasoner: {
      isEnabled: () => true,
      compileQuerySemantics: async () => ({
        shouldRecall: false,
        focusedQuery: "",
        queryEntities: [
          { name: "n", type: "concept", role: "subject" },
          { name: "P", type: "concept", role: "object" },
        ],
        queryShape: {
          timeframe: "timeless",
          granularity: "summary",
          referentialMode: "anchored",
          evidenceNeed: "chunk",
        },
        primaryRoute: "none",
      }),
    },
  });

  assert.equal(result.shouldRecall, true);
  assert.deepEqual(result.queryEntities, []);
  assert.ok(result.primaryRoute);
  assert.ok(Object.keys(result.routeWeights).length > 0);
  assert.ok(result.candidateSurfaces.includes("chunk"));
});

test("query compiler prompt input compacts long queries without scaffold raw-text leakage", () => {
  const middleMarker = "MIDDLE_ONLY_RECALL_MARKER_DO_NOT_LEAK";
  const query = `${"head context ".repeat(220)}\n${middleMarker}\n${"tail instruction ".repeat(220)}`;
  const fallback = compileQueryWithoutSemanticFallback(query);

  const promptInput = buildQueryCompilerPromptInput(query, fallback);
  const serialized = JSON.stringify(promptInput);

  assert.equal(promptInput.envelope.rawLength, query.length);
  assert.ok(promptInput.envelope.rawHash);
  assert.equal(promptInput.envelope.truncated, true);
  assert.ok(serialized.length < query.length);
  assert.equal(serialized.includes(middleMarker), false);
  assert.equal(serialized.includes(query), false);
  assert.ok(promptInput.scaffold.queryText.length < 400);
  assert.ok(promptInput.scaffold.focusedQuery.length < 260);
});

test("long query recall does not run a second LLM scan in the hook hot path", async () => {
  const query = `${"alpha ".repeat(500)}\nMEMORY_MEMX_DEEPSEEK_PROVIDER_MIDDLE_ANCHOR\n${"omega ".repeat(500)}`;
  let longScanCalled = false;

  const result = await compileQuery({
    query,
    ctx: minimalCtx(),
    reasoner: {
      isEnabled: () => true,
      compileQuerySemantics: async () => ({
        shouldRecall: false,
        focusedQuery: "",
        queryEntities: [],
        queryShape: {
          timeframe: "timeless",
          granularity: "summary",
          referentialMode: "anchored",
          evidenceNeed: "chunk",
        },
        primaryRoute: "none",
      }),
      compileLongQuerySemantics: async () => {
        longScanCalled = true;
        throw new Error("long query scan must not run in before_prompt_build");
      },
    },
  });

  assert.equal(longScanCalled, false);
  assert.equal(result.shouldRecall, true);
  assert.ok(result.primaryRoute);
});

test("normalization does not synthesize reported_detail facts without LLM assertions", () => {
  const outputs = normalizeCandidate(
    {
      candidateId: "candidate_no_shadow_fact",
      source: {
        kind: "user",
        sessionKey: "s1",
      },
      observedAt,
      rawText: "OpenClaw budget is 123 and DeepSeek provider changed today.",
      normalizedText: "openclaw budget is 123 and deepseek provider changed today.",
      eventType: "conversation_turn",
      structuredHints: {
        entities: [],
        timeHints: [],
        semanticDraft: {
          sourceRef: "user:turn-no-shadow",
          assertionDrafts: [],
          correctionDrafts: [],
          supportSpans: [],
          compilerProvenance: {
            source: "llm",
            mode: "llm",
          },
        },
      },
      metadata: {
        sourceRef: "user:turn-no-shadow",
      },
      classification: "episodic-event",
      policy: {
        salienceScore: 0.9,
        expectedFutureUtility: 0.9,
        sensitivityScore: 0,
        stabilityScore: 0.9,
        action: "durable",
        reasons: ["test"],
        explicitIntent: true,
        captureAuthorized: true,
      },
      confidence: 0.9,
      scope: "agent:main",
    },
    minimalCtx(),
  );

  assert.equal(outputs.facts.some((fact) => fact.predicate === "reported_detail"), false);
});

test("normalization materializes LLM fact-like assertions even without relation drafts", () => {
  const outputs = normalizeCandidate(
    {
      candidateId: "candidate_llm_fact_assertion",
      source: {
        kind: "user",
        sessionKey: "s1",
      },
      observedAt,
      rawText: "InvoicePilot 默认数据库是 PostgreSQL。",
      normalizedText: "invoicepilot 默认数据库是 postgresql",
      eventType: "conversation_turn",
      structuredHints: {
        entities: [
          { name: "InvoicePilot", type: "project" },
          { name: "PostgreSQL", type: "service" },
        ],
        semanticDraft: {
          sourceRef: "user:turn-llm-fact",
          assertionDrafts: [
            {
              draftId: "draft-llm-fact",
              sourceRef: "user:turn-llm-fact",
              familyHint: "fact_like",
              timeframeHint: "current",
              entityHints: [
                { name: "InvoicePilot", type: "project" },
                { name: "PostgreSQL", type: "service" },
              ],
              slotHints: ["default_database"],
              valueHint: "PostgreSQL",
              confidence: 0.95,
              lineage: {
                sourceKind: "chunk",
                sourceId: "chunk-llm-fact",
                sourceRef: "user:turn-llm-fact",
              },
            },
          ],
          correctionDrafts: [],
          relationDrafts: [],
          supportSpans: [{ sourceRef: "user:turn-llm-fact", text: "InvoicePilot 默认数据库是 PostgreSQL。" }],
          compilerProvenance: {
            source: "llm",
            mode: "llm",
          },
        },
        materializationHint: {
          sourceRef: "user:turn-llm-fact",
          primaryFamily: "fact_like",
          timeframeHint: "current",
        },
      },
      metadata: {
        sourceRef: "user:turn-llm-fact",
      },
      classification: "stable-fact",
      policy: {
        salienceScore: 0.95,
        expectedFutureUtility: 0.9,
        sensitivityScore: 0,
        stabilityScore: 0.9,
        action: "stable_fact",
        reasons: ["semantic-draft-adapter:stable-fact"],
        explicitIntent: true,
        captureAuthorized: true,
      },
      confidence: 0.95,
      scope: "agent:main",
    },
    minimalCtx(),
  );

  assert.equal(outputs.facts.length, 1);
  assert.equal(outputs.facts[0].predicate, "has_default_database");
  assert.equal(outputs.facts[0].canonicalSubject, "invoicepilot");
  assert.equal(outputs.facts[0].canonicalObject, "postgresql");
  assert.match(outputs.facts[0].provenanceText, /PostgreSQL/);
  assert.ok(outputs.vectorDocs.some((doc) => doc.docKind === "fact" && /postgresql/i.test(doc.text)));
});

test("normalization materializes LLM deictic corrections onto the resolved entity slot", () => {
  const outputs = normalizeCandidate(
    {
      candidateId: "candidate_llm_deictic_update",
      source: {
        kind: "user",
        sessionKey: "s1",
      },
      observedAt,
      rawText: "这个导出格式以后改成 Arrow IPC，不要再用 Parquet。",
      normalizedText: "这个导出格式以后改成 arrow ipc 不要再用 parquet",
      eventType: "conversation_turn",
      structuredHints: {
        entities: [
          { name: "InvoicePilot", type: "project" },
          { name: "Arrow IPC", type: "concept" },
        ],
        semanticDraft: {
          sourceRef: "user:turn-llm-correction",
          assertionDrafts: [
            {
              draftId: "draft-llm-correction-assertion",
              sourceRef: "user:turn-llm-correction",
              familyHint: "fact_like",
              timeframeHint: "current",
              entityHints: [
                { name: "InvoicePilot", type: "project" },
                { name: "Arrow IPC", type: "concept" },
              ],
              slotHints: ["export_format"],
              valueHint: "Arrow IPC",
              confidence: 0.94,
              lineage: {
                sourceKind: "chunk",
                sourceId: "chunk-llm-correction",
                sourceRef: "user:turn-llm-correction",
              },
            },
          ],
          correctionDrafts: [
            {
              sourceRef: "user:turn-llm-correction",
              correction: {
                timeframe: "current",
                targetKind: "fact",
                canonicalKey: "InvoicePilot.export_format",
                predicate: "has_export_format",
                priorValue: "Parquet",
                nextValue: "Arrow IPC",
                confidence: 0.94,
              },
              confidence: 0.94,
              lineage: {
                sourceKind: "chunk",
                sourceId: "chunk-llm-correction",
                sourceRef: "user:turn-llm-correction",
              },
            },
          ],
          relationDrafts: [],
          supportSpans: [
            {
              sourceRef: "user:turn-llm-correction",
              text: "这个导出格式以后改成 Arrow IPC，不要再用 Parquet。",
            },
          ],
          compilerProvenance: {
            source: "llm",
            mode: "llm",
          },
        },
        materializationHint: {
          sourceRef: "user:turn-llm-correction",
          primaryFamily: "fact_like",
          timeframeHint: "current",
          replacementMode: "supersede_fact",
        },
        correctionHint: true,
        correction: {
          timeframe: "current",
          targetKind: "fact",
          canonicalKey: "InvoicePilot.export_format",
          predicate: "has_export_format",
          priorValue: "Parquet",
          nextValue: "Arrow IPC",
          confidence: 0.94,
        },
      },
      metadata: {
        sourceRef: "user:turn-llm-correction",
      },
      classification: "stable-fact",
      policy: {
        salienceScore: 0.95,
        expectedFutureUtility: 0.9,
        sensitivityScore: 0,
        stabilityScore: 0.9,
        action: "stable_fact",
        reasons: ["semantic-draft-adapter:fact-correction"],
        explicitIntent: true,
        captureAuthorized: true,
      },
      confidence: 0.94,
      scope: "agent:main",
    },
    minimalCtx(),
  );

  assert.ok(
    outputs.facts.some(
      (fact) =>
        fact.canonicalSubject === "invoicepilot" &&
        fact.predicate === "has_export_format" &&
        fact.canonicalObject === "arrow ipc",
    ),
  );
  assert.equal(
    outputs.facts.some(
      (fact) => fact.canonicalSubject === "user" && fact.predicate === "has_export_format",
    ),
    false,
  );
});

test("normalization canonicalizes LLM attribute predicate aliases before materialization", () => {
  const outputs = normalizeCandidate(
    {
      candidateId: "candidate_llm_attribute_alias",
      source: {
        kind: "user",
        sessionKey: "s1",
      },
      observedAt,
      rawText: "这个导出格式改成 JSONL，不再用 CSV。",
      normalizedText: "这个导出格式改成 jsonl 不再用 csv",
      eventType: "conversation_turn",
      structuredHints: {
        entities: [
          { name: "BeaconAccept", type: "project" },
          { name: "JSONL", type: "concept" },
        ],
        semanticDraft: {
          sourceRef: "user:turn-llm-alias",
          assertionDrafts: [
            {
              draftId: "draft-llm-alias-assertion",
              sourceRef: "user:turn-llm-alias",
              familyHint: "fact_like",
              timeframeHint: "current",
              entityHints: [
                { name: "BeaconAccept", type: "project" },
                { name: "JSONL", type: "concept" },
              ],
              slotHints: ["uses_export_format"],
              valueHint: "JSONL",
              confidence: 0.92,
            },
          ],
          correctionDrafts: [
            {
              sourceRef: "user:turn-llm-alias",
              correction: {
                timeframe: "current",
                targetKind: "fact",
                canonicalKey: "BeaconAccept.exportFormat",
                predicate: "uses_export_format",
                priorValue: "CSV",
                nextValue: "JSONL",
                confidence: 0.92,
              },
              confidence: 0.92,
            },
          ],
          relationDrafts: [],
          supportSpans: [{ sourceRef: "user:turn-llm-alias", text: "这个导出格式改成 JSONL，不再用 CSV。" }],
          compilerProvenance: {
            source: "llm",
            mode: "llm",
          },
        },
        materializationHint: {
          sourceRef: "user:turn-llm-alias",
          primaryFamily: "fact_like",
          timeframeHint: "current",
          replacementMode: "supersede_fact",
        },
        correctionHint: true,
        correction: {
          timeframe: "current",
          targetKind: "fact",
          canonicalKey: "BeaconAccept.exportFormat",
          predicate: "uses_export_format",
          priorValue: "CSV",
          nextValue: "JSONL",
          confidence: 0.92,
        },
      },
      metadata: {
        sourceRef: "user:turn-llm-alias",
      },
      classification: "stable-fact",
      policy: {
        salienceScore: 0.95,
        expectedFutureUtility: 0.9,
        sensitivityScore: 0,
        stabilityScore: 0.9,
        action: "stable_fact",
        reasons: ["semantic-draft-adapter:fact-correction"],
        explicitIntent: true,
        captureAuthorized: true,
      },
      confidence: 0.92,
      scope: "agent:main",
    },
    minimalCtx(),
  );

  assert.ok(
    outputs.facts.some(
      (fact) =>
        fact.canonicalSubject === "beaconaccept" &&
        fact.predicate === "has_export_format" &&
        fact.canonicalObject === "jsonl",
    ),
  );
  assert.equal(outputs.facts.some((fact) => fact.predicate === "uses_export_format"), false);
});

test("normalization canonicalizes multilingual default component slots and persists fact support lineage", () => {
  const outputs = normalizeCandidate(
    {
      candidateId: "candidate_llm_multilingual_slot_alias",
      source: {
        kind: "user",
        sessionKey: "s1",
      },
      observedAt,
      rawText: "这个就不要再考虑了，之后 BlueHarbor 支付服务默认用 Pulsar。",
      normalizedText: "这个就不要再考虑了 之后 blueharbor 支付服务默认用 pulsar",
      eventType: "conversation_turn",
      structuredHints: {
        entities: [
          { name: "BlueHarbor 支付服务", type: "service" },
          { name: "Pulsar", type: "concept" },
        ],
        semanticDraft: {
          sourceRef: "user:turn-llm-multilingual-alias",
          assertionDrafts: [
            {
              draftId: "draft-llm-multilingual-alias-assertion",
              sourceRef: "user:turn-llm-multilingual-alias",
              familyHint: "fact_like",
              timeframeHint: "current",
              entityHints: [
                { name: "BlueHarbor 支付服务", type: "service" },
                { name: "Pulsar", type: "concept" },
              ],
              slotHints: ["默认消息队列"],
              valueHint: "Pulsar",
              confidence: 0.92,
            },
          ],
          correctionDrafts: [
            {
              sourceRef: "user:turn-llm-multilingual-alias",
              correction: {
                timeframe: "current",
                targetKind: "fact",
                canonicalKey: "BlueHarbor 支付服务.default_message_queue",
                predicate: "set_default",
                priorValue: "NATS",
                nextValue: "Pulsar",
                confidence: 0.92,
              },
              confidence: 0.92,
            },
          ],
          relationDrafts: [],
          supportSpans: [
            {
              sourceRef: "user:turn-llm-multilingual-alias",
              text: "之后 BlueHarbor 支付服务默认用 Pulsar。",
            },
          ],
          compilerProvenance: {
            source: "llm",
            mode: "llm",
          },
        },
        materializationHint: {
          sourceRef: "user:turn-llm-multilingual-alias",
          primaryFamily: "fact_like",
          timeframeHint: "current",
          replacementMode: "supersede_fact",
        },
        correctionHint: true,
        correction: {
          timeframe: "current",
          targetKind: "fact",
          canonicalKey: "BlueHarbor 支付服务.default_message_queue",
          predicate: "set_default",
          priorValue: "NATS",
          nextValue: "Pulsar",
          confidence: 0.92,
        },
      },
      metadata: {
        sourceRef: "user:turn-llm-multilingual-alias",
      },
      classification: "stable-fact",
      policy: {
        salienceScore: 0.95,
        expectedFutureUtility: 0.9,
        sensitivityScore: 0,
        stabilityScore: 0.9,
        action: "stable_fact",
        reasons: ["semantic-draft-adapter:fact-correction"],
        explicitIntent: true,
        captureAuthorized: true,
      },
      confidence: 0.92,
      scope: "agent:main",
    },
    minimalCtx(),
  );

  const blueHarborFacts = outputs.facts.filter((fact) => fact.canonicalSubject === "blueharbor 支付服务");
  assert.ok(blueHarborFacts.length >= 1);
  assert.deepEqual([...new Set(blueHarborFacts.map((fact) => fact.predicate))], [
    "has_default_message_queue",
  ]);
  assert.ok(blueHarborFacts.every((fact) => fact.canonicalObject === "pulsar"));
  assert.ok(
    blueHarborFacts.every((fact) =>
      fact.objectValueJson?.supportContentRefs?.includes("user:turn-llm-multilingual-alias"),
    ),
  );
  assert.ok(
    blueHarborFacts.every((fact) =>
      String(fact.objectValueJson?.supportText ?? "").includes("BlueHarbor"),
    ),
  );
});

test("normalization maps task queue and bare CJK queue slots to the canonical message queue slot", () => {
  const outputs = normalizeCandidate(
    {
      candidateId: "candidate_llm_task_queue_alias",
      source: {
        kind: "user",
        sessionKey: "s1",
      },
      observedAt,
      rawText: "BlueHarbor 支付服务的任务队列默认用 Pulsar。",
      normalizedText: "blueharbor 支付服务的任务队列默认用 pulsar",
      eventType: "conversation_turn",
      structuredHints: {
        entities: [
          { name: "BlueHarbor 支付服务", type: "service" },
          { name: "Pulsar", type: "concept" },
        ],
        semanticDraft: {
          sourceRef: "user:turn-llm-task-queue-alias",
          assertionDrafts: [
            {
              draftId: "draft-llm-task-queue-alias-assertion",
              sourceRef: "user:turn-llm-task-queue-alias",
              familyHint: "fact_like",
              timeframeHint: "current",
              entityHints: [
                { name: "BlueHarbor 支付服务", type: "service" },
                { name: "Pulsar", type: "concept" },
              ],
              slotHints: ["default_task_queue", "队列"],
              valueHint: "Pulsar",
              confidence: 0.92,
            },
          ],
          correctionDrafts: [],
          relationDrafts: [],
          supportSpans: [
            {
              sourceRef: "user:turn-llm-task-queue-alias",
              text: "BlueHarbor 支付服务的任务队列默认用 Pulsar。",
            },
          ],
          compilerProvenance: {
            source: "llm",
            mode: "llm",
          },
        },
        materializationHint: {
          sourceRef: "user:turn-llm-task-queue-alias",
          primaryFamily: "fact_like",
          timeframeHint: "current",
        },
      },
      metadata: {
        sourceRef: "user:turn-llm-task-queue-alias",
      },
      classification: "stable-fact",
      policy: {
        salienceScore: 0.95,
        expectedFutureUtility: 0.9,
        sensitivityScore: 0,
        stabilityScore: 0.9,
        action: "stable_fact",
        reasons: ["semantic-draft-adapter:stable-fact"],
        explicitIntent: true,
        captureAuthorized: true,
      },
      confidence: 0.92,
      scope: "agent:main",
    },
    minimalCtx(),
  );

  const predicates = [...new Set(outputs.facts.map((fact) => fact.predicate))];
  assert.deepEqual(predicates, ["has_default_message_queue"]);
  assert.ok(outputs.facts.every((fact) => fact.canonicalObject === "pulsar"));
});

test("reasoner summaries and topic judgments do not rebuild semantics without LLM", async () => {
  const reasoner = new MemxReasoner(minimalConfig(), {
    debug() {},
    info() {},
    warn() {},
  });

  const chunkSummary = await reasoner.summarizeChunk(
    "记住：DeepSeek 是当前 provider，OpenClaw 是项目。",
    "user",
  );
  const taskSummary = await reasoner.summarizeTask([
    {
      chunkId: "chunk_1",
      agentId: "main",
      scope: "agent:main",
      sessionKey: "s1",
      turnId: "turn3",
      seq: 0,
      role: "user",
      chunkKind: "message",
      content: "继续 OpenClaw 的 DeepSeek provider 配置。",
      summary: "",
      contentHash: "hash_1",
      taskId: "task_1",
      dedupStatus: "active",
      mergeCount: 0,
      sourceRef: "user:turn3",
      createdAt: observedAt,
      updatedAt: observedAt,
    },
  ]);
  const topicDecision = await reasoner.judgeNewTopic(
    "OpenClaw memory plugin setup",
    "DeepSeek provider config",
  );

  assert.equal(chunkSummary, "记住：DeepSeek 是当前 provider，OpenClaw 是项目。");
  assert.equal(taskSummary.title, "Conversation task");
  assert.equal(taskSummary.summary, "");
  assert.equal(taskSummary.metadataJson.summarySource, "llm_unavailable");
  assert.equal(topicDecision, null);
});
