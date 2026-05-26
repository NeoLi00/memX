import assert from "node:assert/strict";
import test from "node:test";
import { assembleEvidencePackets } from "../dist/.runtime/src/pipeline/evidenceAssembler.mjs";
import { collectBehavioralGuidance } from "../dist/.runtime/src/pipeline/memoryObjects.mjs";
import { compileQueryWithoutSemanticFallback } from "../dist/.runtime/src/pipeline/queryCompiler.mjs";

function queryAnalysis(query, answerMode = "single_fact") {
  return {
    ...compileQueryWithoutSemanticFallback(query),
    answerMode,
    evidenceFidelity: "medium",
    evidenceCoverage: "minimal",
    supportNeed: 0.4,
    ambiguityLevel: 0.1,
    evidencePlan: {
      operation: {
        type: "return_value",
        description: "Return the value directly supported by filled evidence slots.",
      },
      slots: [
        {
          id: "query_context",
          role: "query_context",
          requiredRole: "query_context",
          description: "Subject or situation the answer must be bound to.",
          subjectHints: [query],
          relationHints: ["query context"],
          capabilityQueries: [],
          negativeHints: [],
          requiredFields: ["query_context"],
          preferredLayers: ["chunk"],
          fallbackLayers: ["chunk"],
          minEvidence: 1,
        },
        {
          id: "answer_value",
          role: "answer_value",
          requiredRole: "answer_value",
          description: "Evidence that can directly answer the query.",
          subjectHints: [query],
          relationHints: [],
          capabilityQueries: [],
          negativeHints: [],
          requiredFields: ["answer_value"],
          preferredLayers: ["chunk"],
          fallbackLayers: ["chunk"],
          minEvidence: 1,
        },
      ],
    },
    semanticBridges: [],
  };
}

test("query compiler treats distributed attribute questions as multi evidence", () => {
  const compiled = compileQueryWithoutSemanticFallback(
    "云杉票据的默认数据库、默认消息队列、失败重试策略分别是什么？",
  );

  assert.equal(compiled.answerMode, "multi_evidence");
});

test("query compiler turns single-attribute questions into attribute lookup contracts", () => {
  const compiled = compileQueryWithoutSemanticFallback("SolsticeGrid 的默认消息队列是什么？");

  assert.equal(compiled.answerMode, "attribute_lookup");
  const answerSlot = compiled.evidencePlan?.slots.find((slot) => slot.id === "answer_value");
  assert.ok(answerSlot);
  const contractText = [
    answerSlot.description,
    ...(answerSlot.relationHints ?? []),
    ...answerSlot.requiredFields,
  ].join(" ");
  assert.match(contractText, /default_message_queue|default message queue|消息队列/i);
  assert.equal(
    answerSlot.requiredFields.includes("has_default_message_queue") &&
      answerSlot.requiredFields.includes("uses_default_message_queue"),
    false,
    "predicate alternatives belong in relation hints, not simultaneously required fields",
  );
});

function chunkCandidate(query, overrides = {}) {
  return {
    id: "event:chunk:test",
    surface: "chunk",
    text: "[answer] 旧任务：Work only inside /Users/dali/.openclaw/workspace/notecheck-lab.",
    rawText: "user: Work only inside /Users/dali/.openclaw/workspace/notecheck-lab.",
    metadata: { role: "user" },
    sourceRef: "user:test",
    mergedSourceRefs: ["user:test"],
    observedAt: "2026-05-12T00:00:00.000Z",
    excerptAnchors: [query],
    priority: 0.35,
    goalScore: 0.3,
    semanticScore: 0.3,
    coverage: {
      requiredHits: [],
      missingRequired: [],
      coverageScore: 1,
      answerMode: "single_fact",
    },
    slotCoverage: [
      {
        slotId: "query_context",
        requiredHits: [],
        missingRequired: [query],
        coverageScore: 0.19,
        filled: false,
      },
      {
        slotId: "answer_value",
        requiredHits: [],
        missingRequired: [query],
        coverageScore: 0.23,
        filled: false,
      },
    ],
    filledSlotIds: [],
    injectionScore: 0.36,
    source: "candidate",
    role: "protected",
    ...overrides,
  };
}

test("unfilled stale task instructions are not injected as priority evidence", () => {
  const query = "请在当前工作区的 notecheck-lab 工程里检查中文标题/锚点修复是否完整";

  const result = assembleEvidencePackets({
    queryAnalysis: queryAnalysis(query),
    promptEvidence: [chunkCandidate(query)],
    now: "2026-05-13T00:00:00.000Z",
  });

  assert.equal(result.packets.length, 1);
  assert.equal(result.packets[0].injected, false);
  assert.equal(result.packets[0].coverage.filled, false);
  assert.equal(result.promptEvidence[0].injected, false);
  assert.equal(result.promptEvidence[0].role, "support");
});

test("low-confidence filled packets are withheld from prompt injection", () => {
  const query = "青石报表默认导出格式是什么？";

  const result = assembleEvidencePackets({
    queryAnalysis: queryAnalysis(query),
    promptEvidence: [
      chunkCandidate(query, {
        text: "[answer] RedMapNotebook uses DigestQueue.",
        rawText: "assistant: RedMapNotebook uses DigestQueue.",
        sourceRef: "assistant:old",
        mergedSourceRefs: ["assistant:old"],
        priority: 0.12,
        goalScore: 0.18,
        semanticScore: 0.16,
        injectionScore: 0.2,
        slotCoverage: [
          {
            slotId: "query_context",
            requiredHits: ["青石报表"],
            missingRequired: [],
            coverageScore: 0.32,
            filled: true,
          },
          {
            slotId: "answer_value",
            requiredHits: ["导出格式"],
            missingRequired: [],
            coverageScore: 0.34,
            filled: true,
          },
        ],
        filledSlotIds: ["query_context", "answer_value"],
      }),
    ],
    now: "2026-05-13T00:00:00.000Z",
  });

  assert.equal(result.packets.length, 1);
  assert.equal(result.packets[0].coverage.filled, true);
  assert.equal(result.packets[0].injected, false);
  assert.equal(result.promptEvidence[0].injected, false);
  assert.equal(result.promptEvidence[0].role, "support");
});

test("off-target packets are reported as dropped evidence instead of eligible recall evidence", () => {
  const query =
    "LyraLedger 的默认消息队列、导出格式、API 超时和审计日志保留时间分别是什么？";
  const result = assembleEvidencePackets({
    queryAnalysis: {
      ...queryAnalysis(query, "multi_evidence"),
      queryEntities: [{ name: "LyraLedger", type: "project", role: "subject" }],
    },
    promptEvidence: [
      chunkCandidate(query, {
        id: "fact:probequeue-default",
        surface: "fact",
        text: "probequeue has default message queue redis",
        rawText: "probequeue has default message queue redis",
        metadata: {
          recallLayer: "fact",
          canonicalSubject: "ProbeQueue",
          predicate: "has_default_message_queue",
          canonicalObject: "redis",
        },
        sourceRef: "user:turn-probequeue",
        mergedSourceRefs: ["user:turn-probequeue"],
        priority: 0.92,
        goalScore: 0.88,
        semanticScore: 0.88,
        injectionScore: 0.88,
        slotEvidenceRole: "answer_value",
        slotCoverage: [
          {
            slotId: "query_context",
            requiredHits: [],
            missingRequired: ["LyraLedger"],
            coverageScore: 0.02,
            filled: false,
          },
          {
            slotId: "answer_value",
            requiredHits: ["default message queue"],
            missingRequired: ["LyraLedger"],
            coverageScore: 0.26,
            filled: false,
          },
        ],
        filledSlotIds: ["answer_value"],
      }),
    ],
    now: "2026-05-13T00:00:00.000Z",
  });

  assert.equal(result.packets[0].injected, false);
  assert.equal(result.packets[0].dropReason, "unbound-query-context");
  assert.deepEqual(result.audit.eligibleEvidencePackets, []);
  assert.equal(result.audit.droppedEvidencePackets.length, 1);
  assert.match(result.audit.droppedEvidencePackets[0].primaryText, /probequeue/);
  assert.match(result.audit.droppedEvidencePackets[0].dropReason, /unbound-query-context/);
});

test("assistant non-answer chunks are not injected as answer evidence", () => {
  const query = "银杉账单默认数据库是什么？";

  const result = assembleEvidencePackets({
    queryAnalysis: queryAnalysis(query),
    promptEvidence: [
      chunkCandidate(query, {
        id: "assistant-non-answer",
        surface: "chunk",
        text: '我没有关于"银杉账单"系统的信息，无法在不查看文件的情况下回答。',
        rawText: 'assistant: 我没有关于"银杉账单"系统的信息，无法在不查看文件的情况下回答。',
        sourceRef: "assistant:non-answer",
        mergedSourceRefs: ["assistant:non-answer"],
        priority: 1,
        goalScore: 1,
        semanticScore: 1,
        injectionScore: 1,
        slotEvidenceRole: "answer_value",
        slotCoverage: [
          {
            slotId: "query_context",
            requiredHits: ["银杉账单"],
            missingRequired: [],
            coverageScore: 0.9,
            filled: true,
          },
          {
            slotId: "answer_value",
            requiredHits: ["默认数据库"],
            missingRequired: [],
            coverageScore: 0.9,
            filled: true,
          },
        ],
        filledSlotIds: ["query_context", "answer_value"],
      }),
      chunkCandidate(query, {
        id: "fact-yinsan-db",
        surface: "fact",
        text: "银杉账单 has default database mysql",
        rawText: "银杉账单 has default database mysql",
        metadata: { recallLayer: "fact" },
        sourceRef: "user:yinsan-db",
        mergedSourceRefs: ["user:yinsan-db"],
        priority: 0.62,
        goalScore: 0.62,
        semanticScore: 0.62,
        injectionScore: 0.62,
        slotEvidenceRole: "answer_value",
        slotCoverage: [
          {
            slotId: "query_context",
            requiredHits: ["银杉账单"],
            missingRequired: [],
            coverageScore: 0.82,
            filled: true,
          },
          {
            slotId: "answer_value",
            requiredHits: ["默认数据库", "mysql"],
            missingRequired: [],
            coverageScore: 0.82,
            filled: true,
          },
        ],
        filledSlotIds: ["query_context", "answer_value"],
      }),
    ],
    now: "2026-05-13T00:00:00.000Z",
  });

  const injectedLines = result.packets
    .filter((packet) => packet.injected)
    .flatMap((packet) => packet.displayLines ?? []);
  assert.match(injectedLines.join("\n"), /default database mysql/);
  assert.doesNotMatch(injectedLines.join("\n"), /没有关于/);
});

test("assistant-authored chunks cannot satisfy canonical attribute lookup answers", () => {
  const query = "MeridianForge 现在告警渠道是什么？";
  const analysis = compileQueryWithoutSemanticFallback(query);
  assert.equal(analysis.answerMode, "attribute_lookup");
  const result = assembleEvidencePackets({
    queryAnalysis: analysis,
    promptEvidence: [
      chunkCandidate(query, {
        id: "event:chunk:meridian-assistant-old",
        surface: "chunk",
        text: "MeridianForge 默认告警渠道是 Email/Webhook。",
        rawText: "assistant: MeridianForge 默认告警渠道是 Email/Webhook。",
        metadata: { role: "assistant" },
        sourceRef: "assistant:turn-meridian-old",
        mergedSourceRefs: ["assistant:turn-meridian-old"],
        priority: 1,
        goalScore: 1,
        semanticScore: 1,
        injectionScore: 1,
        slotEvidenceRole: "answer_value",
        slotCoverage: [
          {
            slotId: "query_context",
            requiredHits: ["MeridianForge"],
            missingRequired: [],
            coverageScore: 0.96,
            filled: true,
          },
          {
            slotId: "answer_value",
            requiredHits: ["告警渠道", "Email/Webhook"],
            missingRequired: [],
            coverageScore: 0.96,
            filled: true,
          },
        ],
        filledSlotIds: ["query_context", "answer_value"],
      }),
    ],
    now: "2026-05-13T00:00:00.000Z",
  });

  assert.equal(result.packets[0].injected, false);
  assert.equal(result.promptEvidence[0].injected, false);
  assert.equal(result.promptEvidence[0].role, "support");
  assert.match(result.packets[0].dropReason ?? "", /assistant-authored|missing-answer-value/);
});

test("filled evidence packets are still injected", () => {
  const query = "请检查 notecheck-lab 中文锚点修复是否通过测试";
  const result = assembleEvidencePackets({
    queryAnalysis: queryAnalysis(query),
    promptEvidence: [
      chunkCandidate(query, {
        text: "notecheck 中文锚点修复已通过 62 项测试。",
        rawText: "assistant: notecheck 中文锚点修复已通过 62 项测试。",
        priority: 0.7,
        goalScore: 0.72,
        semanticScore: 0.74,
        injectionScore: 0.74,
        slotEvidenceRole: "answer_value",
        slotCoverage: [
          {
            slotId: "query_context",
            requiredHits: [query],
            missingRequired: [],
            coverageScore: 0.9,
            filled: true,
          },
          {
            slotId: "answer_value",
            requiredHits: ["通过 62 项测试"],
            missingRequired: [],
            coverageScore: 0.95,
            filled: true,
          },
        ],
        filledSlotIds: ["query_context", "answer_value"],
      }),
    ],
    now: "2026-05-13T00:00:00.000Z",
  });

  assert.equal(result.packets.length, 1);
  assert.equal(result.packets[0].injected, true);
  assert.equal(result.packets[0].coverage.filled, true);
  assert.equal(result.promptEvidence[0].injected, true);
  assert.equal(result.promptEvidence[0].role, "protected");
});

test("canonical fact is preferred over raw turn chunk for direct fact answers", () => {
  const query = "CedarLedger 的默认缓存是什么？";
  const fact = chunkCandidate(query, {
    id: "fact:cedar-cache",
    surface: "fact",
    text: "CedarLedger has default cache redis",
    rawText: "CedarLedger has default cache redis",
    metadata: { recallLayer: "fact" },
    sourceRef: "user:turn-cedar-cache",
    mergedSourceRefs: ["user:turn-cedar-cache"],
    priority: 0.62,
    goalScore: 0.64,
    semanticScore: 0.64,
    injectionScore: 0.64,
    slotEvidenceRole: "answer_value",
    slotCoverage: [
      {
        slotId: "query_context",
        requiredHits: ["CedarLedger"],
        missingRequired: [],
        coverageScore: 0.78,
        filled: true,
      },
      {
        slotId: "answer_value",
        requiredHits: ["默认缓存", "Redis"],
        missingRequired: [],
        coverageScore: 0.78,
        filled: true,
      },
    ],
    filledSlotIds: ["query_context", "answer_value"],
  });
  const rawChunk = chunkCandidate(query, {
    id: "event:chunk:cedar-cache",
    surface: "chunk",
    text: "请记住：CedarLedger 的默认缓存是 Redis。",
    rawText: "user: 请记住：CedarLedger 的默认缓存是 Redis。",
    metadata: { role: "user" },
    sourceRef: "user:turn-cedar-cache",
    mergedSourceRefs: ["user:turn-cedar-cache"],
    priority: 0.9,
    goalScore: 0.86,
    semanticScore: 0.86,
    injectionScore: 0.86,
    slotEvidenceRole: "answer_value",
    slotCoverage: [
      {
        slotId: "query_context",
        requiredHits: ["CedarLedger"],
        missingRequired: [],
        coverageScore: 0.86,
        filled: true,
      },
      {
        slotId: "answer_value",
        requiredHits: ["默认缓存", "Redis"],
        missingRequired: [],
        coverageScore: 0.86,
        filled: true,
      },
    ],
    filledSlotIds: ["query_context", "answer_value"],
  });

  const result = assembleEvidencePackets({
    queryAnalysis: queryAnalysis(query),
    promptEvidence: [rawChunk, fact],
    now: "2026-05-13T00:00:00.000Z",
  });

  const injected = result.packets.find((packet) => packet.injected);
  assert.ok(injected);
  assert.equal(injected.answerCandidate.surface, "fact");
  assert.match(injected.displayLines.join("\n"), /CedarLedger has default cache redis/);
  assert.doesNotMatch(injected.displayLines.join("\n"), /请记住/);
});

test("attribute lookup does not inject a same-subject fact for the wrong requested slot", () => {
  const query = "SolsticeGrid 的默认消息队列是什么？";
  const analysis = compileQueryWithoutSemanticFallback(query);
  const wrongFact = chunkCandidate(query, {
    id: "fact:solstice-regression-command",
    surface: "fact",
    text: "solsticegrid has regression command npm run verify:solstice",
    rawText: "solsticegrid has regression command npm run verify:solstice",
    metadata: {
      recallLayer: "fact",
      canonicalSubject: "SolsticeGrid",
      predicate: "has_regression_command",
      canonicalObject: "npm run verify:solstice",
    },
    sourceRef: "user:turn-solstice-command",
    mergedSourceRefs: ["user:turn-solstice-command"],
    priority: 1,
    goalScore: 1,
    semanticScore: 1,
    injectionScore: 1,
    slotEvidenceRole: "answer_value",
    slotCoverage: [
      {
        slotId: "query_context",
        requiredHits: ["SolsticeGrid"],
        missingRequired: [],
        coverageScore: 0.96,
        filled: true,
      },
      {
        slotId: "answer_value",
        requiredHits: ["SolsticeGrid", "默认消息队列"],
        missingRequired: [],
        coverageScore: 0.96,
        filled: true,
      },
    ],
    filledSlotIds: ["query_context", "answer_value"],
  });
  const rightFact = chunkCandidate(query, {
    id: "fact:solstice-default-message-queue",
    surface: "fact",
    text: "solsticegrid has default message queue rabbitmq",
    rawText: "solsticegrid has default message queue rabbitmq",
    metadata: {
      recallLayer: "fact",
      canonicalSubject: "SolsticeGrid",
      predicate: "has_default_message_queue",
      canonicalObject: "rabbitmq",
    },
    sourceRef: "user:turn-solstice-queue",
    mergedSourceRefs: ["user:turn-solstice-queue"],
    priority: 0.54,
    goalScore: 0.52,
    semanticScore: 0.52,
    injectionScore: 0.54,
    slotEvidenceRole: "answer_value",
    slotCoverage: [
      {
        slotId: "query_context",
        requiredHits: ["SolsticeGrid"],
        missingRequired: [],
        coverageScore: 0.78,
        filled: true,
      },
      {
        slotId: "answer_value",
        requiredHits: ["default message queue", "rabbitmq"],
        missingRequired: [],
        coverageScore: 0.78,
        filled: true,
      },
    ],
    filledSlotIds: ["query_context", "answer_value"],
  });

  const result = assembleEvidencePackets({
    queryAnalysis: analysis,
    promptEvidence: [wrongFact, rightFact],
    now: "2026-05-13T00:00:00.000Z",
  });

  const injectedText = result.packets
    .filter((packet) => packet.injected)
    .flatMap((packet) => packet.displayLines ?? [])
    .join("\n");
  assert.match(injectedText, /default message queue rabbitmq/i);
  assert.doesNotMatch(injectedText, /regression command/i);
});

test("multi-attribute factual questions inject sibling canonical facts", () => {
  const query = "云杉票据的默认数据库、默认消息队列、失败重试策略分别是什么？";
  const fact = (id, text, relation, object) =>
    chunkCandidate(query, {
      id,
      surface: "fact",
      text,
      rawText: text,
      metadata: { recallLayer: "fact", canonicalSubject: "云杉票据", predicate: relation, canonicalObject: object },
      sourceRef: "user:turn-yunsan",
      mergedSourceRefs: ["user:turn-yunsan"],
      priority: 0.72,
      goalScore: 0.72,
      semanticScore: 0.72,
      injectionScore: 0.72,
      slotEvidenceRole: "answer_value",
      slotCoverage: [
        {
          slotId: "query_context",
          requiredHits: ["云杉票据"],
          missingRequired: [],
          coverageScore: 0.86,
          filled: true,
        },
        {
          slotId: "answer_value",
          requiredHits: [relation, object],
          missingRequired: [],
          coverageScore: 0.82,
          filled: true,
        },
      ],
      filledSlotIds: ["query_context", "answer_value"],
    });

  const result = assembleEvidencePackets({
    queryAnalysis: queryAnalysis(query, "multi_evidence"),
    promptEvidence: [
      fact("fact-yunsan-db", "云杉票据 has default database mysql", "has_default_database", "mysql"),
      fact("fact-yunsan-queue", "云杉票据 has default message queue pulsar", "has_default_message_queue", "pulsar"),
      fact("fact-yunsan-retry", "云杉票据 has retry strategy exponential_backoff", "has_retry_strategy", "exponential_backoff"),
    ],
    now: "2026-05-13T00:00:00.000Z",
  });

  const injectedLines = result.packets
    .filter((packet) => packet.injected)
    .flatMap((packet) => packet.displayLines ?? []);
  assert.equal(injectedLines.length, 3);
  assert.match(injectedLines.join("\n"), /default database mysql/);
  assert.match(injectedLines.join("\n"), /default message queue pulsar/);
  assert.match(injectedLines.join("\n"), /retry strategy exponential_backoff/);
});

test("multi-attribute factual recall prefers canonical facts over same-source assistant restatements", () => {
  const query =
    "LyraLedger 的默认消息队列、导出格式、API 超时和审计日志保留时间分别是什么？";
  const sourceRef = "user:turn-lyra";
  const fact = (id, text, relation, object) =>
    chunkCandidate(query, {
      id,
      surface: "fact",
      text,
      rawText: text,
      metadata: { recallLayer: "fact", canonicalSubject: "LyraLedger", predicate: relation, canonicalObject: object },
      sourceRef,
      mergedSourceRefs: [sourceRef],
      priority: 0.76,
      goalScore: 0.74,
      semanticScore: 0.74,
      injectionScore: 0.74,
      slotEvidenceRole: "answer_value",
      slotCoverage: [
        {
          slotId: "query_context",
          requiredHits: ["LyraLedger"],
          missingRequired: [],
          coverageScore: 0.86,
          filled: true,
        },
        {
          slotId: "answer_value",
          requiredHits: [relation, object],
          missingRequired: [],
          coverageScore: 0.84,
          filled: true,
        },
      ],
      filledSlotIds: ["query_context", "answer_value"],
    });
  const assistantRestatement = chunkCandidate(query, {
    id: "event:chunk:lyra-assistant",
    surface: "chunk",
    text: "我理解的 LyraLedger 约束如下：默认消息队列 Kafka，默认导出格式 Parquet，API 超时 14 秒，审计日志保留 45 天。",
    rawText:
      "assistant: 我理解的 LyraLedger 约束如下：默认消息队列 Kafka，默认导出格式 Parquet，API 超时 14 秒，审计日志保留 45 天。",
    metadata: { role: "assistant" },
    sourceRef: "assistant:turn-lyra",
    mergedSourceRefs: [sourceRef, "assistant:turn-lyra"],
    priority: 1,
    goalScore: 1,
    semanticScore: 1,
    injectionScore: 1,
    slotEvidenceRole: "answer_value",
    slotCoverage: [
      {
        slotId: "query_context",
        requiredHits: ["LyraLedger"],
        missingRequired: [],
        coverageScore: 0.9,
        filled: true,
      },
      {
        slotId: "answer_value",
        requiredHits: ["Kafka", "Parquet", "14 秒", "45 天"],
        missingRequired: [],
        coverageScore: 0.92,
        filled: true,
      },
    ],
    filledSlotIds: ["query_context", "answer_value"],
  });

  const result = assembleEvidencePackets({
    queryAnalysis: queryAnalysis(query, "multi_evidence"),
    promptEvidence: [
      assistantRestatement,
      fact("fact-lyra-queue", "lyraledger has default message queue kafka", "has_default_message_queue", "kafka"),
      fact("fact-lyra-format", "lyraledger has export format parquet", "has_export_format", "parquet"),
      fact("fact-lyra-timeout", "lyraledger has api timeout 14s", "has_api_timeout", "14s"),
      fact(
        "fact-lyra-retention",
        "lyraledger has audit log retention days 45 days",
        "has_audit_log_retention_days",
        "45 days",
      ),
    ],
    now: "2026-05-13T00:00:00.000Z",
  });

  const injected = result.packets.filter((packet) => packet.injected);
  const injectedText = injected.flatMap((packet) => packet.displayLines ?? []).join("\n");
  assert.equal(injected.length, 1);
  assert.equal(injected[0].answerCandidate.surface, "fact");
  assert.match(injectedText, /default message queue kafka/);
  assert.match(injectedText, /export format parquet/);
  assert.match(injectedText, /api timeout 14s/);
  assert.match(injectedText, /audit log retention days 45 days/);
  assert.doesNotMatch(injectedText, /我理解的 LyraLedger/);
});

test("multi-attribute canonical facts hide raw support text from prompt injection", () => {
  const query = "PineFlowMature 现在默认用什么消息队列？归档格式是什么？";
  const queryAnalysis = compileQueryWithoutSemanticFallback(query);
  const oldSourceRef = "user:pineflow-initial";
  const updateSourceRef = "user:pineflow-update";
  const slotCoverage = (answerSlotId, score) => [
    {
      slotId: "query_context",
      requiredHits: ["PineFlowMature"],
      missingRequired: [],
      coverageScore: 0.62,
      filled: true,
    },
    {
      slotId: "answer_archive_format",
      requiredHits: ["archive_format"],
      missingRequired: [],
      coverageScore: answerSlotId === "answer_archive_format" ? score : 0.57,
      filled: true,
    },
    {
      slotId: "answer_default_message_queue",
      requiredHits: ["default_message_queue"],
      missingRequired: [],
      coverageScore: answerSlotId === "answer_default_message_queue" ? score : 0.57,
      filled: true,
    },
  ];
  const fact = ({ id, text, sourceRef, predicate, object, answerSlotId }) =>
    chunkCandidate(query, {
      id,
      surface: "fact",
      text,
      rawText: text,
      metadata: {
        recallLayer: "fact",
        canonicalSubject: "PineFlowMature",
        predicate,
        canonicalObject: object,
        supportRefs: [sourceRef],
      },
      sourceRef,
      mergedSourceRefs: [sourceRef],
      priority: 0.75,
      goalScore: 0.74,
      semanticScore: 0.74,
      injectionScore: 0.75,
      slotEvidenceRole: "answer_value",
      slotCoverage: slotCoverage(answerSlotId, 0.74),
      filledSlotIds: ["query_context", "answer_archive_format", "answer_default_message_queue"],
    });
  const raw = ({ id, text, sourceRef }) =>
    chunkCandidate(query, {
      id,
      surface: "chunk",
      text,
      rawText: `user: ${text}`,
      metadata: { role: "user" },
      sourceRef,
      mergedSourceRefs: [sourceRef],
      priority: 1,
      goalScore: 1,
      semanticScore: 1,
      injectionScore: 1,
      slotEvidenceRole: "answer_value",
      slotCoverage: slotCoverage("answer_default_message_queue", 0.58),
      filledSlotIds: ["query_context", "answer_archive_format", "answer_default_message_queue"],
    });

  const result = assembleEvidencePackets({
    queryAnalysis,
    promptEvidence: [
      raw({
        id: "event:chunk:initial",
        sourceRef: oldSourceRef,
        text: "请记住：PineFlowMature 默认消息队列是 NATS，归档格式是 Parquet。",
      }),
      raw({
        id: "event:chunk:update",
        sourceRef: updateSourceRef,
        text: "PineFlowMature 的队列选择不要再沿用，后面默认改成 Pulsar。归档格式保持不变。",
      }),
      fact({
        id: "fact:archive",
        sourceRef: oldSourceRef,
        text: "pineflowmature has archive format parquet",
        predicate: "has_archive_format",
        object: "parquet",
        answerSlotId: "answer_archive_format",
      }),
      fact({
        id: "fact:queue",
        sourceRef: updateSourceRef,
        text: "pineflowmature has default message queue pulsar",
        predicate: "has_default_message_queue",
        object: "pulsar",
        answerSlotId: "answer_default_message_queue",
      }),
    ],
    now: "2026-05-26T00:00:00.000Z",
  });

  const injectedText = result.packets
    .filter((packet) => packet.injected)
    .flatMap((packet) => packet.displayLines ?? [])
    .join("\n");
  assert.match(injectedText, /archive format parquet/i);
  assert.match(injectedText, /default message queue pulsar/i);
  assert.doesNotMatch(injectedText, /NATS/i);
  assert.doesNotMatch(injectedText, /请记住|队列选择不要再沿用/);
});

test("canonical facts suppress cross-source assistant restatements", () => {
  const query = "PineFlowMature 现在默认用什么消息队列？归档格式是什么？";
  const sourceRef = "user:pineflow-update";
  const fact = (id, text, predicate, object) =>
    chunkCandidate(query, {
      id,
      surface: "fact",
      text,
      rawText: text,
      metadata: {
        recallLayer: "fact",
        canonicalSubject: "PineFlowMature",
        predicate,
        canonicalObject: object,
      },
      sourceRef,
      mergedSourceRefs: [sourceRef],
      priority: 0.76,
      goalScore: 0.74,
      semanticScore: 0.74,
      injectionScore: 0.74,
      slotEvidenceRole: "answer_value",
      slotCoverage: [
        {
          slotId: "query_context",
          requiredHits: ["PineFlowMature"],
          missingRequired: [],
          coverageScore: 0.86,
          filled: true,
        },
        {
          slotId: "answer_value",
          requiredHits: [predicate, object],
          missingRequired: [],
          coverageScore: 0.84,
          filled: true,
        },
      ],
      filledSlotIds: ["query_context", "answer_value"],
    });
  const assistantRestatement = chunkCandidate(query, {
    id: "event:chunk:assistant-update",
    surface: "chunk",
    text: "已更新：PineFlowMature 后续默认消息队列改为 Pulsar，归档格式仍保持 Parquet。",
    rawText:
      "assistant: 已更新：PineFlowMature 后续默认消息队列改为 Pulsar，归档格式仍保持 Parquet。",
    metadata: { role: "assistant" },
    sourceRef: "assistant:pineflow-update",
    mergedSourceRefs: ["assistant:pineflow-update"],
    priority: 1,
    goalScore: 1,
    semanticScore: 1,
    injectionScore: 1,
    slotEvidenceRole: "answer_value",
    slotCoverage: [
      {
        slotId: "query_context",
        requiredHits: ["PineFlowMature"],
        missingRequired: [],
        coverageScore: 0.9,
        filled: true,
      },
      {
        slotId: "answer_value",
        requiredHits: ["Pulsar", "Parquet"],
        missingRequired: [],
        coverageScore: 0.92,
        filled: true,
      },
    ],
    filledSlotIds: ["query_context", "answer_value"],
  });

  const result = assembleEvidencePackets({
    queryAnalysis: queryAnalysis(query, "multi_evidence"),
    promptEvidence: [
      assistantRestatement,
      fact(
        "fact-pineflow-queue",
        "pineflowmature has default message queue pulsar",
        "has_default_message_queue",
        "pulsar",
      ),
      fact("fact-pineflow-archive", "pineflowmature has archive format parquet", "has_archive_format", "parquet"),
    ],
    now: "2026-05-26T00:00:00.000Z",
  });

  const injectedText = result.packets
    .filter((packet) => packet.injected)
    .flatMap((packet) => packet.displayLines ?? [])
    .join("\n");
  assert.match(injectedText, /default message queue pulsar/i);
  assert.match(injectedText, /archive format parquet/i);
  assert.doesNotMatch(injectedText, /已更新/);
});

test("multi-attribute recall can promote weak same-source facts through raw turn support", () => {
  const query = "银杉账单的默认数据库、默认消息队列、失败重试策略分别是什么？";
  const sourceRef = "user:turn-yinsan";
  const weakFact = (id, text, relation, object) =>
    chunkCandidate(query, {
      id,
      surface: "fact",
      text,
      rawText: text,
      metadata: {
        recallLayer: "fact",
        canonicalSubject: "银杉账单",
        predicate: relation,
        canonicalObject: object,
        supportRefs: [sourceRef],
      },
      sourceRef,
      mergedSourceRefs: [sourceRef],
      priority: 0.22,
      goalScore: 0.2,
      semanticScore: 0.2,
      injectionScore: 0.22,
      slotEvidenceRole: undefined,
      slotCoverage: [],
      filledSlotIds: [],
    });
  const rawTurn = chunkCandidate(query, {
    id: "event:chunk:yinsan-write",
    surface: "chunk",
    text: "不用查看文件。我们在做一个账单同步服务的接口设计。请记住：银杉账单的默认数据库是 MySQL，默认消息队列是 Pulsar，失败重试策略是指数退避。只回复：已记录。",
    rawText:
      "user: 不用查看文件。我们在做一个账单同步服务的接口设计。请记住：银杉账单的默认数据库是 MySQL，默认消息队列是 Pulsar，失败重试策略是指数退避。只回复：已记录。",
    metadata: { role: "user" },
    sourceRef,
    mergedSourceRefs: [sourceRef],
    priority: 0.92,
    goalScore: 0.86,
    semanticScore: 0.86,
    injectionScore: 0.86,
    slotEvidenceRole: "answer_value",
    slotCoverage: [
      {
        slotId: "query_context",
        requiredHits: ["银杉账单"],
        missingRequired: [],
        coverageScore: 0.82,
        filled: true,
      },
      {
        slotId: "answer_value",
        requiredHits: ["默认数据库", "默认消息队列", "失败重试策略"],
        missingRequired: [],
        coverageScore: 0.86,
        filled: true,
      },
    ],
    filledSlotIds: ["query_context", "answer_value"],
  });
  const queryEcho = chunkCandidate(query, {
    id: "event:chunk:yinsan-query",
    surface: "chunk",
    text: "不用查看文件。银杉账单的默认数据库、默认消息队列、失败重试策略分别是什么？只用一行中文回答。",
    rawText:
      "user: 不用查看文件。银杉账单的默认数据库、默认消息队列、失败重试策略分别是什么？只用一行中文回答。",
    sourceRef: "user:turn-yinsan-query",
    mergedSourceRefs: ["user:turn-yinsan-query"],
    priority: 1,
    goalScore: 1,
    semanticScore: 1,
    injectionScore: 1,
    slotEvidenceRole: "answer_value",
    filledSlotIds: ["query_context", "answer_value"],
  });
  const assistantNonAnswer = chunkCandidate(query, {
    id: "event:chunk:yinsan-non-answer",
    surface: "chunk",
    text: '我没有关于"银杉账单"系统的信息，无法在不查看文件的情况下回答。',
    rawText: 'assistant: 我没有关于"银杉账单"系统的信息，无法在不查看文件的情况下回答。',
    sourceRef: "assistant:turn-yinsan-non-answer",
    mergedSourceRefs: ["assistant:turn-yinsan-non-answer"],
    priority: 0.95,
    goalScore: 0.95,
    semanticScore: 0.95,
    injectionScore: 0.95,
    slotEvidenceRole: "answer_value",
    filledSlotIds: ["query_context", "answer_value"],
  });

  const result = assembleEvidencePackets({
    queryAnalysis: queryAnalysis(query, "multi_evidence"),
    promptEvidence: [
      queryEcho,
      assistantNonAnswer,
      rawTurn,
      weakFact("fact-yinsan-db", "银杉账单 has default database mysql", "has_default_database", "mysql"),
      weakFact(
        "fact-yinsan-queue",
        "银杉账单 has default message queue pulsar",
        "has_default_message_queue",
        "pulsar",
      ),
      weakFact(
        "fact-yinsan-retry",
        "银杉账单 has retry strategy 指数退避",
        "has_retry_strategy",
        "指数退避",
      ),
    ],
    now: "2026-05-13T00:00:00.000Z",
  });

  const injectedLines = result.packets
    .filter((packet) => packet.injected)
    .flatMap((packet) => packet.displayLines ?? []);
  const rendered = injectedLines.join("\n");
  assert.match(rendered, /default database mysql/);
  assert.match(rendered, /default message queue pulsar/);
  assert.match(rendered, /retry strategy 指数退避/);
  assert.doesNotMatch(rendered, /没有关于/);
  assert.doesNotMatch(rendered, /分别是什么/);
});

test("canonical fact can use same-source raw text as hidden support for direct fact answers", () => {
  const query = "MapleLedger 的默认队列是什么？";
  const fact = chunkCandidate(query, {
    id: "fact:maple-queue",
    surface: "fact",
    text: "mapleledger has default queue keydb",
    rawText: "mapleledger has default queue keydb",
    metadata: { recallLayer: "fact", supportText: "fact_like current MapleLedger KeyDB default_queue" },
    sourceRef: "user:turn-maple-queue",
    mergedSourceRefs: ["user:turn-maple-queue"],
    priority: 0.42,
    goalScore: 0.4,
    semanticScore: 0.4,
    injectionScore: 0.42,
    slotEvidenceRole: "answer_value",
    slotCoverage: [
      {
        slotId: "query_context",
        requiredHits: [],
        missingRequired: ["MapleLedger 默认队列"],
        coverageScore: 0.34,
        filled: false,
      },
      {
        slotId: "answer_value",
        requiredHits: [],
        missingRequired: ["MapleLedger 默认队列"],
        coverageScore: 0.38,
        filled: false,
      },
    ],
    filledSlotIds: [],
  });
  const rawChunk = chunkCandidate(query, {
    id: "event:chunk:maple-queue",
    surface: "chunk",
    text: "请记住：MapleLedger 的默认队列是 KeyDB。",
    rawText: "user: 请记住：MapleLedger 的默认队列是 KeyDB。",
    metadata: { role: "user" },
    sourceRef: "user:turn-maple-queue",
    mergedSourceRefs: ["user:turn-maple-queue"],
    priority: 0.96,
    goalScore: 0.9,
    semanticScore: 0.9,
    injectionScore: 0.9,
    slotEvidenceRole: "answer_value",
    slotCoverage: [
      {
        slotId: "query_context",
        requiredHits: ["MapleLedger 默认队列", query],
        missingRequired: [],
        coverageScore: 0.82,
        filled: true,
      },
      {
        slotId: "answer_value",
        requiredHits: ["MapleLedger 默认队列", query],
        missingRequired: [],
        coverageScore: 0.86,
        filled: true,
      },
    ],
    filledSlotIds: ["query_context", "answer_value"],
  });

  const result = assembleEvidencePackets({
    queryAnalysis: queryAnalysis(query),
    promptEvidence: [rawChunk, fact],
    now: "2026-05-13T00:00:00.000Z",
  });

  const injected = result.packets.find((packet) => packet.injected);
  assert.ok(injected);
  assert.equal(injected.answerCandidate.surface, "fact");
  assert.match(injected.displayLines.join("\n"), /mapleledger has default queue keydb/);
  assert.doesNotMatch(injected.displayLines.join("\n"), /请记住/);
  assert.deepEqual(result.packets.filter((packet) => packet.injected), [injected]);
});

test("task-scoped workflow guidance is excluded from ambient reply guidance", () => {
  const workflowFact = {
    factId: "workflow_fact",
    canonicalSubject: "user",
    predicate: "has_workflow_guidance",
    objectValueJson: {
      guidance: {
        guidanceType: "generic_preference",
        guidanceText: "When this workflow pattern applies, solve the old math problem this way.",
      },
    },
    updatedAt: "2026-05-12T00:00:00.000Z",
  };
  const languageFact = {
    factId: "language_fact",
    canonicalSubject: "user",
    predicate: "prefers_language",
    objectValueJson: {
      guidance: {
        guidanceType: "language",
        guidanceText: "Default to Chinese responses unless the current turn asks otherwise.",
      },
    },
    updatedAt: "2026-05-13T00:00:00.000Z",
  };
  const store = {
    beliefRepo: {
      listByAgent() {
        return [];
      },
    },
    factRepo: {
      query() {
        return [workflowFact, languageFact];
      },
    },
  };

  assert.deepEqual(
    collectBehavioralGuidance(store, {
      agentId: "main",
      scopes: ["agent:main"],
    }),
    ["Default to Chinese responses unless the current turn asks otherwise."],
  );
});
