import assert from "node:assert/strict";
import test from "node:test";
import { assembleEvidencePackets } from "../dist/.runtime/src/pipeline/evidenceAssembler.mjs";
import { collectBehavioralGuidance } from "../dist/.runtime/src/pipeline/memoryObjects.mjs";
import { compileQueryWithoutSemanticFallback } from "../dist/.runtime/src/pipeline/queryCompiler.mjs";

function queryAnalysis(query) {
  return {
    ...compileQueryWithoutSemanticFallback(query),
    answerMode: "single_fact",
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
