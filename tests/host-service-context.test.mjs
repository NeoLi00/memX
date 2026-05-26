import assert from "node:assert/strict";
import test from "node:test";

function row(id, text) {
  return {
    id,
    text,
    score: 0.9,
    scope: "agent:test",
    observedAt: "2026-05-21T00:00:00.000Z",
  };
}

function packet(packetId, text, entityAliases = []) {
  return {
    packetId,
    slotId: `${packetId}-slot`,
    operationType: "return_value",
    role: "answer",
    injected: true,
    primaryText: text,
    supportingTexts: [],
    sourceRefs: [`fact:${packetId}`],
    entityAliases,
    coverage: { filled: true, missing: [], confidence: 0.9 },
    grade: {
      retrievalScore: 0.9,
      answerScore: 0.9,
      contextBindingScore: 0.9,
      slotCoverageScore: 0.9,
      authorityScore: 0.9,
      finalScore: 0.9,
    },
  };
}

test("native recall context can focus injected evidence on LLM query entities", async () => {
  const { focusRecallBundleForQueryEntities } = await import(
    "../dist/.runtime/src/host/service.mjs"
  );
  assert.equal(typeof focusRecallBundleForQueryEntities, "function");

  const bundle = {
    routeType: "mixed",
    routeConfidence: 0.9,
    states: [row("state-redmap", "RedMapNotebook 默认队列是 NATS")],
    tasks: [],
    facts: [
      row("fact-claude", "ClaudeProbe 默认输出格式是 Parquet"),
      row("fact-redmap", "RedMapNotebook 默认输出格式是 SQLite"),
    ],
    events: [
      row("event-claude", "用户在 ClaudeProbe 任务里确认了 Parquet"),
      row("event-redmap", "用户在 RedMapNotebook 任务里讨论了 DigestQueue"),
    ],
    graph: {
      nodes: [],
      edges: [],
      pathCandidates: [],
      paths: [
        "ClaudeProbe -> default format -> Parquet",
        "RedMapNotebook -> default queue -> NATS",
      ],
    },
    alternates: [],
    diagnostics: [],
    behavioralGuidance: [
      "ClaudeProbe 相关问题优先回答 Parquet。",
      "RedMapNotebook 相关问题优先检查 DigestQueue。",
    ],
    recalledChunkIds: [],
    recalledChunkTexts: [],
    promptEvidence: [],
    evidencePackets: [
      packet("packet-claude", "ClaudeProbe 默认输出格式是 Parquet", ["ClaudeProbe"]),
      packet("packet-redmap", "RedMapNotebook 默认输出格式是 SQLite", ["RedMapNotebook"]),
    ],
    renderedBlock: "",
  };
  const focused = focusRecallBundleForQueryEntities(
    {
      queryEntities: [{ name: "ClaudeProbe", type: "project", role: "subject" }],
      suppressedEntities: [],
    },
    bundle,
  );

  assert.deepEqual(
    focused.facts.map((entry) => entry.id),
    ["fact-claude"],
  );
  assert.deepEqual(
    focused.events.map((entry) => entry.id),
    ["event-claude"],
  );
  assert.deepEqual(
    focused.states.map((entry) => entry.id),
    [],
  );
  assert.deepEqual(focused.graph.paths, ["ClaudeProbe -> default format -> Parquet"]);
  assert.deepEqual(focused.behavioralGuidance, ["ClaudeProbe 相关问题优先回答 Parquet。"]);
  assert.deepEqual(
    focused.evidencePackets.map((entry) => entry.packetId),
    ["packet-claude"],
  );
});

test("native recall focus accepts compound entity aliases when strong evidence mentions one alias", async () => {
  const { assessNativeContextEligibility, focusRecallBundleForQueryEntities } = await import(
    "../dist/.runtime/src/host/service.mjs"
  );
  const query = "FrostBridge/霜桥同步 的 default queue、API timeout 和导出格式分别是什么？";
  const bundle = {
    routeType: "factual",
    routeConfidence: 0.91,
    queryText: query,
    queryAnchors: ["FrostBridge/霜桥同步"],
    states: [],
    tasks: [],
    facts: [
      row("fact-frost-queue", "frostbridge has default message queue kafka"),
      row("fact-frost-timeout", "frostbridge has api timeout 12 seconds"),
      row("fact-frost-format", "frostbridge has export format csv"),
    ],
    events: [],
    graph: { nodes: [], edges: [], paths: [], pathCandidates: [] },
    alternates: [],
    diagnostics: [],
    behavioralGuidance: [],
    recalledChunkIds: [],
    recalledChunkTexts: [],
    promptEvidence: [],
    evidencePackets: [
      {
        ...packet(
          "packet-frost",
          [
            "[answer] frostbridge has default message queue kafka",
            "[answer] frostbridge has api timeout 12 seconds",
            "[answer] frostbridge has export format csv",
          ].join("\n"),
        ),
        displayLines: [
          "[answer] frostbridge has default message queue kafka",
          "[answer] frostbridge has api timeout 12 seconds",
          "[answer] frostbridge has export format csv",
        ],
        sourceRefs: ["fact:frostbridge"],
        layers: ["fact"],
      },
    ],
    renderedBlock: "",
  };

  const focused = focusRecallBundleForQueryEntities(
    {
      queryEntities: [{ name: "FrostBridge/霜桥同步", type: "project", role: "subject" }],
      suppressedEntities: [],
    },
    bundle,
  );

  assert.deepEqual(
    focused.facts.map((entry) => entry.id),
    ["fact-frost-queue", "fact-frost-timeout", "fact-frost-format"],
  );
  assert.deepEqual(
    focused.evidencePackets.map((entry) => entry.packetId),
    ["packet-frost"],
  );
  const eligibility = assessNativeContextEligibility(
    query,
    {
      queryEntities: [{ name: "FrostBridge/霜桥同步", type: "project", role: "subject" }],
      suppressedEntities: [],
    },
    focused,
  );
  assert.equal(eligibility.eligible, true);
  assert.equal(eligibility.reason, "llm-query-entities");
});

test("native recall context withholds weak source-grounded assembled evidence", async () => {
  const { assessNativeContextEligibility } = await import("../dist/.runtime/src/host/service.mjs");

  const result = assessNativeContextEligibility(
    "上次确认的默认数据库是什么？",
    {
      queryEntities: [],
      suppressedEntities: [],
    },
    {
      routeType: "mixed",
      routeConfidence: 0.32,
      states: [],
      tasks: [],
      facts: [],
      events: [],
      graph: { nodes: [], edges: [], pathCandidates: [], paths: [] },
      alternates: [],
      diagnostics: [],
      behavioralGuidance: [],
      recalledChunkIds: [],
      recalledChunkTexts: [],
      promptEvidence: [],
      evidencePackets: [
        {
          ...packet("packet-db", "InvoicePilot 默认数据库是 PostgreSQL"),
          coverage: { filled: false, missing: ["query_context"], confidence: 0.44 },
          grade: {
            retrievalScore: 0.48,
            answerScore: 0.45,
            contextBindingScore: 0.36,
            slotCoverageScore: 0.42,
            authorityScore: 0.7,
            finalScore: 0.42,
          },
          displayLines: ["[answer] InvoicePilot 默认数据库是 PostgreSQL"],
          sourceRefs: ["chunk:user:turn-1:0"],
        },
      ],
      renderedBlock: "[answer] InvoicePilot 默认数据库是 PostgreSQL",
    },
  );

  assert.equal(result.eligible, false);
  assert.equal(result.reason, "weak-evidence");
});

test("native recall context trusts strong source-grounded assembled evidence", async () => {
  const { assessNativeContextEligibility } = await import("../dist/.runtime/src/host/service.mjs");

  const result = assessNativeContextEligibility(
    "InvoicePilot 默认数据库是什么？",
    {
      queryEntities: [{ name: "InvoicePilot", type: "project", role: "subject" }],
      suppressedEntities: [],
    },
    {
      routeType: "mixed",
      routeConfidence: 0.42,
      states: [],
      tasks: [],
      facts: [],
      events: [],
      graph: { nodes: [], edges: [], pathCandidates: [], paths: [] },
      alternates: [],
      diagnostics: [],
      behavioralGuidance: [],
      recalledChunkIds: [],
      recalledChunkTexts: [],
      promptEvidence: [],
      evidencePackets: [
        {
          ...packet("packet-db", "InvoicePilot 默认数据库是 PostgreSQL", ["InvoicePilot"]),
          coverage: { filled: true, missing: [], confidence: 0.74 },
          grade: {
            retrievalScore: 0.78,
            answerScore: 0.76,
            contextBindingScore: 0.72,
            slotCoverageScore: 0.74,
            authorityScore: 0.82,
            finalScore: 0.74,
          },
          displayLines: ["[answer] InvoicePilot 默认数据库是 PostgreSQL"],
          sourceRefs: ["fact:invoicepilot-db"],
        },
      ],
      renderedBlock: "[answer] InvoicePilot 默认数据库是 PostgreSQL",
    },
  );

  assert.equal(result.eligible, true);
  assert.equal(result.reason, "llm-query-entities");
});

test("native recall fallback withholds strong evidence that has no query anchor support", async () => {
  const { assessNativeContextEligibility } = await import("../dist/.runtime/src/host/service.mjs");

  const result = assessNativeContextEligibility(
    "请用中文简短回答：今天我们测试一个虚构项目 AsterFlow，先只确认你收到这个名字。",
    {
      queryEntities: [],
      suppressedEntities: [],
      compilerProvenance: {
        source: "deterministic",
        mode: "fallback",
        reasons: ["query-compile-llm-timeout"],
      },
    },
    {
      routeType: "mixed",
      routeConfidence: 0.72,
      states: [],
      tasks: [],
      facts: [],
      events: [],
      graph: { nodes: [], edges: [], pathCandidates: [], paths: [] },
      alternates: [],
      diagnostics: [],
      behavioralGuidance: [],
      recalledChunkIds: [],
      recalledChunkTexts: [],
      promptEvidence: [],
      evidencePackets: [
        {
          ...packet("packet-old-project", "银杏仪表盘 has default database sqlite"),
          coverage: { filled: true, missing: [], confidence: 0.91 },
          grade: {
            retrievalScore: 0.91,
            answerScore: 0.91,
            contextBindingScore: 0.91,
            slotCoverageScore: 0.91,
            authorityScore: 0.91,
            finalScore: 0.91,
          },
          displayLines: ["[answer] 银杏仪表盘 has default database sqlite"],
          sourceRefs: ["fact:old-project-default-db"],
        },
      ],
      renderedBlock: "[answer] 银杏仪表盘 has default database sqlite",
    },
  );

  assert.equal(result.eligible, false);
  assert.equal(result.reason, "query-anchor-mismatch");
});

test("native recall fallback withholds same-entity evidence for a different requested attribute", async () => {
  const { assessNativeContextEligibility } = await import("../dist/.runtime/src/host/service.mjs");

  const result = assessNativeContextEligibility(
    "不用查看文件。现在蓝杉账本默认队列是什么？只回答队列名。",
    {
      queryEntities: [],
      suppressedEntities: [],
      compilerProvenance: {
        source: "deterministic",
        mode: "fallback",
        reasons: ["query-compile-llm-timeout"],
      },
    },
    {
      routeType: "mixed",
      routeConfidence: 0.72,
      states: [],
      tasks: [],
      facts: [],
      events: [],
      graph: { nodes: [], edges: [], pathCandidates: [], paths: [] },
      alternates: [],
      diagnostics: [],
      behavioralGuidance: [],
      recalledChunkIds: [],
      recalledChunkTexts: [],
      promptEvidence: [],
      evidencePackets: [
        {
          ...packet("packet-export-format", "蓝杉账本 has export format parquet"),
          coverage: { filled: true, missing: [], confidence: 0.91 },
          grade: {
            retrievalScore: 0.91,
            answerScore: 0.91,
            contextBindingScore: 0.91,
            slotCoverageScore: 0.91,
            authorityScore: 0.91,
            finalScore: 0.91,
          },
          displayLines: ["[answer] 蓝杉账本 has export format parquet"],
          sourceRefs: ["fact:lan-shan-export-format"],
        },
      ],
      renderedBlock: "[answer] 蓝杉账本 has export format parquet",
    },
  );

  assert.equal(result.eligible, false);
  assert.equal(result.reason, "degraded-query-semantic-mismatch");
});

test("native recall fallback accepts cross-lingual canonical attribute evidence", async () => {
  const { assessNativeContextEligibility } = await import("../dist/.runtime/src/host/service.mjs");

  const result = assessNativeContextEligibility(
    "OrchidLedger 的默认消息队列是什么？",
    {
      queryEntities: [],
      suppressedEntities: [],
      answerMode: "attribute_lookup",
      evidencePlan: {
        slots: [
          {
            id: "answer_value",
            role: "answer_value",
            requestedAttributeSlots: ["default_message_queue"],
          },
        ],
      },
      compilerProvenance: {
        source: "deterministic",
        mode: "fallback",
        reasons: ["query-compile-llm-timeout"],
      },
    },
    {
      routeType: "mixed",
      routeConfidence: 0.72,
      states: [],
      tasks: [],
      facts: [],
      events: [],
      graph: { nodes: [], edges: [], pathCandidates: [], paths: [] },
      alternates: [],
      diagnostics: [],
      behavioralGuidance: [],
      recalledChunkIds: [],
      recalledChunkTexts: [],
      promptEvidence: [],
      evidencePackets: [
        {
          ...packet("packet-orchid-queue", "orchidledger has default message queue nats"),
          coverage: { filled: true, missing: [], confidence: 0.91 },
          grade: {
            retrievalScore: 0.91,
            answerScore: 0.91,
            contextBindingScore: 0.91,
            slotCoverageScore: 0.91,
            authorityScore: 0.91,
            finalScore: 0.91,
          },
          displayLines: ["[answer] orchidledger has default message queue nats"],
          sourceRefs: ["fact:orchidledger-default-message-queue"],
        },
      ],
      renderedBlock: "[answer] orchidledger has default message queue nats",
    },
  );

  assert.equal(result.eligible, true);
  assert.equal(result.reason, "strong-evidence");
});

test("native recall context withholds unbound strong evidence for generic generation requests", async () => {
  const { assessNativeContextEligibility } = await import("../dist/.runtime/src/host/service.mjs");

  const result = assessNativeContextEligibility(
    "Now switch to English. Give me a compact checklist for reviewing API pagination behavior.",
    {
      queryEntities: [],
      suppressedEntities: [],
      queryShape: {
        timeframe: "timeless",
        granularity: "summary",
        referentialMode: "anchored",
        evidenceNeed: "chunk",
      },
      compilerProvenance: { source: "llm", mode: "llm" },
    },
    {
      routeType: "mixed",
      routeConfidence: 0.72,
      states: [],
      tasks: [],
      facts: [],
      events: [],
      graph: { nodes: [], edges: [], pathCandidates: [], paths: [] },
      alternates: [],
      diagnostics: [],
      behavioralGuidance: [],
      recalledChunkIds: [],
      recalledChunkTexts: [],
      promptEvidence: [],
      evidencePackets: [
        {
          ...packet("packet-orion", "OrionDesk has default vector db Qdrant", ["OrionDesk"]),
          displayLines: ["[answer] OrionDesk has default vector db Qdrant"],
          sourceRefs: ["fact:oriondesk-vector-db"],
        },
      ],
      renderedBlock: "[answer] OrionDesk has default vector db Qdrant",
    },
  );

  assert.equal(result.eligible, false);
  assert.equal(result.reason, "unbound-context");
});

test("native recall context honors LLM context exclusions for prior projects", async () => {
  const { assessNativeContextEligibility } = await import("../dist/.runtime/src/host/service.mjs");

  const result = assessNativeContextEligibility(
    "Now switch to English. Give me a compact checklist for reviewing API pagination behavior; do not mention any previous project.",
    {
      queryEntities: [],
      suppressedEntities: [],
      contextExclusions: [
        {
          kind: "prior_project",
          label: "previous project",
          reason: "The user explicitly excluded prior project context.",
        },
      ],
      queryShape: {
        timeframe: "timeless",
        granularity: "summary",
        referentialMode: "anchored",
        evidenceNeed: "chunk",
      },
      compilerProvenance: { source: "llm", mode: "llm" },
    },
    {
      routeType: "mixed",
      routeConfidence: 0.72,
      states: [],
      tasks: [],
      facts: [],
      events: [],
      graph: { nodes: [], edges: [], pathCandidates: [], paths: [] },
      alternates: [],
      diagnostics: [],
      behavioralGuidance: [],
      recalledChunkIds: [],
      recalledChunkTexts: [],
      promptEvidence: [],
      evidencePackets: [
        {
          ...packet("packet-orion", "OrionDesk has default vector db Qdrant", ["OrionDesk"]),
          displayLines: ["[answer] OrionDesk has default vector db Qdrant"],
          sourceRefs: ["fact:oriondesk-vector-db"],
        },
      ],
      renderedBlock: "[answer] OrionDesk has default vector db Qdrant",
    },
  );

  assert.equal(result.eligible, false);
  assert.equal(result.reason, "excluded-context");
});

test("native recall context withholds direct rows when packet assembly has no injectable evidence", async () => {
  const { assessNativeContextEligibility } = await import("../dist/.runtime/src/host/service.mjs");

  const result = assessNativeContextEligibility(
    "AuroraAccept 现在默认数据库、告警渠道和导出格式分别是什么？",
    {
      queryEntities: [],
      suppressedEntities: [],
    },
    {
      routeType: "mixed",
      routeConfidence: 0.22,
      states: [],
      tasks: [],
      facts: [
        row("fact-db", "AuroraAccept 默认数据库是 PostgreSQL"),
        row("fact-channel", "AuroraAccept 告警渠道是 Slack"),
      ],
      events: [],
      graph: { nodes: [], edges: [], pathCandidates: [], paths: [] },
      alternates: [],
      diagnostics: [],
      behavioralGuidance: [],
      recalledChunkIds: [],
      recalledChunkTexts: [],
      promptEvidence: [],
      evidencePackets: [],
      renderedBlock: "AuroraAccept 默认数据库是 PostgreSQL",
    },
  );

  assert.equal(result.eligible, false);
  assert.equal(result.reason, "no-injected-packets");
});

test("native recall context renders only injected evidence packets", async () => {
  const { formatNativeRecallContext } = await import("../dist/.runtime/src/host/service.mjs");

  const context = formatNativeRecallContext(
    {
      routeType: "mixed",
      routeConfidence: 0.9,
      states: [],
      tasks: [],
      facts: [row("fact-old", "银杏仪表盘 默认导出格式是 CSV")],
      events: [],
      graph: {
        nodes: [],
        edges: [],
        pathCandidates: [],
        paths: ["银杏仪表盘 --contradicts--> NimbusLedger"],
      },
      alternates: [],
      diagnostics: [],
      behavioralGuidance: [],
      recalledChunkIds: [],
      recalledChunkTexts: [],
      promptEvidence: [],
      evidencePackets: [
        {
          ...packet("packet-nimbus", "NimbusLedger 默认导出格式是 Arrow IPC", ["NimbusLedger"]),
          displayLines: ["[answer] NimbusLedger 默认导出格式是 Arrow IPC"],
          sourceRefs: ["fact:nimbus-export"],
        },
      ],
      renderedBlock: "",
    },
    1200,
  );

  assert.match(context, /NimbusLedger/);
  assert.match(context, /Arrow IPC/);
  assert.doesNotMatch(context, /银杏仪表盘/);
});
