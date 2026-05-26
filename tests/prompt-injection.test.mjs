import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";
import plugin from "../dist/index.mjs";
import { extractPromptQuery } from "../dist/.runtime/src/index.mjs";
import { captureAgentEndTurn } from "../dist/.runtime/src/pipeline/turnCapture.mjs";
import { stripInjectedHistoricalBlock } from "../dist/.runtime/src/security/escaping.mjs";

function createLogger() {
  return {
    debug() {},
    info() {},
    warn() {},
    error() {},
  };
}

test("memory recall registers only the prompt-build hook plus agent_end capture", async () => {
  const hooks = [];

  await plugin.register({
    config: {},
    pluginConfig: {},
    logger: createLogger(),
    registerTool() {},
    registerCli() {},
    on(name) {
      hooks.push(name);
    },
    registerService() {},
  });

  assert.equal(hooks.filter((name) => name === "before_prompt_build").length, 1);
  assert.equal(hooks.filter((name) => name === "before_agent_start").length, 0);
  assert.equal(hooks.filter((name) => name === "agent_end").length, 1);
});

test("memX prepend context markers are stripped before reading the user query", () => {
  const prompt = [
    "<!-- MEMX_CONTEXT_START -->",
    "## memX Memory",
    "- remembered historical fact",
    "<!-- MEMX_CONTEXT_END -->",
    "",
    "What should I do next?",
  ].join("\n");

  assert.equal(stripInjectedHistoricalBlock(prompt), "What should I do next?");
});

test("prompt query extraction keeps multi-paragraph task statements", () => {
  const prompt = [
    "You are solving an olympiad problem.",
    "",
    "Problem C8. Let n be a positive integer. Given an n x n board, the unit cell in the top left corner is initially coloured black, and the other cells are coloured white. In each operation, choose a 2 x 2 square with exactly one black cell and colour the remaining three cells black. Determine all values of n such that the whole board can become black.",
    "",
    "Give a rigorous solution.",
  ].join("\n");

  const query = extractPromptQuery({ prompt });

  assert.match(query, /Problem C8/);
  assert.match(query, /n x n board/);
  assert.match(query, /2 x 2 square/);
  assert.match(query, /Give a rigorous solution/);
});

test("turn capture excludes memX prepend context from the captured user message", () => {
  const prompt = [
    "<!-- MEMX_CONTEXT_START -->",
    "## memX Memory",
    "- previous answer evidence",
    "<!-- MEMX_CONTEXT_END -->",
    "",
    "Please continue the current task.",
  ].join("\n");

  const captured = captureAgentEndTurn({
    agentId: "main",
    scope: "global",
    sessionKey: "agent:main:test",
    turnId: "turn_test",
    observedAt: "2026-05-11T00:00:00.000Z",
    messages: [
      { role: "user", content: prompt },
      { role: "assistant", content: "Continuing the task." },
    ],
  });

  assert.deepEqual(
    captured.map((entry) => ({ role: entry.role, content: entry.content })),
    [
      { role: "user", content: "Please continue the current task." },
      { role: "assistant", content: "Continuing the task." },
    ],
  );
});

test("turn capture suppresses concise assistant memory acknowledgements", () => {
  const captured = captureAgentEndTurn({
    agentId: "main",
    scope: "global",
    sessionKey: "agent:main:test",
    turnId: "turn_ack",
    observedAt: "2026-05-11T00:00:00.000Z",
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

  assert.deepEqual(
    captured.map((entry) => ({ role: entry.role, content: entry.content })),
    [
      {
        role: "user",
        content: "工程记录：AuroraBridge 的默认告警通道是 RelayNine。",
      },
    ],
  );
});

test("compiled recall hook returns prependContext rather than overriding systemPrompt", async () => {
  const compiled = await readFile(new URL("../dist/.runtime/src/index.mjs", import.meta.url), "utf8");

  assert.match(compiled, /prependContext:\s*_logPromptContext/);
  assert.doesNotMatch(compiled, /systemPrompt:\s*_logSystemPrompt/);
});
