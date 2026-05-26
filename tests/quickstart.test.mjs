import assert from "node:assert/strict";
import { existsSync, mkdtempSync, readFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";

test("OpenClaw quickstart config uses unified LLM flags and default local embedding", async () => {
  const { applyOpenClawQuickstartConfig } = await import("../dist/.runtime/src/host/quickstart.mjs");
  const existingAgents = {
    defaults: {
      model: { primary: "existing/main-model", fallbacks: ["existing/fallback"] },
      models: { "existing/main-model": { alias: "Existing Main" } },
    },
  };

  const next = applyOpenClawQuickstartConfig(
    { agents: existingAgents },
    {
      llmProvider: "openai-compatible",
      llmBaseUrl: "https://llm.example.com/v1",
      llmModel: "fast-memory-model",
      llmApiKey: "sk-test",
    },
  );

  assert.deepEqual(next.agents, existingAgents);
  assert.equal(next.models, undefined);

  assert.deepEqual(next.plugins.allow, ["memx"]);
  assert.equal(next.plugins.slots.memory, "memx");

  const memx = next.plugins.entries.memx.config;
  assert.equal(memx.advanced.llmClassifierEnabled, true);
  assert.equal(memx.advanced.llmProvider, "openai-compatible");
  assert.equal(memx.advanced.llmBaseURL, "https://llm.example.com/v1");
  assert.equal(memx.advanced.llmApiKey, "sk-test");
  assert.equal(memx.advanced.llmClassifierModel, "fast-memory-model");
  assert.equal(memx.advanced.enableCompatibilityMemoryTools, false);
  assert.equal(memx.embedding.provider, "sentence-transformers-local");
  assert.equal(memx.embedding.model, "intfloat/multilingual-e5-small");
  assert.match(memx.embedding.localPythonBin, /\.openclaw\/memx\/\.venv\/bin\/python$/);
});

test("OpenClaw quickstart can store provider key as an env SecretRef", async () => {
  const { applyOpenClawQuickstartConfig } = await import("../dist/.runtime/src/host/quickstart.mjs");
  const { memxConfigSchema } = await import("../dist/.runtime/src/config.mjs");

  const next = applyOpenClawQuickstartConfig(
    {},
    {
      llmProvider: "anthropic",
      llmBaseUrl: "https://api.anthropic.com/v1",
      llmModel: "claude-fast",
      llmApiKeyEnv: "ANTHROPIC_API_KEY",
    },
  );

  const advanced = next.plugins.entries.memx.config.advanced;
  assert.equal(advanced.llmProvider, "anthropic");
  assert.deepEqual(advanced.llmApiKey, {
    source: "env",
    provider: "default",
    id: "ANTHROPIC_API_KEY",
  });
  assert.equal(advanced.llmClassifierModel, "claude-fast");

  const validation = memxConfigSchema.validate(next.plugins.entries.memx.config);
  assert.equal(validation.ok, true);
  assert.deepEqual(validation.value.advanced.llmApiKey, {
    source: "env",
    provider: "default",
    id: "ANTHROPIC_API_KEY",
  });
});

test("OpenClaw quickstart command plan is exec-form only", async () => {
  const { buildOpenClawQuickstartSteps } = await import("../dist/.runtime/src/host/quickstart.mjs");

  const steps = buildOpenClawQuickstartSteps({
    llmProvider: "openai-compatible",
    llmBaseUrl: "https://llm.example.com/v1",
    llmModel: "fast-memory-model",
    llmApiKey: "sk-test",
    homeDir: "/tmp/home",
    configPath: "/tmp/openclaw.json",
    pluginInstallSource: "/tmp/current-memx-package",
  });

  assert.deepEqual(
    steps.map((step) => [step.command, step.args]),
    [
      ["python3", ["-m", "venv", "/tmp/home/.openclaw/memx/.venv"]],
      [
        "/tmp/home/.openclaw/memx/.venv/bin/python",
        ["-m", "pip", "install", "-U", "pip", "sentence-transformers", "torch"],
      ],
      ["openclaw", ["plugins", "install", "/tmp/current-memx-package"]],
      ["openclaw", ["gateway", "restart"]],
      ["openclaw", ["memx", "doctor", "--deep"]],
    ],
  );
  for (const step of steps) {
    assert.equal(typeof step.command, "string");
    assert.ok(Array.isArray(step.args));
    assert.doesNotMatch(step.command, /\s/);
  }
});

test("OpenClaw quickstart writes config and redacts plaintext API key from result", async () => {
  const { runOpenClawQuickstart } = await import("../dist/.runtime/src/host/quickstart.mjs");
  const dir = mkdtempSync(join(tmpdir(), "memx-quickstart-"));
  const configPath = join(dir, "openclaw.json");
  const calls = [];

  const result = await runOpenClawQuickstart(
    {
      llmProvider: "openai-compatible",
      llmBaseUrl: "https://llm.example.com/v1",
      llmModel: "fast-memory-model",
      llmApiKey: "sk-secret",
      configPath,
      homeDir: dir,
      pluginInstallSource: "/tmp/current-memx-package",
      skipEmbeddingDeps: true,
      skipRestart: true,
      skipDoctor: true,
    },
    {
      runCommand: async (command, args) => {
        assert.equal(existsSync(configPath), false, "plugin install should run before writing memx slot config");
        calls.push({ command, args });
        return { code: 0, stdout: "", stderr: "" };
      },
    },
  );

  assert.deepEqual(calls, [
    { command: "openclaw", args: ["plugins", "install", "/tmp/current-memx-package"] },
  ]);
  assert.equal(existsSync(configPath), true);
  const written = JSON.parse(readFileSync(configPath, "utf8"));
  assert.equal(written.models, undefined);
  assert.equal(written.plugins.entries.memx.config.advanced.llmApiKey, "sk-secret");
  assert.doesNotMatch(JSON.stringify(result), /sk-secret/);
  assert.equal(result.ok, true);
});

test("OpenClaw quickstart installs a sanitized package snapshot by default", async () => {
  const { runOpenClawQuickstart, resolveCurrentPackageRoot } = await import(
    "../dist/.runtime/src/host/quickstart.mjs"
  );
  const dir = mkdtempSync(join(tmpdir(), "memx-quickstart-snapshot-"));
  const configPath = join(dir, "openclaw.json");
  const packageRoot = resolveCurrentPackageRoot();
  let installSource = "";

  await runOpenClawQuickstart(
    {
      llmProvider: "openai-compatible",
      llmBaseUrl: "https://llm.example.com/v1",
      llmModel: "fast-memory-model",
      llmApiKey: "sk-secret",
      configPath,
      homeDir: dir,
      skipEmbeddingDeps: true,
      skipRestart: true,
      skipDoctor: true,
    },
    {
      runCommand: async (_command, args) => {
        installSource = args[2];
        assert.notEqual(installSource, packageRoot);
        assert.equal(existsSync(join(installSource, "package.json")), true);
        assert.equal(existsSync(join(installSource, "dist", "index.mjs")), true);
        assert.equal(existsSync(join(installSource, "src")), false);
        assert.equal(existsSync(join(installSource, ".codex-plugin")), false);
        assert.equal(existsSync(join(installSource, ".claude-plugin")), false);
        assert.equal(existsSync(join(installSource, "hooks")), false);
        assert.equal(existsSync(join(installSource, "dist", ".runtime", "src", "bin")), false);
        assert.equal(existsSync(join(installSource, "dist", ".runtime", "src", "host")), false);
        const snapshotPackage = JSON.parse(readFileSync(join(installSource, "package.json"), "utf8"));
        assert.equal(snapshotPackage.bin, undefined);
        assert.deepEqual(snapshotPackage.openclaw.extensions, ["./dist/index.mjs"]);
        const pluginEntry = readFileSync(join(installSource, "dist", ".runtime", "src", "index.mjs"), "utf8");
        assert.doesNotMatch(pluginEntry, /host\/(?:quickstart|standaloneQuickstart|hookRunner|mcpProtocol|httpServer)/);
        return { code: 0, stdout: "", stderr: "" };
      },
    },
  );

  assert.equal(existsSync(installSource), false);
});

test("OpenClaw quickstart dry run reports that no changes were applied", async () => {
  const { runOpenClawQuickstart } = await import("../dist/.runtime/src/host/quickstart.mjs");

  const result = await runOpenClawQuickstart({
    llmProvider: "ollama",
    llmBaseUrl: "http://127.0.0.1:11434",
    llmModel: "qwen2.5:7b",
    skipEmbeddingDeps: true,
    dryRun: true,
  });

  assert.match(result.nextStep, /Dry run only/);
  assert.equal(result.llmApiKey, null);
});

test("OpenClaw quickstart keeps legacy provider-id and memx-model aliases", async () => {
  const { applyOpenClawQuickstartConfig } = await import("../dist/.runtime/src/host/quickstart.mjs");

  const next = applyOpenClawQuickstartConfig(
    {},
    {
      providerId: "legacy-provider",
      baseUrl: "https://legacy.example.com/v1",
      agentModel: "legacy-main",
      memxModel: "legacy-memory",
      apiKey: "sk-legacy",
    },
  );

  assert.equal(next.models, undefined);
  assert.equal(next.agents, undefined);
  assert.equal(next.plugins.entries.memx.config.advanced.llmProvider, "openai-compatible");
  assert.equal(next.plugins.entries.memx.config.advanced.llmBaseURL, "https://legacy.example.com/v1");
  assert.equal(
    next.plugins.entries.memx.config.advanced.llmClassifierModel,
    "legacy-memory",
  );
});

test("OpenClaw quickstart migrates legacy memory-memx config into memx", async () => {
  const { applyOpenClawQuickstartConfig } = await import("../dist/.runtime/src/host/quickstart.mjs");

  const next = applyOpenClawQuickstartConfig(
    {
      plugins: {
        allow: ["memory-memx"],
        slots: { memory: "memory-memx" },
        entries: {
          "memory-memx": {
            enabled: true,
            hooks: { allowPromptInjection: true },
            config: {
              embedding: { provider: "off" },
              advanced: { enableTurnScheduler: false },
            },
          },
        },
      },
    },
    {
      llmProvider: "openai-compatible",
      llmBaseUrl: "https://llm.example.com/v1",
      llmModel: "fast-memory-model",
      llmApiKey: "sk-test",
    },
  );

  assert.deepEqual(next.plugins.allow, ["memx"]);
  assert.equal(next.plugins.slots.memory, "memx");
  assert.equal(next.plugins.entries["memory-memx"], undefined);
  assert.equal(next.plugins.entries.memx.enabled, true);
  assert.equal(next.plugins.entries.memx.hooks.allowPromptInjection, true);
  assert.equal(next.plugins.entries.memx.config.advanced.enableTurnScheduler, true);
});
