import assert from "node:assert/strict";
import { existsSync, mkdtempSync, readFileSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";

test("standalone quickstart builds independent LLM and local embedding config", async () => {
  const { applyStandaloneMemxQuickstartConfig } = await import(
    "../dist/.runtime/src/host/standaloneQuickstart.mjs"
  );

  const next = applyStandaloneMemxQuickstartConfig(
    {},
    {
      llmProvider: "openai-compatible",
      llmBaseUrl: "https://llm.example.com/v1",
      llmModel: "fast-memory-model",
      llmApiKey: "sk-standalone",
      homeDir: "/tmp/home",
    },
  );

  assert.equal(next.advanced.llmProvider, "openai-compatible");
  assert.equal(next.advanced.llmBaseURL, "https://llm.example.com/v1");
  assert.equal(next.advanced.llmClassifierModel, "fast-memory-model");
  assert.equal(next.advanced.llmApiKey, "sk-standalone");
  assert.equal(next.embedding.provider, "sentence-transformers-local");
  assert.equal(next.embedding.model, "intfloat/multilingual-e5-small");
  assert.equal(next.embedding.localPythonBin, "/tmp/home/.memx/.venv/bin/python");
});

test("standalone service config resolves LLM without OpenClaw config", async () => {
  const { runStandaloneMemxQuickstart } = await import("../dist/.runtime/src/host/standaloneQuickstart.mjs");
  const { createServiceConfigFromEnv } = await import("../dist/.runtime/src/host/service.mjs");
  const { loadJudgeModelConfig } = await import("../dist/.runtime/src/pipeline/judgeModelConfig.mjs");
  const dir = mkdtempSync(join(tmpdir(), "memx-standalone-"));
  const configPath = join(dir, "config.json");

  await runStandaloneMemxQuickstart({
    target: "mcp",
    configPath,
    homeDir: dir,
    llmProvider: "openai-compatible",
    llmBaseUrl: "https://llm.example.com/v1",
    llmModel: "fast-memory-model",
    llmApiKey: "sk-standalone",
    skipEmbeddingDeps: true,
    skipServiceStart: true,
  });

  const config = createServiceConfigFromEnv({ MEMX_CONFIG_PATH: configPath });
  const judge = loadJudgeModelConfig(config, { debug() {}, warn() {} });

  assert.equal(judge.provider, "openai-compatible");
  assert.equal(judge.baseUrl, "https://llm.example.com/v1");
  assert.equal(judge.model, "fast-memory-model");
  assert.equal(judge.apiKey, "sk-standalone");
});

test("standalone service config migrates legacy agent-wide scopes from disk", async () => {
  const { createServiceConfigFromEnv } = await import("../dist/.runtime/src/host/service.mjs");
  const { writeFile } = await import("node:fs/promises");
  const dir = mkdtempSync(join(tmpdir(), "memx-standalone-legacy-scope-"));
  const configPath = join(dir, "config.json");
  await writeFile(
    configPath,
    JSON.stringify({
      defaultScope: "agent:{agentId}",
      allowedScopes: ["global", "agent:{agentId}", "session:{sessionKey}", "project:{project}"],
    }),
    "utf8",
  );

  const config = createServiceConfigFromEnv({ MEMX_CONFIG_PATH: configPath });

  assert.equal(config.defaultScope, "workspace:{workspace}");
  assert.deepEqual(config.allowedScopes, [
    "workspace:{workspace}",
    "agent:{agentId}",
    "session:{sessionKey}",
    "project:{project}",
  ]);
});

test("standalone quickstart migrates query compiler timeout settings to the native hook sub-budget", async () => {
  const { applyStandaloneMemxQuickstartConfig } = await import(
    "../dist/.runtime/src/host/standaloneQuickstart.mjs"
  );

  const next = applyStandaloneMemxQuickstartConfig(
    {
      advanced: {
        queryCompilerHotPathTimeoutMs: 8000,
      },
    },
    {
      llmProvider: "openai-compatible",
      llmBaseUrl: "https://llm.example.com/v1",
      llmModel: "fast-memory-model",
      llmApiKey: "sk-standalone",
      homeDir: "/tmp/home",
    },
  );

  assert.equal(next.advanced.queryCompilerHotPathTimeoutMs, 5000);
});

test("standalone quickstart migrates legacy agent-wide scopes to workspace isolation", async () => {
  const { applyStandaloneMemxQuickstartConfig } = await import(
    "../dist/.runtime/src/host/standaloneQuickstart.mjs"
  );

  const next = applyStandaloneMemxQuickstartConfig(
    {
      defaultScope: "agent:{agentId}",
      allowedScopes: ["global", "agent:{agentId}", "session:{sessionKey}", "project:{project}"],
    },
    {
      llmProvider: "openai-compatible",
      llmBaseUrl: "https://llm.example.com/v1",
      llmModel: "fast-memory-model",
      llmApiKey: "sk-standalone",
      homeDir: "/tmp/home",
    },
  );

  assert.equal(next.defaultScope, "workspace:{workspace}");
  assert.deepEqual(next.allowedScopes, [
    "workspace:{workspace}",
    "agent:{agentId}",
    "session:{sessionKey}",
    "project:{project}",
  ]);
});

test("standalone quickstart can configure Codex in one command", async () => {
  const { runStandaloneMemxQuickstart } = await import("../dist/.runtime/src/host/standaloneQuickstart.mjs");
  const dir = mkdtempSync(join(tmpdir(), "memx-codex-"));
  const configPath = join(dir, "config.json");
  const codexConfigPath = join(dir, "codex.toml");
  const staleCodexCache = join(dir, ".codex", "plugins", "cache", "memx", "memx", "2026.3.15");
  await import("node:fs/promises").then(({ mkdir, writeFile }) =>
    mkdir(staleCodexCache, { recursive: true }).then(() =>
      writeFile(join(staleCodexCache, "stale.txt"), "old manifest", "utf8"),
    ),
  );
  const calls = [];

  const result = await runStandaloneMemxQuickstart(
    {
      target: "codex",
      configPath,
      codexConfigPath,
      homeDir: dir,
      llmProvider: "openai-compatible",
      llmBaseUrl: "https://llm.example.com/v1",
      llmModel: "fast-memory-model",
      llmApiKey: "sk-standalone",
      skipEmbeddingDeps: true,
      skipServiceStart: true,
    },
    {
      runCommand: async (command, args) => {
        calls.push({ command, args });
        return { code: 0 };
      },
    },
  );

  assert.equal(existsSync(configPath), true);
  assert.equal(existsSync(codexConfigPath), false, "Codex native hook install should not add an MCP server by default");
  const written = JSON.parse(readFileSync(configPath, "utf8"));
  assert.equal(written.advanced.llmBaseURL, "https://llm.example.com/v1");
  assert.equal(written.embedding.model, "intfloat/multilingual-e5-small");
  assert.equal(existsSync(join(dir, ".memx", "runtime", "src", "bin", "memx-mcp.mjs")), true);
  assert.equal(existsSync(join(dir, ".memx", "runtime", "src", "bin", "memx-hook.mjs")), true);
  assert.equal(
    existsSync(join(dir, ".memx", "codex-marketplace", ".agents", "plugins", "marketplace.json")),
    true,
  );
  assert.equal(existsSync(staleCodexCache), false, "quickstart should invalidate stale Codex plugin cache");
  const hookJson = readFileSync(
    join(dir, ".memx", "codex-marketplace", "plugins", "memx", "hooks", "hooks.codex.json"),
    "utf8",
  );
  const rootHookJson = readFileSync(
    join(dir, ".memx", "codex-marketplace", "plugins", "memx", "hooks.json"),
    "utf8",
  );
  assert.equal(rootHookJson, hookJson, "root hooks.json should mirror the compatibility hook file");
  assert.match(hookJson, new RegExp(join(dir, ".memx", "runtime", "src", "bin", "memx-hook.mjs").replace(/[.*+?^${}()|[\]\\]/g, "\\$&")));
  const hookConfig = JSON.parse(hookJson);
  const userPromptHook = hookConfig.hooks.UserPromptSubmit[0].hooks[0];
  assert.equal("args" in userPromptHook, false);
  assert.match(userPromptHook.command, /memx-hook\.mjs'? codex UserPromptSubmit\b/);
  assert.match(userPromptHook.command, /--hook-config /);
  assert.equal(
    existsSync(join(dir, ".memx", "hook-runtime.json")),
    true,
    "native hooks should receive quickstart memxUrl without relying on shell env",
  );
  const hookRuntime = JSON.parse(readFileSync(join(dir, ".memx", "hook-runtime.json"), "utf8"));
  assert.equal(hookRuntime.memxUrl, "http://127.0.0.1:3878");
  assert.deepEqual(
    {
      hookTimeoutMs: hookRuntime.hookTimeoutMs,
      hookContextTimeoutMs: hookRuntime.hookContextTimeoutMs,
      hookObserveTimeoutMs: hookRuntime.hookObserveTimeoutMs,
      hookQueryCompilerTimeoutMs: hookRuntime.hookQueryCompilerTimeoutMs,
    },
    {
      hookTimeoutMs: 8000,
      hookContextTimeoutMs: undefined,
      hookObserveTimeoutMs: undefined,
      hookQueryCompilerTimeoutMs: undefined,
    },
  );
  const pluginManifest = JSON.parse(
    readFileSync(join(dir, ".memx", "codex-marketplace", "plugins", "memx", ".codex-plugin", "plugin.json"), "utf8"),
  );
  assert.equal("mcpServers" in pluginManifest, false);
  assert.equal("skills" in pluginManifest, false);
  assert.equal(pluginManifest.hooks, "./hooks.json");
  assert.equal(pluginManifest.interface.displayName, "memX");
  assert.equal(pluginManifest.interface.shortDescription, "Local semantic memory for coding agents.");
  assert.deepEqual(calls, [
    { command: "codex", args: ["plugin", "remove", "memx@memx"] },
    { command: "codex", args: ["plugin", "marketplace", "remove", "memx"] },
    {
      command: "codex",
      args: ["plugin", "marketplace", "add", join(dir, ".memx", "codex-marketplace")],
    },
    { command: "codex", args: ["plugin", "add", "memx@memx"] },
  ]);
  assert.equal(result.codexPlugin.installed, true);
  assert.equal(result.hostConfig.path, undefined);
  assert.ok(Array.isArray(result.lifecycleNotes));
  assert.match(result.lifecycleNotes.join("\n"), /trust/i);
  assert.match(result.lifecycleNotes.join("\n"), /codex exec/i);
  assert.doesNotMatch(JSON.stringify(result), /sk-standalone/);
});

test("standalone quickstart starts the local service after installing native hooks", async () => {
  const { runStandaloneMemxQuickstart } = await import("../dist/.runtime/src/host/standaloneQuickstart.mjs");
  const dir = mkdtempSync(join(tmpdir(), "memx-codex-service-"));
  const configPath = join(dir, "config.json");
  const serviceStarts = [];

  const result = await runStandaloneMemxQuickstart(
    {
      target: "codex",
      configPath,
      homeDir: dir,
      llmProvider: "openai-compatible",
      llmBaseUrl: "https://llm.example.com/v1",
      llmModel: "fast-memory-model",
      llmApiKey: "sk-standalone",
      skipEmbeddingDeps: true,
    },
    {
      runCommand: async () => ({ code: 0 }),
      ensureService: async (options) => {
        serviceStarts.push(options);
        return {
          ok: true,
          alreadyRunning: false,
          url: "http://127.0.0.1:3878",
          pid: 12345,
          pidPath: join(dir, ".memx", "service.json"),
          logPath: join(dir, ".memx", "memx-server.log"),
        };
      },
    },
  );

  assert.equal(serviceStarts.length, 1);
  assert.equal(serviceStarts[0].configPath, configPath);
  assert.equal(serviceStarts[0].runtimeDir, join(dir, ".memx", "runtime"));
  assert.equal(serviceStarts[0].url, "http://127.0.0.1:3878");
  assert.equal(result.service.ok, true);
  assert.equal(result.service.pid, 12345);
  assert.doesNotMatch(String(result.nextStep ?? ""), /Start memx-server/i);
});

test("Codex native quickstart removes stale MCP config when lifecycle hooks own memory", async () => {
  const { runStandaloneMemxQuickstart } = await import("../dist/.runtime/src/host/standaloneQuickstart.mjs");
  const dir = mkdtempSync(join(tmpdir(), "memx-codex-stale-mcp-"));
  const configPath = join(dir, "config.json");
  const codexConfigPath = join(dir, "codex.toml");
  writeFileSync(
    codexConfigPath,
    [
      'model = "gpt-5.5"',
      "",
      "[mcp_servers.memx]",
      'command = "npx"',
      'args = ["-y", "-p", "github:NeoLi00/memX", "memx-mcp"]',
      "",
      "[mcp_servers.memx.env]",
      'MEMX_MCP_TOOLS = "full"',
      "",
      "[mcp_servers.other]",
      'command = "other-mcp"',
      "",
    ].join("\n"),
    "utf8",
  );

  const result = await runStandaloneMemxQuickstart(
    {
      target: "codex",
      configPath,
      codexConfigPath,
      homeDir: dir,
      llmProvider: "openai-compatible",
      llmBaseUrl: "https://llm.example.com/v1",
      llmModel: "fast-memory-model",
      llmApiKey: "sk-standalone",
      skipEmbeddingDeps: true,
      skipServiceStart: true,
    },
    {
      runCommand: async () => ({ code: 0 }),
    },
  );

  const toml = readFileSync(codexConfigPath, "utf8");
  assert.doesNotMatch(toml, /\[mcp_servers\.memx\]/);
  assert.doesNotMatch(toml, /MEMX_MCP_TOOLS/);
  assert.match(toml, /\[mcp_servers\.other\]/);
  assert.equal(result.hostConfig.path, codexConfigPath);
  assert.equal(result.hostConfig.mcpTools, "none");
});

test("standalone quickstart installs Claude Code native plugin hooks in one command", async () => {
  const { runStandaloneMemxQuickstart } = await import("../dist/.runtime/src/host/standaloneQuickstart.mjs");
  const dir = mkdtempSync(join(tmpdir(), "memx-claude-"));
  const configPath = join(dir, "config.json");
  const claudeConfigPath = join(dir, "claude.json");
  const staleClaudeCache = join(dir, ".claude", "plugins", "cache", "memx-local", "memx", "2026.3.15");
  await import("node:fs/promises").then(({ mkdir, writeFile }) =>
    mkdir(staleClaudeCache, { recursive: true }).then(() =>
      writeFile(join(staleClaudeCache, "stale.txt"), "old manifest", "utf8"),
    ),
  );
  const calls = [];

  const result = await runStandaloneMemxQuickstart(
    {
      target: "claude-code",
      configPath,
      claudeConfigPath,
      homeDir: dir,
      llmProvider: "openai-compatible",
      llmBaseUrl: "https://llm.example.com/v1",
      llmModel: "fast-memory-model",
      llmApiKey: "sk-standalone",
      skipEmbeddingDeps: true,
      skipServiceStart: true,
    },
    {
      runCommand: async (command, args) => {
        calls.push({ command, args });
        return { code: 0 };
      },
    },
  );

  assert.equal(existsSync(configPath), true);
  assert.equal(existsSync(claudeConfigPath), false, "Claude native plugin should provide MCP without duplicate .claude.json server");
  assert.equal(existsSync(join(dir, ".memx", "runtime", "src", "bin", "memx-hook.mjs")), true);
  assert.equal(
    existsSync(join(dir, ".memx", "claude-marketplace", ".claude-plugin", "marketplace.json")),
    true,
  );
  assert.equal(existsSync(staleClaudeCache), false, "quickstart should invalidate stale Claude plugin cache");
  assert.equal(
    existsSync(join(dir, ".memx", "claude-marketplace", "plugins", "memx", ".claude-plugin", "plugin.json")),
    true,
  );
  assert.equal(
    existsSync(join(dir, ".memx", "claude-marketplace", "plugins", "memx", "hooks", "hooks.json")),
    true,
  );
  assert.equal(
    existsSync(
      join(
        dir,
        ".memx",
        "claude-marketplace",
        "plugins",
        "memx",
        "dist",
        ".runtime",
        "src",
        "bin",
        "memx-mcp.mjs",
      ),
    ),
    true,
  );
  const marketplace = JSON.parse(
    readFileSync(join(dir, ".memx", "claude-marketplace", ".claude-plugin", "marketplace.json"), "utf8"),
  );
  assert.equal(marketplace.name, "memx");
  assert.equal(marketplace.plugins[0].source, "./plugins/memx");
  const pluginManifest = JSON.parse(
    readFileSync(join(dir, ".memx", "claude-marketplace", "plugins", "memx", ".claude-plugin", "plugin.json"), "utf8"),
  );
  assert.equal("hooks" in pluginManifest, false);
  assert.equal("mcpServers" in pluginManifest, false);
  assert.equal("skills" in pluginManifest, false);
  assert.equal(
    existsSync(join(dir, ".memx", "claude-marketplace", "plugins", "memx", ".mcp.json")),
    false,
  );
  const claudeSettingsPath = join(dir, ".claude", "settings.json");
  const claudeSettings = JSON.parse(readFileSync(claudeSettingsPath, "utf8"));
  assert.equal(claudeSettings.autoMemoryEnabled, false);
  assert.equal(claudeSettings.env.CLAUDE_CODE_DISABLE_AUTO_MEMORY, "1");
  assert.equal(existsSync(join(dir, ".memx", "claude-settings-backup.json")), true);
  const hookJson = readFileSync(
    join(dir, ".memx", "claude-marketplace", "plugins", "memx", "hooks", "hooks.json"),
    "utf8",
  );
  assert.match(hookJson, /claude-code UserPromptSubmit/);
  assert.match(hookJson, /--hook-config/);
  assert.equal(
    existsSync(join(dir, ".memx", "claude-marketplace", "plugins", "memx", ".memx-hook.json")),
    true,
    "Claude plugin snapshot should carry the hook runtime config",
  );
  const hookRuntime = JSON.parse(
    readFileSync(
      join(dir, ".memx", "claude-marketplace", "plugins", "memx", ".memx-hook.json"),
      "utf8",
    ),
  );
  assert.equal(hookRuntime.memxUrl, "http://127.0.0.1:3878");
  assert.equal(hookRuntime.hookTimeoutMs, 8000);
  assert.match(hookJson, /node \\"\$\{CLAUDE_PLUGIN_ROOT\}\/dist\/\.runtime\/src\/bin\/memx-hook\.mjs\\" claude-code/);
  assert.deepEqual(calls, [
    { command: "claude", args: ["plugin", "uninstall", "memx@memx"] },
    { command: "claude", args: ["plugin", "uninstall", "memx"] },
    { command: "claude", args: ["plugin", "marketplace", "remove", "memx"] },
    {
      command: "claude",
      args: ["plugin", "marketplace", "add", join(dir, ".memx", "claude-marketplace")],
    },
    { command: "claude", args: ["plugin", "install", "memx@memx"] },
  ]);
  assert.equal(result.claudePlugin.installed, true);
  assert.equal(result.hostConfig.settingsPath, claudeSettingsPath);
  assert.doesNotMatch(JSON.stringify(result), /sk-standalone/);
});

test("Claude native quickstart removes stale MCP server while preserving unrelated servers", async () => {
  const { runStandaloneMemxQuickstart } = await import("../dist/.runtime/src/host/standaloneQuickstart.mjs");
  const dir = mkdtempSync(join(tmpdir(), "memx-claude-stale-mcp-"));
  const configPath = join(dir, "config.json");
  const claudeConfigPath = join(dir, "claude.json");
  writeFileSync(
    claudeConfigPath,
    `${JSON.stringify(
      {
        mcpServers: {
          memx: {
            command: "npx",
            args: ["-y", "-p", "github:NeoLi00/memX", "memx-mcp"],
          },
          other: {
            command: "other-mcp",
          },
        },
      },
      null,
      2,
    )}\n`,
    "utf8",
  );

  const result = await runStandaloneMemxQuickstart(
    {
      target: "claude-code",
      configPath,
      claudeConfigPath,
      homeDir: dir,
      llmProvider: "openai-compatible",
      llmBaseUrl: "https://llm.example.com/v1",
      llmModel: "fast-memory-model",
      llmApiKey: "sk-standalone",
      skipEmbeddingDeps: true,
      skipServiceStart: true,
    },
    {
      runCommand: async () => ({ code: 0 }),
    },
  );

  const claudeConfig = JSON.parse(readFileSync(claudeConfigPath, "utf8"));
  assert.equal("memx" in claudeConfig.mcpServers, false);
  assert.equal(claudeConfig.mcpServers.other.command, "other-mcp");
  assert.equal(result.hostConfig.path, claudeConfigPath);
  assert.equal(result.hostConfig.mcpTools, "none");
});

test("standalone quickstart local embedding install plan is exec-form only", async () => {
  const { buildStandaloneMemxQuickstartSteps } = await import(
    "../dist/.runtime/src/host/standaloneQuickstart.mjs"
  );

  const steps = buildStandaloneMemxQuickstartSteps({
    target: "mcp",
    homeDir: "/tmp/home",
    llmProvider: "openai-compatible",
    llmBaseUrl: "https://llm.example.com/v1",
    llmModel: "fast-memory-model",
    llmApiKey: "sk-standalone",
  });

  assert.deepEqual(
    steps.map((step) => [step.command, step.args]),
    [
      ["python3", ["-m", "venv", "/tmp/home/.memx/.venv"]],
      [
        "/tmp/home/.memx/.venv/bin/python",
        ["-m", "pip", "install", "-U", "pip", "sentence-transformers", "torch"],
      ],
    ],
  );
  for (const step of steps) {
    assert.equal(typeof step.command, "string");
    assert.ok(Array.isArray(step.args));
    assert.doesNotMatch(step.command, /\s/);
  }
});

test("standalone quickstart dry run shows no API key for local Ollama", async () => {
  const { runStandaloneMemxQuickstart } = await import("../dist/.runtime/src/host/standaloneQuickstart.mjs");

  const result = await runStandaloneMemxQuickstart({
    target: "mcp",
    llmProvider: "ollama",
    llmBaseUrl: "http://127.0.0.1:11434",
    llmModel: "qwen2.5:7b",
    skipEmbeddingDeps: true,
    dryRun: true,
  });

  assert.equal(result.llmApiKey, null);
});

test("standalone quickstart keeps generic MCP on the full tool surface", async () => {
  const { runStandaloneMemxQuickstart } = await import("../dist/.runtime/src/host/standaloneQuickstart.mjs");

  const result = await runStandaloneMemxQuickstart({
    target: "mcp",
    llmProvider: "ollama",
    llmBaseUrl: "http://127.0.0.1:11434",
    llmModel: "qwen2.5:7b",
    skipEmbeddingDeps: true,
    dryRun: true,
  });

  assert.equal(result.mcpConfig.mcpServers.memx.env.MEMX_MCP_TOOLS, "full");
});

test("standalone quickstart can explicitly expose full MCP tools for native hosts", async () => {
  const { runStandaloneMemxQuickstart } = await import("../dist/.runtime/src/host/standaloneQuickstart.mjs");
  const dir = mkdtempSync(join(tmpdir(), "memx-codex-full-mcp-"));
  const configPath = join(dir, "config.json");
  const codexConfigPath = join(dir, "codex.toml");

  await runStandaloneMemxQuickstart(
    {
      target: "codex",
      configPath,
      codexConfigPath,
      homeDir: dir,
      llmProvider: "openai-compatible",
      llmBaseUrl: "https://llm.example.com/v1",
      llmModel: "fast-memory-model",
      llmApiKey: "sk-standalone",
      mcpTools: "full",
      skipEmbeddingDeps: true,
      skipServiceStart: true,
    },
    {
      runCommand: async () => ({ code: 0 }),
    },
  );

  const toml = readFileSync(codexConfigPath, "utf8");
  assert.match(toml, /MEMX_MCP_TOOLS = "full"/);
});
