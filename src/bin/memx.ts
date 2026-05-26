#!/usr/bin/env node

import { homedir } from "node:os";
import { join } from "node:path";

function usage(): string {
  return [
    "Usage: memx <command>",
    "",
    "Commands:",
    "  server              Start the local memX REST service",
    "  service status      Report the managed local memX service state",
    "  service stop        Stop the managed local memX service",
    "  service restart     Restart the managed local memX service",
    "  mcp                 Start the memX MCP stdio server",
    "  hook <host> <event>  Run a native hook bridge",
    "  quickstart openclaw Configure OpenClaw-only quickstart settings",
    "  quickstart codex    Configure standalone memX for Codex",
    "  quickstart claude-code Configure standalone memX for Claude Code",
    "  quickstart mcp      Configure standalone memX and print generic MCP JSON",
    "  uninstall openclaw  Cleanly remove memX from OpenClaw config and plugin install",
    "  uninstall codex     Cleanly remove memX from Codex native hooks, MCP config, and service",
    "  uninstall claude-code Cleanly remove memX from Claude Code native hooks, MCP config, and service",
    "  connect codex       Wire Codex MCP config",
    "  connect claude-code Wire Claude Code MCP config",
  ].join("\n");
}

function readOption(argv: string[], name: string): string | undefined {
  const index = argv.indexOf(name);
  if (index < 0) {
    return undefined;
  }
  return argv[index + 1];
}

function hasFlag(argv: string[], name: string): boolean {
  return argv.includes(name);
}

function parseOpenClawQuickstartOptions(argv: string[]) {
  const embeddingProvider = readOption(argv, "--embedding-provider");
  const llmProvider = readOption(argv, "--llm-provider");
  return {
    llmProvider: llmProvider as
      | "openai-compatible"
      | "anthropic"
      | "google"
      | "ollama"
      | undefined,
    llmBaseUrl: readOption(argv, "--llm-base-url"),
    llmModel: readOption(argv, "--llm-model"),
    llmApiKey: readOption(argv, "--llm-api-key"),
    llmApiKeyEnv: readOption(argv, "--llm-api-key-env"),
    providerId: readOption(argv, "--provider-id"),
    baseUrl: readOption(argv, "--base-url"),
    apiKey: readOption(argv, "--api-key"),
    apiKeyEnv: readOption(argv, "--api-key-env"),
    agentModel: readOption(argv, "--agent-model"),
    memxModel: readOption(argv, "--memx-model"),
    embeddingProvider: embeddingProvider as
      | "local"
      | "off"
      | "openai-compatible"
      | "ollama"
      | "sentence-transformers-local"
      | undefined,
    embeddingModel: readOption(argv, "--embedding-model"),
    embeddingPythonBin: readOption(argv, "--embedding-python"),
    embeddingCacheDir: readOption(argv, "--embedding-cache-dir"),
    embeddingDevice: readOption(argv, "--embedding-device") as
      | "auto"
      | "cpu"
      | "mps"
      | "cuda"
      | undefined,
    configPath: readOption(argv, "--config"),
    openclawBin: readOption(argv, "--openclaw-bin"),
    pythonBin: readOption(argv, "--python"),
    pluginInstallSource: readOption(argv, "--plugin-install-source"),
    skipEmbeddingDeps:
      hasFlag(argv, "--skip-embedding-deps") || hasFlag(argv, "--no-install-embedding-deps"),
    skipPluginInstall: hasFlag(argv, "--skip-plugin-install"),
    skipRestart: hasFlag(argv, "--skip-restart"),
    skipDoctor: hasFlag(argv, "--skip-doctor"),
    dryRun: hasFlag(argv, "--dry-run"),
  };
}

function parseStandaloneQuickstartOptions(
  target: "codex" | "claude-code" | "mcp",
  argv: string[],
) {
  const embeddingProvider = readOption(argv, "--embedding-provider");
  const llmProvider = readOption(argv, "--llm-provider");
  return {
    target,
    llmProvider: llmProvider as
      | "openai-compatible"
      | "anthropic"
      | "google"
      | "ollama"
      | undefined,
    llmBaseUrl: readOption(argv, "--llm-base-url"),
    llmModel: readOption(argv, "--llm-model"),
    llmApiKey: readOption(argv, "--llm-api-key"),
    llmApiKeyEnv: readOption(argv, "--llm-api-key-env"),
    embeddingProvider: embeddingProvider as
      | "local"
      | "off"
      | "openai-compatible"
      | "ollama"
      | "sentence-transformers-local"
      | undefined,
    embeddingModel: readOption(argv, "--embedding-model"),
    embeddingBaseUrl: readOption(argv, "--embedding-base-url"),
    embeddingApiKey: readOption(argv, "--embedding-api-key"),
    embeddingApiKeyEnv: readOption(argv, "--embedding-api-key-env"),
    embeddingPythonBin: readOption(argv, "--embedding-python"),
    embeddingCacheDir: readOption(argv, "--embedding-cache-dir"),
    embeddingDevice: readOption(argv, "--embedding-device") as
      | "auto"
      | "cpu"
      | "mps"
      | "cuda"
      | undefined,
    embeddingOllamaBaseUrl: readOption(argv, "--embedding-ollama-base-url"),
    runtimeDir: readOption(argv, "--runtime-dir"),
    codexBin: readOption(argv, "--codex-bin"),
    codexMarketplaceDir: readOption(argv, "--codex-marketplace-dir"),
    skipCodexPluginInstall: hasFlag(argv, "--skip-codex-plugin-install"),
    claudeBin: readOption(argv, "--claude-bin"),
    claudeMarketplaceDir: readOption(argv, "--claude-marketplace-dir"),
    skipClaudePluginInstall: hasFlag(argv, "--skip-claude-plugin-install"),
    configPath: readOption(argv, "--config"),
    codexConfigPath: readOption(argv, "--codex-config"),
    claudeConfigPath: readOption(argv, "--claude-config"),
    homeDir: readOption(argv, "--home"),
    pythonBin: readOption(argv, "--python"),
    memxUrl: readOption(argv, "--memx-url"),
    memxSecret: readOption(argv, "--memx-secret"),
    mcpTools: readOption(argv, "--mcp-tools") as "full" | "lifecycle-safe" | "none" | undefined,
    skipEmbeddingDeps:
      hasFlag(argv, "--skip-embedding-deps") || hasFlag(argv, "--no-install-embedding-deps"),
    skipServiceStart: hasFlag(argv, "--skip-service-start") || hasFlag(argv, "--no-start-service"),
    dryRun: hasFlag(argv, "--dry-run"),
  };
}

function parseUninstallOptions(target: "openclaw" | "codex" | "claude-code", argv: string[]) {
  return {
    target,
    configPath:
      readOption(argv, "--config") ??
      (target === "codex"
        ? readOption(argv, "--codex-config")
        : target === "claude-code"
          ? readOption(argv, "--claude-config")
          : undefined),
    codexBin: readOption(argv, "--codex-bin"),
    claudeBin: readOption(argv, "--claude-bin"),
    homeDir: readOption(argv, "--home"),
    memxUrl: readOption(argv, "--memx-url"),
    memxSecret: readOption(argv, "--memx-secret"),
    codexMarketplaceDir: readOption(argv, "--codex-marketplace-dir"),
    claudeMarketplaceDir: readOption(argv, "--claude-marketplace-dir"),
    openclawBin: readOption(argv, "--openclaw-bin"),
    skipPluginUninstall: hasFlag(argv, "--skip-plugin-uninstall"),
    dryRun: hasFlag(argv, "--dry-run"),
  };
}

function parseServiceOptions(argv: string[]) {
  const homeDir = readOption(argv, "--home") ?? homedir();
  return {
    homeDir,
    url: readOption(argv, "--memx-url"),
    secret: readOption(argv, "--memx-secret"),
    runtimeDir: readOption(argv, "--runtime-dir") ?? join(homeDir, ".memx", "runtime"),
    configPath: readOption(argv, "--config") ?? join(homeDir, ".memx", "config.json"),
    nodeBin: readOption(argv, "--node"),
  };
}

async function main(argv = process.argv.slice(2)): Promise<void> {
  const command = argv[0];
  if (!command || command === "--help" || command === "-h") {
    console.log(usage());
    return;
  }
  if (command === "server") {
    const { startMemxHttpServer } = await import("../host/httpServer.js");
    await startMemxHttpServer();
    return;
  }
  if (command === "service") {
    const action = argv[1];
    const options = parseServiceOptions(argv.slice(2));
    const {
      ensureMemxService,
      readMemxServiceStatus,
      stopMemxService,
    } = await import("../host/serviceManager.js");
    if (action === "status") {
      const status = await readMemxServiceStatus(options);
      console.log(JSON.stringify({ target: "service", action, ...status }, null, 2));
      return;
    }
    if (action === "stop") {
      const status = await stopMemxService(options);
      console.log(JSON.stringify({ target: "service", action, ...status }, null, 2));
      return;
    }
    if (action === "restart" || action === "start") {
      const status = await ensureMemxService(options);
      console.log(JSON.stringify({ target: "service", action, ...status }, null, 2));
      if (!status.ok) {
        process.exitCode = 1;
      }
      return;
    }
    throw new Error(`unknown service action: ${action ?? ""}`);
  }
  if (command === "mcp") {
    const { startMcpStdio } = await import("../host/mcpStdio.js");
    await startMcpStdio();
    return;
  }
  if (command === "hook") {
    const { runMemxHook } = await import("../host/hookRunner.js");
    await runMemxHook(argv.slice(1));
    return;
  }
  if (command === "quickstart") {
    const target = argv[1];
    if (target === "openclaw") {
      const { runOpenClawQuickstart } = await import("../host/quickstart.js");
      const result = await runOpenClawQuickstart(parseOpenClawQuickstartOptions(argv.slice(2)));
      console.log(JSON.stringify(result, null, 2));
      return;
    }
    if (target === "codex" || target === "claude-code" || target === "mcp") {
      const { runStandaloneMemxQuickstart } = await import("../host/standaloneQuickstart.js");
      const result = await runStandaloneMemxQuickstart(
        parseStandaloneQuickstartOptions(target, argv.slice(2)),
      );
      console.log(JSON.stringify(result, null, 2));
      return;
    }
    throw new Error(`unknown quickstart target: ${target ?? ""}`);
  }
  if (command === "uninstall") {
    const target = argv[1];
    const {
      runClaudeCodeUninstall,
      runCodexUninstall,
      runOpenClawUninstall,
    } = await import("../host/uninstall.js");
    if (target === "openclaw") {
      const result = await runOpenClawUninstall(parseUninstallOptions(target, argv.slice(2)));
      console.log(JSON.stringify(result, null, 2));
      return;
    }
    if (target === "codex") {
      const result = await runCodexUninstall(parseUninstallOptions(target, argv.slice(2)));
      console.log(JSON.stringify(result, null, 2));
      return;
    }
    if (target === "claude-code" || target === "claude") {
      const result = await runClaudeCodeUninstall(
        parseUninstallOptions("claude-code", argv.slice(2)),
      );
      console.log(JSON.stringify(result, null, 2));
      return;
    }
    throw new Error(`unknown uninstall target: ${target ?? ""}`);
  }
  if (command === "connect") {
    const { connectClaudeCodeConfig, connectCodexConfig } = await import("../host/connect.js");
    const target = argv[1];
    if (target === "codex") {
      console.log(`Wired Codex: ${connectCodexConfig()}`);
      return;
    }
    if (target === "claude-code" || target === "claude") {
      console.log(`Wired Claude Code: ${connectClaudeCodeConfig()}`);
      return;
    }
    throw new Error(`unknown connect target: ${target ?? ""}`);
  }
  throw new Error(`unknown command: ${command}`);
}

main().catch((error) => {
  console.error(error instanceof Error ? error.message : String(error));
  process.exit(1);
});
