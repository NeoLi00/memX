import { spawn } from "node:child_process";
import { existsSync } from "node:fs";
import { cp, mkdir, readFile, readdir, rename, rm, writeFile } from "node:fs/promises";
import { homedir, platform } from "node:os";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { DEFAULT_MEMORY_CONFIG } from "../config.js";
import {
  deriveNativeHookBudget,
  MEMX_NATIVE_HOOK_TIMEOUT_MS,
  MEMX_NATIVE_HOOK_TIMEOUT_SECONDS,
} from "../timeouts.js";
import type { MemoryEmbeddingProvider, MemoryLlmProvider, MemoryPluginConfig } from "../types.js";
import {
  applyClaudeJsonConnect,
  applyClaudeJsonDisconnect,
  applyCodexTomlConnect,
  applyCodexTomlDisconnect,
  buildGenericMcpConfig,
  type McpCommandConfig,
  type McpToolsProfile,
} from "./connect.js";
import {
  normalizeStandaloneScopeDefaults,
  STANDALONE_ALLOWED_SCOPES,
  STANDALONE_DEFAULT_SCOPE,
} from "./standaloneConfig.js";
import {
  ensureMemxService,
  type MemxServiceStartOptions,
  type MemxServiceStatus,
} from "./serviceManager.js";

const DEFAULT_CONFIG_PATH = join(homedir(), ".memx", "config.json");
const DEFAULT_DB_PATH = join(homedir(), ".memx", "{agentId}", "memx.sqlite");
const DEFAULT_MEMX_URL = "http://127.0.0.1:3878";
const DEFAULT_EMBEDDING_MODEL = "intfloat/multilingual-e5-small";
const DEFAULT_RUNTIME_DIRNAME = "runtime";
const DEFAULT_CODEX_MARKETPLACE_DIRNAME = "codex-marketplace";
const DEFAULT_CLAUDE_MARKETPLACE_DIRNAME = "claude-marketplace";
const CLAUDE_SETTINGS_BACKUP = "claude-settings-backup.json";
const MEMX_PLUGIN_VERSION = "2026.3.15";

export type StandaloneQuickstartTarget = "codex" | "claude-code" | "mcp";

export type StandaloneMemxQuickstartOptions = {
  target: StandaloneQuickstartTarget;
  llmProvider?: MemoryLlmProvider;
  llmBaseUrl?: string;
  llmModel?: string;
  llmApiKey?: string;
  llmApiKeyEnv?: string;
  embeddingProvider?: "local" | MemoryEmbeddingProvider;
  embeddingModel?: string;
  embeddingBaseUrl?: string;
  embeddingApiKey?: string;
  embeddingApiKeyEnv?: string;
  embeddingPythonBin?: string;
  embeddingCacheDir?: string;
  embeddingDevice?: "auto" | "cpu" | "mps" | "cuda";
  embeddingOllamaBaseUrl?: string;
  runtimeDir?: string;
  codexBin?: string;
  codexMarketplaceDir?: string;
  skipCodexPluginInstall?: boolean;
  claudeBin?: string;
  claudeMarketplaceDir?: string;
  skipClaudePluginInstall?: boolean;
  configPath?: string;
  codexConfigPath?: string;
  claudeConfigPath?: string;
  homeDir?: string;
  pythonBin?: string;
  memxUrl?: string;
  memxSecret?: string;
  mcpTools?: McpToolsProfile;
  skipEmbeddingDeps?: boolean;
  skipServiceStart?: boolean;
  dryRun?: boolean;
};

type NormalizedStandaloneOptions = Required<
  Pick<
    StandaloneMemxQuickstartOptions,
    | "target"
    | "llmProvider"
    | "llmBaseUrl"
    | "llmModel"
    | "configPath"
    | "homeDir"
    | "pythonBin"
    | "memxUrl"
    | "runtimeDir"
    | "codexBin"
    | "codexMarketplaceDir"
    | "claudeBin"
    | "claudeMarketplaceDir"
  >
> &
  Omit<
    StandaloneMemxQuickstartOptions,
    | "target"
    | "llmProvider"
    | "llmBaseUrl"
    | "llmModel"
    | "configPath"
    | "homeDir"
    | "pythonBin"
    | "memxUrl"
    | "runtimeDir"
    | "codexBin"
    | "codexMarketplaceDir"
    | "claudeBin"
    | "claudeMarketplaceDir"
  > & {
    embeddingProvider: MemoryEmbeddingProvider;
    embeddingModel: string;
    embeddingPythonBin: string;
  };

export type StandaloneQuickstartCommandStep = {
  key: string;
  command: string;
  args: string[];
};

export type StandaloneQuickstartCommandResult = {
  code: number;
  stdout?: string;
  stderr?: string;
};

export type StandaloneQuickstartDeps = {
  runCommand?: (command: string, args: string[]) => Promise<StandaloneQuickstartCommandResult>;
  ensureService?: (options: MemxServiceStartOptions) => Promise<MemxServiceStatus>;
};

function trimOrUndefined(value: string | undefined): string | undefined {
  const trimmed = value?.trim();
  return trimmed ? trimmed : undefined;
}

function localVenvDir(homeDir: string): string {
  return join(homeDir, ".memx", ".venv");
}

function localVenvPython(homeDir: string): string {
  const venv = localVenvDir(homeDir);
  return platform() === "win32" ? join(venv, "Scripts", "python.exe") : join(venv, "bin", "python");
}

function localRuntimeDir(homeDir: string): string {
  return join(homeDir, ".memx", DEFAULT_RUNTIME_DIRNAME);
}

function localCodexMarketplaceDir(homeDir: string): string {
  return join(homeDir, ".memx", DEFAULT_CODEX_MARKETPLACE_DIRNAME);
}

function localClaudeMarketplaceDir(homeDir: string): string {
  return join(homeDir, ".memx", DEFAULT_CLAUDE_MARKETPLACE_DIRNAME);
}

function hookRuntimeConfigPath(homeDir: string): string {
  return join(homeDir, ".memx", "hook-runtime.json");
}

function currentRuntimeRoot(): string {
  return fileURLToPath(new URL("../../", import.meta.url));
}

function localRuntimeMcpCommand(runtimeDir: string): McpCommandConfig {
  return {
    command: process.execPath,
    args: [join(runtimeDir, "src", "bin", "memx-mcp.mjs")],
  };
}

function localRuntimeHookCommand(runtimeDir: string): McpCommandConfig {
  return {
    command: process.execPath,
    args: [join(runtimeDir, "src", "bin", "memx-hook.mjs")],
  };
}

function shellQuote(value: string): string {
  if (/^[A-Za-z0-9_/:=.,@%+-]+$/u.test(value)) {
    return value;
  }
  return `'${value.replaceAll("'", "'\\''")}'`;
}

function hookCommandLine(
  commandConfig: McpCommandConfig,
  host: "codex" | "claude-code",
  eventName: string,
  hookConfigPath?: string,
): string {
  return [
    ...[commandConfig.command, ...commandConfig.args],
    host,
    eventName,
    ...(hookConfigPath ? ["--hook-config", hookConfigPath] : []),
  ]
    .map(shellQuote)
    .join(" ");
}

function normalizeEmbeddingProvider(
  provider: StandaloneMemxQuickstartOptions["embeddingProvider"],
): MemoryEmbeddingProvider {
  if (!provider || provider === "local") {
    return "sentence-transformers-local";
  }
  return provider;
}

function secretValue(value: string | undefined, envName: string | undefined): string | undefined {
  const direct = trimOrUndefined(value);
  const env = trimOrUndefined(envName);
  if (direct && env) {
    throw new Error("use either a direct API key or an API key env var, not both");
  }
  if (env) {
    return `\${${env}}`;
  }
  return direct;
}

function isExpectedMissingCleanup(result: StandaloneQuickstartCommandResult): boolean {
  const detail = `${result.stderr ?? ""}\n${result.stdout ?? ""}`;
  return /not found|not installed|not configured/iu.test(detail);
}

function normalizeOptions(
  options: StandaloneMemxQuickstartOptions,
): NormalizedStandaloneOptions {
  const llmProvider = options.llmProvider ?? "openai-compatible";
  const llmBaseUrl = trimOrUndefined(options.llmBaseUrl);
  const llmModel = trimOrUndefined(options.llmModel);
  if (!llmBaseUrl) {
    throw new Error("standalone quickstart requires --llm-base-url");
  }
  if (!llmModel) {
    throw new Error("standalone quickstart requires --llm-model");
  }
  if (
    options.mcpTools &&
    options.mcpTools !== "full" &&
    options.mcpTools !== "lifecycle-safe" &&
    options.mcpTools !== "none"
  ) {
    throw new Error("standalone quickstart requires --mcp-tools to be full, lifecycle-safe, or none");
  }
  const homeDir = options.homeDir ?? homedir();
  const embeddingProvider = normalizeEmbeddingProvider(options.embeddingProvider);
  return {
    ...options,
    target: options.target,
    llmProvider,
    llmBaseUrl,
    llmModel,
    embeddingProvider,
    embeddingModel: trimOrUndefined(options.embeddingModel) ?? DEFAULT_EMBEDDING_MODEL,
    embeddingPythonBin:
      trimOrUndefined(options.embeddingPythonBin) ??
      (embeddingProvider === "sentence-transformers-local" ? localVenvPython(homeDir) : ""),
    configPath: trimOrUndefined(options.configPath) ?? DEFAULT_CONFIG_PATH,
    homeDir,
    pythonBin: trimOrUndefined(options.pythonBin) ?? "python3",
    memxUrl: trimOrUndefined(options.memxUrl) ?? DEFAULT_MEMX_URL,
    runtimeDir: trimOrUndefined(options.runtimeDir) ?? localRuntimeDir(homeDir),
    codexBin: trimOrUndefined(options.codexBin) ?? "codex",
    codexMarketplaceDir:
      trimOrUndefined(options.codexMarketplaceDir) ?? localCodexMarketplaceDir(homeDir),
    claudeBin: trimOrUndefined(options.claudeBin) ?? "claude",
    claudeMarketplaceDir:
      trimOrUndefined(options.claudeMarketplaceDir) ?? localClaudeMarketplaceDir(homeDir),
  };
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return Boolean(value) && typeof value === "object" && !Array.isArray(value);
}

function deepMerge<T>(base: T, override: unknown): T {
  if (!isRecord(base) || !isRecord(override)) {
    return (override === undefined ? base : override) as T;
  }
  const output: Record<string, unknown> = { ...base };
  for (const [key, value] of Object.entries(override)) {
    output[key] = key in output ? deepMerge(output[key], value) : value;
  }
  return output as T;
}

function baseStandaloneConfig(): MemoryPluginConfig {
  const config = structuredClone(DEFAULT_MEMORY_CONFIG);
  config.dbPath = DEFAULT_DB_PATH;
  config.defaultScope = STANDALONE_DEFAULT_SCOPE;
  config.allowedScopes = [...STANDALONE_ALLOWED_SCOPES];
  config.advanced.llmClassifierEnabled = true;
  config.advanced.enableTurnScheduler = true;
  config.advanced.enableCompatibilityMemoryTools = false;
  return config;
}

export function applyStandaloneMemxQuickstartConfig(
  input: unknown,
  rawOptions: StandaloneMemxQuickstartOptions,
): MemoryPluginConfig {
  const options = normalizeOptions(rawOptions);
  const next = normalizeStandaloneScopeDefaults(deepMerge(baseStandaloneConfig(), input));
  next.advanced.queryCompilerHotPathTimeoutMs =
    DEFAULT_MEMORY_CONFIG.advanced.queryCompilerHotPathTimeoutMs;
  next.advanced.llmProvider = options.llmProvider;
  next.advanced.llmBaseURL = options.llmBaseUrl;
  next.advanced.llmClassifierModel = options.llmModel;
  next.advanced.llmApiKey = secretValue(options.llmApiKey, options.llmApiKeyEnv);
  next.embedding.provider = options.embeddingProvider;
  next.embedding.model = options.embeddingModel;
  if (options.embeddingProvider === "openai-compatible") {
    next.embedding.baseURL = trimOrUndefined(options.embeddingBaseUrl);
    next.embedding.apiKey = secretValue(options.embeddingApiKey, options.embeddingApiKeyEnv);
  }
  if (options.embeddingProvider === "ollama") {
    next.embedding.ollamaBaseURL = trimOrUndefined(options.embeddingOllamaBaseUrl);
  }
  if (options.embeddingProvider === "sentence-transformers-local") {
    next.embedding.localPythonBin = options.embeddingPythonBin;
    next.embedding.localDevice = options.embeddingDevice ?? "auto";
    if (trimOrUndefined(options.embeddingCacheDir)) {
      next.embedding.localCacheDir = trimOrUndefined(options.embeddingCacheDir);
    }
  }
  return next;
}

export function buildStandaloneMemxQuickstartSteps(
  rawOptions: StandaloneMemxQuickstartOptions,
): StandaloneQuickstartCommandStep[] {
  const options = normalizeOptions(rawOptions);
  if (options.embeddingProvider !== "sentence-transformers-local" || options.skipEmbeddingDeps) {
    return [];
  }
  return [
    {
      key: "embedding-venv",
      command: options.pythonBin,
      args: ["-m", "venv", localVenvDir(options.homeDir)],
    },
    {
      key: "embedding-deps",
      command: options.embeddingPythonBin,
      args: ["-m", "pip", "install", "-U", "pip", "sentence-transformers", "torch"],
    },
  ];
}

async function readJson(path: string): Promise<Record<string, unknown>> {
  if (!existsSync(path)) {
    return {};
  }
  return JSON.parse(await readFile(path, "utf8")) as Record<string, unknown>;
}

async function writeAtomic(path: string, text: string): Promise<void> {
  await mkdir(dirname(path), { recursive: true });
  const tmp = `${path}.tmp-${process.pid}-${Date.now()}`;
  await writeFile(tmp, text, "utf8");
  await rename(tmp, path);
}

async function writeHookRuntimeConfig(
  path: string,
  options: NormalizedStandaloneOptions,
): Promise<void> {
  const budget = deriveNativeHookBudget(MEMX_NATIVE_HOOK_TIMEOUT_MS);
  const config = {
    memxUrl: options.memxUrl,
    ...(options.memxSecret ? { memxSecret: options.memxSecret } : {}),
    hookTimeoutMs: budget.hookTimeoutMs,
  };
  await writeAtomic(path, `${JSON.stringify(config, null, 2)}\n`);
}

function settingSnapshot(record: Record<string, unknown>, key: string): Record<string, unknown> {
  return Object.hasOwn(record, key)
    ? { present: true, value: record[key] }
    : { present: false };
}

function claudeSettingsPath(homeDir: string): string {
  return join(homeDir, ".claude", "settings.json");
}

function claudeSettingsBackupPath(homeDir: string): string {
  return join(homeDir, ".memx", CLAUDE_SETTINGS_BACKUP);
}

async function writeClaudeNativeSettings(options: NormalizedStandaloneOptions): Promise<Record<string, string>> {
  const path = claudeSettingsPath(options.homeDir);
  const backupPath = claudeSettingsBackupPath(options.homeDir);
  const current = await readJson(path);
  const currentEnv = isRecord(current.env) ? current.env : {};
  if (!existsSync(backupPath)) {
    await writeAtomic(
      backupPath,
      `${JSON.stringify(
        {
          autoMemoryEnabled: settingSnapshot(current, "autoMemoryEnabled"),
          env: {
            CLAUDE_CODE_DISABLE_AUTO_MEMORY: settingSnapshot(
              currentEnv,
              "CLAUDE_CODE_DISABLE_AUTO_MEMORY",
            ),
          },
        },
        null,
        2,
      )}\n`,
    );
  }
  const next = {
    ...current,
    autoMemoryEnabled: false,
    env: {
      ...currentEnv,
      CLAUDE_CODE_DISABLE_AUTO_MEMORY: "1",
    },
  };
  await writeAtomic(path, `${JSON.stringify(next, null, 2)}\n`);
  return { path, backupPath };
}

async function disconnectCodexMcpConfig(path: string): Promise<string | undefined> {
  if (!existsSync(path)) {
    return undefined;
  }
  const current = await readFile(path, "utf8");
  const next = applyCodexTomlDisconnect(current);
  if (next !== current.trimEnd()) {
    await writeAtomic(path, next ? `${next}\n` : "");
  }
  return path;
}

async function disconnectClaudeMcpConfig(path: string): Promise<string | undefined> {
  if (!existsSync(path)) {
    return undefined;
  }
  const current = await readJson(path);
  const next = applyClaudeJsonDisconnect(current);
  await writeAtomic(path, `${JSON.stringify(next, null, 2)}\n`);
  return path;
}

async function installStandaloneRuntime(runtimeDir: string): Promise<string> {
  const tmp = `${runtimeDir}.tmp-${process.pid}-${Date.now()}`;
  await rm(tmp, { recursive: true, force: true });
  await mkdir(dirname(runtimeDir), { recursive: true });
  await cp(currentRuntimeRoot(), tmp, { recursive: true });
  await rm(runtimeDir, { recursive: true, force: true });
  await rename(tmp, runtimeDir);
  return runtimeDir;
}

async function removeCachedMemxPlugin(cacheRoot: string): Promise<void> {
  let marketplaceEntries: string[];
  try {
    marketplaceEntries = await readdir(cacheRoot);
  } catch {
    return;
  }
  await Promise.all(
    marketplaceEntries.map((marketplace) =>
      rm(join(cacheRoot, marketplace, "memx", MEMX_PLUGIN_VERSION), {
        recursive: true,
        force: true,
      }),
    ),
  );
}

async function removeCodexPluginCache(homeDir: string): Promise<void> {
  await removeCachedMemxPlugin(join(homeDir, ".codex", "plugins", "cache"));
}

async function removeClaudePluginCache(homeDir: string): Promise<void> {
  await removeCachedMemxPlugin(join(homeDir, ".claude", "plugins", "cache"));
}

function hookEntry(
  commandConfig: McpCommandConfig,
  host: "codex" | "claude-code",
  eventName: string,
  hookConfigPath?: string,
  statusMessage?: string,
) {
  return {
    hooks: [
      {
        type: "command",
        command: hookCommandLine(commandConfig, host, eventName, hookConfigPath),
        timeout: MEMX_NATIVE_HOOK_TIMEOUT_SECONDS,
        ...(statusMessage ? { statusMessage } : {}),
      },
    ],
  };
}

function codexHooksConfig(
  commandConfig: McpCommandConfig,
  hookConfigPath?: string,
): Record<string, unknown> {
  return {
    hooks: {
      SessionStart: [
        hookEntry(
          commandConfig,
          "codex",
          "SessionStart",
          hookConfigPath,
          "memx: opening memory session",
        ),
      ],
      UserPromptSubmit: [
        hookEntry(
          commandConfig,
          "codex",
          "UserPromptSubmit",
          hookConfigPath,
          "memx: recalling memory",
        ),
      ],
      PreToolUse: [
        {
          matcher: "Edit|Write|Read|Glob|Grep|apply_patch|exec_command",
          ...hookEntry(commandConfig, "codex", "PreToolUse", hookConfigPath),
        },
      ],
      PostToolUse: [hookEntry(commandConfig, "codex", "PostToolUse", hookConfigPath)],
      PreCompact: [hookEntry(commandConfig, "codex", "PreCompact", hookConfigPath)],
      Stop: [hookEntry(commandConfig, "codex", "Stop", hookConfigPath)],
    },
  };
}

function codexPluginManifest(includeMcp = false): Record<string, unknown> {
  const manifest: Record<string, unknown> = {
    name: "memx",
    version: MEMX_PLUGIN_VERSION,
    description:
      "memX: local-first semantic memory for coding agents. Native Codex lifecycle hooks.",
    author: { name: "Neo Li" },
    license: "MIT",
    homepage: "https://github.com/NeoLi00/memX",
    repository: "https://github.com/NeoLi00/memX",
    hooks: "./hooks.json",
    interface: {
      displayName: "memX",
      shortDescription: "Local semantic memory for coding agents.",
      longDescription:
        "memX adds local-first lifecycle memory through native hooks. It recalls relevant context before each turn and writes completed turns after the agent responds.",
      developerName: "Neo Li",
      category: "Productivity",
      capabilities: ["Read", "Write"],
      websiteURL: "https://github.com/NeoLi00/memX",
      privacyPolicyURL: "https://github.com/NeoLi00/memX",
      termsOfServiceURL: "https://github.com/NeoLi00/memX",
      defaultPrompt: ["Use memX automatic memory hooks."],
      brandColor: "#2563EB",
    },
  };
  if (includeMcp) {
    manifest.mcpServers = "./.mcp.json";
  }
  return manifest;
}

function claudePluginManifest(includeMcp = false): Record<string, unknown> {
  const manifest: Record<string, unknown> = {
    name: "memx",
    version: MEMX_PLUGIN_VERSION,
    description:
      "memX: local-first semantic memory for coding agents. Native Claude Code lifecycle hooks.",
    author: { name: "Neo Li" },
    license: "MIT",
    homepage: "https://github.com/NeoLi00/memX",
    repository: "https://github.com/NeoLi00/memX",
  };
  if (includeMcp) {
    manifest.mcpServers = "./.mcp.json";
  }
  return manifest;
}

function codexMarketplaceManifest(): Record<string, unknown> {
  return {
    name: "memx",
    interface: {
      displayName: "memX",
    },
    plugins: [
      {
        name: "memx",
        source: { source: "local", path: "./plugins/memx" },
        policy: { installation: "AVAILABLE", authentication: "ON_INSTALL" },
        category: "Productivity",
      },
    ],
  };
}

function claudeMarketplaceManifest(): Record<string, unknown> {
  return {
    $schema: "https://anthropic.com/claude-code/marketplace.schema.json",
    name: "memx",
    description: "memX local plugin marketplace for Claude Code native lifecycle hooks.",
    owner: { name: "Neo Li" },
    plugins: [
      {
        name: "memx",
        description: "memX: local-first semantic memory for coding agents.",
        author: { name: "Neo Li" },
        source: "./plugins/memx",
        category: "productivity",
        homepage: "https://github.com/NeoLi00/memX",
      },
    ],
  };
}

function claudeHookConfigArg(path: string): string {
  return path.includes("${CLAUDE_PLUGIN_ROOT}") ? `"${path}"` : shellQuote(path);
}

function claudeHooksConfig(
  hookConfigPath = "${CLAUDE_PLUGIN_ROOT}/.memx-hook.json",
): Record<string, unknown> {
  const hook = (eventName: string) => ({
    hooks: [
      {
        type: "command",
        command: `node "\${CLAUDE_PLUGIN_ROOT}/dist/.runtime/src/bin/memx-hook.mjs" claude-code ${eventName} --hook-config ${claudeHookConfigArg(hookConfigPath)}`,
        timeout: MEMX_NATIVE_HOOK_TIMEOUT_SECONDS,
      },
    ],
  });
  return {
    hooks: {
      SessionStart: [hook("SessionStart")],
      UserPromptSubmit: [hook("UserPromptSubmit")],
      PreToolUse: [
        {
          matcher: "Edit|Write|Read|Glob|Grep",
          ...hook("PreToolUse"),
        },
      ],
      PostToolUse: [hook("PostToolUse")],
      PostToolUseFailure: [hook("PostToolUseFailure")],
      PreCompact: [hook("PreCompact")],
      SubagentStart: [hook("SubagentStart")],
      SubagentStop: [hook("SubagentStop")],
      Notification: [hook("Notification")],
      TaskCompleted: [hook("TaskCompleted")],
      Stop: [hook("Stop")],
      SessionEnd: [hook("SessionEnd")],
    },
  };
}

function claudeMcpConfig(
  options: NormalizedStandaloneOptions,
  mcpTools: McpToolsProfile,
): Record<string, unknown> {
  return {
    mcpServers: {
      memx: {
        command: process.execPath,
        args: ["${CLAUDE_PLUGIN_ROOT}/dist/.runtime/src/bin/memx-mcp.mjs"],
        env: {
          MEMX_URL: options.memxUrl,
          MEMX_SECRET: options.memxSecret ?? "",
          MEMX_MCP_TOOLS: mcpTools,
        },
      },
    },
  };
}

async function installClaudeMarketplaceSnapshot(
  options: NormalizedStandaloneOptions,
  mcpTools: McpToolsProfile,
): Promise<string> {
  const includeMcp = mcpTools !== "none";
  const marketplaceDir = options.claudeMarketplaceDir;
  const pluginDir = join(marketplaceDir, "plugins", "memx");
  const tmpPluginDir = join("plugins", "memx");
  const tmp = `${marketplaceDir}.tmp-${process.pid}-${Date.now()}`;
  await rm(tmp, { recursive: true, force: true });
  await mkdir(join(tmp, ".claude-plugin"), { recursive: true });
  await mkdir(join(tmp, tmpPluginDir, ".claude-plugin"), { recursive: true });
  await mkdir(join(tmp, tmpPluginDir, "hooks"), { recursive: true });
  await writeFile(
    join(tmp, ".claude-plugin", "marketplace.json"),
    `${JSON.stringify(claudeMarketplaceManifest(), null, 2)}\n`,
    "utf8",
  );
  await writeFile(
    join(tmp, tmpPluginDir, ".claude-plugin", "plugin.json"),
    `${JSON.stringify(claudePluginManifest(includeMcp), null, 2)}\n`,
    "utf8",
  );
  if (includeMcp) {
    await writeFile(
      join(tmp, tmpPluginDir, ".mcp.json"),
      `${JSON.stringify(claudeMcpConfig(options, mcpTools), null, 2)}\n`,
      "utf8",
    );
  }
  await writeFile(
    join(tmp, tmpPluginDir, "hooks", "hooks.json"),
    `${JSON.stringify(claudeHooksConfig(), null, 2)}\n`,
    "utf8",
  );
  await writeHookRuntimeConfig(join(tmp, tmpPluginDir, ".memx-hook.json"), options);
  await mkdir(join(tmp, tmpPluginDir, "dist"), { recursive: true });
  await cp(currentRuntimeRoot(), join(tmp, tmpPluginDir, "dist", ".runtime"), { recursive: true });
  await rm(marketplaceDir, { recursive: true, force: true });
  await mkdir(dirname(marketplaceDir), { recursive: true });
  await rename(tmp, marketplaceDir);
  return pluginDir;
}

async function installCodexMarketplaceSnapshot(
  options: NormalizedStandaloneOptions,
  hookCommandConfig: McpCommandConfig,
  mcpTools: McpToolsProfile,
  hookConfigPath?: string,
): Promise<string> {
  const includeMcp = mcpTools !== "none";
  const marketplaceDir = options.codexMarketplaceDir;
  const pluginDir = join(marketplaceDir, "plugins", "memx");
  const tmp = `${marketplaceDir}.tmp-${process.pid}-${Date.now()}`;
  await rm(tmp, { recursive: true, force: true });
  await mkdir(join(tmp, ".agents", "plugins"), { recursive: true });
  await mkdir(join(tmp, "plugins", "memx", ".codex-plugin"), { recursive: true });
  await mkdir(join(tmp, "plugins", "memx", "hooks"), { recursive: true });
  const hooksJson = `${JSON.stringify(codexHooksConfig(hookCommandConfig, hookConfigPath), null, 2)}\n`;
  await writeFile(
    join(tmp, ".agents", "plugins", "marketplace.json"),
    `${JSON.stringify(codexMarketplaceManifest(), null, 2)}\n`,
    "utf8",
  );
  await writeFile(
    join(tmp, "plugins", "memx", ".codex-plugin", "plugin.json"),
    `${JSON.stringify(codexPluginManifest(includeMcp), null, 2)}\n`,
    "utf8",
  );
  if (includeMcp) {
    await writeFile(
      join(tmp, "plugins", "memx", ".mcp.json"),
      `${JSON.stringify(claudeMcpConfig(options, mcpTools), null, 2)}\n`,
      "utf8",
    );
  }
  await writeFile(
    join(tmp, "plugins", "memx", "hooks", "hooks.codex.json"),
    hooksJson,
    "utf8",
  );
  await writeFile(
    join(tmp, "plugins", "memx", "hooks.json"),
    hooksJson,
    "utf8",
  );
  await rm(marketplaceDir, { recursive: true, force: true });
  await mkdir(dirname(marketplaceDir), { recursive: true });
  await rename(tmp, marketplaceDir);
  return pluginDir;
}

async function installCodexPlugin(
  options: NormalizedStandaloneOptions,
  hookCommandConfig: McpCommandConfig,
  mcpTools: McpToolsProfile,
  hookConfigPath: string,
  runCommand: (command: string, args: string[]) => Promise<StandaloneQuickstartCommandResult>,
  runBestEffortCommand: (command: string, args: string[]) => Promise<StandaloneQuickstartCommandResult>,
): Promise<Record<string, unknown>> {
  await installCodexMarketplaceSnapshot(options, hookCommandConfig, mcpTools, hookConfigPath);
  await removeCodexPluginCache(options.homeDir);
  const warnings: string[] = [];
  const bestEffort = async (args: string[]) => {
    const result = await runBestEffortCommand(options.codexBin, args);
    if (result.code !== 0 && !isExpectedMissingCleanup(result)) {
      const detail = (result.stderr || result.stdout || "").trim();
      warnings.push(
        `${options.codexBin} ${args.join(" ")} exited ${result.code}${detail ? `: ${detail}` : ""}`,
      );
    }
  };
  await bestEffort(["plugin", "remove", "memx@memx"]);
  await bestEffort(["plugin", "marketplace", "remove", "memx"]);
  const addMarketplace = await runCommand(options.codexBin, [
    "plugin",
    "marketplace",
    "add",
    options.codexMarketplaceDir,
  ]);
  if (addMarketplace.code !== 0) {
    throw new Error(
      `standalone quickstart step failed: codex-plugin-marketplace (${options.codexBin} plugin marketplace add ${options.codexMarketplaceDir}) exited ${addMarketplace.code}`,
    );
  }
  const addPlugin = await runCommand(options.codexBin, ["plugin", "add", "memx@memx"]);
  if (addPlugin.code !== 0) {
    throw new Error(
      `standalone quickstart step failed: codex-plugin-install (${options.codexBin} plugin add memx@memx) exited ${addPlugin.code}`,
    );
  }
  return {
    marketplaceDir: options.codexMarketplaceDir,
    installed: true,
    warnings,
  };
}

async function installClaudePlugin(
  options: NormalizedStandaloneOptions,
  mcpTools: McpToolsProfile,
  runCommand: (command: string, args: string[]) => Promise<StandaloneQuickstartCommandResult>,
  runBestEffortCommand: (command: string, args: string[]) => Promise<StandaloneQuickstartCommandResult>,
): Promise<Record<string, unknown>> {
  await installClaudeMarketplaceSnapshot(options, mcpTools);
  await removeClaudePluginCache(options.homeDir);
  const warnings: string[] = [];
  const bestEffort = async (args: string[]) => {
    const result = await runBestEffortCommand(options.claudeBin, args);
    if (result.code !== 0 && !isExpectedMissingCleanup(result)) {
      const detail = (result.stderr || result.stdout || "").trim();
      warnings.push(
        `${options.claudeBin} ${args.join(" ")} exited ${result.code}${detail ? `: ${detail}` : ""}`,
      );
    }
  };
  await bestEffort(["plugin", "uninstall", "memx@memx"]);
  await bestEffort(["plugin", "uninstall", "memx"]);
  await bestEffort(["plugin", "marketplace", "remove", "memx"]);
  const addMarketplace = await runCommand(options.claudeBin, [
    "plugin",
    "marketplace",
    "add",
    options.claudeMarketplaceDir,
  ]);
  if (addMarketplace.code !== 0) {
    throw new Error(
      `standalone quickstart step failed: claude-plugin-marketplace (${options.claudeBin} plugin marketplace add ${options.claudeMarketplaceDir}) exited ${addMarketplace.code}`,
    );
  }
  const addPlugin = await runCommand(options.claudeBin, ["plugin", "install", "memx@memx"]);
  if (addPlugin.code !== 0) {
    throw new Error(
      `standalone quickstart step failed: claude-plugin-install (${options.claudeBin} plugin install memx@memx) exited ${addPlugin.code}`,
    );
  }
  return {
    marketplaceDir: options.claudeMarketplaceDir,
    installed: true,
    warnings,
  };
}

async function defaultRunCommand(
  command: string,
  args: string[],
): Promise<StandaloneQuickstartCommandResult> {
  return new Promise((resolve, reject) => {
    const child = spawn(command, args, { shell: false, stdio: "inherit" });
    child.once("error", reject);
    child.once("close", (code) => resolve({ code: code ?? 1 }));
  });
}

async function defaultRunCommandQuiet(
  command: string,
  args: string[],
): Promise<StandaloneQuickstartCommandResult> {
  return new Promise((resolve) => {
    const child = spawn(command, args, { shell: false, stdio: "pipe" });
    let stdout = "";
    let stderr = "";
    child.stdout?.on("data", (chunk) => {
      stdout += String(chunk);
    });
    child.stderr?.on("data", (chunk) => {
      stderr += String(chunk);
    });
    child.once("error", (error) => {
      resolve({ code: 1, stderr: error.message });
    });
    child.once("close", (code) => resolve({ code: code ?? 1, stdout, stderr }));
  });
}

async function writeHostConfig(
  options: NormalizedStandaloneOptions,
  commandConfig: McpCommandConfig,
  codexPlugin: Record<string, unknown> | null,
  claudePlugin: Record<string, unknown> | null,
  mcpTools: McpToolsProfile,
): Promise<Record<string, unknown> | null> {
  if (options.target === "codex") {
    if (codexPlugin && mcpTools === "none") {
      const path = await disconnectCodexMcpConfig(
        trimOrUndefined(options.codexConfigPath) ?? join(options.homeDir, ".codex", "config.toml"),
      );
      return { host: "codex", path, codexPlugin, mcpTools };
    }
    const path = trimOrUndefined(options.codexConfigPath) ?? join(options.homeDir, ".codex", "config.toml");
    const current = existsSync(path) ? await readFile(path, "utf8") : "";
    await writeAtomic(
      path,
      applyCodexTomlConnect(
        current,
        options.memxUrl,
        options.memxSecret ?? "",
        commandConfig,
        mcpTools,
      ),
    );
    return { host: "codex", path, codexPlugin };
  }
  if (options.target === "claude-code") {
    if (claudePlugin) {
      const path = await disconnectClaudeMcpConfig(
        trimOrUndefined(options.claudeConfigPath) ?? join(options.homeDir, ".claude.json"),
      );
      const settings = await writeClaudeNativeSettings(options);
      return {
        host: "claude-code",
        path,
        claudePlugin,
        mcpTools,
        settingsPath: settings.path,
        settingsBackupPath: settings.backupPath,
      };
    }
    const path = trimOrUndefined(options.claudeConfigPath) ?? join(options.homeDir, ".claude.json");
    const current = await readJson(path);
    const next = applyClaudeJsonConnect(
      current,
      options.memxUrl,
      options.memxSecret ?? "",
      commandConfig,
      mcpTools,
    );
    await writeAtomic(path, `${JSON.stringify(next, null, 2)}\n`);
    return { host: "claude-code", path };
  }
  return null;
}

function redactSummary(
  options: NormalizedStandaloneOptions,
  steps: StandaloneQuickstartCommandStep[],
  commandConfig: McpCommandConfig,
  codexPlugin: Record<string, unknown> | null,
  claudePlugin: Record<string, unknown> | null,
  mcpTools: McpToolsProfile,
) {
  const lifecycleNotes = buildLifecycleNotes(options, mcpTools);
  return {
    target: options.target,
    configPath: options.configPath,
    llmProvider: options.llmProvider,
    llmBaseUrl: options.llmBaseUrl,
    llmModel: options.llmModel,
    llmApiKey: options.llmApiKeyEnv
      ? { source: "env", id: options.llmApiKeyEnv }
      : options.llmApiKey
        ? "plaintext-redacted"
        : null,
    embeddingProvider: options.embeddingProvider,
    embeddingModel: options.embeddingModel,
    embeddingPythonBin: options.embeddingPythonBin || null,
    memxUrl: options.memxUrl,
    mcpTools,
    runtimeDir: options.runtimeDir,
    codexPlugin,
    claudePlugin,
    lifecycleNotes,
    steps,
    mcpConfig:
      options.target === "mcp"
        ? buildGenericMcpConfig(options.memxUrl, options.memxSecret ?? "", commandConfig, mcpTools)
        : undefined,
  };
}

function buildLifecycleNotes(
  options: NormalizedStandaloneOptions,
  mcpTools: McpToolsProfile,
): string[] {
  const notes: string[] = [];
  if (options.target === "codex" && !options.skipCodexPluginInstall) {
    notes.push(
      "Codex may ask you to trust the memX plugin on first launch; this is the host trust gate and quickstart cannot bypass it.",
    );
    notes.push(
      "Use an interactive Codex TUI session to validate lifecycle hooks; codex exec can follow a different host path and is not a reliable lifecycle acceptance test.",
    );
  }
  if (
    (options.target === "codex" && !options.skipCodexPluginInstall) ||
    (options.target === "claude-code" && !options.skipClaudePluginInstall)
  ) {
    notes.push(
      mcpTools === "none"
        ? "Native hooks are installed and MCP memory tools are hidden by default to avoid duplicate recall/write."
        : "MCP memory tools are exposed because --mcp-tools was explicitly set.",
    );
  }
  return notes;
}

function defaultMcpToolsForOptions(options: NormalizedStandaloneOptions): McpToolsProfile {
  if (options.mcpTools) {
    return options.mcpTools;
  }
  if (options.target === "codex" && !options.skipCodexPluginInstall) {
    return "none";
  }
  if (options.target === "claude-code" && !options.skipClaudePluginInstall) {
    return "none";
  }
  return "full";
}

export async function runStandaloneMemxQuickstart(
  rawOptions: StandaloneMemxQuickstartOptions,
  deps: StandaloneQuickstartDeps = {},
): Promise<Record<string, unknown>> {
  const options = normalizeOptions(rawOptions);
  const mcpTools = defaultMcpToolsForOptions(options);
  const current = await readJson(options.configPath);
  const next = applyStandaloneMemxQuickstartConfig(current, options);
  const steps = buildStandaloneMemxQuickstartSteps(options);
  const commandConfig = localRuntimeMcpCommand(options.runtimeDir);
  const hookCommandConfig = localRuntimeHookCommand(options.runtimeDir);
  let hostConfig: Record<string, unknown> | null = null;
  let codexPlugin: Record<string, unknown> | null = null;
  let claudePlugin: Record<string, unknown> | null = null;
  let service: MemxServiceStatus | null = null;
  if (!options.dryRun) {
    await installStandaloneRuntime(options.runtimeDir);
    await writeAtomic(options.configPath, `${JSON.stringify(next, null, 2)}\n`);
    const hookConfigPath = hookRuntimeConfigPath(options.homeDir);
    await writeHookRuntimeConfig(hookConfigPath, options);
    const runCommand = deps.runCommand ?? defaultRunCommand;
    const runBestEffortCommand = deps.runCommand ?? defaultRunCommandQuiet;
    if (options.target === "codex" && !options.skipCodexPluginInstall) {
      codexPlugin = await installCodexPlugin(
        options,
        hookCommandConfig,
        mcpTools,
        hookConfigPath,
        runCommand,
        runBestEffortCommand,
      );
    }
    if (options.target === "claude-code" && !options.skipClaudePluginInstall) {
      claudePlugin = await installClaudePlugin(options, mcpTools, runCommand, runBestEffortCommand);
    }
    hostConfig = await writeHostConfig(options, commandConfig, codexPlugin, claudePlugin, mcpTools);
    for (const step of steps) {
      const result = await runCommand(step.command, step.args);
      if (result.code !== 0) {
        throw new Error(
          `standalone quickstart step failed: ${step.key} (${step.command} ${step.args.join(" ")}) exited ${result.code}`,
        );
      }
    }
    if (!options.skipServiceStart) {
      const ensureService = deps.ensureService ?? ensureMemxService;
      service = await ensureService({
        homeDir: options.homeDir,
        runtimeDir: options.runtimeDir,
        configPath: options.configPath,
        url: options.memxUrl,
        secret: options.memxSecret,
      });
      if (!service.ok) {
        throw new Error(
          `standalone quickstart step failed: service-start (${service.error ?? "health check failed"})`,
        );
      }
    }
  }
  return {
    ok: true,
    dryRun: Boolean(options.dryRun),
    ...redactSummary(options, steps, commandConfig, codexPlugin, claudePlugin, mcpTools),
    hostConfig,
    service,
    nextStep: service?.ok
      ? "Use the configured host; memX service is running."
      : "Run `memx service restart` with this config, then use the configured host.",
  };
}
