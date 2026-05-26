import { spawn } from "node:child_process";
import { existsSync } from "node:fs";
import { cp, mkdir, mkdtemp, readFile, rename, rm, writeFile } from "node:fs/promises";
import { homedir, platform, tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { DEFAULT_MEMORY_CONFIG } from "../config.js";
import {
  LEGACY_MEMX_PLUGIN_ID,
  MEMX_PLUGIN_ID,
  withoutLegacyPluginIds,
} from "../identity.js";
import type { MemoryLlmProvider, MemoryPluginConfig } from "../types.js";

const PLUGIN_ID = MEMX_PLUGIN_ID;
const DEFAULT_CONFIG_PATH = join(homedir(), ".openclaw", "openclaw.json");
const DEFAULT_EMBEDDING_MODEL = "intfloat/multilingual-e5-small";

type SecretRef = { source: "env"; provider: "default"; id: string };

type OpenClawConfigLike = {
  agents?: Record<string, unknown>;
  models?: Record<string, unknown>;
  plugins?: {
    allow?: string[];
    slots?: Record<string, string>;
    entries?: Record<string, Record<string, unknown>>;
    [key: string]: unknown;
  };
  [key: string]: unknown;
};

export type OpenClawQuickstartOptions = {
  llmProvider?: MemoryLlmProvider;
  llmBaseUrl?: string;
  llmModel?: string;
  llmApiKey?: string;
  llmApiKeyEnv?: string;
  /** @deprecated Use llmProvider plus the default provider id. */
  providerId?: string;
  /** @deprecated Use llmBaseUrl. */
  baseUrl?: string;
  /** @deprecated Use llmApiKey. */
  apiKey?: string;
  /** @deprecated Use llmApiKeyEnv. */
  apiKeyEnv?: string;
  /** @deprecated Ignored. OpenClaw quickstart no longer changes the agent's primary model. */
  agentModel?: string;
  /** @deprecated Use llmModel. */
  memxModel?: string;
  embeddingProvider?: "local" | MemoryPluginConfig["embedding"]["provider"];
  embeddingModel?: string;
  embeddingPythonBin?: string;
  embeddingCacheDir?: string;
  embeddingDevice?: "auto" | "cpu" | "mps" | "cuda";
  configPath?: string;
  homeDir?: string;
  openclawBin?: string;
  pythonBin?: string;
  pluginInstallSource?: string;
  skipEmbeddingDeps?: boolean;
  skipPluginInstall?: boolean;
  skipRestart?: boolean;
  skipDoctor?: boolean;
  dryRun?: boolean;
};

type NormalizedOpenClawQuickstartOptions = Required<
  Pick<
    OpenClawQuickstartOptions,
    | "llmProvider"
    | "llmBaseUrl"
    | "llmModel"
    | "embeddingModel"
    | "configPath"
    | "homeDir"
    | "openclawBin"
    | "pythonBin"
    | "pluginInstallSource"
  >
> &
  Omit<
    OpenClawQuickstartOptions,
    | "llmProvider"
    | "llmBaseUrl"
    | "llmModel"
    | "embeddingModel"
    | "configPath"
    | "homeDir"
    | "openclawBin"
    | "pythonBin"
    | "pluginInstallSource"
  > & {
    embeddingProvider: MemoryPluginConfig["embedding"]["provider"];
    embeddingPythonBin: string;
    llmApiKey?: string;
    llmApiKeyEnv?: string;
  };

export type QuickstartCommandStep = {
  key: string;
  command: string;
  args: string[];
};

export type QuickstartCommandResult = {
  code: number;
  stdout?: string;
  stderr?: string;
};

export type QuickstartDeps = {
  runCommand?: (command: string, args: string[]) => Promise<QuickstartCommandResult>;
};

function asConfig(input: unknown): OpenClawConfigLike {
  return input && typeof input === "object" && !Array.isArray(input)
    ? (structuredClone(input) as OpenClawConfigLike)
    : {};
}

function trimOrUndefined(value: string | undefined): string | undefined {
  const trimmed = value?.trim();
  return trimmed ? trimmed : undefined;
}

function localVenvDir(homeDir: string): string {
  return join(homeDir, ".openclaw", "memx", ".venv");
}

function localVenvPython(homeDir: string): string {
  const venv = localVenvDir(homeDir);
  return platform() === "win32" ? join(venv, "Scripts", "python.exe") : join(venv, "bin", "python");
}

const OPENCLAW_SAFE_PACKAGE_FILES = [
  "dist",
  "openclaw.plugin.json",
  "README.md",
  "README-ch.md",
  "ARCHITECTURE.md",
  "ARCHITECTURE-ch.md",
  "assets",
];

const OPENCLAW_STANDALONE_PACKAGE_PATHS = [
  ".agents",
  ".codex-plugin",
  ".claude-plugin",
  ".mcp.json",
  "hooks.json",
  "hooks",
  "skills",
  "dist/.runtime/src/bin",
  "dist/.runtime/src/host",
];

export function resolveCurrentPackageRoot(): string {
  return join(dirname(fileURLToPath(import.meta.url)), "..", "..", "..", "..");
}

async function prepareOpenClawInstallPackage(packageRoot: string): Promise<string> {
  const targetDir = await mkdtemp(join(tmpdir(), "memx-openclaw-plugin-"));
  const packageJsonPath = join(packageRoot, "package.json");
  const rawPackage = JSON.parse(await readFile(packageJsonPath, "utf8")) as {
    files?: unknown;
  };
  const entries = [
    "package.json",
    ...(Array.isArray(rawPackage.files)
      ? rawPackage.files.filter((entry): entry is string => typeof entry === "string")
      : []),
  ];
  for (const entry of new Set(entries)) {
    const source = join(packageRoot, entry);
    if (!existsSync(source)) {
      continue;
    }
    await cp(source, join(targetDir, entry), { recursive: true });
  }
  for (const entry of OPENCLAW_STANDALONE_PACKAGE_PATHS) {
    await rm(join(targetDir, entry), { recursive: true, force: true });
  }
  const sanitizedPackage: Record<string, unknown> = {
    ...(rawPackage as Record<string, unknown>),
    files: OPENCLAW_SAFE_PACKAGE_FILES,
  };
  delete sanitizedPackage.bin;
  delete sanitizedPackage.scripts;
  await writeFile(
    join(targetDir, "package.json"),
    `${JSON.stringify(sanitizedPackage, null, 2)}\n`,
    "utf8",
  );
  return targetDir;
}

function normalizeEmbeddingProvider(
  provider: OpenClawQuickstartOptions["embeddingProvider"],
): MemoryPluginConfig["embedding"]["provider"] {
  if (!provider || provider === "local") {
    return "sentence-transformers-local";
  }
  return provider;
}

function normalizeLlmProvider(provider: string | undefined): MemoryLlmProvider | undefined {
  const trimmed = trimOrUndefined(provider);
  if (
    trimmed === "openai-compatible" ||
    trimmed === "anthropic" ||
    trimmed === "google" ||
    trimmed === "ollama"
  ) {
    return trimmed;
  }
  return undefined;
}

function normalizeOptions(
  options: OpenClawQuickstartOptions,
): NormalizedOpenClawQuickstartOptions {
  const rawLlmProvider = trimOrUndefined(options.llmProvider);
  const parsedLlmProvider = normalizeLlmProvider(rawLlmProvider);
  if (rawLlmProvider && !parsedLlmProvider) {
    throw new Error(
      "unsupported --llm-provider. Expected openai-compatible, anthropic, google, or ollama",
    );
  }
  const llmProvider = parsedLlmProvider ?? (options.providerId ? "openai-compatible" : undefined);
  if (!llmProvider) {
    throw new Error(
      "quickstart requires --llm-provider (openai-compatible, anthropic, google, or ollama)",
    );
  }
  const llmBaseUrl = trimOrUndefined(options.llmBaseUrl) ?? trimOrUndefined(options.baseUrl);
  if (!llmBaseUrl) {
    throw new Error("quickstart requires --llm-base-url");
  }
  const llmModel = trimOrUndefined(options.llmModel) ?? trimOrUndefined(options.memxModel);
  if (!llmModel) {
    throw new Error("quickstart requires --llm-model");
  }
  const llmApiKey = trimOrUndefined(options.llmApiKey) ?? trimOrUndefined(options.apiKey);
  const llmApiKeyEnv = trimOrUndefined(options.llmApiKeyEnv) ?? trimOrUndefined(options.apiKeyEnv);
  if (llmApiKey && llmApiKeyEnv) {
    throw new Error("use either --llm-api-key or --llm-api-key-env, not both");
  }
  if (!llmApiKey && !llmApiKeyEnv && llmProvider !== "ollama") {
    throw new Error("quickstart requires --llm-api-key or --llm-api-key-env");
  }
  const homeDir = options.homeDir ?? homedir();
  const embeddingProvider = normalizeEmbeddingProvider(options.embeddingProvider);
  return {
    ...options,
    llmProvider,
    llmBaseUrl,
    llmApiKey,
    llmApiKeyEnv,
    llmModel,
    embeddingProvider,
    embeddingModel: trimOrUndefined(options.embeddingModel) ?? DEFAULT_EMBEDDING_MODEL,
    embeddingPythonBin:
      trimOrUndefined(options.embeddingPythonBin) ??
      (embeddingProvider === "sentence-transformers-local" ? localVenvPython(homeDir) : ""),
    configPath: trimOrUndefined(options.configPath) ?? DEFAULT_CONFIG_PATH,
    homeDir,
    openclawBin: trimOrUndefined(options.openclawBin) ?? "openclaw",
    pythonBin: trimOrUndefined(options.pythonBin) ?? "python3",
    pluginInstallSource: trimOrUndefined(options.pluginInstallSource) ?? resolveCurrentPackageRoot(),
  };
}

function apiKeyValue(options: NormalizedOpenClawQuickstartOptions): string | SecretRef | undefined {
  const envName = trimOrUndefined(options.llmApiKeyEnv);
  if (envName) {
    return { source: "env", provider: "default", id: envName };
  }
  return trimOrUndefined(options.llmApiKey);
}

function memxEntry(
  currentEntry: Record<string, unknown> | undefined,
  options: NormalizedOpenClawQuickstartOptions,
): Record<string, unknown> {
  const base = structuredClone(DEFAULT_MEMORY_CONFIG);
  const existingConfig =
    currentEntry?.config && typeof currentEntry.config === "object"
      ? (currentEntry.config as Record<string, unknown>)
      : {};
  const existingEmbedding =
    existingConfig.embedding && typeof existingConfig.embedding === "object"
      ? (existingConfig.embedding as Record<string, unknown>)
      : {};
  const existingAdvanced =
    existingConfig.advanced && typeof existingConfig.advanced === "object"
      ? (existingConfig.advanced as Record<string, unknown>)
      : {};
  const embedding = {
    ...base.embedding,
    ...existingEmbedding,
    provider: options.embeddingProvider,
    model: options.embeddingModel,
    ...(options.embeddingProvider === "sentence-transformers-local"
      ? {
          localPythonBin: options.embeddingPythonBin,
          ...(options.embeddingCacheDir?.trim()
            ? { localCacheDir: options.embeddingCacheDir.trim() }
            : {}),
          localDevice: options.embeddingDevice ?? base.embedding.localDevice,
        }
      : {}),
  };
  return {
    ...(currentEntry ?? {}),
    enabled: true,
    hooks: {
      ...((currentEntry?.hooks as Record<string, unknown> | undefined) ?? {}),
      allowPromptInjection: true,
    },
    config: {
      ...base,
      ...existingConfig,
      embedding,
      advanced: {
        ...base.advanced,
        ...existingAdvanced,
        enableTurnScheduler: true,
        enableCompatibilityMemoryTools: false,
        llmClassifierEnabled: true,
        llmProvider: options.llmProvider,
        llmBaseURL: options.llmBaseUrl,
        llmApiKey: apiKeyValue(options),
        llmClassifierModel: options.llmModel,
      },
    },
  };
}

function currentMemxEntry(
  entries: Record<string, Record<string, unknown>> | undefined,
): Record<string, unknown> | undefined {
  return entries?.[PLUGIN_ID] ?? entries?.[LEGACY_MEMX_PLUGIN_ID];
}

export function applyOpenClawQuickstartConfig(
  input: unknown,
  rawOptions: OpenClawQuickstartOptions,
): OpenClawConfigLike {
  const options = normalizeOptions(rawOptions);
  const next = asConfig(input);
  const allow = new Set(withoutLegacyPluginIds(next.plugins?.allow));
  allow.add(PLUGIN_ID);
  const entries = { ...(next.plugins?.entries ?? {}) };
  const existingEntry = currentMemxEntry(entries);
  delete entries[LEGACY_MEMX_PLUGIN_ID];

  return {
    ...next,
    plugins: {
      ...(next.plugins ?? {}),
      allow: [...allow],
      slots: {
        ...(next.plugins?.slots ?? {}),
        memory: PLUGIN_ID,
      },
      entries: {
        ...entries,
        [PLUGIN_ID]: memxEntry(existingEntry, options),
      },
    },
  };
}

export function buildOpenClawQuickstartSteps(
  rawOptions: OpenClawQuickstartOptions,
): QuickstartCommandStep[] {
  const options = normalizeOptions(rawOptions);
  const steps: QuickstartCommandStep[] = [];
  if (options.embeddingProvider === "sentence-transformers-local" && !options.skipEmbeddingDeps) {
    steps.push({
      key: "embedding-venv",
      command: options.pythonBin,
      args: ["-m", "venv", localVenvDir(options.homeDir)],
    });
    steps.push({
      key: "embedding-deps",
      command: options.embeddingPythonBin,
      args: ["-m", "pip", "install", "-U", "pip", "sentence-transformers", "torch"],
    });
  }
  if (!options.skipPluginInstall) {
    steps.push({
      key: "plugin-install",
      command: options.openclawBin,
      args: ["plugins", "install", options.pluginInstallSource],
    });
  }
  if (!options.skipRestart) {
    steps.push({
      key: "gateway-restart",
      command: options.openclawBin,
      args: ["gateway", "restart"],
    });
  }
  if (!options.skipDoctor) {
    steps.push({
      key: "doctor",
      command: options.openclawBin,
      args: ["memx", "doctor", "--deep"],
    });
  }
  return steps;
}

async function readConfig(path: string): Promise<OpenClawConfigLike> {
  if (!existsSync(path)) {
    return {};
  }
  return asConfig(JSON.parse(await readFile(path, "utf8")));
}

async function writeAtomicJson(path: string, value: unknown): Promise<void> {
  await mkdir(dirname(path), { recursive: true });
  const tmp = `${path}.tmp-${process.pid}-${Date.now()}`;
  await writeFile(tmp, `${JSON.stringify(value, null, 2)}\n`, "utf8");
  await rename(tmp, path);
}

async function defaultRunCommand(
  command: string,
  args: string[],
): Promise<QuickstartCommandResult> {
  return new Promise((resolve, reject) => {
    const child = spawn(command, args, { shell: false, stdio: "inherit" });
    child.once("error", reject);
    child.once("close", (code) => resolve({ code: code ?? 1 }));
  });
}

function publicSummary(options: NormalizedOpenClawQuickstartOptions, steps: QuickstartCommandStep[]) {
  return {
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
    steps,
  };
}

function isPreConfigStep(step: QuickstartCommandStep): boolean {
  return step.key === "embedding-venv" || step.key === "embedding-deps" || step.key === "plugin-install";
}

async function runQuickstartStep(
  step: QuickstartCommandStep,
  runCommand: (command: string, args: string[]) => Promise<QuickstartCommandResult>,
): Promise<void> {
  const result = await runCommand(step.command, step.args);
  if (result.code !== 0) {
    throw new Error(
      `quickstart step failed: ${step.key} (${step.command} ${step.args.join(" ")}) exited ${result.code}`,
    );
  }
}

export async function runOpenClawQuickstart(
  rawOptions: OpenClawQuickstartOptions,
  deps: QuickstartDeps = {},
): Promise<Record<string, unknown>> {
  const options = normalizeOptions(rawOptions);
  const current = await readConfig(options.configPath);
  const next = applyOpenClawQuickstartConfig(current, options);
  let preparedInstallSource: string | undefined;
  const runtimeOptions =
    !options.dryRun && !options.skipPluginInstall && !rawOptions.pluginInstallSource
      ? {
          ...options,
          pluginInstallSource: (preparedInstallSource = await prepareOpenClawInstallPackage(
            options.pluginInstallSource,
          )),
        }
      : options;
  const steps = buildOpenClawQuickstartSteps(runtimeOptions);
  if (!options.dryRun) {
    try {
      const runCommand = deps.runCommand ?? defaultRunCommand;
      for (const step of steps.filter(isPreConfigStep)) {
        await runQuickstartStep(step, runCommand);
      }
      await writeAtomicJson(options.configPath, next);
      for (const step of steps.filter((step) => !isPreConfigStep(step))) {
        await runQuickstartStep(step, runCommand);
      }
    } finally {
      if (preparedInstallSource) {
        await rm(preparedInstallSource, { recursive: true, force: true });
      }
    }
  }
  return {
    ok: true,
    dryRun: Boolean(options.dryRun),
    configPath: options.configPath,
    ...publicSummary(options, steps),
    nextStep: options.dryRun
      ? "Dry run only; rerun without --dry-run to write config and execute the planned steps."
      : options.skipRestart
        ? "Restart OpenClaw so the updated memX config is applied."
        : "OpenClaw was restarted; run openclaw tui or your normal client.",
  };
}
