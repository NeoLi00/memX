import { MemoryEmbeddingProvider, MemoryLlmProvider, MemoryPluginConfig } from "../types.mjs";
import { McpToolsProfile } from "./connect.mjs";
import { MemxServiceStartOptions, MemxServiceStatus } from "./serviceManager.mjs";

//#region src/host/standaloneQuickstart.d.ts
type StandaloneQuickstartTarget = "codex" | "claude-code" | "mcp";
type StandaloneMemxQuickstartOptions = {
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
type StandaloneQuickstartCommandStep = {
  key: string;
  command: string;
  args: string[];
};
type StandaloneQuickstartCommandResult = {
  code: number;
  stdout?: string;
  stderr?: string;
};
type StandaloneQuickstartDeps = {
  runCommand?: (command: string, args: string[]) => Promise<StandaloneQuickstartCommandResult>;
  ensureService?: (options: MemxServiceStartOptions) => Promise<MemxServiceStatus>;
};
declare function applyStandaloneMemxQuickstartConfig(input: unknown, rawOptions: StandaloneMemxQuickstartOptions): MemoryPluginConfig;
declare function buildStandaloneMemxQuickstartSteps(rawOptions: StandaloneMemxQuickstartOptions): StandaloneQuickstartCommandStep[];
declare function runStandaloneMemxQuickstart(rawOptions: StandaloneMemxQuickstartOptions, deps?: StandaloneQuickstartDeps): Promise<Record<string, unknown>>;
//#endregion
export { StandaloneMemxQuickstartOptions, StandaloneQuickstartCommandResult, StandaloneQuickstartCommandStep, StandaloneQuickstartDeps, StandaloneQuickstartTarget, applyStandaloneMemxQuickstartConfig, buildStandaloneMemxQuickstartSteps, runStandaloneMemxQuickstart };