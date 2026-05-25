import { GenericMcpConfig, McpCommandConfig, McpToolsProfile, applyClaudeJsonConnect, applyClaudeJsonDisconnect, applyCodexTomlConnect, applyCodexTomlDisconnect, buildGenericMcpConfig, connectClaudeCodeConfig, connectCodexConfig, hasCodexMemxBlock } from "./host/connect.mjs";
import { MemxHostId, MemxHostMessage, MemxTurnEnvelope, normalizeHookPayload, normalizeObservePayload } from "./host/hookPayload.mjs";
import { MEMX_MCP_TOOLS, MemxMcpHandlerDeps, MemxMcpProxy, MemxMcpToolsProfile, defaultMemxProxy, handleMcpRequest } from "./host/mcpProtocol.mjs";
import { EvidenceBundle } from "./types.mjs";
import { OpenClawQuickstartOptions, QuickstartCommandResult, QuickstartCommandStep, QuickstartDeps, applyOpenClawQuickstartConfig, buildOpenClawQuickstartSteps, resolveCurrentPackageRoot, runOpenClawQuickstart } from "./host/quickstart.mjs";
import { StandaloneMemxQuickstartOptions, StandaloneQuickstartCommandResult, StandaloneQuickstartCommandStep, StandaloneQuickstartDeps, StandaloneQuickstartTarget, applyStandaloneMemxQuickstartConfig, buildStandaloneMemxQuickstartSteps, runStandaloneMemxQuickstart } from "./host/standaloneQuickstart.mjs";
import { MemxAgentRequest, MemxHostService, MemxRecallRequest, MemxServiceOptions, assessNativeContextEligibility, createServiceConfigFromEnv, focusRecallBundleForQueryEntities, formatNativeRecallContext, stableHostTurnId } from "./host/service.mjs";
import { compileTurnSemantics } from "./pipeline/turnSemanticCompiler.mjs";
import { sanitizeChunkSummaryResult } from "./pipeline/reasoner.mjs";
import { OpenClawPluginDefinition } from "openclaw/plugin-sdk/core";

//#region src/index.d.ts
declare function evidencePlanRuleLines(bundle: EvidenceBundle): string[];
declare function extractPromptQuery(event: {
  prompt?: string;
  messages?: unknown[];
}): string;
declare function createMemoryMemxPlugin(): OpenClawPluginDefinition;
declare const memoryMemxPlugin: OpenClawPluginDefinition;
//#endregion
export { createMemoryMemxPlugin, evidencePlanRuleLines, extractPromptQuery, memoryMemxPlugin };