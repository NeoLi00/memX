import { EvidenceBundle } from "./types.mjs";
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