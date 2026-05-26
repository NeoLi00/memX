import type { MemoryPluginConfig } from "../types.js";

export const STANDALONE_DEFAULT_SCOPE = "workspace:{workspace}";
export const STANDALONE_ALLOWED_SCOPES = [
  "workspace:{workspace}",
  "agent:{agentId}",
  "session:{sessionKey}",
  "project:{project}",
] as const;

const LEGACY_AGENT_WIDE_SCOPE = "agent:{agentId}";
const LEGACY_AGENT_WIDE_ALLOWED_SCOPES = [
  "global",
  "agent:{agentId}",
  "session:{sessionKey}",
  "project:{project}",
] as const;

function sameScopeSet(left: readonly string[] | undefined, right: readonly string[]): boolean {
  if (!left || left.length !== right.length) {
    return false;
  }
  const leftSet = new Set(left);
  return right.every((scope) => leftSet.has(scope));
}

export function normalizeStandaloneScopeDefaults(config: MemoryPluginConfig): MemoryPluginConfig {
  if (
    config.defaultScope === LEGACY_AGENT_WIDE_SCOPE &&
    sameScopeSet(config.allowedScopes, LEGACY_AGENT_WIDE_ALLOWED_SCOPES)
  ) {
    return {
      ...config,
      defaultScope: STANDALONE_DEFAULT_SCOPE,
      allowedScopes: [...STANDALONE_ALLOWED_SCOPES],
    };
  }
  return config;
}
