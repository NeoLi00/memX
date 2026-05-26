import type { MemoryPluginConfig, ScopeVars } from "../types.js";
import { resolveUserPath, stableHash } from "../support.js";

export function workspaceScopeValue(workspaceDir: string | undefined): string {
  const trimmed = workspaceDir?.trim();
  if (!trimmed) {
    return "default";
  }
  return stableHash([resolveUserPath(trimmed)]).slice(0, 16);
}

export function scopeVarsForContext(input: {
  agentId?: string;
  sessionKey?: string;
  project?: string;
  workspaceDir?: string;
}): ScopeVars {
  return {
    agentId: input.agentId,
    sessionKey: input.sessionKey,
    project: input.project,
    workspace: workspaceScopeValue(input.workspaceDir),
  };
}

export function renderTemplate(input: string, vars: ScopeVars): string {
  return input
    .replaceAll("{agentId}", vars.agentId ?? "")
    .replaceAll("{sessionKey}", vars.sessionKey ?? "")
    .replaceAll("{project}", vars.project ?? "")
    .replaceAll("{workspace}", vars.workspace ?? "");
}

export function resolveDefaultScope(config: MemoryPluginConfig, vars: ScopeVars): string {
  return renderTemplate(config.defaultScope, vars).trim();
}

export function resolveAllowedScopes(config: MemoryPluginConfig, vars: ScopeVars): string[] {
  const seen = new Set<string>();
  for (const entry of config.allowedScopes) {
    const resolved = renderTemplate(entry, vars).trim();
    if (!resolved) {
      continue;
    }
    seen.add(resolved);
  }
  const defaultScope = resolveDefaultScope(config, vars);
  if (defaultScope) {
    seen.add(defaultScope);
  }
  return [...seen];
}

export function isScopeAllowed(
  scope: string,
  config: MemoryPluginConfig,
  vars: ScopeVars,
): boolean {
  return resolveAllowedScopes(config, vars).includes(scope.trim());
}

export function defaultRetrievalScopes(config: MemoryPluginConfig, vars: ScopeVars): string[] {
  const allowed = resolveAllowedScopes(config, vars);
  const scopes = new Set<string>();
  const fallbackScope = resolveDefaultScope(config, vars);
  if (fallbackScope && allowed.includes(fallbackScope)) {
    scopes.add(fallbackScope);
  }
  const workspaceScope = vars.workspace ? `workspace:${vars.workspace}` : "";
  if (workspaceScope && allowed.includes(workspaceScope)) {
    scopes.add(workspaceScope);
  }
  const sessionScope = vars.sessionKey ? `session:${vars.sessionKey}` : "";
  if (sessionScope && allowed.includes(sessionScope)) {
    scopes.add(sessionScope);
  }
  const projectScope = vars.project ? `project:${vars.project}` : "";
  if (projectScope && allowed.includes(projectScope)) {
    scopes.add(projectScope);
  }
  if (fallbackScope === "global" && allowed.includes("global")) {
    scopes.add("global");
  }
  return [...scopes];
}
