import { resolveUserPath, stableHash } from "../support.mjs";
//#region src/security/scopes.ts
function workspaceScopeValue(workspaceDir) {
	const trimmed = workspaceDir?.trim();
	if (!trimmed) return "default";
	return stableHash([resolveUserPath(trimmed)]).slice(0, 16);
}
function scopeVarsForContext(input) {
	return {
		agentId: input.agentId,
		sessionKey: input.sessionKey,
		project: input.project,
		workspace: workspaceScopeValue(input.workspaceDir)
	};
}
function renderTemplate(input, vars) {
	return input.replaceAll("{agentId}", vars.agentId ?? "").replaceAll("{sessionKey}", vars.sessionKey ?? "").replaceAll("{project}", vars.project ?? "").replaceAll("{workspace}", vars.workspace ?? "");
}
function resolveDefaultScope(config, vars) {
	return renderTemplate(config.defaultScope, vars).trim();
}
function resolveAllowedScopes(config, vars) {
	const seen = /* @__PURE__ */ new Set();
	for (const entry of config.allowedScopes) {
		const resolved = renderTemplate(entry, vars).trim();
		if (!resolved) continue;
		seen.add(resolved);
	}
	const defaultScope = resolveDefaultScope(config, vars);
	if (defaultScope) seen.add(defaultScope);
	return [...seen];
}
function isScopeAllowed(scope, config, vars) {
	return resolveAllowedScopes(config, vars).includes(scope.trim());
}
function defaultRetrievalScopes(config, vars) {
	const allowed = resolveAllowedScopes(config, vars);
	const scopes = /* @__PURE__ */ new Set();
	const fallbackScope = resolveDefaultScope(config, vars);
	if (fallbackScope && allowed.includes(fallbackScope)) scopes.add(fallbackScope);
	const workspaceScope = vars.workspace ? `workspace:${vars.workspace}` : "";
	if (workspaceScope && allowed.includes(workspaceScope)) scopes.add(workspaceScope);
	const sessionScope = vars.sessionKey ? `session:${vars.sessionKey}` : "";
	if (sessionScope && allowed.includes(sessionScope)) scopes.add(sessionScope);
	const projectScope = vars.project ? `project:${vars.project}` : "";
	if (projectScope && allowed.includes(projectScope)) scopes.add(projectScope);
	if (fallbackScope === "global" && allowed.includes("global")) scopes.add("global");
	return [...scopes];
}
//#endregion
export { defaultRetrievalScopes, isScopeAllowed, renderTemplate, resolveDefaultScope, scopeVarsForContext };
