//#region src/host/standaloneConfig.ts
const STANDALONE_DEFAULT_SCOPE = "workspace:{workspace}";
const STANDALONE_ALLOWED_SCOPES = [
	"workspace:{workspace}",
	"agent:{agentId}",
	"session:{sessionKey}",
	"project:{project}"
];
const LEGACY_AGENT_WIDE_SCOPE = "agent:{agentId}";
const LEGACY_AGENT_WIDE_ALLOWED_SCOPES = [
	"global",
	"agent:{agentId}",
	"session:{sessionKey}",
	"project:{project}"
];
function sameScopeSet(left, right) {
	if (!left || left.length !== right.length) return false;
	const leftSet = new Set(left);
	return right.every((scope) => leftSet.has(scope));
}
function normalizeStandaloneScopeDefaults(config) {
	if (config.defaultScope === LEGACY_AGENT_WIDE_SCOPE && sameScopeSet(config.allowedScopes, LEGACY_AGENT_WIDE_ALLOWED_SCOPES)) return {
		...config,
		defaultScope: STANDALONE_DEFAULT_SCOPE,
		allowedScopes: [...STANDALONE_ALLOWED_SCOPES]
	};
	return config;
}
//#endregion
export { STANDALONE_ALLOWED_SCOPES, STANDALONE_DEFAULT_SCOPE, normalizeStandaloneScopeDefaults };
