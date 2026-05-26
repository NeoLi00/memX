const MEMX_NATIVE_HOOK_TIMEOUT_MS = 8 * 1e3;
const MEMX_NATIVE_HOOK_COMPILER_RESERVE_MS = 1e3;
const MEMX_NATIVE_HOOK_QUERY_COMPILER_MAX_MS = 5e3;
const MEMX_TRANSCRIPT_CAPTURE_TIMEOUT_MS = 1200;
function finitePositive(value) {
	return Number.isFinite(value) && value > 0 ? value : MEMX_NATIVE_HOOK_TIMEOUT_MS;
}
function deriveNativeHookHttpTimeoutMs(hookTimeoutMs) {
	return Math.max(250, finitePositive(hookTimeoutMs) - 250);
}
function deriveNativeHookQueryCompilerTimeoutMs(httpTimeoutMs) {
	return Math.max(250, Math.min(MEMX_NATIVE_HOOK_QUERY_COMPILER_MAX_MS, finitePositive(httpTimeoutMs) - MEMX_NATIVE_HOOK_COMPILER_RESERVE_MS));
}
function deriveNativeHookBudget(hookTimeoutMs) {
	const hook = finitePositive(hookTimeoutMs);
	const contextTimeoutMs = deriveNativeHookHttpTimeoutMs(hook);
	return {
		hookTimeoutMs: hook,
		contextTimeoutMs,
		observeTimeoutMs: contextTimeoutMs,
		queryCompilerTimeoutMs: deriveNativeHookQueryCompilerTimeoutMs(contextTimeoutMs)
	};
}
//#endregion
export { MEMX_NATIVE_HOOK_TIMEOUT_MS, MEMX_TRANSCRIPT_CAPTURE_TIMEOUT_MS, deriveNativeHookBudget, deriveNativeHookQueryCompilerTimeoutMs };
