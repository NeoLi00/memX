const MEMX_NATIVE_HOOK_TIMEOUT_MS = 8 * 1e3;
const MEMX_NATIVE_HOOK_QUERY_COMPILER_MAX_MS = 4e3;
const MEMX_TRANSCRIPT_CAPTURE_TIMEOUT_MS = 1200;
function finitePositive(value) {
	return Number.isFinite(value) && value > 0 ? value : MEMX_NATIVE_HOOK_TIMEOUT_MS;
}
function deriveNativeHookHttpTimeoutMs(hookTimeoutMs) {
	return Math.max(250, finitePositive(hookTimeoutMs) - 500);
}
function deriveNativeHookQueryCompilerTimeoutMs(httpTimeoutMs) {
	return Math.max(250, Math.min(MEMX_NATIVE_HOOK_QUERY_COMPILER_MAX_MS, finitePositive(httpTimeoutMs) - 250));
}
//#endregion
export { MEMX_NATIVE_HOOK_TIMEOUT_MS, MEMX_TRANSCRIPT_CAPTURE_TIMEOUT_MS, deriveNativeHookHttpTimeoutMs, deriveNativeHookQueryCompilerTimeoutMs };
