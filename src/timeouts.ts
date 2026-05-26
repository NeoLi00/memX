export const MEMX_NATIVE_HOOK_TIMEOUT_SECONDS = 8;
export const MEMX_NATIVE_HOOK_TIMEOUT_MS = MEMX_NATIVE_HOOK_TIMEOUT_SECONDS * 1000;
export const MEMX_NATIVE_HOOK_HTTP_RESERVE_MS = 250;
export const MEMX_NATIVE_HOOK_COMPILER_RESERVE_MS = 1000;
export const MEMX_NATIVE_HOOK_QUERY_COMPILER_MAX_MS = 5000;
export const MEMX_TRANSCRIPT_CAPTURE_TIMEOUT_MS = 1200;
export const MEMX_TRANSCRIPT_CAPTURE_INTERVAL_MS = 50;

export type NativeHookBudget = {
  hookTimeoutMs: number;
  contextTimeoutMs: number;
  observeTimeoutMs: number;
  queryCompilerTimeoutMs: number;
};

function finitePositive(value: number): number {
  return Number.isFinite(value) && value > 0 ? value : MEMX_NATIVE_HOOK_TIMEOUT_MS;
}

export function deriveNativeHookHttpTimeoutMs(hookTimeoutMs: number): number {
  return Math.max(250, finitePositive(hookTimeoutMs) - MEMX_NATIVE_HOOK_HTTP_RESERVE_MS);
}

export function deriveNativeHookQueryCompilerTimeoutMs(httpTimeoutMs: number): number {
  return Math.max(
    250,
    Math.min(
      MEMX_NATIVE_HOOK_QUERY_COMPILER_MAX_MS,
      finitePositive(httpTimeoutMs) - MEMX_NATIVE_HOOK_COMPILER_RESERVE_MS,
    ),
  );
}

export function deriveNativeHookBudget(hookTimeoutMs: number): NativeHookBudget {
  const hook = finitePositive(hookTimeoutMs);
  const contextTimeoutMs = deriveNativeHookHttpTimeoutMs(hook);
  return {
    hookTimeoutMs: hook,
    contextTimeoutMs,
    observeTimeoutMs: contextTimeoutMs,
    queryCompilerTimeoutMs: deriveNativeHookQueryCompilerTimeoutMs(contextTimeoutMs),
  };
}
