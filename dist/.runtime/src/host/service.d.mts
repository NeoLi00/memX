import { MemxTurnEnvelope } from "./hookPayload.mjs";
import { EvidenceBundle, MemoryPluginConfig, MemxLogger, QueryCompileResult } from "../types.mjs";

//#region src/host/service.d.ts
type MemxServiceOptions = {
  config?: MemoryPluginConfig;
  logger?: MemxLogger;
};
type MemxRecallRequest = {
  query: string;
  limit?: number;
  hostId?: string;
  actorId?: string;
  sessionId?: string;
  workspaceDir?: string;
  project?: string;
  hotPathTimeoutMs?: number;
};
type MemxAgentRequest = {
  hostId?: string;
  actorId?: string;
  sessionId?: string;
  workspaceDir?: string;
  project?: string;
};
declare function createServiceConfigFromEnv(env?: NodeJS.ProcessEnv): MemoryPluginConfig;
declare function formatNativeRecallContext(bundle: EvidenceBundle, maxChars: number): string;
type NativeContextEligibility = {
  eligible: boolean;
  reason: string;
  bestScore: number;
};
declare function focusRecallBundleForQueryEntities(queryAnalysis: Pick<QueryCompileResult, "queryEntities">, bundle: EvidenceBundle): EvidenceBundle;
declare function assessNativeContextEligibility(_query: string, queryAnalysis: QueryCompileResult, bundle: EvidenceBundle): NativeContextEligibility;
declare class MemxHostService {
  private readonly config;
  private readonly logger;
  private readonly manager;
  private readonly pendingWrites;
  constructor(options?: MemxServiceOptions);
  close(): Promise<void>;
  private pendingWriteKey;
  private hasPendingWrite;
  private enqueuePendingWrite;
  private waitForPendingWrites;
  observe(input: unknown): Promise<Record<string, unknown>>;
  recall(request: MemxRecallRequest): Promise<Record<string, unknown>>;
  remember(request: Record<string, unknown>): Promise<Record<string, unknown>>;
  forget(request: Record<string, unknown>): Promise<Record<string, unknown>>;
  stats(request?: MemxAgentRequest): Promise<Record<string, unknown>>;
  audit(limit?: number, request?: MemxAgentRequest): Promise<Record<string, unknown>>;
  context(request: MemxRecallRequest): Promise<Record<string, unknown>>;
}
declare function stableHostTurnId(envelope: MemxTurnEnvelope): string;
//#endregion
export { MemxAgentRequest, MemxHostService, MemxRecallRequest, MemxServiceOptions, assessNativeContextEligibility, createServiceConfigFromEnv, focusRecallBundleForQueryEntities, formatNativeRecallContext, stableHostTurnId };