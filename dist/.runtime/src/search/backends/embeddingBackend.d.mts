import { EmbeddingConfig, MemxLogger, RetrievalBackend, RetrievalSearchParams, SearchHit, VectorDocRecord } from "../../types.mjs";
import { VectorRepo } from "../../db/repositories/vectorRepo.mjs";

//#region src/search/backends/embeddingBackend.d.ts
type EmbedMode = "query" | "passage";
type LocalEmbeddingWorkerCleanupStats = {
  checkedStateFiles: number;
  removedStateFiles: number;
  stoppedWorkers: number;
};
declare function localEmbeddingWorkerRegistryDir(): string;
declare function localEmbeddingWorkerStatePath(config: EmbeddingConfig, registryDir?: string, ownerPid?: number): string;
declare function cleanupStaleLocalEmbeddingWorkers(params?: {
  registryDir?: string;
  logger?: MemxLogger;
  currentPid?: number;
  legacyStateDirs?: string[];
  legacyStateMaxAgeMs?: number;
}): Promise<LocalEmbeddingWorkerCleanupStats>;
declare class OptionalEmbeddingBackend implements RetrievalBackend {
  private readonly repo;
  private readonly embedding;
  private readonly logger;
  private readonly lexical;
  private readonly localWorkerLease;
  private readonly localWorker;
  private readonly queryEmbeddingCache;
  private warnedUnavailable;
  private acceptingUpserts;
  private localUnavailableForProcess;
  private closed;
  private upsertQueue;
  private localPrewarm;
  constructor(repo: VectorRepo, embedding: EmbeddingConfig, logger: MemxLogger);
  upsertDocs(docs: VectorDocRecord[]): void;
  flushPendingUpserts(): Promise<void>;
  close(): Promise<void>;
  deleteDocs(docIds: string[]): void;
  keywordSearch(params: RetrievalSearchParams): SearchHit[];
  similaritySearch(params: RetrievalSearchParams): Promise<SearchHit[]>;
  hybridSearch(params: RetrievalSearchParams): Promise<SearchHit[]>;
  embedTextsBatch(texts: string[], mode?: EmbedMode): Promise<number[][]>;
  prewarmLocalEmbeddings(): Promise<void>;
  private isEmbeddingDisabledForProcess;
  private handleEmbeddingFailure;
  private getCachedQueryEmbedding;
  private warnOnce;
  private embedTexts;
}
//#endregion
export { LocalEmbeddingWorkerCleanupStats, OptionalEmbeddingBackend, cleanupStaleLocalEmbeddingWorkers, localEmbeddingWorkerRegistryDir, localEmbeddingWorkerStatePath };