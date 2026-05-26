import { SearchHit, VectorDocRecord } from "../../types.mjs";
import { MemxDbClient } from "../client.mjs";

//#region src/db/repositories/vectorRepo.d.ts
declare class VectorRepo {
  private readonly db;
  constructor(db: MemxDbClient);
  private appendDocFilters;
  upsertDocs(docs: VectorDocRecord[]): void;
  deleteDocs(docIds: string[]): void;
  keywordSearch(params: {
    agentId: string;
    scopes: string[];
    query: string;
    limit: number;
    readEpoch?: number;
    docKinds?: VectorDocRecord["docKind"][];
    docTypes?: string[];
  }): SearchHit[];
  listDocs(params: {
    agentId: string;
    scopes: string[];
    limit?: number;
    readEpoch?: number;
    docKinds?: VectorDocRecord["docKind"][];
    docTypes?: string[];
  }): VectorDocRecord[];
  upsertEmbedding(params: {
    docId: string;
    agentId: string;
    scope: string;
    embedding: number[];
    updatedAt: string;
  }): void;
  listEmbeddings(params: {
    agentId: string;
    scopes: string[];
    limit: number;
    readEpoch?: number;
    docKinds?: VectorDocRecord["docKind"][];
    docTypes?: string[];
  }): Array<{
    docId: string;
    scope: string;
    embedding: number[];
    updatedAt: string;
  }>;
  getDoc(docId: string, readEpoch?: number): VectorDocRecord | null;
}
//#endregion
export { VectorRepo };