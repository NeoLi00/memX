import * as _$node_sqlite0 from "node:sqlite";
import { DatabaseSync, StatementSync } from "node:sqlite";

//#region src/db/client.d.ts
declare function requireNodeSqlite(): typeof _$node_sqlite0;
declare class MemxDbClient {
  readonly dbPath: string;
  readonly db: DatabaseSync;
  constructor(dbPath: string);
  static open(dbPath: string): Promise<MemxDbClient>;
  private initialize;
  private migrate;
  prepare(sql: string): StatementSync;
  exec(sql: string): void;
  withTransaction<T>(run: () => T): T;
  currentMemoryEpoch(agentId: string): number;
  nextMemoryEpoch(agentId: string, updatedAt?: string): number;
  close(): void;
}
//#endregion
export { MemxDbClient };