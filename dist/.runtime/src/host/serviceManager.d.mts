//#region src/host/serviceManager.d.ts
type MemxServiceStartOptions = {
  homeDir: string;
  runtimeDir: string;
  configPath: string;
  url?: string;
  secret?: string;
  nodeBin?: string;
  healthTimeoutMs?: number;
  startTimeoutMs?: number;
  stopTimeoutMs?: number;
};
type MemxServiceStatus = {
  ok: boolean;
  alreadyRunning: boolean;
  url: string;
  pid?: number;
  pidPath: string;
  logPath: string;
  error?: string;
};
declare function serviceRecordPath(homeDir: string): string;
declare function serviceLogPath(homeDir: string): string;
declare function ensureMemxService(options: MemxServiceStartOptions): Promise<MemxServiceStatus>;
declare function readMemxServiceStatus(options: Pick<MemxServiceStartOptions, "homeDir" | "url" | "secret" | "healthTimeoutMs">): Promise<MemxServiceStatus>;
declare function stopMemxService(options: Pick<MemxServiceStartOptions, "homeDir" | "url" | "secret" | "healthTimeoutMs" | "stopTimeoutMs">): Promise<MemxServiceStatus>;
//#endregion
export { MemxServiceStartOptions, MemxServiceStatus };