import { createServer, type IncomingMessage, type ServerResponse } from "node:http";
import { URL } from "node:url";
import { MEMX_NATIVE_HOOK_TIMEOUT_MS } from "../timeouts.js";
import { MemxHostService, type MemxAgentRequest, type MemxRecallRequest } from "./service.js";

const DEFAULT_PORT = 3878;
const DEFAULT_SHUTDOWN_TIMEOUT_MS = MEMX_NATIVE_HOOK_TIMEOUT_MS;

function json(res: ServerResponse, status: number, payload: unknown): void {
  res.writeHead(status, { "content-type": "application/json; charset=utf-8" });
  res.end(JSON.stringify(payload));
}

async function readBody(req: IncomingMessage): Promise<unknown> {
  const chunks: Buffer[] = [];
  for await (const chunk of req) {
    chunks.push(typeof chunk === "string" ? Buffer.from(chunk) : chunk);
  }
  const text = Buffer.concat(chunks).toString("utf8").trim();
  if (!text) {
    return {};
  }
  return JSON.parse(text);
}

function authorized(req: IncomingMessage): boolean {
  const secret = process.env["MEMX_SECRET"];
  if (!secret) {
    return true;
  }
  return req.headers.authorization === `Bearer ${secret}`;
}

function agentRequestFromSearch(url: URL): MemxAgentRequest {
  return {
    hostId: url.searchParams.get("hostId") ?? undefined,
    actorId: url.searchParams.get("actorId") ?? undefined,
    sessionId: url.searchParams.get("sessionId") ?? undefined,
    workspaceDir: url.searchParams.get("workspaceDir") ?? undefined,
    project: url.searchParams.get("project") ?? undefined,
  };
}

export async function startMemxHttpServer(options: { port?: number; host?: string } = {}): Promise<void> {
  const service = new MemxHostService();
  const port = options.port ?? Number(process.env["MEMX_PORT"] || DEFAULT_PORT);
  const host = options.host ?? process.env["MEMX_HOST"] ?? "127.0.0.1";
  const server = createServer(async (req, res) => {
    try {
      if (!authorized(req)) {
        json(res, 401, { ok: false, error: "unauthorized" });
        return;
      }
      const url = new URL(req.url ?? "/", `http://${req.headers.host ?? "localhost"}`);
      if (req.method === "GET" && (url.pathname === "/v1/health" || url.pathname === "/health")) {
        json(res, 200, { ok: true, service: "memx" });
        return;
      }
      if (req.method === "GET" && url.pathname === "/v1/stats") {
        json(res, 200, await service.stats(agentRequestFromSearch(url)));
        return;
      }
      if (req.method === "GET" && url.pathname === "/v1/audit") {
        json(
          res,
          200,
          await service.audit(Number(url.searchParams.get("limit") ?? 50), agentRequestFromSearch(url)),
        );
        return;
      }
      if (req.method === "POST" && url.pathname === "/v1/observe") {
        json(res, 200, await service.observe(await readBody(req)));
        return;
      }
      if (req.method === "POST" && url.pathname === "/v1/context") {
        json(res, 200, await service.context((await readBody(req)) as MemxRecallRequest));
        return;
      }
      if (req.method === "POST" && url.pathname === "/v1/recall") {
        json(res, 200, await service.recall((await readBody(req)) as MemxRecallRequest));
        return;
      }
      if (req.method === "POST" && url.pathname === "/v1/remember") {
        json(res, 200, await service.remember((await readBody(req)) as Record<string, unknown>));
        return;
      }
      if (req.method === "POST" && url.pathname === "/v1/forget") {
        json(res, 200, await service.forget((await readBody(req)) as Record<string, unknown>));
        return;
      }
      json(res, 404, { ok: false, error: "not found" });
    } catch (error) {
      json(res, 500, { ok: false, error: error instanceof Error ? error.message : String(error) });
    }
  });
  await new Promise<void>((resolve) => {
    server.listen(port, host, resolve);
  });
  console.error(`memx: server listening on http://${host}:${port}`);

  const close = async (signal: NodeJS.Signals) => {
    const configuredTimeoutMs = Number(
      process.env["MEMX_SHUTDOWN_TIMEOUT_MS"] || DEFAULT_SHUTDOWN_TIMEOUT_MS,
    );
    const timeoutMs = Number.isFinite(configuredTimeoutMs)
      ? Math.max(500, configuredTimeoutMs)
      : DEFAULT_SHUTDOWN_TIMEOUT_MS;
    server.close();
    let timeout: NodeJS.Timeout | undefined;
    await Promise.race([
      service.close(),
      new Promise<void>((resolve) => {
        timeout = setTimeout(() => {
          console.error(`memx: shutdown timed out after ${timeoutMs}ms (${signal})`);
          resolve();
        }, timeoutMs);
        timeout.unref();
      }),
    ]);
    if (timeout) {
      clearTimeout(timeout);
    }
  };
  let closing = false;
  const shutdown = (signal: NodeJS.Signals) => {
    if (closing) {
      process.exit(1);
    }
    closing = true;
    void close(signal).then(() => process.exit(0));
  };
  process.once("SIGINT", shutdown);
  process.once("SIGTERM", shutdown);
}
