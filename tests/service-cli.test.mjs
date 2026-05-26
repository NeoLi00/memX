import assert from "node:assert/strict";
import { mkdirSync, mkdtempSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { spawn } from "node:child_process";
import { setTimeout as delay } from "node:timers/promises";
import test from "node:test";

function runMemxCli(args) {
  return new Promise((resolve) => {
    const child = spawn(process.execPath, ["dist/.runtime/src/bin/memx.mjs", ...args], {
      cwd: new URL("..", import.meta.url),
      stdio: ["ignore", "pipe", "pipe"],
    });
    let stdout = "";
    let stderr = "";
    child.stdout.on("data", (chunk) => {
      stdout += String(chunk);
    });
    child.stderr.on("data", (chunk) => {
      stderr += String(chunk);
    });
    child.once("close", (code) => resolve({ code, stdout, stderr }));
  });
}

test("memx service status reports the managed local service state as JSON", async () => {
  const dir = mkdtempSync(join(tmpdir(), "memx-service-status-"));

  const result = await runMemxCli([
    "service",
    "status",
    "--home",
    dir,
    "--memx-url",
    "http://127.0.0.1:9",
  ]);

  assert.equal(result.code, 0);
  const json = JSON.parse(result.stdout);
  assert.equal(json.ok, false);
  assert.equal(json.target, "service");
  assert.equal(json.action, "status");
  assert.equal(json.url, "http://127.0.0.1:9");
  assert.match(json.error, /not running|health check failed/i);
});

test("memx service commands default to the recorded managed service URL", async () => {
  const dir = mkdtempSync(join(tmpdir(), "memx-service-recorded-url-"));
  mkdirSync(join(dir, ".memx"), { recursive: true });
  writeFileSync(
    join(dir, ".memx", "service.json"),
    `${JSON.stringify({
      pid: 99999999,
      url: "http://127.0.0.1:9",
      configPath: join(dir, ".memx", "config.json"),
      startedAt: new Date().toISOString(),
      runtimeDir: join(dir, ".memx", "runtime"),
    })}\n`,
    "utf8",
  );

  const status = await runMemxCli(["service", "status", "--home", dir]);
  assert.equal(status.code, 0);
  assert.equal(JSON.parse(status.stdout).url, "http://127.0.0.1:9");

  writeFileSync(
    join(dir, ".memx", "service.json"),
    `${JSON.stringify({
      pid: 99999999,
      url: "http://127.0.0.1:9",
      configPath: join(dir, ".memx", "config.json"),
      startedAt: new Date().toISOString(),
      runtimeDir: join(dir, ".memx", "runtime"),
    })}\n`,
    "utf8",
  );
  const stop = await runMemxCli(["service", "stop", "--home", dir]);
  assert.equal(stop.code, 0);
  assert.equal(JSON.parse(stop.stdout).url, "http://127.0.0.1:9");
});

test("memx service stop is idempotent when no managed service is running", async () => {
  const dir = mkdtempSync(join(tmpdir(), "memx-service-stop-"));

  const result = await runMemxCli([
    "service",
    "stop",
    "--home",
    dir,
    "--memx-url",
    "http://127.0.0.1:9",
  ]);

  assert.equal(result.code, 0);
  const json = JSON.parse(result.stdout);
  assert.equal(json.ok, true);
  assert.equal(json.target, "service");
  assert.equal(json.action, "stop");
  assert.equal(json.url, "http://127.0.0.1:9");
});

test("memx service stop terminates a managed service that ignores SIGTERM", async () => {
  const dir = mkdtempSync(join(tmpdir(), "memx-service-stop-stubborn-"));
  const child = spawn(process.execPath, [
    "-e",
    "process.on('SIGTERM', () => {}); setInterval(() => {}, 1000);",
  ], {
    stdio: "ignore",
  });
  await delay(100);
  mkdirSync(join(dir, ".memx"), { recursive: true });
  writeFileSync(
    join(dir, ".memx", "service.json"),
    `${JSON.stringify({
      pid: child.pid,
      url: "http://127.0.0.1:9",
      configPath: join(dir, ".memx", "config.json"),
      startedAt: new Date().toISOString(),
      runtimeDir: join(dir, ".memx", "runtime"),
    })}\n`,
    "utf8",
  );

  try {
    const result = await runMemxCli([
      "service",
      "stop",
      "--home",
      dir,
      "--memx-url",
      "http://127.0.0.1:9",
    ]);

    assert.equal(result.code, 0);
    const json = JSON.parse(result.stdout);
    assert.equal(json.ok, true);
    await delay(100);
    assert.throws(() => process.kill(child.pid, 0), /ESRCH/);
  } finally {
    if (child.pid) {
      try {
        process.kill(child.pid, "SIGKILL");
      } catch {}
    }
  }
});
