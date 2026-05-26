import assert from "node:assert/strict";
import { existsSync, mkdirSync, mkdtempSync, readFileSync, utimesSync, writeFileSync } from "node:fs";
import { rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";

function logger() {
  return {
    debug() {},
    info() {},
    warn() {},
    error() {},
  };
}

test("local embedding worker cleanup removes stale registry entries for dead owners", async () => {
  const {
    cleanupStaleLocalEmbeddingWorkers,
    localEmbeddingWorkerStatePath,
  } = await import("../dist/.runtime/src/search/backends/embeddingBackend.mjs");
  const registryDir = mkdtempSync(join(tmpdir(), "memx-embedder-registry-test-"));
  const statePath = localEmbeddingWorkerStatePath(
    {
      provider: "sentence-transformers-local",
      model: "intfloat/multilingual-e5-small",
      localPythonBin: "python3",
      localDevice: "cpu",
    },
    registryDir,
  );
  writeFileSync(
    statePath,
    `${JSON.stringify({
      url: "http://127.0.0.1:9",
      token: "stale-token",
      pid: 999999991,
      ownerPid: 999999992,
      workerKey: "stale-worker-key",
    })}\n`,
    "utf8",
  );

  const result = await cleanupStaleLocalEmbeddingWorkers({
    registryDir,
    legacyStateDirs: [],
    logger: logger(),
  });

  assert.equal(result.removedStateFiles, 1);
  assert.equal(result.stoppedWorkers, 0);
  assert.equal(existsSync(statePath), false);
  await rm(registryDir, { recursive: true, force: true });
});

test("local embedding worker state paths are stable per model and isolated per configuration", async () => {
  const { localEmbeddingWorkerStatePath } = await import(
    "../dist/.runtime/src/search/backends/embeddingBackend.mjs"
  );
  const registryDir = mkdtempSync(join(tmpdir(), "memx-embedder-path-test-"));
  const base = {
    provider: "sentence-transformers-local",
    model: "intfloat/multilingual-e5-small",
    localPythonBin: "python3",
    localDevice: "cpu",
  };
  const first = localEmbeddingWorkerStatePath(base, registryDir);
  const second = localEmbeddingWorkerStatePath(base, registryDir);
  const differentDevice = localEmbeddingWorkerStatePath(
    {
      ...base,
      localDevice: "mps",
    },
    registryDir,
  );

  assert.equal(first, second);
  assert.notEqual(first, differentDevice);
  assert.match(readFileSync(new URL("../package.json", import.meta.url), "utf8"), /"memx"/i);
  await rm(registryDir, { recursive: true, force: true });
});

test("local embedding cleanup removes stale legacy tmp state files from older releases", async () => {
  const { cleanupStaleLocalEmbeddingWorkers } = await import(
    "../dist/.runtime/src/search/backends/embeddingBackend.mjs"
  );
  const registryDir = mkdtempSync(join(tmpdir(), "memx-embedder-registry-test-"));
  const legacyDir = mkdtempSync(join(tmpdir(), "memx-embedder-legacy-test-"));
  mkdirSync(registryDir, { recursive: true });
  const legacyStatePath = join(legacyDir, "memx-embedder-legacy-token.json");
  writeFileSync(
    legacyStatePath,
    `${JSON.stringify({
      url: "http://127.0.0.1:9",
      token: "legacy-token",
    })}\n`,
    "utf8",
  );
  const stale = new Date(Date.now() - 60 * 60 * 1000);
  utimesSync(legacyStatePath, stale, stale);

  const result = await cleanupStaleLocalEmbeddingWorkers({
    registryDir,
    legacyStateDirs: [legacyDir],
    legacyStateMaxAgeMs: 1000,
    logger: logger(),
  });

  assert.equal(result.checkedStateFiles, 1);
  assert.equal(result.removedStateFiles, 1);
  assert.equal(existsSync(legacyStatePath), false);
  await rm(registryDir, { recursive: true, force: true });
  await rm(legacyDir, { recursive: true, force: true });
});

test("local embedding cleanup keeps fresh legacy tmp state files without process metadata", async () => {
  const { cleanupStaleLocalEmbeddingWorkers } = await import(
    "../dist/.runtime/src/search/backends/embeddingBackend.mjs"
  );
  const registryDir = mkdtempSync(join(tmpdir(), "memx-embedder-registry-test-"));
  const legacyDir = mkdtempSync(join(tmpdir(), "memx-embedder-legacy-test-"));
  const legacyStatePath = join(legacyDir, "memx-embedder-fresh-token.json");
  writeFileSync(
    legacyStatePath,
    `${JSON.stringify({
      url: "http://127.0.0.1:9",
      token: "legacy-token",
    })}\n`,
    "utf8",
  );

  const result = await cleanupStaleLocalEmbeddingWorkers({
    registryDir,
    legacyStateDirs: [legacyDir],
    legacyStateMaxAgeMs: 60 * 60 * 1000,
    logger: logger(),
  });

  assert.equal(result.checkedStateFiles, 1);
  assert.equal(result.removedStateFiles, 0);
  assert.equal(existsSync(legacyStatePath), true);
  await rm(registryDir, { recursive: true, force: true });
  await rm(legacyDir, { recursive: true, force: true });
});
