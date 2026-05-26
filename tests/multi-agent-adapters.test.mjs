import assert from "node:assert/strict";
import { spawn } from "node:child_process";
import { appendFileSync, existsSync, mkdtempSync, readdirSync, readFileSync, writeFileSync } from "node:fs";
import { createServer } from "node:http";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";

const root = new URL("..", import.meta.url);
const rootPath = root.pathname;

function readJson(relativePath) {
  return JSON.parse(readFileSync(join(rootPath, relativePath), "utf8"));
}

function collectCommandHooks(hooksConfig) {
  const commandHooks = [];
  for (const entries of Object.values(hooksConfig.hooks)) {
    for (const entry of entries) {
      for (const hook of entry.hooks ?? []) {
        if (hook.type === "command") {
          commandHooks.push(hook);
        }
      }
    }
  }
  return commandHooks;
}

function encodeMcpFrame(payload) {
  const json = JSON.stringify(payload);
  return `Content-Length: ${Buffer.byteLength(json, "utf8")}\r\n\r\n${json}`;
}

function decodeMcpFrame(output) {
  const marker = "\r\n\r\n";
  const headerEnd = output.indexOf(marker);
  assert.notEqual(headerEnd, -1);
  const header = output.slice(0, headerEnd);
  const length = Number(header.match(/content-length:\s*(\d+)/i)?.[1]);
  assert.ok(Number.isInteger(length));
  const body = output.slice(headerEnd + marker.length, headerEnd + marker.length + length);
  return JSON.parse(body);
}

test("package ships standalone bins and native plugin assets", () => {
  const pkg = readJson("package.json");

  assert.equal(pkg.name, "@neoli00/memx");
  assert.equal(pkg.openclaw.install.npmSpec, "@neoli00/memx");
  assert.deepEqual(Object.keys(pkg.bin).sort(), [
    "memx",
    "memx-hook",
    "memx-mcp",
    "memx-server",
  ]);
  assert.equal(pkg.bin.memx, "dist/.runtime/src/bin/memx.mjs");
  assert.equal(pkg.bin["memx-mcp"], "dist/.runtime/src/bin/memx-mcp.mjs");

  for (const path of [".codex-plugin", ".claude-plugin", ".mcp.json", "hooks.json", "hooks", "skills"]) {
    assert.ok(pkg.files.includes(path), `package files must include ${path}`);
    assert.ok(existsSync(join(rootPath, path)), `${path} should exist`);
  }
});

test("standalone server bundle does not require OpenClaw at runtime", () => {
  const embeddingBackend = readFileSync(
    join(rootPath, "dist/.runtime/src/search/backends/embeddingBackend.mjs"),
    "utf8",
  );

  assert.doesNotMatch(embeddingBackend, /openclaw\/plugin-sdk/);
});

test("Codex native plugin manifest is hook-only by default", () => {
  const manifest = readJson(".codex-plugin/plugin.json");
  assert.equal(manifest.name, "memx");
  assert.equal(manifest.homepage, "https://github.com/NeoLi00/memX");
  assert.equal(manifest.repository, "https://github.com/NeoLi00/memX");
  assert.equal(manifest.hooks, "./hooks.json");
  assert.equal(manifest.interface.displayName, "memX");
  assert.equal(manifest.interface.shortDescription, "Local semantic memory for coding agents.");
  assert.ok(Array.isArray(manifest.interface.capabilities));
  assert.equal("mcpServers" in manifest, false);
  assert.equal("skills" in manifest, false);

  assert.deepEqual(readJson("hooks.json"), readJson("hooks/hooks.codex.json"));
  const hooks = readJson("hooks.json").hooks;
  const supported = new Set([
    "SessionStart",
    "UserPromptSubmit",
    "PreToolUse",
    "PostToolUse",
    "PermissionRequest",
    "PreCompact",
    "PostCompact",
    "Stop",
  ]);
  assert.deepEqual(Object.keys(hooks).sort(), [
    "PostToolUse",
    "PreCompact",
    "PreToolUse",
    "SessionStart",
    "Stop",
    "UserPromptSubmit",
  ]);
  for (const event of Object.keys(hooks)) {
    assert.ok(supported.has(event), `unsupported Codex hook event: ${event}`);
  }
  for (const hook of collectCommandHooks({ hooks })) {
    assert.match(hook.command, /^node "\$\{CLAUDE_PLUGIN_ROOT\}\/dist\/\.runtime\/src\/bin\/memx-hook\.mjs" codex /);
    assert.equal("args" in hook, false, "Codex hooks should use Codex-compatible command strings");
    assert.equal(hook.timeout, 8);
  }
});

test("Claude Code native plugin manifest is hook-only by default", () => {
  const manifest = readJson(".claude-plugin/plugin.json");
  assert.equal(manifest.name, "memx");
  assert.equal(manifest.homepage, "https://github.com/NeoLi00/memX");
  assert.equal(manifest.repository, "https://github.com/NeoLi00/memX");
  assert.equal("mcpServers" in manifest, false);
  assert.equal("skills" in manifest, false);
  assert.equal(
    "hooks" in manifest,
    false,
    "Claude Code auto-loads hooks/hooks.json; declaring it in the manifest duplicates the hook file",
  );

  const hooks = readJson("hooks/hooks.json").hooks;
  for (const event of [
    "SessionStart",
    "UserPromptSubmit",
    "PreToolUse",
    "PostToolUse",
    "PostToolUseFailure",
    "PreCompact",
    "SubagentStart",
    "SubagentStop",
    "Notification",
    "TaskCompleted",
    "Stop",
    "SessionEnd",
  ]) {
    assert.ok(event in hooks, `missing Claude Code hook: ${event}`);
  }
  for (const hook of collectCommandHooks({ hooks })) {
    assert.match(hook.command, /^node "\$\{CLAUDE_PLUGIN_ROOT\}\/dist\/\.runtime\/src\/bin\/memx-hook\.mjs" claude-code /);
    assert.equal("args" in hook, false, "Claude hooks should use command strings");
    assert.equal(hook.timeout, 8);
  }
});

test("native MCP config runs the local memX MCP binary from plugin root", () => {
  const mcp = readJson(".mcp.json");
  const entry = mcp.mcpServers.memx;
  assert.equal(entry.command, "node");
  assert.deepEqual(entry.args, ["${CLAUDE_PLUGIN_ROOT}/dist/.runtime/src/bin/memx-mcp.mjs"]);
  assert.equal(entry.env.MEMX_URL, "${MEMX_URL}");
  assert.equal(entry.env.MEMX_SECRET, "${MEMX_SECRET}");
  assert.equal(entry.env.MEMX_MCP_TOOLS, "none");
});

test("host protocol normalizes user turns and ignores tool hooks by default", async () => {
  const { normalizeHookPayload } = await import("../dist/.runtime/src/host/hookPayload.mjs");

  const codex = normalizeHookPayload("codex", "UserPromptSubmit", {
    session_id: "codex-session",
    cwd: "/tmp/project",
    prompt: "请记住 Notebook validator 要支持 pytest",
  });
  assert.equal(codex.hostId, "codex");
  assert.equal(codex.sessionId, "codex-session");
  assert.equal(codex.workspaceDir, "/tmp/project");
  assert.equal(codex.messages[0].role, "user");

  const claude = normalizeHookPayload("claude-code", "PostToolUse", {
    session_id: "claude-session",
    cwd: "/tmp/project",
    tool_name: "Write",
    tool_input: { file_path: "validator.py" },
    tool_response: "ok",
  });
  assert.equal(claude.hostId, "claude-code");
  assert.deepEqual(claude.messages, []);
});

test("Codex UserPromptSubmit hook injects recalled context and defers write until Stop", async () => {
  const calls = [];
  const server = createServer(async (req, res) => {
    const chunks = [];
    for await (const chunk of req) {
      chunks.push(chunk);
    }
    const bodyText = Buffer.concat(chunks).toString("utf8");
    calls.push({
      path: req.url,
      body: bodyText ? JSON.parse(bodyText) : {},
    });
    res.writeHead(200, { "content-type": "application/json" });
    if (req.url === "/v1/context") {
      res.end(
        JSON.stringify({
          ok: true,
          prependContext: "## memX Memory\n- notebook validator prefers pytest fixtures",
        }),
      );
      return;
    }
    res.end(JSON.stringify({ ok: true }));
  });
  await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
  const address = server.address();
  assert.ok(address && typeof address === "object");

  const child = spawn(process.execPath, [join(rootPath, "dist/.runtime/src/bin/memx-hook.mjs"), "codex", "UserPromptSubmit"], {
    env: {
      ...process.env,
      MEMX_URL: `http://127.0.0.1:${address.port}`,
      MEMX_HOOK_TIMEOUT_MS: "2000",
    },
    stdio: ["pipe", "pipe", "pipe"],
  });
  let stdout = "";
  let stderr = "";
  child.stdout.setEncoding("utf8");
  child.stderr.setEncoding("utf8");
  child.stdout.on("data", (chunk) => {
    stdout += chunk;
  });
  child.stderr.on("data", (chunk) => {
    stderr += chunk;
  });
  child.stdin.end(
    JSON.stringify({
      session_id: "codex-session",
      cwd: "/tmp/project",
      prompt: "继续 notebook validator 任务",
    }),
  );

  const exitCode = await new Promise((resolve, reject) => {
    const timeout = setTimeout(() => {
      child.kill("SIGKILL");
      reject(new Error("memx-hook UserPromptSubmit test timed out"));
    }, 5000);
    child.on("error", (error) => {
      clearTimeout(timeout);
      reject(error);
    });
    child.on("exit", (code) => {
      clearTimeout(timeout);
      resolve(code);
    });
  });
  await new Promise((resolve) => server.close(resolve));

  assert.equal(exitCode, 0, stderr);
  assert.deepEqual(calls.map((call) => call.path), ["/v1/context"]);
  assert.equal(calls.find((call) => call.path === "/v1/context").body.query, "继续 notebook validator 任务");
  const output = JSON.parse(stdout);
  assert.equal(output.hookSpecificOutput.hookEventName, "UserPromptSubmit");
  assert.equal(
    output.hookSpecificOutput.additionalContext,
    "## memX Memory\n- notebook validator prefers pytest fixtures",
  );
});

test("native hook reads memX URL from quickstart hook runtime config", async () => {
  const calls = [];
  const server = createServer(async (req, res) => {
    const chunks = [];
    for await (const chunk of req) {
      chunks.push(chunk);
    }
    const bodyText = Buffer.concat(chunks).toString("utf8");
    calls.push({
      path: req.url,
      body: bodyText ? JSON.parse(bodyText) : {},
    });
    res.writeHead(200, { "content-type": "application/json" });
    res.end(JSON.stringify({ ok: true, prependContext: "## memX Memory\n- hook config works" }));
  });
  await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
  const address = server.address();
  assert.ok(address && typeof address === "object");
  const dir = mkdtempSync(join(tmpdir(), "memx-hook-config-"));
  const hookConfigPath = join(dir, "hook-runtime.json");
  writeFileSync(
    hookConfigPath,
    JSON.stringify({
      memxUrl: `http://127.0.0.1:${address.port}`,
      hookTimeoutMs: 2000,
    }),
    "utf8",
  );

  const child = spawn(
    process.execPath,
    [
      join(rootPath, "dist/.runtime/src/bin/memx-hook.mjs"),
      "codex",
      "UserPromptSubmit",
      "--hook-config",
      hookConfigPath,
    ],
    {
      env: {
        ...process.env,
        MEMX_URL: "",
        MEMX_HOOK_TIMEOUT_MS: "",
      },
      stdio: ["pipe", "pipe", "pipe"],
    },
  );
  let stdout = "";
  let stderr = "";
  child.stdout.setEncoding("utf8");
  child.stderr.setEncoding("utf8");
  child.stdout.on("data", (chunk) => {
    stdout += chunk;
  });
  child.stderr.on("data", (chunk) => {
    stderr += chunk;
  });
  child.stdin.end(
    JSON.stringify({
      session_id: "codex-hook-config-session",
      cwd: "/tmp/project",
      prompt: "继续 hook config 验证",
    }),
  );

  const exitCode = await new Promise((resolve, reject) => {
    const timeout = setTimeout(() => {
      child.kill("SIGKILL");
      reject(new Error("memx-hook config test timed out"));
    }, 5000);
    child.on("error", (error) => {
      clearTimeout(timeout);
      reject(error);
    });
    child.on("exit", (code) => {
      clearTimeout(timeout);
      resolve(code);
    });
  });
  await new Promise((resolve) => server.close(resolve));

  assert.equal(exitCode, 0, stderr);
  assert.deepEqual(calls.map((call) => call.path), ["/v1/context"]);
  const output = JSON.parse(stdout);
  assert.match(output.hookSpecificOutput.additionalContext, /hook config works/);
});

test("UserPromptSubmit persists the pending user turn before slow recall can consume the hook budget", async () => {
  const calls = [];
  const server = createServer(async (req, res) => {
    const chunks = [];
    for await (const chunk of req) {
      chunks.push(chunk);
    }
    const bodyText = Buffer.concat(chunks).toString("utf8");
    calls.push({
      path: req.url,
      body: bodyText ? JSON.parse(bodyText) : {},
    });
    res.writeHead(200, { "content-type": "application/json" });
    setTimeout(() => {
      res.end(JSON.stringify({ ok: true, prependContext: "" }));
    }, 1000);
  });
  await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
  const address = server.address();
  assert.ok(address && typeof address === "object");
  const pendingDir = mkdtempSync(join(tmpdir(), "memx-pending-before-recall-"));

  const child = spawn(process.execPath, [join(rootPath, "dist/.runtime/src/bin/memx-hook.mjs"), "codex", "UserPromptSubmit"], {
    env: {
      ...process.env,
      MEMX_URL: `http://127.0.0.1:${address.port}`,
      MEMX_HOOK_TIMEOUT_MS: "5000",
      MEMX_PENDING_DIR: pendingDir,
    },
    stdio: ["pipe", "pipe", "pipe"],
  });
  child.stdin.end(
    JSON.stringify({
      session_id: "codex-session",
      cwd: "/tmp/project",
      prompt: "记住 PendingFirstProbe 的默认格式是 avro",
    }),
  );

  await new Promise((resolve) => setTimeout(resolve, 150));
  assert.equal(calls[0]?.path, "/v1/context");
  const pendingFiles = readdirSync(pendingDir).filter((name) => name.endsWith(".json"));
  assert.equal(pendingFiles.length, 1);
  const pending = JSON.parse(readFileSync(join(pendingDir, pendingFiles[0]), "utf8"));
  assert.match(pending.messages[0].content, /PendingFirstProbe/);

  child.kill("SIGKILL");
  await new Promise((resolve) => child.on("exit", resolve));
  await new Promise((resolve) => server.close(resolve));
});

test("Codex Stop hook writes a complete pending user and assistant turn", async () => {
  const calls = [];
  const server = createServer(async (req, res) => {
    const chunks = [];
    for await (const chunk of req) {
      chunks.push(chunk);
    }
    const bodyText = Buffer.concat(chunks).toString("utf8");
    calls.push({
      path: req.url,
      body: bodyText ? JSON.parse(bodyText) : {},
    });
    res.writeHead(200, { "content-type": "application/json" });
    if (req.url === "/v1/context") {
      res.end(
        JSON.stringify({
          ok: true,
          prependContext: "## memX Memory\n- notebook validator prefers pytest fixtures",
        }),
      );
      return;
    }
    res.end(JSON.stringify({ ok: true }));
  });
  await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
  const address = server.address();
  assert.ok(address && typeof address === "object");
  const pendingDir = mkdtempSync(join(tmpdir(), "memx-pending-turn-"));
  const env = {
    ...process.env,
    MEMX_URL: `http://127.0.0.1:${address.port}`,
    MEMX_HOOK_TIMEOUT_MS: "2000",
    MEMX_PENDING_DIR: pendingDir,
  };

  async function runHook(eventName, payload) {
    const child = spawn(process.execPath, [join(rootPath, "dist/.runtime/src/bin/memx-hook.mjs"), "codex", eventName], {
      env,
      stdio: ["pipe", "pipe", "pipe"],
    });
    let stderr = "";
    child.stderr.setEncoding("utf8");
    child.stderr.on("data", (chunk) => {
      stderr += chunk;
    });
    child.stdin.end(JSON.stringify(payload));
    const exitCode = await new Promise((resolve, reject) => {
      const timeout = setTimeout(() => {
        child.kill("SIGKILL");
        reject(new Error(`memx-hook ${eventName} test timed out`));
      }, 5000);
      child.on("error", (error) => {
        clearTimeout(timeout);
        reject(error);
      });
      child.on("exit", (code) => {
        clearTimeout(timeout);
        resolve(code);
      });
    });
    assert.equal(exitCode, 0, stderr);
  }

  await runHook("UserPromptSubmit", {
    session_id: "codex-session",
    cwd: "/tmp/project",
    prompt: "继续 notebook validator 任务",
  });
  await runHook("Stop", {
    session_id: "codex-session",
    cwd: "/tmp/project",
    assistant_response: "已补上 pytest fixtures，并记录了验证命令。",
  });
  await new Promise((resolve) => server.close(resolve));

  assert.deepEqual(calls.map((call) => call.path), ["/v1/context", "/v1/observe"]);
  const observe = calls.find((call) => call.path === "/v1/observe").body;
  assert.deepEqual(
    observe.messages.map((message) => message.role),
    ["user", "assistant"],
  );
  assert.match(observe.messages[0].content, /notebook validator/);
  assert.match(observe.messages[1].content, /pytest fixtures/);
  assert.deepEqual(observe.metadata.memxRecall.injectedTexts, [
    "## memX Memory\n- notebook validator prefers pytest fixtures",
    "notebook validator prefers pytest fixtures",
  ]);
});

test("Stop hook completes pending turns from Codex and Claude transcripts", async () => {
  const calls = [];
  const server = createServer(async (req, res) => {
    const chunks = [];
    for await (const chunk of req) {
      chunks.push(chunk);
    }
    const bodyText = Buffer.concat(chunks).toString("utf8");
    calls.push({
      path: req.url,
      body: bodyText ? JSON.parse(bodyText) : {},
    });
    res.writeHead(200, { "content-type": "application/json" });
    res.end(JSON.stringify({ ok: true }));
  });
  await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
  const address = server.address();
  assert.ok(address && typeof address === "object");
  const pendingDir = mkdtempSync(join(tmpdir(), "memx-pending-transcript-"));
  const env = {
    ...process.env,
    MEMX_URL: `http://127.0.0.1:${address.port}`,
    MEMX_HOOK_TIMEOUT_MS: "2000",
    MEMX_PENDING_DIR: pendingDir,
  };

  async function runHook(host, eventName, payload) {
    const child = spawn(process.execPath, [join(rootPath, "dist/.runtime/src/bin/memx-hook.mjs"), host, eventName], {
      env,
      stdio: ["pipe", "pipe", "pipe"],
    });
    let stderr = "";
    child.stderr.setEncoding("utf8");
    child.stderr.on("data", (chunk) => {
      stderr += chunk;
    });
    child.stdin.end(JSON.stringify(payload));
    const exitCode = await new Promise((resolve, reject) => {
      const timeout = setTimeout(() => {
        child.kill("SIGKILL");
        reject(new Error(`memx-hook ${host} ${eventName} test timed out`));
      }, 5000);
      child.on("error", (error) => {
        clearTimeout(timeout);
        reject(error);
      });
      child.on("exit", (code) => {
        clearTimeout(timeout);
        resolve(code);
      });
    });
    assert.equal(exitCode, 0, stderr);
  }

  const codexTranscript = join(pendingDir, "codex.jsonl");
  writeFileSync(
    codexTranscript,
    [
      JSON.stringify({
        type: "response_item",
        payload: {
          type: "message",
          role: "user",
          content: [{ type: "input_text", text: "继续 QilinLedger 任务" }],
        },
      }),
      JSON.stringify({
        type: "response_item",
        payload: {
          type: "message",
          role: "assistant",
          content: [{ type: "output_text", text: "确认：QilinLedger 继续使用 Kafka 默认队列。" }],
        },
      }),
    ].join("\n"),
    "utf8",
  );
  await runHook("codex", "UserPromptSubmit", {
    session_id: "codex-transcript-session",
    cwd: "/tmp/project",
    prompt: "继续 QilinLedger 任务",
  });
  await runHook("codex", "Stop", {
    session_id: "codex-transcript-session",
    cwd: "/tmp/project",
    transcript_path: codexTranscript,
  });

  const claudeTranscript = join(pendingDir, "claude.jsonl");
  writeFileSync(
    claudeTranscript,
    [
      JSON.stringify({
        type: "user",
        message: { role: "user", content: "继续 MingLedger 任务" },
      }),
      JSON.stringify({
        type: "assistant",
        message: {
          role: "assistant",
          content: [{ type: "text", text: "确认：MingLedger 以后默认 Pulsar。" }],
        },
      }),
    ].join("\n"),
    "utf8",
  );
  await runHook("claude-code", "UserPromptSubmit", {
    session_id: "claude-transcript-session",
    cwd: "/tmp/project",
    prompt: "继续 MingLedger 任务",
  });
  await runHook("claude-code", "Stop", {
    session_id: "claude-transcript-session",
    cwd: "/tmp/project",
    transcript_path: claudeTranscript,
  });
  await new Promise((resolve) => server.close(resolve));

  const observes = calls.filter((call) => call.path === "/v1/observe").map((call) => call.body);
  assert.equal(observes.length, 2);
  assert.deepEqual(
    observes.map((body) => body.messages.map((message) => message.role)),
    [
      ["user", "assistant"],
      ["user", "assistant"],
    ],
  );
  assert.match(observes[0].messages[1].content, /Kafka/);
  assert.match(observes[1].messages[1].content, /Pulsar/);
  assert.equal(observes[0].metadata.transcriptAssistantCapture, "codex");
  assert.equal(observes[1].metadata.transcriptAssistantCapture, "claude-code");
});

test("Claude Stop hook uses the pending transcript path when Stop payload omits it", async () => {
  const calls = [];
  const server = createServer(async (req, res) => {
    const chunks = [];
    for await (const chunk of req) {
      chunks.push(chunk);
    }
    const bodyText = Buffer.concat(chunks).toString("utf8");
    calls.push({
      path: req.url,
      body: bodyText ? JSON.parse(bodyText) : {},
    });
    res.writeHead(200, { "content-type": "application/json" });
    res.end(JSON.stringify({ ok: true }));
  });
  await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
  const address = server.address();
  assert.ok(address && typeof address === "object");
  const pendingDir = mkdtempSync(join(tmpdir(), "memx-claude-pending-transcript-"));
  const env = {
    ...process.env,
    MEMX_URL: `http://127.0.0.1:${address.port}`,
    MEMX_HOOK_TIMEOUT_MS: "2000",
    MEMX_PENDING_DIR: pendingDir,
  };

  async function runHook(eventName, payload) {
    const child = spawn(process.execPath, [join(rootPath, "dist/.runtime/src/bin/memx-hook.mjs"), "claude-code", eventName], {
      env,
      stdio: ["pipe", "pipe", "pipe"],
    });
    let stderr = "";
    child.stderr.setEncoding("utf8");
    child.stderr.on("data", (chunk) => {
      stderr += chunk;
    });
    child.stdin.end(JSON.stringify(payload));
    const exitCode = await new Promise((resolve, reject) => {
      const timeout = setTimeout(() => {
        child.kill("SIGKILL");
        reject(new Error(`memx-hook claude-code ${eventName} test timed out`));
      }, 5000);
      child.on("error", (error) => {
        clearTimeout(timeout);
        reject(error);
      });
      child.on("exit", (code) => {
        clearTimeout(timeout);
        resolve(code);
      });
    });
    assert.equal(exitCode, 0, stderr);
  }

  const claudeTranscript = join(pendingDir, "claude-real-format.jsonl");
  writeFileSync(
    claudeTranscript,
    [
      JSON.stringify({
        type: "queue-operation",
        operation: "enqueue",
        sessionId: "claude-real-transcript-session",
        content: "不用查看文件。请只回复：收到",
      }),
      JSON.stringify({
        type: "user",
        message: { role: "user", content: "不用查看文件。请只回复：收到" },
        sessionId: "claude-real-transcript-session",
      }),
      JSON.stringify({
        type: "assistant",
        message: {
          role: "assistant",
          content: [{ type: "thinking", thinking: "The user asked for a short acknowledgment." }],
        },
        sessionId: "claude-real-transcript-session",
      }),
      JSON.stringify({
        type: "assistant",
        message: {
          role: "assistant",
          content: [{ type: "text", text: "收到" }],
        },
        sessionId: "claude-real-transcript-session",
      }),
      JSON.stringify({
        type: "last-prompt",
        lastPrompt: "不用查看文件。请只回复：收到",
        sessionId: "claude-real-transcript-session",
      }),
    ].join("\n"),
    "utf8",
  );

  await runHook("UserPromptSubmit", {
    session_id: "claude-real-transcript-session",
    cwd: "/tmp/project",
    prompt: "不用查看文件。请只回复：收到",
    transcript_path: claudeTranscript,
  });
  await runHook("Stop", {
    session_id: "claude-real-transcript-session",
    cwd: "/tmp/project",
  });
  await new Promise((resolve) => server.close(resolve));

  const observes = calls.filter((call) => call.path === "/v1/observe").map((call) => call.body);
  assert.equal(observes.length, 1);
  assert.deepEqual(
    observes[0].messages.map((message) => message.role),
    ["user", "assistant"],
  );
  assert.equal(observes[0].messages[1].content, "收到");
  assert.equal(observes[0].metadata.transcriptPath, claudeTranscript);
  assert.equal(observes[0].metadata.transcriptAssistantCapture, "claude-code");
  assert.deepEqual(readdirSync(pendingDir).filter((name) => name.endsWith(".json")), []);
});

test("Stop hook waits briefly for Claude transcript assistant before writing the completed turn", async () => {
  const calls = [];
  const server = createServer(async (req, res) => {
    const chunks = [];
    for await (const chunk of req) {
      chunks.push(chunk);
    }
    const bodyText = Buffer.concat(chunks).toString("utf8");
    calls.push({
      path: req.url,
      body: bodyText ? JSON.parse(bodyText) : {},
    });
    res.writeHead(200, { "content-type": "application/json" });
    res.end(JSON.stringify({ ok: true }));
  });
  await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
  const address = server.address();
  assert.ok(address && typeof address === "object");
  const pendingDir = mkdtempSync(join(tmpdir(), "memx-delayed-transcript-"));
  const env = {
    ...process.env,
    MEMX_URL: `http://127.0.0.1:${address.port}`,
    MEMX_HOOK_TIMEOUT_MS: "2000",
    MEMX_TRANSCRIPT_CAPTURE_TIMEOUT_MS: "600",
    MEMX_TRANSCRIPT_CAPTURE_INTERVAL_MS: "40",
    MEMX_PENDING_DIR: pendingDir,
  };

  async function runHook(eventName, payload) {
    const child = spawn(process.execPath, [join(rootPath, "dist/.runtime/src/bin/memx-hook.mjs"), "claude-code", eventName], {
      env,
      stdio: ["pipe", "pipe", "pipe"],
    });
    let stderr = "";
    child.stderr.setEncoding("utf8");
    child.stderr.on("data", (chunk) => {
      stderr += chunk;
    });
    child.stdin.end(JSON.stringify(payload));
    const exitCode = await new Promise((resolve, reject) => {
      const timeout = setTimeout(() => {
        child.kill("SIGKILL");
        reject(new Error(`memx-hook claude-code ${eventName} test timed out`));
      }, 5000);
      child.on("error", (error) => {
        clearTimeout(timeout);
        reject(error);
      });
      child.on("exit", (code) => {
        clearTimeout(timeout);
        resolve(code);
      });
    });
    assert.equal(exitCode, 0, stderr);
  }

  const claudeTranscript = join(pendingDir, "claude-delayed.jsonl");
  writeFileSync(
    claudeTranscript,
    `${JSON.stringify({
      type: "user",
      message: { role: "user", content: "继续 DelayedLedger 任务" },
    })}\n`,
    "utf8",
  );
  await runHook("UserPromptSubmit", {
    session_id: "claude-delayed-session",
    cwd: "/tmp/project",
    prompt: "继续 DelayedLedger 任务",
  });
  setTimeout(() => {
    appendFileSync(
      claudeTranscript,
      `${JSON.stringify({
        type: "assistant",
        message: {
          role: "assistant",
          content: [{ type: "text", text: "确认：DelayedLedger 默认导出格式是 parquet。" }],
        },
      })}\n`,
      "utf8",
    );
  }, 120).unref();
  await runHook("Stop", {
    session_id: "claude-delayed-session",
    cwd: "/tmp/project",
    transcript_path: claudeTranscript,
  });
  await new Promise((resolve) => server.close(resolve));

  const observes = calls.filter((call) => call.path === "/v1/observe").map((call) => call.body);
  assert.equal(observes.length, 1);
  assert.deepEqual(
    observes[0].messages.map((message) => message.role),
    ["user", "assistant"],
  );
  assert.match(observes[0].messages[1].content, /parquet/);
});

test("Stop hook keeps pending turns out of memory when assistant output is unavailable", async () => {
  const calls = [];
  const server = createServer(async (req, res) => {
    const chunks = [];
    for await (const chunk of req) {
      chunks.push(chunk);
    }
    const bodyText = Buffer.concat(chunks).toString("utf8");
    calls.push({
      path: req.url,
      body: bodyText ? JSON.parse(bodyText) : {},
    });
    res.writeHead(200, { "content-type": "application/json" });
    res.end(JSON.stringify({ ok: true }));
  });
  await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
  const address = server.address();
  assert.ok(address && typeof address === "object");
  const pendingDir = mkdtempSync(join(tmpdir(), "memx-missing-transcript-"));
  const env = {
    ...process.env,
    MEMX_URL: `http://127.0.0.1:${address.port}`,
    MEMX_HOOK_TIMEOUT_MS: "2000",
    MEMX_TRANSCRIPT_CAPTURE_TIMEOUT_MS: "120",
    MEMX_TRANSCRIPT_CAPTURE_INTERVAL_MS: "30",
    MEMX_PENDING_DIR: pendingDir,
  };

  async function runHook(eventName, payload) {
    const child = spawn(process.execPath, [join(rootPath, "dist/.runtime/src/bin/memx-hook.mjs"), "claude-code", eventName], {
      env,
      stdio: ["pipe", "pipe", "pipe"],
    });
    let stderr = "";
    child.stderr.setEncoding("utf8");
    child.stderr.on("data", (chunk) => {
      stderr += chunk;
    });
    child.stdin.end(JSON.stringify(payload));
    const exitCode = await new Promise((resolve, reject) => {
      const timeout = setTimeout(() => {
        child.kill("SIGKILL");
        reject(new Error(`memx-hook claude-code ${eventName} test timed out`));
      }, 5000);
      child.on("error", (error) => {
        clearTimeout(timeout);
        reject(error);
      });
      child.on("exit", (code) => {
        clearTimeout(timeout);
        resolve(code);
      });
    });
    assert.equal(exitCode, 0, stderr);
  }

  const claudeTranscript = join(pendingDir, "claude-missing.jsonl");
  writeFileSync(
    claudeTranscript,
    `${JSON.stringify({
      type: "user",
      message: { role: "user", content: "继续 MissingAssistantLedger 任务" },
    })}\n`,
    "utf8",
  );
  await runHook("UserPromptSubmit", {
    session_id: "claude-missing-session",
    cwd: "/tmp/project",
    prompt: "继续 MissingAssistantLedger 任务",
  });
  await runHook("Stop", {
    session_id: "claude-missing-session",
    cwd: "/tmp/project",
    transcript_path: claudeTranscript,
  });
  await new Promise((resolve) => server.close(resolve));

  assert.deepEqual(calls.map((call) => call.path), ["/v1/context"]);
});

test("pending user turns are queued instead of overwritten when a Stop lacks assistant output", async () => {
  const calls = [];
  const server = createServer(async (req, res) => {
    const chunks = [];
    for await (const chunk of req) {
      chunks.push(chunk);
    }
    const bodyText = Buffer.concat(chunks).toString("utf8");
    calls.push({
      path: req.url,
      body: bodyText ? JSON.parse(bodyText) : {},
    });
    res.writeHead(200, { "content-type": "application/json" });
    if (req.url === "/v1/context") {
      res.end(JSON.stringify({ ok: true, prependContext: "" }));
      return;
    }
    res.end(JSON.stringify({ ok: true }));
  });
  await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
  const address = server.address();
  assert.ok(address && typeof address === "object");
  const pendingDir = mkdtempSync(join(tmpdir(), "memx-pending-queue-"));
  const env = {
    ...process.env,
    MEMX_URL: `http://127.0.0.1:${address.port}`,
    MEMX_HOOK_TIMEOUT_MS: "2000",
    MEMX_PENDING_DIR: pendingDir,
  };

  async function runHook(eventName, payload) {
    const child = spawn(process.execPath, [join(rootPath, "dist/.runtime/src/bin/memx-hook.mjs"), "codex", eventName], {
      env,
      stdio: ["pipe", "pipe", "pipe"],
    });
    let stderr = "";
    child.stderr.setEncoding("utf8");
    child.stderr.on("data", (chunk) => {
      stderr += chunk;
    });
    child.stdin.end(JSON.stringify(payload));
    const exitCode = await new Promise((resolve, reject) => {
      const timeout = setTimeout(() => {
        child.kill("SIGKILL");
        reject(new Error(`memx-hook queued ${eventName} test timed out`));
      }, 5000);
      child.on("error", (error) => {
        clearTimeout(timeout);
        reject(error);
      });
      child.on("exit", (code) => {
        clearTimeout(timeout);
        resolve(code);
      });
    });
    assert.equal(exitCode, 0, stderr);
  }

  await runHook("UserPromptSubmit", {
    session_id: "queued-session",
    cwd: "/tmp/project",
    timestamp: "2026-05-26T10:00:00.000Z",
    prompt: "记住 QueueProbeOne 默认队列是 Kafka",
  });
  await runHook("Stop", {
    session_id: "queued-session",
    cwd: "/tmp/project",
  });
  await runHook("UserPromptSubmit", {
    session_id: "queued-session",
    cwd: "/tmp/project",
    timestamp: "2026-05-26T10:01:00.000Z",
    prompt: "记住 QueueProbeTwo 默认队列是 Pulsar",
  });
  await runHook("Stop", {
    session_id: "queued-session",
    cwd: "/tmp/project",
    assistant_response: "已记录 QueueProbeTwo 默认队列是 Pulsar。",
  });
  await new Promise((resolve) => server.close(resolve));

  const observes = calls.filter((call) => call.path === "/v1/observe").map((call) => call.body);
  assert.equal(observes.length, 1);
  assert.match(observes[0].messages[0].content, /QueueProbeTwo/);
  assert.doesNotMatch(observes[0].messages[0].content, /QueueProbeOne/);
  const pendingFile = readdirSync(pendingDir).find((name) => name.endsWith(".json"));
  assert.ok(pendingFile, "first pending turn should remain queued for a later transcript/session flush");
  const pending = JSON.parse(readFileSync(join(pendingDir, pendingFile), "utf8"));
  assert.equal(pending.pendingTurns.length, 1);
  assert.match(pending.pendingTurns[0].messages[0].content, /QueueProbeOne/);
});

test("next prompt flushes an earlier pending turn after transcript assistant appears late", async () => {
  const calls = [];
  const server = createServer(async (req, res) => {
    const chunks = [];
    for await (const chunk of req) {
      chunks.push(chunk);
    }
    const bodyText = Buffer.concat(chunks).toString("utf8");
    calls.push({
      path: req.url,
      body: bodyText ? JSON.parse(bodyText) : {},
    });
    res.writeHead(200, { "content-type": "application/json" });
    if (req.url === "/v1/context") {
      res.end(JSON.stringify({ ok: true, prependContext: "" }));
      return;
    }
    res.end(JSON.stringify({ ok: true }));
  });
  await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
  const address = server.address();
  assert.ok(address && typeof address === "object");
  const pendingDir = mkdtempSync(join(tmpdir(), "memx-late-assistant-flush-"));
  const env = {
    ...process.env,
    MEMX_URL: `http://127.0.0.1:${address.port}`,
    MEMX_HOOK_TIMEOUT_MS: "2500",
    MEMX_TRANSCRIPT_CAPTURE_TIMEOUT_MS: "0",
    MEMX_PENDING_DIR: pendingDir,
  };

  async function runHook(eventName, payload) {
    const child = spawn(process.execPath, [join(rootPath, "dist/.runtime/src/bin/memx-hook.mjs"), "claude-code", eventName], {
      env,
      stdio: ["pipe", "pipe", "pipe"],
    });
    let stderr = "";
    child.stderr.setEncoding("utf8");
    child.stderr.on("data", (chunk) => {
      stderr += chunk;
    });
    child.stdin.end(JSON.stringify(payload));
    const exitCode = await new Promise((resolve, reject) => {
      const timeout = setTimeout(() => {
        child.kill("SIGKILL");
        reject(new Error(`memx-hook late assistant ${eventName} test timed out`));
      }, 5000);
      child.on("error", (error) => {
        clearTimeout(timeout);
        reject(error);
      });
      child.on("exit", (code) => {
        clearTimeout(timeout);
        resolve(code);
      });
    });
    assert.equal(exitCode, 0, stderr);
  }

  const transcript = join(pendingDir, "late-output.jsonl");
  writeFileSync(
    transcript,
    `${JSON.stringify({
      type: "user",
      message: { role: "user", content: "继续 LateAssistantLedger 任务" },
    })}\n`,
    "utf8",
  );
  await runHook("UserPromptSubmit", {
    session_id: "late-assistant-session",
    cwd: "/tmp/project",
    prompt: "继续 LateAssistantLedger 任务",
  });
  await runHook("Stop", {
    session_id: "late-assistant-session",
    cwd: "/tmp/project",
    transcript_path: transcript,
  });
  appendFileSync(
    transcript,
    `${JSON.stringify({
      type: "assistant",
      message: { role: "assistant", content: "已完成 LateAssistantLedger 的任务总结。" },
    })}\n`,
    "utf8",
  );
  await runHook("UserPromptSubmit", {
    session_id: "late-assistant-session",
    cwd: "/tmp/project",
    prompt: "现在继续下一个自然任务",
  });
  await new Promise((resolve) => server.close(resolve));

  const observes = calls.filter((call) => call.path === "/v1/observe").map((call) => call.body);
  assert.equal(observes.length, 1);
  assert.match(observes[0].messages[0].content, /LateAssistantLedger/);
  assert.match(observes[0].messages[1].content, /任务总结/);
  assert.doesNotMatch(JSON.stringify(observes[0].messages), /现在继续下一个自然任务/);
  const pendingFile = readdirSync(pendingDir).find((name) => name.endsWith(".json"));
  assert.ok(pendingFile, "the new prompt should remain pending for its own Stop hook");
  const pending = JSON.parse(readFileSync(join(pendingDir, pendingFile), "utf8"));
  assert.equal(pending.pendingTurns.length, 1);
  assert.match(pending.pendingTurns[0].messages[0].content, /现在继续下一个自然任务/);
});

test("PreToolUse denies attempts to use host-native memory stores instead of memX lifecycle memory", async () => {
  const child = spawn(process.execPath, [join(rootPath, "dist/.runtime/src/bin/memx-hook.mjs"), "codex", "PreToolUse"], {
    env: {
      ...process.env,
      MEMX_URL: "http://127.0.0.1:9",
      MEMX_HOOK_TIMEOUT_MS: "2000",
    },
    stdio: ["pipe", "pipe", "pipe"],
  });
  let stdout = "";
  let stderr = "";
  child.stdout.setEncoding("utf8");
  child.stderr.setEncoding("utf8");
  child.stdout.on("data", (chunk) => {
    stdout += chunk;
  });
  child.stderr.on("data", (chunk) => {
    stderr += chunk;
  });
  child.stdin.end(
    JSON.stringify({
      session_id: "codex-native-memory-bypass",
      cwd: "/tmp/project",
      tool_name: "exec_command",
      tool_input: {
        cmd: 'mkdir -p "$HOME/.codex/memories" && printf "RiverMap uses Kafka" >> "$HOME/.codex/memories/user_facts.txt"',
      },
    }),
  );

  const exitCode = await new Promise((resolve, reject) => {
    const timeout = setTimeout(() => {
      child.kill("SIGKILL");
      reject(new Error("memx-hook PreToolUse bypass guard test timed out"));
    }, 5000);
    child.on("error", (error) => {
      clearTimeout(timeout);
      reject(error);
    });
    child.on("exit", (code) => {
      clearTimeout(timeout);
      resolve(code);
    });
  });

  assert.equal(exitCode, 0, stderr);
  const output = JSON.parse(stdout);
  assert.equal(output.hookSpecificOutput.hookEventName, "PreToolUse");
  assert.equal(output.hookSpecificOutput.permissionDecision, "deny");
  assert.match(output.hookSpecificOutput.permissionDecisionReason, /memX/i);
});

test("Codex UserPromptSubmit hook recalls before observing the current prompt", async () => {
  let currentPromptObserved = false;
  const server = createServer(async (req, res) => {
    const chunks = [];
    for await (const chunk of req) {
      chunks.push(chunk);
    }
    const bodyText = Buffer.concat(chunks).toString("utf8");
    const body = bodyText ? JSON.parse(bodyText) : {};
    res.writeHead(200, { "content-type": "application/json" });
    if (req.url === "/v1/observe") {
      currentPromptObserved = body.messages?.some((message) =>
        String(message.content ?? "").includes("NebulaLedger"),
      );
      res.end(JSON.stringify({ ok: true }));
      return;
    }
    if (req.url === "/v1/context") {
      await new Promise((resolve) => setTimeout(resolve, 80));
      res.end(
        JSON.stringify({
          ok: true,
          prependContext: currentPromptObserved
            ? "## memX Memory\n- leaked current prompt: NebulaLedger"
            : "## memX Memory\n- previous memory only",
        }),
      );
      return;
    }
    res.end(JSON.stringify({ ok: true }));
  });
  await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
  const address = server.address();
  assert.ok(address && typeof address === "object");

  const child = spawn(process.execPath, [join(rootPath, "dist/.runtime/src/bin/memx-hook.mjs"), "codex", "UserPromptSubmit"], {
    env: {
      ...process.env,
      MEMX_URL: `http://127.0.0.1:${address.port}`,
      MEMX_HOOK_TIMEOUT_MS: "2000",
    },
    stdio: ["pipe", "pipe", "pipe"],
  });
  let stdout = "";
  let stderr = "";
  child.stdout.setEncoding("utf8");
  child.stderr.setEncoding("utf8");
  child.stdout.on("data", (chunk) => {
    stdout += chunk;
  });
  child.stderr.on("data", (chunk) => {
    stderr += chunk;
  });
  child.stdin.end(
    JSON.stringify({
      session_id: "codex-session",
      cwd: "/tmp/project",
      prompt: "NebulaLedger 的校验命令是什么？",
    }),
  );

  const exitCode = await new Promise((resolve, reject) => {
    const timeout = setTimeout(() => {
      child.kill("SIGKILL");
      reject(new Error("memx-hook current prompt ordering test timed out"));
    }, 5000);
    child.on("error", (error) => {
      clearTimeout(timeout);
      reject(error);
    });
    child.on("exit", (code) => {
      clearTimeout(timeout);
      resolve(code);
    });
  });
  await new Promise((resolve) => server.close(resolve));

  assert.equal(exitCode, 0, stderr);
  const output = JSON.parse(stdout);
  assert.equal(
    output.hookSpecificOutput.additionalContext,
    "## memX Memory\n- previous memory only",
  );
});

test("MCP handler exposes memX tools and proxies calls to REST", async () => {
  const { handleMcpRequest } = await import("../dist/.runtime/src/host/mcpProtocol.mjs");
  const calls = [];
  const proxy = async (path, init) => {
    calls.push({ path, init });
    return { ok: true, path };
  };

  const list = await handleMcpRequest(
    { jsonrpc: "2.0", id: 1, method: "tools/list" },
    { proxy },
  );
  const toolNames = list.result.tools.map((tool) => tool.name).sort();
  assert.deepEqual(toolNames, [
    "memx_audit",
    "memx_forget",
    "memx_observe",
    "memx_recall",
    "memx_remember",
    "memx_stats",
  ]);

  const call = await handleMcpRequest(
    {
      jsonrpc: "2.0",
      id: 2,
      method: "tools/call",
      params: { name: "memx_recall", arguments: { query: "Notebook validator", limit: 3 } },
    },
    { proxy },
  );
  assert.equal(calls[0].path, "/v1/recall");
  assert.equal(JSON.parse(calls[0].init.body).query, "Notebook validator");
  assert.equal(call.result.content[0].type, "text");

  await handleMcpRequest(
    {
      jsonrpc: "2.0",
      id: 3,
      method: "tools/call",
      params: {
        name: "memx_audit",
        arguments: { limit: 5, hostId: "codex", actorId: "memx-shared", sessionId: "s1" },
      },
    },
    { proxy },
  );
  assert.equal(
    calls[1].path,
    "/v1/audit?limit=5&hostId=codex&actorId=memx-shared&sessionId=s1",
  );

  const initialized = await handleMcpRequest({
    jsonrpc: "2.0",
    method: "notifications/initialized",
    params: {},
  });
  assert.equal(initialized, null);
});

test("MCP none profile hides all tools for native lifecycle plugins", async () => {
  const { handleMcpRequest } = await import("../dist/.runtime/src/host/mcpProtocol.mjs");
  const previous = process.env.MEMX_MCP_TOOLS;
  process.env.MEMX_MCP_TOOLS = "none";
  try {
    const list = await handleMcpRequest({ jsonrpc: "2.0", id: 1, method: "tools/list" });
    const toolNames = list.result.tools.map((tool) => tool.name).sort();
    assert.deepEqual(toolNames, []);

    const init = await handleMcpRequest({ jsonrpc: "2.0", id: 0, method: "initialize" });
    assert.deepEqual(init.result.capabilities, {});

    const recall = await handleMcpRequest({
      jsonrpc: "2.0",
      id: 2,
      method: "tools/call",
      params: { name: "memx_recall", arguments: { query: "Notebook validator" } },
    });
    assert.equal(recall.error.code, -32601);
    assert.match(recall.error.message, /not available/i);

    const remember = await handleMcpRequest({
      jsonrpc: "2.0",
      id: 3,
      method: "tools/call",
      params: { name: "memx_remember", arguments: { content: "Use pnpm" } },
    });
    assert.equal(remember.error.code, -32601);
    assert.match(remember.error.message, /not available/i);

    const audit = await handleMcpRequest({
      jsonrpc: "2.0",
      id: 4,
      method: "tools/call",
      params: { name: "memx_audit", arguments: { limit: 5 } },
    });
    assert.equal(audit.error.code, -32601);
    assert.match(audit.error.message, /not available/i);
  } finally {
    if (previous === undefined) {
      delete process.env.MEMX_MCP_TOOLS;
    } else {
      process.env.MEMX_MCP_TOOLS = previous;
    }
  }
});

test("standalone host service isolates default memory by native host", async () => {
  const { MemxHostService } = await import("../dist/.runtime/src/host/service.mjs");
  const dir = mkdtempSync(join(tmpdir(), "memx-host-scope-"));
  const service = new MemxHostService({
    config: {
      dbPath: join(dir, "{agentId}", "memx.sqlite"),
      defaultScope: "agent:{agentId}",
      allowedScopes: ["agent:{agentId}", "session:{sessionKey}"],
      embedding: { provider: "off" },
      advanced: {
        llmClassifierEnabled: false,
        llmProvider: "openai-compatible",
        llmClassifierModel: "unused",
      },
    },
    logger: { warn() {}, info() {}, debug() {}, error() {} },
  });
  try {
    const codex = await service.observe({
      hostId: "codex",
      sessionId: "same-session",
      messages: [{ role: "user", content: "记住 Codex 使用 pnpm" }],
    });
    const claude = await service.observe({
      hostId: "claude-code",
      sessionId: "same-session",
      messages: [{ role: "user", content: "记住 Claude 使用 uv" }],
    });

    assert.notEqual(codex.actorId, claude.actorId);
    assert.match(codex.actorId, /^codex--/);
    assert.match(claude.actorId, /^claude-code--/);
    assert.equal(existsSync(join(dir, codex.actorId, "memx.sqlite")), true);
    assert.equal(existsSync(join(dir, claude.actorId, "memx.sqlite")), true);

    const codexStats = await service.stats({
      hostId: "codex",
      actorId: "memx-shared",
      sessionId: "stats",
    });
    const claudeAudit = await service.audit(10, {
      hostId: "claude-code",
      actorId: "memx-shared",
      sessionId: "audit",
    });
    assert.equal(codexStats.agentId, codex.actorId);
    assert.equal(claudeAudit.agentId, claude.actorId);
  } finally {
    await service.close();
  }
});

test("standalone host service returns no injected context when recall has no evidence", async () => {
  const { DEFAULT_MEMORY_CONFIG } = await import("../dist/.runtime/src/config.mjs");
  const { MemxHostService } = await import("../dist/.runtime/src/host/service.mjs");
  const dir = mkdtempSync(join(tmpdir(), "memx-empty-recall-"));
  const config = structuredClone(DEFAULT_MEMORY_CONFIG);
  config.dbPath = join(dir, "{agentId}", "memx.sqlite");
  config.embedding.provider = "off";
  config.advanced.llmClassifierEnabled = false;
  config.advanced.enableMaintenanceJobs = false;
  config.advanced.enableTurnSemanticCompiler = false;
  config.advanced.enableQueryCompiler = false;
  config.advanced.enableEmbeddingCandidates = false;
  config.advanced.enableEmbeddingClustering = false;
  const service = new MemxHostService({
    config,
    logger: { warn() {}, info() {}, debug() {}, error() {} },
  });
  try {
    const result = await service.context({
      hostId: "codex",
      actorId: "memx-shared",
      sessionId: "empty-session",
      query: "NebulaLedger 的校验命令是什么？",
    });

    assert.equal(result.prependContext, "");
    assert.equal(result.recall.context, "");
  } finally {
    await service.close();
  }
});

test("native context injection follows retrieval evidence without a memory-intent gate", async () => {
  const { assessNativeContextEligibility } = await import("../dist/.runtime/src/host/service.mjs");
  const { compileQueryWithoutSemanticFallback } = await import(
    "../dist/.runtime/src/pipeline/queryCompiler.mjs"
  );
  const strongPacket = {
    packetId: "packet-1",
    slotId: "answer",
    operationType: "lookup",
    role: "answer",
    injected: true,
    primaryText:
      "项目 BlueWhaleLedger 的回归校验命令是 npm run verify:ledger，幂等键字段叫 operationFingerprint，批量导入必须先跑 dry-run。",
    supportingTexts: [],
    sourceRefs: ["user:1"],
    layers: ["chunk"],
    displayLines: [
      "[answer] 项目 BlueWhaleLedger 的回归校验命令是 npm run verify:ledger，幂等键字段叫 operationFingerprint，批量导入必须先跑 dry-run。",
    ],
    coverage: { filled: true, missing: [], confidence: 0.88 },
    grade: {
      retrievalScore: 0.82,
      answerScore: 0.78,
      contextBindingScore: 0.72,
      slotCoverageScore: 0.88,
      authorityScore: 0.8,
      finalScore: 0.78,
    },
  };
  const weakPacket = {
    ...strongPacket,
    packetId: "packet-weak",
    coverage: { filled: false, missing: ["answer"], confidence: 0.2 },
    grade: {
      ...strongPacket.grade,
      finalScore: 0.14,
    },
  };
  const strongBundle = {
    routeConfidence: 0.44,
    evidencePackets: [strongPacket],
  };
  const weakBundle = {
    routeConfidence: 0.24,
    evidencePackets: [weakPacket],
  };

  const fallback = compileQueryWithoutSemanticFallback(
    "把“批量导入完成”改成更适合日志的短句。",
  );
  const generic = assessNativeContextEligibility(
    "把“批量导入完成”改成更适合日志的短句。",
    fallback,
    weakBundle,
  );
  assert.equal(generic.eligible, false);
  assert.equal(generic.reason, "weak-evidence");

  const strong = assessNativeContextEligibility(
    "把“批量导入完成”改成更适合日志的短句。",
    fallback,
    strongBundle,
  );
  assert.equal(strong.eligible, true);
  assert.equal(strong.reason, "strong-evidence");

  const anchored = assessNativeContextEligibility(
    "BlueWhaleLedger 的回归校验命令是什么？",
    {
      ...compileQueryWithoutSemanticFallback("BlueWhaleLedger 的回归校验命令是什么？"),
      queryEntities: [{ name: "BlueWhaleLedger", type: "project", role: "subject" }],
    },
    strongBundle,
  );
  assert.equal(anchored.eligible, true);
  assert.equal(anchored.reason, "llm-query-entities");

  const entitySupported = assessNativeContextEligibility(
    "BlueWhaleLedger 的回归校验命令是什么？",
    {
      ...compileQueryWithoutSemanticFallback("BlueWhaleLedger 的回归校验命令是什么？"),
      queryEntities: [{ name: "BlueWhaleLedger", type: "project", role: "subject" }],
    },
    {
      routeConfidence: 0.24,
      evidencePackets: [
        {
          ...weakPacket,
          coverage: { filled: true, missing: [], confidence: 0.36 },
        },
      ],
    },
  );
  assert.equal(entitySupported.eligible, false);
  assert.equal(entitySupported.reason, "weak-evidence");

  const suppressedAnchor = assessNativeContextEligibility(
    "现在先不谈 BlueWhaleLedger。帮我起三个 API 名字。",
    {
      ...compileQueryWithoutSemanticFallback("现在先不谈 BlueWhaleLedger。帮我起三个 API 名字。"),
      queryEntities: [{ name: "BlueWhaleLedger", type: "project", role: "context" }],
      suppressedEntities: [
        { name: "BlueWhaleLedger", type: "project", reason: "user excluded this topic" },
      ],
    },
    strongBundle,
  );
  assert.equal(suppressedAnchor.eligible, false);
  assert.equal(suppressedAnchor.reason, "suppressed-entity-anchor");
});

test("MCP stdio accepts standard Content-Length framed requests", async () => {
  const child = spawn(process.execPath, [join(rootPath, "dist/.runtime/src/bin/memx-mcp.mjs")], {
    env: {
      ...process.env,
      MEMX_URL: "http://127.0.0.1:9",
      MEMX_SECRET: "",
    },
    stdio: ["pipe", "pipe", "pipe"],
  });
  let stdout = "";
  let stderr = "";
  child.stdout.setEncoding("utf8");
  child.stderr.setEncoding("utf8");
  child.stdout.on("data", (chunk) => {
    stdout += chunk;
  });
  child.stderr.on("data", (chunk) => {
    stderr += chunk;
  });

  child.stdin.end(
    encodeMcpFrame({
      jsonrpc: "2.0",
      id: 1,
      method: "initialize",
      params: {
        protocolVersion: "2024-11-05",
        capabilities: {},
        clientInfo: { name: "node-test", version: "0" },
      },
    }),
  );

  const exitCode = await new Promise((resolve, reject) => {
    const timeout = setTimeout(() => {
      child.kill("SIGKILL");
      reject(new Error("memx-mcp framed stdio test timed out"));
    }, 5000);
    child.on("error", (error) => {
      clearTimeout(timeout);
      reject(error);
    });
    child.on("close", (code) => {
      clearTimeout(timeout);
      resolve(code);
    });
  });

  assert.equal(exitCode, 0);
  assert.equal(stderr, "");
  const response = decodeMcpFrame(stdout);
  assert.equal(response.id, 1);
  assert.equal(response.result.serverInfo.name, "memx");
  assert.equal(response.result.capabilities.tools instanceof Object, true);
});

test("connect helpers generate Codex TOML and generic MCP JSON without duplicating blocks", async () => {
  const {
    applyCodexTomlConnect,
    buildGenericMcpConfig,
    hasCodexMemxBlock,
  } = await import("../dist/.runtime/src/host/connect.mjs");

  const first = applyCodexTomlConnect("");
  const second = applyCodexTomlConnect(first);
  assert.equal(hasCodexMemxBlock(second), true);
  assert.equal((second.match(/\[mcp_servers\.memx\]/g) ?? []).length, 1);
  assert.match(second, /github:NeoLi00\/memX/);

  const generic = buildGenericMcpConfig();
  assert.equal(generic.mcpServers.memx.command, "npx");
  assert.deepEqual(generic.mcpServers.memx.args, [
    "-y",
    "-p",
    "github:NeoLi00/memX",
    "memx-mcp",
  ]);
});
