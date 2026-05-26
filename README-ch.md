<p align="center">
  <img src="./assets/memx-cover.svg" alt="memX - AI agent 的自学习，自维护记忆" width="920">
</p>

<p align="center">
  <a href="./README.md">English</a> · <a href="./README-ch.md">中文</a> ·
  <a href="./ARCHITECTURE-ch.md">架构深读</a>
</p>

---

memX 把完成后的工作 turn 编译成结构化、可检索、可维护的长期记忆，并只把当前 query 需要的证据注入 prompt。
它原生适配 Codex、Claude Code、OpenClaw，也能通过同一层本地记忆接入任何支持 MCP 的 client。

## Benchmark

<table align="center">
  <thead>
    <tr>
      <th>测试集</th>
      <th>范围</th>
      <th>R@3 成功率</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><strong>LongMemEval-S</strong></td>
      <td>长上下文记忆召回</td>
      <td><strong>94.2%</strong></td>
    </tr>
    <tr>
      <td><strong>真实工程 case</strong></td>
      <td>30 个 case，每个 20+ turns</td>
      <td><strong>100%</strong></td>
    </tr>
  </tbody>
</table>

## 架构

<p align="center">
  <img src="./assets/memx-overview-zh.svg" alt="memX 粗粒度架构" width="920">
</p>

## Agent 支持

<table align="center">
  <tr>
    <td align="center" width="56"><img src="./assets/agent-logos/codex.png" alt="Codex logo" width="34"></td>
    <td><strong>Codex</strong></td>
    <td><sub>原生 hooks，默认隐藏 MCP</sub></td>
  </tr>
  <tr>
    <td align="center" width="56"><img src="./assets/agent-logos/claude-code.png" alt="Claude Code logo" width="34"></td>
    <td><strong>Claude Code</strong></td>
    <td><sub>原生 hooks，默认隐藏 MCP</sub></td>
  </tr>
  <tr>
    <td align="center" width="56"><img src="./assets/agent-logos/openclaw.png" alt="OpenClaw logo" width="34"></td>
    <td><strong>OpenClaw</strong></td>
    <td><sub>原生 + hooks</sub></td>
  </tr>
  <tr>
    <td align="center" width="56"><strong>MCP</strong></td>
    <td><strong>MCP clients</strong></td>
    <td><sub>任何兼容 MCP 的 client</sub></td>
  </tr>
</table>

## 快速启动

依赖：Node.js 22.14+ 或 Node 24。OpenClaw 安装需要 OpenClaw 2026.3.25+。只有使用默认本地
embedding 运行时时才需要 Python 3。

README 命令默认使用 GitHub package spec。每次全新执行都会拉取 GitHub 当前代码，因此安装流程
不需要等待 npm publish。以后如果明确要用 npm 发布通道，再把 `github:NeoLi00/memX`
换成 `@neoli00/memx`。

先把下面这些值填好，再运行对应命令：

- `--llm-provider`：memX 要调用的 LLM provider adapter。可选 `openai-compatible`、
  `anthropic`、`google` 或 `ollama`。
- `--llm-base-url`：provider 的 base URL。例如 `https://api.openai.com/v1`、
  `https://api.anthropic.com/v1`、`https://generativelanguage.googleapis.com/v1beta`，或者
  Ollama 的 `http://127.0.0.1:11434`。
- `--llm-model`：memX 用来做记忆编译、召回规划和维护的模型。建议选速度快、成本低、JSON 输出
  稳定的模型。
- `--llm-api-key`：provider API key。如果不想写明文，用
  `--llm-api-key-env PROVIDER_API_KEY`，配置里会保存环境变量引用。本地 Ollama 可以不传 key。

默认 embedding 是本地 `sentence-transformers-local`，模型 `intfloat/multilingual-e5-small`。
只有想覆盖默认值时才需要额外传 `--embedding-provider` 和 `--embedding-model`。使用 `--dry-run`
可以先预览会写入哪些文件、会执行哪些 exec-form 命令。

对 Codex 和 Claude Code，native hooks 是默认生命周期入口，自动负责召回和 turn 捕获。Quickstart
会一次性安装 native plugin、写入共享 memX 配置、启动或刷新 managed local memX service，并默认用
`--mcp-tools none` 隐藏 MCP 记忆工具。这样既避免重复召回/重复写入，也避免 agent 把 audit 数据当作
旁路记忆读取。只有明确希望 agent 看见完整 MCP 工具集时，才加 `--mcp-tools full`。通用 MCP
quickstart 默认仍是 `full`，因为它没有 native lifecycle hooks。默认 native 记忆也会按 host
隔离，所以 Codex 和 Claude Code 不会共用同一个本地数据库，除非你主动覆盖 database path 和
actor 设置。

如果 `http://127.0.0.1:3878` 已经被一个不受当前 quickstart 管理的 memX-compatible service 占用，
quickstart 会停止并提示冲突，而不是静默复用旧服务。你可以先停掉旧服务，或传一个空闲本地地址，
例如 `--memx-url http://127.0.0.1:3888`。

### Claude Code

这条命令会一次性写入共享 memX 配置、生成本地 Claude Code plugin marketplace、安装 native
生命周期 hooks，并启动 managed local memX service。

```bash
npx -y -p github:NeoLi00/memX memx quickstart claude-code \
  --llm-provider openai-compatible \
  --llm-base-url https://llm.example.com/v1 \
  --llm-model fast-memory-model \
  --llm-api-key sk-your-provider-key
```

### Codex

这条命令会一次性写入共享 memX 配置、生成本地 Codex plugin marketplace、安装 native 生命周期
hooks，并启动 managed local memX service。

```bash
npx -y -p github:NeoLi00/memX memx quickstart codex \
  --llm-provider openai-compatible \
  --llm-base-url https://llm.example.com/v1 \
  --llm-model fast-memory-model \
  --llm-api-key sk-your-provider-key
```

### OpenClaw

```bash
npx -y -p github:NeoLi00/memX memx quickstart openclaw \
  --llm-provider openai-compatible \
  --llm-base-url https://llm.example.com/v1 \
  --llm-model fast-memory-model \
  --llm-api-key sk-your-provider-key
```

### 通用 MCP

这个路径用于没有 memX native lifecycle adapter 的 MCP client。Quickstart 会写入共享 memX
配置、启动 managed local memX service，并输出可以放进 MCP client 的 server 配置。

```bash
npx -y -p github:NeoLi00/memX memx quickstart mcp \
  --llm-provider openai-compatible \
  --llm-base-url https://llm.example.com/v1 \
  --llm-model fast-memory-model \
  --llm-api-key sk-your-provider-key
```

### Service 管理

```bash
npx -y -p github:NeoLi00/memX memx service status
npx -y -p github:NeoLi00/memX memx service restart
npx -y -p github:NeoLi00/memX memx service stop
```

如果 quickstart 时用了非默认路径或端口，管理 service 时也传同样的 `--home`、`--memx-url` 和
`--memx-secret`。

## 干净卸载

每个卸载命令都会先备份目标配置文件，然后只删除 memX 自己写入的条目。Claude Code 和 Codex
清理会同时停止 managed local service、卸载 native plugin、移除本地 marketplace，并删除生成的
marketplace snapshot。
OpenClaw 清理会额外删掉残留的 `memx` / `memory-memx` slot、allow 和 entry 引用，并在
OpenClaw 仍能识别插件时 best-effort 删除当前和旧版插件文件。

```bash
npx -y -p github:NeoLi00/memX memx uninstall openclaw
npx -y -p github:NeoLi00/memX memx uninstall codex
npx -y -p github:NeoLi00/memX memx uninstall claude-code
```

加 `--dry-run` 可以预览；如果使用非默认配置路径，加 `--config /path/to/config`。

## memX 能做什么

- **长期记住工作上下文**：项目决策、用户偏好、任务状态、长 source segments 和原始证据都能
  保留来源关系。
- **连接相关对象**：项目、仓库、工具、文件、资源、卡点和结果可以形成实体与关系边。
- **学习协作模式**：重复出现的证据可以变成可复用建议，同时保留支撑来源。
- **自动维护记忆**：纠错可以替代旧事实，稳定证据可以提升，过期任务状态不会长期压过当前状态。
- **紧凑召回证据**：事实、事件、状态、片段、关系、资源和已学习模式一起参与召回，再压成小段
  evidence 注入。
