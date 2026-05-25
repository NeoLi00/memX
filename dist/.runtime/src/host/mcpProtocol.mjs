//#region src/host/mcpProtocol.ts
const DEFAULT_URL = "http://127.0.0.1:3878";
function stringProp(description) {
	return {
		type: "string",
		description
	};
}
function numberProp(description) {
	return {
		type: "number",
		description
	};
}
const MEMX_MCP_TOOLS = [
	{
		name: "memx_recall",
		description: "Recall relevant memX memory across working state, facts, events, and graph.",
		inputSchema: {
			type: "object",
			properties: {
				query: stringProp("Focused recall query."),
				limit: numberProp("Maximum number of returned items."),
				hostId: stringProp("Optional host identifier such as codex or claude-code."),
				actorId: stringProp("Optional actor identifier."),
				sessionId: stringProp("Optional host session identifier.")
			},
			required: ["query"]
		}
	},
	{
		name: "memx_remember",
		description: "Store reusable memory through the memX semantic write pipeline.",
		inputSchema: {
			type: "object",
			properties: {
				content: stringProp("Memory content to store."),
				type: stringProp("Optional memory type hint."),
				hostId: stringProp("Optional host identifier such as codex or claude-code."),
				actorId: stringProp("Optional actor identifier."),
				sessionId: stringProp("Optional host session identifier.")
			},
			required: ["content"]
		}
	},
	{
		name: "memx_observe",
		description: "Append a host turn or lifecycle event to memX.",
		inputSchema: {
			type: "object",
			properties: {
				hostId: stringProp("Host identifier, for example codex or claude-code."),
				sessionId: stringProp("Host session identifier."),
				messages: {
					type: "array",
					description: "Turn messages to observe."
				}
			}
		}
	},
	{
		name: "memx_forget",
		description: "Delete or tombstone a memory object.",
		inputSchema: {
			type: "object",
			properties: {
				kind: stringProp("Memory kind: doc, event, fact, or state."),
				id: stringProp("Memory object id.")
			},
			required: ["id"]
		}
	},
	{
		name: "memx_stats",
		description: "Return memX store statistics for the requested host-scoped actor.",
		inputSchema: {
			type: "object",
			properties: {
				hostId: stringProp("Optional host identifier such as codex or claude-code."),
				actorId: stringProp("Optional actor identifier."),
				sessionId: stringProp("Optional host session identifier.")
			}
		}
	},
	{
		name: "memx_audit",
		description: "Return recent memX audit signals and maintenance activity for the requested host-scoped actor.",
		inputSchema: {
			type: "object",
			properties: {
				limit: numberProp("Maximum number of audit rows."),
				hostId: stringProp("Optional host identifier such as codex or claude-code."),
				actorId: stringProp("Optional actor identifier."),
				sessionId: stringProp("Optional host session identifier.")
			}
		}
	}
];
const LIFECYCLE_SAFE_MCP_TOOLS = new Set([
	"memx_forget",
	"memx_stats",
	"memx_audit"
]);
function activeToolsProfile() {
	const raw = (process.env["MEMX_MCP_TOOLS"] || "").trim().toLowerCase();
	if (raw === "none" || raw === "off" || raw === "disabled") return "none";
	if (raw === "lifecycle-safe" || raw === "native" || raw === "safe") return "lifecycle-safe";
	return "full";
}
function toolsForProfile(profile) {
	if (profile === "none") return [];
	if (profile === "lifecycle-safe") return MEMX_MCP_TOOLS.filter((tool) => LIFECYCLE_SAFE_MCP_TOOLS.has(tool.name));
	return MEMX_MCP_TOOLS;
}
function asRecord(value) {
	return value && typeof value === "object" && !Array.isArray(value) ? value : {};
}
function jsonResponse(id, result) {
	return {
		jsonrpc: "2.0",
		id,
		result
	};
}
function errorResponse(id, code, message) {
	return {
		jsonrpc: "2.0",
		id,
		error: {
			code,
			message
		}
	};
}
function textResult(payload) {
	return { content: [{
		type: "text",
		text: JSON.stringify(payload, null, 2)
	}] };
}
function callBody(args, extra) {
	return {
		method: "POST",
		headers: { "content-type": "application/json" },
		body: JSON.stringify({
			...args,
			...extra ?? {}
		})
	};
}
function hostScopedQuery(args, keys = [
	"hostId",
	"actorId",
	"sessionId"
]) {
	const params = new URLSearchParams();
	for (const key of keys) {
		const value = args[key];
		if (typeof value === "string" && value.trim()) params.set(key, value.trim());
	}
	const query = params.toString();
	return query ? `?${query}` : "";
}
function pathForTool(name, args) {
	switch (name) {
		case "memx_recall": return {
			path: "/v1/recall",
			init: callBody(args)
		};
		case "memx_remember": return {
			path: "/v1/remember",
			init: callBody(args)
		};
		case "memx_observe": return {
			path: "/v1/observe",
			init: callBody(args)
		};
		case "memx_forget": return {
			path: "/v1/forget",
			init: callBody(args)
		};
		case "memx_stats": return {
			path: `/v1/stats${hostScopedQuery(args)}`,
			init: { method: "GET" }
		};
		case "memx_audit": {
			const limit = typeof args.limit === "number" ? Math.trunc(args.limit) : 50;
			return {
				path: `/v1/audit${hostScopedQuery({
					...args,
					limit: String(Math.max(1, Math.min(limit, 200)))
				}, [
					"limit",
					"hostId",
					"actorId",
					"sessionId"
				])}`,
				init: { method: "GET" }
			};
		}
		default: throw new Error(`unknown tool: ${name}`);
	}
}
function authHeaders() {
	const secret = process.env["MEMX_SECRET"];
	return secret ? { authorization: `Bearer ${secret}` } : {};
}
async function defaultMemxProxy(path, init) {
	const url = (process.env["MEMX_URL"] || DEFAULT_URL).replace(/\/+$/u, "");
	const response = await fetch(`${url}${path}`, {
		...init,
		headers: {
			...init.headers,
			...authHeaders()
		},
		signal: AbortSignal.timeout(15e3)
	});
	if (!response.ok) throw new Error(`${init.method ?? "GET"} ${path} -> ${response.status} ${response.statusText}`);
	const text = await response.text();
	return text ? JSON.parse(text) : null;
}
async function handleMcpRequest(request, deps = {}) {
	const id = request.id ?? null;
	try {
		if (request.method === "notifications/initialized") return null;
		if (request.method === "initialize") return jsonResponse(id, {
			protocolVersion: "2024-11-05",
			serverInfo: {
				name: "memx",
				version: "2026.3.15"
			},
			capabilities: toolsForProfile(activeToolsProfile()).length > 0 ? { tools: {} } : {}
		});
		if (request.method === "ping") return jsonResponse(id, {});
		if (request.method === "tools/list") return jsonResponse(id, { tools: toolsForProfile(activeToolsProfile()) });
		if (request.method === "tools/call") {
			const params = asRecord(request.params);
			const name = typeof params.name === "string" ? params.name : "";
			const args = asRecord(params.arguments);
			if (!toolsForProfile(activeToolsProfile()).some((tool) => tool.name === name)) return errorResponse(id, -32601, `tool not available in this memX MCP profile: ${name}`);
			const { path, init } = pathForTool(name, args);
			return jsonResponse(id, textResult(await (deps.proxy ?? defaultMemxProxy)(path, init)));
		}
		return errorResponse(id, -32601, `method not found: ${request.method ?? "unknown"}`);
	} catch (error) {
		return errorResponse(id, -32e3, error instanceof Error ? error.message : String(error));
	}
}
//#endregion
export { MEMX_MCP_TOOLS, defaultMemxProxy, handleMcpRequest };
