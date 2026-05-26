import type { AnyAgentTool } from "openclaw/plugin-sdk/core";
import { runConsolidation } from "./pipeline/consolidate.js";
import { buildStoredFactObjectValueJson } from "./pipeline/normalize.js";
import { compileQuery } from "./pipeline/queryCompiler.js";
import { retrieveEvidence } from "./pipeline/retrieve.js";
import { expandStateKeyAliases } from "./pipeline/semantics.js";
import { buildOperationContext, type MemxRuntimeManager } from "./runtime.js";
import { isScopeAllowed, resolveDefaultScope, scopeVarsForContext } from "./security/scopes.js";
import { nowIso, normalizeName, normalizeText, stableHash } from "./support.js";
import { jsonToolResult, readBoolean, readNumber, readString, stringEnum, Type } from "./tooling.js";
import type { MemoryPluginConfig, MemxLogger } from "./types.js";

const FORGET_KINDS = ["doc", "event", "fact", "state"] as const;
type ToolContext = {
  agentId?: string;
  sessionKey?: string;
  workspaceDir?: string;
};

function resolveToolContext(
  toolCtx: ToolContext,
  config: MemoryPluginConfig,
): ReturnType<typeof buildOperationContext> {
  return buildOperationContext(config, {
    agentId: toolCtx.agentId,
    sessionKey: toolCtx.sessionKey,
    workspaceDir: toolCtx.workspaceDir,
  });
}

function resolveScope(
  scope: string | undefined,
  config: MemoryPluginConfig,
  ctx: NonNullable<ReturnType<typeof buildOperationContext>>,
) {
  if (!scope) {
    return resolveDefaultScope(config, scopeVarsForContext(ctx));
  }
  if (!isScopeAllowed(scope, config, scopeVarsForContext(ctx))) {
    throw new Error(`scope not allowed: ${scope}`);
  }
  return scope;
}

function countTable(
  store: Awaited<ReturnType<MemxRuntimeManager["getStore"]>>,
  table: string,
): number {
  return Number(
    (store.client.prepare(`SELECT COUNT(*) AS count FROM ${table}`).get() as { count: number })
      .count ?? 0,
  );
}

export function createMemxTools(params: {
  toolCtx: ToolContext;
  config: MemoryPluginConfig;
  manager: MemxRuntimeManager;
  logger: MemxLogger;
}): AnyAgentTool[] | null {
  const opCtx = resolveToolContext(params.toolCtx, params.config);
  if (!opCtx) {
    params.logger.warn("memx: tool factory skipped because agent context is unavailable");
    return null;
  }

  const getStore = async () => params.manager.getStore(opCtx);

  const memoryStateGet: AnyAgentTool = {
    name: "memory_state_get",
    label: "Memory State Get",
    description: "Read current-state memory entries for this agent and scope.",
    parameters: Type.Object({
      key: Type.Optional(Type.String()),
      scope: Type.Optional(Type.String()),
    }),
    async execute(_toolCallId, rawParams) {
      try {
        const params = rawParams as Record<string, unknown>;
        const scope = resolveScope(readString(params, "scope"), opCtx.config, opCtx);
        const store = await getStore();
        const requestedKey = readString(params, "key");
        const states = requestedKey
          ? expandStateKeyAliases(requestedKey).flatMap((key) =>
              store.stateRepo.get({
                agentId: opCtx.agentId,
                scopes: [scope],
                key,
              }),
            )
          : store.stateRepo.get({
              agentId: opCtx.agentId,
              scopes: [scope],
            });
        return jsonToolResult({ states });
      } catch (error) {
        return jsonToolResult({ error: String(error) });
      }
    },
  };

  const memoryStateSet: AnyAgentTool = {
    name: "memory_state_set",
    label: "Memory State Set",
    description: "Write a session or durable current-state memory value.",
    parameters: Type.Object({
      key: Type.String(),
      valueText: Type.Optional(Type.String()),
      valueJson: Type.Optional(Type.String()),
      scope: Type.Optional(Type.String()),
      durable: Type.Optional(Type.Boolean()),
      ttlHours: Type.Optional(Type.Number()),
    }),
    async execute(_toolCallId, rawParams) {
      try {
        const params = rawParams as Record<string, unknown>;
        const scope = resolveScope(readString(params, "scope"), opCtx.config, opCtx);
        const key = readString(params, "key");
        if (!key) {
          return jsonToolResult({ error: "key required" });
        }
        const durable = readBoolean(params, "durable") === true;
        const store = await getStore();
        let value: Record<string, unknown>;
        const valueJson = readString(params, "valueJson");
        if (valueJson) {
          try {
            value = JSON.parse(valueJson) as Record<string, unknown>;
          } catch {
            return jsonToolResult({ error: "valueJson must be valid JSON" });
          }
        } else {
          value = { value: readString(params, "valueText") };
        }
        const updatedAt = nowIso();
        store.client.withTransaction(() => {
          store.stateRepo.upsert({
            key,
            valueJson: value,
            scope,
            agentId: opCtx.agentId,
            stateKind: durable ? "durable" : "session",
            confidence: 0.95,
            sourceRef: "tool:memory_state_set",
            updatedAt,
            expiresAt: durable
              ? undefined
              : store.stateRepo.createExpiry(
                  updatedAt,
                  readNumber(params, "ttlHours") ?? opCtx.config.stateTtlHours,
                ),
          });
          store.retrievalBackend.upsertDocs([
            {
              docId: `state:${key}`,
              docKind: "state",
              sourceId: key,
              scope,
              agentId: opCtx.agentId,
              text: `${key}: ${JSON.stringify(value)}`,
              metadataJson: {
                memxDocType: "state",
                key,
                scope,
                observedAt: updatedAt,
                confidence: 0.95,
              },
              createdAt: updatedAt,
              updatedAt,
            },
          ]);
        });
        return jsonToolResult({ ok: true, key, scope, durable });
      } catch (error) {
        return jsonToolResult({ error: String(error) });
      }
    },
  };

  const memoryFactUpsert: AnyAgentTool = {
    name: "memory_fact_upsert",
    label: "Memory Fact Upsert",
    description: "Create or update a durable fact.",
    parameters: Type.Object({
      subject: Type.String(),
      predicate: Type.String(),
      object: Type.Optional(Type.String()),
      scope: Type.Optional(Type.String()),
    }),
    async execute(_toolCallId, rawParams) {
      try {
        const params = rawParams as Record<string, unknown>;
        const subject = readString(params, "subject");
        const predicate = readString(params, "predicate");
        const object = readString(params, "object");
        if (!subject || !predicate) {
          return jsonToolResult({ error: "subject and predicate required" });
        }
        const scope = resolveScope(readString(params, "scope"), opCtx.config, opCtx);
        const store = await getStore();
        const updatedAt = nowIso();
        const factId = stableHash([opCtx.agentId, scope, subject, predicate, object ?? ""]);
        const objectValueJson = buildStoredFactObjectValueJson({
          subject,
          predicate,
          object,
        });
        store.client.withTransaction(() => {
          store.factRepo.upsert(
            {
              factId,
              canonicalSubject: normalizeName(subject),
              predicate: normalizeName(predicate),
              canonicalObject: object ? normalizeName(object) : undefined,
              objectValueJson,
              scope,
              agentId: opCtx.agentId,
              confidence: 0.95,
              status: "active",
              validFrom: updatedAt,
              createdAt: updatedAt,
              updatedAt,
              sourceRef: "tool:memory_fact_upsert",
              provenanceText: `${subject} ${predicate} ${object ?? ""}`.trim(),
            },
            "tool-upsert",
          );
          store.retrievalBackend.upsertDocs([
            {
              docId: `fact:${factId}`,
              docKind: "fact",
              sourceId: factId,
              scope,
              agentId: opCtx.agentId,
              text: `${subject} ${predicate}${object ? ` ${object}` : ""}`,
              metadataJson: {
                memxDocType: "fact",
                predicate: normalizeName(predicate),
                scope,
                observedAt: updatedAt,
                confidence: 0.95,
              },
              createdAt: updatedAt,
              updatedAt,
            },
          ]);
        });
        return jsonToolResult({ ok: true, factId, scope });
      } catch (error) {
        return jsonToolResult({ error: String(error) });
      }
    },
  };

  const memoryFactQuery: AnyAgentTool = {
    name: "memory_fact_query",
    label: "Memory Fact Query",
    description: "Search durable facts.",
    parameters: Type.Object({
      query: Type.Optional(Type.String()),
      predicate: Type.Optional(Type.String()),
      scope: Type.Optional(Type.String()),
      limit: Type.Optional(Type.Number()),
      includeHistorical: Type.Optional(Type.Boolean()),
    }),
    async execute(_toolCallId, rawParams) {
      try {
        const params = rawParams as Record<string, unknown>;
        const scope = resolveScope(readString(params, "scope"), opCtx.config, opCtx);
        const store = await getStore();
        const facts = store.factRepo.query({
          agentId: opCtx.agentId,
          scopes: [scope],
          text: readString(params, "query"),
          predicate: readString(params, "predicate"),
          limit: readNumber(params, "limit") ?? 8,
          includeHistorical: readBoolean(params, "includeHistorical") ?? false,
        });
        return jsonToolResult({ facts });
      } catch (error) {
        return jsonToolResult({ error: String(error) });
      }
    },
  };

  const memoryEventAppend: AnyAgentTool = {
    name: "memory_event_append",
    label: "Memory Event Append",
    description: "Append an episodic event to memory.",
    parameters: Type.Object({
      text: Type.String(),
      eventType: Type.Optional(Type.String()),
      scope: Type.Optional(Type.String()),
    }),
    async execute(_toolCallId, rawParams) {
      try {
        const params = rawParams as Record<string, unknown>;
        const text = readString(params, "text");
        if (!text) {
          return jsonToolResult({ error: "text required" });
        }
        const scope = resolveScope(readString(params, "scope"), opCtx.config, opCtx);
        const store = await getStore();
        const observedAt = nowIso();
        const eventId = stableHash([opCtx.agentId, scope, normalizeText(text), observedAt]);
        store.client.withTransaction(() => {
          store.eventRepo.append({
            eventId,
            agentId: opCtx.agentId,
            scope,
            eventType: readString(params, "eventType") ?? "manual_event",
            text,
            normalizedText: normalizeText(text),
            observedAt,
            sourceKind: "tool",
            sourceRef: "tool:memory_event_append",
            confidence: 0.95,
            metadataJson: {},
          });
          store.retrievalBackend.upsertDocs([
            {
              docId: `event:${eventId}`,
              docKind: "event",
              sourceId: eventId,
              scope,
              agentId: opCtx.agentId,
              text,
              metadataJson: { scope, observedAt, confidence: 0.95 },
              createdAt: observedAt,
              updatedAt: observedAt,
            },
          ]);
        });
        return jsonToolResult({ ok: true, eventId, scope });
      } catch (error) {
        return jsonToolResult({ error: String(error) });
      }
    },
  };

  const memoryEventSearch: AnyAgentTool = {
    name: "memory_event_search",
    label: "Memory Event Search",
    description: "Search episodic events.",
    parameters: Type.Object({
      query: Type.Optional(Type.String()),
      eventType: Type.Optional(Type.String()),
      scope: Type.Optional(Type.String()),
      limit: Type.Optional(Type.Number()),
    }),
    async execute(_toolCallId, rawParams) {
      try {
        const params = rawParams as Record<string, unknown>;
        const scope = resolveScope(readString(params, "scope"), opCtx.config, opCtx);
        const store = await getStore();
        const events = store.eventRepo.search({
          agentId: opCtx.agentId,
          scopes: [scope],
          text: readString(params, "query"),
          eventType: readString(params, "eventType"),
          limit: readNumber(params, "limit") ?? 8,
        });
        return jsonToolResult({ events });
      } catch (error) {
        return jsonToolResult({ error: String(error) });
      }
    },
  };

  const memoryGraphQuery: AnyAgentTool = {
    name: "memory_graph_query",
    label: "Memory Graph Query",
    description: "Run graph-aware retrieval for explanatory questions.",
    parameters: Type.Object({
      query: Type.String(),
      scope: Type.Optional(Type.String()),
      maxHops: Type.Optional(Type.Number()),
    }),
    async execute(_toolCallId, rawParams) {
      try {
        const params = rawParams as Record<string, unknown>;
        const scope = readString(params, "scope");
        const query = readString(params, "query");
        if (!query) {
          return jsonToolResult({ error: "query required" });
        }
        const scopedCtx = {
          ...opCtx,
          scopes: scope ? [resolveScope(scope, opCtx.config, opCtx)] : opCtx.scopes,
        };
        const store = await getStore();
        const bundle = await retrieveEvidence(store, scopedCtx, query);
        return jsonToolResult({
          routeType: bundle.routeType,
          graph: bundle.graph,
          diagnostics: bundle.diagnostics,
        });
      } catch (error) {
        return jsonToolResult({ error: String(error) });
      }
    },
  };

  const memoryRecall: AnyAgentTool = {
    name: "memory_recall",
    label: "Memory Recall",
    description:
      "Unified memX memory search across working state, facts, episodic history, and graph relations. Use this when automatic memory recall was insufficient. Prefer a short, focused query you generate yourself.",
    parameters: Type.Object({
      query: Type.String(),
      scope: Type.Optional(Type.String()),
      maxResults: Type.Optional(Type.Number()),
    }),
    async execute(_toolCallId, rawParams) {
      try {
        const params = rawParams as Record<string, unknown>;
        const originalQuery = readString(params, "query");
        if (!originalQuery) {
          return jsonToolResult({ error: "query required" });
        }
        const scopedCtx = {
          ...opCtx,
          scopes: readString(params, "scope")
            ? [resolveScope(readString(params, "scope"), opCtx.config, opCtx)]
            : opCtx.scopes,
        };
        const store = await getStore();
        const compiled = await compileQuery({
          query: originalQuery,
          ctx: scopedCtx,
          reasoner: store.reasoner,
        });
        const query = compiled.focusedQuery;
        const bundle = await retrieveEvidence(store, scopedCtx, originalQuery, query, {
          queryAnalysis: compiled,
        });
        const maxResults = Math.max(1, Math.min(readNumber(params, "maxResults") ?? 6, 12));
        return jsonToolResult({
          routeType: bundle.routeType,
          routeConfidence: bundle.routeConfidence,
          originalQuery,
          focusedQuery: query,
          compilerProvenance: compiled.compilerProvenance,
          behavioralGuidance: bundle.behavioralGuidance.slice(0, maxResults),
          states: bundle.states.slice(0, maxResults),
          tasks: bundle.tasks.slice(0, maxResults),
          facts: bundle.facts.slice(0, maxResults),
          events: bundle.events.slice(0, maxResults),
          graph: {
            paths: bundle.graph.paths.slice(0, maxResults),
            pathCandidates: bundle.graph.pathCandidates.slice(0, maxResults).map((path) => ({
              pathId: path.pathId,
              summary: path.summary,
              score: path.score,
              features: path.features,
              reasons: path.reasons,
            })),
            edges: bundle.graph.edges.slice(0, maxResults),
          },
          diagnostics: bundle.diagnostics,
          suggestedFollowUp:
            bundle.routeConfidence < 0.58 ||
            bundle.diagnostics.some((entry) => entry.includes("conflict"))
              ? "If this is still insufficient, refine the query and call memory_recall again with a shorter, more specific search phrase."
              : "Use the returned memory directly in your answer. Only call a narrower memory tool if you need exact details.",
        });
      } catch (error) {
        return jsonToolResult({ error: String(error) });
      }
    },
  };

  const memoryForget: AnyAgentTool = {
    name: "memory_forget",
    label: "Memory Forget",
    description: "Delete or tombstone memory records.",
    parameters: Type.Object({
      kind: Type.Optional(stringEnum(FORGET_KINDS)),
      id: Type.String(),
      scope: Type.Optional(Type.String()),
    }),
    async execute(_toolCallId, rawParams) {
      try {
        const params = rawParams as Record<string, unknown>;
        const id = readString(params, "id");
        if (!id) {
          return jsonToolResult({ error: "id required" });
        }
        const kind = readString(params, "kind") ?? "doc";
        const scope = readString(params, "scope");
        const scoped = scope ? resolveScope(scope, opCtx.config, opCtx) : undefined;
        const store = await getStore();
        let deleted = 0;
        switch (kind) {
          case "state":
            deleted = store.stateRepo.delete({ agentId: opCtx.agentId, scope: scoped, key: id });
            store.vectorRepo.deleteDocs([`state:${id}`]);
            break;
          case "fact":
            deleted = store.factRepo.markDeleted({
              agentId: opCtx.agentId,
              scope: scoped,
              factId: id,
            });
            store.vectorRepo.deleteDocs([`fact:${id}`]);
            break;
          case "event":
            deleted = store.eventRepo.delete({
              agentId: opCtx.agentId,
              scope: scoped,
              eventId: id,
            });
            store.vectorRepo.deleteDocs([`event:${id}`]);
            break;
          default:
            store.vectorRepo.deleteDocs([id]);
            deleted = 1;
            break;
        }
        return jsonToolResult({ ok: true, deleted, kind, id });
      } catch (error) {
        return jsonToolResult({ error: String(error) });
      }
    },
  };

  const memoryInspect: AnyAgentTool = {
    name: "memory_inspect",
    label: "Memory Inspect",
    description: "Inspect a stored memory document.",
    parameters: Type.Object({
      id: Type.String(),
    }),
    async execute(_toolCallId, rawParams) {
      try {
        const params = rawParams as Record<string, unknown>;
        const id = readString(params, "id");
        if (!id) {
          return jsonToolResult({ error: "id required" });
        }
        const store = await getStore();
        const doc =
          store.vectorRepo.getDoc(id) ??
          store.vectorRepo.getDoc(`fact:${id}`) ??
          store.vectorRepo.getDoc(`event:${id}`) ??
          store.vectorRepo.getDoc(`state:${id}`);
        return jsonToolResult(doc ?? { error: "not found", id });
      } catch (error) {
        return jsonToolResult({ error: String(error) });
      }
    },
  };

  const memoryStats: AnyAgentTool = {
    name: "memory_stats",
    label: "Memory Stats",
    description: "Show memory database statistics.",
    parameters: Type.Object({}),
    async execute() {
      try {
        const store = await getStore();
        return jsonToolResult({
          agentId: opCtx.agentId,
          dbPath: opCtx.dbPath,
          scopes: opCtx.scopes,
          taskCount: countTable(store, "conversation_tasks"),
          chunkCount: countTable(store, "conversation_chunks"),
          stateCount: countTable(store, "state_kv"),
          factCount: countTable(store, "facts"),
          eventCount: countTable(store, "episodic_events"),
          edgeCount: countTable(store, "graph_edges"),
          vectorDocCount: countTable(store, "vector_docs"),
        });
      } catch (error) {
        return jsonToolResult({ error: String(error) });
      }
    },
  };

  const memorySearch: AnyAgentTool = {
    name: "memory_search",
    label: "Memory Search",
    description:
      "Search stored memory and return snippet hits compatible with memory_search flows.",
    parameters: Type.Object({
      query: Type.String(),
      maxResults: Type.Optional(Type.Number()),
      minScore: Type.Optional(Type.Number()),
    }),
    async execute(_toolCallId, rawParams) {
      try {
        const params = rawParams as Record<string, unknown>;
        const query = readString(params, "query");
        if (!query) {
          return jsonToolResult({ error: "query required", disabled: true });
        }
        const store = await getStore();
        const hits = await store.retrievalBackend.hybridSearch({
          agentId: opCtx.agentId,
          scopes: opCtx.scopes,
          query,
          limit: readNumber(params, "maxResults") ?? 6,
        });
        const minScore = readNumber(params, "minScore") ?? 0;
        const results = hits
          .filter((hit) => hit.score >= minScore)
          .map((hit) => ({
            path: `memx://${hit.docId}`,
            startLine: 1,
            endLine: 1,
            score: hit.score,
            snippet: hit.text,
          }));
        return jsonToolResult({ results });
      } catch (error) {
        return jsonToolResult({ results: [], disabled: true, error: String(error) });
      }
    },
  };

  const memoryGet: AnyAgentTool = {
    name: "memory_get",
    label: "Memory Get",
    description: "Read a stored memory snippet by memx:// document path.",
    parameters: Type.Object({
      path: Type.String(),
    }),
    async execute(_toolCallId, rawParams) {
      try {
        const params = rawParams as Record<string, unknown>;
        const pathValue = readString(params, "path");
        if (!pathValue) {
          return jsonToolResult({ error: "path required", disabled: true });
        }
        const docId = pathValue.replace(/^memx:\/\//, "");
        const store = await getStore();
        const doc = store.vectorRepo.getDoc(docId);
        if (!doc) {
          return jsonToolResult({ path: pathValue, text: "", disabled: true, error: "not found" });
        }
        return jsonToolResult({
          path: pathValue,
          text: doc.text,
        });
      } catch (error) {
        return jsonToolResult({ text: "", disabled: true, error: String(error) });
      }
    },
  };

  const tools: AnyAgentTool[] = [
    memoryStateGet,
    memoryStateSet,
    memoryFactUpsert,
    memoryFactQuery,
    memoryEventAppend,
    memoryEventSearch,
    memoryGraphQuery,
    memoryForget,
    memoryInspect,
    memoryStats,
  ];
  if (params.config.advanced.enableExplicitRecallTool) {
    tools.unshift(memoryRecall);
  }
  if (params.config.advanced.enableCompatibilityMemoryTools) {
    tools.push(memorySearch, memoryGet);
  }
  return tools;
}
