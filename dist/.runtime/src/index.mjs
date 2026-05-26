import { MEMX_BRAND_NAME, MEMX_PLUGIN_ID } from "./identity.mjs";
import { nowIso, randomId, truncateText } from "./support.mjs";
import { semanticTextSimilarity } from "./pipeline/semantic/textSimilarity.mjs";
import { emitBackgroundRetrievalSignals } from "./pipeline/signalLedger.mjs";
import "./pipeline/constants.mjs";
import { compileQuery } from "./pipeline/queryCompiler.mjs";
import "./pipeline/turnSemanticCompiler.mjs";
import "./pipeline/reasoner.mjs";
import { formatMemxContextBlock, stripInjectedHistoricalBlock } from "./security/escaping.mjs";
import { resolveDefaultScope, scopeVarsForContext } from "./security/scopes.mjs";
import { MemxRuntimeManager, buildOperationContext } from "./runtime.mjs";
import { registerMemxCli } from "./cli/registerCli.mjs";
import { DEFAULT_MEMORY_CONFIG, memxConfigSchema } from "./config.mjs";
import { selectAgentEndMessagesForCapture } from "./pipeline/agentEndMessages.mjs";
import { readMessageText, stripInboundMetadata } from "./pipeline/messageText.mjs";
import { shouldSkipMemxForHeartbeat } from "./pipeline/heartbeatFilter.mjs";
import { sanitizeFocusedRecallQuery, summarizeBackgroundRecallBundle } from "./pipeline/retrieveTracing.mjs";
import { buildBackgroundRecallBundle, hasBackgroundRecallMaterial, retrieveEvidence } from "./pipeline/retrieve.mjs";
import { captureAgentEndTurn } from "./pipeline/turnCapture.mjs";
import { createMemxTools } from "./plugin-tools.mjs";
//#region src/index.ts
function shouldSuggestExplicitRecallTool(config) {
	return config.advanced.enableExplicitRecallTool && config.advanced.suggestExplicitRecallTool;
}
function resolveConfig(api) {
	const parsed = memxConfigSchema.safeParse?.(api.pluginConfig ?? {});
	if (parsed?.success) return parsed.data;
	if (parsed && !parsed.success) {
		const message = parsed.error?.issues?.map((issue) => `${issue.path.join(".")}: ${issue.message}`).join("; ");
		api.logger.warn(`memx: invalid config, using defaults (${message ?? "parse error"})`);
	}
	return DEFAULT_MEMORY_CONFIG;
}
function buildPromptSection(title, lines) {
	if (lines.length === 0) return "";
	return [`## ${title}`, ...lines].join("\n");
}
function truncatePromptLines(lines, maxChars) {
	if (maxChars <= 0 || lines.length === 0) return [];
	const selected = [];
	let used = 0;
	for (const line of lines) {
		const normalized = line.trim();
		if (!normalized) continue;
		const addition = (selected.length > 0 ? 1 : 0) + normalized.length;
		if (used + addition <= maxChars) {
			selected.push(normalized);
			used += addition;
			continue;
		}
		if (selected.length === 0) selected.push(truncateText(normalized, Math.max(24, maxChars)));
		break;
	}
	return selected;
}
function packPromptSections(params) {
	const budget = Math.max(400, Math.trunc(params.maxChars));
	const sections = params.sections.map((section) => {
		const titleCost = section.title.length + 5;
		return {
			...section,
			minLines: Math.max(0, section.minLines ?? 0),
			lines: truncatePromptLines(section.lines, Math.max(24, section.maxChars - titleCost))
		};
	}).filter((section) => section.lines.length > 0);
	const render = () => [params.instructionBlock, ...sections.map((section) => buildPromptSection(section.title, section.lines))].filter(Boolean).join("\n\n").trim();
	let combined = render();
	while (combined.length > budget) {
		const shrinkable = sections.map((section, index) => ({
			section,
			index
		})).filter(({ section }) => section.lines.length > section.minLines);
		if (shrinkable.length === 0) break;
		shrinkable.sort((left, right) => {
			if (left.section.priority !== right.section.priority) return left.section.priority - right.section.priority;
			return right.section.lines.length - left.section.lines.length;
		});
		sections[shrinkable[0].index].lines.pop();
		combined = render();
	}
	if (combined.length > budget) {
		const droppable = sections.map((section, index) => ({
			section,
			index
		})).filter(({ section }) => section.minLines === 0 && section.lines.length > 0);
		while (combined.length > budget && droppable.length > 0) {
			droppable.sort((left, right) => left.section.priority - right.section.priority);
			const next = droppable.shift();
			if (!next) break;
			sections[next.index].lines = [];
			combined = render();
		}
	}
	return combined.length > budget ? truncateText(combined, budget) : combined;
}
function hasRecallMaterial(bundle) {
	return bundle.evidencePackets.some((packet) => packet.injected && !packet.dropReason);
}
function hasStrongProtectedPromptEvidence(bundle) {
	return bundle.evidencePackets.some((packet) => packet.injected && !packet.dropReason && packet.coverage.filled && packet.coverage.confidence >= .5);
}
function shouldUseCompactEvidencePrompt(bundle) {
	return hasStrongProtectedPromptEvidence(bundle) && bundle.turnMode === "memory_qa" && (bundle.routeType === "factual" || bundle.routeType === "temporal" || bundle.routeType === "mixed");
}
function answerModeRuleLines(queryAnalysis, bundle, promptEvidenceItems) {
	if (promptEvidenceItems.length === 0) return [];
	if (queryAnalysis.answerMode === "count_aggregate") return ["- Count distinct relevant remembered events when multiple evidence lines apply; do not count duplicate source references twice."];
	return [];
}
function evidencePlanRuleLines(bundle) {
	const audit = bundle.evidencePlanAudit;
	if (!audit) return [];
	const missing = audit.slots.filter((slot) => !slot.filled);
	if (missing.length === 0) return [];
	if (audit.operation.type === "tailor_advice") {
		const contextualSlotFilled = audit.slots.some((slot) => slot.filled && slot.slotId !== "current_need");
		const missingHistoricalSlots = missing.filter((slot) => slot.slotId !== "current_need");
		if (contextualSlotFilled && missingHistoricalSlots.length === 0) return ["- The current user message supplies the current need. Use injected remembered resources, constraints, or prior advice as personalization context; do not reject them just because the exact current problem was not previously discussed."];
		if (missingHistoricalSlots.length > 0 && missingHistoricalSlots.length !== missing.length) return [`- Partial memory evidence slots: ${missingHistoricalSlots.map((slot) => slot.slotId).join(", ")}. The current user message supplies the current need; prefer injected answer evidence when present, and only say memory evidence is incomplete when the missing historical context is needed.`];
	}
	return [`- Partial memory evidence slots: ${missing.map((slot) => slot.slotId).join(", ")}. Prefer injected answer evidence when present; if only partial unrelated evidence is available, say memory evidence is incomplete instead of guessing.`];
}
function promptEvidencePromptItems(bundle, limit = 4) {
	const items = [];
	const seenLines = /* @__PURE__ */ new Set();
	const isExactDuplicate = (line) => {
		const key = line.replace(/\s+/gu, " ").trim().toLowerCase();
		if (seenLines.has(key)) return true;
		seenLines.add(key);
		return false;
	};
	const injectedPackets = bundle.evidencePackets.filter((packet) => packet.injected && !packet.dropReason).sort((left, right) => (right.grade?.finalScore ?? right.coverage.confidence) - (left.grade?.finalScore ?? left.coverage.confidence) || right.coverage.confidence - left.coverage.confidence || left.slotId.localeCompare(right.slotId));
	for (const packet of injectedPackets) {
		const line = evidencePacketPromptLine(packet);
		if (isExactDuplicate(line)) continue;
		items.push({
			packet,
			line
		});
		if (items.length >= limit) break;
	}
	if (items.length > 0) return items;
	return items;
}
function evidencePacketPromptLine(packet) {
	const displayLines = (packet.displayLines ?? []).filter((line) => line.trim().length > 0);
	if (displayLines.length > 0) return displayLines.map((line) => line.trim().startsWith("-") ? line.trim() : `- ${line.trim()}`).join("\n");
	const slot = `[slot:${packet.slotId}]`;
	const date = packet.resolvedDate ?? packet.observedAt?.slice(0, 10);
	const datePart = date ? ` [date:${date}]` : "";
	const quantity = typeof packet.quantityHint === "number" ? ` [quantity:${packet.quantityHint}${packet.unitHint ? ` ${packet.unitHint}` : ""}]` : "";
	const support = packet.supportingTexts.length > 0 ? ` | support: ${truncateText(packet.supportingTexts[0], 180)}` : "";
	return `- ${slot}${datePart}${quantity} ${truncateText(packet.primaryText, 420)}${support}`;
}
function formatProbeLogContext(ctx) {
	if (!ctx) return "";
	const rendered = [
		["agentId", ctx.agentId],
		["sessionKey", ctx.sessionKey],
		["runId", ctx.runId]
	].filter(([, value]) => typeof value === "string" && value.length > 0).map(([key, value]) => `${key}=${JSON.stringify(value)}`);
	return rendered.length > 0 ? ` ${rendered.join(" ")}` : "";
}
function _logPromptContext(logger, tag, prompt, probeContext) {
	const promptContext = formatMemxContextBlock(prompt);
	logger.info(`memx: PROBE prependContext [${tag}] chars=${promptContext.length}${formatProbeLogContext(probeContext)}\n--- BEGIN PREPENDCONTEXT ---\n${promptContext}\n--- END PREPENDCONTEXT ---`);
	return promptContext;
}
function buildBackgroundRecallPrompt(config, bundle, query, queryAnalysis, maxChars) {
	const instructions = [
		`## ${MEMX_BRAND_NAME} Memory`,
		`${MEMX_BRAND_NAME} is the active memory backend for this run.`,
		"Treat the following as remembered context from prior turns and use it directly when it helps answer the user.",
		"Use remembered preferences and current working context when they fit, but do not narrate memory internals unless the user explicitly asks.",
		"Auto-recall found background memory, but no final packet-qualified evidence was selected for prompt injection.",
		"Use the background memory below only when it directly fits the current turn."
	];
	const sections = [];
	if (bundle.behavioralGuidance.length > 0) sections.push({
		title: "Reply Guidance",
		lines: bundle.behavioralGuidance.slice(0, 4).map((line) => `- ${line}`),
		maxChars: 520,
		priority: 80,
		minLines: 1
	});
	const followUpLines = shouldSuggestExplicitRecallTool(config) ? [`- Auto-recall found no final evidence packet. If prior context still matters, call memory_recall with a focused query such as "${truncateText(queryAnalysis.focusedQuery || "user preferences", 120)}".`] : ["- Auto-recall found no final evidence packet. Answer conservatively from the current turn and the background memory above instead of speculating about missing prior context."];
	return packPromptSections({
		instructionBlock: instructions.join("\n"),
		sections: [...sections, {
			title: "If More Memory Is Needed",
			lines: followUpLines,
			maxChars: 260,
			priority: 4,
			minLines: 0
		}],
		maxChars: Math.max(700, maxChars)
	});
}
function buildMemxImplicitRecallPrompt(config, bundle, background, queryAnalysis, maxChars) {
	const instructionBlock = [
		`## ${MEMX_BRAND_NAME} Memory`,
		`${MEMX_BRAND_NAME} is the active memory backend for this run.`,
		"Treat the following as remembered context from prior turns and use it directly when it helps answer the user.",
		"Use remembered preferences and current working context when they fit, but do not narrate memory internals unless the user explicitly asks.",
		"Do not ask the user to restate remembered preferences or prior work context unless the current request clearly conflicts with them.",
		"Do not narrate memory searches, database inspection, plugin debugging, or workspace file inspection unless the user explicitly asks for those internals.",
		"Do not assume workspace MEMORY.md or memory/*.md files are the active memory backend unless the user explicitly asks about those files.",
		"Ignore bootstrap/setup scaffolding such as BOOTSTRAP.md, IDENTITY.md, USER.md, MEMORY.md, and memory/*.md when deciding what to remember or how to answer, unless the user explicitly asks about those files."
	].join("\n");
	const recallBudget = Math.max(220, bundle.budgetPlan?.totalPromptChars ?? maxChars);
	const followUpBudget = Math.max(120, Math.floor(recallBudget * .14));
	const compactEvidencePrompt = shouldUseCompactEvidencePrompt(bundle);
	const promptEvidenceItems = promptEvidencePromptItems(bundle, queryAnalysis.evidencePlan?.operation.type === "derive" || queryAnalysis.evidencePlan?.operation.type === "compare" || queryAnalysis.evidencePlan?.operation.type === "aggregate" ? 6 : compactEvidencePrompt ? 3 : 4);
	const promptEvidenceLines = promptEvidenceItems.map((item) => item.line);
	const answerRuleLines = [...answerModeRuleLines(queryAnalysis, bundle, promptEvidenceItems), ...evidencePlanRuleLines(bundle)];
	const sections = [{
		title: "Priority Evidence",
		lines: promptEvidenceLines,
		maxChars: Math.max(compactEvidencePrompt ? 720 : 360, Math.floor(recallBudget * .78)),
		priority: promptEvidenceLines.length > 0 ? 96 : 0,
		minLines: promptEvidenceLines.length > 0 ? 1 : 0
	}, {
		title: "Answer Rule",
		lines: answerRuleLines,
		maxChars: 360,
		priority: answerRuleLines.length > 0 ? 95 : 0,
		minLines: answerRuleLines.length > 0 ? 1 : 0
	}].filter((section) => section.lines.length > 0);
	const followUpSection = compactEvidencePrompt ? null : bundle.routeConfidence < .58 || bundle.diagnostics.some((entry) => entry.includes("conflict")) ? {
		title: "If More Memory Is Needed",
		lines: shouldSuggestExplicitRecallTool(config) ? [`- Auto-recall may be incomplete (${bundle.diagnostics[0] ?? "selection"}). If answering still depends on prior context, call memory_recall with a shorter, more specific query like "${truncateText(queryAnalysis.focusedQuery || "user preferences", 120)}".`, "- Use narrower tools only when you need exact details: memory_state_get, memory_fact_query, memory_event_search, or memory_graph_query."] : [`- Auto-recall may be incomplete (${bundle.diagnostics[0] ?? "selection"}). If uncertainty remains, answer conservatively and prefer already-grounded memory over speculation.`],
		maxChars: Math.max(120, followUpBudget),
		priority: 8,
		minLines: 0
	} : null;
	return packPromptSections({
		instructionBlock,
		sections: followUpSection ? [...sections, followUpSection] : sections,
		maxChars: Math.max(900, Math.min(maxChars + 800, instructionBlock.length + recallBudget + followUpBudget + 40))
	});
}
function buildRecallTraceLine(params) {
	const background = summarizeBackgroundRecallBundle(params.background);
	return [
		`query="${truncateText(params.rawQuery, 96)}"`,
		`full=${String(params.fullRecall)}`,
		`shouldRecall=${String(params.queryAnalysis.shouldRecall)}`,
		`primaryRoute=${params.queryAnalysis.primaryRoute ?? "unknown"}`,
		`turnMode=${params.queryAnalysis.turnMode}`,
		`shape=${params.queryAnalysis.queryShape.timeframe}/${params.queryAnalysis.queryShape.granularity}/${params.queryAnalysis.queryShape.referentialMode}/${params.queryAnalysis.queryShape.evidenceNeed}`,
		`focused="${truncateText(params.queryAnalysis.focusedQuery || params.rawQuery, 96)}"`,
		`entities=${params.queryAnalysis.queryEntities.length}`,
		`background=g${background.guidanceCount ?? 0}/s${Array.isArray(background.stateIds) ? background.stateIds.length : 0}/t${Array.isArray(background.taskIds) ? background.taskIds.length : 0}`,
		`granularity=${params.queryAnalysis.answerGranularity}/${params.queryAnalysis.evidenceFidelity}`,
		params.bundle ? `route=${params.bundle.routeType}:${params.bundle.routeConfidence.toFixed(2)} diag=${params.bundle.diagnostics[0] ?? "none"}` : "route=none"
	].join(" ");
}
function extractPromptQuery(event) {
	const prompt = event.prompt?.trim() ?? "";
	const promptCandidate = (() => {
		if (!prompt) return "";
		return stripInboundMetadata(stripInjectedHistoricalBlock(prompt));
	})();
	const messages = Array.isArray(event.messages) ? event.messages : [];
	const userCandidates = [];
	for (const message of messages) {
		if (!message || typeof message !== "object") continue;
		const entry = message;
		if (entry.role !== "user") continue;
		const text = stripInboundMetadata(stripInjectedHistoricalBlock(readMessageText(entry.content)));
		if (text.trim()) userCandidates.push(text.trim());
	}
	if (userCandidates.length === 0) return promptCandidate;
	if (!promptCandidate) return userCandidates[0] ?? "";
	const best = userCandidates.map((candidate) => {
		const similarity = semanticTextSimilarity(candidate, promptCandidate);
		const containment = candidate.includes(promptCandidate) || promptCandidate.includes(candidate) ? .95 : 0;
		return {
			candidate,
			score: Math.max(similarity, containment)
		};
	}).sort((left, right) => right.score - left.score)[0];
	return best && best.score >= .18 ? best.candidate : promptCandidate;
}
function createMemoryMemxPlugin() {
	return {
		id: MEMX_PLUGIN_ID,
		name: MEMX_BRAND_NAME,
		description: "Local-first multi-tier memory slot aligned with MemOS-style turn capture and recall",
		kind: "memory",
		configSchema: memxConfigSchema,
		register(api) {
			const config = resolveConfig(api);
			if (!config.enabled) {
				api.logger.info("memx: disabled");
				return;
			}
			const manager = new MemxRuntimeManager(api.logger);
			const warned = /* @__PURE__ */ new Set();
			const warnOnce = (key, message) => {
				if (warned.has(key)) return;
				warned.add(key);
				api.logger.warn(message);
			};
			api.registerTool((toolCtx) => createMemxTools({
				toolCtx,
				config,
				manager,
				logger: api.logger
			}), {
				names: [
					...config.advanced.enableExplicitRecallTool ? ["memory_recall"] : [],
					"memory_state_get",
					"memory_state_set",
					"memory_fact_upsert",
					"memory_fact_query",
					"memory_event_append",
					"memory_event_search",
					"memory_graph_query",
					"memory_forget",
					"memory_inspect",
					"memory_stats",
					...config.advanced.enableCompatibilityMemoryTools ? ["memory_search", "memory_get"] : []
				],
				optional: true
			});
			api.registerCli(({ program }) => {
				registerMemxCli({
					program,
					pluginConfig: config,
					appConfig: api.config,
					manager
				});
			}, { commands: ["memx"] });
			const recallHandler = async (event, ctx) => {
				if (!config.autoRecall) return;
				if (shouldSkipMemxForHeartbeat(event, ctx)) return;
				const opCtx = buildOperationContext(config, {
					agentId: ctx.agentId,
					sessionKey: ctx.sessionKey,
					workspaceDir: ctx.workspaceDir,
					channelId: ctx.channelId,
					runId: ctx.runId
				});
				if (!opCtx) {
					warnOnce("recall.no-agent", "memx: skipping recall without agentId");
					return;
				}
				try {
					const t0 = performance.now();
					const store = await manager.getStore(opCtx);
					const recallCtx = {
						...opCtx,
						readEpoch: store.client.currentMemoryEpoch(opCtx.agentId)
					};
					api.logger.debug?.(`memx: recall snapshot read_epoch=${recallCtx.readEpoch}`);
					const tStore = performance.now();
					const tFlush = performance.now();
					const rawQuery = extractPromptQuery(event);
					if (!rawQuery) return;
					const backgroundBundle = buildBackgroundRecallBundle(store, recallCtx);
					const tBackground = performance.now();
					const queryAnalysis = await compileQuery({
						query: rawQuery,
						ctx: recallCtx,
						reasoner: store.reasoner
					});
					queryAnalysis.focusedQuery = sanitizeFocusedRecallQuery(rawQuery, queryAnalysis.focusedQuery);
					const traceLine = buildRecallTraceLine({
						rawQuery,
						queryAnalysis,
						fullRecall: true,
						background: backgroundBundle
					});
					api.logger.debug?.(`memx: recall trace ${traceLine}`);
					const bundle = await retrieveEvidence(store, recallCtx, rawQuery, queryAnalysis.focusedQuery || rawQuery, {
						recallMode: "full",
						background: backgroundBundle,
						queryAnalysis
					});
					const promptBackground = backgroundBundle;
					const tFullRecall = performance.now();
					manager.rememberRecall(ctx.agentId, ctx.sessionKey, {
						chunkIds: bundle.recalledChunkIds,
						texts: bundle.recalledChunkTexts
					});
					const tEnd = performance.now();
					api.logger.info(`memx: TIMING recall total=${(tEnd - t0).toFixed(0)}ms store=${(tStore - t0).toFixed(0)}ms flush=${(tFlush - tStore).toFixed(0)}ms bg=${(tBackground - tFlush).toFixed(0)}ms compile=${(tFullRecall - tBackground).toFixed(0)}ms build=${(tEnd - tFullRecall).toFixed(0)}ms query="${rawQuery.slice(0, 60)}"`);
					if (!hasRecallMaterial(bundle)) {
						if (hasBackgroundRecallMaterial(backgroundBundle)) {
							emitBackgroundRetrievalSignals(store, recallCtx, {
								auditId: randomId("audit"),
								routeType: bundle.routeType,
								bundle: backgroundBundle
							});
							return { prependContext: _logPromptContext(api.logger, "full-no-material-bg", buildBackgroundRecallPrompt(config, promptBackground, rawQuery, queryAnalysis, config.maxInjectedChars), {
								agentId: ctx.agentId,
								sessionKey: ctx.sessionKey,
								runId: ctx.runId
							}) };
						}
						return;
					}
					return { prependContext: _logPromptContext(api.logger, "full-recall", buildMemxImplicitRecallPrompt(config, bundle, promptBackground, queryAnalysis, config.maxInjectedChars), {
						agentId: ctx.agentId,
						sessionKey: ctx.sessionKey,
						runId: ctx.runId
					}) };
				} catch (error) {
					const errorMsg = error instanceof Error ? error.message : String(error);
					const errorStack = error instanceof Error ? error.stack : void 0;
					api.logger.warn(`memx: recall failed: ${errorMsg}${errorStack ? `\n${errorStack}` : ""}`);
					return { prependContext: formatMemxContextBlock("[memx: recall unavailable due to internal error]") };
				}
			};
			api.on("before_prompt_build", recallHandler);
			api.on("agent_end", async (event, ctx) => {
				const allMessages = Array.isArray(event.messages) ? event.messages : [];
				if (shouldSkipMemxForHeartbeat({ messages: allMessages }, ctx)) return;
				if (allMessages.length === 0) return;
				const lastMsg = allMessages[allMessages.length - 1];
				if (lastMsg && typeof lastMsg.role === "string") {
					const content = typeof lastMsg.content === "string" ? lastMsg.content : JSON.stringify(lastMsg.content);
					api.logger.info(`memx: PROBE agent_end${formatProbeLogContext({
						agentId: ctx.agentId,
						sessionKey: ctx.sessionKey,
						runId: ctx.runId
					})} lastMsg role=${lastMsg.role} len=${content.length}\n--- BEGIN REPLY ---\n${content.slice(0, 2e3)}\n--- END REPLY ---`);
				}
				const opCtx = buildOperationContext(config, {
					agentId: ctx.agentId,
					sessionKey: ctx.sessionKey,
					workspaceDir: ctx.workspaceDir,
					channelId: ctx.channelId,
					runId: ctx.runId
				});
				if (!opCtx) {
					warnOnce("agent_end.no-agent", "memx: skipping capture without agentId");
					return;
				}
				try {
					const tA0 = performance.now();
					const store = await manager.getStore(opCtx);
					const tAStore = performance.now();
					const sessionKey = ctx.sessionKey ?? "default";
					const newMessages = selectAgentEndMessagesForCapture({
						messages: allMessages,
						ctx,
						cursors: manager,
						agentId: ctx.agentId,
						sessionKey
					});
					if (newMessages.length === 0) return;
					let captured = [];
					if (config.advanced.enableTurnScheduler) {
						const recall = manager.consumeRecall(ctx.agentId, ctx.sessionKey);
						const captureScope = resolveDefaultScope(config, scopeVarsForContext(opCtx));
						captured = captureAgentEndTurn({
							agentId: ctx.agentId,
							scope: captureScope,
							sessionKey,
							turnId: randomId("turn"),
							observedAt: nowIso(),
							messages: newMessages,
							recalledTexts: recall.texts
						});
						if (captured.length > 0) {
							await store.turnScheduler.enqueue(opCtx, captured);
							await store.turnScheduler.flush();
						}
						const tAFlush = performance.now();
						api.logger.info(`memx: TIMING agent_end flush total=${(tAFlush - tA0).toFixed(0)}ms store=${(tAStore - tA0).toFixed(0)}ms msgs=${newMessages.length} captured=${captured.length}`);
					}
					if (config.advanced.enableMaintenanceJobs) {
						const turnId = captured[0]?.turnId;
						const observedAt = captured.at(-1)?.observedAt ?? nowIso();
						if (turnId) await manager.recordMaintenanceTurn(opCtx, {
							store,
							turnId,
							observedAt
						});
					}
				} catch (error) {
					api.logger.warn(`memx: agent_end capture failed (${String(error)})`);
				}
			});
			api.registerService({
				id: MEMX_PLUGIN_ID,
				start() {
					api.logger.info("memx: initialized");
				},
				async stop() {
					await manager.closeAll();
				}
			});
		}
	};
}
const memoryMemxPlugin = createMemoryMemxPlugin();
//#endregion
export { createMemoryMemxPlugin, memoryMemxPlugin as default, evidencePlanRuleLines, extractPromptQuery };
