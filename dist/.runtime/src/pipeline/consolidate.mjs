import { normalizeText, nowIso, objectRecord, randomId, stableHash } from "../support.mjs";
import { refreshEntityProfileDocs } from "./entityProfile.mjs";
import { buildEntityMention, resolveEntityMention } from "./entityResolver.mjs";
import { snapshotMemoryLlmBudgetAudit } from "./llmBudgetAudit.mjs";
import { buildMaintenanceContractMetadata, summarizeMaintenanceContractDiagnostics, uniqueMaintenanceRefs } from "./maintenanceContract.mjs";
import { deriveStrategyHypotheses } from "./strategyHypotheses.mjs";
import { emitBeliefMaintenanceSignals } from "./signalLedger.mjs";
import { aggregateBeliefs } from "./beliefAggregation.mjs";
import { buildTaskSummaryEvidenceSet, taskSummaryMetadataFields, taskSummaryNeedsUpgrade, taskSummarySource, taskSummaryUpgradePriority } from "./taskSummary.mjs";
//#region src/pipeline/consolidate.ts
const MAX_CONSOLIDATION_CONFIRMATIONS_PER_KIND = 6;
const MAX_TASK_SUMMARY_UPGRADES_PER_RUN = 3;
const TASK_SUMMARY_TIMEOUT_MS = 1e4;
function olderThanDays(days, now = (/* @__PURE__ */ new Date()).toISOString()) {
	const date = new Date(now);
	if (!Number.isFinite(date.getTime())) {
		const fallback = /* @__PURE__ */ new Date();
		fallback.setUTCDate(fallback.getUTCDate() - days);
		return fallback.toISOString();
	}
	date.setUTCDate(date.getUTCDate() - days);
	return date.toISOString();
}
function shouldSkipLlmTaskSummaryUpgrade(ctx) {
	return typeof ctx.runId === "string" && ctx.runId.startsWith("lme-replay:");
}
function eventStructuredHints(event) {
	const structured = objectRecord(objectRecord(event.metadataJson)?.memxStructuredHints);
	return structured ? structured : null;
}
function sortPendingGroups(groups) {
	return [...groups].sort((left, right) => {
		if (right.events.length !== left.events.length) return right.events.length - left.events.length;
		const rightObservedAt = right.events.at(-1)?.observedAt ?? "";
		const leftObservedAt = left.events.at(-1)?.observedAt ?? "";
		return rightObservedAt.localeCompare(leftObservedAt);
	});
}
function summarizeMaintenanceCalls(audit, labels) {
	return (audit?.calls ?? []).filter((entry) => entry.stage === "maintenance_async" && labels.includes(entry.label)).reduce((summary, entry) => ({
		callCount: summary.callCount + 1,
		estimatedPromptTokens: summary.estimatedPromptTokens + (entry.estimatedPromptTokens ?? 0),
		estimatedCompletionTokens: summary.estimatedCompletionTokens + (entry.estimatedCompletionTokens ?? 0),
		estimatedTotalTokens: summary.estimatedTotalTokens + (entry.estimatedTotalTokens ?? 0),
		elapsedMs: summary.elapsedMs + (entry.elapsedMs ?? 0)
	}), {
		callCount: 0,
		estimatedPromptTokens: 0,
		estimatedCompletionTokens: 0,
		estimatedTotalTokens: 0,
		elapsedMs: 0
	});
}
function structuredEventEvidenceSummary(event) {
	const metadata = objectRecord(event.metadataJson);
	const temporalFacet = objectRecord(metadata?.memxTemporalFacet);
	return typeof temporalFacet?.summary === "string" && temporalFacet.summary.trim() || typeof metadata?.memxStructuredSummary === "string" && metadata.memxStructuredSummary.trim() || `${event.eventType} @ ${event.observedAt}`;
}
function latestObservedAt(events) {
	return events.reduce((latest, event) => {
		if (!latest) return event.observedAt;
		return event.observedAt > latest ? event.observedAt : latest;
	}, void 0);
}
function missingLlmBatchDecision() {
	return "defer";
}
function factConfirmationKey(preference) {
	return `${preference.predicate}:${normalizeText(preference.object)}`;
}
function relationConfirmationKey(relation) {
	return `${normalizeText(relation.subject)}:${relation.predicate}:${normalizeText(relation.object)}`;
}
function collectStructuredConfirmations(params) {
	const touchedFactKeys = /* @__PURE__ */ new Set();
	const touchedRelationKeys = /* @__PURE__ */ new Set();
	for (const event of params.deltaEvents) {
		const hints = eventStructuredHints(event);
		if (!hints) continue;
		if (params.ctx.config.advanced.enableFactPromotion && hints.preference) touchedFactKeys.add(factConfirmationKey(hints.preference));
		const relations = params.ctx.config.advanced.enableGraphPromotion && Array.isArray(hints.relations) ? hints.relations : hints.relation ? [hints.relation] : [];
		for (const relation of relations) touchedRelationKeys.add(relationConfirmationKey(relation));
	}
	const allowFactGroups = !params.batchMode || touchedFactKeys.size > 0;
	const allowRelationGroups = !params.batchMode || touchedRelationKeys.size > 0;
	const factGroups = /* @__PURE__ */ new Map();
	const relationGroups = /* @__PURE__ */ new Map();
	for (const event of params.recentEvents) {
		const hints = eventStructuredHints(event);
		if (!hints) continue;
		const preferences = allowFactGroups && params.ctx.config.advanced.enableFactPromotion && hints.preference ? [hints.preference] : [];
		for (const preference of preferences) {
			const key = factConfirmationKey(preference);
			if (params.batchMode && touchedFactKeys.size > 0 && !touchedFactKeys.has(key)) continue;
			const existing = factGroups.get(key);
			if (existing) {
				existing.events.push(event);
				existing.text = structuredEventEvidenceSummary(event);
			} else factGroups.set(key, {
				text: structuredEventEvidenceSummary(event),
				events: [event],
				preference
			});
		}
		const relations = allowRelationGroups && params.ctx.config.advanced.enableGraphPromotion && Array.isArray(hints.relations) ? hints.relations : allowRelationGroups && hints.relation ? [hints.relation] : [];
		for (const relation of relations) {
			const key = relationConfirmationKey(relation);
			if (params.batchMode && touchedRelationKeys.size > 0 && !touchedRelationKeys.has(key)) continue;
			const existing = relationGroups.get(key);
			if (existing) {
				existing.events.push(event);
				existing.text = structuredEventEvidenceSummary(event);
			} else relationGroups.set(key, {
				text: structuredEventEvidenceSummary(event),
				events: [event],
				relation
			});
		}
	}
	const pendingFacts = sortPendingGroups([...factGroups.values()].filter((entry) => entry.events.length >= 2));
	const pendingRelations = sortPendingGroups([...relationGroups.values()].filter((entry) => entry.events.length >= 2));
	return {
		pendingFacts,
		pendingRelations,
		touchedFactKeys,
		touchedRelationKeys,
		skippedReasons: [...pendingFacts.length === 0 && pendingRelations.length === 0 ? ["no_structured_confirmation_candidates"] : [], ...params.batchMode && touchedFactKeys.size === 0 && touchedRelationKeys.size === 0 ? ["no_touched_structured_keys_in_delta"] : []]
	};
}
function syncSupersededFactBeliefs(store, ctx) {
	let beliefsSuperseded = 0;
	const factBeliefs = store.beliefRepo.listByAgent({ agentId: ctx.agentId }).filter((belief) => belief.memoryKind === "fact" && belief.contentRef && (belief.stage === "active" || belief.stage === "probationary"));
	for (const belief of factBeliefs) {
		const matchingFact = store.factRepo.findBySemanticKey({
			agentId: ctx.agentId,
			scope: belief.scope,
			canonicalSubject: belief.semanticKey.split(":")[0] ?? "",
			predicate: belief.semanticKey.split(":").slice(1).join(":"),
			includeHistorical: true
		}).find((fact) => fact.factId === belief.contentRef);
		if (matchingFact && matchingFact.status === "superseded") {
			store.beliefRepo.markSupersededByContentRef({
				agentId: ctx.agentId,
				contentRef: belief.contentRef,
				updatedAt: ctx.now
			});
			beliefsSuperseded += 1;
		}
	}
	return beliefsSuperseded;
}
async function runConsolidation(store, ctx, options = {}) {
	const runStartedAt = nowIso();
	const runId = store.auditRepo.startMaintenance({
		agentId: ctx.agentId,
		sessionKey: options.batch?.sessionKey ?? ctx.sessionKey,
		jobType: "consolidate",
		stats: {},
		startedAt: runStartedAt
	});
	const stats = {
		...options.batch ? { batch: {
			...options.batch,
			delta: {
				eventsConsidered: 0,
				tasksConsidered: 0,
				lowerWatermarks: options.batch.lowerWatermarks,
				upperWatermarks: options.batch.upperWatermarks
			}
		} } : {},
		expiredStates: 0,
		promotedFacts: 0,
		promotedEdges: 0,
		promotedStates: 0,
		prunedEdges: 0,
		beliefsUpserted: 0,
		beliefSignalsProcessed: 0,
		beliefsNeedingReevaluation: 0,
		beliefsPromoted: 0,
		beliefsDemoted: 0,
		beliefsQuarantined: 0,
		beliefsSuperseded: 0,
		beliefsDecaying: 0,
		strategiesUpserted: 0,
		activeStrategies: 0,
		candidateStrategies: 0,
		quarantinedStrategies: 0,
		stageTimingsMs: {
			hygiene: 0,
			beliefAggregation: 0,
			semanticUpgrade: 0,
			structureDerivation: 0,
			total: 0
		},
		budgets: {
			maxConsolidationConfirmationsPerKind: MAX_CONSOLIDATION_CONFIRMATIONS_PER_KIND,
			eligibleFactConfirmations: 0,
			eligibleRelationConfirmations: 0,
			attemptedFactConfirmations: 0,
			attemptedRelationConfirmations: 0,
			maxTaskSummaryUpgrades: MAX_TASK_SUMMARY_UPGRADES_PER_RUN,
			eligibleTaskSummaryUpgrades: 0,
			attemptedTaskSummaryUpgrades: 0,
			skippedTaskSummaryUpgrades: 0,
			taskSummaryTimeoutMs: TASK_SUMMARY_TIMEOUT_MS,
			maxStrategyEmbeddingCandidates: 0,
			strategyEmbeddingCandidatesEmbedded: 0
		},
		semanticUpgrade: {
			factConfirmDeferred: 0,
			relationConfirmDeferred: 0,
			factConfirmRejected: 0,
			relationConfirmRejected: 0,
			confirmFallbackTriggered: false,
			taskSummariesUpgraded: 0,
			taskSummaryUpgradeFailures: 0,
			taskSummaryUpgradeTimedOut: 0,
			skippedReasons: [],
			confirmationGroups: {
				factGroupKeys: [],
				relationGroupKeys: [],
				selectedFactGroupKeys: [],
				selectedRelationGroupKeys: []
			},
			llm: {
				taskSummaryUpgrade: {
					callCount: 0,
					estimatedPromptTokens: 0,
					estimatedCompletionTokens: 0,
					estimatedTotalTokens: 0,
					elapsedMs: 0
				},
				consolidationConfirm: {
					callCount: 0,
					estimatedPromptTokens: 0,
					estimatedCompletionTokens: 0,
					estimatedTotalTokens: 0,
					elapsedMs: 0
				}
			}
		},
		authoritySources: {
			hygiene: ["deterministic_aggregated"],
			beliefAggregation: ["deterministic_aggregated"],
			semanticUpgrade: ["llm_confirmed"],
			structureDerivation: ["deterministic_aggregated"]
		},
		semanticSources: {
			hygiene: ["deterministic_lifecycle"],
			beliefAggregation: ["deterministic_lifecycle"],
			semanticUpgrade: ["upstream_structured", "llm_upgrade"],
			structureDerivation: ["upstream_structured"]
		}
	};
	const promotedMaintenanceMetadata = [];
	const totalStart = performance.now();
	try {
		const hygieneStart = performance.now();
		stats.expiredStates = store.stateRepo.expireSessionStates(ctx.agentId, ctx.now);
		const sessionKey = options.batch?.sessionKey;
		const historySince = olderThanDays(Math.max(7, ctx.config.episodicDedupWindowDays), ctx.now);
		const boundedHistorySince = options.batch ? void 0 : historySince;
		const recentEvents = store.eventRepo.search({
			agentId: ctx.agentId,
			scopes: ctx.scopes,
			...sessionKey ? { sessionKey } : {},
			limit: 96,
			...boundedHistorySince ? { since: boundedHistorySince } : {},
			...options.batch?.upperWatermarks.event ? { until: options.batch.upperWatermarks.event } : {}
		});
		const deltaEvents = options.batch ? store.eventRepo.search({
			agentId: ctx.agentId,
			scopes: ctx.scopes,
			...sessionKey ? { sessionKey } : {},
			limit: 96,
			...boundedHistorySince ? { since: boundedHistorySince } : {},
			...options.batch.lowerWatermarks.event ? { after: options.batch.lowerWatermarks.event } : {},
			...options.batch.upperWatermarks.event ? { until: options.batch.upperWatermarks.event } : {}
		}) : recentEvents;
		if (stats.batch) stats.batch.delta.eventsConsidered = deltaEvents.length;
		const confirmationCollection = collectStructuredConfirmations({
			ctx,
			recentEvents,
			deltaEvents,
			batchMode: Boolean(options.batch)
		});
		const pendingFacts = confirmationCollection.pendingFacts;
		const pendingRelations = confirmationCollection.pendingRelations;
		stats.semanticUpgrade.skippedReasons.push(...confirmationCollection.skippedReasons);
		stats.semanticUpgrade.confirmationGroups.factGroupKeys = pendingFacts.map((entry) => factConfirmationKey(entry.preference));
		stats.semanticUpgrade.confirmationGroups.relationGroupKeys = pendingRelations.map((entry) => relationConfirmationKey(entry.relation));
		stats.budgets.eligibleFactConfirmations = pendingFacts.length;
		stats.budgets.eligibleRelationConfirmations = pendingRelations.length;
		stats.stageTimingsMs.hygiene += performance.now() - hygieneStart;
		const semanticUpgradeStart = performance.now();
		const selectedFacts = pendingFacts.slice(0, MAX_CONSOLIDATION_CONFIRMATIONS_PER_KIND);
		const selectedRelations = pendingRelations.slice(0, MAX_CONSOLIDATION_CONFIRMATIONS_PER_KIND);
		stats.semanticUpgrade.confirmationGroups.selectedFactGroupKeys = selectedFacts.map((entry) => factConfirmationKey(entry.preference));
		stats.semanticUpgrade.confirmationGroups.selectedRelationGroupKeys = selectedRelations.map((entry) => relationConfirmationKey(entry.relation));
		stats.budgets.attemptedFactConfirmations = selectedFacts.length;
		stats.budgets.attemptedRelationConfirmations = selectedRelations.length;
		const confirmationItems = [...selectedFacts.map((entry) => ({
			id: `fact:${factConfirmationKey(entry.preference)}`,
			kind: "fact",
			predicate: entry.preference.predicate,
			object: entry.preference.object,
			supportCount: entry.events.length,
			latestObservedAt: latestObservedAt(entry.events),
			structuredSummaries: [...new Set(entry.events.slice(0, 4).map((event) => structuredEventEvidenceSummary(event)))],
			sourceRefs: entry.events.slice(0, 6).map((event) => event.sourceRef)
		})), ...selectedRelations.map((entry) => ({
			id: `relation:${relationConfirmationKey(entry.relation)}`,
			kind: "relation",
			predicate: entry.relation.predicate,
			object: entry.relation.object,
			subject: entry.relation.subject,
			supportCount: entry.events.length,
			latestObservedAt: latestObservedAt(entry.events),
			structuredSummaries: [...new Set(entry.events.slice(0, 4).map((event) => structuredEventEvidenceSummary(event)))],
			sourceRefs: entry.events.slice(0, 6).map((event) => event.sourceRef)
		}))];
		const confirmationDecisions = confirmationItems.length ? await store.reasoner.confirmConsolidationBatch(confirmationItems, {
			stage: "maintenance_async",
			audit: ctx.llmBudgetAudit
		}) : null;
		if (confirmationItems.length > 0 && !confirmationDecisions) stats.semanticUpgrade.confirmFallbackTriggered = true;
		const factConfirmResults = selectedFacts.map((entry) => {
			const id = `fact:${factConfirmationKey(entry.preference)}`;
			return {
				entry,
				decision: confirmationDecisions?.get(id)?.decision ?? missingLlmBatchDecision()
			};
		});
		const relationConfirmResults = selectedRelations.map((entry) => {
			const id = `relation:${relationConfirmationKey(entry.relation)}`;
			return {
				entry,
				decision: confirmationDecisions?.get(id)?.decision ?? missingLlmBatchDecision()
			};
		});
		if (!shouldSkipLlmTaskSummaryUpgrade(ctx)) {
			const rankedTasks = store.taskRepo.listActive({
				agentId: ctx.agentId,
				scopes: ctx.scopes,
				...sessionKey ? { sessionKey } : {},
				limit: 12
			}).map((task) => {
				const chunks = store.chunkRepo.listByTask(task.taskId);
				const evidence = buildTaskSummaryEvidenceSet({
					eventRepo: store.eventRepo,
					task,
					chunks,
					now: ctx.now,
					readEpoch: ctx.readEpoch
				});
				return {
					task,
					chunks,
					evidence,
					needsUpgrade: taskSummaryNeedsUpgrade({
						task,
						evidence,
						now: ctx.now
					}),
					priority: taskSummaryUpgradePriority({
						task,
						evidence
					}),
					summarySource: taskSummarySource(task.metadataJson),
					touchedByDelta: !options.batch?.lowerWatermarks.task || task.updatedAt > options.batch.lowerWatermarks.task
				};
			}).filter((entry) => entry.needsUpgrade && (!options.batch || entry.touchedByDelta || entry.summarySource === "heuristic_fallback" || !entry.summarySource)).sort((left, right) => {
				if (right.priority !== left.priority) return right.priority - left.priority;
				return right.task.updatedAt.localeCompare(left.task.updatedAt);
			});
			if (stats.batch) stats.batch.delta.tasksConsidered = rankedTasks.length;
			stats.budgets.eligibleTaskSummaryUpgrades = rankedTasks.length;
			const selectedTasks = rankedTasks.slice(0, MAX_TASK_SUMMARY_UPGRADES_PER_RUN);
			stats.budgets.attemptedTaskSummaryUpgrades = selectedTasks.length;
			stats.budgets.skippedTaskSummaryUpgrades = Math.max(0, rankedTasks.length - selectedTasks.length);
			if (selectedTasks.length > 0) {
				let didTimeout = false;
				try {
					const llmSummaries = await Promise.race([store.reasoner.summarizeTaskEvidenceBatch(selectedTasks.map((entry) => entry.evidence), {
						stage: "maintenance_async",
						audit: ctx.llmBudgetAudit
					}), new Promise((resolve) => setTimeout(() => {
						didTimeout = true;
						resolve(null);
					}, TASK_SUMMARY_TIMEOUT_MS))]);
					if (!llmSummaries) if (didTimeout) stats.semanticUpgrade.taskSummaryUpgradeTimedOut += 1;
					else stats.semanticUpgrade.taskSummaryUpgradeFailures += selectedTasks.length;
					else for (const { task, evidence } of selectedTasks) {
						const llmSummary = llmSummaries.get(task.taskId);
						if (!llmSummary) {
							stats.semanticUpgrade.taskSummaryUpgradeFailures += 1;
							continue;
						}
						store.taskRepo.update(task.taskId, {
							...task,
							title: llmSummary.title,
							summary: llmSummary.summary,
							metadataJson: {
								...task.metadataJson,
								...llmSummary.metadataJson,
								...taskSummaryMetadataFields({
									summarySource: "maintenance_llm",
									summaryQuality: "stable",
									summaryBasisFingerprint: evidence.fingerprint,
									observedAt: ctx.now,
									...evidence.compilerTaskSummary ? {
										compilerTaskSummary: evidence.compilerTaskSummary.summary,
										compilerTaskSummaryConfidence: evidence.compilerTaskSummary.confidence
									} : {}
								})
							},
							updatedAt: ctx.now
						});
						stats.semanticUpgrade.taskSummariesUpgraded += 1;
					}
				} catch {
					stats.semanticUpgrade.taskSummaryUpgradeFailures += selectedTasks.length;
				}
			}
		}
		stats.stageTimingsMs.semanticUpgrade += performance.now() - semanticUpgradeStart;
		const hygieneFinalizeStart = performance.now();
		for (const { entry, decision } of factConfirmResults) {
			if (decision === "defer") {
				stats.semanticUpgrade.factConfirmDeferred += 1;
				continue;
			}
			if (decision === "reject") {
				stats.semanticUpgrade.factConfirmRejected += 1;
				continue;
			}
			const factId = stableHash([
				ctx.agentId,
				entry.events[0]?.scope,
				"user",
				entry.preference.predicate,
				entry.preference.object
			]);
			const sourceRefs = uniqueMaintenanceRefs(entry.events.map((event) => event.sourceRef));
			const firstSourceRef = sourceRefs[0] ?? randomId("event");
			const promotionEpoch = store.client.nextMemoryEpoch(ctx.agentId, ctx.now);
			const factMetadata = buildMaintenanceContractMetadata({
				existing: {
					canonicalObjectText: entry.preference.object,
					supportCount: entry.events.length
				},
				sourceRef: firstSourceRef,
				supportContentRefs: sourceRefs,
				supportRefs: sourceRefs,
				derivedFromIds: sourceRefs,
				semanticSource: "upstream_structured",
				semanticSources: ["upstream_structured", "llm_upgrade"],
				authoritySource: "llm_confirmed",
				generatedFrom: "structured_event_confirmation",
				recallLayer: "fact",
				answerEligibleByDefault: true,
				materializedEpoch: promotionEpoch
			});
			store.factRepo.upsert({
				factId,
				canonicalSubject: "user",
				predicate: entry.preference.predicate,
				canonicalObject: normalizeText(entry.preference.object),
				objectValueJson: factMetadata,
				scope: entry.events[0]?.scope ?? ctx.scopes[0] ?? `agent:${ctx.agentId}`,
				agentId: ctx.agentId,
				confidence: .78,
				status: "active",
				validFrom: entry.events[0]?.observedAt,
				createdAt: entry.events[0]?.observedAt ?? ctx.now,
				updatedAt: ctx.now,
				materializedEpoch: promotionEpoch,
				sourceRef: firstSourceRef,
				provenanceText: entry.events[0]?.text ?? entry.text
			}, "promotion-from-structured-events");
			promotedMaintenanceMetadata.push(factMetadata);
			stats.promotedFacts += 1;
		}
		for (const { entry, decision } of relationConfirmResults) {
			if (decision === "defer") {
				stats.semanticUpgrade.relationConfirmDeferred += 1;
				continue;
			}
			if (decision === "reject") {
				stats.semanticUpgrade.relationConfirmRejected += 1;
				continue;
			}
			const scope = entry.events[0]?.scope ?? ctx.scopes[0] ?? `agent:${ctx.agentId}`;
			const sourceRefs = uniqueMaintenanceRefs(entry.events.map((event) => event.sourceRef));
			const evidenceRef = sourceRefs[0] ?? randomId("event");
			const subjectResolution = resolveEntityMention(store, ctx, buildEntityMention({
				ctx,
				scope,
				rawText: entry.relation.subject,
				proposedType: "unknown",
				semanticRole: "subject",
				sourceRef: evidenceRef,
				supportText: entry.events[0]?.text ?? entry.text,
				observedAt: entry.events[0]?.observedAt ?? ctx.now,
				sessionKey: entry.events[0]?.sessionKey,
				metadataJson: {
					generatedFrom: "structured_relation_confirmation",
					relationType: entry.relation.predicate,
					relationRole: "subject",
					weakProposedType: true
				}
			}));
			const objectResolution = resolveEntityMention(store, ctx, buildEntityMention({
				ctx,
				scope,
				rawText: entry.relation.object,
				proposedType: "unknown",
				semanticRole: "object",
				sourceRef: evidenceRef,
				supportText: entry.events[0]?.text ?? entry.text,
				observedAt: entry.events[0]?.observedAt ?? ctx.now,
				sessionKey: entry.events[0]?.sessionKey,
				metadataJson: {
					generatedFrom: "structured_relation_confirmation",
					relationType: entry.relation.predicate,
					relationRole: "object",
					weakProposedType: true
				}
			}));
			if (subjectResolution.method === "uncertain" || objectResolution.method === "uncertain") {
				stats.semanticUpgrade.skippedReasons.push(`graph_endpoint_uncertain:${entry.relation.predicate}`);
				continue;
			}
			const subjectEntity = subjectResolution.entity;
			const objectEntity = objectResolution.entity;
			const promotionEpoch = store.client.nextMemoryEpoch(ctx.agentId, ctx.now);
			const edgeMetadata = buildMaintenanceContractMetadata({
				existing: {
					supportCount: entry.events.length,
					relationType: entry.relation.predicate,
					sourceName: entry.relation.subject,
					targetName: entry.relation.object,
					sourceMentionId: subjectResolution.mention.mentionId,
					targetMentionId: objectResolution.mention.mentionId,
					sourceResolutionMethod: subjectResolution.method,
					targetResolutionMethod: objectResolution.method,
					sourceResolutionConfidence: subjectResolution.confidence,
					targetResolutionConfidence: objectResolution.confidence
				},
				sourceRef: evidenceRef,
				supportContentRefs: sourceRefs,
				supportRefs: sourceRefs,
				derivedFromIds: sourceRefs,
				semanticSource: "upstream_structured",
				semanticSources: ["upstream_structured", "llm_upgrade"],
				authoritySource: "llm_confirmed",
				generatedFrom: "structured_relation_confirmation",
				recallLayer: "graph",
				answerEligibleByDefault: true,
				materializedEpoch: promotionEpoch
			});
			if (store.graphRepo.upsertEdge({
				edgeId: stableHash([
					ctx.agentId,
					scope,
					subjectEntity.entityId,
					entry.relation.predicate,
					objectEntity.entityId
				]),
				srcEntityId: subjectEntity.entityId,
				relType: entry.relation.predicate,
				dstEntityId: objectEntity.entityId,
				scope,
				agentId: ctx.agentId,
				confidence: .76,
				evidenceRef,
				validFrom: entry.events[0]?.observedAt,
				createdAt: entry.events[0]?.observedAt ?? ctx.now,
				updatedAt: ctx.now,
				sourceKind: "extracted",
				materializedEpoch: promotionEpoch,
				metadataJson: edgeMetadata
			}).action === "created") stats.promotedEdges += 1;
			promotedMaintenanceMetadata.push(edgeMetadata);
			refreshEntityProfileDocs(store, ctx, [subjectEntity.entityId, objectEntity.entityId]);
		}
		stats.prunedEdges = store.graphRepo.pruneLowConfidence({
			agentId: ctx.agentId,
			olderThan: olderThanDays(30, ctx.now),
			maxConfidence: .35
		});
		stats.stageTimingsMs.hygiene += performance.now() - hygieneFinalizeStart;
		const beliefStart = performance.now();
		emitBeliefMaintenanceSignals(store, ctx);
		const beliefStats = aggregateBeliefs(store, ctx, { ...options.batch ? {
			signalWindow: {
				...sessionKey ? { sessionKey } : {},
				...options.batch.lowerWatermarks.signal ? { after: options.batch.lowerWatermarks.signal } : {},
				...options.batch.upperWatermarks.signal ? { until: options.batch.upperWatermarks.signal } : {}
			},
			scopes: ctx.scopes
		} : {} });
		stats.beliefsUpserted = beliefStats.beliefsUpserted;
		stats.beliefSignalsProcessed = beliefStats.signalsProcessed;
		stats.beliefsNeedingReevaluation = beliefStats.beliefsNeedingReevaluation;
		stats.beliefsPromoted = beliefStats.beliefsPromoted;
		stats.beliefsDemoted = beliefStats.beliefsDemoted;
		stats.beliefsQuarantined = beliefStats.beliefsQuarantined;
		stats.beliefsSuperseded = beliefStats.beliefsSuperseded;
		stats.beliefsDecaying = beliefStats.beliefsDecaying;
		stats.beliefsSuperseded += syncSupersededFactBeliefs(store, ctx);
		stats.stageTimingsMs.beliefAggregation += performance.now() - beliefStart;
		const structureStart = performance.now();
		const strategyStats = await deriveStrategyHypotheses(store, ctx);
		stats.strategiesUpserted = strategyStats.strategiesUpserted;
		stats.activeStrategies = strategyStats.activeStrategies;
		stats.candidateStrategies = strategyStats.candidateStrategies;
		stats.quarantinedStrategies = strategyStats.quarantinedStrategies;
		stats.budgets.maxStrategyEmbeddingCandidates = strategyStats.embeddingBudget;
		stats.budgets.strategyEmbeddingCandidatesEmbedded = strategyStats.embeddingCandidatesEmbedded;
		stats.authoritySources.structureDerivation = [strategyStats.authoritySource];
		stats.semanticSources.structureDerivation = strategyStats.semanticSources.length > 0 ? strategyStats.semanticSources : ["deterministic_lifecycle"];
		stats.stageTimingsMs.structureDerivation += performance.now() - structureStart;
		stats.stageTimingsMs.total = performance.now() - totalStart;
		const maintenanceContractDiagnostics = summarizeMaintenanceContractDiagnostics(promotedMaintenanceMetadata);
		stats.maintenanceContractDiagnostics = maintenanceContractDiagnostics;
		stats.recallFacingDiagnostics = {
			recallVisible: maintenanceContractDiagnostics.recallVisibleCount > 0,
			answerEligibleByDefault: maintenanceContractDiagnostics.answerEligibleByDefaultCount > 0,
			sourceRefsForExpansion: maintenanceContractDiagnostics.sourceRefsForExpansion,
			recallLayers: maintenanceContractDiagnostics.recallLayers
		};
		const llmBudget = snapshotMemoryLlmBudgetAudit(ctx.llmBudgetAudit);
		stats.semanticUpgrade.llm = {
			taskSummaryUpgrade: summarizeMaintenanceCalls(llmBudget, ["task-summary-batch"]),
			consolidationConfirm: summarizeMaintenanceCalls(llmBudget, ["consolidation-confirm-batch"])
		};
		store.auditRepo.finishMaintenance({
			runId,
			agentId: ctx.agentId,
			jobType: "consolidate",
			statsJson: {
				...stats,
				llmBudget
			},
			startedAt: runStartedAt,
			completedAt: nowIso(),
			status: "completed"
		});
		return stats;
	} catch (error) {
		store.auditRepo.finishMaintenance({
			runId,
			agentId: ctx.agentId,
			jobType: "consolidate",
			statsJson: {
				...stats,
				llmBudget: snapshotMemoryLlmBudgetAudit(ctx.llmBudgetAudit),
				error: String(error)
			},
			startedAt: runStartedAt,
			completedAt: nowIso(),
			status: "failed"
		});
		throw error;
	}
}
//#endregion
export { runConsolidation };
