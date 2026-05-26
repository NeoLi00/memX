import { nowIso, objectRecord, stableHash } from "../support.mjs";
import { refreshEntityProfileDocs } from "./entityProfile.mjs";
import { normalizeGraphRelationType } from "./semantic/heuristics.mjs";
import { buildEntityMention, resolveEntityMention } from "./entityResolver.mjs";
import { buildMaintenanceContractMetadata, summarizeMaintenanceContractDiagnostics } from "./maintenanceContract.mjs";
import "./semantics.mjs";
import { inferStrategyHypothesisStage } from "./strategyHypotheses.mjs";
import { buildOutcomeEventCandidate } from "./outcomeHypotheses.mjs";
import { writeCandidate } from "./write.mjs";
//#region src/pipeline/abstractionPromotion.ts
const PROMOTION_POLICY_VERSION = "memx-authority-v3";
function currentSourceEpoch(store, ctx) {
	return ctx.readEpoch ?? store.client.currentMemoryEpoch(ctx.agentId);
}
function candidateStageCounts(store, agentId) {
	return {
		active: store.abstractionRepo.countByAgent({
			agentId,
			stages: ["active"]
		}),
		candidate: store.abstractionRepo.countByAgent({
			agentId,
			stages: ["candidate"]
		}),
		decaying: store.abstractionRepo.countByAgent({
			agentId,
			stages: ["decaying"]
		}),
		probationary: store.abstractionRepo.countByAgent({
			agentId,
			stages: ["probationary"]
		}),
		quarantined: store.abstractionRepo.countByAgent({
			agentId,
			stages: ["quarantined"]
		}),
		superseded: store.abstractionRepo.countByAgent({
			agentId,
			stages: ["superseded"]
		})
	};
}
function eligibleForPromotion(candidate) {
	return candidate.stage === "probationary" || candidate.stage === "active";
}
function stringArray(value) {
	if (!Array.isArray(value)) return [];
	return value.filter((entry) => typeof entry === "string" && entry.trim().length > 0);
}
function stringValue(value) {
	return typeof value === "string" && value.trim() ? value.trim() : void 0;
}
function numberValue(value) {
	return typeof value === "number" && Number.isFinite(value) ? value : void 0;
}
function generatedFromValue(value, fallback) {
	if (typeof value === "string" && value.trim()) return value.trim();
	if (Array.isArray(value)) {
		const entries = value.filter((entry) => typeof entry === "string" && entry.trim().length > 0);
		if (entries.length > 0) return entries;
	}
	return fallback;
}
function parseWorkflowDomainKey(candidate) {
	const metadataDomainKey = stringValue(candidate.metadataJson.domainKey);
	if (metadataDomainKey) return metadataDomainKey;
	const prefix = `workflow_pattern:${candidate.scope}:`;
	if (candidate.semanticKey.startsWith(prefix)) return candidate.semanticKey.slice(prefix.length) || void 0;
}
function inferDerivedStateKind(candidate, stateKey) {
	const metadataStateKind = stringValue(candidate.metadataJson.promotedStateKind);
	if (metadataStateKind === "session" || metadataStateKind === "durable") return metadataStateKind;
	if (stateKey === "project.active_project" && candidate.confidence >= .78 && candidate.stabilityScore >= .68 && candidate.contradictionScore <= .18) return "durable";
	return "session";
}
function buildPromotionMetadata(candidate, params) {
	const promotion = objectRecord(candidate.metadataJson.promotion) ?? {};
	const semanticSource = stringValue(candidate.metadataJson.semanticSource) ?? "upstream_structured";
	const semanticSources = Array.isArray(candidate.metadataJson.semanticSources) ? candidate.metadataJson.semanticSources.filter((entry) => typeof entry === "string" && entry.trim().length > 0) : void 0;
	return {
		...buildMaintenanceContractMetadata({
			existing: candidate.metadataJson,
			sourceRef: params.sourceRef,
			supportContentRefs: candidate.supportContentRefs,
			supportBeliefIds: candidate.supportBeliefIds,
			derivedFromIds: params.derivedFromIds,
			semanticSource,
			semanticSources,
			authoritySource: "deterministic_aggregated",
			generatedFrom: generatedFromValue(candidate.metadataJson.generatedFrom, "abstraction_promotion"),
			recallLayer: params.targetKind === "strategy" ? "strategy" : params.targetKind,
			answerEligibleByDefault: params.targetKind !== "strategy",
			materializedEpoch: params.materializedEpoch,
			derivationPolicyVersion: PROMOTION_POLICY_VERSION
		}),
		...params.extra,
		promotion: {
			...promotion,
			targetKind: params.targetKind,
			targetRef: params.targetRef,
			sourceRef: params.sourceRef,
			supportContentRefs: candidate.supportContentRefs,
			supportBeliefIds: candidate.supportBeliefIds,
			derivedAtEpoch: params.derivedAtEpoch,
			derivedFromKind: params.derivedFromKind,
			derivedFromIds: params.derivedFromIds,
			derivationPolicyVersion: PROMOTION_POLICY_VERSION,
			firstPromotedAt: stringValue(promotion.firstPromotedAt) ?? stringValue(promotion.promotedAt) ?? params.promotedAt,
			lastPromotedAt: params.promotedAt,
			...params.extra
		}
	};
}
function updateCandidatePromotionMetadata(store, candidate, metadataJson, updatedAt, stage = candidate.stage, lineage) {
	store.abstractionRepo.upsert({
		...candidate,
		stage,
		...lineage ? {
			materializedEpoch: lineage.materializedEpoch,
			derivedFromMinEpoch: Math.min(candidate.derivedFromMinEpoch ?? lineage.derivedAtEpoch, lineage.derivedAtEpoch),
			derivedFromMaxEpoch: Math.max(candidate.derivedFromMaxEpoch ?? lineage.derivedAtEpoch, lineage.derivedAtEpoch),
			derivedFromKind: lineage.derivedFromKind,
			derivedFromIds: lineage.derivedFromIds,
			derivedAtEpoch: lineage.derivedAtEpoch,
			derivationPolicyVersion: PROMOTION_POLICY_VERSION
		} : {},
		metadataJson,
		updatedAt
	});
}
function parseOutcomeHypothesisCandidate(store, candidate) {
	const metadata = objectRecord(candidate.metadataJson);
	const taskId = stringValue(metadata?.taskId);
	const outcomeKey = stringValue(metadata?.outcomeKey);
	const eventType = stringValue(metadata?.eventType);
	const phase = stringValue(metadata?.phase);
	if (!taskId || !outcomeKey || !eventType || !phase) return null;
	const task = store.taskRepo.get(taskId);
	if (!task) return null;
	return {
		task,
		promotedOutcome: {
			eventType,
			summary: candidate.summary,
			phase,
			closureScore: numberValue(metadata?.closureScore) ?? candidate.confidence,
			verificationScore: numberValue(metadata?.verificationScore) ?? candidate.usefulnessScore,
			contradictionRisk: numberValue(metadata?.contradictionRisk) ?? candidate.contradictionScore,
			confidence: numberValue(metadata?.confidence) ?? candidate.confidence,
			promotionScore: numberValue(metadata?.promotionScore) ?? candidate.usefulnessScore,
			evidenceChunkIds: stringArray(metadata?.evidenceChunkIds),
			outcomeKey
		},
		observedAt: stringValue(metadata?.observedAt) ?? candidate.updatedAt,
		sourceRef: `abstraction_candidate:${candidate.candidateId}`,
		targetRef: `task:${task.taskId}:outcome:${outcomeKey}`,
		alreadyEmitted: task.metadataJson.lastEmittedOutcomeKey === outcomeKey
	};
}
function parseGraphHypothesisCandidate(candidate) {
	const metadata = objectRecord(candidate.metadataJson);
	const relationValue = stringValue(metadata?.relationType);
	const normalizedRelation = relationValue ? normalizeGraphRelationType(relationValue) : null;
	const sourceName = stringValue(metadata?.sourceName);
	const targetName = stringValue(metadata?.targetName);
	const relationType = normalizedRelation?.relationType;
	const durableRelationType = relationType === "depends_on" || relationType === "blocks" || relationType === "caused_by" || relationType === "uses" || relationType === "part_of" || relationType === "owner_of" || relationType === "supersedes" || relationType === "contradicts" || relationType === "resolved_by" || relationType === "related_to" || relationType === "reads" ? relationType : null;
	if (!durableRelationType || !sourceName || !targetName) return null;
	return {
		relationType: durableRelationType,
		relationSlot: stringValue(metadata?.relationSlot),
		rawPredicate: normalizedRelation?.rawPredicate,
		relationClass: stringValue(metadata?.relationClass),
		sourceName,
		targetName,
		sourceType: stringValue(metadata?.sourceType),
		targetType: stringValue(metadata?.targetType),
		observedAt: stringValue(metadata?.firstSeenAt) ?? candidate.updatedAt
	};
}
function entityTypeValue(value) {
	switch (value) {
		case "person":
		case "project":
		case "tool":
		case "service":
		case "language":
		case "framework":
		case "concept":
		case "organization": return value;
		default: return "unknown";
	}
}
function shouldReplaceState(existing, candidate, sourceRef) {
	if (!existing) return true;
	if (existing.sourceRef === sourceRef) return true;
	const lastSeenAt = stringValue(candidate.metadataJson.lastSeenAt) ?? candidate.updatedAt;
	const existingUpdated = Date.parse(existing.updatedAt);
	const candidateObserved = Date.parse(lastSeenAt);
	if (Number.isFinite(existingUpdated) && Number.isFinite(candidateObserved) && existingUpdated > candidateObserved && existing.confidence >= candidate.confidence - .02) return false;
	return existing.confidence + .08 < candidate.confidence;
}
function promoteDerivedState(store, ctx, candidate) {
	const stateKey = stringValue(candidate.metadataJson.stateKey);
	const valueJson = objectRecord(candidate.metadataJson.valueJson);
	if (!stateKey || !valueJson) return false;
	const stateKind = inferDerivedStateKind(candidate, stateKey);
	const sourceRef = `abstraction_candidate:${candidate.candidateId}`;
	const sourceEpoch = currentSourceEpoch(store, ctx);
	if (!shouldReplaceState(store.stateRepo.get({
		agentId: ctx.agentId,
		scopes: [candidate.scope],
		key: stateKey,
		includeExpired: true,
		now: ctx.now,
		readEpoch: sourceEpoch
	}).at(0), candidate, sourceRef)) return false;
	const promotionEpoch = store.client.nextMemoryEpoch(ctx.agentId, ctx.now);
	const promotedState = {
		key: stateKey,
		valueJson,
		scope: candidate.scope,
		agentId: ctx.agentId,
		stateKind,
		confidence: candidate.confidence,
		sourceRef,
		updatedAt: ctx.now,
		materializedEpoch: promotionEpoch,
		expiresAt: stateKind === "session" ? store.stateRepo.createExpiry(ctx.now, ctx.config.stateTtlHours) : void 0
	};
	store.stateRepo.upsert(promotedState);
	updateCandidatePromotionMetadata(store, candidate, buildPromotionMetadata(candidate, {
		targetKind: "state",
		targetRef: `${candidate.scope}:${stateKey}`,
		sourceRef,
		promotedAt: ctx.now,
		derivedAtEpoch: sourceEpoch,
		derivedFromKind: "abstraction_candidate",
		derivedFromIds: [candidate.candidateId],
		materializedEpoch: promotionEpoch,
		extra: {
			stateKey,
			stateKind,
			supportLastSeenAt: stringValue(candidate.metadataJson.lastSeenAt) ?? candidate.updatedAt
		}
	}), ctx.now, candidate.stage, {
		materializedEpoch: promotionEpoch,
		derivedAtEpoch: sourceEpoch,
		derivedFromKind: "abstraction_candidate",
		derivedFromIds: [candidate.candidateId]
	});
	return true;
}
function promoteGraphHypothesis(store, ctx, candidate) {
	const parsed = parseGraphHypothesisCandidate(candidate);
	if (!parsed) return false;
	const sourceRef = `abstraction_candidate:${candidate.candidateId}`;
	const sourceResolution = resolveEntityMention(store, ctx, buildEntityMention({
		ctx,
		scope: candidate.scope,
		rawText: parsed.sourceName,
		proposedType: entityTypeValue(parsed.sourceType),
		semanticRole: parsed.sourceType === "project" ? "project" : "subject",
		sourceRef,
		supportText: candidate.summary,
		observedAt: parsed.observedAt,
		metadataJson: {
			abstractionCandidateId: candidate.candidateId,
			relationType: parsed.relationType,
			relationRole: "source"
		}
	}));
	const targetResolution = resolveEntityMention(store, ctx, buildEntityMention({
		ctx,
		scope: candidate.scope,
		rawText: parsed.targetName,
		proposedType: entityTypeValue(parsed.targetType),
		semanticRole: parsed.targetType === "project" ? "project" : "object",
		sourceRef,
		supportText: candidate.summary,
		observedAt: parsed.observedAt,
		metadataJson: {
			abstractionCandidateId: candidate.candidateId,
			relationType: parsed.relationType,
			relationRole: "target"
		}
	}));
	if (sourceResolution.method === "uncertain" || targetResolution.method === "uncertain") return false;
	const sourceEntity = sourceResolution.entity;
	const targetEntity = targetResolution.entity;
	const sourceEpoch = currentSourceEpoch(store, ctx);
	const promotionEpoch = store.client.nextMemoryEpoch(ctx.agentId, ctx.now);
	const edgeId = stableHash([
		ctx.agentId,
		candidate.scope,
		sourceEntity.entityId,
		parsed.relationType,
		parsed.relationSlot ?? "",
		targetEntity.entityId
	]);
	const edgeMetadata = buildPromotionMetadata(candidate, {
		targetKind: "graph",
		targetRef: edgeId,
		sourceRef,
		promotedAt: ctx.now,
		derivedAtEpoch: sourceEpoch,
		derivedFromKind: "abstraction_candidate",
		derivedFromIds: [candidate.candidateId],
		materializedEpoch: promotionEpoch,
		extra: {
			relationType: parsed.relationType,
			...parsed.relationSlot ? { relationSlot: parsed.relationSlot } : {},
			sourceEntityId: sourceEntity.entityId,
			targetEntityId: targetEntity.entityId,
			sourceName: parsed.sourceName,
			targetName: parsed.targetName,
			sourceMentionId: sourceResolution.mention.mentionId,
			targetMentionId: targetResolution.mention.mentionId,
			sourceResolutionMethod: sourceResolution.method,
			targetResolutionMethod: targetResolution.method,
			sourceResolutionConfidence: sourceResolution.confidence,
			targetResolutionConfidence: targetResolution.confidence,
			relationClass: parsed.relationClass
		}
	});
	store.graphRepo.upsertEdge({
		edgeId,
		srcEntityId: sourceEntity.entityId,
		relType: parsed.relationType,
		dstEntityId: targetEntity.entityId,
		relationSlot: parsed.relationSlot,
		scope: candidate.scope,
		agentId: ctx.agentId,
		confidence: candidate.confidence,
		validFrom: parsed.observedAt,
		evidenceRef: sourceRef,
		rawRelationType: parsed.rawPredicate,
		sourceKind: parsed.relationClass === "observed" ? "extracted" : "synthesized",
		createdAt: ctx.now,
		updatedAt: ctx.now,
		materializedEpoch: promotionEpoch,
		metadataJson: edgeMetadata
	});
	refreshEntityProfileDocs(store, ctx, [sourceEntity.entityId, targetEntity.entityId]);
	updateCandidatePromotionMetadata(store, candidate, edgeMetadata, ctx.now, candidate.stage, {
		materializedEpoch: promotionEpoch,
		derivedAtEpoch: sourceEpoch,
		derivedFromKind: "abstraction_candidate",
		derivedFromIds: [candidate.candidateId]
	});
	return true;
}
function promoteWorkflowPattern(store, ctx, candidate) {
	const domainKey = parseWorkflowDomainKey(candidate);
	if (!domainKey) return false;
	const supportTaskIds = candidate.supportContentRefs.filter((ref) => ref.startsWith("task:")).map((ref) => ref.slice(5));
	const strategyId = stableHash([
		ctx.agentId,
		candidate.scope,
		domainKey
	]);
	const sourceEpoch = currentSourceEpoch(store, ctx);
	const promotionEpoch = store.client.nextMemoryEpoch(ctx.agentId, ctx.now);
	const stage = inferStrategyHypothesisStage({
		confidence: candidate.confidence,
		usefulnessScore: candidate.usefulnessScore,
		stabilityScore: candidate.stabilityScore,
		contradictionScore: candidate.contradictionScore,
		groundedByRoles: stringArray(candidate.metadataJson.groundedByRoles),
		taskPhase: stringValue(candidate.metadataJson.taskPhase),
		explicitInstruction: Boolean(candidate.metadataJson.explicitInstruction),
		groundedResolution: Boolean(candidate.metadataJson.groundedResolution)
	});
	const sourceRef = `abstraction_candidate:${candidate.candidateId}`;
	const strategyMetadata = buildMaintenanceContractMetadata({
		existing: {
			...candidate.metadataJson,
			promotedFromCandidateId: candidate.candidateId,
			promotionSourceRef: sourceRef,
			supportContentRefs: candidate.supportContentRefs,
			supportBeliefIds: candidate.supportBeliefIds,
			promotedAt: ctx.now
		},
		sourceRef,
		supportContentRefs: candidate.supportContentRefs,
		supportBeliefIds: candidate.supportBeliefIds,
		derivedFromIds: [candidate.candidateId],
		semanticSource: stringValue(candidate.metadataJson.semanticSource) ?? "upstream_structured",
		semanticSources: Array.isArray(candidate.metadataJson.semanticSources) ? candidate.metadataJson.semanticSources.filter((entry) => typeof entry === "string" && entry.trim().length > 0) : void 0,
		authoritySource: "deterministic_aggregated",
		generatedFrom: generatedFromValue(candidate.metadataJson.generatedFrom, "workflow_pattern_promotion"),
		recallLayer: "strategy",
		answerEligibleByDefault: false,
		materializedEpoch: promotionEpoch,
		derivationPolicyVersion: PROMOTION_POLICY_VERSION
	});
	const strategyRecord = {
		strategyId,
		agentId: ctx.agentId,
		scope: candidate.scope,
		domainKey,
		summary: candidate.summary,
		supportBeliefIds: candidate.supportBeliefIds,
		supportTaskIds,
		confidence: candidate.confidence,
		usefulnessScore: candidate.usefulnessScore,
		stabilityScore: candidate.stabilityScore,
		contradictionScore: candidate.contradictionScore,
		stage,
		derivedFromMinEpoch: sourceEpoch,
		derivedFromMaxEpoch: sourceEpoch,
		materializedEpoch: promotionEpoch,
		derivedFromKind: "abstraction_candidate",
		derivedFromIds: [candidate.candidateId],
		derivedAtEpoch: sourceEpoch,
		derivationPolicyVersion: PROMOTION_POLICY_VERSION,
		metadataJson: strategyMetadata,
		createdAt: ctx.now,
		updatedAt: ctx.now
	};
	store.strategyRepo.upsert(strategyRecord);
	updateCandidatePromotionMetadata(store, candidate, buildPromotionMetadata(candidate, {
		targetKind: "strategy",
		targetRef: strategyId,
		sourceRef,
		promotedAt: ctx.now,
		derivedAtEpoch: sourceEpoch,
		derivedFromKind: "abstraction_candidate",
		derivedFromIds: [candidate.candidateId],
		materializedEpoch: promotionEpoch,
		extra: {
			domainKey,
			strategyStage: stage,
			supportTaskIds
		}
	}), ctx.now, candidate.stage, {
		materializedEpoch: promotionEpoch,
		derivedAtEpoch: sourceEpoch,
		derivedFromKind: "abstraction_candidate",
		derivedFromIds: [candidate.candidateId]
	});
	return true;
}
function promoteOutcomeHypothesis(store, ctx, candidate) {
	const parsed = parseOutcomeHypothesisCandidate(store, candidate);
	if (!parsed) return false;
	if (!parsed.alreadyEmitted) {
		if (writeCandidate(store, ctx, buildOutcomeEventCandidate(parsed.task, parsed.promotedOutcome, parsed.observedAt)).events === 0) return false;
	}
	const sourceEpoch = currentSourceEpoch(store, ctx);
	const promotionEpoch = store.client.nextMemoryEpoch(ctx.agentId, ctx.now);
	updateCandidatePromotionMetadata(store, candidate, buildPromotionMetadata(candidate, {
		targetKind: "event",
		targetRef: parsed.targetRef,
		sourceRef: parsed.sourceRef,
		promotedAt: ctx.now,
		derivedAtEpoch: sourceEpoch,
		derivedFromKind: "abstraction_candidate",
		derivedFromIds: [candidate.candidateId],
		materializedEpoch: promotionEpoch,
		extra: {
			taskId: parsed.task.taskId,
			outcomeKey: parsed.promotedOutcome.outcomeKey,
			eventType: parsed.promotedOutcome.eventType,
			phase: parsed.promotedOutcome.phase,
			evidenceChunkIds: parsed.promotedOutcome.evidenceChunkIds
		}
	}), ctx.now, "active", {
		materializedEpoch: promotionEpoch,
		derivedAtEpoch: sourceEpoch,
		derivedFromKind: "abstraction_candidate",
		derivedFromIds: [candidate.candidateId]
	});
	store.taskRepo.update(parsed.task.taskId, {
		metadataJson: {
			...parsed.task.metadataJson,
			lastEmittedOutcomeKey: parsed.promotedOutcome.outcomeKey,
			lastEmittedOutcomeAt: ctx.now,
			lastPromotedOutcomeHypothesisId: candidate.candidateId
		},
		updatedAt: ctx.now
	});
	return true;
}
function runAbstractionPromotion(store, ctx, options = {}) {
	const runStartedAt = nowIso();
	const runId = store.auditRepo.startMaintenance({
		agentId: ctx.agentId,
		sessionKey: options.batch?.sessionKey ?? ctx.sessionKey,
		jobType: "abstraction-promotion",
		stats: {},
		startedAt: runStartedAt
	});
	try {
		if (options.batch && options.deltaTriggered === false) {
			const skippedStats = {
				candidatesEvaluated: 0,
				deltaTriggered: false,
				skippedNoRelevantDelta: true,
				activeCandidates: 0,
				probationaryCandidates: 0,
				candidateCandidates: 0,
				decayingCandidates: 0,
				quarantinedCandidates: 0,
				supersededCandidates: 0,
				promotedGraphs: 0,
				promotedOutcomes: 0,
				promotedStates: 0,
				promotedStrategies: 0,
				promotedConcepts: 0,
				skippedCandidates: []
			};
			store.auditRepo.finishMaintenance({
				runId,
				agentId: ctx.agentId,
				jobType: "abstraction-promotion",
				statsJson: {
					...options.batch ? { batch: options.batch } : {},
					...skippedStats
				},
				startedAt: runStartedAt,
				completedAt: nowIso(),
				status: "completed"
			});
			return skippedStats;
		}
		const candidates = options.candidateIds ? options.candidateIds.map((candidateId) => store.abstractionRepo.getById(candidateId)).filter((candidate) => candidate !== void 0) : store.abstractionRepo.listByAgent({
			agentId: ctx.agentId,
			scopes: ctx.scopes,
			limit: 64
		});
		const stages = candidateStageCounts(store, ctx.agentId);
		const maintenanceContractDiagnostics = summarizeMaintenanceContractDiagnostics(candidates.map((candidate) => candidate.metadataJson));
		const stats = {
			candidatesEvaluated: candidates.length,
			...options.batch ? { deltaTriggered: options.deltaTriggered !== false } : {},
			skippedNoRelevantDelta: false,
			activeCandidates: stages.active,
			probationaryCandidates: stages.probationary,
			candidateCandidates: stages.candidate,
			decayingCandidates: stages.decaying,
			quarantinedCandidates: stages.quarantined,
			supersededCandidates: stages.superseded,
			promotedGraphs: 0,
			promotedOutcomes: 0,
			promotedStates: 0,
			promotedStrategies: 0,
			promotedConcepts: 0,
			skippedCandidates: [],
			maintenanceContractDiagnostics,
			recallFacingDiagnostics: {
				recallVisible: maintenanceContractDiagnostics.recallVisibleCount > 0,
				answerEligibleByDefault: maintenanceContractDiagnostics.answerEligibleByDefaultCount > 0,
				sourceRefsForExpansion: maintenanceContractDiagnostics.sourceRefsForExpansion,
				recallLayers: maintenanceContractDiagnostics.recallLayers
			}
		};
		for (const candidate of candidates) {
			if (!eligibleForPromotion(candidate)) {
				stats.skippedCandidates?.push({
					candidateId: candidate.candidateId,
					abstractionType: candidate.abstractionType,
					reason: "stage_not_promotable"
				});
				continue;
			}
			if (candidate.abstractionType === "outcome_hypothesis") {
				if (promoteOutcomeHypothesis(store, ctx, candidate)) stats.promotedOutcomes += 1;
				else stats.skippedCandidates?.push({
					candidateId: candidate.candidateId,
					abstractionType: candidate.abstractionType,
					reason: "outcome_not_grounded_or_already_materialized"
				});
				continue;
			}
			if (candidate.abstractionType === "derived_state") {
				if (promoteDerivedState(store, ctx, candidate)) stats.promotedStates += 1;
				else stats.skippedCandidates?.push({
					candidateId: candidate.candidateId,
					abstractionType: candidate.abstractionType,
					reason: "state_promotion_blocked"
				});
				continue;
			}
			if (candidate.abstractionType === "graph_hypothesis") {
				if (promoteGraphHypothesis(store, ctx, candidate)) stats.promotedGraphs += 1;
				else stats.skippedCandidates?.push({
					candidateId: candidate.candidateId,
					abstractionType: candidate.abstractionType,
					reason: "graph_promotion_blocked_or_unresolved"
				});
				continue;
			}
			if (candidate.abstractionType === "workflow_pattern") {
				if (promoteWorkflowPattern(store, ctx, candidate)) stats.promotedStrategies += 1;
				else stats.skippedCandidates?.push({
					candidateId: candidate.candidateId,
					abstractionType: candidate.abstractionType,
					reason: "workflow_pattern_not_grounded_or_invalid"
				});
				continue;
			}
			if (candidate.abstractionType === "concept_candidate") stats.skippedCandidates?.push({
				candidateId: candidate.candidateId,
				abstractionType: candidate.abstractionType,
				reason: "not_promotable_yet"
			});
		}
		store.auditRepo.finishMaintenance({
			runId,
			agentId: ctx.agentId,
			jobType: "abstraction-promotion",
			statsJson: {
				...options.batch ? { batch: options.batch } : {},
				...stats
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
			jobType: "abstraction-promotion",
			statsJson: {
				...options.batch ? { batch: options.batch } : {},
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
export { runAbstractionPromotion };
