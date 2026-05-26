import { nowIso } from "../support.mjs";
import { runAbstractionJobs } from "./abstractionJobs.mjs";
import { runAbstractionPromotion } from "./abstractionPromotion.mjs";
import { runConsolidation } from "./consolidate.mjs";
import { runSourceSegmentSemanticExtraction } from "./sourceSegmentSemanticExtraction.mjs";
//#region src/pipeline/maintenanceBatch.ts
const SEMANTIC_RETRY_BATCH_LIMIT = 32;
function uniqueTurnIds(turnIds) {
	return [...new Set(turnIds.filter((turnId) => turnId.trim().length > 0))];
}
async function runAutomaticMaintenanceBatch(store, ctx, batch) {
	const retryJobs = store.auditRepo.listRetryableSemanticWriteJobs({
		agentId: ctx.agentId,
		sessionKey: batch.sessionKey,
		now: ctx.now,
		limit: SEMANTIC_RETRY_BATCH_LIMIT
	});
	const effectiveCtx = retryJobs.length > 0 ? {
		...ctx,
		scopes: uniqueTurnIds([...ctx.scopes, ...retryJobs.map((job) => job.scope)])
	} : ctx;
	const retryTurnIds = uniqueTurnIds(retryJobs.map((job) => job.turnId));
	const sourceSegmentTurnIds = uniqueTurnIds([...batch.turnIds, ...retryTurnIds]);
	const effectiveBatch = {
		...batch,
		turnIds: sourceSegmentTurnIds,
		turnCount: sourceSegmentTurnIds.length
	};
	const sourceSegmentStartedAt = nowIso();
	const sourceSegmentRunId = store.auditRepo.startMaintenance({
		agentId: ctx.agentId,
		sessionKey: batch.sessionKey,
		jobType: "source-segment-semantic-extraction",
		startedAt: sourceSegmentStartedAt,
		stats: {
			sessionKey: batch.sessionKey,
			turnIds: batch.turnIds,
			turnCount: batch.turnCount,
			repairTurnIds: retryTurnIds,
			retryJobCount: retryJobs.length,
			sourceSegmentTurnIds,
			reason: batch.reason,
			status: "started"
		}
	});
	let sourceSegmentStats;
	try {
		sourceSegmentStats = await runSourceSegmentSemanticExtraction(store, effectiveCtx, {
			sessionKey: batch.sessionKey,
			turnIds: sourceSegmentTurnIds
		});
		store.auditRepo.finishMaintenance({
			runId: sourceSegmentRunId,
			agentId: ctx.agentId,
			sessionKey: batch.sessionKey,
			jobType: "source-segment-semantic-extraction",
			startedAt: sourceSegmentStartedAt,
			completedAt: nowIso(),
			status: "completed",
			statsJson: {
				...sourceSegmentStats,
				sessionKey: batch.sessionKey,
				turnIds: batch.turnIds,
				turnCount: batch.turnCount,
				repairTurnIds: retryTurnIds,
				retryJobCount: retryJobs.length,
				sourceSegmentTurnIds,
				reason: batch.reason
			}
		});
	} catch (error) {
		store.auditRepo.finishMaintenance({
			runId: sourceSegmentRunId,
			agentId: ctx.agentId,
			sessionKey: batch.sessionKey,
			jobType: "source-segment-semantic-extraction",
			startedAt: sourceSegmentStartedAt,
			completedAt: nowIso(),
			status: "failed",
			statsJson: {
				sessionKey: batch.sessionKey,
				turnIds: batch.turnIds,
				turnCount: batch.turnCount,
				repairTurnIds: retryTurnIds,
				retryJobCount: retryJobs.length,
				sourceSegmentTurnIds,
				reason: batch.reason,
				error: error instanceof Error ? error.message : String(error)
			}
		});
		throw error;
	}
	const consolidationStats = await runConsolidation(store, effectiveCtx, { batch: effectiveBatch });
	const deltaTriggered = sourceSegmentStats.candidatesWritten > 0 || (consolidationStats.batch?.delta.eventsConsidered ?? 0) > 0 || (consolidationStats.batch?.delta.tasksConsidered ?? 0) > 0 || consolidationStats.promotedFacts > 0 || consolidationStats.promotedEdges > 0 || consolidationStats.promotedStates > 0 || consolidationStats.beliefSignalsProcessed > 0 || consolidationStats.beliefsNeedingReevaluation > 0 || consolidationStats.beliefsUpserted > 0 || consolidationStats.semanticUpgrade.taskSummariesUpgraded > 0;
	runAbstractionPromotion(store, effectiveCtx, {
		batch: effectiveBatch,
		candidateIds: (await runAbstractionJobs(store, effectiveCtx, {
			refineWithLlm: false,
			batch: effectiveBatch,
			deltaTriggered
		})).materializedCandidateIds ?? [],
		deltaTriggered
	});
}
//#endregion
export { runAutomaticMaintenanceBatch };
