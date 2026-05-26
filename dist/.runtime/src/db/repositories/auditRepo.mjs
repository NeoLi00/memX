import { randomId, safeJsonParse, stableHash } from "../../support.mjs";
//#region src/db/repositories/auditRepo.ts
var AuditRepo = class {
	db;
	constructor(db) {
		this.db = db;
	}
	semanticWriteJobId(params) {
		return stableHash([
			params.agentId,
			params.sessionKey,
			params.turnId,
			params.jobType
		]);
	}
	recordPolicyDecision(params) {
		this.db.prepare(`INSERT INTO policy_decisions(
          decision_id, agent_id, session_key, source_ref, candidate_hash, salience_score, utility_score, chosen_action, reasons_json, created_at, metadata_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`).run(randomId("decision"), params.agentId, params.sessionKey ?? null, params.sourceRef, stableHash([params.candidateText]), params.decision.salienceScore, params.decision.expectedFutureUtility, params.decision.action, JSON.stringify(params.decision.reasons), params.createdAt, JSON.stringify(params.metadataJson ?? {}));
	}
	recordRetrieval(audit) {
		this.db.prepare(`INSERT INTO retrieval_audit(
          audit_id, agent_id, session_key, scope, route_type, query_text, query_hash, selected_items_json, injected_chars, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`).run(audit.auditId, audit.agentId, audit.sessionKey ?? null, audit.scope, audit.routeType, audit.queryText, audit.queryHash, JSON.stringify(audit.selectedItemsJson), audit.injectedChars, audit.createdAt);
	}
	annotateLatestRetrievalInjection(params) {
		const queryHash = stableHash([params.queryText]);
		const row = this.db.prepare(`SELECT audit_id, selected_items_json, injected_chars
           FROM retrieval_audit
          WHERE agent_id = ?
            AND query_hash = ?
            ${params.sessionKey ? "AND session_key = ?" : ""}
          ORDER BY created_at DESC, rowid DESC
          LIMIT 1`).get(...params.sessionKey ? [
			params.agentId,
			queryHash,
			params.sessionKey
		] : [params.agentId, queryHash]);
		if (!row) return;
		const selectedItems = safeJsonParse(row.selected_items_json, {});
		selectedItems.nativeContextInjection = {
			candidateChars: params.candidateChars,
			actualInjectedChars: params.actualInjectedChars,
			actualContextPreview: params.actualContextPreview ?? "",
			finalInjectedPackets: params.finalInjectedPackets ?? [],
			finalDiagnostics: params.finalDiagnostics ?? [],
			eligible: params.eligible,
			reason: params.reason,
			finalizedAt: params.finalizedAt
		};
		this.db.prepare(`UPDATE retrieval_audit
            SET selected_items_json = ?,
                injected_chars = ?
          WHERE audit_id = ?`).run(JSON.stringify(selectedItems), params.actualInjectedChars, row.audit_id);
	}
	recordSignal(signal) {
		this.db.prepare(`INSERT OR IGNORE INTO memory_signal_events(
          signal_id, agent_id, scope, session_key, signal_type, memory_kind, content_ref, semantic_key, value, source_ref, metadata_json, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`).run(signal.signalId, signal.agentId, signal.scope, signal.sessionKey ?? null, signal.signalType, signal.memoryKind, signal.contentRef ?? null, signal.semanticKey, signal.value, signal.sourceRef, JSON.stringify(signal.metadataJson), signal.createdAt);
	}
	recordSemanticWriteAttemptStart(params) {
		const sessionKey = params.sessionKey ?? "default";
		const jobType = params.jobType ?? "turn_semantic_extraction";
		const jobId = this.semanticWriteJobId({
			agentId: params.agentId,
			sessionKey,
			turnId: params.turnId,
			jobType
		});
		const sourceRefs = [...new Set(params.sourceRefs.filter((entry) => entry.trim()))];
		const existing = this.db.prepare(`SELECT status, attempt_count
           FROM semantic_write_jobs
          WHERE job_id = ?`).get(jobId);
		if (existing?.status === "succeeded") return {
			jobId,
			attemptCount: existing.attempt_count,
			claimed: false
		};
		if (params.retryOnly && existing?.status !== "pending" && existing?.status !== "retrying") return {
			jobId,
			attemptCount: existing?.attempt_count ?? 0,
			claimed: false
		};
		this.db.prepare(`INSERT INTO semantic_write_jobs(
          job_id, agent_id, session_key, scope, turn_id, job_type, source_refs_json, input_hash,
          status, attempt_count, last_error, result_json, next_attempt_at, created_at, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'running', 1, NULL, '{}', NULL, ?, ?)
        ON CONFLICT(job_id) DO UPDATE SET
          scope = excluded.scope,
          source_refs_json = excluded.source_refs_json,
          input_hash = excluded.input_hash,
          status = 'running',
          attempt_count = semantic_write_jobs.attempt_count + 1,
          last_error = NULL,
          next_attempt_at = NULL,
          updated_at = excluded.updated_at`).run(jobId, params.agentId, sessionKey, params.scope, params.turnId, jobType, JSON.stringify(sourceRefs), params.inputHash, params.startedAt, params.startedAt);
		return {
			jobId,
			attemptCount: this.db.prepare(`SELECT attempt_count FROM semantic_write_jobs WHERE job_id = ?`).get(jobId)?.attempt_count ?? 1,
			claimed: true
		};
	}
	finishSemanticWriteAttempt(params) {
		const sessionKey = params.sessionKey ?? "default";
		const jobType = params.jobType ?? "turn_semantic_extraction";
		const jobId = this.semanticWriteJobId({
			agentId: params.agentId,
			sessionKey,
			turnId: params.turnId,
			jobType
		});
		this.db.prepare(`UPDATE semantic_write_jobs
            SET status = ?,
                last_error = ?,
                result_json = ?,
                next_attempt_at = ?,
                updated_at = ?
          WHERE job_id = ?`).run(params.status, params.error ?? null, JSON.stringify(params.resultJson ?? {}), params.nextAttemptAt ?? null, params.completedAt, jobId);
	}
	listSemanticWriteJobs(params) {
		const limit = Math.max(1, Math.min(Math.trunc(params.limit ?? 50), 200));
		const values = [params.agentId];
		let sql = `
      SELECT job_id, agent_id, session_key, scope, turn_id, job_type, source_refs_json, input_hash,
             status, attempt_count, last_error, result_json, next_attempt_at, created_at, updated_at
        FROM semantic_write_jobs
       WHERE agent_id = ?
    `;
		if (params.sessionKey) {
			sql += " AND session_key = ?";
			values.push(params.sessionKey);
		}
		if (params.statuses && params.statuses.length > 0) {
			sql += ` AND status IN (${params.statuses.map(() => "?").join(", ")})`;
			values.push(...params.statuses);
		}
		sql += ` ORDER BY updated_at DESC, rowid DESC LIMIT ${limit}`;
		return this.db.prepare(sql).all(...values).map((row) => {
			const record = row;
			return {
				jobId: record.job_id,
				agentId: record.agent_id,
				sessionKey: record.session_key,
				scope: record.scope,
				turnId: record.turn_id,
				jobType: record.job_type,
				sourceRefs: safeJsonParse(record.source_refs_json, []),
				inputHash: record.input_hash,
				status: record.status,
				attemptCount: record.attempt_count,
				lastError: record.last_error ?? void 0,
				resultJson: safeJsonParse(record.result_json, {}),
				nextAttemptAt: record.next_attempt_at ?? void 0,
				createdAt: record.created_at,
				updatedAt: record.updated_at
			};
		});
	}
	listRetryableSemanticWriteJobs(params) {
		const limit = Math.max(1, Math.min(Math.trunc(params.limit ?? 50), 200));
		const values = [params.agentId];
		let sql = `
      SELECT job_id, agent_id, session_key, scope, turn_id, job_type, source_refs_json, input_hash,
             status, attempt_count, last_error, result_json, next_attempt_at, created_at, updated_at
        FROM semantic_write_jobs
       WHERE agent_id = ?
         AND status IN ('pending', 'retrying')
         AND (next_attempt_at IS NULL OR next_attempt_at <= ?)
    `;
		values.push(params.now);
		if (params.sessionKey) {
			sql += " AND session_key = ?";
			values.push(params.sessionKey);
		}
		sql += ` ORDER BY updated_at ASC, rowid ASC LIMIT ${limit}`;
		return this.db.prepare(sql).all(...values).map((row) => {
			const record = row;
			return {
				jobId: record.job_id,
				agentId: record.agent_id,
				sessionKey: record.session_key,
				scope: record.scope,
				turnId: record.turn_id,
				jobType: record.job_type,
				sourceRefs: safeJsonParse(record.source_refs_json, []),
				inputHash: record.input_hash,
				status: record.status,
				attemptCount: record.attempt_count,
				lastError: record.last_error ?? void 0,
				resultJson: safeJsonParse(record.result_json, {}),
				nextAttemptAt: record.next_attempt_at ?? void 0,
				createdAt: record.created_at,
				updatedAt: record.updated_at
			};
		});
	}
	listSignals(params) {
		const values = [params.agentId];
		let sql = `
      SELECT signal_id, agent_id, scope, session_key, signal_type, memory_kind, content_ref, semantic_key, value,
             source_ref, metadata_json, created_at
        FROM memory_signal_events
       WHERE agent_id = ?
    `;
		if (params.sessionKey) {
			sql += " AND session_key = ?";
			values.push(params.sessionKey);
		}
		if (params.signalTypes && params.signalTypes.length > 0) {
			sql += ` AND signal_type IN (${params.signalTypes.map(() => "?").join(", ")})`;
			values.push(...params.signalTypes);
		}
		if (params.after) {
			sql += " AND created_at > ?";
			values.push(params.after);
		}
		if (params.until) {
			sql += " AND created_at <= ?";
			values.push(params.until);
		}
		sql += " ORDER BY created_at ASC, rowid ASC";
		if (params.limit) sql += ` LIMIT ${Math.max(1, Math.trunc(params.limit))}`;
		return this.db.prepare(sql).all(...values).map((row) => {
			const record = row;
			return {
				signalId: record.signal_id,
				agentId: record.agent_id,
				scope: record.scope,
				sessionKey: record.session_key ?? void 0,
				signalType: record.signal_type,
				memoryKind: record.memory_kind,
				contentRef: record.content_ref ?? void 0,
				semanticKey: record.semantic_key,
				value: record.value,
				sourceRef: record.source_ref,
				metadataJson: safeJsonParse(record.metadata_json, {}),
				createdAt: record.created_at
			};
		});
	}
	listSignalsForTargets(params) {
		if (params.targets.length === 0) return [];
		const values = [params.agentId];
		let sql = `
      SELECT signal_id, agent_id, scope, session_key, signal_type, memory_kind, content_ref, semantic_key, value,
             source_ref, metadata_json, created_at
        FROM memory_signal_events
       WHERE agent_id = ?
         AND (${params.targets.map((target) => {
			values.push(target.memoryKind, target.contentRef ?? null, target.semanticKey);
			return `(
        memory_kind = ?
        AND (
          (content_ref IS NOT NULL AND content_ref = ?)
          OR (content_ref IS NULL AND semantic_key = ?)
        )
      )`;
		}).join(" OR ")})
    `;
		if (params.until) {
			sql += " AND created_at <= ?";
			values.push(params.until);
		}
		sql += " ORDER BY created_at ASC, rowid ASC";
		return this.db.prepare(sql).all(...values).map((row) => {
			const record = row;
			return {
				signalId: record.signal_id,
				agentId: record.agent_id,
				scope: record.scope,
				sessionKey: record.session_key ?? void 0,
				signalType: record.signal_type,
				memoryKind: record.memory_kind,
				contentRef: record.content_ref ?? void 0,
				semanticKey: record.semantic_key,
				value: record.value,
				sourceRef: record.source_ref,
				metadataJson: safeJsonParse(record.metadata_json, {}),
				createdAt: record.created_at
			};
		});
	}
	latestSignalCreatedAt(params) {
		const values = [params.agentId];
		let sql = `
      SELECT MAX(created_at) AS createdAt
        FROM memory_signal_events
       WHERE agent_id = ?
    `;
		if (params.sessionKey) {
			sql += " AND session_key = ?";
			values.push(params.sessionKey);
		}
		return this.db.prepare(sql).get(...values)?.createdAt ?? void 0;
	}
	listRetrievals(params) {
		const limit = Math.max(1, Math.min(Math.trunc(params.limit ?? 50), 200));
		const values = [params.agentId];
		let sql = `
      SELECT audit_id, agent_id, session_key, scope, route_type, query_text, query_hash, selected_items_json,
             injected_chars, created_at
        FROM retrieval_audit
       WHERE agent_id = ?
    `;
		if (params.sessionKey) {
			sql += " AND session_key = ?";
			values.push(params.sessionKey);
		}
		sql += ` ORDER BY created_at DESC, rowid DESC LIMIT ${limit}`;
		return this.db.prepare(sql).all(...values).map((row) => {
			const record = row;
			return {
				auditId: record.audit_id,
				agentId: record.agent_id,
				sessionKey: record.session_key ?? void 0,
				scope: record.scope,
				routeType: record.route_type,
				queryText: record.query_text,
				queryHash: record.query_hash,
				selectedItemsJson: safeJsonParse(record.selected_items_json, {}),
				injectedChars: record.injected_chars,
				createdAt: record.created_at
			};
		});
	}
	listPolicyDecisions(params) {
		const limit = Math.max(1, Math.min(Math.trunc(params.limit ?? 50), 200));
		const values = [params.agentId];
		let sql = `
      SELECT decision_id, agent_id, session_key, source_ref, candidate_hash, salience_score, utility_score,
             chosen_action, reasons_json, created_at, metadata_json
        FROM policy_decisions
       WHERE agent_id = ?
    `;
		if (params.sessionKey) {
			sql += " AND session_key = ?";
			values.push(params.sessionKey);
		}
		sql += ` ORDER BY created_at DESC, rowid DESC LIMIT ${limit}`;
		return this.db.prepare(sql).all(...values).map((row) => {
			const record = row;
			return {
				decisionId: record.decision_id,
				agentId: record.agent_id,
				sessionKey: record.session_key ?? void 0,
				sourceRef: record.source_ref,
				candidateHash: record.candidate_hash,
				salienceScore: record.salience_score,
				utilityScore: record.utility_score,
				chosenAction: record.chosen_action,
				reasons: safeJsonParse(record.reasons_json, []),
				metadataJson: safeJsonParse(record.metadata_json, {}),
				createdAt: record.created_at
			};
		});
	}
	listMaintenanceRuns(params) {
		const limit = Math.max(1, Math.min(Math.trunc(params.limit ?? 50), 200));
		const values = [params.agentId];
		let sql = `
      SELECT run_id, agent_id, session_key, job_type, stats_json, started_at, completed_at, status
        FROM maintenance_runs
       WHERE agent_id = ?
    `;
		if (params.sessionKey) {
			sql += " AND session_key = ?";
			values.push(params.sessionKey);
		}
		sql += ` ORDER BY started_at DESC, rowid DESC LIMIT ${limit}`;
		return this.db.prepare(sql).all(...values).map((row) => {
			const record = row;
			return {
				runId: record.run_id,
				agentId: record.agent_id,
				sessionKey: record.session_key ?? void 0,
				jobType: record.job_type,
				statsJson: safeJsonParse(record.stats_json, {}),
				startedAt: record.started_at,
				completedAt: record.completed_at ?? void 0,
				status: record.status
			};
		});
	}
	markExpiredRunningMaintenanceRunsInterrupted(params) {
		const rows = this.db.prepare(`SELECT run_id, stats_json
           FROM maintenance_runs
          WHERE agent_id = ?
            AND status = 'running'
            AND started_at <= ?`).all(params.agentId, params.startedBefore);
		if (rows.length === 0) return 0;
		const update = this.db.prepare(`UPDATE maintenance_runs
          SET stats_json = ?, completed_at = ?, status = 'failed'
        WHERE run_id = ?`);
		for (const row of rows) {
			const stats = safeJsonParse(row.stats_json, {});
			update.run(JSON.stringify({
				...stats,
				recovery: {
					reason: params.reason,
					recoveredAt: params.completedAt
				}
			}), params.completedAt, row.run_id);
		}
		return rows.length;
	}
	startMaintenance(params) {
		const runId = randomId("maintenance");
		this.db.prepare(`INSERT INTO maintenance_runs(
          run_id, agent_id, session_key, job_type, stats_json, started_at, completed_at, status
        ) VALUES (?, ?, ?, ?, ?, ?, NULL, 'running')`).run(runId, params.agentId, params.sessionKey ?? null, params.jobType, JSON.stringify(params.stats), params.startedAt);
		return runId;
	}
	finishMaintenance(run) {
		this.db.prepare(`UPDATE maintenance_runs
            SET stats_json = ?, completed_at = ?, status = ?
          WHERE run_id = ?`).run(JSON.stringify(run.statsJson), run.completedAt ?? null, run.status, run.runId);
	}
};
//#endregion
export { AuditRepo };
