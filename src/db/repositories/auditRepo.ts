import { randomId, safeJsonParse, stableHash } from "../../support.js";
import type {
  MaintenanceRunRecord,
  MemoryPolicyDecision,
  MemorySignalEventRecord,
  RetrievalAuditRecord,
  SemanticWriteJobRecord,
  SemanticWriteJobStatus,
} from "../../types.js";
import type { MemxDbClient } from "../client.js";

export class AuditRepo {
  constructor(private readonly db: MemxDbClient) {}

  private semanticWriteJobId(params: {
    agentId: string;
    sessionKey: string;
    turnId: string;
    jobType: SemanticWriteJobRecord["jobType"];
  }): string {
    return stableHash([params.agentId, params.sessionKey, params.turnId, params.jobType]);
  }

  recordPolicyDecision(params: {
    agentId: string;
    sessionKey?: string;
    sourceRef: string;
    candidateText: string;
    decision: MemoryPolicyDecision;
    createdAt: string;
    metadataJson?: Record<string, unknown>;
  }): void {
    this.db
      .prepare(
        `INSERT INTO policy_decisions(
          decision_id, agent_id, session_key, source_ref, candidate_hash, salience_score, utility_score, chosen_action, reasons_json, created_at, metadata_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
      )
      .run(
        randomId("decision"),
        params.agentId,
        params.sessionKey ?? null,
        params.sourceRef,
        stableHash([params.candidateText]),
        params.decision.salienceScore,
        params.decision.expectedFutureUtility,
        params.decision.action,
        JSON.stringify(params.decision.reasons),
        params.createdAt,
        JSON.stringify(params.metadataJson ?? {}),
      );
  }

  recordRetrieval(audit: RetrievalAuditRecord): void {
    this.db
      .prepare(
        `INSERT INTO retrieval_audit(
          audit_id, agent_id, session_key, scope, route_type, query_text, query_hash, selected_items_json, injected_chars, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
      )
      .run(
        audit.auditId,
        audit.agentId,
        audit.sessionKey ?? null,
        audit.scope,
        audit.routeType,
        audit.queryText,
        audit.queryHash,
        JSON.stringify(audit.selectedItemsJson),
        audit.injectedChars,
        audit.createdAt,
      );
  }

  annotateLatestRetrievalInjection(params: {
    agentId: string;
    sessionKey?: string;
    queryText: string;
    actualInjectedChars: number;
    candidateChars: number;
    actualContextPreview?: string;
    finalInjectedPackets?: unknown[];
    finalDiagnostics?: string[];
    eligible: boolean;
    reason?: string;
    finalizedAt: string;
  }): void {
    const queryHash = stableHash([params.queryText]);
    const row = this.db
      .prepare(
        `SELECT audit_id, selected_items_json, injected_chars
           FROM retrieval_audit
          WHERE agent_id = ?
            AND query_hash = ?
            ${params.sessionKey ? "AND session_key = ?" : ""}
          ORDER BY created_at DESC, rowid DESC
          LIMIT 1`,
      )
      .get(...(params.sessionKey ? [params.agentId, queryHash, params.sessionKey] : [params.agentId, queryHash])) as
      | { audit_id: string; selected_items_json: string; injected_chars: number }
      | undefined;
    if (!row) {
      return;
    }
    const selectedItems = safeJsonParse<Record<string, unknown>>(row.selected_items_json, {});
    selectedItems.nativeContextInjection = {
      candidateChars: params.candidateChars,
      actualInjectedChars: params.actualInjectedChars,
      actualContextPreview: params.actualContextPreview ?? "",
      finalInjectedPackets: params.finalInjectedPackets ?? [],
      finalDiagnostics: params.finalDiagnostics ?? [],
      eligible: params.eligible,
      reason: params.reason,
      finalizedAt: params.finalizedAt,
    };
    this.db
      .prepare(
        `UPDATE retrieval_audit
            SET selected_items_json = ?,
                injected_chars = ?
          WHERE audit_id = ?`,
      )
      .run(JSON.stringify(selectedItems), params.actualInjectedChars, row.audit_id);
  }

  recordSignal(signal: MemorySignalEventRecord): void {
    this.db
      .prepare(
        `INSERT OR IGNORE INTO memory_signal_events(
          signal_id, agent_id, scope, session_key, signal_type, memory_kind, content_ref, semantic_key, value, source_ref, metadata_json, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
      )
      .run(
        signal.signalId,
        signal.agentId,
        signal.scope,
        signal.sessionKey ?? null,
        signal.signalType,
        signal.memoryKind,
        signal.contentRef ?? null,
        signal.semanticKey,
        signal.value,
        signal.sourceRef,
        JSON.stringify(signal.metadataJson),
        signal.createdAt,
      );
  }

  recordSemanticWriteAttemptStart(params: {
    agentId: string;
    sessionKey?: string;
    scope: string;
    turnId: string;
    jobType?: SemanticWriteJobRecord["jobType"];
    sourceRefs: string[];
    inputHash: string;
    startedAt: string;
    retryOnly?: boolean;
  }): { jobId: string; attemptCount: number; claimed: boolean } {
    const sessionKey = params.sessionKey ?? "default";
    const jobType = params.jobType ?? "turn_semantic_extraction";
    const jobId = this.semanticWriteJobId({
      agentId: params.agentId,
      sessionKey,
      turnId: params.turnId,
      jobType,
    });
    const sourceRefs = [...new Set(params.sourceRefs.filter((entry) => entry.trim()))];
    const existing = this.db
      .prepare(
        `SELECT status, attempt_count
           FROM semantic_write_jobs
          WHERE job_id = ?`,
      )
      .get(jobId) as { status: SemanticWriteJobStatus; attempt_count: number } | undefined;
    if (existing?.status === "succeeded") {
      return { jobId, attemptCount: existing.attempt_count, claimed: false };
    }
    if (params.retryOnly && existing?.status !== "pending" && existing?.status !== "retrying") {
      return { jobId, attemptCount: existing?.attempt_count ?? 0, claimed: false };
    }
    this.db
      .prepare(
        `INSERT INTO semantic_write_jobs(
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
          updated_at = excluded.updated_at`,
      )
      .run(
        jobId,
        params.agentId,
        sessionKey,
        params.scope,
        params.turnId,
        jobType,
        JSON.stringify(sourceRefs),
        params.inputHash,
        params.startedAt,
        params.startedAt,
      );
    const row = this.db
      .prepare(`SELECT attempt_count FROM semantic_write_jobs WHERE job_id = ?`)
      .get(jobId) as { attempt_count: number } | undefined;
    return { jobId, attemptCount: row?.attempt_count ?? 1, claimed: true };
  }

  finishSemanticWriteAttempt(params: {
    agentId: string;
    sessionKey?: string;
    turnId: string;
    jobType?: SemanticWriteJobRecord["jobType"];
    status: Exclude<SemanticWriteJobStatus, "pending" | "running">;
    error?: string;
    resultJson?: Record<string, unknown>;
    nextAttemptAt?: string;
    completedAt: string;
  }): void {
    const sessionKey = params.sessionKey ?? "default";
    const jobType = params.jobType ?? "turn_semantic_extraction";
    const jobId = this.semanticWriteJobId({
      agentId: params.agentId,
      sessionKey,
      turnId: params.turnId,
      jobType,
    });
    this.db
      .prepare(
        `UPDATE semantic_write_jobs
            SET status = ?,
                last_error = ?,
                result_json = ?,
                next_attempt_at = ?,
                updated_at = ?
          WHERE job_id = ?`,
      )
      .run(
        params.status,
        params.error ?? null,
        JSON.stringify(params.resultJson ?? {}),
        params.nextAttemptAt ?? null,
        params.completedAt,
        jobId,
      );
  }

  listSemanticWriteJobs(params: {
    agentId: string;
    sessionKey?: string;
    statuses?: SemanticWriteJobStatus[];
    limit?: number;
  }): SemanticWriteJobRecord[] {
    const limit = Math.max(1, Math.min(Math.trunc(params.limit ?? 50), 200));
    const values: Array<string | number> = [params.agentId];
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
    return this.db
      .prepare(sql)
      .all(...values)
      .map((row) => {
        const record = row as {
          job_id: string;
          agent_id: string;
          session_key: string;
          scope: string;
          turn_id: string;
          job_type: SemanticWriteJobRecord["jobType"];
          source_refs_json: string;
          input_hash: string;
          status: SemanticWriteJobStatus;
          attempt_count: number;
          last_error: string | null;
          result_json: string;
          next_attempt_at: string | null;
          created_at: string;
          updated_at: string;
        };
        return {
          jobId: record.job_id,
          agentId: record.agent_id,
          sessionKey: record.session_key,
          scope: record.scope,
          turnId: record.turn_id,
          jobType: record.job_type,
          sourceRefs: safeJsonParse<string[]>(record.source_refs_json, []),
          inputHash: record.input_hash,
          status: record.status,
          attemptCount: record.attempt_count,
          lastError: record.last_error ?? undefined,
          resultJson: safeJsonParse<Record<string, unknown>>(record.result_json, {}),
          nextAttemptAt: record.next_attempt_at ?? undefined,
          createdAt: record.created_at,
          updatedAt: record.updated_at,
        } satisfies SemanticWriteJobRecord;
      });
  }

  listRetryableSemanticWriteJobs(params: {
    agentId: string;
    sessionKey?: string;
    now: string;
    limit?: number;
  }): SemanticWriteJobRecord[] {
    const limit = Math.max(1, Math.min(Math.trunc(params.limit ?? 50), 200));
    const values: Array<string | number> = [params.agentId];
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
    return this.db
      .prepare(sql)
      .all(...values)
      .map((row) => {
        const record = row as {
          job_id: string;
          agent_id: string;
          session_key: string;
          scope: string;
          turn_id: string;
          job_type: SemanticWriteJobRecord["jobType"];
          source_refs_json: string;
          input_hash: string;
          status: SemanticWriteJobStatus;
          attempt_count: number;
          last_error: string | null;
          result_json: string;
          next_attempt_at: string | null;
          created_at: string;
          updated_at: string;
        };
        return {
          jobId: record.job_id,
          agentId: record.agent_id,
          sessionKey: record.session_key,
          scope: record.scope,
          turnId: record.turn_id,
          jobType: record.job_type,
          sourceRefs: safeJsonParse<string[]>(record.source_refs_json, []),
          inputHash: record.input_hash,
          status: record.status,
          attemptCount: record.attempt_count,
          lastError: record.last_error ?? undefined,
          resultJson: safeJsonParse<Record<string, unknown>>(record.result_json, {}),
          nextAttemptAt: record.next_attempt_at ?? undefined,
          createdAt: record.created_at,
          updatedAt: record.updated_at,
        } satisfies SemanticWriteJobRecord;
      });
  }

  listSignals(params: {
    agentId: string;
    sessionKey?: string;
    signalTypes?: MemorySignalEventRecord["signalType"][];
    after?: string;
    until?: string;
    limit?: number;
  }): MemorySignalEventRecord[] {
    const values: Array<string | number | null> = [params.agentId];
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
    if (params.limit) {
      sql += ` LIMIT ${Math.max(1, Math.trunc(params.limit))}`;
    }
    return this.db
      .prepare(sql)
      .all(...values)
      .map((row) => {
        const record = row as {
          signal_id: string;
          agent_id: string;
          scope: string;
          session_key: string | null;
          signal_type: MemorySignalEventRecord["signalType"];
          memory_kind: MemorySignalEventRecord["memoryKind"];
          content_ref: string | null;
          semantic_key: string;
          value: number;
          source_ref: string;
          metadata_json: string;
          created_at: string;
        };
        return {
          signalId: record.signal_id,
          agentId: record.agent_id,
          scope: record.scope,
          sessionKey: record.session_key ?? undefined,
          signalType: record.signal_type,
          memoryKind: record.memory_kind,
          contentRef: record.content_ref ?? undefined,
          semanticKey: record.semantic_key,
          value: record.value,
          sourceRef: record.source_ref,
          metadataJson: safeJsonParse<Record<string, unknown>>(record.metadata_json, {}),
          createdAt: record.created_at,
        } satisfies MemorySignalEventRecord;
      });
  }

  listSignalsForTargets(params: {
    agentId: string;
    targets: Array<{
      memoryKind: MemorySignalEventRecord["memoryKind"];
      contentRef?: string;
      semanticKey: string;
    }>;
    until?: string;
  }): MemorySignalEventRecord[] {
    if (params.targets.length === 0) {
      return [];
    }
    const values: Array<string | number | null> = [params.agentId];
    const targetClauses = params.targets.map((target) => {
      values.push(target.memoryKind, target.contentRef ?? null, target.semanticKey);
      return `(
        memory_kind = ?
        AND (
          (content_ref IS NOT NULL AND content_ref = ?)
          OR (content_ref IS NULL AND semantic_key = ?)
        )
      )`;
    });
    let sql = `
      SELECT signal_id, agent_id, scope, session_key, signal_type, memory_kind, content_ref, semantic_key, value,
             source_ref, metadata_json, created_at
        FROM memory_signal_events
       WHERE agent_id = ?
         AND (${targetClauses.join(" OR ")})
    `;
    if (params.until) {
      sql += " AND created_at <= ?";
      values.push(params.until);
    }
    sql += " ORDER BY created_at ASC, rowid ASC";
    return this.db
      .prepare(sql)
      .all(...values)
      .map((row) => {
        const record = row as {
          signal_id: string;
          agent_id: string;
          scope: string;
          session_key: string | null;
          signal_type: MemorySignalEventRecord["signalType"];
          memory_kind: MemorySignalEventRecord["memoryKind"];
          content_ref: string | null;
          semantic_key: string;
          value: number;
          source_ref: string;
          metadata_json: string;
          created_at: string;
        };
        return {
          signalId: record.signal_id,
          agentId: record.agent_id,
          scope: record.scope,
          sessionKey: record.session_key ?? undefined,
          signalType: record.signal_type,
          memoryKind: record.memory_kind,
          contentRef: record.content_ref ?? undefined,
          semanticKey: record.semantic_key,
          value: record.value,
          sourceRef: record.source_ref,
          metadataJson: safeJsonParse<Record<string, unknown>>(record.metadata_json, {}),
          createdAt: record.created_at,
        } satisfies MemorySignalEventRecord;
      });
  }

  latestSignalCreatedAt(params: {
    agentId: string;
    sessionKey?: string;
  }): string | undefined {
    const values: Array<string> = [params.agentId];
    let sql = `
      SELECT MAX(created_at) AS createdAt
        FROM memory_signal_events
       WHERE agent_id = ?
    `;
    if (params.sessionKey) {
      sql += " AND session_key = ?";
      values.push(params.sessionKey);
    }
    const row = this.db.prepare(sql).get(...values) as { createdAt: string | null } | undefined;
    return row?.createdAt ?? undefined;
  }

  listRetrievals(params: { agentId: string; sessionKey?: string; limit?: number }): RetrievalAuditRecord[] {
    const limit = Math.max(1, Math.min(Math.trunc(params.limit ?? 50), 200));
    const values: string[] = [params.agentId];
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
    return this.db
      .prepare(sql)
      .all(...values)
      .map((row) => {
        const record = row as {
          audit_id: string;
          agent_id: string;
          session_key: string | null;
          scope: string;
          route_type: RetrievalAuditRecord["routeType"];
          query_text: string;
          query_hash: string;
          selected_items_json: string;
          injected_chars: number;
          created_at: string;
        };
        return {
          auditId: record.audit_id,
          agentId: record.agent_id,
          sessionKey: record.session_key ?? undefined,
          scope: record.scope,
          routeType: record.route_type,
          queryText: record.query_text,
          queryHash: record.query_hash,
          selectedItemsJson: safeJsonParse<Record<string, unknown>>(
            record.selected_items_json,
            {},
          ),
          injectedChars: record.injected_chars,
          createdAt: record.created_at,
        } satisfies RetrievalAuditRecord;
      });
  }

  listPolicyDecisions(params: { agentId: string; sessionKey?: string; limit?: number }): Array<Record<string, unknown>> {
    const limit = Math.max(1, Math.min(Math.trunc(params.limit ?? 50), 200));
    const values: string[] = [params.agentId];
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
    return this.db
      .prepare(sql)
      .all(...values)
      .map((row) => {
        const record = row as {
          decision_id: string;
          agent_id: string;
          session_key: string | null;
          source_ref: string;
          candidate_hash: string;
          salience_score: number;
          utility_score: number;
          chosen_action: string;
          reasons_json: string;
          created_at: string;
          metadata_json: string;
        };
        return {
          decisionId: record.decision_id,
          agentId: record.agent_id,
          sessionKey: record.session_key ?? undefined,
          sourceRef: record.source_ref,
          candidateHash: record.candidate_hash,
          salienceScore: record.salience_score,
          utilityScore: record.utility_score,
          chosenAction: record.chosen_action,
          reasons: safeJsonParse<string[]>(record.reasons_json, []),
          metadataJson: safeJsonParse<Record<string, unknown>>(record.metadata_json, {}),
          createdAt: record.created_at,
        };
      });
  }

  listMaintenanceRuns(params: { agentId: string; sessionKey?: string; limit?: number }): MaintenanceRunRecord[] {
    const limit = Math.max(1, Math.min(Math.trunc(params.limit ?? 50), 200));
    const values: string[] = [params.agentId];
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
    return this.db
      .prepare(sql)
      .all(...values)
      .map((row) => {
        const record = row as {
          run_id: string;
          agent_id: string;
          session_key: string | null;
          job_type: string;
          stats_json: string;
          started_at: string;
          completed_at: string | null;
          status: MaintenanceRunRecord["status"];
        };
        return {
          runId: record.run_id,
          agentId: record.agent_id,
          sessionKey: record.session_key ?? undefined,
          jobType: record.job_type,
          statsJson: safeJsonParse<Record<string, unknown>>(record.stats_json, {}),
          startedAt: record.started_at,
          completedAt: record.completed_at ?? undefined,
          status: record.status,
        } satisfies MaintenanceRunRecord;
      });
  }

  markExpiredRunningMaintenanceRunsInterrupted(params: {
    agentId: string;
    completedAt: string;
    startedBefore: string;
    reason: string;
  }): number {
    const rows = this.db
      .prepare(
        `SELECT run_id, stats_json
           FROM maintenance_runs
          WHERE agent_id = ?
            AND status = 'running'
            AND started_at <= ?`,
      )
      .all(params.agentId, params.startedBefore) as Array<{
      run_id: string;
      stats_json: string;
    }>;
    if (rows.length === 0) {
      return 0;
    }
    const update = this.db.prepare(
      `UPDATE maintenance_runs
          SET stats_json = ?, completed_at = ?, status = 'failed'
        WHERE run_id = ?`,
    );
    for (const row of rows) {
      const stats = safeJsonParse<Record<string, unknown>>(row.stats_json, {});
      update.run(
        JSON.stringify({
          ...stats,
          recovery: {
            reason: params.reason,
            recoveredAt: params.completedAt,
          },
        }),
        params.completedAt,
        row.run_id,
      );
    }
    return rows.length;
  }

  startMaintenance(params: {
    agentId: string;
    sessionKey?: string;
    jobType: string;
    stats: Record<string, unknown>;
    startedAt: string;
  }): string {
    const runId = randomId("maintenance");
    this.db
      .prepare(
        `INSERT INTO maintenance_runs(
          run_id, agent_id, session_key, job_type, stats_json, started_at, completed_at, status
        ) VALUES (?, ?, ?, ?, ?, ?, NULL, 'running')`,
      )
      .run(runId, params.agentId, params.sessionKey ?? null, params.jobType, JSON.stringify(params.stats), params.startedAt);
    return runId;
  }

  finishMaintenance(run: MaintenanceRunRecord): void {
    this.db
      .prepare(
        `UPDATE maintenance_runs
            SET stats_json = ?, completed_at = ?, status = ?
          WHERE run_id = ?`,
      )
      .run(JSON.stringify(run.statsJson), run.completedAt ?? null, run.status, run.runId);
  }
}
