//#region src/db/migrations.ts
const MEMX_MIGRATIONS = [
	{
		version: 1,
		description: "initial memx schema",
		sql: `
      CREATE TABLE IF NOT EXISTS schema_migrations (
        version INTEGER PRIMARY KEY,
        description TEXT NOT NULL,
        applied_at TEXT NOT NULL
      );

      CREATE TABLE IF NOT EXISTS state_kv (
        key TEXT NOT NULL,
        value_json TEXT NOT NULL,
        scope TEXT NOT NULL,
        agent_id TEXT NOT NULL,
        state_kind TEXT NOT NULL CHECK (state_kind IN ('session', 'durable')),
        confidence REAL NOT NULL,
        source_ref TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        expires_at TEXT,
        PRIMARY KEY (agent_id, scope, key)
      );

      CREATE TABLE IF NOT EXISTS facts (
        fact_id TEXT PRIMARY KEY,
        canonical_subject TEXT NOT NULL,
        predicate TEXT NOT NULL,
        canonical_object TEXT,
        object_value_json TEXT,
        scope TEXT NOT NULL,
        agent_id TEXT NOT NULL,
        confidence REAL NOT NULL,
        status TEXT NOT NULL CHECK (status IN ('active', 'superseded', 'deleted', 'uncertain')),
        valid_from TEXT,
        valid_to TEXT,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
      );

      CREATE TABLE IF NOT EXISTS fact_versions (
        version_id TEXT PRIMARY KEY,
        fact_id TEXT NOT NULL,
        prior_snapshot_json TEXT NOT NULL,
        changed_at TEXT NOT NULL,
        change_reason TEXT NOT NULL,
        source_ref TEXT NOT NULL,
        FOREIGN KEY (fact_id) REFERENCES facts(fact_id) ON DELETE CASCADE
      );

      CREATE TABLE IF NOT EXISTS episodic_events (
        event_id TEXT PRIMARY KEY,
        agent_id TEXT NOT NULL,
        scope TEXT NOT NULL,
        event_type TEXT NOT NULL,
        text TEXT NOT NULL,
        normalized_text TEXT NOT NULL,
        observed_at TEXT NOT NULL,
        valid_from TEXT,
        valid_to TEXT,
        source_kind TEXT NOT NULL,
        source_ref TEXT NOT NULL,
        session_key TEXT,
        tool_name TEXT,
        confidence REAL NOT NULL,
        metadata_json TEXT NOT NULL
      );

      CREATE TABLE IF NOT EXISTS entities (
        entity_id TEXT PRIMARY KEY,
        canonical_name TEXT NOT NULL,
        entity_type TEXT NOT NULL,
        normalized_name TEXT NOT NULL,
        aliases_json TEXT NOT NULL,
        confidence REAL NOT NULL,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
      );

      CREATE TABLE IF NOT EXISTS entity_aliases (
        alias_id TEXT PRIMARY KEY,
        entity_id TEXT NOT NULL,
        alias_text TEXT NOT NULL,
        normalized_alias TEXT NOT NULL,
        FOREIGN KEY (entity_id) REFERENCES entities(entity_id) ON DELETE CASCADE
      );

      CREATE TABLE IF NOT EXISTS graph_edges (
        edge_id TEXT PRIMARY KEY,
        src_entity_id TEXT NOT NULL,
        rel_type TEXT NOT NULL,
        dst_entity_id TEXT NOT NULL,
        scope TEXT NOT NULL,
        agent_id TEXT NOT NULL,
        confidence REAL NOT NULL,
        valid_from TEXT,
        valid_to TEXT,
        evidence_ref TEXT NOT NULL,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        FOREIGN KEY (src_entity_id) REFERENCES entities(entity_id) ON DELETE CASCADE,
        FOREIGN KEY (dst_entity_id) REFERENCES entities(entity_id) ON DELETE CASCADE
      );

      CREATE TABLE IF NOT EXISTS vector_docs (
        doc_id TEXT PRIMARY KEY,
        doc_kind TEXT NOT NULL CHECK (doc_kind IN ('fact', 'event', 'edge', 'state')),
        source_id TEXT NOT NULL,
        scope TEXT NOT NULL,
        agent_id TEXT NOT NULL,
        text TEXT NOT NULL,
        metadata_json TEXT NOT NULL,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
      );

      CREATE TABLE IF NOT EXISTS vector_embeddings (
        doc_id TEXT PRIMARY KEY,
        agent_id TEXT NOT NULL,
        scope TEXT NOT NULL,
        dimensions INTEGER NOT NULL,
        embedding_json TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        FOREIGN KEY (doc_id) REFERENCES vector_docs(doc_id) ON DELETE CASCADE
      );

      CREATE TABLE IF NOT EXISTS retrieval_audit (
        audit_id TEXT PRIMARY KEY,
        agent_id TEXT NOT NULL,
        scope TEXT NOT NULL,
        route_type TEXT NOT NULL,
        query_text TEXT NOT NULL,
        query_hash TEXT NOT NULL,
        selected_items_json TEXT NOT NULL,
        injected_chars INTEGER NOT NULL,
        created_at TEXT NOT NULL
      );

      CREATE TABLE IF NOT EXISTS policy_decisions (
        decision_id TEXT PRIMARY KEY,
        agent_id TEXT NOT NULL,
        source_ref TEXT NOT NULL,
        candidate_hash TEXT NOT NULL,
        salience_score REAL NOT NULL,
        utility_score REAL NOT NULL,
        chosen_action TEXT NOT NULL,
        reasons_json TEXT NOT NULL,
        created_at TEXT NOT NULL
      );

      CREATE TABLE IF NOT EXISTS maintenance_runs (
        run_id TEXT PRIMARY KEY,
        agent_id TEXT NOT NULL,
        job_type TEXT NOT NULL,
        stats_json TEXT NOT NULL,
        started_at TEXT NOT NULL,
        completed_at TEXT,
        status TEXT NOT NULL
      );

      CREATE INDEX IF NOT EXISTS idx_state_scope_agent ON state_kv(agent_id, scope, updated_at);
      CREATE INDEX IF NOT EXISTS idx_state_expires ON state_kv(expires_at);
      CREATE INDEX IF NOT EXISTS idx_fact_scope_agent ON facts(agent_id, scope, predicate, status);
      CREATE INDEX IF NOT EXISTS idx_event_scope_agent ON episodic_events(agent_id, scope, observed_at);
      CREATE INDEX IF NOT EXISTS idx_event_norm ON episodic_events(agent_id, scope, normalized_text, observed_at);
      CREATE INDEX IF NOT EXISTS idx_entity_norm ON entities(normalized_name, entity_type);
      CREATE INDEX IF NOT EXISTS idx_alias_norm ON entity_aliases(normalized_alias);
      CREATE INDEX IF NOT EXISTS idx_edge_scope_agent ON graph_edges(agent_id, scope, rel_type, updated_at);
      CREATE INDEX IF NOT EXISTS idx_vector_scope_agent ON vector_docs(agent_id, scope, doc_kind, updated_at);
      CREATE INDEX IF NOT EXISTS idx_embedding_scope_agent ON vector_embeddings(agent_id, scope, updated_at);
      CREATE INDEX IF NOT EXISTS idx_retrieval_audit_agent ON retrieval_audit(agent_id, created_at);
      CREATE INDEX IF NOT EXISTS idx_policy_agent ON policy_decisions(agent_id, created_at);
      CREATE INDEX IF NOT EXISTS idx_maintenance_agent ON maintenance_runs(agent_id, started_at);

      CREATE VIRTUAL TABLE IF NOT EXISTS vector_docs_fts USING fts5(
        doc_id UNINDEXED,
        doc_kind UNINDEXED,
        source_id UNINDEXED,
        scope UNINDEXED,
        agent_id UNINDEXED,
        text
      );
    `
	},
	{
		version: 2,
		description: "conversation chunks and tasks",
		sql: `
      CREATE TABLE IF NOT EXISTS conversation_tasks (
        task_id TEXT PRIMARY KEY,
        agent_id TEXT NOT NULL,
        scope TEXT NOT NULL,
        session_key TEXT NOT NULL,
        title TEXT NOT NULL,
        summary TEXT NOT NULL,
        status TEXT NOT NULL CHECK (status IN ('active', 'completed', 'skipped')),
        started_at TEXT NOT NULL,
        ended_at TEXT,
        updated_at TEXT NOT NULL,
        metadata_json TEXT NOT NULL DEFAULT '{}'
      );

      CREATE TABLE IF NOT EXISTS conversation_chunks (
        chunk_id TEXT PRIMARY KEY,
        agent_id TEXT NOT NULL,
        scope TEXT NOT NULL,
        session_key TEXT NOT NULL,
        turn_id TEXT NOT NULL,
        seq INTEGER NOT NULL,
        role TEXT NOT NULL CHECK (role IN ('user', 'assistant', 'tool')),
        tool_name TEXT,
        chunk_kind TEXT NOT NULL DEFAULT 'message',
        content TEXT NOT NULL,
        summary TEXT NOT NULL,
        content_hash TEXT NOT NULL,
        task_id TEXT,
        dedup_status TEXT NOT NULL CHECK (dedup_status IN ('active', 'duplicate', 'merged')),
        dedup_target TEXT,
        dedup_reason TEXT,
        merge_count INTEGER NOT NULL DEFAULT 0,
        last_hit_at TEXT,
        source_ref TEXT NOT NULL,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        FOREIGN KEY (task_id) REFERENCES conversation_tasks(task_id) ON DELETE SET NULL
      );

      CREATE INDEX IF NOT EXISTS idx_conv_tasks_agent_scope_status
        ON conversation_tasks(agent_id, scope, status, updated_at);
      CREATE INDEX IF NOT EXISTS idx_conv_tasks_session
        ON conversation_tasks(agent_id, session_key, updated_at);
      CREATE INDEX IF NOT EXISTS idx_conv_chunks_agent_scope_created
        ON conversation_chunks(agent_id, scope, created_at);
      CREATE INDEX IF NOT EXISTS idx_conv_chunks_session_turn
        ON conversation_chunks(agent_id, session_key, turn_id, seq);
      CREATE INDEX IF NOT EXISTS idx_conv_chunks_hash
        ON conversation_chunks(agent_id, scope, role, content_hash);
      CREATE INDEX IF NOT EXISTS idx_conv_chunks_task
        ON conversation_chunks(task_id, created_at);
      CREATE INDEX IF NOT EXISTS idx_conv_chunks_dedup
        ON conversation_chunks(agent_id, scope, dedup_status, updated_at);
    `
	},
	{
		version: 3,
		description: "self-learning signal ledger",
		sql: `
      CREATE TABLE IF NOT EXISTS memory_signal_events (
        signal_id TEXT PRIMARY KEY,
        agent_id TEXT NOT NULL,
        scope TEXT NOT NULL,
        signal_type TEXT NOT NULL,
        memory_kind TEXT NOT NULL,
        content_ref TEXT,
        semantic_key TEXT NOT NULL,
        value REAL NOT NULL,
        source_ref TEXT NOT NULL,
        metadata_json TEXT NOT NULL,
        created_at TEXT NOT NULL
      );

      CREATE INDEX IF NOT EXISTS idx_signal_events_agent_type_created
        ON memory_signal_events(agent_id, signal_type, created_at);
      CREATE INDEX IF NOT EXISTS idx_signal_events_agent_semantic_created
        ON memory_signal_events(agent_id, semantic_key, created_at);
    `
	},
	{
		version: 4,
		description: "memory beliefs",
		sql: `
      CREATE TABLE IF NOT EXISTS memory_beliefs (
        belief_id TEXT PRIMARY KEY,
        agent_id TEXT NOT NULL,
        scope TEXT NOT NULL,
        memory_kind TEXT NOT NULL,
        content_ref TEXT,
        semantic_key TEXT NOT NULL,
        stage TEXT NOT NULL CHECK (stage IN ('candidate', 'probationary', 'active', 'decaying', 'superseded', 'quarantined')),
        prior_confidence REAL NOT NULL,
        posterior_confidence REAL NOT NULL,
        usefulness_score REAL NOT NULL,
        stability_score REAL NOT NULL,
        contradiction_score REAL NOT NULL,
        outcome_support_score REAL NOT NULL,
        source_reliability REAL NOT NULL,
        first_seen_at TEXT NOT NULL,
        last_seen_at TEXT NOT NULL,
        last_used_at TEXT,
        use_count INTEGER NOT NULL DEFAULT 0,
        reevaluation_due_at TEXT,
        metadata_json TEXT NOT NULL,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
      );

      CREATE INDEX IF NOT EXISTS idx_memory_beliefs_agent_stage_updated
        ON memory_beliefs(agent_id, stage, updated_at);
      CREATE INDEX IF NOT EXISTS idx_memory_beliefs_agent_semantic
        ON memory_beliefs(agent_id, semantic_key, updated_at);
      CREATE INDEX IF NOT EXISTS idx_memory_beliefs_agent_content
        ON memory_beliefs(agent_id, memory_kind, content_ref, updated_at);
    `
	},
	{
		version: 5,
		description: "strategy hypotheses",
		sql: `
      CREATE TABLE IF NOT EXISTS strategy_hypotheses (
        strategy_id TEXT PRIMARY KEY,
        agent_id TEXT NOT NULL,
        scope TEXT NOT NULL,
        domain_key TEXT NOT NULL,
        summary TEXT NOT NULL,
        support_belief_ids_json TEXT NOT NULL,
        support_task_ids_json TEXT NOT NULL,
        confidence REAL NOT NULL,
        usefulness_score REAL NOT NULL,
        stability_score REAL NOT NULL,
        contradiction_score REAL NOT NULL,
        stage TEXT NOT NULL CHECK (stage IN ('candidate', 'active', 'superseded', 'quarantined')),
        metadata_json TEXT NOT NULL,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
      );

      CREATE INDEX IF NOT EXISTS idx_strategy_hypotheses_agent_stage_updated
        ON strategy_hypotheses(agent_id, stage, updated_at);
      CREATE INDEX IF NOT EXISTS idx_strategy_hypotheses_agent_domain
        ON strategy_hypotheses(agent_id, scope, domain_key, updated_at);
    `
	},
	{
		version: 6,
		description: "abstraction candidates",
		sql: `
      CREATE TABLE IF NOT EXISTS abstraction_candidates (
        candidate_id TEXT PRIMARY KEY,
        agent_id TEXT NOT NULL,
        scope TEXT NOT NULL,
        abstraction_type TEXT NOT NULL CHECK (abstraction_type IN ('derived_state', 'workflow_pattern', 'concept_candidate')),
        semantic_key TEXT NOT NULL,
        summary TEXT NOT NULL,
        support_content_refs_json TEXT NOT NULL,
        support_belief_ids_json TEXT NOT NULL,
        confidence REAL NOT NULL,
        usefulness_score REAL NOT NULL,
        stability_score REAL NOT NULL,
        contradiction_score REAL NOT NULL,
        stage TEXT NOT NULL CHECK (stage IN ('candidate', 'probationary', 'active', 'decaying', 'superseded', 'quarantined')),
        metadata_json TEXT NOT NULL,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
      );

      CREATE INDEX IF NOT EXISTS idx_abstraction_candidates_agent_stage_updated
        ON abstraction_candidates(agent_id, stage, updated_at);
      CREATE INDEX IF NOT EXISTS idx_abstraction_candidates_agent_type
        ON abstraction_candidates(agent_id, abstraction_type, updated_at);
      CREATE INDEX IF NOT EXISTS idx_abstraction_candidates_agent_semantic
        ON abstraction_candidates(agent_id, semantic_key, updated_at);
    `
	},
	{
		version: 7,
		description: "extend abstraction candidates with graph hypotheses",
		sql: `
      ALTER TABLE abstraction_candidates RENAME TO abstraction_candidates_v6;

      CREATE TABLE abstraction_candidates (
        candidate_id TEXT PRIMARY KEY,
        agent_id TEXT NOT NULL,
        scope TEXT NOT NULL,
        abstraction_type TEXT NOT NULL CHECK (abstraction_type IN ('derived_state', 'workflow_pattern', 'concept_candidate', 'graph_hypothesis')),
        semantic_key TEXT NOT NULL,
        summary TEXT NOT NULL,
        support_content_refs_json TEXT NOT NULL,
        support_belief_ids_json TEXT NOT NULL,
        confidence REAL NOT NULL,
        usefulness_score REAL NOT NULL,
        stability_score REAL NOT NULL,
        contradiction_score REAL NOT NULL,
        stage TEXT NOT NULL CHECK (stage IN ('candidate', 'probationary', 'active', 'decaying', 'superseded', 'quarantined')),
        metadata_json TEXT NOT NULL,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
      );

      INSERT INTO abstraction_candidates(
        candidate_id, agent_id, scope, abstraction_type, semantic_key, summary,
        support_content_refs_json, support_belief_ids_json, confidence, usefulness_score,
        stability_score, contradiction_score, stage, metadata_json, created_at, updated_at
      )
      SELECT
        candidate_id, agent_id, scope, abstraction_type, semantic_key, summary,
        support_content_refs_json, support_belief_ids_json, confidence, usefulness_score,
        stability_score, contradiction_score, stage, metadata_json, created_at, updated_at
      FROM abstraction_candidates_v6;

      DROP TABLE abstraction_candidates_v6;

      CREATE INDEX IF NOT EXISTS idx_abstraction_candidates_agent_stage_updated
        ON abstraction_candidates(agent_id, stage, updated_at);
      CREATE INDEX IF NOT EXISTS idx_abstraction_candidates_agent_type
        ON abstraction_candidates(agent_id, abstraction_type, updated_at);
      CREATE INDEX IF NOT EXISTS idx_abstraction_candidates_agent_semantic
        ON abstraction_candidates(agent_id, semantic_key, updated_at);
    `
	},
	{
		version: 8,
		description: "extend abstraction candidates with outcome hypotheses",
		sql: `
      ALTER TABLE abstraction_candidates RENAME TO abstraction_candidates_v7;

      CREATE TABLE abstraction_candidates (
        candidate_id TEXT PRIMARY KEY,
        agent_id TEXT NOT NULL,
        scope TEXT NOT NULL,
        abstraction_type TEXT NOT NULL CHECK (abstraction_type IN ('derived_state', 'workflow_pattern', 'concept_candidate', 'graph_hypothesis', 'outcome_hypothesis')),
        semantic_key TEXT NOT NULL,
        summary TEXT NOT NULL,
        support_content_refs_json TEXT NOT NULL,
        support_belief_ids_json TEXT NOT NULL,
        confidence REAL NOT NULL,
        usefulness_score REAL NOT NULL,
        stability_score REAL NOT NULL,
        contradiction_score REAL NOT NULL,
        stage TEXT NOT NULL CHECK (stage IN ('candidate', 'probationary', 'active', 'decaying', 'superseded', 'quarantined')),
        metadata_json TEXT NOT NULL,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
      );

      INSERT INTO abstraction_candidates(
        candidate_id, agent_id, scope, abstraction_type, semantic_key, summary,
        support_content_refs_json, support_belief_ids_json, confidence, usefulness_score,
        stability_score, contradiction_score, stage, metadata_json, created_at, updated_at
      )
      SELECT
        candidate_id, agent_id, scope, abstraction_type, semantic_key, summary,
        support_content_refs_json, support_belief_ids_json, confidence, usefulness_score,
        stability_score, contradiction_score, stage, metadata_json, created_at, updated_at
      FROM abstraction_candidates_v7;

      DROP TABLE abstraction_candidates_v7;

      CREATE INDEX IF NOT EXISTS idx_abstraction_candidates_agent_stage_updated
        ON abstraction_candidates(agent_id, stage, updated_at);
      CREATE INDEX IF NOT EXISTS idx_abstraction_candidates_agent_type
        ON abstraction_candidates(agent_id, abstraction_type, updated_at);
      CREATE INDEX IF NOT EXISTS idx_abstraction_candidates_agent_semantic
        ON abstraction_candidates(agent_id, semantic_key, updated_at);
    `
	},
	{
		version: 9,
		description: "FTS5 sync triggers and missing FK indexes",
		sql: `
      -- Safety-net trigger: keep FTS in sync when vector_docs rows are deleted
      -- outside the application layer (e.g. direct SQL, CASCADE).
      -- INSERT/UPDATE sync is handled in-app by VectorRepo to avoid double-insertion
      -- with SQLite's ON CONFLICT DO UPDATE trigger semantics.
      CREATE TRIGGER IF NOT EXISTS trg_vector_docs_fts_delete
        AFTER DELETE ON vector_docs
      BEGIN
        DELETE FROM vector_docs_fts WHERE doc_id = OLD.doc_id;
      END;

      -- FK indexes for DELETE CASCADE / JOIN performance.
      CREATE INDEX IF NOT EXISTS idx_fact_versions_fact_id
        ON fact_versions(fact_id);
      CREATE INDEX IF NOT EXISTS idx_entity_aliases_entity_id
        ON entity_aliases(entity_id);
    `
	},
	{
		version: 10,
		description: "add predicate_verb and predicate_topic columns to facts",
		sql: `
      ALTER TABLE facts ADD COLUMN predicate_verb TEXT;
      ALTER TABLE facts ADD COLUMN predicate_topic TEXT;

      -- Backfill: extract canonical verb prefix and topic from existing predicates.
      -- predicate_verb = first underscore-delimited token if it is a canonical verb.
      -- predicate_topic = remaining tokens with underscores replaced by spaces.
      UPDATE facts
      SET
        predicate_verb = CASE
          WHEN instr(predicate, '_') > 0
           AND substr(predicate, 1, instr(predicate, '_') - 1)
               IN ('prefers','uses','has','depends')
          THEN substr(predicate, 1, instr(predicate, '_') - 1)
          ELSE NULL
        END,
        predicate_topic = CASE
          WHEN instr(predicate, '_') > 0
           AND substr(predicate, 1, instr(predicate, '_') - 1)
               IN ('prefers','uses','has','depends')
          THEN replace(substr(predicate, instr(predicate, '_') + 1), '_', ' ')
          ELSE NULL
        END;

      -- Index for O(log n) verb-gated supersession lookup.
      CREATE INDEX IF NOT EXISTS idx_facts_verb_subject
        ON facts(agent_id, scope, canonical_subject, predicate_verb);
    `
	},
	{
		version: 11,
		description: "canonicalize legacy preference predicates in facts",
		sql: `
      -- Retire legacy preference predicate names so new and historical facts
      -- participate in the same verb/topic supersession family.
      UPDATE facts
      SET predicate = CASE
        WHEN predicate = 'prefers_style' THEN 'prefers_response_style'
        WHEN predicate = 'prefers_charset' THEN 'prefers_output_charset'
        ELSE predicate
      END
      WHERE predicate IN ('prefers_style', 'prefers_charset');

      -- Recompute derived columns after canonicalization and fill any legacy NULLs.
      UPDATE facts
      SET
        predicate_verb = CASE
          WHEN instr(predicate, '_') > 0
           AND substr(predicate, 1, instr(predicate, '_') - 1)
               IN ('prefers','uses','has','depends')
          THEN substr(predicate, 1, instr(predicate, '_') - 1)
          ELSE NULL
        END,
        predicate_topic = CASE
          WHEN instr(predicate, '_') > 0
           AND substr(predicate, 1, instr(predicate, '_') - 1)
               IN ('prefers','uses','has','depends')
          THEN replace(substr(predicate, instr(predicate, '_') + 1), '_', ' ')
          ELSE NULL
        END
      WHERE predicate_verb IS NULL
         OR predicate_topic IS NULL
         OR predicate IN ('prefers_response_style', 'prefers_output_charset');
    `
	},
	{
		version: 12,
		description: "add relation_slot to graph edges for role-aware replacements",
		sql: `
      ALTER TABLE graph_edges ADD COLUMN relation_slot TEXT;

      CREATE INDEX IF NOT EXISTS idx_graph_edges_slot
        ON graph_edges(agent_id, scope, src_entity_id, rel_type, relation_slot, updated_at);
    `
	},
	{
		version: 13,
		description: "memory epochs and promotion lineage",
		sql: `
      CREATE TABLE IF NOT EXISTS memory_epoch_heads (
        agent_id TEXT PRIMARY KEY,
        current_epoch INTEGER NOT NULL DEFAULT 0,
        updated_at TEXT NOT NULL
      );

      INSERT OR IGNORE INTO memory_epoch_heads(agent_id, current_epoch, updated_at)
      SELECT agent_id, 0, CURRENT_TIMESTAMP
      FROM (
        SELECT agent_id FROM state_kv
        UNION
        SELECT agent_id FROM facts
        UNION
        SELECT agent_id FROM episodic_events
        UNION
        SELECT agent_id FROM graph_edges
        UNION
        SELECT agent_id FROM vector_docs
        UNION
        SELECT agent_id FROM memory_beliefs
        UNION
        SELECT agent_id FROM strategy_hypotheses
        UNION
        SELECT agent_id FROM abstraction_candidates
      );

      ALTER TABLE state_kv ADD COLUMN materialized_epoch INTEGER NOT NULL DEFAULT 0;
      ALTER TABLE facts ADD COLUMN materialized_epoch INTEGER NOT NULL DEFAULT 0;
      ALTER TABLE episodic_events ADD COLUMN materialized_epoch INTEGER NOT NULL DEFAULT 0;
      ALTER TABLE graph_edges ADD COLUMN materialized_epoch INTEGER NOT NULL DEFAULT 0;
      ALTER TABLE vector_docs ADD COLUMN materialized_epoch INTEGER NOT NULL DEFAULT 0;

      ALTER TABLE memory_beliefs ADD COLUMN derived_from_min_epoch INTEGER NOT NULL DEFAULT 0;
      ALTER TABLE memory_beliefs ADD COLUMN derived_from_max_epoch INTEGER NOT NULL DEFAULT 0;
      ALTER TABLE memory_beliefs ADD COLUMN materialized_epoch INTEGER NOT NULL DEFAULT 0;

      ALTER TABLE strategy_hypotheses ADD COLUMN derived_from_min_epoch INTEGER NOT NULL DEFAULT 0;
      ALTER TABLE strategy_hypotheses ADD COLUMN derived_from_max_epoch INTEGER NOT NULL DEFAULT 0;
      ALTER TABLE strategy_hypotheses ADD COLUMN materialized_epoch INTEGER NOT NULL DEFAULT 0;
      ALTER TABLE strategy_hypotheses ADD COLUMN derived_from_kind TEXT;
      ALTER TABLE strategy_hypotheses ADD COLUMN derived_from_ids_json TEXT NOT NULL DEFAULT '[]';
      ALTER TABLE strategy_hypotheses ADD COLUMN derived_at_epoch INTEGER NOT NULL DEFAULT 0;
      ALTER TABLE strategy_hypotheses ADD COLUMN derivation_policy_version TEXT;

      ALTER TABLE abstraction_candidates ADD COLUMN derived_from_min_epoch INTEGER NOT NULL DEFAULT 0;
      ALTER TABLE abstraction_candidates ADD COLUMN derived_from_max_epoch INTEGER NOT NULL DEFAULT 0;
      ALTER TABLE abstraction_candidates ADD COLUMN materialized_epoch INTEGER NOT NULL DEFAULT 0;
      ALTER TABLE abstraction_candidates ADD COLUMN derived_from_kind TEXT;
      ALTER TABLE abstraction_candidates ADD COLUMN derived_from_ids_json TEXT NOT NULL DEFAULT '[]';
      ALTER TABLE abstraction_candidates ADD COLUMN derived_at_epoch INTEGER NOT NULL DEFAULT 0;
      ALTER TABLE abstraction_candidates ADD COLUMN derivation_policy_version TEXT;

      CREATE INDEX IF NOT EXISTS idx_state_epoch
        ON state_kv(agent_id, scope, materialized_epoch, updated_at);
      CREATE INDEX IF NOT EXISTS idx_fact_epoch
        ON facts(agent_id, scope, materialized_epoch, updated_at);
      CREATE INDEX IF NOT EXISTS idx_event_epoch
        ON episodic_events(agent_id, scope, materialized_epoch, observed_at);
      CREATE INDEX IF NOT EXISTS idx_graph_epoch
        ON graph_edges(agent_id, scope, materialized_epoch, updated_at);
      CREATE INDEX IF NOT EXISTS idx_vector_docs_epoch
        ON vector_docs(agent_id, scope, materialized_epoch, updated_at);
      CREATE INDEX IF NOT EXISTS idx_beliefs_epoch
        ON memory_beliefs(agent_id, materialized_epoch, updated_at);
      CREATE INDEX IF NOT EXISTS idx_strategy_epoch
        ON strategy_hypotheses(agent_id, materialized_epoch, updated_at);
      CREATE INDEX IF NOT EXISTS idx_abstraction_epoch
        ON abstraction_candidates(agent_id, materialized_epoch, updated_at);
    `
	},
	{
		version: 14,
		description: "policy decision metadata audit",
		sql: `
      ALTER TABLE policy_decisions
        ADD COLUMN metadata_json TEXT NOT NULL DEFAULT '{}';
    `
	},
	{
		version: 15,
		description: "maintenance batching state and session-scoped signal cursors",
		sql: `
      ALTER TABLE memory_signal_events
        ADD COLUMN session_key TEXT;

      CREATE INDEX IF NOT EXISTS idx_signal_events_agent_session_created
        ON memory_signal_events(agent_id, session_key, created_at);

      CREATE TABLE IF NOT EXISTS maintenance_scheduler_state (
        agent_id TEXT NOT NULL,
        session_key TEXT NOT NULL,
        pending_turn_count INTEGER NOT NULL DEFAULT 0,
        pending_turn_ids_json TEXT NOT NULL DEFAULT '[]',
        first_pending_observed_at TEXT,
        last_pending_observed_at TEXT,
        inflight_turn_count INTEGER NOT NULL DEFAULT 0,
        inflight_turn_ids_json TEXT NOT NULL DEFAULT '[]',
        inflight_reason TEXT,
        inflight_started_at TEXT,
        inflight_first_observed_at TEXT,
        inflight_last_observed_at TEXT,
        last_event_watermark TEXT,
        last_signal_watermark TEXT,
        last_task_watermark TEXT,
        status TEXT NOT NULL DEFAULT 'idle',
        lease_owner TEXT,
        lease_expires_at TEXT,
        last_completed_at TEXT,
        updated_at TEXT NOT NULL,
        PRIMARY KEY(agent_id, session_key)
      );

      CREATE INDEX IF NOT EXISTS idx_maintenance_scheduler_status
        ON maintenance_scheduler_state(agent_id, status, lease_expires_at, updated_at);
    `
	},
	{
		version: 16,
		description: "entity resolution lineage",
		sql: `
      CREATE TABLE IF NOT EXISTS entity_mentions (
        mention_id TEXT PRIMARY KEY,
        agent_id TEXT NOT NULL,
        scope TEXT NOT NULL,
        raw_text TEXT NOT NULL,
        normalized_text TEXT NOT NULL,
        proposed_type TEXT NOT NULL,
        semantic_role TEXT NOT NULL,
        source_ref TEXT NOT NULL,
        support_text TEXT NOT NULL,
        session_key TEXT,
        turn_index INTEGER,
        observed_at TEXT NOT NULL,
        resolved_entity_id TEXT,
        resolution_method TEXT,
        confidence REAL NOT NULL,
        candidate_ids_json TEXT NOT NULL DEFAULT '[]',
        blockers_json TEXT NOT NULL DEFAULT '[]',
        metadata_json TEXT NOT NULL DEFAULT '{}',
        FOREIGN KEY (resolved_entity_id) REFERENCES entities(entity_id) ON DELETE SET NULL
      );

      CREATE TABLE IF NOT EXISTS entity_alias_sources (
        alias_source_id TEXT PRIMARY KEY,
        entity_id TEXT NOT NULL,
        alias_text TEXT NOT NULL,
        normalized_alias TEXT NOT NULL,
        source_ref TEXT NOT NULL,
        confidence REAL NOT NULL,
        created_at TEXT NOT NULL,
        metadata_json TEXT NOT NULL DEFAULT '{}',
        FOREIGN KEY (entity_id) REFERENCES entities(entity_id) ON DELETE CASCADE
      );

      CREATE TABLE IF NOT EXISTS entity_identity_links (
        link_id TEXT PRIMARY KEY,
        src_entity_id TEXT NOT NULL,
        dst_entity_id TEXT NOT NULL,
        link_type TEXT NOT NULL CHECK (link_type IN ('same_as', 'duplicate_of', 'possible_same_as', 'supersedes')),
        confidence REAL NOT NULL,
        evidence_ref TEXT NOT NULL,
        status TEXT NOT NULL CHECK (status IN ('active', 'rejected')),
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        metadata_json TEXT NOT NULL DEFAULT '{}',
        FOREIGN KEY (src_entity_id) REFERENCES entities(entity_id) ON DELETE CASCADE,
        FOREIGN KEY (dst_entity_id) REFERENCES entities(entity_id) ON DELETE CASCADE
      );

      CREATE INDEX IF NOT EXISTS idx_entity_mentions_agent_norm
        ON entity_mentions(agent_id, normalized_text);
      CREATE INDEX IF NOT EXISTS idx_entity_mentions_source_ref
        ON entity_mentions(source_ref);
      CREATE INDEX IF NOT EXISTS idx_entity_mentions_resolved
        ON entity_mentions(resolved_entity_id);
      CREATE INDEX IF NOT EXISTS idx_entity_alias_sources_norm
        ON entity_alias_sources(normalized_alias);
      CREATE INDEX IF NOT EXISTS idx_entity_identity_links_src
        ON entity_identity_links(src_entity_id, link_type);
      CREATE INDEX IF NOT EXISTS idx_entity_identity_links_dst
        ON entity_identity_links(dst_entity_id, link_type);
    `
	},
	{
		version: 17,
		description: "entity profile vector docs",
		sql: `
      PRAGMA foreign_keys=OFF;

      CREATE TABLE IF NOT EXISTS vector_docs_v17 (
        doc_id TEXT PRIMARY KEY,
        doc_kind TEXT NOT NULL CHECK (doc_kind IN ('fact', 'event', 'edge', 'state', 'entity_profile')),
        source_id TEXT NOT NULL,
        scope TEXT NOT NULL,
        agent_id TEXT NOT NULL,
        text TEXT NOT NULL,
        metadata_json TEXT NOT NULL,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        materialized_epoch INTEGER NOT NULL DEFAULT 0
      );

      INSERT OR IGNORE INTO vector_docs_v17(
        doc_id, doc_kind, source_id, scope, agent_id, text, metadata_json, created_at, updated_at, materialized_epoch
      )
      SELECT doc_id, doc_kind, source_id, scope, agent_id, text, metadata_json, created_at, updated_at,
             COALESCE(materialized_epoch, 0)
        FROM vector_docs;

      DROP TABLE vector_docs;
      ALTER TABLE vector_docs_v17 RENAME TO vector_docs;

      CREATE INDEX IF NOT EXISTS idx_vector_scope_agent ON vector_docs(agent_id, scope, doc_kind, updated_at);
      CREATE INDEX IF NOT EXISTS idx_vector_epoch
        ON vector_docs(agent_id, scope, materialized_epoch, updated_at);

      CREATE TRIGGER IF NOT EXISTS trg_vector_docs_fts_delete
        AFTER DELETE ON vector_docs
      BEGIN
        DELETE FROM vector_docs_fts WHERE doc_id = OLD.doc_id;
      END;

      PRAGMA foreign_keys=ON;
    `
	},
	{
		version: 18,
		description: "graph edge maintenance contract metadata",
		sql: `
      ALTER TABLE graph_edges ADD COLUMN metadata_json TEXT NOT NULL DEFAULT '{}';
    `
	},
	{
		version: 19,
		description: "source segment lineage for long captured turns",
		sql: `
      CREATE TABLE IF NOT EXISTS source_segments (
        segment_id TEXT PRIMARY KEY,
        source_group_id TEXT NOT NULL,
        parent_source_ref TEXT NOT NULL,
        chunk_id TEXT NOT NULL,
        agent_id TEXT NOT NULL,
        scope TEXT NOT NULL,
        session_key TEXT NOT NULL,
        turn_id TEXT NOT NULL,
        seq INTEGER NOT NULL,
        role TEXT NOT NULL CHECK(role IN ('user','assistant','tool')),
        tool_name TEXT,
        segment_index INTEGER NOT NULL,
        char_start INTEGER NOT NULL,
        char_end INTEGER NOT NULL,
        text TEXT NOT NULL,
        content_hash TEXT NOT NULL,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        metadata_json TEXT NOT NULL DEFAULT '{}',
        FOREIGN KEY (chunk_id) REFERENCES conversation_chunks(chunk_id) ON DELETE CASCADE
      );

      CREATE INDEX IF NOT EXISTS idx_source_segments_group
        ON source_segments(agent_id, scope, source_group_id, segment_index);
      CREATE INDEX IF NOT EXISTS idx_source_segments_parent
        ON source_segments(agent_id, scope, parent_source_ref, segment_index);
      CREATE INDEX IF NOT EXISTS idx_source_segments_chunk
        ON source_segments(chunk_id, segment_index);
      CREATE INDEX IF NOT EXISTS idx_source_segments_session_turn
        ON source_segments(agent_id, session_key, turn_id, seq, segment_index);
      CREATE INDEX IF NOT EXISTS idx_source_segments_hash
        ON source_segments(agent_id, scope, content_hash);
    `
	},
	{
		version: 20,
		description: "session-scoped audit rows",
		sql: `
      ALTER TABLE retrieval_audit
        ADD COLUMN session_key TEXT;
      ALTER TABLE policy_decisions
        ADD COLUMN session_key TEXT;
      ALTER TABLE maintenance_runs
        ADD COLUMN session_key TEXT;

      CREATE INDEX IF NOT EXISTS idx_retrieval_audit_agent_session
        ON retrieval_audit(agent_id, session_key, created_at);
      CREATE INDEX IF NOT EXISTS idx_policy_agent_session
        ON policy_decisions(agent_id, session_key, created_at);
      CREATE INDEX IF NOT EXISTS idx_maintenance_agent_session
        ON maintenance_runs(agent_id, session_key, started_at);
    `
	},
	{
		version: 21,
		description: "durable semantic write job ledger",
		sql: `
      CREATE TABLE IF NOT EXISTS semantic_write_jobs (
        job_id TEXT PRIMARY KEY,
        agent_id TEXT NOT NULL,
        session_key TEXT NOT NULL,
        scope TEXT NOT NULL,
        turn_id TEXT NOT NULL,
        job_type TEXT NOT NULL CHECK (job_type IN ('turn_semantic_extraction')),
        source_refs_json TEXT NOT NULL DEFAULT '[]',
        input_hash TEXT NOT NULL,
        status TEXT NOT NULL CHECK (status IN ('pending', 'running', 'succeeded', 'retrying', 'failed')),
        attempt_count INTEGER NOT NULL DEFAULT 0,
        last_error TEXT,
        result_json TEXT NOT NULL DEFAULT '{}',
        next_attempt_at TEXT,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
      );

      CREATE UNIQUE INDEX IF NOT EXISTS idx_semantic_write_jobs_turn
        ON semantic_write_jobs(agent_id, session_key, turn_id, job_type);
      CREATE INDEX IF NOT EXISTS idx_semantic_write_jobs_status
        ON semantic_write_jobs(agent_id, session_key, status, updated_at);
    `
	}
];
//#endregion
export { MEMX_MIGRATIONS };
