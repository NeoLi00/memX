import { normalizeName, randomId, safeJsonParse, stableHash } from "../../support.mjs";
import { projectAliasVariants, projectIdentityKey } from "../../pipeline/projectIdentity.mjs";
import { normalizeGraphRelationType } from "../../pipeline/semantic/heuristics.mjs";
import { buildGraphPathCandidates } from "../../pipeline/graphPathEngine.mjs";
import { ENTITY_TYPES } from "../../types.mjs";
//#region src/db/repositories/graphRepo.ts
function entityTypeValue(value) {
	return ENTITY_TYPES.includes(value) ? value : "unknown";
}
function mergeStringArrays(...values) {
	const out = /* @__PURE__ */ new Set();
	for (const value of values) {
		if (typeof value === "string" && value.trim()) {
			out.add(value.trim());
			continue;
		}
		if (!Array.isArray(value)) continue;
		for (const entry of value) if (typeof entry === "string" && entry.trim()) out.add(entry.trim());
	}
	return [...out];
}
var GraphRepo = class {
	db;
	constructor(db) {
		this.db = db;
	}
	upsertEntity(entity) {
		const aliases = [...new Set([...entity.aliases, ...entity.entityType === "project" ? projectAliasVariants(entity.canonicalName) : []].filter(Boolean))];
		this.db.prepare(`INSERT INTO entities(
          entity_id, canonical_name, entity_type, normalized_name, aliases_json, confidence, created_at, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(entity_id) DO UPDATE SET
          canonical_name = excluded.canonical_name,
          entity_type = excluded.entity_type,
          normalized_name = excluded.normalized_name,
          aliases_json = excluded.aliases_json,
          confidence = excluded.confidence,
          updated_at = excluded.updated_at`).run(entity.entityId, entity.canonicalName, entity.entityType, entity.normalizedName, JSON.stringify(aliases), entity.confidence, (/* @__PURE__ */ new Date()).toISOString(), (/* @__PURE__ */ new Date()).toISOString());
		for (const alias of aliases) this.db.prepare(`INSERT OR IGNORE INTO entity_aliases(alias_id, entity_id, alias_text, normalized_alias)
           VALUES (?, ?, ?, ?)`).run(randomId("alias"), entity.entityId, alias, normalizeName(alias));
	}
	findEntityByName(name, entityType) {
		const normalized = normalizeName(name);
		const stored = this.lookupEntityByName(name, entityType);
		if (stored) return stored;
		return {
			entityId: stableHash([normalized]),
			canonicalName: name.trim(),
			entityType: entityTypeValue(entityType),
			normalizedName: normalized,
			aliases: [],
			confidence: .6
		};
	}
	lookupEntityByName(name, preferredType) {
		const normalized = normalizeName(name);
		const row = this.db.prepare(`SELECT entity_id, canonical_name, entity_type, normalized_name, aliases_json, confidence, created_at, updated_at
           FROM entities
          WHERE normalized_name = ?
             OR entity_id IN (
               SELECT entity_id FROM entity_aliases WHERE normalized_alias = ?
               UNION
               SELECT entity_id FROM entity_alias_sources WHERE normalized_alias = ?
             )
          ORDER BY
            CASE
              WHEN entity_type = 'project' THEN 2
              WHEN entity_type != 'unknown' THEN 1
              ELSE 0
            END DESC,
            confidence DESC
          LIMIT 1`).get(normalized, normalized, normalized);
		if (row) {
			if (!preferredType || row.entity_type === preferredType) return this.toEntity(row);
			if (preferredType === "project" && row.entity_type === "unknown") return this.toEntity(row);
		}
		const compact = projectIdentityKey(name);
		if (!compact) return row ? this.toEntity(row) : null;
		const fuzzy = this.db.prepare(`SELECT entity_id, canonical_name, entity_type, normalized_name, aliases_json, confidence, created_at, updated_at
           FROM entities
          WHERE entity_type = 'project'
             OR entity_type = 'unknown'
          ORDER BY
            CASE
              WHEN entity_type = 'project' THEN 1
              ELSE 0
            END DESC,
            confidence DESC,
            updated_at DESC
          LIMIT 256`).all().find((candidate) => {
			if (projectIdentityKey(candidate.canonical_name) === compact) return true;
			return safeJsonParse(candidate.aliases_json, []).some((alias) => projectIdentityKey(alias) === compact);
		});
		return fuzzy ? this.toEntity(fuzzy) : row ? this.toEntity(row) : null;
	}
	getEntityById(entityId) {
		const row = this.db.prepare(`SELECT entity_id, canonical_name, entity_type, normalized_name, aliases_json, confidence, created_at, updated_at
           FROM entities
          WHERE entity_id = ?
          LIMIT 1`).get(entityId);
		return row ? this.toEntity(row) : null;
	}
	lookupEntityByNormalizedName(normalizedName, preferredType) {
		const row = this.db.prepare(`SELECT entity_id, canonical_name, entity_type, normalized_name, aliases_json, confidence, created_at, updated_at
           FROM entities
          WHERE normalized_name = ?
          ORDER BY
            CASE
              WHEN entity_type = ? THEN 2
              WHEN entity_type != 'unknown' THEN 1
              ELSE 0
            END DESC,
            confidence DESC,
            updated_at DESC
          LIMIT 1`).get(normalizedName, preferredType ?? "");
		if (!row) return null;
		if (!preferredType || row.entity_type === preferredType || row.entity_type === "unknown") return this.toEntity(row);
		return null;
	}
	lookupEntityByAlias(normalizedAlias, preferredType) {
		const aliasKey = normalizeName(normalizedAlias);
		const row = this.db.prepare(`SELECT e.entity_id, e.canonical_name, e.entity_type, e.normalized_name, e.aliases_json, e.confidence, e.created_at, e.updated_at
           FROM entities e
          WHERE e.entity_id IN (
            SELECT entity_id FROM entity_aliases WHERE normalized_alias = ?
            UNION
            SELECT entity_id FROM entity_alias_sources WHERE normalized_alias = ?
          )
          ORDER BY
            CASE
              WHEN e.entity_type = ? THEN 2
              WHEN e.entity_type != 'unknown' THEN 1
              ELSE 0
            END DESC,
            e.confidence DESC,
            e.updated_at DESC
          LIMIT 1`).get(aliasKey, aliasKey, preferredType ?? "");
		if (!row) return null;
		if (!preferredType || row.entity_type === preferredType || row.entity_type === "unknown") return this.toEntity(row);
		return null;
	}
	searchEntitiesByQuery(query, limit = 6) {
		const normalizedQuery = normalizeName(query);
		if (!normalizedQuery) return [];
		return this.db.prepare(`SELECT entity_id, canonical_name, entity_type, normalized_name, aliases_json, confidence, created_at, updated_at
           FROM entities
          ORDER BY confidence DESC, updated_at DESC
          LIMIT 128`).all().filter((row) => {
			if (normalizedQuery.includes(row.normalized_name) || row.normalized_name.includes(normalizedQuery)) return true;
			return safeJsonParse(row.aliases_json, []).some((alias) => {
				const normalizedAlias = normalizeName(alias);
				return normalizedAlias.length > 0 && (normalizedQuery.includes(normalizedAlias) || normalizedAlias.includes(normalizedQuery));
			});
		}).slice(0, limit).map((row) => this.toEntity(row));
	}
	listResolvedMentionsByNormalized(params) {
		return this.db.prepare(`SELECT mention_id, agent_id, scope, raw_text, normalized_text, proposed_type, semantic_role, source_ref,
                support_text, session_key, turn_index, observed_at, resolved_entity_id, resolution_method,
                confidence, candidate_ids_json, blockers_json, metadata_json
           FROM entity_mentions
          WHERE agent_id = ?
            AND scope = ?
            AND normalized_text = ?
            AND resolved_entity_id IS NOT NULL
            ${params.sessionKey ? "AND session_key = ?" : ""}
          ORDER BY observed_at DESC
          LIMIT ?`).all(params.agentId, params.scope, params.normalizedText, ...params.sessionKey ? [params.sessionKey] : [], params.limit ?? 8).map((row) => this.toEntityMention(row));
	}
	upsertEntityMention(mention) {
		this.db.prepare(`INSERT INTO entity_mentions(
          mention_id, agent_id, scope, raw_text, normalized_text, proposed_type, semantic_role, source_ref,
          support_text, session_key, turn_index, observed_at, resolved_entity_id, resolution_method,
          confidence, candidate_ids_json, blockers_json, metadata_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(mention_id) DO UPDATE SET
          resolved_entity_id = excluded.resolved_entity_id,
          resolution_method = excluded.resolution_method,
          confidence = excluded.confidence,
          candidate_ids_json = excluded.candidate_ids_json,
          blockers_json = excluded.blockers_json,
          metadata_json = excluded.metadata_json`).run(mention.mentionId, mention.agentId, mention.scope, mention.rawText, mention.normalizedText, mention.proposedType, mention.semanticRole, mention.sourceRef, mention.supportText, mention.sessionKey ?? null, mention.turnIndex ?? null, mention.observedAt, mention.resolvedEntityId ?? null, mention.resolutionMethod ?? null, mention.confidence, JSON.stringify(mention.candidateIds), JSON.stringify(mention.blockers), JSON.stringify(mention.metadataJson));
	}
	upsertEntityAliasSource(params) {
		const normalizedAlias = normalizeName(params.aliasText);
		if (!normalizedAlias) return;
		const aliasSourceId = stableHash([
			"entity-alias-source",
			params.entityId,
			normalizedAlias,
			params.sourceRef
		]);
		this.db.prepare(`INSERT INTO entity_alias_sources(
          alias_source_id, entity_id, alias_text, normalized_alias, source_ref, confidence, created_at, metadata_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(alias_source_id) DO UPDATE SET
          confidence = MAX(confidence, excluded.confidence),
          metadata_json = excluded.metadata_json`).run(aliasSourceId, params.entityId, params.aliasText, normalizedAlias, params.sourceRef, params.confidence, params.createdAt, JSON.stringify(params.metadataJson ?? {}));
	}
	upsertIdentityLink(params) {
		if (params.srcEntityId === params.dstEntityId) return;
		const linkId = stableHash([
			"entity-identity-link",
			params.srcEntityId,
			params.dstEntityId,
			params.linkType,
			params.evidenceRef
		]);
		this.db.prepare(`INSERT INTO entity_identity_links(
          link_id, src_entity_id, dst_entity_id, link_type, confidence, evidence_ref, status,
          created_at, updated_at, metadata_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(link_id) DO UPDATE SET
          confidence = MAX(confidence, excluded.confidence),
          status = excluded.status,
          updated_at = excluded.updated_at,
          metadata_json = excluded.metadata_json`).run(linkId, params.srcEntityId, params.dstEntityId, params.linkType, params.confidence, params.evidenceRef, params.status ?? "active", params.at, params.at, JSON.stringify(params.metadataJson ?? {}));
	}
	linkedEntityIds(params) {
		const linkTypes = params.linkTypes ?? [
			"same_as",
			"duplicate_of",
			"supersedes"
		];
		if (linkTypes.length === 0) return [];
		const placeholders = linkTypes.map(() => "?").join(", ");
		const rows = this.db.prepare(`SELECT CASE
                  WHEN src_entity_id = ? THEN dst_entity_id
                  ELSE src_entity_id
                END AS entity_id
           FROM entity_identity_links
          WHERE status = ?
            AND link_type IN (${placeholders})
            AND (src_entity_id = ? OR dst_entity_id = ?)
          ORDER BY confidence DESC, updated_at DESC
          LIMIT ?`).all(params.entityId, params.status ?? "active", ...linkTypes, params.entityId, params.entityId, params.limit ?? 8);
		return [...new Set(rows.map((row) => row.entity_id).filter(Boolean))];
	}
	findSupersedingEntity(entityId) {
		const row = this.db.prepare(`SELECT e.entity_id, e.canonical_name, e.entity_type, e.normalized_name, e.aliases_json,
                e.confidence, e.created_at, e.updated_at
           FROM entity_identity_links l
           JOIN entities e ON e.entity_id = l.src_entity_id
          WHERE l.dst_entity_id = ?
            AND l.link_type = 'supersedes'
            AND l.status = 'active'
          ORDER BY l.confidence DESC, l.updated_at DESC
          LIMIT 1`).get(entityId);
		return row ? this.toEntity(row) : null;
	}
	listMentionsForEntity(params) {
		return this.db.prepare(`SELECT mention_id, agent_id, scope, raw_text, normalized_text, proposed_type, semantic_role, source_ref,
                support_text, session_key, turn_index, observed_at, resolved_entity_id, resolution_method,
                confidence, candidate_ids_json, blockers_json, metadata_json
           FROM entity_mentions
          WHERE resolved_entity_id = ?
          ORDER BY observed_at DESC, confidence DESC
          LIMIT ?`).all(params.entityId, params.limit ?? 8).map((row) => this.toEntityMention(row));
	}
	listAliasSourcesForEntity(params) {
		return this.db.prepare(`SELECT alias_source_id, entity_id, alias_text, normalized_alias, source_ref, confidence, created_at, metadata_json
           FROM entity_alias_sources
          WHERE entity_id = ?
          ORDER BY confidence DESC, created_at DESC
          LIMIT ?`).all(params.entityId, params.limit ?? 8).map((row) => ({
			aliasText: row.alias_text,
			normalizedAlias: row.normalized_alias,
			sourceRef: row.source_ref,
			confidence: row.confidence,
			createdAt: row.created_at,
			metadataJson: safeJsonParse(row.metadata_json, {})
		}));
	}
	upsertEdge(edge) {
		const existing = this.db.prepare(`SELECT edge_id, src_entity_id, rel_type, dst_entity_id, scope, agent_id, confidence, valid_from, valid_to,
                relation_slot, evidence_ref, created_at, updated_at, materialized_epoch, metadata_json
           FROM graph_edges
          WHERE src_entity_id = ?
            AND rel_type = ?
            AND dst_entity_id = ?
            AND scope = ?
            AND agent_id = ?
            AND COALESCE(relation_slot, '') = COALESCE(?, '')
          LIMIT 1`).get(edge.srcEntityId, edge.relType, edge.dstEntityId, edge.scope, edge.agentId, edge.relationSlot ?? null);
		if (existing) {
			const existingMetadata = safeJsonParse(existing.metadata_json, {});
			const incomingMetadata = edge.metadataJson ?? {};
			const sourceRefs = mergeStringArrays(existingMetadata.sourceRefs, incomingMetadata.sourceRefs, existing.evidence_ref, edge.evidenceRef);
			const supportRefs = mergeStringArrays(existingMetadata.supportRefs, incomingMetadata.supportRefs, existingMetadata.supportContentRefs, incomingMetadata.supportContentRefs, existing.evidence_ref, edge.evidenceRef);
			this.db.prepare(`UPDATE graph_edges
              SET confidence = ?, valid_from = COALESCE(valid_from, ?), valid_to = ?, evidence_ref = ?, relation_slot = ?, updated_at = ?, materialized_epoch = ?, metadata_json = ?
            WHERE edge_id = ?`).run(Math.max(existing.confidence, edge.confidence), edge.validFrom ?? null, edge.validTo ?? existing.valid_to, edge.evidenceRef, edge.relationSlot ?? existing.relation_slot, edge.updatedAt, edge.materializedEpoch ?? existing.materialized_epoch, JSON.stringify({
				...existingMetadata,
				...incomingMetadata,
				sourceRefs,
				supportRefs
			}), existing.edge_id);
			return { action: "merged" };
		}
		if (edge.relationSlot) this.db.prepare(`UPDATE graph_edges
              SET valid_to = COALESCE(valid_to, ?), updated_at = ?
            WHERE src_entity_id = ?
              AND rel_type = ?
              AND scope = ?
              AND agent_id = ?
              AND COALESCE(relation_slot, '') = COALESCE(?, '')
              AND dst_entity_id != ?
              AND (valid_to IS NULL OR valid_to > ?)`).run(edge.validFrom ?? edge.updatedAt, edge.updatedAt, edge.srcEntityId, edge.relType, edge.scope, edge.agentId, edge.relationSlot, edge.dstEntityId, edge.validFrom ?? edge.updatedAt);
		this.db.prepare(`INSERT INTO graph_edges(
          edge_id, src_entity_id, rel_type, dst_entity_id, relation_slot, scope, agent_id, confidence, valid_from,
          valid_to, evidence_ref, created_at, updated_at, materialized_epoch, metadata_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(edge_id) DO UPDATE SET
          confidence = MAX(confidence, excluded.confidence),
          valid_from = COALESCE(graph_edges.valid_from, excluded.valid_from),
          evidence_ref = excluded.evidence_ref,
          updated_at = excluded.updated_at,
          materialized_epoch = excluded.materialized_epoch,
          metadata_json = excluded.metadata_json`).run(edge.edgeId, edge.srcEntityId, edge.relType, edge.dstEntityId, edge.relationSlot ?? null, edge.scope, edge.agentId, edge.confidence, edge.validFrom ?? null, edge.validTo ?? null, edge.evidenceRef, edge.createdAt, edge.updatedAt, edge.materializedEpoch ?? 0, JSON.stringify(edge.metadataJson ?? {}));
		return { action: "created" };
	}
	closeActiveSlotEdges(params) {
		const result = this.db.prepare(`UPDATE graph_edges
            SET valid_to = COALESCE(valid_to, ?), updated_at = ?
          WHERE src_entity_id = ?
            AND rel_type = ?
            AND scope = ?
            AND agent_id = ?
            AND COALESCE(relation_slot, '') = COALESCE(?, '')
            AND (valid_to IS NULL OR valid_to > ?)`).run(params.validTo, params.validTo, params.srcEntityId, params.relType, params.scope, params.agentId, params.relationSlot, params.validTo);
		return Number(result.changes ?? 0);
	}
	expandNeighborhood(params) {
		if (params.seedEntityIds.length === 0 || params.scopes.length === 0) return {
			nodes: [],
			edges: [],
			pathCandidates: [],
			paths: []
		};
		const scopePlaceholders = params.scopes.map(() => "?").join(", ");
		const seenNodes = new Set(params.seedEntityIds);
		const nodes = /* @__PURE__ */ new Map();
		const edges = /* @__PURE__ */ new Map();
		let frontier = [...params.seedEntityIds];
		for (let hop = 0; hop < params.maxHops; hop += 1) {
			if (frontier.length === 0 || edges.size >= params.maxEdges || nodes.size >= params.maxNodes) break;
			const frontierPlaceholders = frontier.map(() => "?").join(", ");
			const rows = this.db.prepare(`SELECT edge_id, src_entity_id, rel_type, dst_entity_id, scope, agent_id, confidence, valid_from,
                  valid_to, relation_slot, evidence_ref, created_at, updated_at, materialized_epoch, metadata_json
             FROM graph_edges
            WHERE agent_id = ?
              AND scope IN (${scopePlaceholders})
              AND (valid_to IS NULL OR valid_to > ?)
              ${typeof params.readEpoch === "number" ? "AND materialized_epoch <= ?" : ""}
              AND (src_entity_id IN (${frontierPlaceholders}) OR dst_entity_id IN (${frontierPlaceholders}))
            ORDER BY confidence DESC, updated_at DESC
            LIMIT ${Math.max(params.maxEdges * 2, 16)}`).all(params.agentId, ...params.scopes, params.now ?? (/* @__PURE__ */ new Date()).toISOString(), ...typeof params.readEpoch === "number" ? [params.readEpoch] : [], ...frontier, ...frontier);
			frontier = [];
			for (const row of rows) {
				if (edges.size >= params.maxEdges) break;
				const relation = normalizeGraphRelationType(row.rel_type);
				if (!relation) continue;
				if (!seenNodes.has(row.src_entity_id) || !seenNodes.has(row.dst_entity_id)) {
					const next = seenNodes.has(row.src_entity_id) ? row.dst_entity_id : row.src_entity_id;
					if (!seenNodes.has(next) && nodes.size < params.maxNodes) {
						frontier.push(next);
						seenNodes.add(next);
					}
				}
				edges.set(row.edge_id, {
					edgeId: row.edge_id,
					srcNodeId: row.src_entity_id,
					srcEntityId: row.src_entity_id,
					relType: relation.relationType,
					dstNodeId: row.dst_entity_id,
					dstEntityId: row.dst_entity_id,
					relationSlot: row.relation_slot ?? void 0,
					confidence: row.confidence,
					evidenceRef: row.evidence_ref,
					updatedAt: row.updated_at,
					sourceKind: "stored",
					metadata: safeJsonParse(row.metadata_json, {})
				});
			}
		}
		for (const entityId of seenNodes) {
			const row = this.db.prepare(`SELECT entity_id, canonical_name, entity_type, normalized_name, aliases_json, confidence, created_at, updated_at
             FROM entities WHERE entity_id = ?`).get(entityId);
			if (!row) continue;
			nodes.set(row.entity_id, {
				nodeId: row.entity_id,
				nodeKind: "entity",
				entityId: row.entity_id,
				name: row.canonical_name,
				type: row.entity_type,
				confidence: row.confidence
			});
		}
		const edgeList = [...edges.values()].sort((left, right) => right.confidence - left.confidence || (right.updatedAt ? Date.parse(right.updatedAt) : 0) - (left.updatedAt ? Date.parse(left.updatedAt) : 0));
		const pathCandidates = buildGraphPathCandidates({
			seedNodeIds: params.seedEntityIds,
			nodes,
			edges: edgeList,
			now: params.now ?? (/* @__PURE__ */ new Date()).toISOString(),
			maxPaths: Math.max(6, Math.min(params.maxEdges, 12)),
			maxHops: Math.min(2, Math.max(1, params.maxHops))
		});
		return {
			nodes: [...nodes.values()],
			edges: edgeList,
			pathCandidates,
			paths: pathCandidates.map((path) => path.summary)
		};
	}
	pruneLowConfidence(params) {
		const result = this.db.prepare(`DELETE FROM graph_edges
          WHERE agent_id = ?
            AND confidence <= ?
            AND updated_at < ?`).run(params.agentId, params.maxConfidence, params.olderThan);
		return Number(result.changes ?? 0);
	}
	toEntity(row) {
		return {
			entityId: row.entity_id,
			canonicalName: row.canonical_name,
			entityType: entityTypeValue(row.entity_type),
			normalizedName: row.normalized_name,
			aliases: safeJsonParse(row.aliases_json, []),
			confidence: row.confidence
		};
	}
	toEntityMention(row) {
		return {
			mentionId: row.mention_id,
			agentId: row.agent_id,
			scope: row.scope,
			rawText: row.raw_text,
			normalizedText: row.normalized_text,
			proposedType: row.proposed_type,
			semanticRole: row.semantic_role,
			sourceRef: row.source_ref,
			supportText: row.support_text,
			sessionKey: row.session_key ?? void 0,
			turnIndex: row.turn_index ?? void 0,
			observedAt: row.observed_at,
			resolvedEntityId: row.resolved_entity_id ?? void 0,
			resolutionMethod: row.resolution_method ?? void 0,
			confidence: row.confidence,
			candidateIds: safeJsonParse(row.candidate_ids_json, []),
			blockers: safeJsonParse(row.blockers_json, []),
			metadataJson: safeJsonParse(row.metadata_json, {})
		};
	}
};
//#endregion
export { GraphRepo };
