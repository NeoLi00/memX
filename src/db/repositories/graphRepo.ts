import { buildGraphPathCandidates } from "../../pipeline/graphPathEngine.js";
import { projectAliasVariants, projectIdentityKey } from "../../pipeline/projectIdentity.js";
import { normalizeGraphRelationType } from "../../pipeline/semantic/heuristics.js";
import { normalizeName, randomId, safeJsonParse, stableHash } from "../../support.js";
import { ENTITY_TYPES } from "../../types.js";
import type {
  EntityType,
  GraphEvidence,
  GraphEvidenceEdge,
  GraphEvidenceNode,
  EntityIdentityLinkType,
  EntityMention,
  EntityResolutionMethod,
  NormalizedEntity,
  NormalizedGraphEdge,
} from "../../types.js";
import type { MemxDbClient } from "../client.js";

type EntityRow = {
  entity_id: string;
  canonical_name: string;
  entity_type: string;
  normalized_name: string;
  aliases_json: string;
  confidence: number;
  created_at: string;
  updated_at: string;
};

function entityTypeValue(value: string | undefined): EntityType {
  return ENTITY_TYPES.includes(value as EntityType) ? (value as EntityType) : "unknown";
}

type EdgeRow = {
  edge_id: string;
  src_entity_id: string;
  rel_type: string;
  dst_entity_id: string;
  relation_slot: string | null;
  scope: string;
  agent_id: string;
  confidence: number;
  valid_from: string | null;
  valid_to: string | null;
  evidence_ref: string;
  created_at: string;
  updated_at: string;
  materialized_epoch: number;
  metadata_json: string;
};

type EntityMentionRow = {
  mention_id: string;
  agent_id: string;
  scope: string;
  raw_text: string;
  normalized_text: string;
  proposed_type: string;
  semantic_role: string;
  source_ref: string;
  support_text: string;
  session_key: string | null;
  turn_index: number | null;
  observed_at: string;
  resolved_entity_id: string | null;
  resolution_method: string | null;
  confidence: number;
  candidate_ids_json: string;
  blockers_json: string;
  metadata_json: string;
};

type EntityAliasSourceRow = {
  alias_source_id: string;
  entity_id: string;
  alias_text: string;
  normalized_alias: string;
  source_ref: string;
  confidence: number;
  created_at: string;
  metadata_json: string;
};

type ExpandedGraphEdge = GraphEvidenceEdge;

function mergeStringArrays(...values: unknown[]): string[] {
  const out = new Set<string>();
  for (const value of values) {
    if (typeof value === "string" && value.trim()) {
      out.add(value.trim());
      continue;
    }
    if (!Array.isArray(value)) {
      continue;
    }
    for (const entry of value) {
      if (typeof entry === "string" && entry.trim()) {
        out.add(entry.trim());
      }
    }
  }
  return [...out];
}

export class GraphRepo {
  constructor(private readonly db: MemxDbClient) {}

  upsertEntity(entity: NormalizedEntity): void {
    const aliases = [
      ...new Set(
        [
          ...entity.aliases,
          ...(entity.entityType === "project" ? projectAliasVariants(entity.canonicalName) : []),
        ].filter(Boolean),
      ),
    ];
    this.db
      .prepare(
        `INSERT INTO entities(
          entity_id, canonical_name, entity_type, normalized_name, aliases_json, confidence, created_at, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(entity_id) DO UPDATE SET
          canonical_name = excluded.canonical_name,
          entity_type = excluded.entity_type,
          normalized_name = excluded.normalized_name,
          aliases_json = excluded.aliases_json,
          confidence = excluded.confidence,
          updated_at = excluded.updated_at`,
      )
      .run(
        entity.entityId,
        entity.canonicalName,
        entity.entityType,
        entity.normalizedName,
        JSON.stringify(aliases),
        entity.confidence,
        new Date().toISOString(),
        new Date().toISOString(),
      );

    for (const alias of aliases) {
      this.db
        .prepare(
          `INSERT OR IGNORE INTO entity_aliases(alias_id, entity_id, alias_text, normalized_alias)
           VALUES (?, ?, ?, ?)`,
        )
        .run(randomId("alias"), entity.entityId, alias, normalizeName(alias));
    }
  }

  findEntityByName(name: string, entityType?: string): NormalizedEntity | null {
    const normalized = normalizeName(name);
    const stored = this.lookupEntityByName(name, entityType);
    if (stored) {
      return stored;
    }
    const entityId = stableHash([normalized]);
    return {
      entityId,
      canonicalName: name.trim(),
      entityType: entityTypeValue(entityType),
      normalizedName: normalized,
      aliases: [],
      confidence: 0.6,
    };
  }

  lookupEntityByName(name: string, preferredType?: string): NormalizedEntity | null {
    const normalized = normalizeName(name);
    const row = this.db
      .prepare(
        `SELECT entity_id, canonical_name, entity_type, normalized_name, aliases_json, confidence, created_at, updated_at
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
          LIMIT 1`,
      )
      .get(normalized, normalized, normalized) as EntityRow | undefined;
    if (row) {
      if (!preferredType || row.entity_type === preferredType) {
        return this.toEntity(row);
      }
      if (preferredType === "project" && row.entity_type === "unknown") {
        return this.toEntity(row);
      }
    }
    const compact = projectIdentityKey(name);
    if (!compact) {
      return row ? this.toEntity(row) : null;
    }
    const projectRows = this.db
      .prepare(
        `SELECT entity_id, canonical_name, entity_type, normalized_name, aliases_json, confidence, created_at, updated_at
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
          LIMIT 256`,
      )
      .all() as EntityRow[];
    const fuzzy = projectRows.find((candidate) => {
      if (projectIdentityKey(candidate.canonical_name) === compact) {
        return true;
      }
      const aliases = safeJsonParse<string[]>(candidate.aliases_json, []);
      return aliases.some((alias) => projectIdentityKey(alias) === compact);
    });
    return fuzzy ? this.toEntity(fuzzy) : row ? this.toEntity(row) : null;
  }

  getEntityById(entityId: string): NormalizedEntity | null {
    const row = this.db
      .prepare(
        `SELECT entity_id, canonical_name, entity_type, normalized_name, aliases_json, confidence, created_at, updated_at
           FROM entities
          WHERE entity_id = ?
          LIMIT 1`,
      )
      .get(entityId) as EntityRow | undefined;
    return row ? this.toEntity(row) : null;
  }

  lookupEntityByNormalizedName(
    normalizedName: string,
    preferredType?: string,
  ): NormalizedEntity | null {
    const row = this.db
      .prepare(
        `SELECT entity_id, canonical_name, entity_type, normalized_name, aliases_json, confidence, created_at, updated_at
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
          LIMIT 1`,
      )
      .get(normalizedName, preferredType ?? "") as EntityRow | undefined;
    if (!row) {
      return null;
    }
    if (!preferredType || row.entity_type === preferredType || row.entity_type === "unknown") {
      return this.toEntity(row);
    }
    return null;
  }

  lookupEntityByAlias(normalizedAlias: string, preferredType?: string): NormalizedEntity | null {
    const aliasKey = normalizeName(normalizedAlias);
    const row = this.db
      .prepare(
        `SELECT e.entity_id, e.canonical_name, e.entity_type, e.normalized_name, e.aliases_json, e.confidence, e.created_at, e.updated_at
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
          LIMIT 1`,
      )
      .get(aliasKey, aliasKey, preferredType ?? "") as EntityRow | undefined;
    if (!row) {
      return null;
    }
    if (!preferredType || row.entity_type === preferredType || row.entity_type === "unknown") {
      return this.toEntity(row);
    }
    return null;
  }

  searchEntitiesByQuery(query: string, limit = 6): NormalizedEntity[] {
    const normalizedQuery = normalizeName(query);
    if (!normalizedQuery) {
      return [];
    }
    const rows = this.db
      .prepare(
        `SELECT entity_id, canonical_name, entity_type, normalized_name, aliases_json, confidence, created_at, updated_at
           FROM entities
          ORDER BY confidence DESC, updated_at DESC
          LIMIT 128`,
      )
      .all() as EntityRow[];
    return rows
      .filter((row) => {
        if (
          normalizedQuery.includes(row.normalized_name) ||
          row.normalized_name.includes(normalizedQuery)
        ) {
          return true;
        }
        const aliases = safeJsonParse<string[]>(row.aliases_json, []);
        return aliases.some((alias) => {
          const normalizedAlias = normalizeName(alias);
          return (
            normalizedAlias.length > 0 &&
            (normalizedQuery.includes(normalizedAlias) ||
              normalizedAlias.includes(normalizedQuery))
          );
        });
      })
      .slice(0, limit)
      .map((row) => this.toEntity(row));
  }

  listResolvedMentionsByNormalized(params: {
    agentId: string;
    scope: string;
    normalizedText: string;
    sessionKey?: string;
    limit?: number;
  }): EntityMention[] {
    const rows = this.db
      .prepare(
        `SELECT mention_id, agent_id, scope, raw_text, normalized_text, proposed_type, semantic_role, source_ref,
                support_text, session_key, turn_index, observed_at, resolved_entity_id, resolution_method,
                confidence, candidate_ids_json, blockers_json, metadata_json
           FROM entity_mentions
          WHERE agent_id = ?
            AND scope = ?
            AND normalized_text = ?
            AND resolved_entity_id IS NOT NULL
            ${params.sessionKey ? "AND session_key = ?" : ""}
          ORDER BY observed_at DESC
          LIMIT ?`,
      )
      .all(
        params.agentId,
        params.scope,
        params.normalizedText,
        ...(params.sessionKey ? [params.sessionKey] : []),
        params.limit ?? 8,
      ) as EntityMentionRow[];
    return rows.map((row) => this.toEntityMention(row));
  }

  upsertEntityMention(mention: EntityMention): void {
    this.db
      .prepare(
        `INSERT INTO entity_mentions(
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
          metadata_json = excluded.metadata_json`,
      )
      .run(
        mention.mentionId,
        mention.agentId,
        mention.scope,
        mention.rawText,
        mention.normalizedText,
        mention.proposedType,
        mention.semanticRole,
        mention.sourceRef,
        mention.supportText,
        mention.sessionKey ?? null,
        mention.turnIndex ?? null,
        mention.observedAt,
        mention.resolvedEntityId ?? null,
        mention.resolutionMethod ?? null,
        mention.confidence,
        JSON.stringify(mention.candidateIds),
        JSON.stringify(mention.blockers),
        JSON.stringify(mention.metadataJson),
      );
  }

  upsertEntityAliasSource(params: {
    entityId: string;
    aliasText: string;
    sourceRef: string;
    confidence: number;
    createdAt: string;
    metadataJson?: Record<string, unknown>;
  }): void {
    const normalizedAlias = normalizeName(params.aliasText);
    if (!normalizedAlias) {
      return;
    }
    const aliasSourceId = stableHash([
      "entity-alias-source",
      params.entityId,
      normalizedAlias,
      params.sourceRef,
    ]);
    this.db
      .prepare(
        `INSERT INTO entity_alias_sources(
          alias_source_id, entity_id, alias_text, normalized_alias, source_ref, confidence, created_at, metadata_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(alias_source_id) DO UPDATE SET
          confidence = MAX(confidence, excluded.confidence),
          metadata_json = excluded.metadata_json`,
      )
      .run(
        aliasSourceId,
        params.entityId,
        params.aliasText,
        normalizedAlias,
        params.sourceRef,
        params.confidence,
        params.createdAt,
        JSON.stringify(params.metadataJson ?? {}),
      );
  }

  upsertIdentityLink(params: {
    srcEntityId: string;
    dstEntityId: string;
    linkType: EntityIdentityLinkType;
    confidence: number;
    evidenceRef: string;
    status?: "active" | "rejected";
    at: string;
    metadataJson?: Record<string, unknown>;
  }): void {
    if (params.srcEntityId === params.dstEntityId) {
      return;
    }
    const linkId = stableHash([
      "entity-identity-link",
      params.srcEntityId,
      params.dstEntityId,
      params.linkType,
      params.evidenceRef,
    ]);
    this.db
      .prepare(
        `INSERT INTO entity_identity_links(
          link_id, src_entity_id, dst_entity_id, link_type, confidence, evidence_ref, status,
          created_at, updated_at, metadata_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(link_id) DO UPDATE SET
          confidence = MAX(confidence, excluded.confidence),
          status = excluded.status,
          updated_at = excluded.updated_at,
          metadata_json = excluded.metadata_json`,
      )
      .run(
        linkId,
        params.srcEntityId,
        params.dstEntityId,
        params.linkType,
        params.confidence,
        params.evidenceRef,
        params.status ?? "active",
        params.at,
        params.at,
        JSON.stringify(params.metadataJson ?? {}),
      );
  }

  linkedEntityIds(params: {
    entityId: string;
    linkTypes?: EntityIdentityLinkType[];
    status?: "active" | "rejected";
    limit?: number;
  }): string[] {
    const linkTypes = params.linkTypes ?? ["same_as", "duplicate_of", "supersedes"];
    if (linkTypes.length === 0) {
      return [];
    }
    const placeholders = linkTypes.map(() => "?").join(", ");
    const rows = this.db
      .prepare(
        `SELECT CASE
                  WHEN src_entity_id = ? THEN dst_entity_id
                  ELSE src_entity_id
                END AS entity_id
           FROM entity_identity_links
          WHERE status = ?
            AND link_type IN (${placeholders})
            AND (src_entity_id = ? OR dst_entity_id = ?)
          ORDER BY confidence DESC, updated_at DESC
          LIMIT ?`,
      )
      .all(
        params.entityId,
        params.status ?? "active",
        ...linkTypes,
        params.entityId,
        params.entityId,
        params.limit ?? 8,
      ) as Array<{ entity_id: string }>;
    return [...new Set(rows.map((row) => row.entity_id).filter(Boolean))];
  }

  findSupersedingEntity(entityId: string): NormalizedEntity | null {
    const row = this.db
      .prepare(
        `SELECT e.entity_id, e.canonical_name, e.entity_type, e.normalized_name, e.aliases_json,
                e.confidence, e.created_at, e.updated_at
           FROM entity_identity_links l
           JOIN entities e ON e.entity_id = l.src_entity_id
          WHERE l.dst_entity_id = ?
            AND l.link_type = 'supersedes'
            AND l.status = 'active'
          ORDER BY l.confidence DESC, l.updated_at DESC
          LIMIT 1`,
      )
      .get(entityId) as EntityRow | undefined;
    return row ? this.toEntity(row) : null;
  }

  listMentionsForEntity(params: { entityId: string; limit?: number }): EntityMention[] {
    const rows = this.db
      .prepare(
        `SELECT mention_id, agent_id, scope, raw_text, normalized_text, proposed_type, semantic_role, source_ref,
                support_text, session_key, turn_index, observed_at, resolved_entity_id, resolution_method,
                confidence, candidate_ids_json, blockers_json, metadata_json
           FROM entity_mentions
          WHERE resolved_entity_id = ?
          ORDER BY observed_at DESC, confidence DESC
          LIMIT ?`,
      )
      .all(params.entityId, params.limit ?? 8) as EntityMentionRow[];
    return rows.map((row) => this.toEntityMention(row));
  }

  listAliasSourcesForEntity(params: { entityId: string; limit?: number }): Array<{
    aliasText: string;
    normalizedAlias: string;
    sourceRef: string;
    confidence: number;
    createdAt: string;
    metadataJson: Record<string, unknown>;
  }> {
    const rows = this.db
      .prepare(
        `SELECT alias_source_id, entity_id, alias_text, normalized_alias, source_ref, confidence, created_at, metadata_json
           FROM entity_alias_sources
          WHERE entity_id = ?
          ORDER BY confidence DESC, created_at DESC
          LIMIT ?`,
      )
      .all(params.entityId, params.limit ?? 8) as EntityAliasSourceRow[];
    return rows.map((row) => ({
      aliasText: row.alias_text,
      normalizedAlias: row.normalized_alias,
      sourceRef: row.source_ref,
      confidence: row.confidence,
      createdAt: row.created_at,
      metadataJson: safeJsonParse<Record<string, unknown>>(row.metadata_json, {}),
    }));
  }

  upsertEdge(edge: NormalizedGraphEdge): { action: "created" | "merged" } {
    const existing = this.db
      .prepare(
        `SELECT edge_id, src_entity_id, rel_type, dst_entity_id, scope, agent_id, confidence, valid_from, valid_to,
                relation_slot, evidence_ref, created_at, updated_at, materialized_epoch, metadata_json
           FROM graph_edges
          WHERE src_entity_id = ?
            AND rel_type = ?
            AND dst_entity_id = ?
            AND scope = ?
            AND agent_id = ?
            AND COALESCE(relation_slot, '') = COALESCE(?, '')
          LIMIT 1`,
      )
      .get(
        edge.srcEntityId,
        edge.relType,
        edge.dstEntityId,
        edge.scope,
        edge.agentId,
        edge.relationSlot ?? null,
      ) as EdgeRow | undefined;

    if (existing) {
      const existingMetadata = safeJsonParse<Record<string, unknown>>(existing.metadata_json, {});
      const incomingMetadata = edge.metadataJson ?? {};
      const sourceRefs = mergeStringArrays(
        existingMetadata.sourceRefs,
        incomingMetadata.sourceRefs,
        existing.evidence_ref,
        edge.evidenceRef,
      );
      const supportRefs = mergeStringArrays(
        existingMetadata.supportRefs,
        incomingMetadata.supportRefs,
        existingMetadata.supportContentRefs,
        incomingMetadata.supportContentRefs,
        existing.evidence_ref,
        edge.evidenceRef,
      );
      this.db
        .prepare(
          `UPDATE graph_edges
              SET confidence = ?, valid_from = COALESCE(valid_from, ?), valid_to = ?, evidence_ref = ?, relation_slot = ?, updated_at = ?, materialized_epoch = ?, metadata_json = ?
            WHERE edge_id = ?`,
        )
        .run(
          Math.max(existing.confidence, edge.confidence),
          edge.validFrom ?? null,
          edge.validTo ?? existing.valid_to,
          edge.evidenceRef,
          edge.relationSlot ?? existing.relation_slot,
          edge.updatedAt,
          edge.materializedEpoch ?? existing.materialized_epoch,
          JSON.stringify({
            ...existingMetadata,
            ...incomingMetadata,
            sourceRefs,
            supportRefs,
          }),
          existing.edge_id,
        );
      return { action: "merged" };
    }

    if (edge.relationSlot) {
      this.db
        .prepare(
          `UPDATE graph_edges
              SET valid_to = COALESCE(valid_to, ?), updated_at = ?
            WHERE src_entity_id = ?
              AND rel_type = ?
              AND scope = ?
              AND agent_id = ?
              AND COALESCE(relation_slot, '') = COALESCE(?, '')
              AND dst_entity_id != ?
              AND (valid_to IS NULL OR valid_to > ?)`,
        )
        .run(
          edge.validFrom ?? edge.updatedAt,
          edge.updatedAt,
          edge.srcEntityId,
          edge.relType,
          edge.scope,
          edge.agentId,
          edge.relationSlot,
          edge.dstEntityId,
          edge.validFrom ?? edge.updatedAt,
        );
    }

    this.db
      .prepare(
        `INSERT INTO graph_edges(
          edge_id, src_entity_id, rel_type, dst_entity_id, relation_slot, scope, agent_id, confidence, valid_from,
          valid_to, evidence_ref, created_at, updated_at, materialized_epoch, metadata_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(edge_id) DO UPDATE SET
          confidence = MAX(confidence, excluded.confidence),
          valid_from = COALESCE(graph_edges.valid_from, excluded.valid_from),
          evidence_ref = excluded.evidence_ref,
          updated_at = excluded.updated_at,
          materialized_epoch = excluded.materialized_epoch,
          metadata_json = excluded.metadata_json`,
      )
      .run(
        edge.edgeId,
        edge.srcEntityId,
        edge.relType,
        edge.dstEntityId,
        edge.relationSlot ?? null,
        edge.scope,
        edge.agentId,
        edge.confidence,
        edge.validFrom ?? null,
        edge.validTo ?? null,
        edge.evidenceRef,
        edge.createdAt,
        edge.updatedAt,
        edge.materializedEpoch ?? 0,
        JSON.stringify(edge.metadataJson ?? {}),
      );
    return { action: "created" };
  }

  closeActiveSlotEdges(params: {
    agentId: string;
    scope: string;
    srcEntityId: string;
    relType: string;
    relationSlot: string;
    validTo: string;
  }): number {
    const result = this.db
      .prepare(
        `UPDATE graph_edges
            SET valid_to = COALESCE(valid_to, ?), updated_at = ?
          WHERE src_entity_id = ?
            AND rel_type = ?
            AND scope = ?
            AND agent_id = ?
            AND COALESCE(relation_slot, '') = COALESCE(?, '')
            AND (valid_to IS NULL OR valid_to > ?)`,
      )
      .run(
        params.validTo,
        params.validTo,
        params.srcEntityId,
        params.relType,
        params.scope,
        params.agentId,
        params.relationSlot,
        params.validTo,
      );
    return Number(result.changes ?? 0);
  }

  expandNeighborhood(params: {
    agentId: string;
    scopes: string[];
    seedEntityIds: string[];
    maxHops: number;
    maxEdges: number;
    maxNodes: number;
    now?: string;
    readEpoch?: number;
  }): GraphEvidence {
    if (params.seedEntityIds.length === 0 || params.scopes.length === 0) {
      return { nodes: [], edges: [], pathCandidates: [], paths: [] };
    }
    const scopePlaceholders = params.scopes.map(() => "?").join(", ");
    const seenNodes = new Set<string>(params.seedEntityIds);
    const nodes = new Map<string, GraphEvidenceNode>();
    const edges = new Map<string, ExpandedGraphEdge>();
    let frontier = [...params.seedEntityIds];

    for (let hop = 0; hop < params.maxHops; hop += 1) {
      if (frontier.length === 0 || edges.size >= params.maxEdges || nodes.size >= params.maxNodes) {
        break;
      }
      const frontierPlaceholders = frontier.map(() => "?").join(", ");
      const rows = this.db
        .prepare(
          `SELECT edge_id, src_entity_id, rel_type, dst_entity_id, scope, agent_id, confidence, valid_from,
                  valid_to, relation_slot, evidence_ref, created_at, updated_at, materialized_epoch, metadata_json
             FROM graph_edges
            WHERE agent_id = ?
              AND scope IN (${scopePlaceholders})
              AND (valid_to IS NULL OR valid_to > ?)
              ${typeof params.readEpoch === "number" ? "AND materialized_epoch <= ?" : ""}
              AND (src_entity_id IN (${frontierPlaceholders}) OR dst_entity_id IN (${frontierPlaceholders}))
            ORDER BY confidence DESC, updated_at DESC
            LIMIT ${Math.max(params.maxEdges * 2, 16)}`,
        )
        .all(
          params.agentId,
          ...params.scopes,
          params.now ?? new Date().toISOString(),
          ...(typeof params.readEpoch === "number" ? [params.readEpoch] : []),
          ...frontier,
          ...frontier,
        ) as EdgeRow[];

      frontier = [];
      for (const row of rows) {
        if (edges.size >= params.maxEdges) {
          break;
        }
        const relation = normalizeGraphRelationType(row.rel_type);
        if (!relation) {
          continue;
        }
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
          relationSlot: row.relation_slot ?? undefined,
          confidence: row.confidence,
          evidenceRef: row.evidence_ref,
          updatedAt: row.updated_at,
          sourceKind: "stored",
          metadata: safeJsonParse<Record<string, unknown>>(row.metadata_json, {}),
        });
      }
    }

    for (const entityId of seenNodes) {
      const row = this.db
        .prepare(
          `SELECT entity_id, canonical_name, entity_type, normalized_name, aliases_json, confidence, created_at, updated_at
             FROM entities WHERE entity_id = ?`,
        )
        .get(entityId) as EntityRow | undefined;
      if (!row) {
        continue;
      }
      nodes.set(row.entity_id, {
        nodeId: row.entity_id,
        nodeKind: "entity",
        entityId: row.entity_id,
        name: row.canonical_name,
        type: row.entity_type,
        confidence: row.confidence,
      });
    }

    const edgeList = [...edges.values()].sort(
      (left, right) =>
        right.confidence - left.confidence ||
        (right.updatedAt ? Date.parse(right.updatedAt) : 0) -
          (left.updatedAt ? Date.parse(left.updatedAt) : 0),
    );
    const pathCandidates = buildGraphPathCandidates({
      seedNodeIds: params.seedEntityIds,
      nodes,
      edges: edgeList,
      now: params.now ?? new Date().toISOString(),
      maxPaths: Math.max(6, Math.min(params.maxEdges, 12)),
      maxHops: Math.min(2, Math.max(1, params.maxHops)),
    });

    return {
      nodes: [...nodes.values()],
      edges: edgeList,
      pathCandidates,
      paths: pathCandidates.map((path) => path.summary),
    };
  }

  pruneLowConfidence(params: {
    agentId: string;
    olderThan: string;
    maxConfidence: number;
  }): number {
    const result = this.db
      .prepare(
        `DELETE FROM graph_edges
          WHERE agent_id = ?
            AND confidence <= ?
            AND updated_at < ?`,
      )
      .run(params.agentId, params.maxConfidence, params.olderThan);
    return Number(result.changes ?? 0);
  }

  private toEntity(row: EntityRow): NormalizedEntity {
    return {
      entityId: row.entity_id,
      canonicalName: row.canonical_name,
      entityType: entityTypeValue(row.entity_type),
      normalizedName: row.normalized_name,
      aliases: safeJsonParse<string[]>(row.aliases_json, []),
      confidence: row.confidence,
    };
  }

  private toEntityMention(row: EntityMentionRow): EntityMention {
    return {
      mentionId: row.mention_id,
      agentId: row.agent_id,
      scope: row.scope,
      rawText: row.raw_text,
      normalizedText: row.normalized_text,
      proposedType: row.proposed_type as EntityMention["proposedType"],
      semanticRole: row.semantic_role as EntityMention["semanticRole"],
      sourceRef: row.source_ref,
      supportText: row.support_text,
      sessionKey: row.session_key ?? undefined,
      turnIndex: row.turn_index ?? undefined,
      observedAt: row.observed_at,
      resolvedEntityId: row.resolved_entity_id ?? undefined,
      resolutionMethod: (row.resolution_method ?? undefined) as EntityResolutionMethod | undefined,
      confidence: row.confidence,
      candidateIds: safeJsonParse<string[]>(row.candidate_ids_json, []),
      blockers: safeJsonParse<string[]>(row.blockers_json, []),
      metadataJson: safeJsonParse<Record<string, unknown>>(row.metadata_json, {}),
    };
  }
}
