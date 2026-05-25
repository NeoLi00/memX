import { tokenizeSearchTerms } from "../../pipeline/semantics.js";
import { objectRecord, safeJsonParse } from "../../support.js";
import type { NormalizedFact } from "../../types.js";
import type { MemxDbClient } from "../client.js";

type FactRow = {
  fact_id: string;
  canonical_subject: string;
  predicate: string;
  canonical_object: string | null;
  object_value_json: string | null;
  scope: string;
  agent_id: string;
  confidence: number;
  status: "active" | "superseded" | "deleted" | "uncertain";
  valid_from: string | null;
  valid_to: string | null;
  created_at: string;
  updated_at: string;
  materialized_epoch: number;
};

const SEARCH_STOPWORDS = new Set([
  "a",
  "an",
  "and",
  "do",
  "does",
  "i",
  "is",
  "last",
  "me",
  "my",
  "of",
  "on",
  "the",
  "to",
  "was",
  "what",
  "之前",
  "以前",
  "偏好",
  "回答",
  "格式",
  "风格",
  "什么",
]);

function tokenizeSearch(text: string): string[] {
  return tokenizeSearchTerms(text, SEARCH_STOPWORDS);
}

function stringArray(value: unknown): string[] {
  return Array.isArray(value)
    ? value.filter((entry): entry is string => typeof entry === "string" && entry.trim().length > 0)
    : [];
}

function uniqueStrings(values: Array<string | undefined | null>): string[] {
  return [
    ...new Set(
      values
        .filter((value): value is string => typeof value === "string" && value.trim().length > 0)
        .map((value) => value.trim()),
    ),
  ];
}

function sourceRefsFromFactMetadata(metadata: Record<string, unknown> | undefined): string[] {
  if (!metadata) {
    return [];
  }
  const semanticAssertion = objectRecord(metadata.semanticAssertion);
  return uniqueStrings([
    typeof metadata.sourceRef === "string" ? metadata.sourceRef : undefined,
    typeof semanticAssertion?.sourceRef === "string" ? semanticAssertion.sourceRef : undefined,
    ...stringArray(metadata.sourceRefs),
    ...stringArray(metadata.supportRefs),
    ...stringArray(metadata.supportContentRefs),
  ]);
}

function supportTextFromFactMetadata(metadata: Record<string, unknown> | undefined): string {
  if (!metadata) {
    return "";
  }
  const semanticAssertion = objectRecord(metadata.semanticAssertion);
  const semanticSupport =
    typeof semanticAssertion?.supportText === "string" ? semanticAssertion.supportText.trim() : "";
  if (semanticSupport) {
    return semanticSupport;
  }
  return typeof metadata.supportText === "string" ? metadata.supportText.trim() : "";
}

/**
 * Canonical verb prefixes the LLM policy prompt constrains predicates to.
 * Two facts with the same subject AND same verb AND similar topics are
 * treated as covering the same semantic dimension and supersede each other.
 * Facts with DIFFERENT verbs (e.g. `prefers_` vs `uses_`) are independent
 * dimensions and are never fused.
 */
const ALLOWED_VERB_PREFIXES = new Set(["prefers", "uses", "has", "depends"]);

/**
 * Verb tokens from pre-constraint LLM output (before the policy prompt
 * required canonical verbs).  Used in topic extraction for legacy facts.
 */
const LEGACY_VERB_TOKENS = new Set([
  "follows",
  "likes",
  "wants",
  "needs",
  "is",
  "does",
  "adopts",
  "sets",
  "gets",
  "configures",
]);

const COEXISTING_FACT_PREDICATES = new Set(["reported_detail"]);
const OBJECT_SCOPED_FACT_PREDICATES = new Set(["has_resource"]);

function slotFromObjectValueJson(value: Record<string, unknown> | undefined): string | null {
  const graph = objectRecord(value?.graph);
  const explicitSlot =
    typeof graph?.relationSlot === "string" && graph.relationSlot.trim()
      ? graph.relationSlot.trim()
      : typeof value?.componentRole === "string" && value.componentRole.trim()
        ? value.componentRole.trim()
        : null;
  return explicitSlot;
}

function factSupersessionSlot(
  fact: Pick<NormalizedFact, "predicate" | "objectValueJson">,
): string | null {
  const verb = FactRepo.predicateVerb(fact.predicate);
  if (verb !== "uses" && verb !== "depends") {
    return null;
  }
  return slotFromObjectValueJson(fact.objectValueJson);
}

function slotsCompatible(
  nextFact: Pick<NormalizedFact, "predicate" | "objectValueJson">,
  priorFact: Pick<NormalizedFact, "predicate" | "objectValueJson">,
): boolean {
  const nextSlot = factSupersessionSlot(nextFact);
  const priorSlot = factSupersessionSlot(priorFact);
  if (!nextSlot && !priorSlot) {
    return true;
  }
  return nextSlot === priorSlot;
}

function normalizedObjectText(value: string | null | undefined): string {
  return (value ?? "")
    .trim()
    .toLowerCase()
    .replace(/[^\p{L}\p{N}]+/gu, " ")
    .replace(/\s+/g, " ")
    .trim();
}

function objectTokens(value: string | null | undefined): Set<string> {
  return new Set(tokenizeSearch(normalizedObjectText(value)));
}

function objectValuesCompete(
  left: string | null | undefined,
  right: string | null | undefined,
): boolean {
  const leftText = normalizedObjectText(left);
  const rightText = normalizedObjectText(right);
  if (!leftText || !rightText) {
    return true;
  }
  if (leftText === rightText) {
    return true;
  }
  const leftTokens = objectTokens(leftText);
  const rightTokens = objectTokens(rightText);
  if (leftTokens.size === 0 || rightTokens.size === 0) {
    return true;
  }
  let overlap = 0;
  for (const token of leftTokens) {
    if (rightTokens.has(token)) {
      overlap += 1;
    }
  }
  const smaller = Math.min(leftTokens.size, rightTokens.size);
  const union = leftTokens.size + rightTokens.size - overlap;
  const containment = overlap / smaller;
  const jaccard = overlap / union;
  return (overlap >= 2 && containment >= 0.8) || jaccard >= 0.72;
}

function factsCompeteForPredicateValue(
  nextFact: Pick<NormalizedFact, "predicate" | "canonicalObject">,
  priorFact: Pick<NormalizedFact, "predicate" | "canonicalObject">,
): boolean {
  if (!OBJECT_SCOPED_FACT_PREDICATES.has(nextFact.predicate)) {
    return true;
  }
  if (nextFact.predicate !== priorFact.predicate) {
    return false;
  }
  return objectValuesCompete(nextFact.canonicalObject, priorFact.canonicalObject);
}

export class FactRepo {
  constructor(private readonly db: MemxDbClient) {}

  static predicateAllowsCoexistingFacts(predicate: string): boolean {
    return COEXISTING_FACT_PREDICATES.has(predicate);
  }

  static predicateCoexistsAcrossObjects(predicate: string): boolean {
    return OBJECT_SCOPED_FACT_PREDICATES.has(predicate);
  }

  get(factId: string): NormalizedFact | null {
    const row = this.db
      .prepare(
        `SELECT fact_id, canonical_subject, predicate, canonical_object, object_value_json, scope,
                agent_id, confidence, status, valid_from, valid_to, created_at, updated_at, materialized_epoch
           FROM facts
          WHERE fact_id = ?`,
      )
      .get(factId) as FactRow | undefined;
    return row ? this.toFact(row) : null;
  }

  supersedeActiveBySubjectAndPredicate(params: {
    agentId: string;
    scope: string;
    canonicalSubject: string;
    predicate: string;
    updatedAt: string;
    sourceRef: string;
    changeReason: string;
  }): number {
    const existing = this.findActiveBySemanticKey({
      agentId: params.agentId,
      scope: params.scope,
      canonicalSubject: params.canonicalSubject,
      predicate: params.predicate,
    });
    for (const prior of existing) {
      this.db
        .prepare(
          "UPDATE facts SET status = 'superseded', valid_to = ?, updated_at = ? WHERE fact_id = ?",
        )
        .run(params.updatedAt, params.updatedAt, prior.factId);
      this.db
        .prepare(
          `INSERT INTO fact_versions(
            version_id, fact_id, prior_snapshot_json, changed_at, change_reason, source_ref
          ) VALUES (?, ?, ?, ?, ?, ?)`,
        )
        .run(
          `${prior.factId}:${params.updatedAt}`,
          prior.factId,
          JSON.stringify(prior),
          params.updatedAt,
          params.changeReason,
          params.sourceRef,
        );
    }
    return existing.length;
  }

  /**
   * Extract the canonical verb prefix from a predicate.
   * Returns the verb if the first underscore-delimited token is in
   * `ALLOWED_VERB_PREFIXES`, otherwise null (legacy or unstructured predicates).
   * `prefers_code_style` → `"prefers"`, `follows_coding_style` → null
   */
  static predicateVerb(predicate: string): string | null {
    const underscore = predicate.indexOf("_");
    if (underscore === -1) return null;
    const first = predicate.slice(0, underscore);
    return ALLOWED_VERB_PREFIXES.has(first) ? first : null;
  }

  /**
   * Extract the topic portion of a predicate by stripping the leading verb prefix.
   * Both canonical (`ALLOWED_VERB_PREFIXES`) and legacy verbs are stripped.
   * Remaining underscores are converted to spaces.
   * `prefers_code_style` → `"code style"`, `follows_coding_style` → `"coding style"`.
   */
  static predicateTopic(predicate: string): string {
    const underscore = predicate.indexOf("_");
    if (underscore === -1) return predicate;
    const first = predicate.slice(0, underscore);
    if (ALLOWED_VERB_PREFIXES.has(first) || LEGACY_VERB_TOKENS.has(first)) {
      return predicate
        .slice(underscore + 1)
        .replace(/_/g, " ")
        .trim();
    }
    return predicate.replace(/_/g, " ").trim();
  }

  findBySemanticKey(params: {
    agentId: string;
    scope: string;
    canonicalSubject: string;
    predicate: string;
    includeHistorical?: boolean;
  }): NormalizedFact[] {
    const rows = this.db
      .prepare(
        `SELECT fact_id, canonical_subject, predicate, canonical_object, object_value_json, scope,
                agent_id, confidence, status, valid_from, valid_to, created_at, updated_at, materialized_epoch
           FROM facts
          WHERE agent_id = ?
            AND scope = ?
            AND canonical_subject = ?
            AND predicate = ?
            AND (${params.includeHistorical ? "status != 'deleted'" : "status IN ('active', 'uncertain')"})
          ORDER BY updated_at DESC`,
      )
      .all(params.agentId, params.scope, params.canonicalSubject, params.predicate) as FactRow[];
    return rows.map((row) => this.toFact(row));
  }

  findActiveBySemanticKey(params: {
    agentId: string;
    scope: string;
    canonicalSubject: string;
    predicate: string;
  }): NormalizedFact[] {
    return this.findBySemanticKey(params);
  }

  findActiveBySubject(params: {
    agentId: string;
    scope: string;
    canonicalSubject: string;
  }): NormalizedFact[] {
    const rows = this.db
      .prepare(
        `SELECT fact_id, canonical_subject, predicate, canonical_object, object_value_json, scope,
                agent_id, confidence, status, valid_from, valid_to, created_at, updated_at, materialized_epoch
           FROM facts
          WHERE agent_id = ?
            AND scope = ?
            AND canonical_subject = ?
            AND status IN ('active', 'uncertain')
          ORDER BY updated_at DESC`,
      )
      .all(params.agentId, params.scope, params.canonicalSubject) as FactRow[];
    return rows.map((row) => this.toFact(row));
  }

  upsert(
    fact: NormalizedFact,
    changeReason: string,
  ): { action: "created" | "updated" | "versioned" } {
    const predicateAllowsCoexistence = FactRepo.predicateAllowsCoexistingFacts(fact.predicate);
    const predicateCoexistsAcrossObjects = FactRepo.predicateCoexistsAcrossObjects(fact.predicate);
    const existing = this.findActiveBySemanticKey({
      agentId: fact.agentId,
      scope: fact.scope,
      canonicalSubject: fact.canonicalSubject,
      predicate: fact.predicate,
    });

    const exact = existing.find(
      (entry) =>
        entry.canonicalObject === fact.canonicalObject &&
        JSON.stringify(entry.objectValueJson ?? null) ===
          JSON.stringify(fact.objectValueJson ?? null),
    );
    if (exact) {
      this.db
        .prepare(
          `UPDATE facts
              SET confidence = ?, status = ?, valid_from = ?, valid_to = ?, updated_at = ?
            WHERE fact_id = ?`,
        )
        .run(
          fact.confidence,
          fact.status,
          fact.validFrom ?? null,
          fact.validTo ?? null,
          fact.updatedAt,
          exact.factId,
      );
      return { action: "updated" };
    }

    if (predicateAllowsCoexistence) {
      this.db
        .prepare(
          `INSERT OR REPLACE INTO facts(
            fact_id, canonical_subject, predicate, predicate_verb, predicate_topic,
            canonical_object, object_value_json, scope, agent_id,
            confidence, status, valid_from, valid_to, created_at, updated_at, materialized_epoch
          ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
        )
        .run(
          fact.factId,
          fact.canonicalSubject,
          fact.predicate,
          FactRepo.predicateVerb(fact.predicate),
          FactRepo.predicateTopic(fact.predicate) || null,
          fact.canonicalObject ?? null,
          fact.objectValueJson ? JSON.stringify(fact.objectValueJson) : null,
          fact.scope,
          fact.agentId,
          fact.confidence,
          fact.status,
          fact.validFrom ?? null,
          fact.validTo ?? null,
          fact.createdAt,
          fact.updatedAt,
          fact.materializedEpoch ?? 0,
        );
      return { action: "created" };
    }

    const competingExisting = predicateCoexistsAcrossObjects
      ? existing.filter((prior) => factsCompeteForPredicateValue(fact, prior))
      : existing;
    if (competingExisting.length > 0) {
      for (const prior of competingExisting) {
        this.db
          .prepare(
            "UPDATE facts SET status = 'superseded', valid_to = ?, updated_at = ? WHERE fact_id = ?",
          )
          .run(fact.updatedAt, fact.updatedAt, prior.factId);
        this.db
          .prepare(
            `INSERT INTO fact_versions(
              version_id, fact_id, prior_snapshot_json, changed_at, change_reason, source_ref
            ) VALUES (?, ?, ?, ?, ?, ?)`,
          )
          .run(
            `${prior.factId}:${fact.updatedAt}`,
            prior.factId,
            JSON.stringify(prior),
            fact.updatedAt,
            changeReason,
            fact.sourceRef,
          );
      }
    }

    this.db
      .prepare(
        `INSERT OR REPLACE INTO facts(
          fact_id, canonical_subject, predicate, predicate_verb, predicate_topic,
          canonical_object, object_value_json, scope, agent_id,
          confidence, status, valid_from, valid_to, created_at, updated_at, materialized_epoch
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
      )
      .run(
        fact.factId,
        fact.canonicalSubject,
        fact.predicate,
        FactRepo.predicateVerb(fact.predicate),
        FactRepo.predicateTopic(fact.predicate) || null,
        fact.canonicalObject ?? null,
        fact.objectValueJson ? JSON.stringify(fact.objectValueJson) : null,
        fact.scope,
        fact.agentId,
        fact.confidence,
        fact.status,
        fact.validFrom ?? null,
        fact.validTo ?? null,
        fact.createdAt,
        fact.updatedAt,
        fact.materializedEpoch ?? 0,
      );
    return { action: competingExisting.length > 0 ? "versioned" : "created" };
  }

  query(params: {
    agentId: string;
    scopes: string[];
    text?: string;
    predicate?: string;
    limit?: number;
    includeHistorical?: boolean;
    readEpoch?: number;
  }): NormalizedFact[] {
    if (params.scopes.length === 0) {
      return [];
    }
    const placeholders = params.scopes.map(() => "?").join(", ");
    const values: Array<string | number> = [params.agentId, ...params.scopes];
    let sql = `
      SELECT fact_id, canonical_subject, predicate, canonical_object, object_value_json, scope,
             agent_id, confidence, status, valid_from, valid_to, created_at, updated_at, materialized_epoch
        FROM facts
       WHERE agent_id = ?
         AND scope IN (${placeholders})
         AND status != 'deleted'
    `;
    if (!params.includeHistorical) {
      sql += " AND status IN ('active', 'uncertain')";
    }
    if (params.predicate) {
      sql += " AND predicate = ?";
      values.push(params.predicate);
    }
    if (typeof params.readEpoch === "number") {
      sql += " AND materialized_epoch <= ?";
      values.push(params.readEpoch);
    }
    sql += " ORDER BY updated_at DESC";
    sql += ` LIMIT ${Math.max(24, Math.trunc((params.limit ?? 6) * 6))}`;
    const rows = this.db.prepare(sql).all(...values) as FactRow[];
    const facts = rows.map((row) => this.toFact(row));
    if (!params.text) {
      return facts.slice(0, params.limit ?? facts.length);
    }
    const terms = tokenizeSearch(params.text);
    if (terms.length === 0) {
      return facts.slice(0, params.limit ?? facts.length);
    }
    const matched = facts
      .map((fact) => {
        const haystack = [
          fact.canonicalSubject,
          fact.predicate,
          fact.canonicalObject,
          fact.objectValueJson ? JSON.stringify(fact.objectValueJson) : "",
        ]
          .filter(Boolean)
          .join(" ")
          .toLowerCase();
        const score = terms.reduce((acc, term) => acc + (haystack.includes(term) ? 1 : 0), 0);
        return { fact, score };
      })
      .filter((entry) => entry.score > 0)
      .sort((left, right) => right.score - left.score)
      .slice(0, params.limit ?? facts.length)
      .map((entry) => entry.fact);
    return matched.length > 0 ? matched : facts.slice(0, params.limit ?? facts.length);
  }

  markDeleted(params: {
    agentId: string;
    scope?: string;
    factId?: string;
    subject?: string;
  }): number {
    const clauses = ["agent_id = ?"];
    const values: Array<string | number> = [params.agentId];
    if (params.scope) {
      clauses.push("scope = ?");
      values.push(params.scope);
    }
    if (params.factId) {
      clauses.push("fact_id = ?");
      values.push(params.factId);
    }
    if (params.subject) {
      clauses.push("canonical_subject = ?");
      values.push(params.subject);
    }
    const result = this.db
      .prepare(
        `UPDATE facts SET status = 'deleted', updated_at = CURRENT_TIMESTAMP WHERE ${clauses.join(" AND ")}`,
      )
      .run(...values);
    return Number(result.changes ?? 0);
  }

  private toFact(row: FactRow): NormalizedFact {
    const objectValueJson = safeJsonParse<Record<string, unknown> | undefined>(
      row.object_value_json,
      undefined,
    );
    const sourceRefs = sourceRefsFromFactMetadata(objectValueJson);
    return {
      factId: row.fact_id,
      canonicalSubject: row.canonical_subject,
      predicate: row.predicate,
      canonicalObject: row.canonical_object ?? undefined,
      objectValueJson,
      scope: row.scope,
      agentId: row.agent_id,
      confidence: row.confidence,
      status: row.status,
      validFrom: row.valid_from ?? undefined,
      validTo: row.valid_to ?? undefined,
      createdAt: row.created_at,
      updatedAt: row.updated_at,
      materializedEpoch: row.materialized_epoch,
      sourceRef: sourceRefs[0] ?? "",
      provenanceText: supportTextFromFactMetadata(objectValueJson),
    };
  }
}
