import { objectRecord, safeJsonParse } from "../../support.mjs";
import { tokenizeSearchTerms } from "../../pipeline/semantic/heuristics.mjs";
import "../../pipeline/semantics.mjs";
//#region src/db/repositories/factRepo.ts
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
	"什么"
]);
function tokenizeSearch(text) {
	return tokenizeSearchTerms(text, SEARCH_STOPWORDS);
}
function stringArray(value) {
	return Array.isArray(value) ? value.filter((entry) => typeof entry === "string" && entry.trim().length > 0) : [];
}
function uniqueStrings(values) {
	return [...new Set(values.filter((value) => typeof value === "string" && value.trim().length > 0).map((value) => value.trim()))];
}
function sourceRefsFromFactMetadata(metadata) {
	if (!metadata) return [];
	const semanticAssertion = objectRecord(metadata.semanticAssertion);
	return uniqueStrings([
		typeof metadata.sourceRef === "string" ? metadata.sourceRef : void 0,
		typeof semanticAssertion?.sourceRef === "string" ? semanticAssertion.sourceRef : void 0,
		...stringArray(metadata.sourceRefs),
		...stringArray(metadata.supportRefs),
		...stringArray(metadata.supportContentRefs)
	]);
}
function supportTextFromFactMetadata(metadata) {
	if (!metadata) return "";
	const semanticAssertion = objectRecord(metadata.semanticAssertion);
	const semanticSupport = typeof semanticAssertion?.supportText === "string" ? semanticAssertion.supportText.trim() : "";
	if (semanticSupport) return semanticSupport;
	return typeof metadata.supportText === "string" ? metadata.supportText.trim() : "";
}
/**
* Canonical verb prefixes the LLM policy prompt constrains predicates to.
* Two facts with the same subject AND same verb AND similar topics are
* treated as covering the same semantic dimension and supersede each other.
* Facts with DIFFERENT verbs (e.g. `prefers_` vs `uses_`) are independent
* dimensions and are never fused.
*/
const ALLOWED_VERB_PREFIXES = new Set([
	"prefers",
	"uses",
	"has",
	"depends"
]);
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
	"configures"
]);
const COEXISTING_FACT_PREDICATES = new Set(["reported_detail"]);
const OBJECT_SCOPED_FACT_PREDICATES = new Set(["has_resource"]);
function normalizedObjectText(value) {
	return (value ?? "").trim().toLowerCase().replace(/[^\p{L}\p{N}]+/gu, " ").replace(/\s+/g, " ").trim();
}
function objectTokens(value) {
	return new Set(tokenizeSearch(normalizedObjectText(value)));
}
function objectValuesCompete(left, right) {
	const leftText = normalizedObjectText(left);
	const rightText = normalizedObjectText(right);
	if (!leftText || !rightText) return true;
	if (leftText === rightText) return true;
	const leftTokens = objectTokens(leftText);
	const rightTokens = objectTokens(rightText);
	if (leftTokens.size === 0 || rightTokens.size === 0) return true;
	let overlap = 0;
	for (const token of leftTokens) if (rightTokens.has(token)) overlap += 1;
	const smaller = Math.min(leftTokens.size, rightTokens.size);
	const union = leftTokens.size + rightTokens.size - overlap;
	const containment = overlap / smaller;
	const jaccard = overlap / union;
	return overlap >= 2 && containment >= .8 || jaccard >= .72;
}
function factsCompeteForPredicateValue(nextFact, priorFact) {
	if (!OBJECT_SCOPED_FACT_PREDICATES.has(nextFact.predicate)) return true;
	if (nextFact.predicate !== priorFact.predicate) return false;
	return objectValuesCompete(nextFact.canonicalObject, priorFact.canonicalObject);
}
var FactRepo = class FactRepo {
	db;
	constructor(db) {
		this.db = db;
	}
	static predicateAllowsCoexistingFacts(predicate) {
		return COEXISTING_FACT_PREDICATES.has(predicate);
	}
	static predicateCoexistsAcrossObjects(predicate) {
		return OBJECT_SCOPED_FACT_PREDICATES.has(predicate);
	}
	get(factId) {
		const row = this.db.prepare(`SELECT fact_id, canonical_subject, predicate, canonical_object, object_value_json, scope,
                agent_id, confidence, status, valid_from, valid_to, created_at, updated_at, materialized_epoch
           FROM facts
          WHERE fact_id = ?`).get(factId);
		return row ? this.toFact(row) : null;
	}
	supersedeActiveBySubjectAndPredicate(params) {
		const existing = this.findActiveBySemanticKey({
			agentId: params.agentId,
			scope: params.scope,
			canonicalSubject: params.canonicalSubject,
			predicate: params.predicate
		});
		for (const prior of existing) {
			this.db.prepare("UPDATE facts SET status = 'superseded', valid_to = ?, updated_at = ? WHERE fact_id = ?").run(params.updatedAt, params.updatedAt, prior.factId);
			this.db.prepare(`INSERT INTO fact_versions(
            version_id, fact_id, prior_snapshot_json, changed_at, change_reason, source_ref
          ) VALUES (?, ?, ?, ?, ?, ?)`).run(`${prior.factId}:${params.updatedAt}`, prior.factId, JSON.stringify(prior), params.updatedAt, params.changeReason, params.sourceRef);
		}
		return existing.length;
	}
	/**
	* Extract the canonical verb prefix from a predicate.
	* Returns the verb if the first underscore-delimited token is in
	* `ALLOWED_VERB_PREFIXES`, otherwise null (legacy or unstructured predicates).
	* `prefers_code_style` → `"prefers"`, `follows_coding_style` → null
	*/
	static predicateVerb(predicate) {
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
	static predicateTopic(predicate) {
		const underscore = predicate.indexOf("_");
		if (underscore === -1) return predicate;
		const first = predicate.slice(0, underscore);
		if (ALLOWED_VERB_PREFIXES.has(first) || LEGACY_VERB_TOKENS.has(first)) return predicate.slice(underscore + 1).replace(/_/g, " ").trim();
		return predicate.replace(/_/g, " ").trim();
	}
	findBySemanticKey(params) {
		return this.db.prepare(`SELECT fact_id, canonical_subject, predicate, canonical_object, object_value_json, scope,
                agent_id, confidence, status, valid_from, valid_to, created_at, updated_at, materialized_epoch
           FROM facts
          WHERE agent_id = ?
            AND scope = ?
            AND canonical_subject = ?
            AND predicate = ?
            AND (${params.includeHistorical ? "status != 'deleted'" : "status IN ('active', 'uncertain')"})
          ORDER BY updated_at DESC`).all(params.agentId, params.scope, params.canonicalSubject, params.predicate).map((row) => this.toFact(row));
	}
	findActiveBySemanticKey(params) {
		return this.findBySemanticKey(params);
	}
	findActiveBySubject(params) {
		return this.db.prepare(`SELECT fact_id, canonical_subject, predicate, canonical_object, object_value_json, scope,
                agent_id, confidence, status, valid_from, valid_to, created_at, updated_at, materialized_epoch
           FROM facts
          WHERE agent_id = ?
            AND scope = ?
            AND canonical_subject = ?
            AND status IN ('active', 'uncertain')
          ORDER BY updated_at DESC`).all(params.agentId, params.scope, params.canonicalSubject).map((row) => this.toFact(row));
	}
	upsert(fact, changeReason) {
		const predicateAllowsCoexistence = FactRepo.predicateAllowsCoexistingFacts(fact.predicate);
		const predicateCoexistsAcrossObjects = FactRepo.predicateCoexistsAcrossObjects(fact.predicate);
		const existing = this.findActiveBySemanticKey({
			agentId: fact.agentId,
			scope: fact.scope,
			canonicalSubject: fact.canonicalSubject,
			predicate: fact.predicate
		});
		const exact = existing.find((entry) => entry.canonicalObject === fact.canonicalObject && JSON.stringify(entry.objectValueJson ?? null) === JSON.stringify(fact.objectValueJson ?? null));
		if (exact) {
			this.db.prepare(`UPDATE facts
              SET confidence = ?, status = ?, valid_from = ?, valid_to = ?, updated_at = ?
            WHERE fact_id = ?`).run(fact.confidence, fact.status, fact.validFrom ?? null, fact.validTo ?? null, fact.updatedAt, exact.factId);
			return { action: "updated" };
		}
		if (predicateAllowsCoexistence) {
			this.db.prepare(`INSERT OR REPLACE INTO facts(
            fact_id, canonical_subject, predicate, predicate_verb, predicate_topic,
            canonical_object, object_value_json, scope, agent_id,
            confidence, status, valid_from, valid_to, created_at, updated_at, materialized_epoch
          ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`).run(fact.factId, fact.canonicalSubject, fact.predicate, FactRepo.predicateVerb(fact.predicate), FactRepo.predicateTopic(fact.predicate) || null, fact.canonicalObject ?? null, fact.objectValueJson ? JSON.stringify(fact.objectValueJson) : null, fact.scope, fact.agentId, fact.confidence, fact.status, fact.validFrom ?? null, fact.validTo ?? null, fact.createdAt, fact.updatedAt, fact.materializedEpoch ?? 0);
			return { action: "created" };
		}
		const competingExisting = predicateCoexistsAcrossObjects ? existing.filter((prior) => factsCompeteForPredicateValue(fact, prior)) : existing;
		if (competingExisting.length > 0) for (const prior of competingExisting) {
			this.db.prepare("UPDATE facts SET status = 'superseded', valid_to = ?, updated_at = ? WHERE fact_id = ?").run(fact.updatedAt, fact.updatedAt, prior.factId);
			this.db.prepare(`INSERT INTO fact_versions(
              version_id, fact_id, prior_snapshot_json, changed_at, change_reason, source_ref
            ) VALUES (?, ?, ?, ?, ?, ?)`).run(`${prior.factId}:${fact.updatedAt}`, prior.factId, JSON.stringify(prior), fact.updatedAt, changeReason, fact.sourceRef);
		}
		this.db.prepare(`INSERT OR REPLACE INTO facts(
          fact_id, canonical_subject, predicate, predicate_verb, predicate_topic,
          canonical_object, object_value_json, scope, agent_id,
          confidence, status, valid_from, valid_to, created_at, updated_at, materialized_epoch
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`).run(fact.factId, fact.canonicalSubject, fact.predicate, FactRepo.predicateVerb(fact.predicate), FactRepo.predicateTopic(fact.predicate) || null, fact.canonicalObject ?? null, fact.objectValueJson ? JSON.stringify(fact.objectValueJson) : null, fact.scope, fact.agentId, fact.confidence, fact.status, fact.validFrom ?? null, fact.validTo ?? null, fact.createdAt, fact.updatedAt, fact.materializedEpoch ?? 0);
		return { action: competingExisting.length > 0 ? "versioned" : "created" };
	}
	query(params) {
		if (params.scopes.length === 0) return [];
		const placeholders = params.scopes.map(() => "?").join(", ");
		const values = [params.agentId, ...params.scopes];
		let sql = `
      SELECT fact_id, canonical_subject, predicate, canonical_object, object_value_json, scope,
             agent_id, confidence, status, valid_from, valid_to, created_at, updated_at, materialized_epoch
        FROM facts
       WHERE agent_id = ?
         AND scope IN (${placeholders})
         AND status != 'deleted'
    `;
		if (!params.includeHistorical) sql += " AND status IN ('active', 'uncertain')";
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
		const facts = this.db.prepare(sql).all(...values).map((row) => this.toFact(row));
		if (!params.text) return facts.slice(0, params.limit ?? facts.length);
		const terms = tokenizeSearch(params.text);
		if (terms.length === 0) return facts.slice(0, params.limit ?? facts.length);
		const matched = facts.map((fact) => {
			const haystack = [
				fact.canonicalSubject,
				fact.predicate,
				fact.canonicalObject,
				fact.objectValueJson ? JSON.stringify(fact.objectValueJson) : ""
			].filter(Boolean).join(" ").toLowerCase();
			return {
				fact,
				score: terms.reduce((acc, term) => acc + (haystack.includes(term) ? 1 : 0), 0)
			};
		}).filter((entry) => entry.score > 0).sort((left, right) => right.score - left.score).slice(0, params.limit ?? facts.length).map((entry) => entry.fact);
		return matched.length > 0 ? matched : facts.slice(0, params.limit ?? facts.length);
	}
	markDeleted(params) {
		const clauses = ["agent_id = ?"];
		const values = [params.agentId];
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
		const result = this.db.prepare(`UPDATE facts SET status = 'deleted', updated_at = CURRENT_TIMESTAMP WHERE ${clauses.join(" AND ")}`).run(...values);
		return Number(result.changes ?? 0);
	}
	toFact(row) {
		const objectValueJson = safeJsonParse(row.object_value_json, void 0);
		const sourceRefs = sourceRefsFromFactMetadata(objectValueJson);
		return {
			factId: row.fact_id,
			canonicalSubject: row.canonical_subject,
			predicate: row.predicate,
			canonicalObject: row.canonical_object ?? void 0,
			objectValueJson,
			scope: row.scope,
			agentId: row.agent_id,
			confidence: row.confidence,
			status: row.status,
			validFrom: row.valid_from ?? void 0,
			validTo: row.valid_to ?? void 0,
			createdAt: row.created_at,
			updatedAt: row.updated_at,
			materializedEpoch: row.materialized_epoch,
			sourceRef: sourceRefs[0] ?? "",
			provenanceText: supportTextFromFactMetadata(objectValueJson)
		};
	}
};
//#endregion
export { FactRepo };
