import type { MemxStoreBundle } from "../runtime.js";
import { clamp01, normalizeName, normalizeText, stableHash, truncateText } from "../support.js";
import type {
  EntityDisambiguationDecision,
  EntityMention,
  EntityMentionSemanticRole,
  EntityResolutionCandidate,
  EntityResolutionMethod,
  EntityResolutionResult,
  EntityType,
  MemoryOperationContext,
  NormalizedEntity,
} from "../types.js";
import { refreshEntityProfileDocs } from "./entityProfile.js";
import { projectAliasVariants, projectIdentityKey } from "./projectIdentity.js";
import { semanticTextSimilarity } from "./semantic/textSimilarity.js";

type BuildEntityMentionParams = {
  ctx: MemoryOperationContext;
  scope: string;
  rawText: string;
  proposedType?: EntityType;
  semanticRole: EntityMentionSemanticRole;
  sourceRef: string;
  supportText: string;
  observedAt: string;
  sessionKey?: string;
  turnIndex?: number;
  metadataJson?: Record<string, unknown>;
};

type ResolveOptions = {
  persist?: boolean;
  createIfMissing?: boolean;
  disambiguate?: (input: {
    mention: EntityMention;
    candidates: EntityResolutionCandidate[];
  }) => EntityDisambiguationDecision | null;
};

const LATIN_ENTITY_DESCRIPTOR_WORDS = new Set([
  "api",
  "app",
  "billing",
  "cache",
  "client",
  "component",
  "database",
  "db",
  "engine",
  "gateway",
  "module",
  "package",
  "payment",
  "pipeline",
  "platform",
  "plugin",
  "project",
  "queue",
  "repo",
  "repository",
  "service",
  "server",
  "storage",
  "system",
  "tool",
  "webhook",
  "webhooks",
  "worker",
  "workspace",
]);

const CJK_FAMILY_RE =
  /[\p{Script=Han}\p{Script=Hiragana}\p{Script=Katakana}\p{Script=Hangul}]/u;
const LATIN_TOKEN_RE = /[A-Za-z][A-Za-z0-9_.-]{2,}/gu;
const CAMEL_OR_CODE_RE =
  /(?:[A-Z][a-z0-9]+[A-Z][A-Za-z0-9]*|[A-Za-z]+[0-9][A-Za-z0-9]*|[A-Za-z0-9]+[_.-][A-Za-z0-9_.-]+)/u;

function typedEntityId(normalizedText: string, proposedType: EntityType): string {
  if (proposedType === "unknown") {
    return stableHash(["entity", normalizedText]);
  }
  return stableHash(["entity", proposedType, normalizedText]);
}

function buildNewEntity(mention: EntityMention): NormalizedEntity {
  const aliases =
    mention.proposedType === "project" ? projectAliasVariants(mention.rawText.trim()) : [];
  return {
    entityId: typedEntityId(mention.normalizedText, mention.proposedType),
    canonicalName: mention.rawText.trim(),
    entityType: mention.proposedType,
    normalizedName: mention.normalizedText,
    aliases,
    confidence: Math.max(0.55, Math.min(0.8, mention.confidence)),
  };
}

function usesWeakProposedType(mention: EntityMention): boolean {
  return mention.metadataJson.weakProposedType === true;
}

function preferredLookupType(mention: EntityMention): EntityType | undefined {
  return mention.proposedType === "unknown" ? undefined : mention.proposedType;
}

function mentionHasProjectDescriptorAlias(mention: EntityMention): boolean {
  const compactNormalized = mention.normalizedText.replace(/\s+/g, "");
  const projectKey = projectIdentityKey(mention.rawText);
  return Boolean(projectKey) && projectKey !== compactNormalized;
}

function entityMentionProfileQuery(mention: EntityMention): string {
  const nearbyEntityNames = stringArray(mention.metadataJson.nearbyEntityNames).join(" ");
  const nearbyRelationHints = stringArray(mention.metadataJson.nearbyRelationHints).join(" ");
  return [
    mention.rawText,
    mention.proposedType,
    mention.semanticRole,
    mention.supportText,
    nearbyEntityNames,
    nearbyRelationHints,
  ]
    .filter(Boolean)
    .join("\n");
}

function typeCompatibilityScore(mention: EntityMention, entity: NormalizedEntity): number {
  if (mention.proposedType === "unknown" || entity.entityType === "unknown") {
    return 0.74;
  }
  return mention.proposedType === entity.entityType ? 1 : 0;
}

function aliasScore(mention: EntityMention, entity: NormalizedEntity): number {
  const normalized = mention.normalizedText;
  if (!normalized) return 0;
  return entity.aliases.some((alias) => normalizeName(alias) === normalized) ? 1 : 0;
}

function exactNameScore(mention: EntityMention, entity: NormalizedEntity): number {
  if (!mention.normalizedText) return 0;
  return entity.normalizedName === mention.normalizedText ? 1 : 0;
}

function recencyScore(updatedAt: unknown, nowIso: string): number {
  if (typeof updatedAt !== "string") return 0.42;
  const updated = Date.parse(updatedAt);
  const now = Date.parse(nowIso);
  if (!Number.isFinite(updated) || !Number.isFinite(now)) return 0.42;
  const ageDays = Math.max(0, (now - updated) / 86_400_000);
  return clamp01(Math.exp(-ageDays / 180));
}

function profileVectorCandidates(
  store: MemxStoreBundle,
  ctx: MemoryOperationContext,
  mention: EntityMention,
): EntityResolutionCandidate[] {
  const query = entityMentionProfileQuery(mention);
  const docs = store.vectorRepo.listDocs({
    agentId: ctx.agentId,
    scopes: [mention.scope],
    limit: 256,
    readEpoch: ctx.readEpoch,
    docKinds: ["entity_profile"],
    docTypes: ["entity_profile"],
  });
  const candidates: EntityResolutionCandidate[] = [];
  for (const doc of docs) {
    const entityId =
      typeof doc.metadataJson.entityId === "string" ? doc.metadataJson.entityId : doc.sourceId;
    const entity = store.graphRepo.getEntityById(entityId);
    if (!entity) {
      continue;
    }
    const exact = exactNameScore(mention, entity);
    const alias = aliasScore(mention, entity);
    const embedding = Math.max(
      semanticTextSimilarity(query, doc.text),
      semanticTextSimilarity(`${mention.rawText} ${mention.supportText}`, doc.text),
    );
    if (embedding < 0.6 && exact === 0 && alias === 0) {
      continue;
    }
    const typeCompatibility = typeCompatibilityScore(mention, entity);
    const contradictionPenalty = typeCompatibility === 0 ? 0.35 : 0;
    const scopeSessionTaskFit = doc.scope === mention.scope ? 1 : 0.45;
    const cooccurrenceOverlap = semanticTextSimilarity(mention.supportText, doc.text);
    const graphNeighborhoodOverlap = Math.max(
      0,
      ...stringArray(doc.metadataJson.relationNeighborIds).map((neighbor) =>
        normalizeText(query).includes(normalizeText(neighbor)) ? 1 : 0,
      ),
    );
    const recency = recencyScore(doc.updatedAt, ctx.now);
    const score = clamp01(
      exact * 0.2 +
        alias * 0.18 +
        embedding * 0.22 +
        typeCompatibility * 0.12 +
        scopeSessionTaskFit * 0.1 +
        cooccurrenceOverlap * 0.08 +
        graphNeighborhoodOverlap * 0.06 +
        recency * 0.04 -
        contradictionPenalty,
    );
    candidates.push({
      entity,
      score,
      exactNameScore: exact,
      aliasScore: alias,
      embeddingScore: embedding,
      typeCompatibility,
      scopeSessionTaskFit,
      cooccurrenceOverlap,
      graphNeighborhoodOverlap,
      recency,
      contradictionPenalty,
      source: "profile_vector",
      metadataJson: {
        docId: doc.docId,
        profileText: truncateText(doc.text, 360),
      },
    });
  }
  return candidates.sort((left, right) => right.score - left.score).slice(0, 15);
}

function strongNonEmbeddingSupport(candidate: EntityResolutionCandidate): number {
  return clamp01(
    candidate.exactNameScore * 0.2 +
      candidate.aliasScore * 0.18 +
      candidate.typeCompatibility * 0.12 +
      candidate.scopeSessionTaskFit * 0.1 +
      candidate.cooccurrenceOverlap * 0.08 +
      candidate.graphNeighborhoodOverlap * 0.06 +
      candidate.recency * 0.04,
  );
}

function candidateScores(candidates: EntityResolutionCandidate[]): Array<Record<string, unknown>> {
  return candidates.map((candidate) => ({
    entityId: candidate.entity.entityId,
    canonicalName: candidate.entity.canonicalName,
    score: candidate.score,
    exactNameScore: candidate.exactNameScore,
    aliasScore: candidate.aliasScore,
    embeddingScore: candidate.embeddingScore,
    typeCompatibility: candidate.typeCompatibility,
    scopeSessionTaskFit: candidate.scopeSessionTaskFit,
    cooccurrenceOverlap: candidate.cooccurrenceOverlap,
    graphNeighborhoodOverlap: candidate.graphNeighborhoodOverlap,
    recency: candidate.recency,
    contradictionPenalty: candidate.contradictionPenalty,
    source: candidate.source,
  }));
}

function resolutionResult(params: {
  mention: EntityMention;
  entity: NormalizedEntity;
  method: EntityResolutionMethod;
  confidence: number;
  candidateEntityIds?: string[];
  blockers?: string[];
  createdEntity?: boolean;
  candidates?: EntityResolutionCandidate[];
  llmDecision?: EntityDisambiguationDecision | null;
}): EntityResolutionResult {
  const resolvedEntityId =
    params.method === "uncertain" && params.createdEntity !== true
      ? undefined
      : params.entity.entityId;
  return {
    mention: {
      ...params.mention,
      resolvedEntityId,
      resolutionMethod: params.method,
      confidence: params.confidence,
      candidateIds: params.candidateEntityIds ?? [params.entity.entityId],
      blockers: params.blockers ?? [],
      metadataJson: {
        ...params.mention.metadataJson,
        entityResolution: {
          method: params.method,
          confidence: params.confidence,
          candidateEntityIds: params.candidateEntityIds ?? [params.entity.entityId],
          ...(params.candidates ? { candidateScores: candidateScores(params.candidates) } : {}),
          ...(params.llmDecision ? { llmDecision: params.llmDecision } : {}),
        },
      },
    },
    entity: params.entity,
    method: params.method,
    confidence: params.confidence,
    candidateEntityIds: params.candidateEntityIds ?? [params.entity.entityId],
    blockers: params.blockers ?? [],
    createdEntity: params.createdEntity ?? false,
  };
}

function stringArray(value: unknown): string[] {
  if (!Array.isArray(value)) return [];
  return value.filter(
    (entry): entry is string => typeof entry === "string" && entry.trim().length > 0,
  );
}

function isDescriptorToken(value: string): boolean {
  return LATIN_ENTITY_DESCRIPTOR_WORDS.has(normalizeText(value));
}

function entityIdentityAnchorsFromText(value: string): string[] {
  const anchors = new Set<string>();
  const rawTokens = value.match(LATIN_TOKEN_RE) ?? [];
  const hasCjk = CJK_FAMILY_RE.test(value);
  for (const token of rawTokens) {
    const normalized = normalizeName(token);
    if (normalized.length < 3 || isDescriptorToken(normalized)) {
      continue;
    }
    if (CAMEL_OR_CODE_RE.test(token) || hasCjk) {
      anchors.add(normalized);
    }
  }
  return [...anchors];
}

function entityIdentityAnchors(entity: NormalizedEntity): string[] {
  return [
    ...new Set(
      [entity.canonicalName, entity.normalizedName, ...entity.aliases].flatMap((entry) =>
        entityIdentityAnchorsFromText(entry),
      ),
    ),
  ];
}

function identityAnchorCandidates(
  store: MemxStoreBundle,
  mention: EntityMention,
): EntityResolutionCandidate[] {
  const mentionAnchors = entityIdentityAnchorsFromText(mention.rawText);
  if (mentionAnchors.length === 0) {
    return [];
  }
  const candidates = new Map<string, EntityResolutionCandidate>();
  for (const anchor of mentionAnchors) {
    for (const entity of store.graphRepo.searchEntitiesByQuery(anchor, 12)) {
      const entityAnchors = new Set(entityIdentityAnchors(entity));
      if (!entityAnchors.has(anchor)) {
        continue;
      }
      const typeCompatibility = typeCompatibilityScore(mention, entity);
      if (typeCompatibility === 0) {
        continue;
      }
      const exact = exactNameScore(mention, entity);
      const alias = aliasScore(mention, entity);
      const candidate: EntityResolutionCandidate = {
        entity,
        score: clamp01(
          0.62 +
            typeCompatibility * 0.14 +
            entity.confidence * 0.1 +
            (exact > 0 ? 0.08 : 0) +
            (alias > 0 ? 0.06 : 0),
        ),
        exactNameScore: exact,
        aliasScore: alias,
        embeddingScore: 0,
        typeCompatibility,
        scopeSessionTaskFit: 0.72,
        cooccurrenceOverlap: 0,
        graphNeighborhoodOverlap: 0,
        recency: 0.5,
        contradictionPenalty: 0,
        source: "identity_anchor",
        metadataJson: { sharedAnchor: anchor, mentionAnchors, entityAnchors: [...entityAnchors] },
      };
      const existing = candidates.get(entity.entityId);
      if (!existing || candidate.score > existing.score) {
        candidates.set(entity.entityId, candidate);
      }
    }
  }
  return [...candidates.values()].sort((left, right) => right.score - left.score).slice(0, 6);
}

function persistResolution(
  store: MemxStoreBundle,
  ctx: MemoryOperationContext,
  result: EntityResolutionResult,
): void {
  store.graphRepo.upsertEntity(result.entity);
  store.graphRepo.upsertEntityMention(result.mention);

  const normalizedAlias = normalizeName(result.mention.rawText);
  if (
    normalizedAlias &&
    normalizedAlias !== result.entity.normalizedName &&
    (result.method === "alias" ||
      result.method === "project_identity" ||
      result.method === "identity_anchor" ||
      result.method === "identity_link" ||
      result.method === "llm_candidate")
  ) {
    store.graphRepo.upsertEntityAliasSource({
      entityId: result.entity.entityId,
      aliasText: result.mention.rawText,
      sourceRef: result.mention.sourceRef,
      confidence: result.confidence,
      createdAt: result.mention.observedAt,
      metadataJson: {
        method: result.method,
        mentionId: result.mention.mentionId,
      },
    });
  }

  for (const link of result.identityLinks ?? []) {
    store.graphRepo.upsertIdentityLink({
      srcEntityId: link.srcEntityId,
      dstEntityId: link.dstEntityId,
      linkType: link.linkType,
      confidence: link.confidence,
      evidenceRef: link.evidenceRef,
      status: link.status,
      at: ctx.now,
      metadataJson: link.metadataJson,
    });
  }
  refreshEntityProfileDocs(store, ctx, [
    result.entity.entityId,
    ...(result.identityLinks ?? []).flatMap((link) => [link.srcEntityId, link.dstEntityId]),
  ]);
}

function supersedingResolution(
  store: MemxStoreBundle,
  matched: NormalizedEntity,
): { entity: NormalizedEntity; confidence: number; candidateEntityIds: string[] } | null {
  const superseding = store.graphRepo.findSupersedingEntity(matched.entityId);
  if (!superseding) {
    return null;
  }
  return {
    entity: superseding,
    confidence: Math.max(0.8, Math.min(0.92, superseding.confidence)),
    candidateEntityIds: [superseding.entityId, matched.entityId],
  };
}

export function buildEntityMention(params: BuildEntityMentionParams): EntityMention {
  const rawText = params.rawText.trim();
  const normalizedText = normalizeName(rawText);
  const mentionId = stableHash([
    "entity-mention",
    params.ctx.agentId,
    params.scope,
    params.sourceRef,
    params.semanticRole,
    normalizedText,
    rawText,
  ]);
  return {
    mentionId,
    agentId: params.ctx.agentId,
    scope: params.scope,
    rawText,
    normalizedText,
    proposedType: params.proposedType ?? "unknown",
    semanticRole: params.semanticRole,
    sourceRef: params.sourceRef,
    supportText: truncateText(params.supportText, 720),
    sessionKey: params.sessionKey ?? params.ctx.sessionKey,
    turnIndex: params.turnIndex,
    observedAt: params.observedAt,
    confidence: 0.72,
    candidateIds: [],
    blockers: normalizedText ? [] : ["empty_normalized_text"],
    metadataJson: params.metadataJson ?? {},
  };
}

export function resolveEntityMention(
  store: MemxStoreBundle,
  ctx: MemoryOperationContext,
  mention: EntityMention,
  options: ResolveOptions = {},
): EntityResolutionResult {
  const persist = options.persist ?? true;
  const createIfMissing = options.createIfMissing ?? true;
  if (!mention.normalizedText) {
    const entity = {
      entityId: stableHash(["entity", "uncertain", mention.mentionId]),
      canonicalName: mention.rawText || "unknown",
      entityType: mention.proposedType,
      normalizedName: mention.normalizedText,
      aliases: [],
      confidence: 0.1,
    };
    const result = resolutionResult({
      mention,
      entity,
      method: "uncertain",
      confidence: 0.1,
      blockers: ["empty_normalized_text"],
      createdEntity: false,
    });
    if (persist) {
      store.graphRepo.upsertEntityMention(result.mention);
    }
    return result;
  }

  const exact = store.graphRepo.lookupEntityByNormalizedName(
    mention.normalizedText,
    preferredLookupType(mention),
  );
  if (exact) {
    const superseding = supersedingResolution(store, exact);
    if (superseding) {
      const result = resolutionResult({
        mention,
        entity: superseding.entity,
        method: "identity_link",
        confidence: superseding.confidence,
        candidateEntityIds: superseding.candidateEntityIds,
      });
      if (persist) persistResolution(store, ctx, result);
      return result;
    }
    const result = resolutionResult({
      mention,
      entity: exact,
      method: "exact",
      confidence: Math.max(0.82, exact.confidence),
    });
    if (persist) persistResolution(store, ctx, result);
    return result;
  }
  if (usesWeakProposedType(mention) && mention.proposedType !== "unknown") {
    const exactAnyType = store.graphRepo.lookupEntityByNormalizedName(mention.normalizedText);
    if (exactAnyType) {
      const superseding = supersedingResolution(store, exactAnyType);
      if (superseding) {
        const result = resolutionResult({
          mention,
          entity: superseding.entity,
          method: "identity_link",
          confidence: superseding.confidence,
          candidateEntityIds: superseding.candidateEntityIds,
        });
        if (persist) persistResolution(store, ctx, result);
        return result;
      }
      const result = resolutionResult({
        mention,
        entity: exactAnyType,
        method: "exact",
        confidence: Math.max(0.8, exactAnyType.confidence),
      });
      if (persist) persistResolution(store, ctx, result);
      return result;
    }
  }

  const alias = store.graphRepo.lookupEntityByAlias(
    mention.normalizedText,
    preferredLookupType(mention),
  );
  if (alias) {
    const superseding = supersedingResolution(store, alias);
    if (superseding) {
      const result = resolutionResult({
        mention,
        entity: superseding.entity,
        method: "identity_link",
        confidence: superseding.confidence,
        candidateEntityIds: superseding.candidateEntityIds,
      });
      if (persist) persistResolution(store, ctx, result);
      return result;
    }
    const result = resolutionResult({
      mention,
      entity: alias,
      method: "alias",
      confidence: Math.max(0.78, alias.confidence),
    });
    if (persist) persistResolution(store, ctx, result);
    return result;
  }

  const projectCandidate =
    mention.proposedType === "project" ||
    mention.semanticRole === "project" ||
    mentionHasProjectDescriptorAlias(mention)
      ? store.graphRepo.lookupEntityByName(mention.rawText, "project")
      : null;
  if (
    projectCandidate &&
    projectIdentityKey(projectCandidate.canonicalName) === projectIdentityKey(mention.rawText)
  ) {
    const result = resolutionResult({
      mention,
      entity: projectCandidate,
      method: "project_identity",
      confidence: Math.max(0.76, projectCandidate.confidence),
    });
    if (persist) persistResolution(store, ctx, result);
    return result;
  }

  const anchoredCandidates = identityAnchorCandidates(store, mention);
  const topAnchorCandidate = anchoredCandidates[0];
  const secondAnchorCandidate = anchoredCandidates[1];
  if (
    topAnchorCandidate &&
    topAnchorCandidate.score >= 0.8 &&
    (!secondAnchorCandidate || topAnchorCandidate.score - secondAnchorCandidate.score >= 0.08)
  ) {
    const result = resolutionResult({
      mention,
      entity: topAnchorCandidate.entity,
      method: "identity_anchor",
      confidence: Math.max(0.8, topAnchorCandidate.score),
      candidateEntityIds: anchoredCandidates.map((candidate) => candidate.entity.entityId),
      candidates: anchoredCandidates,
    });
    if (persist) persistResolution(store, ctx, result);
    return result;
  }

  const cooccurring = store.graphRepo
    .listResolvedMentionsByNormalized({
      agentId: mention.agentId,
      scope: mention.scope,
      normalizedText: mention.normalizedText,
      sessionKey: mention.sessionKey,
      limit: 1,
    })
    .at(0);
  if (cooccurring?.resolvedEntityId) {
    const entity = store.graphRepo.getEntityById(cooccurring.resolvedEntityId);
    const typeCompatible =
      !entity ||
      mention.proposedType === "unknown" ||
      entity.entityType === "unknown" ||
      entity.entityType === mention.proposedType;
    if (entity && typeCompatible) {
      const result = resolutionResult({
        mention,
        entity,
        method: "cooccurrence",
        confidence: Math.max(0.7, Math.min(0.86, cooccurring.confidence)),
      });
      if (persist) persistResolution(store, ctx, result);
      return result;
    }
  }

  const profileCandidates = profileVectorCandidates(store, ctx, mention);
  const topProfile = profileCandidates[0];
  if (
    topProfile &&
    topProfile.score >= 0.86 &&
    topProfile.contradictionPenalty === 0 &&
    strongNonEmbeddingSupport(topProfile) >= 0.34
  ) {
    const result = resolutionResult({
      mention,
      entity: topProfile.entity,
      method: "embedding_candidate",
      confidence: Math.max(0.78, topProfile.score),
      candidateEntityIds: profileCandidates.map((candidate) => candidate.entity.entityId),
      candidates: profileCandidates,
    });
    if (persist) persistResolution(store, ctx, result);
    return result;
  }
  const inAmbiguousProfileBand =
    topProfile &&
    topProfile.embeddingScore >= 0.6 &&
    topProfile.score >= 0.5 &&
    topProfile.score < 0.86;
  if (topProfile && inAmbiguousProfileBand && options.disambiguate) {
    const decision = options.disambiguate({
      mention,
      candidates: profileCandidates.slice(0, 5),
    });
    if (decision?.decision === "match" && decision.matchedEntityId && decision.confidence >= 0.78) {
      const matched = profileCandidates.find(
        (candidate) => candidate.entity.entityId === decision.matchedEntityId,
      );
      if (matched) {
        const result = resolutionResult({
          mention,
          entity: matched.entity,
          method: "llm_candidate",
          confidence: decision.confidence,
          candidateEntityIds: profileCandidates.map((candidate) => candidate.entity.entityId),
          candidates: profileCandidates,
          llmDecision: decision,
        });
        if (persist) persistResolution(store, ctx, result);
        return result;
      }
    }
    if (decision?.decision === "no_match") {
      profileCandidates.length = 0;
    }
  }

  const nameSearchCandidates: EntityResolutionCandidate[] = store.graphRepo
    .searchEntitiesByQuery(mention.rawText, 6)
    .map((candidate) => ({
      entity: candidate,
      score: 0.5,
      exactNameScore: exactNameScore(mention, candidate),
      aliasScore: aliasScore(mention, candidate),
      embeddingScore: 0,
      typeCompatibility: typeCompatibilityScore(mention, candidate),
      scopeSessionTaskFit: 0.5,
      cooccurrenceOverlap: 0,
      graphNeighborhoodOverlap: 0,
      recency: 0.4,
      contradictionPenalty: typeCompatibilityScore(mention, candidate) === 0 ? 0.35 : 0,
      source: "name_search",
    }));
  const candidates = [...profileCandidates, ...nameSearchCandidates]
    .sort((left, right) => right.score - left.score)
    .filter(
      (candidate, index, all) =>
        all.findIndex((entry) => entry.entity.entityId === candidate.entity.entityId) === index,
    );
  const entity = createIfMissing
    ? buildNewEntity(mention)
    : {
        entityId: stableHash(["entity", "uncertain", mention.mentionId]),
        canonicalName: mention.rawText,
        entityType: mention.proposedType,
        normalizedName: mention.normalizedText,
        aliases: [],
        confidence: 0.2,
      };
  const result = resolutionResult({
    mention,
    entity,
    method: createIfMissing ? "new_entity" : "uncertain",
    confidence: createIfMissing ? entity.confidence : 0.2,
    candidateEntityIds: candidates.map((candidate) => candidate.entity.entityId),
    blockers: candidates.length > 0 ? ["candidate_shortlist_unresolved"] : [],
    createdEntity: createIfMissing,
    candidates,
  });
  if (candidates.length > 0) {
    result.identityLinks = candidates
      .filter((candidate) => candidate.entity.entityId !== entity.entityId)
      .map((candidate) => ({
        srcEntityId: entity.entityId,
        dstEntityId: candidate.entity.entityId,
        linkType: "possible_same_as" as const,
        confidence: Math.min(0.61, Math.max(0.42, candidate.score)),
        evidenceRef: mention.sourceRef,
        status: "active" as const,
        metadataJson: {
          reason: "candidate-shortlist-without-confident-resolution",
          mentionId: mention.mentionId,
          candidateScore: candidate.score,
          candidateSource: candidate.source,
        },
      }));
  }
  if (persist) {
    if (result.method === "uncertain" && !result.createdEntity) {
      store.graphRepo.upsertEntityMention(result.mention);
    } else {
      persistResolution(store, ctx, result);
    }
  }
  return result;
}

export function resolveEntityMentions(
  store: MemxStoreBundle,
  ctx: MemoryOperationContext,
  mentions: EntityMention[],
  options: ResolveOptions = {},
): EntityResolutionResult[] {
  return mentions.map((mention) => resolveEntityMention(store, ctx, mention, options));
}
