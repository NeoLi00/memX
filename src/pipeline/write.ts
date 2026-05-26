import type { MemxStoreBundle } from "../runtime.js";
import { normalizeName, objectRecord, stableHash } from "../support.js";
import type { ClassifiedCandidate, EntityType, MemoryOperationContext } from "../types.js";
import type {
  NormalizedEntity,
  NormalizedFact,
  NormalizedGraphEdge,
  NormalizedState,
  VectorDocRecord,
} from "../types.js";
import { shouldDeriveProjectProfileArtifacts } from "./authority.js";
import {
  attributeSlotFromPredicate,
  semanticFactPredicatesForAttributeSlot,
} from "./attributeSlots.js";
import { refreshEntityProfileDocs } from "./entityProfile.js";
import { buildEntityMention, resolveEntityMention } from "./entityResolver.js";
import { snapshotMemoryLlmBudgetAudit } from "./llmBudgetAudit.js";
import { normalizeCandidate } from "./normalize.js";
import {
  isProjectProfileStateKey,
  projectAliasVariants,
  projectCodeFromStateKey,
  resolveProjectReference,
} from "./projectIdentity.js";
import { emitWriteMaterializationSignals } from "./signalLedger.js";
import { buildVectorDocMetadata } from "./vectorDocMetadata.js";

export type WriteSummary = {
  states: number;
  facts: number;
  events: number;
  entities: number;
  edges: number;
  vectorDocs: number;
};

function dedupWindowStart(observedAt: string, days: number): string {
  const date = new Date(observedAt);
  date.setUTCDate(date.getUTCDate() - days);
  return date.toISOString();
}

function projectProfileStateKey(projectCode: string): string {
  return `project.${projectCode.trim()}`;
}

function mergeProjectProfileStateValue(
  existing: Record<string, unknown> | undefined,
  incoming: Record<string, unknown>,
): Record<string, unknown> {
  const merged: Record<string, unknown> = {
    ...(existing ?? {}),
    ...incoming,
  };
  const components = {
    ...(objectRecord(existing?.components) ?? {}),
    ...(objectRecord(incoming.components) ?? {}),
  };
  for (const [slot, value] of Object.entries(components)) {
    if (value === null || value === "") {
      delete components[slot];
    }
  }
  if (Object.keys(components).length > 0) {
    merged.components = components;
  } else {
    delete merged.components;
  }
  delete merged.action;
  delete merged.target;
  delete merged.replacement;
  delete merged.update;
  return merged;
}

function buildSourceRef(candidate: ClassifiedCandidate): string {
  if (typeof candidate.metadata?.sourceRef === "string" && candidate.metadata.sourceRef.trim()) {
    return candidate.metadata.sourceRef.trim();
  }
  return `${candidate.source.kind}:${candidate.source.messageId ?? candidate.source.runId ?? candidate.candidateId}`;
}

function stringArray(value: unknown): string[] {
  return Array.isArray(value)
    ? value.filter((entry): entry is string => typeof entry === "string" && entry.trim().length > 0)
    : [];
}

function sourceRefRole(sourceRef: string | undefined): "user" | "assistant" | "tool" | "unknown" {
  const normalized = sourceRef?.trim().toLowerCase() ?? "";
  if (normalized.startsWith("user:")) {
    return "user";
  }
  if (normalized.startsWith("assistant:")) {
    return "assistant";
  }
  if (normalized.startsWith("tool:")) {
    return "tool";
  }
  return "unknown";
}

function factSourceRefs(fact: NormalizedFact): string[] {
  const metadata = objectRecord(fact.objectValueJson);
  const semanticAssertion = objectRecord(metadata?.semanticAssertion);
  return [
    ...new Set(
      [
        fact.sourceRef,
        typeof metadata?.sourceRef === "string" ? metadata.sourceRef : undefined,
        typeof semanticAssertion?.sourceRef === "string" ? semanticAssertion.sourceRef : undefined,
        ...stringArray(metadata?.sourceRefs),
        ...stringArray(metadata?.supportRefs),
        ...stringArray(metadata?.supportContentRefs),
      ].filter((entry): entry is string => Boolean(entry?.trim())),
    ),
  ];
}

function factHasNonAssistantGrounding(fact: NormalizedFact): boolean {
  return factSourceRefs(fact).some((sourceRef) => {
    const role = sourceRefRole(sourceRef);
    return role === "user" || role === "tool";
  });
}

function sourceRefsHaveNonAssistantGrounding(sourceRefs: string[]): boolean {
  return sourceRefs.some((sourceRef) => {
    const role = sourceRefRole(sourceRef);
    return role === "user" || role === "tool";
  });
}

function factIsAssistantOnly(fact: NormalizedFact): boolean {
  const refs = factSourceRefs(fact);
  return refs.some((sourceRef) => sourceRefRole(sourceRef) === "assistant") && !factHasNonAssistantGrounding(fact);
}

function assistantFactConflictsWithGroundedFact(
  store: MemxStoreBundle,
  fact: NormalizedFact,
): boolean {
  if (sourceRefRole(fact.sourceRef) !== "assistant") {
    return false;
  }
  const existing = store.factRepo.findActiveBySemanticKey({
    agentId: fact.agentId,
    scope: fact.scope,
    canonicalSubject: fact.canonicalSubject,
    predicate: fact.predicate,
  });
  return existing.some(
    (prior) =>
      factHasNonAssistantGrounding(prior) &&
      (prior.canonicalObject ?? JSON.stringify(prior.objectValueJson ?? null)) !==
      (fact.canonicalObject ?? JSON.stringify(fact.objectValueJson ?? null)),
  );
}

function shouldSkipFactForAuthority(store: MemxStoreBundle, fact: NormalizedFact): boolean {
  return factIsAssistantOnly(fact) || assistantFactConflictsWithGroundedFact(store, fact);
}

function edgeSourceRefs(edge: NormalizedGraphEdge): string[] {
  return [
    ...new Set(
      [
        edge.evidenceRef,
        ...stringArray(edge.metadataJson?.sourceRefs),
        ...stringArray(edge.metadataJson?.supportRefs),
      ].filter((entry): entry is string => Boolean(entry?.trim())),
    ),
  ];
}

function edgeIsAssistantOnly(edge: NormalizedGraphEdge): boolean {
  const refs = edgeSourceRefs(edge);
  return (
    refs.some((sourceRef) => sourceRefRole(sourceRef) === "assistant") &&
    !sourceRefsHaveNonAssistantGrounding(refs)
  );
}

function semanticCompilerProvenance(candidate: Pick<ClassifiedCandidate, "metadata" | "structuredHints">):
  | Record<string, unknown>
  | undefined {
  const draftProvenance = objectRecord(candidate.structuredHints?.semanticDraft?.compilerProvenance);
  if (draftProvenance) {
    return draftProvenance;
  }
  const metadataProvenance = objectRecord(candidate.metadata?.turnSemanticCompiler);
  if (metadataProvenance) {
    return metadataProvenance;
  }
  const frame = objectRecord(candidate.metadata?.turnSemanticFrame);
  return objectRecord(frame?.compilerProvenance);
}

export function policyAuditSourceMetadata(
  candidate: Pick<ClassifiedCandidate, "metadata" | "structuredHints">,
): Record<string, unknown> {
  const semanticDraft = candidate.structuredHints?.semanticDraft;
  if (!semanticDraft) {
    return {
      decisionSource: "structured_adapter",
      materializationSource: "structured_adapter",
    };
  }
  const compiler = semanticCompilerProvenance(candidate);
  return {
    decisionSource: "llm_semantic_adapter",
    semanticSource: typeof compiler?.source === "string" ? compiler.source : "llm",
    semanticMode: typeof compiler?.mode === "string" ? compiler.mode : "llm",
    materializationSource: "structured_adapter",
  };
}

function candidateTurnIndex(candidate: ClassifiedCandidate): number | undefined {
  const value = candidate.metadata?.turnIndex;
  return typeof value === "number" && Number.isFinite(value) ? value : undefined;
}

function buildProjectProfileFact(params: {
  ctx: MemoryOperationContext;
  candidate: ClassifiedCandidate;
  subject: string;
  predicate: string;
  object?: string;
  objectValueJson?: Record<string, unknown>;
}): NormalizedFact {
  return {
    factId: stableHash([
      params.ctx.agentId,
      params.candidate.scope,
      params.subject,
      params.predicate,
      params.object ?? JSON.stringify(params.objectValueJson ?? {}),
    ]),
    canonicalSubject: normalizeName(params.subject),
    predicate: params.predicate,
    canonicalObject: params.object ? normalizeName(params.object) : undefined,
    objectValueJson: params.objectValueJson,
    scope: params.candidate.scope,
    agentId: params.ctx.agentId,
    confidence: params.candidate.confidence,
    status: "active",
    validFrom: params.candidate.observedAt,
    createdAt: params.candidate.observedAt,
    updatedAt: params.candidate.observedAt,
    sourceRef: buildSourceRef(params.candidate),
    provenanceText: params.candidate.rawText,
  };
}

function buildProjectProfileDocs(params: {
  ctx: MemoryOperationContext;
  fact: NormalizedFact;
}): VectorDocRecord {
  return {
    docId: `fact:${params.fact.factId}`,
    docKind: "fact",
    sourceId: params.fact.factId,
    scope: params.fact.scope,
    agentId: params.fact.agentId,
    text: params.fact.canonicalObject
      ? `${params.fact.canonicalSubject} ${params.fact.predicate} ${params.fact.canonicalObject}`
      : `${params.fact.canonicalSubject} ${params.fact.predicate} ${JSON.stringify(params.fact.objectValueJson ?? {})}`,
    metadataJson: buildVectorDocMetadata({
      docType: "fact",
      confidence: params.fact.confidence,
      observedAt: params.fact.updatedAt,
      lineage: {
        canonicalKind: "fact",
        canonicalId: params.fact.factId,
        sourceKind: "fact",
        sourceId: params.fact.factId,
        sourceRef: params.fact.sourceRef,
        materializedEpoch: params.fact.materializedEpoch,
      },
    }),
    createdAt: params.fact.createdAt,
    updatedAt: params.fact.updatedAt,
  };
}

function normalizedFactStableId(fact: NormalizedFact): string {
  return stableHash([
    fact.agentId,
    fact.scope,
    fact.canonicalSubject,
    fact.predicate,
    fact.canonicalObject ?? JSON.stringify(fact.objectValueJson ?? {}),
  ]);
}

function formatFactVectorTextForWrite(fact: NormalizedFact): string {
  return fact.canonicalObject
    ? `${fact.canonicalSubject} ${fact.predicate} ${fact.canonicalObject}`
    : `${fact.canonicalSubject} ${fact.predicate} ${JSON.stringify(fact.objectValueJson ?? {})}`;
}

function metadataWithoutCanonicalObject(metadata: Record<string, unknown>): Record<string, unknown> {
  const next = { ...metadata };
  delete next.canonicalObject;
  return next;
}

function rewriteFactVectorDoc(
  doc: VectorDocRecord,
  fact: NormalizedFact,
): VectorDocRecord {
  const metadata = objectRecord(doc.metadataJson) ?? {};
  const lineage = objectRecord(metadata.lineage) ?? {};
  const sourceRef =
    fact.sourceRef ??
    (typeof metadata.sourceRef === "string" && metadata.sourceRef.trim()
      ? metadata.sourceRef.trim()
      : undefined);
  const mergedMetadata = {
    ...metadata,
    canonicalSubject: fact.canonicalSubject,
    predicate: fact.predicate,
    ...(fact.canonicalObject ? { canonicalObject: fact.canonicalObject } : {}),
    canonicalKind: "fact",
    canonicalId: fact.factId,
    sourceKind: "fact",
    sourceId: fact.factId,
    ...(sourceRef ? { sourceRef } : {}),
    lineage: {
      ...lineage,
      canonicalKind: "fact",
      canonicalId: fact.factId,
      sourceKind: "fact",
      sourceId: fact.factId,
      ...(sourceRef ? { sourceRef } : {}),
      ...(typeof fact.materializedEpoch === "number"
        ? { materializedEpoch: fact.materializedEpoch }
        : {}),
    },
  };
  const nextMetadata = fact.canonicalObject
    ? mergedMetadata
    : metadataWithoutCanonicalObject(mergedMetadata);
  return {
    ...doc,
    docId: `fact:${fact.factId}`,
    sourceId: fact.factId,
    text: formatFactVectorTextForWrite(fact),
    metadataJson: nextMetadata,
    updatedAt: fact.updatedAt,
  };
}

function mapCanonicalInputName(
  mappings: Map<string, string>,
  input: string | undefined,
  canonical: string,
): void {
  const inputKey = normalizeName(input ?? "");
  const canonicalKey = normalizeName(canonical);
  if (!inputKey || !canonicalKey) {
    return;
  }
  mappings.set(inputKey, canonicalKey);
}

function rewriteFactsForResolvedEntities(
  normalized: { facts: NormalizedFact[]; vectorDocs: VectorDocRecord[] },
  canonicalNameByInputName: Map<string, string>,
): void {
  if (canonicalNameByInputName.size === 0 || normalized.facts.length === 0) {
    return;
  }
  const replacementsByOldFactId = new Map<string, NormalizedFact>();
  normalized.facts = normalized.facts.map((fact) => {
    const canonicalSubject = canonicalNameByInputName.get(fact.canonicalSubject);
    const canonicalObject = fact.canonicalObject
      ? canonicalNameByInputName.get(fact.canonicalObject)
      : undefined;
    if (
      (!canonicalSubject || canonicalSubject === fact.canonicalSubject) &&
      (!canonicalObject || canonicalObject === fact.canonicalObject)
    ) {
      return fact;
    }
    const rewritten: NormalizedFact = {
      ...fact,
      canonicalSubject: canonicalSubject ?? fact.canonicalSubject,
      canonicalObject: canonicalObject ?? fact.canonicalObject,
    };
    rewritten.factId = normalizedFactStableId(rewritten);
    replacementsByOldFactId.set(fact.factId, rewritten);
    return rewritten;
  });
  if (replacementsByOldFactId.size === 0) {
    return;
  }
  normalized.vectorDocs = normalized.vectorDocs.map((doc) => {
    if (doc.docKind !== "fact") {
      return doc;
    }
    const sourceReplacement = replacementsByOldFactId.get(doc.sourceId);
    const docReplacement =
      sourceReplacement ??
      (doc.docId.startsWith("fact:")
        ? replacementsByOldFactId.get(doc.docId.slice("fact:".length))
        : undefined);
    return docReplacement ? rewriteFactVectorDoc(doc, docReplacement) : doc;
  });
}

function shouldResolveFactEntityName(value: string | undefined): value is string {
  const normalized = normalizeName(value ?? "");
  return Boolean(normalized && normalized !== "user" && normalized !== "assistant" && normalized !== "system");
}

function addResolvedFactEntityNameMappings(
  store: MemxStoreBundle,
  ctx: MemoryOperationContext,
  candidate: ClassifiedCandidate,
  facts: NormalizedFact[],
  canonicalNameByInputName: Map<string, string>,
): void {
  const sourceRef = buildSourceRef(candidate);
  const turnIndex = candidateTurnIndex(candidate);
  const seen = new Set<string>();
  for (const fact of facts) {
    for (const [role, rawName] of [
      ["subject", fact.canonicalSubject],
      ["object", fact.canonicalObject],
    ] as const) {
      if (!shouldResolveFactEntityName(rawName)) {
        continue;
      }
      const normalizedName = normalizeName(rawName);
      if (seen.has(normalizedName)) {
        continue;
      }
      seen.add(normalizedName);
      const result = resolveEntityMention(
        store,
        ctx,
        buildEntityMention({
          ctx,
          scope: fact.scope,
          rawText: rawName,
          proposedType: "unknown",
          semanticRole: role,
          sourceRef: fact.sourceRef ?? sourceRef,
          supportText: fact.provenanceText ?? candidate.rawText,
          observedAt: fact.updatedAt,
          sessionKey: candidate.source.sessionKey,
          turnIndex,
          metadataJson: {
            reason: "fact-canonical-name-resolution",
            factId: fact.factId,
            predicate: fact.predicate,
            role,
          },
        }),
        {
          createIfMissing: false,
        },
      );
      if (result.method !== "uncertain" && result.entity.normalizedName) {
        mapCanonicalInputName(canonicalNameByInputName, rawName, result.entity.normalizedName);
      }
    }
  }
}

function buildProjectProfileEdgeDoc(edge: NormalizedGraphEdge): VectorDocRecord {
  return {
    docId: `edge:${edge.edgeId}`,
    docKind: "edge",
    sourceId: edge.edgeId,
    scope: edge.scope,
    agentId: edge.agentId,
    text: `${edge.srcEntityId} ${edge.relType} ${edge.dstEntityId}`,
    metadataJson: buildVectorDocMetadata({
      docType: "edge",
      confidence: edge.confidence,
      observedAt: edge.updatedAt,
      lineage: {
        canonicalKind: "graph_edge",
        canonicalId: edge.edgeId,
        sourceKind: "graph_edge",
        sourceId: edge.edgeId,
        sourceRef: edge.evidenceRef,
        materializedEpoch: edge.materializedEpoch,
      },
      extra: {
        relationType: edge.relType,
        ...(edge.relationSlot ? { relationSlot: edge.relationSlot } : {}),
        sourceKind: edge.sourceKind ?? "extracted",
      },
    }),
    createdAt: edge.createdAt,
    updatedAt: edge.updatedAt,
  };
}

function dedupeById<T extends { [key: string]: unknown }>(
  items: T[],
  getId: (item: T) => string,
): T[] {
  const seen = new Set<string>();
  const deduped: T[] = [];
  for (const item of items) {
    const id = getId(item);
    if (seen.has(id)) {
      continue;
    }
    seen.add(id);
    deduped.push(item);
  }
  return deduped;
}

function resolveKnownProjectNames(
  store: MemxStoreBundle,
  ctx: MemoryOperationContext,
  candidate: ClassifiedCandidate,
  normalized: ReturnType<typeof normalizeCandidate>,
): string[] {
  const currentProject =
    typeof candidate.metadata?.currentProject === "string" &&
    candidate.metadata.currentProject.trim()
      ? candidate.metadata.currentProject.trim()
      : undefined;
  const currentProjectProfile = objectRecord(candidate.metadata?.currentProjectProfile);
  const storedProjectStates = store.stateRepo
    .get({
      agentId: ctx.agentId,
      scopes: ctx.scopes,
      includeExpired: true,
      now: candidate.observedAt,
    })
    .filter((state) => isProjectProfileStateKey(state.key))
    .map((state) => projectCodeFromStateKey(state.key))
    .filter((entry): entry is string => Boolean(entry));
  const incomingProjectStates = normalized.states
    .map((state) => projectCodeFromStateKey(state.key))
    .filter((entry): entry is string => Boolean(entry));
  return [
    currentProject,
    typeof currentProjectProfile?.projectCode === "string"
      ? currentProjectProfile.projectCode.trim()
      : undefined,
    ...storedProjectStates,
    ...incomingProjectStates,
  ].filter((entry): entry is string => Boolean(entry?.trim()));
}

function reconcileProjectReferences(
  store: MemxStoreBundle,
  ctx: MemoryOperationContext,
  candidate: ClassifiedCandidate,
  normalized: ReturnType<typeof normalizeCandidate>,
): void {
  const currentProject =
    typeof candidate.metadata?.currentProject === "string" &&
    candidate.metadata.currentProject.trim()
      ? candidate.metadata.currentProject.trim()
      : undefined;
  const knownProjects = resolveKnownProjectNames(store, ctx, candidate, normalized);
  const projectNameMap = new Map<string, string>();
  const rewriteProjectName = (value: string, allowDescriptorAlias = false): string => {
    const canonical = resolveProjectReference(value, {
      currentProject,
      knownProjects,
      allowDescriptorAlias,
    });
    if (canonical !== value) {
      projectNameMap.set(value, canonical);
    }
    return canonical;
  };

  for (const state of normalized.states) {
    if (isProjectProfileStateKey(state.key)) {
      const projectCode = projectCodeFromStateKey(state.key);
      if (!projectCode) {
        continue;
      }
      const canonical = rewriteProjectName(projectCode, true);
      state.key = projectProfileStateKey(canonical);
      state.valueJson = {
        ...state.valueJson,
        projectCode: canonical,
      };
      continue;
    }
    if (
      state.key === "project.active_project" &&
      typeof state.valueJson.project === "string" &&
      state.valueJson.project.trim()
    ) {
      state.valueJson = {
        ...state.valueJson,
        project: rewriteProjectName(state.valueJson.project, true),
      };
    }
  }

  for (const fact of normalized.facts) {
    const canonical = rewriteProjectName(
      fact.canonicalSubject,
      Boolean(attributeSlotFromPredicate(fact.predicate)),
    );
    fact.canonicalSubject = normalizeName(canonical);
  }

  for (const entity of normalized.entities) {
    const shouldTreatAsProject =
      entity.entityType === "project" ||
      knownProjects.some((project) => normalizeName(project) === entity.normalizedName);
    if (!shouldTreatAsProject) {
      continue;
    }
    const canonical = rewriteProjectName(entity.canonicalName, true);
    if (canonical !== entity.canonicalName) {
      entity.aliases = [
        ...new Set([...entity.aliases, ...projectAliasVariants(entity.canonicalName)]),
      ];
      entity.canonicalName = canonical;
      entity.normalizedName = normalizeName(canonical);
    }
    entity.entityType = "project";
  }
}

function buildProjectProfileArtifacts(params: {
  ctx: MemoryOperationContext;
  candidate: ClassifiedCandidate;
  state: NormalizedState;
}): {
  entities: NormalizedEntity[];
  facts: NormalizedFact[];
  edges: NormalizedGraphEdge[];
  vectorDocs: VectorDocRecord[];
} {
  const projectCode = projectCodeFromStateKey(params.state.key);
  if (!projectCode) {
    return { entities: [], facts: [], edges: [], vectorDocs: [] };
  }
  const entities: NormalizedEntity[] = [
    {
      entityId: stableHash([normalizeName(projectCode)]),
      canonicalName: projectCode,
      entityType: "project",
      normalizedName: normalizeName(projectCode),
      aliases: [
        ...new Set([
          ...projectAliasVariants(projectCode),
          ...((Array.isArray(params.state.valueJson.historicalAliases)
            ? params.state.valueJson.historicalAliases
            : typeof params.state.valueJson.historicalAliases === "string"
              ? [params.state.valueJson.historicalAliases]
              : []) as string[]),
        ]),
      ],
      confidence: Math.max(0.75, params.state.confidence),
    },
  ];
  const facts: NormalizedFact[] = [];
  const edges: NormalizedGraphEdge[] = [];
  const vectorDocs: VectorDocRecord[] = [];
  const version =
    typeof params.state.valueJson.version === "string" ? params.state.valueJson.version.trim() : "";
  const launchDate =
    typeof params.state.valueJson.launchDate === "string"
      ? params.state.valueJson.launchDate.trim()
      : "";
  const internalTrialDate =
    typeof params.state.valueJson.internalTrialDate === "string"
      ? params.state.valueJson.internalTrialDate.trim()
      : "";
  const historicalAliases = Array.isArray(params.state.valueJson.historicalAliases)
    ? params.state.valueJson.historicalAliases.filter(
        (entry): entry is string => typeof entry === "string" && entry.trim().length > 0,
      )
    : typeof params.state.valueJson.historicalAliases === "string" &&
        params.state.valueJson.historicalAliases.trim()
      ? [params.state.valueJson.historicalAliases.trim()]
      : [];
  if (version) {
    const fact = buildProjectProfileFact({
      ctx: params.ctx,
      candidate: params.candidate,
      subject: projectCode,
      predicate: "has_version",
      object: version,
    });
    facts.push(fact);
    vectorDocs.push(buildProjectProfileDocs({ ctx: params.ctx, fact }));
  }
  if (launchDate) {
    const fact = buildProjectProfileFact({
      ctx: params.ctx,
      candidate: params.candidate,
      subject: projectCode,
      predicate: "has_launch_date",
      object: launchDate,
    });
    facts.push(fact);
    vectorDocs.push(buildProjectProfileDocs({ ctx: params.ctx, fact }));
  }
  if (internalTrialDate) {
    const fact = buildProjectProfileFact({
      ctx: params.ctx,
      candidate: params.candidate,
      subject: projectCode,
      predicate: "had_internal_trial_date",
      object: internalTrialDate,
    });
    facts.push(fact);
    vectorDocs.push(buildProjectProfileDocs({ ctx: params.ctx, fact }));
  }
  for (const alias of historicalAliases) {
    const fact = buildProjectProfileFact({
      ctx: params.ctx,
      candidate: params.candidate,
      subject: projectCode,
      predicate: "has_historical_alias",
      object: alias,
    });
    facts.push(fact);
    vectorDocs.push(buildProjectProfileDocs({ ctx: params.ctx, fact }));
  }
  const components = objectRecord(params.state.valueJson.components) ?? {};
  for (const [slot, value] of Object.entries(components)) {
    if (typeof value !== "string" || !value.trim()) {
      continue;
    }
    const object = value.trim();
    const fact = buildProjectProfileFact({
      ctx: params.ctx,
      candidate: params.candidate,
      subject: projectCode,
      predicate: `uses_${slot}`,
      object,
      objectValueJson: {
        componentRole: slot,
        graph: {
          relationType: "uses",
          relationSlot: slot,
          sourceKind: "extracted",
        },
      },
    });
    facts.push(fact);
    vectorDocs.push(buildProjectProfileDocs({ ctx: params.ctx, fact }));
    const componentEntity: NormalizedEntity = {
      entityId: stableHash([normalizeName(object)]),
      canonicalName: object,
      entityType: "unknown",
      normalizedName: normalizeName(object),
      aliases: [],
      confidence: Math.max(0.7, params.state.confidence),
    };
    entities.push(componentEntity);
    const edge: NormalizedGraphEdge = {
      edgeId: stableHash([
        params.ctx.agentId,
        params.candidate.scope,
        entities[0]!.entityId,
        "uses",
        slot,
        componentEntity.entityId,
      ]),
      srcEntityId: entities[0]!.entityId,
      relType: "uses",
      relationSlot: slot,
      dstEntityId: componentEntity.entityId,
      scope: params.candidate.scope,
      agentId: params.ctx.agentId,
      confidence: params.candidate.confidence,
      validFrom: params.candidate.observedAt,
      evidenceRef: buildSourceRef(params.candidate),
      sourceKind: "extracted",
      createdAt: params.candidate.observedAt,
      updatedAt: params.candidate.observedAt,
    };
    edges.push(edge);
    vectorDocs.push(buildProjectProfileEdgeDoc(edge));
  }
  return { entities, facts, edges, vectorDocs };
}

function entitySemanticRole(
  entity: NormalizedEntity,
): "project" | "person" | "resource" | "support" {
  if (entity.entityType === "project") return "project";
  if (entity.entityType === "person") return "person";
  if (entity.entityType === "tool" || entity.entityType === "service") return "resource";
  return "support";
}

function entityTypeFromStructuredHint(type: string | undefined): EntityType {
  switch (type) {
    case "person":
    case "project":
    case "tool":
    case "service":
    case "language":
    case "framework":
    case "concept":
    case "organization":
    case "unknown":
      return type;
    default:
      return "unknown";
  }
}

function semanticRoleFromEntityType(type: EntityType): "project" | "person" | "resource" | "support" {
  if (type === "project") return "project";
  if (type === "person") return "person";
  if (type === "tool" || type === "service") return "resource";
  return "support";
}

function persistStructuredEntityMentions(
  store: MemxStoreBundle,
  ctx: MemoryOperationContext,
  candidate: ClassifiedCandidate,
): number {
  const sourceRef = buildSourceRef(candidate);
  const turnIndex = candidateTurnIndex(candidate);
  const seen = new Set<string>();
  let created = 0;
  for (const hint of candidate.structuredHints?.entities ?? []) {
    const rawText = typeof hint.name === "string" ? hint.name.trim() : "";
    if (!rawText) {
      continue;
    }
    const proposedType = entityTypeFromStructuredHint(hint.type);
    const key = `${proposedType}:${normalizeName(rawText)}`;
    if (seen.has(key)) {
      continue;
    }
    seen.add(key);
    const result = resolveEntityMention(
      store,
      ctx,
      buildEntityMention({
        ctx,
        scope: candidate.scope,
        rawText,
        proposedType,
        semanticRole: semanticRoleFromEntityType(proposedType),
        sourceRef,
        supportText: candidate.rawText,
        observedAt: candidate.observedAt,
        sessionKey: candidate.source.sessionKey,
        turnIndex,
        metadataJson: {
          reason: "structured-entity-mention",
        },
      }),
    );
    if (result.createdEntity) {
      created += 1;
    }
  }
  return created;
}

function resolveGraphEntities(
  store: MemxStoreBundle,
  ctx: MemoryOperationContext,
  candidate: ClassifiedCandidate,
  entities: NormalizedEntity[],
  edges: NormalizedGraphEdge[],
): {
  entities: NormalizedEntity[];
  edges: NormalizedGraphEdge[];
  canonicalNameByInputName: Map<string, string>;
} {
  const canonicalNameByInputName = new Map<string, string>();
  if (entities.length === 0) {
    return { entities, edges, canonicalNameByInputName };
  }
  const sourceRef = buildSourceRef(candidate);
  const turnIndex = candidateTurnIndex(candidate);
  const entityIdMap = new Map<string, string>();
  const canonicalEntities = new Map<string, NormalizedEntity>();
  for (const entity of entities) {
    const mention = buildEntityMention({
      ctx,
      scope: candidate.scope,
      rawText: entity.canonicalName,
      proposedType: entity.entityType,
      semanticRole: entitySemanticRole(entity),
      sourceRef,
      supportText: candidate.rawText,
      observedAt: candidate.observedAt,
      sessionKey: candidate.source.sessionKey,
      turnIndex,
      metadataJson: {
        originalEntityId: entity.entityId,
        aliases: entity.aliases,
      },
    });
    const result = resolveEntityMention(store, ctx, mention);
    const aliases = [...new Set([...result.entity.aliases, ...entity.aliases])];
    const canonical = {
      ...result.entity,
      entityType:
        result.entity.entityType === "unknown" && entity.entityType !== "unknown"
          ? entity.entityType
          : result.entity.entityType,
      aliases,
      confidence: Math.max(result.entity.confidence, entity.confidence, result.confidence),
    };
    store.graphRepo.upsertEntity(canonical);
    for (const alias of entity.aliases) {
      store.graphRepo.upsertEntityAliasSource({
        entityId: canonical.entityId,
        aliasText: alias,
        sourceRef,
        confidence: Math.max(0.7, result.confidence),
        createdAt: candidate.observedAt,
        metadataJson: {
          mentionId: result.mention.mentionId,
          method: result.method,
          originalEntityId: entity.entityId,
        },
      });
    }
    mapCanonicalInputName(canonicalNameByInputName, entity.normalizedName, canonical.normalizedName);
    mapCanonicalInputName(canonicalNameByInputName, entity.canonicalName, canonical.normalizedName);
    mapCanonicalInputName(canonicalNameByInputName, result.mention.rawText, canonical.normalizedName);
    for (const alias of entity.aliases) {
      mapCanonicalInputName(canonicalNameByInputName, alias, canonical.normalizedName);
    }
    entityIdMap.set(entity.entityId, canonical.entityId);
    canonicalEntities.set(canonical.entityId, canonical);
  }
  const rewrittenEdges = edges.map((edge) => ({
    ...edge,
    srcEntityId: entityIdMap.get(edge.srcEntityId) ?? edge.srcEntityId,
    dstEntityId: entityIdMap.get(edge.dstEntityId) ?? edge.dstEntityId,
  }));
  return {
    entities: [...canonicalEntities.values()],
    edges: rewrittenEdges,
    canonicalNameByInputName,
  };
}

function persistIdentityLinkForGraphEdge(
  store: MemxStoreBundle,
  ctx: MemoryOperationContext,
  edge: NormalizedGraphEdge,
): void {
  if (edge.relType !== "supersedes") {
    return;
  }
  const current = store.graphRepo.getEntityById(edge.srcEntityId);
  const historical = store.graphRepo.getEntityById(edge.dstEntityId);
  if (!current || !historical) {
    return;
  }
  store.graphRepo.upsertIdentityLink({
    srcEntityId: current.entityId,
    dstEntityId: historical.entityId,
    linkType: "supersedes",
    confidence: Math.max(0.74, edge.confidence),
    evidenceRef: edge.evidenceRef,
    status: "active",
    at: ctx.now,
    metadataJson: {
      generatedFrom: "structured_graph_relation",
      relationType: edge.relType,
      relationSlot: edge.relationSlot,
      edgeId: edge.edgeId,
    },
  });
  store.graphRepo.upsertEntityAliasSource({
    entityId: current.entityId,
    aliasText: historical.canonicalName,
    sourceRef: edge.evidenceRef,
    confidence: Math.max(0.74, edge.confidence),
    createdAt: edge.updatedAt,
    metadataJson: {
      generatedFrom: "structured_graph_relation",
      relationType: edge.relType,
      edgeId: edge.edgeId,
      historicalEntityId: historical.entityId,
    },
  });
  refreshEntityProfileDocs(store, ctx, [current.entityId, historical.entityId]);
}

export function writeCandidate(
  store: MemxStoreBundle,
  ctx: MemoryOperationContext,
  candidate: ClassifiedCandidate,
): WriteSummary {
  const normalized = normalizeCandidate(candidate, ctx);
  reconcileProjectReferences(store, ctx, candidate, normalized);
  const summary: WriteSummary = {
    states: 0,
    facts: 0,
    events: 0,
    entities: 0,
    edges: 0,
    vectorDocs: 0,
  };
  const insertedEventIds = new Set<string>();
  const projectStateTransitions: Array<{
    state: NormalizedState;
    previous?: Record<string, unknown>;
  }> = [];

  store.client.withTransaction(() => {
    const materializedEpoch = store.client.nextMemoryEpoch(ctx.agentId, candidate.observedAt);
    const touchedEntityIds = new Set<string>();
    const insertedEvents: typeof normalized.events = [];

    for (const state of normalized.states) {
      state.materializedEpoch = materializedEpoch;
      if (isProjectProfileStateKey(state.key)) {
        const existing = store.stateRepo.get({
          agentId: state.agentId,
          scopes: [state.scope],
          key: state.key,
          includeExpired: true,
          now: state.updatedAt,
        })[0];
        if (existing) {
          projectStateTransitions.push({
            state,
            previous: existing.valueJson,
          });
          state.valueJson = mergeProjectProfileStateValue(existing.valueJson, state.valueJson);
          state.confidence = Math.max(existing.confidence, state.confidence);
        } else {
          projectStateTransitions.push({ state });
        }
      }
      if (state.stateKind === "session" && !state.expiresAt) {
        state.expiresAt = store.stateRepo.createExpiry(state.updatedAt, ctx.config.stateTtlHours);
      }
      store.stateRepo.upsert(state);
      summary.states += 1;
    }

    const profileArtifacts = projectStateTransitions.flatMap(({ state }) => {
      if (!shouldDeriveProjectProfileArtifacts(state)) {
        return [];
      }
      const built = buildProjectProfileArtifacts({
        ctx,
        candidate,
        state,
      });
      return [built];
    });
    normalized.entities = dedupeById(
      [...normalized.entities, ...profileArtifacts.flatMap((entry) => entry.entities)],
      (entity) => entity.entityId,
    );
    normalized.facts = dedupeById(
      [...normalized.facts, ...profileArtifacts.flatMap((entry) => entry.facts)],
      (fact) => fact.factId,
    );
    normalized.edges = dedupeById(
      [...normalized.edges, ...profileArtifacts.flatMap((entry) => entry.edges)],
      (edge) => edge.edgeId,
    );
    normalized.vectorDocs = dedupeById(
      [...normalized.vectorDocs, ...profileArtifacts.flatMap((entry) => entry.vectorDocs)],
      (doc) => doc.docId,
    );
    for (const fact of normalized.facts) {
      fact.materializedEpoch = materializedEpoch;
    }
    for (const event of normalized.events) {
      event.materializedEpoch = materializedEpoch;
    }
    for (const edge of normalized.edges) {
      edge.materializedEpoch = materializedEpoch;
    }
    for (const doc of normalized.vectorDocs) {
      doc.materializedEpoch = materializedEpoch;
      doc.metadataJson = {
        ...doc.metadataJson,
        materializedEpoch,
        ...(doc.metadataJson.lineage &&
        typeof doc.metadataJson.lineage === "object" &&
        !Array.isArray(doc.metadataJson.lineage)
          ? {
              lineage: {
                ...(doc.metadataJson.lineage as Record<string, unknown>),
                materializedEpoch,
              },
            }
        : {}),
      };
    }
    summary.entities += persistStructuredEntityMentions(store, ctx, candidate);
    const refreshedGraph = resolveGraphEntities(
      store,
      ctx,
      candidate,
      normalized.entities,
      normalized.edges,
    );
    normalized.entities = refreshedGraph.entities;
    normalized.edges = refreshedGraph.edges;
    addResolvedFactEntityNameMappings(
      store,
      ctx,
      candidate,
      normalized.facts,
      refreshedGraph.canonicalNameByInputName,
    );
    rewriteFactsForResolvedEntities(normalized, refreshedGraph.canonicalNameByInputName);

    for (const transition of projectStateTransitions) {
      const projectCode = projectCodeFromStateKey(transition.state.key);
      if (!projectCode) {
        continue;
      }
      const previousComponents = objectRecord(transition.previous?.components) ?? {};
      const nextComponents = objectRecord(transition.state.valueJson.components) ?? {};
      const projectEntity = resolveEntityMention(
        store,
        ctx,
        buildEntityMention({
          ctx,
          scope: transition.state.scope,
          rawText: projectCode,
          proposedType: "project",
          semanticRole: "project",
          sourceRef: buildSourceRef(candidate),
          supportText: candidate.rawText,
          observedAt: transition.state.updatedAt,
          sessionKey: candidate.source.sessionKey,
          turnIndex: candidateTurnIndex(candidate),
          metadataJson: {
            reason: "project-profile-transition",
          },
        }),
      ).entity;
      for (const [slot, value] of Object.entries(previousComponents)) {
        if (!Object.prototype.hasOwnProperty.call(nextComponents, slot)) {
          store.factRepo.supersedeActiveBySubjectAndPredicate({
            agentId: ctx.agentId,
            scope: transition.state.scope,
            canonicalSubject: normalizeName(projectCode),
            predicate: `uses_${slot}`,
            updatedAt: transition.state.updatedAt,
            sourceRef: buildSourceRef(candidate),
            changeReason: "project-profile-slot-removed",
          });
          if (projectEntity) {
            store.graphRepo.closeActiveSlotEdges({
              agentId: ctx.agentId,
              scope: transition.state.scope,
              srcEntityId: projectEntity.entityId,
              relType: "uses",
              relationSlot: slot,
              validTo: transition.state.updatedAt,
            });
          }
          continue;
        }
        const nextValue = nextComponents[slot];
        if (
          typeof value === "string" &&
          value.trim() &&
          typeof nextValue === "string" &&
          nextValue.trim() &&
          normalizeName(value) !== normalizeName(nextValue)
        ) {
          store.factRepo.supersedeActiveBySubjectAndPredicate({
            agentId: ctx.agentId,
            scope: transition.state.scope,
            canonicalSubject: normalizeName(projectCode),
            predicate: `uses_${slot}`,
            updatedAt: transition.state.updatedAt,
            sourceRef: buildSourceRef(candidate),
            changeReason: "project-profile-slot-replaced",
          });
        }
      }
    }

    const skippedFactIds = new Set<string>();
    const persistedFacts: NormalizedFact[] = [];
    for (const fact of normalized.facts) {
      if (shouldSkipFactForAuthority(store, fact)) {
        skippedFactIds.add(fact.factId);
        continue;
      }
      const attributeSlot = attributeSlotFromPredicate(fact.predicate);
      if (attributeSlot && fact.canonicalObject) {
        for (const siblingPredicate of semanticFactPredicatesForAttributeSlot(attributeSlot)) {
          if (siblingPredicate === fact.predicate) {
            continue;
          }
          store.factRepo.supersedeActiveBySubjectAndPredicate({
            agentId: fact.agentId,
            scope: fact.scope,
            canonicalSubject: fact.canonicalSubject,
            predicate: siblingPredicate,
            updatedAt: fact.updatedAt,
            sourceRef: fact.sourceRef,
            changeReason: "attribute-slot-sibling-replaced",
          });
        }
      }
      const { action } = store.factRepo.upsert(fact, "normalized-update");
      // When a fact supersedes an older version, sync the belief state
      if (action === "versioned") {
        store.beliefRepo.markSupersededByContentRef({
          agentId: ctx.agentId,
          contentRef: fact.factId,
          updatedAt: fact.updatedAt,
        });
      }
      persistedFacts.push(fact);
      summary.facts += 1;
    }

    for (const entity of normalized.entities) {
      store.graphRepo.upsertEntity(entity);
      touchedEntityIds.add(entity.entityId);
      summary.entities += 1;
    }

    const skippedEdgeIds = new Set<string>();
    const persistedEdges: NormalizedGraphEdge[] = [];
    for (const edge of normalized.edges) {
      if (edgeIsAssistantOnly(edge)) {
        skippedEdgeIds.add(edge.edgeId);
        continue;
      }
      store.graphRepo.upsertEdge(edge);
      persistIdentityLinkForGraphEdge(store, ctx, edge);
      touchedEntityIds.add(edge.srcEntityId);
      touchedEntityIds.add(edge.dstEntityId);
      persistedEdges.push(edge);
      summary.edges += 1;
    }

    for (const event of normalized.events) {
      const duplicate = store.eventRepo.findNearDuplicate({
        agentId: event.agentId,
        scope: event.scope,
        normalizedText: event.normalizedText,
        observedAfter: dedupWindowStart(event.observedAt, ctx.config.episodicDedupWindowDays),
      });
      if (duplicate) {
        continue;
      }
      store.eventRepo.append(event);
      insertedEventIds.add(event.eventId);
      insertedEvents.push(event);
      summary.events += 1;
    }

    const filteredDocs = normalized.vectorDocs.filter((doc) => {
      if (
        doc.docKind === "fact" &&
        (skippedFactIds.has(doc.sourceId) || skippedFactIds.has(doc.docId.replace(/^fact:/u, "")))
      ) {
        return false;
      }
      if (doc.docKind !== "event") {
        if (
          doc.docKind === "edge" &&
          (skippedEdgeIds.has(doc.sourceId) || skippedEdgeIds.has(doc.docId.replace(/^edge:/u, "")))
        ) {
          return false;
        }
        return true;
      }
      return [...insertedEventIds].some((eventId) => `event:${eventId}` === doc.docId);
    });
    refreshEntityProfileDocs(store, ctx, [...touchedEntityIds]);
    store.retrievalBackend.upsertDocs(filteredDocs);
    summary.vectorDocs += filteredDocs.length;
    emitWriteMaterializationSignals(store, ctx, {
      states: normalized.states,
      facts: persistedFacts,
      events: insertedEvents,
      graphEdges: persistedEdges,
      materializedEpoch,
    });
  });

  if (ctx.config.advanced.enableTelemetryAudit) {
    store.auditRepo.recordPolicyDecision({
      agentId: ctx.agentId,
      sessionKey: ctx.sessionKey,
      sourceRef: `${candidate.source.kind}:${candidate.candidateId}`,
      candidateText: candidate.rawText,
      decision: candidate.policy,
      createdAt: candidate.observedAt,
      metadataJson: {
        ...policyAuditSourceMetadata(candidate),
        materializedBy: "normalize/write",
        materializationOutcome: summary,
        classification: candidate.classification,
        structuredHints: candidate.structuredHints ?? {},
        turnSemanticFrame:
          candidate.metadata && typeof candidate.metadata.turnSemanticFrame === "object"
            ? candidate.metadata.turnSemanticFrame
            : undefined,
        semanticDraftConsumed: candidate.structuredHints?.semanticDraft
          ? {
              sourceRef: candidate.structuredHints.semanticDraft.sourceRef,
              families: [
                ...new Set(
                  candidate.structuredHints.semanticDraft.assertionDrafts.map(
                    (entry) => entry.familyHint,
                  ),
                ),
              ],
              timeframes: [
                ...new Set(
                  candidate.structuredHints.semanticDraft.assertionDrafts.map(
                    (entry) => entry.timeframeHint,
                  ),
                ),
              ],
              corrections: candidate.structuredHints.semanticDraft.correctionDrafts.map(
                (entry) => ({
                  timeframe: entry.correction.timeframe,
                  targetKind: entry.correction.targetKind,
                  predicate: entry.correction.predicate,
                }),
              ),
            }
          : undefined,
        materializationHint: candidate.structuredHints?.materializationHint,
        llmBudget: snapshotMemoryLlmBudgetAudit(ctx.llmBudgetAudit),
        turnSemanticCompile:
          candidate.metadata && typeof candidate.metadata.turnSemanticCompiler === "object"
            ? candidate.metadata.turnSemanticCompiler
            : undefined,
      },
    });
  }

  return summary;
}
