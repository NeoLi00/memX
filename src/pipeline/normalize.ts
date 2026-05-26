import { redactSensitiveText } from "../security/pii.js";
import {
  clamp01,
  isValidEntityName,
  normalizeName,
  normalizeText,
  normalizedTerms,
  objectRecord,
  randomId,
  stableHash,
  truncateText,
} from "../support.js";
import type {
  ClassifiedCandidate,
  MemoryCandidateCorrectionHint,
  MemoryCandidateMaterializationHint,
  MemoryAdviceSignalHint,
  MemoryCandidateSemanticDraft,
  EntityType,
  GraphRelationType,
  MemoryCandidateStructuredHints,
  MemoryOperationContext,
  MemoryCandidateDecisionHint,
  MemoryGuidanceFacet,
  MemoryGuidanceType,
  MemoryCandidatePreferenceHint,
  MemoryCandidateRelationHint,
  MemoryResourceAssertionHint,
  MemoryCandidateWorkflowHint,
  MemoryWorkflowStateKind,
  NormalizedEntity,
  NormalizedEvent,
  NormalizedFact,
  NormalizedGraphEdge,
  NormalizedState,
  TurnSemanticAssertionFamilyHint,
  TurnSemanticAssertionDraft,
  VectorDocRecord,
} from "../types.js";
import {
  shouldDeriveRelationFact,
  shouldMaterializePreferenceFact,
  sanitizeWorkflowHint,
} from "./authority.js";
import {
  canonicalAttributeSlot,
  normalizeSemanticFactPredicate,
} from "./attributeSlots.js";
import { sourceRefsFromMaintenanceMetadata, uniqueMaintenanceRefs } from "./maintenanceContract.js";
import { describeStateValue } from "./memoryObjectsHelpers.js";
import {
  isProjectProfileStateKey,
  projectAliasVariants,
  projectCodeFromStateKey,
  projectNamesMatch,
  resolveProjectReference,
} from "./projectIdentity.js";
import { canonicalStateKey } from "./semantic/heuristics.js";
import { canonicalizePreferenceHint } from "./semantics.js";
import { stateCurrentnessVectorMetadata } from "./stateLifecycle.js";
import { buildVectorDocMetadata } from "./vectorDocMetadata.js";

type NormalizedOutputs = {
  states: NormalizedState[];
  facts: NormalizedFact[];
  events: NormalizedEvent[];
  entities: NormalizedEntity[];
  edges: NormalizedGraphEdge[];
  vectorDocs: VectorDocRecord[];
};

function subjectUser(): string {
  return "user";
}

function buildEntity(name: string, type: EntityType = "unknown"): NormalizedEntity | null {
  if (!isValidEntityName(name)) {
    return null;
  }
  const normalizedName = normalizeName(name);
  return {
    // Hash by normalizedName only — same name with different types should
    // share one entityId so the DB upsert merges them instead of creating duplicates.
    entityId: stableHash([normalizedName]),
    canonicalName: name.trim(),
    entityType: type,
    normalizedName,
    aliases: [],
    confidence: 0.75,
  };
}

function projectProfileStateKey(projectCode: string): string {
  return `project.${projectCode.trim()}`;
}

function mergeAliases(existing: string[], aliases: string[]): string[] {
  return [...new Set([...existing, ...aliases].filter(Boolean))];
}

function sanitizeProjectComponentMap(
  components: Record<string, unknown> | undefined,
): Record<string, string | null> {
  const sanitized: Record<string, string | null> = {};
  for (const [slot, rawValue] of Object.entries(components ?? {})) {
    const normalizedSlot = normalizeText(slot)
      .replace(/[^\p{L}\p{N}]+/gu, "_")
      .replace(/^_+|_+$/g, "");
    if (!normalizedSlot) {
      continue;
    }
    if (typeof rawValue === "string" && rawValue.trim()) {
      sanitized[normalizedSlot] = rawValue.trim();
      continue;
    }
    if (rawValue === null) {
      sanitized[normalizedSlot] = null;
    }
  }
  return sanitized;
}

function chineseMonthToNumber(token: string): number | null {
  const normalized = token.trim();
  if (/^\d{1,2}$/.test(normalized)) {
    const month = Number(normalized);
    return month >= 1 && month <= 12 ? month : null;
  }
  const table: Record<string, number> = {
    一: 1,
    二: 2,
    三: 3,
    四: 4,
    五: 5,
    六: 6,
    七: 7,
    八: 8,
    九: 9,
    十: 10,
    十一: 11,
    十二: 12,
  };
  return table[normalized] ?? null;
}

function extractRelativeMonthRefs(text: string, observedAt: string): string[] {
  const observed = new Date(observedAt);
  const baseYear = Number.isFinite(observed.getUTCFullYear())
    ? observed.getUTCFullYear()
    : new Date().getUTCFullYear();
  const matches = [
    ...text.matchAll(
      /(前年|去年|今年|明年|this year|last year|next year)\s*([一二三四五六七八九十]{1,3}|\d{1,2})月/giu,
    ),
  ];
  const resolved: string[] = [];
  for (const match of matches) {
    const marker = (match[1] ?? "").toLowerCase();
    const month = chineseMonthToNumber(match[2] ?? "");
    if (!month) {
      continue;
    }
    let yearOffset = 0;
    if (marker === "前年") yearOffset = -2;
    else if (marker === "去年" || marker === "last year") yearOffset = -1;
    else if (marker === "明年" || marker === "next year") yearOffset = 1;
    const year = baseYear + yearOffset;
    resolved.push(`${year}-${String(month).padStart(2, "0")}`);
  }
  return resolved;
}

function extractHistoricalAliases(text: string): string[] {
  const aliases = [
    ...text.matchAll(/[`"'“”‘’]([A-Za-z][A-Za-z0-9_.-]{1,40})[`"'“”‘’]/g),
    ...text.matchAll(
      /(?:代号|别名|旧称|旧名|内部名|alias|codename)\s*(?:是|为|叫)?\s*([A-Za-z][A-Za-z0-9_.-]{1,40})/giu,
    ),
  ]
    .map((match) => match[1]?.trim() ?? "")
    .filter(Boolean);
  return [...new Set(aliases)];
}

function canonicalProjectProfileFieldKey(rawKey: string): string {
  const compact = normalizeText(rawKey)
    .replace(/[^\p{L}\p{N}]+/gu, "_")
    .replace(/^_+|_+$/g, "");
  switch (compact) {
    case "publicbetadate":
    case "public_beta_date":
    case "launchdate":
    case "launch_date":
    case "publiclaunchdate":
    case "public_launch_date":
      return "launchDate";
    case "internaltestdate":
    case "internal_test_date":
    case "internaltrialdate":
    case "internal_trial_date":
    case "internalbetadate":
    case "internal_beta_date":
      return "internalTrialDate";
    case "historicalcodename":
    case "historical_codename":
    case "historicalalias":
    case "historical_alias":
    case "codename":
    case "alias":
      return "historicalAliases";
    default:
      return compact;
  }
}

function mergeStringArray(existing: unknown, incoming: string[]): string[] | undefined {
  const base = Array.isArray(existing)
    ? existing.filter(
        (entry): entry is string => typeof entry === "string" && entry.trim().length > 0,
      )
    : [];
  const merged = [...new Set([...base, ...incoming.map((entry) => entry.trim()).filter(Boolean)])];
  return merged.length > 0 ? merged : undefined;
}

function inferProjectComponentUpdate(
  update: Record<string, unknown>,
  base: Record<string, unknown> | undefined,
): Record<string, string | null> {
  const normalized = sanitizeProjectComponentMap(objectRecord(update.components));
  const target =
    typeof update.target === "string" && update.target.trim() ? update.target.trim() : undefined;
  const replacement =
    typeof update.replacement === "string" && update.replacement.trim()
      ? update.replacement.trim()
      : undefined;
  const slot =
    typeof update.componentRole === "string" && update.componentRole.trim()
      ? update.componentRole.trim()
      : typeof update.slot === "string" && update.slot.trim()
        ? update.slot.trim()
        : undefined;
  const baseComponents = objectRecord(base?.components) ?? {};
  const resolvedSlot =
    slot ??
    Object.entries(baseComponents).find(
      ([, value]) =>
        typeof value === "string" && target && normalizeName(value) === normalizeName(target),
    )?.[0];
  if (!resolvedSlot) {
    return normalized;
  }
  normalized[resolvedSlot] = replacement ?? null;
  return normalized;
}

function sanitizeProjectProfileValue(
  value: Record<string, unknown>,
  base: Record<string, unknown> | undefined,
): Record<string, unknown> {
  const sanitized: Record<string, unknown> = {};
  const projectCode =
    typeof value.projectCode === "string" && value.projectCode.trim()
      ? value.projectCode.trim()
      : typeof base?.projectCode === "string" && base.projectCode.trim()
        ? base.projectCode.trim()
        : undefined;
  const version =
    typeof value.version === "string" && value.version.trim()
      ? value.version.trim()
      : typeof base?.version === "string" && base.version.trim()
        ? base.version.trim()
        : undefined;
  const launchDate =
    typeof value.launchDate === "string" && value.launchDate.trim()
      ? value.launchDate.trim()
      : typeof base?.launchDate === "string" && base.launchDate.trim()
        ? base.launchDate.trim()
        : undefined;
  const componentUpdates = inferProjectComponentUpdate(value, base);
  if (projectCode) {
    sanitized.projectCode = projectCode;
  }
  if (version) {
    sanitized.version = version;
  }
  if (launchDate) {
    sanitized.launchDate = launchDate;
  }
  if (Object.keys(componentUpdates).length > 0) {
    sanitized.components = componentUpdates;
  }
  for (const [rawKey, rawValue] of Object.entries(value)) {
    if (
      rawKey === "projectCode" ||
      rawKey === "version" ||
      rawKey === "launchDate" ||
      rawKey === "components" ||
      rawKey === "action" ||
      rawKey === "target" ||
      rawKey === "replacement" ||
      rawKey === "componentRole" ||
      rawKey === "slot"
    ) {
      continue;
    }
    const key = canonicalProjectProfileFieldKey(rawKey);
    if (!key) {
      continue;
    }
    if (key === "historicalAliases") {
      const aliases = Array.isArray(rawValue)
        ? rawValue.filter(
            (entry): entry is string => typeof entry === "string" && entry.trim().length > 0,
          )
        : typeof rawValue === "string" && rawValue.trim()
          ? [rawValue.trim()]
          : [];
      const mergedAliases = mergeStringArray(base?.historicalAliases, aliases);
      if (mergedAliases) {
        sanitized.historicalAliases = mergedAliases;
      }
      continue;
    }
    if (typeof rawValue === "string" && rawValue.trim()) {
      sanitized[key] = rawValue.trim();
      continue;
    }
    if (typeof rawValue === "number" || typeof rawValue === "boolean") {
      sanitized[key] = rawValue;
      continue;
    }
    if (rawValue === null) {
      sanitized[key] = null;
    }
  }
  return sanitized;
}

function buildCorrectionProjectProfileUpdate(params: {
  candidate: ClassifiedCandidate;
  correction: MemoryCandidateCorrectionHint;
  projectCode: string;
  currentValue: Record<string, unknown> | undefined;
}): Record<string, unknown> | null {
  const update: Record<string, unknown> = { projectCode: params.projectCode };
  const nextValue = params.correction.nextValue?.trim();
  switch (params.correction.predicate) {
    case "uses_async_framework":
      if (nextValue) {
        update.components = { async_framework: nextValue };
      }
      break;
    case "uses_primary_language":
      if (nextValue) {
        update.components = { primary_language: nextValue };
      }
      break;
    case "uses_cache":
      if (nextValue) {
        update.components = { cache: nextValue };
      }
      break;
    case "has_launch_date": {
      const [launchDate, internalTrialDate] = extractRelativeMonthRefs(
        params.candidate.rawText,
        params.candidate.observedAt,
      );
      if (launchDate) {
        update.launchDate = launchDate;
      }
      if (internalTrialDate) {
        update.internalTrialDate = internalTrialDate;
      }
      if (!launchDate && nextValue) {
        update.launchDate = nextValue;
      }
      break;
    }
    case "has_historical_alias": {
      const aliases = extractHistoricalAliases(params.candidate.rawText);
      const mergedAliases = mergeStringArray(params.currentValue?.historicalAliases, aliases);
      if (mergedAliases) {
        update.historicalAliases = mergedAliases;
      }
      break;
    }
    case "has_product_name":
      if (nextValue) {
        update.projectCode = nextValue;
      }
      break;
    default:
      return null;
  }
  const sanitized = sanitizeProjectProfileValue(update, params.currentValue);
  return Object.keys(sanitized).length > 0 ? sanitized : null;
}

function mergeProjectProfileValue(
  base: Record<string, unknown> | undefined,
  update: Record<string, unknown>,
): Record<string, unknown> {
  const incoming = sanitizeProjectProfileValue(update, base);
  const merged: Record<string, unknown> = {
    ...(base ?? {}),
    ...incoming,
  };
  const mergedComponents = {
    ...(objectRecord(base?.components) ?? {}),
    ...(objectRecord(incoming.components) ?? {}),
  };
  for (const [slot, value] of Object.entries(mergedComponents)) {
    if (value === null || value === "") {
      delete mergedComponents[slot];
    }
  }
  if (Object.keys(mergedComponents).length > 0) {
    merged.components = mergedComponents;
  } else {
    delete merged.components;
  }
  return merged;
}

function projectProfileFieldPaths(value: Record<string, unknown>, prefix = ""): string[] {
  const paths: string[] = [];
  for (const [key, rawValue] of Object.entries(value)) {
    const path = prefix ? `${prefix}.${key}` : key;
    if (rawValue && typeof rawValue === "object" && !Array.isArray(rawValue)) {
      paths.push(...projectProfileFieldPaths(objectRecord(rawValue) ?? {}, path));
      continue;
    }
    paths.push(path);
  }
  return paths;
}

function readProjectProfilePath(value: Record<string, unknown> | undefined, path: string): unknown {
  const segments = path.split(".");
  let cursor: unknown = value;
  for (const segment of segments) {
    if (!cursor || typeof cursor !== "object" || Array.isArray(cursor)) {
      return undefined;
    }
    cursor = objectRecord(cursor)?.[segment];
  }
  return cursor;
}

function writeProjectProfilePath(
  target: Record<string, unknown>,
  path: string,
  value: unknown,
): void {
  const segments = path.split(".");
  let cursor: Record<string, unknown> = target;
  for (const [index, segment] of segments.entries()) {
    if (index === segments.length - 1) {
      cursor[segment] = value;
      return;
    }
    const next = objectRecord(cursor[segment]) ?? {};
    cursor[segment] = next;
    cursor = next;
  }
}

function preserveProjectProfilePaths(params: {
  merged: Record<string, unknown>;
  authoritative: Record<string, unknown> | undefined;
  protectedPaths: Iterable<string>;
}): Record<string, unknown> {
  const result = { ...params.merged };
  for (const path of params.protectedPaths) {
    const authoritativeValue = readProjectProfilePath(params.authoritative, path);
    if (authoritativeValue === undefined) {
      continue;
    }
    writeProjectProfilePath(result, path, authoritativeValue);
  }
  return result;
}

function deriveProjectProfileNote(params: {
  candidateText: string;
  projectCode: string | undefined;
  currentValue: Record<string, unknown>;
}): string | undefined {
  if (!params.projectCode) {
    return undefined;
  }
  if (typeof params.currentValue.note === "string" && params.currentValue.note.trim()) {
    return undefined;
  }
  const detailKeys = Object.keys(params.currentValue).filter(
    (key) => !["projectCode", "status"].includes(key),
  );
  if (detailKeys.length > 0) {
    return undefined;
  }
  const trimmed = params.candidateText.trim();
  if (!trimmed) {
    return undefined;
  }
  const normalizedProject = normalizeName(params.projectCode);
  const normalizedText = normalizeName(trimmed);
  if (normalizedText === normalizedProject) {
    return undefined;
  }
  return truncateText(trimmed, 220);
}

function relationFactPredicate(relation: MemoryCandidateRelationHint): string {
  if (relation.predicate === "uses" && relation.relationSlot) {
    return `uses_${relation.relationSlot}`;
  }
  if (relation.predicate === "related_to" && relation.rawPredicate?.trim()) {
    return normalizeText(relation.rawPredicate).replace(/[^\p{L}\p{N}]+/gu, "_");
  }
  return relation.predicate;
}

function formatStateVectorText(state: NormalizedState): string {
  return `${state.key}: ${describeStateValue(state.key, state.valueJson)}`;
}

function formatFactObject(predicate: string, object?: string): string {
  if (!object) {
    return "";
  }
  if (predicate === "prefers_output_order") {
    return object.replace(/\b(zh|en) first (zh|en) second\b/i, "$1 first, $2 second");
  }
  return object;
}

function formatFactVectorText(fact: NormalizedFact): string {
  if (fact.predicate === "has_resource" && fact.objectValueJson) {
    return resourceVectorTextFromRecord(fact.objectValueJson, {
      subject: fact.canonicalSubject,
      resource: fact.canonicalObject,
    });
  }
  return fact.canonicalObject
    ? `${fact.canonicalSubject} ${fact.predicate} ${formatFactObject(fact.predicate, fact.canonicalObject)}`
    : `${fact.canonicalSubject} ${fact.predicate} ${JSON.stringify(fact.objectValueJson ?? {})}`;
}

function factVectorSupportMetadata(fact: NormalizedFact): Record<string, unknown> {
  const sourceRefs = uniqueMaintenanceRefs([
    ...sourceRefsFromMaintenanceMetadata(fact.objectValueJson),
    fact.sourceRef,
  ]);
  const semanticAssertion = objectRecord(fact.objectValueJson?.semanticAssertion);
  const supportText =
    typeof semanticAssertion?.supportText === "string" && semanticAssertion.supportText.trim()
      ? semanticAssertion.supportText.trim()
      : typeof fact.objectValueJson?.supportText === "string" && fact.objectValueJson.supportText.trim()
        ? fact.objectValueJson.supportText.trim()
        : fact.provenanceText?.trim();
  return {
    ...(sourceRefs[0] ? { sourceRef: sourceRefs[0] } : {}),
    ...(sourceRefs.length > 0
      ? {
          sourceRefs,
          supportRefs: sourceRefs,
          supportContentRefs: sourceRefs,
        }
      : {}),
    ...(supportText ? { supportText } : {}),
  };
}

function buildFact(params: {
  ctx: MemoryOperationContext;
  candidate: ClassifiedCandidate;
  subject: string;
  predicate: string;
  object?: string;
  objectValueJson?: Record<string, unknown>;
  confidence?: number;
  sourceRef?: string;
  provenanceText?: string;
}): NormalizedFact {
  const sourceRef = params.sourceRef ?? sourceRefForCandidate(params.candidate);
  const provenanceText = params.provenanceText ?? params.candidate.rawText;
  const supportRefs = uniqueMaintenanceRefs([
    ...sourceRefsFromMaintenanceMetadata(params.objectValueJson),
    sourceRef,
  ]);
  const objectValueJson =
    params.objectValueJson || supportRefs.length > 0 || provenanceText.trim()
      ? {
          ...(params.objectValueJson ?? {}),
          ...(supportRefs[0] ? { sourceRef: supportRefs[0] } : {}),
          ...(supportRefs.length > 0
            ? {
                sourceRefs: supportRefs,
                supportRefs,
                supportContentRefs: supportRefs,
              }
            : {}),
          ...(provenanceText.trim() &&
          typeof params.objectValueJson?.supportText !== "string" &&
          !objectRecord(params.objectValueJson?.semanticAssertion)?.supportText
            ? { supportText: truncateText(provenanceText.trim(), 520) }
            : {}),
        }
      : undefined;
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
    objectValueJson,
    scope: params.candidate.scope,
    agentId: params.ctx.agentId,
    confidence: params.confidence ?? params.candidate.confidence,
    status: "active",
    validFrom: params.candidate.observedAt,
    createdAt: params.candidate.observedAt,
    updatedAt: params.candidate.observedAt,
    sourceRef,
    provenanceText,
  };
}

function sourceRefForCandidate(candidate: ClassifiedCandidate): string {
  return typeof candidate.metadata?.sourceRef === "string" && candidate.metadata.sourceRef.trim()
    ? candidate.metadata.sourceRef.trim()
    : `${candidate.source.kind}:${candidate.source.messageId ?? candidate.source.runId ?? candidate.candidateId}`;
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

function compactResourceKey(value: string): string {
  return normalizeText(value)
    .replace(/[^\p{L}\p{N}]+/gu, "_")
    .replace(/^_+|_+$/g, "")
    .slice(0, 96);
}

function resourceStateKind(assertion: MemoryResourceAssertionHint): MemoryWorkflowStateKind {
  return assertion.ownershipStatus === "considering" ? "session" : "durable";
}

function resourceEntityType(assertion: MemoryResourceAssertionHint): EntityType {
  void assertion;
  return "concept";
}

function resourceMetadata(assertion: MemoryResourceAssertionHint): Record<string, unknown> {
  return {
    resource: assertion.resource,
    owner: assertion.owner,
    ownershipStatus: assertion.ownershipStatus,
    semanticStatus: assertion.semanticStatus,
    sourceRef: assertion.sourceRef,
    supportText: assertion.supportText,
    ...(assertion.resourceType ? { resourceType: assertion.resourceType } : {}),
    ...(assertion.domains && assertion.domains.length > 0 ? { domains: assertion.domains } : {}),
    ...(assertion.affordances && assertion.affordances.length > 0
      ? { affordances: assertion.affordances }
      : {}),
  };
}

function stringArrayFromRecord(value: unknown): string[] {
  return Array.isArray(value)
    ? value.filter((entry): entry is string => typeof entry === "string" && entry.trim().length > 0)
    : [];
}

function resourceVectorTextFromRecord(
  record: Record<string, unknown>,
  fallback: { subject?: string; resource?: string } = {},
): string {
  const owner =
    typeof record.owner === "string" && record.owner.trim() ? record.owner : fallback.subject;
  const resource =
    typeof record.resource === "string" && record.resource.trim()
      ? record.resource
      : fallback.resource;
  const ownership =
    typeof record.ownershipStatus === "string" && record.ownershipStatus.trim()
      ? record.ownershipStatus
      : undefined;
  const resourceType =
    typeof record.resourceType === "string" && record.resourceType.trim()
      ? record.resourceType.trim()
      : undefined;
  const domains = stringArrayFromRecord(record.domains).slice(0, 6).join(", ");
  const affordances = stringArrayFromRecord(record.affordances).slice(0, 6).join(", ");
  const supportText =
    typeof record.supportText === "string" && record.supportText.trim()
      ? record.supportText.trim()
      : undefined;
  return truncateText(
    [
      `${owner ?? "user"} has_resource ${resource ?? "resource"}${ownership ? ` (${ownership})` : ""}`,
      resourceType ? `resource type: ${resourceType}` : undefined,
      domains ? `domains: ${domains}` : undefined,
      affordances ? `affordances: ${affordances}` : undefined,
      supportText ? `evidence: ${supportText}` : undefined,
    ]
      .filter(Boolean)
      .join(" | "),
    520,
  );
}

function resourceVectorMetadata(
  record: Record<string, unknown> | undefined,
): Record<string, unknown> {
  if (!record || record.signalKind !== "resourceAssertion") {
    return {};
  }
  const keys = [
    "resource",
    "owner",
    "ownershipStatus",
    "semanticStatus",
    "sourceRef",
    "supportText",
    "resourceType",
    "domains",
    "affordances",
    "supportRefs",
    "signalKind",
  ];
  return Object.fromEntries(
    keys.map((key) => [key, record[key]] as const).filter(([, value]) => value !== undefined),
  );
}

function resourceText(assertion: MemoryResourceAssertionHint): string {
  const status =
    assertion.ownershipStatus === "recently_acquired"
      ? "recently acquired"
      : assertion.ownershipStatus;
  return truncateText(
    [
      `${assertion.owner} has_resource ${assertion.resource} (${status})`,
      assertion.resourceType ? `resource type: ${assertion.resourceType}` : undefined,
      assertion.domains && assertion.domains.length > 0
        ? `domains: ${assertion.domains.slice(0, 5).join(", ")}`
        : undefined,
      assertion.affordances && assertion.affordances.length > 0
        ? `affordances: ${assertion.affordances.slice(0, 5).join(", ")}`
        : undefined,
      `evidence: ${assertion.supportText}`,
    ]
      .filter(Boolean)
      .join(" | "),
    520,
  );
}

function buildResourceFact(
  ctx: MemoryOperationContext,
  candidate: ClassifiedCandidate,
  assertion: MemoryResourceAssertionHint,
): NormalizedFact {
  return buildFact({
    ctx,
    candidate,
    subject: assertion.owner || subjectUser(),
    predicate: "has_resource",
    object: assertion.resource,
    objectValueJson: {
      ...resourceMetadata(assertion),
      currentnessHint: assertion.ownershipStatus === "considering" ? "unknown" : "current",
      supportRefs: [assertion.sourceRef],
      signalKind: "resourceAssertion",
    },
    confidence: assertion.confidence,
    sourceRef: assertion.sourceRef,
    provenanceText: assertion.supportText,
  });
}

function buildResourceState(
  ctx: MemoryOperationContext,
  candidate: ClassifiedCandidate,
  assertion: MemoryResourceAssertionHint,
): NormalizedState | null {
  const resourceKey = compactResourceKey(assertion.resource);
  if (!resourceKey) {
    return null;
  }
  return {
    key: canonicalStateKey(`${assertion.owner || subjectUser()}.resource.${resourceKey}`),
    valueJson: {
      ...resourceMetadata(assertion),
      signalKind: "resourceAssertion",
    },
    scope: candidate.scope,
    agentId: ctx.agentId,
    stateKind: resourceStateKind(assertion),
    confidence: assertion.confidence,
    sourceRef: assertion.sourceRef,
    updatedAt: candidate.observedAt,
  };
}

function buildResourceEvent(
  ctx: MemoryOperationContext,
  candidate: ClassifiedCandidate,
  assertion: MemoryResourceAssertionHint,
): NormalizedEvent {
  const text = resourceText(assertion);
  return {
    eventId: stableHash([
      ctx.agentId,
      candidate.scope,
      "resource_observation",
      assertion.sourceRef,
      assertion.owner,
      assertion.resource,
      assertion.ownershipStatus,
    ]),
    agentId: ctx.agentId,
    scope: candidate.scope,
    eventType: "resource_observation",
    text,
    normalizedText: normalizeText(text),
    observedAt: candidate.observedAt,
    sourceKind: candidate.source.kind,
    sourceRef: assertion.sourceRef,
    sessionKey: candidate.source.sessionKey,
    confidence: assertion.confidence,
    metadataJson: {
      ...resourceMetadata(assertion),
      memxStructuredSummary: text,
      memxRetrievalDetailExcerpt: assertion.supportText,
      signalKind: "resourceAssertion",
    },
  };
}

function adviceSignalText(signal: MemoryAdviceSignalHint): string {
  return truncateText(
    [
      signal.problemContext ? `problem: ${signal.problemContext}` : undefined,
      signal.userResources && signal.userResources.length > 0
        ? `resources: ${signal.userResources.join(", ")}`
        : undefined,
      signal.assistantRecommendation
        ? `recommendation: ${signal.assistantRecommendation}`
        : undefined,
      signal.domains && signal.domains.length > 0
        ? `domains: ${signal.domains.join(", ")}`
        : undefined,
      signal.supportText ? `evidence: ${signal.supportText}` : undefined,
    ]
      .filter(Boolean)
      .join(" | "),
    420,
  );
}

function buildAdviceSignalFact(
  ctx: MemoryOperationContext,
  candidate: ClassifiedCandidate,
  signal: MemoryAdviceSignalHint,
): NormalizedFact | null {
  const text = adviceSignalText(signal);
  if (!text) {
    return null;
  }
  return buildFact({
    ctx,
    candidate,
    subject: subjectUser(),
    predicate: "has_advice_signal",
    object: text,
    objectValueJson: {
      problemContext: signal.problemContext,
      userResources: signal.userResources,
      assistantRecommendation: signal.assistantRecommendation,
      domains: signal.domains,
      sourceRefs: signal.sourceRefs,
      supportText: signal.supportText,
      semanticStatus: signal.semanticStatus,
      signalKind: "adviceSignal",
      currentnessHint: "historical",
      supportRefs: signal.sourceRefs,
    },
    confidence: signal.confidence,
    sourceRef: signal.sourceRefs[0] ?? sourceRefForCandidate(candidate),
    provenanceText: signal.supportText ?? text,
  });
}

function preferenceGuidanceType(predicate: string): MemoryGuidanceType {
  switch (predicate) {
    case "prefers_language":
      return "language";
    case "prefers_response_style":
    case "prefers_style":
    case "prefers_code_style":
      return "style";
    case "prefers_output_charset":
    case "prefers_charset":
      return "charset";
    case "prefers_output_order":
      return "output_order";
    default:
      return "generic_preference";
  }
}

function buildPreferenceGuidanceFacet(
  preference: MemoryCandidatePreferenceHint,
): MemoryGuidanceFacet | undefined {
  const guidance = preference.guidance;
  if (guidance?.guidanceText?.trim()) {
    return {
      guidanceType: guidance.guidanceType,
      guidanceText: guidance.guidanceText.trim(),
      confidence: guidance.confidence ?? preference.confidence,
      reason: guidance.reason ?? preference.reason,
    };
  }
  const object = preference.object.trim();
  if (!object) {
    return undefined;
  }
  const guidanceType = preferenceGuidanceType(preference.predicate);
  let guidanceText = "";
  switch (guidanceType) {
    case "language":
      if (object === "bilingual responses") {
        guidanceText =
          "Default to bilingual responses unless the current turn asks for a different language.";
      } else if (object === "chinese responses") {
        guidanceText =
          "Default to Chinese responses unless the current turn asks for a different language.";
      } else if (object === "english responses") {
        guidanceText =
          "Default to English responses unless the current turn asks for a different language.";
      } else {
        guidanceText = `Default to ${object} unless the current turn asks otherwise.`;
      }
      break;
    case "style":
      guidanceText = `Default to ${object} unless the current turn asks otherwise.`;
      break;
    case "charset":
      guidanceText = `Prefer ${object} when it fits the request.`;
      break;
    case "output_order":
      if (object === "zh first, en second") {
        guidanceText = "When replying bilingually, put Chinese first and English second.";
      } else if (object === "en first, zh second") {
        guidanceText = "When replying bilingually, put English first and Chinese second.";
      } else {
        guidanceText = `Honor this remembered output ordering preference when it fits the request: ${object}.`;
      }
      break;
    case "generic_preference":
      guidanceText = `Honor this remembered user preference when it fits the request: ${object}.`;
      break;
  }
  return {
    guidanceType,
    guidanceText,
    confidence: preference.confidence,
    reason: preference.reason,
  };
}

function storedPreferenceHint(params: {
  subject: string;
  predicate: string;
  object?: string;
}): MemoryCandidatePreferenceHint | null {
  const object = params.object?.trim();
  if (!object || normalizeName(params.subject) !== subjectUser()) {
    return null;
  }
  const predicate = normalizeName(params.predicate);
  if (
    predicate === "prefers_language" ||
    predicate === "prefers_response_style" ||
    predicate === "prefers_style" ||
    predicate === "prefers_code_style" ||
    predicate === "prefers_output_charset" ||
    predicate === "prefers_charset" ||
    predicate === "prefers_output_order"
  ) {
    return {
      predicate,
      object,
      confidence: 0.95,
      reason: "explicit tool preference fact",
    };
  }
  if (predicate !== "prefers") {
    return null;
  }
  return {
    predicate,
    object,
    confidence: 0.84,
    reason: "generic preference fact",
  };
}

function buildDecisionGuidanceFacet(
  decision: MemoryCandidateDecisionHint,
): MemoryGuidanceFacet | undefined {
  const summary = decision.summary.trim();
  if (!summary) {
    return undefined;
  }
  return {
    guidanceType: "generic_preference",
    guidanceText: `Honor this remembered user instruction when it fits the request: ${summary}.`,
    confidence: decision.confidence,
    reason: decision.reason,
  };
}

function buildWorkflowGuidanceFacet(
  summary: string,
  confidence?: number,
  reason?: string,
): MemoryGuidanceFacet | undefined {
  const normalized = summary.trim();
  if (!normalized) {
    return undefined;
  }
  return {
    guidanceType: "generic_preference",
    guidanceText: `When this workflow pattern applies, follow this remembered procedure: ${normalized}.`,
    confidence,
    reason: reason ?? "procedure-like semantic draft",
  };
}

function mergeFactObjectValueJson(
  objectValueJson: Record<string, unknown> | undefined,
  guidance: MemoryGuidanceFacet | undefined,
): Record<string, unknown> | undefined {
  if (!guidance) {
    return objectValueJson;
  }
  return {
    ...(objectValueJson ?? {}),
    guidance,
  };
}

export function buildStoredFactObjectValueJson(params: {
  subject: string;
  predicate: string;
  object?: string;
  objectValueJson?: Record<string, unknown>;
}): Record<string, unknown> | undefined {
  const predicate = normalizeName(params.predicate);
  const preference = storedPreferenceHint(params);
  if (!preference) {
    return params.objectValueJson;
  }
  return mergeFactObjectValueJson(params.objectValueJson, buildPreferenceGuidanceFacet(preference));
}

function buildEvent(ctx: MemoryOperationContext, candidate: ClassifiedCandidate): NormalizedEvent {
  const sourceRef = `${candidate.source.kind}:${candidate.source.toolName ?? candidate.source.messageId ?? candidate.candidateId}`;
  const structuredHints = buildEventStructuredHintSnapshot(candidate.structuredHints);
  const structuredSummary = buildEventStructuredSummary(candidate);
  const detailExcerpt = buildEventDetailExcerpt(candidate, structuredSummary);
  const temporalFacet = buildEventTemporalFacet(candidate, structuredSummary);
  return {
    eventId: stableHash([
      ctx.agentId,
      candidate.scope,
      candidate.eventType ?? "event",
      normalizeText(candidate.rawText),
      candidate.observedAt,
    ]),
    agentId: ctx.agentId,
    scope: candidate.scope,
    eventType: candidate.eventType ?? "event",
    text: redactSensitiveText(candidate.rawText, ctx.config.piiMode),
    normalizedText: normalizeText(candidate.rawText),
    observedAt: candidate.observedAt,
    sourceKind: candidate.source.kind,
    sourceRef,
    sessionKey: candidate.source.sessionKey,
    toolName: candidate.source.toolName,
    confidence: candidate.confidence,
    metadataJson: {
      ...(candidate.metadata ?? {}),
      ...(structuredSummary ? { memxStructuredSummary: structuredSummary } : {}),
      ...(detailExcerpt ? { memxRetrievalDetailExcerpt: detailExcerpt } : {}),
      ...(structuredHints ? { memxStructuredHints: structuredHints } : {}),
      ...(temporalFacet ? { memxTemporalFacet: temporalFacet } : {}),
    },
  };
}

function relationSummaryText(relation: MemoryCandidateRelationHint): string {
  const basePredicate =
    relation.predicate === "related_to" && relation.rawPredicate?.trim()
      ? relation.rawPredicate.trim()
      : relation.predicate;
  const predicate = relation.relationSlot
    ? `${basePredicate}[${relation.relationSlot}]`
    : basePredicate;
  return `${relation.subject} ${predicate} ${relation.object}`;
}

function relationSummaryLabel(relation: MemoryCandidateRelationHint): string {
  if (
    relation.predicate === "related_to" &&
    (relation.rawPredicate === "met" ||
      relation.rawPredicate === "introduced_to" ||
      relation.rawPredicate === "contacted" ||
      relation.rawPredicate === "exchanged_numbers_with" ||
      relation.rawPredicate === "followed_up_with")
  ) {
    return "Contact";
  }
  switch (relation.predicate) {
    case "caused_by":
      return "Cause";
    case "contradicts":
      return "Constraint";
    case "depends_on":
      return "Dependency";
    case "resolved_by":
      return "Resolution";
    case "uses":
      return "Component";
    default:
      return "Relation";
  }
}

function conciseEventText(text: string): string {
  const clauses = text
    .split(/[。.!！？?；;\n]/u)
    .map((entry) => entry.trim())
    .filter(Boolean);
  if (clauses.length === 0) {
    return truncateText(text.trim(), 220);
  }
  return truncateText(clauses.slice(0, 2).join(" | "), 220);
}

function semanticDraftSupportTexts(candidate: ClassifiedCandidate): string[] {
  const draft = getSemanticDraft(candidate);
  if (!draft) {
    return [];
  }
  const sourceRef = draft.sourceRef;
  const texts = draft.supportSpans
    .filter((entry) => !sourceRef || entry.sourceRef === sourceRef)
    .map((entry) => entry.text.trim())
    .filter(Boolean);
  return [...new Set(texts)];
}

function fragmentAddsRetrievalDetail(summary: string | undefined, fragment: string): boolean {
  const summaryTerms = new Set(normalizedTerms(summary ?? "", { minLength: 2 }));
  const fragmentTerms = normalizedTerms(fragment, { minLength: 2 });
  if (fragmentTerms.length === 0) {
    return false;
  }
  if (summaryTerms.size === 0) {
    return true;
  }
  const missingTerms = fragmentTerms.filter((term) => !summaryTerms.has(term));
  if (missingTerms.some((term) => /\d/u.test(term))) {
    return true;
  }
  return missingTerms.length >= 2;
}

function buildEventDetailExcerpt(
  candidate: ClassifiedCandidate,
  structuredSummary: string | undefined,
): string | undefined {
  const fragments = [
    ...semanticDraftSupportTexts(candidate),
    conciseEventText(candidate.rawText),
  ].filter(Boolean);
  const selected: string[] = [];
  const seen = new Set<string>();
  for (const fragment of fragments) {
    const normalized = normalizeText(fragment);
    if (!normalized || seen.has(normalized)) {
      continue;
    }
    seen.add(normalized);
    if (!fragmentAddsRetrievalDetail(structuredSummary, fragment)) {
      continue;
    }
    selected.push(fragment);
    if (selected.length >= 2) {
      break;
    }
  }
  if (selected.length === 0) {
    return undefined;
  }
  return truncateText(selected.join(" | "), 260);
}

function buildEventStructuredSummary(candidate: ClassifiedCandidate): string | undefined {
  const hints = candidate.structuredHints;
  const workflows =
    hints?.workflows && hints.workflows.length > 0
      ? hints.workflows
      : hints?.workflow
        ? [hints.workflow]
        : [];
  const relations =
    hints?.relations && hints.relations.length > 0
      ? hints.relations
      : hints?.relation
        ? [hints.relation]
        : [];
  if (relations.length > 0) {
    return truncateText(
      relations
        .slice(0, 2)
        .map((relation) => `${relationSummaryLabel(relation)}: ${relationSummaryText(relation)}`)
        .join(" | "),
      220,
    );
  }
  if (workflows.length > 0) {
    return truncateText(
      workflows
        .slice(0, 2)
        .map((workflow) => {
          const key = canonicalStateKey(workflow.key);
          return `State: ${key}: ${describeStateValue(key, workflow.value)}`;
        })
        .join(" | "),
      220,
    );
  }
  if (hints?.decision?.summary?.trim()) {
    return truncateText(`Decision: ${hints.decision.summary.trim()}`, 220);
  }
  if (hints?.correction) {
    const pieces = [
      `Correction(${hints.correction.timeframe})`,
      hints.correction.priorValue ? `from ${hints.correction.priorValue}` : "",
      hints.correction.nextValue ? `to ${hints.correction.nextValue}` : "",
      hints.correction.canonicalKey ?? hints.correction.predicate ?? "",
    ].filter(Boolean);
    const correctionSummary = truncateText(pieces.join(" "), 220);
    const observationFallback = conciseEventText(candidate.rawText);
    if (fragmentAddsRetrievalDetail(correctionSummary, observationFallback)) {
      return truncateText(`${correctionSummary} | Observation: ${observationFallback}`, 220);
    }
    return correctionSummary;
  }
  if (hints?.preference?.object?.trim()) {
    const preference = canonicalizePreferenceHint(hints.preference);
    return truncateText(
      `Preference: ${preference?.predicate ?? hints.preference.predicate}: ${hints.preference.object.trim()}`,
      220,
    );
  }
  const fallback = conciseEventText(candidate.rawText);
  return fallback ? `Observation: ${fallback}` : undefined;
}

function buildEventTemporalFacet(
  candidate: ClassifiedCandidate,
  structuredSummary: string | undefined,
): Record<string, unknown> | undefined {
  if (!structuredSummary) {
    return undefined;
  }
  const hints = candidate.structuredHints;
  const relations =
    hints?.relations && hints.relations.length > 0
      ? hints.relations
      : hints?.relation
        ? [hints.relation]
        : [];
  const workflows =
    hints?.workflows && hints.workflows.length > 0
      ? hints.workflows
      : hints?.workflow
        ? [hints.workflow]
        : [];
  const role =
    relations.length > 0
      ? relationSummaryLabel(relations[0]!).toLowerCase()
      : workflows.length > 0
        ? "state"
        : hints?.decision
          ? "decision"
          : hints?.preference
            ? "preference"
            : "observation";
  return {
    role,
    summary: structuredSummary,
    structured:
      relations.length > 0 ||
      workflows.length > 0 ||
      Boolean(hints?.decision) ||
      Boolean(hints?.preference),
    relationCount: relations.length,
    workflowCount: workflows.length,
    entityCount: Array.isArray(hints?.entities) ? hints.entities.length : 0,
    timeHintCount: Array.isArray(hints?.timeHints) ? hints.timeHints.length : 0,
  };
}

function buildEventStructuredHintSnapshot(
  hints: MemoryCandidateStructuredHints | undefined,
): Record<string, unknown> | undefined {
  if (!hints) {
    return undefined;
  }
  const workflows =
    hints.workflows && hints.workflows.length > 0
      ? hints.workflows
      : hints.workflow
        ? [hints.workflow]
        : [];
  const snapshot: Record<string, unknown> = {
    ...(Array.isArray(hints.entities) && hints.entities.length > 0
      ? { entities: hints.entities }
      : {}),
    ...(Array.isArray(hints.timeHints) && hints.timeHints.length > 0
      ? { timeHints: hints.timeHints }
      : {}),
    ...(hints.preferenceHint ? { preferenceHint: true } : {}),
    ...(hints.decisionHint ? { decisionHint: true } : {}),
    ...(hints.relationHint ? { relationHint: true } : {}),
    ...(hints.taskStateHint ? { taskStateHint: true } : {}),
    ...(hints.preference ? { preference: hints.preference } : {}),
    ...(workflows.length > 0 ? { workflows } : {}),
    ...(hints.relation ? { relation: hints.relation } : {}),
    ...(Array.isArray(hints.relations) && hints.relations.length > 0
      ? { relations: hints.relations }
      : {}),
    ...(hints.decision ? { decision: hints.decision } : {}),
    ...(hints.correctionHint ? { correctionHint: true } : {}),
    ...(hints.correction ? { correction: hints.correction } : {}),
    ...(semanticDraftConsumedShapeFromDraft(hints.semanticDraft)
      ? { semanticDraft: semanticDraftConsumedShapeFromDraft(hints.semanticDraft) }
      : {}),
    ...(hints.materializationHint ? { materializationHint: hints.materializationHint } : {}),
  };
  return Object.keys(snapshot).length > 0 ? snapshot : undefined;
}

function getWorkflowHints(candidate: ClassifiedCandidate): MemoryCandidateWorkflowHint[] {
  if (getSemanticDraft(candidate) && !hasSemanticDraftFamily(candidate, "workflow")) {
    return [];
  }
  const raw =
    candidate.structuredHints?.workflows && candidate.structuredHints.workflows.length > 0
      ? candidate.structuredHints.workflows
      : candidate.structuredHints?.workflow
        ? [candidate.structuredHints.workflow]
        : [];
  return raw
    .map((workflow) => sanitizeWorkflowHint(workflow))
    .filter((workflow): workflow is MemoryCandidateWorkflowHint => Boolean(workflow));
}

function getRelationHints(candidate: ClassifiedCandidate): MemoryCandidateRelationHint[] {
  if (
    getSemanticDraft(candidate) &&
    !hasSemanticDraftFamily(candidate, "relation_like") &&
    getCorrectionHint(candidate)?.targetKind !== "relation"
  ) {
    return [];
  }
  if (candidate.structuredHints?.relations && candidate.structuredHints.relations.length > 0) {
    return candidate.structuredHints.relations.filter(
      (relation) => relation.polarity !== "negated",
    );
  }
  return candidate.structuredHints?.relation && candidate.structuredHints.relation.polarity !== "negated"
    ? [candidate.structuredHints.relation]
    : [];
}

function getPreferenceHint(candidate: ClassifiedCandidate): MemoryCandidatePreferenceHint | null {
  if (getSemanticDraft(candidate) && !hasSemanticDraftFamily(candidate, "preference")) {
    return null;
  }
  return canonicalizePreferenceHint(candidate.structuredHints?.preference) ?? null;
}

function getDecisionHint(candidate: ClassifiedCandidate): MemoryCandidateDecisionHint | null {
  if (
    getSemanticDraft(candidate) &&
    !hasSemanticDraftFamily(candidate, "fact_like") &&
    !hasSemanticDraftFamily(candidate, "strategy_like")
  ) {
    return null;
  }
  return candidate.structuredHints?.decision ?? null;
}

function getCorrectionHint(candidate: ClassifiedCandidate): MemoryCandidateCorrectionHint | null {
  return candidate.structuredHints?.correction ?? null;
}

function getSemanticDraft(candidate: ClassifiedCandidate): MemoryCandidateSemanticDraft | null {
  return candidate.structuredHints?.semanticDraft ?? null;
}

function getMaterializationHint(
  candidate: ClassifiedCandidate,
): MemoryCandidateMaterializationHint | null {
  return candidate.structuredHints?.materializationHint ?? null;
}

function semanticDraftFamilies(candidate: ClassifiedCandidate): TurnSemanticAssertionFamilyHint[] {
  const draft = getSemanticDraft(candidate);
  const draftFamilies = draft?.assertionDrafts.map((entry) => entry.familyHint) ?? [];
  if ((draft?.relationDrafts?.length ?? 0) > 0) {
    return [
      ...new Set<TurnSemanticAssertionFamilyHint>([...draftFamilies, "relation_like"]),
    ];
  }
  if (draftFamilies.length > 0) {
    return [...new Set<TurnSemanticAssertionFamilyHint>(draftFamilies)];
  }
  return [
    ...new Set<TurnSemanticAssertionFamilyHint>(candidate.structuredHints?.semanticFamilies ?? []),
  ];
}

function hasSemanticDraftFamily(
  candidate: ClassifiedCandidate,
  family: TurnSemanticAssertionFamilyHint,
): boolean {
  return semanticDraftFamilies(candidate).includes(family);
}

function semanticDraftConsumedShapeFromDraft(
  draft: MemoryCandidateSemanticDraft | null | undefined,
): Record<string, unknown> | undefined {
  if (!draft) {
    return undefined;
  }
  return {
    sourceRef: draft.sourceRef,
    assertionDraftIds: draft.assertionDrafts.map((entry) => entry.draftId),
    families: [
      ...new Set([
        ...draft.assertionDrafts.map((entry) => entry.familyHint),
        ...((draft.relationDrafts?.length ?? 0) > 0 ? (["relation_like"] as const) : []),
      ]),
    ],
    timeframes: [...new Set(draft.assertionDrafts.map((entry) => entry.timeframeHint))],
    slotHints: [
      ...new Set(
        draft.assertionDrafts.flatMap((entry) =>
          Array.isArray(entry.slotHints) ? entry.slotHints.filter(Boolean) : [],
        ),
      ),
    ],
    corrections: draft.correctionDrafts.map((entry) => ({
      sourceRef: entry.sourceRef,
      timeframe: entry.correction.timeframe,
      targetKind: entry.correction.targetKind,
      predicate: entry.correction.predicate,
      canonicalKey: entry.correction.canonicalKey,
    })),
    relations: draft.relationDrafts?.map((entry) => ({
      sourceRef: entry.sourceRef,
      subject: entry.relation.subject,
      predicate: entry.relation.predicate,
      object: entry.relation.object,
      relationSlot: entry.relation.relationSlot,
      confidence: entry.confidence ?? entry.relation.confidence,
    })),
    taskProposal: draft.taskProposal
      ? {
          decision: draft.taskProposal.decision,
          targetTaskId: draft.taskProposal.targetTaskId,
          confidence: draft.taskProposal.confidence,
          summary: draft.taskProposal.summary,
          summaryConfidence: draft.taskProposal.summaryConfidence,
        }
      : undefined,
    supportSpanCount: draft.supportSpans.length,
    compiler: draft.compilerProvenance,
  };
}

function semanticDraftConsumedShape(
  candidate: ClassifiedCandidate,
): Record<string, unknown> | undefined {
  return semanticDraftConsumedShapeFromDraft(getSemanticDraft(candidate));
}

const DURABLE_DECISION_SUMMARY_PATTERN =
  /\b(?:must use|always use|never use|default to|stick with|we are going with|from now on|keep using)\b/iu;
const DURABLE_DECISION_SUMMARY_PATTERN_ZH =
  /(?:必须用|只能用|默认(?:改成|设为|采用|用)?|以后(?:都)?用|今后(?:都)?用|统一用|固定用|记着(?:以后)?)/u;
const WORKFLOW_GUIDANCE_PREDICATE = "has_workflow_guidance";

function buildProcedureGuidanceSummary(candidate: ClassifiedCandidate): string | undefined {
  const decision = getDecisionHint(candidate)?.summary?.trim();
  const draftSupport =
    getSemanticDraft(candidate)?.supportSpans?.map((entry) => entry.text.trim()) ?? [];
  const source = decision || draftSupport.find(Boolean) || candidate.rawText.trim();
  if (!source) {
    return undefined;
  }
  const clauses = source
    .split(/[。.!！？?；;\n]/u)
    .map((entry) => entry.trim())
    .filter(Boolean);
  const summary = clauses.length > 0 ? clauses.slice(0, 3).join("；") : source;
  return truncateText(summary, 220);
}

function buildProcedureGuidanceFact(
  ctx: MemoryOperationContext,
  candidate: ClassifiedCandidate,
): NormalizedFact | null {
  if (!hasSemanticDraftFamily(candidate, "strategy_like")) {
    return null;
  }
  const summary = buildProcedureGuidanceSummary(candidate);
  if (!summary) {
    return null;
  }
  const guidance = buildWorkflowGuidanceFacet(
    summary,
    candidate.confidence,
    "strategy-like semantic draft",
  );
  const supportContentRefs = [
    sourceRefForCandidate(candidate),
    ...(typeof candidate.metadata?.chunkId === "string" && candidate.metadata.chunkId.trim()
      ? [`chunk:${candidate.metadata.chunkId.trim()}`]
      : []),
  ];
  return buildFact({
    ctx,
    candidate,
    subject: subjectUser(),
    predicate: WORKFLOW_GUIDANCE_PREDICATE,
    object: summary,
    objectValueJson: {
      ...(guidance ? { guidance } : {}),
      supportContentRefs: [...new Set(supportContentRefs)],
      semanticFamily: "strategy_like",
    },
  });
}

function shouldMaterializeDecisionSummary(
  candidate: ClassifiedCandidate,
  decision: MemoryCandidateDecisionHint | null,
  preference: MemoryCandidatePreferenceHint | null,
  relations: MemoryCandidateRelationHint[],
  correction: MemoryCandidateCorrectionHint | null,
): boolean {
  if (
    candidate.classification !== "stable-fact" ||
    !decision ||
    preference ||
    relations.length > 0 ||
    correction ||
    hasSemanticDraftFamily(candidate, "strategy_like") ||
    candidate.source.kind !== "user"
  ) {
    return false;
  }
  if (
    (candidate.structuredHints?.timeHints?.length ?? 0) > 0 ||
    candidate.rawText.trim().length > 160
  ) {
    return false;
  }
  return (
    DURABLE_DECISION_SUMMARY_PATTERN.test(candidate.rawText) ||
    DURABLE_DECISION_SUMMARY_PATTERN_ZH.test(candidate.rawText)
  );
}

function semanticAssertionSupportText(
  candidate: ClassifiedCandidate,
  draft: MemoryCandidateSemanticDraft,
  assertion: TurnSemanticAssertionDraft,
): string {
  const directSpan = assertion.supportSpans?.find((span) => span.text.trim())?.text;
  const draftSpan = draft.supportSpans.find(
    (span) => span.sourceRef === assertion.sourceRef && span.text.trim(),
  )?.text;
  return truncateText((directSpan || draftSpan || candidate.rawText).trim(), 520);
}

function semanticAssertionSubject(assertion: TurnSemanticAssertionDraft): string {
  const firstEntity = assertion.entityHints?.find((entity) => entity.name.trim());
  return firstEntity?.name.trim() || subjectUser();
}

function semanticAssertionSlots(assertion: TurnSemanticAssertionDraft): string[] {
  const seen = new Set<string>();
  const slots: string[] = [];
  for (const slot of assertion.slotHints ?? []) {
    const predicate = normalizeSemanticFactPredicate(slot);
    if (!predicate || seen.has(predicate)) {
      continue;
    }
    seen.add(predicate);
    slots.push(predicate);
  }
  return slots;
}

function semanticAssertionObjectForSlot(params: {
  assertion: TurnSemanticAssertionDraft;
  subject: string;
  supportText: string;
  slotIndex: number;
  hasExplicitSlot: boolean;
}): string {
  if (params.assertion.valueHint?.trim()) {
    return params.assertion.valueHint.trim();
  }
  const subjectKey = normalizeName(params.subject);
  const objectEntities = (params.assertion.entityHints ?? [])
    .map((entity) => entity.name.trim())
    .filter((name) => name && normalizeName(name) !== subjectKey);
  if (params.hasExplicitSlot) {
    return objectEntities[params.slotIndex] ?? objectEntities[0] ?? params.supportText;
  }
  return params.supportText;
}

function entityTypeFromSemanticHint(type: string | undefined): EntityType | undefined {
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
      return undefined;
  }
}

function semanticEntityHintForName(
  assertion: TurnSemanticAssertionDraft,
  name: string | undefined,
): { name: string; type?: EntityType } | undefined {
  const normalized = normalizeName(name ?? "");
  if (!normalized) {
    return undefined;
  }
  for (const hint of assertion.entityHints ?? []) {
    const hintName = hint.name.trim();
    if (!hintName || normalizeName(hintName) !== normalized) {
      continue;
    }
    return {
      name: hintName,
      ...(entityTypeFromSemanticHint(hint.type) ? { type: entityTypeFromSemanticHint(hint.type) } : {}),
    };
  }
  return undefined;
}

function semanticObjectEntityHint(params: {
  assertion: TurnSemanticAssertionDraft;
  subject: string;
  object: string;
}): { name: string; type?: EntityType } | undefined {
  const subjectName = normalizeName(params.subject);
  const objectName = normalizeName(params.object);
  if (!objectName || objectName === subjectName) {
    return undefined;
  }
  const exact = semanticEntityHintForName(params.assertion, params.object);
  if (exact) {
    return exact;
  }
  if (params.assertion.valueHint?.trim()) {
    return undefined;
  }
  return (params.assertion.entityHints ?? [])
    .map((hint) => ({
      name: hint.name.trim(),
      type: entityTypeFromSemanticHint(hint.type),
    }))
    .find((hint) => hint.name && normalizeName(hint.name) !== subjectName);
}

function shouldMaterializeSemanticAssertionFact(params: {
  candidate: ClassifiedCandidate;
  assertion: TurnSemanticAssertionDraft;
  preference: MemoryCandidatePreferenceHint | null;
  decision: MemoryCandidateDecisionHint | null;
  relations: MemoryCandidateRelationHint[];
  correction: MemoryCandidateCorrectionHint | null;
  workflowHints: MemoryCandidateWorkflowHint[];
}): boolean {
  if (
    params.candidate.classification !== "stable-fact" ||
    params.assertion.familyHint !== "fact_like" ||
    sourceRefRole(params.assertion.sourceRef) === "assistant" ||
    params.preference ||
    params.decision ||
    params.correction ||
    params.workflowHints.length > 0 ||
    (params.candidate.structuredHints?.resourceAssertions?.length ?? 0) > 0 ||
    (params.candidate.structuredHints?.adviceSignals?.length ?? 0) > 0
  ) {
    return false;
  }
  return Boolean(semanticAssertionSubject(params.assertion));
}

function buildSemanticAssertionFacts(
  ctx: MemoryOperationContext,
  candidate: ClassifiedCandidate,
  draft: MemoryCandidateSemanticDraft,
  assertion: TurnSemanticAssertionDraft,
): NormalizedFact[] {
  const supportText = semanticAssertionSupportText(candidate, draft, assertion);
  if (!supportText) {
    return [];
  }
  const subject = semanticAssertionSubject(assertion);
  const slotPredicates = semanticAssertionSlots(assertion);
  const predicates = slotPredicates.length > 0 ? slotPredicates : ["reported_detail"];
  return predicates.map((predicate, index) => {
    const object = semanticAssertionObjectForSlot({
      assertion,
      subject,
      supportText,
      slotIndex: index,
      hasExplicitSlot: slotPredicates.length > 0,
    });
    return buildFact({
      ctx,
      candidate,
      subject,
      predicate,
      object,
      confidence: assertion.confidence,
      sourceRef: assertion.sourceRef,
      provenanceText: supportText,
      objectValueJson: {
        semanticAssertion: {
          familyHint: assertion.familyHint,
          timeframeHint: assertion.timeframeHint,
          entityHints: assertion.entityHints ?? [],
          slotHints: assertion.slotHints ?? [],
          ...(assertion.valueHint ? { valueHint: assertion.valueHint } : {}),
          materializedPredicate: predicate,
          supportText,
        },
        supportContentRefs: [...new Set([assertion.sourceRef, sourceRefForCandidate(candidate)])],
        currentnessHint: assertion.timeframeHint,
      },
    });
  });
}

function relatedCorrectionAssertions(
  draft: MemoryCandidateSemanticDraft | null,
  sourceRef: string,
): TurnSemanticAssertionDraft[] {
  return (draft?.assertionDrafts ?? []).filter((entry) => entry.sourceRef === sourceRef);
}

function correctionSubjectFromCanonicalKey(canonicalKey: string | undefined): string | undefined {
  const parts = canonicalKey
    ?.split(/[.:/#|]+/u)
    .map((entry) => entry.trim())
    .filter(Boolean);
  return parts && parts.length > 1 ? parts[0] : undefined;
}

function correctionSubject(
  correction: MemoryCandidateCorrectionHint,
  assertions: TurnSemanticAssertionDraft[],
): string {
  const assertionSubject = assertions
    .map((assertion) => semanticAssertionSubject(assertion))
    .find((subject) => subject !== subjectUser());
  return assertionSubject ?? correctionSubjectFromCanonicalKey(correction.canonicalKey) ?? subjectUser();
}

function correctionPredicate(
  correction: MemoryCandidateCorrectionHint,
  assertions: TurnSemanticAssertionDraft[],
): string | undefined {
  const canonicalKeyPredicate = normalizeSemanticFactPredicate(correction.canonicalKey);
  const assertionPredicate = assertions.flatMap((assertion) => semanticAssertionSlots(assertion))[0];
  const rawPredicate = normalizeSemanticFactPredicate(correction.predicate);
  return canonicalKeyPredicate ?? assertionPredicate ?? rawPredicate;
}

// Predicates that represent static structural topology (dependency graphs, ownership, architecture).
// For these, the graph edge alone captures the relationship; an episodic event adds noise without
// temporal value. Only temporal/causal predicates (caused_by, resolved_by, supersedes) warrant events.
const STRUCTURAL_TOPOLOGY_PREDICATES = new Set([
  "depends_on",
  "uses",
  "reads",
  "blocks",
  "part_of",
  "owner_of",
  "related_to",
  // Note: "contradicts" is intentionally excluded — semantic conflicts are event-worthy.
]);

const TEMPORAL_RELATION_RAW_PREDICATES = new Set([
  "met",
  "introduced_to",
  "contacted",
  "exchanged_numbers_with",
  "followed_up_with",
]);

function shouldMaterializeObservedEvent(
  candidate: ClassifiedCandidate,
  relations: MemoryCandidateRelationHint[],
): boolean {
  if (candidate.classification === "episodic-event") {
    return true;
  }
  if (candidate.source.kind === "assistant") {
    return false;
  }
  if (
    candidate.structuredHints?.correction &&
    (candidate.structuredHints.correction.timeframe === "historical" ||
      candidate.structuredHints.correction.timeframe === "compare" ||
      (candidate.structuredHints.correction.priorValue &&
        candidate.structuredHints.correction.nextValue))
  ) {
    return true;
  }
  if (relations.length > 0) {
    // Only materialize an event for temporal/causal relations. Pure structural topology
    // (depends_on, uses, blocks, part_of, …) is fully captured by the graph edge — a
    // duplicate event entry adds noise without temporal value.
    return relations.some(
      (r) =>
        !STRUCTURAL_TOPOLOGY_PREDICATES.has(r.predicate) ||
        (r.predicate === "related_to" &&
          Boolean(r.rawPredicate && TEMPORAL_RELATION_RAW_PREDICATES.has(r.rawPredicate))),
    );
  }
  const timeHints = candidate.structuredHints?.timeHints?.length ?? 0;
  return timeHints > 0 && Boolean(candidate.structuredHints?.decision);
}

function buildState(
  ctx: MemoryOperationContext,
  candidate: ClassifiedCandidate,
  workflow: MemoryCandidateWorkflowHint,
  stateKind: MemoryWorkflowStateKind,
): NormalizedState {
  return {
    key: canonicalStateKey(workflow.key),
    valueJson: workflow.value,
    scope: candidate.scope,
    agentId: ctx.agentId,
    stateKind,
    confidence: candidate.confidence,
    sourceRef: `${candidate.source.kind}:${candidate.candidateId}`,
    updatedAt: candidate.observedAt,
    expiresAt: undefined,
  };
}

function resolveWorkflowStateKind(
  candidate: ClassifiedCandidate,
  ctx: MemoryOperationContext,
  workflow: MemoryCandidateWorkflowHint,
): MemoryWorkflowStateKind | null {
  if (workflow.stateKind) {
    return workflow.stateKind;
  }
  if (candidate.policy.action === "durable_state") {
    return "durable";
  }
  if (candidate.policy.action === "session_state") {
    return "session";
  }
  if (!candidate.policy.captureAuthorized || candidate.policy.action === "ignore") {
    return null;
  }
  if (candidate.policy.salienceScore < ctx.config.minSalienceSession) {
    return null;
  }
  return "session";
}

function buildVectorDoc(params: {
  kind: VectorDocRecord["docKind"];
  sourceId: string;
  scope: string;
  agentId: string;
  text: string;
  confidence: number;
  observedAt: string;
  metadataJson?: Record<string, unknown>;
  lineage?: {
    canonicalKind?: "state" | "fact" | "event" | "graph_edge";
    canonicalId?: string;
    sourceKind?: "state" | "fact" | "event" | "graph_edge";
    sourceId?: string;
    sourceRef?: string;
    materializedEpoch?: number;
  };
}): VectorDocRecord {
  const defaultLineage =
    params.kind === "edge"
      ? {
          canonicalKind: "graph_edge" as const,
          canonicalId: params.sourceId,
          sourceKind: "graph_edge" as const,
          sourceId: params.sourceId,
        }
      : params.kind === "entity_profile"
        ? {
            canonicalKind: "entity" as const,
            canonicalId: params.sourceId,
            sourceKind: "vector_doc" as const,
            sourceId: params.sourceId,
          }
        : {
            canonicalKind: params.kind,
            canonicalId: params.sourceId,
            sourceKind: params.kind,
            sourceId: params.sourceId,
          };
  const lineage = {
    ...defaultLineage,
    ...(params.lineage ?? {}),
  };
  return {
    docId: `${params.kind}:${params.sourceId}`,
    docKind: params.kind,
    sourceId: params.sourceId,
    scope: params.scope,
    agentId: params.agentId,
    text: params.text,
    metadataJson: buildVectorDocMetadata({
      docType: params.kind,
      confidence: params.confidence,
      observedAt: params.observedAt,
      lineage,
      extra: params.metadataJson,
    }),
    createdAt: params.observedAt,
    updatedAt: params.observedAt,
  };
}

function buildRelationMetadata(relation: MemoryCandidateRelationHint): Record<string, unknown> {
  return {
    graph: {
      relationType: relation.predicate,
      rawPredicate: relation.rawPredicate ?? relation.predicate,
      ...(relation.relationSlot ? { relationSlot: relation.relationSlot } : {}),
      sourceKind: "extracted",
      confidence: relation.confidence,
      reason: relation.reason,
    },
  };
}

export function normalizeCandidate(
  candidate: ClassifiedCandidate,
  ctx: MemoryOperationContext,
): NormalizedOutputs {
  const text = redactSensitiveText(candidate.rawText, ctx.config.piiMode);
  const outputs: NormalizedOutputs = {
    states: [],
    facts: [],
    events: [],
    entities: [],
    edges: [],
    vectorDocs: [],
  };

  const workflowHints = getWorkflowHints(candidate);
  const currentProject =
    typeof candidate.metadata?.currentProject === "string" &&
    candidate.metadata.currentProject.trim()
      ? candidate.metadata.currentProject.trim()
      : undefined;
  const currentProjectProfile = objectRecord(candidate.metadata?.currentProjectProfile);
  const knownProjectNames = [
    currentProject,
    typeof currentProjectProfile?.projectCode === "string"
      ? currentProjectProfile.projectCode.trim()
      : undefined,
    ...workflowHints.map((workflow) => projectCodeFromStateKey(workflow.key)),
    ...workflowHints.map((workflow) =>
      typeof workflow.value.projectCode === "string"
        ? workflow.value.projectCode.trim()
        : undefined,
    ),
  ].filter((entry): entry is string => Boolean(entry?.trim()));
  const canonicalCurrentProject = currentProject
    ? resolveProjectReference(currentProject, {
        currentProject,
        knownProjects: knownProjectNames,
      })
    : undefined;
  const relations = getRelationHints(candidate)
    .filter((relation) => sourceRefRole(relation.sourceRef) !== "assistant")
    .map((relation) => {
      const canonicalSubject = resolveProjectReference(relation.subject, {
        currentProject: canonicalCurrentProject,
        knownProjects: knownProjectNames,
        allowDescriptorAlias: Boolean(
          relation.relationSlot || canonicalCurrentProject || currentProjectProfile,
        ),
      });
      return canonicalSubject === relation.subject
        ? relation
        : {
            ...relation,
            subject: canonicalSubject,
          };
    });
  const seenRelationFacts = new Set<string>();
  const projectProfiles = new Map<string, Record<string, unknown>>();
  const projectProfileProtectedPaths = new Map<string, Set<string>>();
  const projectProfileBase = (
    key: string,
    projectCode: string | undefined,
  ): Record<string, unknown> | undefined =>
    projectProfiles.get(key) ??
    (projectCode && canonicalCurrentProject && projectCode === canonicalCurrentProject
      ? currentProjectProfile
      : undefined);
  const correction = getCorrectionHint(candidate);
  if (correction?.targetKind === "project_profile") {
    const correctionProjectCode =
      typeof currentProjectProfile?.projectCode === "string" &&
      currentProjectProfile.projectCode.trim()
        ? currentProjectProfile.projectCode.trim()
        : (canonicalCurrentProject ?? knownProjectNames[0]);
    if (correctionProjectCode) {
      const stateKey = projectProfileStateKey(correctionProjectCode);
      const baseProfile = projectProfileBase(stateKey, correctionProjectCode);
      const correctionUpdate = buildCorrectionProjectProfileUpdate({
        candidate,
        correction,
        projectCode: correctionProjectCode,
        currentValue: baseProfile,
      });
      if (correctionUpdate) {
        projectProfiles.set(stateKey, mergeProjectProfileValue(baseProfile, correctionUpdate));
        projectProfileProtectedPaths.set(
          stateKey,
          new Set([
            ...(projectProfileProtectedPaths.get(stateKey) ?? new Set<string>()),
            ...projectProfileFieldPaths(correctionUpdate),
          ]),
        );
      }
    }
  }
  const entityIndexByName = new Map<string, number>();
  const rewriteEntityReferences = (fromId: string, toId: string) => {
    for (const edge of outputs.edges) {
      if (edge.srcEntityId === fromId) {
        edge.srcEntityId = toId;
      }
      if (edge.dstEntityId === fromId) {
        edge.dstEntityId = toId;
      }
    }
  };
  const pushEntity = (
    name: string,
    type?: EntityType,
    aliases: string[] = [],
  ): NormalizedEntity | null => {
    const normalizedName = normalizeName(name);
    const wantsProject =
      type === "project" ||
      (canonicalCurrentProject ? normalizeName(canonicalCurrentProject) === normalizedName : false);
    const nextType: EntityType = wantsProject
      ? "project"
      : (type ?? "unknown");
    const existingIndex = entityIndexByName.get(normalizedName);
    if (existingIndex != null) {
      const existing = outputs.entities[existingIndex]!;
      // Upgrade from "unknown" to any specific type (entityId stays the same
      // since it's based on normalizedName only — no need to rewrite references)
      if (existing.entityType === "unknown" && nextType !== "unknown") {
        existing.entityType = nextType;
      }
      existing.aliases = mergeAliases(existing.aliases, aliases);
      return existing;
    }
    const entity = buildEntity(name, nextType);
    if (!entity) return null;
    entity.aliases = mergeAliases(entity.aliases, aliases);
    entityIndexByName.set(normalizedName, outputs.entities.length);
    outputs.entities.push(entity);
    return entity;
  };

  const pushResourceEdge = (params: {
    srcName: string;
    srcType?: EntityType;
    relType: GraphRelationType;
    dstName: string;
    dstType?: EntityType;
    rawRelationType: string;
    sourceKind: "extracted" | "synthesized";
    confidence: number;
    sourceRef: string;
    text: string;
    metadataJson: Record<string, unknown>;
  }) => {
    const srcEntity = pushEntity(params.srcName, params.srcType);
    const dstEntity = pushEntity(params.dstName, params.dstType);
    if (!srcEntity || !dstEntity) {
      return;
    }
    const edgeId = stableHash([
      ctx.agentId,
      candidate.scope,
      srcEntity.entityId,
      params.relType,
      params.rawRelationType,
      dstEntity.entityId,
      params.sourceRef,
    ]);
    if (!outputs.edges.some((edge) => edge.edgeId === edgeId)) {
      outputs.edges.push({
        edgeId,
        srcEntityId: srcEntity.entityId,
        relType: params.relType,
        dstEntityId: dstEntity.entityId,
        scope: candidate.scope,
        agentId: ctx.agentId,
        confidence: params.confidence,
        validFrom: candidate.observedAt,
        evidenceRef: params.sourceRef,
        rawRelationType: params.rawRelationType,
        sourceKind: params.sourceKind,
        createdAt: candidate.observedAt,
        updatedAt: candidate.observedAt,
      });
    }
    if (!outputs.vectorDocs.some((doc) => doc.docId === `edge:${edgeId}`)) {
      outputs.vectorDocs.push(
        buildVectorDoc({
          kind: "edge",
          sourceId: edgeId,
          scope: candidate.scope,
          agentId: ctx.agentId,
          text: truncateText(params.text, 520),
          confidence: params.confidence,
          observedAt: candidate.observedAt,
          metadataJson: params.metadataJson,
          lineage: {
            canonicalKind: "graph_edge",
            canonicalId: edgeId,
            sourceKind: "graph_edge",
            sourceId: edgeId,
            sourceRef: params.sourceRef,
          },
        }),
      );
    }
  };

  const pushAttributeSlotEdgeFromSemanticAssertion = (params: {
    assertion: TurnSemanticAssertionDraft;
    subject: string;
    predicate: string;
    object: string;
    sourceRef: string;
    supportText: string;
    confidence?: number;
  }) => {
    const slot = canonicalAttributeSlot(params.predicate);
    if (!slot || sourceRefRole(params.sourceRef) === "assistant") {
      return;
    }
    const objectHint = semanticObjectEntityHint({
      assertion: params.assertion,
      subject: params.subject,
      object: params.object,
    });
    if (!objectHint) {
      return;
    }
    const subjectHint = semanticEntityHintForName(params.assertion, params.subject);
    const srcEntity = pushEntity(
      params.subject,
      subjectHint?.type ?? (params.subject === subjectUser() ? "person" : undefined),
      subjectHint?.type === "project" ? projectAliasVariants(params.subject) : [],
    );
    const dstEntity = pushEntity(objectHint.name || params.object, objectHint.type);
    if (!srcEntity || !dstEntity) {
      return;
    }
    const edgeId = stableHash([
      ctx.agentId,
      candidate.scope,
      srcEntity.entityId,
      "uses",
      slot,
      dstEntity.entityId,
    ]);
    if (outputs.edges.some((edge) => edge.edgeId === edgeId)) {
      return;
    }
    outputs.edges.push({
      edgeId,
      srcEntityId: srcEntity.entityId,
      relType: "uses",
      relationSlot: slot,
      dstEntityId: dstEntity.entityId,
      scope: candidate.scope,
      agentId: ctx.agentId,
      confidence: params.confidence ?? candidate.confidence,
      validFrom: candidate.observedAt,
      evidenceRef: params.sourceRef,
      rawRelationType: params.predicate,
      sourceKind: "extracted",
      createdAt: candidate.observedAt,
      updatedAt: candidate.observedAt,
      metadataJson: {
        sourceRefs: [params.sourceRef],
        supportRefs: [params.sourceRef],
        relationType: "uses",
        relationSlot: slot,
        rawPredicate: params.predicate,
        sourceKind: "llm_semantic_assertion",
        semanticAssertion: {
          draftId: params.assertion.draftId,
          familyHint: params.assertion.familyHint,
          timeframeHint: params.assertion.timeframeHint,
          supportText: params.supportText,
        },
      },
    });
  };

  const pushAttributeSlotEdgeFromSemanticCorrection = (params: {
    correction: MemoryCandidateCorrectionHint;
    subject: string;
    predicate: string;
    object: string;
    sourceRef: string;
    supportText: string;
    confidence?: number;
  }) => {
    const slot = canonicalAttributeSlot(params.predicate);
    if (!slot || sourceRefRole(params.sourceRef) === "assistant") {
      return;
    }
    const srcEntity = pushEntity(params.subject, params.subject === subjectUser() ? "person" : undefined);
    const dstEntity = pushEntity(params.object);
    if (!srcEntity || !dstEntity) {
      return;
    }
    const edgeId = stableHash([
      ctx.agentId,
      candidate.scope,
      srcEntity.entityId,
      "uses",
      slot,
      dstEntity.entityId,
    ]);
    if (outputs.edges.some((edge) => edge.edgeId === edgeId)) {
      return;
    }
    outputs.edges.push({
      edgeId,
      srcEntityId: srcEntity.entityId,
      relType: "uses",
      relationSlot: slot,
      dstEntityId: dstEntity.entityId,
      scope: candidate.scope,
      agentId: ctx.agentId,
      confidence: params.confidence ?? candidate.confidence,
      validFrom: candidate.observedAt,
      evidenceRef: params.sourceRef,
      rawRelationType: params.predicate,
      sourceKind: "extracted",
      createdAt: candidate.observedAt,
      updatedAt: candidate.observedAt,
      metadataJson: {
        sourceRefs: [params.sourceRef],
        supportRefs: [params.sourceRef],
        relationType: "uses",
        relationSlot: slot,
        rawPredicate: params.predicate,
        sourceKind: "llm_semantic_correction",
        correction: {
          timeframe: params.correction.timeframe,
          targetKind: params.correction.targetKind,
          ...(params.correction.canonicalKey ? { canonicalKey: params.correction.canonicalKey } : {}),
          ...(params.correction.priorValue ? { priorValue: params.correction.priorValue } : {}),
          nextValue: params.object,
          supportText: params.supportText,
        },
      },
    });
  };

  const pushResourceAssertion = (assertion: MemoryResourceAssertionHint) => {
    const fact = buildResourceFact(ctx, candidate, assertion);
    if (
      !outputs.facts.some(
        (entry) =>
          entry.predicate === fact.predicate &&
          entry.canonicalSubject === fact.canonicalSubject &&
          entry.canonicalObject === fact.canonicalObject &&
          entry.sourceRef === fact.sourceRef,
      )
    ) {
      outputs.facts.push(fact);
    }
    const state = buildResourceState(ctx, candidate, assertion);
    if (state) {
      const existingIndex = outputs.states.findIndex((entry) => entry.key === state.key);
      if (existingIndex >= 0) {
        outputs.states[existingIndex] = state;
      } else {
        outputs.states.push(state);
      }
    }
    outputs.events.push(buildResourceEvent(ctx, candidate, assertion));
    pushEntity(assertion.resource, resourceEntityType(assertion));
    pushResourceEdge({
      srcName: assertion.owner || subjectUser(),
      srcType: assertion.owner === "user" ? "person" : undefined,
      relType: "owner_of",
      dstName: assertion.resource,
      dstType: resourceEntityType(assertion),
      rawRelationType: "has_resource",
      sourceKind: "extracted",
      confidence: assertion.confidence,
      sourceRef: assertion.sourceRef,
      text: `${assertion.owner || subjectUser()} owner_of ${assertion.resource} | evidence: ${assertion.supportText}`,
      metadataJson: {
        relationType: "owner_of",
        rawPredicate: "has_resource",
        sourceKind: "extracted",
        ...resourceMetadata(assertion),
      },
    });
  };

  for (const assertion of candidate.structuredHints?.resourceAssertions ?? []) {
    pushResourceAssertion(assertion);
  }

  for (const signal of candidate.structuredHints?.adviceSignals ?? []) {
    const adviceSignalFact = buildAdviceSignalFact(ctx, candidate, signal);
    if (adviceSignalFact) {
      outputs.facts.push(adviceSignalFact);
    }
  }

  for (const relation of relations) {
    const relationKey = `${normalizeText(relation.subject)}:${relation.predicate}:${relation.relationSlot ?? ""}:${normalizeText(relation.object)}`;
    if (seenRelationFacts.has(relationKey)) {
      continue;
    }
    seenRelationFacts.add(relationKey);
    if (relation.relationSlot) {
      const stateKey = projectProfileStateKey(relation.subject);
      const baseProfile = projectProfileBase(stateKey, relation.subject.trim());
      projectProfiles.set(
        stateKey,
        mergeProjectProfileValue(baseProfile, {
          projectCode: relation.subject.trim(),
          components: {
            [relation.relationSlot]: relation.object.trim(),
          },
        }),
      );
    }
    const subjectAliases = projectAliasVariants(relation.subject);
    if (canonicalCurrentProject && projectNamesMatch(relation.subject, canonicalCurrentProject)) {
      subjectAliases.push(...projectAliasVariants(canonicalCurrentProject));
    }
    const inferredObjectType: EntityType | undefined =
      relation.predicate === "related_to" &&
      relation.rawPredicate &&
      TEMPORAL_RELATION_RAW_PREDICATES.has(relation.rawPredicate)
        ? "person"
        : undefined;
    const srcEntity = pushEntity(
      relation.subject,
      relation.relationSlot ? "project" : relation.subject === subjectUser() ? "person" : undefined,
      subjectAliases,
    );
    const dstEntity = pushEntity(relation.object, inferredObjectType);
    if (!srcEntity || !dstEntity) {
      continue;
    }
    const evidenceRef = relation.sourceRef ?? sourceRefForCandidate(candidate);
    const edgeId = stableHash([
      ctx.agentId,
      candidate.scope,
      srcEntity.entityId,
      relation.predicate,
      relation.relationSlot ?? "",
      dstEntity.entityId,
    ]);
    outputs.edges.push({
      edgeId,
      srcEntityId: srcEntity.entityId,
      relType: relation.predicate,
      dstEntityId: dstEntity.entityId,
      ...(relation.relationSlot ? { relationSlot: relation.relationSlot } : {}),
      scope: candidate.scope,
      agentId: ctx.agentId,
      confidence: candidate.confidence,
      validFrom: candidate.observedAt,
      evidenceRef,
      rawRelationType: relation.rawPredicate,
      sourceKind: "extracted",
      createdAt: candidate.observedAt,
      updatedAt: candidate.observedAt,
      metadataJson: {
        sourceRefs: [evidenceRef],
        supportRefs: [evidenceRef],
      },
    });
    if (shouldDeriveRelationFact(relation)) {
      outputs.facts.push(
        buildFact({
          ctx,
          candidate,
          subject: relation.subject,
          predicate: relationFactPredicate(relation),
          object: relation.object,
          objectValueJson: {
            ...(relation.relationSlot ? { componentRole: relation.relationSlot } : {}),
            ...buildRelationMetadata(relation),
          },
        }),
      );
    }
    outputs.vectorDocs.push(
      buildVectorDoc({
        kind: "edge",
        sourceId: edgeId,
        scope: candidate.scope,
        agentId: ctx.agentId,
        text: relationSummaryText(relation),
        confidence: candidate.confidence,
        observedAt: candidate.observedAt,
        metadataJson: {
          relationType: relation.predicate,
          rawPredicate: relation.rawPredicate ?? relation.predicate,
          ...(relation.relationSlot ? { relationSlot: relation.relationSlot } : {}),
          sourceKind: "extracted",
        },
      }),
    );
  }

  const preference = getPreferenceHint(candidate);
  if (preference && shouldMaterializePreferenceFact(preference)) {
    const guidance = buildPreferenceGuidanceFacet(preference);
    outputs.facts.push(
      buildFact({
        ctx,
        candidate,
        subject: subjectUser(),
        predicate: preference.predicate,
        object: preference.object,
        objectValueJson: guidance ? { guidance } : undefined,
      }),
    );
  }

  const materializationHint = getMaterializationHint(candidate);
  const semanticDraft = getSemanticDraft(candidate);
  if (
    correction &&
    correction.targetKind === "fact" &&
    correction.nextValue?.trim() &&
    (!correction.priorValue ||
      normalizeText(correction.priorValue) !== normalizeText(correction.nextValue.trim()))
  ) {
    const correctionSourceRef =
      semanticDraft?.correctionDrafts[0]?.sourceRef ?? sourceRefForCandidate(candidate);
    if (sourceRefRole(correctionSourceRef) !== "assistant") {
      const correctionAssertions = relatedCorrectionAssertions(semanticDraft, correctionSourceRef);
      const predicate = correctionPredicate(correction, correctionAssertions);
      const correctionFact = buildFact({
        ctx,
        candidate,
        subject: correctionSubject(correction, correctionAssertions),
        predicate: predicate ?? "reported_detail",
        object: correction.nextValue.trim(),
        objectValueJson: {
          correction: {
            timeframe: correction.timeframe,
            ...(correction.priorValue ? { priorValue: correction.priorValue } : {}),
            nextValue: correction.nextValue.trim(),
          },
          replacement: {
            mode: materializationHint?.replacementMode ?? "none",
            targetKind: correction.targetKind,
            ...(predicate ? { predicate } : {}),
            ...(correction.canonicalKey ? { canonicalKey: correction.canonicalKey } : {}),
            ...(correction.priorValue ? { priorValue: correction.priorValue } : {}),
            nextValue: correction.nextValue.trim(),
          },
        },
      });
      if (
        !outputs.facts.some(
          (entry) =>
            entry.canonicalSubject === correctionFact.canonicalSubject &&
            entry.predicate === correctionFact.predicate &&
            entry.canonicalObject === correctionFact.canonicalObject,
        )
      ) {
        outputs.facts.push(correctionFact);
      }
      pushAttributeSlotEdgeFromSemanticCorrection({
        correction,
        subject: correctionFact.canonicalSubject,
        predicate: correctionFact.predicate,
        object: correction.nextValue.trim(),
        sourceRef: correctionSourceRef,
        supportText: correctionFact.provenanceText ?? candidate.rawText,
        confidence: correction.confidence,
      });
      for (const assertion of correctionAssertions) {
        pushAttributeSlotEdgeFromSemanticAssertion({
          assertion,
          subject: correctionSubject(correction, correctionAssertions),
          predicate: predicate ?? correctionFact.predicate,
          object: correction.nextValue.trim(),
          sourceRef: correctionSourceRef,
          supportText: semanticAssertionSupportText(candidate, semanticDraft!, assertion),
          confidence: correction.confidence ?? assertion.confidence,
        });
      }
    }
  }

  const procedureGuidanceFact = buildProcedureGuidanceFact(ctx, candidate);
  if (procedureGuidanceFact) {
    outputs.facts.push(procedureGuidanceFact);
  }

  const decision = getDecisionHint(candidate);
  if (
    decision &&
    shouldMaterializeDecisionSummary(candidate, decision, preference, relations, correction)
  ) {
    const guidance = buildDecisionGuidanceFacet(decision);
    outputs.facts.push(
      buildFact({
        ctx,
        candidate,
        subject: subjectUser(),
        predicate: "decision_summary",
        object: decision.summary,
        objectValueJson: guidance ? { guidance } : undefined,
      }),
    );
  }
  if (semanticDraft) {
    for (const assertion of semanticDraft.assertionDrafts) {
      if (
        !shouldMaterializeSemanticAssertionFact({
          candidate,
          assertion,
          preference,
          decision,
          relations,
          correction,
          workflowHints,
        })
      ) {
        continue;
      }
      for (const assertionFact of buildSemanticAssertionFacts(
        ctx,
        candidate,
        semanticDraft,
        assertion,
      )) {
        if (
          !outputs.facts.some(
            (entry) =>
              entry.predicate === assertionFact.predicate &&
              entry.canonicalSubject === assertionFact.canonicalSubject &&
              entry.canonicalObject === assertionFact.canonicalObject &&
              entry.sourceRef === assertionFact.sourceRef,
          )
        ) {
          outputs.facts.push(assertionFact);
        }
        pushAttributeSlotEdgeFromSemanticAssertion({
          assertion,
          subject: assertionFact.canonicalSubject,
          predicate: assertionFact.predicate,
          object: assertionFact.canonicalObject ?? "",
          sourceRef: assertionFact.sourceRef,
          supportText: assertionFact.provenanceText ?? candidate.rawText,
          confidence: assertion.confidence,
        });
      }
    }
  }

  for (const workflow of workflowHints) {
    const canonicalProjectCode =
      projectCodeFromStateKey(workflow.key) ||
      (typeof workflow.value.projectCode === "string"
        ? workflow.value.projectCode.trim()
        : undefined);
    const resolvedProjectCode = canonicalProjectCode
      ? resolveProjectReference(canonicalProjectCode, {
          currentProject: canonicalCurrentProject,
          knownProjects: knownProjectNames,
          allowDescriptorAlias: Boolean(canonicalCurrentProject || currentProjectProfile),
        })
      : undefined;
    const normalizedWorkflow =
      resolvedProjectCode && isProjectProfileStateKey(workflow.key)
        ? {
            ...workflow,
            key: projectProfileStateKey(resolvedProjectCode),
            value: {
              ...workflow.value,
              projectCode: resolvedProjectCode,
            },
          }
        : workflow;
    if (isProjectProfileStateKey(workflow.key)) {
      const projectCode =
        projectCodeFromStateKey(normalizedWorkflow.key) ??
        normalizedWorkflow.key.slice("project.".length);
      const baseProfile = projectProfileBase(normalizedWorkflow.key, projectCode);
      const sanitizedValue = sanitizeProjectProfileValue(normalizedWorkflow.value, baseProfile);
      const inferredNote = deriveProjectProfileNote({
        candidateText: candidate.rawText,
        projectCode,
        currentValue: sanitizedValue,
      });
      const mergedProfile = mergeProjectProfileValue(baseProfile, {
        ...sanitizedValue,
        ...(projectCode ? { projectCode } : {}),
        ...(inferredNote ? { note: inferredNote } : {}),
      });
      projectProfiles.set(
        normalizedWorkflow.key,
        preserveProjectProfilePaths({
          merged: mergedProfile,
          authoritative: projectProfiles.get(normalizedWorkflow.key),
          protectedPaths: projectProfileProtectedPaths.get(normalizedWorkflow.key) ?? [],
        }),
      );
    }
    const stateKind = resolveWorkflowStateKind(candidate, ctx, normalizedWorkflow);
    if (!stateKind) {
      continue;
    }
    if (isProjectProfileStateKey(normalizedWorkflow.key)) {
      continue;
    }
    const state = buildState(ctx, candidate, normalizedWorkflow, stateKind);
    if (stateKind === "session") {
      state.expiresAt = new Date(
        new Date(candidate.observedAt).getTime() + ctx.config.stateTtlHours * 60 * 60 * 1000,
      ).toISOString();
    }
    const existingIndex = outputs.states.findIndex((entry) => entry.key === state.key);
    if (existingIndex >= 0) {
      outputs.states[existingIndex] = state;
    } else {
      outputs.states.push(state);
    }
  }

  for (const [key, valueJson] of projectProfiles.entries()) {
    const workflow: MemoryCandidateWorkflowHint = {
      key,
      value: valueJson,
      stateKind: "durable",
    };
    const state = buildState(ctx, candidate, workflow, "durable");
    const existingIndex = outputs.states.findIndex((entry) => entry.key === state.key);
    if (existingIndex >= 0) {
      outputs.states[existingIndex] = {
        ...state,
        valueJson: mergeProjectProfileValue(
          outputs.states[existingIndex]?.valueJson,
          state.valueJson,
        ),
      };
    } else {
      outputs.states.push(state);
    }
  }

  if (shouldMaterializeObservedEvent(candidate, relations)) {
    outputs.events.push(buildEvent(ctx, candidate));
  }

  if (candidate.classification === "graph-worthy" && outputs.edges.length === 0) {
    for (const relation of relations) {
      pushEntity(relation.subject);
      pushEntity(relation.object);
    }
  }

  for (const state of outputs.states) {
    const stateCurrentnessMetadata = stateCurrentnessVectorMetadata(state, ctx.now);
    outputs.vectorDocs.push(
      buildVectorDoc({
        kind: "state",
        sourceId: state.key,
        scope: state.scope,
        agentId: state.agentId,
        text: formatStateVectorText(state),
        confidence: state.confidence,
        observedAt: state.updatedAt,
        metadataJson: {
          stateKey: state.key,
          stateKind: state.stateKind,
          ...stateCurrentnessMetadata,
          ...resourceVectorMetadata(state.valueJson),
          ...(state.expiresAt ? { expiresAt: state.expiresAt } : {}),
        },
        lineage: {
          canonicalKind: "state",
          canonicalId: state.key,
          sourceKind: "state",
          sourceId: state.key,
          sourceRef: state.sourceRef,
        },
      }),
    );
  }
  for (const fact of outputs.facts) {
    const currentnessHint =
      typeof fact.objectValueJson?.currentnessHint === "string" &&
      ["current", "historical", "compare", "unknown"].includes(fact.objectValueJson.currentnessHint)
        ? fact.objectValueJson.currentnessHint
        : "current";
    outputs.vectorDocs.push(
      buildVectorDoc({
        kind: "fact",
        sourceId: fact.factId,
        scope: fact.scope,
        agentId: fact.agentId,
        text: formatFactVectorText(fact),
        confidence: fact.confidence,
        observedAt: fact.updatedAt,
        metadataJson: {
          canonicalSubject: fact.canonicalSubject,
          predicate: fact.predicate,
          ...(fact.canonicalObject ? { canonicalObject: fact.canonicalObject } : {}),
          ...factVectorSupportMetadata(fact),
          ...resourceVectorMetadata(fact.objectValueJson),
          activeHint: true,
          supersededHint: false,
          currentnessHint,
        },
        lineage: {
          canonicalKind: "fact",
          canonicalId: fact.factId,
          sourceKind: "fact",
          sourceId: fact.factId,
          sourceRef: fact.sourceRef,
        },
      }),
    );
  }
  for (const event of outputs.events) {
    const structuredSummary =
      typeof event.metadataJson.memxStructuredSummary === "string" &&
      event.metadataJson.memxStructuredSummary.trim()
        ? event.metadataJson.memxStructuredSummary.trim()
        : event.text;
    const detailExcerpt =
      typeof event.metadataJson.memxRetrievalDetailExcerpt === "string" &&
      event.metadataJson.memxRetrievalDetailExcerpt.trim()
        ? event.metadataJson.memxRetrievalDetailExcerpt.trim()
        : undefined;
    outputs.vectorDocs.push(
      buildVectorDoc({
        kind: "event",
        sourceId: event.eventId,
        scope: event.scope,
        agentId: event.agentId,
        text: truncateText(
          [
            `${event.observedAt.slice(0, 10)} ${event.eventType}: ${structuredSummary}`,
            detailExcerpt ? `Detail: ${detailExcerpt}` : undefined,
          ]
            .filter(Boolean)
            .join(" | "),
          520,
        ),
        confidence: event.confidence,
        observedAt: event.observedAt,
        metadataJson: {
          eventType: event.eventType,
          ...(event.sessionKey ? { sessionKey: event.sessionKey } : {}),
          sourceKind: event.sourceKind,
          ...resourceVectorMetadata(event.metadataJson),
          currentnessHint: "historical",
        },
        lineage: {
          canonicalKind: "event",
          canonicalId: event.eventId,
          sourceKind: "event",
          sourceId: event.eventId,
          sourceRef: event.sourceRef,
        },
      }),
    );
  }
  for (const edge of outputs.edges) {
    if (outputs.vectorDocs.some((doc) => doc.docId === `edge:${edge.edgeId}`)) {
      continue;
    }
    outputs.vectorDocs.push(
      buildVectorDoc({
        kind: "edge",
        sourceId: edge.edgeId,
        scope: edge.scope,
        agentId: edge.agentId,
        text: `${edge.srcEntityId} ${edge.relType} ${edge.dstEntityId}`,
        confidence: edge.confidence,
        observedAt: edge.updatedAt,
        metadataJson: {
          relationType: edge.relType,
          rawPredicate: edge.rawRelationType ?? edge.relType,
          ...(edge.relationSlot ? { relationSlot: edge.relationSlot } : {}),
          sourceKind: edge.sourceKind ?? "extracted",
          entityId: edge.srcEntityId,
        },
        lineage: {
          canonicalKind: "graph_edge",
          canonicalId: edge.edgeId,
          sourceKind: "graph_edge",
          sourceId: edge.edgeId,
          sourceRef: edge.evidenceRef,
        },
      }),
    );
  }

  return outputs;
}

export function computeConfidence(candidate: ClassifiedCandidate): number {
  // Boolean-flag base (original approach, scaled down when policy scores are available)
  let boolBase = 0.45;
  if (candidate.source.kind === "user") boolBase += 0.18;
  if (candidate.source.kind === "tool") boolBase += 0.12;
  if (candidate.policy.explicitIntent) boolBase += 0.12;
  if (candidate.structuredHints?.relationHint) boolBase += 0.08;
  if (candidate.structuredHints?.decisionHint || candidate.structuredHints?.preferenceHint) {
    boolBase += 0.08;
  }

  // Policy numeric aggregate (salience 0.4, utility 0.35, stability 0.25)
  const p = candidate.policy;
  const hasPolicyScores =
    typeof p.salienceScore === "number" && typeof p.expectedFutureUtility === "number";
  if (!hasPolicyScores) {
    return clamp01(boolBase);
  }
  const policyScore =
    p.salienceScore * 0.4 +
    p.expectedFutureUtility * 0.35 +
    (typeof p.stabilityScore === "number" ? p.stabilityScore : 0) * 0.25;
  return clamp01(boolBase * 0.35 + policyScore * 0.65);
}
