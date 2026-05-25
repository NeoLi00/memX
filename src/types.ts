export const MEMORY_ROUTE_TYPES = [
  "workflow",
  "factual",
  "explanatory",
  "temporal",
  "mixed",
  "unknown",
] as const;
export const MEMORY_CORRECTION_TIMEFRAMES = ["current", "historical", "compare"] as const;
export const MEMORY_CORRECTION_TARGET_KINDS = [
  "state",
  "fact",
  "relation",
  "project_profile",
  "unknown",
] as const;
export const RECALL_QUERY_TIMEFRAMES = ["current", "historical", "compare", "timeless"] as const;
export const RECALL_QUERY_GRANULARITIES = ["summary", "exact_detail"] as const;
export const RECALL_QUERY_REFERENTIAL_MODES = ["anchored", "deictic"] as const;
export const RECALL_QUERY_EVIDENCE_NEEDS = [
  "workflow_context",
  "canonical_state",
  "factual_history",
  "event_history",
  "relation",
  "chunk",
] as const;
export const TURN_MODES = ["memory_qa", "workspace_task", "mixed"] as const;

export const RECALL_NEED_LEVELS = ["none", "background_only", "shallow", "full"] as const;
export const WORKING_PROJECTION_ROLES = [
  "user_style",
  "active_task",
  "active_blocker",
  "validated_strategy",
  "open_risk",
] as const;

export const MEMORY_ACTIONS = [
  "ignore",
  "session_state",
  "durable_state",
  "stable_fact",
  "episodic_event",
  "graph_relation",
] as const;

export const MEMORY_SIGNAL_TYPES = [
  "self_consistency",
  "retrieval_support",
  "future_usefulness",
  "outcome_feedback",
  "contradiction",
  "temporal_stability",
  "assistant_grounding",
  "demotion",
  "promotion",
  "correction",
  "repeated_use",
  "stale_decay",
] as const;

export const MEMORY_SIGNAL_TARGET_KINDS = [
  "state",
  "task",
  "fact",
  "event",
  "graph_edge",
  "chunk",
] as const;

export const MEMORY_BELIEF_STAGES = [
  "candidate",
  "probationary",
  "active",
  "decaying",
  "superseded",
  "quarantined",
] as const;

export const ABSTRACTION_CANDIDATE_TYPES = [
  "derived_state",
  "workflow_pattern",
  "concept_candidate",
  "graph_hypothesis",
  "outcome_hypothesis",
] as const;

export const ABSTRACTION_CANDIDATE_STAGES = MEMORY_BELIEF_STAGES;

export const STRATEGY_HYPOTHESIS_STAGES = [
  "candidate",
  "active",
  "superseded",
  "quarantined",
] as const;

export const MEMORY_BELIEF_KINDS = [...MEMORY_SIGNAL_TARGET_KINDS, "strategy"] as const;

export const GRAPH_RELATION_TYPES = [
  "depends_on",
  "blocks",
  "caused_by",
  "uses",
  "part_of",
  "owner_of",
  "supersedes",
  "contradicts",
  "resolved_by",
  "related_to",
  "reads",
] as const;

export const GRAPH_SUPPORT_RELATION_TYPES = [
  "supported_by",
  "derived_from",
  "updates",
  "targets",
] as const;

export const ENTITY_TYPES = [
  "person",
  "project",
  "tool",
  "service",
  "language",
  "framework",
  "concept",
  "organization",
  "unknown",
] as const;

export const GRAPH_NODE_KINDS = ["entity", "task", "state", "fact", "event", "outcome"] as const;

export const MEMORY_CLASSES = [
  "current-state",
  "stable-fact",
  "episodic-event",
  "graph-worthy",
  "ignore",
] as const;

export const MEMORY_SOURCE_KINDS = ["user", "assistant", "tool"] as const;
export const MEMORY_SCOPE_TEMPLATES = [
  "global",
  "agent:{agentId}",
  "session:{sessionKey}",
  "project:{project}",
] as const;
export const MEMORY_PII_MODES = ["off", "redact", "allow"] as const;
export const MEMORY_CONSENT_MODES = ["explicit", "implicit", "off"] as const;
export const MEMORY_EMBEDDING_PROVIDERS = [
  "off",
  "openai-compatible",
  "ollama",
  "sentence-transformers-local",
] as const;
export const MEMORY_LLM_PROVIDERS = [
  "openai-compatible",
  "anthropic",
  "google",
  "ollama",
] as const;

export type MemoryRouteType = (typeof MEMORY_ROUTE_TYPES)[number];
export type MemoryCorrectionTimeframe = (typeof MEMORY_CORRECTION_TIMEFRAMES)[number];
export type MemoryCorrectionTargetKind = (typeof MEMORY_CORRECTION_TARGET_KINDS)[number];
export type RecallQueryTimeframe = (typeof RECALL_QUERY_TIMEFRAMES)[number];
export type RecallQueryGranularity = (typeof RECALL_QUERY_GRANULARITIES)[number];
export type RecallQueryReferentialMode = (typeof RECALL_QUERY_REFERENTIAL_MODES)[number];
export type RecallQueryEvidenceNeed = (typeof RECALL_QUERY_EVIDENCE_NEEDS)[number];
export type TurnMode = (typeof TURN_MODES)[number];
export type RecallNeedLevel = (typeof RECALL_NEED_LEVELS)[number];
export type WorkingProjectionRole = (typeof WORKING_PROJECTION_ROLES)[number];
export type MemoryAction = (typeof MEMORY_ACTIONS)[number];
export type MemorySignalType = (typeof MEMORY_SIGNAL_TYPES)[number];
export type MemorySignalTargetKind = (typeof MEMORY_SIGNAL_TARGET_KINDS)[number];
export type MemoryBeliefStage = (typeof MEMORY_BELIEF_STAGES)[number];
export type AbstractionCandidateType = (typeof ABSTRACTION_CANDIDATE_TYPES)[number];
export type AbstractionCandidateStage = (typeof ABSTRACTION_CANDIDATE_STAGES)[number];
export type StrategyHypothesisStage = (typeof STRATEGY_HYPOTHESIS_STAGES)[number];
export type MemoryBeliefKind = (typeof MEMORY_BELIEF_KINDS)[number];
export type GraphRelationType = (typeof GRAPH_RELATION_TYPES)[number];
export type GraphSupportRelationType = (typeof GRAPH_SUPPORT_RELATION_TYPES)[number];
export type GraphTraversalRelationType = GraphRelationType | GraphSupportRelationType;
export type EntityType = (typeof ENTITY_TYPES)[number];
export type GraphNodeKind = (typeof GRAPH_NODE_KINDS)[number];
export type MemoryClass = (typeof MEMORY_CLASSES)[number];
export type MemorySourceKind = (typeof MEMORY_SOURCE_KINDS)[number];
export type MemoryPiiMode = (typeof MEMORY_PII_MODES)[number];
export type MemoryConsentMode = (typeof MEMORY_CONSENT_MODES)[number];
export type MemoryEmbeddingProvider = (typeof MEMORY_EMBEDDING_PROVIDERS)[number];
export type MemoryLlmProvider = (typeof MEMORY_LLM_PROVIDERS)[number];
export type MemoryPrimaryRouteType = Exclude<MemoryRouteType, "mixed" | "unknown">;
export type MemoryWorkflowStateKind = "session" | "durable";
export const MEMX_STATE_LIFECYCLE_KINDS = [
  "durable_profile",
  "session_working",
  "transient_blocker",
  "transient_next_step",
  "derived_maintenance",
  "task_checkpoint",
] as const;
export type MemxStateLifecycleKind = (typeof MEMX_STATE_LIFECYCLE_KINDS)[number];

export type StateCurrentness = {
  lifecycleKind: MemxStateLifecycleKind;
  currentnessScore: number;
  durable: boolean;
  observedAt?: string;
  validFrom?: string;
  expiresAt?: string;
  supersededBy?: string;
  sourceRef?: string;
  supportRefs: string[];
  sessionKey?: string;
  taskId?: string;
  hardExclusions: string[];
  softPenalties: string[];
  answerEligibleByDefault: boolean;
};

export type ScopeVars = {
  agentId?: string;
  sessionKey?: string;
  project?: string;
  workspace?: string;
};

export type EmbeddingConfig = {
  provider: MemoryEmbeddingProvider;
  baseURL?: string;
  apiKey?: string;
  model?: string;
  dimensions?: number;
  headers?: Record<string, string>;
  ollamaBaseURL?: string;
  ollamaModel?: string;
  localPythonBin?: string;
  localCacheDir?: string;
  localDevice?: "auto" | "cpu" | "mps" | "cuda";
};

export type AdvancedMemoryConfig = {
  llmClassifierEnabled: boolean;
  llmClassifierModel?: string;
  llmProvider?: MemoryLlmProvider;
  llmBaseURL?: string;
  llmApiKey?: string | { source: "env"; provider?: string; id: string };
  llmHeaders?: Record<string, string>;
  enableMaintenanceJobs: boolean;
  maintenanceTriggerMode: "batched" | "per_turn";
  maintenanceBatchTurns: number;
  maintenanceIdleFlushMinutes: number;
  enableGraphPromotion: boolean;
  enableFactPromotion: boolean;
  enableTelemetryAudit: boolean;
  enableExplicitRecallTool: boolean;
  suggestExplicitRecallTool: boolean;
  enableCompatibilityMemoryTools: boolean;
  enableTurnScheduler: boolean;
  chunkDedupThreshold: number;
  taskIdleTimeoutMinutes: number;
  recallChunkBudget: number;
  recallTotalObjectBudget: number;
  recallBackgroundCharReserve: number;
  recallPromptBudgetFloor: number;
  recallObjectiveMinWeight: number;
  recallObjectiveOverflowRatio: number;
  recallProbeWorkflowStrongThreshold: number;
  recallProbeWorkflowContinuationThreshold: number;
  recallProbeFactualStrongThreshold: number;
  recallProbeFactualShortQueryThreshold: number;
  recallProbeHybridStrongThreshold: number;
  recallProbeHybridModerateThreshold: number;
  recallProbeEscalateThreshold: number;
  recallProbeContinuationEscalateThreshold: number;
  enableTurnSemanticCompiler: boolean;
  enableQueryCompiler: boolean;
  queryCompilerHotPathTimeoutMs: number;
  enableEmbeddingCandidates: boolean;
  enableEmbeddingClustering: boolean;
  enableHotPathChunkSummaryLlm: boolean;
  enableHotPathTaskSummaryLlm: boolean;
  candidateSurfaceBudgets: CandidateSurfaceBudgetConfig;
};

export type CandidateSurfaceBudgetConfig = {
  state: number;
  fact: number;
  event: number;
  task: number;
  chunk: number;
  graph: number;
  entityAlias: number;
};

export type MemoryPluginConfig = {
  enabled: boolean;
  dbPath: string;
  autoCapture: boolean;
  autoRecall: boolean;
  reflectionEnabled: boolean;
  consentMode: MemoryConsentMode;
  maxInjectedChars: number;
  captureMaxChars: number;
  reflectionMaxChars: number;
  reflectionMaxItems: number;
  piiMode: MemoryPiiMode;
  defaultScope: string;
  allowedScopes: string[];
  minSalienceDurable: number;
  minSalienceSession: number;
  minUtilityForGraph: number;
  maxSensitivityAllowed: number;
  stateTtlHours: number;
  episodicDedupWindowDays: number;
  graphMaxHops: number;
  maxGraphNodes: number;
  maxGraphEdges: number;
  embedding: EmbeddingConfig;
  advanced: AdvancedMemoryConfig;
};

export type MemoryCandidatePreferenceHint = {
  predicate: string;
  object: string;
  guidance?: MemoryGuidanceFacet | null;
  confidence?: number;
  reason?: string;
};

export type MemoryGuidanceType =
  | "language"
  | "style"
  | "charset"
  | "output_order"
  | "generic_preference";

export type MemoryGuidanceFacet = {
  guidanceType: MemoryGuidanceType;
  guidanceText: string;
  confidence?: number;
  reason?: string;
};

export type MemoryCandidateWorkflowHint = {
  key: string;
  value: Record<string, unknown>;
  stateKind?: MemoryWorkflowStateKind;
  confidence?: number;
  reason?: string;
};

export type MemoryResourceOwnershipStatus = "owned" | "recently_acquired" | "uses" | "considering";

export type MemoryResourceSemanticStatus = "observed" | "inferred_affordance";

export type MemoryResourceAssertionHint = {
  owner: "user" | string;
  resource: string;
  ownershipStatus: MemoryResourceOwnershipStatus;
  resourceType?: string;
  domains?: string[];
  affordances?: string[];
  sourceRef: string;
  supportText: string;
  confidence: number;
  semanticStatus: MemoryResourceSemanticStatus;
};

export type MemoryAdviceSignalSemanticStatus = "observed" | "assistant_suggested";

export type MemoryAdviceSignalHint = {
  problemContext?: string;
  userResources?: string[];
  assistantRecommendation?: string;
  domains?: string[];
  sourceRefs: string[];
  supportText?: string;
  confidence: number;
  semanticStatus: MemoryAdviceSignalSemanticStatus;
};

export type MemoryCandidateRelationHint = {
  subject: string;
  predicate: GraphRelationType;
  object: string;
  sourceRef?: string;
  polarity?: "affirmed" | "negated";
  rawPredicate?: string;
  relationSlot?: string;
  confidence?: number;
  reason?: string;
};

export type MemoryCandidateDecisionHint = {
  summary: string;
  confidence?: number;
  reason?: string;
};

export type MemoryCandidateCorrectionHint = {
  timeframe: MemoryCorrectionTimeframe;
  targetKind: MemoryCorrectionTargetKind;
  priorValue?: string;
  nextValue?: string;
  canonicalKey?: string;
  predicate?: string;
  confidence?: number;
  reason?: string;
};

export type MemoryCandidateStructuredHints = {
  entities?: Array<{ name: string; type?: string }>;
  timeHints?: string[];
  preferenceHint?: boolean;
  decisionHint?: boolean;
  relationHint?: boolean;
  taskStateHint?: boolean;
  correctionHint?: boolean;
  semanticFamilies?: TurnSemanticAssertionFamilyHint[];
  semanticTimeframes?: TurnSemanticTimeframeHint[];
  slotHints?: string[];
  preference?: MemoryCandidatePreferenceHint;
  workflow?: MemoryCandidateWorkflowHint;
  workflows?: MemoryCandidateWorkflowHint[];
  relation?: MemoryCandidateRelationHint;
  relations?: MemoryCandidateRelationHint[];
  resourceAssertions?: MemoryResourceAssertionHint[];
  adviceSignals?: MemoryAdviceSignalHint[];
  decision?: MemoryCandidateDecisionHint;
  correction?: MemoryCandidateCorrectionHint;
  semanticDraft?: MemoryCandidateSemanticDraft;
  materializationHint?: MemoryCandidateMaterializationHint;
};

export type MemoryCandidate = {
  candidateId: string;
  source: {
    kind: MemorySourceKind;
    messageId?: string;
    toolName?: string;
    sessionKey?: string;
    runId?: string;
  };
  observedAt: string;
  rawText: string;
  normalizedText?: string;
  eventType?: string;
  structuredHints?: MemoryCandidateStructuredHints;
  metadata?: Record<string, unknown>;
};

export type MemoryCanonicalKind =
  | "state"
  | "fact"
  | "event"
  | "entity"
  | "graph_edge"
  | "task"
  | "chunk";

export type MemoryLineageSourceKind =
  | "chunk"
  | "task"
  | "state"
  | "fact"
  | "event"
  | "entity_alias"
  | "graph_edge"
  | "vector_doc"
  | "alternate";

export type LineageRef = {
  canonicalKind?: MemoryCanonicalKind;
  canonicalId?: string;
  sourceKind: MemoryLineageSourceKind;
  sourceId: string;
  sourceRef?: string;
  materializedEpoch?: number;
};

export type CompilerExecutionMode =
  | "llm"
  | "deterministic"
  | "fallback"
  | "deferred"
  | "degraded"
  | "disabled"
  | "semantic-compiler-authoritative";

export type MemoryCallProvenance = "deterministic" | "llm" | "embedding" | "hybrid";
export type MaintenanceAuthoritySource =
  | "deterministic_aggregated"
  | "llm_confirmed"
  | "llm_upgrade"
  | "embedding_clustered";

export type MaintenanceSemanticSource =
  | "upstream_structured"
  | "embedding_clustered"
  | "llm_upgrade"
  | "deterministic_lifecycle"
  | "lexical_fallback";
export type MemoryLlmCallStage =
  | "query_hot_path"
  | "write_hot_path"
  | "post_answer_writeback"
  | "maintenance_async";

export type MemoryLlmBudgetCall = {
  label: string;
  stage: MemoryLlmCallStage;
  provenance: MemoryCallProvenance;
  mode?: string;
  provider?: string;
  model?: string;
  detail?: string;
  promptChars?: number;
  responseChars?: number;
  estimatedPromptTokens?: number;
  estimatedCompletionTokens?: number;
  estimatedTotalTokens?: number;
  elapsedMs?: number;
  at: string;
};

export type MemoryLlmBudgetAudit = {
  calls: MemoryLlmBudgetCall[];
  hotPathLlmCallCount: number;
  writeHotPathLlmCallCount: number;
  queryHotPathLlmCallCount: number;
  postAnswerWritebackLlmCallCount: number;
  maintenanceLlmCallCount: number;
};

export type CompilerProvenance = {
  source: "llm" | "deterministic" | "hybrid";
  mode: CompilerExecutionMode;
  promptVersion?: string;
  model?: string;
  backendVersion?: string;
  reasons?: string[];
};

export type TurnSemanticAssertionFamilyHint =
  | "workflow"
  | "preference"
  | "fact_like"
  | "event_like"
  | "relation_like"
  | "strategy_like";

export type TurnSemanticTimeframeHint = "current" | "historical" | "compare" | "timeless";

export type TurnSemanticChunkDraft = {
  sourceRef: string;
  summary: string;
  lineage: LineageRef;
};

export type TurnSemanticTaskProposal = {
  decision: "continue" | "resume" | "new" | "none";
  targetTaskId?: string;
  confidence: number;
  summary?: string;
  summaryConfidence?: number;
  reason?: string;
  lineage?: LineageRef;
};

export type TurnSemanticReferenceContextMessage = {
  role: TurnCaptureRole;
  turnId: string;
  sourceRef: string;
  summary?: string;
  textExcerpt?: string;
};

export type TurnSemanticReferenceContextTurn = {
  turnId: string;
  messages: TurnSemanticReferenceContextMessage[];
};

export type TurnSemanticReferenceContext = {
  purpose: "deictic_reference_resolution";
  maxTurns: number;
  turns: TurnSemanticReferenceContextTurn[];
};

export type TurnSemanticAssertionDraft = {
  draftId: string;
  sourceRef: string;
  familyHint: TurnSemanticAssertionFamilyHint;
  timeframeHint: TurnSemanticTimeframeHint;
  entityHints?: Array<{ name: string; type?: string }>;
  slotHints?: string[];
  valueHint?: string;
  supportSpans?: Array<{ text: string; start?: number; end?: number }>;
  confidence?: number;
  lineage: LineageRef;
};

export type TurnSemanticCorrectionDraft = {
  sourceRef: string;
  correction: MemoryCandidateCorrectionHint;
  supportSpans?: Array<{ text: string; start?: number; end?: number }>;
  confidence?: number;
  lineage: LineageRef;
};

export type TurnSemanticRelationDraft = {
  sourceRef: string;
  relation: MemoryCandidateRelationHint;
  supportSpans?: Array<{ text: string; start?: number; end?: number }>;
  confidence?: number;
  lineage: LineageRef;
};

export type TurnSemanticFrame = {
  sourceRefs: string[];
  referenceContext?: TurnSemanticReferenceContext;
  chunkDrafts: TurnSemanticChunkDraft[];
  taskProposal?: TurnSemanticTaskProposal;
  assertionDrafts: TurnSemanticAssertionDraft[];
  correctionDrafts: TurnSemanticCorrectionDraft[];
  relationDrafts?: TurnSemanticRelationDraft[];
  resourceAssertions?: MemoryResourceAssertionHint[];
  adviceSignals?: MemoryAdviceSignalHint[];
  supportSpans: Array<{ sourceRef: string; text: string; start?: number; end?: number }>;
  compilerProvenance: CompilerProvenance;
};

export type MemoryCandidateSemanticDraft = {
  sourceRef: string;
  assertionDrafts: TurnSemanticAssertionDraft[];
  correctionDrafts: TurnSemanticCorrectionDraft[];
  relationDrafts?: TurnSemanticRelationDraft[];
  resourceAssertions?: MemoryResourceAssertionHint[];
  adviceSignals?: MemoryAdviceSignalHint[];
  supportSpans: Array<{ sourceRef: string; text: string; start?: number; end?: number }>;
  taskProposal?: TurnSemanticTaskProposal;
  compilerProvenance?: CompilerProvenance;
};

export type MemoryCandidateMaterializationHint = {
  sourceRef: string;
  primaryFamily?: TurnSemanticAssertionFamilyHint;
  timeframeHint?: TurnSemanticTimeframeHint;
  preferredAction?: MemoryAction;
  preferDurableState?: boolean;
  preferEvent?: boolean;
  preferGuidanceFact?: boolean;
  replacementMode?: "supersede_fact" | "none";
  reasons?: string[];
};

export type MemoryPolicyDecision = {
  salienceScore: number;
  expectedFutureUtility: number;
  sensitivityScore: number;
  stabilityScore: number;
  action: MemoryAction;
  reasons: string[];
  explicitIntent: boolean;
  captureAuthorized: boolean;
};

export type ClassifiedCandidate = MemoryCandidate & {
  classification: MemoryClass;
  policy: MemoryPolicyDecision;
  confidence: number;
  scope: string;
};

export type NormalizedEntity = {
  entityId: string;
  canonicalName: string;
  entityType: EntityType;
  normalizedName: string;
  aliases: string[];
  confidence: number;
};

export type EntityResolutionMethod =
  | "exact"
  | "alias"
  | "project_identity"
  | "identity_anchor"
  | "cooccurrence"
  | "embedding_candidate"
  | "llm_candidate"
  | "identity_link"
  | "new_entity"
  | "uncertain";

export type EntityIdentityLinkType = "same_as" | "duplicate_of" | "possible_same_as" | "supersedes";

export type EntityMentionSemanticRole =
  | "subject"
  | "object"
  | "resource"
  | "project"
  | "person"
  | "query"
  | "support";

export type EntityMention = {
  mentionId: string;
  agentId: string;
  scope: string;
  rawText: string;
  normalizedText: string;
  proposedType: EntityType;
  semanticRole: EntityMentionSemanticRole;
  sourceRef: string;
  supportText: string;
  sessionKey?: string;
  turnIndex?: number;
  observedAt: string;
  resolvedEntityId?: string;
  resolutionMethod?: EntityResolutionMethod;
  confidence: number;
  candidateIds: string[];
  blockers: string[];
  metadataJson: Record<string, unknown>;
};

export type EntityResolutionResult = {
  mention: EntityMention;
  entity: NormalizedEntity;
  method: EntityResolutionMethod;
  confidence: number;
  candidateEntityIds: string[];
  blockers: string[];
  createdEntity: boolean;
  identityLinks?: Array<{
    srcEntityId: string;
    dstEntityId: string;
    linkType: EntityIdentityLinkType;
    confidence: number;
    evidenceRef: string;
    status: "active" | "rejected";
    metadataJson?: Record<string, unknown>;
  }>;
};

export type EntityProfileVectorMetadata = {
  canonicalName: string;
  aliases: string[];
  entityType: EntityType;
  supportRefs: string[];
  relationNeighborIds: string[];
  updatedAt: string;
  confidence: number;
};

export type EntityResolutionCandidate = {
  entity: NormalizedEntity;
  score: number;
  exactNameScore: number;
  aliasScore: number;
  embeddingScore: number;
  typeCompatibility: number;
  scopeSessionTaskFit: number;
  cooccurrenceOverlap: number;
  graphNeighborhoodOverlap: number;
  recency: number;
  contradictionPenalty: number;
  source: "profile_vector" | "identity_anchor" | "name_search" | "mention_history";
  metadataJson?: Record<string, unknown>;
};

export type EntityDisambiguationDecision = {
  matchedEntityId?: string;
  decision: "match" | "no_match" | "uncertain";
  confidence: number;
  rationale: string;
};

export type NormalizedFact = {
  factId: string;
  canonicalSubject: string;
  predicate: string;
  canonicalObject?: string;
  objectValueJson?: Record<string, unknown>;
  scope: string;
  agentId: string;
  confidence: number;
  status: "active" | "superseded" | "deleted" | "uncertain";
  validFrom?: string;
  validTo?: string;
  createdAt: string;
  updatedAt: string;
  materializedEpoch?: number;
  sourceRef: string;
  provenanceText: string;
};

export type NormalizedState = {
  key: string;
  valueJson: Record<string, unknown>;
  scope: string;
  agentId: string;
  stateKind: "session" | "durable";
  confidence: number;
  sourceRef: string;
  updatedAt: string;
  expiresAt?: string;
  materializedEpoch?: number;
};

export type NormalizedEvent = {
  eventId: string;
  agentId: string;
  scope: string;
  eventType: string;
  text: string;
  normalizedText: string;
  observedAt: string;
  validFrom?: string;
  validTo?: string;
  sourceKind: MemorySourceKind;
  sourceRef: string;
  sessionKey?: string;
  toolName?: string;
  confidence: number;
  metadataJson: Record<string, unknown>;
  materializedEpoch?: number;
};

export type NormalizedGraphEdge = {
  edgeId: string;
  srcEntityId: string;
  relType: GraphRelationType;
  dstEntityId: string;
  relationSlot?: string;
  scope: string;
  agentId: string;
  confidence: number;
  validFrom?: string;
  validTo?: string;
  evidenceRef: string;
  rawRelationType?: string;
  sourceKind?: "extracted" | "synthesized";
  createdAt: string;
  updatedAt: string;
  materializedEpoch?: number;
  metadataJson?: Record<string, unknown>;
};

export type VectorDocRecord = {
  docId: string;
  docKind: "fact" | "event" | "edge" | "state" | "entity_profile";
  sourceId: string;
  scope: string;
  agentId: string;
  text: string;
  metadataJson: Record<string, unknown>;
  createdAt: string;
  updatedAt: string;
  materializedEpoch?: number;
};

export type SearchHit = {
  docId: string;
  score: number;
  text: string;
  metadata: Record<string, unknown>;
  backend: "fts" | "embedding" | "hybrid" | "lexical" | "repo";
};

export type RetrievalSearchParams = {
  agentId: string;
  scopes: string[];
  limit: number;
  query: string;
  readEpoch?: number;
  docKinds?: VectorDocRecord["docKind"][];
  docTypes?: string[];
};

export type RetrievalBackend = {
  upsertDocs: (docs: VectorDocRecord[]) => void;
  deleteDocs: (docIds: string[]) => void;
  keywordSearch: (params: RetrievalSearchParams) => SearchHit[];
  similaritySearch: (params: RetrievalSearchParams) => Promise<SearchHit[]>;
  hybridSearch: (params: RetrievalSearchParams) => Promise<SearchHit[]>;
  embedTextsBatch: (texts: string[], mode?: "query" | "passage") => Promise<number[][]>;
  flushPendingUpserts?: () => Promise<void>;
  close?: () => Promise<void>;
};

export type RouteDecision = {
  routeType: MemoryRouteType;
  routeConfidence: number;
  reasons: string[];
};

export type RecallRouteEvaluation = {
  routeType: MemoryPrimaryRouteType;
  priorWeight: number;
  evidenceSupport: number;
  evidenceSufficient: boolean;
  candidateCount: number;
  topRole: "state" | "task" | "fact" | "event" | "graph" | "chunk" | "alternate" | null;
  graphRoleShare: number;
  factRoleShare: number;
  finalScore: number;
  reason: string;
};

export type RecallObjectiveBudget = {
  routeType: MemoryPrimaryRouteType;
  weight: number;
  rawScore: number;
  activated: boolean;
  minObjects: number;
  objectBudget: number;
  minPromptChars: number;
  promptChars: number;
  reasons: string[];
};

export type RecallBudgetPlan = {
  routeDecision: RouteDecision;
  focusedQueries: Record<MemoryPrimaryRouteType, string>;
  routeEvaluations: RecallRouteEvaluation[];
  objectiveBudgets: Record<MemoryPrimaryRouteType, RecallObjectiveBudget>;
  totalObjectBudget: number;
  totalPromptChars: number;
  reservedBackgroundChars: number;
  globalOverflowObjects: number;
};

export type RouteEvidenceCandidate = {
  index: number;
  summary: string;
  role: "state" | "task" | "fact" | "event" | "graph" | "chunk" | "alternate";
  score: number;
  confidence?: number;
  observedAt?: string;
};

export type RoutePriorDecision = {
  primaryRoute: MemoryRouteType;
  secondaryRoutes: MemoryPrimaryRouteType[];
  confidence: number;
  focusedQueries: Partial<Record<MemoryPrimaryRouteType, string>>;
  reason: string;
  judgmentMode?: "llm" | "degraded" | "disabled";
};

export type RouteEvidenceDecision = {
  relevant: number[];
  sufficient: boolean;
  support: number;
  reason: string;
  judgmentMode?: "llm" | "degraded" | "disabled" | "score-driven";
};

export type MemoryRecallPlan = {
  shouldRecall: boolean;
  focusedQuery: string;
  reason: string;
  routeHint?: MemoryRouteType;
  judgmentMode?: "llm" | "degraded" | "disabled";
};

export type RecallQueryShape = {
  timeframe: RecallQueryTimeframe;
  granularity: RecallQueryGranularity;
  referentialMode: RecallQueryReferentialMode;
  evidenceNeed: RecallQueryEvidenceNeed;
};

export type AnswerGranularity = "summary" | "detail";
export type EvidenceFidelity = "low" | "medium" | "high";
export type QueryAnswerMode =
  | "single_fact"
  | "attribute_lookup"
  | "count_aggregate"
  | "multi_evidence";
export type QueryEvidenceCoverage = {
  requiredAnchors: string[];
  optionalAnchors: string[];
  minProtectedItems: number;
  maxProtectedItems: number;
};
export type QueryEntityRole = "subject" | "object" | "context" | "resource";
export type QueryEntityHint = {
  name: string;
  type?: EntityType;
  role?: QueryEntityRole;
};
export type QuerySuppressedEntityHint = {
  name: string;
  type?: EntityType;
  reason?: string;
};
export type CandidateSurface =
  | "state"
  | "fact"
  | "event"
  | "task"
  | "chunk"
  | "snippet"
  | "graph"
  | "entity_alias";

export type QueryEvidenceGoal = {
  goal: string;
  positiveQueries: string[];
  negativeHints?: string[];
  focusAnchors: string[];
  preferredSurfaces: CandidateSurface[];
  fidelity: EvidenceFidelity;
};

export type EvidencePlanLayer =
  | CandidateSurface
  | "control"
  | "strategy"
  | "abstraction"
  | "belief"
  | "snippet";

export type QueryEvidenceSlot = {
  id: string;
  role?:
    | "query_context"
    | "answer_evidence"
    | "answer_value"
    | "answer_event"
    | "time_constraint"
    | "user_resource"
    | "prior_advice"
    | "supporting_context";
  requiredRole?:
    | "query_context"
    | "user_resource"
    | "prior_advice"
    | "answer_value"
    | "answer_event"
    | "time_constraint";
  description: string;
  subjectHints: string[];
  relationHints?: string[];
  capabilityQueries?: string[];
  negativeHints?: string[];
  requiredFields: string[];
  preferredLayers: EvidencePlanLayer[];
  fallbackLayers: EvidencePlanLayer[];
  minEvidence: number;
};

export type QueryEvidenceOperationType =
  | "return_value"
  | "aggregate"
  | "derive"
  | "compare"
  | "relate"
  | "tailor_advice";

export type QueryEvidenceOperation = {
  type: QueryEvidenceOperationType;
  description: string;
};

export type QueryEvidencePlan = {
  slots: QueryEvidenceSlot[];
  operation: QueryEvidenceOperation;
};

export type QuerySemanticBridge = {
  bridgeId: string;
  sourceConcept: string;
  role: NonNullable<QueryEvidenceSlot["requiredRole"]>;
  evidenceShape:
    | "event"
    | "attribute_value"
    | "resource_affordance"
    | "query_context"
    | "time_constraint"
    | "aggregate_item"
    | "causal_explanation"
    | "validation_evidence"
    | "status_answer"
    | "decision_value"
    | "availability_statement";
  retrievalQueries: string[];
  positiveSignals: string[];
  negativeSignals?: string[];
  preferredLayers: EvidencePlanLayer[];
  hypothesisOnly: true;
};

export type QuerySemanticBridgeMatch = {
  bridgeId: string;
  sourceConcept: string;
  role: QuerySemanticBridge["role"];
  evidenceShape: QuerySemanticBridge["evidenceShape"];
  score: number;
  matchedQuery: string;
  positiveSignalScore: number;
  negativeSignalScore: number;
};

export type QueryCompileResult = {
  queryText: string;
  shouldRecall: boolean;
  focusedQuery: string;
  queryEntities: QueryEntityHint[];
  suppressedEntities: QuerySuppressedEntityHint[];
  queryShape: RecallQueryShape;
  primaryRoute?: MemoryPrimaryRouteType;
  answerGranularity: AnswerGranularity;
  evidenceFidelity: EvidenceFidelity;
  routeWeights: Partial<Record<MemoryPrimaryRouteType, number>>;
  anchors: string[];
  candidateSurfaces: CandidateSurface[];
  evidenceGoals: QueryEvidenceGoal[];
  evidencePlan?: QueryEvidencePlan;
  semanticBridges?: QuerySemanticBridge[];
  answerMode?: QueryAnswerMode;
  evidenceCoverage?: QueryEvidenceCoverage;
  detailNeedScore: number;
  supportNeed: number;
  ambiguityLevel: number;
  turnMode: TurnMode;
  compilerProvenance: CompilerProvenance;
};

export type RecallProbeSignals = {
  workflowContinuity: number;
  taskAssociation: number;
  stateAssociation: number;
  contextAssociation: number;
  workflowSimilarity: number;
  factualSimilarity: number;
  temporalSimilarity: number;
  explanatorySimilarity: number;
  topProbeScore: number;
  topProbeDocType?: string;
  backgroundTaskCount: number;
  backgroundStateCount: number;
  factualCandidateCount: number;
  hybridHitCount: number;
};

export type RecallProbeThresholds = {
  workflowStrong: number;
  workflowContinuation: number;
  factualStrong: number;
  factualShortQuery: number;
  hybridStrong: number;
  hybridModerate: number;
  escalate: number;
  continuationEscalate: number;
};

export type RecallProbeDecision = {
  shouldEscalate: boolean;
  probeScore: number;
  reasons: string[];
  focusedQuery: string;
  hintedRoute?: MemoryRouteType;
  signals: RecallProbeSignals;
  thresholds: RecallProbeThresholds;
};

export type RecallControllerSignals = {
  routeSupport: number;
  workflowContinuity: number;
  backgroundContextAvailable: boolean;
  lowSupportSurface: boolean;
  backgroundGuidanceCount: number;
  backgroundStrategyCount: number;
  backgroundTaskCount: number;
  backgroundStateCount: number;
};

export type RecallControllerTrace = {
  needLevel: RecallNeedLevel;
  routeHint?: MemoryRouteType;
  queryShape?: RecallQueryShape;
  routeWeights?: Partial<Record<MemoryPrimaryRouteType, number>>;
  shouldUseBackground: boolean;
  shouldUseShallow: boolean;
  shouldUseFull: boolean;
  legacyTrigger: "plan" | "probe" | "both" | "none";
  legacyFullRecall: boolean;
  divergence?: "legacy_over_recall" | "legacy_under_recall";
  reasons: string[];
  signals: RecallControllerSignals;
};

export type ShallowRecallRouteSummary = {
  routeType: MemoryPrimaryRouteType;
  support: number;
  candidateCount: number;
  projectionSupport: number;
  freshness: number;
  contradictionPressure: number;
  grounding: number;
  topKind?: MemoryObjectKind;
  topObjectId?: string;
  topText?: string;
  topStateKey?: string;
  topFactSubject?: string;
  topFactPredicate?: string;
  topFactObject?: string;
  topGraphNodeNames?: string[];
};

export type ShallowRecallResult = {
  searchQuery: string;
  routeHint?: MemoryRouteType;
  topSupport: number;
  hybridHitCount: number;
  projectionRoles: WorkingProjectionRole[];
  routeSummaries: Record<MemoryPrimaryRouteType, ShallowRecallRouteSummary>;
  reasons: string[];
};

export type RecallQualityGateDecisionKind =
  | "accept_full"
  | "downgrade_background"
  | "reroute"
  | "retry_shallow";

export type RecallQualityGateMetrics = {
  supportDensity: number;
  routeAgreement: number;
  contradictionPressure: number;
  freshness: number;
  grounding: number;
  irrelevantSpill: number;
};

export type RecallQualityGateDecision = {
  decision: RecallQualityGateDecisionKind;
  routeHint?: MemoryRouteType;
  focusedQuery: string;
  confidence: number;
  reasons: string[];
  metrics: RecallQualityGateMetrics;
};

export type EvidenceRow = {
  id: string;
  text: string;
  score: number;
  confidence?: number;
  sourceRef?: string;
  observedAt?: string;
  scope: string;
  provenance?: string;
  lineage?: LineageRef;
};

export type PromptEvidenceCandidate = {
  id: string;
  surface: "fact" | "event" | "chunk" | "snippet";
  text: string;
  rawText?: string;
  scoringText?: string;
  metadata?: Record<string, unknown>;
  sourceRef?: string;
  mergedSourceRefs?: string[];
  observedAt?: string;
  excerptAnchors?: string[];
  lineage?: LineageRef;
  priority: number;
  goalScore: number;
  semanticScore?: number;
  coverage?: {
    requiredHits: string[];
    missingRequired: string[];
    coverageScore: number;
    answerMode: QueryAnswerMode;
  };
  slotCoverage?: Array<{
    slotId: string;
    requiredHits: string[];
    missingRequired: string[];
    coverageScore: number;
    filled: boolean;
  }>;
  filledSlotIds?: string[];
  injectionScore?: number;
  scoreBreakdown?: Record<string, number | string | boolean>;
  bridgeMatches?: QuerySemanticBridgeMatch[];
  slotEvidenceRole?:
    | "query_context"
    | "user_resource"
    | "prior_advice"
    | "answer_value"
    | "answer_event"
    | "time_constraint";
  packetId?: string;
  eligibility?: {
    eligible: boolean;
    role: "answer" | "context" | "resource" | "support";
    blockers: string[];
  };
  grade?: {
    retrievalScore: number;
    answerScore: number;
    contextBindingScore: number;
    temporalFitScore?: number;
    slotCoverageScore: number;
    authorityScore: number;
    softPenaltyScore?: number;
    finalScore: number;
  };
  blockedBy?: string[];
  softPenalties?: string[];
  hardExclusions?: string[];
  source: "candidate" | "selected" | "support_ref" | "projected" | "fallback";
  role: "protected" | "support" | "alternate";
  injected?: boolean;
  selectionReason?: string;
  protectionReason?: string;
  dropReason?: string;
};

export type EvidenceUnitRole =
  | "answer_value"
  | "answer_event"
  | "query_context"
  | "time_constraint"
  | "user_resource"
  | "prior_advice"
  | "support";

export type EvidenceUnitOrigin =
  | "raw_chunk"
  | "canonical_fact"
  | "event"
  | "state"
  | "graph"
  | "belief"
  | "strategy"
  | "derived_summary"
  | "snippet";

export type SourceRefKind =
  | "turn"
  | "chunk"
  | "event"
  | "fact"
  | "state"
  | "graph_edge"
  | "entity"
  | "belief"
  | "strategy"
  | "abstraction_candidate"
  | "task"
  | "query"
  | "prompt_line";

export type NormalizedSourceRef = {
  raw: string;
  kind: SourceRefKind;
  id: string;
  sessionKey?: string;
  turnIndex?: number;
  parentRefs: string[];
};

export type EvidenceUnit = {
  unitId: string;
  surfaceRefs: string[];
  sourceRefs: string[];
  normalizedSourceRefs?: NormalizedSourceRef[];
  supportRefs?: string[];
  normalizedSupportRefs?: NormalizedSourceRef[];
  derivedFromRefs?: string[];
  normalizedDerivedFromRefs?: NormalizedSourceRef[];
  neighborRefs?: string[];
  normalizedNeighborRefs?: NormalizedSourceRef[];
  sessionId?: string;
  turnIndex?: number;
  authorRole?: "user" | "assistant" | "tool" | "memory" | "unknown";
  observedAt?: string;
  rawText: string;
  displayText: string;
  roles: EvidenceUnitRole[];
  origin: EvidenceUnitOrigin;
};

export type EvidenceMatchType = "subject" | "relation" | "capability" | "context" | "answer_source";

export type LayerCandidateHit = {
  id: string;
  layer: EvidencePlanLayer;
  text: string;
  sourceRef?: string;
  sourceRefs?: string[];
  observedAt?: string;
  score: number;
  slotMatches: Array<{
    slotId: string;
    score: number;
    matchedQuery: string;
    matchType?: EvidenceMatchType;
    queryContextOnly?: boolean;
  }>;
  lineage?: LineageRef;
};

export type EvidencePacket = {
  packetId: string;
  slotId: string;
  slotIds?: string[];
  operationType: QueryEvidenceOperationType;
  role: "answer" | "partial" | "support";
  protected?: boolean;
  injected?: boolean;
  answerCandidate?: PromptEvidenceCandidate;
  contextCandidates?: PromptEvidenceCandidate[];
  answerUnits?: EvidenceUnit[];
  contextUnits?: EvidenceUnit[];
  supportUnits?: EvidenceUnit[];
  layers: EvidencePlanLayer[];
  primaryText: string;
  supportingTexts: string[];
  sourceRefs: string[];
  supportSourceRefs?: string[];
  allSourceRefs?: string[];
  normalizedSourceRefs?: NormalizedSourceRef[];
  normalizedSupportSourceRefs?: NormalizedSourceRef[];
  normalizedAllSourceRefs?: NormalizedSourceRef[];
  score?: number;
  scoreBreakdown?: Record<string, number | string | boolean>;
  displayLines?: string[];
  hiddenExactDuplicates?: Array<{
    displayText: string;
    sourceRefs: string[];
    hiddenSourceRefs: string[];
  }>;
  observedAt?: string;
  resolvedDate?: string;
  originalRelativeTime?: string;
  quantityHint?: number;
  unitHint?: string;
  dedupeKey?: string;
  entityAliases?: string[];
  authorRoles?: Array<"user" | "assistant" | "tool" | "memory">;
  coverage: {
    filled: boolean;
    missing: string[];
    confidence: number;
  };
  eligibility?: {
    eligible: boolean;
    role: "answer" | "context" | "resource" | "support";
    blockers: string[];
  };
  grade?: {
    retrievalScore: number;
    answerScore: number;
    contextBindingScore: number;
    temporalFitScore?: number;
    slotCoverageScore: number;
    authorityScore: number;
    softPenaltyScore?: number;
    finalScore: number;
  };
  selectionReason?: string;
  blockedBy?: string[];
  softPenalties?: string[];
  hardExclusions?: string[];
  protectionReason?: string;
  dropReason?: string;
};

export type EvidencePacketSlotAudit = {
  slotId: string;
  queriedLayers: EvidencePlanLayer[];
  layerCandidateCounts: Partial<Record<EvidencePlanLayer, number>>;
  packets: Array<{
    packetId: string;
    role: EvidencePacket["role"];
    layers: EvidencePlanLayer[];
    sourceRefs: string[];
    supportSourceRefs?: string[];
    allSourceRefs?: string[];
    normalizedSourceRefs?: NormalizedSourceRef[];
    normalizedSupportSourceRefs?: NormalizedSourceRef[];
    normalizedAllSourceRefs?: NormalizedSourceRef[];
    coverage: EvidencePacket["coverage"];
    eligibility?: EvidencePacket["eligibility"];
    grade?: EvidencePacket["grade"];
    score?: EvidencePacket["score"];
    scoreBreakdown?: EvidencePacket["scoreBreakdown"];
    displayLines?: string[];
    hiddenExactDuplicates?: EvidencePacket["hiddenExactDuplicates"];
    answerUnits?: EvidenceUnit[];
    contextUnits?: EvidenceUnit[];
    supportUnits?: EvidenceUnit[];
    selectionReason?: string;
    softPenalties?: string[];
    hardExclusions?: string[];
    protected: boolean;
    injected: boolean;
    compilerSignalSeen?: boolean;
    materializedLayers?: EvidencePlanLayer[];
    candidateSeen?: boolean;
    packetAssembled?: boolean;
    dropReason?: string | null;
  }>;
  missing: string[];
};

export type EvidencePacketAudit = {
  operation: QueryEvidenceOperation;
  slots: EvidencePacketSlotAudit[];
  candidatePool?: Array<{
    id: string;
    surface: PromptEvidenceCandidate["surface"];
    sourceRef?: string;
    mergedSourceRefs?: string[];
    normalizedSourceRefs?: NormalizedSourceRef[];
    role: PromptEvidenceCandidate["role"];
    injected?: boolean;
    packetId?: string;
    injectionScore?: number;
    slotEvidenceRole?: PromptEvidenceCandidate["slotEvidenceRole"];
    text: string;
  }>;
  evidenceUnits?: EvidenceUnit[];
  sourceExpansion?: Array<{
    unitId: string;
    sourceRefs: string[];
    normalizedSourceRefs?: NormalizedSourceRef[];
    supportRefs: string[];
    normalizedSupportRefs?: NormalizedSourceRef[];
    derivedFromRefs: string[];
    normalizedDerivedFromRefs?: NormalizedSourceRef[];
    neighborRefs: string[];
    normalizedNeighborRefs?: NormalizedSourceRef[];
    origin: EvidenceUnitOrigin;
    roles: EvidenceUnitRole[];
  }>;
  rankedPackets?: Array<{
    packetId: string;
    injected: boolean;
    score?: number;
    finalScore?: number;
    sourceRefs: string[];
    allSourceRefs: string[];
    normalizedSourceRefs?: NormalizedSourceRef[];
    normalizedAllSourceRefs?: NormalizedSourceRef[];
    selectionReason?: string;
  }>;
  injectedPackets?: string[];
  renderedPromptLines?: Array<{
    lineId: string;
    packetId: string;
    role: EvidenceUnitRole | "unknown";
    line: string;
    sourceRefs: string[];
    normalizedSourceRefs: NormalizedSourceRef[];
    evidenceUnitIds: string[];
  }>;
  hardExclusions?: Array<{ packetId: string; reasons: string[] }>;
  softPenalties?: Array<{ packetId: string; reasons: string[] }>;
  scoreCurve?: Array<{
    rank: number;
    packetId: string;
    injected: boolean;
    finalScore?: number;
    score?: number;
  }>;
  hiddenExactDuplicates?: NonNullable<EvidencePacket["hiddenExactDuplicates"]>;
};

export type EvidencePlanSlotAudit = {
  slotId: string;
  description: string;
  queriedLayers: EvidencePlanLayer[];
  layerCandidateCounts: Partial<Record<EvidencePlanLayer, number>>;
  filled: boolean;
  missingFields: string[];
  coverageScore: number;
  selectedEvidenceIds: string[];
  protectedEvidenceIds: string[];
  injectedEvidenceIds: string[];
  sourceRefs: string[];
};

export type EvidencePlanAudit = {
  operation: QueryEvidenceOperation;
  slots: EvidencePlanSlotAudit[];
};

export const MEMORY_OBJECT_KINDS = [
  "state",
  "task",
  "fact",
  "event",
  "graph_path",
  "chunk",
  "alternate",
] as const;

export type MemoryObjectKind = (typeof MEMORY_OBJECT_KINDS)[number];

export type MemoryObjectAttributes = {
  activeTask?: boolean;
  syntheticWorkflow?: boolean;
  relational?: boolean;
  temporalStructured?: boolean;
  temporalRole?: string;
  guidance?: boolean;
  entityCount?: number;
  graphNodeCount?: number;
  graphEdgeCount?: number;
  docType?: string;
  sourceKind?: MemorySourceKind;
  taskId?: string;
  parentTaskId?: string;
  taskPhase?: string;
  candidateResolution?: string;
  stateKey?: string;
  factSubject?: string;
  factPredicate?: string;
  factObject?: string;
  eventType?: string;
  sessionKey?: string;
};

export type MemoryObjectSemanticProfile = {
  workflowScore: number;
  factualScore: number;
  temporalScore: number;
  explanationScore: number;
  relationDensity: number;
  connectivity: number;
  continuityScore: number;
  recencyScore: number;
  stabilityScore: number;
  guidanceScore: number;
};

export type GraphEvidenceNode = {
  nodeId: string;
  nodeKind: GraphNodeKind;
  name: string;
  type: string;
  sourceObjectId?: string;
  entityId?: string;
  observedAt?: string;
  confidence?: number;
};

export type GraphEvidenceEdge = {
  edgeId: string;
  srcNodeId: string;
  relType: GraphTraversalRelationType;
  dstNodeId: string;
  relationSlot?: string;
  confidence: number;
  evidenceRef?: string;
  updatedAt?: string;
  sourceKind?: "stored" | "synthesized";
  srcEntityId?: string;
  dstEntityId?: string;
  metadata?: Record<string, unknown>;
};

export type GraphEvidence = {
  nodes: GraphEvidenceNode[];
  edges: GraphEvidenceEdge[];
  pathCandidates: GraphPathCandidate[];
  paths: string[];
};

export type GraphPathFeatures = {
  entityMatch: number;
  edgeConfidence: number;
  recency: number;
  pathLengthPenalty: number;
  contradictionPenalty: number;
  supportDiversity: number;
  relationFit: number;
  heterogeneousSupport: number;
};

export type GraphPathCandidate = {
  pathId: string;
  nodeIds: string[];
  edgeIds: string[];
  features: GraphPathFeatures;
  score: number;
  summary: string;
  reasons: string[];
};

export type MemoryObjectBelief = {
  beliefId: string;
  stage: MemoryBeliefStage;
  posteriorConfidence: number;
  usefulnessScore: number;
  stabilityScore: number;
  contradictionScore: number;
  outcomeSupportScore: number;
  lastUsedAt?: string;
  useCount: number;
};

export type MemoryObject = {
  objectId: string;
  kind: MemoryObjectKind;
  routeRole: RouteEvidenceCandidate["role"];
  row: EvidenceRow;
  baseScore: number;
  attributes: MemoryObjectAttributes;
  profile: MemoryObjectSemanticProfile;
  belief?: MemoryObjectBelief;
  graphNodes?: GraphEvidence["nodes"];
  graphEdges?: GraphEvidence["edges"];
  graphPathCandidate?: GraphPathCandidate;
  lineage?: LineageRef;
};

export type MemorySelectionObjective = {
  routeType: MemoryPrimaryRouteType;
  query: string;
  now: string;
  includeHistorical?: boolean;
  broadTemporal?: boolean;
  since?: string;
  currentSessionKey?: string;
};

export type ScheduledMemoryObject = {
  object: MemoryObject;
  objectiveScore: number;
  reasons: string[];
  graphSupport?: number;
  graphPenalty?: number;
};

export type RecallSelectionTraceEntry = {
  objectId: string;
  kind: MemoryObjectKind;
  text: string;
  weightedScore: number;
  maxRouteScore: number;
  selectionReason: string;
  strongestRoute?: MemoryPrimaryRouteType;
  routeScores: Partial<Record<MemoryPrimaryRouteType, number>>;
};

export type RecallSelectionTrace = {
  candidateCountsByRoute: Record<MemoryPrimaryRouteType, number>;
  reserveSelections: Record<MemoryPrimaryRouteType, RecallSelectionTraceEntry[]>;
  overflowSelections: RecallSelectionTraceEntry[];
  droppedHighScore: RecallSelectionTraceEntry[];
};

export type BudgetedMemorySelection = {
  scheduled: ScheduledMemoryObject[];
  trace: RecallSelectionTrace;
};

export type EvidenceBundle = {
  routeType: MemoryRouteType;
  routeConfidence: number;
  queryText?: string;
  queryAnchors?: string[];
  queryShape?: RecallQueryShape;
  routeWeights?: Partial<Record<MemoryPrimaryRouteType, number>>;
  turnMode?: TurnMode;
  snapshotFocus?: boolean;
  states: EvidenceRow[];
  tasks: EvidenceRow[];
  facts: EvidenceRow[];
  events: EvidenceRow[];
  graph: GraphEvidence;
  alternates: EvidenceRow[];
  diagnostics: string[];
  behavioralGuidance: string[];
  recalledChunkIds: string[];
  recalledChunkTexts: string[];
  promptEvidence: PromptEvidenceCandidate[];
  evidencePackets: EvidencePacket[];
  evidencePacketAudit?: EvidencePacketAudit;
  evidencePlanAudit?: EvidencePlanAudit;
  selectedExactSnippetIds?: string[];
  selectedExactSnippetTexts?: string[];
  selectedExactSnippets?: Array<{
    snippetId: string;
    text: string;
    sourceRef?: string;
    lineage: LineageRef;
  }>;
  budgetPlan?: RecallBudgetPlan;
  selectionTrace?: RecallSelectionTrace;
  renderedBlock: string;
};

export type BackgroundRecallBundle = {
  behavioralGuidance: string[];
  strategyGuidance: string[];
  states: EvidenceRow[];
  tasks: EvidenceRow[];
  projectionBlocks: WorkingProjectionBlock[];
};

export type WorkingProjectionBlock = {
  blockId: string;
  role: WorkingProjectionRole;
  title: string;
  lines: string[];
  sourceIds: string[];
};

export type StrategyHypothesisRecord = {
  strategyId: string;
  agentId: string;
  scope: string;
  domainKey: string;
  summary: string;
  supportBeliefIds: string[];
  supportTaskIds: string[];
  confidence: number;
  usefulnessScore: number;
  stabilityScore: number;
  contradictionScore: number;
  stage: StrategyHypothesisStage;
  derivedFromMinEpoch?: number;
  derivedFromMaxEpoch?: number;
  materializedEpoch?: number;
  derivedFromKind?: string;
  derivedFromIds?: string[];
  derivedAtEpoch?: number;
  derivationPolicyVersion?: string;
  metadataJson: Record<string, unknown>;
  createdAt: string;
  updatedAt: string;
};

export type AbstractionCandidateRecord = {
  candidateId: string;
  agentId: string;
  scope: string;
  abstractionType: AbstractionCandidateType;
  semanticKey: string;
  summary: string;
  supportContentRefs: string[];
  supportBeliefIds: string[];
  confidence: number;
  usefulnessScore: number;
  stabilityScore: number;
  contradictionScore: number;
  stage: AbstractionCandidateStage;
  derivedFromMinEpoch?: number;
  derivedFromMaxEpoch?: number;
  materializedEpoch?: number;
  derivedFromKind?: string;
  derivedFromIds?: string[];
  derivedAtEpoch?: number;
  derivationPolicyVersion?: string;
  metadataJson: Record<string, unknown>;
  createdAt: string;
  updatedAt: string;
};

export type TurnCaptureRole = "user" | "assistant" | "tool";

export type TurnCaptureMessage = {
  role: TurnCaptureRole;
  content: string;
  toolName?: string;
  observedAt: string;
  turnId: string;
  sessionKey: string;
  agentId: string;
  scope: string;
  sourceRef: string;
};

export type ConversationChunkStatus = "active" | "duplicate" | "merged";

export type ConversationChunk = {
  chunkId: string;
  agentId: string;
  scope: string;
  sessionKey: string;
  turnId: string;
  seq: number;
  role: TurnCaptureRole;
  toolName?: string;
  chunkKind: "message" | "tool_result" | "summary";
  content: string;
  summary: string;
  contentHash: string;
  taskId?: string;
  dedupStatus: ConversationChunkStatus;
  dedupTarget?: string;
  dedupReason?: string;
  mergeCount: number;
  lastHitAt?: string;
  sourceRef: string;
  createdAt: string;
  updatedAt: string;
};

export type SourceSegmentRecord = {
  segmentId: string;
  sourceGroupId: string;
  parentSourceRef: string;
  chunkId: string;
  agentId: string;
  scope: string;
  sessionKey: string;
  turnId: string;
  seq: number;
  role: TurnCaptureRole;
  toolName?: string;
  segmentIndex: number;
  charStart: number;
  charEnd: number;
  text: string;
  contentHash: string;
  createdAt: string;
  updatedAt: string;
  metadataJson: Record<string, unknown>;
};

export type ConversationTaskStatus = "active" | "completed" | "skipped";
export type ConversationTaskPhase =
  | "investigating"
  | "proposed"
  | "attempting"
  | "validated"
  | "resolved"
  | "reopened";

export type SynthesizedTaskEvent = {
  eventType: string;
  summary: string;
  phase: ConversationTaskPhase;
  closureScore: number;
  verificationScore: number;
  contradictionRisk: number;
  confidence: number;
  promotionScore: number;
  evidenceChunkIds: string[];
  outcomeKey: string;
};

export type ConversationTask = {
  taskId: string;
  agentId: string;
  scope: string;
  sessionKey: string;
  title: string;
  summary: string;
  status: ConversationTaskStatus;
  startedAt: string;
  endedAt?: string;
  updatedAt: string;
  metadataJson: Record<string, unknown>;
};

export type TaskAssignmentSnapshot = {
  taskId: string;
  status: ConversationTaskStatus;
  title: string;
  summary: string;
  updatedAt: string;
  metadataJson: Record<string, unknown>;
  recentContext?: string;
  isActive?: boolean;
};

export type TaskAssignmentDecision = {
  decision: "continue" | "resume" | "new";
  targetTaskId?: string;
  confidence: number;
  reason: string;
};

export type RetrievalAuditRecord = {
  auditId: string;
  agentId: string;
  scope: string;
  routeType: MemoryRouteType;
  queryText: string;
  queryHash: string;
  selectedItemsJson: Record<string, unknown>;
  injectedChars: number;
  createdAt: string;
};

export type CandidateHit = {
  candidateId: string;
  surface: CandidateSurface;
  tier?: "primary" | "alternate";
  text: string;
  score: number;
  retrievalBackend: "embedding" | "fts" | "hybrid" | "lexical" | "repo";
  docId?: string;
  scope: string;
  agentId: string;
  confidence?: number;
  activeHint?: boolean;
  supersededHint?: boolean;
  currentnessHint?: "current" | "historical" | "compare" | "unknown";
  lineage: LineageRef;
  goalMatches?: Array<{
    goal: string;
    score: number;
    matchedQuery: string;
    matchType?: EvidenceMatchType;
  }>;
  slotMatches?: Array<{
    slotId: string;
    score: number;
    matchedQuery: string;
    layer: EvidencePlanLayer;
    matchType?: EvidenceMatchType;
    queryContextOnly?: boolean;
  }>;
  bridgeMatches?: QuerySemanticBridgeMatch[];
  metadata?: Record<string, unknown>;
};

export type MemorySignalEventRecord = {
  signalId: string;
  agentId: string;
  scope: string;
  sessionKey?: string;
  signalType: MemorySignalType;
  memoryKind: MemorySignalTargetKind;
  contentRef?: string;
  semanticKey: string;
  value: number;
  sourceRef: string;
  metadataJson: Record<string, unknown>;
  createdAt: string;
};

export type MemoryBeliefRecord = {
  beliefId: string;
  agentId: string;
  scope: string;
  memoryKind: MemoryBeliefKind;
  contentRef?: string;
  semanticKey: string;
  stage: MemoryBeliefStage;
  priorConfidence: number;
  posteriorConfidence: number;
  usefulnessScore: number;
  stabilityScore: number;
  contradictionScore: number;
  outcomeSupportScore: number;
  sourceReliability: number;
  firstSeenAt: string;
  lastSeenAt: string;
  lastUsedAt?: string;
  useCount: number;
  reevaluationDueAt?: string;
  derivedFromMinEpoch?: number;
  derivedFromMaxEpoch?: number;
  materializedEpoch?: number;
  metadataJson: Record<string, unknown>;
  createdAt: string;
  updatedAt: string;
};

export type MaintenanceRunRecord = {
  runId: string;
  agentId: string;
  jobType: string;
  statsJson: Record<string, unknown>;
  startedAt: string;
  completedAt?: string;
  status: "running" | "completed" | "failed";
};

export type MaintenanceBatchTriggerReason = "threshold" | "idle" | "shutdown";

export type MaintenanceBatchWatermarks = {
  event?: string;
  signal?: string;
  task?: string;
};

export type MaintenanceBatchMetadata = {
  sessionKey: string;
  turnIds: string[];
  turnCount: number;
  reason: MaintenanceBatchTriggerReason;
  firstObservedAt?: string;
  lastObservedAt?: string;
  lowerWatermarks: MaintenanceBatchWatermarks;
  upperWatermarks: MaintenanceBatchWatermarks;
};

export type AbstractionJobStats = {
  eventsConsidered: number;
  taskBeliefsConsidered: number;
  factFamiliesConsidered: number;
  candidatesMaterialized: number;
  materializedCandidateIds?: string[];
  deferredByBudget: number;
  llmCandidatesConsidered: number;
  llmCandidatesRefined: number;
  llmRefinementEnabled?: boolean;
  deltaTriggered?: boolean;
  skippedNoRelevantDelta?: boolean;
  candidateSelection?: {
    cap: number;
    byType: Record<
      string,
      { budget: number; available: number; selected: number; deferred: number }
    >;
  };
  authoritySources?: MaintenanceAuthoritySource[];
  semanticSources?: MaintenanceSemanticSource[];
  maintenanceContractDiagnostics?: Record<string, unknown>;
  recallFacingDiagnostics?: Record<string, unknown>;
  activeCandidates: number;
  probationaryCandidates: number;
  candidateCandidates: number;
  decayingCandidates: number;
  quarantinedCandidates: number;
  supersededCandidates: number;
};

export type AbstractionPromotionStats = {
  candidatesEvaluated: number;
  deltaTriggered?: boolean;
  skippedNoRelevantDelta?: boolean;
  activeCandidates: number;
  probationaryCandidates: number;
  candidateCandidates: number;
  decayingCandidates: number;
  quarantinedCandidates: number;
  supersededCandidates: number;
  promotedGraphs: number;
  promotedOutcomes: number;
  promotedStates: number;
  promotedStrategies: number;
  promotedConcepts: number;
  skippedCandidates?: Array<{
    candidateId: string;
    abstractionType: string;
    reason: string;
  }>;
  maintenanceContractDiagnostics?: Record<string, unknown>;
  recallFacingDiagnostics?: Record<string, unknown>;
};

export type PluginActorContext = {
  agentId?: string;
  sessionKey?: string;
  workspaceDir?: string;
  project?: string;
  runId?: string;
  channelId?: string;
};

export type MemoryOperationContext = Required<Pick<PluginActorContext, "agentId">> &
  PluginActorContext & {
    config: MemoryPluginConfig;
    dbPath: string;
    scopes: string[];
    now: string;
    readEpoch?: number;
    llmBudgetAudit?: MemoryLlmBudgetAudit;
  };

/** Shared logger contract used across memx modules. */
export type MemxLogger = {
  warn: (message: string) => void;
  info?: (message: string) => void;
  debug?: (message: string) => void;
  error?: (message: string) => void;
};
