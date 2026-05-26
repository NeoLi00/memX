import { clamp01, normalizeText } from "../support.mjs";
import { hasExplicitRememberIntent, isLowValueChatter, isQuestionLike } from "./semantic/heuristics.mjs";
import { inferWriteLlmStage, recordMemoryLlmBudgetCall } from "./llmBudgetAudit.mjs";
import { containsLikelySecret, looksLikePromptInjection } from "../security/injection.mjs";
import { canonicalizePreferenceHint } from "./semantics.mjs";
import { sanitizeWorkflowHint } from "./authority.mjs";
import { containsSensitiveValue, sensitivityScore } from "../security/pii.mjs";
import { DEFAULT_SCORES, SENSITIVITY_PROMPT_INJECTION_BOOST, SENSITIVITY_SENSITIVE_VALUE_BOOST } from "./constants.mjs";
import { containsUntrustedBanner } from "../security/escaping.mjs";
//#region src/pipeline/policy.ts
const HYPOTHETICAL_REFERENCE_RULE_PATTERN = /(?:如果我(?:突然)?(?:问|说)|如果之后我(?:问|说)|当我(?:说|问)|以后我(?:说|问)).{0,80}(?:应该知道|应该带上|需要带上|得能接上|你应该|要能关联)/iu;
function isHypotheticalReferenceRule(text) {
	return HYPOTHETICAL_REFERENCE_RULE_PATTERN.test(text);
}
function structuredRelations(candidate) {
	const hints = candidate.structuredHints;
	const relations = hints?.relations && hints.relations.length > 0 ? hints.relations : hints?.relation ? [hints.relation] : void 0;
	if (!relations) return;
	const affirmed = relations.filter((relation) => relation.polarity !== "negated");
	if (affirmed.length === 0) return;
	return affirmed;
}
function derivePolicySignals(candidate) {
	const text = candidate.rawText;
	const explicitIntent = hasExplicitRememberIntent(text);
	const questionLikeUserQuery = candidate.source.kind === "user" && (isQuestionLike(text) || isHypotheticalReferenceRule(text)) && !explicitIntent;
	const cached = candidate.structuredHints;
	const preference = canonicalizePreferenceHint(cached?.preference);
	const workflows = cached?.workflows && cached.workflows.length > 0 ? cached.workflows : cached?.workflow ? [cached.workflow] : [];
	const effectiveWorkflows = questionLikeUserQuery ? [] : workflows;
	const relations = structuredRelations(candidate) ?? [];
	const relation = relations[0];
	const decision = cached?.decision;
	return {
		preference,
		workflow: effectiveWorkflows[0],
		workflows: effectiveWorkflows,
		relations,
		relation,
		decision,
		preferenceHint: Boolean(preference),
		workflowHint: effectiveWorkflows.length > 0,
		relationHint: relations.length > 0,
		decisionHint: Boolean(decision)
	};
}
function enrichCandidateWithSignals(candidate, signals) {
	return {
		...candidate,
		structuredHints: {
			...candidate.structuredHints ?? {},
			preferenceHint: signals.preferenceHint,
			decisionHint: signals.decisionHint,
			relationHint: signals.relationHint,
			taskStateHint: signals.workflowHint,
			preference: signals.preference ?? void 0,
			workflow: signals.workflow ?? void 0,
			workflows: signals.workflows.length > 0 ? signals.workflows : void 0,
			relation: signals.relation ?? void 0,
			relations: signals.relations.length > 0 ? signals.relations : void 0,
			decision: signals.decision ?? void 0
		}
	};
}
function enrichCandidateWithLlmJudgment(candidate, judgment) {
	const explicitIntent = hasExplicitRememberIntent(candidate.rawText);
	const suppressWorkflowHints = candidate.source.kind === "user" && (isQuestionLike(candidate.rawText) || isHypotheticalReferenceRule(candidate.rawText)) && !explicitIntent;
	const existingPreference = canonicalizePreferenceHint(candidate.structuredHints?.preference);
	const llmPreference = canonicalizePreferenceHint(judgment.preference);
	const existingWorkflows = candidate.structuredHints?.workflows && candidate.structuredHints.workflows.length > 0 ? candidate.structuredHints.workflows : candidate.structuredHints?.workflow ? [candidate.structuredHints.workflow] : [];
	const llmWorkflows = judgment.workflows && judgment.workflows.length > 0 ? judgment.workflows : judgment.workflow ? [judgment.workflow] : [];
	const workflows = suppressWorkflowHints ? [] : mergeWorkflowHints(existingWorkflows, llmWorkflows);
	const relations = mergeRelationHints(structuredRelations(candidate) ?? [], judgment.relations && judgment.relations.length > 0 ? judgment.relations : judgment.relation ? [judgment.relation] : []);
	return {
		...candidate,
		structuredHints: {
			...candidate.structuredHints ?? {},
			preferenceHint: Boolean(llmPreference ?? existingPreference),
			decisionHint: Boolean(judgment.decision ?? candidate.structuredHints?.decision),
			relationHint: relations.length > 0,
			taskStateHint: workflows.length > 0,
			preference: llmPreference ?? existingPreference ?? void 0,
			workflow: workflows[0] ?? void 0,
			workflows: workflows.length > 0 ? workflows : void 0,
			relation: relations[0] ?? void 0,
			relations: relations.length > 0 ? relations : void 0,
			decision: judgment.decision ?? candidate.structuredHints?.decision ?? void 0
		}
	};
}
function mergeWorkflowHints(existing, llm) {
	const byKey = /* @__PURE__ */ new Map();
	for (const workflow of existing) {
		const sanitized = sanitizeWorkflowHint(workflow);
		if (sanitized) byKey.set(sanitized.key, sanitized);
	}
	for (const workflow of llm) {
		const sanitized = sanitizeWorkflowHint(workflow);
		if (!sanitized) continue;
		const current = byKey.get(sanitized.key);
		if (!current) {
			byKey.set(sanitized.key, sanitized);
			continue;
		}
		byKey.set(sanitized.key, {
			...current,
			...sanitized,
			value: {
				...current.value ?? {},
				...sanitized.value ?? {}
			},
			confidence: Math.max(current.confidence ?? 0, sanitized.confidence ?? 0),
			reason: sanitized.reason || current.reason,
			stateKind: sanitized.stateKind ?? current.stateKind
		});
	}
	return [...byKey.values()];
}
function relationHintKey(relation) {
	return [
		relation.subject,
		relation.predicate,
		relation.polarity ?? "affirmed",
		relation.object,
		relation.relationSlot ?? "",
		relation.rawPredicate ?? ""
	].map((value) => normalizeText(String(value))).join("|");
}
function mergeRelationHints(existing, llm) {
	const byKey = /* @__PURE__ */ new Map();
	for (const relation of existing) byKey.set(relationHintKey(relation), relation);
	for (const relation of llm) {
		const key = relationHintKey(relation);
		const current = byKey.get(key);
		if (!current) {
			byKey.set(key, relation);
			continue;
		}
		byKey.set(key, {
			...current,
			...relation,
			confidence: Math.max(current.confidence ?? 0, relation.confidence ?? 0),
			reason: relation.reason || current.reason
		});
	}
	return [...byKey.values()];
}
function maxHintConfidence(values) {
	const finite = values.filter((value) => typeof value === "number" && Number.isFinite(value));
	if (finite.length === 0) return;
	return Math.max(...finite);
}
function semanticDraftGateScores(action, hintConfidence) {
	const defaults = defaultScoresForAction(action);
	if (hintConfidence == null) return defaults;
	const confidenceFloor = clamp01(hintConfidence);
	return {
		salienceScore: clamp01(Math.max(defaults.salienceScore - .08, confidenceFloor)),
		expectedFutureUtility: clamp01(Math.max(defaults.expectedFutureUtility - .08, confidenceFloor * .92)),
		stabilityScore: clamp01(Math.max(defaults.stabilityScore - .08, confidenceFloor * .9))
	};
}
function semanticDraft(candidate) {
	return candidate.structuredHints?.semanticDraft;
}
function semanticDraftFamilies(candidate) {
	const draft = semanticDraft(candidate);
	const draftFamilies = draft?.assertionDrafts.map((entry) => entry.familyHint) ?? [];
	if ((draft?.relationDrafts?.length ?? 0) > 0) return [...new Set([...draftFamilies, "relation_like"])];
	if (draftFamilies.length > 0) return [...new Set(draftFamilies)];
	return [...new Set(candidate.structuredHints?.semanticFamilies ?? [])];
}
function semanticDraftTimeframes(candidate) {
	const draftTimeframes = semanticDraft(candidate)?.assertionDrafts.map((entry) => entry.timeframeHint) ?? [];
	if (draftTimeframes.length > 0) return [...new Set(draftTimeframes)];
	return [...new Set(candidate.structuredHints?.semanticTimeframes ?? [])];
}
function semanticDraftMaterializationHint(candidate) {
	return candidate.structuredHints?.materializationHint;
}
function semanticDraftConfidence(candidate) {
	const draft = semanticDraft(candidate);
	return maxHintConfidence([
		maxHintConfidence(draft?.assertionDrafts.map((entry) => entry.confidence).filter((value) => typeof value === "number") ?? []),
		maxHintConfidence(draft?.correctionDrafts.map((entry) => entry.confidence).filter((value) => typeof value === "number") ?? []),
		maxHintConfidence(draft?.relationDrafts?.map((entry) => entry.confidence ?? entry.relation.confidence).filter((value) => typeof value === "number") ?? [])
	]);
}
function structuredSemanticSignalConfidence(candidate) {
	const hints = candidate.structuredHints;
	return maxHintConfidence([
		...(hints?.resourceAssertions ?? []).map((entry) => entry.confidence),
		...(hints?.adviceSignals ?? []).map((entry) => entry.confidence),
		...(semanticDraft(candidate)?.resourceAssertions ?? []).map((entry) => entry.confidence),
		...(semanticDraft(candidate)?.adviceSignals ?? []).map((entry) => entry.confidence)
	]);
}
function workflowActionFromDraft(hint) {
	if (hint?.preferDurableState) return "durable_state";
	return "session_state";
}
function chooseSemanticDraftAction(candidate) {
	const reasons = [];
	const correction = candidate.structuredHints?.correction;
	const families = semanticDraftFamilies(candidate);
	const timeframes = semanticDraftTimeframes(candidate);
	const materializationHint = semanticDraftMaterializationHint(candidate);
	const draftConfidence = semanticDraftConfidence(candidate);
	const primaryFamily = materializationHint?.primaryFamily ?? families[0];
	const correctionConfidence = maxHintConfidence([correction?.confidence, draftConfidence]);
	if (primaryFamily === "workflow") {
		const action = workflowActionFromDraft(materializationHint);
		reasons.push("semantic-draft-adapter:workflow");
		return {
			action,
			reasons,
			scores: semanticDraftGateScores(action, draftConfidence)
		};
	}
	if (correction) {
		if (correction.targetKind === "relation" && primaryFamily === "relation_like" && !materializationHint?.preferEvent) {
			reasons.push("semantic-draft-adapter:relation-correction");
			return {
				action: "graph_relation",
				reasons,
				scores: semanticDraftGateScores("graph_relation", correctionConfidence)
			};
		}
		if (materializationHint?.replacementMode === "supersede_fact" || correction.targetKind === "fact" || correction.targetKind === "state" || correction.targetKind === "project_profile") {
			reasons.push("semantic-draft-adapter:fact-correction");
			return {
				action: "stable_fact",
				reasons,
				scores: semanticDraftGateScores("stable_fact", correctionConfidence)
			};
		}
		if (materializationHint?.preferEvent || correction.timeframe === "historical" || correction.timeframe === "compare" || timeframes.includes("historical") || timeframes.includes("compare")) {
			reasons.push("semantic-draft-adapter:historical-correction");
			return {
				action: "episodic_event",
				reasons,
				scores: semanticDraftGateScores("episodic_event", correctionConfidence)
			};
		}
	}
	if (primaryFamily === "strategy_like") {
		reasons.push("semantic-draft-adapter:strategy-guidance");
		return {
			action: "stable_fact",
			reasons,
			scores: semanticDraftGateScores("stable_fact", draftConfidence)
		};
	}
	if ((candidate.structuredHints?.resourceAssertions?.length ?? 0) > 0) {
		reasons.push("semantic-draft-adapter:resource-assertion");
		return {
			action: "stable_fact",
			reasons,
			scores: semanticDraftGateScores("stable_fact", maxHintConfidence([draftConfidence, structuredSemanticSignalConfidence(candidate)]))
		};
	}
	if ((candidate.structuredHints?.adviceSignals?.length ?? 0) > 0) {
		reasons.push("semantic-draft-adapter:advice-signal");
		return {
			action: "stable_fact",
			reasons,
			scores: semanticDraftGateScores("stable_fact", maxHintConfidence([draftConfidence, structuredSemanticSignalConfidence(candidate)]))
		};
	}
	if (primaryFamily === "preference" || primaryFamily === "fact_like") {
		reasons.push("semantic-draft-adapter:stable-fact");
		return {
			action: "stable_fact",
			reasons,
			scores: semanticDraftGateScores("stable_fact", draftConfidence)
		};
	}
	if (primaryFamily === "relation_like" && !materializationHint?.replacementMode && !materializationHint?.preferEvent) {
		reasons.push("semantic-draft-adapter:relation");
		return {
			action: "graph_relation",
			reasons,
			scores: semanticDraftGateScores("graph_relation", draftConfidence)
		};
	}
	if (primaryFamily === "event_like") {
		reasons.push("semantic-draft-adapter:event");
		return {
			action: "episodic_event",
			reasons,
			scores: semanticDraftGateScores("episodic_event", draftConfidence)
		};
	}
	if (candidate.source.kind === "tool") {
		if (materializationHint?.preferEvent || Boolean(correction)) {
			reasons.push("semantic-draft-adapter:tool-event");
			return {
				action: "episodic_event",
				reasons,
				scores: semanticDraftGateScores("episodic_event", draftConfidence)
			};
		}
		reasons.push("semantic-draft-adapter:tool-ignore");
		return {
			action: "ignore",
			reasons,
			scores: semanticDraftGateScores("ignore", void 0)
		};
	}
	reasons.push(materializationHint?.primaryFamily ? "semantic-draft-adapter:no-safe-owner" : "semantic-draft-adapter:no-compiler-family");
	return {
		action: "ignore",
		reasons,
		scores: semanticDraftGateScores("ignore", draftConfidence)
	};
}
function evaluatePolicyFromSemanticDraft(candidate, ctx, explicitIntent) {
	const enrichedCandidate = enrichCandidateWithSignals(candidate, derivePolicySignals(candidate));
	const gated = chooseSemanticDraftAction(enrichedCandidate);
	return {
		candidate: enrichedCandidate,
		decision: finalizeDecision({
			candidate: enrichedCandidate,
			ctx,
			explicitIntent,
			proposedAction: gated.action,
			salienceScore: gated.scores.salienceScore,
			expectedFutureUtility: gated.scores.expectedFutureUtility,
			stabilityScore: gated.scores.stabilityScore,
			reasons: gated.reasons
		})
	};
}
function naturalCaptureEligible(candidate, action) {
	if (candidate.source.kind !== "user") return false;
	const text = candidate.rawText.trim();
	if (!text || isLowValueChatter(text)) return false;
	return action !== "ignore";
}
function captureAuthorized(candidate, ctx, action, explicitIntent) {
	if (candidate.source.kind === "tool") return true;
	if (explicitIntent) return true;
	if (ctx.config.consentMode === "off") return false;
	if (ctx.config.autoCapture) return true;
	return naturalCaptureEligible(candidate, action);
}
function defaultScoresForAction(action) {
	const entry = DEFAULT_SCORES[action] ?? DEFAULT_SCORES.fallback;
	return {
		salienceScore: entry.salience,
		expectedFutureUtility: entry.utility,
		stabilityScore: entry.stability
	};
}
function finalizeDecision(params) {
	const sensitivity = clamp01(sensitivityScore(params.candidate.rawText) + (containsSensitiveValue(params.candidate.rawText) ? SENSITIVITY_SENSITIVE_VALUE_BOOST : 0) + (looksLikePromptInjection(params.candidate.rawText) && !containsLikelySecret(params.candidate.rawText) ? SENSITIVITY_PROMPT_INJECTION_BOOST : 0));
	const reasons = [...params.reasons];
	const hasInjectedHistory = containsUntrustedBanner(params.candidate.rawText);
	const containsSecret = containsLikelySecret(params.candidate.rawText);
	const containsSensitivePayload = containsSensitiveValue(params.candidate.rawText);
	const injectionLike = looksLikePromptInjection(params.candidate.rawText) && !containsSecret;
	const authorized = captureAuthorized(params.candidate, params.ctx, params.proposedAction, params.explicitIntent);
	if (!authorized) reasons.push("capture_not_authorized");
	if (hasInjectedHistory) reasons.push("contains_injected_history");
	if (injectionLike) reasons.push("instructional_or_sensitive_pattern");
	if (sensitivity > params.ctx.config.maxSensitivityAllowed) reasons.push("high_sensitivity_observed");
	if (containsSecret || containsSensitivePayload) reasons.push("secret_like_payload");
	if (isLowValueChatter(params.candidate.rawText)) reasons.push("low_value_chatter");
	let action = params.proposedAction;
	if (!authorized || hasInjectedHistory) action = "ignore";
	else if (injectionLike || sensitivity > params.ctx.config.maxSensitivityAllowed || containsSecret || containsSensitivePayload) action = "ignore";
	return {
		salienceScore: clamp01(params.salienceScore),
		expectedFutureUtility: clamp01(params.expectedFutureUtility),
		sensitivityScore: sensitivity,
		stabilityScore: clamp01(params.stabilityScore),
		action,
		reasons,
		explicitIntent: params.explicitIntent,
		captureAuthorized: authorized
	};
}
async function evaluatePolicy(candidate, ctx, options = {}) {
	const explicitIntent = hasExplicitRememberIntent(normalizeText(candidate.rawText));
	const questionLikeWithoutExplicitIntent = candidate.source.kind === "user" && (isQuestionLike(candidate.rawText) || isHypotheticalReferenceRule(candidate.rawText)) && !candidate.structuredHints?.correctionHint && !explicitIntent;
	if (ctx.config.advanced.enableTurnSemanticCompiler) {
		const compiledPolicy = evaluatePolicyFromSemanticDraft(candidate, ctx, explicitIntent);
		recordMemoryLlmBudgetCall(ctx.llmBudgetAudit, {
			label: "candidate-policy",
			stage: inferWriteLlmStage(candidate.source.kind),
			provenance: "deterministic",
			mode: "deterministic",
			detail: "candidate-policy consumed the LLM semantic draft through a storage adapter"
		});
		return {
			candidate: compiledPolicy.candidate,
			decision: {
				...compiledPolicy.decision,
				reasons: [...compiledPolicy.decision.reasons, "semantic-compiler-policy-adapter"]
			}
		};
	}
	if (!options.reasoner) {
		recordMemoryLlmBudgetCall(ctx.llmBudgetAudit, {
			label: "candidate-policy",
			stage: inferWriteLlmStage(candidate.source.kind),
			provenance: "deterministic",
			mode: "fallback",
			detail: "candidate-policy requires LLM semantics and failed closed"
		});
		return {
			candidate,
			decision: finalizeDecision({
				candidate,
				ctx,
				explicitIntent,
				proposedAction: "ignore",
				salienceScore: 0,
				expectedFutureUtility: 0,
				stabilityScore: 0,
				reasons: ["llm-only-policy-unavailable"]
			})
		};
	}
	const judgment = await options.reasoner?.judgeCandidatePolicy(candidate, {
		stage: inferWriteLlmStage(candidate.source.kind),
		audit: ctx.llmBudgetAudit
	});
	if (!judgment?.action) return {
		candidate,
		decision: finalizeDecision({
			candidate,
			ctx,
			explicitIntent,
			proposedAction: "ignore",
			salienceScore: 0,
			expectedFutureUtility: 0,
			stabilityScore: 0,
			reasons: ["llm:missing-or-invalid-judgment"]
		})
	};
	const enrichedCandidate = enrichCandidateWithLlmJudgment(candidate, judgment);
	const defaultScores = defaultScoresForAction(judgment.action);
	const reasons = [`llm:${judgment.reason ?? "semantic policy judgment"}`];
	if (enrichedCandidate.source.kind === "user" && isQuestionLike(enrichedCandidate.rawText) && judgment.action !== "ignore") reasons.push("llm_overrode_question_shape");
	const proposedAction = questionLikeWithoutExplicitIntent && (judgment.action === "session_state" || judgment.action === "durable_state") ? "ignore" : judgment.action;
	if (proposedAction === "ignore" && proposedAction !== judgment.action) reasons.push("question_like_state_write_blocked");
	return {
		candidate: enrichedCandidate,
		decision: finalizeDecision({
			candidate: enrichedCandidate,
			ctx,
			explicitIntent,
			proposedAction,
			salienceScore: judgment.salienceScore ?? defaultScores.salienceScore,
			expectedFutureUtility: judgment.expectedFutureUtility ?? defaultScores.expectedFutureUtility,
			stabilityScore: judgment.stabilityScore ?? defaultScores.stabilityScore,
			reasons
		})
	};
}
//#endregion
export { evaluatePolicy };
