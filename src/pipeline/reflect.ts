import { resolveDefaultScope, scopeVarsForContext } from "../security/scopes.js";
import { normalizeText } from "../support.js";
import type { ClassifiedCandidate, MemoryCandidate, MemoryOperationContext } from "../types.js";
import { classifyAction } from "./classify.js";
import { computeConfidence } from "./normalize.js";
import { evaluatePolicy } from "./policy.js";
import type { MemxReasoner } from "./reasoner.js";

export function reflectCandidates(
  candidates: MemoryCandidate[],
  ctx: MemoryOperationContext,
  options?: { reasoner?: MemxReasoner },
): Promise<ClassifiedCandidate[]> {
  return reflectCandidatesInternal(candidates, ctx, options);
}

async function reflectCandidatesInternal(
  candidates: MemoryCandidate[],
  ctx: MemoryOperationContext,
  options?: { reasoner?: MemxReasoner },
): Promise<ClassifiedCandidate[]> {
  const eligible = candidates.slice(-ctx.config.reflectionMaxItems).filter((candidate) => {
    if (candidate.source.kind === "assistant") return false;
    const trimmed = candidate.rawText.trim();
    return trimmed.length > 0 && trimmed.length <= ctx.config.reflectionMaxChars;
  });

  const results = await Promise.all(
    eligible.map(async (candidate) => {
      const policyResult = await evaluatePolicy(candidate, ctx, {
        reasoner: options?.reasoner,
      });
      const trimmed = candidate.rawText.trim();
      const classified: ClassifiedCandidate = {
        ...policyResult.candidate,
        normalizedText: normalizeText(trimmed),
        policy: policyResult.decision,
        scope: resolveDefaultScope(ctx.config, scopeVarsForContext(ctx)),
        classification: classifyAction(policyResult.decision.action),
        confidence: 0,
      };
      classified.confidence = computeConfidence(classified);
      return classified.classification !== "ignore" ? classified : null;
    }),
  );

  return results.filter((r): r is ClassifiedCandidate => r !== null);
}
