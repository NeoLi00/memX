const ENTITY_TYPES = [
	"person",
	"project",
	"tool",
	"service",
	"language",
	"framework",
	"concept",
	"organization",
	"unknown"
];
const MEMORY_SCOPE_TEMPLATES = [
	"global",
	"workspace:{workspace}",
	"agent:{agentId}",
	"session:{sessionKey}",
	"project:{project}"
];
const MEMORY_PII_MODES = [
	"off",
	"redact",
	"allow"
];
const MEMORY_CONSENT_MODES = [
	"explicit",
	"implicit",
	"off"
];
const MEMORY_EMBEDDING_PROVIDERS = [
	"off",
	"openai-compatible",
	"ollama",
	"sentence-transformers-local"
];
const MEMORY_LLM_PROVIDERS = [
	"openai-compatible",
	"anthropic",
	"google",
	"ollama"
];
//#endregion
export { ENTITY_TYPES, MEMORY_CONSENT_MODES, MEMORY_EMBEDDING_PROVIDERS, MEMORY_LLM_PROVIDERS, MEMORY_PII_MODES, MEMORY_SCOPE_TEMPLATES };
