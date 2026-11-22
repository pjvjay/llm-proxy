"""OWASP Top 10 for LLM Applications (2025) -- catalogue and the mapping from
this firewall's checks to it.

Every evaluation the firewall makes (System-1 label, System-2 judgement,
deterministic rule) is tagged with the OWASP categories it provides evidence
for. Tags travel with the decision: response headers, the `firewall` object in
blocked responses, the audit log, /firewall/metrics, and the eval report.

A System-1 decision model (e.g. an OpenJev export in models/decision) can
declare its own label -> OWASP mapping in labels.json ("owasp" key); the
guardrails service returns it with every /v1/decide call and it takes
precedence over DEFAULT_MAP for the labels it names.

Pure stdlib: imported by the addon AND by evals/run_eval.py on the host."""

CATALOG = {
    "LLM01:2025": "Prompt Injection",
    "LLM02:2025": "Sensitive Information Disclosure",
    "LLM03:2025": "Supply Chain",
    "LLM04:2025": "Data and Model Poisoning",
    "LLM05:2025": "Improper Output Handling",
    "LLM06:2025": "Excessive Agency",
    "LLM07:2025": "System Prompt Leakage",
    "LLM08:2025": "Vector and Embedding Weaknesses",
    "LLM09:2025": "Misinformation",
    "LLM10:2025": "Unbounded Consumption",
}

# check / label -> OWASP ids. An empty list means "evaluated, but not an OWASP
# LLM Top-10 category" (content safety), and is reported as such rather than
# forced into a category it doesn't belong to.
DEFAULT_MAP: dict[str, list[str]] = {
    # System-1 labels
    "injection":                ["LLM01:2025"],
    "jailbreak":                ["LLM01:2025"],
    "system_prompt_extraction": ["LLM07:2025", "LLM01:2025"],
    "toxicity":                 [],
    # deterministic checks
    "regex":                    [],              # violent / illegal / sexual keyword categories
    "pii":                      ["LLM02:2025"],
    "system_prompt_leak":       ["LLM07:2025"],
    "output_markup":            ["LLM05:2025"],
    "output_exfiltration":      ["LLM05:2025", "LLM02:2025"],
    "consumption":              ["LLM10:2025"],
    "supply_chain":             ["LLM03:2025"],
    # tool-call rules (tool_policy.py rule names)
    "tool_call":                ["LLM06:2025"],
    "secret_in_args":           ["LLM06:2025", "LLM02:2025"],
    "s2_intent":                ["LLM06:2025"],
}

# Tool names that indicate retrieval from an index / vector store: injected
# content arriving through them is also evidence for LLM08 (RAG poisoning).
_RETRIEVAL_HINTS = ("retriev", "rag", "vector", "embedding", "knowledge", "kb_")


def ids_for(check: str, role: str | None = None, tool: str | None = None,
            model_map: dict | None = None) -> list[str]:
    """OWASP ids for one evaluation, including context-dependent extras."""
    if model_map and check in model_map:
        ids = list(model_map[check])
    elif check in DEFAULT_MAP:
        ids = list(DEFAULT_MAP[check])
    elif check.startswith("tool_") or check in ("shell_destructive", "remote_exec", "sql_destructive",
                                                 "sensitive_path", "ssrf", "url_allow", "allow",
                                                 "deny", "max_arg_chars"):
        ids = list(DEFAULT_MAP["tool_call"])
    else:
        ids = []
    if check in ("injection", "jailbreak") and role == "tool" and tool and \
            any(h in tool.lower() for h in _RETRIEVAL_HINTS):
        ids.append("LLM08:2025")
    return sorted(set(ids))


def describe(ids) -> list[dict]:
    return [{"id": i, "name": CATALOG.get(i, "unknown")} for i in sorted(set(ids))]


def label(ids) -> str:
    """'LLM01:2025 Prompt Injection, LLM07:2025 System Prompt Leakage'"""
    return ", ".join(f"{i} {CATALOG.get(i, '')}".strip() for i in sorted(set(ids)))


# What this deployment does about each category -- used by the eval report and
# docs/owasp-llm-top10.md. status: covered | partial | not covered.
COVERAGE = {
    "LLM01:2025": ("covered", "System-1 injection label on user messages and tool results (indirect), "
                              "System-2 judge for uncertain cases, regex pre-filter, canonicalisation "
                              "(NFKC, zero-width, base64), sliding windows"),
    "LLM02:2025": ("covered", "Presidio PII / secret redaction both directions and in tool results; "
                              "secrets in tool-call arguments; markdown/URL exfiltration in output"),
    "LLM03:2025": ("partial", "SHA-256 manifest verification of model artifacts at guardrails start-up "
                              "(fail closed on mismatch); revision pinning for downloaded models. "
                              "Does not vet the upstream LLM or Python dependencies"),
    "LLM04:2025": ("partial", "Poisoned tool / retrieval content is screened at inference time (LLM01 "
                              "controls). Training-time data or model poisoning is out of scope for an "
                              "inline proxy"),
    "LLM05:2025": ("covered", "Response screening for active markup (script, event handlers, "
                              "javascript: URIs, iframes) and exfiltration-shaped links; neutralised "
                              "before the caller sees them"),
    "LLM06:2025": ("covered", "tool_policy on every tool call (allow/deny, destructive shell/SQL, "
                              "remote exec, sensitive paths, SSRF, secrets, URL allow-list) + System-2 "
                              "intent check for side-effecting tools once session risk is elevated"),
    "LLM07:2025": ("covered", "Extraction attempts tagged on requests; responses checked for verbatim "
                              "system-prompt content (shingle overlap) and withheld"),
    "LLM08:2025": ("partial", "Injection in results of retrieval-type tools is tagged LLM08. No access "
                              "to the vector store itself (permissions, embedding inversion)"),
    "LLM09:2025": ("not covered", "Needs grounding against sources; not an inline-proxy check today"),
    "LLM10:2025": ("covered", "Request size cap, max_tokens clamp, optional per-session rate limit, "
                              "System-2 load shedding"),
}
