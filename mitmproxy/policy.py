"""Per-category action + fail mode. P0 keeps this a small dict; the P1 YAML
policy engine slots in behind the same decide() call.

Fail mode matters: if the classifier service is unreachable, injection and PII
categories FAIL CLOSED (block), toxicity fails open by default (configurable) so
a guardrail outage doesn't take down all legitimate traffic."""
import os

DEFAULT = {
    "injection": {"action": "block", "fail": "closed"},
    "toxicity":  {"action": os.getenv("TOXICITY_ACTION", "block"),
                  "fail":   os.getenv("TOXICITY_FAIL", "open")},
    "pii":       {"action": os.getenv("PII_ACTION", "redact"), "fail": "closed"},
    "regex":     {"action": "block", "fail": "closed"},
    # PII inside a `role: tool` message. Defaults to PII_ACTION; set it to
    # "log" if redacting tool output (e.g. a customer's email the agent must
    # use next) breaks your agent, accepting that the data reaches the provider.
    "pii_tool":  {"action": os.getenv("TOOL_RESULT_PII_ACTION", os.getenv("PII_ACTION", "redact")),
                  "fail": "closed"},
    # Injection found in a `role: tool` message. "redact" withholds just that
    # tool result (the agent keeps going, the payload never reaches the model);
    # "block" refuses the whole call -- which, since the poisoned result stays in
    # the client's history, blocks every subsequent turn of that agent run too.
    "tool_result": {"action": os.getenv("TOOL_RESULT_ACTION", "redact"), "fail": "closed"},
    # A tool call in the model's response that violates tool_policy.
    "tool_call":   {"action": os.getenv("TOOL_CALL_ACTION", "block"), "fail": "closed"},
    # An unexpected exception inside the addon. The old addon let mitmproxy log
    # it and forward the request unscreened; default is now to block.
    "error":       {"action": os.getenv("FIREWALL_ON_ERROR", "block"), "fail": "closed"},
}

THRESHOLDS = {
    "injection": float(os.getenv("INJECTION_THRESHOLD", "0.5")),
    "toxicity":  float(os.getenv("TOXICITY_THRESHOLD", "0.5")),
}

# Labels beyond the built-in ones (e.g. extra heads of a System-1 decision
# model such as an OpenJev export: "jailbreak", "system_prompt_extraction", ...)
# get a generic policy, overridable per label:
#   <LABEL>_ACTION (default block), <LABEL>_FAIL (default open -- extra labels
#   are additive; the core injection label stays fail-closed), <LABEL>_THRESHOLD.
# Also consumption (OWASP LLM10), system-prompt leak (LLM07) and output
# handling (LLM05) categories.
DEFAULT.update({
    "consumption":        {"action": "block", "fail": "closed"},
    "system_prompt_leak": {"action": os.getenv("SYSTEM_PROMPT_LEAK_ACTION", "block"), "fail": "closed"},
    "output_handling":    {"action": os.getenv("OUTPUT_HANDLING_ACTION", "redact"), "fail": "closed"},
})


def label_policy(label: str) -> dict:
    if label in DEFAULT:
        return DEFAULT[label]
    up = label.upper()
    pol = {"action": os.getenv(f"{up}_ACTION", "block"), "fail": os.getenv(f"{up}_FAIL", "open")}
    DEFAULT[label] = pol
    return pol


def threshold(label: str) -> float:
    if label not in THRESHOLDS:
        THRESHOLDS[label] = float(os.getenv(f"{label.upper()}_THRESHOLD", "0.5"))
    return THRESHOLDS[label]

# ---------------------------------------------------------------- System 1 bands
# A System-1 decision model answers fast and attaches a confidence. Two
# thresholds per (role, label) turn that into three outcomes:
#     p <  allow_below  -> allow      (confident, fast path)
#     p >= block_at     -> block      (confident, fast path)
#     otherwise         -> uncertain  -> System 2 judge if enabled, else the
#                                        single THRESHOLDS value (old behaviour)
# Bands are per role because the same sentence means different things in a user
# message and in a tool result. Env: S1_BAND_<ROLE>_<LABEL>="low,high".
def _band(role: str, label: str, default: str) -> tuple[float, float]:
    lo, hi = (float(x) for x in os.getenv(f"S1_BAND_{role.upper()}_{label.upper()}", default).split(","))
    return lo, hi

BANDS = {
    ("user", "injection"):      _band("user", "injection", "0.20,0.80"),
    ("user", "toxicity"):       _band("user", "toxicity", "0.30,0.70"),
    ("tool", "injection"):      _band("tool", "injection", "0.20,0.80"),
    ("assistant", "toxicity"):  _band("assistant", "toxicity", "0.30,0.70"),
}

# Session risk: once a session has had RISK_ESCALATE_AT blocks / withholds, its
# allow-threshold is multiplied by RISK_BAND_FACTOR (more goes to System 2) and
# its tool calls are judged by System 2 as well as tool_policy.
RISK_ESCALATE_AT = int(os.getenv("RISK_ESCALATE_AT", "1"))
RISK_BAND_FACTOR = float(os.getenv("RISK_BAND_FACTOR", "0.5"))


def s1_band(role: str, label: str, p: float, elevated: bool = False) -> str:
    """'allow' | 'block' | 'uncertain' for a System-1 probability."""
    t = threshold(label)
    lo, hi = BANDS.get((role, label), (t, t))
    if elevated:
        lo *= RISK_BAND_FACTOR
    if p >= hi:
        return "block"
    if p < lo:
        return "allow"
    return "uncertain"


def decide(category: str, triggered: bool, available: bool = True) -> str:
    """Return the action to take: 'allow', 'block', 'redact', 'warn', or 'log'."""
    pol = label_policy(category)
    if not available:
        return pol["action"] if pol["fail"] == "closed" else "allow"
    return pol["action"] if triggered else "allow"
