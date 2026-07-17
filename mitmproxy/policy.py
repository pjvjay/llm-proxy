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
}

THRESHOLDS = {
    "injection": float(os.getenv("INJECTION_THRESHOLD", "0.5")),
    "toxicity":  float(os.getenv("TOXICITY_THRESHOLD", "0.5")),
}

def decide(category: str, triggered: bool, available: bool = True) -> str:
    """Return the action to take: 'allow', 'block', 'redact', 'warn', or 'log'."""
    pol = DEFAULT[category]
    if not available:
        return pol["action"] if pol["fail"] == "closed" else "allow"
    return pol["action"] if triggered else "allow"
