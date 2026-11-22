"""Per-category action + fail mode. P0 keeps this a small dict; the P1 YAML
policy engine slots in behind the same decide() call.

Fail mode matters: if the classifier service is unreachable, injection and PII
categories FAIL CLOSED (block), toxicity fails open by default (configurable) so
a guardrail outage doesn't take down all legitimate traffic.

Agentic categories:
  tool_result_injection -- an injection detected inside a ``role: tool``
      message. Default action ``redact`` = quarantine: that tool output is
      replaced with a notice and the call proceeds, so the agent can recover
      instead of the whole run failing. ``block`` rejects the call instead.
  tool_call -- a tool call returned by the model that violates tool_policy.
"""
import os

DEFAULT = {
    "injection": {"action": "block", "fail": "closed"},
    "toxicity":  {"action": os.getenv("TOXICITY_ACTION", "block"),
                  "fail":   os.getenv("TOXICITY_FAIL", "open")},
    "pii":       {"action": os.getenv("PII_ACTION", "redact"), "fail": "closed"},
    "regex":     {"action": "block", "fail": "closed"},
    "tool_result_injection": {"action": os.getenv("TOOL_RESULT_INJECTION_ACTION", "redact"),
                              "fail": "closed"},
    "tool_call": {"action": os.getenv("TOOL_CALL_ACTION", "block"), "fail": "closed"},
}

THRESHOLDS = {
    "injection": float(os.getenv("INJECTION_THRESHOLD", "0.5")),
    "toxicity":  float(os.getenv("TOXICITY_THRESHOLD", "0.5")),
}


def _checks(env, default):
    return tuple(c.strip() for c in os.getenv(env, default).split(",") if c.strip())


# Which checks run on which request-path target. Tool results get injection +
# PII by default: keyword / toxicity rules on arbitrary web pages, files and
# API output would mostly produce false positives (a news article mentions
# "murder"); the response-path toxicity check still covers what the model says.
TARGET_CHECKS = {
    "prompt":      _checks("PROMPT_CHECKS", "regex,injection,toxicity,pii"),
    "tool_result": _checks("TOOL_RESULT_CHECKS", "injection,pii"),
}
ML_CHECKS = ("injection", "toxicity")


def decide(category: str, triggered: bool, available: bool = True) -> str:
    """Return the action to take: 'allow', 'block', 'redact', 'warn', or 'log'."""
    pol = DEFAULT[category]
    if not available:
        return pol["action"] if pol["fail"] == "closed" else "allow"
    return pol["action"] if triggered else "allow"
