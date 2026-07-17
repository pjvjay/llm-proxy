"""LLM firewall -- mitmproxy addon.

Request path : canonicalize -> regex -> parallel(injection, toxicity) -> PII redact
Response path: toxicity + PII redact on model output
Fail-closed on injection & PII; toxicity fail mode is configurable (see policy.py).

Blocked/withheld requests still return a valid OpenAI chat.completion (HTTP 200)
so the caller's SDK doesn't raise."""
import json

from mitmproxy import http

from canonicalize import canonicalize
from classifier_client import classify
from policy import THRESHOLDS, decide
from pii import scan_and_redact
import regex_rules

_PATH = "/v1/chat/completions"

def _blocked(reason: str) -> http.Response:
    body = {
        "id": "blocked-firewall", "object": "chat.completion", "model": "blocked",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant",
                        "content": f"The request was blocked by the LLM firewall ({reason})."},
            "finish_reason": "stop",
        }],
    }
    return http.Response.make(200, json.dumps(body),
                             {"Content-Type": "application/json"})

def _user_text(payload: dict) -> str:
    return "\n".join(m.get("content", "") for m in payload.get("messages", [])
                     if m.get("role") == "user")

async def request(flow: http.HTTPFlow):
    if flow.request.method != "POST" or flow.request.path != _PATH:
        return
    try:
        payload = json.loads(flow.request.get_text())
    except Exception:
        return

    canon, decoded = canonicalize(_user_text(payload))
    scan_text = "\n".join([canon, *decoded])

    # Stage 1 -- regex pre-filter (fail-closed category)
    hit = regex_rules.match(scan_text)
    if hit:
        flow.response = _blocked(hit)
        return

    # Stage 2 -- ML classifiers (run in parallel server-side); fail-closed on injection
    try:
        res = await classify(scan_text)
        available = True
    except Exception:
        res, available = {}, False

    if decide("injection",
              res.get("injection", {}).get("score", 0.0) >= THRESHOLDS["injection"],
              available) != "allow":
        flow.response = _blocked("prompt injection / jailbreak")
        return

    if decide("toxicity",
              res.get("toxicity", {}).get("score", 0.0) >= THRESHOLDS["toxicity"],
              available) != "allow":
        flow.response = _blocked("toxic content")
        return

    # Stage 3 -- PII / secrets: redact before forwarding to the model (or block)
    _, found = scan_and_redact(_user_text(payload))
    if found:
        action = decide("pii", True, True)
        if action == "block":
            flow.response = _blocked("sensitive data in prompt")
            return
        if action == "redact":
            for m in payload.get("messages", []):
                if m.get("role") == "user":
                    m["content"], _ = scan_and_redact(m.get("content", ""))
            flow.request.set_text(json.dumps(payload))

async def response(flow: http.HTTPFlow):
    if flow.request.path != _PATH:
        return
    try:
        payload = json.loads(flow.response.get_text())
    except Exception:
        return

    changed = False
    for choice in payload.get("choices", []):
        msg = choice.get("message", {})
        content = msg.get("content", "")
        if not content:
            continue

        try:
            res = await classify(content, checks=("toxicity",))
            available = True
        except Exception:
            res, available = {}, False

        if decide("toxicity",
                  res.get("toxicity", {}).get("score", 0.0) >= THRESHOLDS["toxicity"],
                  available) != "allow":
            msg["content"] = "[response withheld by the LLM firewall: unsafe content]"
            changed = True
            continue

        redacted, found = scan_and_redact(content)
        if found:
            msg["content"] = redacted
            changed = True

    if changed:
        flow.response.set_text(json.dumps(payload))
