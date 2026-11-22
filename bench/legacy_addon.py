"""Faithful replica of the ORIGINAL firewall_addon.py control flow, kept for
bench_addon.py so the before/after comparison is against real code, not a
description of it. Differences from the original are only what is needed to run
here: it imports the new pii module's *synchronous* scan_and_redact (same
Presidio analyzer; the original used en_core_web_sm, which is slower still) and
emulates the old server-side 512-token truncation by cutting the text it sends
at 2048 characters."""
import json, os
import httpx
from mitmproxy import http
from canonicalize import canonicalize
from policy import THRESHOLDS, decide
from pii import scan_and_redact
import regex_rules

_PATH = "/v1/chat/completions"
_URL = os.getenv("GUARDRAILS_URL", "http://guardrails:8000")
_TIMEOUT = float(os.getenv("GUARDRAILS_TIMEOUT", "2.0"))
_TRUNC = 2048

async def classify(text, checks=("injection", "toxicity")):
    async with httpx.AsyncClient(timeout=_TIMEOUT) as client:          # new client per call (original)
        resp = await client.post(f"{_URL}/v1/classify", json={"text": text[:_TRUNC], "checks": list(checks)})
        resp.raise_for_status()
        return resp.json()["results"]

def _blocked(reason):
    body = {"id": "blocked-firewall", "object": "chat.completion", "model": "blocked",
            "choices": [{"index": 0, "message": {"role": "assistant",
                         "content": f"The request was blocked by the LLM firewall ({reason})."},
                         "finish_reason": "stop"}]}
    return http.Response.make(200, json.dumps(body), {"Content-Type": "application/json"})

def _user_text(payload):
    return "\n".join(m.get("content", "") for m in payload.get("messages", []) if m.get("role") == "user")

async def request(flow):
    if flow.request.method != "POST" or flow.request.path != _PATH:
        return
    try:
        payload = json.loads(flow.request.get_text())
    except Exception:
        return
    canon, decoded = canonicalize(_user_text(payload))
    scan_text = "\n".join([canon, *decoded])
    hit = regex_rules.match(scan_text)
    if hit:
        flow.response = _blocked(hit); return
    try:
        res = await classify(scan_text); available = True
    except Exception:
        res, available = {}, False
    if decide("injection", res.get("injection", {}).get("score", 0.0) >= THRESHOLDS["injection"], available) != "allow":
        flow.response = _blocked("prompt injection / jailbreak"); return
    if decide("toxicity", res.get("toxicity", {}).get("score", 0.0) >= THRESHOLDS["toxicity"], available) != "allow":
        flow.response = _blocked("toxic content"); return
    _, found = scan_and_redact(_user_text(payload))                     # sync, on the loop
    if found:
        action = decide("pii", True, True)
        if action == "block":
            flow.response = _blocked("sensitive data in prompt"); return
        if action == "redact":
            for m in payload.get("messages", []):
                if m.get("role") == "user":
                    m["content"], _ = scan_and_redact(m.get("content", ""))   # analysed again
            flow.request.set_text(json.dumps(payload))

async def response(flow):
    if flow.request.path != _PATH:
        return
    try:
        payload = json.loads(flow.response.get_text())
    except Exception:
        return
    changed = False
    for choice in payload.get("choices", []):
        msg = choice.get("message", {}); content = msg.get("content", "")
        if not content:
            continue
        try:
            res = await classify(content, checks=("toxicity",)); available = True
        except Exception:
            res, available = {}, False
        if decide("toxicity", res.get("toxicity", {}).get("score", 0.0) >= THRESHOLDS["toxicity"], available) != "allow":
            msg["content"] = "[response withheld by the LLM firewall: unsafe content]"; changed = True; continue
        redacted, found = scan_and_redact(content)
        if found:
            msg["content"] = redacted; changed = True
    if changed:
        flow.response.set_text(json.dumps(payload))
