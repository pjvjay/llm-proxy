"""LLM firewall -- mitmproxy addon.

Request path (per screened message, incremental -- see session.py):
    cache lookup -> canonicalize -> regex -> batched classifier calls (one per
    target, in parallel) running concurrently with PII -> decide -> forward
  * ``user`` messages: regex, injection, toxicity, PII   (block / redact)
  * ``tool`` messages: injection, PII                    (quarantine / redact)
Response path:
  * tool calls: deterministic tool policy (tool_policy.py)        -> block
  * text: toxicity + PII, all choices in one batch, concurrently  -> withhold / redact
  * JSON and SSE (``stream: true``) bodies are both screened

Fail-closed: injection, PII and tool-call categories block when their
dependency is down, and any unexpected exception in a hook blocks the request
(or withholds the response) instead of letting mitmproxy forward it unscreened.

Blocked/withheld requests still return a valid OpenAI chat.completion (HTTP 200,
SSE if the client asked to stream) so the caller's SDK doesn't raise."""
from __future__ import annotations

import asyncio
import json
import logging
import os
import time

from mitmproxy import http

import classifier_client
import pii
import regex_rules
import session
import sse
import tool_policy
from canonicalize import canonicalize
from messages import content_text, declared_tools, map_content_text
from policy import ML_CHECKS, TARGET_CHECKS, THRESHOLDS, decide

_PATH = "/v1/chat/completions"
_TIMING_HEADER = os.getenv("FIREWALL_TIMING_HEADER", "0") in ("1", "true", "yes")
QUARANTINE = "[tool output withheld by the LLM firewall: possible prompt injection]"
WITHHELD = "[response withheld by the LLM firewall: unsafe content]"

log = logging.getLogger("firewall")
log.setLevel(os.getenv("FIREWALL_LOG_LEVEL", "INFO"))


# ------------------------------------------------------------------ helpers
def _is_target(flow: http.HTTPFlow) -> bool:
    return (flow.request.method == "POST"
            and flow.request.path.split("?", 1)[0].rstrip("/") == _PATH)


def _completion(content: str, model: str = "blocked") -> dict:
    return {"id": "blocked-firewall", "object": "chat.completion", "model": model,
            "created": int(time.time()),
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": content}}]}


def _blocked(reason: str, stream: bool = False) -> http.Response:
    body = _completion(f"The request was blocked by the LLM firewall ({reason}).")
    if stream:
        return http.Response.make(200, sse.render(body), {"Content-Type": "text/event-stream"})
    return http.Response.make(200, json.dumps(body), {"Content-Type": "application/json"})


def _meta(flow) -> dict:
    return flow.metadata.setdefault("firewall", {})


async def _ml(items):
    """Batched classifier call. -> (results | None, available)"""
    if not items:
        return [], True
    try:
        return await classifier_client.classify_batch(items), True
    except Exception as e:                          # noqa: BLE001 -- any failure = unavailable
        log.warning("guardrails unavailable: %r", e)
        return None, False


def _score(rec, check) -> float:
    return float(((rec or {}).get("scores") or {}).get(check, 0.0))


# ------------------------------------------------------------------ request
async def request(flow: http.HTTPFlow):
    if not _is_target(flow):
        return
    t0 = time.perf_counter()
    try:
        await _screen_request(flow)
    except Exception:                               # fail closed, never forward unscreened
        log.exception("firewall request hook failed; blocking")
        flow.response = _blocked("internal firewall error", _meta(flow).get("stream", False))
    finally:
        m = _meta(flow)
        m["req_ms"] = (time.perf_counter() - t0) * 1000
        if flow.response is not None:
            m["local"] = True                       # response generated here, not upstream
            _log(flow)


async def _screen_request(flow: http.HTTPFlow):
    meta = _meta(flow)
    try:
        payload = json.loads(flow.request.get_text())
    except (ValueError, UnicodeDecodeError):
        flow.response = _blocked("unparseable request body")
        return
    if not isinstance(payload, dict) or not isinstance(payload.get("messages"), list):
        flow.response = _blocked("malformed request")
        return
    meta.update(stream=bool(payload.get("stream")), declared=declared_tools(payload),
                session=session.session_id(flow.request.headers, payload))
    stream = meta["stream"]
    messages = payload["messages"]
    units = session.scan_units(messages)

    # 1. cache lookup ----------------------------------------------------------
    records, todo = {}, []
    for u in units:
        need = tuple(c for c in ML_CHECKS if c in TARGET_CHECKS[u.target])
        rec = session.SCAN_CACHE.get(u.key) if session.ENABLED else None
        if rec is not None and rec["regex"]:
            flow.response = _blocked(rec["regex"], stream)     # cached keyword hit
            return
        if rec is not None and all(c in rec["scores"] for c in need):
            records[u.index] = rec
        else:
            todo.append((u, need))

    # 2. canonicalize + regex on new units (cheap; short-circuits ML) ----------
    ml_items, ml_units = [], []
    for u, need in todo:
        canon, decoded = canonicalize(u.scan_text)
        scan_text = "\n".join([canon, *decoded])
        hit = regex_rules.match(scan_text) if "regex" in TARGET_CHECKS[u.target] else None
        if hit:
            rec = {"regex": hit, "scores": {}}
            records[u.index] = rec
            if session.ENABLED:
                session.SCAN_CACHE.put(u.key, rec, 64)
            flow.response = _blocked(hit, stream)
            return
        if need:
            ml_items.append((scan_text, need))
            ml_units.append(u)
        else:
            records[u.index] = {"regex": None, "scores": {}}

    # 3. classifiers concurrently with PII ---------------------------------------
    # One batched request PER TARGET, in parallel: prompts are short and fail
    # closed (block); tool results can be long and fail to quarantine. Sharing
    # one request let a slow tool result time out the prompt check and block the
    # whole call (B6).
    groups: dict[str, tuple[list, list]] = {}
    for item, u in zip(ml_items, ml_units):
        its, us = groups.setdefault(u.target, ([], []))
        its.append(item)
        us.append(u)
    ml_tasks = {t: asyncio.create_task(_ml(its)) for t, (its, _) in groups.items()}
    pii_units = [u for u in units if "pii" in TARGET_CHECKS[u.target]]
    pii_res = await asyncio.gather(*(_pii_cached(u) for u in pii_units))
    partial = 0
    for target, task in ml_tasks.items():
        ml_res, available = await task
        if not available:
            continue                           # these units keep rec=None -> fail mode
        for u, r in zip(groups[target][1], ml_res):
            partial += any(v.get("truncated") for v in r.values())
            rec = {"regex": None, "scores": {c: float(v["score"]) for c, v in r.items()}}
            records[u.index] = rec
            if session.ENABLED:
                session.SCAN_CACHE.put(u.key, rec, 64)
    if partial:
        log.info("session=%s %d message(s) exceeded the classifier window budget; "
                 "scored head+tail only", meta.get("session"), partial)
    meta.update(units=len(units), scanned=len(ml_units), pii_scanned=len(pii_units))

    # 4. decisions -------------------------------------------------------------
    changed, quarantined = False, set()
    for u in units:
        rec = records.get(u.index)
        ok = rec is not None                   # False only if its ML call failed
        if u.target == "prompt":
            if "injection" in TARGET_CHECKS["prompt"] and decide(
                    "injection", _score(rec, "injection") >= THRESHOLDS["injection"], ok) == "block":
                flow.response = _blocked("prompt injection / jailbreak", stream)
                return
            if "toxicity" in TARGET_CHECKS["prompt"] and decide(
                    "toxicity", _score(rec, "toxicity") >= THRESHOLDS["toxicity"], ok) == "block":
                flow.response = _blocked("toxic content", stream)
                return
        elif u.target == "tool_result" and "injection" in TARGET_CHECKS["tool_result"]:
            action = decide("tool_result_injection",
                            _score(rec, "injection") >= THRESHOLDS["injection"], ok)
            if action == "block":
                flow.response = _blocked("prompt injection in tool result", stream)
                return
            if action == "redact":
                messages[u.index]["content"] = QUARANTINE
                quarantined.add(u.index)
                changed = True

    for u, (redacted, found) in zip(pii_units, pii_res):
        if not found or u.index in quarantined:
            continue
        action = decide("pii", True, True)
        if action == "block":
            flow.response = _blocked("sensitive data in prompt", stream)
            return
        if action == "redact":
            msg = messages[u.index]
            if isinstance(msg.get("content"), str):
                msg["content"] = redacted
            else:                                  # content parts: redact part by part
                msg["content"] = map_content_text(msg.get("content"),
                                                  lambda s: pii.scan_and_redact(s)[0])
            changed = True

    if changed:
        flow.request.set_text(json.dumps(payload))


async def _pii_cached(u) -> tuple[str, list[str]]:
    if session.ENABLED:
        hit = session.PII_CACHE.get(u.pii_key)
        if hit is not None:
            return hit
    res = await pii.scan_and_redact_async(u.text)
    if session.ENABLED:
        session.PII_CACHE.put(u.pii_key, res, len(res[0]) if res[1] else 0)
    return res


# ----------------------------------------------------------------- response
async def response(flow: http.HTTPFlow):
    if not _is_target(flow) or flow.response is None:
        return
    meta = _meta(flow)
    if meta.get("local") or flow.response.status_code != 200:
        return                          # our own block message, or an upstream error
    t0 = time.perf_counter()
    try:
        await _screen_response(flow)
    except Exception:
        log.exception("firewall response hook failed; withholding")
        body = _completion(WITHHELD)
        if "text/event-stream" in flow.response.headers.get("content-type", ""):
            flow.response.set_text(sse.render(body))
        else:
            flow.response.set_text(json.dumps(body))
    finally:
        meta["resp_ms"] = (time.perf_counter() - t0) * 1000
        _log(flow)


async def _screen_response(flow: http.HTTPFlow):
    meta = _meta(flow)
    streamed = "text/event-stream" in flow.response.headers.get("content-type", "")
    raw = flow.response.get_text()
    payload = sse.aggregate(raw) if streamed else json.loads(raw)
    if not payload:
        return                          # SSE without completion chunks (upstream error)
    choices = payload.get("choices") or []
    changed = False

    # 1. tool calls: deterministic, microseconds -------------------------------
    blocked_choices = set()
    for ch in choices:
        msg = ch.get("message") or {}
        calls = tool_policy.tool_calls_of(msg)
        if not calls:
            continue
        verdicts = [tool_policy.check_tool_call(n, a, meta.get("declared")) for n, a in calls]
        bad = [v for v in verdicts if v.blocked]
        if not bad:
            continue
        why = "; ".join(f"{v.name or '?'}: {', '.join(v.violations)}" for v in bad)
        action = decide("tool_call", True)
        log.warning("tool call flagged (%s) session=%s: %s", action, meta.get("session"), why)
        if action == "block":
            # drop ALL calls of the choice: parallel calls may be coordinated
            msg.pop("tool_calls", None)
            msg.pop("function_call", None)
            msg["content"] = f"[tool call blocked by the LLM firewall: {why}]"
            ch["finish_reason"] = "stop"
            blocked_choices.add(id(ch))
            changed = True

    # 2. text: toxicity (one batch) concurrently with PII ----------------------
    texts = [(ch, content_text((ch.get("message") or {}).get("content")))
             for ch in choices if id(ch) not in blocked_choices]
    texts = [(ch, t) for ch, t in texts if t]
    if texts:
        ml_task = asyncio.create_task(_ml([(t, ("toxicity",)) for _, t in texts]))
        pii_res = await asyncio.gather(*(pii.scan_and_redact_async(t) for _, t in texts))
        ml_res, available = await ml_task
        for k, (ch, _) in enumerate(texts):
            msg = ch["message"]
            tox = float(ml_res[k]["toxicity"]["score"]) if available else 0.0
            if decide("toxicity", tox >= THRESHOLDS["toxicity"], available) != "allow":
                msg["content"] = WITHHELD
                changed = True
                continue
            redacted, found = pii_res[k]
            if found and decide("pii", True, True) != "allow":
                msg["content"] = redacted
                changed = True

    if changed:
        flow.response.set_text(sse.render(payload) if streamed else json.dumps(payload))


# ------------------------------------------------------------------ logging
def _log(flow):
    m = _meta(flow)
    if _TIMING_HEADER and flow.response is not None:
        flow.response.headers["x-firewall-ms"] = \
            f"req={m.get('req_ms', 0):.2f};resp={m.get('resp_ms', 0):.2f}"
    log.info("session=%s local=%s units=%s scanned=%s req_ms=%.1f resp_ms=%.1f cache_hit_rate=%.2f",
             m.get("session"), bool(m.get("local")), m.get("units"), m.get("scanned"), m.get("req_ms", 0),
             m.get("resp_ms", 0),
             session.SCAN_CACHE.hits / max(1, session.SCAN_CACHE.hits + session.SCAN_CACHE.misses))


def running():
    """mitmproxy lifecycle hook: warm the PII worker pool in the background."""
    pii.start_pool()


async def done():
    await classifier_client.aclose()
