"""LLM firewall -- mitmproxy addon. Dual-process screening for agentic traffic,
with every evaluation mapped to the OWASP Top 10 for LLM Applications (2025).

System 1 -- every *new* text unit (user message / tool result), inline:
    limits (LLM10) -> canonicalize -> regex (user only)
      -> [ decision model || PII ] -> confidence band per (role, label)
    The decision model (guardrails /v1/decide: the stock classifiers, or a
    System-1 model such as an OpenJev export in models/decision) returns a
    probability for every label relevant to the unit's role in ONE round-trip.
System 2 -- only the uncertain (unit, label) pairs:
    LLM judge (system2.py), inline or raced against the upstream call
    (S2_MODE). Verdicts are cached with the unit; overload / timeouts fall back
    to System 1's leaning.
Response path:
    assistant text  -> [ System-1 toxicity || PII ] -> System 2 if uncertain
                    -> system-prompt leak (LLM07) -> active markup / exfil links (LLM05)
    tool_calls      -> tool_policy (LLM06) + System-2 intent check once the
                       session's risk is elevated

Every check lands in a per-request audit.Evaluation with its OWASP ids; the
summary goes to X-LLM-Firewall-* headers, a `firewall` object in responses the
firewall produced or modified, the audit log and /firewall/metrics.

Blocked / withheld calls still return a valid OpenAI chat.completion (HTTP 200)
so SDKs don't raise. `stream: true` is de-streamed upstream, screened, and
re-emitted as SSE. Unscreened POST endpoints are denied by default. Unexpected
exceptions fail closed (FIREWALL_ON_ERROR)."""
import asyncio
import json
import logging
import os
import time
from collections import Counter

from mitmproxy import http

import audit
import content as C
import limits
import output_checks
import owasp
import pii
import regex_rules
import session
import system2
import tool_policy
from canonicalize import canonicalize
from classifier_client import decide_many
from policy import RISK_ESCALATE_AT, decide, s1_band, threshold

log = logging.getLogger("firewall")

_PATH = "/v1/chat/completions"
_INTERNAL = "/firewall/"


def _labels_env(name: str, default: str) -> tuple[str, ...]:
    return tuple(c.strip() for c in os.getenv(name, default).split(",") if c.strip())


# Labels requested from the decision model per role. Extend these when a
# System-1 model exposes more heads (e.g. LABELS_USER=injection,toxicity,jailbreak).
_LABELS = {
    "user": _labels_env("LABELS_USER", "injection,toxicity"),
    "tool": _labels_env("LABELS_TOOL", os.getenv("TOOL_RESULT_CHECKS", "injection")),
    "assistant": _labels_env("LABELS_ASSISTANT", "toxicity"),
}
_INJECTION_LIKE = {"injection", "jailbreak", "system_prompt_extraction"}
_REASONS = {"injection": "prompt injection / jailbreak", "jailbreak": "prompt injection / jailbreak",
            "toxicity": "toxic content", "system_prompt_extraction": "system prompt extraction"}
# Tools without side effects are low-stakes: once a session's risk is elevated,
# only calls to tools NOT listed here get the System-2 intent check.
_READONLY_TOOLS = {t for t in os.getenv("TOOL_READONLY", "search,read_file,list_files").split(",") if t}
# POST endpoints that are forwarded WITHOUT screening. Everything else except
# the chat-completions path is denied: a legacy /v1/completions call would
# otherwise be a way around every check.
_UNSCREENED_ALLOW = {p.strip() for p in os.getenv("UNSCREENED_POST_ALLOW", "").split(",") if p.strip()}
_ANNOTATE = os.getenv("FIREWALL_ANNOTATE", "0") == "1"

_WITHHELD_TOOL = "[tool result withheld by the LLM firewall: prompt injection]"
_WITHHELD_TEXT = "[response withheld by the LLM firewall: unsafe content]"
_WITHHELD_LEAK = "[response withheld by the LLM firewall: it reproduced the system prompt]"
_WITHHELD_RACE = ("[response withheld by the LLM firewall: a tool result in this turn was "
                  "flagged as prompt injection]")

_cache = session.VerdictCache()
_sessions = session.SessionStore()
STATS: Counter = Counter()          # operational counters; also read by the controlled benchmarks


# ----------------------------------------------------------------- helpers
def _route(flow: http.HTTPFlow) -> str:
    return flow.request.path.split("?", 1)[0].rstrip("/") or "/"


def _ev(flow: http.HTTPFlow) -> audit.Evaluation:
    ev = flow.metadata.get("fw_eval")
    if ev is None:
        ev = flow.metadata["fw_eval"] = audit.Evaluation(flow.metadata.get("fw_session"))
    return ev


def _target(u: C.Unit) -> str:
    if u.role == "assistant":
        return f"choices[{u.msg_idx}]"
    return f"messages[{u.msg_idx}]" + ("" if u.part_idx is None else f".content[{u.part_idx}]")


def _completion(text: str, model: str = "blocked", id_: str = "blocked-firewall") -> dict:
    return {
        "id": id_, "object": "chat.completion", "created": int(time.time()), "model": model,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": text},
                     "finish_reason": "stop"}],
    }


def _to_sse(payload: dict, include_usage: bool = False) -> str:
    """Re-emit a full chat.completion as a chat.completion.chunk stream. The
    trailing usage chunk (choices: []) is sent only if the client asked for it
    with stream_options.include_usage, as OpenAI does -- clients that index
    chunk.choices[0] would otherwise break on it."""
    base = {"id": payload.get("id", "chatcmpl-firewall"), "object": "chat.completion.chunk",
            "created": payload.get("created", int(time.time())),
            "model": payload.get("model", "")}
    events, first = [], True
    for ch in payload.get("choices", []):
        msg = ch.get("message", {}) or {}
        delta = {"role": msg.get("role", "assistant"), "content": msg.get("content")}
        if msg.get("tool_calls"):
            delta["tool_calls"] = [{"index": i, **tc} for i, tc in enumerate(msg["tool_calls"])]
        ev = {**base, "choices": [{"index": ch.get("index", 0), "delta": delta, "finish_reason": None}]}
        if first and "firewall" in payload:
            ev["firewall"] = payload["firewall"]          # extra field; SDKs ignore unknown keys
        first = False
        events.append(ev)
        events.append({**base, "choices": [{"index": ch.get("index", 0), "delta": {},
                                            "finish_reason": ch.get("finish_reason", "stop")}]})
    if include_usage and "usage" in payload:
        events.append({**base, "choices": [], "usage": payload["usage"]})
    return "".join(f"data: {json.dumps(e)}\n\n" for e in events) + "data: [DONE]\n\n"


def _emit(flow: http.HTTPFlow, payload: dict) -> None:
    """Send a response the firewall produced or modified (JSON or SSE), with
    the evaluation attached."""
    ev = _ev(flow)
    payload["firewall"] = ev.summary()
    flow.metadata["fw_done"] = True
    headers = {"Cache-Control": "no-cache", **ev.headers()}
    if flow.metadata.get("fw_stream"):
        flow.response = http.Response.make(200, _to_sse(payload, flow.metadata.get("fw_include_usage", False)),
                                           {"Content-Type": "text/event-stream",
                                            "X-Accel-Buffering": "no", **headers})
    else:
        flow.response = http.Response.make(200, json.dumps(payload),
                                           {"Content-Type": "application/json", **headers})
    audit.record(ev)


def _risk(flow: http.HTTPFlow) -> None:
    sid = flow.metadata.get("fw_session")
    if sid:
        _sessions.bump_risk(sid)


def _block(flow: http.HTTPFlow, reason: str, ids=()) -> None:
    ev = _ev(flow)
    ev.escalate("block", reason)
    flagged = sorted(set(ids) | set(ev.flagged()))
    tag = f"; OWASP {owasp.label(flagged)}" if flagged else ""
    log.warning("blocked session=%s reason=%s owasp=%s", flow.metadata.get("fw_session"), reason,
                ",".join(flagged) or "-")
    STATS["blocked"] += 1
    _risk(flow)
    _emit(flow, _completion(f"The request was blocked by the LLM firewall ({reason}{tag})."))


def _deny_endpoint(flow: http.HTTPFlow, route: str) -> None:
    ev = _ev(flow)
    ev.add("endpoint", "block", detail=f"POST {route} is not screened by the firewall", owasp_ids=[])
    ev.escalate("block", "unscreened endpoint")
    flow.metadata["fw_done"] = True
    flow.response = http.Response.make(
        403, json.dumps({"error": {"message": f"The LLM firewall does not screen {route}; denied. "
                                              "Allow it with UNSCREENED_POST_ALLOW if intended.",
                                   "type": "firewall_denied", "code": "endpoint_not_screened"}}),
        {"Content-Type": "application/json", **ev.headers()})
    audit.record(ev)


# ----------------------------------------------------------------- System 1
def _cacheable(sig: dict) -> dict:
    return {k: v for k, v in sig.items() if k != "scan"}


async def _signals_for(units: list[C.Unit]) -> None:
    """Fill u.signals for every unit: cached ones are reused; new ones get one
    decision-model round-trip (all units, all labels) || one PII job."""
    new = [u for u in units if u.cached is None]
    for u in units:
        if u.cached is not None:
            u.signals = u.cached
    STATS["units_cached"] += len(units) - len(new)
    if not new:
        return
    for u in new:
        canon, decoded = canonicalize(u.text)
        u.signals = {"scan": "\n".join([canon, *decoded])}
        u.signals["regex"] = regex_rules.match(u.signals["scan"]) if u.role == "user" else None

    s1_res, pii_res = await asyncio.gather(
        decide_many([(u.signals["scan"], u.role, u.tool, _LABELS[u.role]) for u in new]),
        pii.scan_many_async([u.text for u in new]),
        return_exceptions=True)
    if isinstance(pii_res, BaseException):
        raise pii_res                                   # PII is fail-closed -> error policy
    available = not isinstance(s1_res, BaseException)
    if not available:
        log.error("decision service unavailable: %r", s1_res)
        STATS["s1_unavailable"] += 1
    else:
        STATS["s1_calls"] += 1
    STATS["units_new"] += len(new)
    for i, u in enumerate(new):
        if available:
            d = s1_res[i]
            u.signals["s1"] = d.get("scores", {})
            u.signals["windows"] = d.get("windows", 0)
            if d.get("owasp"):
                u.signals["owasp_map"] = d["owasp"]        # declared by the decision model
            STATS["s1_passes"] += d.get("passes", 0)
        else:
            u.signals["s1"] = None
        u.signals["redacted"], u.signals["entities"] = pii_res[i]
        if available:                                    # never cache a partial result
            _cache.put(u.key, _cacheable(u.signals))


def _outcome(u: C.Unit, label: str, elevated: bool) -> str:
    """System-1 outcome: allow | block | uncertain | unavailable."""
    s1 = u.signals.get("s1")
    if s1 is None or label not in s1:
        return "unavailable"
    return s1_band(u.role, label, s1[label], elevated)


def _final(u: C.Unit, label: str, elevated: bool) -> tuple[bool, bool]:
    """(triggered, available) after System 2, for policy.decide()."""
    o = _outcome(u, label, elevated)
    if o == "unavailable":
        return False, False
    if o != "uncertain":
        return o == "block", True
    s2 = (u.signals.get("s2") or {}).get(label)
    if s2 is not None:
        return s2 >= system2.THRESHOLD, True
    return u.signals["s1"][label] >= threshold(label), True   # System 1's leaning


def _enforce(u: C.Unit, label: str, elevated: bool) -> bool:
    triggered, available = _final(u, label, elevated)
    action = decide(label, triggered, available)
    if action in ("warn", "log"):
        log.warning("%s %s on %s unit (not enforced)", action, label, u.role)
    return action == "block"


def _label_ids(u: C.Unit, label: str) -> list[str]:
    ids = owasp.ids_for(label, u.role, u.tool, u.signals.get("owasp_map"))
    if label in _INJECTION_LIKE and u.role == "user" and output_checks.extraction_attempt(u.text):
        ids.append("LLM07:2025")            # asks for the system prompt: also System Prompt Leakage
    return sorted(set(ids))


def _record_label(ev: audit.Evaluation, u: C.Unit, label: str, elevated: bool, decision: str) -> None:
    s1 = u.signals.get("s1") or {}
    ev.add(label, decision, role=u.role, target=_target(u), tool=u.tool, score=s1.get(label),
           s1=_outcome(u, label, elevated), s2=(u.signals.get("s2") or {}).get(label),
           owasp_ids=_label_ids(u, label), cached=u.cached is not None)


# ----------------------------------------------------------------- System 2
def _escalations(units: list[C.Unit], elevated: bool) -> list[tuple[C.Unit, str]]:
    if not system2.enabled():
        return []
    out = []
    for u in units:
        if u.role == "user" and u.signals.get("regex"):
            continue                                     # already decided by the regex stage
        for label in _LABELS[u.role]:
            if _outcome(u, label, elevated) == "uncertain" and \
                    label not in (u.signals.get("s2") or {}):
                out.append((u, label))
    return out


async def _escalate(esc: list[tuple[C.Unit, str]]) -> None:
    verdicts = await asyncio.gather(*(
        system2.judge(u.role, label, u.text, s1=u.signals["s1"][label], key=u.key)
        for u, label in esc))
    for (u, label), v in zip(esc, verdicts):
        STATS[f"s2_{v.status}"] += 1
        if v.p is not None:
            u.signals.setdefault("s2", {})[label] = v.p
            _cache.put(u.key, _cacheable(u.signals))


# ----------------------------------------------------------------- request
def _record_pii(ev: audit.Evaluation, u: C.Unit, action: str | None = None) -> None:
    if u.signals.get("entities"):
        action = action or decide("pii" if u.role == "user" else "pii_tool", True, True)
        ev.add("pii", action if action in ("redact", "block") else "log", role=u.role,
               target=_target(u), detail=",".join(u.signals["entities"]), cached=u.cached is not None)
    else:
        ev.add("pii", "allow", role=u.role, target=_target(u), cached=u.cached is not None)


def _apply(flow, msgs, u: C.Unit, labels, elevated: bool) -> str | None:
    """Decide and record the given labels for one request unit. Every label's
    own verdict is recorded even when another label blocks, so the evaluation
    covers every check that ran. Returns 'blocked', 'changed' or None."""
    ev = _ev(flow)
    verdicts = [(label, _enforce(u, label, elevated)) for label in labels]
    blocking = [label for label, hit in verdicts if hit]
    if u.role == "user":
        for label, hit in verdicts:
            _record_label(ev, u, label, elevated, "block" if hit else "allow")
        if blocking:
            _record_pii(ev, u)                          # computed in parallel; record it too
            ids = sorted(set().union(*(_label_ids(u, l) for l in blocking)))
            _block(flow, _REASONS.get(blocking[0], blocking[0]), ids)
            return "blocked"
        return None
    for label, hit in verdicts:
        if not hit:
            _record_label(ev, u, label, elevated, "allow")
    if not blocking:
        return None
    label = blocking[0]
    if decide("tool_result", True, True) == "block":
        _record_label(ev, u, label, elevated, "block")
        _block(flow, "prompt injection in tool result", _label_ids(u, label))
        return "blocked"
    _record_label(ev, u, label, elevated, "withhold")
    ev.escalate("withhold", "prompt injection in tool result")
    log.warning("withheld tool result session=%s msg=%d", flow.metadata.get("fw_session"), u.msg_idx)
    STATS["withheld_tool"] += 1
    _risk(flow)
    C.set_unit_text(msgs, u, _WITHHELD_TOOL)
    return "changed"


async def _screen_request(flow: http.HTTPFlow, payload: dict) -> None:
    msgs = payload.get("messages") or []
    sid, derived = session.session_id(flow.request.headers, payload)
    flow.metadata["fw_session"] = sid
    ev = _ev(flow)
    ev.session = sid

    # OWASP LLM10 -- before any model is called
    changed = False
    for f in limits.check(len(flow.request.raw_content or b""), payload, sid):
        ev.add("consumption", f["action"], detail=f["detail"])
        if f["action"] == "block":
            return _block(flow, f"resource limit: {f['detail']}", owasp.ids_for("consumption"))
        ev.escalate("clamp", f["detail"])
        changed = True

    state = _sessions.touch(sid, session.message_key("system", C.system_text(msgs)), derived)
    if state.system_changed:
        log.warning("system prompt changed mid-session session=%s turn=%d", sid, state.turn)
    elevated = state.risk >= RISK_ESCALATE_AT
    flow.metadata["fw_elevated"] = elevated

    units = session.plan(_cache, msgs, {"user", "tool", "function"})
    t0 = time.perf_counter()
    await _signals_for(units)
    esc = _escalations(units, elevated)
    pending = {(id(u), label) for u, label in esc}

    # Pass 1: System-1-confident decisions and PII, skipping pairs awaiting System 2.
    for u in units:
        if u.role == "user" and u.signals.get("regex"):
            ev.add("regex", "block", role="user", target=_target(u), detail=u.signals["regex"],
                   cached=u.cached is not None)
            for label in _LABELS["user"]:               # model scores were computed too: record them
                if (u.signals.get("s1") or {}).get(label) is not None:
                    _record_label(ev, u, label, elevated, "block" if _enforce(u, label, elevated) else "allow")
            _record_pii(ev, u)
            return _block(flow, u.signals["regex"])
        labels = [l for l in _LABELS[u.role] if (id(u), l) not in pending]
        r = _apply(flow, msgs, u, labels, elevated)
        if r == "blocked":
            return
        if r == "changed":
            changed = True
            continue
        action = decide("pii" if u.role == "user" else "pii_tool", True, True) \
            if u.signals.get("entities") else None
        _record_pii(ev, u, action)
        if action == "block":
            return _block(flow, "sensitive data in prompt", owasp.ids_for("pii"))
        if action == "redact":
            ev.escalate("redact", "sensitive data redacted")
            C.set_unit_text(msgs, u, u.signals["redacted"])
            changed = True

    # Pass 2: System 2 for the uncertain pairs -- raced against upstream or inline.
    if esc:
        STATS["escalations"] += len(esc)
        if system2.MODE == "race":
            flow.metadata["fw_race"] = (asyncio.create_task(_escalate(esc)), esc)
        else:
            await _escalate(esc)
            for u, label in esc:
                r = _apply(flow, msgs, u, [label], elevated)
                if r == "blocked":
                    return
                changed = changed or r == "changed"

    if changed:
        flow.request.set_text(json.dumps(payload))
    log.info("request session=%s turn=%d units=%d new=%d escalated=%d gate_ms=%.1f",
             sid, state.turn, len(units), sum(1 for u in units if u.cached is None),
             len(esc), (time.perf_counter() - t0) * 1000)


# ----------------------------------------------------------------- response
async def _settle_race(flow: http.HTTPFlow) -> bool:
    """Apply raced System-2 verdicts. True if the response was replaced."""
    race = flow.metadata.pop("fw_race", None)
    if not race:
        return False
    task, esc = race
    await task                                          # judge() enforces its own deadline
    elevated = flow.metadata.get("fw_elevated", False)
    ev = _ev(flow)
    for u, label in esc:
        if not _enforce(u, label, elevated):
            _record_label(ev, u, label, elevated, "allow")
            continue
        STATS["race_replaced"] += 1
        if u.role == "user":
            _record_label(ev, u, label, elevated, "block")
            _block(flow, _REASONS.get(label, label), _label_ids(u, label))
        else:
            _record_label(ev, u, label, elevated, "withhold")
            ev.escalate("withhold", "prompt injection in tool result")
            STATS["withheld_tool"] += 1
            _risk(flow)
            _emit(flow, _completion(_WITHHELD_RACE))
        return True
    return False


async def _screen_response(flow: http.HTTPFlow, payload: dict) -> None:
    if await _settle_race(flow):
        return
    ev = _ev(flow)
    elevated = flow.metadata.get("fw_elevated", False)
    req = None
    try:
        req = json.loads(flow.request.get_text())
    except Exception:
        req = {}
    req_msgs = req.get("messages") or []
    sys_text = C.system_text(req_msgs)

    choices = payload.get("choices") or []
    pairs = []
    for ch in choices:
        msg = ch.get("message") or {}
        text = msg.get("content")
        if isinstance(text, str) and text:
            u = C.Unit(ch.get("index", 0), None, "assistant", text)
            u.key = session.message_key("assistant", text)
            u.cached = _cache.get(u.key)
            pairs.append((ch, u))
    units = [u for _, u in pairs]
    await _signals_for(units)
    esc = _escalations(units, elevated)
    if esc:                                             # response is already here: judge inline
        STATS["escalations"] += len(esc)
        await _escalate(esc)

    changed = False
    for ch, u in pairs:
        msg = ch["message"]
        withheld = False
        for label in _LABELS["assistant"]:
            if _enforce(u, label, elevated):
                _record_label(ev, u, label, elevated, "withhold")
                ev.escalate("withhold", "unsafe response content")
                msg["content"] = _WITHHELD_TEXT
                STATS["withheld_text"] += 1
                changed = withheld = True
                break
            _record_label(ev, u, label, elevated, "allow")
        if withheld:
            continue
        # OWASP LLM07: verbatim system prompt in the answer
        if sys_text:
            leaked, frac = output_checks.system_prompt_leak(sys_text, msg["content"])
            action = decide("system_prompt_leak", leaked, True) if leaked else "allow"
            ev.add("system_prompt_leak", "withhold" if action == "block" else ("log" if leaked else "allow"),
                   role="assistant", target=_target(u), score=frac)
            if action == "block":
                ev.escalate("withhold", "response reproduced the system prompt")
                msg["content"] = _WITHHELD_LEAK
                STATS["withheld_leak"] += 1
                _risk(flow)
                changed = True
                continue
        # OWASP LLM05: active markup / exfiltration-shaped links
        hits = output_checks.scan_output(msg["content"])
        if hits:
            action = decide("output_handling", True, True)
            for rule, _, _ in hits:
                ev.add(output_checks.check_kind(rule), "redact" if action == "redact" else action,
                       role="assistant", target=_target(u), detail=rule)
            if action == "block":
                ev.escalate("withhold", "unsafe markup in response")
                msg["content"] = _WITHHELD_TEXT
                changed = True
                continue
            if action == "redact":
                ev.escalate("redact", "unsafe markup neutralised")
                msg["content"] = output_checks.neutralise(msg["content"], hits)
                changed = True
        else:
            ev.add("output_markup", "allow", role="assistant", target=_target(u))
        # OWASP LLM02: PII / secrets in the answer
        if u.signals.get("entities") and decide("pii", True, True) in ("redact", "block"):
            ev.add("pii", "redact", role="assistant", target=_target(u),
                   detail=",".join(u.signals["entities"]))
            ev.escalate("redact", "sensitive data redacted")
            if msg["content"] == u.text:
                msg["content"] = u.signals["redacted"]        # computed with the System-1 call
            else:                                             # markup was neutralised first: redo
                msg["content"], _ = await pii.scan_and_redact_async(msg["content"])
            changed = True
        else:
            ev.add("pii", "allow", role="assistant", target=_target(u))

    goal = None
    for ch in choices:
        msg = ch.get("message") or {}
        calls = list(msg.get("tool_calls") or [])
        if msg.get("function_call"):                    # legacy single-function shape
            calls.append({"function": msg["function_call"]})
        for i, tc in enumerate(calls):
            fn = tc.get("function") or {}
            v = tool_policy.evaluate(fn.get("name", ""), fn.get("arguments", ""))
            reason = v.reason
            if v.allowed and elevated and system2.enabled() and fn.get("name") not in _READONLY_TOOLS:
                if goal is None:
                    goal = C.first_user_text(req_msgs)
                STATS["escalations"] += 1
                jv = await system2.judge("tool_call", "intent",
                                         f"tool={fn.get('name')} arguments={fn.get('arguments')}",
                                         context=f"user goal: {goal}")
                STATS[f"s2_{jv.status}"] += 1
                if jv.block:
                    v = tool_policy.ToolVerdict(False, "action judged inconsistent with the user's goal",
                                                "s2_intent")
                    reason = v.reason
            blocked = not v.allowed and decide("tool_call", True, True) != "allow"
            ev.add(v.rule or "tool_call", "block" if blocked else "allow", role="assistant",
                   target=f"choices[{ch.get('index', 0)}].tool_calls[{i}]", tool=fn.get("name"),
                   detail=reason or None,
                   owasp_ids=owasp.ids_for(v.rule or "tool_call"))
            if blocked:
                log.warning("blocked tool call session=%s tool=%s rule=%s",
                            flow.metadata.get("fw_session"), fn.get("name"), v.rule)
                ev.escalate("block", f"tool call blocked: {reason}")
                STATS["blocked_tool_call"] += 1
                _risk(flow)
                msg.pop("tool_calls", None)
                msg.pop("function_call", None)
                msg["content"] = f"[tool call blocked by the LLM firewall: {reason}]"
                ch["finish_reason"] = "stop"
                changed = True
                break

    if changed or flow.metadata.get("fw_stream") or _ANNOTATE:
        _emit(flow, payload)
    else:
        flow.response.headers.update(ev.headers())
        audit.record(ev)


# ----------------------------------------------------------------- internal endpoints
def _internal(flow: http.HTTPFlow, route: str) -> None:
    """GET /firewall/{health,metrics,owasp} -- answered by the addon, never
    forwarded. nginx refuses /firewall/ from outside; scrape on the internal
    network only."""
    flow.metadata["fw_done"] = True
    if route == "/firewall/health":
        body, ctype = json.dumps({"status": "ok", "cache_entries": len(_cache),
                                  "s2_mode": system2.MODE, "labels": _LABELS}), "application/json"
    elif route == "/firewall/metrics":
        extra = {f"stat_{k}": v for k, v in STATS.items()}
        extra["verdict_cache_entries"] = len(_cache)
        extra.update({f"s2_{k}": v for k, v in system2.STATS.items()})
        body, ctype = audit.prometheus_text(extra), "text/plain; version=0.0.4"
    elif route == "/firewall/owasp":
        body, ctype = json.dumps(audit.owasp_summary(), indent=2), "application/json"
    else:
        flow.response = http.Response.make(404, b"not found", {"Content-Type": "text/plain"})
        return
    flow.response = http.Response.make(200, body, {"Content-Type": ctype})


# ----------------------------------------------------------------- hooks
def load(loader):
    pii.start()


async def request(flow: http.HTTPFlow):
    route = _route(flow)
    if route.startswith(_INTERNAL.rstrip("/")):
        return _internal(flow, route)
    if flow.request.method != "POST":
        return                                          # GET /v1/models etc.: no prompt content
    if route != _PATH:
        if route not in _UNSCREENED_ALLOW:
            _deny_endpoint(flow, route)
        return
    flow.metadata["fw_eval"] = audit.Evaluation()
    try:
        payload = json.loads(flow.request.get_text())
        if not isinstance(payload, dict):
            raise ValueError("payload is not an object")
    except Exception:
        return                                          # let upstream reject malformed JSON
    try:
        if payload.get("stream"):
            payload["stream"] = False                   # de-stream; re-emitted as SSE on the way out
            opts = payload.pop("stream_options", None) or {}
            flow.metadata["fw_include_usage"] = bool(isinstance(opts, dict) and opts.get("include_usage"))
            flow.metadata["fw_stream"] = True
            flow.request.set_text(json.dumps(payload))
        await _screen_request(flow, payload)
    except Exception:
        log.exception("firewall error on request")
        if decide("error", True, True) != "allow":
            _block(flow, "internal error")


async def response(flow: http.HTTPFlow):
    if _route(flow) != _PATH or flow.metadata.get("fw_done") or "fw_eval" not in flow.metadata:
        return
    if flow.response is None or flow.response.status_code != 200:
        ev = _ev(flow)
        if flow.response is not None:
            flow.response.headers.update(ev.headers())
        audit.record(ev)
        return                                          # upstream errors pass through untouched
    try:
        payload = json.loads(flow.response.get_text())
        await _screen_response(flow, payload)
    except Exception:
        log.exception("firewall error on response")
        if decide("error", True, True) != "allow":
            _block(flow, "internal error")


def done():
    pii.shutdown()
