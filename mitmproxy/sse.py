"""Server-sent-event (``stream: true``) chat.completions bodies.

mitmproxy buffers the whole response before the ``response`` hook runs (this
addon does not enable streaming), so a streamed reply arrives here as one SSE
text. The original addon called ``json.loads`` on it, failed, and returned --
i.e. streamed responses, including any tool calls, were never screened.

``aggregate`` folds the chunks into a chat.completion-shaped dict so the same
screening code applies; ``render`` turns a (possibly modified) aggregate back
into a valid SSE stream. Unmodified responses are passed through byte-for-byte.
(True incremental streaming with tool-call buffering is a later step.)
"""
from __future__ import annotations

import json
import time


def events(text: str):
    """Yield the ``data:`` payload of each SSE event."""
    for block in text.replace("\r\n", "\n").split("\n\n"):
        data = [ln[5:].lstrip() for ln in block.split("\n") if ln.startswith("data:")]
        if data:
            yield "\n".join(data)


def aggregate(text: str) -> dict | None:
    """Fold chat.completion.chunk events into one chat.completion dict.
    Returns None if the body contains no completion chunks (e.g. an error)."""
    head, usage, choices = None, None, {}
    for data in events(text):
        if data == "[DONE]":
            continue
        obj = json.loads(data)
        if not isinstance(obj, dict) or "choices" not in obj:
            continue                      # error / keep-alive / unknown event
        head = head or obj
        if obj.get("usage"):
            usage = obj["usage"]
        for ch in obj.get("choices") or []:
            idx = ch.get("index", 0)
            agg = choices.setdefault(idx, {"index": idx, "finish_reason": None,
                                           "message": {"role": "assistant", "content": None}})
            delta = ch.get("delta") or {}
            msg = agg["message"]
            if delta.get("role"):
                msg["role"] = delta["role"]
            if isinstance(delta.get("content"), str):
                msg["content"] = (msg["content"] or "") + delta["content"]
            if isinstance(delta.get("refusal"), str):
                msg["refusal"] = (msg.get("refusal") or "") + delta["refusal"]
            for tc in delta.get("tool_calls") or []:
                calls = msg.setdefault("tool_calls", [])
                i = tc.get("index", len(calls))
                while len(calls) <= i:
                    calls.append({"id": None, "type": "function",
                                  "function": {"name": "", "arguments": ""}})
                cur = calls[i]
                if tc.get("id"):
                    cur["id"] = tc["id"]
                if tc.get("type"):
                    cur["type"] = tc["type"]
                fn = tc.get("function") or {}
                cur["function"]["name"] += fn.get("name") or ""
                cur["function"]["arguments"] += fn.get("arguments") or ""
            fc = delta.get("function_call")
            if isinstance(fc, dict):
                cur = msg.setdefault("function_call", {"name": "", "arguments": ""})
                cur["name"] += fc.get("name") or ""
                cur["arguments"] += fc.get("arguments") or ""
            if ch.get("finish_reason"):
                agg["finish_reason"] = ch["finish_reason"]
    if head is None:
        return None
    return {"id": head.get("id"), "object": "chat.completion", "created": head.get("created"),
            "model": head.get("model"), "system_fingerprint": head.get("system_fingerprint"),
            "choices": [choices[k] for k in sorted(choices)], "usage": usage}


def render(completion: dict) -> str:
    """chat.completion dict -> SSE text (one content chunk + one finish chunk
    per choice, optional usage chunk, then [DONE])."""
    base = {"id": completion.get("id") or "firewall", "object": "chat.completion.chunk",
            "created": completion.get("created") or int(time.time()),
            "model": completion.get("model") or "firewall"}
    if completion.get("system_fingerprint"):
        base["system_fingerprint"] = completion["system_fingerprint"]
    out = []
    for ch in completion.get("choices") or []:
        msg = ch.get("message") or {}
        delta = {"role": msg.get("role", "assistant"), "content": msg.get("content")}
        if msg.get("tool_calls"):
            delta["tool_calls"] = [{**tc, "index": i} for i, tc in enumerate(msg["tool_calls"])]
        if msg.get("function_call"):
            delta["function_call"] = msg["function_call"]
        idx = ch.get("index", 0)
        out.append({**base, "choices": [{"index": idx, "delta": delta, "finish_reason": None}]})
        out.append({**base, "choices": [{"index": idx, "delta": {},
                                         "finish_reason": ch.get("finish_reason") or "stop"}]})
    if completion.get("usage"):
        out.append({**base, "choices": [], "usage": completion["usage"]})
    return "".join(f"data: {json.dumps(c)}\n\n" for c in out) + "data: [DONE]\n\n"
