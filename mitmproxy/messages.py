"""Robust access to chat.completions message content.

The original addon assumed ``message["content"]`` is a ``str``. It can also be
``None`` (assistant turns that only carry ``tool_calls``) or a list of content
parts (multimodal / newer SDKs):  [{"type": "text", "text": "..."}, {"type":
"image_url", ...}]. A list made ``"\\n".join(...)`` raise ``TypeError`` inside
the mitmproxy hook, which mitmproxy logs and then forwards the request
**unscreened** (fail-open). Everything here is total: unknown shapes are
stringified so they are still scanned rather than skipped.
"""
from __future__ import annotations

import json
from typing import Callable, Iterator

# Which roles are screened on the request path, and as which target.
#   prompt      : text a user typed (or a framework put in the user role)
#   tool_result : output of a tool / function the agent called -- the main
#                 indirect prompt-injection vector for agents
# system / developer are operator-controlled; assistant history was screened
# when it was generated (see response path), so neither is rescanned here.
ROLE_TARGET = {"user": "prompt", "tool": "tool_result", "function": "tool_result"}


def content_text(content) -> str:
    """All text carried by a message ``content`` field, joined by newlines."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(t for t in (_part_text(p) for p in content) if t)
    if isinstance(content, dict):
        return _part_text(content)
    return str(content)


def _part_text(part) -> str:
    if isinstance(part, str):
        return part
    if isinstance(part, dict):
        t = part.get("text")
        if isinstance(t, str):
            return t
        if isinstance(t, dict) and isinstance(t.get("value"), str):
            return t["value"]
        if part.get("type") in ("image_url", "input_image", "input_audio", "image", "file"):
            return ""            # non-text modality: out of scope for text classifiers
        # Unknown part type: scan its JSON so nothing slips through unseen.
        return json.dumps(part, ensure_ascii=False)
    return str(part)


def map_content_text(content, fn: Callable[[str], str]):
    """Return ``content`` with ``fn`` applied to every text segment, preserving
    its shape (str stays str, parts stay parts)."""
    if content is None:
        return None
    if isinstance(content, str):
        return fn(content)
    if isinstance(content, list):
        out = []
        for p in content:
            if isinstance(p, str):
                out.append(fn(p))
            elif isinstance(p, dict) and isinstance(p.get("text"), str):
                out.append({**p, "text": fn(p["text"])})
            elif (isinstance(p, dict) and isinstance(p.get("text"), dict)
                  and isinstance(p["text"].get("value"), str)):
                out.append({**p, "text": {**p["text"], "value": fn(p["text"]["value"])}})
            else:
                out.append(p)
        return out
    return fn(content_text(content))


def screened_messages(messages) -> Iterator[tuple[int, str, str]]:
    """Yield (index, target, text) for every message screened on the request path."""
    if not isinstance(messages, list):
        return
    for i, m in enumerate(messages):
        if not isinstance(m, dict):
            continue
        target = ROLE_TARGET.get(m.get("role"))
        if target is None:
            continue
        text = content_text(m.get("content"))
        if text:
            yield i, target, text


def declared_tools(payload: dict) -> set[str] | None:
    """Tool / function names the caller declared in this request, or None if the
    request declares no tools (then no name check is possible)."""
    names = set()
    for t in payload.get("tools") or []:
        if isinstance(t, dict):
            fn = t.get("function") if isinstance(t.get("function"), dict) else t
            if isinstance(fn.get("name"), str):
                names.add(fn["name"])
    for f in payload.get("functions") or []:           # legacy function calling
        if isinstance(f, dict) and isinstance(f.get("name"), str):
            names.add(f["name"])
    return names or None
