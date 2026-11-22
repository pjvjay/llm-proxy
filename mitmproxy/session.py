"""Incremental scanning for agent loops: session identity + a content-addressed
scan cache.

An agent re-sends the whole conversation on every turn. Re-scanning it makes
an N-turn loop cost O(N^2) classifier work; the old addon also joined all user
messages into one string that the classifiers truncate at 512 tokens, so in
long conversations the *newest* user turns were never seen by the ML stages.

Here every screened message becomes a ScanUnit keyed by a cryptographic digest
of (cache version, target, context tail, text). Results are cached per unit, so
each turn only scans what is new, and every message is scanned on its own
(no truncation of later turns).

Design choices (see docs/perf-report.md, B4):

* Content-addressed, not session-keyed. A classifier score is a pure function
  of the text, so sharing results across sessions is safe and raises hit rates
  (identical tool outputs, retries, forked conversations). Session ids are
  still derived -- for logs and for future per-session policy.
* Cryptographic digest (SHA-256, truncated to 128 bits). The key decides
  whether text is screened at all, so it must resist deliberate collisions; a
  non-cryptographic hash (xxHash) would let an attacker craft text colliding
  with a cached benign entry. SHA-256 measured ~2x faster than BLAKE2b here
  (hardware SHA extensions); xxh3 would save ~0.3 ms/turn at turn 40 (B4).
* Context tail. A prompt unit carries the last SCAN_CONTEXT_CHARS of the
  previous user message, so a payload split across two messages ("Ignore all
  previous" | "instructions and ...") is still seen together (tool results:
  SCAN_CONTEXT_CHARS_TOOL, default 0 -- see CONTEXT_CHARS).
* Only *successful* classifier results are cached (never an outage), and
  scores are cached rather than verdicts, so threshold / policy changes take
  effect immediately. Bump SCAN_CACHE_VERSION when models change.
"""
from __future__ import annotations

import hashlib
import os
from collections import OrderedDict
from dataclasses import dataclass

from messages import screened_messages

VERSION = os.getenv("SCAN_CACHE_VERSION", "1")
# Context tail per target. Prompts: multi-turn payload splitting is a known
# jailbreak pattern. Tool results: off by default -- the extra ~60 tokens pushed
# a typical 2 KB tool result (~480 tokens) into a second classifier window,
# doubling its cost (B6), and splitting across tool outputs is far less common.
CONTEXT_CHARS = {
    "prompt": int(os.getenv("SCAN_CONTEXT_CHARS", "256")),
    "tool_result": int(os.getenv("SCAN_CONTEXT_CHARS_TOOL", "0")),
}
MAX_BYTES = int(os.getenv("SCAN_CACHE_MAX_BYTES", str(64 * 1024 * 1024)))
MAX_ENTRIES = int(os.getenv("SCAN_CACHE_MAX_ENTRIES", "200000"))
ENABLED = os.getenv("SCAN_CACHE", "1") not in ("0", "false", "no")
_ENTRY_OVERHEAD = 200   # rough per-entry bookkeeping bytes (key, tuple, dict slot)


def digest(*parts: str) -> bytes:
    h = hashlib.sha256()
    for p in parts:
        b = p.encode("utf-8", "surrogatepass")
        h.update(len(b).to_bytes(8, "little"))      # length-prefix: no ambiguity
        h.update(b)
    return h.digest()[:16]


class LRU:
    """Byte- and entry-bounded LRU. OrderedDict chosen over cachetools /
    FIFO after measurement (B4): same O(1) ops, no dependency, true recency."""

    def __init__(self, max_bytes: int = MAX_BYTES, max_entries: int = MAX_ENTRIES):
        self.max_bytes, self.max_entries = max_bytes, max_entries
        self._d: OrderedDict[bytes, tuple[object, int]] = OrderedDict()
        self.bytes = 0
        self.hits = self.misses = 0

    def get(self, key: bytes):
        item = self._d.get(key)
        if item is None:
            self.misses += 1
            return None
        self._d.move_to_end(key)
        self.hits += 1
        return item[0]

    def put(self, key: bytes, value, size: int = 0):
        size += _ENTRY_OVERHEAD
        if size > self.max_bytes:
            return
        old = self._d.pop(key, None)
        if old is not None:
            self.bytes -= old[1]
        self._d[key] = (value, size)
        self.bytes += size
        while self.bytes > self.max_bytes or len(self._d) > self.max_entries:
            _, (_, s) = self._d.popitem(last=False)
            self.bytes -= s

    def __len__(self):
        return len(self._d)

    def clear(self):
        self._d.clear()
        self.bytes = self.hits = self.misses = 0


SCAN_CACHE = LRU()   # key -> {"regex": str|None, "scores": {check: float}}
PII_CACHE = LRU()    # key -> (redacted_text | None, [entity types])


@dataclass
class ScanUnit:
    index: int          # position in payload["messages"]
    target: str         # "prompt" | "tool_result"
    text: str           # raw message text
    context: str        # tail of previous screened message of the same target
    key: bytes          # classifier-cache key
    pii_key: bytes      # PII-cache key (raw text only: redaction is per message)

    @property
    def scan_text(self) -> str:
        return f"{self.context}\n{self.text}" if self.context else self.text


def scan_units(messages) -> list[ScanUnit]:
    units, prev = [], {}
    for i, target, text in screened_messages(messages):
        n = CONTEXT_CHARS.get(target, 0)
        ctx = prev.get(target, "")[-n:] if n else ""
        units.append(ScanUnit(i, target, text, ctx,
                              digest("scan", VERSION, target, ctx, text),
                              digest("pii", VERSION, text)))
        prev[target] = text
    return units


def session_id(headers, payload: dict) -> str:
    """Best-effort stable id for a conversation: explicit header, OpenAI
    ``user`` / ``metadata.session_id``, else a digest of its opening messages."""
    sid = headers.get("x-session-id") if headers is not None else None
    if sid:
        return str(sid)[:128]
    if isinstance(payload.get("user"), str) and payload["user"]:
        return "user:" + payload["user"][:120]
    md = payload.get("metadata")
    if isinstance(md, dict) and isinstance(md.get("session_id"), str):
        return md["session_id"][:128]
    opening = [str(m.get("content"))[:2000] for m in (payload.get("messages") or [])[:2]
               if isinstance(m, dict)]
    return "derived:" + digest("session", *opening).hex()[:16]
