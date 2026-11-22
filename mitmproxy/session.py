"""Session identity + content-hash verdict cache => incremental scanning.

An agent loop re-sends the entire conversation on every call. Screening the whole
history each time is O(N) per turn and O(N^2) per loop (bench/bench_session.py).
Instead every text unit is keyed by a hash of (role, text); raw signals for a
key (classifier scores, PII findings, redacted text) are cached and the policy
decision is re-derived from them on every turn. Consequences:

* a turn screens only what it has not seen before -- usually one new user
  message or one new tool result;
* the cache is global, not per-session: identical content gets identical
  signals whatever the session, and a redacted message that the client sends
  back unredacted next turn hits the cache and is redacted again for free;
* thresholds / actions are NOT cached, so policy changes apply immediately;
* nothing is cached when the classifier service was unavailable, so the unit is
  retried on the next turn instead of failing closed forever.

The session id itself is only needed for per-session state (turn counter,
system-prompt change detection, and the trajectory / escalation work that
comes later). It is taken from, in order: the X-Session-Id header, the OpenAI
`user` field, `metadata.session_id`, or derived from the system prompt + first
user message."""
import hashlib
import os
import time
from collections import OrderedDict
from dataclasses import dataclass

import content as C

_CACHE_SIZE = int(os.getenv("VERDICT_CACHE_SIZE", "50000"))
_CACHE_TTL = float(os.getenv("VERDICT_CACHE_TTL_S", "1800"))
_SESSION_TTL = float(os.getenv("SESSION_TTL_S", "3600"))


def message_key(role: str, text: str) -> str:
    h = hashlib.sha256()   # hardware-accelerated on most CPUs; see bench_session.py
    h.update(role.encode())
    h.update(b"\x00")
    h.update(text.encode("utf-8", "surrogatepass"))
    return h.hexdigest()


class VerdictCache:
    """LRU + TTL, in-process. Swap for Redis when the proxy is scaled out."""

    def __init__(self, maxsize: int = _CACHE_SIZE, ttl_s: float = _CACHE_TTL):
        self._d: OrderedDict[str, tuple[float, dict]] = OrderedDict()
        self.maxsize, self.ttl = maxsize, ttl_s
        self.hits = self.misses = 0

    def get(self, key: str):
        item = self._d.get(key)
        if item is None:
            self.misses += 1
            return None
        exp, value = item
        if exp < time.monotonic():
            del self._d[key]
            self.misses += 1
            return None
        self._d.move_to_end(key)
        self.hits += 1
        return value

    def put(self, key: str, value: dict) -> None:
        self._d[key] = (time.monotonic() + self.ttl, value)
        self._d.move_to_end(key)
        while len(self._d) > self.maxsize:
            self._d.popitem(last=False)

    def __len__(self):
        return len(self._d)


@dataclass
class SessionState:
    id: str
    turn: int
    system_hash: str
    system_changed: bool
    derived: bool
    risk: int = 0                 # blocks / withholds so far in this session


class SessionStore:
    def __init__(self, ttl_s: float = _SESSION_TTL, maxsize: int = 100000):
        self._d: OrderedDict[str, tuple[float, dict]] = OrderedDict()
        self.ttl, self.maxsize = ttl_s, maxsize

    def touch(self, sid: str, system_hash: str, derived: bool) -> SessionState:
        now = time.monotonic()
        item = self._d.get(sid)
        if item is None or item[0] < now:
            st = {"turn": 0, "system_hash": system_hash, "risk": 0}
        else:
            st = item[1]
        st["turn"] += 1
        changed = st["system_hash"] != system_hash
        st["system_hash"] = system_hash
        self._d[sid] = (now + self.ttl, st)
        self._d.move_to_end(sid)
        while len(self._d) > self.maxsize:
            self._d.popitem(last=False)
        return SessionState(sid, st["turn"], system_hash, changed, derived, st["risk"])

    def bump_risk(self, sid: str) -> None:
        item = self._d.get(sid)
        if item is not None:
            item[1]["risk"] += 1


def session_id(headers, payload: dict) -> tuple[str, bool]:
    """Return (session id, derived?)."""
    sid = headers.get("X-Session-Id") if headers is not None else None
    if sid:
        return str(sid), False
    if isinstance(payload.get("user"), str) and payload["user"]:
        return "user:" + payload["user"], False
    meta = payload.get("metadata")
    if isinstance(meta, dict) and meta.get("session_id"):
        return "meta:" + str(meta["session_id"]), False
    msgs = payload.get("messages", [])
    return "derived:" + message_key("sys+u0", C.system_text(msgs) + "\x00" + C.first_user_text(msgs)), True


def plan(cache: VerdictCache, messages: list, roles: set[str]) -> list[C.Unit]:
    """Units for the given roles with cache lookups attached."""
    out = C.units(messages, roles)
    for u in out:
        u.key = message_key(u.role, u.text)
        u.cached = cache.get(u.key)
    return out
