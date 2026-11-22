"""OWASP LLM10 (Unbounded Consumption) controls on the request path.

  MAX_REQUEST_BYTES      block requests whose body exceeds this (default 4 MB;
                         keep nginx client_max_body_size above it so the firewall,
                         not nginx, answers -- with an OWASP-tagged response)
  MAX_COMPLETION_TOKENS  clamp max_tokens / max_completion_tokens above this
                         (default 4096; 0 = off). TOKEN_LIMIT_ACTION=clamp|block
  RATE_LIMIT_RPM         per-session requests per minute (default 0 = off), a
                         token bucket keyed by the firewall's session id

Pure stdlib, in-process state (per proxy replica)."""
import os
import time

MAX_REQUEST_BYTES = int(os.getenv("MAX_REQUEST_BYTES", str(4 * 1024 * 1024)))
MAX_COMPLETION_TOKENS = int(os.getenv("MAX_COMPLETION_TOKENS", "4096"))
TOKEN_LIMIT_ACTION = os.getenv("TOKEN_LIMIT_ACTION", "clamp")
RATE_LIMIT_RPM = float(os.getenv("RATE_LIMIT_RPM", "0"))

_buckets: dict[str, tuple[float, float]] = {}       # session -> (tokens, last_refill)


def check(raw_len: int, payload: dict, session: str, now: float | None = None) -> list[dict]:
    """Return findings: [{"rule", "action": block|clamp, "detail"}]. Mutates
    `payload` when clamping. An empty list means within limits."""
    out = []
    if MAX_REQUEST_BYTES and raw_len > MAX_REQUEST_BYTES:
        out.append({"rule": "request_bytes", "action": "block",
                    "detail": f"request body {raw_len} bytes > {MAX_REQUEST_BYTES}"})
        return out
    if MAX_COMPLETION_TOKENS:
        for key in ("max_tokens", "max_completion_tokens"):
            v = payload.get(key)
            if isinstance(v, int) and v > MAX_COMPLETION_TOKENS:
                if TOKEN_LIMIT_ACTION == "block":
                    out.append({"rule": key, "action": "block",
                                "detail": f"{key}={v} > {MAX_COMPLETION_TOKENS}"})
                    return out
                payload[key] = MAX_COMPLETION_TOKENS
                out.append({"rule": key, "action": "clamp",
                            "detail": f"{key} {v} -> {MAX_COMPLETION_TOKENS}"})
    if RATE_LIMIT_RPM > 0:
        now = time.monotonic() if now is None else now
        cap = max(1.0, RATE_LIMIT_RPM)
        tokens, last = _buckets.get(session, (cap, now))
        tokens = min(cap, tokens + (now - last) * RATE_LIMIT_RPM / 60.0)
        if tokens < 1.0:
            _buckets[session] = (tokens, now)
            out.append({"rule": "rate_limit", "action": "block",
                        "detail": f"session over {RATE_LIMIT_RPM:g} requests/minute"})
            return out
        _buckets[session] = (tokens - 1.0, now)
        if len(_buckets) > 100_000:                   # bound memory
            _buckets.pop(next(iter(_buckets)))
    return out
