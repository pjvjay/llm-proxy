"""Async client to the guardrails service.

Performance notes (bench/bench_transport.py):

* The old code built a new `httpx.AsyncClient` per call. Constructing a client
  builds an SSL context (loads the CA bundle) even for a plain-http URL --
  ~50 ms on this box -- and then opens a fresh TCP connection. A module-level
  client with keep-alive brings the hop from ~75 ms to ~3 ms p50 and stops
  falling over under concurrency.
* HTTP/2 and Unix sockets measured within noise of HTTP/1.1 keep-alive for this
  traffic pattern (one small request at a time per flow), so the plain option
  is kept. `GUARDRAILS_UDS=/path.sock` switches to a Unix socket if the two
  containers share a volume.
* `decide_many` (System-1 decision endpoint) sends every new unit of a turn in
  ONE round-trip with its role / tool context; the service returns per-label
  probabilities for the labels relevant to each role. `classify_many` is the
  older per-check API, kept for evals.
"""
import asyncio
import os
from collections import defaultdict

import httpx

_URL = os.getenv("GUARDRAILS_URL", "http://guardrails:8000")
_UDS = os.getenv("GUARDRAILS_UDS")
_TIMEOUT = float(os.getenv("GUARDRAILS_TIMEOUT", "5.0"))

_client: httpx.AsyncClient | None = None
_transport_override: httpx.AsyncBaseTransport | None = None


def configure(transport: httpx.AsyncBaseTransport | None = None, url: str | None = None) -> None:
    """Test / benchmark hook: route calls through a given transport (e.g. an
    httpx.MockTransport) and/or a different base URL. Resets the client."""
    global _client, _transport_override, _URL
    _transport_override, _client = transport, None
    if url:
        _URL = url


def _get() -> httpx.AsyncClient:
    global _client
    if _client is None:
        kw = dict(timeout=_TIMEOUT, verify=False,      # plain http; skip SSL-context build
                  limits=httpx.Limits(max_keepalive_connections=16, max_connections=64))
        if _transport_override is not None:
            _client = httpx.AsyncClient(transport=_transport_override, **kw)
        elif _UDS:
            _client = httpx.AsyncClient(transport=httpx.AsyncHTTPTransport(uds=_UDS), **kw)
        else:
            _client = httpx.AsyncClient(**kw)
    return _client


async def decide_many(items: list[tuple[str, str, str | None, tuple[str, ...]]]) -> list[dict]:
    """items: [(text, role, tool, labels)] -> [{"scores": {label: p}, "windows": n, "passes": k}].
    One round-trip for the whole turn. Raises on transport / HTTP error. Labels
    are sent explicitly so a service-side config drift cannot silently drop a check."""
    if not items:
        return []
    resp = await _get().post(f"{_URL}/v1/decide", json={"items": [
        {"text": t, "role": r, "tool": tl, "labels": list(lb)} for t, r, tl, lb in items]})
    resp.raise_for_status()
    return resp.json()["decisions"]


async def classify(text: str, checks=("injection", "toxicity")) -> dict:
    """Single text -> {check: {"score": float}}. Kept for compatibility / evals."""
    return (await classify_many([(text, tuple(checks))]))[0]


async def classify_many(items: list[tuple[str, tuple[str, ...]]]) -> list[dict]:
    """items: [(text, checks)] -> list of {check: {"score": float, "windows": int}},
    same order. Raises on transport / HTTP error (caller applies fail mode)."""
    if not items:
        return []
    groups: dict[tuple[str, ...], list[int]] = defaultdict(list)
    for i, (_, checks) in enumerate(items):
        groups[tuple(checks)].append(i)
    out: list[dict | None] = [None] * len(items)
    client = _get()

    async def one(checks, idxs):
        resp = await client.post(f"{_URL}/v1/classify",
                                 json={"texts": [items[i][0] for i in idxs],
                                       "checks": list(checks)})
        resp.raise_for_status()
        results = resp.json()["results"]
        for i, r in zip(idxs, results):
            out[i] = r

    await asyncio.gather(*(one(c, idxs) for c, idxs in groups.items()))
    return out  # type: ignore[return-value]


async def aclose():
    global _client
    if _client is not None:
        await _client.aclose()
        _client = None
