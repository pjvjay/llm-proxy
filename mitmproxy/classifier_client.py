"""Async client to the guardrails service.

One module-level ``httpx.AsyncClient`` with keep-alive. The original code built
a new client per call; constructing an AsyncClient builds a TLS context and
parses the CA bundle even for plain http:// (measured ~50 ms of CPU that
blocks the event loop -- per call, on both the request and response paths).
See docs/perf-report.md, B1.

``GUARDRAILS_UDS=/path/to.sock`` switches to a Unix domain socket (both
containers must share the socket's volume).

Batch API: every new message of a turn is sent in ONE request; the service
runs all windows of all texts through each model as a batch.
"""
from __future__ import annotations

import os

import httpx

_URL = os.getenv("GUARDRAILS_URL", "http://guardrails:8000")
_UDS = os.getenv("GUARDRAILS_UDS", "")
_TIMEOUT = float(os.getenv("GUARDRAILS_TIMEOUT", "2.0"))

_client: httpx.AsyncClient | None = None


def _get_client() -> httpx.AsyncClient:
    global _client
    if _client is None or _client.is_closed:
        transport = httpx.AsyncHTTPTransport(uds=_UDS, retries=1) if _UDS \
            else httpx.AsyncHTTPTransport(retries=1)
        _client = httpx.AsyncClient(
            base_url="http://guardrails" if _UDS else _URL,
            timeout=_TIMEOUT,
            transport=transport,
            # keep-alive shorter than the server's (--timeout-keep-alive 75)
            limits=httpx.Limits(max_connections=64, max_keepalive_connections=32,
                                keepalive_expiry=60.0),
        )
    return _client


async def _post(body: dict) -> dict:
    client = _get_client()
    # tell the service when we will stop waiting, so it can drop the work
    # instead of computing answers nobody reads (see guardrails/server.py)
    body = {**body, "deadline_ms": max(1.0, _TIMEOUT * 1000 - 50)}
    try:
        resp = await client.post("/v1/classify", json=body)
    except (httpx.RemoteProtocolError, httpx.ReadError, httpx.WriteError, httpx.ConnectError):
        # stale pooled connection closed by the server: one clean retry
        # (classification is a pure function, safe to repeat). Timeouts are
        # NOT retried -- that would double the worst-case latency budget.
        resp = await client.post("/v1/classify", json=body)
    resp.raise_for_status()
    return resp.json()


async def classify(text: str, checks=("injection", "toxicity")) -> dict:
    """Single text -> {check: {"score": float}}  (original API, kept)."""
    data = await _post({"text": text, "checks": list(checks)})
    return data["results"]


async def classify_batch(items: list[tuple[str, tuple[str, ...]]]) -> list[dict]:
    """[(text, checks), ...] -> [{check: {"score": float}}, ...] in the same order.

    Raises if the service is unreachable or returns an incomplete answer, so
    callers apply their fail mode instead of trusting a partial result."""
    if not items:
        return []
    data = await _post({"items": [{"text": t, "checks": list(c)} for t, c in items]})
    results = data.get("items")
    if not isinstance(results, list) or len(results) != len(items):
        raise ValueError("guardrails returned a malformed batch response")
    for (_, checks), r in zip(items, results):
        if any(c not in r for c in checks):
            raise ValueError("guardrails response is missing a requested check")
    return results


async def aclose():
    global _client
    if _client is not None:
        await _client.aclose()
        _client = None
