"""Async client to the guardrails service. One request runs both checks
concurrently server-side, so the added injection stage costs max(), not sum()."""
import os
import httpx

_URL = os.getenv("GUARDRAILS_URL", "http://guardrails:8000")
_TIMEOUT = float(os.getenv("GUARDRAILS_TIMEOUT", "2.0"))

async def classify(text: str, checks=("injection", "toxicity")) -> dict:
    async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
        resp = await client.post(
            f"{_URL}/v1/classify", json={"text": text, "checks": list(checks)}
        )
        resp.raise_for_status()
        return resp.json()["results"]   # {check: {"score": float}}
