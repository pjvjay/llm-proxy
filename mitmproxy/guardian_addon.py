"""
guardian_addon.py
─────────────────
mitmproxy addon that screens every OpenAI /v1/chat/completions request with:

  1. A fast regex keyword classifier (violent acts, illegal activity, sexual content)
  2. IBM Granite Guardian 3.2 toxicity scoring via a local vllm instance

Blocked requests receive a fake-but-valid OpenAI response containing the
block reason so the caller's OpenAI SDK parses it normally.

Environment variables
─────────────────────
  VLLM_URL            Base URL of the vllm OpenAI-compatible server
                      (default: http://vllm:8000)
  TOXICITY_THRESHOLD  0.0–1.0 Guardian score above which a prompt is blocked
                      (default: 0.5)
  GUARDIAN_MODEL      Model served by vllm (default: ibm-granite/granite-guardian-3.2-2b)
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
import time
from typing import Optional

import httpx
from mitmproxy import http

# ── Logging ────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] guardian_addon: %(message)s",
)
log = logging.getLogger("guardian_addon")

# ── Config ─────────────────────────────────────────────────────────────────────

VLLM_URL           = os.environ.get("VLLM_URL", "http://vllm:8000")
TOXICITY_THRESHOLD = float(os.environ.get("TOXICITY_THRESHOLD", "0.5"))
GUARDIAN_MODEL     = os.environ.get("GUARDIAN_MODEL", "ibm-granite/granite-guardian-hap-38m")

# ── Category 1: Violent acts ────────────────────────────────────────────────────

_VIOLENT = [
    re.compile(r'\b(kill|murder|stab|shoot|bomb|attack|assault|behead|decapitate|massacre|torture)\b', re.I),
    re.compile(r'\bhow\s+(to\s+)?(kill|hurt|attack|murder|stab|shoot|maim|harm)\b', re.I),
    re.compile(r'\b(act|acts)\s+of\s+violence\b', re.I),
    re.compile(r'\b(brutally|violently)\s+(attack|kill|assault|murder|beat)\b', re.I),
]

# ── Category 2: Illegal activity inquiries ─────────────────────────────────────

_ILLEGAL = [
    re.compile(
        r'\bhow\s+(do\s+I\s+|to\s+|can\s+I\s+)?'
        r'(make|create|build|synthesize|produce|cook)\b.{0,40}'
        r'\b(meth|methamphetamine|cocaine|heroin|fentanyl|lsd|explosiv|bomb|napalm|c4|poison|ricin)\b',
        re.I,
    ),
    re.compile(r'\bhow\s+(do\s+I\s+|to\s+|can\s+I\s+)?(hack|break\s+into|crack)\b.{0,30}\b(account|system|database|server|password)\b', re.I),
    re.compile(r'\bhow\s+(do\s+I\s+|to\s+|can\s+I\s+)?(steal|rob|burglarise|burgle|shoplift|launder\s+money|forge|counterfeit)\b', re.I),
    re.compile(r'\bhow\s+(do\s+I\s+|to\s+|can\s+I\s+)?(pick\s+(a\s+)?lock|hotwire\s+a\s+car|evade\s+(police|law))\b', re.I),
    re.compile(r'\b(synthesize|manufacture|produce).{0,30}\b(illegal|controlled)\s+(substance|drug|narcotic)\b', re.I),
    re.compile(r'\bhow\s+(do\s+I\s+|to\s+|can\s+I\s+)?commit\s+(fraud|identity\s+theft|tax\s+evasion)\b', re.I),
]

# ── Category 3: Sexual content ──────────────────────────────────────────────────

_SEXUAL = [
    re.compile(r'\b(explicit\s+sexual|sexual\s+explicit|pornograph(ic|y)?)\b', re.I),
    re.compile(r'\b(write|generate|create|produce)\b.{0,20}\b(nude|naked|sexual|erotic|porn)\b.{0,20}\b(image|photo|content|scene|story)\b', re.I),
    re.compile(r'\bnsfw\s+(content|image|story|scene)\b', re.I),
    re.compile(r'\b(sexual\s+intercourse|sex\s+scene)\b.{0,30}\b(describe|write|generate|show)\b', re.I),
    re.compile(r'\b(describe|write|generate)\b.{0,30}\b(sexual\s+intercourse|sex\s+scene)\b', re.I),
    # Sexual slurs and derogatory terms
    re.compile(r'\b(faggot|fag|dyke|tranny|homo|queer)\b', re.I),
    re.compile(r'\b(slut|whore|bitch|cunt|pussy|dick|cock|penis|vagina)\b', re.I),
]

_REASON_LABELS = {
    "violent":  "a description of violent acts",
    "illegal":  "an inquiry on how to perform an illegal activity",
    "sexual":   "sexual content",
}


def keyword_classify(text: str) -> Optional[str]:
    """
    Quick keyword pre-filter.

    Returns one of 'violent', 'illegal', 'sexual', or None.
    Runs in microseconds; used to short-circuit before calling vllm.
    """
    for pattern in _VIOLENT:
        if pattern.search(text):
            return "violent"
    for pattern in _ILLEGAL:
        if pattern.search(text):
            return "illegal"
    for pattern in _SEXUAL:
        if pattern.search(text):
            return "sexual"
    return None


# ── Granite Guardian ────────────────────────────────────────────────────────────

_GUARDIAN_SYSTEM = (
    "You are a safety classifier. "
    "Determine whether the following user message is harmful, toxic, hateful, or otherwise "
    "violates safety guidelines.\n"
    "Respond with exactly one word on the first line: 'Yes' if the content is harmful or toxic, "
    "'No' if it is safe. Do not add any explanation."
)

# vllm readiness retry parameters
_GUARDIAN_READY_RETRIES = 20
_GUARDIAN_READY_WAIT    = 10   # seconds between retries


def _guardian_ready() -> bool:
    """Return True if the vllm health endpoint responds OK."""
    try:
        r = httpx.get(f"{VLLM_URL}/health", timeout=5.0)
        return r.status_code == 200
    except Exception:
        return False


def _wait_for_guardian() -> None:
    """Block until vllm is ready (called once at addon load time)."""
    log.info(f"Waiting for vllm Guardian at {VLLM_URL} …")
    for attempt in range(1, _GUARDIAN_READY_RETRIES + 1):
        if _guardian_ready():
            log.info("vllm Guardian is ready.")
            return
        log.info(f"  attempt {attempt}/{_GUARDIAN_READY_RETRIES} — not ready yet, sleeping {_GUARDIAN_READY_WAIT}s")
        time.sleep(_GUARDIAN_READY_WAIT)
    log.warning(
        "vllm Guardian did not become ready in time. "
        "Guardian checks will be skipped until it responds."
    )


async def query_guardian(prompt_text: str) -> float:
    """
    Query Granite Guardian via the vllm OpenAI-compatible endpoint.

    Returns a toxicity probability in [0.0, 1.0].
    Higher values indicate more likely harmful content.

    Falls open (returns 0.0) if the Guardian service is unreachable,
    so a temporary vllm outage does not block all traffic.
    """
    payload = {
        "model": GUARDIAN_MODEL,
        "messages": [
            {"role": "system", "content": _GUARDIAN_SYSTEM},
            {"role": "user",   "content": prompt_text},
        ],
        "max_tokens":   5,
        "temperature":  0.0,
        "logprobs":     True,
        "top_logprobs": 5,
    }

    t0 = time.perf_counter()
    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(f"{VLLM_URL}/v1/chat/completions", json=payload)
            resp.raise_for_status()
            data = resp.json()
    except httpx.HTTPError as exc:
        log.warning(f"Guardian HTTP error ({exc}); failing open.")
        return 0.0
    except Exception as exc:
        log.warning(f"Guardian unexpected error ({exc}); failing open.")
        return 0.0

    elapsed = time.perf_counter() - t0
    choice  = data["choices"][0]
    raw_text = choice["message"]["content"].strip().lower()
    log.info(f"Guardian raw response: {raw_text!r} ({elapsed:.1f}s)")

    # ── Extract P("Yes") from logprobs if available ─────────────────────────
    score = _extract_yes_probability(choice, raw_text)
    log.info(f"Guardian toxicity score: {score:.4f} (threshold={TOXICITY_THRESHOLD})")
    return score


def _extract_yes_probability(choice: dict, raw_text: str) -> float:
    """
    Pull the probability of the 'Yes' token from the logprobs block.
    Falls back to 0.9 / 0.1 based on the raw text if logprobs are absent.
    """
    try:
        logprobs_content = choice.get("logprobs", {}) or {}
        token_list = logprobs_content.get("content") or []
        if token_list:
            for lp_entry in token_list[0].get("top_logprobs", []):
                token = lp_entry.get("token", "").strip().lower().lstrip("▁")
                if token == "yes":
                    return math.exp(lp_entry["logprob"])
            # 'Yes' not in top_logprobs — fall through to text heuristic
    except Exception as exc:
        log.debug(f"logprob parsing failed ({exc}); using text heuristic")

    return 0.9 if raw_text.startswith("yes") else 0.1


# ── Response construction ───────────────────────────────────────────────────────

def _block_response(category_key: Optional[str]) -> http.Response:
    """
    Build a fake-but-valid OpenAI chat.completion response with the block message.

    Using HTTP 200 (not 4xx) so that the caller's openai SDK parses the
    response body normally rather than raising an API exception.
    """
    if category_key and category_key in _REASON_LABELS:
        message = f"The prompt was blocked because it contained {_REASON_LABELS[category_key]}."
    else:
        message = "The prompt was blocked because it is considered toxic."

    body = {
        "id":      "blocked-00000000",
        "object":  "chat.completion",
        "model":   "blocked",
        "choices": [
            {
                "index":         0,
                "message":       {"role": "assistant", "content": message},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens":     0,
            "completion_tokens": 0,
            "total_tokens":      0,
        },
    }

    return http.Response.make(
        200,
        json.dumps(body).encode("utf-8"),
        {"Content-Type": "application/json"},
    )


# ── Prompt extraction ───────────────────────────────────────────────────────────

def _extract_prompt_text(body: bytes) -> Optional[str]:
    """
    Pull all user/system message content from an OpenAI chat request body.
    Returns a single concatenated string, or None on parse failure.
    """
    try:
        data     = json.loads(body)
        messages = data.get("messages") or []
        parts    = []
        for msg in messages:
            if msg.get("role") not in ("user", "system"):
                continue
            content = msg.get("content", "")
            if isinstance(content, str):
                parts.append(content)
            elif isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        parts.append(block.get("text", ""))
        return " ".join(parts) if parts else None
    except Exception as exc:
        log.debug(f"Could not parse request body: {exc}")
        return None


# ── mitmproxy addon class ───────────────────────────────────────────────────────

class GuardianAddon:
    """
    mitmproxy addon.

    request()  — screen each /v1/chat/completions call before it reaches OpenAI
    response() — log (and optionally screen) every response from OpenAI
    """

    def __init__(self) -> None:
        log.info("GuardianAddon initialising …")
        _wait_for_guardian()
        log.info(
            f"GuardianAddon ready  |  model={GUARDIAN_MODEL}  "
            f"threshold={TOXICITY_THRESHOLD}  vllm={VLLM_URL}"
        )

    # ── Intercept outgoing requests ─────────────────────────────────────────

    async def request(self, flow: http.HTTPFlow) -> None:  # noqa: D401
        """Screen the prompt before it is forwarded to OpenAI."""

        if "/v1/chat/completions" not in flow.request.path:
            return

        log.info(
            f"→ REQUEST  {flow.request.method} {flow.request.path}  "
            f"body={len(flow.request.content)} bytes"
        )

        prompt_text = _extract_prompt_text(flow.request.content)
        if not prompt_text:
            log.info("  Could not extract prompt text — passing through.")
            return

        log.info(f"  Prompt (first 200 chars): {prompt_text[:200]!r}")

        # ── Stage 1: fast keyword classifier ─────────────────────────────
        category = keyword_classify(prompt_text)
        if category:
            log.warning(f"  ✗ BLOCKED by keyword classifier  category={category}")
            flow.response = _block_response(category)
            return

        log.info("  ✓ Keyword classifier: clean")

        # ── Stage 2: Granite Guardian deep check ──────────────────────────
        score = await query_guardian(prompt_text)
        if score >= TOXICITY_THRESHOLD:
            log.warning(f"  ✗ BLOCKED by Guardian  score={score:.4f}")
            flow.response = _block_response(None)   # generic "toxic"
            return

        log.info(f"  ✓ Guardian cleared  score={score:.4f}  — forwarding to OpenAI")

    # ── Intercept incoming responses ────────────────────────────────────────

    def response(self, flow: http.HTTPFlow) -> None:  # noqa: D401
        """Log every response that comes back from OpenAI."""

        if "/v1/chat/completions" not in flow.request.path:
            return

        status = flow.response.status_code
        log.info(f"← RESPONSE  status={status}  body={len(flow.response.content)} bytes")

        # Optionally log the assistant reply (first 200 chars)
        try:
            data    = json.loads(flow.response.content)
            content = data["choices"][0]["message"]["content"]
            log.info(f"  Reply (first 200 chars): {content[:200]!r}")
        except Exception:
            pass


addons = [GuardianAddon()]
