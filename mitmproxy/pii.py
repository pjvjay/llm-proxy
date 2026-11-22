"""PII / secret detection + redaction via Presidio, used on BOTH the request
(before prompts leave for a third-party model) and the response. Adds a custom
recognizer for API keys / bearer tokens on top of Presidio's built-ins.

Performance notes (see docs/perf-report.md, B2):

* NLP engine. ``AnalyzerEngine()`` with no arguments loads ``en_core_web_lg``
  (and downloads it at startup if missing). Every configured entity below is
  pattern / checksum based, so NER adds cost and no detections. We default to a
  tokenizer-only spaCy pipeline (``PII_NLP_MODEL=blank``); measured span parity
  with an ``en_core_web_sm``-architecture pipeline was exact, at 8-16x less cost.
  Set ``PII_NLP_MODEL=en_core_web_sm`` (or a path) if you add NER entities
  such as PERSON / LOCATION.

* Candidate windows. Presidio cost is linear in text length and dominated by
  the phone recognizer (8 region passes of ``phonenumbers``). A cheap superset
  regex finds every place any configured recognizer *could* match; Presidio
  then runs only on small windows around those candidates and offsets are mapped
  back. Text with no candidates costs one regex pass. ``PII_PREFILTER=0``
  disables this (full-text analysis).

  Windows are analysed one call each: a single call over joined windows was
  measured *slower* (Presidio cost is super-linear in text length).

* Analyse once. ``analyze()`` results are reused for redaction instead of
  re-running the analyzer (the old code analysed every message twice).

* Off the event loop. Presidio is CPU-bound pure Python, so threads only
  contend for the GIL (measured: worse than inline under concurrency). The
  prefilter runs inline; texts with candidates go to a small process pool
  (``PII_WORKERS``, default 2; 0 = inline). A broken pool falls back to inline
  analysis -- never to skipping PII.

* ``PII_PHONE_REGIONS`` (e.g. ``US``) narrows the phone recognizer, its
  dominant cost (~2.3x faster on dense text). National-format numbers of
  dropped regions are then missed (``+CC`` international numbers are not), so
  the default keeps Presidio's 8 regions.
"""
from __future__ import annotations

import os
import re
from bisect import bisect_left, bisect_right

from presidio_analyzer import AnalyzerEngine, Pattern, PatternRecognizer, RecognizerResult
from presidio_analyzer.nlp_engine import SpacyNlpEngine
from presidio_anonymizer import AnonymizerEngine

_SECRET_PATTERNS = [
    Pattern("openai_key", r"sk-[A-Za-z0-9]{20,}", 0.9),
    Pattern("aws_key", r"AKIA[0-9A-Z]{16}", 0.9),
    Pattern("bearer", r"(?i)bearer\s+[A-Za-z0-9._\-]{20,}", 0.6),
]
ENTITIES = ["CREDIT_CARD", "US_SSN", "EMAIL_ADDRESS",
            "PHONE_NUMBER", "IP_ADDRESS", "API_SECRET"]

_NLP_MODEL = os.getenv("PII_NLP_MODEL", "blank")
_PREFILTER = os.getenv("PII_PREFILTER", "1") not in ("0", "false", "no")
_MARGIN = int(os.getenv("PII_WINDOW_MARGIN", "96"))
_WORKERS = int(os.getenv("PII_WORKERS", "2"))
_PHONE_REGIONS = tuple(r.strip().upper() for r in os.getenv("PII_PHONE_REGIONS", "").split(",") if r.strip())

# Superset of everything the configured recognizers can match:
#   EMAIL needs '@'; API_SECRET needs 'sk-' / 'AKIA' / 'bearer';
#   IPv6 needs '::' or hex:hex:hex;  CREDIT_CARD / US_SSN / IPv4 / PHONE need
#   >= 4 digits whose digit blocks are separated by <= 4 non-digits
#   (phonenumbers' _PUNCTUATION_LIMIT is {0,4}; the shortest *valid* national
#   number across Presidio's default regions is 4 digits -- DE).
# If you add entities, extend this pattern, or set PII_PREFILTER=0.
_CANDIDATE = re.compile(
    r"@|sk-|AKIA|(?i:bearer)|::|[0-9A-Fa-f]{1,4}:[0-9A-Fa-f]{1,4}:[0-9A-Fa-f]{1,4}"
    r"|\d(?:\D{0,4}\d){3,}"
)
_WS = re.compile(r"\s")


def _nlp_engine() -> SpacyNlpEngine:
    if _NLP_MODEL == "blank":
        import spacy
        eng = SpacyNlpEngine(models=[{"lang_code": "en", "model_name": "blank:en"}])
        eng.nlp = {"en": spacy.blank("en")}      # tokenizer only; no download, no NER
        return eng
    eng = SpacyNlpEngine(models=[{"lang_code": "en", "model_name": _NLP_MODEL}])
    eng.load()
    return eng


_analyzer = AnalyzerEngine(nlp_engine=_nlp_engine(), supported_languages=["en"])
_analyzer.registry.add_recognizer(
    PatternRecognizer(supported_entity="API_SECRET", patterns=_SECRET_PATTERNS)
)
if _PHONE_REGIONS:
    from presidio_analyzer.predefined_recognizers import PhoneRecognizer
    _analyzer.registry.remove_recognizer("PhoneRecognizer")
    _analyzer.registry.add_recognizer(PhoneRecognizer(supported_regions=_PHONE_REGIONS))
_anonymizer = AnonymizerEngine()


def _windows(text: str, margin: int = _MARGIN):
    """Merged [start, end) windows around candidate spans, snapped outward to
    whitespace so no token (and no regex \\b boundary) is cut in half.

    O(n + k log n): windows are merged first, then only merged boundaries are
    snapped via bisect over a whitespace index. (Snapping char-by-char per
    candidate was O(n^2) on hostile input such as 100 KB of '@'.)"""
    n = len(text)
    raw = []
    for m in _CANDIDATE.finditer(text):
        s, e = max(0, m.start() - margin), min(n, m.end() + margin)
        if raw and s <= raw[-1][1]:
            raw[-1][1] = max(raw[-1][1], e)
        else:
            raw.append([s, e])
    if not raw:
        return []
    ws = [m.start() for m in _WS.finditer(text)]
    out = []
    for s, e in raw:
        i = bisect_right(ws, s - 1) - 1          # last whitespace at index <= s-1
        s = ws[i] + 1 if i >= 0 else 0
        j = bisect_left(ws, e)                   # first whitespace at index >= e
        e = ws[j] if j < len(ws) else n
        if out and s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return out


def analyze(text: str, language: str = "en") -> list[RecognizerResult]:
    """Presidio analysis of ``text`` (windowed when the prefilter is enabled)."""
    if not text:
        return []
    if not _PREFILTER:
        return _analyzer.analyze(text=text, entities=ENTITIES, language=language)
    return _analyze_windows(text, _windows(text), language)


def _analyze_windows(text, wins, language="en"):
    out = []
    for s, e in wins:
        for r in _analyzer.analyze(text=text[s:e], entities=ENTITIES, language=language):
            r.start += s
            r.end += s
            out.append(r)
    return out


def redact(text: str, results: list[RecognizerResult]) -> str:
    """Apply redaction using already-computed analyzer results."""
    if not results:
        return text
    return _anonymizer.anonymize(text=text, analyzer_results=results).text


def scan_and_redact(text: str, language: str = "en"):
    """Return (redacted_text, [entity_types_found]). Empty list => nothing found."""
    results = analyze(text, language)
    if not results:
        return text, []
    return redact(text, results), sorted({r.entity_type for r in results})


# ------------------------------------------------------------- async / pool
def _worker_scan(text: str, wins) -> tuple[str, list[str]]:
    """Runs inside a pool process (module imported fresh there)."""
    results = _analyze_windows(text, wins) if wins is not None else analyze(text)
    if not results:
        return text, []
    return redact(text, results), sorted({r.entity_type for r in results})


def _warm() -> bool:
    return bool(_worker_scan("warm up 415-555-0132 a@b.co", None))


_pool = None
_pool_ready = False


def _get_pool():
    global _pool
    if _pool is None and _WORKERS > 0:
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor
        # spawn: never fork a process that is running an event loop + threads
        _pool = ProcessPoolExecutor(_WORKERS, mp_context=mp.get_context("spawn"))
    return _pool


def start_pool(wait: bool = False):
    """Create the pool and import Presidio in every worker up front, so the
    first real request does not pay a multi-second cold start."""
    global _pool_ready
    pool = _get_pool()
    if pool is None:
        return
    futs = [pool.submit(_warm) for _ in range(_WORKERS)]
    pending = set(futs)

    def _done(f):
        global _pool_ready
        pending.discard(f)
        if not pending and all(not x.exception() for x in futs):
            _pool_ready = True
    for f in futs:
        f.add_done_callback(_done)
    if wait:
        for f in futs:
            f.result()


async def scan_and_redact_async(text: str) -> tuple[str, list[str]]:
    """scan_and_redact() without blocking the event loop on Presidio."""
    import asyncio
    from concurrent.futures.process import BrokenProcessPool
    global _pool, _pool_ready
    if not text:
        return text, []
    wins = _windows(text) if _PREFILTER else None
    if wins is not None and not wins:
        return text, []                       # no candidates: nothing to find
    pool = _get_pool() if _pool_ready else None
    if pool is None:                          # inline mode, or pool still warming
        return _worker_scan(text, wins)
    try:
        return await asyncio.get_running_loop().run_in_executor(pool, _worker_scan, text, wins)
    except BrokenProcessPool:
        _pool, _pool_ready = None, False      # rebuild + re-warm in the background
        start_pool()
        return _worker_scan(text, wins)       # fail-safe: inline, never skip
