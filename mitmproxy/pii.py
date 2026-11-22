"""PII / secret detection + redaction via Presidio, used on BOTH the request
(before prompts / tool results leave for a third-party model) and the response.
Adds a custom recognizer for API keys / bearer tokens on top of Presidio's built-ins.

Performance notes (see bench/bench_pii.py for numbers):

* Presidio runs the spaCy pipeline on every analyze() call even when every
  requested entity is pattern-based -- which is the case for the entity list
  below. `PII_NLP_MODEL=blank` (default) uses a tokenizer-only spaCy pipeline,
  which is functionally equivalent for these entities and avoids the NER cost.
  Set `PII_NLP_MODEL=en_core_web_sm` if you add NER entities (PERSON, LOCATION).

* analyze() is CPU-bound, pure Python and holds the GIL. It must never run on
  the mitmproxy event loop (it would serialise every concurrent flow and stall
  streaming). Measured options (bench/bench_pii.py, 32 concurrent flows):
    - inline on the loop: loop stalls up to seconds on long inputs;
    - thread pool: keeps the loop alive but GIL contention between the worker
      threads and the loop thread makes it *slower* than inline under load;
    - process pool: best throughput and tail latency, loop stall ~25-50 ms.
  `PII_EXECUTOR=process` (default) | `thread` | `inline`. Workers are spawned
  (not forked) so they don't inherit mitmproxy's threads; each loads its own
  Presidio at start-up (~1 s). `scan_many_async` screens all new units of a
  turn in one IPC round-trip.

* The original code analysed once to detect and again to redact. `scan_and_redact`
  analyses once and reuses the results for anonymisation.

* tldextract (used by Presidio's URL/email recognisers) tries to fetch the public
  suffix list from the network on first use. In an egress-less container that is
  a multi-second stall on the first request, so `warm_up()` triggers it at load.
"""
import asyncio
import logging
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

from presidio_analyzer import AnalyzerEngine, Pattern, PatternRecognizer
from presidio_analyzer.nlp_engine import NerModelConfiguration, NlpEngineProvider, SpacyNlpEngine
from presidio_anonymizer import AnonymizerEngine

log = logging.getLogger("firewall.pii")

_SECRET_PATTERNS = [
    Pattern("openai_key", r"sk-[A-Za-z0-9_\-]{20,}", 0.9),
    Pattern("aws_key", r"AKIA[0-9A-Z]{16}", 0.9),
    Pattern("github_token", r"gh[pousr]_[A-Za-z0-9]{36,}", 0.9),
    Pattern("bearer", r"(?i)bearer\s+[A-Za-z0-9._\-]{20,}", 0.6),
]
# All pattern-based; none of these need a NER model.
_ENTITIES = ["CREDIT_CARD", "US_SSN", "EMAIL_ADDRESS",
             "PHONE_NUMBER", "IP_ADDRESS", "API_SECRET"]

_NLP_MODEL = os.getenv("PII_NLP_MODEL", "blank")
_EXECUTOR = os.getenv("PII_EXECUTOR", "process")
_WORKERS = int(os.getenv("PII_WORKERS", "2"))
_pool = None


class _BlankSpacyEngine(SpacyNlpEngine):
    """Tokenizer-only spaCy pipeline: no tagger / parser / NER, no model download."""

    def __init__(self):
        super().__init__(models=[{"lang_code": "en", "model_name": "blank"}],
                         ner_model_configuration=NerModelConfiguration(labels_to_ignore=[]))
        self.load()

    def load(self):
        import spacy
        self.nlp = {"en": spacy.blank("en")}

    def is_loaded(self):
        return bool(self.nlp)


def _build_analyzer() -> AnalyzerEngine:
    if _NLP_MODEL == "blank":
        engine = _BlankSpacyEngine()
    else:
        engine = NlpEngineProvider(nlp_configuration={
            "nlp_engine_name": "spacy",
            "models": [{"lang_code": "en", "model_name": _NLP_MODEL}],
        }).create_engine()
    analyzer = AnalyzerEngine(nlp_engine=engine, supported_languages=["en"])
    analyzer.registry.add_recognizer(
        PatternRecognizer(supported_entity="API_SECRET", patterns=_SECRET_PATTERNS))
    return analyzer


_analyzer = _build_analyzer()
_anonymizer = AnonymizerEngine()


def scan_and_redact(text: str, language: str = "en"):
    """Return (redacted_text, [entity_types_found]). Empty list => nothing found.
    Synchronous and CPU-bound -- call via scan_and_redact_async from the event loop."""
    if not text:
        return text, []
    results = _analyzer.analyze(text=text, entities=_ENTITIES, language=language)
    if not results:
        return text, []
    redacted = _anonymizer.anonymize(text=text, analyzer_results=results).text
    return redacted, sorted({r.entity_type for r in results})


def scan_many(texts: list[str], language: str = "en") -> list[tuple[str, list]]:
    return [scan_and_redact(t, language) for t in texts]


def _get_pool():
    global _pool
    if _pool is None:
        if _EXECUTOR == "process":
            # spawned workers must be able to `import pii`; PYTHONPATH is the one
            # channel that survives spawn regardless of how mitmproxy set sys.path.
            here = os.path.dirname(os.path.abspath(__file__))
            os.environ["PYTHONPATH"] = here + os.pathsep + os.environ.get("PYTHONPATH", "")
            _pool = ProcessPoolExecutor(max_workers=_WORKERS,
                                        mp_context=multiprocessing.get_context("spawn"),
                                        initializer=warm_up)
        elif _EXECUTOR == "thread":
            _pool = ThreadPoolExecutor(max_workers=_WORKERS, thread_name_prefix="pii")
    return _pool


async def scan_many_async(texts: list[str], language: str = "en") -> list[tuple[str, list]]:
    """All new units of a turn in one executor job (one IPC round-trip)."""
    if not texts:
        return []
    pool = _get_pool()
    if pool is None:                       # PII_EXECUTOR=inline (tests / debugging)
        return scan_many(texts, language)
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(pool, scan_many, texts, language)


async def scan_and_redact_async(text: str, language: str = "en"):
    return (await scan_many_async([text], language))[0]


def warm_up():
    """Trigger lazy initialisation (regex compilation, tldextract suffix list) off
    the request path. Safe to call more than once; used as the worker initializer."""
    try:
        scan_and_redact("warm-up bob@example.com 4111 1111 1111 1111 10.0.0.1 sk-" + "a" * 24)
    except Exception as exc:  # never fail startup because of warm-up
        log.warning("pii warm-up failed: %s", exc)


def start():
    """Create the worker pool and warm both it and the in-process analyzer.
    Called from the addon's load() hook so the first request doesn't pay for it."""
    warm_up()
    pool = _get_pool()
    if isinstance(pool, ProcessPoolExecutor):
        futs = [pool.submit(warm_up) for _ in range(_WORKERS)]
        for f in futs:
            f.result(timeout=60)


def shutdown():
    global _pool
    if _pool is not None:
        _pool.shutdown(wait=False, cancel_futures=True)
        _pool = None
