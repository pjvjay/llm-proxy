"""Guardrails classifier service.

POST /v1/decide    {"items": [{"text", "role", "tool"}]}  -> {"decisions": [{"scores": {label: p},
                                                                              "windows": n, "passes": k}]}
    System-1 decision endpoint used by the proxy. Returns calibrated per-label
    probabilities for every label relevant to the item's role; the proxy turns
    them into allow / block / escalate with per-role confidence bands.
    DECISION_BACKEND selects how:
      composite        the two existing single-label models (injection, toxicity),
                       heads run concurrently; one forward pass per head per window.
      onnx_multilabel  one multi-label System-1 decision model: ONE forward pass per
                       window produces every label. See _MultiLabelOnnxBackend for
                       the expected model directory layout.

POST /v1/classify  {"texts": [...], "checks": [...]}   -> {"results": [{check: {score, windows}}]}
                   {"text": "...", "checks": [...]}    -> same, single-element list (legacy shape)

Scoring is *windowed*, not truncated: the text is tokenised into overlapping
512-token windows (stride 384) and the score is the max over windows, so an
injection appended after the 512th token of a long tool result is still seen.
Windows are scored in one batched run; bench/bench_windows.py shows batching is
roughly a wash versus sequential runs on CPU, but the tokenizer produces the
batch for free (`return_overflowing_tokens`) so it is the simpler path. Very
long inputs are capped at MAX_WINDOWS windows (first, last, and evenly spaced
ones in between) to bound worst-case cost.

Threading: the old default -- 4 worker threads, each ONNX session using every
core -- runs 4x more compute threads than cores. Worker threads x intra-op
threads is now kept <= the core count: GUARDRAILS_WORKERS (default 2, so the two
composite heads can run concurrently) and ORT_INTRA_THREADS / TORCH_THREADS
(default cores // workers). The best split between the two is hardware-specific
and must be tuned on dedicated hardware, not a shared VM."""
import asyncio
import json
import logging
import os
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from fastapi import FastAPI
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel

import manifest

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"),
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("guardrails")

MODELS_ROOT = os.getenv("MODELS_ROOT", "/models")
def _model_dir(env: str, default: str) -> str:
    """Configured dir if it exists, else the model-init default. Keeps an old
    .env (INJECTION_MODEL_DIR=/models/prompt-guard-2-onnx) working."""
    configured = os.getenv(env)
    if configured and not os.path.isdir(configured) and os.path.isdir(default):
        logging.getLogger("guardrails").warning("%s=%s not found; using %s", env, configured, default)
        return default
    return configured or default


MODEL_DIRS = {
    "injection": _model_dir("INJECTION_MODEL_DIR", f"{MODELS_ROOT}/injection"),
    "toxicity":  _model_dir("TOXICITY_MODEL_DIR", f"{MODELS_ROOT}/toxicity"),
}
POSITIVE = {
    "injection": int(os.getenv("INJECTION_POS_LABEL", "1")),
    "toxicity":  int(os.getenv("TOXICITY_POS_LABEL", "1")),
}
# auto: ONNX if the model dir has an ONNX file (prepare_models.py only keeps
# one that it validated against the PyTorch model), else PyTorch weights.
BACKEND = {
    "injection": os.getenv("BACKEND_INJECTION", "auto"),
    "toxicity":  os.getenv("BACKEND_TOXICITY", "auto"),
}
ONNX_FILE = os.getenv("ONNX_FILE", "model_quantized.onnx")
_ONNX_CANDIDATES = (ONNX_FILE, "model_quantized.onnx", "model.onnx")
_TORCH_WEIGHTS = ("model.safetensors", "pytorch_model.bin")
# OWASP LLM03: refuse to load models whose manifest doesn't match. A missing
# manifest is only a warning unless REQUIRE_MODEL_MANIFEST=1.
REQUIRE_MANIFEST = os.getenv("REQUIRE_MODEL_MANIFEST", "0") == "1"
ALLOW_UNVERIFIED = os.getenv("ALLOW_UNVERIFIED_MODELS", "0") == "1"
WINDOW = int(os.getenv("WINDOW_TOKENS", "512"))
STRIDE = int(os.getenv("WINDOW_STRIDE", "128"))      # HF semantics: overlap between windows
MAX_WINDOWS = int(os.getenv("MAX_WINDOWS", "8"))

_CORES = os.cpu_count() or 4
WORKERS = int(os.getenv("GUARDRAILS_WORKERS", "2"))
INTRA = int(os.getenv("ORT_INTRA_THREADS", str(max(1, _CORES // WORKERS))))
TORCH_THREADS = int(os.getenv("TORCH_THREADS", str(max(1, _CORES // WORKERS))))

# auto: a System-1 decision model (e.g. an OpenJev export) is used when its
# slot contains labels.json; otherwise the composite of the stock classifiers.
DECISION_BACKEND = os.getenv("DECISION_BACKEND", "auto")
DECISION_MODEL_DIR = os.getenv("DECISION_MODEL_DIR", f"{MODELS_ROOT}/decision")
# Labels each role is screened for. Tool results: injection only by default
# (the keyword / toxicity categories are for prompts, not for web pages).
ROLE_LABELS = {
    "user": tuple(os.getenv("LABELS_USER", "injection,toxicity").split(",")),
    "tool": tuple(os.getenv("LABELS_TOOL", "injection").split(",")),
    "assistant": tuple(os.getenv("LABELS_ASSISTANT", "toxicity").split(",")),
}

# Roles routed to the dedicated single-label models even when a multi-label
# decision model is loaded. A multi-label model pays one full pass for every
# label; for a role that needs ONE label (responses: toxicity only) a small
# dedicated head can be cheaper. Which is cheaper depends on the real models --
# see bench/controlled (Sweep A) -- so this is explicit config, not automatic.
SINGLE_LABEL_ROLES = {r for r in os.getenv("DECISION_SINGLE_LABEL_ROLES", "").split(",") if r}

app = FastAPI(title="guardrails")
_backends: dict = {}
_decider = None
_composite = None
_supply_chain: dict = {"status": "not checked"}
METRICS: Counter = Counter()


def _onnx_file(path):
    for name in _ONNX_CANDIDATES:
        if os.path.exists(os.path.join(path, name)):
            return os.path.join(path, name)
    return None


def _backend_kind(name, path):
    kind = BACKEND[name]
    if kind != "auto":
        return kind
    if _onnx_file(path):
        return "onnx"
    if any(os.path.exists(os.path.join(path, w)) for w in _TORCH_WEIGHTS):
        return "torch"
    raise RuntimeError(
        f"no model found for '{name}' in {path} (expected an ONNX file or PyTorch weights). "
        "With docker compose the model-init service prepares it; see README 'Models'.")
_pool = ThreadPoolExecutor(max_workers=WORKERS)


def _softmax(x):
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def _select_windows(n: int) -> list[int]:
    """Indices of windows to score when a text has more than MAX_WINDOWS."""
    if n <= MAX_WINDOWS:
        return list(range(n))
    pts = np.linspace(0, n - 1, MAX_WINDOWS)
    return sorted({int(round(p)) for p in pts})


class _Base:
    def _encode(self, text, framework):
        enc = self.tok(text, truncation=True, max_length=WINDOW, stride=STRIDE,
                       return_overflowing_tokens=True, padding=True,
                       return_tensors=framework)
        enc.pop("overflow_to_sample_mapping", None)
        n = int(enc["input_ids"].shape[0])
        keep = _select_windows(n)
        if len(keep) != n:
            enc = {k: v[keep] for k, v in enc.items()}
        return enc, n


class _TorchBackend(_Base):
    def __init__(self, path):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        torch.set_num_threads(TORCH_THREADS)
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(path)
        self.model = AutoModelForSequenceClassification.from_pretrained(path)
        self.model.eval()

    def score(self, text, positive):
        enc, n = self._encode(text, "pt")
        with self.torch.no_grad():
            logits = self.model(**enc).logits.numpy()
        return float(_softmax(logits)[:, positive].max()), n


class _OnnxBackend(_Base):
    def __init__(self, path):
        import onnxruntime as ort
        from transformers import AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(path)
        so = ort.SessionOptions()
        so.intra_op_num_threads = INTRA
        so.inter_op_num_threads = 1
        self.file = _onnx_file(path)
        self.sess = ort.InferenceSession(self.file, so, providers=["CPUExecutionProvider"])
        self.wanted = {i.name for i in self.sess.get_inputs()}

    def score(self, text, positive):
        enc, n = self._encode(text, "np")
        feeds = {k: np.asarray(v).astype(np.int64) for k, v in enc.items() if k in self.wanted}
        logits = np.asarray(self.sess.run(None, feeds)[0])
        return float(_softmax(logits)[:, positive].max()), n


class _MultiLabelOnnxBackend(_Base):
    """A System-1 decision model: one forward pass -> a decision for every label.

    Model directory layout:
        <dir>/<ONNX_FILE>      sequence classifier, logits [batch, n_labels]
        <dir>/tokenizer files  (HF fast tokenizer)
        <dir>/labels.json      {"labels": ["injection", "toxicity", ...],
                                "activation": "sigmoid" | "softmax",
                                "temperature": {"injection": 1.0, ...},   # optional calibration
                                "context_template": "role: {role} | tool: {tool}\n",  # optional
                                "owasp": {"injection": ["LLM01:2025"], ...},  # optional OWASP map
                                "name": "...", "version": "..."}             # optional, shown in /health
        <dir>/MANIFEST.sha256  optional; verified before loading (OWASP LLM03)
    With "softmax", a label named by "none_label" (e.g. "safe") is dropped from
    the output. "context_template" is prepended to the text when the model was
    trained with role / tool context, so the same sentence can be judged
    differently in a user message and in a tool result.

    If the decision model you deploy ships in another form, this class is the
    only thing that changes: anything exposing decide(text, role, tool) ->
    ({label: probability}, windows, passes) plugs into /v1/decide.
    `passes` counts window-level forward passes (the unit of compute): here it
    equals `windows`; the composite decider pays windows x heads."""

    def __init__(self, path, session=None, tokenizer=None):
        from transformers import AutoTokenizer
        self.tok = tokenizer or AutoTokenizer.from_pretrained(path)
        with open(f"{path}/labels.json") as f:
            cfg = json.load(f)
        self.labels = cfg["labels"]
        self.activation = cfg.get("activation", "sigmoid")
        self.none_label = cfg.get("none_label")
        self.temperature = np.array([float(cfg.get("temperature", {}).get(l, 1.0))
                                     for l in self.labels], dtype=np.float32)
        self.template = cfg.get("context_template")
        self.owasp = {k: list(v) for k, v in (cfg.get("owasp") or {}).items()}
        self.meta = {"name": cfg.get("name", os.path.basename(path.rstrip("/"))),
                     "version": cfg.get("version"), "labels": self.labels}
        if session is None:
            import onnxruntime as ort
            so = ort.SessionOptions()
            so.intra_op_num_threads = INTRA
            so.inter_op_num_threads = 1
            session = ort.InferenceSession(_onnx_file(path), so, providers=["CPUExecutionProvider"])
        self.sess = session
        self.wanted = {i.name for i in self.sess.get_inputs()}

    def decide(self, text, role="user", tool=None, labels=None):
        if self.template:
            text = self.template.format(role=role, tool=tool or "-") + text
        enc, n = self._encode(text, "np")
        feeds = {k: np.asarray(v).astype(np.int64) for k, v in enc.items() if k in self.wanted}
        logits = np.asarray(self.sess.run(None, feeds)[0], dtype=np.float32) / self.temperature
        probs = (1 / (1 + np.exp(-logits))) if self.activation == "sigmoid" else _softmax(logits)
        per_label = probs.max(axis=0)                      # max over windows
        want = set(labels) if labels else None
        out = {l: float(p) for l, p in zip(self.labels, per_label)
               if l != self.none_label and (want is None or l in want)}
        return out, n, n            # n window-passes, every label from the same pass


class _CompositeDecider:
    """System-1 interface over the existing single-label models: one forward
    pass per head; heads for an item run concurrently on the pool."""

    def __init__(self, backends):
        self.backends = backends

    async def decide_async(self, loop, text, labels):
        heads = [l for l in labels if l in self.backends]
        futs = [loop.run_in_executor(_pool, self.backends[l].score, text, POSITIVE[l]) for l in heads]
        res = await asyncio.gather(*futs)
        scores = {l: s for l, (s, _) in zip(heads, res)}
        windows = max((n for _, n in res), default=0)
        return scores, windows, sum(n for _, n in res)


def _verify_supply_chain():
    global _supply_chain
    results = manifest.verify_tree(MODELS_ROOT)
    bad = {p: probs for p, (st, probs) in results.items() if st != "verified"}
    if not results:
        _supply_chain = {"status": "unverified", "detail": f"no {manifest.NAME} under {MODELS_ROOT}"}
        if REQUIRE_MANIFEST and not ALLOW_UNVERIFIED:
            raise RuntimeError(_supply_chain["detail"] + " and REQUIRE_MODEL_MANIFEST=1")
        log.warning("model artifacts unverified: %s", _supply_chain["detail"])
    elif bad:
        _supply_chain = {"status": "mismatch", "detail": bad}
        if not ALLOW_UNVERIFIED:
            raise RuntimeError(f"model manifest verification failed (OWASP LLM03): {bad}")
        log.error("model manifest mismatch, continuing because ALLOW_UNVERIFIED_MODELS=1: %s", bad)
    else:
        _supply_chain = {"status": "verified", "manifests": sorted(results)}


@app.on_event("startup")
def _load():
    global _decider, _composite
    _verify_supply_chain()
    use_decision = DECISION_BACKEND == "onnx_multilabel" or (
        DECISION_BACKEND == "auto" and os.path.exists(os.path.join(DECISION_MODEL_DIR, "labels.json")))
    if use_decision:
        _decider = _MultiLabelOnnxBackend(DECISION_MODEL_DIR)
        log.info("System-1 decision model loaded from %s: %s", DECISION_MODEL_DIR, _decider.meta)
    for name, path in MODEL_DIRS.items():
        if use_decision and not os.path.isdir(path):
            continue                      # single-label models optional with a decision model
        kind = _backend_kind(name, path)
        _backends[name] = _TorchBackend(path) if kind == "torch" else _OnnxBackend(path)
        log.info("loaded %s (%s) from %s", name, kind, path)
    _composite = _CompositeDecider(_backends)
    if _decider is None:
        _decider = _composite
    # Warm-up: the first inference (lazy kernel init, allocator) can take
    # seconds on CPU. Do it here, before /health reports ok, so the first real
    # request can't time out in the proxy and be fail-closed for the wrong reason.
    t0 = time.perf_counter()
    for name, b in _backends.items():
        b.score("warm-up request", POSITIVE[name])
    if isinstance(_decider, _MultiLabelOnnxBackend):
        _decider.decide("warm-up request")
    log.info("warm-up done in %.2fs", time.perf_counter() - t0)


def _score_batch(checks, texts):
    """One pool job per request: all texts x all checks, sequentially on the
    worker's threads. Intra-op parallelism inside each run uses the cores."""
    out = []
    for t in texts:
        r = {}
        for c in checks:
            if c in _backends:
                s, n = _backends[c].score(t, POSITIVE[c])
                r[c] = {"score": s, "windows": n}
        out.append(r)
    return out


class ClassifyRequest(BaseModel):
    text: str | None = None
    texts: list[str] | None = None
    checks: list[str] = ["injection", "toxicity"]


@app.post("/v1/classify")
async def classify(req: ClassifyRequest):
    METRICS["classify_requests_total"] += 1
    texts = req.texts if req.texts is not None else [req.text or ""]
    loop = asyncio.get_event_loop()
    results = await loop.run_in_executor(_pool, _score_batch, req.checks, texts)
    if req.texts is None:                       # legacy single-text shape
        return {"results": results[0]}
    return {"results": results}


class DecideItem(BaseModel):
    text: str
    role: str = "user"
    tool: str | None = None
    labels: list[str] | None = None


class DecideRequest(BaseModel):
    items: list[DecideItem]


@app.post("/v1/decide")
async def decide(req: DecideRequest):
    loop = asyncio.get_event_loop()
    t0 = time.perf_counter()
    out = []
    for it in req.items:
        labels = it.labels or list(ROLE_LABELS.get(it.role, ROLE_LABELS["user"]))
        single = it.role in SINGLE_LABEL_ROLES and all(l in _backends for l in labels)
        if isinstance(_decider, _CompositeDecider) or single:
            scores, n, passes = await (_composite or _decider).decide_async(loop, it.text, labels)
        else:
            scores, n, passes = await loop.run_in_executor(
                _pool, _decider.decide, it.text, it.role, it.tool, labels)
        item = {"scores": scores, "windows": n, "passes": passes}
        declared = getattr(_decider, "owasp", None) if not single else None
        if declared:
            item["owasp"] = {l: declared[l] for l in scores if l in declared}
        out.append(item)
        METRICS["decide_items_total"] += 1
        METRICS["decide_passes_total"] += passes
    METRICS["decide_requests_total"] += 1
    METRICS["decide_seconds_sum"] += time.perf_counter() - t0
    return {"decisions": out}


@app.get("/metrics", response_class=PlainTextResponse)
def metrics():
    lines = []
    for k, v in sorted(METRICS.items()):
        kind = "counter"
        lines += [f"# TYPE guardrails_{k} {kind}", f"guardrails_{k} {v}"]
    lines.append(f'guardrails_supply_chain_verified {1 if _supply_chain.get("status") == "verified" else 0}')
    return "\n".join(lines) + "\n"


@app.get("/health")
def health():
    if isinstance(_decider, _MultiLabelOnnxBackend):
        decision = {"kind": "onnx_multilabel", "dir": DECISION_MODEL_DIR, **_decider.meta,
                    "owasp": _decider.owasp}
    else:
        decision = {"kind": "composite", "labels": sorted(_backends)}
    return {"status": "ok" if (_backends or _decider) else "loading",
            "loaded": {k: type(b).__name__.strip("_").replace("Backend", "").lower()
                       for k, b in _backends.items()},
            "decision_backend": decision, "supply_chain": _supply_chain,
            "workers": WORKERS, "intra_op_threads": INTRA, "window": WINDOW,
            "stride": STRIDE, "max_windows": MAX_WINDOWS}
