"""Guardrails classification service.

Changes vs. the original (see docs/perf-report.md, B3):

* Sliding windows instead of truncation. The original tokenized with
  ``truncation=True, max_length=512``: anything after ~512 tokens was never
  scored, so an injection appended to a long tool result was invisible. Texts
  are now split into overlapping windows (WINDOW_OVERLAP tokens) and the
  text's score is the max over windows.
* Bounded cost. At most MAX_WINDOWS[check] windows per text; beyond that the
  first and last halves are scored (where injected instructions usually sit)
  and the response says ``"truncated": true`` (the proxy logs it). The middle
  of very long texts is then unscored -- a documented residual risk.
* Batched inference. All windows of all texts in a request go through each
  model in batches of up to INFER_BATCH[check] (per backend), padded per batch.
* Batch API. ``{"items": [{"text", "checks"}, ...]}`` scores many texts in one
  round trip; the original ``{"text", "checks"}`` form still works.
* Deadlines. The proxy sends ``deadline_ms`` (its remaining timeout). Work whose
  deadline passed is dropped -- before each forward pass and while queued at
  the gate -- and answered 504. Without this, requests the proxy had already
  abandoned kept the models busy and every later request queued behind them
  (a timeout death spiral seen in the end-to-end benchmark, B6).
* Per-model concurrency gate + thread budget. With the original defaults (4
  worker threads, torch using one intra-op thread per core) four concurrent
  requests ran 16 OpenMP threads on 4 cores and DeBERTa latency collapsed from
  ~1.1 s to ~43 s (B3 Q3). Measured best (B3c): MODEL_CONCURRENCY=2 forward
  passes per model, each with cpus // 2 intra-op threads -- giving one model
  all cores was erratic (any preemption stalls every OpenMP barrier).
  The gate is taken per window batch and is FIFO with direct hand-off (no
  barging by the releasing thread), so an already-waiting request gets the next
  slot; waiters leave the queue at their deadline. At 2 slots its end-to-end
  latency effect vs threading.Semaphore was within noise (perf report 3.9).
"""
import asyncio
import os
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel

MODEL_DIRS = {
    "injection": os.getenv("INJECTION_MODEL_DIR", "/models/prompt-guard-2-onnx"),
    "toxicity":  os.getenv("TOXICITY_MODEL_DIR", "/models/granite-hap-onnx"),
}
POSITIVE = {
    "injection": int(os.getenv("INJECTION_POS_LABEL", "1")),
    "toxicity":  int(os.getenv("TOXICITY_POS_LABEL", "1")),
}
BACKEND = {
    "injection": os.getenv("BACKEND_INJECTION", "torch"),
    "toxicity":  os.getenv("BACKEND_TOXICITY", "onnx"),
}
ONNX_FILE = os.getenv("ONNX_FILE", "model_quantized.onnx")
MAX_LEN = int(os.getenv("MAX_TOKENS", "512"))
OVERLAP = int(os.getenv("WINDOW_OVERLAP", "64"))
# Per-check window caps (B3): DeBERTa-base on CPU costs ~0.9-1.1 s per 512-token
# window, so more than ~2 windows cannot fit GUARDRAILS_TIMEOUT=2 s. 2 = head +
# tail (the original scored the head only); the cheap toxicity model gets 8.
MAX_WINDOWS = {
    "injection": int(os.getenv("MAX_WINDOWS_INJECTION", "2")),
    "toxicity":  int(os.getenv("MAX_WINDOWS_TOXICITY", "8")),
}
# Per-backend batch size (B3): batching windows made DeBERTa/PyTorch *slower*
# on CPU (5 windows: 5.6 s at batch 1 vs 8.7 s at 4), while the small int8 ONNX
# model gained a little from batch 4.
INFER_BATCH = {
    "injection": int(os.getenv("INFER_BATCH_INJECTION", "1")),
    "toxicity":  int(os.getenv("INFER_BATCH_TOXICITY", "4")),
}
MODEL_CONCURRENCY = int(os.getenv("MODEL_CONCURRENCY", "2"))  # 0 = unbounded (original)


def available_cpus() -> int:
    """CPUs this process may use: cgroup v2 quota, else affinity, else count."""
    try:
        quota, period = open("/sys/fs/cgroup/cpu.max").read().split()
        if quota != "max":
            return max(1, int(int(quota) / int(period)))
    except (OSError, ValueError):
        pass
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except AttributeError:
        return os.cpu_count() or 1


# 0 = auto: split the CPU budget across concurrent forward passes
INTRA_OP_THREADS = int(os.getenv("INTRA_OP_THREADS", "0")) or \
    max(1, available_cpus() // max(1, MODEL_CONCURRENCY))


class DeadlineExceeded(Exception):
    pass


class FairGate:
    """Counting semaphore with FIFO hand-off: a released slot goes straight to
    the oldest waiter, so a thread looping over many windows cannot barge back
    in ahead of requests that were already waiting. Waiters can give up at a
    deadline (they leave the queue; a slot handed over in the race is released)."""

    def __init__(self, n: int):
        self._free = n
        self._waiters: deque = deque()
        self._lock = threading.Lock()

    def acquire(self, timeout: float | None = None) -> bool:
        with self._lock:
            if self._free and not self._waiters:
                self._free -= 1
                return True
            ev = threading.Event()
            self._waiters.append(ev)
        if ev.wait(timeout):
            return True                       # slot handed over by release()
        with self._lock:
            if ev.is_set():                   # handed over just as we timed out
                return True
            self._waiters.remove(ev)
            return False

    def release(self):
        with self._lock:
            if self._waiters:
                self._waiters.popleft().set()
            else:
                self._free += 1

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, *exc):
        self.release()


app = FastAPI(title="guardrails")
_backends: dict = {}
_pool = ThreadPoolExecutor(max_workers=int(os.getenv("GUARDRAILS_WORKERS", "4")))


def _softmax(x):
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def select_windows(n: int, max_windows: int) -> list[int]:
    """Indices of windows to score: all, or first/last halves when over budget."""
    if max_windows <= 0 or n <= max_windows:
        return list(range(n))
    head = (max_windows + 1) // 2
    return list(range(head)) + list(range(n - (max_windows - head), n))


class _Backend:
    """Tokenize -> window -> batched forward -> max-over-windows per text."""

    batch = 1
    gate = None
    max_windows = 8

    def __init__(self, path):
        from transformers import AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(path)
        fair = os.getenv("GATE_FAIR", "1") not in ("0", "false", "no")
        self.gate = None if MODEL_CONCURRENCY <= 0 else \
            (FairGate(MODEL_CONCURRENCY) if fair else threading.Semaphore(MODEL_CONCURRENCY))

    def _forward_gated(self, ids, mask, deadline=None):
        if deadline is not None and time.monotonic() > deadline:
            raise DeadlineExceeded
        if self.gate is None:
            return self.forward(ids, mask)
        if isinstance(self.gate, FairGate):
            wait = None if deadline is None else max(0.0, deadline - time.monotonic())
            if not self.gate.acquire(wait):
                raise DeadlineExceeded
        else:
            self.gate.acquire()
        try:
            if deadline is not None and time.monotonic() > deadline:
                raise DeadlineExceeded
            return self.forward(ids, mask)
        finally:
            self.gate.release()

    def encode(self, texts, windowed=True, max_windows=None):
        max_windows = self.max_windows if max_windows is None else max_windows
        kw = dict(truncation=True, max_length=MAX_LEN)
        if windowed:
            kw.update(stride=OVERLAP, return_overflowing_tokens=True)
        enc = self.tok(list(texts), **kw)
        ids, mask = enc["input_ids"], enc["attention_mask"]
        owner = enc.get("overflow_to_sample_mapping") or list(range(len(texts)))
        by_text: dict[int, list[int]] = {}
        for w, t in enumerate(owner):
            by_text.setdefault(t, []).append(w)
        keep, meta = [], []
        for t in range(len(texts)):
            ws = by_text.get(t, [])
            sel = select_windows(len(ws), max_windows)
            keep += [(t, ws[i]) for i in sel]
            meta.append({"windows": len(ws), "truncated": len(sel) < len(ws)})
        return ids, mask, keep, meta

    def score_texts(self, texts, positive, windowed=True, batch=None, max_windows=None,
                    deadline=None):
        """-> (scores per text, meta per text)"""
        batch = batch or self.batch
        ids, mask, keep, meta = self.encode(texts, windowed, max_windows)
        # group windows by length so batches carry little padding
        order = sorted(range(len(keep)), key=lambda k: len(ids[keep[k][1]]))
        best = [0.0] * len(texts)
        for i in range(0, len(order), batch):
            chunk = order[i:i + batch]
            width = max(len(ids[keep[k][1]]) for k in chunk)
            b_ids = np.full((len(chunk), width), self.tok.pad_token_id or 0, dtype=np.int64)
            b_mask = np.zeros((len(chunk), width), dtype=np.int64)
            for r, k in enumerate(chunk):
                w = keep[k][1]
                b_ids[r, :len(ids[w])] = ids[w]
                b_mask[r, :len(mask[w])] = mask[w]
            probs = _softmax(self._forward_gated(b_ids, b_mask, deadline))[:, positive]
            for r, k in enumerate(chunk):
                t = keep[k][0]
                best[t] = max(best[t], float(probs[r]))
        return best, meta


class _TorchBackend(_Backend):
    def __init__(self, path):
        super().__init__(path)
        import torch
        from transformers import AutoModelForSequenceClassification
        torch.set_num_threads(INTRA_OP_THREADS)
        self.torch = torch
        self.model = AutoModelForSequenceClassification.from_pretrained(path)
        self.model.eval()

    def forward(self, ids, mask):
        with self.torch.inference_mode():
            return self.model(input_ids=self.torch.from_numpy(ids),
                              attention_mask=self.torch.from_numpy(mask)).logits.numpy()


class _OnnxBackend(_Backend):
    def __init__(self, path):
        super().__init__(path)
        import onnxruntime as ort
        so = ort.SessionOptions()
        so.intra_op_num_threads = INTRA_OP_THREADS
        self.sess = ort.InferenceSession(f"{path}/{ONNX_FILE}", so,
                                         providers=["CPUExecutionProvider"])
        self.wanted = {i.name for i in self.sess.get_inputs()}

    def forward(self, ids, mask):
        feeds = {"input_ids": ids, "attention_mask": mask}
        if "token_type_ids" in self.wanted:
            feeds["token_type_ids"] = np.zeros_like(ids)
        return np.asarray(self.sess.run(None, {k: v for k, v in feeds.items() if k in self.wanted})[0])


@app.on_event("startup")
def _load():
    for name, path in MODEL_DIRS.items():
        _backends[name] = (_TorchBackend(path) if BACKEND[name] == "torch"
                           else _OnnxBackend(path))
        _backends[name].batch = INFER_BATCH[name]
        _backends[name].max_windows = MAX_WINDOWS[name]


def _score_many(check, texts, deadline=None):
    scores, meta = _backends[check].score_texts(texts, POSITIVE[check], deadline=deadline)
    return [{"score": s, **m} for s, m in zip(scores, meta)]


class Item(BaseModel):
    text: str
    checks: list[str] = ["injection", "toxicity"]


class ClassifyRequest(BaseModel):
    text: str | None = None
    checks: list[str] = ["injection", "toxicity"]
    items: list[Item] | None = None
    deadline_ms: float | None = None      # caller's remaining budget


@app.post("/v1/classify")
async def classify(req: ClassifyRequest):
    deadline = None if not req.deadline_ms else time.monotonic() + req.deadline_ms / 1000
    items = req.items if req.items is not None else [Item(text=req.text or "", checks=req.checks)]
    # one batched job per model, models run concurrently
    per_check: dict[str, list[int]] = {}
    for i, it in enumerate(items):
        for c in it.checks:
            if c in _backends:
                per_check.setdefault(c, []).append(i)
    loop = asyncio.get_running_loop()
    futs = {c: loop.run_in_executor(_pool, _score_many, c, [items[i].text for i in idx], deadline)
            for c, idx in per_check.items()}
    out = [dict() for _ in items]
    try:
        for c, fut in futs.items():
            for i, r in zip(per_check[c], await fut):
                out[i][c] = r
    except DeadlineExceeded:
        for fut in futs.values():             # let siblings finish/abort, don't leak
            try:
                await fut
            except DeadlineExceeded:
                pass
        return JSONResponse({"error": "deadline exceeded"}, status_code=504)
    if req.items is None:
        return {"results": out[0]}
    return {"items": out}


@app.get("/health")
def health():
    return {"status": "ok" if _backends else "loading",
            "loaded": list(_backends.keys())}
