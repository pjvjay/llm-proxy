import asyncio
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from fastapi import FastAPI
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

app = FastAPI(title="guardrails")
_backends: dict = {}
_pool = ThreadPoolExecutor(max_workers=int(os.getenv("GUARDRAILS_WORKERS", "4")))


def _softmax(x):
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


class _TorchBackend:
    def __init__(self, path):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(path)
        self.model = AutoModelForSequenceClassification.from_pretrained(path)
        self.model.eval()

    def score(self, text, positive):
        enc = self.tok(text, return_tensors="pt", truncation=True, max_length=512)
        with self.torch.no_grad():
            logits = self.model(**enc).logits[0].numpy()
        return float(_softmax(logits[None])[0][positive])


class _OnnxBackend:
    def __init__(self, path):
        import onnxruntime as ort
        from transformers import AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(path)
        self.sess = ort.InferenceSession(f"{path}/{ONNX_FILE}",
                                         providers=["CPUExecutionProvider"])

    def score(self, text, positive):
        enc = self.tok(text, return_tensors="np", truncation=True, max_length=512)
        wanted = {i.name for i in self.sess.get_inputs()}
        feeds = {k: v.astype(np.int64) for k, v in enc.items() if k in wanted}
        logits = np.asarray(self.sess.run(None, feeds)[0])
        return float(_softmax(logits)[0][positive])


@app.on_event("startup")
def _load():
    for name, path in MODEL_DIRS.items():
        _backends[name] = (_TorchBackend(path) if BACKEND[name] == "torch"
                           else _OnnxBackend(path))


def _score(check, text):
    return _backends[check].score(text, POSITIVE[check])


class ClassifyRequest(BaseModel):
    text: str
    checks: list[str] = ["injection", "toxicity"]


@app.post("/v1/classify")
async def classify(req: ClassifyRequest):
    loop = asyncio.get_event_loop()
    pending = {c: loop.run_in_executor(_pool, _score, c, req.text)
               for c in req.checks if c in _backends}
    return {"results": {c: {"score": await fut} for c, fut in pending.items()}}


@app.get("/health")
def health():
    return {"status": "ok" if _backends else "loading",
            "loaded": list(_backends.keys())}
