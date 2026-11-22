"""Zero-compute stand-in for the guardrails service, used by bench_transport.py
to isolate transport overhead (connection setup, framing, serialisation) from
model inference. Accepts the same /v1/classify shape as guardrails/server.py."""
import asyncio, math, os
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

# Optional simulated inference cost, per 512-token window (~2048 chars), per check.
_MS = {"injection": float(os.getenv("STUB_MS_INJECTION", "0")),
       "toxicity": float(os.getenv("STUB_MS_TOXICITY", "0"))}
_WINDOW_CHARS = 2048
_MAX_WINDOWS = int(os.getenv("MAX_WINDOWS", "8"))

class Req(BaseModel):
    text: str | None = None
    texts: list[str] | None = None
    checks: list[str] = ["injection", "toxicity"]

@app.post("/v1/classify")
async def classify(req: Req):
    texts = req.texts if req.texts is not None else [req.text or ""]
    delay = sum(min(_MAX_WINDOWS, max(1, math.ceil(len(t) / _WINDOW_CHARS))) * _MS.get(c, 0)
                for t in texts for c in req.checks)
    if delay:
        await asyncio.sleep(delay / 1000)
    res = [{c: {"score": 0.01, "windows": max(1, math.ceil(len(t) / _WINDOW_CHARS))} for c in req.checks} for t in texts]
    return {"results": res if req.texts is not None else res[0]}

@app.get("/health")
async def health():
    return {"status": "ok"}
