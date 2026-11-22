"""Keyword-scoring stand-in for the guardrails service (no models needed).
Same API as guardrails/server.py: {"texts": [...], "checks": [...]}."""
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class Req(BaseModel):
    text: str | None = None
    texts: list[str] | None = None
    checks: list[str] = ["injection", "toxicity"]

def _score(check, text):
    t = text.lower()
    if check == "injection":
        return 0.99 if ("ignore all previous" in t or "reveal your hidden" in t) else 0.02
    if check == "toxicity":
        return 0.97 if ("idiot" in t or "worthless" in t) else 0.01
    return 0.0

@app.post("/v1/classify")
async def classify(req: Req):
    texts = req.texts if req.texts is not None else [req.text or ""]
    res = [{c: {"score": _score(c, t), "windows": 1} for c in req.checks} for t in texts]
    return {"results": res if req.texts is not None else res[0]}

class Item(BaseModel):
    text: str
    role: str = "user"
    tool: str | None = None
    labels: list[str] | None = None

class DecideReq(BaseModel):
    items: list[Item]

@app.post("/v1/decide")
async def decide(req: DecideReq):
    out = []
    for it in req.items:
        labels = it.labels or ["injection", "toxicity"]
        out.append({"scores": {l: _score(l, it.text) for l in labels}, "windows": 1, "passes": 1})
    return {"decisions": out}

@app.get("/health")
async def health():
    return {"status": "ok"}
