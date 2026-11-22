"""Mock guardrails service and System-2 judge, served through httpx.MockTransport
(no sockets). Each charges its declared cost on the virtual clock and counts the
work it was asked to do.

Scores are read from tags embedded in the workload text, e.g.
    "... <<inj=0.55 tox=0.01 truth=1 s2=1>>"
  inj / tox  what System 1 outputs for this text (default 0.02 / 0.01)
  truth      ground truth: 1 = should be stopped                (default 0)
  s2         what the System-2 judge will say (default = truth; set it to
             the opposite to model judge errors)"""
import asyncio
import json
import math
import re
from collections import Counter

import httpx

from costs import COSTS, MAX_WINDOWS, WINDOW_CHARS

_TAG = re.compile(r"<<((?:\s*[a-z0-9]+=[0-9.]+)+)\s*>>")     # <<k=v k=v>> only


def tags(text: str) -> dict:
    out = {}
    for m in _TAG.finditer(text):
        for kv in m.group(1).split():
            k, _, v = kv.partition("=")
            out[k] = float(v)
    return out


def windows(text: str) -> int:
    return min(MAX_WINDOWS, max(1, math.ceil(len(text) / WINDOW_CHARS)))


class MockGuardrails:
    """mode: 'composite' (two single-label models, heads concurrent),
    'multilabel' (one System-1 model, one pass for all labels),
    'none' (no System 1: every label comes back 0.5, i.e. uncertain)."""

    def __init__(self, mode: str, single_label_roles=()):
        self.mode = mode
        self.single_label_roles = set(single_label_roles)   # mirrors DECISION_SINGLE_LABEL_ROLES
        self.c = Counter()

    def _s1(self, text, label):
        t = tags(text)
        if self.mode == "none":
            return 0.5
        return t.get("inj", 0.02) if label == "injection" else t.get("tox", 0.01)

    async def handler(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        self.c["round_trips"] += 1
        self.c["bytes_in"] += len(request.content)
        cost = COSTS["hop_ms"]
        if request.url.path == "/v1/decide":
            out = []
            for it in body["items"]:
                labels = it.get("labels") or ["injection", "toxicity"]
                w = windows(it["text"])
                mode = "composite" if it.get("role") in self.single_label_roles else self.mode
                if mode == "multilabel":
                    passes, ms = w, COSTS["s1_window_ms"] * w
                elif mode == "composite":
                    per = {"injection": COSTS["injection_window_ms"], "toxicity": COSTS["toxicity_window_ms"]}
                    passes, ms = w * len(labels), max(per[l] for l in labels) * w
                else:
                    passes, ms = 0, 0.0
                cost += ms
                self.c["items"] += 1
                self.c["passes"] += passes
                out.append({"scores": {l: self._s1(it["text"], l) for l in labels},
                            "windows": w, "passes": passes})
            await asyncio.sleep(cost / 1000)
            return httpx.Response(200, json={"decisions": out})
        if request.url.path == "/v1/classify":                 # legacy API, old server
            texts = body["texts"] if "texts" in body else [body["text"]]
            checks = body.get("checks", ["injection", "toxicity"])
            per = {"injection": COSTS["injection_window_ms"], "toxicity": COSTS["toxicity_window_ms"]}
            res = []
            for t in texts:
                w = windows(t)
                cost += max(per[c] for c in checks) * w         # old server ran checks concurrently
                self.c["items"] += 1
                self.c["passes"] += w * len(checks)
                res.append({c: {"score": self._s1(t, c)} for c in checks})
            await asyncio.sleep(cost / 1000)
            return httpx.Response(200, json={"results": res if "texts" in body else res[0]})
        return httpx.Response(404)

    def transport(self):
        return httpx.MockTransport(self.handler)


class MockJudge:
    """OpenAI-compatible judge returning first-token logprobs for Yes / No."""

    def __init__(self):
        self.c = Counter()

    async def handler(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        text = body["messages"][-1]["content"]
        t = tags(text)
        verdict = t.get("s2", t.get("truth", 0.0)) >= 0.5
        self.c["calls"] += 1
        await asyncio.sleep((COSTS["hop_ms"] + COSTS["s2_ms"]) / 1000)
        p = 0.95 if verdict else 0.05
        choice = {"index": 0, "message": {"role": "assistant", "content": "Yes" if verdict else "No"},
                  "logprobs": {"content": [{"token": "Yes" if verdict else "No",
                                            "logprob": math.log(max(p, 1 - p)),
                                            "top_logprobs": [{"token": "Yes", "logprob": math.log(p)},
                                                             {"token": "No", "logprob": math.log(1 - p)}]}]},
                  "finish_reason": "stop"}
        return httpx.Response(200, json={"choices": [choice]})

    def transport(self):
        return httpx.MockTransport(self.handler)
