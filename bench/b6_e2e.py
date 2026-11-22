"""B6 -- end to end through real mitmdump: original code vs new code.

  client --HTTP--> mitmdump(addon) --HTTP--> mock upstream (instant reply)
                        \\--HTTP--> guardrails (stub | stand-in models)

Variants
  baseline      : addon + guardrails server from the baseline git commit.
                  Its pii.py calls AnalyzerEngine(), which would download
                  en_core_web_lg; offline we point it at the en_core_web_sm-
                  architecture stand-in instead -- LIGHTER than what it really
                  loads, so baseline numbers here are optimistic.
  new           : this branch (incremental cache ON, PII process pool)
  new_nocache   : this branch with SCAN_CACHE=0 (full per-message rescan)
  new_onnx_base / new_onnx_xsmall[_nocache] : injection model served as int8
                  ONNX (deberta-v3-base / -xsmall size) instead of PyTorch fp32

Classifier modes
  stub     : guardrails answers instantly -> isolates proxy overhead
  standin  : real guardrails server with the stand-in models -> realistic cost

Each agent session uses fresh content (distinct seed), so nothing is served
from cache across sessions. Reported latency = proxied - direct-to-upstream.

    python bench/b6_e2e.py --mode stub
    python bench/b6_e2e.py --mode standin --turns 8 --sessions 2
"""
from __future__ import annotations

import argparse
import asyncio
import os
import shutil
import signal
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import httpx

import common
import corpus

BASELINE_REV = os.getenv("BASELINE_REV", "27d25c7")
UP, GR, MITM = 18801, 18802, 18803
TOOLS = [{"type": "function", "function": {"name": "search", "parameters": {"type": "object"}}}]

UPSTREAM_APP = r'''
import json, time
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
app = FastAPI()
@app.post("/v1/chat/completions")
async def cc(req: Request):
    body = await req.json()
    last_tool = sum(1 for m in body["messages"] if m.get("role") == "tool")
    if last_tool < body.get("max_tool_turns", 0):
        msg = {"role": "assistant", "content": None, "tool_calls": [{"id": f"call_{last_tool}",
               "type": "function", "function": {"name": "search", "arguments": json.dumps({"q": "weather in paris next week"})}}]}
        fr = "tool_calls"
    else:
        msg = {"role": "assistant", "content": "Here is a short summary of what I found about the topic you asked for. " * 3}
        fr = "stop"
    return JSONResponse({"id": "x", "object": "chat.completion", "created": int(time.time()), "model": "mock",
                         "choices": [{"index": 0, "message": msg, "finish_reason": fr}]})
'''

STUB_GUARDRAILS_APP = r'''
from fastapi import FastAPI
from pydantic import BaseModel
app = FastAPI()
class Item(BaseModel):
    text: str = ""
    checks: list[str] = ["injection", "toxicity"]
class Req(BaseModel):
    text: str | None = None
    checks: list[str] = ["injection", "toxicity"]
    items: list[Item] | None = None
@app.post("/v1/classify")
async def classify(r: Req):
    if r.items is not None:
        return {"items": [{c: {"score": 0.01} for c in it.checks} for it in r.items]}
    return {"results": {c: {"score": 0.01} for c in r.checks}}
@app.get("/health")
def health():
    return {"status": "ok"}
'''

BASELINE_PII_PATCH = (
    "_analyzer = AnalyzerEngine()",
    "import os as _os\nfrom presidio_analyzer.nlp_engine import SpacyNlpEngine as _E\n"
    "_e = _E(models=[{'lang_code': 'en', 'model_name': _os.environ['BASELINE_SPACY_MODEL']}]); _e.load()\n"
    "_analyzer = AnalyzerEngine(nlp_engine=_e, supported_languages=['en'])",
)


def git_export(rev, path, dest):
    dest.parent.mkdir(parents=True, exist_ok=True)
    data = subprocess.check_output(["git", "show", f"{rev}:./{path}"], cwd=common.ROOT)
    dest.write_bytes(data)


def prepare_code(tmp: Path):
    base = tmp / "baseline"
    for f in ("firewall_addon.py", "canonicalize.py", "classifier_client.py", "policy.py",
              "pii.py", "regex_rules.py"):
        git_export(BASELINE_REV, f"mitmproxy/{f}", base / "mitmproxy" / f)
    git_export(BASELINE_REV, "guardrails/server.py", base / "guardrails" / "server.py")
    p = base / "mitmproxy" / "pii.py"
    src = p.read_text()
    assert BASELINE_PII_PATCH[0] in src
    p.write_text(src.replace(*BASELINE_PII_PATCH))
    (tmp / "apps").mkdir(exist_ok=True)
    (tmp / "apps" / "upstream.py").write_text(UPSTREAM_APP)
    (tmp / "apps" / "stub.py").write_text(STUB_GUARDRAILS_APP)
    return {"baseline": base, "new": common.ROOT}


def wait_http(url, timeout=120):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            if httpx.get(url, timeout=2).status_code < 500:
                return True
        except Exception:
            pass
        time.sleep(0.5)
    raise RuntimeError(f"timeout waiting for {url}")


class Stack:
    def __init__(self, tmp, code, variant, mode, models, spacy):
        self.procs, self.tmp = [], tmp
        self.variant = variant
        env = {**os.environ, "GUARDRAILS_URL": f"http://127.0.0.1:{GR}", "GUARDRAILS_TIMEOUT": "2.0",
               "BASELINE_SPACY_MODEL": f"{spacy}/sm-arch", "PYTHONUNBUFFERED": "1",
               "FIREWALL_LOG_LEVEL": "WARNING"}
        if variant.endswith("nocache"):
            env["SCAN_CACHE"] = "0"
        if mode == "standin":
            # random-weight stand-ins score ~0.5 on everything; never trigger on
            # score so that blocks only come from fail-closed timeouts/outages
            env.update(INJECTION_THRESHOLD="1.01", TOXICITY_THRESHOLD="1.01")
        root = code["baseline" if variant == "baseline" else "new"]
        py = sys.executable
        self.start([py, "-m", "uvicorn", "upstream:app", "--port", str(UP), "--log-level", "warning"],
                   cwd=tmp / "apps", env=env)
        if mode == "stub":
            self.start([py, "-m", "uvicorn", "stub:app", "--port", str(GR), "--log-level", "warning",
                        "--timeout-keep-alive", "75"], cwd=tmp / "apps", env=env)
        else:
            genv = {**env, "INJECTION_MODEL_DIR": f"{models}/injection-standin-torch",
                    "TOXICITY_MODEL_DIR": f"{models}/toxicity-standin-onnx"}
            for size in ("base", "xsmall"):
                if f"onnx_{size}" in variant:
                    genv.update(BACKEND_INJECTION="onnx",
                                INJECTION_MODEL_DIR=f"{models}/injection-standin-onnx-{size}")
            self.start([py, "-m", "uvicorn", "server:app", "--port", str(GR), "--log-level", "warning",
                        "--timeout-keep-alive", "75"], cwd=root / "guardrails", env=genv)
        wait_http(f"http://127.0.0.1:{GR}/health", 300)
        mitmdump = str(Path(py).parent / "mitmdump")
        self.start([mitmdump, "--set", "termlog_verbosity=warn", "-s", str(root / "mitmproxy" / "firewall_addon.py"),
                    "--listen-port", str(MITM), "--mode", f"reverse:http://127.0.0.1:{UP}",
                    "--set", "keep_host_header=false"], cwd=root / "mitmproxy", env=env)
        wait_http(f"http://127.0.0.1:{UP}/docs")
        self.wait_proxy()

    def wait_proxy(self, timeout=300):
        t0 = time.time()
        while time.time() - t0 < timeout:
            try:
                httpx.post(f"http://127.0.0.1:{MITM}/v1/chat/completions", timeout=60,
                           json={"model": "mock", "messages": [{"role": "user", "content": "ping"}]})
                return
            except httpx.TransportError:
                time.sleep(1)
        raise RuntimeError("mitmdump did not come up")

    def start(self, cmd, cwd, env):
        log = open(self.tmp / f"{self.variant}-{Path(cmd[0]).name}-{len(self.procs)}.log", "w")
        self.procs.append(subprocess.Popen(cmd, cwd=cwd, env=env, stdout=log, stderr=log,
                                           start_new_session=True))

    def stop(self):
        for p in self.procs:
            try:
                os.killpg(p.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        for p in self.procs:
            try:
                p.wait(10)
            except subprocess.TimeoutExpired:
                os.killpg(p.pid, signal.SIGKILL)
        time.sleep(1)


async def agent_session(client, base, seed, turns, tool_chars):
    """Run one agent loop; return [(turn, latency_ms, blocked)]"""
    out = []
    for t, msgs in enumerate(corpus.agent_conversation(turns, tool_chars=tool_chars, seed=seed)):
        body = {"model": "mock", "messages": msgs, "tools": TOOLS, "max_tool_turns": turns}
        t0 = time.perf_counter()
        r = await client.post(f"{base}/v1/chat/completions", json=body)
        ms = (time.perf_counter() - t0) * 1000
        r.raise_for_status()
        blocked = r.json().get("id") == "blocked-firewall"
        out.append((t, ms, blocked))
    return out


async def run_sessions(base, sessions, conc, turns, tool_chars, seed0):
    async with httpx.AsyncClient(timeout=120, limits=httpx.Limits(max_connections=64)) as c:
        sem = asyncio.Semaphore(conc)

        async def one(i):
            async with sem:
                return await agent_session(c, base, seed0 + i, turns, tool_chars)
        return await asyncio.gather(*(one(i) for i in range(sessions)))


def summarize(results, direct):
    by_turn = {}
    blocked = 0
    for sess in results:
        for t, ms, b in sess:
            by_turn.setdefault(t, []).append(ms)
            blocked += b
    d = statistics.median(ms for sess in direct for _, ms, _ in sess)
    all_ms = [ms for sess in results for _, ms, _ in sess]
    return {
        "direct_p50_ms": d,
        "added_p50_ms": common.pct(all_ms, 50) - d,
        "added_p95_ms": common.pct(all_ms, 95) - d,
        "added_by_turn_p50_ms": {t: common.pct(v, 50) - d for t, v in sorted(by_turn.items())},
        "requests": len(all_ms), "blocked": blocked,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["stub", "standin"], default="stub")
    ap.add_argument("--variants", default="baseline,new,new_nocache")
    ap.add_argument("--turns", type=int, default=10)
    ap.add_argument("--sessions", type=int, default=8)
    ap.add_argument("--conc", default="1,8")
    ap.add_argument("--tool-chars", type=int, default=2000)
    ap.add_argument("--models", default="/tmp/models-standin")
    ap.add_argument("--spacy", default="/tmp/spacy-standins")
    ap.add_argument("--warm", type=float, default=25.0, help="seconds to let PII workers warm")
    ap.add_argument("--out", default=None, help="results name (default b6_e2e_<mode>)")
    args = ap.parse_args()
    tmp = Path(tempfile.mkdtemp())
    code = prepare_code(tmp)
    out = {"mode": args.mode, "turns": args.turns, "sessions": args.sessions,
           "tool_chars": args.tool_chars, "rows": []}
    for variant in args.variants.split(","):
        stack = Stack(tmp, code, variant, args.mode, args.models, args.spacy)
        try:
            time.sleep(args.warm)
            # warm-up (distinct seeds, discarded)
            asyncio.run(run_sessions(f"http://127.0.0.1:{MITM}", 2, 2, 2, args.tool_chars, 900_000))
            for conc in [int(c) for c in args.conc.split(",")]:
                seed0 = 10_000 * conc
                direct = asyncio.run(run_sessions(f"http://127.0.0.1:{UP}", args.sessions, conc,
                                                  args.turns, args.tool_chars, seed0))
                prox = asyncio.run(run_sessions(f"http://127.0.0.1:{MITM}", args.sessions, conc,
                                                args.turns, args.tool_chars, seed0))
                s = summarize(prox, direct)
                row = {"variant": variant, "concurrency": conc, **s}
                out["rows"].append(row)
                bt = " ".join(f"{v:.0f}" for v in s["added_by_turn_p50_ms"].values())
                print(f"[{args.mode}] {variant:12s} conc={conc} added p50={s['added_p50_ms']:.1f}ms "
                      f"p95={s['added_p95_ms']:.1f}ms blocked={s['blocked']}/{s['requests']} | by turn: {bt}",
                      flush=True)
        finally:
            stack.stop()
    common.write(args.out or f"b6_e2e_{args.mode}", out)
    shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
