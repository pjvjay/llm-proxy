"""End-to-end proxy overhead: old addon vs new addon on a simulated agent loop.

Drives both addons in-process with real mitmproxy flow objects against
bench/stub_server.py, so everything *except* model inference is real
(HTTP hop, JSON, Presidio, caching, tool-call gating). Two runs:

  1. stub inference cost = 0      -> pure proxy overhead per turn
  2. stub inference cost = README numbers (injection 130 ms, toxicity 33 ms per
     512-token window)            -> what the agent actually waits for

Loop shape: 1 user message, then 11 turns of (assistant tool_call -> 3 KB tool
result), with a short user follow-up every 4th turn. Each turn re-sends the
whole history, like a real agent SDK.

Variants:
  old          original control flow (bench/legacy_addon.py): scans user text only,
               truncated; never looks at tool results; new httpx client per call;
               Presidio twice, on the event loop.
  new-nocache  new addon with the verdict cache cleared every turn = "fix the
               bypasses but rescan everything" -- shows why the cache matters.
  new          new addon as shipped.

Run from repo root:  python bench/bench_addon.py"""
import asyncio, json, os, statistics, subprocess, sys, time, warnings

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
os.environ["GUARDRAILS_URL"] = "http://127.0.0.1:8767"
os.environ.setdefault("GUARDRAILS_TIMEOUT", "10")
sys.path.insert(0, os.path.join(ROOT, "mitmproxy")); sys.path.insert(0, HERE)

import httpx
from mitmproxy import http
from mitmproxy.test import tflow

import firewall_addon as new_fw
import legacy_addon as old_fw
import session

URL = "http://api.openai.com/v1/chat/completions"
TURNS = 12
TOOL_RESULT = ("Search result: the quarterly report lists revenue by region. Contact "
               "ops@example.com for the raw CSV. " * 30)            # ~3.2 KB, 1 email
FOLLOW_UP = "Good, now compare that with last quarter."

def build_history(turn):
    msgs = [{"role": "system", "content": "You are a research agent."},
            {"role": "user", "content": "Find the Q3 revenue figures and summarise them."}]
    for t in range(1, turn + 1):
        msgs.append({"role": "assistant", "content": None, "tool_calls": [{"id": f"c{t}", "type": "function",
                     "function": {"name": "search", "arguments": json.dumps({"q": f"q3 revenue {t}"})}}]})
        msgs.append({"role": "tool", "tool_call_id": f"c{t}", "content": f"[{t}] " + TOOL_RESULT})
        if t % 4 == 0:
            msgs.append({"role": "assistant", "content": "Here is what I found so far."})
            msgs.append({"role": "user", "content": FOLLOW_UP})
    return msgs

def upstream_tool_call(t):
    body = {"id": "x", "object": "chat.completion", "created": 1, "model": "m",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": None,
                         "tool_calls": [{"id": f"c{t+1}", "type": "function",
                                         "function": {"name": "search", "arguments": json.dumps({"q": f"q3 {t+1}"})}}]},
                         "finish_reason": "tool_calls"}]}
    return http.Response.make(200, json.dumps(body), {"Content-Type": "application/json"})

async def run_loop(addon, nocache=False):
    per_turn = []
    for t in range(TURNS):
        if nocache:
            new_fw._cache = session.VerdictCache()
        f = tflow.tflow()
        f.request = http.Request.make("POST", URL, json.dumps({"model": "m", "messages": build_history(t)}).encode(),
                                      {"Content-Type": "application/json", "X-Session-Id": f"bench-{id(addon)}-{nocache}"})
        t0 = time.perf_counter()
        await addon.request(f)
        assert f.response is None, "benign loop must not be blocked"
        f.response = upstream_tool_call(t)
        await addon.response(f)
        per_turn.append((time.perf_counter() - t0) * 1000)
    return per_turn

def _wait(url):
    for _ in range(100):
        try:
            httpx.get(url, timeout=0.5); return
        except Exception:
            time.sleep(0.1)
    raise RuntimeError("stub did not start")

async def main():
    new_fw.load(None)
    for label, env in (("stub inference cost = 0 (pure proxy overhead)", {}),
                       ("stub inference = 130 ms injection / 33 ms toxicity per window", {"STUB_MS_INJECTION": "130", "STUB_MS_TOXICITY": "33"})):
        p = subprocess.Popen([sys.executable, "-m", "uvicorn", "stub_server:app", "--port", "8767", "--log-level", "error"],
                             cwd=HERE, env=dict(os.environ, PYTHONPATH=HERE, **env))
        try:
            _wait("http://127.0.0.1:8767/health")
            print(f"\n### {label}\n")
            print("| variant | turn 1 ms | turn 6 ms | turn 12 ms | mean ms/turn | loop total ms | tool-result chars screened |")
            print("|---|---|---|---|---|---|---|")
            for name, addon, nocache in (("old", old_fw, False), ("new-nocache", new_fw, True), ("new", new_fw, False)):
                # warm (first-call costs: connection, cache) then measure a fresh session
                await run_loop(addon, nocache)
                new_fw._cache = session.VerdictCache()
                xs = await run_loop(addon, nocache)
                screened = 0 if name == "old" else len(TOOL_RESULT) * (TURNS - 1)
                print(f"| {name} | {xs[0]:.0f} | {xs[5]:.0f} | {xs[11]:.0f} | {statistics.mean(xs):.0f} | {sum(xs):.0f} | {screened:,} |")
        finally:
            p.terminate(); p.wait()
    new_fw.done()

if __name__ == "__main__":
    asyncio.run(main())
