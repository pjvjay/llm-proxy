"""B1 -- transport between the mitmproxy addon and the guardrails service.

Options:
  per_call_client   new httpx.AsyncClient per classify() (the original code)
  persistent_tcp    one module-level AsyncClient, keep-alive over TCP
  persistent_uds    one AsyncClient over a Unix domain socket
  in_process        direct function call (floor: no transport at all)

The server is a real uvicorn/FastAPI app with the guardrails endpoint shape
and zero-cost scoring, so only transport + framework overhead is measured.
Also checks idle-connection reuse after the server's keep-alive timeout.
"""
from __future__ import annotations

import asyncio
import os
import subprocess
import sys
import tempfile
import time

import httpx

import common

APP = '''
from fastapi import FastAPI
from pydantic import BaseModel
app = FastAPI()
class Req(BaseModel):
    text: str = ""
    checks: list[str] = ["injection", "toxicity"]
@app.post("/v1/classify")
async def classify(r: Req):
    return {"results": {c: {"score": 0.01} for c in r.checks}}
'''
PAYLOAD = {"text": "What is the capital of France? " * 10, "checks": ["injection", "toxicity"]}


async def run(kind, n, conc, url, uds):
    lat = []
    client = None
    if kind == "persistent_tcp":
        client = httpx.AsyncClient(base_url=url, timeout=5)
    elif kind == "persistent_uds":
        client = httpx.AsyncClient(base_url="http://guardrails", timeout=5,
                                   transport=httpx.AsyncHTTPTransport(uds=uds))

    async def one():
        t0 = time.perf_counter()
        if kind == "per_call_client":
            async with httpx.AsyncClient(timeout=5) as c:
                r = await c.post(f"{url}/v1/classify", json=PAYLOAD)
        elif kind == "in_process":
            await asyncio.sleep(0)
            r = None
        else:
            r = await client.post("/v1/classify", json=PAYLOAD)
        if r is not None:
            r.raise_for_status()
            r.json()
        lat.append((time.perf_counter() - t0) * 1000)

    sem = asyncio.Semaphore(conc)

    async def bounded():
        async with sem:
            await one()
    for _ in range(50):
        await one()
    lat.clear()
    t0 = time.perf_counter()
    await asyncio.gather(*(bounded() for _ in range(n)))
    wall = time.perf_counter() - t0
    if client:
        await client.aclose()
    s = common.summarize(lat)
    s["throughput_rps"] = n / wall
    return s


async def idle_reuse(url, keepalive_s):
    """Does a pooled connection survive the server closing it while idle?"""
    client = httpx.AsyncClient(base_url=url, timeout=5)
    await client.post("/v1/classify", json=PAYLOAD)
    await asyncio.sleep(keepalive_s + 1.5)
    t0 = time.perf_counter()
    try:
        await client.post("/v1/classify", json=PAYLOAD)
        ok, err = True, None
    except Exception as e:             # noqa: BLE001
        ok, err = False, repr(e)
    ms = (time.perf_counter() - t0) * 1000
    await client.aclose()
    return {"ok": ok, "error": err, "ms": ms}


def main():
    tmp = tempfile.mkdtemp()
    open(f"{tmp}/app.py", "w").write(APP)
    uds = f"{tmp}/g.sock"
    port = 18765
    env = {**os.environ, "PYTHONPATH": tmp}
    common_args = [sys.executable, "-m", "uvicorn", "app:app", "--log-level", "warning",
                   "--timeout-keep-alive", "5"]
    procs = [subprocess.Popen(common_args + ["--port", str(port)], env=env, cwd=tmp),
             subprocess.Popen(common_args + ["--uds", uds], env=env, cwd=tmp)]
    time.sleep(3)
    url = f"http://127.0.0.1:{port}"
    out = {"rows": [], "rounds": 3}
    try:
        # Interleave options across rounds so VM noise hits all options alike;
        # report the median of per-round percentiles.
        import statistics
        for conc in (1, 16):
            per = {}
            for rnd in range(3):
                for kind in ("per_call_client", "persistent_tcp", "persistent_uds", "in_process"):
                    if kind == "per_call_client" and rnd > 0:
                        continue                         # slow and unambiguous: one round
                    n = 300 if kind == "per_call_client" else 1500
                    s = asyncio.run(run(kind, n, conc, url, uds))
                    per.setdefault(kind, []).append(s)
                    print(f"round={rnd} conc={conc:2d} {kind:16s} p50={s['p50_ms']:.3f} "
                          f"p95={s['p95_ms']:.3f} rps={s['throughput_rps']:.0f}", flush=True)
            for kind, ss in per.items():
                row = {"option": kind, "concurrency": conc, "rounds": len(ss)}
                for k in ("p50_ms", "p95_ms", "p99_ms", "throughput_rps"):
                    row[k] = statistics.median(x[k] for x in ss)
                out["rows"].append(row)
        out["idle_reuse_after_keepalive_timeout"] = asyncio.run(idle_reuse(url, 5))
        print("idle reuse:", out["idle_reuse_after_keepalive_timeout"])
    finally:
        for p in procs:
            p.terminate()
    common.write("b1_transport", out)


if __name__ == "__main__":
    main()
