"""Transport overhead addon -> guardrails: how much does the HTTP hop cost, and
how much of it is avoidable?

Options compared (all against bench/stub_server.py, which does no inference):
  A. new httpx.AsyncClient per call          (current classifier_client.py)
  B. one module-level client, HTTP/1.1 keep-alive pool
  C. one module-level client, HTTP/2 (needs an h2-capable server: hypercorn)
  D. one module-level client over a Unix domain socket (same host only)

Run:  python bench/bench_transport.py  (starts/stops the stub servers itself)
"""
import asyncio, os, statistics, subprocess, sys, time
import httpx

HERE = os.path.dirname(os.path.abspath(__file__))
TEXT = "Summarise the quarterly report in three bullet points. " * 6   # ~350 chars
N_SEQ = 400
CONC = (1, 8, 32)
N_CONC = 800

def _pct(xs, p):
    xs = sorted(xs); return xs[min(len(xs) - 1, int(p * len(xs)))]

ERRORS = {"n": 0}

async def _one(client_factory, url):
    t0 = time.perf_counter()
    try:
        async with client_factory() as c:
            r = await c.post(url, json={"text": TEXT, "checks": ["injection", "toxicity"]})
            r.raise_for_status()
    except Exception:
        ERRORS["n"] += 1
        return None
    return (time.perf_counter() - t0) * 1000

class _Persist:
    """Wrap a long-lived client so the same code path can 'async with' it."""
    def __init__(self, client): self.c = client
    async def __aenter__(self): return self.c
    async def __aexit__(self, *a): pass

async def run_variant(name, client_factory, url):
    # warm-up
    for _ in range(20):
        await _one(client_factory, url)
    ERRORS["n"] = 0
    seq = [x for x in [await _one(client_factory, url) for _ in range(N_SEQ)] if x is not None]
    rows = [(name, "seq", 1, statistics.median(seq), _pct(seq, 0.95), _pct(seq, 0.99), ERRORS["n"])]
    for conc in CONC[1:]:
        lat = []
        ERRORS["n"] = 0
        async def worker(k):
            for _ in range(k):
                x = await _one(client_factory, url)
                if x is not None: lat.append(x)
        t0 = time.perf_counter()
        await asyncio.gather(*[worker(N_CONC // conc) for _ in range(conc)])
        wall = time.perf_counter() - t0
        rows.append((name, f"conc={conc}", round(N_CONC / wall), statistics.median(lat),
                     _pct(lat, 0.95), _pct(lat, 0.99), ERRORS["n"]))
    return rows

def _wait(url, verify_h2=False):
    for _ in range(100):
        try:
            httpx.get(url, timeout=0.5); return
        except Exception:
            time.sleep(0.1)
    raise RuntimeError(f"server at {url} did not start")

async def main():
    env = dict(os.environ, PYTHONPATH=HERE)
    uds = "/tmp/bench_guard.sock"
    if os.path.exists(uds): os.remove(uds)
    procs = [
        subprocess.Popen([sys.executable, "-m", "uvicorn", "stub_server:app", "--port", "8765",
                          "--log-level", "error"], cwd=HERE, env=env),
        subprocess.Popen([sys.executable, "-m", "uvicorn", "stub_server:app", "--uds", uds,
                          "--log-level", "error"], cwd=HERE, env=env),
    ]
    h2py = os.environ.get("H2_PYTHON")  # python with hypercorn + h2>=4.3, e.g. a venv
    if h2py:
        procs.append(subprocess.Popen([h2py, "-m", "hypercorn", "stub_server:app", "--bind",
                                       "127.0.0.1:8766", "--log-level", "error"], cwd=HERE, env=env))
    try:
        _wait("http://127.0.0.1:8765/health")
        if h2py: _wait("http://127.0.0.1:8766/health")
        time.sleep(0.5)
        url = "http://127.0.0.1:8765/v1/classify"
        rows = []
        rows += await run_variant("A new client/call", lambda: httpx.AsyncClient(timeout=2.0), url)
        p = httpx.AsyncClient(timeout=2.0)
        rows += await run_variant("B keep-alive h1", lambda: _Persist(p), url)
        await p.aclose()
        if h2py:
            p2 = httpx.AsyncClient(timeout=2.0, http2=True)
            rows += await run_variant("C keep-alive h2", lambda: _Persist(p2),
                                      "http://127.0.0.1:8766/v1/classify")
            await p2.aclose()
        pu = httpx.AsyncClient(timeout=2.0, transport=httpx.AsyncHTTPTransport(uds=uds))
        rows += await run_variant("D unix socket", lambda: _Persist(pu), "http://guard/v1/classify")
        await pu.aclose()
        print("| variant | mode | req/s | p50 ms | p95 ms | p99 ms | errors |")
        print("|---|---|---|---|---|---|---|")
        for name, mode, rps, p50, p95, p99, err in rows:
            print(f"| {name} | {mode} | {rps} | {p50:.2f} | {p95:.2f} | {p99:.2f} | {err} |")
    finally:
        for pr in procs:
            pr.terminate()
        for pr in procs:
            pr.wait()

if __name__ == "__main__":
    asyncio.run(main())
