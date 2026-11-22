"""B3 -- long inputs: truncation vs sliding windows, batching, caps, threads.

Q1  cost vs input size for:  truncate(512) [original]  |  windows, batch=1  |
    windows, batch=4  |  windows, batch=8   (all windows, no cap)
Q2  effect of MAX_WINDOWS caps (4 / 8 / 16) on the largest input
Q3  INTRA_OP_THREADS (1 / 2 / 4 / default) x concurrent requests (1 / 4) with
    4 server worker threads -- oversubscription check

Uses the stand-in models (bench/standins/make_models.py): same architecture
and size as the real ones, random weights -> latency is representative.

    python bench/b3_windowing.py --models /tmp/models-standin
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import importlib
import os
import statistics
import sys
import time

import common
import corpus

sys.path.insert(0, str(common.ROOT / "guardrails"))

SIZES = [300, 2_000, 10_000, 50_000]


def load(models, threads=0):
    os.environ.update(INJECTION_MODEL_DIR=f"{models}/injection-standin-torch",
                      TOXICITY_MODEL_DIR=f"{models}/toxicity-standin-onnx",
                      INTRA_OP_THREADS=str(threads))
    import server
    server = importlib.reload(server)
    server._load()
    return server


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="/tmp/models-standin")
    ap.add_argument("--only", default="q1,q2,q3")
    args = ap.parse_args()
    out = {"q1": [], "q2": [], "q3": []}
    srv = load(args.models)
    srv._backends["injection"].batch, srv._backends["toxicity"].batch = 1, 4   # B3 defaults
    texts = {n: corpus.tool_result(n, seed=n, inject_at=0.9) for n in SIZES}

    if "q1" in args.only:
        for check in ("toxicity", "injection"):
            b = srv._backends[check]
            for n, t in texts.items():
                ntok = len(b.tok(t)["input_ids"])
                nwin = b.encode([t], True, 0)[3][0]["windows"]
                for strat, kw in (("truncate_512 (original)", dict(windowed=False, batch=1)),
                                  ("windows_batch1", dict(windowed=True, batch=1, max_windows=0)),
                                  ("windows_batch4", dict(windowed=True, batch=4, max_windows=0)),
                                  ("windows_batch8", dict(windowed=True, batch=8, max_windows=0))):
                    if strat.startswith("windows") and nwin == 1 and strat != "windows_batch1":
                        continue          # identical to batch1 for single-window inputs
                    if check == "injection" and n >= 50_000 and strat in ("windows_batch4", "windows_batch8"):
                        continue          # batch>1 already slower at 10 KB; ~1 min per sample here
                    rep = 20 if (check == "toxicity" or n <= 2000) else (6 if n <= 10_000 else 3)
                    s = common.timeit_ms(lambda: b.score_texts([t], 1, **kw), repeat=rep, warmup=1)
                    row = {"check": check, "chars": n, "tokens": ntok, "windows": nwin,
                           "strategy": strat, **s}
                    out["q1"].append(row)
                    print(f"{check:9s} {n:6d}c {ntok:6d}t {nwin:3d}w {strat:24s} "
                          f"p50={s['p50_ms']:8.1f} p95={s['p95_ms']:8.1f}", flush=True)

    if "q2" in args.only:
        t = texts[50_000]
        for check in ("toxicity", "injection"):
            b = srv._backends[check]
            for cap in (4, 8, 16):
                rep = 10 if check == "toxicity" else 3
                s = common.timeit_ms(lambda: b.score_texts([t], 1, windowed=True, max_windows=cap),
                                     repeat=rep, warmup=1)
                out["q2"].append({"check": check, "chars": 50_000, "max_windows": cap, **s})
                print(f"cap {check:9s} max_windows={cap:2d} p50={s['p50_ms']:8.1f}", flush=True)

    if "q3" in args.only:
        t = texts[2_000]                          # ~1 window: a typical tool result
        for threads in (1, 2, 4, 0):
            srv = load(args.models, threads)
            srv._backends["injection"].batch, srv._backends["toxicity"].batch = 1, 4
            pool = cf.ThreadPoolExecutor(4)       # = GUARDRAILS_WORKERS
            for conc in (1, 4):
                for check in ("toxicity", "injection"):
                    b = srv._backends[check]
                    n_req = 40 if check == "toxicity" else 12
                    lat = []

                    def req():
                        t0 = time.perf_counter()
                        b.score_texts([t], 1)
                        lat.append((time.perf_counter() - t0) * 1000)
                    for _ in range(2):
                        req()
                    lat.clear()
                    t0 = time.perf_counter()
                    futs = []
                    for i in range(n_req):
                        futs.append(pool.submit(req))
                        if len(futs) >= conc:       # keep `conc` requests in flight
                            futs.pop(0).result()
                    for f in futs:
                        f.result()
                    wall = time.perf_counter() - t0
                    s = common.summarize(lat)
                    s["throughput_rps"] = n_req / wall
                    out["q3"].append({"check": check, "intra_op_threads": threads or "default",
                                      "concurrency": conc, **s})
                    print(f"threads={threads or 'dflt'} conc={conc} {check:9s} p50={s['p50_ms']:7.1f} "
                          f"p95={s['p95_ms']:7.1f} rps={s['throughput_rps']:.1f}", flush=True)
            pool.shutdown()
    common.write("b3_windowing", out)


if __name__ == "__main__":
    main()
