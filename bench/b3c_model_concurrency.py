"""B3c -- concurrent requests vs intra-op threads (the B3 Q3 collapse).

Configs (each in a fresh process: torch thread count is process-global):
  original      INTRA_OP_THREADS=default(4)  MODEL_CONCURRENCY=0 (unbounded)
  threads1      INTRA_OP_THREADS=1           MODEL_CONCURRENCY=0
  gate1_t4      INTRA_OP_THREADS=default(4)  MODEL_CONCURRENCY=1   (new default)
  gate2_t2      INTRA_OP_THREADS=2           MODEL_CONCURRENCY=2
Workloads on the injection model (DeBERTa stand-in), 4 server worker threads:
  uniform  : N concurrent 2 KB requests (1 window each), N in {1, 4, 8}
  mixed    : one 10 KB request (5 windows) + 6 short prompts arriving just after
             -> latency of the short ones (head-of-line blocking check)
"""
from __future__ import annotations

import concurrent.futures as cf
import json
import os
import subprocess
import sys
import time

import common
import corpus

CONFIGS = {
    "original": {"MODEL_CONCURRENCY": "0"},
    "threads1": {"MODEL_CONCURRENCY": "0", "INTRA_OP_THREADS": "1"},
    "gate1_t4": {"MODEL_CONCURRENCY": "1"},
    "gate2_t2": {"MODEL_CONCURRENCY": "2", "INTRA_OP_THREADS": "2"},
    "gate1_t2": {"MODEL_CONCURRENCY": "1", "INTRA_OP_THREADS": "2"},
    "gate2_t2_fair": {"MODEL_CONCURRENCY": "2", "INTRA_OP_THREADS": "2", "GATE_FAIR": "1"},
    "gate2_t2_unfair": {"MODEL_CONCURRENCY": "2", "INTRA_OP_THREADS": "2", "GATE_FAIR": "0"},
}


def child(models, name):
    sys.path.insert(0, str(common.ROOT / "guardrails"))
    os.environ.update(INJECTION_MODEL_DIR=f"{models}/injection-standin-torch",
                      TOXICITY_MODEL_DIR=f"{models}/toxicity-standin-onnx")
    import server
    server._load()
    b = server._backends["injection"]
    short, medium, long_ = corpus.text(300, 1), corpus.tool_result(2000, 2), corpus.tool_result(10_000, 3)
    pool = cf.ThreadPoolExecutor(8)
    rows = []

    def timed(text):
        t0 = time.perf_counter()
        b.score_texts([text], 1)
        return (time.perf_counter() - t0) * 1000
    timed(medium)
    for n in (1, 4, 8):
        if name == "original" and n == 8:
            continue                      # collapses (~minutes); n=4 already shows it
        lat, t0 = [], time.perf_counter()
        for _ in range(2 if n > 1 else 6):
            lat += list(pool.map(timed, [medium] * n))
        wall = time.perf_counter() - t0
        s = common.summarize(lat)
        rows.append({"config": name, "workload": f"uniform_{n}", "throughput_rps": len(lat) / wall, **s})
        print(f"{name:9s} uniform n={n} p50={s['p50_ms']:8.1f} p95={s['p95_ms']:8.1f} rps={len(lat)/wall:.2f}", flush=True)
    if name != "original":
        big = pool.submit(timed, long_)
        time.sleep(0.05)
        shorts = [pool.submit(timed, short) for _ in range(6)]
        sl = [f.result() for f in shorts]
        s = common.summarize(sl)
        rows.append({"config": name, "workload": "mixed_short_behind_long", "long_ms": big.result(), **s})
        print(f"{name:9s} mixed: short p50={s['p50_ms']:8.1f} p95={s['p95_ms']:8.1f} long={big.result():.0f}", flush=True)
    print("ROWS" + json.dumps(rows), flush=True)


def main():
    models = sys.argv[1] if len(sys.argv) > 1 else "/tmp/models-standin"
    if len(sys.argv) > 2:
        return child(models, sys.argv[2])
    rows = []
    order = os.getenv("B3C_ORDER", ",".join(CONFIGS)).split(",")
    for rnd, name in enumerate(order):
        env = CONFIGS[name]
        print(f"--- run {rnd}: {name}", flush=True)
        p = subprocess.run([sys.executable, __file__, models, name], env={**os.environ, **env},
                           capture_output=True, text=True, cwd=common.HERE)
        for line in p.stdout.splitlines():
            if line.startswith("ROWS"):
                rows += [{**r, "run": rnd} for r in json.loads(line[4:])]
            else:
                print(line, flush=True)
        if p.returncode:
            print(p.stderr[-2000:])
    common.write(os.getenv("B3C_OUT", "b3c_model_concurrency"), {"rows": rows})


if __name__ == "__main__":
    main()
