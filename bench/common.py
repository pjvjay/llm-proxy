"""Shared helpers for the benchmark suite: timing, percentiles, result files.

All benchmarks write machine-readable JSON to bench/results/ so that
docs/perf-report.md can be regenerated and numbers are auditable."""
from __future__ import annotations

import json
import os
import platform
import statistics
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RESULTS = HERE / "results"
RESULTS.mkdir(exist_ok=True)

sys.path.insert(0, str(ROOT / "mitmproxy"))


def pct(values, p):
    """Nearest-rank percentile (p in 0..100)."""
    if not values:
        return float("nan")
    s = sorted(values)
    k = max(0, min(len(s) - 1, int(round(p / 100 * len(s) + 0.5)) - 1))
    return s[k]


def summarize(samples_ms):
    return {
        "n": len(samples_ms),
        "mean_ms": statistics.fmean(samples_ms),
        "p50_ms": pct(samples_ms, 50),
        "p95_ms": pct(samples_ms, 95),
        "p99_ms": pct(samples_ms, 99),
        "min_ms": min(samples_ms),
    }


def timeit_ms(fn, *, repeat=200, warmup=20):
    for _ in range(warmup):
        fn()
    out = []
    for _ in range(repeat):
        t0 = time.perf_counter_ns()
        fn()
        out.append((time.perf_counter_ns() - t0) / 1e6)
    return summarize(out)


def env_info():
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "cpus": os.cpu_count(),
        "processor": platform.processor() or platform.machine(),
    }


def write(name, data):
    path = RESULTS / f"{name}.json"
    path.write_text(json.dumps({"env": env_info(), **data}, indent=2))
    print(f"-> {path.relative_to(ROOT)}")
    return path


def fmt_table(rows, cols):
    head = "| " + " | ".join(cols) + " |\n|" + "---|" * len(cols)
    body = []
    for r in rows:
        cells = []
        for c in cols:
            v = r.get(c, "")
            cells.append(f"{v:.3f}" if isinstance(v, float) else str(v))
        body.append("| " + " | ".join(cells) + " |")
    return head + "\n" + "\n".join(body)
