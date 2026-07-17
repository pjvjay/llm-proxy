"""Run the labeled corpus through the guardrails service and emit a markdown
report: precision / recall / FPR / F1 + p50/p95 latency per stage. This table is
the point -- it's what turns 'I added a classifier' into 'here's how well it
works and how fast'.

    GUARDRAILS_URL=http://localhost:8000 python run_eval.py
"""
import argparse
import json
import os
import statistics
import time
from pathlib import Path

import httpx

from metrics import confusion, scores

_URL = os.getenv("GUARDRAILS_URL", "http://localhost:8000")
_HERE = Path(__file__).parent

def _load(name: str):
    path = _HERE / "datasets" / name
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]

def _run(check: str, dataset, threshold: float):
    preds, labels, latencies = [], [], []
    with httpx.Client(timeout=10.0) as client:
        for row in dataset:
            t0 = time.perf_counter()
            resp = client.post(f"{_URL}/v1/classify",
                               json={"text": row["text"], "checks": [check]})
            latencies.append((time.perf_counter() - t0) * 1000)
            score = resp.json()["results"].get(check, {}).get("score", 0.0)
            preds.append(score >= threshold)
            labels.append(bool(row["label"]))
    tp, fp, tn, fn = confusion(preds, labels)
    s = scores(tp, fp, tn, fn)
    ordered = sorted(latencies)
    s.update(n=len(dataset), p50_ms=statistics.median(latencies),
             p95_ms=ordered[max(0, int(0.95 * len(ordered)) - 1)])
    return s

def _row(name, s):
    return (f"| {name} | {s['n']} | {s['precision']:.2f} | {s['recall']:.2f} "
            f"| {s['fpr']:.2f} | {s['f1']:.2f} | {s['p50_ms']:.0f} | {s['p95_ms']:.0f} |")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--injection-threshold", type=float, default=0.5)
    ap.add_argument("--toxicity-threshold", type=float, default=0.5)
    ap.add_argument("--out", default=str(_HERE / "reports" / "firewall_eval.md"))
    args = ap.parse_args()

    injection = _load("injection.jsonl") + _load("benign.jsonl")
    toxicity = _load("toxicity.jsonl")
    rows = [
        ("injection", _run("injection", injection, args.injection_threshold)),
        ("toxicity",  _run("toxicity",  toxicity,  args.toxicity_threshold)),
    ]

    header = ("| stage | n | precision | recall | FPR | F1 | p50 ms | p95 ms |\n"
              "|---|---|---|---|---|---|---|---|")
    table = "\n".join(_row(n, s) for n, s in rows)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(f"# Firewall eval\n\n{header}\n{table}\n")
    print(header)
    print(table)
