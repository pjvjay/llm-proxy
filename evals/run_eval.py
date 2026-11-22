"""Run the labelled corpora through the firewall's checks and write a markdown
report: precision / recall / FPR / F1 + latency per stage, each stage mapped to
the OWASP Top 10 for LLM Applications (2025), plus a Top-10 coverage table.

    GUARDRAILS_URL=http://localhost:8000 python evals/run_eval.py
    cat evals/reports/firewall_eval.md

Model-backed stages go through the guardrails service's System-1 endpoint
(/v1/decide) -- the same call the proxy makes -- so whatever decision backend is
loaded (the stock classifiers, or your own ONNX System-1 model in
models/decision) is what gets evaluated; the report records which one.
Deterministic stages (tool calls, system-prompt leak, output handling,
consumption limits, and PII when Presidio is installed) run locally and need no
service (--offline runs only those).

The datasets are small and self-authored: a pipeline-correctness and
calibration check, not a robustness benchmark.
"""
import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

from metrics import confusion, scores

_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE.parent / "mitmproxy"))
import limits          # noqa: E402
import output_checks   # noqa: E402
import owasp           # noqa: E402
import tool_policy     # noqa: E402

_URL = os.getenv("GUARDRAILS_URL", "http://localhost:8000")


def _load(name: str):
    path = _HERE / "datasets" / name
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def _summarise(preds, labels, latencies, n):
    tp, fp, tn, fn = confusion(preds, labels)
    s = scores(tp, fp, tn, fn)
    ordered = sorted(latencies)
    s.update(n=n, p50_ms=statistics.median(latencies),
             p95_ms=ordered[max(0, int(0.95 * len(ordered)) - 1)])
    return s


# ------------------------------------------------------------------ model-backed
def _run_decide(client, label: str, role: str, dataset, threshold: float):
    preds, labels, latencies = [], [], []
    for row in dataset:
        t0 = time.perf_counter()
        resp = client.post(f"{_URL}/v1/decide",
                           json={"items": [{"text": row["text"], "role": role, "labels": [label]}]})
        latencies.append((time.perf_counter() - t0) * 1000)
        resp.raise_for_status()
        score = resp.json()["decisions"][0]["scores"].get(label)
        if score is None:
            raise SystemExit(f"decision backend does not provide label '{label}' "
                             f"(see GET {_URL}/health)")
        preds.append(score >= threshold)
        labels.append(bool(row["label"]))
    return _summarise(preds, labels, latencies, len(dataset))


# ------------------------------------------------------------------ deterministic
def _timed(dataset, fn):
    preds, labels, latencies = [], [], []
    for row in dataset:
        t0 = time.perf_counter()
        preds.append(bool(fn(row)))
        latencies.append((time.perf_counter() - t0) * 1000)
        labels.append(bool(row["label"]))
    return _summarise(preds, labels, latencies, len(dataset))


def _consumption_cases():
    big = {"messages": [{"role": "user", "content": "x" * (limits.MAX_REQUEST_BYTES + 10)}]}
    return [
        {"raw": len(json.dumps(big)), "payload": big, "label": 1},
        {"raw": 200, "payload": {"max_tokens": limits.MAX_COMPLETION_TOKENS * 10}, "label": 1},
        {"raw": 200, "payload": {"max_completion_tokens": 10 ** 6}, "label": 1},
        {"raw": 200, "payload": {"max_tokens": 256}, "label": 0},
        {"raw": 5000, "payload": {"messages": [{"role": "user", "content": "hello"}]}, "label": 0},
        {"raw": 200, "payload": {}, "label": 0},
    ]


def _pii_stage():
    try:
        import pii                              # needs presidio (present in the mitmproxy image)
    except Exception as exc:                    # noqa: BLE001
        return None, f"skipped: Presidio not importable on this host ({type(exc).__name__})"
    pii.warm_up()
    return _timed(_load("pii.jsonl"), lambda r: pii.scan_and_redact(r["text"])[1]), None


# ------------------------------------------------------------------ report
_HEADER = ("| stage | OWASP LLM Top 10 (2025) | n | precision | recall | FPR | F1 | p50 ms | p95 ms |\n"
           "|---|---|---|---|---|---|---|---|---|")


def _owasp_cell(check: str, role: str | None = None, model_map=None) -> str:
    ids = owasp.ids_for(check, role, None, model_map)
    return ", ".join(ids) if ids else "— (content safety; not a Top-10 category)"


def _row(name, ids, s):
    return (f"| {name} | {ids} | {s['n']} | {s['precision']:.2f} | {s['recall']:.2f} "
            f"| {s['fpr']:.2f} | {s['f1']:.2f} | {s['p50_ms']:.1f} | {s['p95_ms']:.1f} |")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--injection-threshold", type=float, default=0.5)
    ap.add_argument("--toxicity-threshold", type=float, default=0.5)
    ap.add_argument("--out", default=str(_HERE / "reports" / "firewall_eval.md"))
    ap.add_argument("--offline", action="store_true", help="only stages that need no service")
    args = ap.parse_args()

    rows, notes, backend, model_map = [], [], None, None

    if not args.offline:
        import httpx
        with httpx.Client(timeout=30.0) as client:
            health = client.get(f"{_URL}/health").json()
            backend = health.get("decision_backend")
            model_map = (backend or {}).get("owasp") or None
            tool_inj = _load("tool_injection.jsonl")
            rows += [
                ("injection (user prompts)", _owasp_cell("injection", "user", model_map),
                 _run_decide(client, "injection", "user", _load("injection.jsonl") + _load("benign.jsonl"),
                             args.injection_threshold)),
                ("toxicity (user prompts)", _owasp_cell("toxicity", "user", model_map),
                 _run_decide(client, "toxicity", "user", _load("toxicity.jsonl"), args.toxicity_threshold)),
                ("injection in tool results (indirect)", _owasp_cell("injection", "tool", model_map),
                 _run_decide(client, "injection", "tool", tool_inj, args.injection_threshold)),
                ("… long tool results, injection after 3k chars", _owasp_cell("injection", "tool", model_map),
                 _run_decide(client, "injection", "tool",
                             [r for r in tool_inj if r.get("kind", "").startswith("long")],
                             args.injection_threshold)),
            ]
            backend["supply_chain"] = health.get("supply_chain", {}).get("status")

    rows += [
        ("tool-call policy", _owasp_cell("tool_call"),
         _timed(_load("tool_calls.jsonl"), lambda r: not tool_policy.evaluate(r["name"], r["arguments"]).allowed)),
        ("system-prompt leak in responses", _owasp_cell("system_prompt_leak"),
         _timed(_load("system_prompt_leak.jsonl"),
                lambda r: output_checks.system_prompt_leak(r["system"], r["response"])[0])),
        ("active markup / exfil links in responses", _owasp_cell("output_exfiltration"),
         _timed(_load("output_handling.jsonl"), lambda r: output_checks.scan_output(r["text"]))),
        ("consumption limits", _owasp_cell("consumption"),
         _timed(_consumption_cases(), lambda r: limits.check(r["raw"], dict(r["payload"]), "eval"))),
    ]
    pii_s, pii_note = _pii_stage()
    if pii_s:
        rows.append(("PII / secrets (Presidio)", _owasp_cell("pii"), pii_s))
    else:
        notes.append(f"PII stage {pii_note}; it runs inside the mitmproxy container "
                     "(`docker compose exec mitmproxy python -c ...`) or with `pip install presidio-analyzer "
                     "presidio-anonymizer`.")

    table = "\n".join(_row(n, ids, s) for n, ids, s in rows)
    if backend:
        if backend.get("kind") == "onnx_multilabel":
            desc = (f"System-1 decision model `{backend.get('name')}`"
                    f"{' v' + str(backend['version']) if backend.get('version') else ''} "
                    f"({backend.get('dir')}), labels {', '.join(backend.get('labels', []))}")
        else:
            desc = f"composite of the stock classifiers ({', '.join(backend.get('labels', []))})"
        src = f"Model-backed stages evaluated via `{_URL}/v1/decide` — {desc}; model artifacts: **{backend.get('supply_chain')}**."
    else:
        src = "Offline run: deterministic stages only."

    cov = ["| ID | Category | Status | Controls in this deployment |", "|---|---|---|---|"]
    for i, name in owasp.CATALOG.items():
        status, controls = owasp.COVERAGE[i]
        cov.append(f"| {i} | {name} | {status} | {controls} |")

    report = "\n".join([
        "# Firewall eval", "",
        src, "",
        _HEADER, table, "",
        "Datasets are small and self-authored: a pipeline-correctness and calibration check, "
        "not a robustness benchmark. Thresholds: injection "
        f"{args.injection_threshold}, toxicity {args.toxicity_threshold}.",
        *(["", *[f"- {n}" for n in notes]] if notes else []),
        "", "## OWASP Top 10 for LLM Applications (2025) — coverage", "",
        *cov, "",
        "Live counts per category: `docker compose exec mitmproxy python -c \"import urllib.request;"
        "print(urllib.request.urlopen('http://localhost:8080/firewall/owasp').read().decode())\"`",
    ]) + "\n"
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report)
    print(_HEADER)
    print(table)
    for n in notes:
        print(f"note: {n}")
    print(f"\nreport: {out}")


if __name__ == "__main__":
    main()
