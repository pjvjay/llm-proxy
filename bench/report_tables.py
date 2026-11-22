"""Render the markdown tables used in docs/perf-report.md from bench/results.

    python bench/report_tables.py > /tmp/tables.md
"""
import json
from pathlib import Path

R = Path(__file__).parent / "results"


def load(name):
    p = R / f"{name}.json"
    return json.loads(p.read_text()) if p.exists() else None


def table(rows, cols, fmt=None):
    fmt = fmt or {}
    out = ["| " + " | ".join(h for h, _ in cols) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        cells = []
        for _, k in cols:
            v = k(r) if callable(k) else r.get(k, "")
            cells.append(fmt.get(k, "{:.1f}").format(v) if isinstance(v, float) else str(v))
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def main():
    b1 = load("b1_transport")
    if b1:
        print("### B1 transport\n")
        print(table(b1["rows"], [("option", "option"), ("concurrency", "concurrency"),
                                 ("p50 ms", "p50_ms"), ("p95 ms", "p95_ms"), ("rps", "throughput_rps")],
                    {"p50_ms": "{:.2f}", "p95_ms": "{:.2f}", "throughput_rps": "{:.0f}"}))
        print("\nidle reuse:", b1.get("idle_reuse_after_keepalive_timeout"), "\n")
    b2 = load("b2_pii")
    if b2:
        print("### B2 PII cost (p50 ms)\n")
        rows = {}
        for r in b2["q1_q2"]:
            rows.setdefault((r["pii_density"], r["size_chars"]), {})[r["variant"]] = r["p50_ms"]
        variants = list(dict.fromkeys(r["variant"] for r in b2["q1_q2"]))
        print("| PII density | chars | " + " | ".join(variants) + " |")
        print("|" + "---|" * (2 + len(variants)))
        for (d, n), v in rows.items():
            print(f"| {d} | {n} | " + " | ".join(f"{v.get(x, float('nan')):.2f}" for x in variants) + " |")
        print("\nparity:", b2["q3"], "\n")
        print(table(b2["q4"], [("variant", "variant"), ("mix", "mix"), ("conc", "concurrency"),
                               ("strategy", "strategy"), ("p50", "p50_ms"), ("p95", "p95_ms"),
                               ("max loop stall", "max_loop_stall_ms"), ("rps", "throughput_rps")],
                    {"throughput_rps": "{:.0f}"}))
        print()
    b3 = load("b3_windowing")
    if b3:
        print("### B3 windowing\n")
        print(table(b3["q1"], [("check", "check"), ("chars", "chars"), ("tokens", "tokens"),
                               ("windows", "windows"), ("strategy", "strategy"),
                               ("p50 ms", "p50_ms"), ("p95 ms", "p95_ms")]))
        print()
        print(table(b3["q2"], [("check", "check"), ("max_windows", "max_windows"), ("p50 ms", "p50_ms")]))
        print()
        print(table(b3["q3"], [("check", "check"), ("intra-op threads", "intra_op_threads"),
                               ("concurrency", "concurrency"), ("p50 ms", "p50_ms"),
                               ("p95 ms", "p95_ms"), ("rps", "throughput_rps")],
                    {"throughput_rps": "{:.2f}"}))
        print()
    for name in ("b3c_model_concurrency", "b3c_model_concurrency_rerun", "b3c_fairness"):
        b = load(name)
        if b:
            print(f"### {name}\n")
            print(table(b["rows"], [("run", lambda r: r.get("run", 0)), ("config", "config"),
                         ("workload", "workload"), ("p50 ms", "p50_ms"),
                         ("p95 ms", "p95_ms"), ("rps", "throughput_rps")],
                        {"throughput_rps": "{:.2f}"}))
            print()
    b3b = load("b3b_injection_models")
    if b3b:
        print("### B3b injection model options\n")
        print("fidelity:", b3b["fidelity_max_abs_logit_diff"], "\n")
        print(table(b3b["rows"], [("variant", "variant"), ("params M", "params_M"), ("tokens", "tokens"),
                                  ("p50 ms", "p50_ms"), ("p95 ms", "p95_ms")]))
        print()
    b4 = load("b4_cache")
    if b4:
        print("### B4 cache machinery\n")
        print(table(b4["q1"], [("bytes", "bytes"), ("hash", "hash"), ("p50 us", "p50_us")], {"p50_us": "{:.2f}"}))
        print()
        print(table(b4["q2"], [("impl", "impl"), ("us/op", "per_get_or_put_us")], {"per_get_or_put_us": "{:.3f}"}))
        print()
        print(table(b4["q3"], [("turn", "turn"), ("messages", "messages"), ("p50 ms", "p50_ms")], {"p50_ms": "{:.3f}"}))
        print()
    b5 = load("b5_tool_policy")
    if b5:
        print("### B5 tool policy\n")
        print(table(b5["q1"], [("input", "input"), ("chars", "chars"), ("matcher", "matcher"),
                               ("p50 ms", "p50_ms")], {"p50_ms": "{:.3f}"}))
        print()
        print(table(b5["q3"], [("body bytes", "body_bytes"), ("lib", "lib"), ("loads ms", "loads_p50_ms"),
                               ("dumps ms", "dumps_p50_ms")], {"loads_p50_ms": "{:.3f}", "dumps_p50_ms": "{:.3f}"}))
        print()
    for mode in ("stub", "standin"):
        b6 = load(f"b6_e2e_{mode}")
        if b6:
            print(f"### B6 end-to-end ({mode}; turns={b6['turns']}, sessions={b6['sessions']}, "
                  f"tool_chars={b6['tool_chars']})\n")
            print(table(b6["rows"], [("variant", "variant"), ("concurrency", "concurrency"),
                                     ("added p50 ms", "added_p50_ms"), ("added p95 ms", "added_p95_ms"),
                                     ("blocked", lambda r: f"{r['blocked']}/{r['requests']}")]))
            for r in b6["rows"]:
                bt = " ".join(f"{v:.0f}" for v in r["added_by_turn_p50_ms"].values())
                print(f"- {r['variant']} conc={r['concurrency']}: blocked {r['blocked']}/{r['requests']}; "
                      f"added p50 by turn: {bt}")
            print()


if __name__ == "__main__":
    main()
