"""B5 -- tool-call gating cost + JSON handling.

Q1  matcher backend: loop (one regex per pattern) | combined (one per category)
    | re2 (combined, linear-time RE2) | re2_loop, over realistic argument sizes
    and two adversarial inputs designed to provoke regex backtracking.
Q2  end-to-end check_tool_call() cost (parse + flatten + canonicalize + match)
Q3  json vs orjson for request / response bodies of agent-history sizes.

    python bench/b5_tool_policy.py
"""
from __future__ import annotations

import json
import time

import common
import corpus
import tool_policy as tp

ARGS = {
    "small(bash cmd ~60B)": json.dumps({"cmd": "grep -rn 'TODO' src/ | head -20 && ls -la build/"}),
    "medium(sql ~2KB)": json.dumps({"q": "SELECT " + ", ".join(f"col{i}" for i in range(300)) + " FROM t WHERE id=1"}),
    "large(write_file 50KB)": json.dumps({"path": "src/app.py", "content": corpus.text(50_000, 3)}),
}
ADVERSARIAL = {
    "adv_rm_flags(60KB)": "rm " + "-a " * 20_000,
    "adv_curl_repeat(100KB)": "curl " * 20_000,
}


def main():
    out = {"q1": [], "q2": [], "q3": []}
    matchers = {k: tp.build_matcher(k) for k in ("loop", "combined", "re2", "re2_loop")}

    # Q1 -------------------------------------------------------------------
    texts = {k: tp.argument_text(v) for k, v in ARGS.items()} | ADVERSARIAL
    for name, text in texts.items():
        for kind, m in matchers.items():
            rep = 200 if len(text) < 5000 else 20
            if name.startswith("adv") and not kind.startswith("re2"):
                t0 = time.perf_counter()
                m.categories(text)                      # one run: may be very slow
                ms = (time.perf_counter() - t0) * 1000
                s = {"n": 1, "p50_ms": ms, "p95_ms": ms, "mean_ms": ms, "p99_ms": ms, "min_ms": ms}
            else:
                s = common.timeit_ms(lambda: m.categories(text), repeat=rep, warmup=3)
            out["q1"].append({"input": name, "chars": len(text), "matcher": kind, **s})
            print(f"{name:24s} {kind:9s} p50={s['p50_ms']:9.3f}ms", flush=True)

    # Q2 -------------------------------------------------------------------
    for name, args in ARGS.items():
        rep = 200 if len(args) < 5000 else 20
        s = common.timeit_ms(lambda: tp.check_tool_call("bash", args, {"bash"}), repeat=rep, warmup=3)
        out["q2"].append({"input": name, **s})
        print(f"check_tool_call {name:24s} p50={s['p50_ms']:.3f}ms p95={s['p95_ms']:.3f}ms")

    # Q3 -------------------------------------------------------------------
    import orjson
    for turns in (1, 10, 40):
        msgs = list(corpus.agent_conversation(turns, tool_chars=3000))[-1]
        body = json.dumps({"model": "gpt-4o", "messages": msgs})
        obj = json.loads(body)
        rep = 100 if len(body) < 100_000 else 30
        for lib, loads, dumps in (("json", json.loads, json.dumps),
                                  ("orjson", orjson.loads, orjson.dumps)):
            s_l = common.timeit_ms(lambda: loads(body), repeat=rep, warmup=3)
            s_d = common.timeit_ms(lambda: dumps(obj), repeat=rep, warmup=3)
            out["q3"].append({"body_bytes": len(body), "lib": lib, "loads_p50_ms": s_l["p50_ms"],
                              "dumps_p50_ms": s_d["p50_ms"]})
            print(f"{len(body):8d}B {lib:6s} loads={s_l['p50_ms']:.3f}ms dumps={s_d['p50_ms']:.3f}ms")
    common.write("b5_tool_policy", out)


if __name__ == "__main__":
    main()
