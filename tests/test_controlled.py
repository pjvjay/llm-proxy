"""Controlled benchmark assertions: each claimed improvement is checked as an
exact, deterministic property of the real proxy code under the declared cost
model (bench/controlled). Runs in well under a minute; no models, no sockets,
no dependence on host load.

    python tests/test_controlled.py"""
import io
import os
import sys
from contextlib import redirect_stdout

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "bench", "controlled"))
import run as R          # noqa: E402
import vclock            # noqa: E402
import workloads as W    # noqa: E402
import system2           # noqa: E402


def main():
    with redirect_stdout(io.StringIO()):
        res = R.main(quick=True)
    w1, w2, w3 = res["w1"], res["w2"], res["w3"]
    L, T0, S1 = "legacy addon", "tier-0 (two classifiers)", "System 1 (multi-label)"
    IN, RA, S2 = "S1 + S2 inline", "S1 + S2 raced", "S2 only (judge everything)"

    # --- System 1: one pass for all labels (structural, independent of model speed)
    assert w1[S1]["passes"] < w1[T0]["passes"], (w1[S1]["passes"], w1[T0]["passes"])
    # --- and, under the declared cost model, lower added latency
    assert w1[S1]["mean"] < w1[T0]["mean"]
    # --- System 2 fixes System 1's mistakes in the uncertainty band...
    assert w1[IN]["fn"] + w1[IN]["fp"] < w1[S1]["fn"] + w1[S1]["fp"]
    # ...judging only the uncertain ones (not everything)
    assert 0 < w1[IN]["s2"] < w1[S2]["s2"] / 5
    # ...and racing hides the judge behind the upstream call: no request pays for it
    # (worst case identical to System 1 alone; means differ only by which requests are blocked)
    assert w1[RA]["max"] == w1[S1]["max"] < w1[IN]["max"]
    assert w1[RA]["mean"] < w1[IN]["mean"] < w1[S2]["mean"]
    assert w1[RA]["fn"] == w1[IN]["fn"] and w1[RA]["fp"] == w1[IN]["fp"]
    assert w1[RA]["wasted"] > 0                      # the price of racing, made visible

    # --- agent loop: old addon lets poisoned tool results reach the model every turn
    assert w2[L]["unneutralised"] > 0 and w2[L]["harmful"] == 2
    for name in (T0, S1, IN, RA, S2):
        assert w2[name]["unneutralised"] == 0, name
    assert w2[S1]["harmful"] == 1                     # destructive shell caught by tool_policy
    assert w2[IN]["harmful"] == 0 and w2[RA]["harmful"] == 0   # goal-inconsistent call needs S2 + session risk
    assert w2[RA]["discarded"] == 1                  # raced turn: model saw it, its output was replaced
    # each distinct escalation judged once (cache); read-only tool calls not judged
    assert w2[IN]["s2"] == 2, w2[IN]["s2"]
    assert w2[S1]["mean"] < w2[T0]["mean"]

    # --- proxy hygiene, as exact operation counts
    assert w2[L]["ops"]["ca_bundle_loads"] > 0 and w2[L]["ops"]["presidio_on_loop_thread"] > 0
    for name in (T0, S1, IN, RA):
        o = w2[name]["ops"]
        assert o.get("ca_bundle_loads", 0) == 0 and o.get("presidio_on_loop_thread", 0) == 0, name

    # --- burst: shedding bounds the tail instead of timing out; racing hides it
    no_shed, shed, raced = (w3["S1 + S2 inline, no shedding"], w3["S1 + S2 inline, shed above queue 8"],
                            w3["S1 + S2 raced, shed above queue 8"])
    assert no_shed["timeout"] > 0 and no_shed["shed"] == 0
    assert shed["shed"] > 0 and shed["timeout"] == 0 and shed["max"] < no_shed["max"]
    assert raced["p95"] < shed["p95"]

    # --- determinism: identical results on repeated runs (no wall-clock dependence)
    for cfg_name, wl in ((RA, "w2"), ("S1 + S2 inline", "w3")):
        outs = []
        for _ in range(2):
            addon, guard, judge = R.setup(R.CONFIGS[cfg_name])
            if wl == "w2":
                outs.append(R.summarise_w2(vclock.run(R.w2(addon)), guard, judge))
            else:
                r = vclock.run(R.w3(addon, W.burst(16)))
                outs.append([(x["added_ms"], x["blocked"]) for x in r] + [dict(system2.STATS)])
        assert outs[0] == outs[1], f"non-deterministic: {cfg_name} {wl}"
    print("test_controlled: OK")


if __name__ == "__main__":
    main()
