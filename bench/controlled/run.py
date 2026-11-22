"""Controlled benchmarks: deterministic, host-independent, seconds to run.

Runs the REAL proxy code (the original addon and the current one) through real
mitmproxy flow objects. Only the network endpoints are mocked (guardrails
service, System-2 judge, upstream LLM), each charging a declared cost
(costs.py) on a virtual clock (vclock.py). Two kinds of output:

  * composed latency -- how the proxy's control flow (sequential, concurrent,
    cached, skipped, escalated, raced, shed) turns the declared costs into
    per-call added latency. Identical on every run and every machine.
  * operation counts -- work the proxy itself does (CA-bundle loads, HTTP
    clients built, Presidio calls on the event-loop thread, round-trips,
    forward passes, judge calls). Deterministic; no timing involved.

Neither is a substitute for a load test on production hardware; they show what
each change does *mechanically*, which a shared VM cannot show reliably.

    python bench/controlled/run.py              # full report -> bench/controlled/REPORT.md
    python bench/controlled/run.py --quick      # smaller workloads (used by tests)
    python bench/controlled/run.py --sweep      # race vs inline across judge / upstream costs
"""
import argparse
import asyncio
import json
import os
import ssl
import statistics
import sys
import threading
import warnings
from collections import Counter
from pathlib import Path

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path[:0] = [str(HERE), str(HERE.parent), str(ROOT / "mitmproxy")]
os.environ.setdefault("GUARDRAILS_URL", "http://guardrails")

import httpx                                         # noqa: E402
from mitmproxy import http                           # noqa: E402
from mitmproxy.test import tflow                     # noqa: E402

import classifier_client                             # noqa: E402
import firewall_addon as new_fw                      # noqa: E402
import legacy_addon as old_fw                        # noqa: E402
import pii                                           # noqa: E402
import session                                       # noqa: E402
import system2                                       # noqa: E402
import vclock                                        # noqa: E402
import workloads as W                                # noqa: E402
from costs import COSTS                              # noqa: E402
from mocks import MockGuardrails, MockJudge          # noqa: E402

URL = "http://api.openai.com/v1/chat/completions"

# ------------------------------------------------------------------ counters
OPS = Counter()
_MAIN = threading.main_thread()


def _install_counters():
    lvl, ldc = ssl.SSLContext.load_verify_locations, ssl.SSLContext.load_default_certs

    def load_verify_locations(self, *a, **k):
        OPS["ca_bundle_loads"] += 1
        return lvl(self, *a, **k)

    def load_default_certs(self, *a, **k):
        OPS["ca_bundle_loads"] += 1
        return ldc(self, *a, **k)
    ssl.SSLContext.load_verify_locations = load_verify_locations
    ssl.SSLContext.load_default_certs = load_default_certs

    init = httpx.AsyncClient.__init__

    def client_init(self, *a, **k):
        OPS["http_clients_built"] += 1
        init(self, *a, **k)
    httpx.AsyncClient.__init__ = client_init

    analyze = pii._analyzer.analyze

    def counted_analyze(*a, **k):
        if threading.current_thread() is _MAIN:
            OPS["presidio_on_loop_thread"] += 1
        return analyze(*a, **k)
    pii._analyzer.analyze = counted_analyze

    scan_many_async = pii.scan_many_async

    async def counted_scan(texts, *a, **k):
        OPS["pii_executor_jobs"] += 1
        return await scan_many_async(texts, *a, **k)
    pii.scan_many_async = counted_scan


# ------------------------------------------------------------------ configs
CONFIGS = {
    "legacy addon":                 dict(addon="legacy"),
    "tier-0 (two classifiers)":     dict(addon="new", s1="composite", s2="off"),
    "System 1 (multi-label)":       dict(addon="new", s1="multilabel", s2="off"),
    "S1 + S2 inline":               dict(addon="new", s1="multilabel", s2="inline"),
    "S1 + S2 raced":                dict(addon="new", s1="multilabel", s2="race"),
    "S2 only (judge everything)":   dict(addon="new", s1="none", s2="inline"),
}


class _LegacyHttpx:
    """Stand-in for the `httpx` module inside legacy_addon: builds real clients
    (so their CA-bundle loads are counted) but routes them to the mock."""

    def __init__(self, transport):
        self._t = transport

    def __getattr__(self, name):
        return getattr(httpx, name)

    def AsyncClient(self, **kw):
        return httpx.AsyncClient(mounts={"all://": self._t}, **kw)


def setup(cfg, max_concurrency=4, max_queue=8, timeout_ms=1500):
    guard = MockGuardrails(cfg.get("s1", "composite"), cfg.get("single_label_roles", ()))
    judge = MockJudge()
    new_fw._cache = session.VerdictCache()
    new_fw._sessions = session.SessionStore()
    new_fw.STATS.clear()
    OPS.clear()
    if cfg["addon"] == "legacy":
        old_fw.httpx = _LegacyHttpx(guard.transport())
        old_fw._URL = "http://guardrails"
        addon = old_fw
    else:
        classifier_client.configure(transport=guard.transport(), url="http://guardrails")
        system2.configure(mode=cfg["s2"], transport=judge.transport(), url="http://judge",
                          timeout_ms=timeout_ms, max_concurrency=max_concurrency, max_queue=max_queue)
        addon = new_fw
    return addon, guard, judge


# ------------------------------------------------------------------ one call
def _text(f) -> str:
    try:
        return json.loads(f.response.get_text())["choices"][0]["message"].get("content") or ""
    except Exception:
        return f.response.get_text() if f.response else ""


async def call(addon, messages, upstream_body, sid=None):
    """Request hook -> (upstream, simulated) -> response hook. Returns a dict
    describing what happened, with simulated timings."""
    loop = asyncio.get_running_loop()
    f = tflow.tflow()
    headers = {"Content-Type": "application/json"}
    if sid:
        headers["X-Session-Id"] = sid
    f.request = http.Request.make("POST", URL, json.dumps({"model": "m", "messages": messages}).encode(), headers)
    t0 = loop.time()
    await addon.request(f)
    forwarded = f.response is None
    sent = f.request.get_text() if forwarded else ""
    replaced = False
    if forwarded:
        await asyncio.sleep(COSTS["upstream_ms"] / 1000)
        f.response = http.Response.make(200, json.dumps(upstream_body), {"Content-Type": "application/json"})
        before = f.response.get_text()
        await addon.response(f)
        replaced = f.response.get_text() != before
    total = (loop.time() - t0) * 1000
    added = round(total - (COSTS["upstream_ms"] if forwarded else 0.0), 6)   # strip float drift
    out = _text(f)
    return {"forwarded": forwarded, "sent": sent, "replaced": replaced, "reply": out,
            "blocked": (not forwarded) or "withheld by the LLM firewall" in out or "blocked by the LLM firewall" in out,
            "added_ms": added}


def _lat(xs):
    xs = sorted(xs)
    return statistics.mean(xs), xs[len(xs) // 2], xs[min(len(xs) - 1, int(0.95 * len(xs)))]


# ------------------------------------------------------------------ W1
def _ok(text):
    return {"id": "x", "object": "chat.completion", "created": 1, "model": "m",
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": text}}]}


async def w1(addon, rows):
    res = []
    for i, (msgs, truth, kind) in enumerate(rows):
        # unique reply per call, so the response-side check is never a free cache hit
        r = await call(addon, msgs, _ok(f"Here is a short summary of item {i}."), sid=f"w1-{i}")
        r.update(truth=truth, kind=kind)
        res.append(r)
    return res


def summarise_w1(res, guard, judge):
    mean, p50, p95 = _lat([r["added_ms"] for r in res])
    fn = sum(1 for r in res if r["truth"] and not r["blocked"])
    fp = sum(1 for r in res if not r["truth"] and r["blocked"])
    wasted = sum(1 for r in res if r["forwarded"] and r["replaced"] and r["blocked"])
    return dict(mean=mean, p50=p50, p95=p95, max=max(r["added_ms"] for r in res), fn=fn, fp=fp, s2=judge.c["calls"],
                passes=guard.c["passes"], trips=guard.c["round_trips"], wasted=wasted)


# ------------------------------------------------------------------ W2
async def w2(addon):
    out = []
    for t in range(W.AGENT_TURNS):
        r = await call(addon, W.agent_history(t), W.agent_response(t), sid="agent-1")
        poisoned_in_ctx = [k for k in (4, 8) if k <= t]
        # marker check, not exact text: tool results are PII-redacted in transit
        exposed = [k for k in poisoned_in_ctx if r["forwarded"] and W.POISON_MARKERS[k] in r["sent"]]
        r.update(turn=t, exposed=exposed,
                 harmful_executed=(t in W.HARMFUL_CALL_TURNS and r["forwarded"] and not r["replaced"]))
        out.append(r)
    return out


def summarise_w2(res, guard, judge):
    mean, _, p95 = _lat([r["added_ms"] for r in res])
    unneutralised = sum(len(r["exposed"]) for r in res if not r["replaced"])
    discarded = sum(len(r["exposed"]) for r in res if r["replaced"])
    harmful = sum(1 for r in res if r["harmful_executed"])
    return dict(mean=mean, p95=p95, unneutralised=unneutralised, discarded=discarded, harmful=harmful,
                s2=judge.c["calls"], passes=guard.c["passes"], trips=guard.c["round_trips"],
                bytes=guard.c["bytes_in"], ops=dict(OPS), turn_ms=[round(r["added_ms"]) for r in res])


# ------------------------------------------------------------------ W3
async def w3(addon, rows):
    res = await asyncio.gather(*(call(addon, m, _ok(f"Hypothetically, no ({i})."), sid=f"b{i}")
                                 for i, (m, _) in enumerate(rows)))
    for r, (_, truth) in zip(res, rows):
        r["truth"] = truth
    return res


# ------------------------------------------------------------------ report
def fmt(x, d=0):
    return f"{x:,.{d}f}"


def main(quick=False, sweep=False, out_path=None):
    _install_counters()
    new_fw.load(None)
    rows = W.prompt_mix(*((50, 5, 3, 10) if quick else (150, 14, 6, 30)))
    report, results = [], {"w1": {}, "w2": {}, "w3": {}}

    def emit(s=""):
        report.append(s)
        print(s)

    emit("# Controlled benchmark report\n")
    emit("Generated by `bench/controlled/run.py`. Simulated latencies are composed from the declared "
         "costs below by the real proxy code on a virtual clock — identical on every run and host. "
         "They exclude the proxy's own CPU time, which is reported as operation counts instead.\n")
    emit("| cost | value | source |\n|---|---|---|")
    src = {"injection_window_ms": "README (measured)", "toxicity_window_ms": "README (measured)"}
    for k, v in COSTS.items():
        emit(f"| {k} | {v:g} | {src.get(k, 'assumption')} |")

    # W1
    emit(f"\n## W1 — single-turn prompt mix ({len(rows)} prompts: benign, clear attacks, "
         f"{sum(1 for r in rows if r[2] == 'ambiguous')} ambiguous; judge wrong on 1 in 10 ambiguous)\n")
    emit("| config | added ms mean | p50 | p95 | missed (FN) | false blocks (FP) | S2 calls | S1 window-passes | upstream calls wasted |")
    emit("|---|---|---|---|---|---|---|---|---|")
    for name, cfg in CONFIGS.items():
        addon, guard, judge = setup(cfg)
        s = summarise_w1(vclock.run(w1(addon, rows)), guard, judge)
        results["w1"][name] = s
        emit(f"| {name} | {fmt(s['mean'])} | {fmt(s['p50'])} | {fmt(s['p95'])} | {s['fn']} | {s['fp']} "
             f"| {s['s2']} | {s['passes']} | {s['wasted']} |")

    # W2
    emit(f"\n## W2 — {W.AGENT_TURNS}-turn agent loop (history re-sent each turn; poisoned tool results "
         "at turns 4 (ambiguous) and 8 (clear); harmful tool calls at turn 6 (goal-inconsistent, no rule "
         "matches) and turn 10 (destructive shell))\n")
    emit("| config | added ms/turn mean | p95 | poisoned-context exposures (model output used) | exposures, output discarded | harmful tool calls passed | S2 calls | S1 window-passes | guardrails round-trips | bytes to guardrails |")
    emit("|---|---|---|---|---|---|---|---|---|---|")
    for name, cfg in CONFIGS.items():
        addon, guard, judge = setup(cfg)
        s = summarise_w2(vclock.run(w2(addon)), guard, judge)
        results["w2"][name] = s
        emit(f"| {name} | {fmt(s['mean'])} | {fmt(s['p95'])} | {s['unneutralised']} | {s['discarded']} "
             f"| {s['harmful']} | {s['s2']} | {s['passes']} | {s['trips']} | {fmt(s['bytes'])} |")

    emit("\n### Proxy work per 12-turn loop (deterministic operation counts)\n")
    emit("| config | CA-bundle loads | HTTP clients built | Presidio calls on event-loop thread | PII executor jobs |")
    emit("|---|---|---|---|---|")
    for name in CONFIGS:
        o = results["w2"][name]["ops"]
        emit(f"| {name} | {o.get('ca_bundle_loads', 0)} | {o.get('http_clients_built', 0)} "
             f"| {o.get('presidio_on_loop_thread', 0)} | {o.get('pii_executor_jobs', 0)} |")

    emit("\n### Added latency per turn (ms)\n")
    emit("| config | " + " | ".join(f"t{t}" for t in range(W.AGENT_TURNS)) + " |")
    emit("|---|" + "---|" * W.AGENT_TURNS)
    for name in CONFIGS:
        emit(f"| {name} | " + " | ".join(str(x) for x in results["w2"][name]["turn_ms"]) + " |")

    # W3
    burst = W.burst(16 if quick else 24)
    emit(f"\n## W3 — burst: {len(burst)} simultaneous ambiguous prompts (every one escalates); "
         "judge limited to 4 concurrent, 1500 ms deadline\n")
    emit("| config | added ms p50 | p95 | max | S2 calls | shed | timed out | FN | FP |")
    emit("|---|---|---|---|---|---|---|---|---|")
    for name, cfg, q in (("S1 + S2 inline, no shedding", CONFIGS["S1 + S2 inline"], 10_000),
                         ("S1 + S2 inline, shed above queue 8", CONFIGS["S1 + S2 inline"], 8),
                         ("S1 + S2 raced, shed above queue 8", CONFIGS["S1 + S2 raced"], 8)):
        addon, guard, judge = setup(cfg, max_queue=q)
        res = vclock.run(w3(addon, burst))
        _, p50, p95 = _lat([r["added_ms"] for r in res])
        s = dict(p50=p50, p95=p95, max=max(r["added_ms"] for r in res), s2=judge.c["calls"],
                 shed=system2.STATS["shed"], timeout=system2.STATS["timeout"],
                 fn=sum(1 for r in res if r["truth"] and not r["blocked"]),
                 fp=sum(1 for r in res if not r["truth"] and r["blocked"]))
        results["w3"][name] = s
        emit(f"| {name} | {fmt(p50)} | {fmt(p95)} | {fmt(s['max'])} | {s['s2']} | {s['shed']} "
             f"| {s['timeout']} | {s['fn']} | {s['fp']} |")

    if sweep:
        base = dict(COSTS)
        emit("\n## Sweep A — how much of System 1's win is structure vs assumed model speed (W1)\n")
        emit("The single-pass structure cuts forward passes regardless of model speed — by half for "
             "roles that need two labels (user messages), not at all for single-label roles (responses), "
             "so −35% on this mix. The *latency* win over tier-0 depends on the System-1 model's "
             "per-window cost, which is an assumption; with a slow System-1 model, route single-label "
             "roles to their dedicated head (`hybrid`).\n")
        emit("`hybrid` = multi-label model for user messages and tool results, the dedicated "
             "33 ms toxicity head for responses (`DECISION_SINGLE_LABEL_ROLES=assistant`).\n")
        emit("| S1 ms/window | tier-0 mean ms | System 1 mean ms | hybrid mean ms | tier-0 passes | System 1 passes | hybrid passes |")
        emit("|---|---|---|---|---|---|---|")
        hybrid = dict(CONFIGS["System 1 (multi-label)"], single_label_roles=("assistant",))
        for s1_ms in (33, 65, 130, 200):
            COSTS.update(s1_window_ms=s1_ms)
            vals = []
            for cfg in (CONFIGS["tier-0 (two classifiers)"], CONFIGS["System 1 (multi-label)"], hybrid):
                addon, guard, judge = setup(cfg)
                vals.append(summarise_w1(vclock.run(w1(addon, rows)), guard, judge))
            emit(f"| {s1_ms} | " + " | ".join(fmt(v["mean"]) for v in vals) + " | "
                 + " | ".join(str(v["passes"]) for v in vals) + " |")
        COSTS.clear(); COSTS.update(base)

        emit("\n## Sweep B — S1 + S2 on W1: added latency mean (ms), inline vs raced\n")
        emit("Racing hides the judge only while it is faster than the upstream call.\n")
        emit("| judge ms | upstream ms | inline | raced |\n|---|---|---|---|")
        for s2_ms in (200, 400, 800, 1600):
            for up in (600, 1200):
                COSTS.update(s2_ms=s2_ms, upstream_ms=up)
                vals = []
                for cfg in (CONFIGS["S1 + S2 inline"], CONFIGS["S1 + S2 raced"]):
                    addon, guard, judge = setup(cfg, timeout_ms=5000)
                    vals.append(summarise_w1(vclock.run(w1(addon, rows)), guard, judge)["mean"])
                emit(f"| {s2_ms} | {up} | {fmt(vals[0])} | {fmt(vals[1])} |")
        COSTS.clear(); COSTS.update(base)

    new_fw.done()
    if out_path:
        Path(out_path).write_text("\n".join(report) + "\n")
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--out", default=str(HERE / "REPORT.md"))
    a = ap.parse_args()
    main(quick=a.quick, sweep=a.sweep, out_path=a.out)
