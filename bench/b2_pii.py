"""B2 -- PII / secrets stage.

Q1  Which NLP engine?   blank tokenizer  vs  en_core_web_sm-architecture
Q2  Analyse once and reuse results for redaction  vs  analyse twice (current code)
Q3  Detection parity between engines for the configured entities
Q4  How to run it from mitmproxy's single asyncio loop under concurrency:
      A inline, sequential with the classifier call           (current code)
      B inline, overlapped (classifier request started first)
      C asyncio.to_thread, overlapped
      D dedicated 1-thread executor, overlapped
      E hybrid: inline if text < 2 KB else thread, overlapped
      F process pool (2 procs), overlapped
    Each simulated flow = PII on its new text + a 35 ms awaited classifier call
    (the README's measured toxicity p50). We report per-flow latency and the
    max event-loop stall (how long *other* flows were frozen).

    python bench/b2_pii.py --spacy /tmp/spacy-standins
"""
from __future__ import annotations

import argparse
import asyncio
import concurrent.futures as cf
import time

from presidio_analyzer import AnalyzerEngine, Pattern, PatternRecognizer
from presidio_analyzer.nlp_engine import SpacyNlpEngine
from presidio_anonymizer import AnonymizerEngine

import common
import corpus

ENTITIES = ["CREDIT_CARD", "US_SSN", "EMAIL_ADDRESS", "PHONE_NUMBER", "IP_ADDRESS", "API_SECRET"]
SECRETS = [Pattern("openai_key", r"sk-[A-Za-z0-9]{20,}", 0.9),
           Pattern("aws_key", r"AKIA[0-9A-Z]{16}", 0.9),
           Pattern("bearer", r"(?i)bearer\s+[A-Za-z0-9._\-]{20,}", 0.6)]
SIZES = [300, 2_000, 10_000, 50_000]


def build(model_path):
    eng = SpacyNlpEngine(models=[{"lang_code": "en", "model_name": model_path}])
    eng.load()
    an = AnalyzerEngine(nlp_engine=eng, supported_languages=["en"])
    an.registry.add_recognizer(PatternRecognizer(supported_entity="API_SECRET", patterns=SECRETS))
    return an


_ANON = AnonymizerEngine()


def redact_twice(an, text):       # mirrors current firewall_addon + pii.py
    res = an.analyze(text=text, entities=ENTITIES, language="en")
    if res:
        res2 = an.analyze(text=text, entities=ENTITIES, language="en")
        return _ANON.anonymize(text=text, analyzer_results=res2).text
    return text


def redact_once(an, text):
    res = an.analyze(text=text, entities=ENTITIES, language="en")
    return _ANON.anonymize(text=text, analyzer_results=res).text if res else text


# ---------------------------------------------------------------- Q4 helpers
_PROC_AN = None


def _proc_init(path):
    global _PROC_AN
    _PROC_AN = build(path)


def _proc_redact(text, windowed=False):
    if windowed:
        return _windowed(_PROC_AN)(text)
    return redact_once(_PROC_AN, text)


async def simulate(strategy, an, texts, concurrency, classifier_ms=35.0, pools=None, windowed=False):
    lags, lat = [], []
    stop = False

    async def lag_probe():
        while not stop:
            t0 = time.perf_counter()
            await asyncio.sleep(0.001)
            lags.append((time.perf_counter() - t0) * 1000 - 1.0)

    async def classify():
        await asyncio.sleep(classifier_ms / 1000)

    async def flow(text):
        t0 = time.perf_counter()
        if strategy == "A_inline_sequential":
            redact_once(an, text)
            await classify()
        else:
            task = asyncio.create_task(classify())
            await asyncio.sleep(0)                      # let the request go out
            if strategy == "B_inline_overlapped":
                redact_once(an, text)
            elif strategy == "C_to_thread":
                await asyncio.to_thread(redact_once, an, text)
            elif strategy == "D_dedicated_thread":
                await asyncio.get_running_loop().run_in_executor(pools["thread1"], redact_once, an, text)
            elif strategy == "E_hybrid_2KB":
                if len(text) < 2048:
                    redact_once(an, text)
                else:
                    await asyncio.to_thread(redact_once, an, text)
            elif strategy == "F_process_pool":
                await asyncio.get_running_loop().run_in_executor(pools["proc"], _proc_redact, text, windowed)
            await task
        lat.append((time.perf_counter() - t0) * 1000)

    probe = asyncio.create_task(lag_probe())
    t0 = time.perf_counter()
    sem = asyncio.Semaphore(concurrency)

    async def bounded(t):
        async with sem:
            await flow(t)
    await asyncio.gather(*(bounded(t) for t in texts))
    wall = time.perf_counter() - t0
    stop = True
    await probe
    s = common.summarize(lat)
    s.update(max_loop_stall_ms=max(lags) if lags else 0.0,
             throughput_rps=len(texts) / wall)
    return s


def _windowed(an):
    """Windowed analysis (the new pii.analyze) over an arbitrary analyzer."""
    import pii
    def run(text):
        out = []
        for s, e in pii._windows(text):
            for r in an.analyze(text=text[s:e], entities=ENTITIES, language="en"):
                r.start += s; r.end += s; out.append(r)
        return _ANON.anonymize(text=text, analyzer_results=out).text if out else text
    return run


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spacy", default="/tmp/spacy-standins")
    ap.add_argument("--skip-q4", action="store_true")
    args = ap.parse_args()
    engines = {"blank": f"{args.spacy}/blank-en", "sm_arch": f"{args.spacy}/sm-arch"}
    ans = {k: build(v) for k, v in engines.items()}
    variants = {
        "sm_arch+twice (~current)": lambda t: redact_twice(ans["sm_arch"], t),
        "sm_arch+once": lambda t: redact_once(ans["sm_arch"], t),
        "blank+once": lambda t: redact_once(ans["blank"], t),
        "blank+once+windowed (new)": _windowed(ans["blank"]),
    }
    out = {"q1_q2": [], "q3": {}, "q4": []}

    # Q1 + Q2 -------------------------------------------------------------
    for density, every in (("none", 0), ("sparse(1/20 sent.)", 20), ("dense(1/4 sent.)", 4)):
        for size in SIZES:
            t = corpus.text(size, seed=size, pii_every=every)
            for name, fn in variants.items():
                if size == 50_000 and name.startswith("sm_arch+twice"):
                    rep = 3
                else:
                    rep = 40 if size <= 2000 else (12 if size <= 10_000 else 5)
                s = common.timeit_ms(lambda: fn(t), repeat=rep, warmup=2)
                row = {"pii_density": density, "size_chars": size, "variant": name, **s}
                out["q1_q2"].append(row)
                print(density, size, name, f"p50={s['p50_ms']:.2f}ms p95={s['p95_ms']:.2f}ms", flush=True)

    # Q3: parity blank vs sm_arch on the configured entity set ----------------
    mism, total = 0, 0
    for seed in range(200):
        t = corpus.text(1500, seed=seed, pii_every=5)
        a = {(r.entity_type, r.start, r.end) for r in ans["blank"].analyze(text=t, entities=ENTITIES, language="en")}
        b = {(r.entity_type, r.start, r.end) for r in ans["sm_arch"].analyze(text=t, entities=ENTITIES, language="en")}
        total += len(b); mism += len(a ^ b)
    out["q3"] = {"docs": 200, "entities_found_sm_arch": total, "span_mismatches_blank_vs_sm_arch": mism}
    print("parity:", out["q3"], flush=True)

    # Q4 --------------------------------------------------------------------
    if not args.skip_q4:
        global redact_once
        pools = {"thread1": cf.ThreadPoolExecutor(1),
                 "proc": cf.ProcessPoolExecutor(2, initializer=_proc_init, initargs=(engines["blank"],))}
        list(pools["proc"].map(_proc_redact, ["warm"] * 4))
        mixes = {
            "typical(300c user / 3KB tool)": [corpus.text(300 if i % 2 else 3000, i, 20) for i in range(64)],
            "large(50KB tool)": [corpus.text(50_000, i, 20) for i in range(16)],
        }
        for label, fn_once in (("blank+once", None), ("blank+once+windowed (new)", variants["blank+once+windowed (new)"])):
            for mix, texts in mixes.items():
                for conc in (1, 16):
                    for strat in ("A_inline_sequential", "B_inline_overlapped", "C_to_thread", "F_process_pool"):
                        orig = redact_once
                        if fn_once is not None:
                            redact_once = lambda an, t, _f=fn_once: _f(t)
                        try:
                            s = asyncio.run(simulate(strat, ans["blank"], texts, conc, pools=pools, windowed=fn_once is not None))
                        finally:
                            redact_once = orig
                        row = {"variant": label, "mix": mix, "concurrency": conc, "strategy": strat, **s}
                        out["q4"].append(row)
                        print(label, mix, conc, strat, f"p50={s['p50_ms']:.1f} p95={s['p95_ms']:.1f} "
                              f"stall={s['max_loop_stall_ms']:.1f} rps={s['throughput_rps']:.0f}", flush=True)
        pools["proc"].shutdown()
    common.write("b2_pii", out)


if __name__ == "__main__":
    main()
