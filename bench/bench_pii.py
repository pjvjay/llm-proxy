"""PII stage cost and placement.

Questions answered:
  1. How much does analysing twice (detect, then redact) cost vs analysing once?
  2. What does the PII stage cost per message, and how does it scale with length?
  3. Where should it run? Options, each measured as *per-flow latency* when N
     flows are in flight and each flow also awaits a 35 ms classifier round-trip
     (simulated with asyncio.sleep so only the PII placement varies):
       a. sync on the event loop, after the classifier   (current addon)
       b. sync on the event loop, but the classifier await overlaps nothing
       c. thread pool, concurrent with the classifier    (new pii.scan_and_redact_async)
       d. process pool, concurrent with the classifier

Run from repo root:  python bench/bench_pii.py
"""
import asyncio, os, statistics, sys, time, warnings
from concurrent.futures import ProcessPoolExecutor

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "mitmproxy"))
import pii  # noqa: E402

CLASSIFIER_MS = 0.035
SHORT = "Please email the invoice to alice@example.com and call +1 415 555 2671 if it bounces."
LONG = (SHORT + " The server at 10.1.2.3 logged card 4111 1111 1111 1111 in the debug output. ") * 8
CLEAN = "Summarise the attached quarterly report in three bullet points, focusing on revenue. " * 4

def _pct(xs, p):
    xs = sorted(xs); return xs[min(len(xs) - 1, int(p * len(xs)))]

def bench_sync(label, fn, n=200):
    for _ in range(10): fn()
    xs = []
    for _ in range(n):
        t0 = time.perf_counter(); fn(); xs.append((time.perf_counter() - t0) * 1000)
    print(f"| {label} | {statistics.median(xs):.2f} | {_pct(xs, 0.95):.2f} |")

def twice(text):   # what firewall_addon.py did: analyze to detect, analyze again to redact
    _, found = pii.scan_and_redact(text)
    if found:
        pii.scan_and_redact(text)

def nlp_only(text):
    pii._analyzer.nlp_engine.process_text(text, "en")

# ---------------------------------------------------------------- placement
async def flow_a(text):               # current: classifier, THEN sync PII on the loop
    await asyncio.sleep(CLASSIFIER_MS)
    pii.scan_and_redact(text)

async def flow_c(text):               # thread pool, concurrent with classifier
    await asyncio.gather(asyncio.sleep(CLASSIFIER_MS), pii.scan_and_redact_async(text))

_ppool = None
async def flow_d(text):               # process pool, concurrent with classifier
    loop = asyncio.get_running_loop()
    await asyncio.gather(asyncio.sleep(CLASSIFIER_MS),
                         loop.run_in_executor(_ppool, pii.scan_and_redact, text))

async def placement(label, flow, text, conc, total=160):
    lat, stall = [], [0.0]
    async def worker(k):
        for _ in range(k):
            t0 = time.perf_counter(); await flow(text); lat.append((time.perf_counter() - t0) * 1000)
    async def heartbeat():            # how long does a 1 ms timer get delayed? = loop blocked
        while True:
            t0 = time.perf_counter(); await asyncio.sleep(0.001)
            stall[0] = max(stall[0], (time.perf_counter() - t0) * 1000 - 1)
    for _ in range(4): await flow(text)
    hb = asyncio.create_task(heartbeat())
    t0 = time.perf_counter()
    await asyncio.gather(*[worker(total // conc) for _ in range(conc)])
    wall = time.perf_counter() - t0
    hb.cancel()
    print(f"| {label} | {conc} | {statistics.median(lat):.1f} | {_pct(lat, 0.95):.1f} | {total / wall:.0f} | {stall[0]:.1f} |")

async def main():
    global _ppool
    pii.warm_up()
    print(f"PII_NLP_MODEL={pii._NLP_MODEL}\n")
    print("### per-call cost (single thread, nothing else running)\n")
    print("| variant | p50 ms | p95 ms |\n|---|---|---|")
    bench_sync("spaCy pipeline only, short", lambda: nlp_only(SHORT))
    bench_sync("analyze+anonymize once, short (2 entities)", lambda: pii.scan_and_redact(SHORT))
    bench_sync("analyze twice (old addon), short", lambda: twice(SHORT))
    bench_sync("analyze+anonymize once, clean 350 chars", lambda: pii.scan_and_redact(CLEAN))
    bench_sync("analyze+anonymize once, long 1.3k chars (32 entities)", lambda: pii.scan_and_redact(LONG))
    bench_sync("analyze twice (old addon), long", lambda: twice(LONG))

    print("\n### placement: per-flow latency with a 35 ms classifier await in the same flow\n")
    _ppool = ProcessPoolExecutor(max_workers=2)
    _ppool.submit(pii.warm_up).result()
    for name, text in (("short, 2 entities, ~1.7 ms", SHORT), ("long, 32 entities, ~48 ms", LONG)):
        print(f"\n**{name}**\n")
        print("| placement | concurrent flows | p50 ms | p95 ms | flows/s | max loop stall ms |\n|---|---|---|---|---|---|")
        for conc in (1, 8, 32):
            await placement("a. sync on loop, after classifier (old)", flow_a, text, conc)
            await placement("c. thread pool ‖ classifier", flow_c, text, conc)
            await placement("d. process pool ‖ classifier", flow_d, text, conc)
    _ppool.shutdown()

if __name__ == "__main__":
    asyncio.run(main())
