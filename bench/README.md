# Benchmarks

There are two kinds of benchmark here, and they are not equally trustworthy.

## 1. Controlled benchmarks — `bench/controlled/` (use these)

Deterministic and host-independent; the full suite runs in ~1.5 minutes, the
quick subset used by `tests/test_controlled.py` in under 30 s.

* The **real** proxy code (original addon and current addon) runs through real
  mitmproxy flow objects. Only the network endpoints — guardrails service,
  System-2 judge, upstream LLM — are mocked, each charging a **declared cost**
  (`costs.py`: README measurements where they exist, explicit assumptions
  otherwise) on a **virtual clock** (`vclock.py`). Latency is therefore
  *composed by the proxy's control flow* — what is sequential, concurrent,
  cached, skipped, escalated, raced or shed — and is identical on every run.
* The proxy's own CPU work is reported as **exact operation counts** (CA-bundle
  loads, HTTP clients built, Presidio calls on the event-loop thread, forward
  passes, judge calls, round-trips, bytes), not timed.
* `tests/test_controlled.py` asserts every claimed improvement and that two
  runs produce identical results.
* Sweeps show how the conclusions move when the assumed costs change.

Results: [`controlled/REPORT.md`](controlled/REPORT.md).

```bash
python bench/controlled/run.py --sweep     # full report -> bench/controlled/REPORT.md
python tests/test_controlled.py            # assertions + determinism check
```

What it cannot tell you: absolute latency on your hardware, CPU saturation,
GC pauses, real model accuracy. Those need a load test on dedicated hardware
with the real models.

## 2. Exploratory wall-clock benchmarks — the `bench_*.py` scripts (indicative only)

These time real work with `perf_counter` and were run on a **shared 4-vCPU
sandbox VM** with visible CPU steal. Shared hosts are not a sound basis for
performance decisions: unrelated load produces spikes that look like results.
The numbers below are kept as a record and as scripts you can re-run on
dedicated hardware; none of the current defaults rests on them alone:

| earlier conclusion | status |
|---|---|
| per-call `httpx.AsyncClient()` rebuilt an SSL context (CA-bundle load) every call | **kept** — confirmed deterministically: 12 CA-bundle loads per 12-turn loop before, 0 after (controlled counts) |
| Presidio must leave the event loop; process pool, not threads | **kept on principle** — pure-Python CPU work holds the GIL, so threads cannot run it in parallel and compete with the loop thread. Controlled count: Presidio calls on the loop thread 12 → 0. Thread-vs-process magnitudes are *not* established by the VM numbers |
| incremental scanning avoids O(N²) work | **kept** — algorithmic; controlled counts: per-turn work constant, not growing |
| guardrails `GUARDRAILS_WORKERS=1` was "fastest" | **reverted** — the noisy data did not justify picking 1 over 2. Default is now 2 workers × cores/2 intra-op threads: the principle (never more compute threads than cores) stands; the split must be tuned on dedicated hardware |
| literal-trigger prefilter for tool-call rules | **kept** — reduces regex evaluations per argument; the ~30× timing is indicative only |

| script | question |
|---|---|
| `bench_transport.py` | what does the addon → guardrails HTTP hop cost, and which transport? |
| `bench_pii.py` | what does Presidio cost, and where should it run? |
| `bench_session.py` | how much work does an agent loop do under each scanning strategy? |
| `bench_windows.py` | truncate vs windows, batched vs sequential, ORT threading under load |
| `bench_tool_policy.py` | what does deterministic tool-call gating cost? |
| `bench_addon.py` | old addon vs new addon, end to end, on a simulated agent loop |

---

### E1. Transport: the 53 ms that wasn't inference

`classifier_client.py` used to create a new `httpx.AsyncClient` per call.
Constructing a client builds an SSL context — it loads the CA bundle — even
when the URL is plain `http://`:

| | ms |
|---|---|
| `httpx.AsyncClient()` construct | 53.4 |
| `ssl.create_default_context()` alone | 54.2 |
| `httpx.AsyncClient(verify=False)` construct | 0.37 |

Against a zero-compute stub of the guardrails API:

| variant | mode | req/s | p50 ms | p95 ms | p99 ms | errors |
|---|---|---|---|---|---|---|
| A new client/call (old) | seq | – | 75.7 | 165.5 | 206.6 | 0 |
| A new client/call (old) | conc=8 | 14 | 558.9 | 862.3 | 1084.4 | 0 |
| A new client/call (old) | conc=32 | 20 | 1579.1 | 2249.3 | 2624.1 | 1 |
| B keep-alive HTTP/1.1 | seq | – | 3.3 | 5.3 | 8.7 | 0 |
| B keep-alive HTTP/1.1 | conc=8 | 473 | 10.0 | 45.3 | 219.4 | 0 |
| B keep-alive HTTP/1.1 | conc=32 | 362 | 71.2 | 185.6 | 392.8 | 0 |
| C keep-alive HTTP/2 | seq | – | 3.5 | 5.1 | 7.1 | 0 |
| C keep-alive HTTP/2 | conc=32 | 314 | 51.9 | 291.9 | 460.9 | 0 |
| D Unix socket | seq | – | 3.3 | 11.0 | 20.3 | 0 |
| D Unix socket | conc=32 | 507 | 54.8 | 117.8 | 134.0 | 0 |

**Decision:** one module-level client, `verify=False` (plain http), keep-alive.
75 ms → 3 ms per hop, and the old variant was paid twice per call (request +
response path). HTTP/2 and Unix sockets are within noise for this traffic
pattern; UDS is available via `GUARDRAILS_UDS` if the containers share a volume.

### E2. PII: cost and placement

Presidio with a tokenizer-only spaCy pipeline (`PII_NLP_MODEL=blank`, the new
default — every entity in the configured list is pattern-based, so NER was pure
overhead; `en_core_web_sm` was not installable offline here so its cost is not
measured, only avoided):

| variant | p50 ms | p95 ms |
|---|---|---|
| spaCy pipeline only, short text | 0.04 | 0.07 |
| analyze + anonymize once, short (2 entities) | 1.74 | 2.14 |
| analyze twice (old addon: detect, then redact) | 3.59 | 5.80 |
| analyze + anonymize once, clean 350 chars | 0.60 | 0.84 |
| analyze + anonymize once, 1.3 KB with 32 entities | 48.0 | 55.3 |
| analyze twice (old addon), same | 93.9 | 111.6 |

Cost is driven by entity *hits*, not length: clean text is sub-millisecond,
entity-dense text is tens of ms. That tail is what decides placement.

Per-flow latency when each flow also awaits a 35 ms classifier call, and the
max delay of a 1 ms timer on the event loop during the run ("loop stall" — what
every *other* flow on the proxy experiences):

**short input (~1.7 ms of PII work)**

| placement | flows | p50 ms | p95 ms | flows/s | max loop stall ms |
|---|---|---|---|---|---|
| a. sync on loop, after classifier (old) | 32 | 101 | 265 | 236 | 197 |
| c. thread pool ‖ classifier | 32 | 224 | 311 | 131 | 51 |
| d. process pool ‖ classifier | 32 | 67 | 138 | 392 | 25 |

**long input (~48 ms of PII work)**

| placement | flows | p50 ms | p95 ms | flows/s | max loop stall ms |
|---|---|---|---|---|---|
| a. sync on loop, after classifier (old) | 8 | 463 | 508 | 18 | 454 |
| c. thread pool ‖ classifier | 8 | 833 | 1765 | 9 | 207 |
| d. process pool ‖ classifier | 8 | 291 | 560 | 24 | 41 |
| a. sync on loop, after classifier (old) | 32 | 1522 | 2343 | 19 | **2398** |
| c. thread pool ‖ classifier | 32 | 2942 | 5045 | 9 | 230 |
| d. process pool ‖ classifier | 32 | 1058 | 2012 | 24 | 52 |

**Decision:** process pool (`PII_EXECUTOR=process`, 2 spawned workers). The
"obvious" fix — `asyncio.to_thread` — is the *worst* option under load: the
GIL convoy between worker threads and the loop thread halves throughput versus
doing nothing. The old inline placement stalls the loop for up to 2.4 s at 32
concurrent flows, which would freeze every other flow, including future
streaming ones. Scale-out path if 2 workers saturate: move PII into the
guardrails service (same process-isolation property, plus the 3 ms hop).

### E3. Incremental scanning for agent loops

30-turn loop, a 900-token tool result per turn, windows of 512 with overlap.
Classifier cost per window from the README (toxicity 33 ms, injection 130 ms):

| strategy | windows scored, whole loop | tokens covered at turn 30 | classifier ms at turn 30 (tox / inj) | whole loop (tox / inj) |
|---|---|---|---|---|
| A truncate (old) | 30 | 60 — **blind** to tool results and to turn ≥ 2 | 36 / 133 | 1 080 / 3 990 |
| B rescan all, windowed | 1 030 | 26 160 | 2 247 / 8 843 | 34 080 / 133 990 |
| C incremental (new) | 88 | 900 | 102 / 393 | 2 994 / 11 530 |

**Decision:** content-hash verdict cache (`session.py`), keyed on
`sha256(role ‖ text)`. The old path was only "fast" because it never looked at
tool results and truncated the rest; the naive fix is quadratic; the cache makes
full coverage cost ≈ one new unit per turn. Hashing is noise (sha256: 4.5 µs per
2.4 KB message, hardware-accelerated; blake2b was 27 µs here, so sha256 won);
cache get/put is ~1–7 µs.

### E4. Long inputs and ONNX threading

Synthetic 4-layer encoder (hidden 384, ~MiniLM/Granite-HAP shape), random
weights — ratios only. Min of 15 runs, one request, 4 intra-op threads:

| tokens | windows | truncate-512 ms | windows sequential ms | windows batched ms | batched / truncate |
|---|---|---|---|---|---|
| 512 | 1 | 48 | 35 | 36 | 0.8× |
| 900 | 3 | 33 | 80 | 105 | 3.1× |
| 2000 | 5 | 32 | 213 | 376 | 11.6× |
| 4000 | 11 | 41 | 671 | 503 | 12.3× |

Windowing is not free — it is the price of not being blind past token 512 — and
batching the windows is a wash on CPU (intra-op threads already saturate the
cores). `MAX_WINDOWS` (default 8) bounds the worst case by sampling first, last
and evenly spaced windows.

Concurrent 512-token requests, 4 cores:

| config | conc | p50 ms | p95 ms | req/s |
|---|---|---|---|---|
| intra_op=4, 1 worker | 1 / 4 / 8 | 46 / 40 / 36 | 121 / 48 / 74 | 15.6 / 24.5 / 22.8 |
| **intra_op=4, 4 workers (old default)** | 1 / 4 / 8 | 95 / **392** / **376** | 115 / 466 / 468 | 10.2 / 9.0 / 9.5 |
| intra_op=1 × 4 sessions, 4 workers | 1 / 4 / 8 | 136 / 137 / 140 | 147 / 192 / 180 | 7.5 / 22.8 / 23.8 |
| intra_op=2 × 2 sessions, 2 workers | 1 / 4 / 8 | 92 / 91 / 87 | 125 / 122 / 121 | 10.3 / 19.6 / 20.9 |

**Decision (revised):** the old default (4 worker threads × ORT's default of
all cores) runs 4× more compute threads than cores; avoiding that is a
principle, not a measurement. The earlier choice of 1 worker rested on these
noisy numbers and has been reverted to 2 workers × cores/2 threads. Tune the
split on dedicated hardware.

### E5. Tool-call gating

| case | evaluate() p50 µs | p99 µs |
|---|---|---|
| small benign (search) | 16 | 68 |
| shell destructive (rule hit) | 12 | 55 |
| http with URL (SSRF check) | 43 | 118 |
| nested 2 KB args (40 leaves), benign | 567 | 893 |
| 20 KB args, benign | 369 | 594 |
| 100 KB args, benign | 1 918 | 5 462 |
| 100 KB args, trigger literal present everywhere | 22 776 | 30 971 |

Without the literal-trigger prefilter, Python `re` cost ~0.55 ms per KB
(20 KB: 11.7 ms; 100 KB: 56 ms) because `\b` + IGNORECASE alternations have no
fast path. **Decision:** each rule declares trigger substrings checked with
C-speed `str.find` before its regex runs — 30× cheaper on benign large
arguments. The pathological case is the residual regex cost; if it shows up in
practice, large-argument scans can move to the PII process pool.

### E6. End to end: old addon vs new addon on a 12-turn agent loop

Real mitmproxy flow objects, real Presidio, real HTTP hop; only inference is
stubbed. Each turn re-sends the whole history (1 user message, then a 3.2 KB
tool result per turn containing 30 email addresses, a user follow-up every 4th
turn). `new-nocache` is the new addon with the verdict cache cleared every turn.

**Stub inference cost = 0 → pure proxy overhead**

| variant | turn 1 ms | turn 6 ms | turn 12 ms | mean ms/turn | loop total ms | tool-result chars screened |
|---|---|---|---|---|---|---|
| old | 81 | 73 | 64 | 74 | 888 | 0 |
| new-nocache | 10 | 102 | 144 | 97 | 1 167 | 33 660 |
| **new** | **11** | **28** | **25** | **27** | **329** | 33 660 |

**Stub inference = 130 ms injection / 33 ms toxicity per window (README numbers)**

| variant | turn 1 ms | turn 6 ms | turn 12 ms | mean ms/turn | loop total ms | tool-result chars screened |
|---|---|---|---|---|---|---|
| old | 235 | 241 | 247 | 296 | 3 552 | 0 |
| new-nocache | 176 | 1 308 | 2 953 | 1 467 | 17 608 | 33 660 |
| **new** | **169** | **272** | **272** | **267** | **3 208** | 33 660 |

Reading it: proxy overhead per turn 74 → 27 ms (the residual is Presidio
redacting 30 emails per tool result, in a worker process); with realistic
inference the new path is slightly *faster* per turn than the old one while
screening every tool result the old one ignored — because it scores only new
content. Fixing the bypass without the cache would have cost 2.9 s by turn 12.
