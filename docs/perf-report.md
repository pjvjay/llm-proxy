# Performance report: steps 1–3 (hygiene, incremental scanning, agentic gating)

This covers the implementation of steps 1–3 from
[openjev-evaluations-design.md](openjev-evaluations-design.md) §7. For every
component I benchmarked several options before choosing one. Raw results are in
`bench/results/*.json` and the scripts are in `bench/`.

## 1. Headline

**End to end through real `mitmdump`** (B6), agent loops with a 2 KB tool
result per turn. "Added" means proxied latency minus direct-to-upstream latency.
The upstream is a mock that replies instantly, so all added latency comes from
the firewall.

Proxy overhead alone (guardrails service stubbed to answer instantly):

| | conc 1: added p50 / p95 | conc 8: added p50 / p95 | what is screened |
|---|---|---|---|
| original code | 125.8 / 157.5 ms | 945 / 1,111 ms | user text only |
| **new** | **21.9 / 46.9 ms** | **126 / 243 ms** | user text + every tool result + tool calls |
| new, cache off | 31.9 / 53.4 ms (11 → 41 ms by turn 10) | 182 / 408 ms | same |

An earlier run, in a faster VM phase and before the per-target request split
(§3.6), measured 83.2 → 15.8 ms and 624 → 83.5 ms. The ratios are the same:
5.3–5.7× at concurrency 1 and 7.5× at 8.

With latency stand-ins of the real models (same architecture and size, random
weights; see §3):

| injection backend | conc 1: added p50 / p95 | conc 4: added p50 / p95 | tool results scanned |
|---|---|---|---|
| original code (DeBERTa-base, PyTorch) | 402 / 768 ms | 853 / 970 ms | **none** |
| new, DeBERTa-base PyTorch (current model) | 2,012 / 2,025 ms ⚠ | 2,006 / 2,018 ms ⚠ | timed out at 2 s → quarantined |
| new, DeBERTa-base **ONNX int8** | 525 / 634 ms | 1,467 / 1,928 ms | all |
| new, xsmall-class (≈ Prompt Guard 2 22M) **ONNX int8** | **294 / 695 ms** | **781 / 937 ms** | all |
| new, xsmall ONNX int8, **cache off** | 149 → 2,096 ms (turn 0 → 7) | at timeout from turn 2 | timed out → quarantined |

What this means:

1. The firewall's own overhead is **5–7.5× lower**, even though it now screens
   much more.
2. **Screening tool results is only practical with incremental scanning.** With
   the cache off, per-turn cost grows with the length of the conversation and
   hits the 2 s timeout by turn 4.
3. **The current injection model (DeBERTa-base on PyTorch) is too slow to
   screen tool results on CPU within a 2 s budget.** It fails safe: the tool
   output is quarantined. With an int8 ONNX export (1.5–3.3× faster, and the
   export was verified faithful; see §3.5) or an xsmall-class model, every tool
   result is scanned, at lower latency than the original code, which scanned
   none. **Before switching models, validate accuracy on the real weights with
   `evals/run_eval.py`.** I measured speed only.

## 2. Method, environment, caveats

* **Hardware.** 4-vCPU Intel Xeon @ 2.6 GHz Firecracker VM, Python 3.12. Under
  load, **CPU steal was about 19%** (`vmstat`), and absolute timings drifted up
  to ~3× between runs. To cope, benchmarks interleave the options across rounds
  and report medians. Treat differences under ~20–30% as noise. Conclusions
  below rest on large or repeated effects.
* **Models.** No HuggingFace access, so `bench/standins/` builds models with the
  real architectures and random weights:
  * a RoBERTa-style 38.5M model (4 layers, hidden 576) exported to int8 ONNX,
    for Granite HAP 38M;
  * the exact `deberta-v3-base` config (184M), on PyTorch.

  Latency depends on architecture, not weights. Calibration: the DeBERTa
  stand-in gives 149 ms p50 on the repo's eval texts, against the README's
  130 ms. The toxicity stand-in is faster than the README's 33 ms (which
  included HTTP), so toxicity costs are, if anything, understated. **Scores are
  meaningless**, so B6 stand-in mode sets the thresholds above 1.0.
* **spaCy.** `en_core_web_sm` could not be downloaded, so I used a randomly
  initialised pipeline with its architecture (same compute) for latency, and
  the tokenizer-only pipeline for parity checks.
* **Baseline.** B6 runs the original addon and guardrails server from the
  baseline git commit. Its `AnalyzerEngine()` would download
  **`en_core_web_lg`**, so it is pointed at the `sm`-architecture stand-in. That
  is lighter than what the baseline really loads, so **baseline numbers are
  optimistic**.
* **One harness bug I found and fixed.** The first B6 runs seeded tool results
  so that different sessions shared content, which produced spurious
  cross-session cache hits. Those runs were discarded. All B6 numbers here come
  from the fixed corpus (0 shared messages across 18 sessions, verified).

## 3. Findings by component

### 3.1 Transport to the guardrails service (B1)

| option | conc 1: p50 / p95 | conc 16: p50 / p95 | rps @16 |
|---|---|---|---|
| new `httpx.AsyncClient` per call (**original**) | 66.8 / 118.8 ms | 871 / 1,372 ms | 15 |
| persistent keep-alive TCP (**chosen**) | 7.6 / 13.9 ms | 56 / 189 ms | 207 |
| persistent Unix socket | 6.0 / 11.1 ms | 43 / 140 ms | 271 |
| in-process call (floor) | 0.01 ms | 0.08 ms | — |

*(Medians of 3 interleaved rounds. In a quieter VM phase: 61 / 2.1 / 1.5 ms.)*

* **Root cause.** Constructing an `httpx.AsyncClient` builds a TLS context and
  parses the CA bundle, even for `http://`. That took **52.7 ms of CPU**, on the
  event loop, for every classify call on both request and response paths.
* **Unix socket.** Better in 2 of 3 rounds and tied in one, a gain of ~1 ms
  against 30–1,000 ms of model time. It is opt-in (`GUARDRAILS_UDS`), not the
  default, because it requires a shared volume.
* **HTTP/2 is not an option.** httpx does not speak cleartext h2c, and uvicorn
  does not serve HTTP/2.
* **Idle keep-alive is safe.** After the server closed an idle pooled
  connection, the next request reconnected cleanly. Client expiry is set to
  60 s and server keep-alive to 75 s. Only stale-connection errors are retried,
  once; timeouts are not, since retrying them would double the worst-case
  latency.

### 3.2 PII / secrets (B2)

**Engine.** `AnalyzerEngine()` silently defaults to `en_core_web_lg`, and the
Dockerfile's `en_core_web_sm` download was never used. None of the six
configured entities needs NER. Across 200 documents (1,049 entities), a
tokenizer-only pipeline had **0 span mismatches** against the
`sm`-architecture pipeline.

**Options, p50 in ms** (sparse PII = one PII item per ~20 sentences):

| PII | chars | sm, analyse twice (~original) | sm, once | blank, once | **blank + candidate windows (new)** |
|---|---|---|---|---|---|
| none | 300 | 19.0 | 18.4 | 0.95 | **0.02** |
| none | 10,000 | 265 | 283 | 19.2 | **0.81** |
| none | 50,000 | 1,609 | 1,540 | 103 | **5.2** |
| sparse | 2,000 | 154 | 104 | 3.6 | **1.0** |
| sparse | 10,000 | 599 | 294 | 34.6 | **22.6** |
| sparse | 50,000 | 3,168 | 2,741 | 337 | **166** |
| dense (1/4) | 2,000 | 130 | 54 | **19.5** | 26.7 |
| dense (1/4) | 10,000 | 710 | 387 | **94.9** | 110.6 |
| dense (1/4) | 50,000 | 3,351 | 1,320 | 542 | **455** |

* **Candidate windows.** A superset regex finds every place a configured
  recognizer *could* match: `@`, `sk-`, `AKIA`, `bearer`, IPv6 hex-colon groups,
  and ≥4 digits with ≤4-character gaps. The digit rule is derived from
  `phonenumbers`' `_PUNCTUATION_LIMIT={0,4}` and the shortest valid national
  number across Presidio's regions, which is 4 digits (DE). Presidio then runs
  only on ±96-character windows, snapped to whitespace.
* **Parity.** Span-for-span parity with full-text analysis is enforced by
  `tests/test_pii_prefilter.py`: edge formats, 400 numeric fuzz cases, and
  corpus documents. The test is verified to fail when the margin is broken.
* **Rejected: joining all windows into one Presidio call.** It was measured
  *slower*, 537 vs 332 ms on dense 50 KB, because Presidio's cost is
  super-linear.
* **Dense PII gets no reliable benefit** (±15–35%, near noise). I didn't add a
  heuristic. That work runs off the event loop and is cached per message.
* **Profile.** The phone recognizer is 57–80% of Presidio's cost: 8 regions,
  each a `phonenumbers` pass. `PII_PHONE_REGIONS=US` makes dense text 2.3×
  faster but misses national formats of other regions (`+CC` formats are still
  found). It is opt-in only.

**Execution model.** Each flow does PII work plus a 35 ms awaited classifier
call. Typical turn, 16 concurrent flows:

| strategy | p50 | p95 | max event-loop stall | rps |
|---|---|---|---|---|
| blank, inline sequential (original pattern) | 51.4 | 80.4 | 60.2 | 200 |
| blank, `asyncio.to_thread` | 158 | 313 | 89.3 | 88 |
| blank + windows, inline | 39.1 | 48.4 | 26.9 | 321 |
| **blank + windows, process pool (2, `spawn`)** | **36.2** | **36.4** | **4.4** | **423** |

Large 50 KB tool results, 16 concurrent: inline p50 1,231 ms with a **2.5 s
event-loop stall**, vs process pool 396 ms with a 1.1 ms stall.

* **Threads lose every time.** Presidio is CPU-bound Python, so they contend
  for the GIL.
* **Implementation.** The prefilter runs inline (text with no candidates never
  leaves the loop). Candidates go to a `spawn` pool, because forking a process
  that runs an event loop and threads is unsafe. Warm-up takes ~15 s, so
  requests are handled inline until the pool reports ready. A broken pool falls
  back to inline analysis, never to skipping PII.

### 3.3 Classifier windowing, batching, threads (B3)

**Truncation was a blind spot.** `truncation=True, max_length=512` meant
anything after ~512 tokens was never scored. The original addon also joined
*all* user messages into one string, so in a long chat **the newest user turns
were never seen by the ML stages**. Both are covered by regression tests.

**Windowing is not free on CPU.** Cost is linear in the number of windows:

| model | 10 KB (5 windows) truncate | 10 KB windowed | 50 KB (24 w) truncate | 50 KB windowed |
|---|---|---|---|---|
| toxicity (38M ONNX int8) | 48.5 ms | 201 ms | 70.2 ms | 983 ms |
| injection (DeBERTa-base PyTorch) | 935 ms | 5,536 ms | 1,124 ms | 26,470 ms |

* **Batching did not help on CPU.** For DeBERTa it made things *worse*: 5
  windows took 5.5 s at batch 1, 6.3 s at batch 4 and 6.3 s at batch 8 (an
  earlier run gave 5.6 / 8.7 / 11.4 s). For the small ONNX model it was within
  noise. So batch size is set per backend, with injection at 1.
* **Per-check window caps.** At ~0.9–1.1 s per window, DeBERTa cannot fit more
  than ~2 windows in `GUARDRAILS_TIMEOUT=2 s`. Injection therefore scores head +
  tail (2 windows), which catches "append the injection at the end" at 2× the
  original's cost; toxicity gets 8. When a cap applies, the service says
  `truncated: true` and the proxy logs it. **Residual risk: the middle of very
  long inputs is not scored by the injection model.** The real fix is a faster
  model (§3.5).

### 3.4 Concurrency collapse in the model server (B3 Q3, B3c)

**This is an issue in the current configuration.** The server runs 4 worker
threads, and PyTorch uses one intra-op thread per core by default. With four
concurrent 482-token requests, that is 16 OpenMP threads on 4 cores:

| intra-op threads | 1 request | 4 concurrent requests |
|---|---|---|
| default (= 4) **original** | 1,143 ms | **43,087 ms** (38× slower) |
| 1 | 1,603 ms | 1,811 ms |
| 2 | 1,003 ms | 1,873 ms |

Short ~120-token prompts degraded less, about 2× end to end (B6 baseline,
conc 1 → 4). Tool results are exactly the inputs that trigger the collapse.

I then compared gating forward passes per model (B3c, two interleaved runs
each):

| config | 1 request p50 | 8 concurrent rps | short request behind a 5-window one, p50 |
|---|---|---|---|
| 1 pass at a time, 4 threads | 2.8–14 s (erratic) | 0.20–0.43 | 6.5–17 s |
| 1 pass at a time, 2 threads | 1.3–1.4 s | 0.33–0.42 | 10–13 s |
| **2 concurrent passes × 2 threads (chosen)** | **1.0–1.5 s** | **0.79–1.48** | **1.5–3.1 s** |

* **Never give one model every core.** Any preemption stalls every OpenMP
  barrier. The default is `MODEL_CONCURRENCY=2` with
  `INTRA_OP_THREADS = CPUs ÷ 2`, read from the cgroup CPU quota.
* **Head-of-line blocking.** With a single slot, short requests waited
  6.5–17 s behind a 5-window request. `threading.Semaphore` also lets the
  releasing thread re-acquire immediately ("barging"). The gate is now FIFO
  with direct hand-off (unit-tested). At the chosen 2 slots, fair and unfair
  gates were indistinguishable end to end (§3.9).
* **A timeout death spiral (found in B6).** The server kept computing requests
  the proxy had already abandoned, and new requests queued behind them. The
  proxy now sends `deadline_ms`. The service drops expired work before each
  forward pass and while a request waits at the gate, and answers 504.

### 3.5 Injection model options (B3b)

| variant | params | 120 tokens | 482 tokens (~2 KB tool result) |
|---|---|---|---|
| DeBERTa-base PyTorch fp32 (**current**) | 184M | 324 ms | 868 ms |
| DeBERTa-base PyTorch dynamic int8 | 184M | 199 ms | 701 ms |
| DeBERTa-base ONNX fp32 | 184M | 195 ms | 805 ms |
| DeBERTa-base ONNX int8 | 184M | **98 ms** | 588 ms |
| xsmall-class PyTorch fp32 | 71M | 127 ms | 344 ms |
| xsmall-class ONNX int8 | 71M | **44 ms** | **272 ms** |

**Export fidelity.** The README says the DeBERTa ONNX export "scores wrong". On
identical weights, `torch.onnx.export` at opset 17 (torch 2.x, transformers
4.48) matched PyTorch to **max |Δlogit| = 4.1e-7 (base) and 7.3e-8 (xsmall)**
over sequence lengths 7–512. That validates the graph. int8 quantisation
changes the numbers by design, so the real model still needs checking with
`evals/run_eval.py`.

### 3.6 Incremental scanning machinery (B4, B6)

* **Cache key hash** (3 KB message): SHA-256 **4.2 µs**, BLAKE2b-128 8.0 µs,
  xxh3-128 0.57 µs. I planned BLAKE2b but switched: SHA-256 is ~2× faster here
  (hardware SHA extensions) and equally collision-resistant.
  * xxh3 is rejected because it's non-cryptographic. An attacker could craft
    text colliding with a cached benign entry and skip screening.
  * The saving would only be ~0.3 ms per turn even at turn 40.
* **LRU implementation**: OrderedDict 0.91 µs/op, `cachetools` 1.48, dict-FIFO
  0.54 (not a true LRU). OrderedDict, bounded by bytes and entry count.
* **Residual per-turn work** (hash every message in the history): 0.03 ms at
  turn 1, **0.85 ms at turn 40** (82 messages).
* **Design choices:**
  * **Content-addressed, not session-keyed.** A score is a function of the text
    only, so hits are safe across sessions.
  * **Only successful results are cached.** An outage is never remembered as
    clean (tested).
  * **Scores, not verdicts.** Threshold changes take effect immediately.
  * **Prompts carry the 256-character tail of the previous user message**, so
    payloads split across turns are caught (tested). Tool results carry none by
    default: the extra ~60 tokens pushed a 2 KB tool result into a second
    window, doubling its cost.
* **One request per target, in parallel.** Prompt and tool-result units used to
  share one request. A slow tool result then timed out the prompt check, which
  blocked the whole call (8/32 blocked in B6). Split, a slow tool result only
  quarantines itself (1/32 blocked).

### 3.7 Tool-call gating (B5)

| input | Python `re`, per pattern | Python `re`, per category | **RE2, per category (chosen)** |
|---|---|---|---|
| 60 B bash command | 0.037 ms | 0.029 ms | **0.017 ms** |
| 2 KB SQL | 1.37 ms | 1.07 ms | **0.054 ms** |
| 50 KB `write_file` | 34.3 ms | 33.3 ms | **0.96 ms** |
| adversarial `"rm " + "-a "*20000` | 88 ms | 90 ms | **0.86 ms** |
| adversarial `"curl " * 20000` (100 KB) | **8,214 ms** | **8,772 ms** | **1.25 ms** |

* **The deny rules were a denial-of-service vector under Python's `re`.** A
  model-emitted tool argument, for example one induced by an injection, could
  freeze the proxy's event loop for 8 seconds. RE2 runs in linear time. Its
  absence falls back to `re` with a warning.
* **End-to-end `check_tool_call`** (parse, flatten, canonicalise, match): 50 KB
  went from 36.2 to **2.95 ms**, and the adversarial input from 8.2 s to
  **5.3 ms**.
* **The same adversarial testing found an O(n²) bug in my own PII window
  snapping** (100 KB of `@` hung). It now merges windows first and snaps via
  bisect, which is linear, and a 6,000-case equivalence test runs against the
  original algorithm.

**JSON.** Agent-history bodies (`json` vs `orjson`, p50):

| body | `json` loads | `json` dumps | `orjson` loads | `orjson` dumps |
|---|---|---|---|---|
| 31 KB | 0.059 ms | 0.119 ms | 0.028 ms | 0.005 ms |
| 130 KB | 0.235 ms | 0.441 ms | 0.135 ms | 0.020 ms |

The total saving is under 0.5 ms even at 130 KB. That isn't worth a new
dependency, so I kept stdlib `json`.

### 3.8 Correctness / safety fixes (all regression-tested)

| issue in original code | effect | fix |
|---|---|---|
| `content` as a list of parts raised `TypeError` inside the hook | mitmproxy logs it and **forwards the request unscreened** | `messages.py` handles every content shape |
| any other exception in a hook | forwarded unscreened | catch-all: **block** (request) / **withhold** (response) |
| `stream: true` responses failed `json.loads` | **streamed replies and tool calls never screened** | SSE aggregate, screen, re-render (clean streams pass byte-for-byte) |
| all user messages joined, then truncated at 512 tokens | newest turns in long chats never scored by ML | per-message units |
| `role: tool` never screened | indirect injection unscreened | injection + PII, quarantine on hit |
| model tool calls never inspected | no action-layer guardrail | `tool_policy.py` |
| path compared exactly (`?query` bypassed) | unscreened | path normalised |
| blocked requests also ran the response hook | a wasted classifier call | skipped for locally-answered flows |
| streamed request blocked with a JSON body | SDK stream parser breaks | block message sent as SSE |

Plus one bug of mine, caught in review before any run. A cached
**regex-hit** record skipped the regex stage, and would have *allowed* a
repeated keyword-blocked message. It now blocks (tested).

### 3.9 Fairness of the model gate (B3c-fair)

Same config (2 concurrent passes × 2 threads), FIFO hand-off gate vs
`threading.Semaphore`, two interleaved rounds each. The metric is the latency
of 6 short prompts submitted just behind a 5-window request. It is normalised
by the same round's uncontended single-request p50, because the VM drifted
1.4 → 2.3 s between rounds.

| gate | round A | round B |
|---|---|---|
| FIFO hand-off | 2.1× | 4.9× |
| `threading.Semaphore` | 5.0× | 2.2× |

**Inconclusive: no measurable difference.** With two slots, the short
requests mostly wait on each other for the free slot, so barging matters
little. The FIFO gate stays for two reasons. First, its bounded-wait ordering
is verified by unit test: an already-waiting request gets the next slot.
Second, it lets a queued request leave at its deadline. It is not claimed as a
speed-up.

## 4. Residual risks and open items

* **Injection model cost.** This is the binding constraint for agentic
  screening on CPU (§1, §3.5). Recommended: evaluate the int8 ONNX export of
  the current model, and an xsmall-class model such as Llama Prompt Guard 2
  22M, with `evals/run_eval.py` plus an indirect-injection dataset (tool
  results). Then raise `MAX_WINDOWS_INJECTION` if budget allows.
* **Window cap.** With the cap, the middle of very long tool results is not
  scored for injection. It is logged.
* **Dense-PII text** sees no speed-up; the cost is off-loop and cached.
* **Streaming is still buffered,** as before. True incremental streaming with
  tool-call buffering is step 7 of the design.
* **Tool-policy patterns are heuristic.** They were tuned to avoid obvious false
  positives (`rm -rf ./build` and `DELETE … WHERE` are allowed), but they need
  review against your agents' real tool traffic. Categories can be disabled
  individually.
* **The cache is per process** (mitmproxy is single-process). Horizontally
  scaled proxies would need a shared store (e.g. Redis, ~0.2 ms RTT vs µs
  in-process).
* **No accuracy numbers.** Stand-in weights are random, so thresholds, int8
  variants and the xsmall model are unvalidated for accuracy.

## 5. Reproduce

```bash
python bench/standins/make_spacy_pipelines.py /tmp/spacy-standins
python bench/standins/make_models.py /tmp/models-standin
python bench/standins/make_injection_onnx.py /tmp/models-standin
cd bench
python b1_transport.py; python b2_pii.py; python b3_windowing.py
python b3b_injection_models.py; python b3c_model_concurrency.py
python b4_cache.py; python b5_tool_policy.py
python b6_e2e.py --mode stub --turns 10 --sessions 8 --conc 1,8
python b6_e2e.py --mode standin --turns 8 --sessions 4 --conc 1,4 \
    --variants baseline,new,new_onnx_base,new_onnx_xsmall,new_onnx_xsmall_nocache
python report_tables.py
```
