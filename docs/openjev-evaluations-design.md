# Design: OpenJev evaluations as low-latency guardrails for agentic traffic

Status: steps 1–3 implemented; the System-1 / System-2 mapping (§8) and tier-1
racing are implemented; OWASP LLM Top-10 (2025) mapping of every evaluation,
audit log and metrics, and the compose deployment (model-init, OpenJev slot in
`models/decision`) are implemented — see README. No OpenJev model exists yet to
load (the PyPI package is a placeholder); the slot is verified with a synthetic
multi-label model.

## 0. A note on OpenJev

At the time of writing, the `openjev` package on PyPI (0.0.1) is a placeholder
(`__version__` only, classifier "Development Status :: 1 - Planning") and the
GitHub repo was not reachable from the environment this was written in. The design
below therefore treats OpenJev as *"a library of evaluators that take
(input, output, context) and return a score / verdict / rationale"* and isolates
it behind one adapter (`evaluators/openjev_adapter.py`). Everything else — tiers,
session cache, streaming, tool-call gating, policy — is independent of OpenJev's
concrete API and is where the latency wins actually come from.

If OpenJev evaluators turn out to be LLM-as-judge style, they belong in tiers 1–2
below. If some are deterministic / small-model, they can also run in tier 0.

---

## 1. What the proxy does today, and where latency goes

`mitmproxy/firewall_addon.py` request path:

```
canonicalize → regex → HTTP → guardrails(injection ‖ toxicity) → Presidio (sync) → forward
```

Things that matter for an agentic caller (an agent loop making 10–50 calls per task):

| # | Observation | Effect |
|---|---|---|
| 1 | `classifier_client.classify()` creates a new `httpx.AsyncClient` per call | new TCP connection per request; no keep-alive; extra ms and p95 jitter |
| 2 | `pii.scan_and_redact()` is synchronous (spaCy NER under Presidio) and is called from the `async` hook — once to detect, then once more per user message to redact | **blocks the mitmproxy event loop**; serialises all concurrent flows; PII work is done twice |
| 3 | Stages run sequentially: classifiers *then* PII | request cost is `sum()` across stages, not `max()` |
| 4 | Every turn re-scans the *entire* user history (`_user_text` joins all user messages) | an N-turn agent loop does O(N²) classifier work; turn 30 scans 30 messages |
| 5 | Only `role == "user"` content is scanned; `role == "tool"` results and `system` are not | **tool results are the main indirect-injection vector in agentic systems and are currently unscreened** |
| 6 | Response `tool_calls` (function name + arguments) are never inspected | the *action* layer — the thing agents actually do — has no guardrail |
| 7 | `_user_text` assumes `content` is a `str`; multimodal / Responses-API style `content: [...]` raises `TypeError` inside the hook | mitmproxy logs the exception and **forwards the request unscreened (fail-open)** |
| 8 | No `stream: true` handling; mitmproxy buffers the full response before `response()` runs | streaming clients lose streaming; time-to-first-token = full generation time |
| 9 | Classifiers truncate at 512 tokens | injection placed at the *end* of a long tool result is silently cut off — a bypass, not just a latency concern |
| 10 | Only `/v1/chat/completions`; no `/v1/responses` | agent SDKs increasingly use the Responses API |
| 11 | Injection on PyTorch DeBERTa: 130 ms p50 / 279 ms p95 (already flagged in README) | dominates the inline budget |

Items 2, 3, 4 and 7 are cheap fixes that would improve latency and safety before
any evaluation framework is added. Items 5, 6, 8 are what "guardrails for
agentic calls" actually requires.

---

## 2. Core idea: separate *gates* from *evaluations*, and tier by latency budget

A guardrail that must block a request cannot cost more than a few tens of ms.
An LLM-judge evaluation that produces a rationale cannot be that fast. Trying to
make one mechanism do both is what produces either slow proxies or useless judges.

```
                 ┌──────────────────────────────────────────────────────────┐
 request ──────▶ │ Tier 0  INLINE GATE        budget ≤ 40 ms p95, fail-closed│
                 │  canonicalize · regex · ONNX classifiers · PII redact    │
                 │  tool-call allowlist · arg rules  (deterministic)        │
                 └──────────────┬───────────────────────────────────────────┘
                                │ cleared
                 ┌──────────────▼───────────────────────────────────────────┐
                 │ Tier 1  RACED EVALUATION   budget = upstream latency      │
                 │  forward to OpenAI immediately; run OpenJev judge         │
                 │  concurrently; if verdict=block arrives before the        │
                 │  upstream reply, discard the reply and return the block   │
                 └──────────────┬───────────────────────────────────────────┘
                                │ response (+ tool_calls) gated the same way
                 ┌──────────────▼───────────────────────────────────────────┐
                 │ Tier 2  SHADOW / TRAJECTORY  off the request path         │
                 │  full OpenJev rubric evals · trajectory evals · audit log │
                 │  feeds: session escalation, metrics, threshold calibration│
                 └──────────────────────────────────────────────────────────┘
```

### Tier 0 — inline gate
Everything that is already here, made concurrent and incremental (see §4). Small
models only. This is what the caller *feels*. OpenJev evaluators go here only if
they are deterministic or tiny-model backed.

### Tier 1 — raced evaluation ("speculative forwarding")
The key trick for agentic latency. The upstream LLM call is typically 500 ms – 10 s.
Any evaluator that finishes inside that window adds **zero** latency to the
call. The proxy forwards the (tier-0-cleared, PII-redacted) request immediately
and runs OpenJev judge evaluators in parallel. Outcomes:

- judge says *block* before upstream returns → drop upstream reply, return the
  standard blocked `chat.completion`;
- upstream returns first → the reply is held until the judge finishes **only if**
  policy says `hold: true` for that evaluator; otherwise the reply is released and
  the verdict is recorded (tier 2 semantics);
- judge times out → per-evaluator `fail: open|closed`.

Constraint: because the prompt has already left for the provider, anything that
exists to prevent *data leaving* (PII/secret redaction) must stay in tier 0.
Tier 1 is for *intent* and *behaviour* judgements — which is exactly what an
LLM-judge is good at and a 38M classifier is not.

### Tier 2 — shadow and trajectory
Runs after the response is released. This is where the richer OpenJev rubrics
live (helpfulness, groundedness, policy adherence, task-completion, tool-use
correctness) and where *agentic-process* evaluation happens: per-session
trajectory checks (loops, goal drift, privilege escalation across turns, budget).
Tier 2 results are not wasted just because they are late — they feed:

1. **Session escalation** — a session flagged by tier 2 gets stricter tier 0/1
   policy on its *next* call (e.g. `hold: true` on the judge, lower thresholds,
   tool allowlist narrowed). The agent loop gives us many chances to intervene.
2. **Audit log / metrics** — the README roadmap items.
3. **Calibration** — §6.

---

## 3. What to evaluate in an agentic call

| Target | Direction | Tier | Evaluators |
|---|---|---|---|
| new `user` messages | in | 0 (+1) | regex, injection, toxicity, PII; OpenJev intent judge |
| new `tool` results | in | 0 (+1) | injection (**windowed**, see §4.5), PII, OpenJev "does this tool output try to instruct the model?" |
| `system` prompt change mid-session | in | 0 | hash compare; flag |
| assistant text | out | 0 (+2) | toxicity, PII; OpenJev quality rubrics |
| `tool_calls` (name + parsed arguments) | out | 0 + 1 | tool allowlist per session/model; arg rules (shell, SQL writes, URLs, paths); OpenJev "is this action consistent with the user's stated goal and the tool results so far?" |
| trajectory (session) | — | 2 | loop detection, turn/token/cost budget, goal drift, OpenJev trajectory rubric |

Tool-call gating is the single most valuable addition for agents: a prompt
injection that *succeeds* manifests as an unexpected tool call, and that is
cheap to detect deterministically (unknown tool, `rm -rf`, `DROP`, external
URL, path traversal) and meaningful to judge (action ≠ intent).

---

## 4. Low-latency mechanics

### 4.1 Session identity + incremental scanning
Derive `session_id` from (in order): `X-Session-Id` header, OpenAI `user` field,
`metadata.session_id`, else `sha256(system_prompt + first user message)`.

Keep a per-session record `{message_hash → EvalResult}` (in-process LRU with
TTL; Redis if the proxy is scaled out). On each turn, hash every message
(`sha256(role + canonical content)`), and evaluate **only the messages not in
the cache**. This makes a 30-turn loop O(N) instead of O(N²) and means a normal
agent turn scans exactly one new user message or one new tool result.

### 4.2 Concurrency inside a turn
All tier-0 evaluators for a turn run under one `asyncio.gather` with a single
deadline; cost becomes `max()`, not `sum()`. Presidio moves off the event loop
(`asyncio.to_thread`, or into the guardrails service so it shares the thread
pool and the keep-alive connection). Redaction reuses the analyzer results from
detection rather than re-analysing.

### 4.3 Transport
One module-level `httpx.AsyncClient(http2=True)` with keep-alive to the
guardrails service (or a Unix socket, since both are on the same host). For
ONNX-only evaluators, an in-process option (`EVAL_INPROC=1`) removes the HTTP hop
entirely at the cost of coupling the addon to onnxruntime.

### 4.4 Cascade with an uncertainty band
Only call the expensive evaluator when the cheap one is unsure:

```
score = onnx_injection(text)
if score < 0.15: pass            # clear
elif score > 0.85: block         # clear
else: run OpenJev judge (tier 1) # ambiguous band only
```

On typical traffic this sends a small fraction of calls to the judge, which
means the judge can be *slower and better* without hurting p50.

### 4.5 Long inputs: window, don't truncate
Replace `truncation=True, max_length=512` with sliding windows (e.g. 384 tokens,
stride 256), batch all windows in one ONNX call, take the max score. Closes the
"append the injection after 512 tokens" bypass; batching keeps the cost close
to a single call for moderate lengths.

### 4.6 Streaming (`stream: true`)
Set `flow.response.stream = inspector` in `responseheaders()`. The inspector:

- passes text deltas through with a small lookback buffer, re-scoring at
  sentence boundaries (toxicity / PII on a sliding window); on a hit, it
  terminates the stream with a withheld-content chunk and `finish_reason: stop`;
- **buffers only `tool_calls` deltas** until the arguments JSON is complete,
  runs tool-call gating, then releases (or replaces) the final chunk.

This restores time-to-first-token for agents while still gating actions.

### 4.7 Judge execution
If OpenJev judges are LLM-backed, run them against a local small model with
prefix caching (the rubric/system prompt is a shared prefix), a single-token
`Yes/No` output with `logprobs` (so the score is `P("Yes")` — the pattern already
in `guardian_addon.py`/`guardian_server.py`), `max_tokens=1`. Latency is then
prefill-only. On CPU this is still hundreds of ms to seconds → tier 1/2 only.
Tier 0 must never wait on a generative model.

### 4.8 Deadlines and fail modes, per evaluator
Every evaluator declares `budget_ms`, `tier`, `action`, `fail` in policy.
`asyncio.wait_for` per evaluator; the tier-0 gate's total deadline is enforced
separately. Exceptions inside the addon are caught and turned into policy
decisions (fixes the current fail-open on `TypeError`).

---

## 5. Proposed module layout

```
mitmproxy/
  firewall_addon.py        thin mitmproxy glue (request / responseheaders / response)
  pipeline.py              tier orchestration: gate(), race(), shadow()
  session.py               session id, message-hash cache, trajectory store
  streaming.py             SSE inspector, tool_call buffering
  evaluators/
    base.py                Evaluator protocol + EvalInput / EvalResult
    registry.py            builds evaluator list from policy.yaml
    classifier.py          injection / toxicity via guardrails service (windowed)
    pii.py                 Presidio, off-loop, single analyse + reuse
    tool_policy.py         deterministic tool-call rules
    openjev_adapter.py     wraps OpenJev evaluators behind Evaluator
policy.yaml                evaluators → targets / tier / budget / action / fail / hold
evals/
  datasets/tool_injection.jsonl   indirect injection via tool results
  datasets/tool_calls.jsonl       safe / unsafe tool calls with user intent
  run_eval.py                     + per-target, per-tier metrics and "added latency"
```

### 5.1 Evaluator protocol

```python
# evaluators/base.py
from dataclasses import dataclass, field
from typing import Literal, Protocol

Target = Literal["prompt", "tool_result", "response", "tool_call", "trajectory"]

@dataclass
class EvalInput:
    session_id: str
    turn: int
    target: Target
    text: str = ""                    # the new content for text targets
    tool_call: dict | None = None     # {"name": ..., "arguments": {...}} for tool_call
    messages: list[dict] = field(default_factory=list)   # full (redacted) history
    user_goal: str = ""               # first user message, for intent checks
    metadata: dict = field(default_factory=dict)

@dataclass
class EvalResult:
    evaluator: str
    score: float                      # 0..1, higher = more unsafe / worse
    verdict: Literal["pass", "flag", "block"]
    reason: str = ""
    latency_ms: float = 0.0
    evidence: dict = field(default_factory=dict)

class Evaluator(Protocol):
    name: str
    targets: tuple[Target, ...]
    async def evaluate(self, inp: EvalInput) -> EvalResult: ...
```

### 5.2 OpenJev adapter (shape only — to be filled in against the real API)

```python
# evaluators/openjev_adapter.py
class OpenJevEvaluator:
    """Adapts one OpenJev evaluator to the proxy's Evaluator protocol.

    Assumptions to confirm against OpenJev's real API:
      - an evaluator object can be constructed once (rubric / model loaded at startup)
      - it exposes a call like  evaluate(input=..., output=..., context=...) -> result
      - result carries a numeric score and/or a pass/fail plus an optional rationale
      - calls are either async or safe to run in a worker thread
    """
    def __init__(self, name, jev_evaluator, targets, threshold=0.5, budget_ms=800):
        self.name, self._jev, self.targets = name, jev_evaluator, targets
        self._threshold, self._budget = threshold, budget_ms / 1000

    async def evaluate(self, inp: EvalInput) -> EvalResult:
        t0 = time.perf_counter()
        payload = _to_openjev_payload(inp)            # map EvalInput -> OpenJev's input shape
        raw = await asyncio.wait_for(
            _maybe_async(self._jev.evaluate, **payload), timeout=self._budget)
        score = _score_from(raw)                      # map OpenJev result -> 0..1
        return EvalResult(self.name, score,
                          "block" if score >= self._threshold else "pass",
                          reason=_rationale_from(raw),
                          latency_ms=(time.perf_counter() - t0) * 1000)
```

### 5.3 Policy (replaces the `DEFAULT` dict in `policy.py`; same `decide()` contract)

```yaml
session:
  id_from: [header:X-Session-Id, body:user, body:metadata.session_id, derive]
  cache_ttl_s: 1800

tiers:
  gate:  { budget_ms: 40 }          # tier 0 total deadline
  race:  { hold_default: false }    # tier 1: release upstream reply if judge is late

evaluators:
  - name: regex            tier: gate  targets: [prompt, tool_result]        action: block  fail: closed
  - name: injection_onnx   tier: gate  targets: [prompt, tool_result]        action: block  fail: closed
                           threshold: 0.5  uncertain_band: [0.15, 0.85]   # escalate to judge
  - name: toxicity_onnx    tier: gate  targets: [prompt, response]           action: block  fail: open
  - name: pii              tier: gate  targets: [prompt, tool_result, response]  action: redact  fail: closed
  - name: tool_rules       tier: gate  targets: [tool_call]                  action: block  fail: closed
                           allow: [search, read_file, http_get]
                           deny_args: { shell: ["rm -rf", "curl .* \\| sh"], sql: ["DROP", "DELETE"] }

  - name: jev_injection_judge  tier: race  targets: [prompt, tool_result]  budget_ms: 800
                               action: block  fail: open  hold: false
                               only_if: injection_onnx.uncertain
  - name: jev_action_intent    tier: race  targets: [tool_call]            budget_ms: 800
                               action: block  fail: closed  hold: true     # always gate actions

  - name: jev_quality          tier: shadow targets: [response]
  - name: jev_trajectory       tier: shadow targets: [trajectory]  every_n_turns: 5
                               on_flag: { escalate: { jev_injection_judge.hold: true, tool_rules.allow: [search] } }
```

### 5.4 Tier-1 race, in the addon

```python
async def request(flow):
    ctx = session.context(flow)                     # id, new messages, turn
    gate = await pipeline.gate(ctx)                 # tier 0, ≤ 40 ms
    if gate.blocked:
        flow.response = blocked(gate.reason); return
    flow.request.set_text(gate.redacted_body)
    # start judges now; they run while OpenAI is generating
    ctx.race = asyncio.create_task(pipeline.race(ctx))

async def response(flow):
    ctx = session.context(flow)
    verdict = await pipeline.settle(ctx.race, hold=ctx.policy.hold)  # returns immediately if not hold
    if verdict.blocked:
        flow.response = blocked(verdict.reason); return
    out = await pipeline.gate_response(ctx, flow.response)   # tool_calls + text, tier 0
    ...
    asyncio.create_task(pipeline.shadow(ctx))      # tier 2, never awaited on the path
```

---

## 6. Use OpenJev *offline* to make the inline gate better

This is the part that turns "we added evaluations" into "the guardrails got
faster and more accurate":

1. Run OpenJev judges in **shadow mode** (tier 2) on real traffic.
2. Log disagreements between the tier-0 classifiers and the judge.
3. Use the judge's labels to (a) re-tune thresholds and the uncertainty band in
   `evals/run_eval.py`, and (b) build a fine-tuning set for the small ONNX
   models on *your* traffic distribution (tool outputs, your tool schemas).
4. Re-measure. The judge's quality migrates into the 30 ms model; the judge is
   invoked less often.

Extend `evals/run_eval.py` to report, per target and per tier:

- precision / recall / FPR / F1 (as today)
- evaluator latency (as today) **and** *added call latency*, which for tier 1 is
  `max(0, judge_ms − upstream_ms)` — the number that actually matters
- time-to-first-token delta under `stream: true`
- judge invocation rate (fraction of calls that hit the uncertain band)

Datasets to add: `tool_injection.jsonl` (instructions hidden in HTML / JSON /
file contents returned by tools, including ones past the 512-token mark) and
`tool_calls.jsonl` (user goal + proposed tool call + label).

---

## 7. Order of work and status

1. **Hygiene — DONE.** Persistent httpx client (the per-call client cost ~53 ms of
   SSL-context construction); Presidio once, in a process pool (thread pool
   measured worse), concurrent with the classifiers; list-valued `content`;
   catch-all → fail-closed; windowed scoring; guardrails threading fixed;
   `stream: true` de-streamed and re-emitted as SSE so it is no longer
   fail-open. Measurements and decisions: `bench/README.md`.
2. **Session + incremental scanning — DONE** (`session.py`, `content.py`).
3. **Tool-result scanning and tool-call gating — DONE** (`tool_policy.py`, addon
   request/response paths; datasets `tool_injection.jsonl`, `tool_calls.jsonl`;
   `evals/run_eval.py` stages). Verify the HF `return_overflowing_tokens` path
   against the real DeBERTa / Granite tokenizers when models are available.
4. **Evaluator protocol + policy.yaml + registry**: refactor existing checks
   onto it; `decide()` keeps its signature so `tests/test_policy.py` still passes.
5. **OpenJev adapter in shadow mode** (tier 2) → collect disagreements.
6. **Tier-1 race** for the uncertainty-band judge and for tool-call intent.
7. **Streaming inspector.**
8. **`/v1/responses` support** (same pipeline, different message shape).

Steps 1–3 are in. Step 6 (tier-1 race) is in as part of §8. Step 5 is reframed by
§8: Jev is a System-1 decision model, so it plugs in at tier 0 through
`/v1/decide`, not as a shadow-mode judge. Remaining: 4 (policy.yaml), 7, 8.

## 8. Jev as a System-1 decision model

Jev is described as a *System One* decision model — in the dual-process sense:
fast, automatic, single-pass, returning a decision with a confidence rather than
deliberating. That changes where it sits relative to §2:

| §2 tier | dual-process role | implementation |
|---|---|---|
| tier 0 gate | **System 1** — every new unit, one pass, all labels | `/v1/decide` (`guardrails/server.py`), `_MultiLabelOnnxBackend`; proxy applies per-role confidence bands (`policy.BANDS`) → allow / block / uncertain |
| tier 1 race | **System 2** — uncertain units only | `mitmproxy/system2.py`: LLM judge (Granite Guardian / vLLM, P(Yes) from logprobs), `S2_MODE=inline\|race`, verdict cache, load shedding, deadline |
| tier 2 shadow | System 2 → System 1 learning | `ESCALATION_LOG` + `evals/calibrate_bands.py` (System 2 as teacher narrows System 1's bands) |

Properties exploited, and where each shows in the controlled benchmarks
(`bench/controlled/REPORT.md`):

1. **One pass, all labels** — forward passes −50% for user messages (two
   labels), −35% on the W1 mix. Independent of model speed.
2. **Calibrated confidence → escalate only the uncertain** — 30 of 200
   prompts judged; errors 14 → 3 (judge modelled as wrong 1 in 10).
3. **Context-conditioned** — role / tool passed with every item;
   `context_template` in `labels.json`; per-role bands.
4. **System 2 is scarce** — raced: no request pays for the judge while it is
   faster than upstream (Sweep B); cached: each distinct content judged once;
   shed: a burst degrades to System 1 instead of queueing past the deadline (W3).
5. **Session risk** — after a block / withhold, the session's bands widen and
   side-effecting tool calls get a System-2 intent check (catches the
   goal-inconsistent `send_email` in W2 that no deterministic rule matches).
   Read-only tools are exempt — the controlled run showed judging every call
   added ~400 ms to every remaining turn.

Where a System-1 model does *not* help: roles needing one label. With a model
much slower per window than the dedicated 33 ms toxicity head, responses get
slower (Sweep A); route them to the dedicated head
(`DECISION_SINGLE_LABEL_ROLES=assistant`).

Still unknown without the real model: Jev's per-window cost and its calibration
on this traffic. Both plug straight into `bench/controlled/costs.py` and
`calibrate_bands.py` once available.

## 9. Deploying alongside latency-sensitive systems

If this proxy sits in front of an agentic/decision-support system that itself
runs in a colocated, low-latency environment (trading infrastructure being the
sharpest version of this), the System-1/System-2 split above is not optional
polish — it's the only way an LLM-based gate can avoid being the slowest thing
in the path. See `docs/infrastructure/colo-deployment.md` for the
hardware/network reference architecture this implies (per-venue pods, local
model inference instead of a public SaaS API in the hot path, and a latency
budget that separates what tuning can buy from what only model choice can).
