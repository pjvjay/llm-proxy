"""Incremental scanning for agent loops.

An agent loop re-sends the whole conversation on every turn. Three strategies
for what the proxy scans on turn N:

  A. truncate   -- concatenate all user text, classifier truncates to 512 tokens
                   (old behaviour). O(1) per turn but BLIND to anything past the
                   first ~512 tokens of the conversation, and never looks at tool
                   results at all.
  B. rescan     -- concatenate all user + tool text, score every 512-token window
                   (what "fix the truncation bypass" does naively). O(N) windows
                   per turn, O(N^2) over the loop.
  C. incremental-- hash each message; score only windows of messages not seen
                   before (new session.py). O(1) windows per turn, O(N) total.

Cost model: per-window classifier cost from the README (int8 ONNX toxicity 33 ms,
PyTorch injection 130 ms) plus ~3 ms of transport per call. We count windows,
which is the hardware-independent number, and also simulate wall time with
asyncio.sleep so the shape is visible.

Run from repo root:  python bench/bench_session.py
"""
import asyncio, hashlib, os, sys, time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "mitmproxy"))
from session import VerdictCache, message_key  # noqa: E402

TURNS = 30
USER_TOKENS = 60          # a short user instruction per turn
TOOL_TOKENS = 900         # a typical tool result (web page / file / API payload)
WINDOW, STRIDE = 512, 384
PER_WINDOW_MS = {"toxicity int8 ONNX": 33, "injection PyTorch": 130}
TRANSPORT_MS = 3

def windows(tokens):
    if tokens <= WINDOW: return 1
    return 1 + -(-(tokens - WINDOW) // STRIDE)

def simulate(strategy):
    """Return list of (windows scored, tokens actually covered) per turn."""
    seen = set(); out = []
    history = []   # (role, tokens)
    for t in range(TURNS):
        history.append(("user" if t == 0 else "tool", USER_TOKENS if t == 0 else TOOL_TOKENS))
        if strategy == "A":
            total = sum(tok for role, tok in history if role == "user")
            out.append((1, min(total, WINDOW)))
        elif strategy == "B":
            total = sum(tok for _, tok in history)
            out.append((windows(total), total))
        else:
            new = [(i, tok) for i, (_, tok) in enumerate(history) if i not in seen]
            seen.update(i for i, _ in new)
            out.append((sum(windows(tok) for _, tok in new), sum(tok for _, tok in new)))
    return out

# ---------------------------------------------------------------- hashing
def bench_hash():
    msg = ("tool result " * 200).encode()          # ~2.4 KB
    print("### message hashing (2.4 KB message)\n")
    print("| hash | µs/msg |\n|---|---|")
    for name in ("md5", "sha1", "sha256", "blake2b", "blake2s"):
        h = getattr(hashlib, name)
        t0 = time.perf_counter()
        for _ in range(5000): h(msg).hexdigest()
        print(f"| {name} | {(time.perf_counter() - t0) / 5000 * 1e6:.1f} |")
    t0 = time.perf_counter()
    for _ in range(5000): hash(msg)
    print(f"| builtin hash() (not stable across processes) | {(time.perf_counter() - t0) / 5000 * 1e6:.1f} |")
    t0 = time.perf_counter()
    for _ in range(5000): message_key("tool", msg.decode())
    print(f"| session.message_key (sha256, role-prefixed) | {(time.perf_counter() - t0) / 5000 * 1e6:.1f} |")

def bench_cache():
    c = VerdictCache(maxsize=10000, ttl_s=600)
    keys = [message_key("user", f"message {i}") for i in range(20000)]
    t0 = time.perf_counter()
    for k in keys: c.put(k, {"score": 0.1})
    put = (time.perf_counter() - t0) / len(keys) * 1e6
    t0 = time.perf_counter()
    for k in keys: c.get(k)
    get = (time.perf_counter() - t0) / len(keys) * 1e6
    print(f"\nVerdictCache: put {put:.2f} µs, get {get:.2f} µs (10k entries, LRU+TTL, half misses)\n")

async def main():
    bench_hash(); bench_cache()
    print(f"### {TURNS}-turn agent loop, {TOOL_TOKENS}-token tool result per turn\n")
    print("| strategy | windows scored, whole loop | tokens covered turn 30 | classifier ms turn 30 (tox / inj) | classifier ms whole loop (tox / inj) |")
    print("|---|---|---|---|---|")
    for strat, label in (("A", "A truncate (old)"), ("B", "B rescan all, windowed"), ("C", "C incremental (new)")):
        sim = simulate(strat)
        total_w = sum(w for w, _ in sim)
        last_w, last_cov = sim[-1]
        def ms(w, per): return w * per + (TRANSPORT_MS if w else 0)
        t30 = " / ".join(f"{ms(last_w, p):.0f}" for p in PER_WINDOW_MS.values())
        loop = " / ".join(f"{sum(ms(w, p) for w, _ in sim):.0f}" for p in PER_WINDOW_MS.values())
        blind = "" if strat != "A" else " (**blind** to tool results and to turn ≥2)"
        print(f"| {label} | {total_w} | {last_cov}{blind} | {t30} | {loop} |")

if __name__ == "__main__":
    asyncio.run(main())
