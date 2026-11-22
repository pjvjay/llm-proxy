"""Cost of deterministic tool-call gating (tier 0). Also compares the two
obvious regex strategies: a list of per-rule patterns (current, gives the rule
name for free) vs one combined alternation.

Run from repo root:  python bench/bench_tool_policy.py"""
import json, os, re, statistics, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "mitmproxy"))
from tool_policy import ToolPolicy, _BUILTIN  # noqa: E402

P = ToolPolicy()
CASES = {
    "small benign (search)": ("search", json.dumps({"query": "weather in Paris tomorrow"})),
    "shell benign": ("shell", json.dumps({"cmd": "git status && ls -la src/"})),
    "shell destructive (hit on rule 1)": ("shell", json.dumps({"cmd": "rm -rf / --no-preserve-root"})),
    "http with url (ssrf check)": ("http_get", json.dumps({"url": "https://api.github.com/repos/a/b", "headers": {"Accept": "application/json"}})),
    "nested 2 KB args, benign": ("batch", json.dumps({"ops": [{"cmd": f"echo {i} && cat notes/{i}.md"} for i in range(40)]})),
    "20 KB args, benign": ("write_file", json.dumps({"path": "out.txt", "content": "lorem ipsum dolor sit amet " * 740})),
    "100 KB args, benign": ("write_file", json.dumps({"path": "out.txt", "content": "lorem ipsum dolor sit amet " * 3700})),
    "100 KB args, 'rm ' present but benign": ("write_file", json.dumps({"path": "out.txt", "content": "perform rm on temp " * 5000})),
}
COMBINED = re.compile("|".join(f"(?P<{r.name}>{r.pattern.pattern})" for r in _BUILTIN), re.I | re.M)

def timeit(fn, n=300):
    for _ in range(50): fn()
    xs = []
    for _ in range(n):
        t0 = time.perf_counter(); fn(); xs.append((time.perf_counter() - t0) * 1e6)
    return statistics.median(xs), sorted(xs)[int(0.99 * n) - 1]

print("| case | evaluate() p50 µs | p99 µs | combined-regex p50 µs |\n|---|---|---|---|")
for label, (name, args) in CASES.items():
    p50, p99 = timeit(lambda: P.evaluate(name, args))
    parsed = ToolPolicy.parse_arguments(args); leaves = ToolPolicy._leaves(parsed, [])
    c50, _ = timeit(lambda: [COMBINED.search(l) for l in leaves])
    print(f"| {label} | {p50:.0f} | {p99:.0f} | {c50:.0f} (regex only, no parse) |")
