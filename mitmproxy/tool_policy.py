"""Deterministic tool-call gating (response path).

When a prompt injection *succeeds* against an agent, it shows up as a tool call:
an undeclared tool, a destructive shell command, a credential read, a secret in
an outbound argument. These checks are cheap (microseconds; see
docs/perf-report.md, B5) and run on every tool call the model returns, before
the agent can execute it.

Checks, in order:
  1. structure   -- tool must be declared in the request's ``tools``; optional
                    allow/deny lists (TOOL_ALLOWLIST / TOOL_DENYLIST)
  2. content     -- argument strings (recursively extracted from the JSON,
                    canonicalized: NFKC, zero-width stripped, base64 decoded)
                    are matched against deny-pattern categories
  3. egress      -- optional URL host allowlist (TOOL_URL_ALLOWLIST)

Limits: this is a deterministic first line. It catches the blatant and the
lightly obfuscated; it does not understand intent. Semantic "is this action
what the user asked for" judging belongs to the tier-1 evaluator (design doc,
section 2).
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from urllib.parse import urlsplit

from canonicalize import canonicalize

# --------------------------------------------------------------------- rules
# Each category is a list of patterns. They must stay RE2-compatible (no
# lookaround / backreferences): RE2 is the default backend -- see MATCHER.
RULES: dict[str, list[str]] = {
    "shell_destructive": [
        r"(?i)\brm\s+(?:-\S+\s+)*-[a-z]*(?:r[a-z]*f|f[a-z]*r)[a-z]*\s+(?:-\S+\s+)*"
        r"(?:/\*?|~/?|\$\{?HOME\}?/?|\*|\.\./?)(?:\s|$|[;&|])",
        r"(?i)--no-preserve-root",
        r"(?i)\bmkfs(?:\.[a-z0-9]+)?\s",
        r"(?i)\bdd\s[^\n]*\bof=/dev/(?:sd|nvme|hd|xvd|disk|mmcblk)",
        r"(?i)>\s*/dev/(?:sd|nvme|hd|xvd)[a-z]",
        r":\(\)\s*\{\s*:\s*\|\s*:\s*&\s*\}\s*;\s*:",
        r"(?i)\bchmod\s+(?:-R\s+)?0?777\s+/(?:\s|$)",
        r"(?i)\b(?:shutdown|reboot|poweroff)\s+(?:-[a-z]+|now)\b",
    ],
    "remote_exec": [
        r"(?i)\b(?:curl|wget)\b[^\n|;]*\|\s*(?:sudo\s+)?(?:ba|z|da|k)?sh\b",
        r"(?i)\b(?:ba|z)?sh\s+-c\s+[\"']?\$\(\s*(?:curl|wget)\b",
        r"(?i)/dev/tcp/[0-9a-z.\-]+/\d+",
        r"(?i)\bnc(?:at)?\s[^\n]*\s-[a-z]*e\s",
        r"(?i)\bpowershell(?:\.exe)?\s[^\n]*-e(?:nc(?:odedcommand)?)?\s",
        r"(?i)\b(?:iex|invoke-expression)\b",
        r"(?i)\bpython[0-9.]*\s+-c\s+[\"'][^\"'\n]*\b(?:socket|subprocess|os\.system|pty\.spawn)\b",
    ],
    "sql_destructive": [
        r"(?i)\bdrop\s+(?:table|database|schema|view|user|index)\b",
        r"(?i)\btruncate\s+(?:table\s+)?[\w.\"`\[]",
        r"(?i)\bdelete\s+from\s+[\w.\"`\[\]]+\s*(?:;|$)",
        r"(?i)\balter\s+table\s[^;\n]*\bdrop\b",
        r"(?i)\bgrant\s+all\b",
    ],
    "credential_access": [
        r"(?i)\.ssh/(?:id_[a-z0-9]+|authorized_keys|known_hosts)\b",
        r"(?i)/etc/(?:shadow|sudoers|gshadow)\b",
        r"(?i)\.aws/credentials\b",
        r"(?i)\.kube/config\b",
        r"(?i)\.(?:netrc|git-credentials|pgpass)\b",
        r"(?i)\.docker/config\.json\b",
        r"(?i)/proc/(?:self|\d+)/environ\b",
        r"(?i)(?:^|[\s/\"'=])\.env(?:\.[a-z]+)?(?:$|[\s\"'])",
        r"(?i)\bprintenv\b",
    ],
    "path_traversal": [
        r"(?:\.\.[/\\]){2,}",
        r"(?i)(?:%2e%2e(?:%2f|%5c|/|\\)){2,}",
    ],
    "secret_in_args": [
        r"sk-[A-Za-z0-9_\-]{20,}",
        r"AKIA[0-9A-Z]{16}",
        r"-----BEGIN (?:[A-Z]+ )?PRIVATE KEY-----",
        r"\bgh[pousr]_[A-Za-z0-9]{36}\b",
        r"\bxox[abprs]-[A-Za-z0-9\-]{10,}",
    ],
}

_DISABLED = {c.strip() for c in os.getenv("TOOL_DISABLE_CATEGORIES", "").split(",") if c.strip()}
_ALLOW = {t.strip() for t in os.getenv("TOOL_ALLOWLIST", "").split(",") if t.strip()}
_DENY = {t.strip() for t in os.getenv("TOOL_DENYLIST", "").split(",") if t.strip()}
_URL_ALLOW = [h.strip().lower() for h in os.getenv("TOOL_URL_ALLOWLIST", "").split(",") if h.strip()]
_REQUIRE_DECLARED = os.getenv("TOOL_REQUIRE_DECLARED", "1") not in ("0", "false", "no")
# RE2 (linear time) by default: Python's backtracking `re` took 8.2 s on a
# 100 KB adversarial argument ("curl " * 20000) -- a model-emitted tool call
# could stall the proxy's event loop. RE2: 1.3 ms. Also ~35x faster on normal
# 50 KB arguments (B5). Falls back to Python `re` if google-re2 is missing.
MATCHER = os.getenv("TOOL_MATCHER", "re2")
_URL = re.compile(r"https?://[^\s\"'<>\\]+", re.I)

# Field separator for flattened arguments: '\n' stops [^\n] runs and the NUL
# stops \s runs, so no pattern can match across two different argument values.
SEP = "\n\x00\n"


# ------------------------------------------------------------------ matchers
def _scoped(p: str) -> str:
    """'(?i)x' -> '(?i:x)' so patterns can be OR-combined (global flags can't)."""
    return f"(?i:{p[4:]})" if p.startswith("(?i)") else f"(?:{p})"


class LoopMatcher:
    """One compiled regex per pattern, searched in turn (like regex_rules.py)."""
    def __init__(self, rules, engine=re):
        self._rx = [(c, engine.compile(_scoped(p))) for c, ps in rules.items() for p in ps]

    def categories(self, text: str) -> list[str]:
        hit = []
        for c, rx in self._rx:
            if c not in hit and rx.search(text):
                hit.append(c)
        return hit


class CombinedMatcher:
    """One regex per category (patterns OR-ed): a single scan per category."""
    def __init__(self, rules, engine=re):
        self._rx = [(c, engine.compile("|".join(_scoped(p) for p in ps))) for c, ps in rules.items()]

    def categories(self, text: str) -> list[str]:
        return [c for c, rx in self._rx if rx.search(text)]


def build_matcher(kind: str = MATCHER, rules=None):
    rules = {c: ps for c, ps in (rules or RULES).items() if c not in _DISABLED}
    if kind == "loop":
        return LoopMatcher(rules)
    if kind == "combined":
        return CombinedMatcher(rules)
    if kind in ("re2", "re2_loop"):
        try:
            import re2  # google-re2: linear-time, immune to catastrophic backtracking
        except ImportError:
            import logging
            logging.getLogger("firewall").warning(
                "google-re2 not installed; tool_policy falls back to Python re "
                "(vulnerable to catastrophic backtracking on hostile input)")
            return CombinedMatcher(rules) if kind == "re2" else LoopMatcher(rules)
        opts = re2.Options()
        opts.log_errors = False
        eng = _Re2Engine(opts)
        return CombinedMatcher(rules, eng) if kind == "re2" else LoopMatcher(rules, eng)
    raise ValueError(f"unknown TOOL_MATCHER {kind!r}")


class _Re2Engine:
    def __init__(self, opts):
        self._opts = opts

    def compile(self, pattern):
        import re2
        return re2.compile(pattern, options=self._opts)


_matcher = build_matcher()


# ---------------------------------------------------------------- extraction
def _strings(obj, out: list[str], depth: int = 0):
    if depth > 32:
        out.append(json.dumps(obj, ensure_ascii=False)[:100_000])
        return
    if isinstance(obj, str):
        out.append(obj)
    elif isinstance(obj, dict):
        for k, v in obj.items():
            out.append(str(k))
            _strings(v, out, depth + 1)
    elif isinstance(obj, list):
        for v in obj:
            _strings(v, out, depth + 1)


def argument_text(arguments) -> str:
    """Flatten tool-call arguments (JSON string or object) to canonical text."""
    if isinstance(arguments, str):
        try:
            obj = json.loads(arguments)
        except ValueError:
            obj = arguments                      # malformed JSON: scan it raw
    else:
        obj = arguments
    parts: list[str] = []
    _strings(obj, parts)
    canon, decoded = canonicalize(SEP.join(parts))
    return SEP.join([canon, *decoded])


# -------------------------------------------------------------------- checks
@dataclass
class ToolVerdict:
    name: str
    violations: list[str] = field(default_factory=list)

    @property
    def blocked(self) -> bool:
        return bool(self.violations)


def _host_allowed(host: str) -> bool:
    host = host.lower().rstrip(".")
    return any(host == a or host.endswith("." + a) for a in _URL_ALLOW)


def check_tool_call(name: str, arguments, declared: set[str] | None = None,
                    matcher=None) -> ToolVerdict:
    v = ToolVerdict(name=name or "")
    if _REQUIRE_DECLARED and declared is not None and name not in declared:
        v.violations.append("undeclared_tool")
    if _ALLOW and name not in _ALLOW:
        v.violations.append("tool_not_allowlisted")
    if name in _DENY:
        v.violations.append("tool_denylisted")
    text = argument_text(arguments)
    v.violations.extend(f"args:{c}" for c in (matcher or _matcher).categories(text))
    if _URL_ALLOW:
        for m in _URL.finditer(text):
            host = urlsplit(m.group(0)).hostname or ""
            if not _host_allowed(host):
                v.violations.append("egress_url_not_allowlisted")
                break
    return v


def tool_calls_of(message: dict) -> list[tuple[str, object]]:
    """(name, arguments) for every tool / function call in an assistant message."""
    out = []
    for tc in message.get("tool_calls") or []:
        fn = (tc or {}).get("function") or {}
        out.append((fn.get("name") or "", fn.get("arguments") or ""))
    fc = message.get("function_call")
    if isinstance(fc, dict):
        out.append((fc.get("name") or "", fc.get("arguments") or ""))
    return out
