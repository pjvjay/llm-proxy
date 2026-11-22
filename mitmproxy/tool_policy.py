"""Tier-0 gating of tool calls the model wants the agent to execute.

A prompt injection that *succeeds* shows up here as an unexpected or dangerous
tool call, which is cheap to detect deterministically. This module is the
deterministic part; a judge ("does this action match the user's goal?") comes
later as a tier-1 evaluator.

What is checked, for every `tool_calls[].function` in a response:
  1. tool name against an allow / deny list
  2. total argument size
  3. every string leaf of the (parsed) arguments against built-in danger rules:
     destructive shell, remote code execution, destructive SQL, sensitive paths /
     traversal, SSRF targets (link-local metadata, loopback, private ranges),
     secrets in arguments (= exfiltration of the user's / tool's credentials)
  4. optional URL domain allow-list

Configuration: env vars for the simple knobs, and an optional JSON file
(TOOL_POLICY_FILE) for per-tool rules:

    {"allow": ["search", "read_file"],           # empty/absent => every tool allowed
     "deny":  ["shell"],
     "url_allow_domains": ["example.com", "api.github.com"],
     "max_arg_chars": 20000,
     "disable_rules": ["sql_destructive"],
     "rules": [{"name": "no_prod", "pattern": "prod-db", "tools": ["sql"], "triggers": ["prod"]}]}

All regexes are compiled once at import. Each built-in rule carries a set of
literal trigger substrings; the regex only runs if one is present, which keeps
large benign arguments (a 100 KB write_file) at well under a millisecond instead
of ~0.5 ms per KB (bench/bench_tool_policy.py). Arguments above
TOOL_MAX_ARG_CHARS (default 256 KB) are rejected outright."""
import ipaddress
import json
import os
import re
from dataclasses import dataclass
from urllib.parse import urlparse

_POLICY_FILE = os.getenv("TOOL_POLICY_FILE")


@dataclass(frozen=True)
class Rule:
    name: str
    pattern: re.Pattern
    tools: tuple[str, ...] = ()          # empty => all tools; else substring match on tool name
    triggers: tuple[str, ...] = ()       # lowercase literals; the regex only runs if one is present


@dataclass
class ToolVerdict:
    allowed: bool
    reason: str = ""
    rule: str = ""


_BUILTIN: list[Rule] = [
    Rule("shell_destructive", re.compile(
        r"(\brm\s+(-[a-z]*r[a-z]*f|-[a-z]*f[a-z]*r)\b[^\n;|&]*\s(/|~|\*|\$HOME|/home|/etc|/var|/usr)"
        r"|\bmkfs(\.\w+)?\b|\bdd\s+if=|:\(\)\s*\{\s*:\|:&\s*\};:|>\s*/dev/(sd|nvme|hd)"
        r"|\b(shutdown|reboot|halt|poweroff)\b|\bchmod\s+(-R\s+)?777\s+/|\bkill\s+-9\s+-1\b)", re.I),
        triggers=("rm ", "rm\t", "mkfs", "dd ", ":(){", "/dev/", "shutdown", "reboot", "halt",
                  "poweroff", "chmod", "kill ")),
    Rule("remote_exec", re.compile(
        r"((curl|wget)\b[^\n|]*\|\s*(sudo\s+)?(ba|z|da)?sh\b"
        r"|\b(ba)?sh\s+-c\s+[\"']?\$\((curl|wget)"
        r"|\bpython[23]?\s+-c\s+[\"'][^\"']*(urllib|requests|socket)[^\"']*exec"
        r"|\beval\s*\(\s*(base64|atob)\b)", re.I),
        triggers=("curl", "wget", "python", "eval")),
    Rule("sql_destructive", re.compile(
        r"\b(DROP\s+(TABLE|DATABASE|SCHEMA)|TRUNCATE\s+TABLE|DELETE\s+FROM\s+\w+\s*;?\s*$"
        r"|ALTER\s+(TABLE|DATABASE)|GRANT\s+ALL)\b", re.I | re.M),
        tools=("sql", "query", "db", "database"),
        triggers=("drop", "truncate", "delete", "alter", "grant")),
    Rule("sensitive_path", re.compile(
        r"(\.\./){2,}|/etc/(passwd|shadow|sudoers)|~?/\.ssh/|\bid_(rsa|ed25519)\b"
        r"|/\.aws/credentials|/\.env\b|\.git/config|/proc/self/environ", re.I),
        triggers=("../", "/etc/", ".ssh", "id_rsa", "id_ed25519", ".aws", ".env", ".git/", "/proc/")),
    Rule("secret_in_args", re.compile(
        r"\bsk-[A-Za-z0-9_\-]{20,}|\bAKIA[0-9A-Z]{16}\b|\bgh[pousr]_[A-Za-z0-9]{36,}"
        r"|(?i:bearer)\s+[A-Za-z0-9._\-]{20,}|-----BEGIN (RSA |EC |OPENSSH )?PRIVATE KEY-----"),
        triggers=("sk-", "akia", "ghp_", "gho_", "ghu_", "ghs_", "ghr_", "bearer", "private key")),
]

_URL_RX = re.compile(r"https?://[^\s\"'<>]+", re.I)
_SSRF_HOSTS = {"localhost", "metadata.google.internal", "metadata"}


def _ssrf_target(url: str) -> bool:
    try:
        host = (urlparse(url).hostname or "").strip("[]").lower()
    except ValueError:
        return True
    if not host:
        return False
    if host in _SSRF_HOSTS or host.endswith(".internal") or host.endswith(".local"):
        return True
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        return False
    return ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved


class ToolPolicy:
    def __init__(self, config: dict | None = None):
        cfg = dict(config or {})
        env_allow = os.getenv("TOOL_ALLOWLIST", "")
        env_deny = os.getenv("TOOL_DENYLIST", "")
        self.allow = set(cfg.get("allow") or [t for t in env_allow.split(",") if t.strip()])
        self.deny = set(cfg.get("deny") or [t for t in env_deny.split(",") if t.strip()])
        self.url_allow = {d.lower() for d in cfg.get("url_allow_domains") or
                          [d for d in os.getenv("TOOL_URL_ALLOW_DOMAINS", "").split(",") if d.strip()]}
        self.max_arg_chars = int(cfg.get("max_arg_chars") or os.getenv("TOOL_MAX_ARG_CHARS", "262144"))
        disabled = set(cfg.get("disable_rules") or [])
        self.rules = [r for r in _BUILTIN if r.name not in disabled]
        for r in cfg.get("rules") or []:
            self.rules.append(Rule(r["name"], re.compile(r["pattern"], re.I), tuple(r.get("tools") or ()),
                                   tuple(t.lower() for t in r.get("triggers") or ())))
        self.ssrf_check = (cfg.get("ssrf_check", os.getenv("TOOL_SSRF_CHECK", "1")) in (True, "1", "true"))

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _leaves(obj, out: list[str]):
        if isinstance(obj, str):
            out.append(obj)
        elif isinstance(obj, dict):
            for k, v in obj.items():
                out.append(str(k))
                ToolPolicy._leaves(v, out)
        elif isinstance(obj, (list, tuple)):
            for v in obj:
                ToolPolicy._leaves(v, out)
        return out

    @staticmethod
    def parse_arguments(arguments) -> object:
        if isinstance(arguments, (dict, list)):
            return arguments
        if not isinstance(arguments, str):
            return ""
        try:
            return json.loads(arguments)
        except ValueError:
            return arguments              # malformed JSON: scan the raw string

    # ------------------------------------------------------------------ verdict
    def evaluate(self, name: str, arguments) -> ToolVerdict:
        name = name or ""
        if self.deny and name in self.deny:
            return ToolVerdict(False, f"tool '{name}' is deny-listed", "deny")
        if self.allow and name not in self.allow:
            return ToolVerdict(False, f"tool '{name}' is not on the allow-list", "allow")

        raw_len = len(arguments) if isinstance(arguments, str) else len(json.dumps(arguments))
        if raw_len > self.max_arg_chars:
            return ToolVerdict(False, f"arguments too large ({raw_len} chars)", "max_arg_chars")

        parsed = self.parse_arguments(arguments)
        leaves = self._leaves(parsed, [])
        lname = name.lower()
        for leaf in leaves:
            low = leaf.lower()
            for rule in self.rules:
                if rule.tools and not any(t in lname for t in rule.tools):
                    continue
                if rule.triggers and not any(t in low for t in rule.triggers):
                    continue                       # literal prefilter: C-speed substring search
                if rule.pattern.search(leaf):
                    return ToolVerdict(False, f"argument matches rule '{rule.name}'", rule.name)
            if (self.ssrf_check or self.url_allow) and "http" in low:
                for url in _URL_RX.findall(leaf):
                    if self.ssrf_check and _ssrf_target(url):
                        return ToolVerdict(False, f"URL targets an internal address: {url[:80]}", "ssrf")
                    if self.url_allow:
                        host = (urlparse(url).hostname or "").lower()
                        if not any(host == d or host.endswith("." + d) for d in self.url_allow):
                            return ToolVerdict(False, f"URL domain not allowed: {host}", "url_allow")
        return ToolVerdict(True)


def load_default() -> ToolPolicy:
    cfg = None
    if _POLICY_FILE:
        with open(_POLICY_FILE) as f:
            cfg = json.load(f)
    return ToolPolicy(cfg)


DEFAULT = load_default()


def evaluate(name: str, arguments) -> ToolVerdict:
    return DEFAULT.evaluate(name, arguments)
