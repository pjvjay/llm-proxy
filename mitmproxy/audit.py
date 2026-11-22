"""Per-request evaluation record, audit log, and metrics.

Every check the firewall runs on a request -- System-1 label scores (allowed
or not), System-2 judgements, deterministic rules, limits -- is appended to
that request's Evaluation as a finding, tagged with OWASP LLM Top-10 ids.
At the end of the flow the Evaluation is:
  * summarised into response headers (X-LLM-Firewall-*),
  * embedded as a `firewall` object in any response the firewall produced or
    modified (or every response, with FIREWALL_ANNOTATE=1),
  * appended to AUDIT_LOG as one JSON line (if set), and
  * folded into counters served at /firewall/metrics (Prometheus text) and
    /firewall/owasp (JSON).
The audit write is a small buffered append; it never blocks on anything
remote."""
import json
import logging
import os
import time
import uuid
from collections import Counter

import owasp

log = logging.getLogger("firewall.audit")

AUDIT_LOG = os.getenv("AUDIT_LOG")                  # e.g. /var/log/firewall/audit.jsonl
AUDIT_TEXT = os.getenv("AUDIT_LOG_TEXT", "0") == "1"

COUNTS: Counter = Counter()          # (metric, labels-tuple) -> value
_STARTED = time.time()


class Evaluation:
    def __init__(self, session: str | None = None):
        self.id = uuid.uuid4().hex[:16]
        self.session = session
        self.started = time.time()
        self.findings: list[dict] = []
        self.action = "allow"            # allow | redact | withhold | block
        self.reason = ""
        self.recorded = False

    _RANK = {"allow": 0, "log": 0, "warn": 0, "clamp": 1, "redact": 1, "withhold": 2, "block": 3}

    def add(self, check: str, decision: str, *, role: str | None = None, target=None,
            score: float | None = None, s1: str | None = None, s2: float | None = None,
            owasp_ids=None, detail: str | None = None, cached: bool = False, tool: str | None = None,
            model_map: dict | None = None) -> dict:
        ids = owasp_ids if owasp_ids is not None else owasp.ids_for(check, role, tool, model_map)
        f = {"check": check, "decision": decision, "owasp": sorted(set(ids))}
        for k, v in (("role", role), ("target", target), ("tool", tool), ("s1", s1), ("detail", detail)):
            if v is not None:
                f[k] = v
        if score is not None:
            f["score"] = round(float(score), 4)
        if s2 is not None:
            f["s2"] = round(float(s2), 4)
        if cached:
            f["cached"] = True
        self.findings.append(f)
        return f

    def escalate(self, action: str, reason: str) -> None:
        if self._RANK.get(action, 0) >= self._RANK.get(self.action, 0):
            if self._RANK.get(action, 0) > self._RANK.get(self.action, 0) or not self.reason:
                self.reason = reason
            self.action = action

    def flagged(self) -> list[str]:
        ids = set()
        for f in self.findings:
            if f["decision"] not in ("allow", "log"):
                ids.update(f["owasp"])
        return sorted(ids)

    def evaluated(self) -> list[str]:
        ids = set()
        for f in self.findings:
            ids.update(f["owasp"])
        return sorted(ids)

    def summary(self) -> dict:
        return {"request_id": self.id, "action": self.action, "reason": self.reason,
                "owasp": owasp.describe(self.flagged()),
                "owasp_evaluated": self.evaluated(),
                "evaluations": self.findings}

    def headers(self) -> dict:
        return {"X-LLM-Firewall-Request-Id": self.id,
                "X-LLM-Firewall-Action": self.action,
                "X-LLM-Firewall-OWASP": ",".join(self.flagged()) or "none"}


def record(ev: Evaluation) -> None:
    if ev.recorded:
        return
    ev.recorded = True
    COUNTS[("requests_total", (("action", ev.action),))] += 1
    for f in ev.findings:
        COUNTS[("evaluations_total", (("check", f["check"]), ("decision", f["decision"])))] += 1
        if f["decision"] not in ("allow", "log"):
            for i in f["owasp"]:
                COUNTS[("owasp_flagged_total", (("category", i),))] += 1
        for i in f["owasp"]:
            COUNTS[("owasp_evaluated_total", (("category", i),))] += 1
    if AUDIT_LOG:
        line = {"ts": ev.started, "ms": round((time.time() - ev.started) * 1000, 2),
                "session": ev.session, **ev.summary()}
        if not AUDIT_TEXT:
            for f in line["evaluations"]:
                f.pop("text", None)
        try:
            with open(AUDIT_LOG, "a", buffering=1) as fh:
                fh.write(json.dumps(line) + "\n")
        except OSError as exc:
            log.warning("audit write failed: %s", exc)


def prometheus_text(extra: dict | None = None) -> str:
    lines = ["# TYPE llm_firewall_up gauge", "llm_firewall_up 1",
             "# TYPE llm_firewall_uptime_seconds gauge",
             f"llm_firewall_uptime_seconds {time.time() - _STARTED:.0f}"]
    seen = set()
    for (metric, labels), v in sorted(COUNTS.items()):
        name = f"llm_firewall_{metric}"
        if name not in seen:
            lines.append(f"# TYPE {name} counter")
            seen.add(name)
        lab = ",".join(f'{k}="{val}"' for k, val in labels)
        lines.append(f"{name}{{{lab}}} {v}")
    for k, v in (extra or {}).items():
        lines.append(f"llm_firewall_{k} {v}")
    return "\n".join(lines) + "\n"


def owasp_summary() -> dict:
    flagged = {i: 0 for i in owasp.CATALOG}
    evaluated = {i: 0 for i in owasp.CATALOG}
    for (metric, labels), v in COUNTS.items():
        if metric in ("owasp_flagged_total", "owasp_evaluated_total"):
            cat = dict(labels)["category"]
            (flagged if metric == "owasp_flagged_total" else evaluated)[cat] = v
    return {"categories": [{"id": i, "name": n, "status": owasp.COVERAGE[i][0],
                            "controls": owasp.COVERAGE[i][1],
                            "evaluated": evaluated[i], "flagged": flagged[i]}
                           for i, n in owasp.CATALOG.items()]}
