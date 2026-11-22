"""Fit System-1 confidence bands from data -- the 'learn' half of the
System 1 / System 2 loop.

Input (JSONL, either shape per line):
  {"s1": 0.42, "label": 1}             labelled eval data scored by System 1
  {"s1": 0.42, "s2": 0.93, ...}        ESCALATION_LOG records: System 2 is the teacher
Optionally filter log records with --role / --label.

Finds (allow_below, block_at) that minimise the share of items escalated to
System 2, subject to: among items System 1 decides on its own (outside the band),
the error rate is at most --max-error. Exhaustive search over the observed
scores, so it is exact for the data given.

Caveat for escalation-log input: those items were all inside the CURRENT band,
so the fit can only narrow the band, never widen it -- which is the direction
that saves System-2 calls.

    python evals/calibrate_bands.py escalations.jsonl --role tool --label injection --max-error 0.02
"""
import argparse
import json
import sys


def fit_band(points, max_error=0.02):
    """points: [(s1, y)] with y in {0,1}. Returns dict with lo, hi, escalated, error."""
    pts = sorted(points)
    n = len(pts)
    if n == 0:
        raise ValueError("no data")
    cands = sorted({0.0, 1.0000001, *(p for p, _ in pts)})
    # prefix counts: positives / negatives strictly below each candidate
    best = None
    for i, lo in enumerate(cands):
        below = [(p, y) for p, y in pts if p < lo]
        fn = sum(y for _, y in below)                       # malicious we'd allow
        for hi in cands[i:]:
            above = [(p, y) for p, y in pts if p >= hi]
            fp = sum(1 - y for _, y in above)               # benign we'd block
            decided = len(below) + len(above)
            err = (fn + fp) / decided if decided else 0.0
            if err > max_error:
                continue
            esc = (n - decided) / n
            key = (esc, err, hi - lo)
            if best is None or key < best[0]:
                best = (key, dict(lo=lo, hi=min(hi, 1.0), escalated=esc, error=err,
                                  decided=decided, n=n, fn=fn, fp=fp))
    return best[1]


def load(path, role=None, label=None):
    pts = []
    for line in open(path):
        if not line.strip():
            continue
        r = json.loads(line)
        lab = r.get("label")                    # label NAME in escalation logs, 0/1 in eval data
        if role and r.get("role") is not None and r["role"] != role:
            continue
        if label and isinstance(lab, str) and lab != label:
            continue
        if r.get("s1") is None:
            continue
        if "s2" in r:
            if r["s2"] is None:                 # shed / timeout: no teacher signal
                continue
            pts.append((float(r["s1"]), int(r["s2"] >= 0.5)))
        elif isinstance(r.get("label"), int):
            pts.append((float(r["s1"]), int(r["label"])))
    return pts


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--role")
    ap.add_argument("--label")
    ap.add_argument("--max-error", type=float, default=0.02)
    a = ap.parse_args()
    pts = load(a.path, a.role, a.label)
    if not pts:
        sys.exit("no usable records")
    r = fit_band(pts, a.max_error)
    print(json.dumps(r, indent=2))
    role, lab = (a.role or "user").upper(), (a.label or "injection").upper()
    print(f"\nS1_BAND_{role}_{lab}={r['lo']:.3f},{r['hi']:.3f}")
