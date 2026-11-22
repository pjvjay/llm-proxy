"""Windowed (prefiltered) PII analysis must find exactly the same spans as
full-text Presidio analysis. Needs presidio (no spaCy model download)."""
import random
import sys

sys.path.insert(0, "mitmproxy")
sys.path.insert(0, "bench")
import pii  # noqa: E402

EDGE = [
    "call +44 20 7946 0958 now", "call 020 7946 0958", "ring +49 30 123456",
    "Tel: 030 1234567", "(415) 555-0132", "415.555.0132", "415 555 0132 x123",
    "+1-800-555-0199 ext. 4", "+33 1 42 68 53 00", "+972-3-123-4567", "+91 98765 43210",
    "+55 11 91234-5678", "phone４１５５５５０１３２end", "１２３-４５-６７８９",
    "ssn 078-05-1120", "ssn 078 05 1120", "ssn 078.05.1120", "ssn 078051120",
    "cc 4111-1111-1111-1111", "cc 5500 0000 0000 0004", "cc 378282246310005",
    "ip 192.168.0.1", "ip 8.8.8.8", "v6 2001:db8:85a3::8a2e:370:7334", "v6 ::1",
    "v6 fe80::1ff:fe23:4567:890a", "v6 2001:0db8:0000:0000:0000:ff00:0042:8329",
    "x@y.io", "first.last+tag@sub.example.co.uk.", "<a href='mailto:z@q.com'>",
    "sk-abcdefghijklmnopqrstuvwxyz123456", "AKIAABCDEFGHIJKLMNOP", "Bearer abcdefghijklmnopqrstuvwxyz.123",
    "numbers 12 34 56 78 90 in a row", "1.2.3.4", "version 10.0.19045.3693",
    "ts 2024-06-01T12:34:56Z id 998877", "a" * 500 + " 415-555-0132 " + "b" * 500,
]
_PUNCT = " -.()/+x\u2010\u2212\uff0d~\t"


def _spans(rs):
    return {(r.entity_type, r.start, r.end) for r in rs}


def _full(text):
    return pii._analyzer.analyze(text=text, entities=pii.ENTITIES, language="en")


def _check(text):
    a, b = _spans(_full(text)), _spans(pii.analyze(text))
    assert a == b, f"mismatch for {text[:120]!r}: full={a} windowed={b}"
    return len(a)


def test_edge_cases():
    found = sum(_check(t) for t in EDGE)
    found += _check(" filler words ".join(EDGE))
    assert found > 40


def test_fuzz_numeric():
    rnd = random.Random(7)
    for _ in range(400):
        parts = []
        for _ in range(rnd.randint(1, 8)):
            parts.append("".join(rnd.choice("0123456789") for _ in range(rnd.randint(1, 5))))
            parts.append("".join(rnd.choice(_PUNCT) for _ in range(rnd.randint(0, 5))))
        _check("word " + "".join(parts) + " word")


def test_corpus_parity():
    import corpus
    for seed in range(60):
        _check(corpus.text(3000, seed=seed, pii_every=4))
        _check(corpus.tool_result(2000, seed=seed))


if __name__ == "__main__":
    test_edge_cases(); test_fuzz_numeric(); test_corpus_parity()
    print("test_pii_prefilter: OK")


def test_windows_linear_on_hostile_input():
    import time
    for t in ("@" * 100_000, "1xxxx" * 20_000, "1xxxxx" * 20_000, "a:b:c" * 20_000,
              ("word " * 20 + "a@b.co ") * 2000):
        t0 = time.perf_counter()
        pii._windows(t)
        assert time.perf_counter() - t0 < 1.0, t[:20]


def test_windows_snap_to_whitespace():
    t = "aaaa bbbb x@y.co cccc dddd"
    (s, e), = pii._windows(t, margin=2)
    assert t[s:e] == "bbbb x@y.co"          # '@' +/- 2 chars, snapped outward


def _windows_reference(text, margin):
    """The original char-by-char algorithm, kept as an oracle."""
    spans, n = [], len(text)
    for m in pii._CANDIDATE.finditer(text):
        s, e = max(0, m.start() - margin), min(n, m.end() + margin)
        while s > 0 and not text[s - 1].isspace():
            s -= 1
        while e < n and not text[e].isspace():
            e += 1
        if spans and s <= spans[-1][1]:
            spans[-1][1] = max(spans[-1][1], e)
        else:
            spans.append([s, e])
    return spans


def test_windows_match_reference_algorithm():
    rnd = random.Random(3)
    alphabet = "ab @:1.-\n\t" + "0123456789"
    for _ in range(2000):
        t = "".join(rnd.choice(alphabet) for _ in range(rnd.randint(0, 120)))
        for margin in (0, 3, 17):
            assert pii._windows(t, margin) == _windows_reference(t, margin), (t, margin)
