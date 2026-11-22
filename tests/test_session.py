import session
from session import LRU, digest, scan_units


def test_digest_is_length_prefixed():
    assert digest("ab", "c") != digest("a", "bc")
    assert len(digest("x")) == 16


def test_lru_byte_budget_and_recency():
    c = LRU(max_bytes=3 * (100 + 200), max_entries=100)
    for k in (b"a", b"b", b"c"):
        c.put(k, k, 100)
    c.get(b"a")                       # a becomes most recent
    c.put(b"d", b"d", 100)            # evicts b (least recent)
    assert c.get(b"b") is None and c.get(b"a") == b"a" and c.get(b"d") == b"d"
    assert c.bytes <= c.max_bytes


def test_lru_rejects_oversized():
    c = LRU(max_bytes=500)
    c.put(b"big", "x", 10_000)
    assert len(c) == 0


def test_units_roles_targets_and_context():
    msgs = [{"role": "system", "content": "sys"},
            {"role": "user", "content": "one"},
            {"role": "assistant", "content": "a"},
            {"role": "tool", "content": "t1"},
            {"role": "user", "content": "two"}]
    us = scan_units(msgs)
    assert [(u.index, u.target) for u in us] == [(1, "prompt"), (3, "tool_result"), (4, "prompt")]
    assert us[2].context.endswith("one") and us[1].context == ""
    assert us[2].scan_text == "one\ntwo"


def test_key_stable_across_turns_and_context_sensitive():
    m1 = [{"role": "user", "content": "hello"}]
    m2 = m1 + [{"role": "assistant", "content": "hi"}, {"role": "user", "content": "next"}]
    assert scan_units(m1)[0].key == scan_units(m2)[0].key       # history -> cache hit
    other = [{"role": "user", "content": "different"}, {"role": "user", "content": "next"}]
    assert scan_units(m2)[1].key != scan_units(other)[1].key    # same text, other context


def test_session_id_sources():
    assert session.session_id({"x-session-id": "abc"}, {}) == "abc"
    assert session.session_id({}, {"user": "u1"}) == "user:u1"
    a = session.session_id({}, {"messages": [{"role": "user", "content": "x"}]})
    b = session.session_id({}, {"messages": [{"role": "user", "content": "x"}, {"role": "user", "content": "y"}]})
    assert a.startswith("derived:") and b.startswith("derived:") and a != b
