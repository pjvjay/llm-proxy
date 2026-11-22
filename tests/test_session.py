import sys, time
sys.path.insert(0, "mitmproxy")
import session as S

def test_message_key_is_role_scoped_and_stable():
    assert S.message_key("user", "hi") == S.message_key("user", "hi")
    assert S.message_key("user", "hi") != S.message_key("tool", "hi")

def test_cache_lru_and_ttl():
    c = S.VerdictCache(maxsize=2, ttl_s=0.05)
    c.put("a", {"x": 1}); c.put("b", {"x": 2}); c.put("c", {"x": 3})
    assert c.get("a") is None and c.get("c") == {"x": 3}        # LRU evicted "a"
    time.sleep(0.06)
    assert c.get("c") is None                                   # TTL expired

def test_plan_marks_seen_units():
    c = S.VerdictCache()
    msgs = [{"role": "user", "content": "turn 1"}]
    u1 = S.plan(c, msgs, {"user", "tool"})
    assert u1[0].cached is None
    c.put(u1[0].key, {"scores": {}})
    msgs += [{"role": "assistant", "content": "ok"}, {"role": "tool", "content": "result"}]
    u2 = S.plan(c, msgs, {"user", "tool"})
    assert [u.cached is not None for u in u2] == [True, False]   # only the tool result is new

def test_session_id_sources():
    assert S.session_id({"X-Session-Id": "abc"}, {})[0] == "abc"
    assert S.session_id({}, {"user": "u1"})[0] == "user:u1"
    assert S.session_id({}, {"metadata": {"session_id": 7}})[0] == "meta:7"
    sid, derived = S.session_id({}, {"messages": [{"role": "user", "content": "x"}]})
    assert derived and sid.startswith("derived:")

def test_session_store_turns_and_system_change():
    st = S.SessionStore()
    a = st.touch("s", "h1", False); b = st.touch("s", "h1", False); c = st.touch("s", "h2", False)
    assert (a.turn, b.turn, c.turn) == (1, 2, 3)
    assert not b.system_changed and c.system_changed

if __name__ == "__main__":
    test_message_key_is_role_scoped_and_stable(); test_cache_lru_and_ttl()
    test_plan_marks_seen_units(); test_session_id_sources(); test_session_store_turns_and_system_change()
    print("test_session: OK")
