import sys
sys.path.insert(0, "mitmproxy")
import content as C

def test_string_and_list_content():
    msgs = [{"role": "system", "content": "be nice"},
            {"role": "user", "content": "hello"},
            {"role": "user", "content": [{"type": "text", "text": "part one"},
                                         {"type": "image_url", "image_url": {"url": "x"}},
                                         {"type": "text", "text": "part two"}]},
            {"role": "assistant", "content": None, "tool_calls": [{"id": "1"}]},
            {"role": "tool", "tool_call_id": "1", "content": "tool says hi"}]
    us = C.units(msgs, {"user", "tool"})
    assert [(u.msg_idx, u.part_idx, u.text) for u in us] == [
        (1, None, "hello"), (2, 0, "part one"), (2, 2, "part two"), (4, None, "tool says hi")]
    assert C.message_text(msgs[2]) == "part one\npart two"
    assert C.message_text(msgs[3]) == ""          # null content must not raise

def test_set_unit_text_round_trips_into_parts():
    msgs = [{"role": "user", "content": [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]}]
    us = C.units(msgs, {"user"})
    C.set_unit_text(msgs, us[1], "REDACTED")
    assert msgs[0]["content"][1]["text"] == "REDACTED" and msgs[0]["content"][0]["text"] == "a"

def test_helpers():
    msgs = [{"role": "developer", "content": "dev"}, {"role": "user", "content": "first"},
            {"role": "user", "content": "second"}]
    assert C.system_text(msgs) == "dev"
    assert C.first_user_text(msgs) == "first"

if __name__ == "__main__":
    test_string_and_list_content(); test_set_unit_text_round_trips_into_parts(); test_helpers()
    print("test_content: OK")
