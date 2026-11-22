from messages import content_text, declared_tools, map_content_text, screened_messages


def test_content_text_shapes():
    assert content_text(None) == ""
    assert content_text("a") == "a"
    assert content_text([{"type": "text", "text": "a"}, {"type": "image_url", "image_url": {}},
                         "b", {"type": "text", "text": {"value": "c"}}]) == "a\nb\nc"
    assert "weird" in content_text([{"type": "custom", "payload": "weird"}])


def test_map_content_text_preserves_shape():
    parts = [{"type": "text", "text": "a"}, {"type": "image_url", "image_url": {"url": "u"}}]
    out = map_content_text(parts, str.upper)
    assert out == [{"type": "text", "text": "A"}, {"type": "image_url", "image_url": {"url": "u"}}]
    assert map_content_text("x", str.upper) == "X" and map_content_text(None, str.upper) is None


def test_screened_messages_skips_non_dicts_and_empty():
    msgs = ["junk", {"role": "user", "content": ""}, {"role": "tool", "content": "t"}]
    assert list(screened_messages(msgs)) == [(2, "tool_result", "t")]
    assert list(screened_messages(None)) == []


def test_declared_tools():
    assert declared_tools({}) is None
    assert declared_tools({"tools": [{"type": "function", "function": {"name": "a"}}],
                           "functions": [{"name": "b"}]}) == {"a", "b"}
