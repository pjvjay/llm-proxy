import json

import sse


def _stream(chunks):
    return "".join(f"data: {json.dumps(c)}\n\n" for c in chunks) + "data: [DONE]\n\n"


def test_aggregate_text_and_tool_calls():
    text = _stream([
        {"id": "a", "object": "chat.completion.chunk", "model": "m",
         "choices": [{"index": 0, "delta": {"role": "assistant", "content": "Hel"}}]},
        {"id": "a", "object": "chat.completion.chunk", "model": "m",
         "choices": [{"index": 0, "delta": {"content": "lo"}}]},
        {"id": "a", "object": "chat.completion.chunk", "model": "m", "choices": [{"index": 0, "delta": {
            "tool_calls": [{"index": 0, "id": "c", "type": "function",
                            "function": {"name": "f", "arguments": "{\"x\""}}]}}]},
        {"id": "a", "object": "chat.completion.chunk", "model": "m", "choices": [{"index": 0, "delta": {
            "tool_calls": [{"index": 0, "function": {"arguments": ": 1}"}}]}}]},
        {"id": "a", "object": "chat.completion.chunk", "model": "m",
         "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]},
        {"id": "a", "object": "chat.completion.chunk", "model": "m", "choices": [],
         "usage": {"total_tokens": 5}},
    ])
    agg = sse.aggregate(text)
    msg = agg["choices"][0]["message"]
    assert msg["content"] == "Hello"
    assert msg["tool_calls"][0]["function"] == {"name": "f", "arguments": "{\"x\": 1}"}
    assert agg["choices"][0]["finish_reason"] == "tool_calls"
    assert agg["usage"] == {"total_tokens": 5}
    # render -> aggregate round trip
    assert sse.aggregate(sse.render(agg))["choices"] == agg["choices"]


def test_error_stream_returns_none():
    assert sse.aggregate('data: {"error": {"message": "x"}}\n\n') is None


def test_crlf_and_multiline():
    text = 'data: {"id":"a","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"x"}}]}\r\n\r\ndata: [DONE]\r\n\r\n'
    assert sse.aggregate(text)["choices"][0]["message"]["content"] == "x"
