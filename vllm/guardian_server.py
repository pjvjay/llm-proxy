from __future__ import annotations
import logging, math, os, time, uuid
from typing import Any, Optional
import torch, uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = os.environ.get("GUARDIAN_MODEL", "ibm-granite/granite-guardian-hap-38m")
HF_TOKEN = os.environ.get("HF_TOKEN") or None
PORT     = int(os.environ.get("PORT", "8000"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] guardian_server: %(message)s")
log = logging.getLogger("guardian_server")

tokenizer: Optional[Any] = None
model:     Optional[Any] = None
model_ready = False

app = FastAPI(title="Guardian Inference Server")

@app.get("/health")
def health():
    if not model_ready:
        raise HTTPException(status_code=503, detail="Model not loaded yet")
    return {"status": "ok", "model": MODEL_ID}

class Message(BaseModel):
    role:    str
    content: str

class ChatRequest(BaseModel):
    model:        str   = MODEL_ID
    messages:     list[Message]
    max_tokens:   int   = 16
    temperature:  float = 0.0
    logprobs:     bool  = False
    top_logprobs: int   = 0

@app.post("/v1/chat/completions")
async def chat_completions(req: ChatRequest):
    if not model_ready:
        raise HTTPException(status_code=503, detail="Model not loaded yet")
    prompt    = _build_prompt(req.messages)
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    t0 = time.perf_counter()
    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_new_tokens=req.max_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            pad_token_id=tokenizer.eos_token_id,
            return_dict_in_generate=True,
            output_scores=(req.logprobs and req.top_logprobs > 0),
        )
    elapsed    = time.perf_counter() - t0
    new_tokens = output.sequences[0][input_ids.shape[-1]:]
    reply_text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    log.info(f"Generated: {reply_text!r}  ({elapsed:.1f}s)")

    logprobs_block = None
    if req.logprobs and req.top_logprobs > 0 and output.scores:
        first_scores = output.scores[0][0]
        log_probs    = torch.log_softmax(first_scores, dim=-1)
        top_vals, top_ids = torch.topk(log_probs, k=min(req.top_logprobs, log_probs.shape[-1]))
        top_logprobs_list = [{"token": tokenizer.decode([tid.item()]), "logprob": val.item(), "bytes": None}
                             for tid, val in zip(top_ids, top_vals)]
        first_token_id = new_tokens[0].item() if len(new_tokens) > 0 else top_ids[0].item()
        logprobs_block = {"content": [{"token": tokenizer.decode([first_token_id]),
                                        "logprob": log_probs[first_token_id].item(),
                                        "bytes": None,
                                        "top_logprobs": top_logprobs_list}]}
    return JSONResponse(content={
        "id": f"chatcmpl-{uuid.uuid4().hex[:8]}", "object": "chat.completion",
        "created": int(time.time()), "model": MODEL_ID,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": reply_text},
                     "logprobs": logprobs_block, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": input_ids.shape[-1], "completion_tokens": len(new_tokens),
                  "total_tokens": input_ids.shape[-1] + len(new_tokens)},
    })

def _build_prompt(messages):
    msg_dicts = [{"role": m.role, "content": m.content} for m in messages]
    try:
        return tokenizer.apply_chat_template(msg_dicts, tokenize=False, add_generation_prompt=True)
    except Exception:
        parts = []
        for m in messages:
            parts.append(f"{'System' if m.role=='system' else 'User' if m.role=='user' else 'Assistant'}: {m.content}")
        parts.append("Assistant:")
        return "\n".join(parts)

def load_model():
    global tokenizer, model, model_ready
    log.info(f"Loading tokenizer: {MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=HF_TOKEN)
    log.info(f"Loading model on CPU (~4.5 GB download on first run)...")
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, token=HF_TOKEN,
                torch_dtype=torch.float32, device_map="cpu", low_cpu_mem_usage=True)
    model.eval()
    model_ready = True
    log.info("Model ready.")

if __name__ == "__main__":
    load_model()
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="warning")
