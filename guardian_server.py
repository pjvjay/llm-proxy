"""
guardian_server.py
──────────────────
Lightweight OpenAI-compatible inference server for IBM Granite Guardian 3.2.

Exposes:
  GET  /health                  — readiness probe (returns 200 once model is loaded)
  POST /v1/chat/completions     — OpenAI-compatible chat endpoint with logprobs support

The server is intentionally minimal — it only implements what guardian_addon.py
needs. It is not a general-purpose OpenAI proxy.

Environment variables:
  GUARDIAN_MODEL   HuggingFace model ID  (default: ibm-granite/granite-guardian-3.2-2b)
  HF_TOKEN         HuggingFace token     (optional, for gated models)
  PORT             Server port           (default: 8000)
"""

from __future__ import annotations

import logging
import math
import os
import time
import uuid
from typing import Any, Optional

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# ── Config ─────────────────────────────────────────────────────────────────────

MODEL_ID = os.environ.get("GUARDIAN_MODEL", "ibm-granite/granite-guardian-3.2-2b")
HF_TOKEN = os.environ.get("HF_TOKEN") or None
PORT     = int(os.environ.get("PORT", "8000"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] guardian_server: %(message)s",
)
log = logging.getLogger("guardian_server")

# ── Global model state ─────────────────────────────────────────────────────────

tokenizer: Optional[Any] = None
model:     Optional[Any] = None
model_ready = False

# ── FastAPI app ─────────────────────────────────────────────────────────────────

app = FastAPI(title="Guardian Inference Server")


@app.get("/health")
def health():
    if not model_ready:
        raise HTTPException(status_code=503, detail="Model not loaded yet")
    return {"status": "ok", "model": MODEL_ID}


# ── Request / response schemas (OpenAI-compatible subset) ──────────────────────

class Message(BaseModel):
    role:    str
    content: str

class ChatRequest(BaseModel):
    model:       str = MODEL_ID
    messages:    list[Message]
    max_tokens:  int  = 16
    temperature: float = 0.0
    logprobs:    bool  = False
    top_logprobs: int  = 0


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatRequest):
    if not model_ready:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    # ── Build prompt from messages ──────────────────────────────────────────
    prompt = _build_prompt(req.messages)

    t0 = time.perf_counter()
    input_ids = tokenizer.encode(prompt, return_tensors="pt")

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

    elapsed = time.perf_counter() - t0

    # ── Decode the new tokens only ──────────────────────────────────────────
    new_tokens  = output.sequences[0][input_ids.shape[-1]:]
    reply_text  = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    log.info(f"Generated: {reply_text!r}  ({elapsed:.1f}s)")

    # ── Build logprobs block if requested ───────────────────────────────────
    logprobs_block = None
    if req.logprobs and req.top_logprobs > 0 and output.scores:
        first_scores = output.scores[0][0]          # logits for first new token
        log_probs    = torch.log_softmax(first_scores, dim=-1)
        top_vals, top_ids = torch.topk(log_probs, k=min(req.top_logprobs, log_probs.shape[-1]))

        top_logprobs_list = [
            {
                "token":   tokenizer.decode([tid.item()]),
                "logprob": val.item(),
                "bytes":   None,
            }
            for tid, val in zip(top_ids, top_vals)
        ]

        # The token that was actually chosen
        first_token_id  = new_tokens[0].item() if len(new_tokens) > 0 else top_ids[0].item()
        first_token_lp  = log_probs[first_token_id].item()

        logprobs_block = {
            "content": [
                {
                    "token":       tokenizer.decode([first_token_id]),
                    "logprob":     first_token_lp,
                    "bytes":       None,
                    "top_logprobs": top_logprobs_list,
                }
            ]
        }

    # ── Assemble OpenAI-compatible response ─────────────────────────────────
    prompt_tokens     = input_ids.shape[-1]
    completion_tokens = len(new_tokens)

    response = {
        "id":      f"chatcmpl-{uuid.uuid4().hex[:8]}",
        "object":  "chat.completion",
        "created": int(time.time()),
        "model":   MODEL_ID,
        "choices": [
            {
                "index":         0,
                "message":       {"role": "assistant", "content": reply_text},
                "logprobs":      logprobs_block,
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens":     prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens":      prompt_tokens + completion_tokens,
        },
    }
    return JSONResponse(content=response)


# ── Prompt builder ──────────────────────────────────────────────────────────────

def _build_prompt(messages: list[Message]) -> str:
    """
    Convert a list of chat messages into a single prompt string using the
    Granite chat template format.

    Falls back to a simple concatenation if the tokenizer has no chat template.
    """
    msg_dicts = [{"role": m.role, "content": m.content} for m in messages]
    try:
        return tokenizer.apply_chat_template(
            msg_dicts,
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception:
        # Fallback: simple System/User/Assistant format
        parts = []
        for m in messages:
            if m.role == "system":
                parts.append(f"System: {m.content}")
            elif m.role == "user":
                parts.append(f"User: {m.content}")
            elif m.role == "assistant":
                parts.append(f"Assistant: {m.content}")
        parts.append("Assistant:")
        return "\n".join(parts)


# ── Model loader ────────────────────────────────────────────────────────────────

def load_model() -> None:
    global tokenizer, model, model_ready

    log.info(f"Loading tokenizer: {MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        token=HF_TOKEN,
    )

    log.info(f"Loading model: {MODEL_ID}  (device=cpu, dtype=float32)")
    log.info("This will take several minutes on first run while the model downloads (~4.5 GB).")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        token=HF_TOKEN,
        torch_dtype=torch.float32,
        device_map="cpu",
        low_cpu_mem_usage=True,
    )
    model.eval()

    model_ready = True
    log.info("Model ready. Server accepting requests.")


# ── Entry point ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    load_model()
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="warning")
