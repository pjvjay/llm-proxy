# ── Guardian inference server (CPU, transformers-based) ───────────────────────
#
# Replaces vllm with a lightweight FastAPI server that loads Granite Guardian
# directly via HuggingFace transformers. Exposes the same OpenAI-compatible
# /v1/chat/completions and /health endpoints so guardian_addon.py is unchanged.
#
# Image size:  ~2 GB  (vs ~12 GB for vllm + CUDA)
# Build time:  ~5 min (vs ~25 min for vllm)
# Cold start:  model download ~4.5 GB on first run (cached in Docker volume)
# Inference:   ~10–40 s per request on CPU
#
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
    && rm -rf /var/lib/apt/lists/*

# CPU-only torch first — prevents pip from pulling the 2 GB CUDA wheel
RUN pip install --no-cache-dir \
        torch==2.4.0 \
        --index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir \
        transformers==4.46.3 \
        accelerate==1.1.1 \
        sentencepiece==0.2.0 \
        fastapi==0.115.5 \
        uvicorn==0.32.1 \
        pydantic==2.9.2

COPY guardian_server.py /app/guardian_server.py
WORKDIR /app

EXPOSE 8000

# Model is downloaded from HuggingFace on first startup and cached in the
# volume mounted at /root/.cache/huggingface (see docker-compose.yml).
CMD ["python", "guardian_server.py"]
