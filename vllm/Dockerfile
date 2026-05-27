FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
    && rm -rf /var/lib/apt/lists/*

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

CMD ["python", "guardian_server.py"]
