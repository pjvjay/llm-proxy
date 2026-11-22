FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY server.py .
# Models are mounted at /models (see docker-compose volume) after export_onnx.py
EXPOSE 8000
# keep-alive longer than the proxy client's pool expiry (60 s) so pooled
# connections are closed by the client, not reset under it
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000", "--timeout-keep-alive", "75"]
