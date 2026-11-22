FROM python:3.11-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1 \
    HF_HOME=/hf-cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY server.py manifest.py prepare_models.py ./
# Models live in /models (bind-mounted ./models). The `model-init` service runs
# prepare_models.py from this same image to download / export / verify them
# before this service starts; see docker-compose.yml.
EXPOSE 8000
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
