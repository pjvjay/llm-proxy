FROM python:3.11-slim

RUN pip install --no-cache-dir httpx==0.27.0

COPY client.py /app/client.py
WORKDIR /app

# Keep the container alive; run `docker compose exec client python client.py`
# to fire the demo at any time.
CMD ["tail", "-f", "/dev/null"]
