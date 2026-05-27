FROM python:3.11-slim

RUN pip install --no-cache-dir mitmproxy==10.4.2 httpx==0.27.0

RUN mkdir -p /addon
COPY guardian_addon.py /addon/guardian_addon.py

EXPOSE 8080

# Run mitmdump in reverse-proxy mode pointing at OpenAI.
# The guardian addon intercepts every /v1/chat/completions request.
CMD ["mitmdump", \
     "--mode", "reverse:https://api.openai.com", \
     "--listen-host", "0.0.0.0", \
     "--listen-port", "8080", \
     "-s", "/addon/guardian_addon.py", \
     "--set", "connection_strategy=lazy"]
