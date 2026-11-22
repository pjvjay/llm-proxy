# Python 3.12: mitmproxy 11.1 requires it (on 3.11 pip silently falls back to 11.0).
FROM python:3.12-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1 \
    UPSTREAM_URL=https://api.openai.com
COPY requirements.txt .
# PII uses a tokenizer-only spaCy pipeline by default (PII_NLP_MODEL=blank), so
# no spaCy model download is needed. Build with --build-arg SPACY_MODEL=en_core_web_sm
# if you add NER entities (PERSON, LOCATION) to pii.py.
ARG SPACY_MODEL=
RUN pip install --no-cache-dir -r requirements.txt \
    && if [ -n "$SPACY_MODEL" ]; then python -m spacy download "$SPACY_MODEL"; fi \
    && mkdir -p /var/log/firewall
COPY *.py ./
EXPOSE 8080
# connection_strategy=lazy: only connect upstream when a request is actually
# forwarded, so blocked requests don't open an upstream connection at all.
# (Blocked requests are answered either way -- verified with an unreachable
# upstream under both strategies.) `exec` so mitmdump gets SIGTERM.
CMD ["sh", "-c", "exec mitmdump -s firewall_addon.py --listen-port 8080 --mode \"reverse:${UPSTREAM_URL}\" --set keep_host_header=false --set connection_strategy=lazy"]
