FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
# PII uses a tokenizer-only spaCy pipeline by default (PII_NLP_MODEL=blank):
# every configured entity is pattern-based, so no model download is needed.
# To use NER entities, build with --build-arg SPACY_MODEL=en_core_web_sm and
# set PII_NLP_MODEL=en_core_web_sm.
ARG SPACY_MODEL=""
RUN pip install --no-cache-dir -r requirements.txt \
    && if [ -n "$SPACY_MODEL" ]; then python -m spacy download "$SPACY_MODEL"; fi
COPY *.py ./
EXPOSE 8080
CMD ["mitmdump", "-s", "firewall_addon.py", "--listen-port", "8080", \
     "--mode", "reverse:https://api.openai.com", "--set", "keep_host_header=false"]
