FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && python -m spacy download en_core_web_sm
COPY *.py ./
EXPOSE 8080
CMD ["mitmdump", "-s", "firewall_addon.py", "--listen-port", "8080", \
     "--mode", "reverse:https://api.openai.com", "--set", "keep_host_header=false"]
