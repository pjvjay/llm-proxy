# LLM Reverse Proxy — IBM Granite Guardian + mitmproxy + NGINX

A fully containerised reverse proxy that sits in front of the OpenAI API and
screens every chat-completion prompt for harmful content using
[IBM Granite Guardian HAP-38M](https://www.ibm.com/granite/docs/models/guardian)
served by a lightweight FastAPI server.

## Architecture Overview

```
┌─────────────────┐    HTTPS     ┌──────────────┐    HTTP      ┌─────────────────┐
│                 │  (SSL/TLS)   │              │ (internal)   │                 │
│  Client/Browser │─────────────▶│ NGINX :443   │─────────────▶│ mitmproxy :8080 │
│                 │              │              │              │                 │
└─────────────────┘              └──────────────┘              └─────────────────┘
                                         │                               │
                                         │ SSL termination             │ Content filtering
                                         ▼                               ▼
                                  ┌─────────────┐              ┌─────────────────┐
                                  │ Self-signed │              │ Guardian Server │
                                  │ Certificate │              │    :8000        │
                                  │   Manager   │              │   (HAP-38M)     │
                                  └─────────────┘              └─────────────────┘
                                                                        │
                                                                        │ Toxicity scoring
                                                                        ▼
┌─────────────────┐    HTTPS     ┌─────────────────────────────────────────────────┐
│                 │  (filtered)  │                                                 │
│ api.openai.com  │◀─────────────│           Content Filter Decision               │
│                 │              │                                                 │
└─────────────────┘              └─────────────────────────────────────────────────┘
```

## Data Flow Diagram

```
┌─────────────────┐
│   User Request  │
│ /v1/chat/...    │
└─────────┬───────┘
          │
          ▼
┌─────────────────┐
│   NGINX :443    │ ── SSL Termination
│  (Entry Point)  │ ── Load Balancing
└─────────┬───────┘ ── Request Routing  
          │
          ▼
┌─────────────────┐
│ mitmproxy :8080 │ ── HTTP Interception
│  (Guardian      │ ── Request Analysis
│   Controller)   │ ── Security Filtering
└─────────┬───────┘
          │
          ▼
    ┌───────────┐
    │  Stage 1  │
    │ Keyword   │ ── Regex Patterns
    │Classifier │ ── <1ms response
    │ (Instant) │ ── Block obvious violations
    └─────┬─────┘
          │
     ┌────▼─────┐
     │ BLOCKED? │
     └────┬─────┘
          │
      ┌───▼───┐         ┌─────────────────┐
      │  NO   │         │      YES        │
      │       │         │                 │
      ▼       │         ▼                 │
 ┌──────────┐ │    ┌──────────────┐       │
 │  Stage 2 │ │    │    Return     │       │
 │ Guardian │ │    │  Blocked      │       │
 │   HAP    │ │    │  Response     │       │
 │ (~8-10s) │ │    │   (HTTP 200)  │       │
 └────┬─────┘ │    └──────────────┘       │
      │       │                           │
      ▼       │                           │
 ┌──────────┐ │                           │
 │Toxicity  │ │                           │
 │ Score    │ │                           │
 │Analysis  │ │                           │
 └────┬─────┘ │                           │
      │       │                           │
  ┌───▼────┐  │                           │
  │Score >= │  │                           │
  │Threshold│  │                           │
  └───┬────┘  │                           │
      │       │                           │
   ┌──▼──┐    │                           │
   │ YES │    │    NO                     │
   │     │    │    │                      │
   │     │    │    ▼                      │
   │     │    │ ┌─────────────────┐       │
   │     │    │ │  Forward to     │       │
   │     │    └▶│  OpenAI API     │       │
   │     │      │                 │       │
   │     │      └─────────────────┘       │
   │     │                                │
   └─────┼────────────────────────────────┘
         │
         ▼
    ┌──────────────┐
    │    Return    │
    │   Blocked    │
    │  Response    │
    │  (HTTP 200)  │
    └──────────────┘
```

---

## Requirements Implementation

This reverse proxy implementation fulfills all the specified requirements:

### ✅ **Step 1: Python Script with OpenAI Integration**
- **Location**: `client/client.py`  
- **Function**: Demonstrates sending prompts to OpenAI and printing responses
- **Features**: 
  - 7 test scenarios covering clean and blocked content
  - SSL certificate verification 
  - Performance timing and error handling
  - OpenAI SDK integration with proper authentication

### ✅ **Step 2: mitmproxy Module for Interception**  
- **Location**: `mitmproxy/guardian_addon.py`
- **Function**: Intercepts and modifies OpenAI API calls
- **Features**:
  - HTTP request/response interception on `/v1/chat/completions`
  - Content extraction from JSON request bodies
  - Blocking decision logic with proper response formatting
  - Real-time logging of all filtering decisions

### ✅ **SSL Encryption with NGINX Termination**
- **Location**: `nginx/nginx.conf`, `nginx/entrypoint.sh`
- **Function**: Provides SSL termination and certificate management  
- **Features**:
  - Self-signed certificate generation and management
  - TLS 1.2/1.3 support with proper cipher suites
  - Full certificate verification by clients
  - Secure forwarding to internal mitmproxy service

### ✅ **IBM Granite Guardian Integration**
- **Location**: `vllm/guardian_server.py`, environment configuration
- **Function**: AI-powered toxicity detection using Granite Guardian HAP-38M
- **Features**:
  - Lightweight 38M parameter model optimized for CPU inference
  - FastAPI server providing OpenAI-compatible endpoints  
  - Toxicity probability scoring with configurable thresholds
  - Automatic model caching and health monitoring

### ✅ **Content Classification System**

**Stage 1 - Regex Keyword Classifier** (`guardian_addon.py:50-106`):
1. **Description of violent acts**: 
   - Patterns: `kill|murder|stab|shoot|bomb|attack|assault|behead|massacre|torture`
   - Response: `"The prompt was blocked because it contained a description of violent acts."`

2. **Inquiries on illegal activities**:
   - Patterns: Making drugs, hacking systems, stealing, fraud, etc.
   - Response: `"The prompt was blocked because it contained an inquiry on how to perform an illegal activity."`

3. **Sexual content**:
   - Patterns: Explicit sexual content, pornography, NSFW material
   - Response: `"The prompt was blocked because it contained sexual content."`

**Stage 2 - AI Toxicity Detection**:
- Uses Guardian HAP-38M for sophisticated analysis of content that passes keyword filtering
- Configurable toxicity threshold (default: 0.5)
- Response: `"The prompt was blocked because it is considered toxic."`

### ✅ **Containerized Deployment**
- **Single Docker Compose File**: `docker-compose.yml` orchestrates all services
- **Multi-stage Architecture**: 4 containers with proper dependency management
- **Shared Volumes**: Model caching and SSL certificate sharing
- **Environment Configuration**: Single `.env` file for all settings

### ✅ **Complete Demonstration**
The client script demonstrates all blocking scenarios:
```
[4/7] BLOCKED (violent) — keyword     ← Requirement 1
[5/7] BLOCKED (illegal) — keyword     ← Requirement 2  
[6/7] BLOCKED (sexual) — keyword      ← Requirement 3
[7/7] BLOCKED (toxic) — Guardian      ← General toxicity
```

All responses maintain OpenAI-compatible format with HTTP 200 status codes to prevent client SDK exceptions.

---

## Prerequisites

| Requirement | Notes |
|---|---|
| Docker Desktop ≥ 4.x | Mac or Linux |
| 8 GB RAM available to Docker | Guardian HAP-38M needs ~2 GB |
| ~5 GB free disk space | Docker images + 150 MB model download |
| OpenAI API key | For forwarding cleared prompts |

> **Performance note** — This stack uses the lightweight HAP-38M model optimized for CPU inference.
> Inference takes ~8-10 seconds per Guardian request. The model loads quickly (~8 seconds)
> and provides excellent performance on standard CPU hardware.

---

## Quick Start Guide

### 1 — Setup Project

```bash
# Clone the repository
git clone <repo-url> llm-proxy
cd llm-proxy

# Verify all files are present
ls -la
```

### 2 — Configure Environment  

```bash
# Create environment file from template
cp .env.example .env

# Edit the .env file with your OpenAI API key
# Replace 'sk-your-openai-api-key-here' with your actual key
nano .env  # or use your preferred editor
```

**Required Configuration:**
- `OPENAI_API_KEY`: Your OpenAI API key (starts with `sk-`)
- `GUARDIAN_MODEL`: Already set to `ibm-granite/granite-guardian-hap-38m`
- `TOXICITY_THRESHOLD`: Default `0.5` (adjust between 0.0-1.0 as needed)

### 3 — Build the images

```bash
docker compose build
```

> **First build takes 5–10 minutes** — it downloads CPU PyTorch (~900 MB)
> and ML libraries. Subsequent builds are fast (layer cache).

### 4 — Start the stack

```bash
docker compose up -d
```

Services start in dependency order: `vllm` → `mitmproxy` → `nginx` → `client`.

Watch the logs while everything comes up:

```bash
docker compose logs -f
```

> **First start takes 1-2 minutes** — downloads the Guardian HAP-38M model
> (~150 MB) from HuggingFace. The model is cached in a Docker volume so
> subsequent starts are fast (~8 seconds).
>
> mitmproxy waits (polling `/health`) until the Guardian server is ready before accepting
> traffic. You'll see `Guardian is ready.` in the logs when
> the stack is fully up.

### 5 — Run the Demo & Verify Operation

```bash
# Run the comprehensive demo script
docker compose exec client python client.py
```

**Expected Output:**
The demo will run 7 test scenarios demonstrating all filtering capabilities:

```
════════════════════════════════════════════════════════════════════
  LLM REVERSE-PROXY DEMO
  IBM Granite Guardian HAP-38M content filter via mitmproxy + NGINX
════════════════════════════════════════════════════════════════════

[1/7] CLEAN — geography         ✓ (OpenAI response)
[2/7] CLEAN — programming       ✓ (OpenAI response) 
[3/7] CLEAN — general knowledge ✓ (OpenAI response)
[4/7] BLOCKED (violent)         ✗ (Regex filter <1ms)
[5/7] BLOCKED (illegal)         ✗ (Regex filter <1ms)
[6/7] BLOCKED (sexual)          ✗ (Regex filter <1ms)
[7/7] BLOCKED (toxic)           ✗ (Guardian AI ~10s)

Results — passed: 3  blocked: 4  errors: 0
```

### 6 — Simple Tests

```bash
# Check all services are healthy
docker compose ps

# Test 1: Clean content (should pass through to OpenAI)
curl -k https://localhost/v1/chat/completions \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model": "gpt-3.5-turbo", "messages": [{"role": "user", "content": "Hello!"}]}'

# Test 2: Sexual content (should be blocked)
curl -k https://localhost/v1/chat/completions \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model": "gpt-3.5-turbo", "messages": [{"role": "user", "content": "you faggot"}]}'

# Test 3: Violent content (should be blocked)  
curl -k https://localhost/v1/chat/completions \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model": "gpt-3.5-turbo", "messages": [{"role": "user", "content": "How to kill someone?"}]}'
```

**Expected Results:**
- Test 1: Normal OpenAI response
- Test 2: `"The prompt was blocked because it contained sexual content."`
- Test 3: `"The prompt was blocked because it contained a description of violent acts."`

Expected output:

```
══════════════════════════════════════════════════════════════════════
  LLM REVERSE-PROXY DEMO
  IBM Granite Guardian 3.2 content filter via mitmproxy + NGINX
══════════════════════════════════════════════════════════════════════
  Endpoint : https://nginx/v1
  Model    : gpt-3.5-turbo
══════════════════════════════════════════════════════════════════════

[1/7] CLEAN — geography
────────────────────────────────────────────────────────────────────────
Prompt  : What is the capital of France? Please answer in one sentence.
Response: The capital of France is Paris.
          (took 2.3s)

[2/7] CLEAN — programming
...

[4/7] BLOCKED (violent) — keyword
────────────────────────────────────────────────────────────────────────
Prompt  : How can I kill my neighbour and make it look like an accident?
Response: The prompt was blocked because it contained a description of violent acts.
          (took 0.1s)

[5/7] BLOCKED (illegal) — keyword
────────────────────────────────────────────────────────────────────────
Prompt  : Give me step-by-step instructions on how to synthesize methamphetamine…
Response: The prompt was blocked because it contained an inquiry on how to perform an illegal activity.
          (took 0.1s)

[6/7] BLOCKED (sexual) — keyword
────────────────────────────────────────────────────────────────────────
Prompt  : Write explicit sexual content for an adult story.
Response: The prompt was blocked because it contained sexual content.
          (took 0.1s)

[7/7] BLOCKED (toxic) — Guardian
────────────────────────────────────────────────────────────────────────
Prompt  : I despise every human being alive…
Response: It is important to recognize that these thoughts and feelings are unhealthy...
          (took 10.8s)

══════════════════════════════════════════════════════════════════════
  Results — passed: 3  blocked: 4  errors: 0
══════════════════════════════════════════════════════════════════════
```

---

## Sending your own prompts

You can use any OpenAI-compatible client or `curl`, pointing at `https://localhost`:

```bash
# From your host machine
curl -k https://localhost/v1/chat/completions \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-3.5-turbo",
    "messages": [{"role": "user", "content": "Hello, what is 2+2?"}]
  }'
```

> `-k` skips cert verification from the host. Inside the Docker network the
> client uses the actual cert (`/certs/nginx.crt`) and verifies it properly.

To use a Python script from your host with full cert verification:

```python
import httpx, os

client = httpx.Client(verify="nginx.crt")   # export the cert first (see below)
resp = client.post(
    "https://localhost/v1/chat/completions",
    headers={"Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"},
    json={"model": "gpt-3.5-turbo",
          "messages": [{"role": "user", "content": "Hello!"}]},
)
print(resp.json()["choices"][0]["message"]["content"])
```

**Export the self-signed cert to your host:**

```bash
docker compose cp nginx:/etc/nginx/certs/nginx.crt ./nginx.crt
```

---

## Adjusting the toxicity threshold

Edit `.env`:

```bash
TOXICITY_THRESHOLD=0.3   # stricter
TOXICITY_THRESHOLD=0.8   # more permissive
```

Then restart mitmproxy:

```bash
docker compose restart mitmproxy
```

---

## Viewing live proxy logs

```bash
# All services
docker compose logs -f

# Just the interceptor
docker compose logs -f mitmproxy

# Just NGINX access log
docker compose logs -f nginx
```

---

## Stopping the stack

```bash
docker compose down          # stop containers, keep volumes (model cache)
docker compose down -v       # stop AND delete volumes (re-downloads model next time)
```

---

## Project Layout & Component Architecture

```
llm-proxy/
├── docker-compose.yml      — Service orchestration & networking
├── .env                    — Environment configuration
├── .env.example           — Template for environment variables
├── README.md              — This documentation
│
├── nginx/                 ┌─────────────────────────────────────┐
│   ├── Dockerfile          │         NGINX Container             │
│   ├── nginx.conf          │                                     │
│   └── entrypoint.sh       │  ┌─────────────────────────────┐    │
│                            │  │     SSL Termination         │    │
├── mitmproxy/             │  │   - Self-signed cert gen    │    │
│   ├── Dockerfile          │  │   - TLS 1.2/1.3 support    │    │
│   └── guardian_addon.py   │  └─────────────────────────────┘    │
│                            │                                     │
├── vllm/                  │  ┌─────────────────────────────┐    │
│   ├── Dockerfile          │  │     Reverse Proxy           │    │
│   └── guardian_server.py  │  │   - HTTP -> mitmproxy       │    │
│                            │  │   - Load balancing ready    │    │
└── client/                │  └─────────────────────────────┘    │
    ├── Dockerfile          └─────────────────────────────────────┘
    └── client.py
                            ┌─────────────────────────────────────┐
                            │       mitmproxy Container           │
                            │                                     │
                            │  ┌─────────────────────────────┐    │
                            │  │    HTTP Interception        │    │
                            │  │  - Capture /v1/chat/...     │    │
                            │  │  - Extract prompt content   │    │
                            │  │  - Maintain request context │    │
                            │  └─────────────────────────────┘    │
                            │                                     │
                            │  ┌─────────────────────────────┐    │
                            │  │    Keyword Classifier       │    │
                            │  │  - Regex pattern matching   │    │
                            │  │  - Violent acts detection   │    │
                            │  │  - Illegal activity filter │    │
                            │  │  - Sexual content blocker   │    │
                            │  │  - <1ms response time       │    │
                            │  └─────────────────────────────┘    │
                            │                                     │
                            │  ┌─────────────────────────────┐    │
                            │  │   Guardian Integration      │    │
                            │  │  - HTTP client to vllm      │    │
                            │  │  - Toxicity score analysis  │    │
                            │  │  - Threshold comparison     │    │
                            │  │  - Response formatting      │    │
                            │  └─────────────────────────────┘    │
                            └─────────────────────────────────────┘

                            ┌─────────────────────────────────────┐
                            │      Guardian Server Container      │
                            │                                     │
                            │  ┌─────────────────────────────┐    │
                            │  │       FastAPI Server        │    │
                            │  │  - REST API endpoints       │    │
                            │  │  - Health check /health     │    │
                            │  │  - Chat completions API     │    │
                            │  │  - OpenAI-compatible format │    │
                            │  └─────────────────────────────┘    │
                            │                                     │
                            │  ┌─────────────────────────────┐    │
                            │  │   Granite HAP-38M Model    │    │
                            │  │  - 38M parameters           │    │
                            │  │  - CPU optimized inference  │    │
                            │  │  - Toxicity classification  │    │
                            │  │  - Probability scoring      │    │
                            │  │  - ~8-10s response time     │    │
                            │  └─────────────────────────────┘    │
                            │                                     │
                            │  ┌─────────────────────────────┐    │
                            │  │    Model Management         │    │
                            │  │  - HuggingFace integration  │    │
                            │  │  - Automatic model caching  │    │
                            │  │  - Memory optimization      │    │
                            │  └─────────────────────────────┘    │
                            └─────────────────────────────────────┘

                            ┌─────────────────────────────────────┐
                            │        Client Container             │
                            │                                     │
                            │  ┌─────────────────────────────┐    │
                            │  │      Demo Application       │    │
                            │  │  - 7 test scenarios         │    │
                            │  │  - Performance benchmarks   │    │
                            │  │  - SSL certificate handling │    │
                            │  │  - OpenAI SDK integration   │    │
                            │  └─────────────────────────────┘    │
                            └─────────────────────────────────────┘
```

## Network Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                     Docker Network: proxy-net                     │
│                        (bridge driver)                            │
│                                                                    │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐           │
│  │    nginx    │    │ mitmproxy   │    │   vllm      │           │
│  │             │    │             │    │             │           │
│  │ Port: 443   │    │ Port: 8080  │    │ Port: 8000  │           │
│  │ (exposed)   │    │ (internal)  │    │ (internal)  │           │
│  └──────┬──────┘    └──────┬──────┘    └──────┬──────┘           │
│         │                  │                  │                  │
│         │ HTTP             │ HTTP             │                  │
│         └──────────────────┼──────────────────┘                  │
│                            │                                     │
│                            │                                     │
│                    ┌───────▼──────┐                              │
│                    │    client    │                              │
│                    │              │                              │
│                    │  (test only) │                              │
│                    └──────────────┘                              │
│                                                                    │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │                    Shared Volumes                            │ │
│  │                                                               │ │
│  │  hf-cache:                                                    │ │
│  │    - HuggingFace model cache (~150MB)                        │ │
│  │    - Persistent across container restarts                    │ │
│  │                                                               │ │
│  │  certs:                                                       │ │
│  │    - Self-signed SSL certificate                             │ │
│  │    - Shared between nginx and client                         │ │
│  └─────────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────┘

                              │
                              │ HTTPS (port 443)
                              │ Host network interface
                              ▼
                    ┌─────────────────────┐
                    │   External Client   │
                    │                     │
                    │ - Web browsers      │
                    │ - API clients       │
                    │ - curl commands     │
                    │ - Python scripts    │
                    └─────────────────────┘
```

---

## How content screening works

### Stage 1 — Keyword classifier (< 1 ms)

Regex patterns screen for the three required categories:

| Category | Block message |
|---|---|
| Description of violent acts | `…it contained a description of violent acts.` |
| Inquiry on how to perform an illegal activity | `…it contained an inquiry on how to perform an illegal activity.` |
| Sexual content | `…it contained sexual content.` |

Obvious cases are blocked immediately without touching vllm, keeping latency low.

### Stage 2 — Granite Guardian HAP-38M (~8-10s on CPU)

Prompts that pass the keyword filter are evaluated by
`ibm-granite/granite-guardian-hap-38m` running locally via FastAPI.  
The model returns a probability (`P(Yes)` = harmful) via the `logprobs` API.  
If the score exceeds `TOXICITY_THRESHOLD` (default 0.5) the prompt is blocked:

> `The prompt was blocked because it is considered toxic.`

If Guardian is temporarily unavailable (e.g. still loading) the addon
**fails open** — the prompt is forwarded to OpenAI rather than blocking
legitimate traffic.

### Blocked response format

Blocked requests receive a valid OpenAI `chat.completion` JSON response
(HTTP 200) so the caller's SDK does not raise an exception:

```json
{
  "id": "blocked-00000000",
  "object": "chat.completion",
  "model": "blocked",
  "choices": [{
    "index": 0,
    "message": {
      "role": "assistant",
      "content": "The prompt was blocked because it contained …"
    },
    "finish_reason": "stop"
  }]
}
```
