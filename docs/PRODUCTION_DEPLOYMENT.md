# Production Deployment Guide for InferenceGuard

This guide covers deployment, containerization, and configuration management for InferenceGuard services.
The system provides real-time inferential privacy evaluation and prompt rewriting.
Services run in containerized environments with health probes and resource limits.

## Architecture Overview

InferenceGuard consists of two primary models and coordination layers:
1. Risk Classifier: RoBERTa multi-task encoder hosted on Hugging Face Hub or served locally.
2. Rewriter Engine: Qwen instruction model hosted on Hugging Face Hub executing cue-conditioned rewriting.
3. API Gateway: FastAPI web service exposing REST endpoints and health checks.

## Deployment with Docker

Build the optimized multi-stage production container image:

```bash
docker build -t inference-guard-api:latest .
```

Run the container:

```bash
docker run -d \
    --name inference_guard \
    -p 8000:8000 \
    -e RISK_MODEL_ID="your-org/inferenceguard-roberta-risk" \
    -e DEVICE="cpu" \
    inference-guard-api:latest
```

## Deployment with Docker Compose

For local development or single-node staging, use Docker Compose:

```bash
docker compose up -d
```

Verify service liveness:

```bash
curl http://localhost:8000/health
curl http://localhost:8000/ready
```

## Health Probes and Monitoring

The API provides two dedicated probe endpoints:
- `/health`: Liveness probe returning HTTP 200 when the web process is running.
- `/ready`: Readiness probe verifying models are loaded in memory and hardware is initialized.
