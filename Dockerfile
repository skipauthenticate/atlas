ARG APP_BASE_IMAGE=python:3.11-slim

FROM ${APP_BASE_IMAGE} AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    ATLAS_VOICE_DATA_DIR=/app/data \
    ATLAS_VOICE_MODELS_DIR=/app/models \
    ATLAS_VOICE_HF_CACHE=/root/.cache/huggingface

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg curl ca-certificates git git-lfs \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md LICENSE ./
COPY atlas_voice ./atlas_voice

FROM base AS web
RUN pip install --upgrade pip \
    && pip install .

FROM base AS worker
RUN pip install --upgrade pip \
    && pip install ".[worker]"
