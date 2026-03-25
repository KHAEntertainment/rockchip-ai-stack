FROM python:3.11-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    gcc \
    g++ \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
COPY shared/requirements-base.txt ./shared/
RUN sed -i 's|../shared/requirements-base.txt|./shared/requirements-base.txt|g' requirements.txt && \
    pip install --no-cache-dir -r requirements.txt

FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH="/app" \
    MODEL_DIR=/models/whisper \
    STORAGE_PATH=/data/audio \
    DEFAULT_BACKEND=whisper_cpp

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

RUN groupadd -g 1001 appuser && \
    useradd -r -m -s /bin/bash -u 1001 -g 1001 appuser

COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

WORKDIR /app
COPY --chown=appuser:appuser . .

USER appuser
EXPOSE 8002
CMD ["uvicorn", "audio_analyzer.main:app", "--host", "0.0.0.0", "--port", "8002"]