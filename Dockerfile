FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    GRADIO_ANALYTICS_ENABLED=False \
    MSS_HOST=0.0.0.0 \
    MSS_USE_GPU=0

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg libsndfile1 libsamplerate0-dev build-essential cmake \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt requirements-base.txt ./
RUN python -m pip install --upgrade pip \
    && python -m pip install torch==2.14.0 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cpu \
    && python -m pip install -r requirements.txt \
    && python -m pip check \
    && python -c "import torch, torchvision, onnx2torch"
RUN useradd --create-home --uid 10001 appuser \
    && mkdir -p /app/checkpoints /app/.cache /app/results \
    && chown -R appuser:appuser /app
COPY --chown=appuser:appuser inference/ ./inference/
USER appuser
EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:7860/', timeout=4)"
CMD ["python", "-m", "inference.app"]
