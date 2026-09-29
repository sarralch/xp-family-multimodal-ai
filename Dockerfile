# syntax=docker/dockerfile:1.7
FROM python:3.12-slim AS base

# Tesseract with French + English language data for label OCR.
RUN apt-get update \
    && apt-get install -y --no-install-recommends tesseract-ocr tesseract-ocr-fra tesseract-ocr-eng \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.12 /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    PATH="/opt/venv/bin:$PATH" \
    HF_HOME=/cache/huggingface \
    CACHE_DIR=/cache/xp \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Dependencies first (cached layer), exactly as pinned in uv.lock.
COPY pyproject.toml uv.lock README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --extra ml --extra ocr --extra web

COPY src ./src
COPY web ./web
COPY configs ./configs
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --extra ml --extra ocr --extra web

RUN useradd --create-home --uid 1000 app && mkdir -p /cache && chown -R app /cache /app
USER app

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"
CMD ["uvicorn", "xp_family.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
