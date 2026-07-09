# Argus — CPU-only auto-tagging service.
# Multi-stage: build a venv with the ML extra (CPU torch), then a slim runtime.
# Model weights are NOT baked here — mount them from the NAS at /models
# (see tasks 1.1 / 5.3), or add a bake step if you prefer a fat image.

FROM python:3.11-slim AS build

# uv for fast, reproducible installs.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# git: the `ml` extra pins RAM++ via a git+https reference, so the builder needs it.
RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

ENV UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

WORKDIR /app
COPY pyproject.toml README.md LICENSE ./
COPY src ./src

# CPU torch wheels (no CUDA) from the pytorch cpu index; `unsafe-best-match` lets
# uv pick torch from it while resolving everything else from PyPI.
RUN uv venv /opt/venv \
    && VIRTUAL_ENV=/opt/venv uv pip install --no-cache \
       --extra-index-url https://download.pytorch.org/whl/cpu \
       --index-strategy unsafe-best-match \
       '.[ml,grpc]'

FROM python:3.11-slim AS runtime

# --create-home so the non-root user has a writable HF cache: RAM++ pulls the
# `bert-base-uncased` tokenizer at load time and must be able to write it.
RUN useradd --system --uid 10001 --create-home argus
COPY --from=build /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/home/argus/.cache/huggingface

# Weights are mounted here at runtime.
VOLUME ["/models"]
USER argus

EXPOSE 8081
HEALTHCHECK --interval=30s --timeout=3s --start-period=120s \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8081/health/ready').status==200 else 1)"

ENTRYPOINT ["argus"]
