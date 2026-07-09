# Argus — CPU-only auto-tagging service.
# Multi-stage: build a venv with the ML extra (CPU torch), then a slim runtime.
# Model weights are NOT baked here — mount them from the NAS at /models
# (see tasks 1.1 / 5.3), or add a bake step if you prefer a fat image.

FROM python:3.12-slim AS build

# uv for fast, reproducible installs.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

ENV UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    PIP_INDEX_URL=https://pypi.org/simple \
    # CPU torch wheels (no CUDA) — keeps the image lean, honors "no GPU".
    PIP_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cpu

WORKDIR /app
COPY pyproject.toml README.md LICENSE ./
COPY src ./src

RUN uv venv /opt/venv \
    && VIRTUAL_ENV=/opt/venv uv pip install --no-cache '.[ml,grpc]'

FROM python:3.12-slim AS runtime

RUN useradd --system --uid 10001 argus
COPY --from=build /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1

# Weights are mounted here at runtime.
VOLUME ["/models"]
USER argus

EXPOSE 8081
HEALTHCHECK --interval=30s --timeout=3s --start-period=120s \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8081/health/ready').status==200 else 1)"

ENTRYPOINT ["argus"]
