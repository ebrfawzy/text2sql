# ── uv-powered production image ───────────────────────────────────
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS base

# Prevent Python from writing .pyc and enable unbuffered stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_HTTP_TIMEOUT=120

WORKDIR /app

# ── Install dependencies (layer caching) ─────────────────────────
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project --extra api --extra athena --extra benchmark

# ── Copy application source ─────────────────────────────────────
COPY src/ ./src/
COPY configs/ ./configs/
RUN uv sync --frozen --no-dev --extra api --extra athena --extra benchmark

# ── Default: run the CLI. Override with docker-compose command. ──
ENTRYPOINT ["uv", "run", "--no-sync", "text2sql"]
CMD ["--help"]
