FROM python:3.13-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:0.12.19 /uv /usr/local/bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=0

WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-default-groups --group web

FROM python:3.13-slim

ENV TZ=Europe/Moscow \
    PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1

RUN useradd --system --no-create-home appuser

WORKDIR /app

COPY --from=builder /app/.venv ./.venv
COPY pyproject.toml ./
COPY src/ ./src/

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --retries=3 --start-period=10s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=3)"

CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000", "--no-access-log"]
