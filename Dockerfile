FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl git nodejs npm ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

RUN useradd -m -u 1000 appuser
WORKDIR /app

COPY pyproject.toml ./
RUN uv pip install --system --no-cache google-adk google-genai fastapi "uvicorn[standard]" pydantic python-dotenv

COPY . .
RUN mkdir -p /app/data && chown -R appuser:appuser /app
USER appuser

EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8080/healthz || exit 1

CMD ["uvicorn", "app.core.server:app", "--host", "0.0.0.0", "--port", "8080"]
