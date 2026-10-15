"""Run `python3 setup_clean_files.py` after downloading from Google Drive to generate clean plain-text Dockerfile, pyproject.toml, and .env files."""

from pathlib import Path

for docx in Path(".").glob("*.docx"):
    docx.unlink()

Path("pyproject.toml").write_text(
'''[project]
name = "universal-adk-system-design-v3"
version = "3.0.0"
description = "17-Layer Production Google ADK System Design Framework with 1-File Type-C Cartridge Architecture"
readme = "SYSTEM_DESIGN_ARCHITECTURE.md"
requires-python = ">=3.11"
dependencies = [
    "google-adk>=0.5.0",
    "google-genai>=1.0.0",
    "fastapi>=0.115.0",
    "uvicorn[standard]>=0.30.0",
    "pydantic>=2.8.0",
    "python-dotenv>=1.0.0",
    "pytest>=8.0.0",
]
''', encoding="utf-8")

Path("Dockerfile").write_text(
'''FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \\
    curl git nodejs npm ca-certificates \\
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
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \\
    CMD curl -f http://localhost:8080/healthz || exit 1

CMD ["uvicorn", "app.core.server:app", "--host", "0.0.0.0", "--port", "8080"]
''', encoding="utf-8")

if not Path(".env").exists():
    Path(".env").write_text(
        'GOOGLE_GENAI_USE_VERTEXAI=FALSE\nGOOGLE_API_KEY="PASTE_YOUR_GEMINI_API_KEY_HERE"\nADK_MODEL_NAME="gemini-2.5-flash"\n',
        encoding="utf-8",
    )

print("✅ Clean Dockerfile, pyproject.toml, and .env generated!")
