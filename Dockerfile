FROM python:3.11.9-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    UV_PROJECT_ENVIRONMENT=/opt/maintai-venv \
    PATH="/opt/maintai-venv/bin:$PATH"

WORKDIR /app

# Install dependencies first for better layer caching, then the package.
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
COPY configs ./configs

RUN pip install "uv==0.12.5" \
    && uv sync --frozen --no-dev --no-editable \
    && useradd --create-home --uid 10001 maintai \
    && mkdir -p /app/data/uploads /app/artifacts /mlflow/artifacts \
    && chown -R maintai:maintai /app /mlflow

USER maintai

EXPOSE 8000

# Overridable default command; docker-compose overrides it for ui/migrate.
CMD ["uvicorn", "maintai.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
