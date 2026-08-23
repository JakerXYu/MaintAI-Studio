FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Install dependencies first for better layer caching, then the package.
COPY pyproject.toml README.md ./
COPY src ./src
COPY configs ./configs

RUN pip install --upgrade pip \
    && pip install -e .

EXPOSE 8000

# Overridable default command; docker-compose overrides it for ui/migrate.
CMD ["uvicorn", "maintai.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
