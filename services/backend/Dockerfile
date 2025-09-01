FROM python:3.13.7

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    POETRY_NO_INTERACTION=1 \
    POETRY_VENV_IN_PROJECT=0 \
    POETRY_CACHE_DIR=/tmp/poetry_cache

# Install system dependencies including Poetry
RUN apt-get update && apt-get install -y \
    curl \
    build-essential \
    gcc \
    g++ \
    && pip install poetry \
    && rm -rf /var/lib/apt/lists/*

RUN groupadd -r artichat && useradd -r -g artichat artichat

WORKDIR /app
COPY pyproject.toml poetry.lock ./
COPY app/ ./app/

RUN poetry install --only=main

# Change ownership to non-root user
RUN chown -R artichat:artichat /app

# Switch to non-root user
USER artichat

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8082/health || exit 1

# Expose port
EXPOSE 8082

CMD ["poetry", "run", "python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8082"]
