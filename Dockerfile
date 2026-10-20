# One image for both the API and the Streamlit UI; docker-compose.yml picks the command.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Dependencies first so code changes don't invalidate the pip layer.
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY alembic.ini .
COPY migrations ./migrations
COPY src ./src

RUN useradd --create-home --uid 1000 app && chown -R app:app /app
USER app

EXPOSE 8003 8501

# Default: run migrations, then the API. The UI service overrides this command.
CMD ["sh", "-c", "alembic upgrade head && exec uvicorn src.api.app:app --host 0.0.0.0 --port 8003"]
