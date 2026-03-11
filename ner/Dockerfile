FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1     PIP_NO_CACHE_DIR=1     HF_HOME=/app/.cache/huggingface

WORKDIR /app
COPY requirements.txt ./
RUN pip install -r requirements.txt

COPY app/ ./app/
COPY models/ ./models/

RUN useradd -m -u 1000 app \
    && mkdir -p /app/.cache/huggingface \
    && chown -R app:app /app/.cache /app/models

USER app

EXPOSE 8004
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8004"]
