# GPU-Queue: STT queue service met Parakeet-TDT-0.6b-v3
# Beheert GPU-toegang: wacht tot supervisor idle, dan STT verwerken

FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/

RUN useradd -m -u 1000 app && chown -R app:app /app
USER app

EXPOSE 8003

HEALTHCHECK --interval=30s --timeout=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8003/health')"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8003"]
