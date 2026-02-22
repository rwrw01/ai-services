FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1     PIP_NO_CACHE_DIR=1     HF_HOME=/app/.cache/huggingface

RUN apt-get update && apt-get install -y         ffmpeg     && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt ./
RUN pip install -r requirements.txt &&     find /usr/local/lib/python3.11/site-packages/nvidia -name "lib" -type d 2>/dev/null     | sort -u > /etc/ld.so.conf.d/nvidia-cuda.conf && ldconfig

COPY app/ ./app/

RUN useradd -m -u 1000 app &&     mkdir -p /app/.cache/huggingface &&     chown -R app:app /app/.cache

USER app

EXPOSE 8001
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8001"]
