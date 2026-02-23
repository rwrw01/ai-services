FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/.cache/huggingface \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    protobuf-compiler \
    libprotobuf-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt ./

# Install torch CUDA build first, then the rest
RUN pip install torch torchaudio --extra-index-url https://download.pytorch.org/whl/cu121 && \
    pip install -r requirements.txt

# Download Piper model at build time so the container starts instantly
RUN python -c "\
from huggingface_hub import hf_hub_download; \
import os; \
os.makedirs('/app/models', exist_ok=True); \
hf_hub_download('rhasspy/piper-voices', 'nl/nl_BE/nathalie/medium/nl_BE-nathalie-medium.onnx', local_dir='/app/models', local_dir_use_symlinks=False); \
hf_hub_download('rhasspy/piper-voices', 'nl/nl_BE/nathalie/medium/nl_BE-nathalie-medium.onnx.json', local_dir='/app/models', local_dir_use_symlinks=False); \
print('Piper model downloaded')"

# Flatten model files to /app/models/
RUN find /app/models -name "nl_BE-nathalie-medium.onnx" -not -path "/app/models/nl_BE-nathalie-medium.onnx" \
    -exec cp {} /app/models/nl_BE-nathalie-medium.onnx \; 2>/dev/null || true && \
    find /app/models -name "nl_BE-nathalie-medium.onnx.json" -not -path "/app/models/nl_BE-nathalie-medium.onnx.json" \
    -exec cp {} /app/models/nl_BE-nathalie-medium.onnx.json \; 2>/dev/null || true

COPY app/ ./app/

RUN useradd -m -u 1000 app && \
    mkdir -p /app/.cache/huggingface /data/tts-cache && \
    chown -R app:app /app /data

USER app
EXPOSE 8002
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8002"]
