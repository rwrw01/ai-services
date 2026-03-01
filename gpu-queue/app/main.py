"""GPU-Queue: STT queue service met Parakeet-TDT-0.6b-v3.

Ontvangt audio, queued STT-requests, verwerkt wanneer supervisor idle is.
"""

import asyncio
import logging
import os
import tempfile
import uuid
from collections import deque
from enum import Enum

from fastapi import FastAPI, HTTPException, Request, Response

from .gpu_arbiter import acquire_gpu, is_supervisor_idle, wait_for_gpu
from .stt_engine import ParakeetSTT

logging.basicConfig(
    level=getattr(logging, os.environ.get("LOG_LEVEL", "INFO").upper(), logging.INFO),
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(title="GPU-Queue", version="0.1.0")
stt = ParakeetSTT()


# --- Job state ---

class JobStatus(str, Enum):
    queued = "queued"
    processing = "processing"
    completed = "completed"
    failed = "failed"


jobs: dict[str, dict] = {}
queue: deque[str] = deque()
_processor_task: asyncio.Task | None = None


# --- Endpoints ---

@app.post("/transcribe")
async def transcribe(request: Request):
    """Ontvang audio en queue voor STT-verwerking."""
    content_type = request.headers.get("content-type", "")
    body = await request.body()

    if not body:
        raise HTTPException(status_code=400, detail="No audio data received")

    # Bepaal extensie op basis van content-type
    ext = ".wav"
    if "ogg" in content_type:
        ext = ".ogg"
    elif "mp3" in content_type or "mpeg" in content_type:
        ext = ".mp3"
    elif "webm" in content_type:
        ext = ".webm"
    elif "flac" in content_type:
        ext = ".flac"

    # Sla audio op in tijdelijk bestand
    job_id = uuid.uuid4().hex[:12]
    audio_dir = os.environ.get("AUDIO_TMP_DIR", "/tmp/gpu-queue-audio")
    os.makedirs(audio_dir, exist_ok=True)
    audio_path = os.path.join(audio_dir, f"{job_id}{ext}")

    with open(audio_path, "wb") as f:
        f.write(body)

    jobs[job_id] = {
        "status": JobStatus.queued,
        "audio_path": audio_path,
        "text": None,
        "error": None,
        "duration_s": None,
    }
    queue.append(job_id)

    logger.info("Job %s queued (%.1f KB, %s)", job_id, len(body) / 1024, ext)
    return {"job_id": job_id, "status": "queued", "position": len(queue)}


@app.get("/transcribe/{job_id}")
async def get_result(job_id: str):
    """Poll voor resultaat van een STT-job."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    job = jobs[job_id]
    result = {"job_id": job_id, "status": job["status"]}

    if job["status"] == JobStatus.queued:
        # Bepaal positie in queue
        try:
            pos = list(queue).index(job_id) + 1
        except ValueError:
            pos = 0
        result["position"] = pos

    elif job["status"] == JobStatus.completed:
        result["text"] = job["text"]
        result["duration_s"] = job["duration_s"]

    elif job["status"] == JobStatus.failed:
        result["error"] = job["error"]

    return result


@app.get("/health")
async def health():
    """Health check endpoint."""
    supervisor_idle = False
    try:
        supervisor_idle = await is_supervisor_idle()
    except Exception:
        pass

    return {
        "status": "ok",
        "queue_size": len(queue),
        "model_loaded": stt.is_loaded,
        "supervisor_idle": supervisor_idle,
    }


# --- Background processor ---

async def process_queue():
    """Verwerkt STT-jobs uit de queue wanneer GPU beschikbaar is."""
    logger.info("Queue processor started")

    while True:
        # Wacht tot er een job in de queue staat
        if not queue:
            await asyncio.sleep(2)
            continue

        job_id = queue[0]  # Peek, niet popleft — pas na succes verwijderen

        if job_id not in jobs:
            queue.popleft()
            continue

        job = jobs[job_id]
        logger.info("Processing job %s — waiting for GPU...", job_id)

        # Wacht tot supervisor idle is en GPU vrij
        gpu_acquired = await wait_for_gpu(timeout=300, poll_interval=5)
        if not gpu_acquired:
            logger.warning("GPU timeout for job %s, retrying later", job_id)
            await asyncio.sleep(30)
            continue

        # Verwijder uit queue, markeer als processing
        queue.popleft()
        job["status"] = JobStatus.processing

        try:
            # Laad STT model als dat nog niet is
            if not stt.is_loaded:
                stt.load()

            # Transcribeer
            result = stt.transcribe(job["audio_path"])
            job["text"] = result["text"]
            job["duration_s"] = result["duration_s"]
            job["status"] = JobStatus.completed
            logger.info("Job %s completed: %d chars", job_id, len(result["text"]))

        except Exception as e:
            job["status"] = JobStatus.failed
            job["error"] = str(e)
            logger.error("Job %s failed: %s", job_id, e)

        finally:
            # Ruim audiobestand op
            try:
                os.unlink(job["audio_path"])
            except OSError:
                pass

        # Als queue leeg is, unload model om VRAM vrij te maken
        if not queue:
            logger.info("Queue empty — unloading STT model")
            stt.unload()


@app.on_event("startup")
async def startup():
    global _processor_task
    _processor_task = asyncio.create_task(process_queue())
    logger.info("GPU-Queue service started")


@app.on_event("shutdown")
async def shutdown():
    if _processor_task:
        _processor_task.cancel()
    stt.unload()
    logger.info("GPU-Queue service stopped")
