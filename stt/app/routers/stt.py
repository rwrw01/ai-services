import logging
import subprocess
import tempfile
from pathlib import Path

import onnx_asr
from fastapi import APIRouter, File, HTTPException, UploadFile

logger = logging.getLogger(__name__)

router = APIRouter()
_vad_model = None

MAX_FILE_BYTES = 200 * 1024 * 1024  # 200 MB
MAX_DURATION_SECS = 5400  # 1 hour 30 minutes

SUPPORTED_TYPES = {
    "audio/wav",
    "audio/wave",
    "audio/x-wav",
    "audio/webm",
    "audio/ogg",
    "audio/mpeg",
    "audio/mp4",
    "audio/aac",
    "application/octet-stream",
}


def load_model() -> None:
    global _vad_model
    logger.info("Loading parakeet-tdt-0.6b-v3 ...")
    model = onnx_asr.load_model("nemo-parakeet-tdt-0.6b-v3")
    logger.info("Model ready. Loading Silero VAD ...")
    vad = onnx_asr.load_vad("silero")
    _vad_model = model.with_vad(vad, max_speech_duration_s=180)
    logger.info("VAD ready.")


def _get_duration(wav_path: str) -> float:
    """Get audio duration in seconds via ffprobe."""
    proc = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", wav_path],
        capture_output=True, text=True,
    )
    try:
        return float(proc.stdout.strip())
    except ValueError:
        return 0.0


@router.post("/api/stt")
async def speech_to_text(audio: UploadFile = File(...)):
    """Transcribe uploaded audio to text using Parakeet TDT 0.6B v3 (ONNX) with VAD."""
    if _vad_model is None:
        raise HTTPException(status_code=503, detail="Model wordt nog geladen, probeer opnieuw")

    base_type = (audio.content_type or "application/octet-stream").split(";")[0].strip()
    if base_type not in SUPPORTED_TYPES:
        raise HTTPException(status_code=415, detail=f"Niet ondersteund formaat: {base_type}")

    audio_bytes = await audio.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Leeg audiobestand")
    if len(audio_bytes) > MAX_FILE_BYTES:
        raise HTTPException(status_code=413, detail="Audiobestand te groot (max 200 MB)")

    wav_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            wav_path = f.name

        proc = subprocess.run(
            ["ffmpeg", "-y", "-i", "pipe:0",
             "-ar", "16000", "-ac", "1", "-f", "wav", wav_path],
            input=audio_bytes,
            capture_output=True,
        )
        if proc.returncode != 0:
            logger.error("ffmpeg error: %s", proc.stderr[-400:].decode(errors="replace"))
            raise HTTPException(status_code=422, detail="Audio conversie mislukt")

        duration = _get_duration(wav_path)
        if duration > MAX_DURATION_SECS:
            raise HTTPException(
                status_code=413,
                detail=f"Audio te lang ({int(duration)}s, max {MAX_DURATION_SECS}s)",
            )

        segments = _vad_model.recognize(wav_path)
        text = " ".join(seg.text for seg in segments)
        logger.info("Transcriptie (%ds, VAD): %r", int(duration), text[:120])
        return {"text": text}

    finally:
        if wav_path:
            Path(wav_path).unlink(missing_ok=True)
