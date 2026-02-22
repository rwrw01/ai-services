import logging
import subprocess
import tempfile
from pathlib import Path

import onnx_asr
from fastapi import APIRouter, File, HTTPException, UploadFile

logger = logging.getLogger(__name__)

router = APIRouter()
_model = None

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
    global _model
    logger.info("Loading parakeet-tdt-0.6b-v3 ...")
    _model = onnx_asr.load_model("nemo-parakeet-tdt-0.6b-v3")
    logger.info("Model ready.")


@router.post("/api/stt")
async def speech_to_text(audio: UploadFile = File(...)):
    """Transcribe uploaded audio to text using Parakeet TDT 0.6B v3 (ONNX)."""
    if _model is None:
        raise HTTPException(status_code=503, detail="Model wordt nog geladen, probeer opnieuw")

    base_type = (audio.content_type or "application/octet-stream").split(";")[0].strip()
    if base_type not in SUPPORTED_TYPES:
        raise HTTPException(status_code=415, detail=f"Niet ondersteund formaat: {base_type}")

    audio_bytes = await audio.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Leeg audiobestand")
    if len(audio_bytes) > 50 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Audiobestand te groot (max 50 MB)")

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

        text = _model.recognize(wav_path)
        logger.info("Transcriptie: %r", str(text)[:120])
        return {"text": str(text)}

    finally:
        if wav_path:
            Path(wav_path).unlink(missing_ok=True)
