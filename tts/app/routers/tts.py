import logging
import subprocess

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import Response

from app.schemas.tts import EngineInfo, EnginesResponse, SynthesizeRequest

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/tts")


def wav_to_mp3(wav_bytes: bytes) -> bytes:
    """Convert WAV audio to MP3 using ffmpeg."""
    proc = subprocess.run(
        ["ffmpeg", "-i", "pipe:0", "-f", "mp3", "-ab", "128k", "-ac", "1", "pipe:1"],
        input=wav_bytes,
        capture_output=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg conversion failed: {proc.stderr.decode()[-200:]}")
    return proc.stdout


@router.post("/synthesize")
async def synthesize(req: SynthesizeRequest, request: Request) -> Response:
    """Convert text to audio using the selected engine."""
    tts: object = request.app.state.tts
    try:
        result = await tts.synthesize(req.text, req.engine, req.voice)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail="TTS synthese mislukt")

    audio = result.audio
    media_type = "audio/wav"

    if req.output_format == "mp3":
        try:
            audio = wav_to_mp3(audio)
            media_type = "audio/mpeg"
        except RuntimeError:
            logger.exception("WAV-to-MP3 conversion failed, returning WAV")

    return Response(
        content=audio,
        media_type=media_type,
        headers={
            "X-Engine-Used": result.engine_used,
            "X-Cached": str(result.cached).lower(),
            "X-Duration-Ms": str(result.duration_ms),
        },
    )


@router.get("/engines", response_model=EnginesResponse)
async def engines(request: Request) -> EnginesResponse:
    """List available TTS engines and their status."""
    tts: object = request.app.state.tts
    available = tts.available_engines()
    engine_list = [
        EngineInfo(
            id=e.engine_id,
            available=e.is_available(),
            quality=e.quality,
            speed=e.speed,
        )
        for e in available
    ]
    return EnginesResponse(engines=engine_list, default=tts._default)
