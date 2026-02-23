import logging

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import Response

from app.schemas.tts import EngineInfo, EnginesResponse, SynthesizeRequest

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/tts")


@router.post("/synthesize")
async def synthesize(req: SynthesizeRequest, request: Request) -> Response:
    """Convert text to WAV audio using the selected engine."""
    tts: object = request.app.state.tts
    try:
        result = await tts.synthesize(req.text, req.engine, req.voice)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail="TTS synthese mislukt")

    return Response(
        content=result.audio,
        media_type="audio/wav",
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
