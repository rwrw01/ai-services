import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.config import settings
from app.routers.tts import router as tts_router
from app.services.audio_cache import AudioCache
from app.services.engines.parkiet import ParkietEngine
from app.services.engines.piper import PiperEngine
from app.services.tts_service import TTSService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    piper: PiperEngine | None = None
    parkiet: ParkietEngine | None = None

    if settings.tts_piper_enabled:
        model_path = f"{settings.tts_models_dir}/{settings.tts_piper_model}.onnx"
        piper = PiperEngine(model_path=model_path)
        logger.info("Piper engine ready (model: %s)", settings.tts_piper_model)

    if settings.tts_parkiet_enabled:
        parkiet = ParkietEngine()
        logger.info(
            "Parkiet engine configured (GPU available: %s)",
            parkiet.is_available(),
        )

    cache = AudioCache(settings.tts_cache_dir, settings.tts_cache_ttl_days)
    app.state.tts = TTSService(piper, parkiet, cache, settings.tts_default_engine)
    logger.info("TTS service ready. Default engine: %s", settings.tts_default_engine)
    yield


app = FastAPI(title="Memories TTS Service", version="0.1.0", lifespan=lifespan)

app.include_router(tts_router)


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "service": "memories-tts"}
