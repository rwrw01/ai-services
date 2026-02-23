import logging
import time
from dataclasses import dataclass

from app.services.audio_cache import AudioCache
from app.services.engines.base import TTSEngine
from app.services.engines.parkiet import ParkietEngine
from app.services.engines.piper import PiperEngine

logger = logging.getLogger(__name__)


@dataclass
class SynthesisResult:
    audio: bytes
    engine_used: str
    cached: bool
    duration_ms: int


class TTSService:
    def __init__(
        self,
        piper: PiperEngine | None,
        parkiet: ParkietEngine | None,
        cache: AudioCache,
        default_engine: str = "piper",
    ) -> None:
        self._piper = piper
        self._parkiet = parkiet
        self._cache = cache
        self._default = default_engine

    def available_engines(self) -> list[TTSEngine]:
        engines: list[TTSEngine] = []
        if self._piper and self._piper.is_available():
            engines.append(self._piper)
        if self._parkiet and self._parkiet.is_available():
            engines.append(self._parkiet)
        return engines

    async def synthesize(
        self, text: str, engine: str = "auto", voice: str = "default"
    ) -> SynthesisResult:
        selected = self._select_engine(engine)
        cached_audio = self._cache.get(selected.engine_id, voice, text)
        if cached_audio:
            return SynthesisResult(
                audio=cached_audio,
                engine_used=selected.engine_id,
                cached=True,
                duration_ms=0,
            )

        t0 = time.monotonic()
        try:
            audio = await selected.synthesize(text, voice)
        except Exception as exc:
            if selected is not self._piper and self._piper and self._piper.is_available():
                logger.warning("Falling back to Piper: %s", exc)
                audio = await self._piper.synthesize(text, voice)
                selected = self._piper
            else:
                raise

        duration_ms = int((time.monotonic() - t0) * 1000)
        self._cache.put(selected.engine_id, voice, text, audio)
        return SynthesisResult(
            audio=audio,
            engine_used=selected.engine_id,
            cached=False,
            duration_ms=duration_ms,
        )

    def _select_engine(self, engine: str) -> TTSEngine:
        if engine == "piper":
            if not self._piper:
                raise ValueError("Piper engine is not enabled")
            return self._piper
        if engine == "parkiet":
            if not self._parkiet or not self._parkiet.is_available():
                raise ValueError("Parkiet engine is not available")
            return self._parkiet
        # "auto": prefer Parkiet if available, else Piper
        if self._parkiet and self._parkiet.is_available():
            return self._parkiet
        if self._piper:
            return self._piper
        raise RuntimeError("No TTS engine available")
