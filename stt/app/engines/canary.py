import asyncio
import logging
import os

import onnx_asr

from app.engines.base import STTEngine

logger = logging.getLogger(__name__)


class CanaryEngine(STTEngine):
    """Local Canary-1B-v2 via ONNX + Silero VAD. Multilingual, higher accuracy."""

    name = "canary"

    def __init__(self):
        self._vad_model = None

    def load(self) -> None:
        logger.info("Loading canary-1b-v2 ...")
        model = onnx_asr.load_model("nemo-canary-1b-v2")
        logger.info("Model ready. Loading Silero VAD ...")
        vad = onnx_asr.load_vad("silero")
        self._vad_model = model.with_vad(vad, max_speech_duration_s=180)
        logger.info("Canary + VAD ready.")

    async def transcribe(self, wav_path: str) -> dict:
        if self._vad_model is None:
            raise RuntimeError("Canary model not loaded")
        language = os.environ.get("CANARY_LANGUAGE", "nl")
        segments = await asyncio.to_thread(
            self._vad_model.recognize, wav_path, language=language,
        )
        text = " ".join(seg.text for seg in segments)
        return {"text": text, "engine": self.name}

    async def health(self) -> dict:
        return {
            "engine": self.name,
            "model_loaded": self._vad_model is not None,
        }
