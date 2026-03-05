import asyncio
import logging

import onnx_asr

from app.engines.base import STTEngine

logger = logging.getLogger(__name__)


class ParakeetEngine(STTEngine):
    """Local Parakeet-TDT-0.6B-v3 via ONNX + Silero VAD."""

    name = "parakeet"

    def __init__(self):
        self._vad_model = None

    def load(self) -> None:
        logger.info("Loading parakeet-tdt-0.6b-v3 ...")
        model = onnx_asr.load_model("nemo-parakeet-tdt-0.6b-v3")
        logger.info("Model ready. Loading Silero VAD ...")
        vad = onnx_asr.load_vad("silero")
        self._vad_model = model.with_vad(vad, max_speech_duration_s=180)
        logger.info("Parakeet + VAD ready.")

    async def transcribe(self, wav_path: str) -> dict:
        if self._vad_model is None:
            raise RuntimeError("Parakeet model not loaded")
        segments = await asyncio.to_thread(self._vad_model.recognize, wav_path)
        text = " ".join(seg.text for seg in segments)
        return {"text": text, "engine": self.name}

    async def health(self) -> dict:
        return {
            "engine": self.name,
            "model_loaded": self._vad_model is not None,
        }
