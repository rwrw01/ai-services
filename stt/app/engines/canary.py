import asyncio
import logging
import os
import re

import onnx_asr

from app.engines.base import STTEngine

logger = logging.getLogger(__name__)

# Collapse 2+ dots (with optional spaces) into a single period
_DOTS_RE = re.compile(r"[.\s]*\.{2,}[.\s]*")
# Collapse multiple spaces
_SPACES_RE = re.compile(r" {2,}")


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
        self._vad_model = model.with_vad(
            vad,
            max_speech_duration_s=180,
            threshold=0.6,
            min_silence_duration_ms=600,
            min_speech_duration_ms=250,
            speech_pad_ms=50,
        )
        logger.info("Canary + VAD ready.")

    async def transcribe(self, wav_path: str) -> dict:
        if self._vad_model is None:
            raise RuntimeError("Canary model not loaded")
        language = os.environ.get("CANARY_LANGUAGE", "nl")
        segments = await asyncio.to_thread(
            self._vad_model.recognize, wav_path, language=language,
        )
        parts = [seg.text.strip() for seg in segments]
        parts = [p for p in parts if p and not all(c in ".… " for c in p)]
        text = " ".join(parts)
        text = _DOTS_RE.sub(". ", text)
        text = _SPACES_RE.sub(" ", text).strip()
        return {"text": text, "engine": self.name}

    async def health(self) -> dict:
        return {
            "engine": self.name,
            "model_loaded": self._vad_model is not None,
        }
