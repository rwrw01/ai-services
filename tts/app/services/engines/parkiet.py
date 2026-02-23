import asyncio
import io
import logging
import time
import wave
from typing import Any

from app.services.engines.base import TTSEngine

logger = logging.getLogger(__name__)

PARKIET_MODEL = "pevers/parkiet"
IDLE_UNLOAD_SECONDS = 300  # unload model after 5 min of inactivity


class ParkietEngine(TTSEngine):
    """TTS via Parkiet — high quality Dutch, GPU-based, lazy loaded."""

    def __init__(self) -> None:
        self._pipeline: Any = None
        self._last_used: float = 0.0
        self._lock = asyncio.Lock()

    @property
    def engine_id(self) -> str:
        return "parkiet"

    @property
    def quality(self) -> str:
        return "high"

    @property
    def speed(self) -> str:
        return "slow"

    def is_available(self) -> bool:
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False

    async def synthesize(self, text: str, voice: str = "default") -> bytes:
        async with self._lock:
            await asyncio.to_thread(self._ensure_loaded)
            if not text.lstrip().startswith("[S"):
                text = f"[S1] {text}"
            wav_bytes = await asyncio.to_thread(self._run_inference, text)
            self._last_used = time.monotonic()
            return wav_bytes

    def _ensure_loaded(self) -> None:
        if self._pipeline is not None:
            return
        logger.info("Loading Parkiet model from %s ...", PARKIET_MODEL)
        try:
            import torch
            from transformers import pipeline as hf_pipeline

            device = "cuda" if torch.cuda.is_available() else "cpu"
            self._pipeline = hf_pipeline(
                "text-to-speech",
                model=PARKIET_MODEL,
                torch_dtype=torch.bfloat16,
                device=device,
            )
            logger.info("Parkiet model loaded on %s", device)
        except Exception as exc:
            logger.error("Failed to load Parkiet: %s", exc)
            raise RuntimeError("Parkiet model kon niet geladen worden") from exc

    def _run_inference(self, text: str) -> bytes:
        result = self._pipeline(text)
        # result["audio"] is a numpy array; result["sampling_rate"] is the rate
        audio_array = result["audio"]
        sample_rate = result["sampling_rate"]
        return _numpy_to_wav(audio_array, sample_rate)

    def unload(self) -> None:
        """Release GPU memory."""
        if self._pipeline is not None:
            logger.info("Unloading Parkiet model to free VRAM")
            self._pipeline = None
            try:
                import torch
                torch.cuda.empty_cache()
            except ImportError:
                pass


def _numpy_to_wav(audio: Any, sample_rate: int) -> bytes:
    import numpy as np

    audio_int16 = (np.clip(audio.squeeze(), -1.0, 1.0) * 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio_int16.tobytes())
    return buf.getvalue()
