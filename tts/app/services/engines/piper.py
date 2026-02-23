import asyncio
import io
import logging
import wave

from app.services.engines.base import TTSEngine

logger = logging.getLogger(__name__)

# Piper outputs 22050 Hz mono 16-bit PCM; we wrap it in a WAV container
PIPER_SAMPLE_RATE = 22050


class PiperEngine(TTSEngine):
    """TTS via Piper — fast, CPU-only, Dutch voice (nl_BE-nathalie-medium)."""

    def __init__(self, model_path: str) -> None:
        self._model_path = model_path

    @property
    def engine_id(self) -> str:
        return "piper"

    @property
    def quality(self) -> str:
        return "basic"

    @property
    def speed(self) -> str:
        return "fast"

    def is_available(self) -> bool:
        return True  # CPU-based, always available

    async def synthesize(self, text: str, voice: str = "default") -> bytes:
        """Run piper as subprocess: text → stdin, raw PCM → stdout."""
        proc = await asyncio.create_subprocess_exec(
            "piper",
            "--model", self._model_path,
            "--output_raw",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        pcm_bytes, stderr = await proc.communicate(input=text.encode())

        if proc.returncode != 0:
            err = stderr.decode(errors="replace")[-300:]
            logger.error("piper error: %s", err)
            raise RuntimeError(f"Piper synthesis failed: {err}")

        return _pcm_to_wav(pcm_bytes, PIPER_SAMPLE_RATE)


def _pcm_to_wav(pcm: bytes, sample_rate: int) -> bytes:
    """Wrap raw 16-bit mono PCM in a WAV container."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sample_rate)
        wf.writeframes(pcm)
    return buf.getvalue()
