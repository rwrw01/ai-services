"""Parakeet-TDT-0.6b-v3 speech-to-text engine via onnx-asr.

Snel, Nederlands-ondersteunend STT model met GPU lifecycle management.
"""

import logging
import subprocess
import tempfile
import time
from pathlib import Path

logger = logging.getLogger(__name__)

MODEL_NAME = "nemo-parakeet-tdt-0.6b-v3"


class ParakeetSTT:
    """Lazy-loaded Parakeet STT model met expliciete GPU lifecycle."""

    def __init__(self):
        self._vad_model = None

    @property
    def is_loaded(self) -> bool:
        return self._vad_model is not None

    def load(self) -> None:
        """Laad het model."""
        if self._vad_model is not None:
            return

        import onnx_asr

        logger.info("Loading Parakeet model: %s", MODEL_NAME)
        start = time.monotonic()

        model = onnx_asr.load_model(MODEL_NAME)
        vad = onnx_asr.load_vad("silero")
        self._vad_model = model.with_vad(vad, max_speech_duration_s=180)

        elapsed = time.monotonic() - start
        logger.info("Parakeet loaded in %.1fs", elapsed)

    def transcribe(self, audio_path: str) -> dict:
        """Transcribeer een audiobestand naar tekst.

        Converteert eerst naar 16kHz mono WAV via ffmpeg.
        """
        if self._vad_model is None:
            raise RuntimeError("Model not loaded — call load() first")

        # Converteer naar 16kHz mono WAV
        wav_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                wav_path = f.name

            proc = subprocess.run(
                ["ffmpeg", "-y", "-i", audio_path,
                 "-ar", "16000", "-ac", "1", "-f", "wav", wav_path],
                capture_output=True,
            )
            if proc.returncode != 0:
                error = proc.stderr[-400:].decode(errors="replace")
                raise RuntimeError(f"ffmpeg conversie mislukt: {error}")

            start = time.monotonic()
            segments = self._vad_model.recognize(wav_path)
            text = " ".join(seg.text for seg in segments)
            elapsed = time.monotonic() - start

            logger.info("Transcribed %d chars in %.1fs", len(text), elapsed)
            return {"text": text, "duration_s": elapsed}

        finally:
            if wav_path:
                Path(wav_path).unlink(missing_ok=True)

    def unload(self) -> None:
        """Unload model en maak GPU VRAM vrij."""
        if self._vad_model is None:
            return

        logger.info("Unloading Parakeet model")
        self._vad_model = None

        try:
            import gc
            gc.collect()
        except Exception:
            pass
