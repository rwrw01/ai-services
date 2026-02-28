import asyncio
import io
import logging
import re
import time
import wave
from typing import Any

from num2words import num2words

from app.services.engines.base import TTSEngine

logger = logging.getLogger(__name__)

PARKIET_MODEL = "pevers/parkiet"
IDLE_UNLOAD_SECONDS = 300  # unload model after 5 min of inactivity

# Regex patterns for text normalization
_RE_URL = re.compile(r"https?://\S+", re.IGNORECASE)
_RE_EMAIL = re.compile(r"\S+@\S+\.\S+")
_RE_ABBREV = re.compile(r"\b([A-Z]{2,})\b")
_RE_NUMBER = re.compile(r"\b\d+([.,]\d+)?\b")
_RE_SPECIAL = re.compile(r"[^\w\s.,!?;:'\-()…]")


# Dutch phonetic letter names for spelling out abbreviations
_LETTER_NAMES = {
    "A": "aa", "B": "bee", "C": "cee", "D": "dee", "E": "ee",
    "F": "ef", "G": "gee", "H": "haa", "I": "ie", "J": "jee",
    "K": "kaa", "L": "el", "M": "em", "N": "en", "O": "oo",
    "P": "pee", "Q": "kuu", "R": "er", "S": "es", "T": "tee",
    "U": "uu", "V": "vee", "W": "wee", "X": "iks", "Y": "ij",
    "Z": "zet",
}


def _expand_abbreviation(m: re.Match) -> str:
    """Spell out uppercase abbreviations phonetically: PZC -> pee zet cee."""
    return " ".join(_LETTER_NAMES.get(ch, ch.lower()) for ch in m.group(1))


def _number_to_words(m: re.Match) -> str:
    """Convert digits to Dutch words: 2026 -> tweeduizend zesentwintig."""
    raw = m.group(0)
    try:
        # Handle decimals with comma (Dutch style)
        if "," in raw:
            raw = raw.replace(",", ".")
        return num2words(float(raw) if "." in raw else int(raw), lang="nl")
    except (ValueError, OverflowError):
        return raw


def _normalize_for_parkiet(text: str) -> str:
    """Normalize text for optimal Parkiet Dutch TTS output.

    Parkiet expects: lowercase, digits as words, no abbreviations/URLs.
    """
    # Extract speaker tags before processing
    tags: list[str] = []
    def _save_tag(m: re.Match) -> str:
        tags.append(m.group(0))
        return f"__TAG{len(tags) - 1}__"
    text = re.sub(r"\[S\d\]", _save_tag, text)

    # Replace URLs and emails
    text = _RE_URL.sub("link", text)
    text = _RE_EMAIL.sub("e-mailadres", text)

    # Expand uppercase abbreviations BEFORE lowercasing (PZC -> p z c)
    text = _RE_ABBREV.sub(_expand_abbreviation, text)

    # Convert numbers to Dutch words (2026 -> tweeduizend zesentwintig)
    text = _RE_NUMBER.sub(_number_to_words, text)

    # Lowercase everything
    text = text.lower()

    # Remove special characters (emoji, etc.) but keep basic punctuation
    text = _RE_SPECIAL.sub("", text)

    # Collapse multiple spaces
    text = re.sub(r"\s+", " ", text).strip()

    # Restore speaker tags
    for i, tag in enumerate(tags):
        text = text.replace(f"__tag{i}__", tag)

    return text


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
            text = _normalize_for_parkiet(text)
            if not text.lstrip().startswith("[S"):
                text = f"[S1] {text}"
            logger.debug("Parkiet input: %s", text[:200])
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
