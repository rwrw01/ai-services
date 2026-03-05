import os

from app.engines.base import STTEngine


def create_engine() -> STTEngine:
    """Factory: create STT engine based on STT_ENGINE env var."""
    engine_name = os.getenv("STT_ENGINE", "parakeet")
    if engine_name == "speechmatics":
        from app.engines.speechmatics_engine import SpeechmaticsEngine
        return SpeechmaticsEngine()
    else:
        from app.engines.parakeet import ParakeetEngine
        return ParakeetEngine()
