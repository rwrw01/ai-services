import os

from app.engines.base import STTEngine


def create_engine() -> STTEngine:
    """Factory: create STT engine based on STT_ENGINE env var."""
    engine_name = os.getenv("STT_ENGINE", "canary")
    if engine_name == "speechmatics":
        from app.engines.speechmatics_engine import SpeechmaticsEngine
        return SpeechmaticsEngine()
    else:
        from app.engines.canary import CanaryEngine
        return CanaryEngine()
