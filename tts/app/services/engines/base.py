from abc import ABC, abstractmethod


class TTSEngine(ABC):
    """Abstract interface for all TTS engines."""

    @abstractmethod
    async def synthesize(self, text: str, voice: str = "default") -> bytes:
        """Generate WAV audio bytes from text."""
        ...

    @abstractmethod
    def is_available(self) -> bool:
        """Whether this engine is currently usable."""
        ...

    @property
    @abstractmethod
    def engine_id(self) -> str:
        """Unique identifier: 'piper' or 'parkiet'."""
        ...

    @property
    @abstractmethod
    def quality(self) -> str:
        """Quality label: 'basic' or 'high'."""
        ...

    @property
    @abstractmethod
    def speed(self) -> str:
        """Speed label: 'fast' or 'slow'."""
        ...
