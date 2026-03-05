from abc import ABC, abstractmethod


class STTEngine(ABC):
    """Abstract base for speech-to-text engines."""

    name: str = "base"

    @abstractmethod
    def load(self) -> None:
        """Load model / validate credentials. Called once at startup."""

    @abstractmethod
    async def transcribe(self, wav_path: str) -> dict:
        """Transcribe a 16 kHz mono WAV file.

        Returns: {"text": "...", "engine": "<engine-name>"}
        """

    @abstractmethod
    async def health(self) -> dict:
        """Return engine health status."""
