from abc import ABC, abstractmethod


class NEREngine(ABC):
    """Abstract base for NER extraction engines."""

    name: str = "base"

    @abstractmethod
    def load(self) -> None:
        """Load models for all configured domains. Called once at startup."""

    @abstractmethod
    async def extract(self, text: str, domain: str) -> dict:
        """Extract structured entities from text for the given domain.

        Returns domain-specific dict, e.g.:
        {"entries": [...], "engine": "robbertje", "domain": "uren"}
        """

    @abstractmethod
    async def health(self) -> dict:
        """Return engine health status."""
