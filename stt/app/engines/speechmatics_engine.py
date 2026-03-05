import asyncio
import logging
import os

from app.engines.base import STTEngine

logger = logging.getLogger(__name__)


class SpeechmaticsEngine(STTEngine):
    """Speechmatics Cloud Batch API via official Python SDK."""

    name = "speechmatics"

    def __init__(self):
        self._api_key: str | None = None

    def load(self) -> None:
        self._api_key = os.environ.get("SPEECHMATICS_API_KEY", "")
        if not self._api_key:
            raise RuntimeError("SPEECHMATICS_API_KEY not set")
        logger.info("Speechmatics engine ready (cloud API).")

    async def transcribe(self, wav_path: str) -> dict:
        if not self._api_key:
            raise RuntimeError("Speechmatics API key not configured")
        text = await asyncio.to_thread(self._batch_transcribe, wav_path)
        return {"text": text, "engine": self.name}

    def _batch_transcribe(self, wav_path: str) -> str:
        """Synchronous batch transcription via Speechmatics SDK."""
        from speechmatics.batch_client import BatchClient
        from speechmatics.models import BatchTranscriptionConfig

        with BatchClient(self._api_key) as client:
            job_id = client.submit_job(
                wav_path,
                BatchTranscriptionConfig(
                    language="auto",
                    operating_point="enhanced",
                ),
            )
            logger.info("Speechmatics job %s submitted", job_id)
            transcript = client.wait_for_completion(
                job_id,
                transcription_format="txt",
            )
        return transcript.strip()

    async def health(self) -> dict:
        ok = bool(self._api_key)
        return {
            "engine": self.name,
            "api_key_set": ok,
        }
