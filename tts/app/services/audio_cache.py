import hashlib
import logging
import time
import unicodedata
from pathlib import Path

logger = logging.getLogger(__name__)


def _cache_key(engine_id: str, voice: str, text: str) -> str:
    normalized = unicodedata.normalize("NFC", text.strip().lower())
    payload = f"{engine_id}:{voice}:{normalized}"
    return hashlib.sha256(payload.encode()).hexdigest()


class AudioCache:
    def __init__(self, cache_dir: str, ttl_days: int = 7) -> None:
        self._root = Path(cache_dir)
        self._ttl_seconds = ttl_days * 86400
        self._root.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        bucket = self._root / key[:2]
        bucket.mkdir(exist_ok=True)
        return bucket / f"{key}.wav"

    def get(self, engine_id: str, voice: str, text: str) -> bytes | None:
        key = _cache_key(engine_id, voice, text)
        path = self._path(key)
        if not path.exists():
            return None
        age = time.time() - path.stat().st_mtime
        if age > self._ttl_seconds:
            path.unlink(missing_ok=True)
            logger.debug("Cache expired: %s", key[:12])
            return None
        logger.debug("Cache hit: %s", key[:12])
        return path.read_bytes()

    def put(self, engine_id: str, voice: str, text: str, audio: bytes) -> None:
        key = _cache_key(engine_id, voice, text)
        path = self._path(key)
        path.write_bytes(audio)
        logger.debug("Cache stored: %s (%d bytes)", key[:12], len(audio))
