"""GPU arbiter: coördineert VRAM-toegang tussen Ollama en STT/TTS."""

import asyncio
import logging
import os

import httpx

logger = logging.getLogger(__name__)

OLLAMA_URL = os.environ.get("OLLAMA_BASE_URL", "http://ollama:11434")
SUPERVISOR_URL = os.environ.get("SUPERVISOR_URL", "http://supervisor:8100")


async def is_supervisor_idle() -> bool:
    """Check of de supervisor geen actieve taken heeft."""
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            # Check pending tasks
            resp = await client.get(f"{SUPERVISOR_URL}/tasks/pending")
            if resp.status_code == 200 and resp.json():
                return False

            # Check running/validating/reviewing tasks
            resp = await client.get(f"{SUPERVISOR_URL}/tasks")
            if resp.status_code == 200:
                for task in resp.json():
                    if task.get("status") in ("running", "validating", "reviewing"):
                        return False

        return True
    except Exception:
        # Supervisor niet bereikbaar → neem aan dat ie idle is
        return True


async def acquire_gpu() -> None:
    """Unload alle Ollama modellen om GPU VRAM vrij te maken."""
    logger.info("Acquiring GPU: unloading Ollama models...")
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.get(f"{OLLAMA_URL}/api/ps")
            resp.raise_for_status()
            models = resp.json().get("models", [])

            for model in models:
                name = model.get("name", "")
                logger.info("Unloading Ollama model: %s", name)
                await client.post(
                    f"{OLLAMA_URL}/api/generate",
                    json={"model": name, "keep_alive": 0},
                )

            if models:
                await asyncio.sleep(2)  # Wacht op VRAM vrijgave
                logger.info("GPU acquired: %d model(s) unloaded", len(models))
            else:
                logger.info("GPU acquired: no models were loaded")

    except Exception as e:
        logger.warning("Could not unload Ollama models: %s", e)


async def wait_for_gpu(timeout: float = 300, poll_interval: float = 5) -> bool:
    """Wacht tot de supervisor idle is en de GPU beschikbaar.

    Returns True als GPU verkregen, False bij timeout.
    """
    import time

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if await is_supervisor_idle():
            await acquire_gpu()
            return True
        await asyncio.sleep(poll_interval)

    logger.warning("Timeout waiting for GPU (%.0fs)", timeout)
    return False
