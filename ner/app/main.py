import asyncio
import logging
import os
import secrets
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, Request

from app.routers.extract import get_engine, init_engine, router as extract_router

logging.basicConfig(level=logging.INFO)

_INTERNAL_KEY = os.getenv("INTERNAL_KEY", "")


async def _verify_internal_key(request: Request):
    if request.url.path == "/health":
        return
    key = request.headers.get("X-Internal-Key")
    if not key or not _INTERNAL_KEY:
        raise HTTPException(401, "Internal key vereist")
    if not secrets.compare_digest(key, _INTERNAL_KEY):
        raise HTTPException(403, "Ongeldige key")


@asynccontextmanager
async def lifespan(app: FastAPI):
    await asyncio.to_thread(init_engine)
    yield


app = FastAPI(
    title="NER Service",
    version="0.2.0",
    lifespan=lifespan,
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
    dependencies=[Depends(_verify_internal_key)],
)
app.include_router(extract_router)


@app.get("/health")
async def health() -> dict:
    eng = get_engine()
    if eng is None:
        return {"status": "loading", "engine": "unknown"}
    engine_health = await eng.health()
    return {"status": "ok", **engine_health}
