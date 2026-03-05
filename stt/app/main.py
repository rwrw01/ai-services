import asyncio
import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routers.stt import get_engine, init_engine, router as stt_router

logging.basicConfig(level=logging.INFO)

_origin = os.getenv("ORIGIN", "http://localhost:3000")
ALLOWED_ORIGINS = list({_origin, "http://localhost:3000", "http://127.0.0.1:3000"})


@asynccontextmanager
async def lifespan(app: FastAPI):
    await asyncio.to_thread(init_engine)
    yield


app = FastAPI(title="STT Service", version="0.3.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)

app.include_router(stt_router)


@app.get("/health")
async def health() -> dict:
    eng = get_engine()
    if eng is None:
        return {"status": "loading", "engine": "unknown"}
    engine_health = await eng.health()
    return {"status": "ok", **engine_health}
