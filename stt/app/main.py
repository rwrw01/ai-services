import asyncio
import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routers.stt import load_model, router as stt_router

logging.basicConfig(level=logging.INFO)

_origin = os.getenv("ORIGIN", "http://localhost:3000")
ALLOWED_ORIGINS = list({_origin, "http://localhost:3000", "http://127.0.0.1:3000"})


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the Parakeet model in a thread so the event loop stays free
    await asyncio.to_thread(load_model)
    yield


app = FastAPI(title="Memories Backend", version="0.2.0", lifespan=lifespan)

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
    return {"status": "ok", "service": "memories-backend"}
