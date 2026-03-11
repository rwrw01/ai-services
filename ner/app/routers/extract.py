import logging

from fastapi import APIRouter, HTTPException
from typing import Literal

from pydantic import BaseModel, Field

from app.engines.robbert import RobBERTEngine

logger = logging.getLogger(__name__)
router = APIRouter()

_engine: RobBERTEngine | None = None


def get_engine() -> RobBERTEngine | None:
    return _engine


def init_engine() -> None:
    global _engine
    _engine = RobBERTEngine()
    _engine.load()


class ExtractRequest(BaseModel):
    domain: Literal["uren", "km"]
    text: str = Field(..., min_length=1, max_length=5_000)


@router.post("/extract")
async def extract(req: ExtractRequest) -> dict:
    engine = get_engine()
    if engine is None:
        raise HTTPException(503, "NER engine not ready")

    try:
        result = await engine.extract(req.text, req.domain)
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception:
        logger.exception("Extraction failed for domain=%s", req.domain)
        raise HTTPException(500, "Extraction failed")

    return result
