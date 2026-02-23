from pydantic import BaseModel, field_validator


class SynthesizeRequest(BaseModel):
    text: str
    engine: str = "auto"   # "piper" | "parkiet" | "auto"
    voice: str = "default"

    @field_validator("engine")
    @classmethod
    def validate_engine(cls, v: str) -> str:
        allowed = {"piper", "parkiet", "auto"}
        if v not in allowed:
            raise ValueError(f"engine must be one of {allowed}")
        return v

    @field_validator("text")
    @classmethod
    def validate_text(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("text may not be empty")
        if len(v) > 5000:
            raise ValueError("text too long (max 5000 characters)")
        return v


class EngineInfo(BaseModel):
    id: str
    available: bool
    quality: str   # "basic" | "high"
    speed: str     # "fast" | "slow"


class EnginesResponse(BaseModel):
    engines: list[EngineInfo]
    default: str
