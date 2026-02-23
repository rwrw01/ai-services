from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    tts_piper_enabled: bool = True
    tts_parkiet_enabled: bool = True
    tts_piper_model: str = "nl_BE-nathalie-medium"
    tts_default_engine: str = "piper"
    tts_cache_ttl_days: int = 7
    tts_cache_dir: str = "/data/tts-cache"
    tts_models_dir: str = "/app/models"

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()
