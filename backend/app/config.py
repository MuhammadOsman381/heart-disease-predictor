from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    app_name: str = "Heart Disease ML API"
    database_url: str = f"sqlite://{BASE_DIR / 'local.db'}"
    supabase_url: str = ""
    supabase_key: str = ""
    supabase_bucket: str = "heart-model-artifacts"
    artifact_dir: Path = BASE_DIR / "artifacts"
    groq_api_key: str = ""
    groq_model: str = "llama-3.3-70b-versatile"

    model_config = SettingsConfigDict(
        env_file=BASE_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    settings.artifact_dir.mkdir(parents=True, exist_ok=True)
    return settings
