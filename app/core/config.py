import json
from functools import lru_cache
from typing import List

from pydantic.v1 import BaseSettings, Field, validator


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    DATABASE_URL: str = Field("sqlite:///./suicideprevention.db", env="DATABASE_URL")
    SECRET_KEY: str = Field("change-me-in-your-local-env", env="SECRET_KEY")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = Field(30, env="ACCESS_TOKEN_EXPIRE_MINUTES")
    CORS_ORIGINS: List[str] = Field(default_factory=lambda: ["*"], env="CORS_ORIGINS")
    # Resolved by app.ml.common.paths. Common relative values like ../ml_models
    # and ./ml_models map to the repository-level ml_models/ directory.
    MODEL_ROOT: str = Field("../ml_models", env="MODEL_ROOT")
    UPLOAD_ROOT: str = Field("./uploaded_audio", env="UPLOAD_ROOT")
    REDIS_URL: str = Field("redis://localhost:6379/0", env="REDIS_URL")
    ENVIRONMENT: str = Field("development", env="ENVIRONMENT")

    @validator("CORS_ORIGINS", pre=True)
    def parse_cors_origins(cls, value):
        if isinstance(value, str):
            value = value.strip()
            if not value:
                return []
            if value.startswith("["):
                return json.loads(value)
            return [origin.strip() for origin in value.split(",") if origin.strip()]
        return value

    @property
    def is_sqlite(self) -> bool:
        return self.DATABASE_URL.startswith("sqlite")

    @property
    def is_production(self) -> bool:
        return self.ENVIRONMENT.lower() == "production"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True

        @classmethod
        def parse_env_var(cls, field_name, raw_val):
            if field_name == "CORS_ORIGINS":
                return raw_val
            return cls.json_loads(raw_val)


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
