from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings loaded from INFERENCE_LAB_* environment variables."""

    backend: Literal["mock", "openai", "transformers"] = "mock"
    model: str = "Qwen/Qwen3-0.6B"

    upstream_url: str = "http://localhost:8001"
    upstream_api_key: str = "not-needed"
    request_timeout_s: float = Field(default=180.0, gt=0)

    mock_first_token_ms: float = Field(default=40.0, ge=0)
    mock_token_ms: float = Field(default=8.0, ge=0)

    transformers_device: str = "auto"
    transformers_dtype: str = "auto"

    model_config = SettingsConfigDict(
        env_prefix="INFERENCE_LAB_",
        env_file=".env",
        extra="ignore",
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
