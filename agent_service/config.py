"""Configuration utilities for the agent service."""

from functools import lru_cache
from pydantic import BaseSettings, Field


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    openrouter_api_key: str | None = Field(
        default=None,
        description="API key used to authenticate against OpenRouter endpoints.",
        env="OPENROUTER_API_KEY",
    )
    openrouter_base_url: str = Field(
        default="https://openrouter.ai/api/v1",
        description="Base URL for OpenRouter compatible chat completions endpoint.",
    )
    llm_model: str = Field(
        default="anthropic/claude-3-sonnet",
        description="Default model identifier sent to the LLM provider.",
    )

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache()
def get_settings() -> Settings:
    """Return cached application settings."""

    return Settings()


__all__ = ["Settings", "get_settings"]
