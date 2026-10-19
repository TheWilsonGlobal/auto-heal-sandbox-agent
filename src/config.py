"""Configuration settings for Auto-Heal Sandbox Agent."""

from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

ENV_PATH = Path(__file__).resolve().parent.parent / ".env"


class Settings(BaseSettings):
    """System-wide configuration settings."""

    # LLM Provider & Settings
    LLM_PROVIDER: str = Field(
        default="gemini",
        description="LLM provider: 'gemini' or 'openai' (local llama.cpp / vLLM)"
    )
    LLM_MODEL: str = Field(
        default="gemini-3.5-flash-lite",
        description="Model identifier (e.g. gemini-3.5-flash-lite)"
    )
    GEMINI_LLM_API_KEY: str = Field(
        default="",
        description="Google Gemini API Key"
    )

    # Legacy / OpenAI-compatible local fallback settings
    LLM_BASE_URL: str = Field(
        default="http://127.0.0.1:5813/v1",
        description="Local OpenAI-compatible base URL for llama.cpp / local model"
    )
    LLM_MODEL_NAME: str = Field(
        default="gemini-3.5-flash-lite",
        description="Target model identifier (backward compatibility)"
    )
    LLM_API_KEY: str = Field(
        default="local",
        description="API key for LLM endpoint (dummy for local inference)"
    )
    LLM_TEMPERATURE: float = Field(
        default=0.1,
        description="Sampling temperature for deterministic code patches"
    )

    # Sandbox Constraints
    SANDBOX_IMAGE: str = Field(
        default="auto-heal-sandbox:latest",
        description="Docker image to run the sandbox containers"
    )
    SANDBOX_FALLBACK_IMAGE: str = Field(
        default="python:3.11-slim",
        description="Fallback Docker image if custom image is not built"
    )
    SANDBOX_MEM_LIMIT: str = Field(
        default="512m",
        description="Container memory limit"
    )
    SANDBOX_NANO_CPUS: int = Field(
        default=1_000_000_000,
        description="Container CPU limit (1 core = 1e9 nano cpus)"
    )
    SANDBOX_NETWORK_MODE: str = Field(
        default="none",
        description="Strict network isolation mode (zero internet)"
    )
    SANDBOX_USER: str = Field(
        default="nobody",
        description="Container execution user for defense-in-depth"
    )
    SANDBOX_TIMEOUT_SECONDS: int = Field(
        default=10,
        description="Timeout execution per command in seconds"
    )

    # Agent Loop Parameters
    MAX_RETRIES: int = Field(
        default=4,
        description="Maximum iterations before terminating healing loop"
    )
    TRACEBACK_MAX_LINES: int = Field(
        default=30,
        description="Number of stderr/stdout traceback lines to retain for model context"
    )


    # FastMCP Server settings
    MCP_SERVER_NAME: str = Field(
        default="auto-heal-sandbox",
        description="MCP Server registered name"
    )

    model_config = SettingsConfigDict(
        env_file=str(ENV_PATH) if ENV_PATH.exists() else ".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )


# Singleton instance
settings = Settings()
