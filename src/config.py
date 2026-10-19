"""Configuration settings for Auto-Heal Sandbox Agent."""

import sys
from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from constants import (
    DEFAULT_LLM_PROVIDER,
    DEFAULT_LLM_MODEL,
    DEFAULT_LLM_BASE_URL,
    DEFAULT_LLM_MODEL_NAME,
    DEFAULT_LLM_API_KEY,
    DEFAULT_LLM_TEMPERATURE,
    SANDBOX_IMAGE,
    SANDBOX_FALLBACK_IMAGE,
    SANDBOX_MEM_LIMIT,
    SANDBOX_NANO_CPUS,
    SANDBOX_NETWORK_MODE,
    SANDBOX_USER,
    SANDBOX_TIMEOUT_SECONDS,
    DEFAULT_MAX_RETRIES,
    DEFAULT_TRACEBACK_MAX_LINES,
    DEFAULT_MCP_SERVER_NAME,
)

ENV_PATH = REPO_ROOT / ".env"


class Settings(BaseSettings):
    """System-wide configuration settings."""

    # LLM Provider & Settings
    LLM_PROVIDER: str = Field(
        default=DEFAULT_LLM_PROVIDER,
        description="LLM provider: 'gemini' or 'openai' (local llama.cpp / vLLM)"
    )
    LLM_MODEL: str = Field(
        default=DEFAULT_LLM_MODEL,
        description="Model identifier (e.g. gemini-3.5-flash-lite)"
    )
    GEMINI_LLM_API_KEY: str = Field(
        default="",
        description="Google Gemini API Key"
    )

    # Legacy / OpenAI-compatible local fallback settings
    LLM_BASE_URL: str = Field(
        default=DEFAULT_LLM_BASE_URL,
        description="Local OpenAI-compatible base URL for llama.cpp / local model"
    )
    LLM_MODEL_NAME: str = Field(
        default=DEFAULT_LLM_MODEL_NAME,
        description="Target model identifier (backward compatibility)"
    )
    LLM_API_KEY: str = Field(
        default=DEFAULT_LLM_API_KEY,
        description="API key for LLM endpoint (dummy for local inference)"
    )
    LLM_TEMPERATURE: float = Field(
        default=DEFAULT_LLM_TEMPERATURE,
        description="Sampling temperature for deterministic code patches"
    )

    # Sandbox Constraints
    SANDBOX_IMAGE: str = Field(
        default=SANDBOX_IMAGE,
        description="Docker image to run the sandbox containers"
    )
    SANDBOX_FALLBACK_IMAGE: str = Field(
        default=SANDBOX_FALLBACK_IMAGE,
        description="Fallback Docker image if custom image is not built"
    )
    SANDBOX_MEM_LIMIT: str = Field(
        default=SANDBOX_MEM_LIMIT,
        description="Container memory limit"
    )
    SANDBOX_NANO_CPUS: int = Field(
        default=SANDBOX_NANO_CPUS,
        description="Container CPU limit (1 core = 1e9 nano cpus)"
    )
    SANDBOX_NETWORK_MODE: str = Field(
        default=SANDBOX_NETWORK_MODE,
        description="Strict network isolation mode (zero internet)"
    )
    SANDBOX_USER: str = Field(
        default=SANDBOX_USER,
        description="Container execution user for defense-in-depth"
    )
    SANDBOX_TIMEOUT_SECONDS: int = Field(
        default=SANDBOX_TIMEOUT_SECONDS,
        description="Timeout execution per command in seconds"
    )

    # Agent Loop Parameters
    MAX_RETRIES: int = Field(
        default=DEFAULT_MAX_RETRIES,
        description="Maximum iterations before terminating healing loop"
    )
    TRACEBACK_MAX_LINES: int = Field(
        default=DEFAULT_TRACEBACK_MAX_LINES,
        description="Number of stderr/stdout traceback lines to retain for model context"
    )

    # FastMCP Server settings
    MCP_SERVER_NAME: str = Field(
        default=DEFAULT_MCP_SERVER_NAME,
        description="MCP Server registered name"
    )


    model_config = SettingsConfigDict(
        env_file=str(ENV_PATH) if ENV_PATH.exists() else ".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )


# Singleton instance
settings = Settings()
