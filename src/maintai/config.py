"""Application configuration.

Settings are loaded with ``pydantic-settings``. Priority (highest first):
init args > environment variables > ``.env`` > ``configs/default.yaml`` >
field defaults. Secrets are never hard-coded and never printed.

The default YAML/env file paths are discovered by walking up from this file to
the project root, so they resolve correctly both in the source tree and under
an editable install. Override with ``MAINTAI_CONFIG_FILE`` if needed.
"""

from __future__ import annotations

import os
from functools import cache
from pathlib import Path

from pydantic import SecretStr, field_validator
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    YamlConfigSettingsSource,
)


def _project_root() -> Path:
    """Locate the repo root by walking up to a ``configs`` dir or pyproject.toml."""
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "configs").is_dir() or (parent / "pyproject.toml").is_file():
            return parent
    # Fallback for non-editable installs (source-layout root).
    return here.parents[2]


def _find_config_file() -> Path:
    override = os.getenv("MAINTAI_CONFIG_FILE")
    if override:
        return Path(override)
    return _project_root() / "configs" / "default.yaml"


def _find_env_file() -> Path:
    return _project_root() / ".env"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(_find_env_file()),
        yaml_file=str(_find_config_file()),
        yaml_file_encoding="utf-8",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    project_name: str = "MaintAI Studio"
    environment: str = "dev"
    random_seed: int = 42

    # Business database. Runtime uses Postgres; tests override with SQLite.
    database_url: str = "sqlite:///./maintai.db"
    mlflow_tracking_uri: str = "http://localhost:5000"
    dataset_storage_path: Path = Path("data/uploads")

    # LLM provider (mock default = offline).
    llm_provider: str = "mock"
    llm_api_key: SecretStr | None = None
    llm_base_url: str | None = None
    llm_model: str | None = None

    api_host: str = "0.0.0.0"
    api_port: int = 8000
    request_id_header: str = "X-Request-ID"
    max_upload_mb: int = 200

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        """Load YAML below environment sources and above field defaults."""
        return (
            init_settings,
            env_settings,
            dotenv_settings,
            YamlConfigSettingsSource(settings_cls),
            file_secret_settings,
        )

    @field_validator("llm_api_key", "llm_base_url", "llm_model", mode="before")
    @classmethod
    def _empty_string_to_none(cls, value: object) -> object:
        if isinstance(value, str) and value.strip() == "":
            return None
        return value


@cache
def get_settings() -> Settings:
    return Settings()
