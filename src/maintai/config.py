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
from typing import Literal

from pydantic import BaseModel, SecretStr, field_validator
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


class DataRulesSettings(BaseModel):
    missing_warning_rate: float = 0.05
    missing_severe_rate: float = 0.20
    duplicate_row_warning_rate: float = 0.01
    duplicate_row_error_rate: float = 0.10
    near_constant_unique_ratio: float = 0.001
    flatline_min_length: int = 10
    flatline_min_ratio: float = 0.05
    outlier_mad_threshold: float = 3.5
    outlier_severe_rate: float = 0.10
    timestamp_gap_factor: float = 5.0
    sampling_cv_threshold: float = 1.0
    asset_imbalance_ratio: float = 10.0
    target_imbalance_ratio: float = 20.0
    min_trainable_samples: int = 100


class SplitSettings(BaseModel):
    test_size: float = 0.20
    validation_size: float = 0.20
    random_state: int = 42


class MLRulesSettings(BaseModel):
    seed: int = 42
    n_jobs: int = 1
    scale_numeric: bool = True
    minimum_recall: float = 0.80


class AnomalySettings(BaseModel):
    seed: int = 42
    contamination: float | str = "auto"
    threshold: float = 3.0
    top_k_features: int = 3
    n_estimators: int = 100


class DriftSettings(BaseModel):
    psi_medium: float = 0.10
    psi_high: float = 0.25
    ks_medium: float = 0.10
    ks_high: float = 0.20
    shift_medium: float = 0.50
    shift_high: float = 1.00
    missingness_medium: float = 0.05
    missingness_high: float = 0.15
    n_bins: int = 10
    epsilon: float = 1e-6


class RecommendSettings(BaseModel):
    performance_drop_threshold: float = 0.05
    anomaly_rate_jump_threshold: float = 0.10
    schedule_days_threshold: int = 30


class MonitoringSettings(BaseModel):
    anomaly: AnomalySettings = AnomalySettings()
    drift: DriftSettings = DriftSettings()
    recommend: RecommendSettings = RecommendSettings()


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(_find_env_file()),
        yaml_file=str(_find_config_file()),
        yaml_file_encoding="utf-8",
        env_file_encoding="utf-8",
        case_sensitive=False,
        env_nested_delimiter="__",
        extra="ignore",
    )

    project_name: str = "MaintAI Studio"
    environment: str = "dev"
    random_seed: int = 42

    # Business database. Runtime uses Postgres; tests override with SQLite.
    database_url: str = "sqlite:///./maintai.db"
    mlflow_tracking_uri: str = "http://localhost:5000"
    mlflow_experiment_name: str = "maintai-p0"
    dataset_storage_path: Path = Path("data/uploads")
    artifact_storage_path: Path = Path("data/artifacts")

    # LLM provider (mock default = offline).
    llm_provider: str = "mock"
    llm_api_key: SecretStr | None = None
    llm_base_url: str | None = None
    llm_model: str | None = None
    llm_thinking: Literal["enabled", "disabled"] | None = None

    api_host: str = "0.0.0.0"
    api_port: int = 8000
    request_id_header: str = "X-Request-ID"
    max_upload_mb: int = 200

    # P1 human-approval gate. Environment-only: never set a real value in YAML.
    # When unset, the approval decision endpoints return 503 (decisions disabled).
    approval_api_token: SecretStr | None = None
    data: DataRulesSettings = DataRulesSettings()
    split: SplitSettings = SplitSettings()
    ml: MLRulesSettings = MLRulesSettings()
    monitoring: MonitoringSettings = MonitoringSettings()

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

    @field_validator(
        "llm_api_key",
        "llm_base_url",
        "llm_model",
        "llm_thinking",
        "approval_api_token",
        mode="before",
    )
    @classmethod
    def _empty_string_to_none(cls, value: object) -> object:
        if isinstance(value, str) and value.strip() == "":
            return None
        return value


@cache
def get_settings() -> Settings:
    return Settings()
