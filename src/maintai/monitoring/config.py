"""Bridge application settings to deterministic monitoring configurations."""

from maintai.config import Settings
from maintai.monitoring.anomaly import AnomalyConfig
from maintai.monitoring.drift import DriftConfig
from maintai.monitoring.recommend import RecommendConfig


def anomaly_config(settings: Settings) -> AnomalyConfig:
    return AnomalyConfig.model_validate(settings.monitoring.anomaly.model_dump())


def drift_config(settings: Settings) -> DriftConfig:
    return DriftConfig.model_validate(settings.monitoring.drift.model_dump())


def recommend_config(settings: Settings) -> RecommendConfig:
    return RecommendConfig.model_validate(settings.monitoring.recommend.model_dump())
