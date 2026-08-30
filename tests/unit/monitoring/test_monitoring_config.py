"""Configuration tests for the monitoring settings (pydantic-settings)."""

from pathlib import Path

import yaml

from maintai.config import Settings
from maintai.monitoring.config import anomaly_config, drift_config, recommend_config


def test_monitoring_defaults_from_yaml():
    s = Settings()
    assert s.monitoring.anomaly.threshold == 3.0
    assert s.monitoring.anomaly.contamination == "auto"
    assert s.monitoring.anomaly.seed == 42
    assert s.monitoring.anomaly.top_k_features == 3
    assert s.monitoring.anomaly.n_estimators == 100
    assert s.monitoring.drift.psi_medium == 0.10
    assert s.monitoring.drift.psi_high == 0.25
    assert s.monitoring.drift.ks_medium == 0.10
    assert s.monitoring.drift.ks_high == 0.20
    assert s.monitoring.drift.shift_medium == 0.50
    assert s.monitoring.drift.shift_high == 1.00
    assert s.monitoring.drift.missingness_medium == 0.05
    assert s.monitoring.drift.missingness_high == 0.15
    assert s.monitoring.drift.n_bins == 10
    assert s.monitoring.recommend.performance_drop_threshold == 0.05
    assert s.monitoring.recommend.anomaly_rate_jump_threshold == 0.10
    assert s.monitoring.recommend.schedule_days_threshold == 30


def test_monitoring_init_override():
    s = Settings(
        monitoring={
            "drift": {"psi_high": 0.99},
            "anomaly": {"threshold": 5.5},
        }
    )
    assert s.monitoring.drift.psi_high == 0.99
    assert s.monitoring.anomaly.threshold == 5.5
    # untouched values keep defaults
    assert s.monitoring.drift.psi_medium == 0.10


def test_monitoring_env_override(monkeypatch):
    monkeypatch.setenv("MONITORING__DRIFT__PSI_HIGH", "0.77")
    s = Settings(_env_file=None)
    assert s.monitoring.drift.psi_high == 0.77


def test_settings_bridge_to_runtime_configs():
    settings = Settings(
        monitoring={
            "anomaly": {"threshold": 5.5, "n_estimators": 25},
            "drift": {"psi_high": 0.77, "missingness_high": 0.22},
            "recommend": {"performance_drop_threshold": 0.12},
        }
    )
    assert anomaly_config(settings).threshold == 5.5
    assert anomaly_config(settings).n_estimators == 25
    assert drift_config(settings).psi_high == 0.77
    assert drift_config(settings).missingness_high == 0.22
    assert recommend_config(settings).performance_drop_threshold == 0.12


def test_default_yaml_contains_monitoring_block():
    path = Path(__file__).resolve().parents[3] / "configs" / "default.yaml"
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    monitoring = data["monitoring"]
    assert monitoring["anomaly"]["threshold"] == 3.0
    assert monitoring["drift"]["psi_high"] == 0.25
    assert monitoring["drift"]["ks_high"] == 0.20
    assert monitoring["drift"]["shift_high"] == 1.00
    assert monitoring["drift"]["missingness_high"] == 0.15
    assert monitoring["recommend"]["schedule_days_threshold"] == 30
