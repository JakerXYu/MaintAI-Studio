"""Deterministic P1 monitoring core for MaintAI Studio.

Pure Python + numpy/pandas/scikit-learn/scipy. Anomaly detection, drift
detection, synthetic production replay, and retraining recommendations. All
results are JSON-safe Pydantic contracts; nothing here trains, deploys, or
persists — those transitions stay behind human approval (see AGENTS.md).
"""

from maintai.monitoring.config import anomaly_config, drift_config, recommend_config

__all__ = ["anomaly_config", "drift_config", "recommend_config"]
