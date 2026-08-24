"""MLflow tracking and registry gateways for MaintAI Studio (P0).

Minimal, deterministic wrappers over the MLflow tracking and model-registry
backends used by the single in-process P0 training worker. Both use
``MlflowClient`` directly to avoid the fluent API's global active-run state.
"""

from maintai.mlops.registry import (
    CANDIDATE_ALIAS,
    CHAMPION_ALIAS,
    MLflowRegistry,
    RegistryError,
    RegistryVersion,
)
from maintai.mlops.tracker import (
    MODEL_ARTIFACT_PATH,
    PHASE_TAG,
    PROJECT_TAG,
    MLflowTracker,
    TrackedRun,
    TrackingError,
)

__all__ = [
    "MLflowTracker",
    "TrackedRun",
    "TrackingError",
    "MLflowRegistry",
    "RegistryVersion",
    "RegistryError",
    "PROJECT_TAG",
    "PHASE_TAG",
    "MODEL_ARTIFACT_PATH",
    "CANDIDATE_ALIAS",
    "CHAMPION_ALIAS",
]
