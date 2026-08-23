"""MLflow tracking gateway for MaintAI Studio (P0).

Minimal, deterministic wrapper over the MLflow tracking backend used by the
single in-process P0 training worker. Uses ``MlflowClient`` directly to avoid
the fluent API's global active-run state.
"""

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
    "PROJECT_TAG",
    "PHASE_TAG",
    "MODEL_ARTIFACT_PATH",
]
