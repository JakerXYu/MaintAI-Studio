"""Integration tests for the P0 MLflow model registry gateway.

Runs against a real MLflow local file backend (``tmp_path``/``mlruns``) with no
network. Records a pipeline through :class:`MLflowTracker`, registers it as a
``candidate`` via :class:`MLflowRegistry`, and verifies version/alias/get/list
behaviour, version increment on a second run, predictable ``models:/`` loading,
strict name validation, champion-alias rejection, stable missing-run handling,
and the healthcheck.
"""

from __future__ import annotations

import contextlib

import mlflow
import numpy as np
import pandas as pd
import pytest
from mlflow.tracking import MlflowClient
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from maintai.ml.preprocess import build_preprocessor
from maintai.mlops import MLflowRegistry, MLflowTracker, RegistryError

EXPERIMENT_NAME = "p0_registry_test"
MODEL_NAME = "failure_model"


def _fit_pipeline():
    rng = np.random.default_rng(0)
    n = 80
    frame = pd.DataFrame(
        {
            "num": rng.normal(0, 1, n),
            "cat": rng.choice(["a", "b"], n),
            "target": (rng.normal(0, 1, n) > 0).astype(int),
        }
    )
    features = ["num", "cat"]
    X = frame[features]
    preprocessor = build_preprocessor(X, features)
    preprocessor.fit(X)
    pipeline = Pipeline(
        [
            ("preprocess", preprocessor),
            ("model", RandomForestClassifier(n_estimators=10, random_state=0)),
        ]
    )
    pipeline.fit(X, frame["target"])
    return pipeline, X


def _log_run(tracker, pipeline, *, f1):
    return tracker.log_model_run(
        model_name=MODEL_NAME,
        task="binary_classification",
        pipeline=pipeline,
        metrics={"f1": f1},
    )


@pytest.fixture()
def registry_uri(tmp_path):
    return (tmp_path / "mlruns").as_uri()


@pytest.fixture()
def tracker(registry_uri):
    return MLflowTracker(registry_uri, EXPERIMENT_NAME)


@pytest.fixture()
def registry(registry_uri):
    return MLflowRegistry(registry_uri)


@contextlib.contextmanager
def _tracking_context(tracking_uri):
    previous = mlflow.get_tracking_uri()
    mlflow.set_tracking_uri(tracking_uri)
    try:
        yield
    finally:
        mlflow.set_tracking_uri(previous)


def test_register_run_creates_candidate_version(registry, tracker):
    pipeline, _ = _fit_pipeline()
    tracked = _log_run(tracker, pipeline, f1=0.9)

    rv = registry.register_run(tracked.run_id, MODEL_NAME)

    assert rv.name == MODEL_NAME
    assert rv.version == "1"
    assert rv.run_id == tracked.run_id
    assert rv.model_uri == f"models:/{MODEL_NAME}/1"
    assert rv.alias == "candidate"
    assert rv.status == "READY"

    # No filesystem paths leak through any returned field.
    for field in (rv.name, rv.version, rv.run_id, rv.model_uri, rv.alias, rv.status):
        assert "file:" not in str(field)

    client = MlflowClient(registry.tracking_uri)
    candidate = client.get_model_version_by_alias(MODEL_NAME, "candidate")
    assert str(candidate.version) == "1"
    assert candidate.run_id == tracked.run_id

    got = registry.get_version(MODEL_NAME, "1")
    assert got.model_uri == f"models:/{MODEL_NAME}/1"
    assert got.alias == "candidate"


def test_second_run_increments_version_and_reassigns_candidate(registry, tracker):
    pipeline, _ = _fit_pipeline()
    first = _log_run(tracker, pipeline, f1=0.5)
    second = _log_run(tracker, pipeline, f1=0.95)

    v1 = registry.register_run(first.run_id, MODEL_NAME)
    v2 = registry.register_run(second.run_id, MODEL_NAME)

    assert v1.version == "1"
    assert v2.version == "2"

    versions = registry.list_versions(MODEL_NAME)
    assert [v.version for v in versions] == ["1", "2"]

    client = MlflowClient(registry.tracking_uri)
    assert str(client.get_model_version_by_alias(MODEL_NAME, "candidate").version) == "2"

    # The candidate alias moved to the latest version; the earlier one is unaliased.
    assert registry.get_version(MODEL_NAME, "1").alias is None
    assert registry.get_version(MODEL_NAME, "2").alias == "candidate"


def test_register_same_run_is_idempotent(registry, tracker):
    pipeline, _ = _fit_pipeline()
    tracked = _log_run(tracker, pipeline, f1=0.9)
    first = registry.register_run(tracked.run_id, MODEL_NAME)
    second = registry.register_run(tracked.run_id, MODEL_NAME)
    assert second.version == first.version
    assert len(registry.list_versions(MODEL_NAME)) == 1


def test_registered_model_loads_predictably(registry, tracker):
    pipeline, X = _fit_pipeline()
    tracked = _log_run(tracker, pipeline, f1=0.9)

    rv = registry.register_run(tracked.run_id, MODEL_NAME)

    with _tracking_context(registry.tracking_uri):
        loaded = mlflow.sklearn.load_model(rv.model_uri)
    np.testing.assert_array_equal(loaded.predict(X), pipeline.predict(X))


@pytest.mark.parametrize(
    "bad_name",
    ["", "bad/name", "bad\\name", "bad name", "bad\x00name", "../escape", "bad*name"],
)
def test_invalid_name_rejected_without_path_leak(registry, bad_name):
    with pytest.raises(RegistryError) as exc_info:
        registry.register_run("some-run-id", bad_name)
    assert "file:" not in str(exc_info.value)


def test_champion_alias_rejected(registry, tracker):
    pipeline, _ = _fit_pipeline()
    tracked = _log_run(tracker, pipeline, f1=0.9)
    registry.register_run(tracked.run_id, MODEL_NAME)

    with pytest.raises(RegistryError, match="approval"):
        registry.set_alias(MODEL_NAME, "champion", "1")

    with pytest.raises(RegistryError, match="candidate"):
        registry.set_alias(MODEL_NAME, "prod", "1")


def test_promote_champion_sets_champion_alias(registry, tracker):
    pipeline, _ = _fit_pipeline()
    tracked = _log_run(tracker, pipeline, f1=0.9)
    registry.register_run(tracked.run_id, MODEL_NAME)

    rv = registry.promote_champion(MODEL_NAME, "1")

    assert rv.alias == "champion"
    assert rv.model_uri == f"models:/{MODEL_NAME}/1"
    client = MlflowClient(registry.tracking_uri)
    assert str(client.get_model_version_by_alias(MODEL_NAME, "champion").version) == "1"


def test_promote_champion_moves_alias_between_versions(registry, tracker):
    pipeline, _ = _fit_pipeline()
    first = _log_run(tracker, pipeline, f1=0.5)
    second = _log_run(tracker, pipeline, f1=0.95)
    registry.register_run(first.run_id, MODEL_NAME)
    registry.register_run(second.run_id, MODEL_NAME)

    registry.promote_champion(MODEL_NAME, "1")
    registry.promote_champion(MODEL_NAME, "2")

    client = MlflowClient(registry.tracking_uri)
    assert str(client.get_model_version_by_alias(MODEL_NAME, "champion").version) == "2"
    assert registry.get_version(MODEL_NAME, "1").alias is None


def test_promote_champion_missing_version_raises_stable_error(registry, tracker):
    pipeline, _ = _fit_pipeline()
    tracked = _log_run(tracker, pipeline, f1=0.9)
    registry.register_run(tracked.run_id, MODEL_NAME)

    with pytest.raises(RegistryError) as exc_info:
        registry.promote_champion(MODEL_NAME, "99")
    assert "file:" not in str(exc_info.value)


def test_missing_run_raises_stable_error(registry):
    with pytest.raises(RegistryError) as exc_info:
        registry.register_run("missing-run-id", MODEL_NAME)
    assert "file:" not in str(exc_info.value)


def test_missing_version_raises_stable_error(registry, tracker):
    pipeline, _ = _fit_pipeline()
    tracked = _log_run(tracker, pipeline, f1=0.9)
    registry.register_run(tracked.run_id, MODEL_NAME)

    with pytest.raises(RegistryError) as exc_info:
        registry.get_version(MODEL_NAME, "99")
    assert "file:" not in str(exc_info.value)


def test_list_versions_unknown_model_is_empty(registry):
    assert registry.list_versions("no_such_model") == []


def test_healthcheck(registry):
    assert registry.healthcheck() is True
