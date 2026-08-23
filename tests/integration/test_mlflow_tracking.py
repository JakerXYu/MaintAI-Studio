"""Integration tests for the P0 MLflow tracking gateway.

Runs against a real MLflow local file backend (``tmp_path``/``mlruns``) with no
network. Verifies params/metrics/tags/artifacts/model round-trip, ``None``
metric skipping, run isolation, stable failure handling (run ``FAILED``, no
active-run residue), ``get_run``, and ``healthcheck``.
"""

from __future__ import annotations

import contextlib
import json
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import pytest
from mlflow.tracking import MlflowClient
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from maintai.ml.preprocess import build_preprocessor
from maintai.mlops import MLflowTracker, TrackingError

EXPERIMENT_NAME = "p0_tracking_test"


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


@pytest.fixture()
def tracking_uri(tmp_path):
    return (tmp_path / "mlruns").as_uri()


@pytest.fixture()
def tracker(tracking_uri):
    return MLflowTracker(tracking_uri, EXPERIMENT_NAME)


@contextlib.contextmanager
def _tracking_context(tracking_uri):
    previous = mlflow.get_tracking_uri()
    mlflow.set_tracking_uri(tracking_uri)
    try:
        yield
    finally:
        mlflow.set_tracking_uri(previous)


def test_log_model_run_records_everything_and_model_loads(tracker):
    pipeline, X = _fit_pipeline()

    tracked = tracker.log_model_run(
        model_name="random_forest",
        task="binary_classification",
        pipeline=pipeline,
        params={
            "seed": 42,
            "n_estimators": 10,
            "np_scalar": np.float64(1.5),
            "features": ["num", "cat"],
            "config": {"a": 1, "b": [1, 2, 3]},
        },
        metrics={"f1": 0.9, "training_time": 1.23, "inference_latency": 0.05, "pr_auc": None},
        tags={"model_family": "random_forest", "dataset_name": "demo"},
        artifacts={
            "evaluation.json": {"f1": 0.9, "confusion_matrix": [[1, 2], [3, 4]]},
            "feature_names.json": ["num", "cat"],
        },
    )

    assert tracked.run_id
    assert tracked.status == "FINISHED"
    assert tracked.model_uri == f"runs:/{tracked.run_id}/model"
    assert tracked.artifact_uri.endswith(f"/{tracked.run_id}/artifacts")

    client = MlflowClient(tracker.tracking_uri)
    run = client.get_run(tracked.run_id)
    data = run.data

    assert data.params["seed"] == "42"
    assert data.params["np_scalar"] == "1.5"
    assert data.params["features"] == '["num","cat"]'
    assert data.params["config"] == '{"a":1,"b":[1,2,3]}'

    assert data.metrics["f1"] == 0.9
    assert data.metrics["training_time"] == 1.23
    assert data.metrics["inference_latency"] == 0.05
    assert "pr_auc" not in data.metrics

    assert data.tags["project"] == "maintai-studio"
    assert data.tags["phase"] == "P0"
    assert data.tags["model_family"] == "random_forest"
    assert data.tags["model_name"] == "random_forest"
    assert data.tags["task"] == "binary_classification"
    assert data.tags["dataset_name"] == "demo"

    artifact_paths = {info.path for info in client.list_artifacts(tracked.run_id)}
    assert {"model", "artifacts"} <= artifact_paths
    model_files = {info.path for info in client.list_artifacts(tracked.run_id, path="model")}
    assert "model/MLmodel" in model_files
    json_files = {info.path for info in client.list_artifacts(tracked.run_id, path="artifacts")}
    assert {
        "artifacts/evaluation.json",
        "artifacts/feature_names.json",
    } <= json_files

    evaluation_path = client.download_artifacts(tracked.run_id, "artifacts/evaluation.json")
    evaluation = json.loads(Path(evaluation_path).read_text(encoding="utf-8"))
    assert evaluation["f1"] == 0.9

    with _tracking_context(tracker.tracking_uri):
        loaded = mlflow.sklearn.load_model(tracked.model_uri)
    np.testing.assert_array_equal(loaded.predict(X), pipeline.predict(X))


def test_none_metric_is_skipped(tracker):
    pipeline, _ = _fit_pipeline()
    tracked = tracker.log_model_run(
        model_name="random_forest",
        task="binary_classification",
        pipeline=pipeline,
        metrics={"f1": 0.8, "roc_auc": None, "pr_auc": None},
    )
    data = MlflowClient(tracker.tracking_uri).get_run(tracked.run_id).data
    assert data.metrics == {"f1": 0.8}


def test_two_runs_are_isolated(tracker):
    pipeline, _ = _fit_pipeline()
    first = tracker.log_model_run(
        model_name="random_forest",
        task="binary_classification",
        pipeline=pipeline,
        params={"marker": "one"},
        metrics={"f1": 0.5},
    )
    second = tracker.log_model_run(
        model_name="random_forest",
        task="binary_classification",
        pipeline=pipeline,
        params={"marker": "two"},
        metrics={"f1": 0.95},
    )

    assert first.run_id != second.run_id
    assert first.experiment_id == second.experiment_id

    client = MlflowClient(tracker.tracking_uri)
    first_data = client.get_run(first.run_id).data
    second_data = client.get_run(second.run_id).data
    assert first_data.params["marker"] == "one"
    assert second_data.params["marker"] == "two"
    assert first_data.metrics["f1"] == 0.5
    assert second_data.metrics["f1"] == 0.95


def test_failed_pipeline_marks_run_failed_without_active_run(tracker):
    with pytest.raises(TrackingError, match="pipeline must be a fitted model"):
        tracker.log_model_run(
            model_name="broken",
            task="binary_classification",
            pipeline=None,
            metrics={"f1": 0.9},
        )

    assert mlflow.active_run() is None

    client = MlflowClient(tracker.tracking_uri)
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    runs = client.search_runs(
        [experiment.experiment_id], order_by=["start_time DESC"], max_results=1
    )
    assert runs[0].info.status == "FAILED"


def test_invalid_artifact_raises_stable_error_without_active_run(tracker):
    pipeline, _ = _fit_pipeline()
    with pytest.raises(TrackingError, match="MLflow tracking failed"):
        tracker.log_model_run(
            model_name="random_forest",
            task="binary_classification",
            pipeline=pipeline,
            artifacts={"evaluation.json": object()},
        )

    assert mlflow.active_run() is None

    client = MlflowClient(tracker.tracking_uri)
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    runs = client.search_runs(
        [experiment.experiment_id], order_by=["start_time DESC"], max_results=1
    )
    assert runs[0].info.status == "FAILED"


def test_get_run_and_healthcheck(tracker):
    pipeline, _ = _fit_pipeline()
    tracked = tracker.log_model_run(
        model_name="random_forest",
        task="binary_classification",
        pipeline=pipeline,
        metrics={"f1": 0.7},
    )

    run = tracker.get_run(tracked.run_id)
    assert run.info.run_id == tracked.run_id
    assert run.data.metrics["f1"] == 0.7
    assert tracker.healthcheck() is True

    with pytest.raises(TrackingError, match="get_run failed"):
        tracker.get_run("missing-run-id")


def test_long_param_is_truncated(tracker):
    pipeline, _ = _fit_pipeline()
    tracked = tracker.log_model_run(
        model_name="random_forest",
        task="binary_classification",
        pipeline=pipeline,
        params={"long": "x" * 2000},
    )
    data = MlflowClient(tracker.tracking_uri).get_run(tracked.run_id).data
    assert data.params["long"].endswith("...[truncated:2000]")
    assert len(data.params["long"]) <= 500 + len("...[truncated:2000]")
