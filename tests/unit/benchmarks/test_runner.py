"""Scania APS benchmark runner tests (tiny synthetic data, fake train)."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from maintai.benchmarks import runner
from maintai.benchmarks.runner import (
    BenchmarkConfig,
    BenchmarkConfigError,
    parse_benchmark_config,
    run_benchmark,
)
from maintai.benchmarks.scania_aps import (
    PREPARED_TEST_FILENAME,
    PREPARED_TRAIN_FILENAME,
    SUMMARY_FILENAME,
)
from maintai.data.profile import profile as real_profile
from maintai.data.quality import assess as real_assess
from maintai.data.schema import infer_schema as real_infer_schema
from maintai.ml.schemas import ModelEvaluation, TrainingResult
from maintai.ml.train import TrainingOutput
from maintai.tasks.infer import recommend_task as real_recommend_task

_MODELS = ("logistic_regression", "random_forest", "xgboost")


def _config(train_rows=6, test_rows=4, n_features=2, **overrides) -> BenchmarkConfig:
    values = dict(
        target="aps_failure",
        n_features=n_features,
        train_rows=train_rows,
        test_rows=test_rows,
        fp_cost=10.0,
        fn_cost=500.0,
        currency_label="challenge cost units",
        threshold=0.5,
        task="binary_classification",
        model_names=_MODELS,
    )
    values.update(overrides)
    return BenchmarkConfig(**values)


def _frame(n_rows, n_features=2, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    y = np.zeros(n_rows, dtype=int)
    y[: n_rows // 2] = 1
    rng.shuffle(y)
    data = {"aps_failure": y}
    for i in range(n_features):
        data[f"f{i}"] = rng.normal(0, 1, n_rows) + y * 1.0
    return pd.DataFrame(data)


def _write_prepared(prepared_dir: Path, train, test, *, with_summary=True) -> None:
    prepared_dir.mkdir(parents=True, exist_ok=True)
    train.to_csv(prepared_dir / PREPARED_TRAIN_FILENAME, index=False)
    test.to_csv(prepared_dir / PREPARED_TEST_FILENAME, index=False)
    if with_summary:
        n_features = len(train.columns) - 1
        summary = {
            "dataset": "scania_aps",
            "target": "aps_failure",
            "n_features": n_features,
            "prepared_at": "2026-01-01T00:00:00+00:00",
            "splits": {
                "train": {
                    "path": str((prepared_dir / PREPARED_TRAIN_FILENAME).resolve()),
                    "shape": [len(train), n_features + 1],
                },
                "test": {
                    "path": str((prepared_dir / PREPARED_TEST_FILENAME).resolve()),
                    "shape": [len(test), n_features + 1],
                },
            },
        }
        (prepared_dir / SUMMARY_FILENAME).write_text(
            json.dumps(summary), encoding="utf-8"
        )


def _fake_train(captured: dict):
    def fake_train(frame, plan):
        captured["frame"] = frame
        captured["plan"] = plan
        evaluations = [
            ModelEvaluation(
                model_name="logistic_regression",
                task="binary_classification",
                status="success",
                metrics={
                    "precision": 0.6,
                    "recall": 0.5,
                    "f1": 0.5455,
                    "roc_auc": 0.8,
                    "pr_auc": 0.7,
                },
                confusion_matrix=[[40, 10], [5, 45]],
                labels=[0, 1],
                positive_label=1,
                inference_latency_ms=1.0,
                training_time_seconds=0.1,
            ),
            ModelEvaluation(
                model_name="random_forest",
                task="binary_classification",
                status="success",
                metrics={
                    "precision": 0.8,
                    "recall": 0.7,
                    "f1": 0.746,
                    "roc_auc": 0.9,
                    "pr_auc": 0.9,
                },
                confusion_matrix=[[48, 2], [8, 42]],
                labels=[0, 1],
                positive_label=1,
                inference_latency_ms=2.0,
                training_time_seconds=0.2,
            ),
            ModelEvaluation(
                model_name="xgboost",
                task="binary_classification",
                status="success",
                metrics={
                    "precision": 0.7,
                    "recall": 0.6,
                    "f1": 0.646,
                    "roc_auc": 0.85,
                    "pr_auc": 0.85,
                },
                confusion_matrix=[[45, 5], [3, 47]],
                labels=[0, 1],
                positive_label=1,
                inference_latency_ms=3.0,
                training_time_seconds=0.3,
            ),
        ]
        result = TrainingResult(
            task="binary_classification",
            target="aps_failure",
            seed=42,
            primary_metric="pr_auc",
            evaluations=evaluations,
            errors=[],
        )
        return TrainingOutput(result=result, models={})

    return fake_train


def _run(tmp_path, monkeypatch, *, train_rows=6, test_rows=4):
    cfg = _config(train_rows=train_rows, test_rows=test_rows)
    train = _frame(train_rows)
    test = _frame(test_rows, seed=7)
    prepared = tmp_path / "prepared"
    _write_prepared(prepared, train, test)
    captured: dict = {}
    monkeypatch.setattr(runner, "train", _fake_train(captured))
    artifacts = tmp_path / "artifacts"
    summary = run_benchmark(prepared_dir=prepared, artifacts_dir=artifacts, config=cfg)
    return cfg, train, test, captured, artifacts, summary


# --- config parsing -------------------------------------------------------------


def test_config_parses_matching_adapter_constants():
    cfg = parse_benchmark_config(runner.CONFIG_PATH)
    assert cfg.target == "aps_failure"
    assert cfg.n_features == 170
    assert cfg.train_rows == 60000
    assert cfg.test_rows == 16000
    assert cfg.fp_cost == 10.0
    assert cfg.fn_cost == 500.0
    assert cfg.threshold == 0.5
    assert cfg.task == "binary_classification"
    assert list(cfg.model_names) == ["logistic_regression", "random_forest", "xgboost"]


def _config_dict() -> dict:
    return {
        "dataset": {
            "name": "scania_aps",
            "target": "aps_failure",
            "n_features": 170,
            "train_rows": 60000,
            "test_rows": 16000,
        },
        "costs": {"fp_cost": 10, "fn_cost": 500, "currency_label": "challenge cost units"},
        "threshold": 0.5,
        "models": {
            "task": "binary_classification",
            "catalog": ["logistic_regression", "random_forest", "xgboost"],
        },
    }


def _write_config(tmp_path, mutate) -> Path:
    data = _config_dict()
    mutate(data)
    path = tmp_path / "scania_aps.yaml"
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return path


@pytest.mark.parametrize(
    "mutate",
    [
        lambda d: d["dataset"].__setitem__("n_features", 169),
        lambda d: d["dataset"].__setitem__("train_rows", 60001),
        lambda d: d["dataset"].__setitem__("target", "failure"),
        lambda d: d["costs"].__setitem__("fp_cost", 11),
        lambda d: d.__setitem__("threshold", 0.6),
        lambda d: d["models"].__setitem__("task", "regression"),
        lambda d: d["models"].__setitem__(
            "catalog", ["logistic_regression", "random_forest"]
        ),
    ],
)
def test_config_mismatch_rejected(tmp_path, mutate):
    with pytest.raises(BenchmarkConfigError, match="mismatch"):
        parse_benchmark_config(_write_config(tmp_path, mutate))


def test_config_missing_key_rejected(tmp_path):
    path = _write_config(tmp_path, lambda d: d["dataset"].pop("n_features"))
    with pytest.raises(BenchmarkConfigError, match="missing"):
        parse_benchmark_config(path)


# --- official split -------------------------------------------------------------


def test_run_uses_official_split_and_no_reshuffle(tmp_path, monkeypatch):
    cfg, train, test, captured, _, summary = _run(tmp_path, monkeypatch)
    plan = captured["plan"]
    assert plan.split.strategy == "official_holdout"
    assert plan.split.train_indices == list(range(6))
    assert plan.split.test_indices == list(range(6, 10))
    assert plan.split.validation_indices == []
    assert plan.target == "aps_failure"
    assert plan.task == "binary_classification"
    assert plan.primary_metric == "pr_auc"
    assert plan.seed == 42
    assert plan.model_names == ["logistic_regression", "random_forest", "xgboost"]

    frame = captured["frame"]
    assert len(frame) == 10
    assert frame.iloc[0]["aps_failure"] == train.iloc[0]["aps_failure"]
    assert frame.iloc[9]["aps_failure"] == test.iloc[3]["aps_failure"]
    assert summary["official_split"] == {"train_rows": 6, "test_rows": 4}


def test_run_profiles_on_train_split_only(tmp_path, monkeypatch):
    cfg = _config(train_rows=6, test_rows=4)
    train = _frame(6)
    test = _frame(4, seed=7)
    prepared = tmp_path / "prepared"
    _write_prepared(prepared, train, test)
    monkeypatch.setattr(runner, "train", _fake_train({}))

    seen: dict[str, int] = {}

    def spy_infer(frame):
        seen["infer"] = len(frame)
        return real_infer_schema(frame)

    def spy_profile(frame, schema=None, top_n=10):
        seen["profile"] = len(frame)
        return real_profile(frame, schema, top_n)

    def spy_assess(frame, schema=None, **kwargs):
        seen["assess"] = len(frame)
        return real_assess(frame, schema, **kwargs)

    def spy_task(frame, target_col=None):
        seen["task"] = len(frame)
        return real_recommend_task(frame, target_col)

    monkeypatch.setattr(runner, "infer_schema", spy_infer)
    monkeypatch.setattr(runner, "profile", spy_profile)
    monkeypatch.setattr(runner, "assess", spy_assess)
    monkeypatch.setattr(runner, "recommend_task", spy_task)

    run_benchmark(prepared_dir=prepared, artifacts_dir=tmp_path / "artifacts", config=cfg)
    assert seen == {"infer": 6, "profile": 6, "assess": 6, "task": 6}


# --- serialization ---------------------------------------------------------------


def test_metrics_and_cost_serialization(tmp_path, monkeypatch):
    _, _, _, _, artifacts, _ = _run(tmp_path, monkeypatch)

    metrics = json.loads((artifacts / "model_metrics.json").read_text(encoding="utf-8"))
    assert metrics["threshold"] == 0.5
    assert metrics["threshold_optimized"] is False
    assert metrics["primary_metric"] == "pr_auc"
    assert metrics["recommended_model"] == "random_forest"
    models = {m["model_name"]: m for m in metrics["models"]}
    lr = models["logistic_regression"]
    assert (lr["tn"], lr["fp"], lr["fn"], lr["tp"]) == (40, 10, 5, 45)
    assert lr["precision"] == 0.6
    assert lr["recall"] == 0.5
    assert lr["roc_auc"] == 0.8
    assert lr["pr_auc"] == 0.7
    assert lr["threshold"] == 0.5
    assert lr["training_time_seconds"] == 0.1
    assert lr["inference_latency_ms"] == 1.0

    cm = json.loads((artifacts / "confusion_matrices.json").read_text(encoding="utf-8"))
    assert cm["models"]["logistic_regression"]["matrix"] == [[40, 10], [5, 45]]
    assert cm["models"]["xgboost"]["tn"] == 45
    assert cm["models"]["xgboost"]["tp"] == 47

    cost = json.loads((artifacts / "cost_comparison.json").read_text(encoding="utf-8"))
    assert cost["metric_best"] == "random_forest"
    assert cost["cost_best"] == "xgboost"
    assert "official IDA 2016 challenge economics" in cost["notes"][0]
    assert cost["assumptions"] == {
        "fn_cost": 500.0,
        "fp_cost": 10.0,
        "currency_label": "challenge cost units",
    }
    rows = {r["model_name"]: r for r in cost["rows"]}
    assert rows["logistic_regression"]["expected_error_cost"] == 5 * 500 + 10 * 10
    assert rows["random_forest"]["expected_error_cost"] == 8 * 500 + 2 * 10
    assert rows["xgboost"]["expected_error_cost"] == 3 * 500 + 5 * 10


def test_output_files_exist(tmp_path, monkeypatch):
    _, _, _, _, artifacts, _ = _run(tmp_path, monkeypatch)
    for name in [
        "dataset_summary.json",
        "model_metrics.json",
        "model_metrics.csv",
        "confusion_matrices.json",
        "cost_comparison.json",
        "benchmark_summary.md",
    ]:
        assert (artifacts / name).is_file()

    csv = pd.read_csv(artifacts / "model_metrics.csv")
    assert list(csv.columns) == [
        "model_name",
        "status",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "pr_auc",
        "tn",
        "fp",
        "fn",
        "tp",
        "threshold",
        "training_time_seconds",
        "inference_latency_ms",
        "primary_metric",
        "primary_metric_value",
    ]
    assert set(csv["model_name"]) == {"logistic_regression", "random_forest", "xgboost"}
    assert csv.loc[csv["model_name"] == "logistic_regression", "tp"].iloc[0] == 45


def test_no_threshold_optimization_claimed(tmp_path, monkeypatch):
    _, _, _, _, artifacts, _ = _run(tmp_path, monkeypatch)

    metrics = json.loads((artifacts / "model_metrics.json").read_text(encoding="utf-8"))
    assert metrics["threshold"] == 0.5
    assert metrics["threshold_optimized"] is False
    assert "no threshold" in metrics["threshold_note"].lower()

    dataset = json.loads((artifacts / "dataset_summary.json").read_text(encoding="utf-8"))
    assert dataset["benchmark"]["threshold"] == 0.5
    assert dataset["benchmark"]["threshold_optimized"] is False

    markdown = (artifacts / "benchmark_summary.md").read_text(encoding="utf-8")
    assert "no threshold optimisation" in markdown.lower()
    assert "0.5" in markdown


def test_dataset_summary_copied_enriched_without_absolute_path(tmp_path, monkeypatch):
    _, _, _, _, artifacts, _ = _run(tmp_path, monkeypatch)
    dataset = json.loads((artifacts / "dataset_summary.json").read_text(encoding="utf-8"))
    # Original prepared summary keys survive the copy.
    assert dataset["dataset"] == "scania_aps"
    assert dataset["prepared_at"] == "2026-01-01T00:00:00+00:00"
    # Enriched benchmark metadata is present.
    assert dataset["benchmark"]["official_split"] == {"train_rows": 6, "test_rows": 4}
    assert dataset["benchmark"]["models"] == ["logistic_regression", "random_forest", "xgboost"]
    # Absolute paths are relativised.
    for key in ("train", "test"):
        path = dataset["splits"][key]["path"]
        assert not Path(path).is_absolute()
        assert path.endswith(("aps_failure_train.csv", "aps_failure_test.csv"))


def test_summary_based_on_actual_result(tmp_path, monkeypatch):
    _, _, _, _, _, summary = _run(tmp_path, monkeypatch)
    assert summary["recommended_model"] == "random_forest"
    assert summary["metric_best_model"] == "random_forest"
    assert summary["cost_best_model"] == "xgboost"
    assert summary["primary_metric"] == "pr_auc"
    assert summary["threshold"] == 0.5
    assert summary["task_recommendation"] == "binary_classification"
    assert len(summary["models"]) == 3


def test_run_with_real_train_on_tiny_data(tmp_path):
    cfg = _config(train_rows=60, test_rows=40, n_features=3)
    train = _frame(60, n_features=3)
    test = _frame(40, n_features=3, seed=7)
    prepared = tmp_path / "prepared"
    _write_prepared(prepared, train, test)
    artifacts = tmp_path / "artifacts"

    summary = run_benchmark(prepared_dir=prepared, artifacts_dir=artifacts, config=cfg)

    assert summary["primary_metric"] == "pr_auc"
    assert summary["recommended_model"] in _MODELS
    assert len(summary["models"]) == 3
    assert all(m["status"] == "success" for m in summary["models"])
    for name in [
        "dataset_summary.json",
        "model_metrics.json",
        "model_metrics.csv",
        "confusion_matrices.json",
        "cost_comparison.json",
        "benchmark_summary.md",
    ]:
        assert (artifacts / name).is_file()
