"""End-to-end deterministic training tests (synthetic small data)."""

import numpy as np
import pandas as pd
import pytest

from maintai.data.schemas import SplitResult
from maintai.data.split import SplitConfig
from maintai.data.split import split as split_data
from maintai.ml.preprocess import PreprocessError
from maintai.ml.recommend import recommend
from maintai.ml.schemas import TrainingPlan
from maintai.ml.train import TrainingError, train


def _classification_frame(n=240, n_pos=120, seed=0):
    rng = np.random.default_rng(seed)
    y = np.zeros(n, dtype=int)
    y[:n_pos] = 1
    rng.shuffle(y)
    return pd.DataFrame(
        {
            "num": rng.normal(0, 1, n) + y * 2.0,
            "cat": rng.choice(["a", "b", "c"], n),
            "flag": rng.choice([True, False], n),
            "target": y,
        }
    )


def _regression_frame(n=240, seed=0):
    rng = np.random.default_rng(seed)
    target = rng.normal(0, 1, n)
    return pd.DataFrame(
        {
            "num": target + rng.normal(0, 0.1, n),
            "cat": rng.choice(["a", "b", "c"], n),
            "flag": rng.choice([True, False], n),
            "target": target,
        }
    )


def _plan(frame, task, **kwargs):
    s = split_data(
        frame,
        target_col="target",
        task_type=task,
        config=SplitConfig(test_size=0.3, validation_size=0.0),
    )
    return TrainingPlan(target="target", task=task, split=s, **kwargs)


def test_three_model_classification_trains_and_compares():
    frame = _classification_frame(n=240, n_pos=120)
    out = train(frame, _plan(frame, "binary_classification", seed=42))
    successes = [e for e in out.result.evaluations if e.status == "success"]
    assert len(successes) == 3
    names = {e.model_name for e in successes}
    assert names == {"logistic_regression", "random_forest", "xgboost"}
    for e in successes:
        assert e.confusion_matrix is not None
        assert e.metrics["f1"] is not None
    comp = recommend(out.result)
    assert len(comp.ranking) == 3
    assert comp.best_model in names


def test_imbalanced_classification_primary_pr_auc():
    frame = _classification_frame(n=300, n_pos=12)
    out = train(frame, _plan(frame, "binary_classification", seed=42))
    assert out.result.primary_metric == "pr_auc"


def test_multiclass_classification_primary_f1():
    rng = np.random.default_rng(1)
    n = 240
    y = rng.integers(0, 3, n)
    frame = pd.DataFrame(
        {"num": rng.normal(0, 1, n) + y, "cat": rng.choice(["a", "b"], n), "target": y}
    )
    out = train(frame, _plan(frame, "multiclass_classification", seed=42))
    assert out.result.primary_metric == "f1"
    assert len([e for e in out.result.evaluations if e.status == "success"]) == 3


def test_regression_three_models():
    frame = _regression_frame(n=240)
    out = train(frame, _plan(frame, "regression", seed=42))
    successes = [e for e in out.result.evaluations if e.status == "success"]
    assert len(successes) == 3
    assert out.result.primary_metric == "rmse"
    assert len(recommend(out.result).ranking) == 3


def test_model_failure_recorded_others_continue():
    frame = _classification_frame(n=240, n_pos=120)
    plan = _plan(
        frame,
        "binary_classification",
        seed=42,
        model_names=["logistic_regression", "not_a_model", "random_forest"],
    )
    out = train(frame, plan)
    statuses = {e.model_name: e.status for e in out.result.evaluations}
    assert statuses["logistic_regression"] == "success"
    assert statuses["random_forest"] == "success"
    assert statuses["not_a_model"] == "error"
    assert any("not_a_model" in error for error in out.result.errors)
    assert set(out.models) == {"logistic_regression", "random_forest"}


def test_all_models_fail_raises():
    frame = _classification_frame(n=240, n_pos=120)
    plan = _plan(frame, "binary_classification", seed=42, model_names=["not_a_model"])
    with pytest.raises(TrainingError, match="no model trained successfully"):
        train(frame, plan)


def test_invalid_split_empty_train_raises():
    frame = _classification_frame(n=240, n_pos=120)
    plan = _plan(frame, "binary_classification", seed=42)
    bad_split = SplitResult(
        strategy="stratified",
        train_indices=[],
        test_indices=list(plan.split.test_indices),
    )
    with pytest.raises(TrainingError, match="must not be empty"):
        train(frame, plan.model_copy(update={"split": bad_split}))


def test_empty_features_raises():
    frame = _classification_frame(n=240, n_pos=120)
    plan = _plan(frame, "binary_classification", excluded=["num", "cat", "flag"])
    with pytest.raises(PreprocessError, match="no usable"):
        train(frame, plan)


def test_invalid_feature_column_raises():
    frame = _classification_frame(n=240, n_pos=120)
    plan = _plan(frame, "binary_classification", features=["ghost"])
    with pytest.raises(PreprocessError, match="not found"):
        train(frame, plan)


def test_fixed_seed_reproducible_and_recommendation_consistent():
    frame = _classification_frame(n=240, n_pos=120)
    plan = _plan(frame, "binary_classification", seed=42)
    out1 = train(frame, plan)
    out2 = train(frame, plan)
    metrics1 = {e.model_name: e.metrics for e in out1.result.evaluations}
    metrics2 = {e.model_name: e.metrics for e in out2.result.evaluations}
    assert metrics1 == metrics2
    assert recommend(out1.result).best_model == recommend(out2.result).best_model


def test_position_split_works_with_non_default_dataframe_index():
    frame = _classification_frame(n=120, n_pos=60)
    plan = _plan(frame, "binary_classification", model_names=["logistic_regression"])
    frame.index = np.arange(1000, 1120)
    out = train(frame, plan)
    assert out.result.evaluations[0].status == "success"


def test_preprocessor_column_roles_use_train_fold_only():
    frame = pd.DataFrame(
        {
            "binary_sensor": [0, 1] * 40 + [2] * 20,
            "target": [0, 1] * 50,
        }
    )
    split = SplitResult(
        strategy="manual",
        train_indices=list(range(80)),
        test_indices=list(range(80, 100)),
    )
    plan = TrainingPlan(
        target="target",
        task="binary_classification",
        split=split,
        model_names=["logistic_regression"],
    )
    out = train(frame, plan)
    transformers = out.models["logistic_regression"].pipeline["preprocess"].transformers_
    transformer_names = {
        name for name, _, _ in transformers
    }
    assert "bool" in transformer_names
    assert "num" not in transformer_names


def test_pr_auc_falls_back_to_f1_when_test_fold_has_one_class():
    frame = pd.DataFrame(
        {
            "feature": np.linspace(0, 1, 100),
            "target": [1] * 5 + [0] * 95,
        }
    )
    split = SplitResult(
        strategy="chronological",
        train_indices=list(range(80)),
        test_indices=list(range(80, 100)),
    )
    plan = TrainingPlan(
        target="target",
        task="binary_classification",
        split=split,
        model_names=["logistic_regression"],
    )
    out = train(frame, plan)
    assert out.result.primary_metric == "f1"
    assert recommend(out.result).best_model == "logistic_regression"


def test_evaluation_records_label_order_and_positive_label():
    frame = _classification_frame(n=120, n_pos=60)
    out = train(
        frame,
        _plan(frame, "binary_classification", model_names=["logistic_regression"]),
    )
    evaluation = out.result.evaluations[0]
    assert evaluation.labels == [0, 1]
    assert evaluation.positive_label == 1
