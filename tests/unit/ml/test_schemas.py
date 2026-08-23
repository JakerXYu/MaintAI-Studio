"""Contract tests for the ML schemas (JSON-safe Pydantic v2)."""

import pytest
from pydantic import ValidationError

from maintai.data.schemas import SplitResult
from maintai.ml.schemas import TrainingPlan, TrainingResult


def _split() -> SplitResult:
    return SplitResult(
        strategy="stratified",
        train_indices=[0, 1, 2],
        test_indices=[3, 4],
    )


def test_training_plan_json_roundtrip():
    plan = TrainingPlan(target="target", task="binary_classification", split=_split())
    data = plan.model_dump()
    assert TrainingPlan.model_validate(data) == plan
    assert isinstance(plan.model_dump_json(), str)


def test_training_result_json_safe():
    result = TrainingResult(
        task="regression",
        target="target",
        seed=42,
        primary_metric="rmse",
        evaluations=[],
    )
    assert result.model_dump()["primary_metric"] == "rmse"
    assert isinstance(result.model_dump_json(), str)


def test_unknown_primary_metric_rejected():
    with pytest.raises(ValidationError):
        TrainingPlan(
            target="target",
            task="regression",
            split=_split(),
            primary_metric="banana",
        )


def test_negative_seed_rejected():
    with pytest.raises(ValidationError):
        TrainingPlan(target="target", task="regression", split=_split(), seed=-1)


def test_known_primary_metric_accepted():
    plan = TrainingPlan(target="target", task="regression", split=_split(), primary_metric="rmse")
    assert plan.primary_metric == "rmse"


def test_primary_metric_must_match_task():
    with pytest.raises(ValidationError, match="not valid"):
        TrainingPlan(
            target="target",
            task="binary_classification",
            split=_split(),
            primary_metric="rmse",
        )
