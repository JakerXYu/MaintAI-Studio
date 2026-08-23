"""Deterministic recommendation tests: recall constraint and tie-breaking."""

from maintai.ml.recommend import recommend
from maintai.ml.schemas import ModelEvaluation, TrainingResult


def _eval(name, *, task="binary_classification", f1=None, recall=0.5, rmse=None, latency=None):
    if rmse is not None:
        metrics = {"mae": rmse, "rmse": rmse, "r2": 0.5}
    else:
        metrics = {
            "precision": 0.5,
            "recall": recall,
            "f1": f1 if f1 is not None else 0.5,
            "roc_auc": None,
            "pr_auc": None,
        }
    return ModelEvaluation(
        model_name=name,
        task=task,
        status="success",
        metrics=metrics,
        inference_latency_ms=latency,
    )


def _cls_result(evaluations, primary="f1"):
    return TrainingResult(
        task="binary_classification",
        target="target",
        seed=42,
        primary_metric=primary,
        evaluations=evaluations,
    )


def test_recall_constraint_filters_lower_recall_model():
    result = _cls_result(
        [_eval("a", f1=0.9, recall=0.4), _eval("b", f1=0.8, recall=0.9)]
    )
    comp = recommend(result, minimum_recall=0.6)
    assert comp.best_model == "b"
    assert any("minimum_recall" in note for note in comp.notes)


def test_tie_break_prefers_simpler_model():
    result = _cls_result([_eval("random_forest", f1=0.5), _eval("logistic_regression", f1=0.5)])
    comp = recommend(result)
    assert comp.best_model == "logistic_regression"


def test_tie_break_prefers_lower_latency():
    result = _cls_result(
        [_eval("model_b", f1=0.5, latency=2.0), _eval("model_a", f1=0.5, latency=1.0)]
    )
    comp = recommend(result)
    assert comp.best_model == "model_a"


def test_tie_break_prefers_model_name():
    result = _cls_result([_eval("model_b", f1=0.5), _eval("model_a", f1=0.5)])
    comp = recommend(result)
    assert comp.best_model == "model_a"


def test_regression_lower_is_better():
    result = TrainingResult(
        task="regression",
        target="target",
        seed=42,
        primary_metric="rmse",
        evaluations=[
            _eval("xgboost", task="regression", rmse=2.0),
            _eval("ridge", task="regression", rmse=1.0),
        ],
    )
    comp = recommend(result)
    assert comp.best_model == "ridge"


def test_no_candidate_returns_none_best():
    result = _cls_result([])
    comp = recommend(result)
    assert comp.best_model is None
    assert comp.ranking == []
