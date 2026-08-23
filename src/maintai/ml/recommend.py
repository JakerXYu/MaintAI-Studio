"""Deterministic best-model recommendation (rule engine, never an LLM).

Selection order: satisfy the classification ``minimum_recall`` constraint, then
rank by the primary metric (higher is better for classification metrics, lower
is better for regression error metrics), then tie-break by model complexity
(simpler preferred), inference latency (lower preferred), and finally model name.
"""

from __future__ import annotations

from maintai.ml import catalog
from maintai.ml.schemas import CLASSIFICATION_TASKS, Comparison, TrainingResult

_LOWER_IS_BETTER: frozenset[str] = frozenset({"rmse", "mae", "mse"})


def _default_primary(task: str) -> str:
    return "rmse" if task == "regression" else "f1"


def _metric_direction(metric: str) -> str:
    return "lower" if metric in _LOWER_IS_BETTER else "higher"


def _fmt(value: float) -> str:
    return f"{value:.6g}"


def _ranking_key(primary: str, direction: str):
    def key(evaluation) -> tuple:
        value = float(evaluation.metrics[primary])
        sign = 1.0 if direction == "lower" else -1.0
        complexity = catalog.model_complexity(evaluation.model_name)
        latency = (
            evaluation.inference_latency_ms
            if evaluation.inference_latency_ms is not None
            else float("inf")
        )
        return (sign * value, complexity, latency, evaluation.model_name)

    return key


def recommend(result: TrainingResult, *, minimum_recall: float = 0.0) -> Comparison:
    """Select the best successful model deterministically with explainable notes."""
    primary = result.primary_metric or _default_primary(result.task)
    direction = _metric_direction(primary)

    candidates = [
        e
        for e in result.evaluations
        if e.status == "success" and e.metrics.get(primary) is not None
    ]

    notes: list[str] = []
    if result.task in CLASSIFICATION_TASKS and minimum_recall > 0:
        before = len(candidates)
        candidates = [
            e
            for e in candidates
            if e.metrics.get("recall") is None or float(e.metrics["recall"]) >= minimum_recall
        ]
        notes.append(
            f"minimum_recall constraint (>={minimum_recall}) kept "
            f"{len(candidates)}/{before} candidate(s)"
        )

    if not candidates:
        return Comparison(
            task=result.task,
            primary_metric=primary,
            best_model=None,
            ranking=[],
            candidates=[],
            notes=notes
            + ["no successful model satisfies the primary metric and recall constraint"],
        )

    ranked = sorted(candidates, key=_ranking_key(primary, direction))
    best = ranked[0]
    notes.append(
        f"primary metric {primary!r} ({direction} is better); "
        f"{len(ranked)} candidate(s) ranked"
    )
    notes.append(
        f"best model {best.model_name!r} with {primary}={_fmt(float(best.metrics[primary]))}"
    )
    if len(ranked) >= 2 and abs(
        float(ranked[0].metrics[primary]) - float(ranked[1].metrics[primary])
    ) < 1e-9:
        notes.append(
            "primary metric tie broken by: model complexity (simpler preferred), "
            "then inference latency (lower preferred), then model name"
        )

    return Comparison(
        task=result.task,
        primary_metric=primary,
        best_model=best.model_name,
        ranking=[e.model_name for e in ranked],
        candidates=[e.model_name for e in candidates],
        notes=notes,
    )
