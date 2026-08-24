"""Shared fixtures for copilot unit tests (offline, deterministic fakes)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from fakes import (
    FakeDatasetService,
    FakeExperimentService,
    FakeModelRegistryService,
    FakePredictionService,
)

from maintai.agent.provider import MockProvider
from maintai.agent.service import CopilotService
from maintai.agent.tools import CopilotTools


@pytest.fixture()
def fakes():
    dataset_service = FakeDatasetService()
    experiment_service = FakeExperimentService()
    model_registry_service = FakeModelRegistryService()
    prediction_service = FakePredictionService(
        deployed=True,
        result={
            "model_id": "m1",
            "model_version": "1",
            "count": 1,
            "records": [
                {
                    "prediction": 1,
                    "confidence": 0.93,
                    "positive_probability": 0.93,
                    "explanation": {"top_positive": [], "top_negative": [], "method": "shap"},
                }
            ],
        },
    )

    dataset_service.add(
        "d1",
        {
            "id": "d1",
            "name": "sensor_data",
            "file_path": "d1.csv",
            "file_hash": "deadbeef",
            "status": "task_recommended",
            "row_count": 240,
            "column_count": 7,
            "quality_score": 82.5,
            "target_column": "failure",
            "task_type": "binary_classification",
            "profile": {
                "task_recommendation": {"recommended_task": "binary_classification"},
                "leakage_report": {"verdict": "block", "excluded_features": ["serial_no"]},
            },
        },
        quality={"dataset_id": "d1", "score": 82.5, "report": {"score": 82.5}},
    )
    experiment_service.add(
        "e1",
        {
            "id": "e1",
            "task_type": "binary_classification",
            "status": "succeeded",
            "primary_metric": "recall",
            "value": 0.91,
            "recommended_model": "xgboost",
            "recommended_run_id": "run-xgboost",
            "model_runs": [
                {
                    "model_name": "xgboost",
                    "primary_metric": "recall",
                    "primary_metric_value": 0.91,
                    "metrics": {"recall": 0.91, "f1": 0.88, "precision": 0.85},
                }
            ],
        },
        {
            "experiment_id": "e1",
            "task": "binary_classification",
            "primary_metric": "recall",
            "best_model": "xgboost",
            "ranking": ["xgboost", "random_forest"],
            "candidates": ["xgboost", "random_forest"],
            "notes": ["best model 'xgboost' with recall=0.91"],
            "recommended_run_id": "run-xgboost",
            "value": 0.91,
        },
    )
    model_registry_service.add(
        "m1",
        {
            "id": "m1",
            "name": "failure-risk-demo",
            "version": "1",
            "deployment_status": "demo_deployed",
            "alias": "candidate",
            "mlflow_model_uri": "models:/failure-risk-demo/1",
        },
    )

    tools = CopilotTools(
        dataset_service=dataset_service,
        experiment_service=experiment_service,
        model_registry_service=model_registry_service,
        prediction_service=prediction_service,
    )
    return SimpleNamespace(
        dataset=dataset_service,
        experiment=experiment_service,
        model=model_registry_service,
        prediction=prediction_service,
        tools=tools,
    )


@pytest.fixture()
def copilot(fakes) -> CopilotService:
    return CopilotService(tools=fakes.tools, provider=MockProvider())
