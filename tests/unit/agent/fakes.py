"""Shared fake application services for copilot unit tests.

These mirror the *read* surface of the real application services and raise the
same public exception classes so the tool allowlist's error handling is
exercised without a database or MLflow backend.
"""

from __future__ import annotations

from maintai.application.datasets import DatasetNotFoundError
from maintai.application.experiments import ExperimentNotFoundError
from maintai.application.models import RegisteredModelNotFoundError
from maintai.application.predictions import ModelNotDeployedError


class FakeDatasetService:
    def __init__(self) -> None:
        self._datasets: dict[str, dict] = {}
        self._qualities: dict[str, dict] = {}
        self.get_calls: list[str] = []
        self.get_quality_calls: list[str] = []

    def add(self, dataset_id: str, dataset: dict, quality: dict | None = None) -> None:
        self._datasets[dataset_id] = dataset
        if quality is not None:
            self._qualities[dataset_id] = quality

    def get(self, dataset_id: str) -> dict:
        self.get_calls.append(dataset_id)
        if dataset_id not in self._datasets:
            raise DatasetNotFoundError(f"dataset {dataset_id!r} not found")
        return self._datasets[dataset_id]

    def get_quality(self, dataset_id: str) -> dict:
        self.get_quality_calls.append(dataset_id)
        if dataset_id not in self._qualities:
            raise DatasetNotFoundError(f"dataset {dataset_id!r} not found")
        return self._qualities[dataset_id]


class FakeExperimentService:
    def __init__(self) -> None:
        self._results: dict[str, dict] = {}
        self._comparisons: dict[str, dict] = {}
        self.get_calls: list[str] = []
        self.comparison_calls: list[str] = []

    def add(self, experiment_id: str, results: dict, comparison: dict) -> None:
        self._results[experiment_id] = results
        self._comparisons[experiment_id] = comparison

    def get(self, experiment_id: str) -> dict:
        self.get_calls.append(experiment_id)
        if experiment_id not in self._results:
            raise ExperimentNotFoundError(f"experiment {experiment_id!r} not found")
        return self._results[experiment_id]

    def comparison(self, experiment_id: str) -> dict:
        self.comparison_calls.append(experiment_id)
        if experiment_id not in self._comparisons:
            raise ExperimentNotFoundError(f"experiment {experiment_id!r} not found")
        return self._comparisons[experiment_id]


class FakeModelRegistryService:
    def __init__(self) -> None:
        self._models: dict[str, dict] = {}
        self.get_calls: list[str] = []

    def add(self, model_id: str, model: dict) -> None:
        self._models[model_id] = model

    def get(self, model_id: str) -> dict:
        self.get_calls.append(model_id)
        if model_id not in self._models:
            raise RegisteredModelNotFoundError(f"registered model {model_id!r} not found")
        return self._models[model_id]


class FakePredictionService:
    def __init__(self, *, deployed: bool = True, result: dict | None = None) -> None:
        self.deployed = deployed
        self.result = result if result is not None else {}
        self.predict_calls: list[tuple[str, list]] = []
        self.persist_values: list[bool] = []

    def predict(self, model_id: str, records: list, *, persist: bool = True) -> dict:
        self.predict_calls.append((model_id, records))
        self.persist_values.append(persist)
        if not self.deployed:
            raise ModelNotDeployedError(
                f"registered model {model_id!r} is not demo-deployed"
            )
        return self.result
