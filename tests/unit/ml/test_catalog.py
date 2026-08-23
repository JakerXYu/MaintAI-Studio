"""Model catalog tests: names, fixed seeds, xgboost-unavailable graceful path."""

import pytest

from maintai.ml import catalog
from maintai.ml.catalog import ModelUnavailableError


def test_catalog_model_names():
    assert catalog.CLASSIFICATION_MODELS == ("logistic_regression", "random_forest", "xgboost")
    assert catalog.REGRESSION_MODELS == ("ridge", "random_forest", "xgboost")
    assert catalog.model_names_for_task("binary_classification") == catalog.CLASSIFICATION_MODELS
    assert catalog.model_names_for_task("regression") == catalog.REGRESSION_MODELS


def test_build_model_has_fixed_small_hyperparameters():
    rf = catalog.build_model("random_forest", "binary_classification", seed=7)
    assert rf.random_state == 7
    assert rf.n_estimators == 100
    lr = catalog.build_model("logistic_regression", "binary_classification", seed=7)
    assert lr.random_state == 7
    ridge = catalog.build_model("ridge", "regression", seed=7)
    assert ridge.random_state == 7


def test_xgboost_has_fixed_seed_and_params():
    xgb = catalog.build_model("xgboost", "binary_classification", seed=3)
    assert xgb.get_params()["random_state"] == 3
    assert xgb.get_params()["n_estimators"] == 100


def test_xgboost_unavailable_is_graceful(monkeypatch):
    monkeypatch.setattr(catalog, "_XGB_AVAILABLE", False)
    monkeypatch.setattr(
        catalog, "_XGB_IMPORT_ERROR", "ModuleNotFoundError: No module named 'xgboost'"
    )
    assert catalog.is_model_available("xgboost") is False
    reason = catalog.unavailable_reason("xgboost")
    assert reason is not None and "unavailable" in reason
    with pytest.raises(ModelUnavailableError, match="unavailable"):
        catalog.build_model("xgboost", "binary_classification", seed=42)


def test_lightgbm_is_not_offered():
    with pytest.raises(ModelUnavailableError):
        catalog.build_model("lightgbm", "binary_classification")


def test_unknown_model_raises():
    with pytest.raises(ModelUnavailableError, match="unknown model"):
        catalog.build_model("not_a_model", "regression")
