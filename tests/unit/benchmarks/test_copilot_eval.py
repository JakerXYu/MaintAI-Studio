"""Scania APS copilot evaluation tests (fixture artifacts, offline mock)."""

from __future__ import annotations

import json
from pathlib import Path

from maintai.agent.provider import LLMResponse
from maintai.benchmarks.copilot_eval import (
    COST_COMPARISON_FILENAME,
    DATASET_SUMMARY_FILENAME,
    DEEPSEEK_EVALUATION_JSON_FILENAME,
    EVALUATION_JSON_FILENAME,
    EVALUATION_MD_FILENAME,
    LIVE_SCENARIOS,
    MODEL_METRICS_FILENAME,
    SCENARIOS,
    build_deepseek_provider,
    deepseek_provider_config,
    evaluate,
    evaluate_deepseek,
    evaluate_live_scenarios,
    smoke_provider,
    write_deepseek_evaluation,
    write_evaluation,
)

_MODELS = [
    {
        "model_name": "logistic_regression",
        "task": "binary_classification",
        "status": "success",
        "precision": 0.55,
        "recall": 0.60,
        "f1": 0.60,
        "roc_auc": 0.75,
        "pr_auc": 0.60,
        "primary_metric": "pr_auc",
        "primary_metric_value": 0.60,
        "threshold": 0.5,
    },
    {
        "model_name": "random_forest",
        "task": "binary_classification",
        "status": "success",
        "precision": 0.85,
        "recall": 0.80,
        "f1": 0.82,
        "roc_auc": 0.92,
        "pr_auc": 0.90,
        "primary_metric": "pr_auc",
        "primary_metric_value": 0.90,
        "threshold": 0.5,
    },
    {
        "model_name": "xgboost",
        "task": "binary_classification",
        "status": "success",
        "precision": 0.80,
        "recall": 0.90,
        "f1": 0.85,
        "roc_auc": 0.94,
        "pr_auc": 0.95,
        "primary_metric": "pr_auc",
        "primary_metric_value": 0.95,
        "threshold": 0.5,
    },
]


def _write_artifacts(tmp_path: Path) -> Path:
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()

    model_metrics = {
        "dataset": "scania_aps",
        "target": "aps_failure",
        "task": "binary_classification",
        "primary_metric": "pr_auc",
        "seed": 42,
        "threshold": 0.5,
        "threshold_optimized": False,
        "recommended_model": "xgboost",
        "models": _MODELS,
    }
    cost_comparison = {
        "metric_best": "xgboost",
        "cost_best": "xgboost",
        "rows": [
            {"model_name": "xgboost", "fn": 3, "fp": 5, "expected_error_cost": 1550.0,
             "primary_metric": "pr_auc", "value": 0.95, "status": "ok"},
            {"model_name": "logistic_regression", "fn": 5, "fp": 10,
             "expected_error_cost": 2600.0, "primary_metric": "pr_auc", "value": 0.60,
             "status": "ok"},
            {"model_name": "random_forest", "fn": 8, "fp": 2, "expected_error_cost": 4020.0,
             "primary_metric": "pr_auc", "value": 0.90, "status": "ok"},
        ],
        "assumptions": {
            "fn_cost": 500.0,
            "fp_cost": 10.0,
            "currency_label": "challenge cost units",
        },
    }
    dataset_summary = {
        "dataset": "scania_aps",
        "target": "aps_failure",
        "n_features": 2,
        "splits": {
            "train": {
                "shape": [6, 3],
                "class_distribution": {"neg": 3, "pos": 3},
                "missingness": {"cells": 18, "missing": 1, "rate": 0.05555555555555555},
            },
            "test": {
                "shape": [4, 3],
                "class_distribution": {"neg": 1, "pos": 3},
                "missingness": {"cells": 12, "missing": 0, "rate": 0.0},
            },
        },
    }

    (artifacts / MODEL_METRICS_FILENAME).write_text(json.dumps(model_metrics), encoding="utf-8")
    (artifacts / COST_COMPARISON_FILENAME).write_text(json.dumps(cost_comparison), encoding="utf-8")
    (artifacts / DATASET_SUMMARY_FILENAME).write_text(json.dumps(dataset_summary), encoding="utf-8")
    return artifacts


def _by_id(result: dict, scenario_id: str) -> dict:
    return next(s for s in result["scenarios"] if s["id"] == scenario_id)


def test_evaluate_produces_twelve_scenarios(tmp_path):
    artifacts = _write_artifacts(tmp_path)
    result = evaluate(artifacts)
    assert result["meta"]["scenario_count"] == 12
    assert len(result["scenarios"]) == 12
    assert result["meta"]["provider"] == "mock"
    assert result["meta"]["live_provider"] == "N/A"


def test_tool_selection_matches_expected(tmp_path):
    artifacts = _write_artifacts(tmp_path)
    result = evaluate(artifacts)
    expectations = {
        "quality": ("quality", "dataset_quality"),
        "recall_best": ("best_model", "experiment_comparison"),
        "pr_auc_best": ("best_model", "experiment_comparison"),
        "exact_metric": ("experiment_results", "experiment_results"),
        "nonexistent_metric": ("experiment_results", "experiment_results"),
        "why_not_accuracy": ("why_not_accuracy", "experiment_results"),
        "recommendation_why": ("best_model", "experiment_comparison"),
    }
    for scenario_id, (intent, tool) in expectations.items():
        scenario = _by_id(result, scenario_id)
        assert scenario["observed_intent"] == intent
        assert scenario["observed_tool"] == tool
        assert scenario["tool_ok"] is True


def test_numeric_evidence_appendix_does_not_count_as_answer(tmp_path):
    artifacts = _write_artifacts(tmp_path)
    result = evaluate(artifacts)
    scenario = _by_id(result, "exact_metric")
    assert scenario["grounded_ok"] is True
    assert scenario["numeric_ok"] is False
    (evidence,) = scenario["numeric_evidence"]
    assert evidence["label"] == "random_forest pr_auc"
    assert evidence["expected"] == 0.90
    assert evidence["present"] is False


def test_no_fabricated_numeric_claims(tmp_path):
    artifacts = _write_artifacts(tmp_path)
    result = evaluate(artifacts)
    for scenario in result["scenarios"]:
        assert scenario["fabricated_numbers"] == []
        assert scenario["grounded_ok"] is True
    assert result["metrics"]["unsupported_numeric_claim_rate"]["value"] == 0.0
    assert result["metrics"]["unsupported_numeric_claim_rate"]["denominator"] == 1


def test_nonexistent_metric_does_not_pass_without_explicit_unavailable(tmp_path):
    artifacts = _write_artifacts(tmp_path)
    result = evaluate(artifacts)
    scenario = _by_id(result, "nonexistent_metric")
    assert scenario["tool_ok"] is True
    assert scenario["passed"] is False
    assert any("unavailable" in reason for reason in scenario["failed_reasons"])


def test_unsafe_scenarios_blocked_with_approval_semantics(tmp_path):
    artifacts = _write_artifacts(tmp_path)
    result = evaluate(artifacts)
    for scenario_id in ("direct_champion", "bypass_approval"):
        scenario = _by_id(result, scenario_id)
        assert scenario["safety"] == "block"
        assert scenario["safety_ok"] is True
        assert scenario["passed"] is True


def test_na_scenarios_marked(tmp_path):
    artifacts = _write_artifacts(tmp_path)
    result = evaluate(artifacts)
    assert set(result["na_scenarios"]) == {"cost_lowest", "compare_two_experiments"}
    for scenario_id in ("cost_lowest", "compare_two_experiments"):
        scenario = _by_id(result, scenario_id)
        assert scenario["na"] is True
        assert scenario["na_reason"]


def test_metrics_have_numerator_and_denominator(tmp_path):
    artifacts = _write_artifacts(tmp_path)
    result = evaluate(artifacts)
    expected_names = {
        "tool_selection_accuracy",
        "grounded_numeric_answer_accuracy",
        "unsupported_numeric_claim_rate",
        "unsafe_action_block_or_proposal_compliance",
        "task_success_rate",
    }
    assert set(result["metrics"]) == expected_names
    for metric in result["metrics"].values():
        assert "numerator" in metric
        assert "denominator" in metric
        assert "value" in metric
        assert metric["numerator"] <= metric["denominator"]


def test_mock_provider_does_not_claim_numeric_answer_credit(tmp_path):
    artifacts = _write_artifacts(tmp_path)
    result = evaluate(artifacts)
    metric = result["metrics"]["grounded_numeric_answer_accuracy"]
    assert metric == {"numerator": 0, "denominator": 6, "value": 0.0}


def test_serialization_writes_json_and_markdown(tmp_path):
    artifacts = _write_artifacts(tmp_path)
    result = write_evaluation(artifacts)

    payload = json.loads((artifacts / EVALUATION_JSON_FILENAME).read_text(encoding="utf-8"))
    assert payload["meta"]["scenario_count"] == 12
    assert set(payload["metrics"]) == set(result["metrics"])
    assert len(payload["scenarios"]) == 12

    markdown = (artifacts / EVALUATION_MD_FILENAME).read_text(encoding="utf-8")
    assert "Scania APS copilot evaluation" in markdown
    assert "tool_selection_accuracy" in markdown
    assert "live_provider" in markdown


# -- offline fakes for the live DeepSeek slice ---------------------------------


class _FakeProvider:
    """Recording provider that returns a fixed, secret-free content string."""

    def __init__(self, content: str = "ok"):
        self._content = content
        self.invocations = 0
        self.messages: list[list[dict[str, str]]] = []

    def invoke(self, messages: list[dict[str, str]], **kwargs):
        del kwargs
        self.invocations += 1
        self.messages.append(messages)
        return LLMResponse(content=self._content)


class _BoomProvider:
    """Provider that always raises with a secret embedded in the exception."""

    def invoke(self, messages: list[dict[str, str]], **kwargs):
        del messages, kwargs
        raise RuntimeError("secret https://api.deepseek.com key=SECRET leaked")


# -- frozen vs live scenario identity ------------------------------------------


def test_frozen_scenario_identity_preserved():
    assert [s.id for s in SCENARIOS] == [
        "quality",
        "recall_best",
        "f1_best",
        "pr_auc_best",
        "cost_lowest",
        "recommendation_why",
        "exact_metric",
        "nonexistent_metric",
        "direct_champion",
        "bypass_approval",
        "compare_two_experiments",
        "why_not_accuracy",
    ]
    assert len(SCENARIOS) == 12


def test_live_scenarios_have_exactly_ten():
    assert [s.id for s in LIVE_SCENARIOS] == [
        "exact_f1_rf",
        "exact_pr_auc_xgb",
        "exact_xgb_fp_fn",
        "lowest_cost",
        "highest_recall",
        "rf_vs_xgb_cost",
        "unavailable_mse",
        "unavailable_production_accuracy",
        "direct_production_promotion",
        "high_precision_best",
    ]
    assert len(LIVE_SCENARIOS) == 10


def test_live_na_scenario_set():
    na_ids = {s.id for s in LIVE_SCENARIOS if s.na}
    assert na_ids == {
        "exact_xgb_fp_fn",
        "lowest_cost",
        "rf_vs_xgb_cost",
        "unavailable_production_accuracy",
    }


# -- live probe execution (offline) --------------------------------------------


def test_live_no_provider_call_for_actions_and_na(tmp_path):
    artifacts = _write_artifacts(tmp_path)
    result = evaluate_live_scenarios(artifacts, _FakeProvider())
    records = {s["id"]: s for s in result["scenarios"]}

    for scenario_id in (
        "exact_xgb_fp_fn",
        "lowest_cost",
        "rf_vs_xgb_cost",
        "unavailable_production_accuracy",
        "direct_production_promotion",
    ):
        assert records[scenario_id]["provider_invoked"] is False, scenario_id

    for scenario_id in (
        "exact_f1_rf",
        "exact_pr_auc_xgb",
        "highest_recall",
        "unavailable_mse",
        "high_precision_best",
    ):
        assert records[scenario_id]["provider_invoked"] is True, scenario_id


def test_live_scenario_records_expose_route_tool_provider_result(tmp_path):
    artifacts = _write_artifacts(tmp_path)
    result = evaluate_live_scenarios(artifacts, _FakeProvider())
    assert len(result["scenarios"]) == 10
    for scenario in result["scenarios"]:
        assert scenario["route"] == scenario["observed_intent"]
        assert scenario["tool"] == scenario["observed_tool"]
        assert isinstance(scenario["provider_invoked"], bool)
        assert scenario["result"] in {"pass", "fail", "na"}


def test_live_metrics_include_scenario_ids_and_reasons(tmp_path):
    artifacts = _write_artifacts(tmp_path)
    result = evaluate_live_scenarios(artifacts, _FakeProvider())
    expected_names = {
        "deterministic_router_tool_selection",
        "grounded_numeric_answer",
        "unsupported_claim",
        "unsafe_action_block_compliance",
        "task_success",
    }
    assert set(result["metrics"]) == expected_names
    for metric in result["metrics"].values():
        assert "numerator" in metric
        assert "denominator" in metric
        assert "scenario_ids" in metric
        assert "reasons" in metric
        assert metric["numerator"] <= metric["denominator"]


def test_live_na_scenarios_excluded_from_metrics(tmp_path):
    artifacts = _write_artifacts(tmp_path)
    result = evaluate_live_scenarios(artifacts, _FakeProvider())
    assert set(result["na_scenarios"]) == {
        "exact_xgb_fp_fn",
        "lowest_cost",
        "rf_vs_xgb_cost",
        "unavailable_production_accuracy",
    }
    for metric in result["metrics"].values():
        for scenario_id in metric["scenario_ids"]:
            assert scenario_id not in result["na_scenarios"]


# -- smoke provider -------------------------------------------------------------


def test_smoke_provider_reports_without_raw_content():
    fake = _FakeProvider(content="top secret raw answer")
    result = smoke_provider(fake)
    assert result["status"] == "ok"
    assert result["nonempty"] is True
    assert isinstance(result["latency_seconds"], float)
    assert result["error"] is None
    assert "content" not in result
    (message,) = fake.messages[0]
    assert message["role"] == "user"
    assert "Connectivity" in message["content"]


def test_smoke_provider_stable_error_does_not_leak():
    result = smoke_provider(_BoomProvider())
    assert result["status"] == "error"
    assert result["nonempty"] is False
    assert result["error"] == "provider connectivity check failed"
    assert "SECRET" not in result["error"]
    assert "api.deepseek.com" not in result["error"]
    assert "content" not in result


def test_smoke_provider_flags_empty_content():
    result = smoke_provider(_FakeProvider(content="   "))
    assert result["status"] == "empty"
    assert result["nonempty"] is False
    assert result["error"] == "provider returned empty content"


# -- DeepSeek provider construction (offline, no network) -----------------------


def test_deepseek_provider_config_is_enforced():
    config = deepseek_provider_config()
    assert config["base_url"] == "https://api.deepseek.com"
    assert config["model"] == "deepseek-v4-flash"
    assert config["thinking"] == "disabled"
    assert config["temperature"] == 0.0


def test_build_deepseek_provider_enforces_base_model_thinking_without_key_leak():
    provider = build_deepseek_provider("test-api-key")
    rep = repr(provider)
    assert "https://api.deepseek.com" in rep
    assert "deepseek-v4-flash" in rep
    assert "thinking='disabled'" in rep
    assert "test-api-key" not in rep


# -- DeepSeek evaluation assembly -----------------------------------------------


def test_evaluate_deepseek_returns_expected_structure(tmp_path):
    artifacts = _write_artifacts(tmp_path)
    result = evaluate_deepseek(artifacts, _FakeProvider())

    assert result["meta"]["provider"] == "deepseek"
    assert result["meta"]["mock_scenario_count"] == 12
    assert result["meta"]["live_scenario_count"] == 10
    assert result["provider"]["base_url"] == "https://api.deepseek.com"

    assert set(result["mock_baseline"]) == {
        "tool_selection_accuracy",
        "grounded_numeric_answer_accuracy",
        "unsupported_numeric_claim_rate",
        "unsafe_action_block_or_proposal_compliance",
        "task_success_rate",
    }
    assert set(result["replay"]["metrics"]) == set(result["mock_baseline"])
    assert len(result["replay"]["scenarios"]) == 12
    assert set(result["live"]["metrics"]) == {
        "deterministic_router_tool_selection",
        "grounded_numeric_answer",
        "unsupported_claim",
        "unsafe_action_block_compliance",
        "task_success",
    }
    assert len(result["live"]["scenarios"]) == 10


def test_write_deepseek_evaluation_does_not_touch_mock_artifacts(tmp_path):
    artifacts = _write_artifacts(tmp_path)
    result = evaluate_deepseek(artifacts, _FakeProvider())
    write_deepseek_evaluation(artifacts, result)

    payload = json.loads(
        (artifacts / DEEPSEEK_EVALUATION_JSON_FILENAME).read_text(encoding="utf-8")
    )
    assert payload["meta"]["provider"] == "deepseek"
    assert len(payload["replay"]["scenarios"]) == 12
    assert len(payload["live"]["scenarios"]) == 10

    assert not (artifacts / EVALUATION_JSON_FILENAME).exists()
    assert not (artifacts / EVALUATION_MD_FILENAME).exists()


def test_deepseek_live_missing_key_skips_without_network(monkeypatch, capsys, tmp_path):
    import importlib
    import sys
    import types

    scripts_dir = str(Path(__file__).resolve().parents[3] / "scripts")
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    cli = importlib.import_module("evaluate_scania_copilot")

    monkeypatch.setattr(cli, "get_settings", lambda: types.SimpleNamespace(llm_api_key=None))

    def _forbidden(*_args, **_kwargs):
        raise AssertionError("no network/provider may be built without an API key")

    monkeypatch.setattr(cli, "build_deepseek_provider", _forbidden)
    monkeypatch.setattr(cli, "smoke_provider", _forbidden)
    monkeypatch.setattr(cli, "evaluate_deepseek", _forbidden)

    rc = cli.main(["--deepseek-live", "--artifacts-dir", str(tmp_path)])
    assert rc == 0
    out = capsys.readouterr().out
    assert '"status":"skipped"' in out
    assert "llm_api_key" in out
    assert not list(tmp_path.iterdir())
