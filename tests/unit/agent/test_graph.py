"""Unit tests for the route → tool → synthesize LangGraph copilot."""

from __future__ import annotations

import pytest

from maintai.agent.graph import (
    ACTION_DEPLOY,
    ACTION_PRODUCTION_PROMOTION,
    ACTION_REGISTER,
    ACTION_TRAIN,
    ACTION_WORK_ORDER,
    INTENT_BEST_MODEL,
    INTENT_DEPLOYMENT_STATUS,
    INTENT_EXPLAIN_PREDICTION,
    INTENT_FIX_BEFORE_DEPLOY,
    INTENT_QUALITY,
    INTENT_TASK,
    INTENT_UNKNOWN,
    INTENT_WHY_NOT_ACCURACY,
)
from maintai.agent.provider import LLMResponse, ProviderError
from maintai.agent.service import CopilotService


class FailingProvider:
    def invoke(self, messages, **kwargs):
        raise ProviderError("LLM provider request failed")


class InventingProvider:
    def invoke(self, messages, **kwargs):
        return LLMResponse(content="The unsupported quality score is 0.999.")


@pytest.mark.parametrize(
    ("question", "kwargs", "intent", "evidence_source", "needle"),
    [
        ("数据质量怎么样？", {"dataset_id": "d1"}, INTENT_QUALITY, "dataset.quality", "82.5"),
        (
            "为什么推荐二分类任务？",
            {"dataset_id": "d1"},
            INTENT_TASK,
            "dataset.task",
            "binary_classification",
        ),
        (
            "哪个模型最好？",
            {"experiment_id": "e1"},
            INTENT_BEST_MODEL,
            "experiment.comparison",
            "xgboost",
        ),
        (
            "为什么不使用准确率？",
            {"experiment_id": "e1"},
            INTENT_WHY_NOT_ACCURACY,
            "experiment.results",
            "recall",
        ),
        (
            "解释这个高风险预测",
            {"model_id": "m1", "record": {"sensor": 0.5}},
            INTENT_EXPLAIN_PREDICTION,
            "prediction.explanation",
            "0.93",
        ),
        (
            "部署前需要修复什么？",
            {"dataset_id": "d1"},
            INTENT_FIX_BEFORE_DEPLOY,
            "dataset.quality",
            "82.5",
        ),
        (
            "模型部署状态？",
            {"model_id": "m1"},
            INTENT_DEPLOYMENT_STATUS,
            "model.deployment_status",
            "demo_deployed",
        ),
    ],
)
def test_canonical_grounded_answers(copilot, question, kwargs, intent, evidence_source, needle):
    response = copilot.chat(user_request=question, **kwargs)
    assert response["intent"] == intent
    assert needle in response["answer"]
    assert response["tool_results"], "expected a tool result"
    assert response["tool_results"][0]["ok"] is True
    assert response["evidence"][0]["source"] == evidence_source
    # The mock narrative contributes no invented figures; the evidence block is
    # always embedded in the final answer.
    assert "Evidence:" in response["answer"]


def test_unknown_intent_insufficient_evidence(copilot):
    response = copilot.chat(user_request="tell me a joke about robots")
    assert response["intent"] == INTENT_UNKNOWN
    assert "Insufficient evidence" in response["answer"]
    assert response["tool_results"] == []
    assert response["evidence"] == []


def test_missing_id_graceful(copilot):
    response = copilot.chat(user_request="数据质量怎么样？")
    assert response["intent"] == INTENT_QUALITY
    assert response["tool_results"][0]["ok"] is False
    assert "missing required input(s): dataset_id" in response["answer"]
    assert "Insufficient evidence" in response["answer"]


def test_provider_failure_still_grounded(fakes):
    service = CopilotService(tools=fakes.tools, provider=FailingProvider())
    response = service.chat(user_request="数据质量怎么样？", dataset_id="d1")
    assert response["intent"] == INTENT_QUALITY
    assert "82.5" in response["answer"]
    assert response["tool_results"][0]["ok"] is True


def test_provider_invented_number_is_discarded(fakes):
    service = CopilotService(tools=fakes.tools, provider=InventingProvider())
    response = service.chat(user_request="数据质量怎么样？", dataset_id="d1")
    assert "0.999" not in response["answer"]
    assert "82.5" in response["answer"]


@pytest.mark.parametrize(
    ("question", "action", "approval_required", "expected_type"),
    [
        ("promote this model to production", ACTION_PRODUCTION_PROMOTION, True, "rejected"),
        ("create a maintenance work order", ACTION_WORK_ORDER, True, "proposal"),
        ("train a model on this dataset", ACTION_TRAIN, False, "proposal"),
        ("register the recommended run", ACTION_REGISTER, False, "proposal"),
        ("deploy this model to demo", ACTION_DEPLOY, False, "proposal"),
    ],
)
def test_actions_not_executed(copilot, fakes, question, action, approval_required, expected_type):
    before = (
        fakes.dataset.get_calls.copy(),
        fakes.experiment.get_calls.copy(),
        fakes.model.get_calls.copy(),
        fakes.prediction.predict_calls.copy(),
    )
    response = copilot.chat(user_request=question)
    assert response["intent"] == action
    assert response["tool_results"] == []
    assert response["proposed_action"]["status"] == "not_executed"
    assert response["proposed_action"]["approval_required"] is approval_required
    assert response["proposed_action"]["type"] == expected_type
    # No allowlisted tool (and no mutation) was executed.
    assert (
        fakes.dataset.get_calls,
        fakes.experiment.get_calls,
        fakes.model.get_calls,
        fakes.prediction.predict_calls,
    ) == before


def test_production_promotion_rejected(copilot):
    response = copilot.chat(user_request="promote this model to production")
    assert response["intent"] == ACTION_PRODUCTION_PROMOTION
    assert response["proposed_action"]["type"] == "rejected"
    assert "not executed" in response["answer"]
    assert "approval" in response["answer"].lower() or "P1" in response["answer"]
