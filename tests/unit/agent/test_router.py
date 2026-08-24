"""Unit tests for the deterministic canonical intent router (中英 keywords)."""

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
    route_intent,
)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("What is the data quality?", INTENT_QUALITY),
        ("数据质量怎么样？", INTENT_QUALITY),
        ("Are there any missing values?", INTENT_QUALITY),
        ("为什么推荐二分类任务？", INTENT_TASK),
        ("Why is this a binary classification task?", INTENT_TASK),
        ("哪个模型最好？", INTENT_BEST_MODEL),
        ("Which model is best?", INTENT_BEST_MODEL),
        ("为什么不使用准确率？", INTENT_WHY_NOT_ACCURACY),
        ("Why not accuracy?", INTENT_WHY_NOT_ACCURACY),
        ("解释这个高风险预测", INTENT_EXPLAIN_PREDICTION),
        ("Explain this high-risk prediction", INTENT_EXPLAIN_PREDICTION),
        ("部署前需要修复什么？", INTENT_FIX_BEFORE_DEPLOY),
        ("What should I fix before deploy?", INTENT_FIX_BEFORE_DEPLOY),
        ("模型部署状态？", INTENT_DEPLOYMENT_STATUS),
        ("What is the deployment status?", INTENT_DEPLOYMENT_STATUS),
    ],
)
def test_canonical_read_intents(text, expected):
    assert route_intent(text) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("train a model on this dataset", ACTION_TRAIN),
        ("训练这个模型", ACTION_TRAIN),
        ("register the recommended run", ACTION_REGISTER),
        ("注册这个模型", ACTION_REGISTER),
        ("deploy this model to demo", ACTION_DEPLOY),
        ("部署这个模型", ACTION_DEPLOY),
        ("promote this model to production", ACTION_PRODUCTION_PROMOTION),
        ("把这个模型上线生产", ACTION_PRODUCTION_PROMOTION),
        ("create a maintenance work order", ACTION_WORK_ORDER),
        ("创建一个维修工单", ACTION_WORK_ORDER),
    ],
)
def test_action_intents(text, expected):
    assert route_intent(text) == expected


def test_unknown_intent():
    assert route_intent("tell me a joke about factory robots") == INTENT_UNKNOWN
    assert route_intent("") == INTENT_UNKNOWN


def test_deterministic_router():
    text = "Which model is best?"
    assert route_intent(text) == route_intent(text)
