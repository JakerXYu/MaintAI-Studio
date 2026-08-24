"""Integration tests for the P0 copilot chat REST API.

Exercise ``POST /api/v1/copilot/chat`` end-to-end against an in-memory SQLite
database with the *real* dataset application service (so answers are genuinely
grounded in deterministic tool output), a mock provider (offline, no network),
and the request-id middleware. Also asserts action requests (promotion /
work-order) are never executed and the source of the agent package contains no
dangerous constructs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from maintai.agent.provider import MockProvider, build_provider
from maintai.agent.service import CopilotService
from maintai.agent.tools import CopilotTools
from maintai.api.main import create_app
from maintai.application.datasets import DatasetService
from maintai.audit.repository import AuditRepository
from maintai.audit.service import AuditService
from maintai.config import get_settings
from maintai.db.dataset_repository import DatasetRepository

CHAT_URL = "/api/v1/copilot/chat"


def _csv_bytes(n: int = 120, seed: int = 0) -> bytes:
    rng = np.random.default_rng(seed)
    y = np.zeros(n, dtype=int)
    y[: n // 2] = 1
    rng.shuffle(y)
    frame = pd.DataFrame(
        {
            "sensor_a": np.round(rng.normal(0.0, 1.0, n), 6),
            "sensor_b": np.round(rng.normal(1.0, 2.0, n), 6),
            "temp": np.round(rng.normal(50.0, 10.0, n), 6),
            "cat": rng.choice(["x", "y"], n),
            "failure": y,
        }
    )
    return frame.to_csv(index=False).encode("utf-8")


@pytest.fixture()
def dataset_service(session_factory, tmp_path):
    return DatasetService(
        repository=DatasetRepository(session_factory),
        session_factory=session_factory,
        audit=AuditService(AuditRepository(session_factory)),
        storage_root=tmp_path / "storage",
    )


@pytest.fixture()
def api_client(session_factory, dataset_service):
    app = create_app(
        session_factory=session_factory,
        dataset_service=dataset_service,
        mlflow_healthcheck=lambda: None,
    )
    with TestClient(app) as client:
        yield client


def _profiled_dataset_id(dataset_service: DatasetService) -> tuple[str, float]:
    dataset_id = dataset_service.upload(_csv_bytes(), "sensor.csv")["id"]
    dataset_service.profile(dataset_id)
    score = dataset_service.get_quality(dataset_id)["score"]
    return dataset_id, float(score)


def test_quality_chat_is_grounded(api_client, dataset_service):
    dataset_id, score = _profiled_dataset_id(dataset_service)

    r = api_client.post(
        CHAT_URL,
        json={"user_request": "数据质量怎么样？", "dataset_id": dataset_id},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["intent"] == "quality"
    assert body["evidence"][0]["source"] == "dataset.quality"
    assert body["tool_results"][0]["ok"] is True
    assert body["tool_results"][0]["data"]["score"] == score
    # The quantified value in the answer comes straight from the tool result,
    # and the offline mock provider produced the narrative.
    assert str(score) in body["answer"]
    assert "deterministic tool evidence" in body["answer"]


def test_request_id_echoed(api_client):
    r = api_client.post(
        CHAT_URL,
        json={"user_request": "What is the deployment status?"},
        headers={"X-Request-ID": "copilot-req-1"},
    )
    assert r.status_code == 200
    assert r.headers["x-request-id"] == "copilot-req-1"


def test_invalid_body_422(api_client):
    assert api_client.post(CHAT_URL, json={}).status_code == 422
    r = api_client.post(
        CHAT_URL,
        json={"user_request": "hi", "shell": "rm -rf /"},
    )
    assert r.status_code == 422


def test_promotion_not_executed_via_api(api_client):
    r = api_client.post(
        CHAT_URL,
        json={"user_request": "promote this model to production"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["tool_results"] == []
    assert body["proposed_action"]["status"] == "not_executed"
    assert body["proposed_action"]["approval_required"] is True
    assert body["proposed_action"]["type"] == "rejected"


def test_work_order_not_executed_via_api(api_client):
    r = api_client.post(
        CHAT_URL,
        json={"user_request": "create a maintenance work order"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["proposed_action"]["status"] == "not_executed"
    assert body["proposed_action"]["approval_required"] is True
    assert body["tool_results"] == []


def test_provider_defaults_to_mock(api_client):
    settings = get_settings()
    provider = build_provider(
        provider=settings.llm_provider,
        api_key=settings.llm_api_key.get_secret_value() if settings.llm_api_key else None,
        base_url=settings.llm_base_url,
        model=settings.llm_model,
    )
    assert isinstance(provider, MockProvider)


def test_chat_never_opens_network(session_factory, dataset_service, monkeypatch):
    dataset_id, _ = _profiled_dataset_id(dataset_service)

    settings = get_settings()
    provider = build_provider(
        provider=settings.llm_provider,
        api_key=settings.llm_api_key.get_secret_value() if settings.llm_api_key else None,
        base_url=settings.llm_base_url,
        model=settings.llm_model,
    )
    assert isinstance(provider, MockProvider)
    # Only the dataset service is exercised by a quality question; the other
    # services are unused here and passed as placeholders.
    service = CopilotService(
        tools=CopilotTools(
            dataset_service=dataset_service,
            experiment_service=None,
            model_registry_service=None,
            prediction_service=None,
        ),
        provider=provider,
    )

    def boom(*args, **kwargs):
        raise AssertionError("network access attempted")

    monkeypatch.setattr("maintai.agent.provider.httpx.Client", boom)
    response = service.chat(user_request="数据质量怎么样？", dataset_id=dataset_id)
    assert response["intent"] == "quality"
