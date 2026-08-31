"""Unit tests for the P1 UI HTTP client methods using ``httpx.MockTransport``.

Covers the P1 endpoints (monitoring, cost comparison, approvals, promotion,
feedback, mock CMMS) for HTTP verb, path, JSON body, and — where the local-demo
human gate applies — the ``Authorization`` / ``X-Human-Actor-ID`` headers. The
bearer token is only ever placed in the outgoing header and never cached.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import httpx
import pytest

from maintai.ui.api_client import APIClient

# (method_name, args, kwargs, expected_method, expected_path, expected_body, expected_headers)
_CASES: list[tuple[str, tuple, dict[str, Any], str, str, dict[str, Any], dict[str, str]]] = [
    (
        "create_monitoring_run",
        ("m1",),
        {"replay_kind": "severe"},
        "POST",
        "/api/v1/monitoring/runs",
        {"model_id": "m1", "replay_kind": "severe"},
        {},
    ),
    ("list_monitoring_runs", (), {}, "GET", "/api/v1/monitoring/runs", None, {}),
    ("get_monitoring_run", ("run-1",), {}, "GET", "/api/v1/monitoring/runs/run-1", None, {}),
    (
        "get_cost_comparison",
        ("exp1", 10.0, 1.0, 0.8),
        {},
        "POST",
        "/api/v1/experiments/exp1/cost-comparison",
        {"fn_cost": 10.0, "fp_cost": 1.0, "minimum_recall": 0.8},
        {},
    ),
    (
        "propose_approval",
        (),
        {
            "action_type": "cmms_work_order",
            "entity_type": "prediction_event",
            "entity_id": "pe-1",
            "requested_by_type": "user",
            "proposed_payload": {"asset_id": "a1"},
        },
        "POST",
        "/api/v1/approvals",
        {
            "action_type": "cmms_work_order",
            "entity_type": "prediction_event",
            "entity_id": "pe-1",
            "requested_by_type": "user",
            "proposed_payload": {"asset_id": "a1"},
        },
        {},
    ),
    ("list_approvals", (), {"status": "pending"}, "GET", "/api/v1/approvals", None, {}),
    (
        "get_approval",
        ("ap-1",),
        {"token": "secret"},
        "GET",
        "/api/v1/approvals/ap-1",
        None,
        {"authorization": "Bearer secret"},
    ),
    (
        "approve_approval",
        ("ap-1", 1),
        {"token": "secret", "actor_id": "alice"},
        "POST",
        "/api/v1/approvals/ap-1/approve",
        {"expected_version": 1},
        {"authorization": "Bearer secret", "x-human-actor-id": "alice"},
    ),
    (
        "reject_approval",
        ("ap-1", 1, "unsafe"),
        {"token": "secret", "actor_id": "alice"},
        "POST",
        "/api/v1/approvals/ap-1/reject",
        {"expected_version": 1, "reason": "unsafe"},
        {"authorization": "Bearer secret", "x-human-actor-id": "alice"},
    ),
    (
        "request_promotion",
        ("rm-1",),
        {},
        "POST",
        "/api/v1/models/rm-1/promotion-request",
        {"requested_by_type": "agent"},
        {},
    ),
    (
        "promote_model",
        ("rm-1", "ap-1"),
        {"token": "secret", "actor_id": "alice"},
        "POST",
        "/api/v1/models/rm-1/promote",
        {"approval_id": "ap-1"},
        {"authorization": "Bearer secret", "x-human-actor-id": "alice"},
    ),
    (
        "submit_feedback",
        ("pe-1", "confirmed_issue"),
        {"actor_id": "tech-1"},
        "POST",
        "/api/v1/predictions/pe-1/feedback",
        {"outcome": "confirmed_issue"},
        {"x-human-actor-id": "tech-1"},
    ),
    ("list_feedback", ("pe-1",), {}, "GET", "/api/v1/predictions/pe-1/feedback", None, {}),
    (
        "draft_work_order",
        ("ap-1",),
        {"token": "secret"},
        "POST",
        "/api/v1/cmms/work-orders/draft",
        {"approval_id": "ap-1"},
        {"authorization": "Bearer secret"},
    ),
    ("list_work_orders", (), {}, "GET", "/api/v1/cmms/work-orders", None, {}),
]


def _client(handler: Callable[[httpx.Request], httpx.Response]) -> APIClient:
    return APIClient(base_url="http://api.test", transport=httpx.MockTransport(handler))


def _json_response(payload: Any) -> httpx.Response:
    return httpx.Response(200, json=payload, headers={"x-request-id": "req-1"})


@pytest.mark.parametrize(
    "method_name,args,kwargs,exp_method,exp_path,exp_body,exp_headers", _CASES
)
def test_p1_client_methods(
    method_name: str,
    args: tuple,
    kwargs: dict[str, Any],
    exp_method: str,
    exp_path: str,
    exp_body: dict[str, Any],
    exp_headers: dict[str, str],
) -> None:
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["method"] = request.method
        captured["path"] = request.url.path
        captured["body"] = request.content
        captured["headers"] = {k.lower(): v for k, v in request.headers.items()}
        captured["query"] = dict(request.url.params)
        return _json_response({"ok": True})

    client = _client(handler)
    try:
        result = getattr(client, method_name)(*args, **kwargs)
        assert result == {"ok": True}
        assert captured["method"] == exp_method
        assert captured["path"] == exp_path
        if exp_body is not None:
            assert json.loads(captured["body"]) == exp_body
        for header_name, expected in exp_headers.items():
            assert captured["headers"].get(header_name) == expected
    finally:
        client.close()


def test_list_approvals_passes_status_query() -> None:
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["query"] = dict(request.url.params)
        return _json_response([])

    client = _client(handler)
    try:
        client.list_approvals(status="pending")
        assert captured["query"] == {"limit": "100", "offset": "0", "status": "pending"}
    finally:
        client.close()


def test_approve_approval_without_token_sends_no_authorization_header() -> None:
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["authorization"] = request.headers.get("authorization")
        return _json_response({"ok": True})

    client = _client(handler)
    try:
        client.approve_approval("ap-1", 1, actor_id="alice")
        assert captured["authorization"] is None
    finally:
        client.close()


def test_token_is_never_stored_on_client() -> None:
    client = _client(lambda r: _json_response({"ok": True}))
    try:
        client.get_approval("ap-1", token="super-secret")
        assert not hasattr(client, "token")
        assert "super-secret" not in repr(client.__dict__)
    finally:
        client.close()
