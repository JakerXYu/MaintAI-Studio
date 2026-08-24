"""Unit tests for the UI HTTP client using ``httpx.MockTransport``.

Every endpoint method is exercised for its HTTP verb, path, and JSON body, plus
error handling (4xx/5xx -> UIAPIError), timeouts, connectivity failures, and
safe URL path quoting (no raw ``/`` injection).
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import httpx
import pytest

from maintai.ui.api_client import DEFAULT_BASE_URL, APIClient, UIAPIError

# (method, args, expected_method, expected_path, expected_json_body or None)
_CASES: list[tuple[str, tuple, str, str, dict[str, Any] | None]] = [
    ("health", (), "GET", "/health", None),
    ("health_live", (), "GET", "/health/live", None),
    ("health_ready", (), "GET", "/health/ready", None),
    ("list_datasets", (), "GET", "/api/v1/datasets", None),
    ("get_dataset", ("abc123",), "GET", "/api/v1/datasets/abc123", None),
    ("profile_dataset", ("abc123",), "POST", "/api/v1/datasets/abc123/profile", None),
    ("get_quality", ("abc123",), "GET", "/api/v1/datasets/abc123/quality", None),
    (
        "recommend_task",
        ("abc123", "target", "asset", "ts"),
        "POST",
        "/api/v1/datasets/abc123/task-recommendation",
        {"target_column": "target", "asset_id_column": "asset", "timestamp_column": "ts"},
    ),
    (
        "create_experiment",
        ("abc123", ["random_forest"], 0.8),
        "POST",
        "/api/v1/experiments",
        {"dataset_id": "abc123", "model_names": ["random_forest"], "minimum_recall": 0.8},
    ),
    ("list_experiments", (), "GET", "/api/v1/experiments", None),
    ("get_experiment", ("exp1",), "GET", "/api/v1/experiments/exp1", None),
    ("get_comparison", ("exp1",), "GET", "/api/v1/experiments/exp1/comparison", None),
    ("list_models", (), "GET", "/api/v1/models", None),
    ("get_model", ("m1",), "GET", "/api/v1/models/m1", None),
    (
        "register_model",
        ("run1", "my-model"),
        "POST",
        "/api/v1/models/run1/register",
        {"name": "my-model"},
    ),
    ("deploy_demo", ("m1",), "POST", "/api/v1/models/m1/deploy-demo", None),
    (
        "predict_single",
        ("m1", {"a": 1}),
        "POST",
        "/api/v1/predict",
        {"model_id": "m1", "records": [{"a": 1}]},
    ),
    (
        "predict_batch",
        ("m1", [{"a": 1}, {"a": 2}]),
        "POST",
        "/api/v1/predict/batch",
        {"model_id": "m1", "records": [{"a": 1}, {"a": 2}]},
    ),
    (
        "copilot_chat",
        ("hello",),
        "POST",
        "/api/v1/copilot/chat",
        {"user_request": "hello"},
    ),
]


def _client(handler: Callable[[httpx.Request], httpx.Response]) -> APIClient:
    return APIClient(base_url="http://api.test", transport=httpx.MockTransport(handler))


def _json_response(status: int, payload: Any) -> httpx.Response:
    return httpx.Response(status, json=payload, headers={"x-request-id": "req-1"})


@pytest.mark.parametrize("method_name,args,exp_method,exp_path,exp_body", _CASES)
def test_client_methods(
    method_name: str,
    args: tuple,
    exp_method: str,
    exp_path: str,
    exp_body: dict[str, Any] | None,
) -> None:
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["method"] = request.method
        captured["path"] = request.url.path
        captured["body"] = request.content
        return _json_response(200, {"ok": True})

    client = _client(handler)
    try:
        result = getattr(client, method_name)(*args)
        assert result == {"ok": True}
        assert captured["method"] == exp_method
        assert captured["path"] == exp_path
        if exp_body is not None:
            assert json.loads(captured["body"]) == exp_body
    finally:
        client.close()


def test_upload_dataset_multipart() -> None:
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["content_type"] = request.headers.get("content-type", "")
        captured["body"] = request.content
        return _json_response(201, {"id": "ds-1"})

    client = _client(handler)
    try:
        result = client.upload_dataset("sensor.csv", b"a,b\n1,2")
        assert result == {"id": "ds-1"}
        assert captured["content_type"].startswith("multipart/form-data")
        assert b"sensor.csv" in captured["body"]
        assert b"application/octet-stream" in captured["body"]
    finally:
        client.close()


def test_list_datasets_query_params() -> None:
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["query"] = dict(request.url.params)
        return _json_response(200, [])

    client = _client(handler)
    try:
        client.list_datasets(limit=10, offset=20)
        assert captured["query"] == {"limit": "10", "offset": "20"}
    finally:
        client.close()


def test_error_status_raises_ui_api_error_with_detail() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return _json_response(422, {"detail": "upload exceeds the byte limit"})

    client = _client(handler)
    try:
        with pytest.raises(UIAPIError) as excinfo:
            client.list_datasets()
        assert excinfo.value.status_code == 422
        assert excinfo.value.detail == "upload exceeds the byte limit"
        assert excinfo.value.request_id == "req-1"
        assert "422" in str(excinfo.value)
    finally:
        client.close()


def test_error_status_without_json_detail() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="oops")

    client = _client(handler)
    try:
        with pytest.raises(UIAPIError) as excinfo:
            client.list_datasets()
        assert excinfo.value.status_code == 500
        assert excinfo.value.detail is None
    finally:
        client.close()


def test_timeout_raises_ui_api_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.TimeoutException("timed out")

    client = _client(handler)
    try:
        with pytest.raises(UIAPIError) as excinfo:
            client.health()
        assert "timed out" in str(excinfo.value)
        assert excinfo.value.status_code is None
    finally:
        client.close()


def test_connectivity_failure_raises_ui_api_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    client = _client(handler)
    try:
        with pytest.raises(UIAPIError) as excinfo:
            client.health()
        assert "unreachable" in str(excinfo.value)
    finally:
        client.close()


def test_path_segment_is_url_quoted() -> None:
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        return _json_response(200, {"id": "x"})

    client = _client(handler)
    try:
        client.get_dataset("a/b")
        url = captured["url"]
        assert "/datasets/a%2Fb" in url
        assert "/datasets/a/b" not in url
    finally:
        client.close()


def test_base_url_from_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MAINTAI_API_URL", "http://env-host:9000")
    client = APIClient(transport=httpx.MockTransport(lambda r: _json_response(200, {})))
    try:
        assert client.base_url == "http://env-host:9000"
    finally:
        client.close()


def test_default_base_url() -> None:
    client = APIClient(transport=httpx.MockTransport(lambda r: _json_response(200, {})))
    try:
        assert client.base_url == DEFAULT_BASE_URL
    finally:
        client.close()


def test_trailing_slash_is_stripped() -> None:
    transport = httpx.MockTransport(lambda r: _json_response(200, {}))
    client = APIClient(base_url="http://api.test/", transport=transport)
    try:
        assert client.base_url == "http://api.test"
    finally:
        client.close()
