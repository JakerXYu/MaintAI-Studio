"""Health endpoint and request-id middleware tests."""

import re

from fastapi.testclient import TestClient

from maintai.api.main import create_app


def test_health_root(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["service"]
    assert body["version"]


def test_health_live(client):
    r = client.get("/health/live")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_health_ready(client):
    r = client.get("/health/ready")
    assert r.status_code == 200
    assert r.json()["status"] == "ready"
    assert r.json()["database"] == "ok"
    assert r.json()["mlflow"] == "ok"


def test_health_ready_unavailable_returns_503():
    def broken_factory():
        raise RuntimeError("db down")

    app = create_app(session_factory=broken_factory, mlflow_healthcheck=lambda: None)
    with TestClient(app) as c:
        r = c.get("/health/ready")
    assert r.status_code == 503
    assert r.json()["status"] == "not_ready"


def test_health_ready_mlflow_unavailable_returns_503(session_factory):
    def broken_mlflow():
        raise RuntimeError("mlflow down")

    app = create_app(
        session_factory=session_factory,
        mlflow_healthcheck=broken_mlflow,
    )
    with TestClient(app) as c:
        r = c.get("/health/ready")
    assert r.status_code == 503
    assert r.json() == {
        "status": "not_ready",
        "database": "ok",
        "mlflow": "unavailable",
    }


def test_request_id_echoed(client):
    r = client.get("/health", headers={"X-Request-ID": "abc123"})
    assert r.status_code == 200
    assert r.headers.get("x-request-id") == "abc123"


def test_request_id_generated_when_missing(client):
    r = client.get("/health")
    assert r.status_code == 200
    rid = r.headers.get("x-request-id")
    assert rid
    assert re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", rid)


def test_request_id_invalid_rejected(client):
    r = client.get("/health", headers={"X-Request-ID": "../etc/passwd"})
    assert r.status_code == 400
    assert r.headers.get("x-request-id")


def test_windows_reserved_request_id_rejected(client):
    r = client.get("/health", headers={"X-Request-ID": "CON"})
    assert r.status_code == 400


def test_unhandled_error_is_redacted_and_correlated(session_factory):
    app = create_app(session_factory=session_factory, mlflow_healthcheck=lambda: None)

    @app.get("/boom")
    def boom():
        raise RuntimeError("sensitive detail")

    with TestClient(app, raise_server_exceptions=False) as c:
        r = c.get("/boom", headers={"X-Request-ID": "failure-123"})
    assert r.status_code == 500
    assert r.headers["X-Request-ID"] == "failure-123"
    assert "sensitive" not in r.text
