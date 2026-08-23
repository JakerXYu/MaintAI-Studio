"""FastAPI application factory and health routes.

FastAPI is the only business entry point. The app factory accepts an injected
``session_factory`` and ``settings`` so tests can bind a SQLite in-memory
database without touching the global config.
"""

from __future__ import annotations

import uuid
from collections.abc import Callable
from urllib.request import urlopen

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from sqlalchemy import text
from sqlalchemy.orm import sessionmaker

from maintai import __version__
from maintai.api.request_id import InvalidRequestIdError, validate_request_id
from maintai.config import Settings, get_settings
from maintai.db.session import get_session_factory


def _mlflow_healthcheck(tracking_uri: str) -> None:
    health_url = f"{tracking_uri.rstrip('/')}/health"
    with urlopen(health_url, timeout=3) as response:  # noqa: S310 - configured service URL
        if response.status != 200:
            raise RuntimeError("MLflow health endpoint is unavailable")


def create_app(
    session_factory: sessionmaker | None = None,
    settings: Settings | None = None,
    mlflow_healthcheck: Callable[[], None] | None = None,
) -> FastAPI:
    settings = settings or get_settings()
    session_factory = session_factory or get_session_factory()
    mlflow_healthcheck = mlflow_healthcheck or (
        lambda: _mlflow_healthcheck(settings.mlflow_tracking_uri)
    )

    app = FastAPI(title=settings.project_name, version=__version__)

    @app.middleware("http")
    async def request_id_middleware(request: Request, call_next):
        request_id = request.headers.get(settings.request_id_header)
        if request_id is None:
            request_id = uuid.uuid4().hex
        else:
            try:
                validate_request_id(request_id)
            except InvalidRequestIdError:
                return JSONResponse(
                    status_code=400,
                    content={"detail": "invalid request id"},
                    headers={settings.request_id_header: uuid.uuid4().hex},
                )
        request.state.request_id = request_id
        try:
            response = await call_next(request)
        except Exception:  # noqa: BLE001 - never expose dependency tracebacks
            response = JSONResponse(
                status_code=500,
                content={
                    "detail": "internal server error",
                    "request_id": request_id,
                },
            )
        response.headers[settings.request_id_header] = request_id
        return response

    @app.get("/health", tags=["health"])
    def health() -> dict:
        return {
            "status": "ok",
            "service": settings.project_name,
            "version": __version__,
        }

    @app.get("/health/live", tags=["health"])
    def health_live() -> dict:
        return {"status": "ok"}

    @app.get("/health/ready", tags=["health"])
    def health_ready() -> JSONResponse:
        try:
            with session_factory() as session:
                session.execute(text("SELECT 1"))
        except Exception:  # noqa: BLE001 - readiness must not raise
            return JSONResponse(
                status_code=503,
                content={
                    "status": "not_ready",
                    "database": "unavailable",
                    "mlflow": "unknown",
                },
            )
        try:
            mlflow_healthcheck()
        except Exception:  # noqa: BLE001 - readiness must not raise
            return JSONResponse(
                status_code=503,
                content={
                    "status": "not_ready",
                    "database": "ok",
                    "mlflow": "unavailable",
                },
            )
        return JSONResponse(
            status_code=200,
            content={"status": "ready", "database": "ok", "mlflow": "ok"},
        )

    return app


app = create_app()
