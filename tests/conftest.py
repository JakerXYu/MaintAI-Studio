"""Shared test fixtures.

Tests use an in-memory SQLite database (test double) via SQLAlchemy; the
runtime Postgres path is exercised by docker-compose, not by pytest.
"""

from __future__ import annotations

import os

# Ensure the module-level app in maintai.api.main never points at Postgres.
os.environ.setdefault("DATABASE_URL", "sqlite+pysqlite:///:memory:")

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

import maintai.db.models  # noqa: F401  (register models on Base.metadata)
from maintai.api.main import create_app
from maintai.db.base import Base


@pytest.fixture()
def engine():
    engine = create_engine(
        "sqlite+pysqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    return engine


@pytest.fixture()
def session_factory(engine):
    return sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


@pytest.fixture()
def client(session_factory):
    app = create_app(session_factory=session_factory, mlflow_healthcheck=lambda: None)
    with TestClient(app) as test_client:
        yield test_client
