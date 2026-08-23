"""Engine and session factory helpers.

Runtime uses Postgres (via ``postgresql+psycopg``); tests use SQLite. The
``build_engine`` helper applies the right connect options per dialect.
"""

from __future__ import annotations

from typing import Any

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from maintai.config import get_settings


def build_engine(database_url: str, *, echo: bool = False) -> Engine:
    kwargs: dict[str, Any] = {"echo": echo}
    if database_url.startswith("sqlite"):
        kwargs["connect_args"] = {"check_same_thread": False}
    else:
        kwargs["pool_pre_ping"] = True
    return create_engine(database_url, **kwargs)


def build_session_factory(engine: Engine) -> sessionmaker[Session]:
    return sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


def get_engine() -> Engine:
    return build_engine(get_settings().database_url)


def get_session_factory() -> sessionmaker[Session]:
    return build_session_factory(get_engine())
