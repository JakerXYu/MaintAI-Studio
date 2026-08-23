"""Minimal schema bootstrap for Phase A.

Uses ``Base.metadata.create_all`` — the smallest correct approach for Phase A.
Alembic is introduced only if migrations are required before P0 freeze.
"""

from __future__ import annotations

from sqlalchemy.engine import Engine

from maintai.config import get_settings

# Import models so they register on Base.metadata before create_all.
from maintai.db import models as _models  # noqa: F401
from maintai.db.base import Base
from maintai.db.session import build_engine


def create_all(engine: Engine | None = None) -> Engine:
    engine = engine or build_engine(get_settings().database_url)
    Base.metadata.create_all(engine)
    return engine


def main() -> int:
    engine = create_all()
    print(f"Schema ready on: {engine.url.render_as_string(hide_password=True)}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
