"""Smoke + source-safety tests for the Streamlit app.

The app is designed to be import-safe (the Streamlit UI only runs under
``if __name__ == "__main__"``), so a subprocess import proves it does not crash
at import time. Source-safety tests assert the UI never imports the data/ML/DB/
application layers, never imports MLflow/sklearn/SQLAlchemy, and never opens
files directly — the UI is an HTTP-only client.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_SRC_DIR = _PROJECT_ROOT / "src"
_UI_DIR = _SRC_DIR / "maintai" / "ui"

_UI_SOURCE_FILES = ("app.py", "api_client.py", "render.py")

# Modules the UI must never import (HTTP-only client boundary).
_FORBIDDEN_IMPORT_TOKENS = (
    "maintai.data",
    "maintai.ml",
    "maintai.db",
    "maintai.application",
    "maintai.agent",
    "maintai.mlops",
    "maintai.tasks",
    "maintai.audit",
    "import mlflow",
    "from mlflow",
    "import sklearn",
    "from sklearn",
    "import sqlalchemy",
    "from sqlalchemy",
    "import xgboost",
    "from xgboost",
    "import shap",
    "from shap",
)

# The UI must not touch the filesystem directly.
_FORBIDDEN_FS_TOKENS = (
    "open(",
    "pathlib",
    "Path(",
    "read_text(",
    "write_text(",
    "read_bytes(",
)


def _read_source(filename: str) -> str:
    path = _UI_DIR / filename
    assert path.exists(), f"missing UI source file: {path}"
    return path.read_text(encoding="utf-8")


def test_app_imports_via_subprocess() -> None:
    """The app module must import cleanly without a Streamlit runtime."""
    env = dict(os.environ)
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(_SRC_DIR) + (os.pathsep + existing if existing else "")
    proc = subprocess.run(
        [sys.executable, "-c", "import maintai.ui.app; print('ok')"],
        cwd=str(_PROJECT_ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, f"import failed:\n{proc.stderr}"
    assert "ok" in proc.stdout


def test_api_client_imports_via_subprocess() -> None:
    env = dict(os.environ)
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(_SRC_DIR) + (os.pathsep + existing if existing else "")
    proc = subprocess.run(
        [sys.executable, "-c", "import maintai.ui.api_client, maintai.ui.render; print('ok')"],
        cwd=str(_PROJECT_ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, f"import failed:\n{proc.stderr}"


@pytest.mark.parametrize("filename", _UI_SOURCE_FILES)
def test_ui_source_has_no_forbidden_imports(filename: str) -> None:
    source = _read_source(filename)
    for token in _FORBIDDEN_IMPORT_TOKENS:
        assert token not in source, f"{filename} imports forbidden {token!r}"


@pytest.mark.parametrize("filename", _UI_SOURCE_FILES)
def test_ui_source_has_no_direct_filesystem_access(filename: str) -> None:
    source = _read_source(filename)
    for token in _FORBIDDEN_FS_TOKENS:
        assert token not in source, f"{filename} accesses the filesystem via {token!r}"


def test_api_client_only_depends_on_httpx() -> None:
    """The client module must use httpx and nothing else transport-related."""
    source = _read_source("api_client.py")
    assert "import httpx" in source
    assert "requests" not in source
    assert "urllib3" not in source
    assert "aiohttp" not in source


def test_app_guards_streamlit_entrypoint() -> None:
    """The Streamlit entrypoint must be behind the __main__ guard."""
    source = _read_source("app.py")
    assert 'if __name__ == "__main__":' in source
