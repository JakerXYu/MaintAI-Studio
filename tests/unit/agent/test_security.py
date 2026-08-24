"""Security guard: the copilot source must not contain dangerous constructs.

The agent package (and the copilot API module) is read-only and must never be
able to evaluate/execute code, spawn subprocesses, talk to the database
directly, or open files. This test reads the source and asserts the forbidden
tokens are absent.
"""

from __future__ import annotations

from pathlib import Path

import pytest


def _project_root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "pyproject.toml").is_file():
            return parent
    raise AssertionError("pyproject.toml not found")


_AGENT_DIR = _project_root() / "src" / "maintai" / "agent"
_API_COPILOT = _project_root() / "src" / "maintai" / "api" / "copilot.py"

_FORBIDDEN = (
    "eval(",
    "exec(",
    "subprocess",
    "sqlalchemy",
    "open(",
    "__import__",
    "os.system",
    "shell=True",
)

_TARGETS = sorted(_AGENT_DIR.glob("*.py")) + [_API_COPILOT]


@pytest.mark.parametrize("path", _TARGETS)
def test_agent_source_has_no_dangerous_constructs(path):
    text = path.read_text(encoding="utf-8")
    for token in _FORBIDDEN:
        assert token not in text, f"{path.name} contains forbidden token {token!r}"
