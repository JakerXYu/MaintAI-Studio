"""Request-ID validation.

Request IDs are user-supplied correlation tokens echoed back on responses. They
must be constrained to a safe charset so they can never be abused as path
basenames, log injection, or Windows reserved device names. Invalid IDs are
rejected with HTTP 400 at the API boundary.
"""

from __future__ import annotations

import re

_REQUEST_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")

# Windows reserves these basenames regardless of extension.
_WINDOWS_RESERVED_BASENAMES = frozenset(
    ["CON", "PRN", "AUX", "NUL"]
    + [f"COM{i}" for i in range(1, 10)]
    + [f"LPT{i}" for i in range(1, 10)]
)


class InvalidRequestIdError(ValueError):
    """Raised when a request ID is not safe to accept."""


def is_valid_request_id(request_id: object) -> bool:
    if not isinstance(request_id, str):
        return False
    if not _REQUEST_ID_RE.match(request_id):
        return False
    return request_id.upper() not in _WINDOWS_RESERVED_BASENAMES


def validate_request_id(request_id: str) -> str:
    if not is_valid_request_id(request_id):
        raise InvalidRequestIdError(
            f"invalid request_id {request_id!r}: must match "
            "^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$ and not be a Windows reserved "
            "device basename"
        )
    return request_id
