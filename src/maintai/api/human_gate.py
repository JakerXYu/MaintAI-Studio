"""Shared local-demo bearer and human actor gate for approved P1 actions."""

from __future__ import annotations

import secrets

from fastapi import HTTPException, Request


def require_bearer(request: Request, token: str | None) -> None:
    """Require the configured local-demo bearer token."""
    if token is None:
        raise HTTPException(
            status_code=503,
            detail="approval decisions are disabled: APPROVAL_API_TOKEN is not configured",
        )
    authorization = request.headers.get("authorization")
    if authorization is None:
        raise HTTPException(status_code=401, detail="missing bearer token")
    scheme, _, credentials = authorization.partition(" ")
    if scheme.lower() != "bearer" or not credentials:
        raise HTTPException(status_code=401, detail="invalid authorization header")
    if not secrets.compare_digest(credentials.encode("utf-8"), token.encode("utf-8")):
        raise HTTPException(status_code=401, detail="invalid bearer token")


def require_human(request: Request, token: str | None) -> str:
    """Enforce the bearer gate and return a non-empty human actor id."""
    require_bearer(request, token)
    actor_id = request.headers.get("x-human-actor-id")
    if actor_id is None or not actor_id.strip():
        raise HTTPException(status_code=422, detail="X-Human-Actor-ID header is required")
    return actor_id.strip()
