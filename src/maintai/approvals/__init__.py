"""Approval subsystem (P1 human-in-the-loop gate).

Exposes the :class:`ApprovalService` plus its stable error surface. The CAS
conflict error is owned by the repository (see
:mod:`maintai.db.approval_repository`) and re-exported here for a single import
surface.
"""

from maintai.approvals.service import (
    ApprovalNotFoundError,
    ApprovalService,
    ApprovalServiceError,
    ApprovalValidationError,
)
from maintai.db.approval_repository import ApprovalConflictError

__all__ = [
    "ApprovalConflictError",
    "ApprovalNotFoundError",
    "ApprovalService",
    "ApprovalServiceError",
    "ApprovalValidationError",
]
