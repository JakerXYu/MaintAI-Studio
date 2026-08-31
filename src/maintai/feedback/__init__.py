"""Technician feedback subsystem (P1 append-only technician feedback capture).

Exposes the :class:`FeedbackService` plus its stable error surface for a single
import surface (mirrors :mod:`maintai.approvals`).
"""

from maintai.feedback.service import (
    FeedbackPredictionNotFoundError,
    FeedbackService,
    FeedbackServiceError,
    FeedbackValidationError,
)

__all__ = [
    "FeedbackPredictionNotFoundError",
    "FeedbackService",
    "FeedbackServiceError",
    "FeedbackValidationError",
]
