"""Pydantic contracts for deterministic task recommendation."""

from __future__ import annotations

from pydantic import BaseModel, Field


class Alternative(BaseModel):
    """A secondary task the user may choose instead."""

    task: str
    confidence: float
    reason: str


class TaskRecommendation(BaseModel):
    """Deterministic classification / regression recommendation."""

    recommended_task: str
    trainable: bool
    confidence: float
    target_column: str | None = None
    evidence: list[str] = Field(default_factory=list)
    alternatives: list[Alternative] = Field(default_factory=list)
