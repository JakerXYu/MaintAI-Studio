"""Keep duplicated HTTP-only UI choices aligned with backend contracts."""

from maintai.feedback.service import FEEDBACK_OUTCOMES
from maintai.ui import render


def test_feedback_outcomes_match_ui() -> None:
    assert set(render.FEEDBACK_OUTCOMES) == FEEDBACK_OUTCOMES
