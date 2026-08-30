"""Unit tests for synthetic production replay."""

import numpy as np
import pandas as pd

from maintai.monitoring.contracts import REPLAY_KINDS
from maintai.monitoring.replay import generate_batches


def _baseline(seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "temp": rng.normal(50.0, 10.0, 200),
            "vib": rng.normal(1.0, 0.3, 200),
            "mode": rng.choice(["a", "b", "c"], 200),
            "failure": rng.integers(0, 2, 200),
        }
    )


def test_four_batches_generated_with_metadata():
    out = generate_batches(_baseline())
    assert set(out.frames) == set(REPLAY_KINDS)
    assert set(out.metadata) == set(REPLAY_KINDS)
    for kind in REPLAY_KINDS:
        assert len(out.frames[kind]) == 200
        assert out.metadata[kind].row_count == 200
        assert out.metadata[kind].transformations
        joined = " ".join(out.metadata[kind].transformations).lower()
        assert "not real production" in joined


def test_input_not_modified():
    base = _baseline()
    snapshot = base.copy(deep=True)
    generate_batches(base, seed=42)
    pd.testing.assert_frame_equal(base, snapshot)


def test_reproducible_with_fixed_seed():
    base = _baseline()
    out1 = generate_batches(base, seed=42)
    out2 = generate_batches(base, seed=42)
    for kind in REPLAY_KINDS:
        pd.testing.assert_frame_equal(out1.frames[kind], out2.frames[kind])


def test_different_seeds_differ():
    base = _baseline()
    out1 = generate_batches(base, seed=1)
    out2 = generate_batches(base, seed=2)
    assert not out1.frames["mild"].equals(out2.frames["mild"])


def test_normal_batch_stays_close_to_baseline():
    base = _baseline()
    out = generate_batches(base, seed=0)
    shift = abs(out.frames["normal"]["temp"].mean() - base["temp"].mean())
    assert shift < 1.0


def test_severe_batch_shifts_more_than_mild():
    base = _baseline()
    out = generate_batches(base, seed=0)
    mild_shift = abs(out.frames["mild"]["temp"].mean() - base["temp"].mean())
    severe_shift = abs(out.frames["severe"]["temp"].mean() - base["temp"].mean())
    assert severe_shift > mild_shift


def test_target_shift_only_when_target_provided():
    base = _baseline()
    base["failure"] = 0  # all negatives
    # Without target, the failure column is treated as a normal numeric feature.
    out = generate_batches(base.drop(columns=["failure"]), seed=0, target="failure")
    # target not in frame -> no target shift, but frame must still be produced
    assert "failure" not in out.frames["increased_failure_risk"].columns


def test_target_shift_increases_failure_rate():
    base = _baseline()
    base["failure"] = 0  # all negatives
    out = generate_batches(base, seed=0, target="failure")
    rate = float(out.frames["increased_failure_risk"]["failure"].mean())
    assert rate > 0.0
    # non-increased batches leave the all-negative target untouched
    for kind in ("normal", "mild", "severe"):
        assert out.frames[kind]["failure"].sum() == 0


def test_non_zero_one_target_is_not_shifted_or_misreported():
    base = _baseline()
    base["failure"] = np.where(base["failure"] == 0, 1, 2)
    out = generate_batches(base, seed=0, target="failure")
    pd.testing.assert_series_equal(
        out.frames["increased_failure_risk"]["failure"],
        base["failure"],
    )
    assert out.metadata["increased_failure_risk"].params["target_shifted"] is False


def test_flatline_is_optional_and_recorded():
    base = _baseline()
    out = generate_batches(base, seed=0, flatline=True)
    assert out.metadata["severe"].params["flatline_injected"] is True
    values = out.frames["severe"]["temp"].to_numpy(dtype=float)
    diffs = np.diff(values)
    assert (diffs == 0).sum() > 0  # contiguous stalled block exists
    # Without flatline the same metadata flag is False
    out_no = generate_batches(base, seed=0, flatline=False)
    assert out_no.metadata["severe"].params["flatline_injected"] is False
