"""Tests for the user-facing risk band translation.

The interesting case is the last one: it pins the behaviour that stopped us
reporting a failed rewrite as our best result in Session 04.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.product.risk_bands import (  # noqa: E402
    Band,
    Cue,
    band_for,
    describe,
    format_delta,
    summarize,
)


@pytest.mark.parametrize(
    "score, expected",
    [
        (0.0, Band.LOW),
        (0.29, Band.LOW),
        (0.30, Band.MEDIUM),  # edge belongs to the higher band
        (0.59, Band.MEDIUM),
        (0.60, Band.HIGH),  # edge belongs to the higher band
        (1.0, Band.HIGH),
    ],
)
def test_band_edges(score, expected):
    assert band_for(score) is expected


@pytest.mark.parametrize("score", [-0.01, 1.01])
def test_band_rejects_out_of_range(score):
    with pytest.raises(ValueError):
        band_for(score)


def test_primary_is_worst_attribute_and_secondary_skips_low():
    summary = summarize(
        {"location": 0.88, "education": 0.79, "occupation": 0.41, "age": 0.12}
    )
    assert summary.primary == "location"
    assert summary.overall == 0.88
    assert summary.overall_band is Band.HIGH
    # age is LOW, so it is not padded into the secondary list
    assert summary.secondary == ("education", "occupation")


def test_unmeasured_attribute_is_omitted_not_zeroed():
    summary = summarize({"location": 0.7})
    assert set(summary.scores) == {"location"}
    assert "age" not in summary.bands  # absent, not reported as a safe 0.0


def test_unknown_attributes_are_ignored_and_empty_input_raises():
    assert summarize({"location": 0.5, "income": 0.9}).scores == {"location": 0.5}
    with pytest.raises(ValueError):
        summarize({"income": 0.9})


def test_ties_resolve_deterministically():
    first = summarize({"location": 0.5, "education": 0.5})
    second = summarize({"education": 0.5, "location": 0.5})
    assert first.primary == second.primary == "education"  # alphabetical on a tie


def test_weak_cues_dropped_and_strong_cues_ranked():
    summary = summarize(
        {"location": 0.8},
        cues=[
            Cue("the", "location", 0.05),  # below the floor
            Cue("Green Line", "location", 0.73),
            {"span": "co-op", "attribute": "occupation", "importance": 0.44},
        ],
    )
    assert [c.span for c in summary.cues] == ["Green Line", "co-op"]


def test_describe_frames_output_as_attacker_inference():
    text = describe(summarize({"location": 0.88, "education": 0.65}))
    assert "an attacker" in text
    assert "could plausibly infer" in text
    # never asserts the attribute as fact about the user
    assert "you are" not in text.lower()


def test_describe_is_quiet_when_nothing_is_risky():
    assert "No strong attribute inference" in describe(summarize({"location": 0.1}))


def test_one_collapsing_attribute_cannot_hide_the_rest():
    """Regression guard for the resume row in session04_usage_log.csv.

    Location fell 0.859 -> 0.006 while the name, email and phone survived. An
    averaged overall would have called that a large win; taking the max keeps
    the message HIGH and the reported overall delta near zero.
    """
    before = summarize({"location": 0.859, "occupation": 0.997, "age": 0.982})
    after = summarize({"location": 0.006, "occupation": 0.987, "age": 0.982})

    assert after.overall_band is Band.HIGH
    assert after.primary == "occupation"

    deltas = format_delta(before, after)
    assert deltas["location"] == 0.853  # the tempting headline
    assert deltas["overall"] == 0.01  # what we actually achieved
