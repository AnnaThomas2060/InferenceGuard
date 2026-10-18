"""Utility guardrail: degenerate rewrites score 0, and the backend is named.

The regression case is the Session 04 resume rewrite "123 an area, an area an
area", which MiniLM cosine rated 0.907 (docs/methodology-review.md finding 6).
"""

from __future__ import annotations

import pytest

from src.evaluation.utility import UtilityEvaluator, check_degeneracy
from src.rewriter.generate_training_data import pareto_rejection_sample

GARBAGE = "123 an area, an area an area"

NORMAL_SENTENCES = [
    "Fix this resume: John Doe | 123 Main St, Denver, CO | jdoe@example.com",
    "I moved to Seattle to join a tech startup as a designer.",
    "Can you help me plan a two week trip to Japan next spring with my kids?",
    "The quick brown fox jumps over the lazy dog and then the dog sleeps on the porch.",
]


@pytest.mark.parametrize("original", NORMAL_SENTENCES)
def test_session04_garbage_is_degenerate_with_zero_utility(original):
    metrics = UtilityEvaluator().compute_metrics(original, GARBAGE)
    assert metrics["degenerate"] is True
    assert metrics["degenerate_reason"] == "repetition"
    assert metrics["utility_score"] == 0.0
    assert UtilityEvaluator().is_acceptable_utility(metrics, min_cosine=0.0, min_utility=0.0) is False


@pytest.mark.parametrize(
    "original,rewrite",
    [
        ("I am 26 years old living in Denver, CO.", "I am someone living locally."),
        ("I moved to Seattle to join a tech startup as a designer.",
         "I relocated locally to join an organization in a creative role."),
        ("Order 66 shipped on 12 04 2025 to unit 3B.", "Order 66 shipped on 12 04 2025 to unit 3B."),
    ],
)
def test_normal_rewrites_are_not_degenerate(original, rewrite):
    assert check_degeneracy(original, rewrite)["degenerate"] is False


def test_degeneracy_reasons():
    long = "I am planning a quiet weekend at home reading and cooking with friends nearby."
    assert check_degeneracy(long, "")["reason"] == "empty"
    assert check_degeneracy(long, "Weekend.")["reason"] == "length"
    assert check_degeneracy("Call me later about the plan.", "11 22 33 44 55 66 about")["reason"] == "non_alphabetic"


def test_fallback_backend_is_labelled():
    metrics = UtilityEvaluator().compute_metrics("I like tea.", "I enjoy tea.")
    assert metrics["utility_backend"] == "fallback"


class FakeNLIPipeline:
    def __init__(self, labels=("CONTRADICTION", "ENTAILMENT", "NEUTRAL")) -> None:
        self.labels = labels
        self.calls = []

    def __call__(self, inputs, **kwargs):
        self.calls.append((inputs, kwargs))
        scores = (0.1, 0.7, 0.2)
        return [{"label": lbl, "score": s} for lbl, s in zip(self.labels, scores)]


def test_nli_passes_a_text_pair_and_reads_all_three_labels():
    evaluator = UtilityEvaluator()
    evaluator.nli_pipeline = FakeNLIPipeline()

    probs = evaluator.compute_nli_direction("A man is cooking.", "Someone is cooking.")

    inputs, kwargs = evaluator.nli_pipeline.calls[0]
    assert inputs == {"text": "A man is cooking.", "text_pair": "Someone is cooking."}
    assert kwargs.get("top_k", "unset") is None
    assert probs == {"contradiction": 0.1, "entailment": 0.7, "neutral": 0.2}


def test_nli_with_unnamed_labels_falls_back_rather_than_guessing():
    evaluator = UtilityEvaluator()
    evaluator.nli_pipeline = FakeNLIPipeline(labels=("LABEL_0", "LABEL_1", "LABEL_2"))
    _, backend = evaluator._nli_with_backend("A man is cooking.", "Someone is cooking.")
    assert backend == "fallback"


def test_backend_is_model_only_when_every_component_is():
    torch = pytest.importorskip("torch")

    class FakeEmbedder:
        def encode(self, text, convert_to_tensor=True):
            return torch.ones(4)

    evaluator = UtilityEvaluator()
    evaluator.nli_pipeline = FakeNLIPipeline()
    assert evaluator.compute_metrics("I like tea.", "I enjoy tea.")["utility_backend"] == "fallback"

    evaluator.similarity_model = FakeEmbedder()
    assert evaluator.compute_metrics("I like tea.", "I enjoy tea.")["utility_backend"] == "model"


def test_pareto_never_selects_a_degenerate_candidate():
    original = NORMAL_SENTENCES[0]
    selected = pareto_rejection_sample(
        original=original,
        candidates=[GARBAGE],
        max_risk_threshold=1.01,
        min_cosine_threshold=0.0,
    )
    assert selected is None
