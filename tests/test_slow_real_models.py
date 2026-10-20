"""Checks against the real models. Skipped unless --run-slow.

These are the only tests that download weights. They exist to confirm, on
the real checkpoints, what the fast tests assume with fakes.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.slow


def test_real_nli_cross_encoder_returns_three_named_probabilities():
    pytest.importorskip("transformers")
    from src.evaluation.utility import UtilityEvaluator

    evaluator = UtilityEvaluator(load_models=True)  # ~1.7 GB for nli-deberta-v3-large
    if evaluator.nli_pipeline is None:
        pytest.skip("NLI model could not be loaded")

    probs, backend = evaluator._nli_with_backend("A man is cooking pasta in a kitchen.", "A person is cooking.")
    assert backend == "model"
    assert set(probs) == {"contradiction", "entailment", "neutral"}
    assert abs(sum(probs.values()) - 1.0) < 1e-3
    assert probs["entailment"] > probs["contradiction"]

    contra, _ = evaluator._nli_with_backend("A man is cooking pasta in a kitchen.", "Nobody is cooking.")
    assert contra["contradiction"] > contra["entailment"]


def test_real_phi4_attacker_answers_from_the_model():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("Phi-4-mini needs a GPU for a practical test")
    from src.evaluation.attacker import Phi4MiniAttacker

    attacker = Phi4MiniAttacker(load_model=True, strict=True)
    res = attacker.predict_attribute("I take the Green Line to Fenway after my shift.", "location")
    assert res["mode"] == "model"
    assert "Exact age or range" not in res["prediction"]
    assert "City, State, or Region" not in res["prediction"]
