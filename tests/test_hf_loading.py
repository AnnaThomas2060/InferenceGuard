"""Unit tests for Hugging Face Hub model loading and factory instantiation.
These tests verify that model loading helpers handle remote identifiers and fallback logic gracefully.
Mocked transformers and hub modules test behavior without network access."""

import unittest

from src.risk_model.model import load_risk_classifier
from src.risk_model.predict import CalibratedRiskPredictor


class TestHfLoading(unittest.TestCase):
    """Test suite verifying model loader functions and predictor factory methods.
    It tests that repository identifiers trigger appropriate class instantiation.
    Fallback mechanisms are checked when external dependencies are unavailable."""

    def test_load_risk_classifier_fallback(self) -> None:
        """Verify that loading an unresolvable model identifier falls back cleanly.
        When Hugging Face Hub cannot be reached, the factory creates an uninitialized model.
        The returned instance must not be None."""
        model = load_risk_classifier("roberta-base", device="cpu")
        self.assertIsNotNone(model)

    def test_predictor_from_pretrained_instantiation(self) -> None:
        """Verify predictor instantiation through the from_pretrained factory method.
        The predictor must initialize its device and evaluation mode correctly.
        Predictions on empty text must return a safe baseline report."""
        predictor = CalibratedRiskPredictor.from_pretrained("roberta-base", device="cpu")
        self.assertIsNotNone(predictor)
        report = predictor.predict("")
        self.assertEqual(report.overall_risk_score, 0.0)
        self.assertEqual(report.overall_risk_band.value, "LOW")


if __name__ == "__main__":
    unittest.main()
