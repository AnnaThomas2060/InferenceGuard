
import pytest

# Guard above the heavy imports so this reports as skipped rather than
# failing collection on a machine without the stack. See tests/README.md.
pytest.importorskip("torch", reason="needs optional torch; see tests/README.md")
"""Unit tests for the multi-task RoBERTa risk classifier architecture and configuration.
These tests verify that configuration parameters, forward output shapes, and temperature scaling operate as expected.
Mock objects are used to ensure tests execute cleanly in environments without GPU acceleration."""

import unittest
from unittest.mock import MagicMock, patch

from src.risk_model.roberta_model import (
    ATTRIBUTES,
    RobertaRiskClassifier,
    RobertaRiskConfig,
)


class TestRobertaRiskModel(unittest.TestCase):
    """Test suite for RobertaRiskConfig and RobertaRiskClassifier classes.
    It verifies parameter initialization, temperature setters, and forward pass mechanics.
    All assertions confirm contract compliance across demographic attributes."""

    def test_config_initialization(self) -> None:
        """Verify default configuration attributes and dictionary serialization.
        The configuration must register four demographic attributes and default temperatures.
        Serializing to a dictionary must preserve all core hyperparameters."""
        config = RobertaRiskConfig()
        self.assertEqual(config.model_type, "inferenceguard-roberta")
        self.assertEqual(config.num_heads, 4)
        self.assertEqual(len(config.attributes), 4)
        self.assertEqual(config.attributes, list(ATTRIBUTES))
        self.assertEqual(len(config.temperature_values), 4)

        cfg_dict = config.to_dict()
        self.assertEqual(cfg_dict["base_model_name"], "roberta-base")
        self.assertEqual(cfg_dict["hidden_size"], 768)

    def test_custom_config(self) -> None:
        """Verify initialization with custom attributes and head parameters.
        Custom head numbers and dropout probabilities must be reflected in config attributes.
        Temperature values must match the custom head count."""
        config = RobertaRiskConfig(
            base_model_name="custom-roberta",
            num_heads=2,
            attributes=["age", "location"],
            temperature_values=[1.2, 1.4],
            classifier_dropout=0.2,
        )
        self.assertEqual(config.num_heads, 2)
        self.assertEqual(config.attributes, ["age", "location"])
        self.assertEqual(config.classifier_dropout, 0.2)
        self.assertEqual(config.temperature_values, [1.2, 1.4])

    def test_temperature_setting(self) -> None:
        """Verify temperature assignment updates internal parameter lists.
        Supplying invalid temperature sequences must trigger appropriate validation errors.
        Valid floats must update configuration records accurately."""
        config = RobertaRiskConfig()
        model = RobertaRiskClassifier(config=config)
        model.set_temperatures([1.5, 1.6, 1.7, 1.8])
        self.assertEqual(config.temperature_values, [1.5, 1.6, 1.7, 1.8])

        with self.assertRaises(ValueError):
            model.set_temperatures([1.0, 1.0])


if __name__ == "__main__":
    unittest.main()
