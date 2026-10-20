# src/risk_model/predict.py
"""Production prediction service for calibrated attribute risk modeling.

Combines multi-task ModernBERT logits with post-hoc temperature scaling,
Shannon entropy and margin adjustment, and syntax-aligned cue attribution.
"""

from __future__ import annotations

from typing import Dict, List, Optional
import torch

from src.contracts.schemas import (
    AttributeName,
    AttributeRiskDistribution,
    AttributedCue,
    RiskBand,
    RiskReport,
)
from src.risk_model.calibrate import AttributeTemperatureScaler


class CalibratedRiskPredictor:
    """Predicts calibrated multi-attribute inferential risk and coordinates cue attribution."""

    def __init__(
        self,
        model,
        tokenizer,
        scaler: AttributeTemperatureScaler,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """Initialize the predictor with model, tokenizer, and scaler."""
        self.model = model.to(device).eval()
        self.tokenizer = tokenizer
        self.scaler = scaler.to(device).eval()
        self.device = device

    @torch.no_grad()
    def predict(self, text: str) -> RiskReport:
        """Compute calibrated multi-attribute risk distributions without cue attribution."""
        return self.predict_with_cues(text=text, attributor=None)

    def predict_with_cues(
        self,
        text: str,
        attributor=None,
        explicit_pii_counts: Optional[Dict[str, int]] = None,
    ) -> RiskReport:
        """Compute calibrated attribute risks and optionally extract high-salience cues."""
        if not text or not text.strip():
            # Return baseline uniform safe distribution
            empty_attrs = {
                attr: AttributeRiskDistribution(
                    attribute=attr,
                    predicted_value="UNKNOWN",
                    calibrated_confidence=0.0,
                    top2_margin=0.0,
                    normalized_entropy=1.0,
                    inferability_risk_score=0.0,
                    risk_band=RiskBand.LOW,
                    class_probabilities={},
                )
                for attr in AttributeName
            }
            return RiskReport(
                attributes=empty_attrs,
                overall_risk_score=0.0,
                overall_risk_band=RiskBand.LOW,
                primary_risk_attribute=None,
                secondary_risk_attributes=[],
                cues=[],
                explicit_pii_counts=explicit_pii_counts or {},
            )

        encoding = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )
        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)

        with torch.no_grad():
            out = self.model(input_ids, attention_mask)
            logits_dict = out["logits"] if isinstance(out, dict) and "logits" in out else out
            attr_distributions = self.scaler.score_logits(logits_dict)

        # Find primary and secondary risks
        sorted_by_risk = sorted(
            attr_distributions.items(),
            key=lambda item: item[1].inferability_risk_score,
            reverse=True,
        )
        primary_attr, primary_dist = sorted_by_risk[0]
        secondary_attrs = [
            attr
            for attr, dist in sorted_by_risk[1:]
            if dist.risk_band in (RiskBand.MEDIUM, RiskBand.HIGH)
        ]

        overall_score = primary_dist.inferability_risk_score
        overall_band = primary_dist.risk_band

        # Extract cues if attributor is provided and risk is elevated
        cues: List[AttributedCue] = []
        if attributor is not None:
            elevated_attrs = [
                attr
                for attr, dist in attr_distributions.items()
                if dist.risk_band in (RiskBand.MEDIUM, RiskBand.HIGH)
            ]
            if elevated_attrs:
                cues = attributor.extract_cues(text=text, target_attributes=elevated_attrs)

        return RiskReport(
            attributes=attr_distributions,
            overall_risk_score=round(overall_score, 4),
            overall_risk_band=overall_band,
            primary_risk_attribute=primary_attr,
            secondary_risk_attributes=secondary_attrs,
            cues=cues,
            explicit_pii_counts=explicit_pii_counts or {},
        )
