# src/conversation/leakage.py
"""Bayesian evidence accumulation and cumulative multi-turn privacy leakage tracking.

Prevents artificial risk dilution on benign follow-ups by updating posterior attribute
beliefs monotonically via tempered log-odds accumulation over turn inferability scores.
"""

from __future__ import annotations

from typing import Dict, Tuple
import numpy as np

from src.contracts.schemas import AttributeName, ConversationState, RiskReport
from src.risk_model.calibrate import AttributeTemperatureScaler
from src.risk_model.taxonomy import ATTRIBUTE_TAXONOMY


class BayesianConversationLeakageTracker:
    """Tracks multi-turn cumulative privacy leakage via tempered Bayesian belief updating."""

    def __init__(self, scaler: AttributeTemperatureScaler):
        """Initialize tracker with temperature scaler for posterior inferability mapping."""
        self.scaler = scaler

    def compute_cumulative_leakage(
        self,
        state: ConversationState,
        turn_risk_report: RiskReport,
        commit_update: bool = False,
    ) -> Tuple[float, float, Dict[AttributeName, float]]:
        """Compute new cumulative conversation leakage, marginal delta, and per-attribute risks.

        Returns:
            Tuple of (new_cumulative_leakage, incremental_delta, per_attribute_cumulative_scores).
        """
        prev_leakage = state.cumulative_leakage_score
        updated_posteriors: Dict[AttributeName, Dict[str, float]] = {}
        per_attr_cumulative_risk: Dict[AttributeName, float] = {}

        for attr, labels in ATTRIBUTE_TAXONOMY.items():
            k = len(labels)
            prior_dict = state.posterior_beliefs.get(
                attr, {lbl: 1.0 / k for lbl in labels}
            )
            prior_vec = np.array([prior_dict.get(lbl, 1.0 / k) for lbl in labels], dtype=np.float64)
            prior_vec = np.clip(prior_vec, 1e-9, 1.0)
            prior_vec /= np.sum(prior_vec)

            turn_dist = turn_risk_report.attributes.get(attr)
            if turn_dist is not None:
                turn_probs = np.array(
                    [turn_dist.class_probabilities.get(lbl, 1.0 / k) for lbl in labels],
                    dtype=np.float64,
                )
                turn_probs = np.clip(turn_probs, 1e-9, 1.0)
                turn_probs /= np.sum(turn_probs)
                evidence_weight = float(turn_dist.inferability_risk_score)
            else:
                turn_probs = np.full(k, 1.0 / k, dtype=np.float64)
                evidence_weight = 0.0

            # Tempered Bayesian log-odds accumulation
            log_posterior = np.log(prior_vec) + evidence_weight * np.log(turn_probs * k)
            log_posterior -= np.max(log_posterior)
            posterior_vec = np.exp(log_posterior)
            posterior_vec /= np.sum(posterior_vec)

            updated_posteriors[attr] = {
                lbl: float(posterior_vec[i]) for i, lbl in enumerate(labels)
            }
            raw_cum_inf = self.scaler._compute_raw_inferability_from_probs(posterior_vec)
            ui_cum_score, _ = self.scaler.map_to_ui_scale(attr, raw_cum_inf)
            per_attr_cumulative_risk[attr] = ui_cum_score

        # Overall conversation leakage is root-mean-square of top-2 most leaked attributes
        sorted_risks = sorted(per_attr_cumulative_risk.values(), reverse=True)
        top_two = sorted_risks[:2] if len(sorted_risks) >= 2 else sorted_risks
        computed_leakage = round(float(np.sqrt(np.mean(np.square(top_two)))), 4) if top_two else 0.0
        # Enforce strict monotonicity: previously disclosed information cannot be un-leaked
        new_leakage = max(prev_leakage, computed_leakage)
        delta = round(new_leakage - prev_leakage, 4)

        if commit_update:
            state.posterior_beliefs = updated_posteriors
            state.cumulative_leakage_score = new_leakage

        return new_leakage, delta, per_attr_cumulative_risk
