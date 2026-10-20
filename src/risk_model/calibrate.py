# src/risk_model/calibrate.py
"""Post-hoc temperature scaling and validation-derived risk threshold calibration.

Optimizes per-attribute scalar temperatures via L-BFGS to minimize Expected Calibration
Error (ECE) and Negative Log-Likelihood (NLL), and derives Youden's J ROC risk bands.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.contracts.schemas import AttributeName, AttributeRiskDistribution, RiskBand
from src.risk_model.taxonomy import ATTRIBUTE_TAXONOMY, ID_TO_LABEL


def compute_expected_calibration_error(
    probs: np.ndarray, labels: np.ndarray, n_bins: int = 15
) -> float:
    """Compute Expected Calibration Error (ECE) over valid non-ignored labels."""
    valid = labels != -100
    if not np.any(valid):
        return 0.0
    probs = probs[valid]
    labels = labels[valid]

    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = (predictions == labels).astype(np.float64)

    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n_samples = len(labels)

    for i in range(n_bins):
        in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        prop_in_bin = np.mean(in_bin)
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(accuracies[in_bin])
            avg_confidence_in_bin = np.mean(confidences[in_bin])
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * (np.sum(in_bin) / n_samples)
    return float(ece)


class AttributeTemperatureScaler(nn.Module):
    """Learns per-attribute temperature T_a via L-BFGS and calibrates inferability risk bands."""

    def __init__(self):
        super().__init__()
        self.log_temps = nn.ParameterDict({
            attr.value: nn.Parameter(torch.zeros(1, dtype=torch.float32))
            for attr in AttributeName
        })
        self.thresholds: Dict[str, Dict[str, float]] = {
            attr.value: {"medium": 0.30, "high": 0.60} for attr in AttributeName
        }

    def get_temperature(self, attr: AttributeName) -> torch.Tensor:
        """Return strictly positive clamped temperature for the given attribute."""
        return torch.exp(self.log_temps[attr.value]).clamp(min=0.05, max=10.0)

    def fit_validation_logits(
        self,
        val_logits: Dict[str, torch.Tensor],
        val_labels: Dict[str, torch.Tensor],
    ) -> Dict[str, Dict[str, float]]:
        """Fit temperatures and derive ROC risk thresholds on validation logits via L-BFGS."""
        diagnostics: Dict[str, Dict[str, float]] = {}

        for attr in AttributeName:
            key = attr.value
            if key not in val_logits or key not in val_labels:
                continue

            logits = val_logits[key].detach().float()
            labels = val_labels[key].detach().long()
            valid_mask = labels != -100
            logits_valid = logits[valid_mask]
            labels_valid = labels[valid_mask]

            if len(labels_valid) == 0:
                continue

            # Pre-calibration diagnostics
            raw_probs = F.softmax(logits_valid, dim=-1).cpu().numpy()
            labels_np = labels_valid.cpu().numpy()
            ece_before = compute_expected_calibration_error(raw_probs, labels_np)
            nll_before = float(F.cross_entropy(logits_valid, labels_valid).item())

            # Optimize log_temp with L-BFGS
            log_temp_param = self.log_temps[key]
            optimizer = torch.optim.LBFGS([log_temp_param], lr=0.05, max_iter=100)

            def eval_closure():
                optimizer.zero_grad()
                temp = torch.exp(log_temp_param).clamp(min=0.05, max=10.0)
                loss = F.cross_entropy(logits_valid / temp, labels_valid)
                loss.backward()
                return loss

            optimizer.step(eval_closure)

            # Post-calibration diagnostics
            learned_temp = float(self.get_temperature(attr).item())
            cal_probs = F.softmax(logits_valid / learned_temp, dim=-1).cpu().numpy()
            ece_after = compute_expected_calibration_error(cal_probs, labels_np)
            nll_after = float(F.cross_entropy(logits_valid / learned_temp, labels_valid).item())

            # Derive Youden's J thresholds
            raw_inferability = np.array([
                self._compute_raw_inferability_from_probs(p) for p in cal_probs
            ])
            correct_indicators = (np.argmax(cal_probs, axis=-1) == labels_np).astype(np.int32)
            med_thresh, high_thresh = self._derive_roc_thresholds(raw_inferability, correct_indicators)
            self.thresholds[key] = {"medium": med_thresh, "high": high_thresh}

            diagnostics[key] = {
                "temperature": round(learned_temp, 4),
                "ece_before": round(ece_before, 4),
                "ece_after": round(ece_after, 4),
                "nll_before": round(nll_before, 4),
                "nll_after": round(nll_after, 4),
                "threshold_medium_raw": round(med_thresh, 4),
                "threshold_high_raw": round(high_thresh, 4),
            }
        return diagnostics

    @staticmethod
    def _compute_raw_inferability_from_probs(probs: np.ndarray, alpha: float = 0.5) -> float:
        """Compute composite inferability accounting for excess confidence, entropy, and margin."""
        k = len(probs)
        sorted_p = np.sort(probs)[::-1]
        p1 = float(sorted_p[0])
        p2 = float(sorted_p[1]) if k > 1 else 0.0
        chance = 1.0 / k
        c_norm = max(0.0, (p1 - chance) / (1.0 - chance)) if chance < 1.0 else 0.0
        eps = 1e-12
        entropy = -np.sum(probs * np.log(np.clip(probs, eps, 1.0))) / math.log(k) if k > 1 else 0.0
        entropy_norm = float(np.clip(entropy, 0.0, 1.0))
        margin = p1 - p2
        score = c_norm * (alpha * (1.0 - entropy_norm) + (1.0 - alpha) * margin)
        return float(np.clip(score, 0.0, 1.0))

    @staticmethod
    def _derive_roc_thresholds(scores: np.ndarray, correct: np.ndarray) -> Tuple[float, float]:
        """Derive optimal operational thresholds via Youden's J statistic and 85% sensitivity."""
        candidate_grid = np.linspace(0.05, 0.95, 91)
        pos = correct == 1
        neg = correct == 0
        if not np.any(pos) or not np.any(neg):
            return 0.25, 0.55

        best_j = -1.0
        high_thresh = 0.55
        med_thresh = 0.25
        min_sens_diff = 1.0

        for th in candidate_grid:
            tpr = float(np.mean(scores[pos] >= th))
            fpr = float(np.mean(scores[neg] >= th))
            j_stat = tpr - fpr
            if j_stat > best_j:
                best_j = j_stat
                high_thresh = float(th)
            if abs(tpr - 0.85) < min_sens_diff:
                min_sens_diff = abs(tpr - 0.85)
                med_thresh = float(th)

        if med_thresh >= high_thresh:
            med_thresh = max(0.02, round(high_thresh * 0.6, 4))
        if med_thresh >= high_thresh:
            high_thresh = max(0.20, high_thresh)
            med_thresh = round(high_thresh * 0.5, 4)

        return med_thresh, high_thresh

    def map_to_ui_scale(self, attr: AttributeName, raw_inferability: float) -> Tuple[float, RiskBand]:
        """Map raw inferability monotonically to UI scale [0, 1] aligned with [0.30, 0.60] bands."""
        th = self.thresholds.get(attr.value, {"medium": 0.30, "high": 0.60})
        m = th["medium"]
        h = th["high"]

        if raw_inferability < m:
            ui_score = 0.30 * (raw_inferability / max(m, 1e-6))
            band = RiskBand.LOW
        elif raw_inferability < h:
            ui_score = 0.30 + 0.30 * ((raw_inferability - m) / max(h - m, 1e-6))
            band = RiskBand.MEDIUM
        else:
            ui_score = 0.60 + 0.40 * ((raw_inferability - h) / max(1.0 - h, 1e-6))
            band = RiskBand.HIGH

        ui_score = float(np.clip(ui_score, 0.0, 1.0))
        return round(ui_score, 4), band

    def score_logits(self, logits_dict: Dict[str, torch.Tensor]) -> Dict[AttributeName, AttributeRiskDistribution]:
        """Convert multi-head model logits into calibrated probability distributions and risk scores."""
        results: Dict[AttributeName, AttributeRiskDistribution] = {}
        for attr in AttributeName:
            key = attr.value
            temp = float(self.get_temperature(attr).item())
            logits = logits_dict[key].detach().float()
            if logits.ndim > 1:
                logits = logits.squeeze(0)
            probs = F.softmax(logits / temp, dim=-1).cpu().numpy()

            k = len(probs)
            sorted_indices = np.argsort(probs)[::-1]
            top1_idx = int(sorted_indices[0])
            top2_idx = int(sorted_indices[1]) if k > 1 else top1_idx

            p1 = float(probs[top1_idx])
            p2 = float(probs[top2_idx])
            margin = round(p1 - p2, 4)
            eps = 1e-12
            entropy_norm = round(
                float(np.clip(-np.sum(probs * np.log(np.clip(probs, eps, 1.0))) / math.log(k), 0.0, 1.0)),
                4,
            )

            raw_inf = self._compute_raw_inferability_from_probs(probs)
            ui_score, band = self.map_to_ui_scale(attr, raw_inf)

            pred_label = ID_TO_LABEL[attr][top1_idx] if band != RiskBand.LOW else "UNKNOWN"
            class_prob_map = {
                ID_TO_LABEL[attr][i]: round(float(probs[i]), 4) for i in range(k)
            }

            results[attr] = AttributeRiskDistribution(
                attribute=attr,
                predicted_value=pred_label,
                calibrated_confidence=round(p1, 4),
                top2_margin=margin,
                normalized_entropy=entropy_norm,
                inferability_risk_score=ui_score,
                risk_band=band,
                class_probabilities=class_prob_map,
            )
        return results

    def save(self, path: Path) -> None:
        """Persist calibrated temperatures and Youden's J thresholds to JSON."""
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "temperatures": {attr.value: float(self.get_temperature(attr).item()) for attr in AttributeName},
            "thresholds": self.thresholds,
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "AttributeTemperatureScaler":
        """Load calibrated temperatures and thresholds from JSON file."""
        scaler = cls()
        payload = json.loads(path.read_text(encoding="utf-8"))
        with torch.no_grad():
            for attr_key, temp_val in payload["temperatures"].items():
                scaler.log_temps[attr_key].copy_(torch.tensor([math.log(temp_val)], dtype=torch.float32))
        scaler.thresholds = payload["thresholds"]
        return scaler


def main() -> None:
    """CLI tool for evaluating or fitting calibration on validation logits."""
    parser = argparse.ArgumentParser(description="Calibrate multi-head risk model.")
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--save_path", type=Path, default=Path("artifacts/calibration_params.json"))
    args = parser.parse_args()
    print(f"Calibration module initialized for split={args.split}. Target save path: {args.save_path}")


if __name__ == "__main__":
    main()
