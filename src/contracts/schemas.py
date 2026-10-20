# src/contracts/schemas.py
"""Shared Pydantic v2 data contracts for cross-module validation.

Enforces typed schemas across classification, conversation state, rewriting,
evaluation, and frontend rendering to prevent brittle dictionary passing.
"""

from __future__ import annotations

from enum import Enum
from typing import Dict, List, Optional
from pydantic import BaseModel, Field, model_validator
from src.product.thresholds import PROVISIONAL_MAX_RISK_THRESHOLD


class AttributeName(str, Enum):
    """Supported demographic attribute families for inferential privacy detection."""
    AGE = "age"
    LOCATION = "location"
    OCCUPATION = "occupation"
    EDUCATION = "education"


class RiskBand(str, Enum):
    """Categorical risk bands based on validation-calibrated inferability thresholds."""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


class RewriteMode(str, Enum):
    """Execution modes for the privacy-preserving rewriter subsystem."""
    ORIGINAL = "original"
    PROMPT_BASELINE = "prompt_baseline"
    BALANCED_SFT = "balanced_sft"
    STRONG_PRIVACY_SFT = "strong_privacy_sft"


class PIIEntitySpan(BaseModel):
    """Represents an explicit PII entity detection with character offsets and pseudonym token."""
    entity_type: str = Field(..., description="Presidio entity type, e.g., PERSON, EMAIL_ADDRESS")
    start: int = Field(..., ge=0, description="Start character offset in original text")
    end: int = Field(..., gt=0, description="End character offset in original text")
    original_surface: str = Field(..., description="Surface string in memory only; never persisted to disk")
    pseudonym_token: str = Field(..., description="Deterministic session token, e.g., PERSON_1")
    confidence: float = Field(..., ge=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_offsets(self) -> "PIIEntitySpan":
        """Validate that end character offset strictly exceeds start offset."""
        if self.end <= self.start:
            raise ValueError(f"Invalid span offsets: start={self.start}, end={self.end}")
        return self


class AttributedCue(BaseModel):
    """High-salience text span attributed to an inferable attribute via Integrated Gradients."""
    span_text: str = Field(..., min_length=1)
    start: int = Field(..., ge=0, description="Start character offset in original message")
    end: int = Field(..., gt=0, description="End character offset in original message")
    attribute: AttributeName
    importance: float = Field(..., ge=0.0, le=1.0, description="Normalized attribution importance score")
    deletion_risk_drop: Optional[float] = Field(
        default=None, description="Empirical drop in calibrated risk when span is masked"
    )


class AttributeRiskDistribution(BaseModel):
    """Calibrated probability distribution and inferability risk score for a single attribute."""
    attribute: AttributeName
    predicted_value: str = Field(..., description="Top-1 predicted coarse class or UNKNOWN")
    calibrated_confidence: float = Field(..., ge=0.0, le=1.0, description="Temperature-scaled top-1 probability")
    top2_margin: float = Field(..., ge=0.0, le=1.0, description="p_(1) - p_(2)")
    normalized_entropy: float = Field(..., ge=0.0, le=1.0, description="Shannon entropy divided by log(K)")
    inferability_risk_score: float = Field(
        ..., ge=0.0, le=1.0, description="Composite calibrated risk score accounting for entropy and margin"
    )
    risk_band: RiskBand
    class_probabilities: Dict[str, float] = Field(default_factory=dict)


class RiskReport(BaseModel):
    """Comprehensive risk assessment across all demographic attributes with attributed cues."""
    attributes: Dict[AttributeName, AttributeRiskDistribution]
    overall_risk_score: float = Field(..., ge=0.0, le=1.0)
    overall_risk_band: RiskBand
    primary_risk_attribute: Optional[AttributeName] = None
    secondary_risk_attributes: List[AttributeName] = Field(default_factory=list)
    cues: List[AttributedCue] = Field(default_factory=list)
    explicit_pii_counts: Dict[str, int] = Field(default_factory=dict)


class TurnHistoryItem(BaseModel):
    """Record of a single conversation turn within the multi-turn session."""
    turn_id: int = Field(..., ge=1)
    text: str = Field(..., min_length=1)
    protected_text: Optional[str] = None
    turn_risk_scores: Dict[AttributeName, float] = Field(default_factory=dict)
    cumulative_leakage_after_turn: float = Field(default=0.0, ge=0.0, le=1.0)


class ConversationState(BaseModel):
    """Session-level conversation state maintaining pseudonym mappings and Bayesian beliefs."""
    conversation_id: str
    turn_count: int = Field(default=0, ge=0)
    entity_map: Dict[str, str] = Field(
        default_factory=dict, description="Normalized surface entity -> stable pseudonym (e.g., PERSON_1)"
    )
    reverse_entity_map: Dict[str, str] = Field(
        default_factory=dict, description="Pseudonym -> generic safe category phrase or placeholder"
    )
    posterior_beliefs: Dict[AttributeName, Dict[str, float]] = Field(
        default_factory=dict, description="Running Bayesian belief distribution over classes per attribute"
    )
    risk_history: List[TurnHistoryItem] = Field(default_factory=list)
    cumulative_leakage_score: float = Field(default=0.0, ge=0.0, le=1.0)


class UtilityReport(BaseModel):
    """Semantic utility and meaning preservation metrics computed between original and rewrite."""
    semantic_similarity: float = Field(..., ge=-1.0, le=1.0)
    forward_entailment_prob: float = Field(..., ge=0.0, le=1.0, description="P(entail | original -> rewrite)")
    backward_entailment_prob: float = Field(..., ge=0.0, le=1.0, description="P(entail | rewrite -> original)")
    bidirectional_entailment: float = Field(..., ge=0.0, le=1.0)
    contradiction_prob: float = Field(..., ge=0.0, le=1.0)
    token_edit_ratio: float = Field(..., ge=0.0, le=1.0)
    utility_preserved: bool
    warning_message: Optional[str] = None


class ProtectionResponse(BaseModel):
    """End-to-end response returned after pseudonymizing, risk-checking, and rewriting text."""
    conversation_id: str
    turn_id: int
    original_text: str
    protected_text: str
    rewrite_mode_used: RewriteMode
    risk_before: RiskReport
    risk_after: RiskReport
    utility: UtilityReport
    pseudonyms_applied: Dict[str, str]
    conversation_leakage_before: float = Field(..., ge=0.0, le=1.0)
    conversation_leakage_after: float = Field(..., ge=0.0, le=1.0)
    incremental_leakage_delta: float
    latency_ms: float


class NormalizedInput(BaseModel):
    """Sanitized input text payload before entity extraction or tokenization."""
    raw_text: str = Field(..., min_length=1)
    cleaned_text: str = Field(..., min_length=1)
    detected_language: str = Field(default="en")
    length_tokens: Optional[int] = None


class RewriteRequest(BaseModel):
    """Client request for privacy-preserving rewriting."""
    text: str = Field(..., min_length=1)
    conversation_id: Optional[str] = None
    session_id: Optional[str] = Field("default", description="Optional conversation session ID")
    mode: RewriteMode = RewriteMode.BALANCED_SFT
    preserve_pseudonyms: bool = True
    max_risk: float = Field(default=PROVISIONAL_MAX_RISK_THRESHOLD, ge=0.0, le=1.0)


class RewriteResponse(BaseModel):
    """Lightweight rewrite response containing sanitized text and operational metadata."""
    protected_text: str
    mode_used: RewriteMode
    cues_mitigated: List[str] = Field(default_factory=list)
    latency_ms: float = 0.0


class EvaluationRecord(BaseModel):
    """Structured record for offline benchmark evaluation across privacy methods."""
    sample_id: str
    original_text: str
    rewritten_text: str
    method: str
    ground_truth: Dict[AttributeName, str] = Field(default_factory=dict)
    adversary_predictions: Dict[AttributeName, str] = Field(default_factory=dict)
    utility: Optional[UtilityReport] = None
    latency_ms: float = 0.0
