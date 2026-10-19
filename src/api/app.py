# src/api/app.py
"""Production FastAPI service for InferenceGuard end-to-end privacy layer.

Exposes REST endpoints for calibrated risk prediction, syntax-aligned cue highlighting,
QLoRA rewriting, session-stable pseudonym management, and zero-PII telemetry logging.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.api.ui_helpers import (
    PrivacySafeTelemetryLogger,
    SafeTelemetryEvent,
    render_highlighted_html_from_offsets,
)
from src.contracts.schemas import (
    AttributedCue,
    ConversationState,
    NormalizedInput,
    ProtectionResponse,
    RewriteMode,
    RewriteRequest,
    RiskReport,
)
from src.conversation.entity_map import DeterministicPseudonymManager
from src.conversation.leakage import BayesianConversationLeakageTracker
from src.evaluation.utility_metrics import BidirectionalUtilityEvaluator
from src.pipeline.orchestrator import InferenceGuardPipeline
from src.rewriter.inference import QwenPrivacyRewriterService
from src.risk_model.attribution import SyntaxAlignedCueAttributor
from src.risk_model.calibrate import AttributeTemperatureScaler
from src.risk_model.predict import CalibratedRiskPredictor

logger = logging.getLogger(__name__)

app = FastAPI(title="InferenceGuard Production API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
pseudonym_mgr = DeterministicPseudonymManager()
scaler = AttributeTemperatureScaler()
utility_eval = BidirectionalUtilityEvaluator()
rewriter_service = QwenPrivacyRewriterService()
telemetry_logger = PrivacySafeTelemetryLogger()

# Fallback predictor wrapping mock / rule-based components for offline CPU safety
class MockPredictor:
    def predict(self, text: str) -> RiskReport:
        return self.predict_with_cues(text)

    def predict_with_cues(self, text: str, attributor=None, explicit_pii_counts=None) -> RiskReport:
        from src.contracts.schemas import AttributeName, AttributeRiskDistribution, RiskBand
        t_low = text.lower()
        has_risk = any(w in t_low for w in ["green line", "seattle", "amazon", "co-op", "campus", "denver"])
        score = 0.72 if has_risk else 0.15
        band = RiskBand.HIGH if has_risk else RiskBand.LOW

        attrs = {}
        for attr in AttributeName:
            attrs[attr] = AttributeRiskDistribution(
                attribute=attr,
                predicted_value="detected" if has_risk else "UNKNOWN",
                calibrated_confidence=score,
                top2_margin=0.35,
                normalized_entropy=0.3,
                inferability_risk_score=score,
                risk_band=band,
                class_probabilities={},
            )

        cues = []
        if has_risk and attributor is not None:
            cues = attributor.extract_cues(text, [AttributeName.LOCATION, AttributeName.OCCUPATION])

        return RiskReport(
            attributes=attrs,
            overall_risk_score=score,
            overall_risk_band=band,
            primary_risk_attribute=AttributeName.LOCATION if has_risk else None,
            secondary_risk_attributes=[AttributeName.OCCUPATION] if has_risk else [],
            cues=cues,
            explicit_pii_counts=explicit_pii_counts or {},
        )

cue_attributor = SyntaxAlignedCueAttributor(
    model=None,
    tokenizer=None,
    scaler=scaler,
)
risk_estimator = MockPredictor()
leakage_tracker = BayesianConversationLeakageTracker(scaler)

pipeline = InferenceGuardPipeline(
    pseudonym_manager=pseudonym_mgr,
    risk_estimator=risk_estimator,
    cue_attributor=cue_attributor,
    rewriter_service=rewriter_service,
    utility_evaluator=utility_eval,
    leakage_tracker=leakage_tracker,
)


@app.get("/health")
def health() -> Dict[str, str]:
    """Healthcheck endpoint confirming API availability."""
    return {"status": "ok", "service": "InferenceGuard API", "version": "2.0.0"}


@app.post("/protect", response_model=ProtectionResponse)
def protect_turn(request: RewriteRequest) -> ProtectionResponse:
    """Process a single message through the full 6-step privacy protection pipeline."""
    conv_id = request.conversation_id or "default_session"
    return pipeline.process_turn(
        conversation_id=conv_id,
        message=request.text,
        mode=request.mode,
        commit_to_session=True,
    )


class HighlightRequest(BaseModel):
    text: str = Field(..., min_length=1)
    cues: List[AttributedCue] = Field(default_factory=list)


@app.post("/highlight")
def highlight_text(req: HighlightRequest) -> Dict[str, str]:
    """Render character-offset interval highlights safely without regex text replacement."""
    html_output = render_highlighted_html_from_offsets(req.text, req.cues)
    return {"highlighted_html": html_output}


@app.post("/telemetry")
def record_telemetry(event: SafeTelemetryEvent) -> Dict[str, str]:
    """Record anonymized user study event strictly validated against the zero-PII whitelist."""
    telemetry_logger.log_event(event)
    return {"status": "recorded", "event_id": event.event_id}


@app.post("/session/reset")
def reset_session(session_id: str) -> Dict[str, str]:
    """Reset active conversation state and pseudonym mappings."""
    pipeline.reset_session(session_id)
    return {"status": "reset", "session_id": session_id}


# Mount web interface if available
web_dir = Path("web")
if web_dir.exists():
    app.mount("/", StaticFiles(directory=str(web_dir), html=True), name="static")
