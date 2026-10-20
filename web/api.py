"""Unified API facade for InferenceGuard web interface and external integrations.

Re-exports the consolidated FastAPI application, contract schemas, and tunneling
helpers from src.api.app, maintaining 100% backward compatibility.
"""

from __future__ import annotations

from src.api.app import (
    AnalyzeRequest,
    AnalyzeResponse,
    HighlightRequest,
    RewriteRequest,
    RewriteResponse,
    app,
    extract_cues,
    extract_pii_spans,
    launch_tunnel,
    to_canonical_entity_type,
)
from src.rewriter.generate_training_data import (
    estimate_privacy_risk,
    generate_candidate_rewrites_heuristic,
    pareto_rejection_sample,
)

__all__ = [
    "app",
    "launch_tunnel",
    "AnalyzeRequest",
    "AnalyzeResponse",
    "RewriteRequest",
    "RewriteResponse",
    "HighlightRequest",
    "extract_pii_spans",
    "extract_cues",
    "to_canonical_entity_type",
    "pareto_rejection_sample",
    "generate_candidate_rewrites_heuristic",
    "estimate_privacy_risk",
]
