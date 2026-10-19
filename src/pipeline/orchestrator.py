# src/pipeline/orchestrator.py
"""Master end-to-end pipeline orchestrator for InferenceGuard.

Unites Presidio explicit PII pseudonymization, calibrated ModernBERT risk modeling,
syntax-aligned Integrated Gradients cue attribution, QLoRA Qwen rewriting,
bidirectional DeBERTa-v3 NLI utility validation, and Bayesian conversation leakage tracking.
"""

from __future__ import annotations

import time
from typing import Any, Dict, Optional

from src.contracts.schemas import (
    ConversationState,
    ProtectionResponse,
    RewriteMode,
    RiskBand,
    RiskReport,
    TurnHistoryItem,
)
from src.conversation.entity_map import DeterministicPseudonymManager
from src.conversation.leakage import BayesianConversationLeakageTracker
from src.evaluation.utility_metrics import BidirectionalUtilityEvaluator
from src.rewriter.inference import QwenPrivacyRewriterService


class InferenceGuardPipeline:
    """Master production orchestrator linking all five privacy engineering subsystems."""

    def __init__(
        self,
        pseudonym_manager: DeterministicPseudonymManager,
        risk_estimator: Any,
        cue_attributor: Any,
        rewriter_service: QwenPrivacyRewriterService,
        utility_evaluator: BidirectionalUtilityEvaluator,
        leakage_tracker: BayesianConversationLeakageTracker,
    ):
        """Initialize the unified protection pipeline with all subsystem components."""
        self.pseudonym_manager = pseudonym_manager
        self.risk_estimator = risk_estimator
        self.cue_attributor = cue_attributor
        self.rewriter_service = rewriter_service
        self.utility_evaluator = utility_evaluator
        self.leakage_tracker = leakage_tracker
        self.sessions: Dict[str, ConversationState] = {}

    def get_or_create_session(self, conversation_id: str) -> ConversationState:
        """Retrieve existing session state or initialize a clean conversation record."""
        if conversation_id not in self.sessions:
            self.sessions[conversation_id] = ConversationState(conversation_id=conversation_id)
        return self.sessions[conversation_id]

    def reset_session(self, conversation_id: str) -> None:
        """Clear active session state and associated pseudonym mappings."""
        self.sessions.pop(conversation_id, None)

    def process_turn(
        self,
        conversation_id: str,
        message: str,
        mode: RewriteMode = RewriteMode.BALANCED_SFT,
        commit_to_session: bool = True,
    ) -> ProtectionResponse:
        """Process a single conversation turn through the full 6-step protection pipeline."""
        t0 = time.perf_counter()
        state = self.get_or_create_session(conversation_id)
        next_turn_id = state.turn_count + 1

        # Step 1: Deterministic Explicit PII Pseudonymization
        pseudo_text, pii_spans, pii_counts = self.pseudonym_manager.pseudonymize_turn(message, state)

        # Step 2: Pre-Rewrite Calibrated Attribute Risk & Cue Attribution
        risk_before: RiskReport = self.risk_estimator.predict_with_cues(
            text=pseudo_text,
            attributor=self.cue_attributor,
            explicit_pii_counts=pii_counts,
        )

        # Step 3: Compute hypothetical conversation leakage BEFORE protection
        leakage_before, _, _ = self.leakage_tracker.compute_cumulative_leakage(
            state=state,
            turn_risk_report=risk_before,
            commit_update=False,
        )

        # Step 4: Cue-Conditioned Privacy Rewriting
        protected_text, _ = self.rewriter_service.protect(
            pseudonymized_text=pseudo_text,
            risk_report=risk_before,
            conversation_state=state,
            mode=mode,
        )

        # Step 5: Post-Rewrite Risk Rescoring & Bidirectional Utility Check
        risk_after: RiskReport = self.risk_estimator.predict_with_cues(
            text=protected_text,
            attributor=self.cue_attributor,
            explicit_pii_counts={k: 0 for k in pii_counts},
        )
        utility_report = self.utility_evaluator.evaluate_pair(pseudo_text, protected_text)

        # Step 6: Update Conversation State with Protected Turn
        leakage_after, delta_after, per_attr_cum = self.leakage_tracker.compute_cumulative_leakage(
            state=state,
            turn_risk_report=risk_after,
            commit_update=commit_to_session,
        )

        if commit_to_session:
            state.turn_count = next_turn_id
            state.risk_history.append(
                TurnHistoryItem(
                    turn_id=next_turn_id,
                    text="[REDACTED_IN_MEMORY_ONLY]",
                    protected_text=protected_text,
                    turn_risk_scores={
                        attr: dist.inferability_risk_score
                        for attr, dist in risk_after.attributes.items()
                    },
                    cumulative_leakage_after_turn=leakage_after,
                )
            )

        total_latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)

        return ProtectionResponse(
            conversation_id=conversation_id,
            turn_id=next_turn_id,
            original_text=message,
            protected_text=protected_text,
            rewrite_mode_used=mode,
            risk_before=risk_before,
            risk_after=risk_after,
            utility=utility_report,
            pseudonyms_applied={s.original_surface: s.pseudonym_token for s in pii_spans},
            conversation_leakage_before=leakage_before,
            conversation_leakage_after=leakage_after,
            incremental_leakage_delta=delta_after,
            latency_ms=total_latency_ms,
        )
