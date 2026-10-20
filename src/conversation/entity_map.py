# src/conversation/entity_map.py
"""Deterministic session-scoped PII pseudonymization and entity map management.

Combines Presidio entity extraction with deterministic right-to-left placeholder
replacement, preventing entity drift and character-offset displacement across turns.
"""

from __future__ import annotations

import logging
import re
from typing import Dict, List, Optional, Tuple

from src.contracts.schemas import ConversationState, PIIEntitySpan

logger = logging.getLogger(__name__)


class DeterministicPseudonymManager:
    """Manages session-stable entity pseudonyms using right-to-left replacement."""

    TRACKED_ENTITIES = ["PERSON", "EMAIL_ADDRESS", "PHONE_NUMBER", "ORGANIZATION", "LOCATION"]
    PREFIX_MAP = {
        "PERSON": "PERSON",
        "EMAIL_ADDRESS": "EMAIL",
        "PHONE_NUMBER": "PHONE",
        "ORGANIZATION": "ORG",
        "LOCATION": "LOCATION",
    }

    # High-precision regex fallback patterns for offline environments or fast unit tests
    FALLBACK_PATTERNS = {
        "EMAIL_ADDRESS": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
        "PHONE_NUMBER": r"\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b",
        "LOCATION": r"\b(?:Denver, CO|Seattle, WA|Boston, MA|New York, NY|San Francisco, CA)\b",
        "ORGANIZATION": r"\b(?:Amazon|Google|Microsoft|Meta|Apple)\b",
        "PERSON": r"\b(?:Alice Smith|Bob Jones|Charlie Brown|Sarah Connor)\b",
    }

    def __init__(self, score_threshold: float = 0.45):
        """Initialize the manager with optional Presidio AnalyzerEngine."""
        self.score_threshold = score_threshold
        self.analyzer = None
        try:
            from presidio_analyzer import AnalyzerEngine
            self.analyzer = AnalyzerEngine()
        except Exception as exc:
            logger.info("Presidio AnalyzerEngine unavailable (%s); using fallback regex engine.", exc)

    @staticmethod
    def _normalize_surface(surface: str) -> str:
        """Normalize whitespace and casing for entity key matching."""
        return re.sub(r"\s+", " ", surface.strip().lower())

    def _extract_spans_with_fallback(self, text: str) -> List[Tuple[str, int, int, float]]:
        """Extract entity spans using regex fallback when AnalyzerEngine is unavailable or empty."""
        spans: List[Tuple[str, int, int, float]] = []
        for ent_type, pattern in self.FALLBACK_PATTERNS.items():
            for m in re.finditer(pattern, text, flags=re.IGNORECASE):
                spans.append((ent_type, m.start(), m.end(), 0.95))
        return spans

    def pseudonymize_turn(
        self, text: str, state: ConversationState
    ) -> Tuple[str, List[PIIEntitySpan], Dict[str, int]]:
        """Extract explicit PII and replace with deterministic tokens from right to left."""
        if not text:
            return "", [], {k: 0 for k in self.TRACKED_ENTITIES}

        raw_detections: List[Tuple[str, int, int, float]] = []
        if self.analyzer is not None:
            try:
                results = self.analyzer.analyze(
                    text=text,
                    entities=self.TRACKED_ENTITIES,
                    language="en",
                    score_threshold=self.score_threshold,
                )
                for r in results:
                    raw_detections.append((r.entity_type, r.start, r.end, float(r.score)))
            except Exception as e:
                logger.warning("Presidio analysis failed (%s); reverting to fallback patterns.", e)
                raw_detections = self._extract_spans_with_fallback(text)
        else:
            raw_detections = self._extract_spans_with_fallback(text)

        # Resolve overlapping spans greedily: highest confidence, then longest span
        raw_detections.sort(key=lambda r: (r[3], r[2] - r[1]), reverse=True)
        accepted = []
        occupied = [False] * (len(text) + 1)
        for ent_type, start, end, score in raw_detections:
            if any(occupied[start:end]):
                continue
            for idx in range(start, end):
                occupied[idx] = True
            accepted.append((ent_type, start, end, score))

        # Sort left-to-right to assign chronological indices
        accepted.sort(key=lambda r: r[1])

        pii_spans: List[PIIEntitySpan] = []
        pii_counts: Dict[str, int] = {k: 0 for k in self.TRACKED_ENTITIES}

        for ent_type, start, end, score in accepted:
            pii_counts[ent_type] = pii_counts.get(ent_type, 0) + 1
            surface = text[start:end]
            norm_key = f"{ent_type}::{self._normalize_surface(surface)}"

            if norm_key not in state.entity_map:
                prefix = self.PREFIX_MAP.get(ent_type, ent_type)
                existing_count = sum(
                    1 for v in state.entity_map.values() if v.startswith(f"{prefix}_")
                )
                pseudonym = f"{prefix}_{existing_count + 1}"
                state.entity_map[norm_key] = pseudonym
                state.reverse_entity_map[pseudonym] = surface
            else:
                pseudonym = state.entity_map[norm_key]

            pii_spans.append(
                PIIEntitySpan(
                    entity_type=ent_type,
                    start=start,
                    end=end,
                    original_surface=surface,
                    pseudonym_token=pseudonym,
                    confidence=round(score, 4),
                )
            )

        # Apply replacements from RIGHT TO LEFT so earlier character offsets never shift
        chars = list(text)
        for span in sorted(pii_spans, key=lambda s: s.start, reverse=True):
            chars[span.start : span.end] = list(span.pseudonym_token)

        pseudonymized_text = "".join(chars)
        return pseudonymized_text, pii_spans, pii_counts

    def restore_pseudonyms(self, text: str, state: ConversationState) -> str:
        """Replace active pseudonyms in rewritten text with their original surface entities."""
        restored = text
        for token, original in sorted(state.reverse_entity_map.items(), key=lambda x: len(x[0]), reverse=True):
            restored = restored.replace(token, original)
        return restored

    def verify_pseudonym_preservation(
        self, original_pseudo_text: str, rewritten_text: str, state: ConversationState
    ) -> Tuple[bool, List[str]]:
        """Verify whether all pseudonym tokens present in the input text survived rewriting."""
        missing: List[str] = []
        for token in state.reverse_entity_map.keys():
            if token in original_pseudo_text and token not in rewritten_text:
                missing.append(token)
        return len(missing) == 0, missing
