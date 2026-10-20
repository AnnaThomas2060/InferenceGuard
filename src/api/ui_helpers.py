# src/api/ui_helpers.py
"""UI rendering helpers and zero-PII user study telemetry logging.

Provides character-offset interval tree highlight rendering free of regex corruption
and enforces strict whitelist schemas on telemetry logs to prevent accidental PII leakage.
"""

from __future__ import annotations

import html
import json
import time
import uuid
from pathlib import Path
from typing import List, Optional
from pydantic import BaseModel, Field

from src.contracts.schemas import AttributedCue


def render_highlighted_html_from_offsets(text: str, cues: List[AttributedCue]) -> str:
    """Render non-overlapping HTML highlights using exact character offsets.

    Greedily prioritizes higher importance scores to resolve overlapping spans
    without regex text substitution or duplicated HTML tag corruption.
    """
    if not text:
        return ""
    if not cues:
        return html.escape(text)

    # Sort by importance descending to accept highest-salience spans first
    sorted_by_imp = sorted(cues, key=lambda c: (c.importance, c.end - c.start), reverse=True)
    accepted: List[AttributedCue] = []
    occupied = [False] * (len(text) + 1)

    for cue in sorted_by_imp:
        s, e = max(0, cue.start), min(len(text), cue.end)
        if s >= e or any(occupied[s:e]):
            continue
        for idx in range(s, e):
            occupied[idx] = True
        accepted.append(cue)

    # Sort accepted intervals left-to-right
    accepted.sort(key=lambda c: c.start)

    fragments: List[str] = []
    cursor = 0
    for cue in accepted:
        if cue.start > cursor:
            fragments.append(html.escape(text[cursor : cue.start]))
        escaped_span = html.escape(text[cue.start : cue.end])
        attr_val = getattr(cue.attribute, "value", str(cue.attribute))
        fragments.append(
            f'<mark class="ig-cue ig-attr-{attr_val}" '
            f'data-attribute="{attr_val}" '
            f'data-importance="{cue.importance:.2f}">'
            f'{escaped_span}<span class="ig-cue-badge">{attr_val}</span></mark>'
        )
        cursor = cue.end

    if cursor < len(text):
        fragments.append(html.escape(text[cursor:]))

    return "".join(fragments)


class SafeTelemetryEvent(BaseModel):
    """Zero-PII event schema strictly whitelisting operational telemetry attributes."""
    event_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp_utc: float = Field(default_factory=time.time)
    session_id: str
    scenario_id: str
    event_type: str = Field(
        ...,
        pattern="^(risk_warning_viewed|rewrite_requested|rewrite_accepted|rewrite_edited|rewrite_rejected)$",
    )
    rewrite_mode: str
    overall_risk_before: float
    overall_risk_after: float
    bidirectional_entailment: float
    utility_warning_shown: bool
    latency_ms: float


class PrivacySafeTelemetryLogger:
    """Enforces zero-PII logging by validating events against a strict whitelist schema."""

    def __init__(self, log_path: Path = Path("user_studies/session07/telemetry_events.jsonl")):
        """Initialize telemetry logger and ensure log directory exists."""
        self.log_path = Path(log_path)

    def log_event(self, event: SafeTelemetryEvent) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        """Serialize and append telemetry event JSONL to disk."""
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(event.model_dump_json() + "\n")
