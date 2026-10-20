# src/api/__init__.py
"""API and UI integration package."""

from src.api.ui_helpers import (
    PrivacySafeTelemetryLogger,
    SafeTelemetryEvent,
    render_highlighted_html_from_offsets,
)

__all__ = [
    "PrivacySafeTelemetryLogger",
    "SafeTelemetryEvent",
    "render_highlighted_html_from_offsets",
]
