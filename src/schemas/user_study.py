"""Pydantic schema definitions and write-time ingestion validation for user study records.

Enforces schema contracts at write time, preventing malformed types, non-conforming
structures, and corrupt JSON array serialization.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field, field_validator, model_validator


SESSION06_SYSTEM_STATES = {
    "risk_positive_rewrite",
    "risk_positive_utility_conflict",
    "risk_negative_passthrough",
}

SESSION06_COMPREHENSION = {
    "correct",
    "partial",
    "incorrect",
    "echo",
}

DEFAULT_PARTICIPANT_RE = re.compile(r"^P\d{2,}$")


class Session06RecordSchema(BaseModel):
    """Pydantic validation contract for Goal-Driven User Study session records."""

    participant_id: str = Field(..., description="Participant ID matching ^P\\d{2,}$")
    scenario_id: str = Field(..., description="Scenario ID (e.g. S-01, S-02)")
    scenario_mode: Optional[str] = Field(None, description="generative, controlled, staged_multiturn")
    session_id: Optional[str] = Field(None, description="Session identifier")
    experiment_id: str = Field("session06_user_eval_v1", description="Experiment configuration ID")
    session: int = Field(6, description="Session integer number")
    date: str = Field(..., description="Date of user study (YYYY-MM-DD)")
    facilitator: Optional[str] = Field(None, description="Facilitator name")
    artifact: Optional[str] = Field(None, description="Path to study session artifact")

    anchors_available: Optional[List[str]] = Field(None, description="Available anchors in scenario")
    anchors_used: Optional[List[str]] = Field(None, description="Anchors disclosed by participant")
    initial_query: Optional[str] = Field(None, description="Raw participant query text")
    query_formulation_seconds: Optional[float] = Field(None, ge=0, description="Time to formulate query")
    evaluator_intervention_pre_submit: Optional[bool] = Field(False, description="Evaluator intervention flag")

    detected_risk: Optional[bool] = Field(None, description="Whether risk was detected")
    risk_score: Optional[float] = Field(None, ge=0.0, le=1.0, description="Normalized risk score in [0, 1]")
    system_state: str = Field(..., description="Observed system state")

    warning_in_their_words: Optional[str] = Field(None, description="Participant interpretation of warning")
    warning_comprehension: Optional[str] = Field(None, description="Comprehension rating")

    requested_rewrite: Optional[bool] = Field(None, description="Whether rewrite was requested")
    rewrite_changed_text: Optional[bool] = Field(None, description="Whether rewrite changed text")
    rewrite_decision: Optional[str] = Field(None, description="accept_as_is, accept_with_edits, reject, not_applicable")
    edit_made: Optional[bool] = Field(None, description="Whether manual edit was made")
    edited_query: Optional[str] = Field(None, description="Edited query text")

    usefulness: Optional[Union[int, float]] = Field(None, ge=0, le=10, description="Self-reported usefulness (1-10)")
    would_use_before_sending: Optional[bool] = Field(None, description="Adoption intent")
    adoption_reason: Optional[str] = Field(None, description="Qualitative reason for adoption decision")
    time_to_decision_seconds: Optional[float] = Field(None, ge=0, description="Time to reach decision in seconds")
    comments: Optional[str] = Field(None, description="Observer notes and participant comments")
    protocol_deviation: Optional[bool] = Field(False, description="Protocol deviation flag")

    @field_validator("participant_id")
    @classmethod
    def validate_participant_id(cls, v: str) -> str:
        if not DEFAULT_PARTICIPANT_RE.match(v):
            raise ValueError(f"participant_id '{v}' does not match pattern ^P\\d{{2,}}$")
        return v

    @field_validator("system_state")
    @classmethod
    def validate_system_state(cls, v: str) -> str:
        if v not in SESSION06_SYSTEM_STATES:
            raise ValueError(f"Invalid system_state '{v}'. Must be one of {sorted(SESSION06_SYSTEM_STATES)}")
        return v

    @field_validator("warning_comprehension")
    @classmethod
    def validate_warning_comprehension(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and v not in SESSION06_COMPREHENSION:
            raise ValueError(f"Invalid warning_comprehension '{v}'. Must be one of {sorted(SESSION06_COMPREHENSION)}")
        return v

    @model_validator(mode="after")
    def validate_cross_field_relationships(self) -> Session06RecordSchema:
        # 1. Anchors subset check
        if self.anchors_available is not None and self.anchors_used is not None:
            avail_set = set(self.anchors_available)
            unknown = set(self.anchors_used) - avail_set
            if unknown:
                raise ValueError(f"anchors_used contains anchors not present in anchors_available: {sorted(unknown)}")

        # 2. Passthrough state constraints
        if self.system_state == "risk_negative_passthrough":
            if self.detected_risk is True:
                raise ValueError("risk_negative_passthrough cannot have detected_risk=True")
            if self.rewrite_changed_text is True:
                raise ValueError("risk_negative_passthrough cannot have rewrite_changed_text=True")
            if self.rewrite_decision is not None and self.rewrite_decision != "not_applicable":
                raise ValueError("risk_negative_passthrough must use rewrite_decision='not_applicable'")

        return self


def _split_anchors(raw: Any) -> Optional[List[str]]:
    """Cleanly split anchors separated by semicolons or commas."""
    if not raw:
        return None
    if isinstance(raw, list):
        if len(raw) == 1 and isinstance(raw[0], str):
            cleaned = raw[0].strip("() \t\n")
            parts = re.split(r"[;,]", cleaned)
            return [p.strip() for p in parts if p.strip()]
        return [str(x).strip("() \t\n") for x in raw]
    if isinstance(raw, str):
        cleaned = raw.strip("() \t\n")
        parts = re.split(r"[;,]", cleaned)
        return [p.strip() for p in parts if p.strip()]
    return None


def _normalize_legacy_record(data: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize legacy nested user study records into flat schema representation."""
    norm = dict(data)
    tc = norm.pop("task_context", None) or {}
    po = norm.pop("participant_observation", None) or {}
    rc = norm.pop("researcher_coding", None) or {}
    proto = norm.pop("protocol", None) or {}

    norm.setdefault("scenario_mode", norm.get("scenario_mode", "controlled"))
    norm.setdefault("experiment_id", "session06_user_eval_v1")
    norm.setdefault("session", 6)

    date_val = str(norm.get("date", ""))
    if "/" in date_val:
        parts = date_val.split("/")
        if len(parts) == 3:
            norm["date"] = f"{parts[2]}-{parts[0].zfill(2)}-{parts[1].zfill(2)}"

    if "anchors_available" in tc:
        norm["anchors_available"] = _split_anchors(tc["anchors_available"])

    if "anchors_used" in tc:
        norm["anchors_used"] = _split_anchors(tc["anchors_used"])

    if "query_formulation_seconds" in tc:
        raw_qfs = str(tc["query_formulation_seconds"]).strip("() \t\n")
        try:
            norm["query_formulation_seconds"] = float(raw_qfs)
        except ValueError:
            pass

    if "evaluator_intervention_pre_submit" in tc:
        val = str(tc["evaluator_intervention_pre_submit"]).strip().lower()
        norm["evaluator_intervention_pre_submit"] = val in {"true", "1", "yes"}

    if "warning_in_their_words" in po:
        norm["warning_in_their_words"] = po["warning_in_their_words"]
    if "warning_comprehension" in po:
        wc = po["warning_comprehension"].strip().lower()
        if wc in SESSION06_COMPREHENSION:
            norm["warning_comprehension"] = wc
    if "rewrite_decision" in po:
        rd = po["rewrite_decision"].strip().lower()
        if rd in {"accept_with_edit", "accept_with_edits"}:
            rd = "accept_with_edits"
        norm["rewrite_decision"] = rd
    if "edit_made" in po:
        val = str(po["edit_made"]).strip().lower()
        norm["edit_made"] = val in {"true", "1", "yes"}
    if "edited_query" in po and po["edited_query"] != "none":
        norm["edited_query"] = po["edited_query"]
    if "usefulness" in po:
        raw_u = str(po["usefulness"]).strip()
        if "-" in raw_u:
            parts = [float(p.strip()) for p in raw_u.split("-") if p.strip()]
            norm["usefulness"] = sum(parts) / len(parts) if parts else 5.0
        else:
            try:
                norm["usefulness"] = float(raw_u)
            except ValueError:
                norm["usefulness"] = 5.0
    if "would_use_before_sending" in po:
        val = str(po["would_use_before_sending"]).strip().lower()
        norm["would_use_before_sending"] = val in {"true", "1", "yes"}
    if "adoption_reason" in po:
        norm["adoption_reason"] = po["adoption_reason"]
    if "time_to_decision_seconds" in po:
        raw_tds = str(po["time_to_decision_seconds"]).strip("() \t\n")
        try:
            norm["time_to_decision_seconds"] = float(raw_tds)
        except ValueError:
            pass

    comments = []
    if rc.get("comments") and rc["comments"] != "none":
        comments.append(rc["comments"])
    if rc.get("notable_model_behavior") and rc["notable_model_behavior"] != "none":
        comments.append(rc["notable_model_behavior"])
    if comments:
        norm["comments"] = "; ".join(comments)

    norm.setdefault("system_state", "risk_positive_rewrite")
    return norm


def validate_session_record(data: Dict[str, Any]) -> Session06RecordSchema:
    """Validate raw dictionary record against Session06RecordSchema."""
    if "task_context" in data or "participant_observation" in data:
        data = _normalize_legacy_record(data)
    return Session06RecordSchema(**data)


def write_session_record(
    record_data: Union[Dict[str, Any], Session06RecordSchema],
    filepath: Union[str, Path],
) -> Session06RecordSchema:
    """Validate and atomically append a session record to a JSON or JSONL file.

    Guarantees that:
    1. Schema validation is enforced at write time (rejects malformed records).
    2. Serialization is valid JSON without comma omission or syntax corruption.
    """
    if isinstance(record_data, dict):
        validated = validate_session_record(record_data)
    else:
        validated = record_data

    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    rec_dict = validated.model_dump(exclude_none=True)

    if str(path).endswith(".jsonl"):
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec_dict) + "\n")
    else:
        # JSON Array format: read existing list, append, atomically rewrite
        records: List[Dict[str, Any]] = []
        if path.exists() and path.stat().st_size > 0:
            with open(path, "r", encoding="utf-8") as f:
                content = f.read().strip()
                if content:
                    records = json.loads(content)
                    if isinstance(records, dict):
                        records = [records]

        records.append(rec_dict)

        # Atomic write via temporary file
        dir_name = path.parent
        with tempfile.NamedTemporaryFile("w", dir=dir_name, delete=False, encoding="utf-8") as tf:
            json.dump(records, tf, indent=2)
            temp_path = tf.name

        os.replace(temp_path, path)

    return validated
