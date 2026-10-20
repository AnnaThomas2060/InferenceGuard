"""Tests for Milestone 5: Operations State and Pseudonyms (F15, F16).

Verifies PersistentPseudonymManager entity types, bidirectional consistency,
cryptographic persistence, and ConversationTracker multi-turn dynamic leakage accumulation.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
import pytest

from src.conversation.pseudonyms import PersistentPseudonymManager
from src.conversation.state import ConversationTracker, SessionManager


def test_pseudonym_manager_entity_types():
    """Verify entity types PER, LOC, ORG, DATE, ID generate structured pseudonyms."""
    with tempfile.TemporaryDirectory() as tmpdir:
        mgr = PersistentPseudonymManager(storage_dir=tmpdir)

        p_per = mgr.get_or_create("Alice Smith", "PER", session_id="s1")
        p_loc = mgr.get_or_create("Denver, CO", "LOC", session_id="s1")
        p_org = mgr.get_or_create("Google", "ORG", session_id="s1")
        p_date = mgr.get_or_create("1998-05-12", "DATE", session_id="s1")
        p_id = mgr.get_or_create("alice@test.com", "ID", session_id="s1")

        assert p_per == "[PER_1]"
        assert p_loc == "[LOC_1]"
        assert p_org == "[ORG_1]"
        assert p_date == "[DATE_1]"
        assert p_id == "[ID_1]"

        # Second person gets sequential index
        p_per2 = mgr.get_or_create("Bob Jones", "PER", session_id="s1")
        assert p_per2 == "[PER_2]"


def test_pseudonym_manager_bidirectional_consistency():
    """Verify bidirectional consistency: pseudonymize -> depseudonymize restores original text."""
    with tempfile.TemporaryDirectory() as tmpdir:
        mgr = PersistentPseudonymManager(storage_dir=tmpdir)
        session_id = "test_sess"

        raw_text = "Alice Smith moved to Denver, CO to work at Google."
        entities = [
            ("Alice Smith", "PER"),
            ("Denver, CO", "LOC"),
            ("Google", "ORG"),
        ]

        pseudo_text = mgr.pseudonymize(raw_text, session_id=session_id, known_entities=entities)
        assert "Alice Smith" not in pseudo_text
        assert "[PER_1]" in pseudo_text
        assert "Denver, CO" not in pseudo_text
        assert "[LOC_1]" in pseudo_text
        assert "Google" not in pseudo_text
        assert "[ORG_1]" in pseudo_text

        # Calling again gives exact same pseudonyms
        pseudo_text_2 = mgr.pseudonymize(raw_text, session_id=session_id, known_entities=entities)
        assert pseudo_text == pseudo_text_2

        # Depseudonymize restores original text
        restored = mgr.depseudonymize(pseudo_text, session_id=session_id)
        assert restored == raw_text


def test_pseudonym_manager_encrypted_persistence():
    """Verify vault is saved to disk and reloaded with encrypted entities."""
    with tempfile.TemporaryDirectory() as tmpdir:
        mgr = PersistentPseudonymManager(storage_dir=tmpdir, master_secret="vault_secret_123")
        mgr.get_or_create("Secret Location", "LOC", session_id="sess_enc")

        vault_file = Path(tmpdir) / "pseudonym_vault.json"
        assert vault_file.exists()

        vault_content = vault_file.read_text(encoding="utf-8")
        # Ensure plaintext real entity is not in disk file
        assert "Secret Location" not in vault_content
        assert "[LOC_1]" in vault_content

        # Reload with fresh manager instance
        mgr_reload = PersistentPseudonymManager(storage_dir=tmpdir, master_secret="vault_secret_123")
        p = mgr_reload.get_or_create("Secret Location", "LOC", session_id="sess_enc")
        assert p == "[LOC_1]"
        depseudo = mgr_reload.depseudonymize("[LOC_1]", session_id="sess_enc")
        assert depseudo == "Secret Location"


def test_conversation_tracker_turn_accumulation():
    """Verify multi-turn independent risk accumulation: P_cum = 1 - prod(1 - P_t)."""
    tracker = ConversationTracker()
    session_id = "multi_turn_1"

    # Turn 1: Mild age clue (e.g. 0.30)
    def mock_estimator_t1(_):
        return {"age": 0.30, "location": 0.05, "occupation": 0.05, "education": 0.05}
    tracker.risk_estimator_fn = mock_estimator_t1
    rec1 = tracker.add_turn("I am finishing college soon.", session_id=session_id)

    assert rec1.turn_number == 1
    assert rec1.cumulative_scores["age"] == pytest.approx(0.30, 1e-4)
    assert rec1.leakage_delta == pytest.approx(0.30, 1e-4)

    # Turn 2: Another age clue (e.g. 0.40)
    # P_cum(age) = 1 - (1 - 0.30)*(1 - 0.40) = 1 - (0.70 * 0.60) = 1 - 0.42 = 0.58
    def mock_estimator_t2(_):
        return {"age": 0.40, "location": 0.05, "occupation": 0.05, "education": 0.05}
    tracker.risk_estimator_fn = mock_estimator_t2
    rec2 = tracker.add_turn("Turning 22 next month.", session_id=session_id)

    assert rec2.turn_number == 2
    assert rec2.cumulative_scores["age"] == pytest.approx(0.58, 1e-3)
    # leakage_delta = 0.58 - 0.30 = 0.28
    assert rec2.leakage_delta == pytest.approx(0.28, 1e-3)


def test_conversation_tracker_entropy_reduction():
    """Verify that accumulating disclosures strictly reduces joint anonymity entropy."""
    tracker = ConversationTracker()
    session_id = "entropy_sess"

    # Turn 1: No cues
    tracker.risk_estimator_fn = lambda _: {"age": 0.05, "location": 0.05, "occupation": 0.05, "education": 0.05}
    rec1 = tracker.add_turn("Hello there.", session_id=session_id)
    initial_entropy = rec1.joint_entropy

    # Turn 2: Major location cue
    tracker.risk_estimator_fn = lambda _: {"age": 0.05, "location": 0.85, "occupation": 0.05, "education": 0.05}
    rec2 = tracker.add_turn("I live right next to the Space Needle.", session_id=session_id)

    assert rec2.joint_entropy < initial_entropy
    assert rec2.cumulative_scores["location"] >= 0.85


def test_conversation_tracker_breach_detection():
    """Verify is_breached flag is set when cumulative risk >= 0.60 or entropy < 1.50."""
    tracker = ConversationTracker(high_risk_threshold=0.60)
    session_id = "breach_sess"

    # Turn 1: Low risk
    tracker.risk_estimator_fn = lambda _: {"age": 0.20, "location": 0.10, "occupation": 0.10, "education": 0.10}
    rec1 = tracker.add_turn("General comment.", session_id=session_id)
    assert rec1.is_breached is False
    assert rec1.risk_band in ("LOW", "MEDIUM")

    # Turn 2: High risk disclosure
    tracker.risk_estimator_fn = lambda _: {"age": 0.20, "location": 0.75, "occupation": 0.10, "education": 0.10}
    rec2 = tracker.add_turn("Living in downtown Seattle, WA.", session_id=session_id)
    assert rec2.is_breached is True
    assert rec2.risk_band == "HIGH"


def test_session_manager_backward_compatibility():
    """Verify lightweight SessionManager backward-compatible methods."""
    with tempfile.TemporaryDirectory() as tmpdir:
        sm = SessionManager(storage_dir=tmpdir)
        sess_id = sm.start_session(user_id="user_1")
        assert sess_id in sm.sessions

        sm.add_message(sess_id, "user", "Hello", timestamp=100.0)
        assert len(sm.sessions[sess_id].messages) == 1

        sm.end_session(sess_id)
        assert sm.sessions[sess_id].status == "ended"


def test_bayesian_evidence_gating_prevents_false_positive_alarm_on_20_innocent_turns():
    """Verify Bayesian Evidence Gating prevents background noise from compounding into breach alarms.

    Over 20 innocent conversational turns, each turn outputs baseline background noise P_t <= 0.05.
    Ungated naive multiplication would compound to 1 - (1 - 0.05)^20 = 0.642, falsely triggering
    a high-risk breach alarm. With Bayesian Evidence Gating, background noise is not compounded,
    preventing false positive breach alerts.
    """
    tracker = ConversationTracker(high_risk_threshold=0.60)
    session_id = "innocent_20_turns"
    tracker.risk_estimator_fn = lambda _: {"age": 0.05, "location": 0.05, "occupation": 0.05, "education": 0.05}

    last_rec = None
    for turn_idx in range(1, 21):
        last_rec = tracker.add_turn(f"Innocent greeting number {turn_idx}.", session_id=session_id)

    assert last_rec is not None
    assert last_rec.turn_number == 20
    # Cumulative risk must not have compounded to 0.642
    assert last_rec.overall_cumulative_risk <= 0.05
    assert last_rec.is_breached is False
    assert last_rec.residual_anonymity_capacity >= 3.8
    assert last_rec.residual_anonymity_capacity == last_rec.joint_entropy


def test_bayesian_evidence_gating_monotonicity_under_baseline_fluctuation():
    """Verify that cumulative scores do not decrease when turn scores fluctuate below noise floor."""
    tracker = ConversationTracker(high_risk_threshold=0.60)
    session_id = "monotonic_sess"

    # Turn 1: 0.04
    tracker.risk_estimator_fn = lambda _: {"age": 0.04, "location": 0.01, "occupation": 0.01, "education": 0.01}
    rec1 = tracker.add_turn("Turn 1", session_id=session_id)
    assert rec1.cumulative_scores["age"] == pytest.approx(0.04, 1e-4)

    # Turn 2: 0.01 (lower baseline must not decrease cumulative risk from 0.04 to 0.01)
    tracker.risk_estimator_fn = lambda _: {"age": 0.01, "location": 0.01, "occupation": 0.01, "education": 0.01}
    rec2 = tracker.add_turn("Turn 2", session_id=session_id)
    assert rec2.cumulative_scores["age"] == pytest.approx(0.04, 1e-4)

    # Turn 3: 0.05 (updates to max baseline 0.05)
    tracker.risk_estimator_fn = lambda _: {"age": 0.05, "location": 0.01, "occupation": 0.01, "education": 0.01}
    rec3 = tracker.add_turn("Turn 3", session_id=session_id)
    assert rec3.cumulative_scores["age"] == pytest.approx(0.05, 1e-4)

    # Turn 4: 0.02 (retains 0.05)
    tracker.risk_estimator_fn = lambda _: {"age": 0.02, "location": 0.01, "occupation": 0.01, "education": 0.01}
    rec4 = tracker.add_turn("Turn 4", session_id=session_id)
    assert rec4.cumulative_scores["age"] == pytest.approx(0.05, 1e-4)

    # Turn 5: 0.30 (genuine disclosure above noise floor)
    tracker.risk_estimator_fn = lambda _: {"age": 0.30, "location": 0.01, "occupation": 0.01, "education": 0.01}
    rec5 = tracker.add_turn("Turn 5", session_id=session_id)
    assert rec5.cumulative_scores["age"] == pytest.approx(0.30, 1e-4)

    # Turn 6: 0.02 (noise floor; does not compound on top of 0.30)
    tracker.risk_estimator_fn = lambda _: {"age": 0.02, "location": 0.01, "occupation": 0.01, "education": 0.01}
    rec6 = tracker.add_turn("Turn 6", session_id=session_id)
    assert rec6.cumulative_scores["age"] == pytest.approx(0.30, 1e-4)




