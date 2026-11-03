"""Guards for the dataset pin and the data-preparation path.

Issues #21 and #62. Deliberately dependency-free: these run on a clone with
nothing installed, which is the state a new contributor is actually in and
the state in which both issues were originally hit.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.dataset_pin import (  # noqa: E402
    DATASET_ID,
    missing_data_message,
    pinned_revision,
)

# Files that name the SynthPAI revision. All must agree; there is one dataset.
REVISION_DECLARATIONS = [
    Path("configs/models.yaml"),
    Path("configs/experiments/session06.yaml"),
    Path("configs/experiments/session08.yaml"),
    Path("scripts/dataset_report.py"),
]

# Anything that reaches the hub for SynthPAI. A local load_dataset("json", ...)
# is a different thing and is not covered.
HUB_LOAD = re.compile(r"load_dataset\(\s*(?:\"RobinSta/SynthPAI\"|'RobinSta/SynthPAI'|DATASET_ID)")


# --- the pin itself -------------------------------------------------------

def test_a_revision_is_pinned_and_looks_like_a_commit():
    revision = pinned_revision()
    assert re.fullmatch(r"[0-9a-f]{7,40}", revision), revision


def test_every_declaration_of_the_revision_agrees():
    """It is written out longhand in several files; they must not drift.

    A config pointing at one revision while the download uses another is the
    same failure as having no pin, and harder to spot.
    """
    expected = pinned_revision()
    disagreements = []
    for rel in REVISION_DECLARATIONS:
        path = ROOT / rel
        if not path.exists():
            continue
        found = set(re.findall(r"\b([0-9a-f]{40})\b", path.read_text(encoding="utf-8")))
        for revision in found:
            if revision != expected:
                disagreements.append(f"{rel} names {revision[:12]}, expected {expected[:12]}")
    assert not disagreements, "\n".join(disagreements)


def test_no_unpinned_hub_download_of_the_dataset():
    """The core of issue #21.

    The seed was never the weak link. `train_test_split` is deterministic but
    shuffles the list it is handed, and that order comes from dataset
    iteration order -- so an unpinned load changes the committed split without
    anything failing.
    """
    offenders = []
    for path in sorted(ROOT.glob("src/**/*.py")) + sorted(ROOT.glob("scripts/*.py")):
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if HUB_LOAD.search(line) and "revision" not in line:
                offenders.append(f"{path.relative_to(ROOT)}:{lineno}: {line.strip()}")
    assert not offenders, (
        "SynthPAI is downloaded without a revision:\n"
        + "\n".join(offenders)
        + "\n\nPass revision=pinned_revision() from src.data.dataset_pin."
    )


# --- the committed split artifacts ---------------------------------------

def test_the_two_committed_split_artifacts_agree():
    """`artifacts/profile_splits.json` and `data/manifests/split_manifest.json`
    are two recordings of one split, written by two different code paths.

    Nothing kept them in step. If they ever diverge, "the test split" stops
    having a single meaning, and every held-out claim inherits the ambiguity.
    """
    a_path = ROOT / "artifacts" / "profile_splits.json"
    m_path = ROOT / "data" / "manifests" / "split_manifest.json"
    if not (a_path.exists() and m_path.exists()):
        pytest.skip("one of the two split artifacts is not committed")

    artifacts = json.loads(a_path.read_text(encoding="utf-8"))
    manifest = json.loads(m_path.read_text(encoding="utf-8"))

    for split in ("train", "val", "test"):
        from_artifacts = set(artifacts[split])
        from_manifest = set(manifest[split]["profile_ids"])
        assert from_artifacts == from_manifest, (
            f"{split}: the two artifacts disagree -- "
            f"{len(from_artifacts - from_manifest)} only in profile_splits.json, "
            f"{len(from_manifest - from_artifacts)} only in split_manifest.json"
        )


def test_the_splits_remain_profile_disjoint():
    path = ROOT / "artifacts" / "profile_splits.json"
    if not path.exists():
        pytest.skip("artifacts/profile_splits.json is not committed")
    splits = json.loads(path.read_text(encoding="utf-8"))
    train, val, test = (set(splits[k]) for k in ("train", "val", "test"))
    assert not train & val and not train & test and not val & test
    assert len(train) + len(val) + len(test) == len(train | val | test)


# --- the data-preparation path -------------------------------------------

def test_the_missing_data_error_says_how_to_fix_it():
    """Issue #62. The old error named the missing file and stopped there."""
    message = missing_data_message("data/raw/synthpai.jsonl")
    assert "scripts/prepare_data.py" in message
    assert DATASET_ID in message
    assert pinned_revision()[:12] in message


def test_prepare_data_script_exists_and_is_self_contained():
    script = ROOT / "scripts" / "prepare_data.py"
    assert script.exists(), "issue #62 asks for an automated preparation step"
    source = script.read_text(encoding="utf-8")
    assert "pinned_revision" in source, "prepare_data must use the pin, not its own revision"
    assert "hf_hub_download" in source


def test_src_data_imports_without_the_heavy_stack():
    """`dataset_pin` exists partly to explain what to install.

    It cannot do that if importing it requires what is not installed, which
    it did until `src/data/__init__.py` stopped eagerly importing `splits`.
    """
    import subprocess

    result = subprocess.run(
        [sys.executable, "-c", "import src.data.dataset_pin as m; m.pinned_revision()"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
