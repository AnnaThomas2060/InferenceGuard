"""The one place that answers "which SynthPAI are we using".

The revision is declared in `configs/models.yaml`. It was also written out
longhand in two experiment configs, a script and a test, while the two code
paths that actually download the dataset -- `src/data/splits.py` and
`scripts/make_artifacts.py` -- passed no revision at all and silently took
whatever the hub served that day.

That is issue #21. The seed was never the weak link: `train_test_split` is
deterministic, but it shuffles the list it is handed, and that list's order
comes from dataset iteration order. Pin the revision and the seed is
sufficient; leave it unpinned and the seed guarantees nothing.

Read with a regex rather than a YAML parser so that nothing in the data path
depends on PyYAML being installed, and so the guard tests that use this can
run on a machine with no dependencies at all.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODELS_YAML = ROOT / "configs" / "models.yaml"

DATASET_ID = "RobinSta/SynthPAI"

# Where prepare_data.py puts the file, and where everything else looks for it.
DEFAULT_DATA_PATH = ROOT / "data" / "raw" / "synthpai.jsonl"

_REVISION_RE = re.compile(
    r"^dataset:\s*$.*?^\s*revision:\s*([0-9a-f]{7,40})\s*(?:#.*)?$",
    re.MULTILINE | re.DOTALL,
)


def pinned_revision() -> str:
    """The SynthPAI commit every download must ask for.

    Raises rather than defaulting. A wrong-but-plausible revision is worse
    than a crash: it would produce a different split with the same seed and
    nothing downstream would notice.
    """
    if not MODELS_YAML.exists():
        raise FileNotFoundError(f"configs/models.yaml is missing; expected at {MODELS_YAML}")
    match = _REVISION_RE.search(MODELS_YAML.read_text(encoding="utf-8"))
    if not match:
        raise ValueError(
            "configs/models.yaml has no dataset.revision pin. Every SynthPAI "
            "download must name a revision -- see issue #21."
        )
    return match.group(1)


def missing_data_message(path: str | Path) -> str:
    """What to print when the dataset file is not on disk.

    Issue #62 was not that the failure was silent -- it raised a
    FileNotFoundError. It was that the error told you what was missing and
    nothing about how to get it, in a repository where the answer is one
    command.
    """
    return (
        f"SynthPAI input file not found: {path}\n"
        f"\n"
        f"Fetch it at the pinned revision first:\n"
        f"\n"
        f"    python scripts/prepare_data.py\n"
        f"\n"
        f"That downloads {DATASET_ID} at revision {pinned_revision()[:12]} "
        f"(pinned in configs/models.yaml) to {DEFAULT_DATA_PATH.relative_to(ROOT)}. "
        f"The dataset is CC-BY-NC-SA-4.0 and is not committed to this repository."
    )
