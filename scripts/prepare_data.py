"""Fetch SynthPAI at the pinned revision. Run this before anything else.

    python scripts/prepare_data.py

Closes the gap in issue #62: the SFT pipeline needed `data/raw/synthpai.jsonl`
and nothing in the repository created it, so a fresh clone failed at the first
step with an error that said what was missing and not how to get it.

The dataset is CC-BY-NC-SA-4.0 and deliberately not committed. The revision
comes from `configs/models.yaml` rather than being written here, so there is
one answer to "which SynthPAI" across the whole project -- see issue #21.

Idempotent: if the file is already present and the right size, it does
nothing unless `--force` is given.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data.dataset_pin import (  # noqa: E402
    DATASET_ID,
    DEFAULT_DATA_PATH,
    pinned_revision,
)

FILENAME = "synthpai.jsonl"

# Sanity floor, not a checksum. The published file is ~19 MB; anything much
# smaller means a truncated download or an HTML error page saved as JSONL,
# which would otherwise surface much later as a confusing parse failure.
MIN_BYTES = 5_000_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--dest",
        type=Path,
        default=DEFAULT_DATA_PATH,
        help=f"where to write the file (default: {DEFAULT_DATA_PATH})",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="re-download even if a plausible file is already there",
    )
    return parser.parse_args()


def verify(path: Path) -> tuple[bool, str]:
    """Cheap checks that the file on disk is the dataset and not debris."""
    if not path.exists():
        return False, "missing"
    size = path.stat().st_size
    if size < MIN_BYTES:
        return False, f"only {size:,} bytes, expected at least {MIN_BYTES:,}"
    with path.open(encoding="utf-8") as handle:
        first = handle.readline()
    if not first.lstrip().startswith("{"):
        return False, "first line is not a JSON object"
    return True, f"{size:,} bytes"


def main() -> int:
    args = parse_args()
    revision = pinned_revision()
    dest = args.dest

    ok, detail = verify(dest)
    if ok and not args.force:
        print(f"Already present: {dest} ({detail})")
        print("Pass --force to download it again.")
        return 0
    if not ok and dest.exists():
        print(f"Replacing unusable file at {dest}: {detail}")

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print(
            "huggingface_hub is not installed. Install it with:\n"
            "\n"
            "    pip install huggingface_hub\n",
            file=sys.stderr,
        )
        return 1

    print(f"Downloading {DATASET_ID} at revision {revision[:12]} ...")
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        fetched = hf_hub_download(
            repo_id=DATASET_ID,
            filename=FILENAME,
            repo_type="dataset",
            revision=revision,
            local_dir=str(dest.parent),
        )
    except Exception as exc:  # network, auth, a revision that no longer exists
        print(f"Download failed: {exc}", file=sys.stderr)
        print(
            f"\nIf the revision is the problem, check dataset.revision in "
            f"configs/models.yaml. Do not simply drop the pin -- an unpinned "
            f"load changes the profile split without failing (issue #21).",
            file=sys.stderr,
        )
        return 1

    fetched_path = Path(fetched)
    if fetched_path.resolve() != dest.resolve():
        dest.write_bytes(fetched_path.read_bytes())

    ok, detail = verify(dest)
    if not ok:
        print(f"Downloaded file looks wrong: {detail}", file=sys.stderr)
        return 1

    print(f"Ready: {dest} ({detail})")
    print(f"  dataset  {DATASET_ID}")
    print(f"  revision {revision}")
    print("\nNext: python scripts/make_artifacts.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
