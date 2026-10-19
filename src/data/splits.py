# src/data/splits.py
"""Cryptographic profile-disjoint dataset splitting and manifest generation.

Partitions comments strictly by author profile_id using stratified hashing
to prevent personal style and author identity leakage between train, val, and test.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
import numpy as np


def extract_profile_id(record: Dict[str, Any]) -> str:
    """Extract author profile ID from comment record across SynthPAI schema variations."""
    for key in ("profile_id", "author", "username", "user_id"):
        if key in record and record[key]:
            return str(record[key])
    raise KeyError("Record missing valid profile_id or author identifier.")


def create_profile_disjoint_splits(
    records: List[Dict[str, Any]],
    output_dir: Path,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    seed: int = 641,
    manifest_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Partition records strictly by author profile_id using demographic stratification.

    Guarantees that train, val, and test splits share zero author profile IDs,
    writes split JSONL files, and generates cryptographic SHA-256 manifest hashes.
    """
    if not records:
        raise ValueError("Cannot partition empty records list.")

    profile_to_records: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    profile_strata: Dict[str, str] = {}

    for rec in records:
        pid = extract_profile_id(rec)
        profile_to_records[pid].append(rec)
        if pid not in profile_strata:
            attrs = rec.get("attributes", {})
            age_bucket = str(attrs.get("age", rec.get("age", "unk")))
            occ_bucket = str(attrs.get("occupation", rec.get("occupation", "unk")))
            profile_strata[pid] = f"{age_bucket}__{occ_bucket}"

    strata_groups: Dict[str, List[str]] = defaultdict(list)
    for pid, stratum in sorted(profile_strata.items()):
        strata_groups[stratum].append(pid)

    rng = np.random.default_rng(seed)
    train_pids: Set[str] = set()
    val_pids: Set[str] = set()
    test_pids: Set[str] = set()

    for _, pids in sorted(strata_groups.items()):
        pids_arr = list(pids)
        rng.shuffle(pids_arr)
        n = len(pids_arr)
        n_train = max(1, int(round(n * train_ratio)))
        n_val = max(0, int(round(n * val_ratio)))
        if n_train + n_val >= n and n >= 3:
            n_train = n - 2
            n_val = 1

        train_pids.update(pids_arr[:n_train])
        val_pids.update(pids_arr[n_train : n_train + n_val])
        test_pids.update(pids_arr[n_train + n_val :])

    # Enforce strict zero-leakage disjointness invariants
    assert train_pids.isdisjoint(val_pids), "Profile overlap detected between train and val splits!"
    assert train_pids.isdisjoint(test_pids), "Profile overlap detected between train and test splits!"
    assert val_pids.isdisjoint(test_pids), "Profile overlap detected between val and test splits!"

    output_dir.mkdir(parents=True, exist_ok=True)
    split_map = {"train": train_pids, "val": val_pids, "test": test_pids}
    manifest_hashes: Dict[str, Any] = {}

    for split_name, pid_set in split_map.items():
        split_file = output_dir / f"{split_name}.jsonl"
        hasher = hashlib.sha256()
        count = 0
        with split_file.open("w", encoding="utf-8", newline="\n") as f:
            for pid in sorted(pid_set):
                for row in profile_to_records[pid]:
                    row_copy = dict(row)
                    row_copy["split"] = split_name
                    row_copy["profile_id"] = pid
                    line_str = json.dumps(row_copy, sort_keys=True) + "\n"
                    hasher.update(line_str.encode("utf-8"))
                    f.write(line_str)
                    count += 1
        manifest_hashes[split_name] = {
            "num_profiles": len(pid_set),
            "num_comments": count,
            "sha256": hasher.hexdigest(),
            "profile_ids": sorted(pid_set),
        }

    # Write manifest in output_dir and data/manifests
    manifest_json = json.dumps(manifest_hashes, indent=2)
    (output_dir / "split_manifest.json").write_text(manifest_json, encoding="utf-8")

    target_manifest_dir = manifest_dir or (Path("data") / "manifests")
    target_manifest_dir.mkdir(parents=True, exist_ok=True)
    (target_manifest_dir / "split_manifest.json").write_text(manifest_json, encoding="utf-8")

    return manifest_hashes


def load_or_fetch_synthpai(local_path: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Load SynthPAI comments from local jsonl or fetch from Hugging Face repository."""
    if local_path and local_path.exists():
        records: List[Dict[str, Any]] = []
        with local_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records

    try:
        from datasets import load_dataset
        ds = load_dataset("RobinSta/SynthPAI")
        records = []
        for split_key in ds.keys():
            for row in ds[split_key]:
                records.append(dict(row))
        return records
    except Exception as exc:
        raise RuntimeError(f"Could not load SynthPAI from local or HuggingFace: {exc}") from exc


def main() -> None:
    """CLI entrypoint for generating profile-disjoint splits."""
    parser = argparse.ArgumentParser(description="Generate profile-disjoint SynthPAI splits.")
    parser.add_argument("--input_path", type=Path, default=Path("data/raw/synthpai.jsonl"))
    parser.add_argument("--output_dir", type=Path, default=Path("data/splits"))
    parser.add_argument("--manifest_dir", type=Path, default=Path("data/manifests"))
    parser.add_argument("--seed", type=int, default=641)
    args = parser.parse_args()

    records = load_or_fetch_synthpai(args.input_path)
    manifest = create_profile_disjoint_splits(
        records=records,
        output_dir=args.output_dir,
        manifest_dir=args.manifest_dir,
        seed=args.seed,
    )
    print(f"Successfully generated profile-disjoint splits for {len(records)} comments.")
    for split_name, info in manifest.items():
        print(f"  {split_name}: {info['num_profiles']} profiles, {info['num_comments']} comments")


if __name__ == "__main__":
    main()
