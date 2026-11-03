from __future__ import annotations

import hashlib
import json
from pathlib import Path

from scripts.check_eval_readiness import (
    ADAPTER_MANIFEST_SCHEMA,
    check_readiness,
)


def write_bytes(path: Path, data: bytes) -> str:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_bytes(data)

    return hashlib.sha256(
        data
    ).hexdigest()


def write_json(path: Path, value) -> str:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    payload = (
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")

    path.write_bytes(payload)

    return hashlib.sha256(
        payload
    ).hexdigest()


def make_fixture(tmp_path: Path):
    train_path = (
        tmp_path
        / "data/splits/train.jsonl"
    )

    test_path = (
        tmp_path
        / "data/splits/test.jsonl"
    )

    sft_path = (
        tmp_path
        / "data/sft/teacher_v1/train.jsonl"
    )

    train_hash = write_bytes(
        train_path,
        b'{"profile_id":"train"}\n',
    )

    test_hash = write_bytes(
        test_path,
        b'{"profile_id":"test"}\n',
    )

    sft_hash = write_bytes(
        sft_path,
        b'{"messages":[]}\n',
    )

    split_manifest_path = (
        tmp_path
        / "data/manifests/split_manifest.json"
    )

    write_json(
        split_manifest_path,
        {
            "train": {
                "sha256":
                    train_hash,
            },
            "test": {
                "sha256":
                    test_hash,
            },
        },
    )

    teacher_path = (
        tmp_path
        / "data/sft/teacher_v1/"
        "manifest_train.json"
    )

    teacher_manifest = {
        "split": "train",
        "git_dirty": False,
        "teacher": {
            "model_id":
                "Qwen/Qwen2.5-7B-Instruct",
            "revision":
                "1" * 40,
        },
        "nli": {
            "model_id":
                "cross-encoder/"
                "nli-deberta-v3-large",
            "revision":
                "2" * 40,
        },
        "input": {
            "path":
                "data/splits/train.jsonl",
            "sha256":
                train_hash,
        },
        "outputs": {
            "sft": {
                "path":
                    "data/sft/teacher_v1/"
                    "train.jsonl",
                "sha256":
                    sft_hash,
            }
        },
    }

    teacher_hash = write_json(
        teacher_path,
        teacher_manifest,
    )

    write_json(
        tmp_path
        / "artifacts/profile_splits.json",
        {
            "seed": 42,
        },
    )

    config = {
        "experiment_id":
            "session08_heldout_eval_v1",

        "session": 8,

        "experiment_type":
            "heldout_evaluation",

        "schema_version":
            "session08_v1",

        "random_seed": 42,

        "dataset": {
            "name":
                "RobinSta/SynthPAI",
            "revision":
                "a" * 40,
            "split":
                "test",
        },

        "split_provenance": {
            "profile_split_file":
                "artifacts/profile_splits.json",

            "manifest_file":
                "data/manifests/"
                "split_manifest.json",

            "test_split_file":
                "data/splits/test.jsonl",

            "expected_seed":
                42,

            "require_profile_disjoint":
                True,

            "require_manifest_hashes":
                True,
        },

        # Intentionally still 1.5B.
        # The held-out evaluation target below
        # must be allowed to select 0.5B explicitly.
        "rewriter": {
            "name":
                "Qwen/Qwen2.5-1.5B-Instruct",

            "checkpoint":
                "artifacts/rewriter_qlora",
        },

        "risk_model": {
            "name":
                "ModernBERT-base",

            "checkpoint":
                "artifacts/risk_model/best_model",
        },

        "training_provenance": {
            "teacher_manifest_file":
                "data/sft/teacher_v1/"
                "manifest_train.json",

            "training_dataset_file":
                "data/sft/teacher_v1/"
                "train.jsonl",

            "require_teacher_manifest_clean_git":
                True,

            "require_training_dataset_hash_match":
                True,

            "require_train_split_hash_match":
                True,

            "forbid_attacker_as_teacher":
                True,

            "require_teacher_revision":
                True,
        },

        "student_provenance": {
            "evaluation_target":
                "local_student",

            "targets": {
                "local_student": {
                    "role":
                        "local_first",

                    "base_model":
                        "Qwen/Qwen2.5-0.5B-Instruct",

                    "adapter_dir":
                        "artifacts/rewriter_qlora",

                    "adapter_manifest_file":
                        "artifacts/rewriter_qlora/"
                        "training_manifest.json",
                },

                "quality_ceiling": {
                    "role":
                        "quality_ceiling",

                    "base_model":
                        "Qwen/Qwen2.5-1.5B-Instruct",

                    "adapter_dir":
                        "artifacts/"
                        "rewriter_qlora_1p5b",

                    "adapter_manifest_file":
                        "artifacts/"
                        "rewriter_qlora_1p5b/"
                        "training_manifest.json",
                },
            },

            "require_adapter_manifest":
                True,

            "require_base_model_match":
                True,

            "require_training_dataset_hash_match":
                True,

            "require_teacher_manifest_hash_match":
                True,

            "require_adapter_file_hashes":
                True,
        },

        "attacker": {
            "name":
                "microsoft/"
                "Phi-4-mini-instruct",

            "required_mode":
                "model",
        },

        "utility_evaluator": {
            "semantic_similarity_model":
                "sentence-transformers/"
                "all-MiniLM-L6-v2",

            "nli_model":
                "cross-encoder/"
                "nli-deberta-v3-large",

            "required_backend_for_nli_claims":
                "model",
        },

        "evaluation": {
            "profile_disjoint_test_only":
                True,

            "heavy_models_required":
                True,

            "allow_diagnostic_fallbacks":
                True,
        },

        "reportability": {
            "privacy": {
                "require_attacker_model_match":
                    True,

                "require_attacker_mode_model":
                    True,

                "require_reportable_as_held_out":
                    True,

                "require_test_split":
                    True,

                "require_pinned_dataset_revision":
                    True,

                "require_split_manifest":
                    True,
            },

            "utility": {
                "require_model_backend_for_nli_claims":
                    True,

                "fallback_metrics_diagnostic_only":
                    True,
            },

            "rewrite": {
                "require_engine_provenance":
                    True,

                "unchanged_high_risk_is_not_protected":
                    True,
            },
        },

        "metrics": {
            "privacy": [],
            "utility": [],
            "provenance": [],
            "performance": [],
        },

        "output": {
            "root":
                "experiments/session08",
        },

        "validation": {
            "fail_on_missing_required_fields":
                True,

            "fail_on_dataset_revision_mismatch":
                True,

            "fail_on_split_mismatch":
                True,

            "fail_on_missing_split_manifest":
                True,

            "fail_on_attacker_model_mismatch":
                True,

            "fail_on_non_model_attacker_for_privacy_claim":
                True,

            "fail_on_false_heldout_reportability":
                True,

            "block_nli_claims_on_fallback_backend":
                True,

            "require_rewrite_engine_provenance":
                True,

            "require_git_commit_sha":
                True,
        },
    }

    return {
        "config":
            config,

        "sft_hash":
            sft_hash,

        "teacher_hash":
            teacher_hash,

        "adapter_dir":
            tmp_path
            / "artifacts/rewriter_qlora",
    }


def add_registration(
    fx,
    *,
    include_artifact: bool,
):
    adapter_dir = fx[
        "adapter_dir"
    ]

    config_hash = write_json(
        adapter_dir
        / "adapter_config.json",

        {
            "base_model_name_or_path":
                "Qwen/Qwen2.5-0.5B-Instruct",
        },
    )

    model_path = (
        adapter_dir
        / "adapter_model.safetensors"
    )

    model_hash = hashlib.sha256(
        b"real-adapter"
    ).hexdigest()

    if include_artifact:
        write_bytes(
            model_path,
            b"real-adapter",
        )

    manifest = {
        "schema_version":
            ADAPTER_MANIFEST_SCHEMA,

        "base_model_id":
            "Qwen/Qwen2.5-0.5B-Instruct",

        "training_dataset": {
            "path":
                "data/sft/teacher_v1/"
                "train.jsonl",

            "sha256":
                fx["sft_hash"],
        },

        "teacher_manifest": {
            "path":
                "data/sft/teacher_v1/"
                "manifest_train.json",

            "sha256":
                fx["teacher_hash"],
        },

        "adapter_files": {
            "adapter_config.json":
                config_hash,

            "adapter_model.safetensors":
                model_hash,
        },
    }

    write_json(
        adapter_dir
        / "training_manifest.json",

        manifest,
    )


def test_0p5b_target_is_explicit_even_when_rewriter_default_is_1p5b(
    tmp_path,
):
    fx = make_fixture(tmp_path)

    result = check_readiness(
        fx["config"],
        root=tmp_path,
    )

    assert (
        result["checks"][
            "evaluation_target"
        ]["status"]
        == "PASS"
    )

    assert (
        result["provenance"][
            "evaluation_target_id"
        ]
        == "local_student"
    )

    assert (
        result["provenance"][
            "evaluation_target_base_model_id"
        ]
        == "Qwen/Qwen2.5-0.5B-Instruct"
    )


def test_registration_can_pass_while_artifact_availability_blocks(
    tmp_path,
):
    fx = make_fixture(tmp_path)

    add_registration(
        fx,
        include_artifact=False,
    )

    result = check_readiness(
        fx["config"],
        root=tmp_path,
    )

    assert (
        result["checks"][
            "adapter_registration"
        ]["status"]
        == "PASS"
    )

    assert (
        result["checks"][
            "artifact_availability"
        ]["status"]
        == "BLOCKED"
    )

    assert (
        result["overall_status"]
        == "BLOCKED"
    )


def test_available_hash_locked_target_becomes_ready(
    tmp_path,
):
    fx = make_fixture(tmp_path)

    add_registration(
        fx,
        include_artifact=True,
    )

    result = check_readiness(
        fx["config"],
        root=tmp_path,
    )

    assert (
        result["checks"][
            "evaluation_target"
        ]["status"]
        == "PASS"
    )

    assert (
        result["checks"][
            "adapter_registration"
        ]["status"]
        == "PASS"
    )

    assert (
        result["checks"][
            "artifact_availability"
        ]["status"]
        == "PASS"
    )

    assert (
        result["checks"][
            "student_adapter"
        ]["status"]
        == "PASS"
    )

    assert (
        result["overall_status"]
        == "READY"
    )


def test_unknown_evaluation_target_blocks(
    tmp_path,
):
    fx = make_fixture(tmp_path)

    fx["config"][
        "student_provenance"
    ][
        "evaluation_target"
    ] = "does_not_exist"

    result = check_readiness(
        fx["config"],
        root=tmp_path,
    )

    assert (
        result["checks"][
            "evaluation_target"
        ]["status"]
        == "BLOCKED"
    )

    assert (
        result["overall_status"]
        == "BLOCKED"
    )
