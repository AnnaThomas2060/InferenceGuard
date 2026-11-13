from __future__ import annotations

import json
from pathlib import Path

import yaml

from scripts.check_production_evidence import (
    check_production_evidence,
)


RISK_MODEL = "AT707/InferenceGuard-Roberta-Risk"
REWRITER_MODEL = "AT707/inferenceguard-rewriter-v1"
REWRITER_BASE = "Qwen/Qwen2.5-1.5B-Instruct"

RISK_REVISION = "a" * 40
REWRITER_REVISION = "b" * 40


def write_json(
    path: Path,
    value,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        json.dumps(
            value,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def write_yaml(
    path: Path,
    value,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        yaml.safe_dump(
            value,
            sort_keys=False,
        ),
        encoding="utf-8",
    )


def make_fixture(
    tmp_path: Path,
):
    compose = {
        "services": {
            "inference-guard-api": {
                "environment": [
                    (
                        "RISK_MODEL_ID="
                        "${RISK_MODEL_ID:-"
                        + RISK_MODEL
                        + "}"
                    ),
                    (
                        "RISK_MODEL_REVISION="
                        "${RISK_MODEL_REVISION:-"
                        + RISK_REVISION
                        + "}"
                    ),
                    (
                        "REWRITER_MODEL_ID="
                        "${REWRITER_MODEL_ID:-"
                        + REWRITER_MODEL
                        + "}"
                    ),
                    (
                        "REWRITER_MODEL_REVISION="
                        "${REWRITER_MODEL_REVISION:-"
                        + REWRITER_REVISION
                        + "}"
                    ),
                ]
            }
        }
    }

    write_yaml(
        tmp_path / "docker-compose.yml",
        compose,
    )

    risk_eval = {
        "model_id":
            RISK_MODEL,

        "revision":
            RISK_REVISION,

        "split":
            "val",

        "metrics": {
            "macro": {
                "f1": 0.70,
                "pr_auc": 0.75,
                "roc_auc": 0.85,
            }
        },

        "fitted_bands_in_sample": {
            "age": {
                "usable": True,
                "degraded": False,
            },

            "location": {
                "usable": True,
                "degraded": False,
            },

            "occupation": {
                "usable": True,
                "degraded": False,
            },

            "education": {
                "usable": True,
                "degraded": False,
            },
        },
    }

    risk_eval_path = (
        tmp_path
        / "artifacts/risk_eval/risk.json"
    )

    write_json(
        risk_eval_path,
        risk_eval,
    )

    readiness = {
        "overall_status":
            "READY",

        "provenance": {
            "evaluation_target_base_model_id":
                REWRITER_BASE,
        },
    }

    readiness_path = (
        tmp_path
        / "artifacts/rewriter/readiness.json"
    )

    write_json(
        readiness_path,
        readiness,
    )

    rewriter_eval_path = (
        tmp_path
        / "artifacts/rewriter/evaluation.json"
    )

    write_json(
        rewriter_eval_path,
        {
            "model_id":
                REWRITER_MODEL,

            "revision":
                REWRITER_REVISION,

            "reportable":
                True,
        },
    )

    heldout_path = (
        tmp_path
        / "artifacts/heldout/result.json"
    )

    write_json(
        heldout_path,
        {
            "attacker":
                "microsoft/Phi-4-mini-instruct",

            "reportable":
                True,
        },
    )

    config = {
        "schema_version":
            "session10_production_evidence_v1",

        "release_id":
            "test_release",

        "serving": {
            "compose_file":
                "docker-compose.yml",

            "service":
                "inference-guard-api",

            "risk_model_id_env":
                "RISK_MODEL_ID",

            "risk_model_revision_env":
                "RISK_MODEL_REVISION",

            "rewriter_model_id_env":
                "REWRITER_MODEL_ID",

            "rewriter_model_revision_env":
                "REWRITER_MODEL_REVISION",
        },

        "risk_model": {
            "served_model_id":
                RISK_MODEL,

            "evaluation_artifact":
                "artifacts/risk_eval/risk.json",

            "require_revision_pin":
                True,

            "require_revision_match":
                True,

            "require_usable_fitted_bands":
                True,
        },

        "rewriter": {
            "served_model_id":
                REWRITER_MODEL,

            "expected_base_model_id":
                REWRITER_BASE,

            "historical_readiness_artifact":
                "artifacts/rewriter/readiness.json",

            "require_readiness_base_model_match":
                True,

            "require_evaluation_artifact":
                True,

            "evaluation_artifact":
                "artifacts/rewriter/evaluation.json",
        },

        "heldout": {
            "require_reportable_result":
                True,

            "result_artifact":
                "artifacts/heldout/result.json",
        },

        "output": {
            "evidence_file":
                "artifacts/production/evidence.json",
        },
    }

    return {
        "config":
            config,

        "compose_path":
            tmp_path / "docker-compose.yml",

        "risk_eval_path":
            risk_eval_path,

        "readiness_path":
            readiness_path,

        "heldout_path":
            heldout_path,
    }


def test_complete_matching_evidence_is_ready(
    tmp_path,
):
    fx = make_fixture(tmp_path)

    result = check_production_evidence(
        fx["config"],
        root=tmp_path,
    )

    assert (
        result["overall_status"]
        == "READY"
    )

    assert all(
        check["status"] == "PASS"
        for check
        in result["checks"].values()
    )


def test_missing_served_revisions_blocks(
    tmp_path,
):
    fx = make_fixture(tmp_path)

    compose = yaml.safe_load(
        fx["compose_path"].read_text(
            encoding="utf-8"
        )
    )

    environment = (
        compose["services"]
        ["inference-guard-api"]
        ["environment"]
    )

    environment[:] = [
        item
        for item in environment
        if not item.startswith(
            (
                "RISK_MODEL_REVISION=",
                "REWRITER_MODEL_REVISION=",
            )
        )
    ]

    write_yaml(
        fx["compose_path"],
        compose,
    )

    result = check_production_evidence(
        fx["config"],
        root=tmp_path,
    )

    assert (
        result["checks"][
            "revision_pinning"
        ]["status"]
        == "BLOCKED"
    )

    assert any(
        "risk model revision is not pinned"
        in reason.lower()
        for reason
        in result["blocking_reasons"]
    )

    assert any(
        "rewriter revision is not pinned"
        in reason.lower()
        for reason
        in result["blocking_reasons"]
    )


def test_wrong_served_risk_revision_blocks(
    tmp_path,
):
    fx = make_fixture(tmp_path)

    compose = yaml.safe_load(
        fx["compose_path"].read_text(
            encoding="utf-8"
        )
    )

    environment = (
        compose["services"]
        ["inference-guard-api"]
        ["environment"]
    )

    for i, item in enumerate(
        environment
    ):
        if item.startswith(
            "RISK_MODEL_REVISION="
        ):
            environment[i] = (
                "RISK_MODEL_REVISION="
                "${RISK_MODEL_REVISION:-"
                + ("c" * 40)
                + "}"
            )

    write_yaml(
        fx["compose_path"],
        compose,
    )

    result = check_production_evidence(
        fx["config"],
        root=tmp_path,
    )

    assert (
        result["checks"][
            "risk_evaluation_evidence"
        ]["status"]
        == "BLOCKED"
    )

    assert any(
        "does not match the evaluated revision"
        in reason
        for reason
        in result["blocking_reasons"]
    )


def test_unusable_risk_bands_block(
    tmp_path,
):
    fx = make_fixture(tmp_path)

    risk_eval = json.loads(
        fx["risk_eval_path"].read_text(
            encoding="utf-8"
        )
    )

    risk_eval[
        "fitted_bands_in_sample"
    ][
        "location"
    ][
        "usable"
    ] = False

    risk_eval[
        "fitted_bands_in_sample"
    ][
        "location"
    ][
        "degraded"
    ] = True

    write_json(
        fx["risk_eval_path"],
        risk_eval,
    )

    result = check_production_evidence(
        fx["config"],
        root=tmp_path,
    )

    assert (
        result["checks"][
            "risk_calibration_evidence"
        ]["status"]
        == "BLOCKED"
    )

    assert any(
        "location"
        in reason
        for reason
        in result["blocking_reasons"]
    )


def test_readiness_for_different_rewriter_blocks(
    tmp_path,
):
    fx = make_fixture(tmp_path)

    readiness = json.loads(
        fx["readiness_path"].read_text(
            encoding="utf-8"
        )
    )

    readiness[
        "provenance"
    ][
        "evaluation_target_base_model_id"
    ] = (
        "Qwen/Qwen2.5-0.5B-Instruct"
    )

    write_json(
        fx["readiness_path"],
        readiness,
    )

    result = check_production_evidence(
        fx["config"],
        root=tmp_path,
    )

    assert (
        result["checks"][
            "rewriter_evidence"
        ]["status"]
        == "BLOCKED"
    )

    assert any(
        "different rewriter base model"
        in reason
        for reason
        in result["blocking_reasons"]
    )


def test_missing_heldout_result_blocks(
    tmp_path,
):
    fx = make_fixture(tmp_path)

    fx["heldout_path"].unlink()

    result = check_production_evidence(
        fx["config"],
        root=tmp_path,
    )

    assert (
        result["checks"][
            "heldout_evidence"
        ]["status"]
        == "BLOCKED"
    )

    assert any(
        "Held-out production evaluation "
        "artifact is missing"
        in reason
        for reason
        in result["blocking_reasons"]
    )
