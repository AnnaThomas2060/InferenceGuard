from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml


SCHEMA_VERSION = "session10_production_evidence_v1"

ENV_DEFAULT_RE = re.compile(
    r"^\$\{(?P<name>[A-Za-z_][A-Za-z0-9_]*):-"
    r"(?P<default>.*)\}$"
)


def load_yaml(path: Path) -> Dict[str, Any]:
    data = yaml.safe_load(
        path.read_text(encoding="utf-8")
    )

    if not isinstance(data, dict):
        raise ValueError(
            f"{path} must contain a YAML mapping"
        )

    return data


def load_json(path: Path) -> Dict[str, Any]:
    data = json.loads(
        path.read_text(encoding="utf-8")
    )

    if not isinstance(data, dict):
        raise ValueError(
            f"{path} must contain a JSON object"
        )

    return data


def resolve(root: Path, value: str) -> Path:
    path = Path(value)

    if path.is_absolute():
        return path

    return root / path


def new_check() -> Dict[str, Any]:
    return {
        "status": "PASS",
        "blocking_reasons": [],
        "warnings": [],
    }


def block(
    check: Dict[str, Any],
    message: str,
) -> None:
    check["status"] = "BLOCKED"
    check["blocking_reasons"].append(
        message
    )


def warn(
    check: Dict[str, Any],
    message: str,
) -> None:
    check["warnings"].append(
        message
    )


def parse_compose_environment(
    compose: Dict[str, Any],
    service_name: str,
) -> Dict[str, str]:
    services = compose.get("services", {})

    if not isinstance(services, dict):
        return {}

    service = services.get(
        service_name,
        {},
    )

    if not isinstance(service, dict):
        return {}

    raw_environment = service.get(
        "environment",
        {},
    )

    result: Dict[str, str] = {}

    if isinstance(
        raw_environment,
        list,
    ):
        for item in raw_environment:
            if (
                not isinstance(item, str)
                or "=" not in item
            ):
                continue

            key, value = item.split(
                "=",
                1,
            )

            result[
                key.strip()
            ] = value.strip()

    elif isinstance(
        raw_environment,
        dict,
    ):
        for key, value in raw_environment.items():
            result[str(key)] = (
                ""
                if value is None
                else str(value)
            )

    return result


def configured_default(
    environment: Dict[str, str],
    key: str,
) -> Optional[str]:
    if key not in environment:
        return None

    value = environment[key]

    match = ENV_DEFAULT_RE.fullmatch(
        value
    )

    if match is not None:
        return match.group("default")

    return value or None


def check_production_evidence(
    config: Dict[str, Any],
    root: Path = Path("."),
) -> Dict[str, Any]:
    root = root.resolve()

    checks = {
        "serving_identity":
            new_check(),

        "revision_pinning":
            new_check(),

        "risk_evaluation_evidence":
            new_check(),

        "risk_calibration_evidence":
            new_check(),

        "rewriter_evidence":
            new_check(),

        "heldout_evidence":
            new_check(),
    }

    provenance: Dict[str, Any] = {}

    serving_cfg = config.get(
        "serving",
        {},
    )

    risk_cfg = config.get(
        "risk_model",
        {},
    )

    rewriter_cfg = config.get(
        "rewriter",
        {},
    )

    heldout_cfg = config.get(
        "heldout",
        {},
    )

    # ---------------------------------------------------------
    # Serving identity
    # ---------------------------------------------------------

    serving_check = checks[
        "serving_identity"
    ]

    compose_path_value = serving_cfg.get(
        "compose_file"
    )

    compose = None
    environment: Dict[str, str] = {}

    if not compose_path_value:
        block(
            serving_check,
            "Serving compose file is not configured.",
        )

    else:
        compose_path = resolve(
            root,
            compose_path_value,
        )

        if not compose_path.exists():
            block(
                serving_check,
                "Serving compose file is missing: "
                f"{compose_path_value}",
            )

        else:
            compose = load_yaml(
                compose_path
            )

            environment = (
                parse_compose_environment(
                    compose,
                    serving_cfg.get(
                        "service",
                        "",
                    ),
                )
            )

    risk_id_key = serving_cfg.get(
        "risk_model_id_env"
    )

    rewriter_id_key = serving_cfg.get(
        "rewriter_model_id_env"
    )

    configured_risk_id = (
        configured_default(
            environment,
            risk_id_key,
        )
        if risk_id_key
        else None
    )

    configured_rewriter_id = (
        configured_default(
            environment,
            rewriter_id_key,
        )
        if rewriter_id_key
        else None
    )

    provenance[
        "served_risk_model_id"
    ] = configured_risk_id

    provenance[
        "served_rewriter_model_id"
    ] = configured_rewriter_id

    if (
        configured_risk_id
        != risk_cfg.get(
            "served_model_id"
        )
    ):
        block(
            serving_check,
            "Configured risk model does not match "
            "the production evidence contract: "
            f"{configured_risk_id!r} != "
            f"{risk_cfg.get('served_model_id')!r}",
        )

    if (
        configured_rewriter_id
        != rewriter_cfg.get(
            "served_model_id"
        )
    ):
        block(
            serving_check,
            "Configured rewriter does not match "
            "the production evidence contract: "
            f"{configured_rewriter_id!r} != "
            f"{rewriter_cfg.get('served_model_id')!r}",
        )

    # ---------------------------------------------------------
    # Revision pinning
    # ---------------------------------------------------------

    revision_check = checks[
        "revision_pinning"
    ]

    risk_revision_key = (
        serving_cfg.get(
            "risk_model_revision_env"
        )
    )

    rewriter_revision_key = (
        serving_cfg.get(
            "rewriter_model_revision_env"
        )
    )

    risk_revision = (
        configured_default(
            environment,
            risk_revision_key,
        )
        if risk_revision_key
        else None
    )

    rewriter_revision = (
        configured_default(
            environment,
            rewriter_revision_key,
        )
        if rewriter_revision_key
        else None
    )

    provenance[
        "served_risk_model_revision"
    ] = risk_revision

    provenance[
        "served_rewriter_revision"
    ] = rewriter_revision

    if (
        risk_cfg.get(
            "require_revision_pin"
        )
        and not risk_revision
    ):
        block(
            revision_check,
            "Served risk model revision is not pinned.",
        )

    if not rewriter_revision:
        block(
            revision_check,
            "Served rewriter revision is not pinned.",
        )

    # ---------------------------------------------------------
    # Risk evaluation evidence
    # ---------------------------------------------------------

    risk_eval_check = checks[
        "risk_evaluation_evidence"
    ]

    risk_eval = None

    risk_eval_path_value = (
        risk_cfg.get(
            "evaluation_artifact"
        )
    )

    if not risk_eval_path_value:
        block(
            risk_eval_check,
            "Risk evaluation artifact is not registered.",
        )

    else:
        risk_eval_path = resolve(
            root,
            risk_eval_path_value,
        )

        if not risk_eval_path.exists():
            block(
                risk_eval_check,
                "Risk evaluation artifact is missing: "
                f"{risk_eval_path_value}",
            )

        else:
            risk_eval = load_json(
                risk_eval_path
            )

    if risk_eval is not None:
        eval_model_id = (
            risk_eval.get(
                "model_id"
            )
        )

        eval_revision = (
            risk_eval.get(
                "revision"
            )
        )

        provenance[
            "risk_evaluation_model_id"
        ] = eval_model_id

        provenance[
            "risk_evaluation_revision"
        ] = eval_revision

        provenance[
            "risk_evaluation_split"
        ] = risk_eval.get(
            "split"
        )

        provenance[
            "risk_evaluation_macro"
        ] = (
            risk_eval.get(
                "metrics",
                {},
            ).get(
                "macro",
                {},
            )
        )

        if (
            eval_model_id
            != risk_cfg.get(
                "served_model_id"
            )
        ):
            block(
                risk_eval_check,
                "Risk evaluation artifact belongs "
                "to a different model ID.",
            )

        if not eval_revision:
            block(
                risk_eval_check,
                "Risk evaluation artifact does not "
                "record a model revision.",
            )

        if (
            risk_cfg.get(
                "require_revision_match"
            )
            and risk_revision
            and eval_revision
            and risk_revision != eval_revision
        ):
            block(
                risk_eval_check,
                "Served risk revision does not match "
                "the evaluated revision: "
                f"{risk_revision!r} != "
                f"{eval_revision!r}",
            )

    # ---------------------------------------------------------
    # Risk calibration evidence
    # ---------------------------------------------------------

    calibration_check = checks[
        "risk_calibration_evidence"
    ]

    if (
        risk_cfg.get(
            "require_usable_fitted_bands"
        )
    ):
        if risk_eval is None:
            block(
                calibration_check,
                "Risk calibration cannot be verified "
                "without an evaluation artifact.",
            )

        else:
            fitted = risk_eval.get(
                "fitted_bands_in_sample"
            )

            if not isinstance(
                fitted,
                dict,
            ):
                block(
                    calibration_check,
                    "Risk evaluation does not contain "
                    "fitted-band evidence.",
                )

            else:
                unusable = sorted(
                    attribute
                    for attribute, result
                    in fitted.items()
                    if (
                        not isinstance(
                            result,
                            dict,
                        )
                        or result.get(
                            "usable"
                        )
                        is not True
                    )
                )

                provenance[
                    "unusable_risk_bands"
                ] = unusable

                if unusable:
                    block(
                        calibration_check,
                        "Risk-band calibration is not "
                        "production-usable for: "
                        + ", ".join(
                            unusable
                        ),
                    )

    # ---------------------------------------------------------
    # Rewriter evidence
    # ---------------------------------------------------------

    rewriter_check = checks[
        "rewriter_evidence"
    ]

    historical_readiness_value = (
        rewriter_cfg.get(
            "historical_readiness_artifact"
        )
    )

    if not historical_readiness_value:
        block(
            rewriter_check,
            "Rewriter readiness evidence is not registered.",
        )

    else:
        readiness_path = resolve(
            root,
            historical_readiness_value,
        )

        if not readiness_path.exists():
            block(
                rewriter_check,
                "Rewriter readiness artifact is missing: "
                f"{historical_readiness_value}",
            )

        else:
            readiness = load_json(
                readiness_path
            )

            readiness_base = (
                readiness.get(
                    "provenance",
                    {},
                ).get(
                    "evaluation_target_base_model_id"
                )
            )

            provenance[
                "historical_rewriter_readiness_base_model"
            ] = readiness_base

            if (
                rewriter_cfg.get(
                    "require_readiness_base_model_match"
                )
                and readiness_base
                != rewriter_cfg.get(
                    "expected_base_model_id"
                )
            ):
                block(
                    rewriter_check,
                    "Historical readiness evidence "
                    "covers a different rewriter base model: "
                    f"{readiness_base!r} != "
                    f"{rewriter_cfg.get('expected_base_model_id')!r}",
                )

    rewriter_eval_value = (
        rewriter_cfg.get(
            "evaluation_artifact"
        )
    )

    if (
        rewriter_cfg.get(
            "require_evaluation_artifact"
        )
    ):
        if not rewriter_eval_value:
            block(
                rewriter_check,
                "No reportable evaluation artifact "
                "is registered for the served rewriter.",
            )

        else:
            rewriter_eval_path = resolve(
                root,
                rewriter_eval_value,
            )

            if not rewriter_eval_path.exists():
                block(
                    rewriter_check,
                    "Served-rewriter evaluation artifact "
                    "is missing: "
                    f"{rewriter_eval_value}",
                )

    # ---------------------------------------------------------
    # Held-out evidence
    # ---------------------------------------------------------

    heldout_check = checks[
        "heldout_evidence"
    ]

    heldout_value = heldout_cfg.get(
        "result_artifact"
    )

    if (
        heldout_cfg.get(
            "require_reportable_result"
        )
    ):
        if not heldout_value:
            block(
                heldout_check,
                "No reportable held-out production "
                "evaluation result is registered.",
            )

        else:
            heldout_path = resolve(
                root,
                heldout_value,
            )

            if not heldout_path.exists():
                block(
                    heldout_check,
                    "Held-out production evaluation "
                    "artifact is missing: "
                    f"{heldout_value}",
                )

    # ---------------------------------------------------------
    # Aggregate
    # ---------------------------------------------------------

    blocking_reasons: List[str] = []
    warnings: List[str] = []

    for name, check in checks.items():
        blocking_reasons.extend(
            f"{name}: {reason}"
            for reason
            in check[
                "blocking_reasons"
            ]
        )

        warnings.extend(
            f"{name}: {message}"
            for message
            in check["warnings"]
        )

    return {
        "schema_version":
            SCHEMA_VERSION,

        "release_id":
            config.get(
                "release_id"
            ),

        "overall_status":
            (
                "READY"
                if not blocking_reasons
                else "BLOCKED"
            ),

        "checks":
            checks,

        "provenance":
            provenance,

        "blocking_reasons":
            blocking_reasons,

        "warnings":
            warnings,
    }


def print_result(
    result: Dict[str, Any],
) -> None:
    print(
        "\n=== SESSION 10 PRODUCTION EVIDENCE ==="
    )

    for name, check in result[
        "checks"
    ].items():
        print(
            f"{name.replace('_', ' ').title()}: "
            f"{check['status']}"
        )

    print(
        "\nOVERALL:",
        result["overall_status"],
    )

    if result["blocking_reasons"]:
        print("\nBlocking reasons:")

        for reason in result[
            "blocking_reasons"
        ]:
            print("-", reason)

    if result["warnings"]:
        print("\nWarnings:")

        for message in result[
            "warnings"
        ]:
            print("-", message)


def main() -> int:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        default=(
            "configs/production/"
            "session10.yaml"
        ),
    )

    parser.add_argument(
        "--output",
        default=None,
    )

    args = parser.parse_args()

    root = Path(".").resolve()

    config = load_yaml(
        resolve(
            root,
            args.config,
        )
    )

    result = check_production_evidence(
        config,
        root=root,
    )

    print_result(
        result
    )

    output_value = (
        args.output
        or config.get(
            "output",
            {},
        ).get(
            "evidence_file"
        )
    )

    if output_value:
        output_path = resolve(
            root,
            output_value,
        )

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        output_path.write_text(
            json.dumps(
                result,
                indent=2,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )

        print(
            "\nWrote production evidence:",
            output_path,
        )

    return (
        0
        if result["overall_status"]
        == "READY"
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
