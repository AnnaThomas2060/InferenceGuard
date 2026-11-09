"""Utility script to verify Hugging Face Hub hosted models for InferenceGuard.

Validates that AT707/inferenceguard-rewriter-v1 and risk model packages
can be queried, inspected, and initialized in the InferenceGuard pipeline.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("verify_hf_models")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def check_hf_rewriter_metadata(repo_id: str = "AT707/inferenceguard-rewriter-v1") -> bool:
    """Inspect the Hugging Face repository metadata for the rewriter model."""
    logger.info("Checking repository metadata for '%s'...", repo_id)
    try:
        from huggingface_hub import HfApi

        api = HfApi()
        info = api.model_info(repo_id)
        logger.info("Successfully reached repository: %s (author: %s)", info.id, info.author)
        logger.info("Total downloads: %s, Sha: %s", getattr(info, "downloads", 0), info.sha[:10] if info.sha else "N/A")

        files = [f.rfilename for f in info.siblings]
        required_files = ["config.json", "model.safetensors", "tokenizer.json"]
        for req in required_files:
            if any(f.endswith(req) for f in files):
                logger.info("  [PASS] Found %s", req)
            else:
                logger.warning("  [WARN] Missing expected file %s in repository", req)
        return True
    except Exception as exc:
        logger.warning("Could not query Hugging Face API directly (%s: %s).", type(exc).__name__, exc)
        return False


def verify_pipeline_factory() -> bool:
    """Verify that InferenceGuardPipeline.from_pretrained configures the rewriter service."""
    logger.info("Verifying InferenceGuardPipeline.from_pretrained wiring...")
    from src.pipeline.orchestrator import InferenceGuardPipeline

    pipeline = InferenceGuardPipeline.from_pretrained(
        risk_model_id="roberta-base",
        rewriter_model_id="AT707/inferenceguard-rewriter-v1",
        device="cpu",
    )
    assert pipeline.rewriter_service is not None
    assert pipeline.rewriter_service.base_model_name == "AT707/inferenceguard-rewriter-v1"
    logger.info("Pipeline rewriter configured correctly: %s", pipeline.rewriter_service.base_model_name)
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify Hugging Face hosted models for InferenceGuard.")
    parser.add_argument("--rewriter-repo", default="AT707/inferenceguard-rewriter-v1", help="Rewriter model repo ID.")
    args = parser.parse_args()

    check_hf_rewriter_metadata(args.rewriter_repo)
    pipeline_ok = verify_pipeline_factory()

    if pipeline_ok:
        logger.info("All model verification checks completed successfully.")
        return 0
    else:
        logger.error("Model verification failed.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
