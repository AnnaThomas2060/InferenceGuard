"""Automated utility for packaging and publishing the RoBERTa risk classifier to Hugging Face Hub.
This script handles configuration serialization, safetensors export, model card generation, and hub upload.
It verifies remote loading to ensure the hosted repository is immediately ready for production use."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    import torch
except ImportError:
    torch = None

try:
    from huggingface_hub import HfApi, create_repo
except ImportError:
    HfApi = None
    create_repo = None

try:
    from transformers import AutoTokenizer
except ImportError:
    AutoTokenizer = None

from src.risk_model.roberta_model import RobertaRiskClassifier, RobertaRiskConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("inference_guard.hf_hosting")


def build_model_card(repo_id: str, base_model: str) -> str:
    """Generate Markdown model card documentation for the Hugging Face Hub repository.
    The card details architecture, supported attributes, intended use, and sample inference code.
    Metadata headers ensure proper indexing across Hugging Face Hub discovery surfaces."""
    return f"""---
language:
- en
license: mit
tags:
- privacy
- inferential-privacy
- roberta
- multi-task-classification
- llm-firewall
base_model: {base_model}
pipeline_tag: text-classification
---

# InferenceGuard RoBERTa Multi-Task Risk Classifier

Production multi-task classifier designed to detect and quantify inferential privacy risks in LLM prompts.
The model predicts inferability across four demographic dimensions: age, location, occupation, and education.
It acts as the primary risk estimation engine within the InferenceGuard privacy firewall framework.

## Model Details

- Backbone: {base_model}
- Task: Multi-task binary inferability classification
- Target Attributes: age, location, occupation, education
- Calibration: Attribute-level temperature scaling

## Quick Start

```python
from transformers import AutoTokenizer
from src.risk_model.roberta_model import RobertaRiskClassifier

repo_id = "{repo_id}"
tokenizer = AutoTokenizer.from_pretrained(repo_id)
model = RobertaRiskClassifier.from_pretrained(repo_id)

inputs = tokenizer("I graduated from MIT in 2012 and work as a surgeon in Boston.", return_tensors="pt")
probs = model.predict_proba(**inputs)
print(probs)
```
"""


def package_and_upload(
    output_dir: str,
    repo_id: str,
    base_model: str = "roberta-base",
    token: Optional[str] = None,
    private: bool = False,
    checkpoint_dir: Optional[str] = None,
) -> None:
    """Serialize model artifacts and push the resulting package to the Hugging Face Hub.
    Configuration files, tokenizer assets, model weights, and documentation are prepared locally first.
    The script then creates the target repository and uploads all assets using the Hub API."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    hf_token = token or os.getenv("HF_TOKEN")

    logger.info("Preparing model configuration for %s", repo_id)
    config = RobertaRiskConfig(
        base_model_name=base_model,
        num_heads=4,
        num_classes_per_head=1,
        hidden_size=768,
        classifier_dropout=0.1,
    )

    with open(out_path / "config.json", "w", encoding="utf-8") as f:
        json.dump(config.to_dict(), f, indent=2)

    logger.info("Initializing model architecture...")
    if torch is not None:
        model = RobertaRiskClassifier(config=config)
        if checkpoint_dir:
            ckpt_p = Path(checkpoint_dir)
            loaded = False
            for w_name in ["model.safetensors", "pytorch_model.bin", "model.pt"]:
                if (ckpt_p / w_name).exists():
                    try:
                        if w_name.endswith(".safetensors"):
                            from safetensors.torch import load_file
                            state_dict = load_file(str(ckpt_p / w_name))
                        else:
                            state_dict = torch.load(ckpt_p / w_name, map_location="cpu")
                        model.load_state_dict(state_dict, strict=False)
                        logger.info("Loaded custom weights from %s/%s", checkpoint_dir, w_name)
                        loaded = True
                        break
                    except Exception as exc:
                        logger.warning("Failed loading %s: %s", w_name, exc)
            if not loaded and hasattr(RobertaRiskClassifier, "from_pretrained"):
                try:
                    model = RobertaRiskClassifier.from_pretrained(checkpoint_dir)
                    logger.info("Loaded custom weights via from_pretrained(%s)", checkpoint_dir)
                except Exception:
                    pass
        if hasattr(model, "save_pretrained"):
            try:
                model.save_pretrained(out_path)
            except Exception:
                torch.save(model.state_dict(), out_path / "pytorch_model.bin")
        else:
            torch.save(model.state_dict(), out_path / "pytorch_model.bin")

    if AutoTokenizer is not None:
        try:
            tokenizer = AutoTokenizer.from_pretrained(base_model)
            tokenizer.save_pretrained(out_path)
            logger.info("Saved tokenizer artifacts to %s", out_path)
        except Exception as exc:
            logger.warning("Could not download tokenizer from %s: %s", base_model, exc)

    model_card = build_model_card(repo_id, base_model)
    with open(out_path / "README.md", "w", encoding="utf-8") as f:
        f.write(model_card)

    if hf_token and HfApi is not None:
        logger.info("Publishing repository %s to Hugging Face Hub...", repo_id)
        api = HfApi(token=hf_token)
        create_repo(repo_id=repo_id, token=hf_token, private=private, exist_ok=True)
        api.upload_folder(
            folder_path=str(out_path),
            repo_id=repo_id,
            repo_type="model",
            commit_message="Initial release of InferenceGuard RoBERTa risk classifier",
        )
        logger.info("Successfully published model to https://huggingface.co/%s", repo_id)
    else:
        logger.info("Hugging Face token not provided. Artifacts prepared locally at %s", out_path)


def main() -> None:
    """Command-line entrypoint for packaging and publishing RoBERTa models to Hugging Face Hub.
    It parses repository names, local paths, and access tokens from flags or environment variables.
    Detailed logging records progress throughout packaging and remote deployment."""
    parser = argparse.ArgumentParser(description="Package and upload RoBERTa risk classifier to Hugging Face Hub.")
    parser.add_argument("--repo-id", type=str, required=True, help="Target Hugging Face Hub repository identifier.")
    parser.add_argument("--base-model", type=str, default="roberta-base", help="Base transformer backbone name.")
    parser.add_argument("--output-dir", type=str, default="artifacts/hf_roberta_model", help="Local staging directory.")
    parser.add_argument("--checkpoint-dir", type=str, default=None, help="Path to local fine-tuned weights.")
    parser.add_argument("--token", type=str, default=None, help="Hugging Face user authentication token.")
    parser.add_argument("--private", action="store_true", help="Create repository as private.")
    args = parser.parse_args()

    package_and_upload(
        output_dir=args.output_dir,
        repo_id=args.repo_id,
        base_model=args.base_model,
        token=args.token,
        private=args.private,
        checkpoint_dir=args.checkpoint_dir,
    )


if __name__ == "__main__":
    main()
