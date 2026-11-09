# Hosting the RoBERTa Risk Model on Hugging Face Hub

This guide details the procedure for packaging, verifying, and hosting the multi-task RoBERTa risk classifier on Hugging Face Hub.
The model predicts binary inferability across age, location, occupation, and education attributes.
Hosting on the Hub enables decoupled cloud serving, automatic caching, and reproducible production deployments.

## Prerequisites

Ensure all required libraries are installed before running deployment scripts:

```bash
pip install -r requirements.txt
huggingface-cli login
```

Set your Hugging Face write token as an environment variable:

```bash
export HF_TOKEN="your_huggingface_write_token"
```

## Option 1: Automated Packaging and Upload via Script

Use the built-in hosting utility to package weights and push them directly:

```bash
python scripts/host_roberta_on_hf.py \
    --repo-id "your-org/inferenceguard-roberta-risk" \
    --base-model "roberta-base" \
    --checkpoint-dir "artifacts/risk_model"
```

To create a private repository, append the `--private` flag.

## Option 2: Uploading Standalone Package Directory

The `hf_roberta_risk_model/` directory contains a standalone package ready for upload.
It includes `configuration_roberta_risk.py` and `modeling_roberta_risk.py` for remote code execution.

Run the standalone uploader:

```bash
cd hf_roberta_risk_model
python push_to_hub.py --repo-id "your-org/inferenceguard-roberta-risk"
```

## Loading the Model in Production

Once published, load the model directly in code:

```python
from transformers import AutoModel, AutoTokenizer

repo_id = "AT707/InferenceGuard-Roberta-Risk"
tokenizer = AutoTokenizer.from_pretrained(repo_id)
model = AutoModel.from_pretrained(repo_id, trust_remote_code=True)

inputs = tokenizer("Prompt text to evaluate", return_tensors="pt")
probabilities = model.predict_proba(**inputs)
print(probabilities)
```
