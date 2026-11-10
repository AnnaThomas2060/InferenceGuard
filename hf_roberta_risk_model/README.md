---
language:
- en
license: mit
tags:
- privacy
- inferential-privacy
- roberta
- multi-task-classification
- llm-firewall
base_model: roberta-base
pipeline_tag: text-classification
---

# InferenceGuard RoBERTa Multi-Task Risk Classifier

Production multi-task classifier designed to detect and quantify inferential privacy risks in LLM prompts.
The model predicts inferability across four demographic dimensions: age, location, occupation, and education.
It acts as the primary risk estimation engine within the InferenceGuard privacy firewall framework.

## Model Details

- Backbone: roberta-base
- Task: Multi-task binary inferability classification
- Target Attributes: age, location, occupation, education
- Calibration: Attribute-level temperature scaling

## Quick Start

```python
from transformers import AutoModel, AutoTokenizer

repo_id = "your-username/inferenceguard-roberta-risk"
tokenizer = AutoTokenizer.from_pretrained(repo_id)
model = AutoModel.from_pretrained(repo_id, trust_remote_code=True)

inputs = tokenizer("I work as a software engineer in Seattle and graduated in 2018.", return_tensors="pt")
probs = model.predict_proba(**inputs)
print(probs)
```
