---
language:
- en
license: apache-2.0
tags:
- privacy
- anonymization
- causal-lm
- qwen2
- text-generation
- inferenceguard
base_model: Qwen/Qwen2.5-1.5B-Instruct
pipeline_tag: text-generation
---

# InferenceGuard Rewriter v1 (`inferenceguard-rewriter-v1`)

`AT707/inferenceguard-rewriter-v1` is a fine-tuned causal language model based on [`Qwen/Qwen2.5-1.5B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) specialized for **inferential privacy protection and text rewriting**.

Developed as the core generative component of the **InferenceGuard** multi-stage privacy engineering framework, this model rewrites user prompts to eliminate subtle linguistic cues and demographic leakage across 7 sensitive attribute dimensions while maintaining downstream semantic utility and conversational naturalness.

---

## Model Details

- **Base Architecture**: Qwen2 (`Qwen2ForCausalLM`)
- **Base Model**: `Qwen/Qwen2.5-1.5B-Instruct`
- **Parameter Count**: 1.54 Billion (1,543,714,816 parameters)
- **Vocabulary Size**: 151,936
- **Hidden Size**: 1,536
- **Layers**: 28 transformer blocks
- **Attention Heads**: 12 (query) / 2 (key-value)
- **Format**: Hugging Face `safetensors` (float16 / bfloat16)
- **Context Length**: 32,768 tokens (eval/rewrite target capped at 512 tokens)

---

## Protected Sensitive Dimensions

The model mitigates inferential leakage across seven demographic and personal dimensions:
1. **Age**: Colloquial generational markers, milestone years, retirement phrasing.
2. **Gender**: Gendered hobbies, stylistic traits, or stereotypical expressions.
3. **Location / Geographic Locale**: Regional slang, transit references, local institutions.
4. **Education Level**: Degree abbreviations, campus idioms, collegiate jargon.
5. **Occupation**: Role-specific technical jargon, workplace hierarchy indicators.
6. **Income Bracket**: Wealth status markers, spending habits, financial cues.
7. **Relationship Status**: Spousal references, dating markers, family dynamics.

---

## Intended Use

- **Agentic Pre-processing**: Sanitizing prompts before transmitting them to third-party commercial LLM APIs.
- **Enterprise Data Protection**: Stripping inferential metadata from customer support queries and internal documents.
- **Academic Research**: Benchmarking generative defense mechanisms against inferential privacy adversaries.

---

## Quickstart & Inference

### 1. Installation

```bash
pip install torch transformers accelerate
```

### 2. Python Inference Snippet

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "AT707/inferenceguard-rewriter-v1"

tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
    device_map="auto",
    trust_remote_code=True,
)

SYSTEM_PROMPT = (
    "You are a privacy-preserving text rewriter. Your task is to rewrite the input text "
    "to remove or generalize implicit personal cues (such as age, gender, location, education, "
    "occupation, income, and relationship status) while strictly preserving the user's core intent, "
    "meaning, and tone. Maintain any explicit pseudonym tokens verbatim. Return ONLY the rewritten text."
)

input_text = "I'm a junior doctor working 80-hour shifts at the hospital in downtown Chicago."

messages = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user", "content": input_text},
]

prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=128,
        do_sample=False,
        repetition_penalty=1.12,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
    )

gen_ids = outputs[0][inputs["input_ids"].shape[1]:]
rewritten_text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

print("Original :", input_text)
print("Rewritten:", rewritten_text)
```

---

## Integration in InferenceGuard Pipeline

InferenceGuard loads this checkpoint via `InferenceGuardPipeline.from_pretrained()`:

```python
from src.pipeline.orchestrator import InferenceGuardPipeline

pipeline = InferenceGuardPipeline.from_pretrained(
    risk_model_id="roberta-base",
    rewriter_model_id="AT707/inferenceguard-rewriter-v1",
    device="cpu",
)

response = pipeline.process_turn(
    conversation_id="session_01",
    message="I just moved to Austin after getting my master's in CS.",
)
print("Sanitized prompt:", response.protected_text)
```

---

## Training Methodology

- **Distillation Strategy**: Distilled from multi-stage teacher candidate rankings scored under dual criteria:
  1. *Privacy Gain*: Empirical drop in adversarial classifier accuracy across target attributes.
  2. *Utility Preservation*: Bidirectional NLI semantic consistency evaluated by DeBERTa-v3 cross-encoder.
- **Fine-Tuning Objective**: Completion-only cross-entropy loss masking prompt tokens (`DataCollatorForCompletionOnlyLM`).
- **Quantization & PEFT**: 4-bit NF4 double-quantization during initial exploration, exported to standalone merged full-weight checkpoint.
