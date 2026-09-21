<p align="center"> <img src="img/inferential_privacy.png" width="120" /> </p> <h1 align="center">InferenceGuard: Local-First Privacy Layer for Inferential Privacy</h1> <p align="center"> <strong>explicit PII removal to risk estimation to privacy-preserving rewrite</strong> </p> <p align="center">
<a href="https://colab.research.google.com/drive/1l_hrmfaKqt69D_8QrEa3KCCYpPrWiG8y?usp=sharing" target="_blank"><img src="https://img.shields.io/badge/notebook-open%20in%20colab-F9AB00?style=flat&logo=googlecolab&logoColor=white" alt="Colab"></a> 
<img src="https://img.shields.io/badge/hardware-A100%20%7C%20T4-76B900?style=flat&logo=nvidia&logoColor=white" alt="Hardware"> <img src="https://img.shields.io/badge/dataset-SynthPAI%207.8k-blue?style=flat" alt="Dataset"> <img src="https://img.shields.io/badge/models-ModernBERT%20%7C%20Qwen3.5-blueviolet?style=flat" alt="Models"> </p> <p align="center"> <a href="#what-it-is">What</a> • <a href="#setup">Setup</a> • <a href="#run-it">Run It</a> </p> 

---

## What It Is

Most privacy tools only redact explicit PII like names, emails, and phone numbers. They miss inferential privacy, which is what a strong LLM can infer from ordinary clues.

> "My co-op ends in December, and I take the Green Line to campus most mornings."

There is no name or address here, but an attacker can still infer Boston from Green Line, undergraduate status from campus language, student occupation from co-op, and age range 18 to 24.

**InferenceGuard** is a local-first layer that runs before text leaves user control. It estimates how inferable attributes like Age, Location, Occupation, and Education are, highlights the cues that cause risk, and produces a rewritten version that reduces leakage while keeping intent. For example, Green Line becomes public transportation and co-op becomes work placement.

The core idea is to protect what text implies, not just what it explicitly says, and to handle multi-turn leakage where individually harmless messages accumulate over a conversation.

## Setup
 
**Dataset:** SynthPAI `RobinSta/SynthPAI` - 300 synthetic profiles, 103 threads, 7,823 comments, CC-BY-NC-SA-4.0  
**Models (to be implemented):** ModernBERT-base for risk, Qwen3.5-0.8B / 2B for rewriter, Phi-4-mini-instruct for held-out attacker, Presidio for explicit PII baseline  
**Key dependency:** `datasets`, `presidio-analyzer`, `presidio-anonymizer`, `spacy`, `transformers`, `peft`, `trl`, `scikit-learn`, `pandas`, `matplotlib`

Install once in Colab or local venv:

```bash
%pip install -q datasets huggingface_hub presidio-analyzer presidio-anonymizer spacy scikit-learn pandas matplotlib seaborn transformers torch peft trl
python -m spacy download en_core_web_lg
```

## Run It

1. Open the notebook via the Colab badge above
2. Set runtime to **Runtime > Change runtime type > A100 GPU** (T4 works for baselines)
3. Run all cells

