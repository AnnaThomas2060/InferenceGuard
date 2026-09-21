---
team: InferenceGuard
session: 04
date: 2026-09-21
members:
  - name: Anna Thomas
    github: AnnaThomas2060
    hat: Data&Eval
  - name: 
    github: 
    hat: Engineering
  - name:
    github: 
    hat: Product
  - name:
    github: 
    hat: Users&Research
north_star:
  metric: location inference risk delta (before_loc - after_loc) + utility preservation (cosine sim)
  value: 0.559 delta example (0.749 -> 0.19) in vertical slice / avg ~0.54 on test phrases
  previous: 0.09 delta with Presidio-only baseline (Session 03)
---

## Shipped this week
- **Dataset foundation + taxonomy frozen**: Loaded `RobinSta/SynthPAI` (300 profiles, 103 threads, 7,823 comments), robust parsing for dict / JSON string / double-stringified `profile` field, audited distributions, frozen coarse taxonomy to `artifacts/taxonomy.json` — age [18-24,25-34,35-44,45-54,55+], education [high_school, undergraduate, graduate_professional, other_unknown], occupation [student, technology, healthcare, education, business_finance, service, other], location [USA_Northeast, USA_South, USA_Midwest, USA_West, Europe, Other] (evidence: `artifacts/taxonomy.json`, lines 204-212 in `inferenceguard_implementation.py`)

- Profile-disjoint splits: Deterministic split by author/username to prevent author-style memorization — 70% train / 15% val / 15% test, seed 42, saved to `artifacts/profile_splits.json` with ~300 unique profiles verified. 

- Explicit PII baseline: Integrated Presidio Analyzer + Anonymizer, demonstrated gap: `My co-op ends in December, and I take the Green Line to campus` -> 0 entities detected despite high inferential risk.

- Risk estimator v0: `MoritzLaurer/ModernBERT-base-zeroshot-v2.0` zero-shot + `dbmdz/bert-large-cased-finetuned-conll03-english` NER. 5 dimensions: location, education, occupation, age, name. Implements `detect_entities()`, `modernbert_predict()` returning `confidence=max(risk_scores)`, `evidence`, `has_hard_leak` (PER/LOC/ORG)

- Privacy-preserving rewriter v0: `Qwen/Qwen2.5-0.5B-Instruct` with privacy system prompt + regex `_hard_scrub()` safety net + `rewrite_until_safe(threshold=0.4, max_attempts=3)` that always rewrites from ORIGINAL and picks best by `(has_hard_leak, confidence)`

- End-to-end vertical slice: `text -> Presidio -> risk estimation -> rewrite -> re-scoring` with structured JSON output `{original, protected, before_loc, after_loc, delta, util_sim, evidence_before, evidence_after}` and live interactive loop logging to `timestamped_usage_log.csv` using `all-MiniLM-L6-v2` cosine for utility.

## User evidence
- What we have: Interactive CLI test loop (`input("You: ")`) that captures before/after risk and protected text. No formal user study with external users yet.
- Raw artifact: `timestamped_usage_log.csv` (generated at runtime, contains timestamp, original, protected, before_loc, after_loc, delta, util_sim, evidence). An example log is committed to the repo e.g. `data/logs/session04_usage_log_2026-09-21.csv`.
- What we discovered - 
  - The current state of the model needs to be improved if the utility of the user sentence needs to be preserved.
  - Explicit PII only checks PER/LOC/ORG, misses EMAIL, PHONE, ACCOUNT, MRN, MONEY as seen in live tests.

- What we plan to improve - 
  - Model: Fine-tune ModernBERT multi-task with temp scaling + ECE, use Integrated Gradients for cues.
  - Pipeline: Two-stage : Presidio for explicit PII + inferential rewriter, QLoRA Qwen 0.8B with 2B fallback.

## Metrics snapshot
- Location risk (zero-shot): Before 0.88 -> After 0.34 on canonical example `Green Line / co-op` (delta 0.54) (was 0.88 with Presidio-only)
- Example vertical slice: Before_loc 0.749 -> After_loc 0.19, delta 0.559, util_sim 0.82 (from docstring lines 602-613)
- Explicit PII baseline ASR reduction: ~9% — confirms need for inferential protection (Observation 4)
- Conversation accumulation: Turn 1 risk 0.24 -> Turn 4 risk 0.72 with simple heuristic (Observation 5)
- Measured on: The reduction in inference risk measured as before minus after for location, plus semantic similarity between original and protected text to ensure meaning is preserv
- Is this the same model that is running in the product? No, Current risk model is `ModernBERT-base-zeroshot-v2.0` proxy; final will be fine-tuned ModernBERT-base multi-task classifier with temperature scaling/ECE. Rewriter is base Qwen2.5-0.5B-Instruct, not yet QLoRA fine-tuned. Held-out adversary `Phi-4-mini-instruct` not yet implemented.

## What did not work
- Presidio detects zero spans on high-risk inferential utterances like "I take the Green Line every morning" (location risk 0.88) — traditional PII tooling insufficient.
- Qwen2.5-0.5B leaks original entities despite system prompt; required regex scrub net and `scrub_entities` union from original detection. Model too small for reliable generalization, may need fallback to 2B.
- Coarse taxonomy loses nuance — e.g. "Bachelor of Science in Computer Science, Northeastern University" -> undergraduate bucket discards institution signal that matters for privacy.
- No calibration yet, zero-shot scores not ECE measured, thresholds 0.4 / LOW 0-0.29 MEDIUM 0.3-0.59 HIGH 0.6-1.0 are uncalibrated.
- Profile parsing brittle, HF loader returns mixed types (dict, JSON string, Python repr); initial `attr_df` extraction failed and only returned `_author`.


## Challenges / blockers
- Need GPU for ModernBERT + Qwen pipeline + SentenceTransformer together; local inference memory pressure
- Need SFT data generation from train-only profiles (cannot use test profiles) scored by privacy reduction + utility — methodology for filtering synthetic pairs not defined
- Held-out evaluation with Phi-4-mini-instruct and utility metrics (DeBERTa-v3 NLI cross-encoder) not yet implemented
- Multi-turn leakage history UI and stable pseudonym mapping not started
- User studies must use synthetic scenarios only, without logging real sensitive messages — IRB/safety process unclear

## Next week's goal
- Train ModernBERT-base 4-head classifier on train_p with calibration (temperature scaling + reliability diagram + ECE) and implement Integrated Gradients cue attribution to replace token-masking baseline. Generate first SFT dataset for QLoRA on Qwen3.5-0.8B.

## Individual contributions
- Anna Thomas (Data&Eval):
    - Parsed SynthPAI (handled dict/JSON-string profile field), built DataFrame audit, finalized and froze coarse taxonomy for 4 aspects: age bucketing, education, occupation, location regions.
    - Implemented risk baseline: `RISK_DIMENSIONS` with ModernBERT zero-shot, `risk_scores` for 5 dims, `confidence = max(values)`, hard leak check PER/LOC/ORG, evidence extraction.
    - Implemented rewriter baseline: Qwen2.5-0.5B `qwen_rewrite` + `rewrite_until_safe` with safety key `(has_hard_leak, confidence)`, threshold 0.4, 3 attempts, and end-to-end pipeline `text -> PII -> risk -> rewrite -> rescore`.

- <add_name> (Engineering): 
-  <add_name> (Product | Engineering): 
-  <add_name> (Users&Research): 

## Lean canvas changes (if any)
- User segment: No change. We continue to focus on privacy-conscious users who share personal context when interacting with LLMs.
- Problem and value proposition: Problem is refined. Explicit PII redaction is not enough because inferential leakage happens through ordinary clues that Presidio misses, for example Green Line implies Boston location with zero explicit entities. Risk also accumulates across turns from 0.24 at turn 1 to 0.72 by turn 4. Value proposition is now clearer as a local-first system that runs ModernBERT and Qwen 0.5B on device, highlights risk cues, rewrites text, and quantifies the privacy utility tradeoff with risk delta and utility similarity.
- Risks and mitigation: Two new risks added. First, SynthPAI is synthetic so results may not transfer to real populations. Second, Qwen 0.5B has quality limits and leaks entities, requiring our _hard_scrub fallback. Mitigation is to consider a 2B fallback model if quality remains low.

