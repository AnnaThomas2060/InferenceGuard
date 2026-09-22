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
  value: 0.852 delta (location risk 0.859 -> 0.006) with util_sim 0.907 (from dev-test logs)
  previous: n/a — first measurement
---

## Shipped this week
- **Dataset foundation + taxonomy frozen**: Loaded `RobinSta/SynthPAI` (300 profiles, 103 threads, 7,823 comments), robust parsing for dict / JSON string / double-stringified `profile` field, audited distributions, frozen coarse taxonomy to `artifacts/taxonomy.json` — age [18-24,25-34,35-44,45-54,55+,unknown], education [high_school, undergraduate, graduate_professional, other_unknown], occupation [student, technology, healthcare, education, business_finance, service, other], location [USA_Northeast, USA_South, USA_Midwest, USA_West, Europe, Other, unknown] (evidence: `artifacts/taxonomy.json`, lines 204-212 in `InferenceGuard_Implementation.ipynb`)

- Profile-disjoint splits: Deterministic split by author/username to prevent author-style memorization — 70% train / 15% val / 15% test, seed 42, saved to `artifacts/profile_splits.json` with ~300 unique profiles verified. 

- Explicit PII baseline: Integrated Presidio Analyzer + Anonymizer. Demonstrated gap: On the phrase "My co-op ends in December, and I take the Green Line to campus", Presidio correctly caught explicit PII (detecting "December" as DATE_TIME and "Green Line" as LOCATION), but missed the inferred data we are trying to solve (e.g., "co-op" implying student status, or the combination implying Boston/undergraduate).

- Risk estimator v0: `MoritzLaurer/ModernBERT-base-zeroshot-v2.0` zero-shot + `dbmdz/bert-large-cased-finetuned-conll03-english` NER. 5 dimensions: location, education, occupation, age, name. Implements `detect_entities()`, `modernbert_predict()` returning `confidence=max(risk_scores)`, `evidence`, `has_hard_leak` (PER/LOC/ORG)

- Privacy-preserving rewriter v0: `Qwen/Qwen2.5-0.5B-Instruct` with privacy system prompt + regex `_hard_scrub()` safety net + `rewrite_until_safe(threshold=0.4, max_attempts=3)` that always rewrites from ORIGINAL and picks best by `(has_hard_leak, confidence)`

- End-to-end vertical slice: `text -> Presidio -> risk estimation -> rewrite -> re-scoring` with structured JSON output `{original, protected, before_loc, after_loc, delta, util_sim, evidence_before, evidence_after}` and live interactive loop logging to `timestamped_usage_log.csv` using `all-MiniLM-L6-v2` cosine for utility.

## User evidence
- What we have: Interactive CLI test loop (`input("You: ")`) that captures before/after risk and protected text. 
- Raw artifact: `timestamped_usage_log.csv` (generated at runtime, contains timestamp, original, protected, before_loc, after_loc, delta, util_sim, evidence). An example log is committed to the repo e.g. `data/logs/session04_usage_log_2026-09-21.csv`.
- What we discovered - 
  - The models used for the rewriter and risk estimator are currently zero-shot prompted and not yet finetuned. Based on the responses in the user test logs, we observe that finetuning is heavily required. For instance, the v0 model over-scrubs terms like "Type 2 diabetes" to "an entity diabetes", and occasionally leaks PII in complex resumes.
  - Explicit PII only checks PER/LOC/ORG, misses EMAIL, PHONE, ACCOUNT, MRN, MONEY as seen in live tests.

- What we plan to improve - 
  - Model: Fine-tune ModernBERT multi-task with temp scaling + ECE, use Integrated Gradients for cues.
  - Pipeline: Two-stage : Presidio for explicit PII + inferential rewriter, QLoRA Qwen 0.8B with 2B fallback.

## Metrics snapshot
- Location risk (zero-shot): Location risk 0.859 -> 0.006 (delta 0.852) on resume dev-test log.
- Example vertical slice: Confidence 0.997 -> 0.987, util_sim 0.907 (from `data/logs/session04_usage_log.csv`).
- Measured on: The reduction in inference risk measured as before minus after for location, plus semantic similarity between original and protected text to ensure meaning is preserv
- Is this the same model that is running in the product? No, Current risk model is `ModernBERT-base-zeroshot-v2.0` proxy; final will be fine-tuned ModernBERT-base multi-task classifier with temperature scaling/ECE. Rewriter is base Qwen2.5-0.5B-Instruct, not yet QLoRA fine-tuned. Held-out adversary `Phi-4-mini-instruct` not yet implemented.

## What did not work
- Traditional PII tooling is insufficient for inferred data: Presidio successfully catches explicit entities like dates and locations, but leaves inferential cues like "co-op" and "campus" untouched, which are what reveal student status, education, and age range.
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
- Problem and value proposition: Problem is refined. Explicit PII redaction is not enough because inferential leakage happens through ordinary clues that Presidio misses, for example Green Line implies Boston location even when explicit PII is scrubbed. Value proposition is now clearer as a local-first system that runs ModernBERT and Qwen 0.5B on device, highlights risk cues, rewrites text, and quantifies the privacy utility tradeoff with risk delta and utility similarity.
- Risks and mitigation: Two new risks added. First, SynthPAI is synthetic so results may not transfer to real populations. Second, Qwen 0.5B has quality limits and leaks entities, requiring our _hard_scrub fallback. Mitigation is to consider a 2B fallback model if quality remains low.

