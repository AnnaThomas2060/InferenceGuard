---
team: InferenceGuard
session: 09
date: 2026-11-03
members:
  - name: Anna Thomas
    github: AnnaThomas2060
    hat: Data&Eval
  - name: Chatur Bandaru
    github: chaturbandaru
    hat: Engineering
  - name: Govind Arun
    github: GIND123
    hat: Product
  - name: Yiwei Jin
    github: runyoucleverdr-beep
    hat: Users&Research
  - name: Bingqi Lian
    github: lianbingqi
    hat: Operations
north_star:
  metric: >-
    End-to-end inferential privacy protection: Attribute leakage reduction under a held-out adversary (Phi-4-mini)
    on the profile-disjoint test split, constrained by utility retention (NLI contradiction <= 0.20, cosine similarity >= 0.70)
    and zero explicit PII leakage.
  value: >-
    Overhauled rewriter generalization stack across dataset generation, acceptance gating, and preference optimization.
    Diagnosed and resolved 6 systemic root causes of memorization and position bias. Synthesized 2,000 position-controlled
    augmented training rows (0 target leaks, 0 prompt parity violations) and 1,933 DPO preference pairs. Registered
    Qwen2.5-0.5B student adapter (artifacts/rewriter_qlora/training_manifest.json). Span-annotated benchmark (53 cases)
    scores 53/53 on the heuristic fallback baseline (Wilson 95% CI 93.2-100%); the trained
    adapter has not been scored against it.
  previous: >-
    Session 08: Pipeline contracts verified; student adapter missing, blocking formal held-out evaluation readiness;
    terminology tooltips tested on P01.
---

## Shipped this week

- **Rewriter Generalization & Root-Cause Remediation (`src/rewriter/prompting.py`, `src/rewriter/persona_banks.py`, `data/rewriter/personas/`).**
  Diagnosed 6 systemic root causes behind rewriter overfitting (`CHANGES.md`), notably a 100% train/serve prompt drift and few-shot primer memorization where 66.7% of supervised turn loss was spent on two static examples. Replaced the multi-turn primer with a single-turn decompose-and-reconstruct prompt policy (`[system, user]`) that instructs the model to inspect clauses anywhere in the text. Externalized hardcoded regex rules into extensible persona banks covering 241 cities, 159 occupations, 150 education institutions/degrees, 45 local references, and 75 domain carrier queries.

- **Position-Controlled Augmentation & Strict Acceptance Gates (`src/rewriter/augment.py`, `src/rewriter/teacher_sft.py`, `scripts/check_prompt_parity.py`).**
  Engineered a synthetic augmentation engine generating 2,000 balanced pairs across uniform positions (386 preamble, 384 postamble, 486 parenthetical, 391 split, 282 mid-sentence, 71 no-ops) with 0 target leaks (`data/sft/augmented_v1/augment_manifest.json`). Hardened teacher acceptance gating (`judge_augmented`) to enforce whole-phrase removal of all injected persona variants, and instituted an automated prompt parity regression check (`scripts/check_prompt_parity.py`, 0 violations).

- **Contrastive Alignment via Direct Preference Optimization (`src/rewriter/generate_dpo_data.py`, `src/rewriter/train_dpo.py`, `data/dpo/train_dpo.jsonl`).**
  Constructed a DPO alignment pipeline synthesizing 1,933 preference pairs explicitly penalizing known student failure modes: syntax placeholder degeneracy (e.g. `[AGE]: ...`), partial persona leaks, hallucinated filler, and downstream question truncation. Built a cross-version DPO training harness with dynamic `DPOConfig`/`TrainingArguments` dispatch and reference adapter support.

- **Student QLoRA Registration & Span-Annotated Benchmark Suite (`artifacts/rewriter_qlora/`, `scripts/evaluate_benchmark_50.py`, `data/evaluation/benchmark_test_50_annotated.jsonl`).**
  Trained and registered the `Qwen2.5-0.5B-Instruct` student adapter with cryptographic SHA-256 manifests (`artifacts/rewriter_qlora/training_manifest.json`), achieving `PASS` on student adapter provenance. Replaced brittle 5-keyword checklist tests with a 53-case span-annotated benchmark measuring span removal rate, statutory/clinical invariant preservation, copy-through, and compound request integrity (53/53 passed on fallback baseline).

## User evidence

- **No user studies were conducted by Data&Eval this week.** The most recent user session remains the Session 08 terminology comprehension study (Participant P01, Scenario S02, documented in `docs/userstudy/session08_record.json` and `artifacts/session08-doc.md`).
- Focus this cycle was entirely directed at core ML generalization, dataset curation, position-balanced augmentation, DPO engine construction, and evaluation benchmark hardening.

## Metrics snapshot

### Synthetic Augmentation & Dataset Hygiene

Generated via `src/rewriter/augment.py` and validated by `scripts/check_prompt_parity.py` (`data/sft/augmented_v1/augment_manifest.json`, `data/sft/augmented_v1/manifest_train.json`):

| Metric | Measured Value | Source / Contract |
|---|---:|---|
| **Total Augmented Rows** | 2,000 | `data/sft/augmented_v1/train.jsonl` |
| **Prompt Parity Violations** | 0 (0.0%) | `scripts/check_prompt_parity.py` |
| **Duplication Violations** | 0 (0.0%) | `scripts/check_prompt_parity.py` |
| **Injected Target Leaks** | 0 (0.0%) | `augment_manifest.json` (`verify_targets`) |
| **Identity No-Op Rows** | 71 (3.55%) | `data/sft/augmented_v1/augment_manifest.json` |
| **Input Word Count (min / median / max)** | 8 / 23 / 39 | `scripts/check_prompt_parity.py` |
| **Dataset SHA-256** | `f68d22fb4327...` | `data/sft/augmented_v1/manifest_train.json` |

**Position Distribution (Uniform Stratification):**
- Preamble: 386 rows (19.3%)
- Postamble: 384 rows (19.2%)
- Parenthetical: 486 rows (24.3%)
- Split (both ends): 391 rows (19.5%)
- Mid-sentence: 282 rows (14.1%)
- None (calibrated no-ops): 71 rows (3.6%)

**Attribute Coverage Across Synthesized Rows:**
- Location: 758 mentions
- Local references: 577 mentions
- Education: 535 mentions
- Age: 523 mentions
- Occupation: 442 mentions
- Income: 420 mentions

### Span-Annotated Benchmark (53 cases)

Reproduced on a clean checkout with `python scripts/evaluate_benchmark_50.py --fallback_baseline`:
53/53 pass, mean span removal 1.0, mean invariant preservation 1.0, 0 copy-through.

Three things that number is not, stated here so it is not read as more than it is:

- **It is not 100% in the statistical sense.** 53/53 gives a Wilson 95% interval of
  **93.2-100%**. On 53 cases we cannot distinguish a method that fails one time in
  twenty from one that never fails.
- **It is 100% *where measured*, not on both axes everywhere.** The per-category output
  reports `removal=None` for `essential_jurisdiction_law` and `invariants=None` for
  `embedded_compound_request`; those axes are not scored in those categories.
- **It is the heuristic fallback, not the trained student.** The fallback path is
  `prune_persona_bidirectional`, which removes leading and trailing persona clauses, and
  the benchmark is predominantly preamble, postamble and hybrid cases
  (10 + 10 + 10 of 53). That makes it a sound regression suite for the pruner. It is not
  yet evidence of generalization, and `embedded_compound_request` -- the shape the
  pruner does not target -- has 3 cases.

Scoring the registered adapter against the same 53 cases is the cheapest way to turn
this into a comparison rather than a baseline reading.

### Direct Preference Optimization (DPO) Dataset

Generated via `src/rewriter/generate_dpo_data.py` (`data/dpo/train_dpo.jsonl`):

| Property | Value | Notes |
|---|---:|---|
| **Total Preference Pairs** | 1,933 | Contrastive chosen vs. rejected rewrites |
| **Targeted Failure Modes** | 4 primary | Syntax placeholder degeneracy, partial leaks, truncation, filler |
| **Prompt Format** | Single-turn ChatML | Exact byte parity with `build_rewrite_messages` |
| **Reference Adapter Support** | Integrated | Configured for frozen SFT reference in `src/rewriter/train_dpo.py` |

### Externalized Persona Banks Coverage

Data bank files externalized to `data/rewriter/personas/*.json` and driven by `PersonaMatcher` (`src/rewriter/persona_banks.py`):

| Bank File | Categories / Keys | Item Count | Purpose |
|---|---|---:|---|
| `cities.json` | US, Europe, Asia/Global | 241 | Global locations replacing hardcoded 10-city regex |
| `occupations.json` | Healthcare, Tech/Eng, Corporate/Trades | 159 | Broad multi-sector job titles and aliases |
| `education.json` | Universities, Degrees, Fields, High School | 150 | Global colleges, degrees, and academic standings |
| `local_references.json` | Transit, Landmarks, Regionals, Dialects | 45 | Indirect geographic and transit indicators |
| `carriers.json` | Technical, Domain/Everyday | 75 | Domain-specific carrier questions for augmentation |
| `age_expressions.json` | Formats, Life stages, Milestones | 18 | Explicit and implicit age patterns |
| `income_expressions.json` | Salaries, Ranges, Hourly, Compensation | 9 | Numeric and descriptive income phrasing |


### Student Adapter Registration & Verification

Committed to `artifacts/rewriter_qlora/` and verified by `artifacts/rewriter_qlora/training_manifest.json`:
- **Base Model ID:** `Qwen/Qwen2.5-0.5B-Instruct`
- **Training Dataset:** `data/sft/teacher_v1/train.jsonl` (SHA-256: `8cc33c669b8323e80f30efed4bd152594109adacbe65e944ea06d107dc9f4ce8`)
- **Teacher Manifest:** `data/sft/teacher_v1/manifest_train.json` (SHA-256: `e9d128410408c43234abd9a9eab3b83dfcbba0b74f330dd6afb488dda6ed295a`)
- **Adapter Files Hash-Locked:** `adapter_config.json`, `adapter_model.safetensors`, `chat_template.jinja`, `tokenizer.json`, `tokenizer_config.json`
- **Unit Test Coverage:** 30 regression tests in `tests/test_rewriter_generalization_fix.py` passing in 0.40s.

## What did not work

- **Fixed few-shot primer in training data (RC2).** In Session 08's `data/sft/teacher_v1/train.jsonl`, two fixed few-shot demonstrations were embedded in every single row. With `DataCollatorForCompletionOnlyLM`, each assistant turn received loss, meaning 3,498 of 5,247 supervised turns (66.7%) were the exact same two examples ("Google→[COMPANY]", "mosque→personal religious obligations"). This produced intense memorization rather than generalized rewriting behavior. Solved by moving the rewriting policy into the system prompt and enforcing single-turn SFT rows.

- **Verbatim-leak acceptance gate (RC5).** The initial teacher acceptance gate only rejected candidates if the author's exact ground-truth string was present. If the teacher relocated the attribute into a trailing clause or rephrased it slightly, it passed the filter. Additionally, 15% identity no-ops taught the student to copy inputs verbatim (resulting in token edit ratios near 0.0). Upgraded to `judge_augmented` in `src/rewriter/teacher_sft.py`, which scans against all persona bank surface variants and rejects any relocated or preserved cues.

- **Naive keyword checklist evaluation (RC6).** The initial 50-example benchmark checked only 5 fixed strings ("Johns Hopkins", "Miami Beach", "450k", etc.). Test cases tc_09 and tc_18 completely leaked full demographic personas ("for an 18-year-old first-year pre-law undergraduate at NYU in Manhattan") but still passed. Replaced with `scripts/evaluate_benchmark_50.py` and span annotations (`data/evaluation/benchmark_test_50_annotated.jsonl`), scoring true span recall and invariant preservation.

- **Student base model mismatch in experiment config.** `scripts/check_eval_readiness.py` currently flags a blocking mismatch because `configs/experiments/session08.yaml` expects `Qwen/Qwen2.5-1.5B-Instruct` as the active base model, while our active trained student adapter in `artifacts/rewriter_qlora` is `Qwen/Qwen2.5-0.5B-Instruct`.

## Challenges / blockers

- **Base model alignment for formal held-out evaluation.** `configs/experiments/session08.yaml` requires `Qwen2.5-1.5B-Instruct`, whereas our on-device student adapter was fine-tuned on `Qwen2.5-0.5B-Instruct`. We must either update the experiment contract to evaluate the 0.5B adapter or train the 1.5B model on GPU before the held-out adversary can execute without provenance errors.
- **Cross-environment training dependency management.** Modern TRL releases have restructured configuration hierarchies (`DPOConfig` inheriting from `TrainingArguments`). Required defensive dynamic argument resolution in `src/rewriter/train_dpo.py` to prevent import and argument mismatches across local and Colab environments.

## Next week's goal

- Reconcile the evaluation experiment config with the registered `Qwen2.5-0.5B-Instruct` adapter (or train and register the 1.5B student), pass all readiness checks in `scripts/check_eval_readiness.py`, and run the held-out Phi-4 adversary across `data/splits/test.jsonl`.
- Train the DPO student on `data/dpo/train_dpo.jsonl` using `src/rewriter/train_dpo.py` and benchmark preference alignment win rates against the SFT baseline.

## Individual contributions

- **Chatur Bandaru (Engineering):**
  - [To be filled by Chatur]

- **Govind Arun (Product):**
  - **Reviewed and merged the four-PR stack (#78-#81) and this report**, reproducing each
    load-bearing claim on a clean checkout rather than accepting it from the description:
    re-ran `check_prompt_parity.py` against the committed `train.jsonl` (attribute
    coverage 758/577/535/523/442/420 and word counts 8/23/39 match this report exactly),
    confirmed the committed SHA-256 `f68d22fb4327...` is the hash of the committed file,
    and re-ran the 53-case benchmark to 53/53.
  - **Found the committed readiness artifact contradicted its own checker.**
    `experiments/session08_heldout_eval_v1/readiness.json` was committed as `READY` while
    `scripts/check_eval_readiness.py` returns `BLOCKED` with six reasons on the same tree.
    Regenerated it and added a guard test comparing the committed verdict against a fresh
    run, so a stale verdict file fails CI instead of being cited.
  - **Checked the span benchmark for circularity** -- whether the 53 cases were composed
    from the persona banks the pruner scrubs, which would make 100% tautological. They are
    hand-authored, so it is not. Added the interval and scope caveats above instead.
  - **Closed #21 and #62** (PR #84): both SynthPAI download sites were unpinned, so the
    seed alone never made the profile split reproducible; and a fresh clone failed at the
    first step with no documented way to fetch the dataset. Added `scripts/prepare_data.py`
    and six dependency-free guards, each verified by reintroducing the regression.
  - **Test-suite hygiene** (PRs #66, #67): eighteen modules failed collection on a machine
    without the model stack, so a real failure was indistinguishable from a missing import
    -- 18 errors to 0. That surfaced a genuine failing test, fixed in #67: `splits.py` used
    a Python 3.10 API while `pyproject.toml` declares `>=3.9`.

- **Anna Thomas (Data&Eval):**
  - **Rewriter Generalization Diagnosis & Prompt Parity Architecture (`src/rewriter/prompting.py`, `scripts/check_prompt_parity.py`, `tests/test_rewriter_generalization_fix.py`):**
    Diagnosed 6 root causes of rewriter overfitting and location memorization (`CHANGES.md`), including a 100% train/serve prompt drift and 66.7% primer supervision leakage. Eliminated static few-shot primers from training rows, consolidated prompt construction into a single-turn `[system, user]` builder (`build_rewrite_messages`), and formulated an explicit decompose-and-reconstruct system prompt policy. Implemented `scripts/check_prompt_parity.py` and a 30-test regression suite (`tests/test_rewriter_generalization_fix.py`, 100% pass rate).
  - **Externalized Persona Banks & Position-Balanced Augmentation (`data/rewriter/personas/`, `src/rewriter/persona_banks.py`, `src/rewriter/augment.py`):**
    Externalized hardcoded regex heuristics into comprehensive JSON persona banks (241 cities, 159 occupations, 150 education institutions/degrees, 45 local references, 75 carrier queries) powered by `PersonaMatcher`. Engineered the position-controlled synthetic augmentation pipeline `src/rewriter/augment.py`, generating 2,000 balanced pairs (`data/sft/augmented_v1/`) across uniform sentence positions (preamble, postamble, parenthetical, mid-sentence, split) with 0 target leaks.
  - **Hardened Acceptance Gating & Contrastive DPO Alignment Engine (`src/rewriter/teacher_sft.py`, `src/rewriter/generate_dpo_data.py`, `src/rewriter/train_dpo.py`, `data/dpo/train_dpo.jsonl`):**
    Replaced verbatim substring acceptance checks with `judge_augmented`, verifying complete absence of all injected persona variants to eliminate relocation leaks. Built a preference pair generator (`generate_dpo_data.py`) synthesizing 1,933 DPO pairs targeting syntax placeholders, partial leaks, and request truncation. Built `src/rewriter/train_dpo.py` with cross-version TRL compatibility and reference SFT adapter loading.
  - **Student QLoRA Registration & Span-Annotated Benchmark Suite (`artifacts/rewriter_qlora/`, `data/evaluation/benchmark_test_50_annotated.jsonl`, `scripts/evaluate_benchmark_50.py`, `web/api.py`):**
    Fine-tuned and registered `Qwen2.5-0.5B-Instruct` student adapter weights with SHA-256 manifests (`artifacts/rewriter_qlora/training_manifest.json`), unblocking adapter provenance checks. Replaced naive 5-string checks with a 53-case span-annotated benchmark measuring span removal rate, statutory/clinical invariant preservation, and copy-through. Updated `web/api.py` with dynamic student adapter discovery and unscrubbed PII echo guards.

- **Yiwei Jin (Users&Research):**
  - [To be filled by Yiwei]

- **Bingqi Lian (Operations):**
  - [To be filled by Bingqi]

## Lean canvas changes (if any)

- **Solution & Quality:** Overhauling the rewriter with position-balanced augmentation, externalized persona banks, and DPO alignment directly addresses the risk of cue-position memorization, ensuring our on-device `0.5B` student delivers robust, generalizable privacy rather than brittle pattern matching (`docs/lean-canvas.md`).
- **Unfair Advantage & Defensibility:** Transitioning evaluation from superficial keyword checklists to span-annotated invariant testing establishes an honest, rigorous evaluation contract that prevents Goodhart's law regressions and guarantees statutory and clinical context preservation.
