# Improvements backlog: observed, not yet implemented

**Owner:** Anna Thomas (Data&Eval) · **Branch:** `improvements/modernbert-addition` · **Status:** descriptive only

> **These items are NOT implemented.** They are improvements noticed while reviewing
> the code and docs during the ModernBERT addition work. **No source, notebook, or
> config has been changed to address any of them.** Each one requires your explicit
> approval before any work begins. Per the request: *"list them out and give them to
> me ... and you may work on them once I approve it."*

Every item below cites a real file, line, or document that exists in this repository.
No benchmark numbers were invented. This review is static only: the `INTEGRATIONS_ONLY`
sandbox blocks `pip`/PyPI and `download.pytorch.org`, so `torch`/`transformers`/etc. are
not importable and the `pytest` suite could not be run here. Items that say "once the
suite can run" depend on that.

## How to read this

- **Effort:** S (a few hours), M (about a day), L (multi-day / needs a trained model or run).
- **Priority:** P0 (do first: correctness, safety, or blocks the thesis number), P1 (should do), P2 (nice to have / cosmetic).
- **Status:** every item is *pending approval / not implemented*.

## Priority summary

| # | Title | Effort | Priority |
|---|---|---|---|
| 1 | Run the held-out adversary and produce the north-star leakage number | L | P0 |
| 2 | Commit or publish trained model artifacts with checksums | M | P0 |
| 3 | Session lifecycle & ops gaps: `DELETE /session/{id}`, TTL, CORS default | M | P0 |
| 4 | Reconcile risk-estimator label source (profile attrs vs `reviews.human`) | M | P1 |
| 5 | Consolidate the two calibration modules (`calibrate.py` vs `calibration.py`) | M | P1 |
| 6 | Resolve the dead `prompt_baseline.py` module + stale model id | S | P1 |
| 7 | Fix location / education taxonomy mismatch vs the SynthPAI paper | L | P1 |
| 8 | Multi-turn attack curve (1/2/4/8 turns) + complete final comparison matrix | L | P1 |
| 9 | Add the PromptPAI secondary eval set (design requirement 5) | M | P2 |
| 10 | Purge stale `Qwen3.5-0.8B` references from the implementation notebook | S | P2 |
| 11 | Full refresh of the README "Run It" / Colab narrative | S | P2 |

---

## P0 (do first)

### 1. Run the held-out adversary and produce the north-star leakage number
- **Evidence:** `docs/engineering/production-readiness.md` §5 and §7 ("Held-out evaluation ... full run pending"); `src/evaluation/attacker.py` (held-out Phi-4 harness, `reportable_as_held_out`), `src/evaluation/run.py:220` (`run_comparative_benchmark`). `reports/comparative_benchmark_val.json` currently carries fallback-adversary zeros (see `docs/requirements-coverage.md` items 8 to 10).
- **Why it matters:** This is the single number the thesis rests on: attribute leakage under a held-out adversary with a surviving utility guardrail. The harness is instrumented but has never been run with `mode: "model"` on the test split, so no trustworthy privacy result exists yet.
- **Effort:** L (needs a trained rewriter adapter + GPU run). **Priority:** P0.
- **Status:** *pending approval / not implemented.*

### 2. Commit or publish trained model artifacts with checksums
- **Evidence:** `docs/engineering/production-readiness.md` §7 "Model packaging" ("Risk checkpoint and LoRA adapter live on Colab/Drive; **neither is committed**. Without them the API silently uses the rule-based risk estimate and the heuristic rewriter"); `context.json` snapshot note ("Trained model checkpoints and LoRA adapters are NOT committed").
- **Why it matters:** The API silently falls back to untrained heads / heuristic rewrite, so a fresh clone cannot reproduce any reported result. The new `InferenceGuard_modernbert_train_and_host.ipynb` partially addresses this by hosting on the Hub, but a versioned, checksum'd release path (and a `/health` check that refuses to start or flags a missing model) is still open.
- **Effort:** M. **Priority:** P0.
- **Status:** *pending approval / not implemented.*

### 3. Session lifecycle & operations gaps (`DELETE /session/{id}`, TTL, CORS default)
- **Evidence:** `docs/engineering/production-readiness.md` §6 (Session reset "Partial ... no API endpoint to clear a session, and memory grows until restart. Needed: a `DELETE /session/{id}` and a TTL"; CORS "Default is still `allow_origins=["*"]` with `allow_credentials=True`") and §7 ("Session lifecycle: In-memory, unbounded"). Confirmed in code: `web/api.py` `allowed_origins()` returns `["*"]` by default and `app.add_middleware(CORSMiddleware, allow_origins=allowed_origins(), allow_credentials=True, ...)`.
- **Why it matters:** Unbounded in-memory session growth is a memory-leak / DoS risk, and the permissive default CORS combined with `allow_credentials=True` lets any web page call the API from a visitor's browser. Fine for the Colab demo, not for real user text.
- **Effort:** M. **Priority:** P0.
- **Status:** *pending approval / not implemented.*

---

## P1 (should do)

### 4. Reconcile risk-estimator label source (profile attributes vs `reviews.human` inferability)
- **Evidence:** `context.json` key_patterns ("multitask uses profile attributes mapped via `map_to_taxonomy_id`; binary path uses `reviews.human` inferability"); `src/risk_model/train.py` multitask path; `docs/methodology-review.md` and the design doc favor the `reviews.human` inferability labels. `docs/requirements-coverage.md` item 4.
- **Why it matters:** Training the multi-task heads on profile attributes rather than the human-annotated inferability labels changes what "risk" means and whether the reported F1/ROC-AUC are comparable to the SynthPAI paper. The two paths should be reconciled to one documented decision before the north-star run.
- **Effort:** M. **Priority:** P1.
- **Status:** *implemented (2026-11, with #7).* `train_risk_model` now defaults to `model_type="inferability_labels"` (alias `"binary_classifier"`), training the 4-head BCE classifier on `reviews.human` inferability labels as the authoritative reported signal; the categorical `multitask_modernbert` path is retained and documented as auxiliary/diagnostic. Rationale in `docs/methodology-review.md` Addendum 2026-11, decision #4.

### 5. Consolidate the two calibration modules
- **Evidence:** Production path `src/risk_model/calibrate.py` (`AttributeTemperatureScaler`, `compute_expected_calibration_error`), imported by `src/api/app.py`, `src/risk_model/predict.py`, `src/conversation/leakage.py`, `src/risk_model/attribution.py`, and three tests. Orphan `src/risk_model/calibration.py` (`TemperatureScaler`, `compute_ece`, `generate_reliability_diagram`, `CalibrationMetrics`) is referenced **only** by `tests/test_calibration.py` and `tests/test_e2e_pipeline.py`; `grep -rn 'risk_model.calibration' src scripts web` returns no matches (FEAT-004 findings).
- **Why it matters:** Two parallel calibration implementations are a maintenance hazard and a source of confusion about which one is authoritative. Fold the useful ECE / reliability-diagram logic into `calibrate.py`, then delete the orphan and its redundant test.
- **Effort:** M. **Priority:** P1.
- **Note:** The orphan's only safety net is a test that cannot be executed in this sandbox, so deletion must wait until the full `pytest` suite can be run.
- **Status:** *pending approval / not implemented.*

### 6. Resolve the dead `prompt_baseline.py` module + stale model id
- **Evidence:** `src/rewriter/prompt_baseline.py` class `PromptBaselineRewriter`: `grep` shows it is **never instantiated**; the `prompt_baseline` eval method in `src/evaluation/run.py:292-294` uses `rewriter.protect(..., mode=RewriteMode.PROMPT_BASELINE)` via `src/rewriter/inference.py:989`, not this class. The module also carries a stale default `model_name="Qwen/Qwen3.5-0.8B"` (`src/rewriter/prompt_baseline.py:25`), a vision-language model per `docs/methodology-review.md` §1, overridden to `DEFAULT_BASE_MODEL` only on tokenizer-load failure.
- **Why it matters:** Dead code that looks live is misleading, and the stale default model id misrepresents the Ablation B baseline. Either delete the module or re-wire `run.py` to actually use it, and in either case fix the default to `Qwen/Qwen2.5-1.5B-Instruct` / `DEFAULT_BASE_MODEL`.
- **Effort:** S. **Priority:** P1.
- **Note:** Confirm against a runnable suite before deleting.
- **Status:** *pending approval / not implemented.*

### 7. Fix location / education taxonomy mismatch vs the SynthPAI paper
- **Evidence:** `docs/methodology-review.md` §2: location frozen in `artifacts/taxonomy.json` as `[USA_Northeast, USA_South, USA_Midwest, USA_West, Europe, Other, unknown]` ("Four of our seven classes cover 7% of the data ... collapses to Europe vs Other"); recommendation "do not classify location" (free-text city/country scoring). Education is `[high_school, undergraduate, graduate_professional, other_unknown]` vs the paper's `high school / college / master / PhD`. Implemented in `src/risk_model/taxonomy.py`. `docs/requirements-coverage.md` divergences 2 and 3.
- **Why it matters:** A location-protection number measured on a Europe-vs-Other split is near meaningless and not comparable to the SynthPAI paper; the education scale is similarly off. This affects the credibility of the risk-model and leakage metrics.
- **Effort:** L (changes labels, taxonomy, training, and the attacker scoring). **Priority:** P1.
- **Status:** *implemented (2026-11, with #4).* `ATTRIBUTE_TAXONOMY[EDUCATION]` is now the paper's four-way scale `high_school / college / master / phd` (default bucket for unmatched text documented as `high_school`); `map_to_taxonomy_id`, the adversary heuristic, and the demo ground-truth rows use the new keys; `artifacts/taxonomy.json` regenerated from `ATTRIBUTE_TAXONOMY`. Location: the reportable metric is the free-text generate-and-score path (`Phi4MiniAttacker.evaluate_success`); the coarse location buckets are retained only as an auxiliary/diagnostic head. Residual (dedicated generative location head) flagged in `docs/methodology-review.md` Addendum 2026-11, decision #7.

### 8. Multi-turn attack curve (1/2/4/8 turns) + complete the final comparison matrix
- **Evidence:** Multi-turn leakage implemented in `src/conversation/leakage.py`; staged scenario in `docs/experiment_protocol.md` §16; comparison harness `src/evaluation/run.py:220` over methods `("raw", "presidio", "prompt_baseline", "qlora_sft")`. But `docs/requirements-coverage.md` items 8 and 10 note there is **no committed "after 1/2/4/8 turns" leakage curve** and the committed matrix carries fallback zeros on the val split, not held-out test numbers. The design's matrix rows (Raw / Presidio / Prompt-only / Fine-tuned / Conversation-aware) are not all populated with reportable numbers.
- **Why it matters:** The turn-by-turn curve and the full comparison matrix are core deliverables of the thesis evidence chain; without them the "conversation-aware" advantage is unquantified.
- **Effort:** L (depends on items 1 and 2). **Priority:** P1.
- **Status:** *pending approval / not implemented.*

---

## P2 (nice to have)

### 9. Add the PromptPAI secondary evaluation set (design requirement 5)
- **Evidence:** `grep -rniE 'promptpai' .` returns no source/data matches anywhere in the repo (only the requirement note in `docs/requirements-coverage.md` item 5). The primary SynthPAI profile-disjoint splits exist (`data/splits/`, `data/manifests/split_manifest.json`), but the secondary PromptPAI set called for by design requirement 5 is absent.
- **Why it matters:** A second, independent eval set strengthens the generalization claim and is an explicit design requirement; its absence is a scope gap the reports never claim to have closed.
- **Effort:** M (data acquisition + ingestion + manifest). **Priority:** P2.
- **Status:** *pending approval / not implemented.*

### 10. Purge stale `Qwen3.5-0.8B` references from the implementation notebook
- **Evidence:** `InferenceGuard_Implementation.ipynb` still says the rewriter "will be replaced by a fine-tuned Qwen3.5-0.8B model" and "QLoRA fine-tune Qwen3.5-0.8B (2B as fallback)". `docs/methodology-review.md` §1 established Qwen3.5-0.8B/2B are vision-language models and the project uses text-only `Qwen2.5-1.5B-Instruct`. `reports/session04.md:84` carries the same stale phrasing (historical; may be left as a dated record).
- **Why it matters:** The implementation notebook is reader-facing documentation; the wrong model id contradicts `configs/models.yaml` and `DEFAULT_BASE_MODEL` and misleads anyone reproducing the work. (Session reports are historical and can stay as-is.)
- **Effort:** S. **Priority:** P2.
- **Status:** *pending approval / not implemented.*

### 11. Full refresh of the README "Run It" / Colab narrative
- **Evidence:** `README.md` "Run It" section (line 30+) still leads with a Colab/Drive-centric narrative. The minimal dead-path fix (replacing `data/raw/synthpai.jsonl` / `artifacts/profile_splits.json` with `data/splits/*.jsonl` and pointing at the new notebook) was already made in FEAT-004 (commit 7cc2c0f); a fuller refresh of the surrounding narrative is still open.
- **Why it matters:** The narrative still frames Colab as the primary path when production-readiness.md calls Colab "a stand-in ... described as a demo, not as the product." A refreshed quickstart (local-first install, then the ModernBERT notebook, then the Hub-hosted artifacts) would match the current architecture.
- **Effort:** S. **Priority:** P2.
- **Status:** *pending approval / not implemented.*

---

*Reminder: nothing above has been implemented. Approve the items you want, in any order,
and that work can begin.*
