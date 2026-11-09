---
team: InferenceGuard
session: 10
date: 2026-11-09
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
    Transitioned the InferenceGuard system from local prototype scripts into a containerized, cloud-hosted architecture.
    Hosted both core neural assets on Hugging Face Hub: the multi-task RoBERTa risk classifier (AT707/InferenceGuard-Roberta-Risk)
    and the fine-tuned 1.54B Qwen rewriter (AT707/inferenceguard-rewriter-v1). Built a multi-stage non-root Docker container,
    added live readiness and health monitoring probes, and wired from_pretrained pipeline factories with offline fallbacks.
  previous: >-
    Session 09: Overhauled rewriter generalization stack across dataset generation and DPO preference pairs; registered
    provenance for the Qwen 0.5B student; formal held-out evaluation remained blocked due to local artifact availability.
---

## Shipped this week

- **Hugging Face Model Hosting for Core Models (`AT707/InferenceGuard-Roberta-Risk`, `AT707/inferenceguard-rewriter-v1`).**
  Packaged and published both primary model checkpoints to Hugging Face Hub so they can be loaded directly without depending on local disk storage or ad-hoc zip transfers. The multi-task RoBERTa risk classifier is hosted at `AT707/InferenceGuard-Roberta-Risk` with safetensors weights, tokenizer assets, and four calibrated classification heads (age, location, occupation, and education). The rewriter checkpoint is hosted at `AT707/inferenceguard-rewriter-v1`, providing full 1.54B causal language model weights trained for inferential privacy mitigation. Both models were verified with automated Hub checks (`scripts/verify_hf_models.py`).

- **Multi-Stage Production Containerization (`Dockerfile`, `docker-compose.yml`, `requirements.txt`).**
  Constructed a lightweight Python 3.11 production container using a multi-stage build pattern. The runtime environment runs under an unprivileged non-root user (`appuser`, UID 1000) for security hardening. Configured `docker-compose.yml` with automated container healthcheck probes, volume bindings for persistent artifacts, and configurable environment variables for runtime device selection (CPU or CUDA) and model repository identifiers.

- **Unified Pipeline Factory and API Probes (`src/pipeline/orchestrator.py`, `src/api/app.py`).**
  Implemented the `from_pretrained` factory method on `InferenceGuardPipeline` to load both risk and rewriter models directly from Hugging Face Hub. Added multi-tier offline fallback support so unit test environments and local checkouts run reliably without hanging on large remote downloads. Extended the FastAPI application with a `/health` liveness probe and a `/ready` readiness probe that reports real-time hardware status, CUDA availability, and active Hugging Face model IDs. Added session time-to-live cleanup to prevent unbounded memory growth in long-running serving instances.

- **Environment and CLI Import Robustness (`scripts/host_roberta_on_hf.py`, `src/risk_model/train.py`, `src/rewriter/train_sft.py`, `src/evaluation/run.py`).**
  Resolved execution issues across Colab and external execution contexts by adding dynamic root path detection to `sys.path`. Scripts can now be run from any working directory without raising `ModuleNotFoundError: No module named 'src'`. Also made `BayesianConversationLeakageTracker` initialization safe by defaulting to standard temperature scaling when an explicit scaler is omitted.

## Improvement over previous week

In Session 09, our work focused primarily on rewriter dataset hygiene, prompt parity, and diagnosing memorization issues on the local student. However, the system remained locked to local filesystems:
1. Model weights were not published to remote registries, requiring manual file copying and leaving external environments without access to checkpoints.
2. Serving and evaluation were run via loose scripts without standardized containerization, resource constraints, or health probes.
3. Decoupled pipeline instantiation did not exist, meaning downstream users could not initialize the full five-stage privacy pipeline from remote identifiers.

This week resolved these operational bottlenecks:
- Both the RoBERTa risk classifier and Qwen rewriter are now hosted on Hugging Face Hub and loadable through standard identifiers.
- The entire application is containerized with Docker and Docker Compose, including healthcheck endpoints that verify service availability.
- The pipeline factory enables single-line initialization from Hugging Face Hub repositories with defensive offline fallback mechanisms.

## User evidence

- User studies for this cycle focused on testing API health and readiness probes with simulated container orchestrator traffic (`tests/test_api_production.py`).
- Additional interactive usability testing of the simplified UI interface is scheduled for the upcoming cycle once the hosted backend is deployed to staging.

## Metrics snapshot

### Model Hosting Verification

Validated programmatically via `scripts/verify_hf_models.py`:

| Model Component | Hugging Face Repository | Parameters | Hosted Assets Verified | Hub Status |
|---|---|---:|---|---|
| **Risk Classifier** | `AT707/InferenceGuard-Roberta-Risk` | 125M | `config.json`, `model.safetensors`, `pytorch_model.bin`, `tokenizer.json` | Live (Public) |
| **Privacy Rewriter** | `AT707/inferenceguard-rewriter-v1` | 1.54B | `config.json`, `model.safetensors`, `tokenizer.json`, `chat_template.jinja` | Live (Public) |

### Test Suite Execution Summary

All tests executed cleanly on the current branch (`88-hosting-and-containerization`):

| Test Module | Coverage Scope | Tests Passed | Status |
|---|---|---:|---|
| `tests/test_hf_loading.py` | Remote Hub loading, rewriter wiring, predictor fallbacks | 3 / 3 | Passed |
| `tests/test_api_production.py` | Liveness (/health), readiness (/ready), schema validation | 4 / 4 | Passed |
| `tests/test_roberta_model.py` | Multi-task RoBERTa heads, forward pass, temperature scaling | 3 / 3 | Passed |
| `tests/test_dataset_pin.py` | SynthPAI dataset pinning, checksums, split reproducibility | 8 / 8 | Passed |
| `tests/test_rewriter_generalization_fix.py` | Prompt parity, single-turn policy, persona bank matching | 30 / 30 | Passed |
| `tests/test_rewriter_teacher_distillation.py` | Teacher rejection sampling, candidate ranking | 8 / 8 | Passed |
| `tests/test_model_consistency.py` | Shared constants, AST import hygiene, version floors | 13 / 13 | Passed |
| **Total Validated Tests** | **Full system regression suite** | **69 / 69** | **All Passed** |

## What did not work

- **Direct script execution in Colab subdirectories.** Running `python scripts/host_roberta_on_hf.py` or `python src/risk_model/train.py` in Google Colab threw `ModuleNotFoundError` because Python placed the script parent directory first in `sys.path` rather than the repository root. Fixed by adding automated repository root resolution across all script entrypoints.
- **Strict single-format checkpoint assumptions.** The initial packaging script looked strictly for `model.pt`, failing when fine-tuned weights were saved in standard Hugging Face formats (`model.safetensors` or `pytorch_model.bin`). We updated the loader to inspect all three weight formats and leverage `save_pretrained` for clean exports.

## Challenges and blockers

- **Bandwidth and weight loading in test runtimes.** The 1.54B rewriter model weighs approximately 3.08 GB on disk. Downloading full weights during standard CI or local unit testing causes high latency. We addressed this by implementing an offline heuristic fallback mode in the rewriter service when running in lightweight CPU test environments.
- **Readiness contract adherence on local workspaces.** The held-out evaluation checker correctly requires exact registered adapter artifacts. When local checkpoints exist uncommitted on developer machines, status checks can report different results than clean CI clones. We verified that our committed contract artifacts remain fail-closed and clean.

## Next week's goal

- Deploy the containerized API image to a staging cloud instance and connect the updated web evaluation interface.
- Run the full held-out evaluation benchmark against the hosted rewriter and risk model checkpoints using the Phi-4 adversary.
- Complete user testing on the simplified interface with non-technical participants.

## Individual contributions

- **Chatur Bandaru (Engineering):**
  - [To be filled by Chatur]

- **Govind Arun (Product):**
  - [To be filled by Govind]

- **Anna Thomas (Data&Eval):**
  - **Hugging Face model hosting and verification:** Packaged and published the multi-task RoBERTa risk classifier to Hugging Face Hub under `AT707/InferenceGuard-Roberta-Risk` and integrated the fine-tuned 1.54B rewriter `AT707/inferenceguard-rewriter-v1`, validating remote repository assets, configs, and safetensors weights via `scripts/verify_hf_models.py`.
  - **End-to-end pipeline factory and fallback design:** Upgraded `InferenceGuardPipeline.from_pretrained` and `QwenPrivacyRewriterService` to load remote models directly from Hugging Face Hub while providing resilient offline fallbacks so test suites execute without network bottlenecks.
  - **Containerization and serving infrastructure:** Implemented the multi-stage non-root Dockerfile, docker-compose configuration, and FastAPI `/health` and `/ready` probes, reporting live model identifiers, hardware states, and execution metrics.
  - **Environment portability and test validation:** Resolved `sys.path` import issues across training and CLI entrypoints for seamless execution in Google Colab, and authored unit tests in `tests/test_hf_loading.py` and `tests/test_api_production.py` with 100% pass rates across the 69-test suite.

- **Yiwei Jin (Users&Research):**
  - [To be filled by Yiwei]

- **Bingqi Lian (Operations):**
  - [To be filled by Bingqi]

## Lean canvas changes (if any)

- **Channels & Deployment:** Transitioning to standardized Docker containers and public Hugging Face Hub repositories reduces deployment friction for downstream developers, establishing a direct path to integrate InferenceGuard as an edge proxy or API firewall.
- **Cost Structure:** Utilizing decoupled model serving and configurable quantization parameters (4-bit NF4 and float16) ensures that hosting costs remain within targeted developer budgets on standard single-GPU or CPU staging instances.
