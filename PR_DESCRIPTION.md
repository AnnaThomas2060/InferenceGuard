# Pull Request: Production Containerization and Hugging Face Model Hosting (#88)

## Overview
This PR completes the end-to-end productionization, containerization, and remote model hosting for InferenceGuard. It decouples the system from local checkpoint dependencies by hosting both neural models on Hugging Face Hub, introduces a secure multi-stage Docker environment with health probes, implements unified pipeline factory loading with defensive offline fallbacks, and fixes environment portability across local and Colab runtimes.

---

## What I Did

### 1. Production Containerization & Environment Configuration
- **Multi-Stage Dockerfile**: Implemented a hardened Python 3.11-slim container with build-stage dependency isolation, running under an unprivileged non-root user (`appuser`, UID 1000).
- **Docker Compose**: Configured `docker-compose.yml` with port mapping (`8000:8000`), persistent volume mounts for artifacts, restart policies, and liveness healthchecks.
- **Dependency Pinning**: Pinned runtime serving dependencies in `requirements.txt` and updated `pyproject.toml`.
- **Environment Management**: Updated `.env.example` with standard defaults for ports, device selection (`cpu`/`cuda`), and Hugging Face model IDs.

### 2. Hugging Face Model Hosting for Core Models
- **RoBERTa Multi-Task Risk Classifier (`AT707/InferenceGuard-Roberta-Risk`)**:
  - Packaged and published the 125M RoBERTa risk classifier to Hugging Face Hub with calibrated classification heads across 4 attributes (`age`, `location`, `occupation`, `education`).
  - Created standalone package directory (`hf_roberta_risk_model/`) with custom architecture definitions (`configuration_roberta_risk.py`, `modeling_roberta_risk.py`) and upload utilities.
  - Implemented `scripts/host_roberta_on_hf.py` supporting `safetensors`, `pytorch_model.bin`, and `model.pt` formats.
- **Privacy Rewriter Checkpoint (`AT707/inferenceguard-rewriter-v1`)**:
  - Integrated the fine-tuned 1.54B Qwen causal language model checkpoint hosted on Hugging Face Hub.
  - Formally registered the remote repository in `configs/models.yaml` under `rewriter.hub_model`.

### 3. Pipeline Factory & Dynamic Service Serving
- **`InferenceGuardPipeline.from_pretrained`**: Built a factory method in `src/pipeline/orchestrator.py` that initializes the full privacy protection pipeline directly from Hugging Face Hub repositories.
- **Defensive Offline Fallbacks**: Added multi-tier offline fallback handling so that test runners and lightweight environments do not stall on multi-gigabyte remote downloads.
- **Session Management**: Implemented automatic time-to-live (TTL) expiration in the pipeline to prevent memory leaks during long-running serving.

### 4. FastAPI Production Health Probes
- **`/health`**: Liveness probe returning HTTP 200 and service timestamps to verify container process health.
- **`/ready`**: Readiness probe checking GPU/CUDA availability, device placement, active `model_id` (`AT707/InferenceGuard-Roberta-Risk`), and `rewriter_model_id` (`AT707/inferenceguard-rewriter-v1`).
- **`/analyze`**: Standardized request/response schemas for external callers with structured inferential risk reports.

### 5. Portability & CLI Entrypoint Fixes
- **Colab/Subdirectory Import Fixes**: Resolved `ModuleNotFoundError: No module named 'src'` by dynamically registering the repository root in `sys.path` across `scripts/host_roberta_on_hf.py`, `src/risk_model/train.py`, `src/rewriter/train_sft.py`, and `src/evaluation/run.py`.
- **Leakage Tracker Resilience**: Made `BayesianConversationLeakageTracker` initialization safe with automatic default scaler fallback.

### 6. Automated Verification & Test Coverage
- **Automated Hub Verifier**: Built `scripts/verify_hf_models.py` which queries Hugging Face APIs to confirm both repositories are accessible and contain all necessary assets (`config.json`, `model.safetensors`, `tokenizer.json`).
- **Unit Test Suites**: Added and verified tests in `tests/test_hf_loading.py`, `tests/test_roberta_model.py`, and `tests/test_api_production.py`.

---

## Verification & Test Results

All 69 system tests passed cleanly on the branch:
- `tests/test_hf_loading.py`: 3 passed
- `tests/test_api_production.py`: 4 passed
- `tests/test_roberta_model.py`: 3 passed
- `tests/test_dataset_pin.py`: 8 passed
- `tests/test_rewriter_generalization_fix.py`: 30 passed
- `tests/test_rewriter_teacher_distillation.py`: 8 passed
- `tests/test_model_consistency.py`: 13 passed

Hugging Face Hub verification (`python scripts/verify_hf_models.py`):
```text
[INFO] Successfully reached repository: AT707/InferenceGuard-Roberta-Risk
[INFO]   [PASS] Found config.json
[INFO]   [PASS] Found model.safetensors
[INFO]   [PASS] Found tokenizer.json
[INFO] Successfully reached repository: AT707/inferenceguard-rewriter-v1
[INFO]   [PASS] Found config.json
[INFO]   [PASS] Found model.safetensors
[INFO]   [PASS] Found tokenizer.json
[INFO] Pipeline rewriter configured correctly: AT707/inferenceguard-rewriter-v1
[INFO] All model verification checks completed successfully.
```

---

## What Still Needs to Be Done

1. **Deploy Staging Container**:
   - Run `docker compose up -d` on the target staging instance and verify health probes externally via reverse proxy/load balancer.
2. **Execute Full Held-Out Adversary Benchmark**:
   - Run the Phi-4-mini held-out evaluation across `data/splits/test.jsonl` hitting the hosted model checkpoints to generate official Session 10 metrics.
3. **Connect Frontend to Container Gateway**:
   - Point the redesigned web interface to the containerized `/protect` and `/analyze` endpoints.
4. **Hugging Face Hub Model Cards**:
   - Add/customize description documentation directly on Hugging Face Hub for `AT707/inferenceguard-rewriter-v1` and `AT707/InferenceGuard-Roberta-Risk`.

---

## PR Checklist
- [x] Code is clean, formatted, and free of regression bugs
- [x] All 69 unit and integration tests pass
- [x] Multi-stage Dockerfile and docker-compose configurations tested
- [x] Models verified on Hugging Face Hub
- [x] Session 10 progress report added (`reports/session10.md`)
- [x] No binary model weights committed to git tracking
