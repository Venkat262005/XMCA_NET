# XMCA-Net: Implementation Status & Task Roadmap

This document provides a transparent, authoritative status of all research components in XMCA-Net.

---

## 1. Task Progress Matrix

| Task ID | Task Description | Status | Target Environment | Deliverables |
|---|---|---|---|---|
| **Task 0** | Build repository foundation from zero | **COMPLETED** | Local Workspace / Sandbox | Configs, schemas, tests, docs, bootstrap notebook, git commit |
| **Task 1** | Dataset acquisition, checksum verification & tier sampling | **PENDING** | Local & Remote | Raw data download scripts, sha256 manifests for 100/300/600/1000 tiers |
| **Task 2** | Kaggle dual-T4 bootstrap verification | **PENDING** | Kaggle Dual T4 | Executed `XMCA_BOOTSTRAP_KAGGLE.ipynb` & persisted hardware verification report |
| **Task 3** | VLM generation pipeline (LLaVA, Qwen2-VL, InternVL) | **PENDING** | Kaggle Dual T4 | Generation scripts, matched ID outputs across 4 tiers |
| **Task 4** | Consistency diagnostics ($S_P, S_R, S_U$) | **PENDING** | Kaggle Dual T4 | Feature extractors, cross-attention grounding, knowledge verification |
| **Task 5** | Calibrated detection combiner & attribution head | **PENDING** | Kaggle Dual T4 | Calibrated combiner, supervised attribution head on POPE & FaithScore |
| **Task 6** | Baseline implementations & comparative benchmark | **PENDING** | Kaggle Dual T4 | Comparative evaluations (AUROC, PR-AUC, ECE, Attribution Acc) |
| **Task 7** | Final synthesis, reporting & paper artifact compilation | **PENDING** | Local / Kaggle | Comprehensive benchmark tables, ablation studies, camera-ready artifacts |

---

## 2. Module Status Breakdown

| Module | File Location | Implementation Status | Notes |
|---|---|---|---|
| `src.utils.config` | `src/utils/config.py` | **ACTIVE** | Strict Pydantic validation, checks 4 datasets and 3 models |
| `src.utils.environment` | `src/utils/environment.py` | **ACTIVE** | Enforces Kaggle dual-T4 requirement and blocks local inference |
| `src.utils.manifest` | `src/utils/manifest.py` | **ACTIVE** | Validates matched evaluation sample IDs across models |
| `src.data.schema` | `src/data/schema.py` | **ACTIVE** | Strict schemas for POPE, FaithScore, A-OKVQA, InfoSeek |
| `src.data.loader` | `src/data/loader.py` | **PENDING** | Pending Task 1; raises `FileNotFoundError` / `NotImplementedError` |
| `src.generation` | `src/generation/generator.py` | **PENDING** | Pending Task 3; strictly blocked on local Mac |
| `src.perception` | `src/perception/consistency.py` | **PENDING** | Pending Task 4; placeholder scores strictly prohibited |
| `src.retrieval` | `src/retrieval/consistency.py` | **PENDING** | Pending Task 4; placeholder scores strictly prohibited |
| `src.utilization` | `src/utilization/consistency.py` | **PENDING** | Pending Task 4; placeholder scores strictly prohibited |
| `src.combiner` | `src/combiner/combiner.py` | **PENDING** | Pending Task 5; no hardcoded probabilities allowed |
| `src.baselines` | `src/baselines/baselines.py` | **PENDING** | Pending Task 6; no dummy baseline functions |
| `src.evaluation` | `src/evaluation/metrics.py` | **PENDING** | Pending Task 6; genuine metric computations only |

---

## 3. Strict Research Constraints in Effect

- **Zero Synthetic Data**: Under no circumstances will synthetic or simulated dataset samples be added.
- **Zero Dummy Scores**: All un-implemented diagnostic and combiner methods raise `NotImplementedError`. No hardcoded floating-point numbers or mock probability returns.
- **Hardware Isolation**: Research model inference and feature extraction remain 100% restricted to Kaggle dual NVIDIA T4 GPUs.
