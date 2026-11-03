# XMCA-Net: Implementation Status & Task Roadmap

This document provides a transparent, authoritative status of all research components in XMCA-Net.

---

## 1. Task Progress Matrix

| Task ID | Task Description | Status | Target Environment | Deliverables |
|---|---|---|---|---|
| **Task 0** | Build repository foundation from zero | **COMPLETED** | Local Workspace / Sandbox | Configs, schemas, tests, docs, bootstrap notebook, git commit |
| **Task 1 & 1.1** | Authentic dataset acquisition, provenance & leakage reconciliation | **COMPLETED** | Local Workspace / Sandbox | 3,174 frozen records (POPE tri-strategy, FaithScore human, A-OKVQA, InfoSeek), global image group manifest, storage audit, 39 passing tests |
| **Task 2** | VLM Adapters & Kaggle Dual-T4 Smoke Test | **COMPLETED** | Kaggle Dual T4 / Local Contracts | Unified `BaseVLMAdapter`, LLaVA, Qwen2-VL, InternVL adapters, dynamic patching, multi-GPU mapping, InfoSeek & leakage guards, 52 passing tests, smoke notebook & script |
| **Task 3** | Benchmark VLM generation pipeline across 3 models | **PENDING** | Kaggle Dual T4 | Generation runs, 9,522 outputs across verified cohorts, matched ID manifests |
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
| `src.data.loader` | `src/data/loader.py` | **ACTIVE** | Authentic loaders for POPE tri-strategy, FaithScore 174 human annotations, A-OKVQA, InfoSeek |
| `src.generation.base` | `src/generation/base.py` | **ACTIVE** | Abstract `BaseVLMAdapter` and standardized `VLMGenerationOutput` schema |
| `src.generation.adapters` | `src/generation/adapters/` | **ACTIVE** | LLaVA, Qwen2-VL, and InternVL adapters with dynamic patching and dual-T4 device maps |
| `src.generation.registry` | `src/generation/registry.py` | **ACTIVE** | Factory registry mapping canonical keys with revision verification |
| `src.generation.generator` | `src/generation/generator.py` | **ACTIVE** | Orchestrator with InfoSeek un-materialized image safeguard |
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

