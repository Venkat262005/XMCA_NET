# XMCA-Net — Independent Repository Audit and Implementation Gap Report

**Audit Date**: November 4, 2026  
**Repository Branch**: `feature/xmca-full-pipeline`  
**Current Baseline Git Commit**: `9937638` (origin/main)  
**Execution Environment**: Local Verification on macOS (Sandboxed Unit/Contract Testing Only; Hardware Execution strictly on Kaggle dual NVIDIA T4 GPUs).

---

## 1. Executive Summary

A comprehensive forensic audit of the `XMCA_NET_SRC` codebase was conducted prior to full benchmark pipeline implementation. The repository is in an authentic, uncompromised state with:
- **Zero fabricated data**: No synthetic labels, mock probability scores, dummy features, or fake latency calculations exist.
- **54 passing tests** across data schema, configuration, manifests, model registries, and VLM adapters.
- **Frozen cohorts**: Exactly 3,174 evaluation records (1,000 POPE tri-strategy, 174 FaithScore human-annotated, 1,000 A-OKVQA, and 1,000 InfoSeek).
- **Physical image group partitions**: 2,379 unique image groups properly partitioned into 1,917 validation groups (combiner training eligible) and 462 held-out test groups (including all 24 overlapping image groups with strictly zero leakage: `eligible_for_combiner_training = False`, `eligible_for_calibration = False`).
- **Kaggle vs. Local Isolation**: Dry-run outputs are quarantined in `tests/fixtures/task2_dry_run/`, leaving `artifacts/generations/task2_verified/` and production generation directories strictly pristine.

This report establishes the implementation blueprint for the remaining 12 stages required to complete the XMCA-Net research pipeline.

---

## 2. Test Suite & Verification Status

The current local test suite was executed in `.venv`:
```bash
.venv/bin/pytest -v
```
**Result**: **54 passed in 1.70s**
- `tests/test_config.py` (5 tests): Configuration validation and schema checks.
- `tests/test_dataset_schema.py` (7 tests): Standardized dataset schema integrity.
- `tests/test_manifest.py` (4 tests): Manifest loading and hashing.
- `tests/test_model_registry.py` (3 tests): Model registry definitions for LLaVA-1.5-7B, Qwen2-VL-7B, InternVL2-2B, InternVL2-8B.
- `tests/test_strict_mode.py` (7 tests): Strict anti-fabrication assertions and guardrails.
- `tests/test_task1_provenance.py` (13 tests): Dataset provenance, image SHA-256 integrity, cohort tier distribution.
- `tests/test_vlm_adapters.py` (15 tests): Abstract interface, tokenization/prompting contracts, multi-GPU device mapping, and dynamic image resolution slicing.

---

## 3. Hardware & Execution Boundary Verification

| Component | Policy | Current Status | Forensic Verification |
| :--- | :--- | :--- | :--- |
| **Model Weight Loading** | Kaggle Dual T4 only | Strictly Enforced | No weights loaded locally; mock tests only test contract shapes |
| **Autoregressive Generation** | Kaggle Dual T4 only | Strictly Enforced | Production generation dirs contain `.gitkeep` only |
| **GPU Telemetry** | Live CUDA API calls | Verified | `src/generation/telemetry.py` queries `torch.cuda` / `nvidia-smi` |
| **Image Leakage Guard** | 24 overlapping groups | Held-out test only | `artifacts/manifests/global_image_group_manifest.parquet` |
| **InfoSeek Guard** | Access restricted | Blocked without disk image | Explicitly guarded in adapter inference routines |

---

## 4. Comprehensive 12-Stage Pipeline Gap Analysis

| Stage | Description | Current Status | Implementation Requirements |
| :--- | :--- | :--- | :--- |
| **Stage 1** | Repository Audit & Reconciliation | **Completed** | Full gap analysis and audit verification report. |
| **Stage 2** | Data & Authentic Knowledge-Base Validation | **Partially Complete** | Cohort manifests exist (3,174 samples). Need authentic RAG knowledge base for A-OKVQA & InfoSeek; explicit `NOT_APPLICABLE` for POPE/FaithScore. |
| **Stage 3** | Sequential Full VLM Generation Infrastructure | **Partially Complete** | Single-probe smoke test notebook exists; need full benchmark batch runner with process exit & memory flushing per model. |
| **Stage 4** | Generation Quality & Ground Truth Verification | **Pending Execution** | Verification scripts to check generation completion, exact ID alignment, and evaluation correctness. |
| **Stage 5** | Consistency Diagnostics ($S_P, S_R, S_U$) | **Pending Implementation** | Atomic claim extraction, OWL-ViT visual grounding ($S_P$), dense/lexical retrieval relevance ($S_R$), DeBERTa-v3 NLI entailment ($S_U$). |
| **Stage 6** | Diagnostic Feature Extraction & Alignment | **Pending Implementation** | Multi-axis feature vector compilation, missingness masks (`has_rag=False` for POPE/FaithScore), and schema validation. |
| **Stage 7** | XMCA-Net Combiner Training & Calibration | **Pending Implementation** | Multi-layer perceptron / logistic combiner with missingness masks, validation-only temperature scaling / isotonic calibration. |
| **Stage 8** | Failure Cause Attribution Head | **Pending Implementation** | Supervised attribution head ($P$ vs $R$ vs $U$) active strictly on ground-truth labeled cohorts. |
| **Stage 9** | Component Ablation Study | **Pending Implementation** | Ablation matrix: ($S_P$ only, $S_R$ only, $S_U$ only, pairwise, full XMCA-Net). |
| **Stage 10** | Comprehensive Evaluation Across Dimensions | **Pending Implementation** | AUROC, PR-AUC, ECE, Attribution Accuracy across dataset cohorts, model families, and progressive sample tiers. |
| **Stage 11** | End-to-End Kaggle Pipeline Orchestration | **Pending Implementation** | Self-contained, sequential multi-stage Kaggle notebook running all models and diagnostics within T4 constraints. |
| **Stage 12** | Research Artifacts & Final Audit Sign-Off | **Pending Execution** | Final research tables, LaTeX-ready figures, and independent audit sign-off. |

---

## 5. Architectural Blueprints for Missing Core Modules

### 5.1 Authentic Knowledge Base (`src/retrieval/corpus.py`)
- Documents constructed strictly from authentic sources:
  - A-OKVQA: verified rationales and train-split annotations.
  - InfoSeek: genuine Wikipedia article passages associated with target entities.
- Zero synthetic passages. POPE and FaithScore receive explicit empty sets with `has_rag_evidence = False`.
- Inverted index (lexical BM25) and dense index (BGE-base-en-v1.5) with passage metadata: `doc_id`, `source_url`, `passage_text`, `passage_boundaries`, `entity_association`, `sha256`, `corpus_split_eligibility`.

### 5.2 Atomic Claim Extraction (`src/claims/extractor.py`)
- Decomposes generated VLM answers into atomic factual claims:
  - Categories: `object_presence`, `attribute`, `action`, `relationship`, `count`, `location`, `external_fact`.
  - Claim verification states: `supported`, `contradicted`, `insufficient_evidence`, `unverifiable`, `not_applicable`.

### 5.3 Tri-Axis Consistency Diagnostics
- **$S_P$ (Perception Consistency, `src/perception/consistency.py`)**: Regional visual grounding using open-vocabulary detection (OWL-ViT) or regional similarity (SigLIP), checking token-object alignment.
- **$S_R$ (Retrieval Consistency, `src/retrieval/consistency.py`)**: Question-evidence relevance, retrieved coverage, and ranking diagnostics.
- **$S_U$ (Utilization Consistency, `src/utilization/consistency.py`)**: NLI entailment scoring (DeBERTa-v3) comparing generated claims to retrieved passages, detecting ignored evidence or unfaithful hallucination.

### 5.4 Combiner & Attribution Architecture (`src/combiner/`)
- Combiner input: $[f_P, f_R, f_U, m_R, m_U]$ where $m$ are explicit missingness indicator masks.
- Calibration: Temperature scaling or isotonic regression on validation image groups only.
- Attribution head: Categorical cross-entropy predicting root cause ($P$ vs $R$ vs $U$) evaluated against datasets with explicit ground truths.

---

## 6. Audit Verdict

**Stage 1 Sign-Off**: **APPROVED**  
The codebase is clean, authentic, and verified. Full pipeline implementation proceeds to Stage 2 (Knowledge Base & RAG Corpus Construction) and subsequent stages.

