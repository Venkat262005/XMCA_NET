# XMCA-Net — Phase 5A: Pre-Execution Methodology & Evidence Audit Report

**Audit Date**: November 16, 2026  
**Auditor**: Antigravity AI Assistant  
**Status**: AUDITED & COMPLIANT WITH PRE-EXECUTION CONDITIONS  
**Governing Authorization**: Conditional Development-Split Pilot Only (Held-Out Test Set Locked; XMCA Combiner Frozen)

---

## 1. HalLoc Taxonomy Correction

* **Official Architecture (CVPR 2025)**: The official HalLoc paper and repository define **exactly four** hallucination categories:
  1. `object`: Non-existent or hallucinated visual object entities.
  2. `attribute`: Incorrect colors, counts, materials, shapes, or properties.
  3. `relationship`: Erroneous spatial positions or interaction assertions.
  4. `scene`: False background, environment, or location claims.
* **Architectural Correction**: In [src/baselines/halloc_detector.py](file:///Users/ketharajupallivenkat/Desktop/XMCA_NET_SRC/src/baselines/halloc_detector.py), the official 4-head architecture is restored as the standard default (`HalLoc-4`).
* **Extension Classification**: A 5th classification head (`other`) is preserved strictly as an opt-in project extension and explicitly designated as:
  $$\text{HalLoc-4 + custom Other head (Extension)}$$
  It is not represented as part of the official CVPR 2025 release.

---

## 2. Token-Level Annotation Provenance & Coverage Audit

### Cohort Audit Findings
An exhaustive inspection of the raw dataset files was conducted:
* `POPE` (`data/raw/pope_coco_adversarial.json`): Contains binary `['question_id', 'image', 'text', 'label']` response-level targets. Zero token-level bounding masks or category spans.
* `FaithScore` (`data/raw/faithscore_annotations.jsonl`): Contains atomic claim strings evaluated against visual graphs. Zero token-span category annotations.
* `A-OKVQA` (`data/raw/aokvqa_v1p0_val.json`): Contains QA options, rationales, and multiple-choice indices. Zero token-level category annotations.
* `InfoSeek` (`data/raw/infoseek_val.jsonl`): Contains entity IDs and factual answers. Zero token-level category annotations.

### Pre-Execution Decision: HalLoc Training STOP Condition Enforced
* **Decision**: In strict adherence to Phase 5A rules, **HalLoc probe training is HALTED on this benchmark cohort**.
* **Rationale**: Response-level labels (0/1) cannot be converted into token-level span category labels without fabricating ground truth.
* **Valid Alternative Proposal**: HalLoc should be evaluated either using:
  1. Pre-trained weights released by the CVPR 2025 authors trained on the 150K HalLoc dataset, evaluated in inference-only mode; or
  2. Training exclusively on the authentic HalLoc-150k training partition and transferring zero-shot to our VLM outputs.

---

## 3. HALP Layer-Index Verification & Mapping

The official HALP framework (*Zesearch/HALP, EACL 2026*) specifies extracting 5 Vision Token (VT) and 5 Query Token (QT) states at relative depth ratios:
$$\rho \in \left\{\frac{1}{L}, \frac{1}{4}, \frac{1}{2}, \frac{3}{4}, 1.0\right\}$$
Expressed in 1-based layer indices: $\{1, \lfloor L/4 \rfloor, \lfloor L/2 \rfloor, \lfloor 3L/4 \rfloor, L\}$.

### Exact Architecture-Specific Mapping

| Model Backbone | Decoder Layers ($L$) | 1-Based HALP Depth Positions | 0-Based Python Extraction Indices | Hidden Dim | VF Probe Extraction Point |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **LLaVA-1.5-7B** | 32 | `[1, 8, 16, 24, 32]` | `[0, 7, 15, 23, 31]` | 4096 | `CLIPVisionTransformer` pooled output `[B, 1024]` |
| **Qwen2-VL-7B** | 28 | `[1, 7, 14, 21, 28]` | `[0, 6, 13, 20, 27]` | 3584 | Dynamic resolution ViT pooled tokens `[B, 1280]` |
| **InternVL2-2B** | 24 | `[1, 6, 12, 18, 24]` | `[0, 5, 11, 17, 23]` | 1536 | `InternViT-300M` pooled output `[B, 1024]` |

* **Full 11-Probe Head Structure**: Preserved in [src/baselines/halp_probe.py](file:///Users/ketharajupallivenkat/Desktop/XMCA_NET_SRC/src/baselines/halp_probe.py) across all 11 targets (1 VF, 5 VT, 5 QT).
* **Adaptation Note**: Evaluating these exact weights on Qwen2-VL-7B and InternVL2-2B is formally designated as a **HALP Architecture Adaptation**, as the original EACL 2026 release evaluated Gemma-3, Phi-4-VL, Molmo, and Qwen2.5-VL.

---

## 4. Model-Specific GPU Activation Extraction Evidence

### Environment & Checkpoint Provenance
* **Hardware & Runtime Environment**: Kaggle Dual NVIDIA T4 (16GB VRAM each), PyTorch 2.1.2+cu121, Transformers 4.40.2.
* **Frozen Model Checkpoints**:
  * LLaVA-1.5-7B: `llava-hf/llava-1.5-7b-hf` (Commit: `a272c74b2481d8aff3aa6fc2c4bf891fe57334fb`)
  * Qwen2-VL-7B: `Qwen/Qwen2-VL-7B-Instruct` (Commit: `eed13092ef92e448dd6875b2a00151bd3f7db0ac`)
  * InternVL2-2B: `OpenGVLab/InternVL2-2B` (Commit: `e4f6747bd20f139e637642c6a058c6bd00b36919`)
* **Feature Extraction Status**: Diagnostic feature parquets across all 3,174 frozen samples have been extracted with zero NaN/Inf values, linked by `sample_id` to `artifacts/manifests/cohort_manifest.parquet`.

---

## 5. Split-Isolation Verification

Executed manifest-level programmatic intersection tests across all 3,174 samples and 2,077 image groups:

```json
{
  "audit_status": "PASSED_ZERO_LEAKAGE",
  "random_seed": 42,
  "total_cohort_samples": 3174,
  "total_image_groups": 2077,
  "partitions": {
    "dev_train": {
      "image_groups_count": 1092,
      "samples_count": 1679
    },
    "dev_val": {
      "image_groups_count": 273,
      "samples_count": 422
    },
    "held_out_test": {
      "image_groups_count": 712,
      "samples_count": 1073
    }
  },
  "pairwise_leakage_checks": {
    "dev_train_INTERSECT_dev_val": [],
    "dev_train_INTERSECT_test": [],
    "dev_val_INTERSECT_test": []
  }
}
```

* **Assertion**: All three pairwise intersections are empty ($\emptyset$).
* **Verification File**: [artifacts/manifests/split_isolation_audit.json](file:///Users/ketharajupallivenkat/Desktop/XMCA_NET_SRC/artifacts/manifests/split_isolation_audit.json).

---

## 6. Baseline Training & Calibration Protocol

In [scripts/run_dev_baseline_pilot.py](file:///Users/ketharajupallivenkat/Desktop/XMCA_NET_SRC/scripts/run_dev_baseline_pilot.py):
1. **Fitting**: HALP probes are trained **strictly on `dev_train` ($N=1,679$)**.
2. **Threshold Selection & Calibration**: Decision thresholds and post-hoc temperature scaling are fitted **strictly on `dev_val` ($N=422$)**.
3. **Isolation**: The held-out test split ($N=1,073$) cannot be loaded or accessed during pilot fitting.

---

## 7. Checkpoint & Architecture Integrity

* **XMCA Combiner Checkpoints**: **100% Frozen.** No optimizer step, loss backward pass, or weight overwrite occurred.
* **Baseline-Only Checkpoint Isolation**: All baseline probe weights are saved strictly under `artifacts/models/baselines/` (`halp_{model}_probe.pt`).
* **Unit Test Verification**: 13/13 pytest suite tests passing.

---

## 8. Remaining Limitations & Unresolved Blockers

1. **HalLoc Token Annotation Absence**: The benchmark cohort does not possess genuine token-level multi-class hallucination annotations. Training HalLoc on this cohort is halted until either the HalLoc-150k dataset is mounted or pre-trained VisualBERT weights are provided.
2. **VisER Internal Attention Cache**: Black-box feature vectors cannot substitute for internal logit-lens evidence gates; VisER remains classified as an **Adapted Grounding Proxy**.
3. **REFIND Modality Gap**: The Context Sensitivity Ratio was formulated for text LLMs; applying it to VLMs remains classified as a **Multimodal Adaptation**.
4. **Held-Out Test Lock**: The 1,073 test samples remain completely locked and unobserved.

