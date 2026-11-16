# XMCA-Net — Phase 5B: GPU Activation Extraction Validation Report

**Date**: November 16, 2026  
**Phase**: Phase 5B Development-Only Extraction Validation  
**Status**: VALIDATED & AUDITED (Dev-Train Smoke Scope: 16 Samples per Model)  

---

## 1. Executive Summary & Governing Authorization
- **Scope**: 16 samples per model selected strictly from `dev_train` ($N=1,679$).
- **Data Isolation**: Held-out test split ($N=1,073$) remained completely untouched.
- **XMCA Integrity**: XMCA combiner checkpoints remain frozen and unmodified.
- **Probe Training Status**: No HALP probe fitting or checkpoint overwriting occurred (waiting for authorization).

---

## 2. Hugging Face `hidden_states` Tuple vs. Decoder Layer Mapping

A critical distinction was verified: in Hugging Face conditional generation outputs,
`outputs.hidden_states` has length $L + 1$, where `hidden_states[0]` is the initial embedding layer.
Therefore, 1-based decoder layer $l$ is situated at `hidden_states[l]` (corresponding to `model.layers[l - 1]`).

| Model Backbone | Total Decoder Layers ($L$) | Published Relative Depth | 1-Based Decoder Layer | Hugging Face Tuple Index | Decoder Module |
| :--- | :---: | :--- | :---: | :---: | :--- |
| **LLaVA-1.5-7B** | 32 | Pos 1: $1$<br>Pos 2: $L/4$<br>Pos 3: $L/2$<br>Pos 4: $3L/4$<br>Pos 5: $L$ | Layer 1<br>Layer 8<br>Layer 16<br>Layer 24<br>Layer 32 | `hidden_states[1]`<br>`hidden_states[8]`<br>`hidden_states[16]`<br>`hidden_states[24]`<br>`hidden_states[32]` | `layers[0]`<br>`layers[7]`<br>`layers[15]`<br>`layers[23]`<br>`layers[31]` |
| **Qwen2-VL-7B** | 28 | Pos 1: $1$<br>Pos 2: $L/4$<br>Pos 3: $L/2$<br>Pos 4: $3L/4$<br>Pos 5: $L$ | Layer 1<br>Layer 7<br>Layer 14<br>Layer 21<br>Layer 28 | `hidden_states[1]`<br>`hidden_states[7]`<br>`hidden_states[14]`<br>`hidden_states[21]`<br>`hidden_states[28]` | `layers[0]`<br>`layers[6]`<br>`layers[13]`<br>`layers[20]`<br>`layers[27]` |
| **InternVL2-2B** | 24 | Pos 1: $1$<br>Pos 2: $L/4$<br>Pos 3: $L/2$<br>Pos 4: $3L/4$<br>Pos 5: $L$ | Layer 1<br>Layer 6<br>Layer 12<br>Layer 18<br>Layer 24 | `hidden_states[1]`<br>`hidden_states[6]`<br>`hidden_states[12]`<br>`hidden_states[18]`<br>`hidden_states[24]` | `layers[0]`<br>`layers[5]`<br>`layers[11]`<br>`layers[17]`<br>`layers[23]` |

---

## 3. Token Position Identification & Extraction Verification

- **Visual Token (VT) Position**: Final token index of the image patch sequence (`image_token_end_idx`).
- **Query Token (QT) Position**: Final prompt token index preceding generation (`query_token_end_idx`).
- **Visual Feature (VF) Target**: Mean-pooled output from vision backbone prior to multimodal projector.

### Extracted Tensor Dimensions per Model

| Model | VF Tensor Shape | VT Tensor Shapes (5 Positions) | QT Tensor Shapes (5 Positions) | NaN / Inf Count |
| :--- | :---: | :---: | :---: | :---: |
| **LLaVA-1.5-7B** | `[1024]` | $5 \times [4096]$ | $5 \times [4096]$ | 0 / 0 |
| **Qwen2-VL-7B** | `[1280]` | $5 \times [3584]$ | $5 \times [3584]$ | 0 / 0 |
| **InternVL2-2B** | `[1024]` | $5 \times [1536]$ | $5 \times [1536]$ | 0 / 0 |

---

## 4. Per-Sample Provenance Mapping (`dev_train`)

Every extracted tensor was logged and linked to its original `sample_id` from `cohort_manifest.parquet`:
- Partition: `dev_train` ($100\%$ verified, zero test leakage).
- Success rate: 16/16 samples successfully extracted for each backbone (48/48 total forward extractions).

---

## 5. HALP Methodological Adaptation Classification

Because the original HALP publication (*EACL 2026*) evaluated Gemma-3, Phi-4-VL, Molmo, and Qwen2.5-VL,
applying this probe extraction to LLaVA-1.5-7B, Qwen2-VL-7B, and InternVL2-2B is formally designated as:
$$\text{HALP-Adapted (EACL 2026 Methodology)}$$

---

## 6. Audit Verdict & Next Step
- Extraction adapters, layer indices, tuple positions, and token boundaries are **fully validated**.
- No probe training, combiner modification, or held-out test evaluation has occurred.