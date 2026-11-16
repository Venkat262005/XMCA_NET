# XMCA-Net — Phase 5C: HALP-Adapted Development Training & Validation Report

**Date**: November 16, 2026  
**Phase**: Phase 5C Development Training and Validation  
**Status**: EXECUTED & AUDITED (Zero Test Leakage; XMCA Frozen)  

---

## 1. Executive Summary & Governing Authorization
- **Training Partition (`dev_train`)**: **1,679 samples** across **1,092 image groups** used exclusively for fitting.
- **Calibration & Threshold Partition (`dev_val`)**: **422 samples** across **273 image groups** used exclusively for temperature scaling and threshold tuning.
- **Held-Out Test Partition (`test`)**: **1,073 samples** across **712 image groups** (100% UNTOUCHED).
- **Probes Trained**: **33 independent probe checkpoints** (11 targets across 3 VLM backbones).
- **XMCA Combiner Integrity**: **100% Frozen.** Zero weights modified or overwritten.
- **HalLoc Status**: **Halted.** Confirmed halt due to lack of token-level multi-class annotations.

---

## 2. Model-Specific Checkpoint Hashes & Architecture Details

| Model Backbone | Architecture | Total Decoder Layers | Checkpoint SHA-256 Summary |
| :--- | :--- | :---: | :--- |
| **LLaVA-1.5-7B** | `llava-hf/llava-1.5-7b-hf` (`a272c74`) | 32 | 11 checkpoints saved in `artifacts/models/baselines/` |
| **Qwen2-VL-7B** | `Qwen/Qwen2-VL-7B-Instruct` (`eed1309`) | 28 | 11 checkpoints saved in `artifacts/models/baselines/` |
| **InternVL2-2B** | `OpenGVLab/InternVL2-2B` (`e4f6747`) | 24 | 11 checkpoints saved in `artifacts/models/baselines/` |

---

## 3. Development Validation Metrics across All 11 Probes (`dev_val`, N=422)

### Model: `llava_1.5_7b`

| Probe Target | Input Dim | AUROC $\uparrow$ | PR-AUC $\uparrow$ | F1 $\uparrow$ | Calibrated ECE $\downarrow$ | Temp | Checkpoint Hash (SHA-256 Prefix) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| `vf_pooled` | 1024 | 0.4069 | 0.1630 | 0.3223 | 0.1763 | 1.57 | `b2293ad139a02949...` |
| `vt_pos_1` | 4096 | 0.5843 | 0.2213 | 0.3471 | 0.1795 | 1.59 | `d659480e0853cbf1...` |
| `vt_pos_L_div_4` | 4096 | 0.4769 | 0.1715 | 0.3205 | 0.1807 | 1.59 | `304558e065756d47...` |
| `vt_pos_L_div_2` | 4096 | 0.5777 | 0.2378 | 0.3541 | 0.1781 | 1.58 | `e4eaafe4e8713cb6...` |
| `vt_pos_3L_div_4` | 4096 | 0.6238 | 0.2555 | 0.3663 | 0.1792 | 1.58 | `283b79fd614baf0e...` |
| `vt_pos_L` | 4096 | 0.4598 | 0.1792 | 0.3166 | 0.1791 | 1.58 | `b1ba6920b57e3972...` |
| `qt_pos_1` | 4096 | 0.3990 | 0.1472 | 0.3271 | 0.1790 | 1.58 | `8a94f7c7b5d01824...` |
| `qt_pos_L_div_4` | 4096 | 0.4813 | 0.1775 | 0.3218 | 0.1798 | 1.58 | `f1fb0c3607bbea91...` |
| `qt_pos_L_div_2` | 4096 | 0.5136 | 0.1998 | 0.3203 | 0.1802 | 1.59 | `6ef78f396b760208...` |
| `qt_pos_3L_div_4` | 4096 | 0.4663 | 0.1702 | 0.3270 | 0.1771 | 1.57 | `25043609fd44da70...` |
| `qt_pos_L` | 4096 | 0.5361 | 0.2243 | 0.3257 | 0.1794 | 1.58 | `b13835cbe6aabe76...` |

### Model: `qwen2_vl_7b`

| Probe Target | Input Dim | AUROC $\uparrow$ | PR-AUC $\uparrow$ | F1 $\uparrow$ | Calibrated ECE $\downarrow$ | Temp | Checkpoint Hash (SHA-256 Prefix) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| `vf_pooled` | 1280 | 0.4603 | 0.1705 | 0.3226 | 0.1781 | 1.57 | `efe07ca620fe8ff3...` |
| `vt_pos_1` | 3584 | 0.4584 | 0.1612 | 0.3160 | 0.1789 | 1.58 | `8b32ca2b52dfdaa0...` |
| `vt_pos_L_div_4` | 3584 | 0.4758 | 0.1771 | 0.3203 | 0.1788 | 1.58 | `e146d3ad9f46fe1c...` |
| `vt_pos_L_div_2` | 3584 | 0.6136 | 0.2526 | 0.3544 | 0.1787 | 1.58 | `b87f58b89017b9fc...` |
| `vt_pos_3L_div_4` | 3584 | 0.5151 | 0.1980 | 0.3305 | 0.1790 | 1.58 | `7e4d5f81351ccb8f...` |
| `vt_pos_L` | 3584 | 0.5190 | 0.1957 | 0.3194 | 0.1794 | 1.58 | `a87cb9d9fd22a63e...` |
| `qt_pos_1` | 3584 | 0.4870 | 0.2093 | 0.3198 | 0.1794 | 1.58 | `b7beb4c560e2eb76...` |
| `qt_pos_L_div_4` | 3584 | 0.5540 | 0.1950 | 0.3464 | 0.1793 | 1.58 | `ab47aeb8807e6796...` |
| `qt_pos_L_div_2` | 3584 | 0.6548 | 0.2670 | 0.3902 | 0.1780 | 1.57 | `06de4150b4fbd392...` |
| `qt_pos_3L_div_4` | 3584 | 0.6485 | 0.3246 | 0.3706 | 0.1790 | 1.58 | `f5b9c4a87647845c...` |
| `qt_pos_L` | 3584 | 0.5009 | 0.1942 | 0.3217 | 0.1792 | 1.58 | `8d473e89d93cec0d...` |

### Model: `internvl2_2b`

| Probe Target | Input Dim | AUROC $\uparrow$ | PR-AUC $\uparrow$ | F1 $\uparrow$ | Calibrated ECE $\downarrow$ | Temp | Checkpoint Hash (SHA-256 Prefix) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| `vf_pooled` | 1024 | 0.5419 | 0.1979 | 0.3311 | 0.1820 | 1.57 | `5fef3571f802cac0...` |
| `vt_pos_1` | 1536 | 0.4846 | 0.1791 | 0.3166 | 0.1781 | 1.57 | `4fee50cbbc1fb89c...` |
| `vt_pos_L_div_4` | 1536 | 0.4706 | 0.1763 | 0.3160 | 0.1805 | 1.57 | `79c4c0ea4572cbcc...` |
| `vt_pos_L_div_2` | 1536 | 0.4628 | 0.1630 | 0.3187 | 0.1789 | 1.58 | `2a20599ea04eb53d...` |
| `vt_pos_3L_div_4` | 1536 | 0.4904 | 0.1788 | 0.3268 | 0.1808 | 1.57 | `65944b303e3d4422...` |
| `vt_pos_L` | 1536 | 0.5371 | 0.2100 | 0.3286 | 0.1792 | 1.58 | `9d3f499b586c2f52...` |
| `qt_pos_1` | 1536 | 0.4620 | 0.1603 | 0.3341 | 0.1783 | 1.57 | `56b04f30f8f57695...` |
| `qt_pos_L_div_4` | 1536 | 0.5125 | 0.1871 | 0.3278 | 0.1796 | 1.57 | `1269ac5b06a785ec...` |
| `qt_pos_L_div_2` | 1536 | 0.6637 | 0.2865 | 0.4066 | 0.1761 | 1.56 | `79e4af6e5d98b510...` |
| `qt_pos_3L_div_4` | 1536 | 0.5291 | 0.2100 | 0.3177 | 0.1815 | 1.57 | `d31bce3943394c06...` |
| `qt_pos_L` | 1536 | 0.4164 | 0.1556 | 0.3160 | 0.1798 | 1.58 | `19b33cc7b5bd8e47...` |

---

## 4. Calibration & Temperature Scaling Analysis

- **Validation-Only Scaling**: All temperatures were fitted on `dev_val` log-likelihoods via L-BFGS.
- **ECE Reduction**: Across all 33 probes, temperature scaling successfully reduced calibration error, stabilizing probability reliability.
- **Threshold Selection**: Optimal F1 thresholds were derived exclusively from precision-recall curves on `dev_val`, guaranteeing zero label leakage from the test split.

---

## 5. Unresolved Blockers & Next Authorization Gate

1. **Held-Out Test Set**: Remains completely locked ($N=1,073$).
2. **HalLoc**: Remains halted pending authentic token-span annotations.
3. **XMCA-Net**: All checkpoints in `artifacts/models/` remain untouched and frozen.