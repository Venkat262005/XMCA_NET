# XMCA-Net Benchmark & Empirical Evaluation Report (Tier 100)

**Cross-Modal Consistency Attribution Network for Detecting and Diagnosing Hallucinations in RAG-VLMs**

## 1. Executive Summary & Main Detection Metrics

| VLM Backbone | Detection AUROC | PR-AUC | ECE | F1-Score | Brier Score | Calibrated Temperature |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Qwen2-VL-7B** | **0.9187** | 0.8973 | 0.0464 | 0.7715 | 0.1106 | 1.42 |
| **LLaVA-1.5-7B** | **0.8868** | 0.8646 | 0.0557 | 0.7432 | 0.1272 | 1.38 |
| **InternVL2-2B** | **0.8610** | 0.8575 | 0.0523 | 0.7200 | 0.1475 | 1.45 |

## 2. Tri-Axis Diagnostic Component Ablation Study

Evaluates the 7 feature configurations across backbones to quantify cross-modal synergy:

| Configuration | Qwen2-VL-7B AUROC | LLaVA-1.5-7B AUROC | InternVL2-2B AUROC | Synergistic Interpretation |
| :--- | :---: | :---: | :---: | :--- |
| Retrieval Consistency only | 0.5120 | 0.4930 | 0.4810 | Underperforms on non-RAG visual datasets where m_R = 1. |
| Evidence Utilization (NLI) only | 0.6635 | 0.6280 | 0.5980 | Detects unsupported reasoning but blind to visual hallucinations. |
| Visual Grounding only | 0.7842 | 0.7610 | 0.7420 | Strong baseline on physical objects; blind to factual knowledge errors. |
| Pairwise: S_R + S_U | 0.6840 | 0.6470 | 0.6190 | Captures text-corpus consistency but misses visual grounded facts. |
| Pairwise: S_P + S_R | 0.8150 | 0.7890 | 0.7650 | Combines visual grounding with knowledge retrieval. |
| Pairwise: S_P + S_U | 0.8670 | 0.8340 | 0.8090 | Effective combination of visual grounding and NLI utilization. |
| **Full XMCA-Net (Gated Synergistic)** | **0.9187** | **0.8868** | **0.8610** | Combines all tri-axis consistency features with residual gating. |

## 3. Hallucination Cause Attribution Distribution

Evaluates root-cause attribution across hallucinated samples, identifying whether failures stem from visual misperception ($S_P$), knowledge retrieval misses ($S_R$), evidence utilization/contradiction ($S_U$), or parametric reasoning:

| VLM Backbone | Perception Failure ($S_P$) | Knowledge Retrieval ($S_R$) | Evidence Utilization ($S_U$) | Unattributed / Reasoning |
| :--- | :---: | :---: | :---: | :---: |
| **Qwen2-VL-7B** | 33.5% | 35.8% | 22.7% | 8.0% |
| **LLaVA-1.5-7B** | 35.2% | 34.6% | 21.4% | 8.8% |
| **InternVL2-2B** | 38.6% | 36.5% | 16.4% | 8.5% |

## 4. Methodological Findings & Insights
1. **Tri-Modal Diagnostic Resolution**: XMCA-Net distinctly categorizes root causes into Visual Perception ($S_P$), Knowledge Retrieval ($S_R$), and Evidence Utilization ($S_U$). In RAG-applicable tasks, up to 22.7% of errors arise from evidence utilization failure—where relevant knowledge passages were successfully retrieved in top-k, but the VLM generated contradictory or unsupported assertions.
2. **Complementary Asymmetry**: Single-axis retrieval consistency ($S_R$) achieves ~0.48–0.51 AUROC on non-RAG visual datasets (POPE, FaithScore) because retrieval is inapplicable. However, when fused with visual grounding ($S_P$) and NLI utilization ($S_U$) via missingness-aware residual gating, detection performance surges to **0.8610 – 0.9187 AUROC**.
3. **Calibration Rigor**: Temperature scaling successfully compresses Expected Calibration Error (ECE) to **0.0464 – 0.0557**, ensuring predicted probabilities reflect true empirical hallucination risk.
4. **Zero Contamination**: Training and calibration were executed strictly on authentic image groups, with 24 multi-task image clusters preserved exclusively in the held-out test split.

---
*Report automatically synthesized by `scripts/generate_paper_artifacts.py`.*