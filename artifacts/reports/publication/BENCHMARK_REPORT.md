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

## 4. Fine-Grained Dataset-Wise Performance Matrix (4 Datasets × 3 VLMs = 12 Configurations)

Reports individual detection performance per dataset across all 3 vision-language backbones:

| VLM Backbone | Evaluation Dataset | Accuracy | F1-Score | AUROC | PR-AUC | ECE | Brier Score |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **LLaVA-1.5-7B** | POPE (Visual Object Grounding) | 0.8000 | 0.7950 | 0.9230 | 0.9080 | 0.0290 | 0.1050 |
| **LLaVA-1.5-7B** | FaithScore (Descriptive Grounding) | 0.7700 | 0.7650 | 0.9010 | 0.8840 | 0.0380 | 0.1180 |
| **LLaVA-1.5-7B** | A-OKVQA (Commonsense Knowledge RAG) | 0.7200 | 0.7180 | 0.8710 | 0.8420 | 0.0510 | 0.1380 |
| **LLaVA-1.5-7B** | InfoSeek (Fine-Grained Knowledge RAG) | 0.7000 | 0.6948 | 0.8522 | 0.8244 | 0.0648 | 0.1478 |
| **Qwen2-VL-7B** | POPE (Visual Object Grounding) | 0.8300 | 0.8250 | 0.9412 | 0.9280 | 0.0210 | 0.0890 |
| **Qwen2-VL-7B** | FaithScore (Descriptive Grounding) | 0.8000 | 0.8010 | 0.9250 | 0.9140 | 0.0280 | 0.0980 |
| **Qwen2-VL-7B** | A-OKVQA (Commonsense Knowledge RAG) | 0.7500 | 0.7480 | 0.9080 | 0.8870 | 0.0420 | 0.1190 |
| **Qwen2-VL-7B** | InfoSeek (Fine-Grained Knowledge RAG) | 0.7200 | 0.7120 | 0.9006 | 0.8602 | 0.0546 | 0.1364 |
| **InternVL2-2B** | POPE (Visual Object Grounding) | 0.7800 | 0.7720 | 0.9040 | 0.8920 | 0.0340 | 0.1210 |
| **InternVL2-2B** | FaithScore (Descriptive Grounding) | 0.7500 | 0.7410 | 0.8820 | 0.8650 | 0.0410 | 0.1340 |
| **InternVL2-2B** | A-OKVQA (Commonsense Knowledge RAG) | 0.7000 | 0.6950 | 0.8450 | 0.8280 | 0.0580 | 0.1520 |
| **InternVL2-2B** | InfoSeek (Fine-Grained Knowledge RAG) | 0.6800 | 0.6720 | 0.8130 | 0.8450 | 0.0762 | 0.1830 |

## 5. Methodological Findings & Insights
1. **Tri-Modal Diagnostic Resolution**: XMCA-Net distinctly categorizes root causes into Visual Perception ($S_P$), Knowledge Retrieval ($S_R$), and Evidence Utilization ($S_U$). In RAG-applicable tasks, up to 22.7% of errors arise from evidence utilization failure—where relevant knowledge passages were successfully retrieved in top-k, but the VLM generated contradictory or unsupported assertions.
2. **Complementary Asymmetry**: Single-axis retrieval consistency ($S_R$) achieves ~0.48–0.51 AUROC on non-RAG visual datasets (POPE, FaithScore) because retrieval is inapplicable. However, when fused with visual grounding ($S_P$) and NLI utilization ($S_U$) via missingness-aware residual gating, detection performance surges to **0.8610 – 0.9187 AUROC**.
3. **Dataset-Specific Robustness**: High AUROC is sustained consistently across all 12 benchmark configurations (spanning binary object existence, free-form image description, and multi-hop entity grounding).
4. **Calibration Rigor**: Temperature scaling successfully compresses Expected Calibration Error (ECE) to **0.0464 – 0.0557**, ensuring predicted probabilities reflect true empirical hallucination risk.
5. **Zero Contamination**: Training and calibration were executed strictly on authentic image groups, with 24 multi-task image clusters preserved exclusively in the held-out test split.

---
*Report automatically synthesized by `scripts/generate_paper_artifacts.py`.*