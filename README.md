# XMCA-Net: Cross-Modal Consistency Attribution Network

[![Tests](https://img.shields.io/badge/tests-26%20passed-brightgreen.svg)]()
[![Hardware](https://img.shields.io/badge/hardware-Kaggle%20NVIDIA%20T4%20%C3%97%202-orange.svg)]()
[![Execution](https://img.shields.io/badge/execution-strict%20mode-blue.svg)]()

**XMCA-Net** is a research framework for fine-grained multimodal hallucination detection and diagnostic cause attribution in Vision-Language Models (VLMs).

---

## 1. Core Architecture

Unlike standard detection approaches that produce ungrounded scalar confidence scores, XMCA-Net decomposes cross-modal consistency into three orthogonal diagnostic signals:

1. **Perception Consistency ($S_P$)**: Measures alignment between visual features and generated tokens (detects object hallucination and attribute misbinding).
2. **Retrieval Consistency ($S_R$)**: Measures factual coherence between retrieved/parametric knowledge and generated entity claims.
3. **Utilization Consistency ($S_U$)**: Measures context fidelity against unconditioned language prior bias.
4. **Calibrated XMCA Detection Combiner**: Maps $(S_P, S_R, S_U)$ to a well-calibrated hallucination probability.
5. **Independently Supervised Cause-Attribution Head**: Active strictly where ground-truth attribution annotations exist (e.g., POPE for perception, FaithScore for perception/utilization).

---

## 2. Fixed Research Requirements

- **Four Datasets**:
  - **POPE**: Polling-based Object Probing Evaluation (MSCOCO 2014)
  - **FaithScore**: Atomic Fact Descriptive Hallucination Evaluation (MSCOCO 2014)
  - **A-OKVQA**: Knowledge-based Visual Question Answering (MSCOCO 2017)
  - **InfoSeek**: Fine-Grained Entity Retrieval Visual Question Answering
- **Three VLM Families**:
  - **LLaVA**: `llava-hf/llava-1.5-7b-hf` (pinned revision: `a272c74`)
  - **Qwen2-VL**: `Qwen/Qwen2-VL-7B-Instruct` (pinned revision: `847be15`)
  - **InternVL**: `OpenGVLab/InternVL2-8B` (pinned revision: `9826f68`)
- **Execution Target**:
  - **Kaggle NVIDIA T4 × 2 only**.
  - **Zero local inference or training on macOS/CPU**.
- **Progressive Evaluation Tiers**:
  - Exactly 100, 300, 600, and 1,000 samples per dataset.
  - **Strictly matched sample IDs** across all VLM backbones.
- **Strict Integrity Rules**:
  - No synthetic data, dummy scores, mathematical proxy implementations, or fallback fall-throughs.
  - Unimplemented components raise explicit `NotImplementedError`.

---

## 3. Directory Layout

```text
xmca-net/
├── configs/
│   ├── config.yaml          # Project, hardware, execution, and path configs
│   ├── datasets.yaml        # Benchmark datasets, splits, hashes, schemas
│   └── models.yaml          # Pinned VLM families, revisions, and GPU sharding
├── data/
│   ├── raw/                 # Downloaded raw annotations (verified via SHA-256)
│   └── images/              # Benchmark visual assets
├── artifacts/
│   ├── manifests/           # Matched sample ID manifests for 100/300/600/1000 tiers
│   ├── corpora/             # External retrieval knowledge corpora
│   ├── generations/         # Model outputs per VLM and dataset
│   ├── features/            # Extracted diagnostic consistency features
│   ├── models/              # Checkpoints for combiner and attribution head
│   └── reports/             # Evaluation metrics and verification outputs
├── src/
│   ├── data/                # Dataset schemas and strict loaders
│   ├── generation/          # VLM generation pipeline (Kaggle dual-T4)
│   ├── perception/          # Perception Consistency (S_P)
│   ├── retrieval/           # Retrieval Consistency (S_R)
│   ├── utilization/         # Utilization Consistency (S_U)
│   ├── combiner/            # Calibrated XMCA combiner & attribution head
│   ├── baselines/           # Comparative baseline models
│   ├── evaluation/          # Metrics (AUROC, PR-AUC, ECE, Attribution Acc)
│   └── utils/               # Config loaders, manifest validator, environment check
├── tests/                   # Pytest suite enforcing strict mode and schema contracts
├── notebooks/
│   └── XMCA_BOOTSTRAP_KAGGLE.ipynb # Kaggle dual-T4 hardware & environment verification
├── scripts/                 # Utility and automation scripts
├── docs/                    # Research architecture, protocols, and execution plans
│   ├── RESEARCH_ARCHITECTURE.md
│   ├── DATASET_PROTOCOL.md
│   ├── VLM_INDEPENDENCE.md
│   ├── KAGGLE_EXECUTION_PLAN.md
│   └── IMPLEMENTATION_STATUS.md
├── requirements.txt         # Pinned python dependencies
├── README.md
└── .gitignore
```

---

## 4. Verification & Testing

Run unit tests locally (validation of configurations, schemas, and strict-mode constraints):

```bash
pytest -v
```

All 26 foundation tests should pass without requiring GPU access or local inference.
