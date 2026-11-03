# XMCA-Net: Dataset & Sampling Protocol

This protocol specifies the data acquisition, validation, sampling, and attribution ground-truth rules for the four benchmark datasets in XMCA-Net.

---

## 1. Benchmark Suite Overview

The evaluation suite consists of **exactly four datasets**, chosen to probe distinct aspects of multimodal perception, knowledge retrieval, and context utilization:

| Dataset | Full Name | Primary Task | Sample Splits | Ground-Truth Attribution? |
|---|---|---|---|---|
| **POPE** | Polling-based Object Probing Evaluation | Object Hallucination Detection | `test` (adversarial) | Yes (Perception) |
| **FaithScore** | FaithScore Atomic Fact Evaluation | Descriptive Hallucination | `test` | Yes (Perception & Utilization) |
| **A-OKVQA** | Augmented OK-VQA | Knowledge Visual Question Answering | `val` | No (Outcome label only) |
| **InfoSeek** | Information-Seeking VQA | Fine-Grained Entity Retrieval VQA | `val` | No (Outcome label only) |

---

## 2. Progressive Evaluation Tiers

To ensure computational feasibility on dual NVIDIA T4 GPUs and progressive validation of scaling trends, evaluation is conducted at four fixed sample tiers:

$$\mathcal{T} \in \{100, 300, 600, 1000\}$$

### Tier Ingestion Rules:
1. **Nested Subsets**: The 100-sample subset is a strict prefix/subset of the 300-sample subset, which is a subset of 600, which is a subset of 1,000.
   $$\mathcal{S}_{100} \subset \mathcal{S}_{300} \subset \mathcal{S}_{600} \subset \mathcal{S}_{1000}$$
2. **Cross-Model ID Matching**: The exact same sample IDs ($\mathcal{S}_k$) must be evaluated across all three VLM backbones (LLaVA, Qwen2-VL, InternVL). Divergent sample sets across models are strictly invalid and rejected by `src/utils/manifest.py`.
3. **Deterministic Sampling**: Stratified pseudo-random sampling is seeded with `random_seed: 42`. The resulting sample IDs are persisted to `artifacts/manifests/` prior to any model inference.

---

## 3. Dataset Specifications

### 3.1 POPE (Polling-based Object Probing Evaluation)
- **Source**: `https://github.com/RUCAIBox/POPE`
- **Images**: MSCOCO 2014 Validation Set.
- **Focus**: Perception consistency ($S_P$).
- **Format**: Binary visual object probe questions ("Is there a dog in the image?"). Ground truth labels are strictly `"yes"` or `"no"`.
- **Attribution Role**: Direct ground truth for perception failures (hallucinating absent objects).

### 3.2 FaithScore
- **Source**: `https://github.com/j-min/FaithScore`
- **Images**: MSCOCO 2014 Validation Set.
- **Focus**: Multi-sentence descriptive hallucination with atomic proposition decomposition.
- **Format**: Image caption / description generation evaluated at atomic fact granularity.
- **Attribution Role**: Fine-grained atomic attribution labels distinguishing between visual perception errors and ungrounded language hallucination.

### 3.3 A-OKVQA
- **Source**: `https://allenai.org/data/a-okvqa`
- **Images**: MSCOCO 2017 Validation Set.
- **Focus**: Knowledge-grounded visual reasoning requiring external world knowledge beyond direct visual appearance.
- **Format**: Multiple-choice and open-ended questions with human rationale annotations.
- **Attribution Role**: Evaluates knowledge retrieval consistency ($S_R$) and utilization consistency ($S_U$). Outcome accuracy only; no fine-grained attribution ground truth.

### 3.4 InfoSeek
- **Source**: `https://github.com/google-research-datasets/infoseek`
- **Images**: OVEN / ImageNet-based fine-grained entity imagery.
- **Focus**: Fine-grained information seeking tied to structured knowledge bases (Wikidata/Wikipedia).
- **Format**: Visual entity questions requiring exact entity retrieval.
- **Attribution Role**: Primary benchmark for retrieval consistency ($S_R$). Outcome accuracy only.

---

## 4. Integrity and Non-Fabrication Rules

1. **No Synthetic Data**: No synthetically manufactured test sets, fake ground-truth labels, or pseudo-labels are permitted.
2. **Checksum Verification**: Raw downloads must be verified with SHA-256 hashes recorded in `configs/datasets.yaml`.
3. **Schema Enforcement**: All samples must pass Pydantic schema validation (`src/data/schema.py`) prior to inclusion in evaluation manifests.
