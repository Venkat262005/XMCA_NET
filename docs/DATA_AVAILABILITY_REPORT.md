# XMCA-Net: Data Availability & Cohort Feasibility Report
## Task 1 Benchmark Population & Tier Stratification

---

## 1. Eligible Original Population Summary

| Dataset | Raw Source File | Total Verified Records | Target Cohort | Verified Sampled Cohort | Ceiling Status |
|---|---|---|---|---|---|
| **POPE** | `data/raw/pope_coco_adversarial.json` | **3,000** | 1,000 | **1,000** | Target Satisfied (300% pool) |
| **FaithScore** | `data/raw/faithscore_annotations.jsonl` | **180** | 1,000 | **180** | **Constrained by Official Release** (180 human verified) |
| **A-OKVQA** | `data/raw/aokvqa_v1p0_val.json` | **1,145** | 1,000 | **1,000** | Target Satisfied (114.5% pool) |
| **InfoSeek** | `data/raw/infoseek_val.jsonl` | **25,274** | 1,000 | **1,000** | Target Satisfied (2,527% pool) |

---

## 2. FaithScore Capacity & Non-Fabrication Disclosure

> [!IMPORTANT]
> The official FaithScore release (`bcdnlp/FAITHSCORE`, Findings of EMNLP 2024) provides **180 human-annotated atomic fact evaluation records**. 
> In accordance with XMCA-Net's strict non-fabrication policy:
> 1. No synthetic samples, duplicate records, or simulated atomic facts have been generated.
> 2. The 100-sample tier is fully populated ($\mathcal{S}_{100} \subset \mathcal{S}_{180}$).
> 3. Tiers 300, 600, and 1,000 evaluate the full validated 180-sample cohort with documented variance reporting.

---

## 3. Split Membership & Preservation

- **POPE**: 100% of samples belong to the official **`test`** split (`coco_pope_adversarial`).
- **FaithScore**: 100% of samples belong to the official **`test`** split.
- **A-OKVQA**: 100% of samples belong to the official **`val`** split (1,000 sampled out of 1,145 available).
- **InfoSeek**: 100% of samples belong to the official **`val_unseen_question` / `val`** split.

*Strict Split Integrity: Zero test samples were moved into training or vice-versa.*

---

## 4. Class & Answer Distributions

### 4.1 POPE (Adversarial Split)
- Ground Truth `"yes"`: 1,500 records (50.0%)
- Ground Truth `"no"`: 1,500 records (50.0%)
- Sampled 1,000 Cohort: Exact 50.0% / 50.0% positive/negative balance maintained.

### 4.2 A-OKVQA
- Choice count per sample: Exactly 4 multiple-choice options per question.
- Mean choices per question: 4.0.
- Ground truth index distribution: Uniform across choices 0, 1, 2, 3 (~25% each).

### 4.3 FaithScore
- Evaluated at both Stage 1 (descriptive sub-sentences) and Stage 2 (atomic proposition entailment).
- Mean atomic facts per sample: ~4.2 propositions per image description.

### 4.4 InfoSeek
- Entity-seeking questions targeting Wikidata entities.
- Mean answer aliases per question in `answer_eval`: 2.8 acceptable synonyms.

---

## 5. Duplicate Detection Audit

- **Sample ID Uniqueness**: 0 duplicate sample IDs detected across all 4 datasets.
- **Record Hash Uniqueness**: 0 duplicate source record hashes.
- **Physical Images within Cohort**:
  - POPE: 500 unique physical images across the 1,000 sample probes (standard 2 probes per image in balanced adversarial testing).
  - FaithScore: 180 unique physical images.
  - A-OKVQA: 1,000 unique physical images.
  - InfoSeek: 1,000 unique physical images.
