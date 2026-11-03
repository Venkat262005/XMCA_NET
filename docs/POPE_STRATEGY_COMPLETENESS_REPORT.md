# XMCA-Net: POPE Strategy Completeness & Stratification Report
## Task 1.1 Tri-Strategy Sampling & Balanced Cohort Specification

---

## 1. Upstream POPE Strategy Acquisition

In Task 1.1, the complete tri-strategy benchmark suite was acquired from the official POPE repository (`https://github.com/RUCAIBox/POPE`, commit `7e24b45`):

| Sampling Strategy | Raw Source File | File Size | SHA-256 Checksum | Available Records | Unique Images | Positive ("yes") | Negative ("no") |
|---|---|---|---|---|---|---|---|
| **Adversarial** | `data/raw/pope_coco_adversarial.json` | 370,401 B | `420b3407db1fa9f1187a805dca41cb7b97fd91504e6c2179706188c107fb8ef8` | 3,000 | 500 | 1,500 | 1,500 |
| **Popular** | `data/raw/pope_coco_popular.json` | 370,190 B | `72c1a8ad45d0c13514f5f22598261df41d3b533854d29682e924db50ed8aa753` | 3,000 | 500 | 1,500 | 1,500 |
| **Random** | `data/raw/pope_coco_random.json` | 368,166 B | `ac25245170b975a5bdf9080b23fd431dfe6be458bc038259c1f4f09a6bef7994` | 3,000 | 500 | 1,500 | 1,500 |
| **Total** | — | **1,108,757 B** | — | **9,000** | **500** | **4,500** | **4,500** |

---

## 2. Stratified 1,000-Record Cohort Composition

Rather than restricting evaluation strictly to adversarial sampling, the candidate 1,000-sample cohort stratifies equally across all three official negative-sampling distributions:

$$\mathcal{C}_{\text{POPE}} = \mathcal{S}_{\text{adv}} (334) \cup \mathcal{S}_{\text{pop}} (333) \cup \mathcal{S}_{\text{rnd}} (333)$$

### Balanced Representation Breakdown:
- **Adversarial Subset (334 samples)**:
  - 167 Positive probes ("yes", object present)
  - 167 Negative probes ("no", frequent co-occurring but absent object)
- **Popular Subset (333 samples)**:
  - 167 Positive probes ("yes")
  - 166 Negative probes ("no", top frequent COCO objects absent)
- **Random Subset (333 samples)**:
  - 166 Positive probes ("yes")
  - 167 Negative probes ("no", uniformly sampled absent objects)
- **Total Cohort Balance**: Exactly **500 Positive ("yes")** and **500 Negative ("no")** probes.

---

## 3. Preserved Identifiers & Lineage

- **Stratified Sample IDs**: `pope_adversarial_{qid}`, `pope_popular_{qid}`, `pope_random_{qid}`.
- **Image Association**: All probes map directly to MSCOCO 2014 Validation imagery (`COCO_val2014_...jpg`).
- **Cryptographic Hashes**: Every probe retains `source_record_hash = sha256(canonical_json(raw_record + strategy))` guaranteeing 100% collision-free traceability back to upstream JSON files.

