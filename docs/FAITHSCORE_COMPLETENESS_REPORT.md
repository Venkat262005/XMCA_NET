# XMCA-Net: FaithScore Completeness & Dual-Resource Audit
## Task 1.1 Benchmark Investigation & Granularity Audit

---

## 1. Overview of FaithScore Resources

The official FaithScore repository (`https://github.com/bcdnlp/FAITHSCORE`, Findings of EMNLP 2024, Jing et al.) provides two distinct categories of evaluation resources:

```
                          +------------------------------------------+
                          |        bcdnlp/FAITHSCORE Resources       |
                          +------------------------------------------+
                                               |
                  +----------------------------+----------------------------+
                  |                                                         |
                  v                                                         v
   [Human Ground-Truth Benchmark]                          [Automated Metric Testbeds]
   - File: `annotation.jsonl`                              - Described in Jing et al. (EMNLP 2024)
   - Scope: 180 lines (174 unique records)                 - Scope: LLaVA-1k & MSCOCO-Cap
   - Images: 30 physical MSCOCO val2014 images             - Ground Truth: None (Model outputs evaluated
   - Supervision: Verified atomic-fact human labels                        by OFA / LLaVA metric)
   - Role in XMCA: Evaluates cause attribution             - Role in XMCA: Un-annotated response testbeds
```

---

## 2. Human Ground-Truth Resource Audit (`annotation.jsonl`)

- **Repository Path**: `bcdnlp/FAITHSCORE/annotation.jsonl`
- **File Size**: 203,676 bytes
- **SHA-256**: `7d68dce9641f92fcdabd6e252cf12ee6ec9b6c1d5c4d306aef83666d299f1b7b`
- **Total Lines**: 180 evaluation records
- **Duplicate Audit**: Deduplication revealed 6 exact duplicate lines:
  - `000000258285`: 2 occurrences
  - `000000225738`: 2 occurrences
  - `000000164255`: 2 occurrences
  - `000000473210`: 2 occurrences
  - `000000214367`: 2 occurrences
  - `000000431165`: 2 occurrences
- **Unique Records**: Exactly **174 unique ground-truth records**.
- **Physical Image Distribution**: Spans exactly **30 unique physical MSCOCO val2014 images** (~5.8 model answer evaluations per physical image).
- **Attribution Labels**:
  - `stage 1`: Human binary judgment of descriptive vs. subjective sub-sentences.
  - `stage 2`: Fine-grained human verification (0 or 1) of individual atomic factual statements extracted from the description.

---

## 3. Automated Evaluation Testbeds (LLaVA-1k & MSCOCO-Cap)

In Section 4 of Jing et al. (EMNLP 2024), the authors construct automated evaluation testbeds:
1. **LLaVA-1k**: 1,000 instruction-following queries prompting candidate LVLMs for detailed image descriptions on MSCOCO images.
2. **MSCOCO-Cap**: Standard image caption generation prompts across MSCOCO validation images.

### Integrity Assessment:
- **Absence of Independent Human Ground Truth**: These benchmarks consist of prompt sets and model-generated candidate texts scored by FaithScore's automated scoring pipeline (using OFA-visual entailment and LLaMA-2 sub-sentence decomposition). They do **not** possess independent human-annotated hallucination ground truth.
- **Protocol Separation**: In accordance with XMCA-Net's strict non-fabrication policy, un-annotated model outputs cannot be treated as ground-truth cause-attribution supervision. The 174 human-annotated records are preserved as the sole legitimate ground-truth attribution cohort.

---

## 4. Statistical Implication: Image-Level Grouping

Because the 174 unique human evaluations originate from 30 physical images:
1. **Clustered Observations**: Evaluations on the same image share identical visual features and perceptual grounding.
2. **Variance Estimation**: All downstream confidence intervals and error bars on FaithScore must utilize **cluster-robust standard errors** grouped by `global_image_id` rather than treating all 174 rows as independent and identically distributed (i.i.d.) observations.
3. **Cohort Representation**: The candidate cohort manifest (`artifacts/manifests/cohort_manifest.parquet`) maintains all 174 unique evaluations with explicit image grouping keys.
