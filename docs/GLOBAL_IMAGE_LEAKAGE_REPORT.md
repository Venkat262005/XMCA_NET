# XMCA-Net: Global Image Leakage & Combiner Partition Audit
## Task 1.1 Physical Image Grouping & Split Conflict Resolution

---

## 1. Theoretical Rationale: Combiner Leakage Mechanics

While freezing Vision-Language Model (VLM) backbones prevents gradient-based parameter corruption, it does **not** protect downstream diagnostic classifiers (e.g. the calibrated XMCA combiner or attribution head) from image-level data leakage.

If physical image $I$ appears in combiner training (e.g., associated with a question from a training split), and the exact same physical image $I$ appears in held-out test evaluation (e.g., associated with a POPE or FaithScore question), the combiner can exploit:
1. Visual feature memorization of image $I$.
2. Perceptual artifact overfitting rather than generalizable consistency attribution.

To ensure unimpeachable scientific validity, **all questions referring to the same physical image must be assigned to the exact same partition group**.

---

## 2. Cross-Dataset Overlap Findings

Auditing all 3,174 cohort samples across POPE (tri-strategy), FaithScore, A-OKVQA, and InfoSeek identified **2,379 unique physical image entities**:

```
+-------------------------------------------------------------------------+
|                  Cohort Physical Image Distribution                     |
+-------------------------------------------------------------------------+
|  Benchmark            | Cohort Questions | Unique Physical Images       |
|-----------------------|------------------|------------------------------|
|  POPE (Tri-Strategy)  | 1,000            | 453 unique COCO 2014 images  |
|  FaithScore           | 174              | 30 unique COCO 2014 images   |
|  A-OKVQA              | 1,000            | 981 unique COCO 2017 images  |
|  InfoSeek             | 1,000            | 1,000 unique OVEN images     |
|-----------------------|------------------|------------------------------|
|  Total Unique Images  | 3,174            | 2,379 global physical groups |
+-------------------------------------------------------------------------+
```

### Pairwise Physical Image Intersections:
- **POPE $\cap$ FaithScore**: 1 shared physical image (`COCO_val2014_000000506483.jpg`).
- **POPE $\cap$ A-OKVQA**: 24 shared physical images (MSCOCO 2014 / 2017 overlaps: `125572`, `369541`, `3845`, `536073`, `549390`, `353180`, `500257`, `222118`, `396338`, `472375`, etc.).
- **FaithScore $\cap$ A-OKVQA**: 0 shared physical images.
- **InfoSeek $\cap$ COCO Datasets**: 0 shared physical images (OVEN disjoint entity collection).

---

## 3. Global Image Group Manifest Contract

The immutable grouping contract is saved at `artifacts/manifests/global_image_group_manifest.parquet`:

```text
Columns:
- global_group_id: Unique physical cluster identifier (grp_00000 to grp_02378)
- global_image_id: Canonical image key (e.g. coco_000000506483, oven_04990048)
- datasets: Comma-separated list of datasets referencing this image
- dataset_count: Integer number of datasets referencing this image
- official_splits: Set of official splits referencing this image
- question_count: Total evaluation queries linked to this physical image
- combiner_partition: "held_out_test" | "validation" | "train_eligible"
- split_conflict: Boolean indicating whether official splits conflict (test vs. val)
- eligible_for_combiner_training: Strict boolean flag
```

---

## 4. Leakage Mitigation Rules

1. **Held-Out Test Primacy**: Any physical image associated with an official test benchmark (POPE or FaithScore) is permanently assigned `combiner_partition = "held_out_test"`.
2. **Strict Training Ineligibility**: All 24 physical images exhibiting split conflict (`split_conflict = True`), as well as all POPE and FaithScore test images, are flagged with:
   $$\texttt{eligible\_for\_combiner\_training} = \text{False}$$
3. **Combiner Training Guarantee**: In Task 8, the calibrated XMCA combiner will be trained strictly on images with $\texttt{eligible\_for\_combiner\_training} = \text{True}$. This guarantees that zero test images ever enter combiner parameter optimization or threshold calibration.
4. **Preservation of Official Splits**: Zero records were modified, re-split, or reassigned across benchmarks.

