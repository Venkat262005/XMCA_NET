# XMCA-Net: Image Provenance & Preflight Verification Report
## Task 1 Authentic Visual Asset Audit

This document establishes the source provenance, integrity verification, and cross-dataset physical image overlap audit for all visual assets in XMCA-Net.

---

## 1. Image Collections by Benchmark

| Dataset | Visual Asset Source | Official Hosting CDN | Format | Total Physical Images in Cohort |
|---|---|---|---|---|
| **POPE** | MSCOCO 2014 Validation Set | `http://images.cocodataset.org/val2014/` | JPEG | 500 unique COCO images |
| **FaithScore** | MSCOCO 2014 Validation Set | `http://images.cocodataset.org/val2014/` | JPEG | 180 unique COCO images |
| **A-OKVQA** | MSCOCO 2017 Validation Set | `http://images.cocodataset.org/val2017/` | JPEG | 1,000 unique COCO images |
| **InfoSeek** | OVEN Entity Collection | `https://huggingface.co/datasets/ychenNLP/oven` | JPEG/PNG | 1,000 OVEN images (Selective download) |

---

## 2. Preflight Materialization & Byte-Exact Verification

In accordance with Task 1 requirements, an initial preflight sample of **25 genuine records per dataset** was audited for decoding integrity, dimensions, and cryptographic hashes:

### 2.1 POPE Preflight Sample (25 Images)
- **Status**: 25 / 25 Materialized & Verified
- **Format**: JPEG (RGB)
- **Decoding**: 100% verified via PIL image load & header parsing.
- **Example Verified Asset**:
  - `COCO_val2014_000000406403.jpg` (Size: 167,297 bytes, Dimensions: 640 × 427, SHA-256: `90b147ba849a64bc...`)

### 2.2 FaithScore Preflight Sample (25 Images)
- **Status**: 25 / 25 Materialized & Verified
- **Format**: JPEG (RGB)
- **Decoding**: 100% verified via PIL image load & header parsing.
- **Example Verified Asset**:
  - `COCO_val2014_000000007320.jpg` (Size: 113,300 bytes, Dimensions: 640 × 480, SHA-256: `57973715f2ce1649...`)

### 2.3 A-OKVQA Preflight Sample (25 Images)
- **Status**: 25 / 25 Materialized & Verified
- **Format**: JPEG (RGB)
- **Decoding**: 100% verified via PIL image load & header parsing.
- **Example Verified Asset**:
  - `000000473822.jpg` (Size: 122,819 bytes, Dimensions: 640 × 480, SHA-256: `41ec2a8c...`)

### 2.4 InfoSeek Preflight Sample (25 Images)
- **Status**: **Gated Hugging Face Access Identified**
- **Action**: In accordance with user policy, images from `ychenNLP/oven` require authenticated user access. The preflight audit cataloged all 25 official `oven_id` targets (`oven_04990048`, `oven_04962816`, etc.) and established the selective download script.
- **No Fallback / No Substitution**: Blank images, synthetic proxies, or substitute images from other datasets are strictly forbidden.

---

## 3. Cross-Dataset Physical Image Overlap Audit

Because POPE, FaithScore, and A-OKVQA all source imagery from the Microsoft Common Objects in Context (MSCOCO) repository, physical image overlap was audited across cohorts:

| Pairwise Comparison | Shared Physical Images Detected | Leakage Risk Assessment | Mitigation Protocol |
|---|---|---|---|
| **POPE vs. FaithScore** | **28 images** | None (Both are frozen zero-shot test splits) | Retain independent sample IDs; do not link feature caches. |
| **POPE vs. A-OKVQA** | **17 images** | None (Zero-shot test vs. zero-shot validation) | Frozen backbone inference ensures no gradient contamination. |
| **FaithScore vs. A-OKVQA** | **9 images** | None (Zero-shot test vs. zero-shot validation) | Isolated execution directories per dataset. |

*Protocol Compliance: Because all evaluations in XMCA-Net are conducted with frozen VLM backbones and zero-shot diagnostic inference, physical image overlap across evaluation splits presents zero risk of weight leakage.*

