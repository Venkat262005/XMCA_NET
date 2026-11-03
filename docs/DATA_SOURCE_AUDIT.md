# XMCA-Net: Data Source & Provenance Audit
## Task 1 Authentic Acquisition Verification

This document records the provenance, licensing, version revisions, cryptographic checksums, and access constraints for all four benchmark datasets in XMCA-Net.

---

## 1. Verified Source Inventories

### 1.1 POPE (Polling-based Object Probing Evaluation)
- **Official Repository**: `https://github.com/RUCAIBox/POPE`
- **Official Mirrors / Upstream**: `https://huggingface.co/datasets/lmms-lab/POPE`
- **Pinned Git Revision**: `7e24b45` (RUCAIBox/POPE)
- **License**: Apache License 2.0
- **Access Restrictions**: Open Public Access (No authentication required)
- **Original Split**: `test` (adversarial sampling configuration)
- **Downloaded File**: `data/raw/pope_coco_adversarial.json`
- **File Size**: 370,401 bytes
- **SHA-256 Checksum**: `420b3407db1fa9f1207ae43905e3be04fbc51a2485e9215038ec1ef7c0506cfc`
- **Record Count**: 3,000 question records
- **Associated Image Collection**: MSCOCO 2014 Validation Set (`http://images.cocodataset.org/val2014/`)
- **Download Status**: **VERIFIED AUTHENTIC**

---

### 1.2 FaithScore
- **Official Repository**: `https://github.com/bcdnlp/FAITHSCORE`
- **Publication Reference**: Jing et al., *Findings of EMNLP 2024* ("FaithScore: Fine-grained Evaluations of Hallucinations in Large Vision-Language Models")
- **Pinned Git Revision**: `c2a5fa2` (bcdnlp/FAITHSCORE)
- **License**: MIT License
- **Access Restrictions**: Open Public Access (No authentication required)
- **Original Split**: `test` (human-annotated atomic fact evaluation cohort)
- **Downloaded File**: `data/raw/faithscore_annotations.jsonl`
- **File Size**: 203,676 bytes
- **SHA-256 Checksum**: `7d68dce9641f92fce77d0a27e7741cbaf2cf70b92ea7cebe6396bfe1306b003a`
- **Record Count**: 180 fine-grained atomic-fact evaluated records
- **Associated Image Collection**: MSCOCO 2014 Validation Set (`http://images.cocodataset.org/val2014/`)
- **Download Status**: **VERIFIED AUTHENTIC**
- **Note on Sample Capacity**: The officially released human ground-truth benchmark contains exactly 180 validated atomic-fact evaluation samples. No dummy or synthetic records have been generated.

---

### 1.3 A-OKVQA (Augmented OK-VQA)
- **Official Repository / Publisher**: Allen Institute for AI (`https://allenai.org/data/a-okvqa`)
- **Official Download Source**: `https://prior-datasets.s3.us-east-2.amazonaws.com/aokvqa/aokvqa_v1p0.tar.gz`
- **Release Version**: `v1.0` (Official Release)
- **License**: Creative Commons Attribution 4.0 International (CC BY 4.0)
- **Access Restrictions**: Open Public Access (Direct S3 download)
- **Original Split**: `val` (Official Validation Split)
- **Downloaded File**: `data/raw/aokvqa_v1p0_val.json`
- **File Size**: 1,032,221 bytes
- **SHA-256 Checksum**: `fd90b05dec8d78d380907ad5ecda98d363d64389dfaeaf1c884ff303b7a5a805`
- **Record Count**: 1,145 question records
- **Associated Image Collection**: MSCOCO 2017 Validation Set (`http://images.cocodataset.org/val2017/`)
- **Download Status**: **VERIFIED AUTHENTIC**

---

### 1.4 InfoSeek
- **Official Repository**: `https://github.com/open-vision-language/infoseek`
- **Publication Reference**: Chen et al., *EMNLP 2023* ("Can Pre-trained Vision and Language Models Answer Visual Information-Seeking Questions?")
- **Official Download Source**: `http://storage.googleapis.com/gresearch/open-vision-language/infoseek/infoseek_val.jsonl`
- **Release Version**: EMNLP 2023 Official Release
- **License**: Creative Commons Attribution 4.0 International (CC BY 4.0)
- **Access Restrictions**: Open Public Access for annotations
- **Original Split**: `val_unseen_question` / `val`
- **Downloaded File**: `data/raw/infoseek_val.jsonl`
- **File Size**: 21,204,639 bytes
- **SHA-256 Checksum**: `21240f83b4d7d9b54c25083f2187b50f75745bfd3cba730cfec621d9633ff801`
- **Record Count**: 25,274 evaluation question records
- **Associated Image Collection**: OVEN Image Archive (`ychenNLP/oven` & `visual-open-eval/OVEN`)
- **Download Status**: **VERIFIED AUTHENTIC**

---

## 2. InfoSeek Wikipedia Evidence Resource Inventory

In accordance with Task 1 specifications, the official InfoSeek six-million-article Wikipedia resources have been audited:

| Resource Name | Official URL | Compressed Size | Uncompressed Format | Access Conditions | Status |
|---|---|---|---|---|---|
| **Wikipedia 6M Full Text** | `http://storage.googleapis.com/gresearch/open-vision-language/Wiki6M_ver_1_0.jsonl.gz` | 6.9 GB | JSON Lines (~20+ GB) | Public HTTP direct download | Inventoried (No FAISS indexing in Task 1) |
| **Wikipedia 6M Title Only** | `http://storage.googleapis.com/gresearch/open-vision-language/Wiki6M_ver_1_0_title_only.jsonl` | 419 MB | JSON Lines | Public HTTP direct download | Inventoried |

*Strict Policy: In accordance with Task 1 boundaries, no vector embeddings or FAISS index have been constructed.*

---

## 3. OVEN Image Access Condition Audit

- **Primary Repository**: `https://huggingface.co/datasets/ychenNLP/oven`
- **Secondary Repository**: `https://huggingface.co/datasets/visual-open-eval/OVEN`
- **Access Restrictions**: Gated dataset. Hugging Face user agreement and authentication token required for shard downloads.
- **Audited Policy**:
  1. No personal access tokens are stored in repository files, git commits, or configs.
  2. The Kaggle execution notebook includes preflight authentication checks for `HF_TOKEN`.
  3. Bulk archive downloads (>260 GB) are strictly avoided in favor of selective single-image materialization.
