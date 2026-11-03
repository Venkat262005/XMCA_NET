# XMCA-Net: InfoSeek Image Access & Authentication Report
## Task 1.1 Gated OVEN Resolution & Kaggle Secrets Protocol

---

## 1. Benchmark Resource Inventory

- **Annotation File**: `data/raw/infoseek_val.jsonl` (21,204,639 bytes, 25,274 question records).
- **SHA-256**: `21240f83b4d7d9b5cc27f1546e486eb7c1eea9216b77a071acab1bfd9fc741c8`.
- **Image Identifier Schema**: Each sample references `image_id: "oven_XXXXXXXX"`, derived from the Open-domain Visual Entity Recognition (OVEN) benchmark.
- **Wikipedia 6M Corpus**:
  - Full Info: `http://storage.googleapis.com/gresearch/open-vision-language/Wiki6M_ver_1_0.jsonl.gz` (6.9 GB compressed).
  - Title Only: `http://storage.googleapis.com/gresearch/open-vision-language/Wiki6M_ver_1_0_title_only.jsonl` (419 MB).

---

## 2. Gated Access Condition Audit

The official OVEN snapshot on Hugging Face (`ychenNLP/oven` and `visual-open-eval/OVEN`) requires user terms acceptance and authenticated access. Unauthenticated requests receive `HTTP 401 Unauthorized` or gated access redirects.

### Secure Authentication Architecture:
```
+------------------------------------------------------------------+
|                   Kaggle Environment (Dual T4)                   |
|                                                                  |
|   +--------------------------+                                   |
|   | Kaggle User Secrets      |                                   |
|   |  Label: "HF_TOKEN"       |                                   |
|   +--------------------------+                                   |
|                 | (In-memory retrieval only)                     |
|                 v                                                |
|   from kaggle_secrets import UserSecretsClient                   |
|   hf_token = UserSecretsClient().get_secret("HF_TOKEN")          |
|                 |                                                |
|                 v                                                |
|   Authenticated Selective Materialization (HTTP Headers)         |
|   Authorization: Bearer <hf_token>                               |
+------------------------------------------------------------------+
```

### Absolute Security Invariants:
1. **Zero Secret Persistence**: No tokens, API keys, or credentials may be written to disk, committed to Git, or outputted in notebook execution logs.
2. **Selective Fetching**: Bulk archive downloads (>260 GB across 8 shards) are strictly prohibited. The 1,000 selected InfoSeek cohort images are materialized selectively using authenticated shard index lookups.
3. **No Substitution**: If access is unconfigured, images remain cataloged as `access_restricted_hf_login_required` in `artifacts/manifests/download_failures.json`. Blank images, dummy pixels, or substitute images are strictly forbidden.

---

## 3. Preflight Cohort Audit (25 Records)

The 25 candidate preflight records (`oven_04990048`, `oven_04962816`, `oven_04967661`, `oven_05001968`, etc.) have been cataloged with:
- Verified entity target questions and Wikidata ground-truth answer aliases.
- Explicit download status in `artifacts/manifests/image_manifest.parquet`.
- Ready for authenticated execution on Kaggle Dual T4 upon credential binding.
