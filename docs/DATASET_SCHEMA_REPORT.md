# XMCA-Net: Dataset Schema & Normalization Report
## Task 1 Data Schema Contracts & Semantic Preservation

This report details the field mappings, schema constraints, and normalization invariants used to unify the four benchmark datasets into `NormalizedRecord` objects.

---

## 1. Unified Normalization Schema Contract

Each sample across all four benchmarks is processed by `src/data/loader.py` into a strictly validated `NormalizedRecord`:

```python
class NormalizedRecord(BaseModel):
    dataset_id: str                      # "pope" | "faithscore" | "a_okvqa" | "infoseek"
    original_sample_id: str              # Original dataset question / annotation identifier
    original_image_id: str               # Original image filename or OVEN ID
    original_split: str                  # Original source split ("test" | "val")
    question: str                        # Natural language query / evaluation prompt
    official_answer_fields: Dict[str, Any] # Ground-truth answers, choices, synonyms
    official_label_fields: Dict[str, Any]  # Ground-truth attribution / binary labels
    source_revision: str                 # Git commit SHA or version string
    source_record_hash: str              # SHA-256 of canonical JSON-serialized raw record
    
    # Optional visual fields (populated upon physical verification)
    image_path: Optional[str] = None
    image_sha256: Optional[str] = None
    image_download_status: str = "pending" # pending | verified | missing | failed
    
    # Raw preservation
    raw_record: Dict[str, Any]           # Exact untouched raw source dictionary
```

---

## 2. Benchmark-Specific Mappings

### 2.1 POPE Normalization
- **Raw Fields**: `question_id`, `image`, `text`, `label`
- **Mapping**:
  - `original_sample_id` $\leftarrow$ `str(question_id)`
  - `original_image_id` $\leftarrow$ `image` (e.g. `COCO_val2014_000000310196.jpg`)
  - `question` $\leftarrow$ `text`
  - `official_answer_fields` $\leftarrow$ `{"label": "yes" | "no"}`
  - `official_label_fields` $\leftarrow$ `{"target_present": True/False, "hallucination_binary": 0/1}`
  - `original_split` $\leftarrow$ `"test"`

### 2.2 FaithScore Normalization
- **Raw Fields**: `id`, `answer`, `stage 1`, `stage 2`
- **Mapping**:
  - `original_sample_id` $\leftarrow$ `str(id)`
  - `original_image_id` $\leftarrow$ `f"COCO_val2014_{id.zfill(12)}.jpg"`
  - `question` $\leftarrow$ `"Describe the image in detail."`
  - `official_answer_fields` $\leftarrow$ `{"answer": raw["answer"]}`
  - `official_label_fields` $\leftarrow$ `{"stage1_descriptive": raw["stage 1"], "stage2_atomic_facts": raw["stage 2"]}`
  - `original_split` $\leftarrow$ `"test"`

### 2.3 A-OKVQA Normalization
- **Raw Fields**: `split`, `image_id`, `question_id`, `question`, `choices`, `correct_choice_idx`, `direct_answers`, `difficult_direct_answer`, `rationales`
- **Mapping**:
  - `original_sample_id` $\leftarrow$ `question_id`
  - `original_image_id` $\leftarrow$ `f"{int(image_id):012d}.jpg"`
  - `question` $\leftarrow$ `question`
  - `official_answer_fields` $\leftarrow$ `{"choices": raw["choices"], "correct_choice_idx": raw["correct_choice_idx"], "direct_answers": raw["direct_answers"], "rationales": raw["rationales"]}`
  - `official_label_fields` $\leftarrow$ `{"correct_choice_idx": raw["correct_choice_idx"]}`
  - `original_split` $\leftarrow$ `split` (`"val"`)

### 2.4 InfoSeek Normalization
- **Raw Fields**: `data_id`, `image_id`, `question`, `answer`, `answer_eval`, `data_split`
- **Mapping**:
  - `original_sample_id` $\leftarrow$ `data_id`
  - `original_image_id` $\leftarrow$ `image_id` (e.g. `oven_04990048`)
  - `question` $\leftarrow$ `question`
  - `official_answer_fields` $\leftarrow$ `{"answer": raw["answer"], "answer_eval": raw["answer_eval"]}`
  - `official_label_fields` $\leftarrow$ `{"answer": raw["answer"]}`
  - `original_split` $\leftarrow$ `data_split` (`"val_unseen_question"` or `"val"`)

---

## 3. Strict Non-Invention Policy

1. **No Imputation**: If an original record does not contain rationales or sub-sentences, the field is either omitted or initialized as an empty structure; it is never filled with synthetic LLM text.
2. **Deterministic Cryptographic Hashing**: Every record contains `source_record_hash = sha256(canonical_json(raw_record))`, guaranteeing byte-level provenance traceable back to original release files.

