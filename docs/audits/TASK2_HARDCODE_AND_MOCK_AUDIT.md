# XMCA-Net: Task 2.0.1 Hardcode & Mock Forensic Audit

**Audit Objective**: Exhaustive static scan for hardcoded answers, proxy scoring, mock responses, fabricated GPU numbers, and hidden shortcuts across the entire codebase.

---

## 1. Forensic Keyword Search Results

Every match of sensitive terms (`Mock`, `MagicMock`, `patch`, `monkeypatch`, `dummy`, `synthetic`, `fake`, `placeholder`, `hardcoded`, `simulated`, `return "A man`, `generated_answer =`, `generation_status =`, `inference_time =`, `peak_gpu_memory`) was forensically audited:

| Match Category | File Location | Line Numbers | Context & Purpose | Enters Research Execution Path? | Severity |
|:---|:---|:---|:---|:---|:---|
| **Defensive Ban Docstrings** | `src/combiner/combiner.py` | 7, 22 | Explicit docstrings prohibiting dummy scores | NO | PASS |
| **Defensive Ban Docstrings** | `src/retrieval/consistency.py` | 5, 18 | Explicit docstrings prohibiting proxy approximations | NO | PASS |
| **Defensive Ban Docstrings** | `src/perception/consistency.py` | 5, 18 | Explicit docstrings prohibiting proxy approximations | NO | PASS |
| **Defensive Ban Docstrings** | `src/utilization/consistency.py` | 6, 19 | Explicit docstrings prohibiting proxy approximations | NO | PASS |
| **Defensive Ban Docstrings** | `src/baselines/baselines.py` | 5, 18 | Explicit docstrings prohibiting dummy baselines | NO | PASS |
| **Strict Config Validator** | `src/utils/config.py` | 15, 20-25 | `allow_synthetic_data: bool = Field(default=False)`; raises error if set True | NO | PASS |
| **POPE Strategy Keywords** | `src/data/loader.py` | 52, 54 | POPE benchmark sampling strategies (`adversarial`, `popular`, `random`) | YES (Authentic data split) | PASS |
| **Simulated Output Quarantine** | `tests/fixtures/task2_dry_run/` | All files | Stored dry-run records generated during contract testing | NO (Quarantined under `tests/fixtures/`) | PASS |
| **Dry-Run CLI Branch** | `scripts/kaggle_vlm_smoke_test.py` | 260-314 | `if dry_run:` branch simulating contracts; writes ONLY to `tests/fixtures/task2_dry_run/` | NO (Requires `--dry-run` flag) | PASS |
| **Unit Test Mocking** | `tests/test_vlm_adapters.py` | 155-190 | `MagicMock` used to verify output schema shape in Python environment lacking GPU | NO (Restricted to test function) | PASS |
| **GPU Check Mocking** | `tests/test_strict_mode.py` | 30-70 | `patch` used to verify strict environment rejection on non-T4 GPUs | NO (Restricted to test function) | PASS |

---

## 2. Deep Dive into Dry-Run vs Research Directory Separation

### 2.1 The Quarantine Check
- `artifacts/generations/task2_verified/`:
  - Contains strictly `.gitkeep`.
  - Zero JSONL records exist in this directory.
- `tests/fixtures/task2_dry_run/`:
  - Contains `llava.jsonl`, `qwen2_vl.jsonl`, `internvl2.jsonl`, and `reports/`.
  - Every simulated answer is explicitly prefixed with `[DRY_RUN answer from ...]`.
  - Every record has `"generation_status": "DRY_RUN_SIMULATED"`.
  - All simulated latency is fixed at `140.0ms` and labeled as simulated.
- **Finding**: There is zero risk of simulated dry-run outputs being mistaken for real VLM inferences or entering downstream evaluation manifests.

### 2.2 CLI Execution Safeguard
In `scripts/kaggle_vlm_smoke_test.py`:
- In lines 309-314:
  ```python
  if dry_run:
      ...
      fixture_file = Path("tests/fixtures/task2_dry_run") / file_map.get(model_key, f"{model_key}.jsonl")
      fixture_file.parent.mkdir(parents=True, exist_ok=True)
      with open(fixture_file, "w", encoding="utf-8") as f:
          for r in records:
              f.write(json.dumps(r) + "\n")
      continue
  ```
- Lines 316-410 (the genuine hardware execution path) runs ONLY when `dry_run=False`, calling `adapter.load()` and `adapter.generate()`, and writing to `artifacts/generations/task2_verified/{target_file}`.
- If GPU hardware is unavailable and `dry_run=False`, the execution halts immediately at line 138 with:
  ```python
  raise RuntimeError(f"Strict Hardware Check Failed: {gpu_msg}")
  ```

---

## 3. Audit Verdict

**VERDICT**: **PASS**  
No hardcoded responses, fabricated latencies, or fake probability scores exist in the production source files or research artifact paths. Mocks are restricted entirely to test assertions.
