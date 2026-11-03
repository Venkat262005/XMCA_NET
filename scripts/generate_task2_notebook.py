"""Generate notebooks/XMCA_TASK2_VLM_SMOKE_TEST.ipynb for Task 2.1 Final Execution."""

import json
from pathlib import Path


def create_task2_notebook():
    cells = []

    def md_cell(source):
        return {
            "cell_type": "markdown",
            "metadata": {},
            "source": source.strip().splitlines(keepends=True),
        }

    def code_cell(source):
        return {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": source.strip().splitlines(keepends=True),
        }

    # Header
    cells.append(md_cell("""# XMCA-Net — Task 2.1: Final Dual-T4 Generation Verification

**Target Accelerator**: Kaggle Dual NVIDIA T4 (2 × 16GB VRAM)  
**Strict Research Protocol**:
- 6 Identical authentic probes across 3 backbones (18 genuine records total):
  - POPE: `pope_adversarial_103` (yes) and `pope_adversarial_104` (no)
  - FaithScore: `000000525439` (Image 1) and `000000097131` (Image 2)
  - A-OKVQA: `3X8TsFBRS7mBsBbyFDzRdb` (Q1) and `4cdYbJFFeKXJr6LbKgcxTa` (Q2)
  - InfoSeek: Verified blocked from image-conditioned inference until OVEN access
- 3 Explicitly declared adapter-validation models:
  - `llava-hf/llava-1.5-7b-hf` (rev `a272c74` / full SHA `a272c74b...`)
  - `Qwen/Qwen2-VL-7B-Instruct` (rev `eed13092...`)
  - `OpenGVLab/InternVL2-2B` (rev `e4f6747b...`)
  - Optional scale test: `OpenGVLab/InternVL2-8B` (rev `6fb9ad69...`) with custom layer mapping
- Outputs:
  - `artifacts/generations/task2_verified/llava.jsonl`
  - `artifacts/generations/task2_verified/qwen2_vl.jsonl`
  - `artifacts/generations/task2_verified/internvl2.jsonl`
  - 6 Reports in `artifacts/reports/`"""))

    # Cell 1: Hardware Validation
    cells.append(md_cell("## 1. Hardware Environment & GPU Verification"))
    cells.append(code_cell("""import os
import sys
import gc
import time
import json
import hashlib
from pathlib import Path
from PIL import Image

import torch

print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")

if not torch.cuda.is_available():
    raise RuntimeError("CUDA accelerator is strictly required. Set Kaggle Accelerator to GPU T4 x 2.")

device_count = torch.cuda.device_count()
print(f"Detected GPU Count: {device_count}")
for i in range(device_count):
    dev_name = torch.cuda.get_device_name(i)
    total_mem = torch.cuda.get_device_properties(i).total_memory / (1024**3)
    print(f"  GPU {i}: {dev_name} ({total_mem:.2f} GiB VRAM)")
    if "T4" not in dev_name:
        print(f"  Warning: Accelerator '{dev_name}' is not an NVIDIA T4.")

if device_count < 2:
    raise RuntimeError(f"Strict requirement: Expected 2 NVIDIA T4 GPUs for dual-T4 execution, found {device_count}.")"""))

    # Cell 2: Dependencies
    cells.append(md_cell("## 2. Environment Dependencies & Kaggle Secrets"))
    cells.append(code_cell('''# Install required packages
!pip install -q "transformers>=4.45.0" "accelerate>=0.30.0" "timm>=0.9.12" "einops>=0.7.0" "bitsandbytes>=0.43.0"'''))

    # Cell 3: Secrets
    cells.append(code_cell("""try:
    from kaggle_secrets import UserSecretsClient
    user_secrets = UserSecretsClient()
    hf_token = user_secrets.get_secret("HF_TOKEN")
    os.environ["HF_TOKEN"] = hf_token
    print("✓ Kaggle Secrets: HF_TOKEN detected and configured.")
except Exception as e:
    print(f"Notice: Kaggle Secrets HF_TOKEN not found ({e}). OVEN access remains gated.")"""))

    # Cell 4: Repository Setup
    cells.append(md_cell("## 3. Codebase Setup & Telemetry Verification"))
    cells.append(code_cell("""repo_root = Path.cwd()
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from src.generation.registry import get_adapter, list_available_adapters
from src.generation.generator import GenerationOrchestrator
from src.generation.telemetry import (
    query_nvidia_smi,
    get_cuda_memory_snapshot,
    reset_peak_memory_stats,
    get_parameter_distribution,
    get_tensor_device_placement,
    ModelMemoryTelemetry,
)
from src.data.loader import verify_partition_isolation
from src.utils.config import load_app_config, load_models_config

app_config = load_app_config()
models_config = load_models_config()

print(f"Available Adapters: {list_available_adapters()}")
print("Initial Hardware Telemetry:")
for gpu in query_nvidia_smi():
    print(f"  GPU {gpu.get('gpu_index')}: {gpu.get('name')} | Used: {gpu.get('memory_used_gib')} GiB / {gpu.get('memory_total_gib')} GiB")"""))

    # Cell 5: Safeguard Verification
    cells.append(md_cell("## 4. Preflight Integrity & Safeguard Verification"))
    cells.append(code_cell("""# 1. InfoSeek Gating Safeguard
try:
    orchestrator = GenerationOrchestrator(app_config, models_config)
    orchestrator.run_inference_sample(
        model_key="llava",
        sample_id="infoseek_val_00004278",
        dataset="infoseek",
        question="What is this object?",
        image_path=Path("data/images/infoseek/unmaterialized.jpg"),
    )
except RuntimeError as e:
    if "InfoSeek image-conditioned inference is BLOCKED" in str(e):
        print("✓ InfoSeek safeguard PASS: Gated OVEN access strictly blocked.")
    else:
        raise e

# 2. Cross-Partition Leakage Isolation Check
is_isolated, violations = verify_partition_isolation(["coco_000000506483"], "calibration")
if not is_isolated and "coco_000000506483" in violations:
    print("✓ Global Image Group Isolation PASS: Overlapping test images excluded from calibration.")
else:
    raise RuntimeError("Partition isolation check failed.")"""))

    # Cell 6: Execute Full Task 2.1 Genuine Generation
    cells.append(md_cell("## 5. Sequential Genuine Generation on Dual T4 GPUs"))
    cells.append(code_cell("""from scripts.kaggle_vlm_smoke_test import execute_task2_smoke_test

# Execute genuine GPU inference across LLaVA-1.5-7B, Qwen2-VL-7B, and InternVL2-2B
acceptance_report = execute_task2_smoke_test(
    models=["llava", "qwen2_vl", "internvl"],
    dry_run=False,
    run_scale_8b=False,
    output_base=Path("artifacts"),
)

print(json.dumps(acceptance_report, indent=2))"""))

    # Cell 7: Inspect Persisted Outputs
    cells.append(md_cell("## 6. Inspect Generated Artifacts & Telemetry Reports"))
    cells.append(code_cell("""!ls -la artifacts/reports/
!ls -la artifacts/generations/task2_verified/"""))

    # Cell 8: Preview Sample Responses
    cells.append(code_cell("""print("=== PREVIEW GENERATED ANSWERS ===")
for model_file in ["llava.jsonl", "qwen2_vl.jsonl", "internvl2.jsonl"]:
    p = Path(f"artifacts/generations/task2_verified/{model_file}")
    if p.exists():
        print(f"\\n--- {model_file} ---")
        with open(p, "r") as f:
            for idx, line in enumerate(f):
                r = json.loads(line)
                print(f"[{r['dataset']}] {r['sample_id']} ({r['latency_ms']:.1f}ms): {r['generated_answer'][:70]}")"""))

    notebook_dict = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.10.12",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }

    out_file = Path("notebooks/XMCA_TASK2_VLM_SMOKE_TEST.ipynb")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(notebook_dict, f, indent=2)
    print(f"Successfully generated {out_file}")


if __name__ == "__main__":
    create_task2_notebook()
