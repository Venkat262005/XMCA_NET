"""Generate notebooks/XMCA_TASK2_VLM_SMOKE_TEST.ipynb for Task 2.1."""

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

    # Title & Overview
    cells.append(md_cell("""# XMCA-Net — Task 2.1: Real Dual-T4 Generation Verification

**Target Hardware**: Kaggle NVIDIA T4 x 2 (Accelerator: GPU T4 x 2)  
**Strict Research Protocol**:
- 6 Identical authentic probes across all 3 VLMs:
  - POPE: 1 yes-labelled (`pope_adversarial_103`) & 1 no-labelled (`pope_adversarial_104`)
  - FaithScore: 2 distinct authentic images (`000000525439` & `000000097131`)
  - A-OKVQA: 2 authentic questions (`3X8TsFBRS7mBsBbyFDzRdb` & `4cdYbJFFeKXJr6LbKgcxTa`)
  - InfoSeek: Verified blocked from image-conditioned inference until OVEN access
- 3 Explicitly declared adapter-validation models:
  - `llava-hf/llava-1.5-7b-hf` (rev `a272c74`)
  - `Qwen/Qwen2-VL-7B-Instruct` (rev `847be15`)
  - `OpenGVLab/InternVL2-2B` (rev `cc06eec`, with 8B split validation)
- Comprehensive GPU memory telemetry:
  - `memory_allocated`, `memory_reserved`, peak watermarks, and driver `nvidia-smi` usage
- Deliverables:
  - `artifacts/generations/task2_verified/`
  - `artifacts/reports/task2_actual_gpu_execution.json`
  - `artifacts/reports/task2_model_load_report.json`
  - `artifacts/reports/task2_memory_telemetry.json`
  - `artifacts/reports/task2_generation_outputs.jsonl`
  - `artifacts/reports/task2_acceptance_report.json`"""))

    # Cell 1: Environment & GPU Verification
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
    raise RuntimeError("CUDA accelerator is strictly required. Ensure Kaggle setting is 'GPU T4 x 2'.")

device_count = torch.cuda.device_count()
print(f"Detected GPU Count: {device_count}")
for i in range(device_count):
    dev_name = torch.cuda.get_device_name(i)
    total_mem = torch.cuda.get_device_properties(i).total_memory / (1024**3)
    print(f"  GPU {i}: {dev_name} ({total_mem:.2f} GiB VRAM)")
    if "T4" not in dev_name:
        print(f"  Warning: Accelerator '{dev_name}' is not an NVIDIA T4.")

if device_count < 2:
    raise RuntimeError(f"Expected at least 2 NVIDIA T4 GPUs for dual-T4 execution, found {device_count}.")"""))

    # Cell 2: Dependencies
    cells.append(code_cell('''# Ensure required packages are present
!pip install -q "transformers>=4.45.0" "accelerate>=0.30.0" "timm>=0.9.12" "einops>=0.7.0"'''))

    # Cell 3: Kaggle Secrets
    cells.append(code_cell("""try:
    from kaggle_secrets import UserSecretsClient
    user_secrets = UserSecretsClient()
    hf_token = user_secrets.get_secret("HF_TOKEN")
    os.environ["HF_TOKEN"] = hf_token
    print("✓ Kaggle Secrets: HF_TOKEN detected and configured.")
except Exception as e:
    print(f"Notice: Kaggle Secrets HF_TOKEN not found ({e}). OVEN access remains gated until configured.")"""))

    # Cell 4: Repository Setup
    cells.append(code_cell("""repo_root = Path.cwd()
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from src.generation.registry import get_adapter, list_available_adapters
from src.generation.generator import GenerationOrchestrator
from src.generation.telemetry import (
    query_nvidia_smi,
    get_cuda_memory_snapshot,
    reset_peak_memory_stats,
    ModelMemoryTelemetry,
)
from src.data.loader import verify_partition_isolation
from src.utils.config import load_app_config, load_models_config

app_config = load_app_config()
models_config = load_models_config()

print(f"Available Adapters: {list_available_adapters()}")
print(f"Initial nvidia-smi: {query_nvidia_smi()}")"""))

    # Cell 5: Safeguard Verifications
    cells.append(code_cell("""# 1. InfoSeek Gating Safeguard
infoseek_sample_blocked = False
try:
    orchestrator = GenerationOrchestrator(app_config, models_config)
    orchestrator.run_inference_sample(
        model_key="llava",
        sample_id="infoseek_val_00004278",
        dataset="infoseek",
        question="What is this object?",
        image_path="data/images/infoseek/unmaterialized.jpg",
    )
except RuntimeError as e:
    if "InfoSeek image-conditioned inference is BLOCKED" in str(e):
        infoseek_sample_blocked = True
        print("✓ InfoSeek safeguard PASS: Un-materialized image access strictly blocked.")
    else:
        raise e

# 2. Cross-Partition Leakage Isolation Check
is_isolated, violations = verify_partition_isolation(["coco_000000506483"], "calibration")
if not is_isolated and "coco_000000506483" in violations:
    print("✓ Global Image Group Isolation PASS: Overlapping test images strictly excluded from calibration.")
else:
    raise RuntimeError("Isolation verification failed.")"""))

    # Cell 6: Execute Full Task 2.1 Suite via CLI runner
    cells.append(md_cell("""### Execute Task 2.1 Real Generation & Telemetry Suite
Executes all 6 probes sequentially across LLaVA-1.5-7B, Qwen2-VL-7B, and InternVL2-2B, capturing all GPU metrics and persisting official deliverables."""))

    cells.append(code_cell("""from scripts.kaggle_vlm_smoke_test import execute_task2_smoke_test

report = execute_task2_smoke_test(
    models=["llava", "qwen2_vl", "internvl"],
    dry_run=False,
    output_base=Path("artifacts"),
)

print(json.dumps(report, indent=2))"""))

    # Cell 7: Inspect Generated Deliverables
    cells.append(code_cell("""!ls -la artifacts/reports/
!ls -la artifacts/generations/task2_verified/"""))

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
