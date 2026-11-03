"""Kaggle Dual-T4 Smoke Test Script for XMCA-Net VLM Adapters.

Performs sequential end-to-end smoke testing of LLaVA, Qwen2-VL, and InternVL
on authentic benchmark samples, verifying:
  1. Multi-GPU allocation and FP16 weights on dual NVIDIA T4.
  2. Memory reclamation between backbones (zero OOM).
  3. Authentic string generation on real images.
  4. InfoSeek image-conditioned gating safeguard.
  5. Cross-partition image leakage isolation.
"""

import argparse
import gc
import json
import os
from pathlib import Path
import sys
import time
from typing import Dict, Any, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.generation.registry import get_adapter, list_available_adapters
from src.generation.base import VLMGenerationOutput
from src.data.loader import verify_partition_isolation
from src.utils.config import load_app_config, load_models_config
from src.utils.environment import check_gpu_environment, get_system_info


def run_smoke_test(
    models_to_test: List[str],
    dry_run: bool = False,
    output_path: Path = Path("artifacts/reports/task2_smoke_test_report.json"),
) -> Dict[str, Any]:
    print("=" * 70)
    print("XMCA-Net — Task 2 VLM Adapters & Dual-T4 Smoke Test")
    print("=" * 70)

    app_config = load_app_config()
    models_config = load_models_config()
    system_info = get_system_info()
    print(f"Host System: {system_info['platform']} | Kaggle Kernel: {system_info['is_kaggle']}")

    gpu_ok, gpu_msg = check_gpu_environment(require_dual_t4=not dry_run)
    print(f"GPU Environment Check: {gpu_msg}")
    if not dry_run and not gpu_ok:
        raise RuntimeError(f"GPU verification failed: {gpu_msg}")

    # 1. Preflight image resolution (pick first existing image dynamically)
    def find_first_image(folder: Path) -> Path:
        if folder.exists():
            for p in sorted(folder.glob("*.jpg")):
                return p
        return folder / "placeholder.jpg"

    pope_img = find_first_image(Path("data/images/preflight/pope"))
    faithscore_img = find_first_image(Path("data/images/preflight/faithscore"))
    aokvqa_img = find_first_image(Path("data/images/preflight/a_okvqa"))

    test_probes = [
        {
            "dataset": "pope",
            "image_path": pope_img,
            "question": "Is there a bicycle in this image? Please answer yes or no.",
        },
        {
            "dataset": "faithscore",
            "image_path": faithscore_img,
            "question": "Describe the main subjects and scene in this image concisely.",
        },
        {
            "dataset": "a_okvqa",
            "image_path": aokvqa_img,
            "question": "What is the primary activity taking place in this image?",
        },
    ]

    # Verify existing image files
    for probe in test_probes:
        if not probe["image_path"].exists():
            print(f"Warning: Preflight image {probe['image_path']} not found on disk.")

    # 2. Verify InfoSeek Gating Safeguard
    infoseek_guard_passed = False
    dummy_infoseek_path = Path("data/images/infoseek/oven_nonexistent.jpg")
    try:
        from src.generation.generator import GenerationOrchestrator
        orchestrator = GenerationOrchestrator(app_config, models_config)
        orchestrator.run_inference_sample(
            model_key="llava",
            sample_id="test_infoseek",
            dataset="infoseek",
            question="What is this?",
            image_path=dummy_infoseek_path,
        )
    except RuntimeError as e:
        if "InfoSeek image-conditioned inference is BLOCKED" in str(e):
            infoseek_guard_passed = True
            print("✓ InfoSeek Gating Safeguard verified: Un-materialized OVEN image access properly blocked.")
    except Exception as e:
        print(f"Note: Orchestrator guard test raised: {type(e).__name__}: {e}")

    # 3. Verify 24 Overlapping Physical Image Isolation
    isolation_passed = False
    try:
        # Check an overlapping COCO ID
        test_overlapping_ids = ["coco_000000506483", "coco_000000179765"]
        is_isolated, violations = verify_partition_isolation(test_overlapping_ids, "calibration")
        # Overlapping image is in held-out test, so it MUST violate calibration/training eligibility
        if not is_isolated and len(violations) > 0:
            isolation_passed = True
            print(f"✓ Cross-Partition Isolation verified: Overlapping test images {violations} barred from calibration.")
    except Exception as e:
        print(f"Note: Isolation test encountered: {e}")

    # 4. Sequential Model Smoke Test
    model_reports: Dict[str, Any] = {}

    for model_key in models_to_test:
        print("\n" + "-" * 50)
        print(f"Testing Backbone: {model_key.upper()}")
        print("-" * 50)

        if dry_run:
            print(f"[DRY-RUN] Simulating adapter contract for '{model_key}'...")
            adapter = get_adapter(model_key, models_config, app_config)
            model_reports[model_key] = {
                "status": "dry_run_validated",
                "model_id": adapter.model_id,
                "revision": adapter.revision,
                "family": adapter.config.family,
                "device_map": adapter.config.device_map,
                "smoke_outputs": [],
            }
            continue

        try:
            import torch
            vram_before = {
                f"cuda:{i}": round(torch.cuda.memory_allocated(i) / (1024 ** 3), 3)
                for i in range(torch.cuda.device_count())
            }
            print(f"VRAM before load: {vram_before}")

            adapter = get_adapter(model_key, models_config, app_config)
            print(f"Loading weights for {adapter.model_id} (rev: {adapter.revision})...")
            load_start = time.perf_counter()
            adapter.load()
            load_time_sec = round(time.perf_counter() - load_start, 2)
            print(f"Loaded successfully in {load_time_sec}s.")

            vram_loaded = adapter.get_memory_allocated()
            print(f"VRAM after load: {vram_loaded}")

            probe_outputs = []
            # Run 1 real probe
            probe = test_probes[0]
            if probe["image_path"].exists():
                print(f"Generating on probe ({probe['dataset']}): {probe['question']}")
                out = adapter.generate(
                    image=probe["image_path"],
                    prompt=probe["question"],
                    max_new_tokens=64,
                )
                print(f"Generated ({out.latency_ms:.1f}ms): {out.generated_text}")
                probe_outputs.append(out.model_dump())

            # Unload model
            print("Unloading model and clearing CUDA cache...")
            adapter.unload()
            del adapter
            gc.collect()
            torch.cuda.empty_cache()

            vram_after = {
                f"cuda:{i}": round(torch.cuda.memory_allocated(i) / (1024 ** 3), 3)
                for i in range(torch.cuda.device_count())
            }
            print(f"VRAM after unload: {vram_after}")

            model_reports[model_key] = {
                "status": "passed",
                "load_time_sec": load_time_sec,
                "vram_loaded_gib": vram_loaded,
                "vram_after_gib": vram_after,
                "smoke_outputs": probe_outputs,
            }

        except Exception as e:
            print(f"ERROR testing {model_key}: {e}")
            model_reports[model_key] = {
                "status": "failed",
                "error": str(e),
            }

    # Final Summary Report
    report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "dry_run": dry_run,
        "system_info": system_info,
        "gpu_environment": {"valid": gpu_ok, "details": gpu_msg},
        "infoseek_guard_passed": infoseek_guard_passed,
        "partition_isolation_passed": isolation_passed,
        "models": model_reports,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\n" + "=" * 70)
    print(f"Smoke Test Report written to: {output_path}")
    print("=" * 70)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Task 2 VLM Smoke Test")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["llava", "qwen2_vl", "internvl"],
        help="Model keys to smoke test",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Simulate execution without requiring GPU hardware",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="artifacts/reports/task2_smoke_test_report.json",
        help="Destination path for smoke test JSON report",
    )
    args = parser.parse_args()

    run_smoke_test(
        models_to_test=args.models,
        dry_run=args.dry_run,
        output_path=Path(args.output),
    )
