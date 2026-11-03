"""Kaggle Dual-T4 Smoke Test Execution Script for XMCA-Net Task 2.1.

Executes real generation across LLaVA-1.5-7B, Qwen2-VL-7B, and InternVL2-2B on
6 identical authentic probes (POPE yes/no, FaithScore 2 distinct images, A-OKVQA 2 questions),
producing 18 records with genuine hardware GPU telemetry and output verification.
"""

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from typing import Dict, Any, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.generation.registry import get_adapter, list_available_adapters
from src.generation.base import VLMGenerationOutput
from src.generation.telemetry import (
    query_nvidia_smi,
    get_cuda_memory_snapshot,
    reset_peak_memory_stats,
    ModelMemoryTelemetry,
)
from src.data.loader import verify_partition_isolation
from src.utils.config import load_app_config, load_models_config
from src.utils.environment import check_gpu_environment, get_system_info


AUTHENTIC_PROBES = [
    {
        "probe_index": 1,
        "dataset": "pope",
        "sample_id": "pope_adversarial_103",
        "image_path": "data/images/preflight/pope/COCO_val2014_000000131115.jpg",
        "expected_sha256": "f99a971a56be876bc62c2c90f26cb6aa3a449d5efc77ef52e0a6b8588b089924",
        "question": "Is there a baseball glove in the image? Please answer yes or no.",
        "ground_truth_metadata": {"label": "yes", "strategy": "adversarial"},
    },
    {
        "probe_index": 2,
        "dataset": "pope",
        "sample_id": "pope_adversarial_104",
        "image_path": "data/images/preflight/pope/COCO_val2014_000000131115.jpg",
        "expected_sha256": "f99a971a56be876bc62c2c90f26cb6aa3a449d5efc77ef52e0a6b8588b089924",
        "question": "Is there a sports ball in the image? Please answer yes or no.",
        "ground_truth_metadata": {"label": "no", "strategy": "adversarial"},
    },
    {
        "probe_index": 3,
        "dataset": "faithscore",
        "sample_id": "faithscore_000000525439_eval_000",
        "image_path": "data/images/preflight/faithscore/COCO_val2014_000000525439.jpg",
        "expected_sha256": "023a7b68203f01cf177bc3888345194e23a47ba7cfefb2775da807da3f2388f2",
        "question": "Describe this image in detail, stating the main subjects, objects, and actions visible.",
        "ground_truth_metadata": {"human_annotated": True, "image_id": "000000525439"},
    },
    {
        "probe_index": 4,
        "dataset": "faithscore",
        "sample_id": "faithscore_000000097131_eval_001",
        "image_path": "data/images/preflight/faithscore/COCO_val2014_000000097131.jpg",
        "expected_sha256": "8aeefef6e7caab578e89c91720c75689171a4680413f2091e6c6a99dd0ba4b0c",
        "question": "Describe this image in detail, stating the main subjects, objects, and actions visible.",
        "ground_truth_metadata": {"human_annotated": True, "image_id": "000000097131"},
    },
    {
        "probe_index": 5,
        "dataset": "a_okvqa",
        "sample_id": "3X8TsFBRS7mBsBbyFDzRdb",
        "image_path": "data/images/preflight/a_okvqa/000000312192.jpg",
        "expected_sha256": "1e698a3dcde04f64793eacefeb9802710182f0ae918db480983049e8235d2cf9",
        "question": "Which is not an ingredient of this dish?",
        "ground_truth_metadata": {"task": "multiple_choice", "image_id": 312192},
    },
    {
        "probe_index": 6,
        "dataset": "a_okvqa",
        "sample_id": "4cdYbJFFeKXJr6LbKgcxTa",
        "image_path": "data/images/preflight/a_okvqa/000000176799.jpg",
        "expected_sha256": "63e94d002954531e2ea11b89733a6a871fd0118060000b5445efe2e7636c447a",
        "question": "In what type of environment are they most likely riding skateboards?",
        "ground_truth_metadata": {"task": "multiple_choice", "image_id": 176799},
    },
]


def verify_file_sha256(path: Path) -> str:
    """Compute and return exact 64-char SHA-256 of file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def execute_task2_smoke_test(
    models: List[str] = None,
    dry_run: bool = False,
    output_base: Path = Path("artifacts"),
) -> Dict[str, Any]:
    models = models or ["llava", "qwen2_vl", "internvl"]
    app_config = load_app_config()
    models_config = load_models_config()

    print("=" * 80)
    print("XMCA-Net — Task 2.1: Real Dual-T4 Generation Verification")
    print(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}")
    print(f"Execution Mode: {'DRY-RUN (Contract / Simulation)' if dry_run else 'GENUINE GPU HARDWARE'}")
    print("=" * 80)

    # Output directories
    generations_dir = output_base / "generations" / "task2_verified"
    reports_dir = output_base / "reports"
    generations_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    # 1. Hardware & GPU Environment Verification
    system_info = get_system_info()
    gpu_ok, gpu_msg = check_gpu_environment(require_dual_t4=not dry_run)
    print(f"Host System: {system_info['platform']} | Kaggle Kernel: {system_info['is_kaggle']}")
    print(f"Accelerator Check: {gpu_msg}")
    if not dry_run and not gpu_ok:
        raise RuntimeError(f"Strict Hardware Check Failed: {gpu_msg}")

    # Initial nvidia-smi capture
    initial_smi = query_nvidia_smi()
    print(f"Hardware nvidia-smi: {initial_smi}")

    actual_gpu_exec_report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "execution_mode": "dry_run" if dry_run else "genuine_dual_t4_gpu",
        "system_info": system_info,
        "gpu_check_passed": gpu_ok,
        "gpu_check_details": gpu_msg,
        "initial_nvidia_smi": initial_smi,
        "target_models": models,
        "precision": "torch.float16",
        "models_config": {
            k: {
                "model_id": v.model_id,
                "revision": v.revision,
                "architecture": v.architecture,
                "role": getattr(v, "role", "adapter_validation"),
                "device_map": v.device_map,
                "max_memory": v.max_memory_per_gpu,
            }
            for k, v in models_config.models.items()
        },
    }

    # 2. Verify Probe Images & SHA-256
    print("\nVerifying Authentic Probe Images...")
    probes_verified = []
    for probe in AUTHENTIC_PROBES:
        img_p = Path(probe["image_path"])
        if not img_p.exists():
            raise FileNotFoundError(f"Missing required authentic image: {img_p}")
        actual_sha = verify_file_sha256(img_p)
        if actual_sha != probe["expected_sha256"]:
            raise ValueError(
                f"SHA-256 mismatch for {img_p}: expected {probe['expected_sha256']}, got {actual_sha}"
            )
        probes_verified.append({
            **probe,
            "verified_sha256": actual_sha,
            "bytes": img_p.stat().st_size,
        })
        print(f"  ✓ [{probe['dataset'].upper()}] {probe['sample_id']} -> {img_p.name} ({actual_sha[:16]}...)")

    # 3. Verify InfoSeek Gating Safeguard
    infoseek_guard_passed = False
    try:
        from src.generation.generator import GenerationOrchestrator
        orchestrator = GenerationOrchestrator(app_config, models_config)
        orchestrator.run_inference_sample(
            model_key="llava",
            sample_id="infoseek_val_00004278",
            dataset="infoseek",
            question="What is the location of this building?",
            image_path=Path("data/images/infoseek/oven_unmaterialized.jpg"),
        )
    except RuntimeError as e:
        if "InfoSeek image-conditioned inference is BLOCKED" in str(e):
            infoseek_guard_passed = True
            print("\n✓ InfoSeek Gating Safeguard Verified: Gated OVEN images correctly blocked from inference.")
        else:
            raise e

    # 4. Verify Cross-Partition Leakage Isolation
    isolation_check_passed = False
    is_isolated, violations = verify_partition_isolation(["coco_000000506483"], "calibration")
    if not is_isolated and "coco_000000506483" in violations:
        isolation_check_passed = True
        print("✓ Cross-Partition Isolation Verified: Overlapping COCO groups barred from calibration/training.")

    # 5. Sequential Execution of the 3 VLMs
    model_load_reports: Dict[str, Any] = {}
    memory_telemetry_reports: Dict[str, Any] = {}
    all_generation_records: List[Dict[str, Any]] = []

    for model_key in models:
        print("\n" + "=" * 60)
        print(f"PROCESSING BACKBONE: {model_key.upper()}")
        print("=" * 60)

        model_cfg = models_config.get_model(model_key)
        telemetry = ModelMemoryTelemetry(model_key, model_cfg.model_id)
        telemetry.capture_baseline()

        if dry_run:
            print(f"[DRY-RUN] Simulating execution contract for {model_cfg.model_id}...")
            model_load_reports[model_key] = {
                "model_key": model_key,
                "model_id": model_cfg.model_id,
                "revision": model_cfg.revision,
                "role": getattr(model_cfg, "role", "adapter_validation"),
                "status": "DRY_RUN_CONTRACT_VALIDATED",
                "load_duration_sec": 0.0,
                "device_map": model_cfg.device_map,
                "precision": model_cfg.dtype,
            }
            # Simulate outputs for contract validation
            model_outputs = []
            for probe in probes_verified:
                rec = {
                    "sample_id": probe["sample_id"],
                    "dataset": probe["dataset"],
                    "model_key": model_key,
                    "model_id": model_cfg.model_id,
                    "revision": model_cfg.revision,
                    "question": probe["question"],
                    "image_path": probe["image_path"],
                    "image_sha256": probe["verified_sha256"],
                    "generated_answer": f"[DRY_RUN simulation for {model_key} on {probe['sample_id']}]",
                    "input_tokens": 128,
                    "output_tokens": 16,
                    "latency_ms": 150.0,
                    "device_placement": model_cfg.device_map,
                    "generation_status": "dry_run_simulated",
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                }
                model_outputs.append(rec)
                all_generation_records.append(rec)

            telemetry.capture_after_load(0.0)
            telemetry.capture_after_inference()
            telemetry.capture_after_unload()
            memory_telemetry_reports[model_key] = telemetry.to_dict()

            # Save per-model JSONL
            model_jsonl = generations_dir / f"{model_key}_task2_samples.jsonl"
            with open(model_jsonl, "w", encoding="utf-8") as f:
                for r in model_outputs:
                    f.write(json.dumps(r) + "\n")
            continue

        # GENUINE GPU EXECUTION
        adapter = get_adapter(model_key, models_config, app_config)
        print(f"Loading {adapter.model_id} (rev: {adapter.revision}) in FP16...")
        t_load_start = time.perf_counter()
        adapter.load()
        load_duration = time.perf_counter() - t_load_start
        print(f"Loaded successfully in {load_duration:.2f}s.")

        telemetry.capture_after_load(load_duration)
        vram_loaded = adapter.get_memory_allocated()
        print(f"VRAM after load: {vram_loaded}")

        model_load_reports[model_key] = {
            "model_key": model_key,
            "model_id": adapter.model_id,
            "revision": adapter.revision,
            "role": getattr(model_cfg, "role", "adapter_validation"),
            "status": "LOADED_SUCCESSFULLY",
            "load_duration_sec": round(load_duration, 2),
            "device_map": str(model_cfg.device_map),
            "precision": model_cfg.dtype,
            "vram_after_load_gib": vram_loaded,
        }

        # Run the 6 identical probes
        model_outputs = []
        for probe in probes_verified:
            print(f"  Generating on {probe['dataset']} probe: '{probe['question'][:50]}...'")
            out: VLMGenerationOutput = adapter.generate(
                image=probe["image_path"],
                prompt=probe["question"],
                max_new_tokens=64,
            )
            print(f"    -> Response ({out.latency_ms:.1f}ms): '{out.generated_text[:60]}...'")

            rec = {
                "sample_id": probe["sample_id"],
                "dataset": probe["dataset"],
                "model_key": model_key,
                "model_id": adapter.model_id,
                "revision": adapter.revision,
                "question": probe["question"],
                "image_path": probe["image_path"],
                "image_sha256": probe["verified_sha256"],
                "generated_answer": out.generated_text,
                "input_tokens": out.input_tokens,
                "output_tokens": out.output_tokens,
                "latency_ms": out.latency_ms,
                "device_placement": str(adapter.config.device_map),
                "device_memory_gib": out.device_memory_gib,
                "generation_status": "SUCCESS",
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
            model_outputs.append(rec)
            all_generation_records.append(rec)

        telemetry.capture_after_inference()

        # Unload model & reclaim memory
        print(f"Unloading {model_key} and clearing CUDA memory...")
        adapter.unload()
        del adapter
        gc.collect()
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

        telemetry.capture_after_unload()
        memory_telemetry_reports[model_key] = telemetry.to_dict()
        print(f"VRAM after unload: {get_cuda_memory_snapshot()}")

        # Save per-model JSONL
        model_jsonl = generations_dir / f"{model_key}_task2_samples.jsonl"
        with open(model_jsonl, "w", encoding="utf-8") as f:
            for r in model_outputs:
                f.write(json.dumps(r) + "\n")

    # 6. Export Deliverable Reports
    with open(reports_dir / "task2_actual_gpu_execution.json", "w", encoding="utf-8") as f:
        json.dump(actual_gpu_exec_report, f, indent=2)

    with open(reports_dir / "task2_model_load_report.json", "w", encoding="utf-8") as f:
        json.dump(model_load_reports, f, indent=2)

    with open(reports_dir / "task2_memory_telemetry.json", "w", encoding="utf-8") as f:
        json.dump(memory_telemetry_reports, f, indent=2)

    # Combined generation outputs JSONL
    combined_jsonl = reports_dir / "task2_generation_outputs.jsonl"
    with open(combined_jsonl, "w", encoding="utf-8") as f:
        for r in all_generation_records:
            f.write(json.dumps(r) + "\n")

    # Acceptance Report
    acceptance_report = {
        "task": "Task 2.1 — Real Dual-T4 Generation Verification",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "execution_mode": "dry_run" if dry_run else "genuine_dual_t4_gpu",
        "total_probes_per_vlm": len(AUTHENTIC_PROBES),
        "total_models_evaluated": len(models),
        "total_records_generated": len(all_generation_records),
        "acceptance_criteria": {
            "dual_t4_gpu_detected": gpu_ok,
            "checkpoint_revisions_pinned": all(m.revision is not None for m in models_config.models.values()),
            "probe_images_sha256_verified": len(probes_verified) == 6,
            "infoseek_safeguard_verified": infoseek_guard_passed,
            "cross_partition_isolation_verified": isolation_check_passed,
            "all_outputs_non_empty": all(len(r["generated_answer"].strip()) > 0 for r in all_generation_records),
            "memory_telemetry_complete": len(memory_telemetry_reports) == len(models),
        },
        "deliverables": [
            str(generations_dir),
            str(reports_dir / "task2_actual_gpu_execution.json"),
            str(reports_dir / "task2_model_load_report.json"),
            str(reports_dir / "task2_memory_telemetry.json"),
            str(reports_dir / "task2_generation_outputs.jsonl"),
            str(reports_dir / "task2_acceptance_report.json"),
        ],
    }

    with open(reports_dir / "task2_acceptance_report.json", "w", encoding="utf-8") as f:
        json.dump(acceptance_report, f, indent=2)

    print("\n" + "=" * 80)
    print("TASK 2.1 EXECUTION COMPLETE — ALL DELIVERABLES GENERATED")
    print(f"Total outputs saved: {len(all_generation_records)}")
    print(f"Deliverables directory: {reports_dir}")
    print("=" * 80)
    return acceptance_report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Task 2.1 Dual-T4 Smoke Execution")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["llava", "qwen2_vl", "internvl"],
        help="List of models to execute",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run contract verification simulation without GPU hardware",
    )
    parser.add_argument(
        "--output-base",
        type=str,
        default="artifacts",
        help="Base directory for artifacts",
    )
    args = parser.parse_args()

    execute_task2_smoke_test(
        models=args.models,
        dry_run=args.dry_run,
        output_base=Path(args.output_base),
    )
