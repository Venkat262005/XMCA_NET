"""GPU Memory Telemetry and Profiling for XMCA-Net.

Measures genuine PyTorch CUDA allocations, PyTorch caching allocator reserves,
peak watermarks, and hardware driver usage via nvidia-smi across dual NVIDIA T4 GPUs.
"""

import subprocess
import shutil
from typing import Dict, Any, List, Optional


def query_nvidia_smi() -> List[Dict[str, Any]]:
    """Query real hardware GPU memory usage via nvidia-smi CLI."""
    nvidia_smi_path = shutil.which("nvidia-smi")
    if not nvidia_smi_path:
        return []

    try:
        cmd = [
            nvidia_smi_path,
            "--query-gpu=index,name,memory.used,memory.total,utilization.gpu",
            "--format=csv,noheader,nounits",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        gpus = []
        for line in result.stdout.strip().splitlines():
            if not line.strip():
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 5:
                gpus.append({
                    "gpu_index": int(parts[0]),
                    "name": parts[1],
                    "memory_used_mb": float(parts[2]),
                    "memory_total_mb": float(parts[3]),
                    "memory_used_gib": round(float(parts[2]) / 1024.0, 3),
                    "memory_total_gib": round(float(parts[3]) / 1024.0, 3),
                    "gpu_utilization_pct": float(parts[4]),
                })
        return gpus
    except Exception as e:
        return [{"error": str(e)}]


def get_parameter_distribution(model: Any) -> Dict[str, Any]:
    """Calculate parameter count and percentage allocated per hardware device."""
    if model is None:
        return {}

    param_counts: Dict[str, int] = {}
    try:
        for _, param in model.named_parameters():
            dev = str(param.device)
            param_counts[dev] = param_counts.get(dev, 0) + param.numel()
        total = sum(param_counts.values())
        return {
            dev: {
                "param_count": count,
                "percentage": round(count / total * 100, 2) if total > 0 else 0.0,
            }
            for dev, count in param_counts.items()
        }
    except Exception as e:
        return {"error": str(e)}


def get_tensor_device_placement(tensors_dict: Dict[str, Any]) -> Dict[str, str]:
    """Inspect and report physical accelerator device for all input tensors."""
    placement = {}
    for k, v in tensors_dict.items():
        if hasattr(v, "device"):
            placement[k] = str(v.device)
    return placement


def get_cuda_memory_snapshot() -> Dict[str, Dict[str, float]]:
    """Capture detailed PyTorch CUDA memory metrics per active device."""
    try:
        import torch
        if not torch.cuda.is_available():
            return {}

        snapshot = {}
        for i in range(torch.cuda.device_count()):
            snapshot[f"cuda:{i}"] = {
                "memory_allocated_gib": round(torch.cuda.memory_allocated(i) / (1024 ** 3), 3),
                "memory_reserved_gib": round(torch.cuda.memory_reserved(i) / (1024 ** 3), 3),
                "max_memory_allocated_gib": round(torch.cuda.max_memory_allocated(i) / (1024 ** 3), 3),
                "max_memory_reserved_gib": round(torch.cuda.max_memory_reserved(i) / (1024 ** 3), 3),
            }
        return snapshot
    except ImportError:
        return {}


def reset_peak_memory_stats() -> None:
    """Reset peak allocation and reservation watermarks on all CUDA devices."""
    try:
        import torch
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                torch.cuda.reset_peak_memory_stats(i)
    except (ImportError, Exception):
        pass


class ModelMemoryTelemetry:
    """Tracks and records full memory lifecycle for a single VLM backbone."""

    def __init__(self, model_key: str, model_id: str):
        self.model_key = model_key
        self.model_id = model_id
        self.baseline_before_load: Dict[str, Any] = {}
        self.after_load: Dict[str, Any] = {}
        self.load_duration_sec: float = 0.0
        self.during_generation_peak: Dict[str, Any] = {}
        self.after_unload: Dict[str, Any] = {}

    def capture_baseline(self) -> None:
        reset_peak_memory_stats()
        self.baseline_before_load = {
            "pytorch": get_cuda_memory_snapshot(),
            "nvidia_smi": query_nvidia_smi(),
        }

    def capture_after_load(self, load_duration_sec: float) -> None:
        self.load_duration_sec = round(load_duration_sec, 2)
        self.after_load = {
            "load_duration_sec": self.load_duration_sec,
            "pytorch": get_cuda_memory_snapshot(),
            "nvidia_smi": query_nvidia_smi(),
        }

    def capture_after_inference(self) -> None:
        self.during_generation_peak = {
            "pytorch_peak": get_cuda_memory_snapshot(),
            "nvidia_smi": query_nvidia_smi(),
        }

    def capture_after_unload(self) -> None:
        self.after_unload = {
            "pytorch": get_cuda_memory_snapshot(),
            "nvidia_smi": query_nvidia_smi(),
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_key": self.model_key,
            "model_id": self.model_id,
            "load_duration_sec": self.load_duration_sec,
            "baseline_before_load": self.baseline_before_load,
            "after_load": self.after_load,
            "during_generation_peak": self.during_generation_peak,
            "after_unload": self.after_unload,
        }

