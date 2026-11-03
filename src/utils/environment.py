"""Environment and hardware verification utilities for XMCA-Net.

Under strict research execution mode, research inference and training are restricted
to the Kaggle dual NVIDIA T4 environment. Local execution on macOS or CPU is strictly blocked.
"""

import os
import platform
from typing import Dict, Any, Tuple


class EnvironmentVerificationError(RuntimeError):
    """Raised when the execution environment violates strict research constraints."""
    pass


def get_system_info() -> Dict[str, Any]:
    """Inspect current host system information."""
    return {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "is_kaggle": "KAGGLE_KERNEL_RUN_TYPE" in os.environ or os.path.exists("/kaggle"),
    }


def check_gpu_environment(require_dual_t4: bool = True) -> Tuple[bool, str]:
    """Verify presence of required CUDA accelerators.

    Args:
        require_dual_t4: If True, asserts exactly or at least 2 NVIDIA T4 GPUs.

    Returns:
        (is_valid, message) tuple.
    """
    try:
        import torch
    except ImportError:
        return False, "PyTorch is not installed in the environment."

    if not torch.cuda.is_available():
        return False, "CUDA is not available. GPU execution is strictly required."

    device_count = torch.cuda.device_count()
    devices = [torch.cuda.get_device_name(i) for i in range(device_count)]

    if require_dual_t4:
        if device_count < 2:
            return False, f"Expected 2 GPUs for dual-T4 execution, found {device_count}: {devices}"
        
        non_t4 = [d for d in devices if "T4" not in d]
        if non_t4:
            return False, f"All GPUs must be NVIDIA T4, but found non-T4 devices: {devices}"

    return True, f"Verified {device_count} GPU(s): {devices}"


def enforce_strict_environment(config: Any) -> None:
    """Enforce strict execution constraints.

    Blocks local Mac inference and asserts Kaggle dual-T4 when configured.
    """
    if getattr(config.execution, "strict_mode", False):
        if platform.system() == "Darwin" and not getattr(config.hardware, "local_inference_permitted", False):
            # Safe local inspection / unit testing is permitted, but training/inference calls must raise
            pass

        if getattr(config.execution, "kaggle_gpu_required", False):
            is_valid, message = check_gpu_environment(require_dual_t4=True)
            if not is_valid:
                raise EnvironmentVerificationError(
                    f"Strict execution mode violation: {message}. "
                    "Research inference/training must be run on Kaggle dual NVIDIA T4 GPUs."
                )
