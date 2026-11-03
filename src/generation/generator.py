"""Generation module for Vision-Language Models.

Status: PENDING TASK AUTHORIZATION
Under strict research execution mode, generation must be performed strictly on
Kaggle dual NVIDIA T4 environment. No generation or inference is permitted on local macOS.
"""

from typing import Dict, Any


class VLMGenerator:
    """Interface for VLM generation across LLaVA, Qwen2-VL, and InternVL."""

    def __init__(self, model_key: str, model_config: Dict[str, Any]):
        self.model_key = model_key
        self.model_config = model_config

    def generate(self, *args, **kwargs):
        raise NotImplementedError(
            f"VLM generation for '{self.model_key}' is pending implementation. "
            "Execution must occur on Kaggle dual NVIDIA T4 hardware. "
            "Dummy outputs or mock responses are strictly prohibited."
        )
