"""Generation package for XMCA-Net.

Exports unified VLM adapters, factory registry, and generation orchestrator.
"""

from src.generation.base import BaseVLMAdapter, VLMGenerationOutput
from src.generation.adapters import (
    LLaVAAdapter,
    Qwen2VLAdapter,
    InternVLAdapter,
    build_internvl2_8b_device_map,
    dynamic_preprocess,
    find_best_aspect_ratio,
)
from src.generation.registry import get_adapter, list_available_adapters
from src.generation.generator import GenerationOrchestrator

__all__ = [
    "BaseVLMAdapter",
    "VLMGenerationOutput",
    "LLaVAAdapter",
    "Qwen2VLAdapter",
    "InternVLAdapter",
    "build_internvl2_8b_device_map",
    "dynamic_preprocess",
    "find_best_aspect_ratio",
    "get_adapter",
    "list_available_adapters",
    "GenerationOrchestrator",
]
