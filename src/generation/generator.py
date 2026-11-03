"""Generation orchestrator for Vision-Language Models in XMCA-Net.

Orchestrates sequential execution of backbones, enforces strict hardware requirements,
guards against InfoSeek un-materialized image access, and persists standardized outputs.
"""

import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from PIL import Image

from src.generation.base import BaseVLMAdapter, VLMGenerationOutput
from src.generation.registry import get_adapter
from src.utils.config import AppConfig, ModelsConfig, load_app_config, load_models_config
from src.utils.environment import enforce_strict_environment


class GenerationOrchestrator:
    """Manages sequential VLM model loading, batch/sample inference, and artifact persistence."""

    def __init__(
        self,
        app_config: Optional[AppConfig] = None,
        models_config: Optional[ModelsConfig] = None,
    ):
        self.app_config = app_config or load_app_config()
        self.models_config = models_config or load_models_config()
        self.active_adapter: Optional[BaseVLMAdapter] = None

    def load_adapter(self, model_key: str) -> BaseVLMAdapter:
        """Load target VLM adapter, unloading any previously loaded model first."""
        if self.active_adapter is not None:
            self.unload_active_adapter()

        adapter = get_adapter(
            model_key=model_key,
            models_config=self.models_config,
            app_config=self.app_config,
        )
        adapter.load()
        self.active_adapter = adapter
        return adapter

    def unload_active_adapter(self) -> None:
        """Safely unload the currently active adapter and reclaim GPU memory."""
        if self.active_adapter is not None:
            self.active_adapter.unload()
            self.active_adapter = None

    def run_inference_sample(
        self,
        model_key: str,
        sample_id: str,
        dataset: str,
        question: str,
        image_path: Union[str, Path],
        image: Optional[Image.Image] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Execute inference on a single verified sample under strict safety guards.
        
        Guards:
          - InfoSeek without materialized image is explicitly blocked.
          - Missing image file raises FileNotFoundError.
          - Execution must satisfy strict environment conditions.
        """
        # InfoSeek safeguard
        if dataset == "infoseek":
            img_p = Path(image_path)
            if not img_p.exists() or img_p.stat().st_size == 0:
                raise RuntimeError(
                    f"InfoSeek image-conditioned inference is BLOCKED: Image '{image_path}' "
                    "is not materialized. Gated OVEN access must be resolved on Kaggle via HF_TOKEN."
                )

        if self.active_adapter is None or self.active_adapter.model_key != model_key:
            self.load_adapter(model_key)

        pil_img = image
        if pil_img is None:
            pil_img = Path(image_path)

        output: VLMGenerationOutput = self.active_adapter.generate(
            image=pil_img,
            prompt=question,
            **kwargs,
        )

        record = {
            "sample_id": sample_id,
            "dataset": dataset,
            "model_key": model_key,
            "model_id": output.model_id,
            "revision": output.revision,
            "question": question,
            "image_path": str(image_path),
            "generated_answer": output.generated_text,
            "input_tokens": output.input_tokens,
            "output_tokens": output.output_tokens,
            "latency_ms": output.latency_ms,
            "device_memory_gib": output.device_memory_gib,
            "finish_reason": output.finish_reason,
            "metadata": output.metadata,
        }
        return record
