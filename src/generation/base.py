"""Base classes and schemas for Vision-Language Model adapters in XMCA-Net.

Defines the unified XMCA interface for multimodal inference across LLaVA,
Qwen2-VL, and InternVL backbones under strict research execution constraints.
"""

from abc import ABC, abstractmethod
import gc
from pathlib import Path
import time
from typing import Dict, Any, Optional, Union, List
from PIL import Image
from pydantic import BaseModel, Field

from src.utils.config import ModelItemConfig, AppConfig, load_app_config
from src.utils.environment import enforce_strict_environment


class VLMGenerationOutput(BaseModel):
    """Standardized output schema for VLM generation across all backbones."""
    model_key: str = Field(..., description="Canonical key: llava, qwen2_vl, or internvl")
    model_id: str = Field(..., description="Hugging Face repository ID")
    revision: str = Field(..., description="Pinned git commit revision")
    prompt: str = Field(..., description="Input textual prompt or question")
    generated_text: str = Field(..., description="Raw generated text from the model")
    finish_reason: Optional[str] = Field("stop", description="Generation stopping condition")
    latency_ms: float = Field(..., description="End-to-end generation latency in milliseconds")
    input_tokens: int = Field(default=0, description="Number of tokens in prompt + visual prefix")
    output_tokens: int = Field(default=0, description="Number of generated tokens")
    device_memory_gib: Dict[str, float] = Field(
        default_factory=dict, description="Allocated VRAM in GiB per CUDA device"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Model-specific auxiliary execution metadata"
    )


class BaseVLMAdapter(ABC):
    """Abstract Base Class establishing the unified XMCA interface for all VLMs.
    
    Subclasses wrap model-specific loading, dynamic resolution preprocessing,
    conversation templating, and multi-GPU tensor placement.
    """

    def __init__(
        self,
        model_key: str,
        model_config: ModelItemConfig,
        app_config: Optional[AppConfig] = None,
    ):
        self.model_key = model_key
        self.config = model_config
        self.app_config = app_config or load_app_config()
        self.model = None
        self.processor = None
        self.tokenizer = None
        self._is_loaded = False

    @property
    def is_loaded(self) -> bool:
        return self._is_loaded

    @property
    def model_id(self) -> str:
        return self.config.model_id

    @property
    def revision(self) -> str:
        if not self.config.revision:
            raise ValueError(f"Strict mode violation: model '{self.model_key}' lacks a pinned git revision.")
        return self.config.revision

    @property
    def dtype_str(self) -> str:
        return self.config.dtype

    def _resolve_image(self, image_input: Union[Image.Image, str, Path]) -> Image.Image:
        """Helper to ensure an open, RGB-mode PIL Image."""
        if isinstance(image_input, (str, Path)):
            path = Path(image_input)
            if not path.exists():
                raise FileNotFoundError(f"Image not found at path: {path}")
            img = Image.open(path)
            return img.convert("RGB")
        elif isinstance(image_input, Image.Image):
            return image_input.convert("RGB")
        else:
            raise TypeError(f"Expected PIL Image, str, or Path, got {type(image_input)}")

    def get_memory_allocated(self) -> Dict[str, float]:
        """Inspect current CUDA memory allocations across all available devices."""
        try:
            import torch
            if not torch.cuda.is_available():
                return {}
            allocations = {}
            for i in range(torch.cuda.device_count()):
                allocated_gib = torch.cuda.memory_allocated(i) / (1024 ** 3)
                allocations[f"cuda:{i}"] = round(allocated_gib, 3)
            return allocations
        except ImportError:
            return {}

    @abstractmethod
    def load(self) -> None:
        """Load tokenizer/processor and model weights onto target accelerators."""
        pass

    def unload(self) -> None:
        """Unload model from memory and flush CUDA cache to ensure zero OOM between models."""
        self.model = None
        self.processor = None
        self.tokenizer = None
        self._is_loaded = False
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
        except ImportError:
            pass

    @abstractmethod
    def format_prompt(self, prompt: str) -> Any:
        """Format text prompt according to model-specific chat template."""
        pass

    @abstractmethod
    def preprocess_image(self, image: Image.Image) -> Any:
        """Apply model-specific visual preprocessing or dynamic patching."""
        pass

    @abstractmethod
    def generate(
        self,
        image: Union[Image.Image, str, Path],
        prompt: str,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        do_sample: Optional[bool] = None,
        **kwargs,
    ) -> VLMGenerationOutput:
        """Execute multimodal inference and return standardized VLMGenerationOutput."""
        pass

