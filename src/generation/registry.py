"""VLM Adapter Registry and Factory for XMCA-Net.

Maps model keys ('llava', 'qwen2_vl', 'internvl') to their respective
adapter implementations, ensuring strict revision and configuration checks.
"""

from typing import Dict, Type, Optional, List

from src.generation.base import BaseVLMAdapter
from src.generation.adapters.llava import LLaVAAdapter
from src.generation.adapters.qwen2_vl import Qwen2VLAdapter
from src.generation.adapters.internvl import InternVLAdapter
from src.utils.config import ModelsConfig, AppConfig, load_models_config, load_app_config


_ADAPTER_REGISTRY: Dict[str, Type[BaseVLMAdapter]] = {
    "llava": LLaVAAdapter,
    "qwen2_vl": Qwen2VLAdapter,
    "internvl": InternVLAdapter,
    "internvl2_8b": InternVLAdapter,
}


def list_available_adapters() -> List[str]:
    """Return list of supported adapter keys."""
    return list(_ADAPTER_REGISTRY.keys())


def get_adapter(
    model_key: str,
    models_config: Optional[ModelsConfig] = None,
    app_config: Optional[AppConfig] = None,
) -> BaseVLMAdapter:
    """Instantiate and return the appropriate VLM adapter for model_key.
    
    Args:
        model_key: Canonical model key ('llava', 'qwen2_vl', 'internvl').
        models_config: Optional ModelsConfig. If None, loads from configs/models.yaml.
        app_config: Optional AppConfig. If None, loads from configs/config.yaml.

    Returns:
        Configured BaseVLMAdapter instance.

    Raises:
        KeyError: If model_key is not in registry or configuration.
        ValueError: If model configuration lacks required pinned revision.
    """
    if model_key not in _ADAPTER_REGISTRY:
        raise KeyError(
            f"Unsupported model key '{model_key}'. "
            f"Available adapters are: {list_available_adapters()}"
        )

    if models_config is None:
        models_config = load_models_config()
    if app_config is None:
        app_config = load_app_config()

    model_config = models_config.get_model(model_key)

    if app_config.execution.require_model_revision and not model_config.revision:
        raise ValueError(
            f"Strict mode violation: model '{model_key}' does not have a pinned commit revision."
        )

    adapter_cls = _ADAPTER_REGISTRY[model_key]
    return adapter_cls(
        model_key=model_key,
        model_config=model_config,
        app_config=app_config,
    )

