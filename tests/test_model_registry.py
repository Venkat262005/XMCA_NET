"""Tests for model registry, explicit missing-model errors, and revision requirements."""

import pytest
from pathlib import Path
from pydantic import ValidationError

from src.utils.config import load_models_config, ModelsConfig, ModelItemConfig


def test_model_registry_valid_access():
    cfg = load_models_config(Path("configs/models.yaml"), require_revision=True)
    for model_name in ["llava", "qwen2_vl", "internvl"]:
        model = cfg.get_model(model_name)
        assert isinstance(model, ModelItemConfig)
        assert model.revision is not None


def test_explicit_missing_model_error():
    cfg = load_models_config(Path("configs/models.yaml"), require_revision=True)
    with pytest.raises(KeyError, match="Explicit missing-model error: model 'gpt4v_unsupported' is not configured"):
        cfg.get_model("gpt4v_unsupported")


def test_missing_model_revision_raises_error():
    invalid_data = {
        "models": {
            "llava": {
                "family": "LLaVA",
                "model_id": "llava-hf/llava-1.5-7b-hf",
                "revision": None, # Missing revision violates strict mode
                "architecture": "LlavaForConditionalGeneration",
                "dtype": "float16",
                "device_map": "auto",
                "multimodal_processor": "llava-hf/llava-1.5-7b-hf"
            },
            "qwen2_vl": {
                "family": "Qwen2-VL",
                "model_id": "Qwen/Qwen2-VL-7B-Instruct",
                "revision": "847be15",
                "architecture": "Qwen2VLForConditionalGeneration",
                "dtype": "float16",
                "device_map": "auto",
                "multimodal_processor": "Qwen/Qwen2-VL-7B-Instruct"
            },
            "internvl": {
                "family": "InternVL",
                "model_id": "OpenGVLab/InternVL2-8B",
                "revision": "9826f68",
                "architecture": "InternVLChatModel",
                "dtype": "float16",
                "device_map": "auto",
                "multimodal_processor": "OpenGVLab/InternVL2-8B"
            }
        }
    }
    # Direct model loading with require_revision=True check
    with pytest.raises(ValueError, match="Strict mode violation: model 'llava' does not have an explicit git commit revision pin"):
        cfg = ModelsConfig(**invalid_data)
        for name, m in cfg.models.items():
            if not m.revision:
                raise ValueError(
                    f"Strict mode violation: model '{name}' does not have an explicit git commit revision pin."
                )

