"""Unit tests for Vision-Language Model adapters and Task 2 requirements.

Verifies:
  1. Adapter registry and factory contracts.
  2. Prompt formatting and dynamic preprocessing across backbones.
  3. Multi-GPU device mapping computation (InternVL2-8B split).
  4. Strict environment enforcement (blocking local Mac execution).
  5. InfoSeek un-materialized image gating guard.
  6. Cross-partition image leakage isolation across all training/calibration partitions.
  7. Generation output schema conformance.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest
from PIL import Image

from src.generation.registry import get_adapter, list_available_adapters
from src.generation.adapters.llava import LLaVAAdapter
from src.generation.adapters.qwen2_vl import Qwen2VLAdapter
from src.generation.adapters.internvl import (
    InternVLAdapter,
    build_internvl2_8b_device_map,
    dynamic_preprocess,
    find_best_aspect_ratio,
)
from src.generation.base import VLMGenerationOutput
from src.generation.generator import GenerationOrchestrator
from src.data.loader import verify_partition_isolation
from src.utils.config import load_app_config, load_models_config, ModelItemConfig
from src.utils.environment import EnvironmentVerificationError


def test_list_available_adapters():
    adapters = list_available_adapters()
    assert set(adapters) == {"llava", "qwen2_vl", "internvl"}


def test_adapter_instantiation():
    models_cfg = load_models_config()
    app_cfg = load_app_config()

    llava = get_adapter("llava", models_cfg, app_cfg)
    assert isinstance(llava, LLaVAAdapter)
    assert llava.model_id == "llava-hf/llava-1.5-7b-hf"
    assert llava.revision == "a272c74"
    assert not llava.is_loaded

    qwen = get_adapter("qwen2_vl", models_cfg, app_cfg)
    assert isinstance(qwen, Qwen2VLAdapter)
    assert qwen.model_id == "Qwen/Qwen2-VL-7B-Instruct"
    assert qwen.revision == "eed13092ef92e448dd6875b2a00151bd3f7db0ac"
    assert not qwen.is_loaded

    internvl = get_adapter("internvl", models_cfg, app_cfg)
    assert isinstance(internvl, InternVLAdapter)
    assert internvl.model_id == "OpenGVLab/InternVL2-2B"
    assert internvl.revision == "e4f6747bd20f139e637642c6a058c6bd00b36919"
    assert not internvl.is_loaded


def test_unknown_adapter_raises_key_error():
    with pytest.raises(KeyError, match="Unsupported model key 'unknown_vlm'"):
        get_adapter("unknown_vlm")


def test_adapter_missing_revision_raises_error():
    models_cfg = load_models_config()
    app_cfg = load_app_config()
    models_cfg.models["llava"].revision = None

    with pytest.raises(ValueError, match="Strict mode violation: model 'llava' does not have a pinned commit revision"):
        get_adapter("llava", models_cfg, app_cfg)


def test_llava_prompt_formatting():
    llava = get_adapter("llava")
    formatted = llava.format_prompt("Is there a cat?")
    assert "Is there a cat?" in formatted
    assert "<image>" in formatted or "USER:" in formatted


def test_qwen2_vl_prompt_formatting():
    qwen = get_adapter("qwen2_vl")
    formatted = qwen.format_prompt("Is there a dog?")
    assert "Is there a dog?" in formatted


def test_internvl_prompt_formatting():
    internvl = get_adapter("internvl")
    formatted = internvl.format_prompt("Is there a car?")
    assert formatted.startswith("<image>\n")
    assert "Is there a car?" in formatted


def test_internvl_aspect_ratio_and_dynamic_preprocess():
    test_img = Image.new("RGB", (800, 600), color="blue")
    best_ratio = find_best_aspect_ratio(
        aspect_ratio=800 / 600,
        target_ratios=[(1, 1), (1, 2), (2, 1), (2, 2)],
        orig_width=800,
        orig_height=600,
        image_size=448,
    )
    assert best_ratio in [(1, 1), (2, 2), (2, 1)]

    patches = dynamic_preprocess(test_img, min_num=1, max_num=4, image_size=448, use_thumbnail=True)
    assert len(patches) >= 2
    for p in patches:
        assert isinstance(p, Image.Image)
        assert p.size == (448, 448)


def test_internvl2_8b_device_map_structure():
    dmap = build_internvl2_8b_device_map(num_gpus=2)
    assert dmap["vision_model"] == 0
    assert dmap["mlp1"] == 0
    assert dmap["language_model.model.embed_tokens"] == 0
    assert dmap["language_model.model.layers.0"] == 0
    assert dmap["language_model.model.layers.31"] == 1
    assert dmap["language_model.output"] == 1
    assert dmap["language_model.lm_head"] == 1


def test_strict_environment_blocks_local_mac_load():
    llava = get_adapter("llava")
    with pytest.raises(EnvironmentVerificationError, match="Strict execution mode violation"):
        llava.load()


def test_infoseek_unmaterialized_image_guard():
    app_cfg = load_app_config()
    models_cfg = load_models_config()
    orchestrator = GenerationOrchestrator(app_cfg, models_cfg)

    with pytest.raises(RuntimeError, match="InfoSeek image-conditioned inference is BLOCKED"):
        orchestrator.run_inference_sample(
            model_key="llava",
            sample_id="test_infoseek_sample",
            dataset="infoseek",
            question="What is this building?",
            image_path="data/images/infoseek/unmaterialized_oven.jpg",
        )


def test_partition_isolation_bars_overlapping_test_images():
    overlapping_id = "coco_000000506483"

    for partition in ["combiner_training", "calibration", "threshold_selection", "early_stopping", "model_selection"]:
        is_isolated, violations = verify_partition_isolation([overlapping_id], partition)
        assert not is_isolated, f"Overlapping image {overlapping_id} must not be eligible for {partition}"
        assert overlapping_id in violations


def test_mock_generation_workflow_conformance():
    import sys
    mock_torch = MagicMock()
    mock_torch.no_grad.return_value.__enter__.return_value = None
    mock_torch.no_grad.return_value.__exit__.return_value = None
    mock_torch.cuda.is_available.return_value = False

    with patch.dict(sys.modules, {"torch": mock_torch}):
        adapter = get_adapter("llava")
        adapter._is_loaded = True
        adapter.model = MagicMock()
        adapter.model.device = "cpu"
        adapter.processor = MagicMock()

        mock_input_ids = MagicMock()
        mock_input_ids.shape = (1, 3)
        mock_input_ids.to.return_value = mock_input_ids
        mock_inputs = {
            "input_ids": mock_input_ids,
            "attention_mask": MagicMock(),
        }
        adapter.processor.return_value = mock_inputs

        mock_output_ids = MagicMock()
        mock_output_ids.__getitem__.return_value.shape = [1, 2]
        adapter.model.generate.return_value = mock_output_ids
        adapter.processor.batch_decode.return_value = ["Yes, there is a cat."]

        img = Image.new("RGB", (224, 224), color="red")
        out = adapter.generate(img, "Is there a cat?")

        assert isinstance(out, VLMGenerationOutput)
        assert out.model_key == "llava"
        assert out.model_id == "llava-hf/llava-1.5-7b-hf"
        assert out.generated_text == "Yes, there is a cat."
        assert out.input_tokens == 3
        assert out.output_tokens == 2
        assert out.latency_ms >= 0.0


def test_telemetry_functions():
    from src.generation.telemetry import (
        query_nvidia_smi,
        get_cuda_memory_snapshot,
        ModelMemoryTelemetry,
    )
    smi = query_nvidia_smi()
    assert isinstance(smi, list)

    snap = get_cuda_memory_snapshot()
    assert isinstance(snap, dict)

    tel = ModelMemoryTelemetry("llava", "llava-hf/llava-1.5-7b-hf")
    tel.capture_baseline()
    tel.capture_after_load(1.23)
    tel.capture_after_inference()
    tel.capture_after_unload()
    d = tel.to_dict()
    assert d["model_key"] == "llava"
    assert d["load_duration_sec"] == 1.23
    assert "baseline_before_load" in d
    assert "after_load" in d
    assert "during_generation_peak" in d
    assert "after_unload" in d


def test_authentic_probe_definitions():
    from scripts.kaggle_vlm_smoke_test import AUTHENTIC_PROBES, verify_file_sha256
    assert len(AUTHENTIC_PROBES) == 6

    # Verify each probe file exists and hash matches
    for p in AUTHENTIC_PROBES:
        img_path = Path(p["image_path"])
        assert img_path.exists(), f"Image {img_path} must exist"
        sha = verify_file_sha256(img_path)
        assert sha == p["expected_sha256"]

