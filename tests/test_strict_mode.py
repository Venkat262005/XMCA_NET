"""Tests for strict-mode enforcement and prohibition of dummy/synthetic artifacts."""

import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
from pydantic import ValidationError

from src.utils.config import ExecutionConfig, load_models_config
from src.utils.environment import (
    enforce_strict_environment,
    check_gpu_environment,
    EnvironmentVerificationError,
)
from src.data.loader import DatasetLoader


def test_disallow_fallbacks_flag():
    with pytest.raises(ValidationError, match="Fallbacks and synthetic data are forbidden"):
        ExecutionConfig(allow_fallbacks=True)


def test_disallow_synthetic_data_flag():
    with pytest.raises(ValidationError, match="Fallbacks and synthetic data are forbidden"):
        ExecutionConfig(allow_synthetic_data=True)


def test_gpu_environment_check_on_cpu():
    mock_torch = MagicMock()
    mock_torch.cuda.is_available.return_value = False
    with patch.dict("sys.modules", {"torch": mock_torch}):
        is_valid, msg = check_gpu_environment(require_dual_t4=True)
        assert is_valid is False
        assert "CUDA is not available" in msg


def test_gpu_environment_check_single_gpu():
    mock_torch = MagicMock()
    mock_torch.cuda.is_available.return_value = True
    mock_torch.cuda.device_count.return_value = 1
    mock_torch.cuda.get_device_name.return_value = "Tesla T4"
    with patch.dict("sys.modules", {"torch": mock_torch}):
        is_valid, msg = check_gpu_environment(require_dual_t4=True)
        assert is_valid is False
        assert "Expected 2 GPUs" in msg


def test_gpu_environment_check_non_t4_gpu():
    mock_torch = MagicMock()
    mock_torch.cuda.is_available.return_value = True
    mock_torch.cuda.device_count.return_value = 2
    mock_torch.cuda.get_device_name.side_effect = ["Tesla T4", "GeForce RTX 3090"]
    with patch.dict("sys.modules", {"torch": mock_torch}):
        is_valid, msg = check_gpu_environment(require_dual_t4=True)
        assert is_valid is False
        assert "non-T4 devices" in msg


def test_enforce_strict_environment_failure():
    class DummyConfig:
        execution = ExecutionConfig(strict_mode=True, kaggle_gpu_required=True)
        hardware = type("HW", (), {"local_inference_permitted": False})()

    mock_torch = MagicMock()
    mock_torch.cuda.is_available.return_value = False
    with patch.dict("sys.modules", {"torch": mock_torch}):
        with pytest.raises(EnvironmentVerificationError, match="Strict execution mode violation"):
            enforce_strict_environment(DummyConfig())


def test_dataset_loader_prohibits_fallbacks_on_missing_file():
    loader = DatasetLoader("pope", Path("non_existent_data_file.json"))
    with pytest.raises(FileNotFoundError, match="Synthetic data and dummy fallbacks are strictly prohibited"):
        loader.load_samples(tier=100)
