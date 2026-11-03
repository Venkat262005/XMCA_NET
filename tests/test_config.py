"""Tests for configuration validation."""

import pytest
from pathlib import Path
from pydantic import ValidationError

from src.utils.config import (
    load_app_config,
    load_datasets_config,
    load_models_config,
    AppConfig,
    DatasetsConfig,
    ModelsConfig,
    EvaluationConfig,
)


def test_load_valid_app_config():
    cfg = load_app_config(Path("configs/config.yaml"))
    assert isinstance(cfg, AppConfig)
    assert cfg.project.name == "XMCA-Net"
    assert cfg.execution.strict_mode is True
    assert cfg.execution.allow_fallbacks is False
    assert cfg.execution.allow_synthetic_data is False
    assert cfg.evaluation.progressive_tiers == [100, 300, 600, 1000]


def test_load_valid_datasets_config():
    cfg = load_datasets_config(Path("configs/datasets.yaml"))
    assert isinstance(cfg, DatasetsConfig)
    expected_datasets = {"pope", "faithscore", "a_okvqa", "infoseek"}
    assert set(cfg.datasets.keys()) == expected_datasets
    for name, ds in cfg.datasets.items():
        assert ds.supported_tiers == [100, 300, 600, 1000]
        assert len(ds.required_fields) > 0


def test_load_valid_models_config():
    cfg = load_models_config(Path("configs/models.yaml"), require_revision=True)
    assert isinstance(cfg, ModelsConfig)
    expected_models = {"llava", "qwen2_vl", "internvl"}
    assert set(cfg.models.keys()) == expected_models
    for name, m in cfg.models.items():
        assert m.revision is not None
        assert len(m.revision) > 0


def test_invalid_evaluation_tiers():
    with pytest.raises(ValidationError):
        EvaluationConfig(progressive_tiers=[50, 100]) # Not the allowed tiers


def test_missing_required_dataset():
    data = {
        "datasets": {
            "pope": {
                "name": "POPE",
                "full_name": "POPE",
                "task_type": "obj",
                "splits": ["test"],
                "supported_tiers": [100, 300, 600, 1000],
                "has_attribution_ground_truth": True,
                "attribution_focus": "perception",
                "source": {"repository": "repo", "data_file": "file", "images_source": "coco"},
                "required_fields": ["id"]
            }
            # Missing faithscore, a_okvqa, infoseek
        }
    }
    with pytest.raises(ValidationError, match="exactly four datasets required"):
        DatasetsConfig(**data)

