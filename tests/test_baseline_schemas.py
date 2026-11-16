"""Unit tests for baseline schemas and contracts."""

import pytest
from src.baselines.common_schema import (
    BaselineDetectionOutput,
    BaselineLocalizationOutput,
    BaselineMitigationOutput,
    ContextSensitivityOutput,
)


def test_baseline_detection_output_schema():
    out = BaselineDetectionOutput(
        sample_id="test_001",
        dataset_id="pope",
        model_name="llava_1.5_7b",
        baseline_id="halp",
        raw_score=1.45,
        calibrated_risk=0.81,
        metadata={"target": "query_tokens"},
    )
    assert out.sample_id == "test_001"
    assert out.raw_score == 1.45
    assert out.calibrated_risk == 0.81


def test_baseline_localization_output_schema():
    out = BaselineLocalizationOutput(
        sample_id="test_002",
        dataset_id="faithscore",
        model_name="qwen2_vl_7b",
        baseline_id="halloc",
        tokens=["a", "red", "apple", "on", "a", "table"],
        token_scores=[0.05, 0.92, 0.10, 0.02, 0.01, 0.03],
        category_token_scores={
            "attribute": [0.01, 0.92, 0.01, 0.01, 0.01, 0.01],
            "object": [0.01, 0.01, 0.10, 0.01, 0.01, 0.01],
        },
    )
    assert len(out.tokens) == 6
    assert out.token_scores[1] == 0.92
    assert out.category_token_scores["attribute"][1] == 0.92


def test_context_sensitivity_output_schema():
    out = ContextSensitivityOutput(
        sample_id="test_003",
        dataset_id="infoseek",
        model_name="internvl2_2b",
        token_csr_scores=[1.2, 0.8, -0.4, 2.1],
        mean_csr=0.925,
        min_csr=-0.4,
        retrieval_reliance_score=1.0,
    )
    assert out.min_csr == -0.4
    assert len(out.token_csr_scores) == 4

