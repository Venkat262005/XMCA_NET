"""Tests for run-manifest validation and matched evaluation sample IDs."""

import pytest
import tempfile
from pathlib import Path
from pydantic import ValidationError

from src.utils.manifest import EvaluationManifest, ModelRunEntry


def test_valid_manifest():
    sample_ids = [f"sample_{i:04d}" for i in range(100)]
    manifest = EvaluationManifest(
        dataset_name="pope",
        sample_tier=100,
        expected_sample_count=100,
        matched_sample_ids=sample_ids,
        model_runs={
            "llava": ModelRunEntry(
                model_key="llava",
                model_id="llava-hf/llava-1.5-7b-hf",
                revision="a272c74",
                sample_ids=sample_ids,
            ),
            "qwen2_vl": ModelRunEntry(
                model_key="qwen2_vl",
                model_id="Qwen/Qwen2-VL-7B-Instruct",
                revision="847be15",
                sample_ids=sample_ids,
            ),
        },
    )
    manifest.validate_matched_ids()
    assert len(manifest.matched_sample_ids) == 100


def test_manifest_mismatched_sample_ids():
    sample_ids = [f"sample_{i:04d}" for i in range(100)]
    divergent_ids = [f"sample_{i:04d}" for i in range(1, 101)] # One ID shifted

    manifest = EvaluationManifest(
        dataset_name="pope",
        sample_tier=100,
        expected_sample_count=100,
        matched_sample_ids=sample_ids,
        model_runs={
            "llava": ModelRunEntry(
                model_key="llava",
                model_id="llava-hf/llava-1.5-7b-hf",
                revision="a272c74",
                sample_ids=sample_ids,
            ),
            "qwen2_vl": ModelRunEntry(
                model_key="qwen2_vl",
                model_id="Qwen/Qwen2-VL-7B-Instruct",
                revision="847be15",
                sample_ids=divergent_ids,
            ),
        },
    )
    with pytest.raises(ValueError, match="Mismatch in evaluation sample IDs"):
        manifest.validate_matched_ids()


def test_manifest_invalid_tier():
    with pytest.raises(ValidationError, match="Sample tier 250 not recognized"):
        EvaluationManifest(
            dataset_name="pope",
            sample_tier=250, # Must be 100, 300, 600, or 1000
            expected_sample_count=250,
            matched_sample_ids=["s1"],
        )


def test_manifest_save_and_load():
    sample_ids = [f"sample_{i:04d}" for i in range(100)]
    manifest = EvaluationManifest(
        dataset_name="faithscore",
        sample_tier=100,
        expected_sample_count=100,
        matched_sample_ids=sample_ids,
        model_runs={
            "llava": ModelRunEntry(
                model_key="llava",
                model_id="llava-hf/llava-1.5-7b-hf",
                revision="a272c74",
                sample_ids=sample_ids,
            )
        },
    )
    with tempfile.TemporaryDirectory() as tmp_dir:
        out_file = Path(tmp_dir) / "manifest_test.json"
        manifest.save(out_file)
        loaded = EvaluationManifest.load(out_file)
        assert loaded.dataset_name == "faithscore"
        assert loaded.sample_tier == 100
        assert loaded.matched_sample_ids == sample_ids

