"""Unit tests for XMCA-Net Component Ablation Study Runner."""

import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import torch

from scripts.run_component_ablation import (
    FEATURE_GROUPS,
    SubsetConfigClassifier,
    train_and_calibrate_classifier,
    predict_calibrated_probabilities,
    run_group_kfold_evaluation,
    to_global_key,
)
from src.combiner.model import TemperatureScaler
from src.evaluation.metrics import MetricEvaluator


def test_subset_config_classifier_forward_and_backward():
    """Verify SubsetConfigClassifier architecture across all feature subset dimensions."""
    for cfg_name, feat_cols in FEATURE_GROUPS.items():
        in_dim = len(feat_cols)
        model = SubsetConfigClassifier(in_dim, hidden_dim=32, dropout=0.1)

        batch_size = 8
        x = torch.randn(batch_size, in_dim)
        logits = model(x)

        assert logits.shape == (batch_size,)
        assert logits.dtype == torch.float32

        loss = logits.sum()
        loss.backward()
        for p in model.parameters():
            if p.requires_grad:
                assert p.grad is not None


def test_train_and_calibrate_classifier():
    """Verify training with class-balanced loss and post-hoc temperature calibration."""
    rng = np.random.RandomState(42)
    n_samples = 60
    n_features = 5

    x = rng.randn(n_samples, n_features).astype(np.float32)
    # Binary labels with some imbalance
    y = np.array([1] * 20 + [0] * 40, dtype=np.int32)

    model, scaler, mean, std = train_and_calibrate_classifier(
        x, y, epochs=15, lr=5e-3, batch_size=16, seed=42
    )

    assert isinstance(model, SubsetConfigClassifier)
    assert isinstance(scaler, TemperatureScaler)
    assert scaler.temperature > 0.0
    assert mean.shape == (n_features,)
    assert std.shape == (n_features,)

    # Test prediction
    test_x = rng.randn(10, n_features).astype(np.float32)
    probs = predict_calibrated_probabilities(model, scaler, mean, std, test_x)

    assert len(probs) == 10
    assert all(0.0 <= p <= 1.0 for p in probs)


def test_to_global_key_mapping():
    """Verify conversion of dataset image IDs to global unique keys."""
    # COCO formats
    assert to_global_key("pope", "COCO_val2014_000000100238.jpg") == "coco_000000100238"
    assert to_global_key("faithscore", "COCO_val2014_000000001268.jpg") == "coco_000000001268"
    assert to_global_key("a_okvqa", "COCO_val2014_000000557016.jpg") == "coco_000000557016"

    # InfoSeek / OVEN formats
    assert to_global_key("infoseek", "oven_001234.jpg") == "oven_001234.jpg"


def test_run_group_kfold_evaluation():
    """Verify image-group disjoint cross-validation out-of-fold probability generation."""
    rng = np.random.RandomState(42)
    n_samples = 40
    feat_cols = ["f1", "f2", "f3"]

    data = {c: rng.randn(n_samples) for c in feat_cols}
    # 8 distinct image groups, each with 5 questions
    groups = [f"img_{i // 5:03d}" for i in range(n_samples)]
    data["global_image_id"] = groups
    data["ground_truth_label"] = rng.choice([0, 1], size=n_samples)

    df = pd.DataFrame(data)
    oof_probs = run_group_kfold_evaluation(df, feat_cols, n_splits=4, seed=42)

    assert len(oof_probs) == n_samples
    assert all(0.0 <= p <= 1.0 for p in oof_probs)
    # Check that out-of-fold probabilities are non-trivial
    assert np.std(oof_probs) > 0.01

