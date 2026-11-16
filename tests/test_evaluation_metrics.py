"""Unit tests for detection and localization evaluation metrics."""

import numpy as np
import pytest

from scripts.evaluate_detection import (
    evaluate_detection_metrics,
    select_optimal_f1_threshold,
    compute_calibration_ece,
)
from scripts.evaluate_localization import evaluate_token_localization


def test_select_optimal_f1_threshold_no_leakage():
    # Validation labels and scores
    val_y = np.array([0, 0, 0, 1, 1, 1])
    val_scores = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])

    thresh = select_optimal_f1_threshold(val_scores, val_y)
    assert 0.3 <= thresh <= 0.8

    # Test set evaluation using validation-selected threshold
    test_y = np.array([0, 1])
    test_scores = np.array([0.2, 0.85])
    metrics = evaluate_detection_metrics(test_y, test_scores, fixed_threshold=thresh)

    assert metrics["auroc"] == 1.0
    assert metrics["f1"] == 1.0


def test_calibration_ece_computation():
    probs = np.array([0.1, 0.2, 0.8, 0.9])
    labels = np.array([0, 0, 1, 1])

    ece = compute_calibration_ece(probs, labels, n_bins=5)
    assert isinstance(ece, float)
    assert 0.0 <= ece <= 0.5


def test_token_localization_metrics():
    y_true = [[1, 1, 0, 0], [0, 1, 0]]
    y_pred = [[1, 0, 0, 0], [0, 1, 0]]

    metrics = evaluate_token_localization(y_true, y_pred)
    assert "token_f1" in metrics
    assert "token_precision" in metrics
    assert "token_recall" in metrics
    assert "token_iou" in metrics
    assert metrics["token_precision"] == 1.0

