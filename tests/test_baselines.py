"""Unit tests for neural single-axis baseline detectors."""

import numpy as np
import pytest
import torch

from src.baselines.baselines import (
    SingleAxisDetector,
    SingleAxisNeuralClassifier,
    BaselineEvaluator,
)
from src.features.extractor import DiagnosticFeatureVector


def test_single_axis_neural_classifier_forward():
    """Verify PyTorch forward pass and gradient backpropagation."""
    model = SingleAxisNeuralClassifier(in_features=5, hidden_dim=16)
    x = torch.randn(8, 5)
    logits = model(x)
    assert logits.shape == (8,)
    loss = logits.sum()
    loss.backward()
    for p in model.parameters():
        if p.requires_grad:
            assert p.grad is not None


def test_single_axis_detector_fit_and_predict():
    """Verify training with class balancing and temperature calibration."""
    detector = SingleAxisDetector("perception", in_features=5)
    rng = np.random.RandomState(42)
    X = rng.randn(40, 5).astype(np.float32)
    y = np.array([1] * 15 + [0] * 25, dtype=np.int32)

    detector.fit(X, y, epochs=10)
    assert detector.is_fitted is True
    assert detector.scaler.temperature > 0.0

    test_sample = rng.randn(5).astype(np.float32)
    prob = detector.predict_proba(test_sample)
    assert 0.0 <= prob <= 1.0


def test_baseline_evaluator_neural_interface():
    """Verify BaselineEvaluator predicts calibrated probabilities using neural heads."""
    evaluator = BaselineEvaluator()

    vec = DiagnosticFeatureVector(
        sample_id="test_sample_1",
        dataset_id="infoseek",
        model_name="llava",
        ground_truth_label=1,
        sp_score=0.25,
        visual_support_ratio=0.2,
        visual_contradiction_ratio=0.7,
        visual_coverage=0.8,
        grounding_uncertainty=0.1,
        sr_score=0.15,
        top_relevance_score=0.15,
        mean_relevance_score=0.1,
        score_margin=0.05,
        evidence_coverage=0.2,
        m_r=0.0,
        su_score=0.1,
        entailment_ratio=0.1,
        contradiction_ratio=0.8,
        omission_ratio=0.1,
        unsupported_claim_ratio=0.8,
        m_u=0.0,
        total_claims=2,
        sp_x_sr=0.0375,
        sp_x_su=0.025,
    )

    # Predictions must be valid probabilities in [0, 1] computed via neural forward pass
    p_sp = evaluator.sp_only(vec)
    p_sr = evaluator.sr_only(vec)
    p_su = evaluator.su_only(vec)

    assert 0.0 <= p_sp <= 1.0
    assert 0.0 <= p_sr <= 1.0
    assert 0.0 <= p_su <= 1.0
