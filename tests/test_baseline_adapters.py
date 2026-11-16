"""Unit tests for baseline adapters and forward passes."""

import numpy as np
import torch
import pytest

from src.baselines.halp_probe import HALPProbeDetector, HALPProbeMLP
from src.baselines.viser_detector import VisERDetector
from src.baselines.halloc_detector import HalLocDetector
from src.baselines.refind_adapter import REFINDDetector
from src.baselines.opera_generation_runner import OperaOverTrustLogitsProcessor
from src.baselines.realign_training import RDPOLoss


def test_halp_probe_training_and_calibration():
    np.random.seed(42)
    torch.manual_seed(42)

    X_train = np.random.randn(100, 32).astype(np.float32)
    y_train = (X_train[:, 0] > 0.0).astype(int)

    detector = HALPProbeDetector(input_dim=32, hidden_dim=16)
    detector.fit(X_train, y_train, epochs=5, batch_size=16)

    score = detector.predict_score(X_train[0])
    prob = detector.predict_calibrated_proba(X_train[0])

    assert isinstance(score, float)
    assert isinstance(prob, float)
    assert 0.0 <= prob <= 1.0


def test_viser_detector_scoring():
    viser = VisERDetector(alpha=0.5)

    # Object 1: high visual evidence (-0.2), low prior (-2.5) -> High Visual Reliance (grounded)
    score1 = viser.compute_object_score("dog", 2, ve_logprob=-0.2, prior_logprob=-2.5)
    assert score1.visual_reliance > 0.0

    # Object 2: low visual evidence (-3.0), high prior (-0.3) -> Low Visual Reliance (hallucinated)
    score2 = viser.compute_object_score("cat", 5, ve_logprob=-3.0, prior_logprob=-0.3)
    assert score2.visual_reliance < 0.0
    assert score2.grounding_score < score1.grounding_score

    # Sequence scoring
    hallucination_risk, objects = viser.score_sequence(
        object_tokens=["dog", "cat"],
        token_indices=[2, 5],
        ve_logprobs=[-0.2, -3.0],
        prior_logprobs=[-2.5, -0.3],
    )
    assert len(objects) == 2
    assert hallucination_risk > 0.0


def test_halloc_multihead_forward():
    # 1. Official 4-head HalLoc architecture (CVPR 2025)
    detector = HalLocDetector(hidden_dim=16, include_custom_other=False)
    x = torch.randn(2, 5, 16)
    out = detector.model(x)

    assert "object" in out
    assert "attribute" in out
    assert "relationship" in out
    assert "scene" in out
    assert "other" not in out
    assert out["object"].shape == (2, 5)

    # 2. HalLoc-4 + custom Other extension
    detector_ext = HalLocDetector(hidden_dim=16, include_custom_other=True)
    out_ext = detector_ext.model(x)
    assert "other" in out_ext
    assert out_ext["other"].shape == (2, 5)


def test_refind_csr_computation():
    refind = REFINDDetector()

    # Grounded token: probability with context is higher than without context
    # e.g. logp with context = -0.5, logp without context = -2.0
    csr_grounded = refind.compute_token_csr(logp_with_context=-0.5, logp_without_context=-2.0)
    assert csr_grounded == 0.25

    # Insensitive / hallucinated token: probability without context is higher
    # e.g. logp with context = -2.5, logp without context = -0.5
    csr_hallucinated = refind.compute_token_csr(logp_with_context=-2.5, logp_without_context=-0.5)
    assert csr_hallucinated == 5.0

    raw_risk, output = refind.score_sequence(
        tokens=["apple", "tree"],
        logprobs_with_context=[-0.5, -0.6],
        logprobs_without_context=[-2.0, -2.2],
    )
    assert len(output.token_csr_scores) == 2
    assert isinstance(raw_risk, float)


def test_opera_logits_processor():
    processor = OperaOverTrustLogitsProcessor(scale_factor=10.0, threshold=5.0)
    input_ids = torch.ones((2, 10), dtype=torch.long)
    # Create an artificial sharp peak in scores
    scores = torch.zeros((2, 50), dtype=torch.float32)
    scores[:, 0] = 12.0
    scores[:, 1] = 0.5

    penalized = processor(input_ids, scores)
    # The peak should be penalized
    assert penalized[0, 0].item() < 12.0


def test_rdpo_loss():
    loss_fn = RDPOLoss(beta=0.1, lambda_vis=0.5)
    p_w = torch.tensor([1.2, 0.8])
    p_l = torch.tensor([0.2, 0.1])
    r_w = torch.tensor([1.0, 0.7])
    r_l = torch.tensor([0.3, 0.2])

    loss = loss_fn(p_w, p_l, r_w, r_l)
    assert isinstance(loss.item(), float)
    assert not torch.isnan(loss)

