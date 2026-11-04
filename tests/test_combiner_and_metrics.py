"""Unit tests for feature extractor, combiner, attribution head, and metrics evaluator."""

import pytest
import numpy as np
from src.features.extractor import DiagnosticFeatureVector, FeatureExtractor
from src.perception.consistency import PerceptionDiagnosticResult
from src.retrieval.consistency import RetrievalDiagnosticResult
from src.utilization.consistency import UtilizationDiagnosticResult
from src.combiner.combiner import XMCADetectionCombiner, XMCAAttributionHead
from src.evaluation.metrics import MetricEvaluator
from src.baselines.baselines import BaselineEvaluator


def test_feature_extractor_and_missingness_masks():
    extractor = FeatureExtractor()

    p_res = PerceptionDiagnosticResult(
        sample_id="s1",
        image_id="img1",
        sp_score=0.9,
        visual_support_ratio=1.0,
        visual_contradiction_ratio=0.0,
        visual_coverage=1.0,
        grounding_uncertainty=0.1,
    )
    r_res = RetrievalDiagnosticResult(
        dataset_id="pope",
        sample_id="s1",
        is_applicable=False,
        sr_score=0.0,
        missingness_mask=1.0,
    )
    u_res = UtilizationDiagnosticResult(
        dataset_id="pope",
        sample_id="s1",
        is_applicable=False,
        su_score=0.0,
        missingness_mask=1.0,
    )

    feat = extractor.extract_features("s1", "pope", "llava", p_res, r_res, u_res, ground_truth_label=0)

    assert feat.sample_id == "s1"
    assert feat.m_r == 1.0
    assert feat.m_u == 1.0
    assert feat.sp_x_sr == 0.0  # Masked interaction
    assert feat.sp_x_su == 0.0

    raw_list = feat.to_feature_list()
    assert len(raw_list) == len(feat.feature_names())


def test_xmca_combiner_and_temperature_calibration():
    combiner = XMCADetectionCombiner()

    # Generate synthetic feature vectors for testing calibration mechanics
    train_vecs = []
    labels = []
    for i in range(20):
        # Even i: faithful (high sp, label 0), Odd i: hallucinated (low sp, label 1)
        sp = 0.9 if i % 2 == 0 else 0.2
        label = 0 if i % 2 == 0 else 1
        vec = DiagnosticFeatureVector(
            sample_id=f"test_{i}",
            dataset_id="pope",
            model_name="llava",
            ground_truth_label=label,
            sp_score=sp,
            visual_support_ratio=sp,
            visual_contradiction_ratio=1.0 - sp,
            visual_coverage=1.0,
            grounding_uncertainty=0.1,
            sr_score=0.0,
            top_relevance_score=0.0,
            mean_relevance_score=0.0,
            score_margin=0.0,
            evidence_coverage=0.0,
            m_r=1.0,
            su_score=0.0,
            entailment_ratio=0.0,
            contradiction_ratio=0.0,
            omission_ratio=0.0,
            unsupported_claim_ratio=0.0,
            m_u=1.0,
            total_claims=1,
            sp_x_sr=0.0,
            sp_x_su=0.0,
        )
        train_vecs.append(vec)
        labels.append(label)

    combiner.fit(train_vecs, labels, epochs=50)
    assert combiner.is_fitted is True

    combiner.calibrate_temperature(train_vecs, labels)
    assert combiner.temperature > 0.0

    # Test prediction on faithful vs hallucinated sample
    pred_faithful = combiner.predict(train_vecs[0])
    pred_hallucinated = combiner.predict(train_vecs[1])

    assert pred_faithful.hallucination_probability < pred_hallucinated.hallucination_probability


def test_attribution_head():
    attr_head = XMCAAttributionHead()
    combiner = XMCADetectionCombiner()

    # Hallucinated POPE sample (perception failure)
    feat_pope = DiagnosticFeatureVector(
        sample_id="pope_fail",
        dataset_id="pope",
        model_name="llava",
        sp_score=0.1,
        visual_support_ratio=0.0,
        visual_contradiction_ratio=1.0,
        visual_coverage=1.0,
        grounding_uncertainty=0.0,
        sr_score=0.0,
        top_relevance_score=0.0,
        mean_relevance_score=0.0,
        score_margin=0.0,
        evidence_coverage=0.0,
        m_r=1.0,
        su_score=0.0,
        entailment_ratio=0.0,
        contradiction_ratio=0.0,
        omission_ratio=0.0,
        unsupported_claim_ratio=0.0,
        m_u=1.0,
        total_claims=1,
        sp_x_sr=0.0,
        sp_x_su=0.0,
    )
    pred = combiner.predict(feat_pope)
    pred.predicted_label = 1
    pred.hallucination_probability = 0.95

    attr_res = attr_head.attribute_cause(feat_pope, pred)
    assert attr_res.attributed_cause == "perception"
    assert attr_res.attribution_probabilities["perception"] == 1.0


def test_metric_evaluator():
    evaluator = MetricEvaluator()
    y_true = [0, 0, 0, 1, 1, 1]
    y_score = [0.1, 0.2, 0.3, 0.7, 0.8, 0.9]

    metrics = evaluator.compute_detection_metrics(y_true, y_score, threshold=0.5)

    assert metrics.auroc == 1.0
    assert metrics.pr_auc == 1.0
    assert metrics.accuracy == 1.0
    assert metrics.f1_score == 1.0
    assert metrics.ece < 0.3
    assert metrics.brier_score < 0.1

    # Test attribution accuracy
    true_causes = ["perception", "perception", "retrieval", "utilization"]
    pred_causes = ["perception", "retrieval", "retrieval", "utilization"]
    attr_metrics = evaluator.compute_attribution_accuracy(true_causes, pred_causes)
    assert attr_metrics["overall_attribution_accuracy"] == 0.75
