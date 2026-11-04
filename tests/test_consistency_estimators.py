"""Unit tests for Perception (S_P), Retrieval (S_R), and Utilization (S_U) estimators."""

import pytest
from src.claims.extractor import AtomicClaim
from src.perception.consistency import PerceptionConsistencyEstimator
from src.retrieval.consistency import RetrievalConsistencyEstimator
from src.retrieval.retriever import RetrievalResult, RetrievedPassage
from src.utilization.consistency import UtilizationConsistencyEstimator


def test_perception_estimator_sp_calculation():
    estimator = PerceptionConsistencyEstimator()
    claims = [
        AtomicClaim(
            claim_id="c1",
            claim_text="There is a cat",
            claim_type="object_presence",
            subject="cat",
        ),
        AtomicClaim(
            claim_id="c2",
            claim_text="The cat is sleeping",
            claim_type="attribute",
            subject="cat",
        ),
    ]

    # Ground truth contains cat -> should be supported
    res_supported = estimator.compute_sp(
        sample_id="s1",
        image_id="img1",
        claims=claims,
        ground_truth_objects=["cat", "couch"],
    )
    assert res_supported.sp_score >= 0.8
    assert res_supported.visual_support_ratio == 1.0
    assert res_supported.visual_contradiction_ratio == 0.0

    # Ground truth contains dog only -> cat should be contradicted
    res_contradicted = estimator.compute_sp(
        sample_id="s2",
        image_id="img2",
        claims=claims,
        ground_truth_objects=["dog"],
    )
    assert res_contradicted.sp_score < 0.5
    assert res_contradicted.visual_contradiction_ratio == 1.0


def test_retrieval_estimator_sr_calculation():
    estimator = RetrievalConsistencyEstimator()

    # Case 1: Non-RAG POPE
    res_pope = RetrievalResult(
        dataset_id="pope",
        sample_id="p1",
        is_applicable=False,
        query="Is there a car?",
        missingness_mask=1.0,
    )
    sr_diag_pope = estimator.compute_sr(res_pope)
    assert sr_diag_pope.is_applicable is False
    assert sr_diag_pope.missingness_mask == 1.0
    assert sr_diag_pope.sr_score == 0.0

    # Case 2: Active RAG with retrieved passages
    passages = [
        RetrievedPassage(
            doc_id="d1",
            passage_text="Norman architecture originated in Normandy and spread to England.",
            relevance_score=12.5,
            rank=1,
            source_url="https://example.org/norman",
            corpus_split_eligibility="eval_held_out",
        ),
        RetrievedPassage(
            doc_id="d2",
            passage_text="Gothic architecture followed Norman architecture in Europe.",
            relevance_score=8.0,
            rank=2,
            source_url="https://example.org/gothic",
            corpus_split_eligibility="eval_held_out",
        ),
    ]
    res_rag = RetrievalResult(
        dataset_id="a_okvqa",
        sample_id="a1",
        is_applicable=True,
        query="What is Norman architecture?",
        retrieved_passages=passages,
        top_score=12.5,
        mean_score=10.25,
        score_margin=4.5,
        evidence_coverage=0.8,
        missingness_mask=0.0,
    )
    sr_diag_rag = estimator.compute_sr(res_rag)
    assert sr_diag_rag.is_applicable is True
    assert sr_diag_rag.missingness_mask == 0.0
    assert sr_diag_rag.sr_score > 0.5
    assert sr_diag_rag.retrieval_failure_category == "sufficiently_retrieved"


def test_utilization_estimator_su_calculation():
    estimator = UtilizationConsistencyEstimator()

    # Case 1: Non-RAG
    su_non_rag = estimator.compute_su(
        dataset_id="faithscore",
        sample_id="fs1",
        claims=[],
        retrieval_result=None,
    )
    assert su_non_rag.is_applicable is False
    assert su_non_rag.missingness_mask == 1.0

    # Case 2: Active RAG with entailed claims
    claims = [
        AtomicClaim(
            claim_id="c1",
            claim_text="Norman architecture has round arches",
            claim_type="attribute",
            subject="norman architecture",
            object_or_value="round arches",
        )
    ]
    passages = [
        RetrievedPassage(
            doc_id="d1",
            passage_text="Norman architecture is known for round arches and thick stone walls.",
            relevance_score=15.0,
            rank=1,
            source_url="https://example.org",
            corpus_split_eligibility="train_only",
        )
    ]
    ret_res = RetrievalResult(
        dataset_id="a_okvqa",
        sample_id="a1",
        is_applicable=True,
        query="Norman architecture characteristics",
        retrieved_passages=passages,
        top_score=15.0,
        missingness_mask=0.0,
    )
    su_res = estimator.compute_su(
        dataset_id="a_okvqa",
        sample_id="a1",
        claims=claims,
        retrieval_result=ret_res,
    )
    assert su_res.is_applicable is True
    assert su_res.missingness_mask == 0.0
    assert su_res.entailment_ratio == 1.0
    assert su_res.su_score >= 0.7
