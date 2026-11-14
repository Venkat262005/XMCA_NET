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

    # Without a physical image (detector inactive), no ground truth should be consulted.
    # All claims receive neutral insufficient-evidence status (conf=0.5) and sp_score=0.5.
    # This verifies Fix 6: GT labels must NOT leak into sp_score when image is absent.
    res_no_image = estimator.compute_sp(
        sample_id="s1",
        image_id="img1",
        claims=claims,
        ground_truth_objects=["cat", "couch"],  # GT provided but must NOT affect sp_score
    )
    # With no image, all claims are insufficient_evidence — sp_score should be 0.5 (neutral)
    assert res_no_image.sp_score == pytest.approx(0.5, abs=0.1), (
        "sp_score must be neutral (0.5) when no image is available. "
        "Ground truth annotations must NOT be consulted as a fallback — "
        "that would leak labels into the feature vector sent to the neural combiner."
    )
    assert res_no_image.visual_contradiction_ratio == 0.0, (
        "No contradictions should be recorded without an active neural detector."
    )

    # Edge case: empty claims list → trivial consistent result
    res_empty = estimator.compute_sp(
        sample_id="s2",
        image_id="img2",
        claims=[],
    )
    assert res_empty.sp_score == 1.0
    assert res_empty.total_claims_evaluated == 0


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
    # Note: The DeBERTa-v3 NLI model is not available in the offline local test environment.
    # The correct behavior per Fix 1 is: when the neural model cannot load, compute_su
    # surfaces a not-applicable result (m_u=1.0) instead of substituting fake lexical scores.
    # In Kaggle (where DeBERTa is loaded), su_score = mean_entailment - mean_contradiction.
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
    # In both outcomes (model loaded or not), is_applicable governs the result:
    if su_res.is_applicable:
        # DeBERTa loaded successfully: verify pure neural NLI output
        # su_score = max(0, mean_entailment - mean_contradiction); no manual coefficients
        assert su_res.missingness_mask == 0.0
        assert 0.0 <= su_res.su_score <= 1.0
        assert su_res.su_score == pytest.approx(
            max(0.0, su_res.entailment_ratio - su_res.contradiction_ratio), abs=0.05
        ), "su_score must equal entailment - contradiction with no manual weighting"
    else:
        # DeBERTa unavailable (expected in offline test env): result correctly marked not-applicable
        # This validates Fix 1: no fake lexical scores are substituted when neural model is offline
        assert su_res.missingness_mask == 1.0, (
            "When DeBERTa-v3 NLI cannot load, su result must be marked not-applicable (m_u=1.0). "
            "No lexical word-overlap approximation may substitute for genuine NLI probabilities."
        )

