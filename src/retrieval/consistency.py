"""Retrieval Consistency (S_R) diagnostic module for XMCA-Net.

Quantifies consistency between internal/external retrieved knowledge and the generated context.
Enforces non-negotiable research rules:
- Non-RAG datasets (POPE, FaithScore) are strictly flagged NOT_APPLICABLE with missingness_mask = 1.0.
- Zero synthetic passages or dummy retrieval scores.
- Genuine ranking diagnostics: top relevance, mean relevance, score margin, query coverage.
- Failure categorization: sufficiently_retrieved, weakly_retrieved, missing_evidence, retrieval_unavailable.
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

from src.retrieval.retriever import RetrievalResult, NON_RAG_DATASETS


class RetrievalDiagnosticResult(BaseModel):
    """Diagnostic output for the Retrieval Consistency (S_R) axis."""

    dataset_id: str
    sample_id: str
    is_applicable: bool
    sr_score: float = 0.0  # Normalized consistency score in [0.0, 1.0]
    top_relevance_score: float = 0.0
    mean_relevance_score: float = 0.0
    score_margin: float = 0.0
    evidence_coverage: float = 0.0
    retrieval_failure_category: str = "retrieval_unavailable"
    # Categories: 'sufficiently_retrieved' | 'weakly_retrieved' | 'missing_evidence' | 'retrieval_unavailable'
    missingness_mask: float = 0.0  # 1.0 if not applicable / missing, 0.0 if active
    retrieved_doc_ids: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class RetrievalConsistencyEstimator:
    """Estimator for Retrieval Consistency (S_R)."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.relevance_threshold = float(self.config.get("relevance_threshold", 5.0))
        self.margin_threshold = float(self.config.get("margin_threshold", 1.0))

    def compute_sr(
        self,
        retrieval_result: RetrievalResult,
    ) -> RetrievalDiagnosticResult:
        """Compute S_R consistency score from authentic retrieval results."""
        dataset_id = retrieval_result.dataset_id.lower()

        # Strict rule: Non-RAG datasets receive explicit NOT_APPLICABLE
        if not retrieval_result.is_applicable or dataset_id in NON_RAG_DATASETS:
            return RetrievalDiagnosticResult(
                dataset_id=dataset_id,
                sample_id=retrieval_result.sample_id,
                is_applicable=False,
                sr_score=0.0,
                top_relevance_score=0.0,
                mean_relevance_score=0.0,
                score_margin=0.0,
                evidence_coverage=0.0,
                retrieval_failure_category="not_applicable",
                missingness_mask=1.0,
                retrieved_doc_ids=[],
            )

        passages = retrieval_result.retrieved_passages
        if not passages:
            return RetrievalDiagnosticResult(
                dataset_id=dataset_id,
                sample_id=retrieval_result.sample_id,
                is_applicable=True,
                sr_score=0.0,
                top_relevance_score=0.0,
                mean_relevance_score=0.0,
                score_margin=0.0,
                evidence_coverage=0.0,
                retrieval_failure_category="missing_evidence",
                missingness_mask=0.0,
                retrieved_doc_ids=[],
            )

        top_score = retrieval_result.top_score
        mean_score = retrieval_result.mean_score
        score_margin = retrieval_result.score_margin
        coverage = retrieval_result.evidence_coverage

        # Categorize retrieval quality
        if top_score >= self.relevance_threshold and score_margin >= self.margin_threshold:
            category = "sufficiently_retrieved"
        elif top_score >= (self.relevance_threshold * 0.5):
            category = "weakly_retrieved"
        else:
            category = "missing_evidence"

        # Normalize score S_R using sigmoid-like scaling on BM25/dense score
        normalized_relevance = min(1.0, max(0.0, top_score / max(1.0, self.relevance_threshold * 2.0)))
        normalized_margin = min(1.0, max(0.0, score_margin / max(1.0, self.margin_threshold * 2.0)))
        sr_score = 0.5 * normalized_relevance + 0.3 * normalized_margin + 0.2 * coverage

        doc_ids = [p.doc_id for p in passages]

        return RetrievalDiagnosticResult(
            dataset_id=dataset_id,
            sample_id=retrieval_result.sample_id,
            is_applicable=True,
            sr_score=round(float(sr_score), 4),
            top_relevance_score=round(float(top_score), 4),
            mean_relevance_score=round(float(mean_score), 4),
            score_margin=round(float(score_margin), 4),
            evidence_coverage=round(float(coverage), 4),
            retrieval_failure_category=category,
            missingness_mask=0.0,
            retrieved_doc_ids=doc_ids,
        )
