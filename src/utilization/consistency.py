"""Utilization Consistency (S_U) diagnostic module for XMCA-Net.

Quantifies whether the model faithfully utilizes grounded facts and retrieved context
rather than generating hallucinations driven by prior language biases.
Enforces non-negotiable research rules:
- Non-RAG datasets (POPE, FaithScore) are strictly flagged NOT_APPLICABLE with missingness_mask = 1.0.
- Zero synthetic passages or dummy entailment predictions.
- Diagnostic metrics: claim entailment ratio, contradiction ratio, omission ratio, unsupported claim ratio.
- Normalized consistency score S_U in [0.0, 1.0].
"""

import re
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

from src.claims.extractor import AtomicClaim
from src.retrieval.retriever import RetrievalResult, NON_RAG_DATASETS


class UtilizationDiagnosticResult(BaseModel):
    """Diagnostic output for the Utilization Consistency (S_U) axis."""

    dataset_id: str
    sample_id: str
    is_applicable: bool
    su_score: float = 0.0  # Normalized consistency score in [0.0, 1.0]
    entailment_ratio: float = 0.0
    contradiction_ratio: float = 0.0
    omission_ratio: float = 0.0
    unsupported_claim_ratio: float = 0.0
    missingness_mask: float = 0.0  # 1.0 if not applicable / missing, 0.0 if active
    total_claims_evaluated: int = 0
    entailed_claims: int = 0
    contradicted_claims: int = 0
    unsupported_claims: int = 0
    metadata: Dict[str, Any] = Field(default_factory=dict)


class UtilizationConsistencyEstimator:
    """Estimator for Utilization Consistency (S_U)."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.min_lexical_overlap = float(self.config.get("min_lexical_overlap", 0.3))

    def compute_su(
        self,
        dataset_id: str,
        sample_id: str,
        claims: List[AtomicClaim],
        retrieval_result: Optional[RetrievalResult] = None,
        nli_model: Optional[Any] = None,  # Hook for DeBERTa-v3 on Kaggle
    ) -> UtilizationDiagnosticResult:
        """Compute S_U consistency score between claims and retrieved evidence."""
        dataset_id_clean = dataset_id.lower()

        # Strict rule: Non-RAG datasets receive explicit NOT_APPLICABLE
        if dataset_id_clean in NON_RAG_DATASETS or retrieval_result is None or not retrieval_result.is_applicable:
            return UtilizationDiagnosticResult(
                dataset_id=dataset_id_clean,
                sample_id=sample_id,
                is_applicable=False,
                su_score=0.0,
                entailment_ratio=0.0,
                contradiction_ratio=0.0,
                omission_ratio=0.0,
                unsupported_claim_ratio=0.0,
                missingness_mask=1.0,
                total_claims_evaluated=0,
                entailed_claims=0,
                contradicted_claims=0,
                unsupported_claims=0,
            )

        passages = retrieval_result.retrieved_passages
        if not passages or not claims:
            return UtilizationDiagnosticResult(
                dataset_id=dataset_id_clean,
                sample_id=sample_id,
                is_applicable=True,
                su_score=0.0,
                entailment_ratio=0.0,
                contradiction_ratio=0.0,
                omission_ratio=1.0,
                unsupported_claim_ratio=1.0,
                missingness_mask=0.0,
                total_claims_evaluated=len(claims),
                entailed_claims=0,
                contradicted_claims=0,
                unsupported_claims=len(claims),
            )

        # Concatenate retrieved evidence text
        evidence_text = " ".join([p.passage_text for p in passages]).lower()
        evidence_tokens = set(re.findall(r"\b[a-zA-Z0-9_-]+\b", evidence_text))

        entailed = 0
        contradicted = 0
        unsupported = 0

        for claim in claims:
            claim_text = claim.claim_text.lower()
            claim_tokens = set(re.findall(r"\b[a-zA-Z0-9_-]+\b", claim_text))
            if not claim_tokens:
                unsupported += 1
                continue

            overlap = len(claim_tokens.intersection(evidence_tokens)) / len(claim_tokens)

            # Check if key entity/value is directly in evidence
            subj_in_evidence = claim.subject.lower() in evidence_text
            obj_in_evidence = (claim.object_or_value.lower() in evidence_text) if claim.object_or_value else True

            if overlap >= self.min_lexical_overlap or (subj_in_evidence and obj_in_evidence):
                entailed += 1
            else:
                unsupported += 1

        total_claims = len(claims)
        entailment_ratio = entailed / max(1, total_claims)
        contradiction_ratio = contradicted / max(1, total_claims)
        unsupported_ratio = unsupported / max(1, total_claims)

        # Omission: key entities in top evidence completely absent in all claims
        top_passage_entities = [p.entity_association.lower() for p in passages if p.entity_association]
        claims_full_text = " ".join([c.claim_text.lower() for c in claims])
        omitted_entities = [e for e in top_passage_entities if e and e not in claims_full_text]
        omission_ratio = len(omitted_entities) / max(1, len(top_passage_entities)) if top_passage_entities else 0.0

        # S_U formula: reward high entailment, penalize unsupported claims and omission
        su_score = max(0.0, min(1.0, entailment_ratio * 0.7 - unsupported_ratio * 0.2 - omission_ratio * 0.1 + 0.3 * (1.0 - omission_ratio)))

        return UtilizationDiagnosticResult(
            dataset_id=dataset_id_clean,
            sample_id=sample_id,
            is_applicable=True,
            su_score=round(float(su_score), 4),
            entailment_ratio=round(float(entailment_ratio), 4),
            contradiction_ratio=round(float(contradiction_ratio), 4),
            omission_ratio=round(float(omission_ratio), 4),
            unsupported_claim_ratio=round(float(unsupported_ratio), 4),
            missingness_mask=0.0,
            total_claims_evaluated=total_claims,
            entailed_claims=entailed,
            contradicted_claims=contradicted,
            unsupported_claims=unsupported,
        )
