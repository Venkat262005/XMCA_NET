"""Feature extraction and vector compilation module for XMCA-Net.

Assembles multi-axis consistency diagnostics into aligned feature vectors with explicit
missingness indicator masks (for POPE/FaithScore non-RAG samples).
"""

from typing import List, Dict, Any, Optional
import pandas as pd
from pydantic import BaseModel, Field

from src.perception.consistency import PerceptionDiagnosticResult
from src.retrieval.consistency import RetrievalDiagnosticResult
from src.utilization.consistency import UtilizationDiagnosticResult


class DiagnosticFeatureVector(BaseModel):
    """Complete diagnostic feature representation for a single benchmark sample."""

    sample_id: str
    dataset_id: str
    model_name: str
    ground_truth_label: Optional[int] = None  # 1 = hallucination, 0 = faithful

    # Perception features
    sp_score: float
    visual_support_ratio: float
    visual_contradiction_ratio: float
    visual_coverage: float
    grounding_uncertainty: float

    # Retrieval features & missingness
    sr_score: float
    top_relevance_score: float
    mean_relevance_score: float
    score_margin: float
    evidence_coverage: float
    m_r: float  # 1.0 if RAG is missing / not applicable, 0.0 if active

    # Utilization features & missingness
    su_score: float
    entailment_ratio: float
    contradiction_ratio: float
    omission_ratio: float
    unsupported_claim_ratio: float
    m_u: float  # 1.0 if RAG is missing / not applicable, 0.0 if active

    # Cross-modal interaction features
    total_claims: int
    sp_x_sr: float
    sp_x_su: float

    def to_feature_list(self) -> List[float]:
        """Convert into numerical feature vector for combiner consumption."""
        return [
            self.sp_score,
            self.visual_support_ratio,
            self.visual_contradiction_ratio,
            self.visual_coverage,
            self.grounding_uncertainty,
            self.sr_score,
            self.top_relevance_score,
            self.mean_relevance_score,
            self.score_margin,
            self.evidence_coverage,
            self.m_r,
            self.su_score,
            self.entailment_ratio,
            self.contradiction_ratio,
            self.omission_ratio,
            self.unsupported_claim_ratio,
            self.m_u,
            float(self.total_claims),
            self.sp_x_sr,
            self.sp_x_su,
        ]

    @staticmethod
    def feature_names() -> List[str]:
        return [
            "sp_score",
            "visual_support_ratio",
            "visual_contradiction_ratio",
            "visual_coverage",
            "grounding_uncertainty",
            "sr_score",
            "top_relevance_score",
            "mean_relevance_score",
            "score_margin",
            "evidence_coverage",
            "m_r",
            "su_score",
            "entailment_ratio",
            "contradiction_ratio",
            "omission_ratio",
            "unsupported_claim_ratio",
            "m_u",
            "total_claims",
            "sp_x_sr",
            "sp_x_su",
        ]


class FeatureExtractor:
    """Assembles diagnostic results into structured feature vectors."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}

    def extract_features(
        self,
        sample_id: str,
        dataset_id: str,
        model_name: str,
        p_res: PerceptionDiagnosticResult,
        r_res: RetrievalDiagnosticResult,
        u_res: UtilizationDiagnosticResult,
        ground_truth_label: Optional[int] = None,
    ) -> DiagnosticFeatureVector:
        """Combine diagnostic outputs into a single aligned feature record."""
        # Cross-modal interaction terms (zeroed if axis is masked/inapplicable)
        sp_x_sr = (p_res.sp_score * r_res.sr_score) if r_res.missingness_mask == 0.0 else 0.0
        sp_x_su = (p_res.sp_score * u_res.su_score) if u_res.missingness_mask == 0.0 else 0.0

        return DiagnosticFeatureVector(
            sample_id=sample_id,
            dataset_id=dataset_id,
            model_name=model_name,
            ground_truth_label=ground_truth_label,
            sp_score=p_res.sp_score,
            visual_support_ratio=p_res.visual_support_ratio,
            visual_contradiction_ratio=p_res.visual_contradiction_ratio,
            visual_coverage=p_res.visual_coverage,
            grounding_uncertainty=p_res.grounding_uncertainty,
            sr_score=r_res.sr_score,
            top_relevance_score=r_res.top_relevance_score,
            mean_relevance_score=r_res.mean_relevance_score,
            score_margin=r_res.score_margin,
            evidence_coverage=r_res.evidence_coverage,
            m_r=r_res.missingness_mask,
            su_score=u_res.su_score,
            entailment_ratio=u_res.entailment_ratio,
            contradiction_ratio=u_res.contradiction_ratio,
            omission_ratio=u_res.omission_ratio,
            unsupported_claim_ratio=u_res.unsupported_claim_ratio,
            m_u=u_res.missingness_mask,
            total_claims=p_res.total_claims_evaluated,
            sp_x_sr=round(float(sp_x_sr), 4),
            sp_x_su=round(float(sp_x_su), 4),
        )

    def to_dataframe(self, vectors: List[DiagnosticFeatureVector]) -> pd.DataFrame:
        """Convert a list of vectors to a pandas DataFrame."""
        return pd.DataFrame([v.model_dump() for v in vectors])

