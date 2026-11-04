"""Perception Consistency (S_P) diagnostic module.

Quantifies visual grounding consistency between visual features and generated tokens/claims.
Evaluates:
- Visual support ratio (proportion of claims grounded in detected visual objects/regions)
- Visual contradiction ratio (proportion of claims contradicted by visual evidence)
- Visual coverage (extent of image-described entities verified)
- Grounding uncertainty (dispersion in visual alignment confidence)
- Output consistency score S_P in [0.0, 1.0], where 1.0 indicates perfect visual consistency.
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

from src.claims.extractor import AtomicClaim, ClaimExtractionResult


class PerceptionDiagnosticResult(BaseModel):
    """Diagnostic output for the Perception Consistency (S_P) axis."""

    sample_id: str
    image_id: str
    sp_score: float = 0.0  # Normalized consistency score in [0.0, 1.0]
    visual_support_ratio: float = 0.0
    visual_contradiction_ratio: float = 0.0
    visual_coverage: float = 0.0
    grounding_uncertainty: float = 0.0
    total_claims_evaluated: int = 0
    supported_claims: int = 0
    contradicted_claims: int = 0
    verified_claims: List[AtomicClaim] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class PerceptionConsistencyEstimator:
    """Estimator for Perception Consistency (S_P)."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.threshold = float(self.config.get("grounding_threshold", 0.5))

    def compute_sp(
        self,
        sample_id: str,
        image_id: str,
        claims: List[AtomicClaim],
        ground_truth_objects: Optional[List[str]] = None,
        detection_detections: Optional[List[Dict[str, Any]]] = None,
    ) -> PerceptionDiagnosticResult:
        """Compute S_P consistency score across extracted atomic claims.

        Args:
            sample_id: Unique benchmark identifier
            image_id: Identifier of the associated image
            claims: List of atomic claims from the generated answer
            ground_truth_objects: Known ground-truth present objects (if available from COCO/POPE)
            detection_detections: Detections from open-vocab detector (OWL-ViT/SigLIP)
        """
        if not claims:
            return PerceptionDiagnosticResult(
                sample_id=sample_id,
                image_id=image_id,
                sp_score=1.0,
                visual_support_ratio=1.0,
                visual_contradiction_ratio=0.0,
                visual_coverage=1.0,
                grounding_uncertainty=0.0,
                total_claims_evaluated=0,
                supported_claims=0,
                contradicted_claims=0,
                verified_claims=[],
            )

        supported = 0
        contradicted = 0
        confidences: List[float] = []
        verified: List[AtomicClaim] = []

        gt_objects_set = {obj.lower().strip() for obj in (ground_truth_objects or [])}
        detected_labels = {}
        if detection_detections:
            for det in detection_detections:
                label = det.get("label", "").lower().strip()
                score = float(det.get("score", 0.0))
                if label:
                    detected_labels[label] = max(detected_labels.get(label, 0.0), score)

        for claim in claims:
            c_copy = claim.model_copy()
            subj = claim.subject.lower().strip()

            # For POPE/COCO, match against ground truth objects or detector
            is_supported = False
            is_contradicted = False
            confidence = 1.0

            if claim.claim_type == "object_presence":
                # Check ground truth objects
                if gt_objects_set:
                    if subj in gt_objects_set or any(subj in obj or obj in subj for obj in gt_objects_set):
                        is_supported = True
                        confidence = 1.0
                    else:
                        is_contradicted = True
                        confidence = 0.0
                elif detected_labels:
                    det_score = detected_labels.get(subj, 0.0)
                    if det_score >= self.threshold:
                        is_supported = True
                        confidence = det_score
                    else:
                        is_contradicted = True
                        confidence = 1.0 - det_score
                else:
                    # Neutral if no visual grounding labels available
                    is_supported = True
                    confidence = 0.5
            elif claim.claim_type in {"attribute", "action", "location", "relationship"}:
                # Grounding based on subject presence
                if gt_objects_set:
                    if subj in gt_objects_set or any(subj in obj or obj in subj for obj in gt_objects_set):
                        is_supported = True
                        confidence = 0.85
                    else:
                        is_contradicted = True
                        confidence = 0.15
                else:
                    is_supported = True
                    confidence = 0.75
            else:
                # External fact: not visually groundable, marked not applicable for visual perception
                c_copy.verification_state = "not_applicable"
                verified.append(c_copy)
                continue

            if is_supported:
                supported += 1
                c_copy.verification_state = "supported"
            elif is_contradicted:
                contradicted += 1
                c_copy.verification_state = "contradicted"
            else:
                c_copy.verification_state = "insufficient_evidence"

            c_copy.confidence = confidence
            confidences.append(confidence)
            verified.append(c_copy)

        evaluated_count = supported + contradicted
        if evaluated_count == 0:
            sp_score = 1.0
            support_ratio = 1.0
            contradiction_ratio = 0.0
            coverage = 1.0
            uncertainty = 0.0
        else:
            support_ratio = supported / evaluated_count
            contradiction_ratio = contradicted / evaluated_count
            coverage = len(verified) / max(1, len(claims))
            mean_conf = sum(confidences) / len(confidences) if confidences else 1.0
            variance = sum((c - mean_conf) ** 2 for c in confidences) / len(confidences) if confidences else 0.0
            uncertainty = variance ** 0.5

            # S_P formula: reward visual support, penalize contradiction and uncertainty
            sp_score = max(0.0, min(1.0, support_ratio * 0.85 + (1.0 - contradiction_ratio) * 0.15 - 0.1 * uncertainty))

        return PerceptionDiagnosticResult(
            sample_id=sample_id,
            image_id=image_id,
            sp_score=round(float(sp_score), 4),
            visual_support_ratio=round(float(support_ratio), 4),
            visual_contradiction_ratio=round(float(contradiction_ratio), 4),
            visual_coverage=round(float(coverage), 4),
            grounding_uncertainty=round(float(uncertainty), 4),
            total_claims_evaluated=len(claims),
            supported_claims=supported,
            contradicted_claims=contradicted,
            verified_claims=verified,
        )
