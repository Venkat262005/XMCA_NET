"""VisER: Visual Evidence and Reliance for Object Hallucination Detection in LVLMs (EMNLP 2026).

Official methodology from AfsanehEB/VisER:
A training-free, two-sided metric framework for object-level hallucination detection.
Evaluates generated object mentions through two complementary mechanisms:
1. Visual Evidence (VE):
   Derives from image-token logit-lens projection, applying an evidence gate that measures
   object-context similarity against intermediate image patch representations.
   VE(o) = Gate(sim(e_o, E_img)) * LogitLens(o | E_img)
2. Visual Reliance (VR):
   Compares image-derived representation support with support derived exclusively from
   the generated-prefix embeddings:
   VR(o) = Support(o | E_img) - Support(o | E_prefix)

A candidate object mention is verified as visually grounded if it exhibits both sufficient
Visual Evidence (VE) through the evidence gate and positive Visual Reliance (VR), proving
the mention does not stem merely from prefix language prior bias.

Note: In black-box inference where internal logit-lens hidden states are unavailable,
we document the conditional likelihood approximation as an adapted proxy.
"""

from typing import List, Dict, Any, Optional, Tuple
import math
import numpy as np
from pydantic import BaseModel

from src.baselines.common_schema import BaselineDetectionOutput
from src.combiner.model import TemperatureScaler
import torch


class VisERObjectScore(BaseModel):
    """Object-mention level VisER diagnostic score."""

    object_token: str
    token_index: int
    visual_evidence_logprob: float      # log p(o | I, q, t_<o)
    language_prior_logprob: float       # log p(o | q, t_<o)
    visual_reliance: float              # VE - prior
    grounding_score: float              # Composite grounding support
    is_hallucinated: Optional[bool] = None


class VisERDetector:
    """VisER two-sided visual evidence and reliance detector."""

    def __init__(self, alpha: float = 0.5):
        """Initialize VisER detector.
        
        Args:
            alpha: Weight balancing Visual Evidence and Visual Reliance.
        """
        self.alpha = alpha
        self.scaler = TemperatureScaler()
        self.is_calibrated = False

    def compute_object_score(
        self,
        token: str,
        token_index: int,
        ve_logprob: float,
        prior_logprob: float,
    ) -> VisERObjectScore:
        """Compute authentic two-sided VisER score for an object mention."""
        vr = float(ve_logprob - prior_logprob)
        # Composite grounding score: high indicates authentic visual support
        # Alpha balances raw visual likelihood (VE) and visual gain over language prior (VR)
        grounding = float(self.alpha * ve_logprob + (1.0 - self.alpha) * vr)
        return VisERObjectScore(
            object_token=token,
            token_index=token_index,
            visual_evidence_logprob=ve_logprob,
            language_prior_logprob=prior_logprob,
            visual_reliance=vr,
            grounding_score=grounding,
        )

    def score_sequence(
        self,
        object_tokens: List[str],
        token_indices: List[int],
        ve_logprobs: List[float],
        prior_logprobs: List[float],
    ) -> Tuple[float, List[VisERObjectScore]]:
        """Compute sequence-level hallucination score from object mentions.
        
        Returns:
            hallucination_score: Continuous score (higher = higher probability of hallucination).
            object_scores: Per-object breakdown.
        """
        if not object_tokens or len(object_tokens) == 0:
            # Fallback for sequences without extracted object mentions: neutral score
            return 0.0, []

        obj_scores = []
        for tok, idx, ve, pr in zip(object_tokens, token_indices, ve_logprobs, prior_logprobs):
            score = self.compute_object_score(tok, idx, ve, pr)
            obj_scores.append(score)

        # Sequence hallucination is governed by the least visually grounded object mention:
        # Minimum grounding score corresponds to maximum hallucination risk:
        min_grounding = min(s.grounding_score for s in obj_scores)
        hallucination_risk = -float(min_grounding)

        return hallucination_risk, obj_scores

    def calibrate(self, val_scores: np.ndarray, val_labels: np.ndarray):
        """Fit post-hoc temperature scaling on validation split."""
        scores_t = torch.tensor(val_scores, dtype=torch.float32)
        labels_t = torch.tensor(val_labels, dtype=torch.float32)
        self.scaler.calibrate(scores_t, labels_t)
        self.is_calibrated = True

    def predict_calibrated_proba(self, raw_score: float) -> float:
        """Convert continuous hallucination risk into calibrated probability."""
        if not self.is_calibrated:
            # Standard sigmoid if not yet calibrated
            return float(1.0 / (1.0 + math.exp(-raw_score)))
        with torch.no_grad():
            s_t = torch.tensor([raw_score], dtype=torch.float32)
            prob = torch.sigmoid(self.scaler(s_t)).item()
        return float(prob)
