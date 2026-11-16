"""REFIND: Retrieval-Augmented Factuality Hallucination Detection (SemEval@ACL 2025).

Official methodology from oneonlee/REFIND:
Measures hallucination risk via the Context Sensitivity Ratio (CSR).
Compares token generation log-probabilities with and without retrieved reference documents D:

                 log p(t_i | D, q, t_<i)
    CSR(t_i) = ---------------------------
               log p(t_i | q, t_<i) + epsilon

Properties:
- When a claim is faithfully grounded in the retrieved document D, the context-conditioned
  probability is significantly higher than the unconditioned probability, producing high CSR.
- When an LLM/VLM hallucinates ungrounded facts or ignores D, the unconditioned language prior
  dominates or contradicts D, resulting in low or inverted CSR.
- Documented VLM adaptation: Conditioned on (Image + Context D + Query) vs (Image + Query).
"""

from typing import List, Dict, Any, Optional, Tuple
import math
import numpy as np
import torch
from pydantic import BaseModel

from src.baselines.common_schema import (
    BaselineDetectionOutput,
    BaselineLocalizationOutput,
    ContextSensitivityOutput,
)
from src.combiner.model import TemperatureScaler


class REFINDDetector:
    """Context Sensitivity Ratio (CSR) hallucination detector (REFIND)."""

    def __init__(self, epsilon: float = 1e-5):
        self.epsilon = epsilon
        self.scaler = TemperatureScaler()
        self.is_calibrated = False

    def compute_token_csr(
        self,
        logp_with_context: float,
        logp_without_context: float,
    ) -> float:
        """Compute the authentic SemEval-2025 Context Sensitivity Ratio for a single token."""
        # Safeguard division by near-zero denominator
        denom = logp_without_context if abs(logp_without_context) > self.epsilon else -self.epsilon
        csr = logp_with_context / denom
        return float(csr)

    def score_sequence(
        self,
        tokens: List[str],
        logprobs_with_context: List[float],
        logprobs_without_context: List[float],
    ) -> Tuple[float, ContextSensitivityOutput]:
        """Compute sequence CSR metrics and overall hallucination risk.
        
        Args:
            tokens: Generated response tokens.
            logprobs_with_context: log p(t_i | D, I, q, t_<i)
            logprobs_without_context: log p(t_i | I, q, t_<i)
            
        Returns:
            hallucination_score: Continuous score (higher = higher hallucination risk).
            context_output: Full CSR breakdown.
        """
        assert len(logprobs_with_context) == len(logprobs_without_context), "Length mismatch"
        n = len(logprobs_with_context)
        if n == 0:
            empty_out = ContextSensitivityOutput(
                sample_id="",
                dataset_id="",
                model_name="",
                token_csr_scores=[],
                mean_csr=1.0,
                min_csr=1.0,
                retrieval_reliance_score=1.0,
            )
            return 0.0, empty_out

        csr_scores = []
        for lp_ctx, lp_no in zip(logprobs_with_context, logprobs_without_context):
            csr_scores.append(self.compute_token_csr(lp_ctx, lp_no))

        mean_csr = float(np.mean(csr_scores))
        min_csr = float(np.min(csr_scores))

        # Overall retrieval reliance score
        retrieval_reliance = float(np.median(csr_scores))

        # Hallucination risk is inversely related to context sensitivity:
        # A token stream insensitive to retrieved facts indicates hallucination or parametric override.
        # Raw risk: negative of mean CSR (or deviation from expected grounded sensitivity)
        raw_hallucination_risk = -mean_csr

        output = ContextSensitivityOutput(
            sample_id="",
            dataset_id="",
            model_name="",
            token_csr_scores=csr_scores,
            mean_csr=mean_csr,
            min_csr=min_csr,
            retrieval_reliance_score=retrieval_reliance,
        )

        return raw_hallucination_risk, output

    def calibrate(self, val_scores: np.ndarray, val_labels: np.ndarray):
        """Fit post-hoc temperature scaling on validation split."""
        scores_t = torch.tensor(val_scores, dtype=torch.float32)
        labels_t = torch.tensor(val_labels, dtype=torch.float32)
        self.scaler.calibrate(scores_t, labels_t)
        self.is_calibrated = True

    def predict_calibrated_proba(self, raw_score: float) -> float:
        """Convert continuous hallucination risk into calibrated probability."""
        if not self.is_calibrated:
            return float(1.0 / (1.0 + math.exp(-raw_score)))
        with torch.no_grad():
            s_t = torch.tensor([raw_score], dtype=torch.float32)
            prob = torch.sigmoid(self.scaler(s_t)).item()
        return float(prob)
