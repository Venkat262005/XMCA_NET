"""Standard baseline detectors for hallucination benchmark comparisons.

Implements established hallucination detection baselines for comparative evaluation:
1. Maximum Softmax Probability (MSP) / Token Logprob baseline
2. Sequence Predictive Entropy baseline
3. Single-axis baseline: S_P only (Perception-only)
4. Single-axis baseline: S_R only (Retrieval-only)
5. Single-axis baseline: S_U only (Utilization-only)
"""

import math
import numpy as np
from typing import List, Dict, Any, Optional
from pydantic import BaseModel

from src.features.extractor import DiagnosticFeatureVector


class BaselineScore(BaseModel):
    """Output score from a baseline detector."""

    sample_id: str
    baseline_name: str
    hallucination_score: float  # Higher means higher probability of hallucination


class BaselineEvaluator:
    """Computes comparison baselines across evaluated samples."""

    @staticmethod
    def sp_only(vector: DiagnosticFeatureVector) -> float:
        """Single-axis Perception baseline: hallucination is 1.0 - S_P."""
        return float(1.0 - vector.sp_score)

    @staticmethod
    def sr_only(vector: DiagnosticFeatureVector) -> float:
        """Single-axis Retrieval baseline: hallucination is 1.0 - S_R (or neutral 0.5 if missing)."""
        if vector.m_r == 1.0:
            return 0.5
        return float(1.0 - vector.sr_score)

    @staticmethod
    def su_only(vector: DiagnosticFeatureVector) -> float:
        """Single-axis Utilization baseline: hallucination is 1.0 - S_U (or neutral 0.5 if missing)."""
        if vector.m_u == 1.0:
            return 0.5
        return float(1.0 - vector.su_score)

    @staticmethod
    def predictive_entropy(token_logprobs: List[float]) -> float:
        """Token logprob entropy baseline."""
        if not token_logprobs:
            return 0.5
        probs = [math.exp(max(-20.0, lp)) for lp in token_logprobs]
        sum_p = sum(probs)
        if sum_p <= 0:
            return 1.0
        norm_p = [p / sum_p for p in probs]
        entropy = -sum(p * math.log(max(1e-9, p)) for p in norm_p)
        # Normalize entropy by max entropy log(N)
        max_entropy = math.log(max(2, len(probs)))
        return float(min(1.0, entropy / max_entropy))

    @staticmethod
    def min_token_logprob(token_logprobs: List[float]) -> float:
        """Minimum token logprob baseline (lowest confidence token)."""
        if not token_logprobs:
            return 0.5
        # Transform lowest logprob to hallucination score in [0, 1]
        min_lp = min(token_logprobs)
        prob = math.exp(max(-20.0, min_lp))
        return float(1.0 - prob)
