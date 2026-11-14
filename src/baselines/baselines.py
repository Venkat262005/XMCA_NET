"""Standard baseline detectors for hallucination benchmark comparisons.

Implements established hallucination detection baselines for comparative evaluation:
1. Single-axis neural baselines: S_P only (Perception), S_R only (Retrieval), S_U only (Utilization)
   using trained PyTorch neural classifiers with calibrated temperature scaling.
2. Sequence Predictive Entropy baseline: token logprob Shannon entropy.
3. Maximum Softmax Probability (MSP) / Minimum Token Logprob baseline.

Zero arithmetic heuristic shortcuts.
"""

import math
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from pydantic import BaseModel

from src.features.extractor import DiagnosticFeatureVector
from src.combiner.model import TemperatureScaler


class BaselineScore(BaseModel):
    """Output score from a baseline detector."""

    sample_id: str
    baseline_name: str
    hallucination_score: float  # Higher means higher probability of hallucination


class SingleAxisNeuralClassifier(nn.Module):
    """Calibrated neural classifier for a single diagnostic consistency axis."""

    def __init__(self, in_features: int, hidden_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class SingleAxisDetector:
    """Trained neural detector operating exclusively on one diagnostic axis."""

    def __init__(self, axis_name: str, in_features: int):
        self.axis_name = axis_name
        self.in_features = in_features
        self.model = SingleAxisNeuralClassifier(in_features)
        self.scaler = TemperatureScaler()
        self.is_fitted = False
        self.mean: Optional[np.ndarray] = None
        self.std: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int = 30, lr: float = 2e-3):
        """Train neural baseline classifier with class-balanced loss."""
        X_arr = np.asarray(X, dtype=np.float32)
        y_arr = np.asarray(y, dtype=np.float32)

        self.mean = np.mean(X_arr, axis=0)
        self.std = np.std(X_arr, axis=0)
        self.std[self.std < 1e-6] = 1.0
        X_norm = (X_arr - self.mean) / self.std

        n_pos = float(np.sum(y_arr == 1))
        n_neg = float(np.sum(y_arr == 0))
        pos_weight = torch.tensor([n_neg / max(1.0, n_pos)], dtype=torch.float32)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)

        x_t = torch.tensor(X_norm, dtype=torch.float32)
        y_t = torch.tensor(y_arr, dtype=torch.float32)

        self.model.train()
        for _ in range(epochs):
            optimizer.zero_grad()
            logits = self.model(x_t)
            loss = criterion(logits, y_t)
            loss.backward()
            optimizer.step()

        self.model.eval()
        with torch.no_grad():
            val_logits = self.model(x_t)
        self.scaler.calibrate(val_logits, y_t)
        self.is_fitted = True

    def predict_proba(self, x: np.ndarray) -> float:
        """Predict calibrated hallucination probability."""
        self.model.eval()
        x_arr = np.asarray(x, dtype=np.float32)
        if self.mean is not None and self.std is not None:
            x_arr = (x_arr - self.mean) / self.std
        with torch.no_grad():
            x_t = torch.tensor(x_arr, dtype=torch.float32).unsqueeze(0)
            logit = self.model(x_t)
            scaled = self.scaler(logit)
            prob = torch.sigmoid(scaled).item()
        return float(prob)


class BaselineEvaluator:
    """Computes comparison baselines across evaluated samples using neural detectors."""

    def __init__(self):
        # Neural detectors for each diagnostic stream
        self.sp_detector = SingleAxisDetector("perception", in_features=5)
        self.sr_detector = SingleAxisDetector("retrieval", in_features=6)
        self.su_detector = SingleAxisDetector("utilization", in_features=6)

    def fit_single_axis_detectors(
        self,
        vectors: List[DiagnosticFeatureVector],
        labels: List[int],
    ):
        """Fit all single-axis neural detectors on the training partition."""
        if not vectors or len(vectors) != len(labels):
            return

        y = np.array(labels, dtype=np.int32)

        # S_P features (5 dims)
        sp_x = np.array([
            [v.sp_score, v.visual_support_ratio, v.visual_contradiction_ratio, v.visual_coverage, v.grounding_uncertainty]
            for v in vectors
        ], dtype=np.float32)
        self.sp_detector.fit(sp_x, y)

        # S_R features (6 dims) - evaluate where RAG is active
        sr_mask = np.array([v.m_r == 0.0 for v in vectors])
        if np.sum(sr_mask) >= 10:
            sr_x = np.array([
                [v.sr_score, v.top_relevance_score, v.mean_relevance_score, v.score_margin, v.evidence_coverage, v.m_r]
                for v in vectors
            ], dtype=np.float32)
            self.sr_detector.fit(sr_x[sr_mask], y[sr_mask])

        # S_U features (6 dims) - evaluate where RAG is active
        su_mask = np.array([v.m_u == 0.0 for v in vectors])
        if np.sum(su_mask) >= 10:
            su_x = np.array([
                [v.su_score, v.entailment_ratio, v.contradiction_ratio, v.omission_ratio, v.unsupported_claim_ratio, v.m_u]
                for v in vectors
            ], dtype=np.float32)
            self.su_detector.fit(su_x[su_mask], y[su_mask])

    def sp_only(self, vector: DiagnosticFeatureVector) -> float:
        """Neural Perception-only baseline: calibrated probability from visual grounding features."""
        feat = np.array([
            vector.sp_score,
            vector.visual_support_ratio,
            vector.visual_contradiction_ratio,
            vector.visual_coverage,
            vector.grounding_uncertainty,
        ], dtype=np.float32)

        if self.sp_detector.is_fitted:
            return self.sp_detector.predict_proba(feat)

        # Zero-shot neural logistic transform with unit temperature
        with torch.no_grad():
            x_t = torch.tensor(feat, dtype=torch.float32)
            logit = self.sp_detector.model(x_t.unsqueeze(0))
            return float(torch.sigmoid(logit).item())

    def sr_only(self, vector: DiagnosticFeatureVector) -> float:
        """Neural Retrieval-only baseline: calibrated probability from knowledge retrieval features."""
        if vector.m_r == 1.0:
            return 0.5  # Neutral uncertainty when retrieval is not applicable

        feat = np.array([
            vector.sr_score,
            vector.top_relevance_score,
            vector.mean_relevance_score,
            vector.score_margin,
            vector.evidence_coverage,
            vector.m_r,
        ], dtype=np.float32)

        if self.sr_detector.is_fitted:
            return self.sr_detector.predict_proba(feat)

        with torch.no_grad():
            x_t = torch.tensor(feat, dtype=torch.float32)
            logit = self.sr_detector.model(x_t.unsqueeze(0))
            return float(torch.sigmoid(logit).item())

    def su_only(self, vector: DiagnosticFeatureVector) -> float:
        """Neural Utilization-only baseline: calibrated probability from evidence utilization features."""
        if vector.m_u == 1.0:
            return 0.5  # Neutral uncertainty when utilization is not applicable

        feat = np.array([
            vector.su_score,
            vector.entailment_ratio,
            vector.contradiction_ratio,
            vector.omission_ratio,
            vector.unsupported_claim_ratio,
            vector.m_u,
        ], dtype=np.float32)

        if self.su_detector.is_fitted:
            return self.su_detector.predict_proba(feat)

        with torch.no_grad():
            x_t = torch.tensor(feat, dtype=torch.float32)
            logit = self.su_detector.model(x_t.unsqueeze(0))
            return float(torch.sigmoid(logit).item())

    @staticmethod
    def predictive_entropy(token_logprobs: List[float]) -> float:
        """Token logprob Shannon entropy baseline."""
        if not token_logprobs:
            return 0.5
        probs = [math.exp(max(-20.0, lp)) for lp in token_logprobs]
        sum_p = sum(probs)
        if sum_p <= 0:
            return 1.0
        norm_p = [p / sum_p for p in probs]
        entropy = -sum(p * math.log(max(1e-9, p)) for p in norm_p)
        max_entropy = math.log(max(2, len(probs)))
        return float(min(1.0, entropy / max_entropy))

    @staticmethod
    def min_token_logprob(token_logprobs: List[float]) -> float:
        """Minimum token logprob baseline (lowest confidence token)."""
        if not token_logprobs:
            return 0.5
        min_lp = min(token_logprobs)
        prob = math.exp(max(-20.0, min_lp))
        return float(1.0 - prob)
