"""Calibrated XMCA detection combiner and cause-attribution module.

Implements authentic PyTorch neural network training, temperature scaling calibration,
and neural multi-class cause-attribution diagnosis:
- Model: XMCANetwork (Residual MLP with missingness-aware gating).
- Calibration: TemperatureScaler optimizing NLL on validation partition via L-BFGS.
- Attribution: XMCAAttributionNetwork supervised on verified ground truth.
- Anti-leakage isolation: training and calibration strictly confined to validation partition.
"""

from typing import Dict, Any, Optional, List, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from pydantic import BaseModel, Field

from src.features.extractor import DiagnosticFeatureVector
from src.combiner.model import XMCANetwork, TemperatureScaler, XMCAAttributionNetwork


class CombinerPrediction(BaseModel):
    """Output prediction of the calibrated neural XMCA combiner."""

    sample_id: str
    hallucination_probability: float  # Calibrated probability in [0.0, 1.0]
    predicted_label: int  # 1 = hallucination, 0 = faithful
    threshold_applied: float
    raw_score: float
    temperature: float = 1.0
    sp_score: float
    sr_score: float
    su_score: float
    m_r: float
    m_u: float


class AttributionPrediction(BaseModel):
    """Output prediction of the neural XMCA cause-attribution head."""

    sample_id: str
    attributed_cause: str  # 'perception' | 'retrieval' | 'utilization' | 'unattributed'
    attribution_probabilities: Dict[str, float] = Field(default_factory=dict)
    confidence: float = 0.0
    evidence_summary: str = ""


class XMCADetectionCombiner:
    """Neural detection combiner integrating S_P, S_R, and S_U into calibrated hallucination probabilities."""

    def __init__(
        self,
        input_dim: int = 20,
        hidden_dim: int = 64,
        device: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        self.config = config or {}
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = XMCANetwork(input_dim=input_dim, hidden_dim=hidden_dim).to(self.device)
        self.scaler = TemperatureScaler().to(self.device)
        self.decision_threshold = 0.5
        self.is_fitted = False

        # Feature normalization parameters
        self.mean: Optional[np.ndarray] = None
        self.std: Optional[np.ndarray] = None

    @property
    def temperature(self) -> float:
        return self.scaler.temperature

    def _normalize_features(self, X: np.ndarray, fit: bool = False) -> np.ndarray:
        if fit:
            self.mean = np.mean(X, axis=0)
            self.std = np.std(X, axis=0)
            self.std[self.std < 1e-6] = 1.0
        if self.mean is not None and self.std is not None:
            return (X - self.mean) / self.std
        return X

    def fit(
        self,
        training_vectors: List[DiagnosticFeatureVector],
        labels: List[int],
        epochs: int = 50,
        batch_size: int = 32,
        lr: float = 1e-3,
        weight_decay: float = 1e-2,
    ):
        """Train XMCA neural network strictly on validation partition image groups."""
        if not training_vectors or len(training_vectors) != len(labels):
            raise ValueError("Training vectors and labels must be non-empty and of equal length.")

        X = np.stack([v.to_feature_list() for v in training_vectors], axis=0)
        X_norm = self._normalize_features(X, fit=True)
        y = np.array(labels, dtype=np.float32)

        # Compute positive class weight for loss balancing
        n_pos = np.sum(y == 1)
        n_neg = np.sum(y == 0)
        pos_weight = torch.tensor([float(n_neg) / max(1.0, float(n_pos))], device=self.device)

        dataset = TensorDataset(
            torch.tensor(X_norm, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32),
        )
        loader = DataLoader(dataset, batch_size=min(batch_size, len(dataset)), shuffle=True)

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        self.model.train()
        for _ in range(epochs):
            for batch_x, batch_y in loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                optimizer.zero_grad()
                logits = self.model(batch_x)
                loss = criterion(logits, batch_y)
                loss.backward()
                optimizer.step()
            scheduler.step()

        self.model.eval()
        self.is_fitted = True

    def calibrate_temperature(
        self,
        val_vectors: List[DiagnosticFeatureVector],
        val_labels: List[int],
    ):
        """Calibrate temperature parameter on validation partition to minimize NLL / ECE."""
        if not val_vectors:
            return

        self.model.eval()
        X = np.stack([v.to_feature_list() for v in val_vectors], axis=0)
        X_norm = self._normalize_features(X, fit=False)
        y = np.array(val_labels, dtype=np.float32)

        with torch.no_grad():
            x_t = torch.tensor(X_norm, dtype=torch.float32, device=self.device)
            val_logits = self.model(x_t)

        y_t = torch.tensor(y, dtype=torch.float32, device=self.device)
        self.scaler.calibrate(val_logits, y_t)

    def set_decision_threshold(self, threshold: float):
        self.decision_threshold = float(max(0.01, min(0.99, threshold)))

    def predict(self, feature_vector: DiagnosticFeatureVector) -> CombinerPrediction:
        """Predict calibrated hallucination probability for a single sample."""
        self.model.eval()
        x = np.array([feature_vector.to_feature_list()], dtype=np.float32)
        x_norm = self._normalize_features(x, fit=False)

        with torch.no_grad():
            x_t = torch.tensor(x_norm, dtype=torch.float32, device=self.device)
            raw_logit = self.model(x_t)
            scaled_logit = self.scaler(raw_logit)
            prob = torch.sigmoid(scaled_logit).item()

        pred_label = 1 if prob >= self.decision_threshold else 0

        return CombinerPrediction(
            sample_id=feature_vector.sample_id,
            hallucination_probability=round(float(prob), 4),
            predicted_label=pred_label,
            threshold_applied=self.decision_threshold,
            raw_score=round(float(raw_logit.item()), 4),
            temperature=round(float(self.scaler.temperature), 3),
            sp_score=feature_vector.sp_score,
            sr_score=feature_vector.sr_score,
            su_score=feature_vector.su_score,
            m_r=feature_vector.m_r,
            m_u=feature_vector.m_u,
        )


class XMCAAttributionHead:
    """Neural cause-attribution head diagnosing primary hallucination failure stage."""

    def __init__(
        self,
        hidden_dim: int = 64,
        device: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        self.config = config or {}
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.network = XMCAAttributionNetwork(hidden_dim=hidden_dim, num_causes=3).to(self.device)
        self.is_trained = False

    def fit(
        self,
        combiner: XMCADetectionCombiner,
        vectors: List[DiagnosticFeatureVector],
        cause_labels: List[int],  # 0: Perception, 1: Retrieval, 2: Utilization
        epochs: int = 40,
        lr: float = 1e-3,
    ):
        """Train attribution head on samples with verified ground-truth cause labels."""
        if not vectors or len(vectors) != len(cause_labels):
            return

        combiner.model.eval()
        X = np.stack([v.to_feature_list() for v in vectors], axis=0)
        X_norm = combiner._normalize_features(X, fit=False)

        with torch.no_grad():
            x_t = torch.tensor(X_norm, dtype=torch.float32, device=self.device)
            representations = combiner.model.forward_representation(x_t)

        y_t = torch.tensor(cause_labels, dtype=torch.long, device=self.device)
        dataset = TensorDataset(representations, y_t)
        loader = DataLoader(dataset, batch_size=min(16, len(dataset)), shuffle=True)

        optimizer = torch.optim.AdamW(self.network.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        self.network.train()
        for _ in range(epochs):
            for batch_rep, batch_y in loader:
                optimizer.zero_grad()
                logits = self.network(batch_rep)
                loss = criterion(logits, batch_y)
                loss.backward()
                optimizer.step()

        self.network.eval()
        self.is_trained = True

    def attribute_cause(
        self,
        feature_vector: DiagnosticFeatureVector,
        prediction: CombinerPrediction,
        combiner: Optional[XMCADetectionCombiner] = None,
    ) -> AttributionPrediction:
        """Attribute root failure cause strictly when hallucination is detected or suspected."""
        # Faithful response: unattributed
        if prediction.predicted_label == 0 and prediction.hallucination_probability < 0.35:
            return AttributionPrediction(
                sample_id=feature_vector.sample_id,
                attributed_cause="unattributed",
                attribution_probabilities={"perception": 0.0, "retrieval": 0.0, "utilization": 0.0},
                confidence=1.0 - prediction.hallucination_probability,
                evidence_summary="Model generated faithful response consistent across diagnostic axes.",
            )

        # Non-RAG datasets (POPE, FaithScore): RAG is masked out, failure is strictly perception
        if feature_vector.m_r == 1.0 and feature_vector.m_u == 1.0:
            return AttributionPrediction(
                sample_id=feature_vector.sample_id,
                attributed_cause="perception",
                attribution_probabilities={"perception": 1.0, "retrieval": 0.0, "utilization": 0.0},
                confidence=round(float(1.0 - feature_vector.sp_score), 4),
                evidence_summary=(
                    f"Visual perception failure (S_P={feature_vector.sp_score:.2f}, "
                    f"contradiction={feature_vector.visual_contradiction_ratio:.2f})."
                ),
            )

        # Neural attribution inference if combiner model available
        if combiner is not None and self.is_trained:
            combiner.model.eval()
            self.network.eval()
            x = np.array([feature_vector.to_feature_list()], dtype=np.float32)
            x_norm = combiner._normalize_features(x, fit=False)

            with torch.no_grad():
                x_t = torch.tensor(x_norm, dtype=torch.float32, device=self.device)
                rep = combiner.model.forward_representation(x_t)
                logits = self.network(rep)
                probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()

            prob_dict = {
                "perception": round(float(probs[0]), 4),
                "retrieval": round(float(probs[1]), 4),
                "utilization": round(float(probs[2]), 4),
            }
            causes = ["perception", "retrieval", "utilization"]
            best_idx = int(np.argmax(probs))
            attributed = causes[best_idx]
            confidence = float(probs[best_idx])
        else:
            # Deterministic multi-axis diagnostic comparison
            p_fail = (1.0 - feature_vector.sp_score) + feature_vector.visual_contradiction_ratio * 0.5
            r_fail = (1.0 - feature_vector.sr_score) if feature_vector.m_r == 0.0 else 0.0
            u_fail = (1.0 - feature_vector.su_score) + feature_vector.contradiction_ratio * 0.5 if feature_vector.m_u == 0.0 else 0.0

            scores = np.array([p_fail, r_fail, u_fail], dtype=np.float64)
            exp_s = np.exp(scores - np.max(scores))
            probs = exp_s / np.sum(exp_s)
            prob_dict = {
                "perception": round(float(probs[0]), 4),
                "retrieval": round(float(probs[1]), 4),
                "utilization": round(float(probs[2]), 4),
            }
            causes = ["perception", "retrieval", "utilization"]
            best_idx = int(np.argmax(scores))
            attributed = causes[best_idx]
            confidence = float(probs[best_idx])

        if attributed == "perception":
            summary = (
                f"Visual perception failure (S_P={feature_vector.sp_score:.2f}, "
                f"visual contradiction={feature_vector.visual_contradiction_ratio:.2f})."
            )
        elif attributed == "retrieval":
            summary = (
                f"Knowledge retrieval failure (S_R={feature_vector.sr_score:.2f}, "
                f"top relevance={feature_vector.top_relevance_score:.2f})."
            )
        else:
            summary = (
                f"Evidence utilization failure (S_U={feature_vector.su_score:.2f}, "
                f"unsupported ratio={feature_vector.unsupported_claim_ratio:.2f})."
            )

        return AttributionPrediction(
            sample_id=feature_vector.sample_id,
            attributed_cause=attributed,
            attribution_probabilities=prob_dict,
            confidence=round(confidence, 4),
            evidence_summary=summary,
        )
