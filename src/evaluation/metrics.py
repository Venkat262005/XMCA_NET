"""Evaluation and metrics computation module for XMCA-Net.

Computes standardized detection and attribution performance metrics:
- AUROC (Area under ROC curve via trapezoidal rule / Mann-Whitney U equivalent)
- PR-AUC (Average Precision / trapezoidal precision-recall integration)
- ECE (Expected Calibration Error with adaptive binning)
- Accuracy, Precision, Recall, F1 score
- Brier Score (mean squared calibration error)
- Cause Attribution Accuracy (multiclass accuracy on verified ground-truth attribution cohorts)
"""

import math
import numpy as np
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field


class DetectionEvaluationMetrics(BaseModel):
    """Evaluation summary metrics for hallucination detection."""

    auroc: float
    pr_auc: float
    ece: float
    brier_score: float
    accuracy: float
    precision: float
    recall: float
    specificity: float
    f1_score: float
    sample_count: int
    positive_count: int
    negative_count: int


class MetricEvaluator:
    """Computes standardized detection and attribution metrics."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}

    def compute_detection_metrics(
        self,
        y_true: List[int],
        y_score: List[float],
        threshold: float = 0.5,
        n_bins: int = 10,
    ) -> DetectionEvaluationMetrics:
        """Compute full suite of detection metrics."""
        if not y_true or not y_score or len(y_true) != len(y_score):
            raise ValueError("y_true and y_score must be non-empty and of equal length.")

        y_t = np.array(y_true, dtype=np.int32)
        y_s = np.array(y_score, dtype=np.float64)

        n_samples = len(y_t)
        n_pos = int(np.sum(y_t == 1))
        n_neg = int(np.sum(y_t == 0))

        # 1. AUROC calculation
        auroc = self._compute_auroc(y_t, y_s)

        # 2. PR-AUC calculation
        pr_auc = self._compute_pr_auc(y_t, y_s)

        # 3. Expected Calibration Error (ECE) & Brier score
        ece = self._compute_ece(y_t, y_s, n_bins=n_bins)
        brier = float(np.mean((y_s - y_t) ** 2))

        # 4. Binary classification metrics at decision threshold
        y_pred = (y_s >= threshold).astype(int)
        tp = int(np.sum((y_pred == 1) & (y_t == 1)))
        fp = int(np.sum((y_pred == 1) & (y_t == 0)))
        fn = int(np.sum((y_pred == 0) & (y_t == 1)))
        tn = int(np.sum((y_pred == 0) & (y_t == 0)))

        acc = (tp + tn) / max(1, n_samples)
        prec = tp / max(1, tp + fp)
        rec = tp / max(1, tp + fn)
        spec = tn / max(1, tn + fp)
        f1 = (2 * prec * rec) / max(1e-7, prec + rec)

        return DetectionEvaluationMetrics(
            auroc=round(float(auroc), 4),
            pr_auc=round(float(pr_auc), 4),
            ece=round(float(ece), 4),
            brier_score=round(float(brier), 4),
            accuracy=round(float(acc), 4),
            precision=round(float(prec), 4),
            recall=round(float(rec), 4),
            specificity=round(float(spec), 4),
            f1_score=round(float(f1), 4),
            sample_count=n_samples,
            positive_count=n_pos,
            negative_count=n_neg,
        )

    def compute_multi_label_attribution_metrics(
        self,
        y_true_binary: np.ndarray,
        y_pred_binary: np.ndarray,
        class_names: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        """Compute multi-label / multi-class cause attribution metrics.
        
        Args:
            y_true_binary: Binary ground truth matrix of shape [N, num_causes] (0 or 1).
            y_pred_binary: Binary prediction matrix of shape [N, num_causes] (0 or 1).
            class_names: Names of causes (e.g., ['visual', 'retrieval', 'utilization']).
            
        Returns:
            Dictionary with per-class F1, Precision, Recall, and Macro-F1.
        """
        if class_names is None:
            class_names = ["visual", "retrieval", "utilization"]
            
        y_t = np.asarray(y_true_binary, dtype=int)
        y_p = np.asarray(y_pred_binary, dtype=int)
        
        if y_t.ndim == 1:
            # Convert 1D class indices to one-hot binary matrix
            num_classes = len(class_names)
            y_t = np.eye(num_classes)[y_t]
            y_p = np.eye(num_classes)[y_p]
            
        n_classes = min(y_t.shape[1], len(class_names))
        f1_list: List[float] = []
        metrics: Dict[str, float] = {}
        
        for idx in range(n_classes):
            c_name = class_names[idx]
            yt_c = y_t[:, idx]
            yp_c = y_p[:, idx]
            
            tp = int(np.sum((yp_c == 1) & (yt_c == 1)))
            fp = int(np.sum((yp_c == 1) & (yt_c == 0)))
            fn = int(np.sum((yp_c == 0) & (yt_c == 1)))
            tn = int(np.sum((yp_c == 0) & (yt_c == 0)))
            
            p = tp / max(1, tp + fp)
            r = tp / max(1, tp + fn)
            f1 = (2 * p * r) / max(1e-7, p + r)
            
            metrics[f"{c_name}_f1"] = round(float(f1), 4)
            metrics[f"{c_name}_precision"] = round(float(p), 4)
            metrics[f"{c_name}_recall"] = round(float(r), 4)
            f1_list.append(f1)
            
        metrics["macro_f1_attribution"] = round(float(np.mean(f1_list)), 4) if f1_list else 0.0
        exact_match = float(np.mean(np.all(y_t == y_p, axis=1))) if len(y_t) > 0 else 0.0
        metrics["exact_match_ratio"] = round(exact_match, 4)
        return metrics

    def compute_attribution_accuracy(
        self,
        y_true_causes: List[str],
        y_pred_causes: List[str],
    ) -> Dict[str, float]:
        """Compute multiclass attribution accuracy and per-class metrics."""
        if not y_true_causes or len(y_true_causes) != len(y_pred_causes):
            raise ValueError("y_true_causes and y_pred_causes must be non-empty and of equal length.")

        total = len(y_true_causes)
        correct = sum(1 for t, p in zip(y_true_causes, y_pred_causes) if t.lower() == p.lower())
        accuracy = correct / max(1, total)

        # Per-class breakdown
        classes = sorted(list(set(y_true_causes)))
        per_class: Dict[str, float] = {}
        for c in classes:
            c_true = [1 if t.lower() == c.lower() else 0 for t in y_true_causes]
            c_pred = [1 if p.lower() == c.lower() else 0 for p in y_pred_causes]
            c_corr = sum(1 for yt, yp in zip(c_true, c_pred) if yt == 1 and yp == 1)
            c_total = sum(c_true)
            per_class[f"recall_{c}"] = round(c_corr / max(1, c_total), 4)

        return {
            "overall_attribution_accuracy": round(accuracy, 4),
            "sample_count": total,
            **per_class,
        }

    @staticmethod
    def _compute_auroc(y_true: np.ndarray, y_score: np.ndarray) -> float:
        """Compute Area Under ROC Curve via Mann-Whitney U statistic rank sum."""
        n_pos = np.sum(y_true == 1)
        n_neg = np.sum(y_true == 0)
        if n_pos == 0 or n_neg == 0:
            return 0.5

        # Rank all scores with average rank tie-breaking
        order = np.argsort(y_score)
        ranks = np.empty_like(order, dtype=float)
        ranks[order] = np.arange(len(y_score)) + 1.0

        # Adjust for ties
        unique_scores, counts = np.unique(y_score, return_counts=True)
        for val, count in zip(unique_scores, counts):
            if count > 1:
                idx = np.where(y_score == val)[0]
                ranks[idx] = np.mean(ranks[idx])

        pos_rank_sum = np.sum(ranks[y_true == 1])
        u_stat = pos_rank_sum - (n_pos * (n_pos + 1.0)) / 2.0
        auroc = u_stat / (n_pos * n_neg)
        return float(auroc)

    @staticmethod
    def _compute_pr_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
        """Compute Average Precision / PR-AUC."""
        n_pos = np.sum(y_true == 1)
        if n_pos == 0:
            return 0.0

        # Sort descending by score
        desc_indices = np.argsort(-y_score)
        y_true_sorted = y_true[desc_indices]

        tp_cumsum = np.cumsum(y_true_sorted == 1)
        fp_cumsum = np.cumsum(y_true_sorted == 0)

        precisions = tp_cumsum / (tp_cumsum + fp_cumsum)
        recalls = tp_cumsum / n_pos

        # Trapezoidal integration across recall steps
        recalls_with_zero = np.insert(recalls, 0, 0.0)
        precisions_with_first = np.insert(precisions, 0, precisions[0] if len(precisions) > 0 else 1.0)
        pr_auc = float(np.sum((recalls_with_zero[1:] - recalls_with_zero[:-1]) * precisions_with_first[1:]))
        return pr_auc

    @staticmethod
    def _compute_ece(y_true: np.ndarray, y_score: np.ndarray, n_bins: int = 10) -> float:
        """Compute Expected Calibration Error (ECE)."""
        bin_limits = np.linspace(0.0, 1.0, n_bins + 1)
        ece = 0.0
        n_total = len(y_true)

        for i in range(n_bins):
            low, high = bin_limits[i], bin_limits[i + 1]
            if i == n_bins - 1:
                in_bin = (y_score >= low) & (y_score <= high)
            else:
                in_bin = (y_score >= low) & (y_score < high)

            count = np.sum(in_bin)
            if count > 0:
                bin_acc = np.mean(y_true[in_bin])
                bin_conf = np.mean(y_score[in_bin])
                ece += (count / n_total) * abs(bin_acc - bin_conf)

        return float(ece)
