"""Evaluation metrics calculator for hallucination detection baselines.

Implements rigorous evaluation protocol:
1. AUROC and PR-AUC computed across complete precision-recall tradeoff.
2. F1 threshold is STRICTLY selected on the validation partition to prevent test label leakage.
3. Calibration (ECE) is evaluated against true binary labels.
4. Specificity, Precision, Recall, Brier score reported.
"""

from typing import Dict, Any, Tuple, List, Optional
import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    precision_recall_curve,
    auc,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    brier_score_loss,
)


def compute_calibration_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    """Compute Expected Calibration Error (ECE)."""
    probs = np.clip(np.asarray(probs, dtype=np.float64), 1e-7, 1.0 - 1e-7)
    labels = np.asarray(labels, dtype=np.float64)
    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(probs)

    for i in range(n_bins):
        low, high = bin_boundaries[i], bin_boundaries[i + 1]
        mask = (probs >= low) & (probs < high) if i < n_bins - 1 else (probs >= low) & (probs <= high)
        bin_count = np.sum(mask)
        if bin_count > 0:
            bin_acc = np.mean(labels[mask])
            bin_conf = np.mean(probs[mask])
            ece += (bin_count / n) * np.abs(bin_acc - bin_conf)

    return float(ece)


def select_optimal_f1_threshold(val_scores: np.ndarray, val_labels: np.ndarray) -> float:
    """Select decision threshold maximizing F1 on the VALIDATION split only.
    
    Prevents test threshold peeking / test-label leakage.
    """
    precisions, recalls, thresholds = precision_recall_curve(val_labels, val_scores)
    f1s = 2 * (precisions * recalls) / (precisions + recalls + 1e-9)
    # Thresholds has len = len(precisions) - 1
    best_idx = int(np.argmax(f1s[:len(thresholds)])) if len(thresholds) > 0 else 0
    return float(thresholds[best_idx]) if len(thresholds) > 0 else 0.5


def evaluate_detection_metrics(
    y_test: np.ndarray,
    scores_test: np.ndarray,
    val_scores: Optional[np.ndarray] = None,
    val_labels: Optional[np.ndarray] = None,
    fixed_threshold: Optional[float] = None,
) -> Dict[str, float]:
    """Calculate full suite of detection metrics on test split."""
    y_true = np.asarray(y_test, dtype=int)
    scores = np.asarray(scores_test, dtype=float)

    # 1. Discrimination (Threshold-independent)
    try:
        auroc = float(roc_auc_score(y_true, scores))
    except Exception:
        auroc = 0.5

    try:
        prec, rec, _ = precision_recall_curve(y_true, scores)
        pr_auc = float(auc(rec, prec))
    except Exception:
        pr_auc = float(np.mean(y_true))

    # 2. Select threshold on validation set (if provided) to avoid test leakage
    if fixed_threshold is not None:
        threshold = fixed_threshold
    elif val_scores is not None and val_labels is not None:
        threshold = select_optimal_f1_threshold(val_scores, val_labels)
    else:
        # Fallback to median score if validation split not explicitly supplied
        threshold = float(np.median(scores))

    # 3. Binary decision metrics
    y_pred = (scores >= threshold).astype(int)
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    precision = float(precision_score(y_true, y_pred, zero_division=0))
    recall = float(recall_score(y_true, y_pred, zero_division=0))

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

    # 4. Calibration & probabilistic loss
    # Normalize scores to [0, 1] range for ECE and Brier if raw logits
    if np.min(scores) < 0.0 or np.max(scores) > 1.0:
        probs = 1.0 / (1.0 + np.exp(-scores))
    else:
        probs = scores

    ece = compute_calibration_ece(probs, y_true)
    brier = float(brier_score_loss(y_true, probs))

    return {
        "auroc": auroc,
        "pr_auc": pr_auc,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "ece": ece,
        "brier": brier,
        "selected_threshold": threshold,
    }

