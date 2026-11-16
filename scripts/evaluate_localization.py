"""Evaluation metrics calculator for token-level and span-level hallucination localization.

Evaluates HalLoc, REFIND, VisER, and XMCA on localized token/span accuracy:
1. Token-level Precision, Recall, and F1.
2. Span-level Intersection over Union (IoU).
3. Category-specific breakdown (Object, Attribute, Relationship, Scene).
"""

from typing import Dict, Any, List, Tuple
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score


def evaluate_token_localization(
    ground_truth_tokens_binary: List[List[int]],
    predicted_tokens_binary: List[List[int]],
) -> Dict[str, float]:
    """Calculate token-level localization metrics across sequences."""
    y_true_flat = []
    y_pred_flat = []

    for true_seq, pred_seq in zip(ground_truth_tokens_binary, predicted_tokens_binary):
        L = min(len(true_seq), len(pred_seq))
        y_true_flat.extend(true_seq[:L])
        y_pred_flat.extend(pred_seq[:L])

    y_t = np.asarray(y_true_flat, dtype=int)
    y_p = np.asarray(y_pred_flat, dtype=int)

    if len(y_t) == 0:
        return {"token_f1": 0.0, "token_precision": 0.0, "token_recall": 0.0, "token_iou": 0.0}

    prec = float(precision_score(y_t, y_p, zero_division=0))
    rec = float(recall_score(y_t, y_p, zero_division=0))
    f1 = float(f1_score(y_t, y_p, zero_division=0))

    intersection = np.logical_and(y_t, y_p).sum()
    union = np.logical_or(y_t, y_p).sum()
    iou = float(intersection / union) if union > 0 else 1.0

    return {
        "token_f1": f1,
        "token_precision": prec,
        "token_recall": rec,
        "token_iou": iou,
    }

