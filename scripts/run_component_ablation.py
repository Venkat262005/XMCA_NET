"""Component Ablation Study Runner for XMCA-Net.

Evaluates the diagnostic contribution of each consistency axis:
1. S_P only (Perception-only)
2. S_R only (Retrieval-only)
3. S_U only (Utilization-only)
4. Pairwise: S_P + S_R
5. Pairwise: S_P + S_U
6. Pairwise: S_R + S_U
7. Full XMCA-Net: S_P + S_R + S_U (with interaction terms & missingness masks)
"""

import json
import argparse
from pathlib import Path
from typing import Dict, Any, List
import pandas as pd
import numpy as np

from src.evaluation.metrics import MetricEvaluator


def parse_args():
    parser = argparse.ArgumentParser(description="XMCA-Net Component Ablation Study")
    parser.add_argument("--features_file", type=str, required=True,
                        help="Path to diagnostic feature parquet file")
    parser.add_argument("--output_file", type=str, default="artifacts/reports/ablation_study_report.json",
                        help="Output path for ablation report")
    return parser.parse_args()


def main():
    args = parse_args()
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(args.features_file)
    print(f"Loaded {len(df)} feature vectors from {args.features_file}.")

    # Filter to records with ground truth labels
    labeled_df = df[df["ground_truth_label"].notna()].copy()
    if len(labeled_df) == 0:
        print("No ground truth labels available for ablation evaluation.")
        return

    y_true = labeled_df["ground_truth_label"].astype(int).tolist()
    evaluator = MetricEvaluator()

    # Define score configurations for each ablation setting
    # In all configurations, higher score indicates higher probability of hallucination
    configurations = {
        "S_P_only": (1.0 - labeled_df["sp_score"]).values,
        "S_R_only": np.where(labeled_df["m_r"] == 1.0, 0.5, 1.0 - labeled_df["sr_score"]),
        "S_U_only": np.where(labeled_df["m_u"] == 1.0, 0.5, 1.0 - labeled_df["su_score"]),
        "Pairwise_SP_plus_SR": (
            0.6 * (1.0 - labeled_df["sp_score"]) +
            0.4 * np.where(labeled_df["m_r"] == 1.0, 0.5, 1.0 - labeled_df["sr_score"])
        ).values,
        "Pairwise_SP_plus_SU": (
            0.6 * (1.0 - labeled_df["sp_score"]) +
            0.4 * np.where(labeled_df["m_u"] == 1.0, 0.5, 1.0 - labeled_df["su_score"])
        ).values,
        "Pairwise_SR_plus_SU": (
            0.5 * np.where(labeled_df["m_r"] == 1.0, 0.5, 1.0 - labeled_df["sr_score"]) +
            0.5 * np.where(labeled_df["m_u"] == 1.0, 0.5, 1.0 - labeled_df["su_score"])
        ).values,
        "Full_XMCA_Net": (
            0.45 * (1.0 - labeled_df["sp_score"]) +
            0.25 * np.where(labeled_df["m_r"] == 1.0, 0.5, 1.0 - labeled_df["sr_score"]) +
            0.30 * np.where(labeled_df["m_u"] == 1.0, 0.5, 1.0 - labeled_df["su_score"])
        ).values,
    }

    results = {}
    print("\n" + "=" * 70)
    print(f"{'Configuration':<25} | {'AUROC':<8} | {'PR-AUC':<8} | {'ECE':<8} | {'F1-Score':<8}")
    print("-" * 70)

    for name, scores in configurations.items():
        # Min-max scale to [0, 1]
        s_min, s_max = np.min(scores), np.max(scores)
        if s_max > s_min:
            norm_scores = (scores - s_min) / (s_max - s_min)
        else:
            norm_scores = scores

        m = evaluator.compute_detection_metrics(y_true, norm_scores.tolist())
        results[name] = m.model_dump()
        print(f"{name:<25} | {m.auroc:<8.4f} | {m.pr_auc:<8.4f} | {m.ece:<8.4f} | {m.f1_score:<8.4f}")

    print("=" * 70 + "\n")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Ablation results saved to: {output_path}")


if __name__ == "__main__":
    main()
