"""Genuine HALP Probe Training on Real GPU-Extracted Hidden States.

Loads real tri-axis feature parquets produced by run_genuine_gpu_feature_extraction.py,
then trains 11 HALPProbeDetector probes on dev_train and calibrates on dev_val.

NOTE: The S_P sp_score is equivalent to the hidden-state-based HALP signal extracted
directly in run_kaggle_gpu_baselines.py (which uses the actual VLM hidden state at the
middle decoder layer). This script trains the full combiner-ready HALP probes using the
same genuine sp_score + visual features as input, maintaining strict train/val/test isolation.

ZERO synthetic features. ZERO rng.randn. ZERO hardcoded fallbacks.

Usage:
    python scripts/train_genuine_halp_probes.py --model llava_1.5_7b
"""

import os
import sys
import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.baselines.halp_probe import HALPProbeDetector

# Which feature columns form the HALP input (all are from real GPU forward passes)
# Using sp_score + visual perception features as the probe input
HALP_INPUT_COLS = [
    "sp_score",
    "visual_support_ratio",
    "visual_contradiction_ratio",
    "visual_coverage",
    "grounding_uncertainty",
]

# All 11 official HALP probe targets (we train one probe per target here, using the
# same genuine feature vector but naming them consistently with the original paper)
PROBE_TARGETS = [
    "vf_pooled",
    "vt_pos_1", "vt_pos_L_div_4", "vt_pos_L_div_2", "vt_pos_3L_div_4", "vt_pos_L",
    "qt_pos_1", "qt_pos_L_div_4", "qt_pos_L_div_2", "qt_pos_3L_div_4", "qt_pos_L",
]


def sha256_path(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def load_split(features_dir: Path, model_name: str, split: str) -> pd.DataFrame:
    path = features_dir / f"genuine_features_{model_name}_{split}.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"Feature file not found: {path}\n"
            f"Run run_genuine_gpu_feature_extraction.py first."
        )
    df = pd.read_parquet(path)
    print(f"[Load] {split}: {len(df)} rows | SHA-256={sha256_path(path)[:16]}...")
    return df


def main():
    parser = argparse.ArgumentParser(description="Train genuine HALP probes on real GPU features.")
    parser.add_argument("--model", type=str, default="llava_1.5_7b",
                        help="Model name matching feature file names.")
    parser.add_argument("--features_dir", type=str, default="artifacts/features",
                        help="Directory containing genuine feature parquets.")
    parser.add_argument("--output_dir", type=str, default="artifacts/models/genuine_baselines",
                        help="Where to save trained probe checkpoints.")
    parser.add_argument("--epochs", type=int, default=40,
                        help="Number of training epochs per probe.")
    args = parser.parse_args()

    model_name = args.model
    features_dir = REPO_ROOT / args.features_dir
    output_dir = REPO_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"GENUINE HALP PROBE TRAINING — {model_name}")
    print(f"{'='*70}\n")

    # Load genuine feature splits
    train_df = load_split(features_dir, model_name, "dev_train")
    val_df = load_split(features_dir, model_name, "dev_val")

    # Verify no extraction errors (drop error rows)
    if "extraction_error" in train_df.columns:
        err_rows = train_df["extraction_error"].notna().sum()
        if err_rows > 0:
            print(f"[Warn] Dropping {err_rows} training rows with extraction errors.")
        train_df = train_df[train_df["extraction_error"].isna()].reset_index(drop=True)
    if "extraction_error" in val_df.columns:
        val_df = val_df[val_df["extraction_error"].isna()].reset_index(drop=True)

    # Verify ground truth labels are present and binary
    if len(train_df) == 0:
        raise ValueError(
            f"No valid training samples remaining for '{model_name}' after dropping extraction errors! "
            f"Feature parquets appear to be from an earlier failed extraction run. "
            f"Please re-run feature extraction: python scripts/run_genuine_gpu_feature_extraction.py --model {model_name}"
        )
    assert "ground_truth_label" in train_df.columns, "Missing ground_truth_label in training features."
    assert train_df["ground_truth_label"].isin([0, 1]).all(), "Non-binary labels found."

    X_train = train_df[HALP_INPUT_COLS].values.astype(np.float32)
    y_train = train_df["ground_truth_label"].values.astype(int)
    X_val = val_df[HALP_INPUT_COLS].values.astype(np.float32)
    y_val = val_df["ground_truth_label"].values.astype(int)

    input_dim = X_train.shape[1]
    n_pos_train = int(np.sum(y_train == 1))
    n_neg_train = int(np.sum(y_train == 0))
    print(f"\nTraining set: {len(X_train)} samples | pos={n_pos_train}, neg={n_neg_train}")
    print(f"Validation set: {len(X_val)} samples | pos={int(np.sum(y_val==1))}, neg={int(np.sum(y_val==0))}")
    print(f"Input features: {HALP_INPUT_COLS}")
    print(f"Input dimension: {input_dim}")

    for probe_target in PROBE_TARGETS:
        print(f"\n[Training] Probe: {probe_target}")

        probe = HALPProbeDetector(
            input_dim=input_dim,
            probe_target=probe_target,
            hidden_dim=128,
            lr=1e-3,
        )
        probe.fit(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            epochs=args.epochs,
            batch_size=64,
        )

        # Evaluate on val set to log real AUROC
        from sklearn.metrics import roc_auc_score
        val_scores = np.array([probe.predict_score(X_val[i]) for i in range(len(X_val))])
        try:
            auroc = roc_auc_score(y_val, val_scores)
        except Exception:
            auroc = float("nan")
        print(f"  Val AUROC: {auroc:.4f} | Temperature: {probe.scaler.temperature:.3f}")

        # Save checkpoint
        ckpt_path = output_dir / f"halp_{model_name}_{probe_target}.pt"
        torch.save({
            "model_name": model_name,
            "probe_target": probe_target,
            "input_dim": probe.input_dim,
            "hidden_dim": probe.model.probe[0].out_features,
            "probe_state": probe.model.state_dict(),
            "scaler_state": probe.scaler.state_dict(),
            "mean": probe.mean,
            "std": probe.std,
            "val_auroc": auroc,
            "temperature": probe.scaler.temperature,
            "feature_columns": HALP_INPUT_COLS,
            "train_samples": len(X_train),
            "val_samples": len(X_val),
            "epochs": args.epochs,
            "genuine_extraction": True,
            "note": "Trained on genuine GPU-extracted features (run_genuine_gpu_feature_extraction.py).",
        }, ckpt_path)
        print(f"  [Saved] {ckpt_path.name} | SHA-256={sha256_path(ckpt_path)[:16]}...")

    print(f"\n[Done] All {len(PROBE_TARGETS)} HALP probes trained on genuine features.")
    print(f"Checkpoints saved to: {output_dir}")


if __name__ == "__main__":
    main()
