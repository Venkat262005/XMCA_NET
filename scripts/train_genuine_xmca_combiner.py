"""Genuine XMCANetwork Combiner Training on Real GPU-Extracted Tri-Axis Features.

Loads genuine feature parquets from run_genuine_gpu_feature_extraction.py,
constructs DiagnosticFeatureVector objects, trains XMCADetectionCombiner on dev_train,
calibrates temperature on dev_val, and saves the checkpoint.

ZERO synthetic features. ZERO rng.randn. ZERO hardcoded fallbacks.
ZERO test-label leakage (test split is never touched in this script).

Usage:
    python scripts/train_genuine_xmca_combiner.py --model llava_1.5_7b
"""

import os
import sys
import argparse
import hashlib
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.features.extractor import DiagnosticFeatureVector
from src.combiner.combiner import XMCADetectionCombiner, XMCAAttributionHead


def sha256_path(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def df_to_diagnostic_vectors(df: pd.DataFrame) -> tuple:
    """Convert genuine feature DataFrame to List[DiagnosticFeatureVector] + labels."""
    vectors = []
    labels = []

    for _, row in df.iterrows():
        try:
            c_val = row.get("cause_label", None)
            cause_label = int(c_val) if (pd.notna(c_val) and c_val is not None) else None
            vec = DiagnosticFeatureVector(
                sample_id=str(row["sample_id"]),
                dataset_id=str(row["dataset_id"]),
                model_name=str(row["model_name"]),
                ground_truth_label=int(row["ground_truth_label"]),
                cause_label=cause_label,
                sp_score=float(row["sp_score"]),
                visual_support_ratio=float(row["visual_support_ratio"]),
                visual_contradiction_ratio=float(row["visual_contradiction_ratio"]),
                visual_coverage=float(row["visual_coverage"]),
                grounding_uncertainty=float(row["grounding_uncertainty"]),
                sr_score=float(row["sr_score"]),
                top_relevance_score=float(row["top_relevance_score"]),
                mean_relevance_score=float(row["mean_relevance_score"]),
                score_margin=float(row["score_margin"]),
                evidence_coverage=float(row["evidence_coverage"]),
                m_r=float(row["m_r"]),
                su_score=float(row["su_score"]),
                entailment_ratio=float(row["entailment_ratio"]),
                contradiction_ratio=float(row["contradiction_ratio"]),
                omission_ratio=float(row["omission_ratio"]),
                unsupported_claim_ratio=float(row["unsupported_claim_ratio"]),
                m_u=float(row["m_u"]),
                total_claims=int(row["total_claims"]),
                sp_x_sr=float(row["sp_x_sr"]),
                sp_x_su=float(row["sp_x_su"]),
            )
            vectors.append(vec)
            labels.append(int(row["ground_truth_label"]))
        except Exception as exc:
            print(f"  [Warn] Skipping row {row.get('sample_id', '?')}: {exc}")

    return vectors, labels


def load_split(features_dir: Path, model_name: str, split: str) -> pd.DataFrame:
    path = features_dir / f"genuine_features_{model_name}_{split}.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"Feature file not found: {path}\n"
            f"Run run_genuine_gpu_feature_extraction.py first."
        )
    df = pd.read_parquet(path)
    if "extraction_error" in df.columns:
        err = df["extraction_error"].notna().sum()
        if err > 0:
            print(f"  [Warn] Dropping {err} rows with extraction errors from {split}.")
        df = df[df["extraction_error"].isna()].reset_index(drop=True)
    print(f"[Load] {split}: {len(df)} rows | SHA-256={sha256_path(path)[:16]}...")
    return df


def main():
    parser = argparse.ArgumentParser(description="Train genuine XMCANetwork combiner on real GPU features.")
    parser.add_argument("--model", type=str, default="llava_1.5_7b")
    parser.add_argument("--features_dir", type=str, default="artifacts/features")
    parser.add_argument("--output_dir", type=str, default="artifacts/models/genuine_combiner")
    parser.add_argument("--epochs", type=int, default=60,
                        help="Training epochs for XMCANetwork.")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    model_name = args.model
    features_dir = REPO_ROOT / args.features_dir
    output_dir = REPO_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"GENUINE XMCA COMBINER TRAINING — {model_name}")
    print(f"{'='*70}\n")

    # Load genuine feature splits (test is NEVER loaded here)
    train_df = load_split(features_dir, model_name, "dev_train")
    val_df = load_split(features_dir, model_name, "dev_val")

    print(f"\nBuilding DiagnosticFeatureVectors...")
    train_vecs, train_labels = df_to_diagnostic_vectors(train_df)
    val_vecs, val_labels = df_to_diagnostic_vectors(val_df)

    n_pos = sum(train_labels)
    n_neg = len(train_labels) - n_pos
    print(f"Training: {len(train_vecs)} vectors | pos={n_pos}, neg={n_neg}")
    print(f"Validation: {len(val_vecs)} vectors | pos={sum(val_labels)}, neg={len(val_labels)-sum(val_labels)}")

    if len(train_vecs) == 0:
        raise RuntimeError("Zero training vectors — cannot train combiner. Run feature extraction first.")
    if len(val_vecs) == 0:
        raise RuntimeError("Zero validation vectors — cannot train combiner. Run feature extraction first.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")

    # Instantiate and train genuine combiner
    combiner = XMCADetectionCombiner(input_dim=20, hidden_dim=128, device=device)

    print(f"\n[Training] XMCANetwork for {args.epochs} epochs with validation tracking...")
    combiner.fit(
        training_vectors=train_vecs,
        labels=train_labels,
        val_vectors=val_vecs,
        val_labels=val_labels,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=1e-2,
    )

    print("[Calibrating] Temperature scaling on validation partition...")
    combiner.calibrate_temperature(val_vectors=val_vecs, val_labels=val_labels)
    print(f"  Learned temperature: {combiner.scaler.temperature:.4f}")

    # Evaluate on val to report genuine val AUROC before saving
    from sklearn.metrics import roc_auc_score
    X_val_np = np.stack([v.to_feature_list() for v in val_vecs], axis=0).astype(np.float32)
    X_val_norm = combiner._normalize_features(X_val_np, fit=False)
    combiner.model.eval()
    with torch.no_grad():
        x_t = torch.tensor(X_val_norm, dtype=torch.float32, device=device)
        logits = combiner.model(x_t).cpu().numpy()
    try:
        val_auroc = roc_auc_score(val_labels, logits)
    except Exception:
        val_auroc = float("nan")
    print(f"\nValidation AUROC (pre-temperature): {val_auroc:.4f}")

    # Save genuine combiner checkpoint
    ckpt_path = output_dir / f"xmca_{model_name}_combiner.pt"
    torch.save({
        "model_name": model_name,
        "model_state": combiner.model.state_dict(),
        "scaler_state": combiner.scaler.state_dict(),
        "mean": combiner.mean,
        "std": combiner.std,
        "decision_threshold": combiner.decision_threshold,
        "val_auroc": val_auroc,
        "temperature": combiner.scaler.temperature,
        "train_samples": len(train_vecs),
        "val_samples": len(val_vecs),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "genuine_extraction": True,
        "note": "Trained on genuine GPU-extracted tri-axis features. Zero synthetic data. Zero test leakage.",
    }, ckpt_path)
    sha = sha256_path(ckpt_path)
    print(f"\n[Saved] {ckpt_path.name} | SHA-256={sha}")

    # Train genuine deep attribution head (Output 2)
    print(f"\n[Training] XMCAAttributionHead for root-cause diagnosis...")
    attribution_head = XMCAAttributionHead(hidden_dim=128, device=device)
    train_cause_vecs = [v for v in train_vecs if v.cause_label is not None and v.ground_truth_label == 1]
    train_cause_lbls = [v.cause_label for v in train_cause_vecs]

    if len(train_cause_vecs) > 0:
        print(f"  Training on {len(train_cause_vecs)} verified failure vectors across causes: "
              f"P={train_cause_lbls.count(0)}, R={train_cause_lbls.count(1)}, U={train_cause_lbls.count(2)}")
        attribution_head.fit(
            combiner=combiner,
            vectors=train_cause_vecs,
            cause_labels=train_cause_lbls,
            epochs=40,
            lr=1e-3,
        )
        attr_ckpt = output_dir / f"xmca_{model_name}_attribution_head.pt"
        torch.save({
            "model_name": model_name,
            "network_state": attribution_head.network.state_dict(),
            "hidden_dim": 128,
            "num_causes": 3,
            "train_samples": len(train_cause_vecs),
            "genuine_extraction": True,
            "note": "Trained on genuine cause annotations. Softmax over [perception, retrieval, utilization].",
        }, attr_ckpt)
        sha_attr = sha256_path(attr_ckpt)
        print(f"  [Saved] {attr_ckpt.name} | SHA-256={sha_attr}")
    else:
        print("  [Warn] No cause labels found in dev_train — run update_genuine_parquet_labels.py first.")

    print(f"\n[Done] Genuine XMCANetwork combiner and attribution head trained and saved.")


if __name__ == "__main__":
    main()
