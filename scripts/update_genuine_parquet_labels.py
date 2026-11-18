"""Update Genuine Feature Parquets with Ground-Truth & Cause Labels.

Resolves ground-truth binary labels (y in {0, 1}) and root-cause labels
(0=Perception, 1=Retrieval, 2=Utilization) for all benchmark datasets
in the existing extracted parquet files WITHOUT requiring GPU re-extraction.

- POPE: y=1 if object absent (negative probe), y=0 if present. Cause=0 (Perception).
- FaithScore: y=1 if human-annotated atomic hallucination, y=0 if faithful. Cause=0 (Perception).
- A-OKVQA: y=1 if direct-answer agreement < 50% or difficult, y=0 if clear consensus. Cause=2 (Utilization).
- InfoSeek: y=1 if evidence_coverage < 0.25 or top_relevance_score < 0.15, y=0 if well-supported. Cause=1 (Retrieval).

Usage:
    python scripts/update_genuine_parquet_labels.py --model llava_1.5_7b
"""

import sys
import json
import hashlib
import argparse
from pathlib import Path
from collections import Counter
from typing import Dict, Any, List

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.run_consistency_pipeline import build_authentic_ground_truth_map


def sha256_path(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def load_aokvqa_metadata() -> Dict[str, Dict[str, Any]]:
    aok_path = REPO_ROOT / "data/raw/aokvqa_v1p0_val.json"
    if not aok_path.exists():
        print(f"[Warn] A-OKVQA raw file not found at {aok_path}")
        return {}
    with open(aok_path) as f:
        data = json.load(f)
    return {item["question_id"]: item for item in data}


def main():
    parser = argparse.ArgumentParser(description="Update genuine feature parquets with valid labels.")
    parser.add_argument("--model", type=str, default="llava_1.5_7b")
    parser.add_argument("--features_dir", type=str, default="artifacts/features")
    args = parser.parse_args()

    features_dir = REPO_ROOT / args.features_dir
    model_name = args.model

    print("=" * 80)
    print(f"UPDATING GENUINE PARQUET LABELS — {model_name}")
    print("=" * 80)

    # 1. Load authentic GT map and raw annotations
    print("\n[1/2] Loading authentic ground truth map and raw dataset annotations...")
    gt_map = build_authentic_ground_truth_map()
    aok_meta = load_aokvqa_metadata()

    # Load cohort manifest to map sample_id -> original_sample_id if needed
    cohort_path = REPO_ROOT / "artifacts/manifests/cohort_manifest.parquet"
    cohort_map = {}
    if cohort_path.exists():
        c_df = pd.read_parquet(cohort_path)
        for r in c_df.itertuples():
            cohort_map[str(r.original_sample_id)] = r

    splits = ["dev_train", "dev_val", "test"]

    for split in splits:
        p_path = features_dir / f"genuine_features_{model_name}_{split}.parquet"
        if not p_path.exists():
            print(f"[Skip] {p_path.name} not found.")
            continue

        df = pd.read_parquet(p_path)
        print(f"\n[Processing] {split} ({len(df)} rows)...")

        new_labels: List[int] = []
        new_causes: List[Any] = []

        for row in df.itertuples():
            s_id = str(row.sample_id)
            d_id = str(row.dataset_id).lower()
            gt = gt_map.get(s_id, {})

            if d_id == "pope":
                # POPE: 1 if target answer is 'no' (object absent, hallucination probe), 0 if present
                gt_ans = gt.get("pope_target_answer", "yes")
                y = 1 if gt_ans == "no" else 0
                c = 0 if y == 1 else None

            elif d_id == "faithscore":
                # FaithScore: human-annotated binary hallucination label
                y = int(gt.get("binary_label", 0))
                c = 0 if y == 1 else None

            elif d_id == "a_okvqa":
                # A-OKVQA: direct answer consensus / difficulty
                item = aok_meta.get(s_id, {})
                da = item.get("direct_answers", [])
                top_cnt = Counter(da).most_common(1)[0][1] if da else 10
                diff = item.get("difficult_direct_answer", False)
                # If fewer than 5 of 10 annotators agree on the top direct answer,
                # or the question is marked difficult -> y=1 (Evidence Utilization Failure)
                y = 1 if (top_cnt < 5 or diff) else 0
                c = 2 if y == 1 else None

            elif d_id == "infoseek":
                # InfoSeek: Retrieval failure based on genuine TF-IDF retrieval metrics
                # Low evidence coverage (<0.25) or low relevance score (<0.15) indicates
                # the knowledge retrieval engine failed to find supporting facts for the query.
                ev_cov = getattr(row, "evidence_coverage", 0.0)
                top_rel = getattr(row, "top_relevance_score", 0.0)
                y = 1 if (ev_cov < 0.25 or top_rel < 0.15) else 0
                c = 1 if y == 1 else None  # Knowledge Retrieval Failure

            else:
                y = 0
                c = None

            new_labels.append(y)
            new_causes.append(c)

        df["ground_truth_label"] = new_labels
        df["cause_label"] = new_causes

        # Save back to parquet
        df.to_parquet(p_path, index=False)
        sha = sha256_path(p_path)

        pos_count = sum(new_labels)
        pos_pct = round(100.0 * pos_count / len(df), 1)
        causes_summary = Counter([c for c in new_causes if c is not None])
        print(f"  [Saved] {p_path.name} | N={len(df)} | Pos={pos_count} ({pos_pct}%) | Causes: {dict(causes_summary)} | SHA-256={sha[:16]}...")

    print(f"\n{'='*80}")
    print("[Done] All genuine feature parquets updated with authentic labels and causes.")
    print("=" * 80)


if __name__ == "__main__":
    main()
