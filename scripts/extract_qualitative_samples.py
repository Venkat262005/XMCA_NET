#!/usr/bin/env python3
"""Extract and format authentic proof samples for qualitative analysis and paper presentation.

Samples:
- 5 Faithful / Grounded (Non-hallucinated)
- 3 Visual Perception Failures
- 3 Knowledge Retrieval Failures
- 3 Evidence Utilization Failures
"""

import os
import sys
import json
import argparse
from pathlib import Path
import pandas as pd
import torch

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.features.extractor import DiagnosticFeatureVector
from src.combiner.combiner import XMCADetectionCombiner, XMCAAttributionHead
from src.data.loader import AuthenticDatasetLoader


def parse_args():
    parser = argparse.ArgumentParser(description="Extract qualitative proof samples")
    parser.add_argument("--tier", type=int, default=1000, help="Benchmark tier (100, 300, 600, 1000)")
    parser.add_argument("--model", type=str, default="llava_1.5_7b", help="Target VLM")
    parser.add_argument("--output_file", type=str, default=None, help="Path to save output JSON")
    return parser.parse_args()


def main():
    args = parse_args()
    tier = args.tier
    model = args.model

    # Canonical model naming
    model_alias = model.replace("-", "_").lower()
    if "llava" in model_alias:
        canonical_model = "llava_1.5_7b"
        gen_dir_name = "llava"
    elif "qwen" in model_alias:
        canonical_model = "qwen2_vl_7b"
        gen_dir_name = "qwen"
    elif "internvl" in model_alias:
        canonical_model = "internvl2_2b"
        gen_dir_name = "internvl"
    else:
        canonical_model = model
        gen_dir_name = model

    # 1. Locate features parquet
    features_candidates = [
        Path(f"artifacts/features/tier_{tier}_{model}_features.parquet"),
        Path(f"artifacts/features/tier_{tier}_{canonical_model}_features.parquet"),
    ]
    features_path = None
    for p in features_candidates:
        if p.exists():
            features_path = p
            break

    if not features_path:
        print(f"Error: Features parquet not found in artifacts/features/ for tier {tier}, model {model}.")
        return

    print(f"Loading features from: {features_path}")
    features_df = pd.read_parquet(features_path)

    # 2. Locate generation records
    gen_candidates = [
        Path(f"artifacts/generations/full_benchmark/tier_{tier}/{gen_dir_name}/generation_records.jsonl"),
        Path(f"artifacts/generations/full_benchmark/tier_{tier}/{gen_dir_name}/{model}.jsonl"),
        Path(f"artifacts/generations/full_benchmark/tier_{tier}/{gen_dir_name}/{canonical_model}.jsonl"),
    ]
    gen_path = None
    for p in gen_candidates:
        if p.exists():
            gen_path = p
            break

    gen_records = {}
    if gen_path:
        print(f"Loading generation records from: {gen_path}")
        with open(gen_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        rec = json.loads(line)
                        sid = rec.get("sample_id") or rec.get("original_sample_id")
                        if sid:
                            gen_records[sid] = rec
                    except Exception:
                        pass
        print(f"Loaded {len(gen_records)} generation records.")

    # 3. Load trained combiner checkpoint if available
    ckpt_candidates = [
        Path(f"artifacts/models/tier_{tier}_{canonical_model}_combiner.pt"),
        Path(f"artifacts/models/tier_{tier}_{model}_combiner.pt"),
    ]
    ckpt_path = None
    for p in ckpt_candidates:
        if p.exists():
            ckpt_path = p
            break

    combiner = XMCADetectionCombiner()
    attr_head = XMCAAttributionHead()

    if ckpt_path:
        print(f"Loading trained neural weights from: {ckpt_path}")
        state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        combiner.model.load_state_dict(state["model_state_dict"])
        combiner.scaler.load_state_dict(state["scaler_state_dict"])
        attr_head.network.load_state_dict(state["attr_head_state_dict"])
        combiner.mean = state.get("mean", combiner.mean)
        combiner.std = state.get("std", combiner.std)
        combiner.temperature = state.get("temperature", combiner.temperature)
        combiner.decision_threshold = state.get("threshold", 0.50)
        combiner.dataset_thresholds = state.get("dataset_thresholds", {})
        combiner.is_fitted = True
    else:
        print("Warning: Trained checkpoint not found. Evaluating with initialized combiner.")

    # 4. Convert dataframe into DiagnosticFeatureVector list and predict
    print("Evaluating diagnostic features and computing cause attributions...")
    samples_data = []

    for _, row in features_df.iterrows():
        sid = row["sample_id"]
        v = DiagnosticFeatureVector(
            sample_id=str(sid),
            dataset_id=str(row["dataset_id"]),
            model_name=str(row["model_name"]),
            ground_truth_label=int(row["ground_truth_label"]) if pd.notnull(row["ground_truth_label"]) else None,
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

        pred = combiner.predict(v, dataset_id=v.dataset_id)
        attr = attr_head.attribute_cause(v, pred, combiner=combiner)

        gen_rec = gen_records.get(sid, {})
        question = gen_rec.get("question") or gen_rec.get("prompt") or "N/A"
        answer = gen_rec.get("generated_text") or gen_rec.get("model_answer") or "N/A"
        image_id = gen_rec.get("image_id") or gen_rec.get("original_image_id") or "N/A"

        samples_data.append({
            "sample_id": sid,
            "dataset_id": v.dataset_id,
            "image_id": image_id,
            "question": question,
            "generated_answer": answer,
            "ground_truth_label": v.ground_truth_label,
            "predicted_label": pred.predicted_label,
            "hallucination_probability": round(pred.hallucination_probability, 4),
            "attributed_cause": attr.attributed_cause,
            "cause_confidence": round(attr.confidence, 4),
            "cause_probabilities": {
                "perception": round(attr.cause_probabilities.get("perception", 0.0), 4),
                "retrieval": round(attr.cause_probabilities.get("retrieval", 0.0), 4),
                "utilization": round(attr.cause_probabilities.get("utilization", 0.0), 4),
            },
            "sp_score": round(v.sp_score, 4),
            "sr_score": round(v.sr_score, 4),
            "su_score": round(v.su_score, 4),
            "m_r": v.m_r,
            "m_u": v.m_u,
        })

    # 5. Extract target cohorts:
    # 5 Faithful (gt=0, pred=0)
    faithful = [s for s in samples_data if s["ground_truth_label"] == 0 and s["predicted_label"] == 0]
    # 3 Perception failures (attributed_cause == "perception", pred=1)
    perception_fails = [s for s in samples_data if s["attributed_cause"] == "perception" and s["predicted_label"] == 1]
    # 3 Retrieval failures (attributed_cause == "retrieval", pred=1)
    retrieval_fails = [s for s in samples_data if s["attributed_cause"] == "retrieval" and s["predicted_label"] == 1]
    # 3 Utilization failures (attributed_cause == "utilization", pred=1)
    utilization_fails = [s for s in samples_data if s["attributed_cause"] == "utilization" and s["predicted_label"] == 1]

    # Select representative samples across different datasets
    def select_diverse(items, n):
        res = []
        seen_ds = set()
        # First pick unique datasets
        for it in items:
            if it["dataset_id"] not in seen_ds:
                res.append(it)
                seen_ds.add(it["dataset_id"])
                if len(res) == n:
                    return res
        # Then fill remaining
        for it in items:
            if it not in res:
                res.append(it)
                if len(res) == n:
                    break
        return res

    selected_faithful = select_diverse(faithful, 5)
    selected_perception = select_diverse(perception_fails, 3)
    selected_retrieval = select_diverse(retrieval_fails, 3)
    selected_utilization = select_diverse(utilization_fails, 3)

    print("\n" + "="*80)
    print("XMCA-NET QUALITATIVE PROOF SAMPLES SUMMARY")
    print("="*80)

    def print_section(title, samples, is_fail=True):
        print(f"\n{'='*30} {title} ({len(samples)} Samples) {'='*30}")
        for idx, s in enumerate(samples, 1):
            print(f"\n--- [{title.split()[0]}] Sample #{idx} ---")
            print(f"Dataset:       {s['dataset_id'].upper()} (ID: {s['sample_id']})")
            print(f"Image ID:      {s['image_id']}")
            print(f"Question:      {s['question']}")
            print(f"VLM Answer:    {s['generated_answer']}")
            print(f"Consistency:   S_P (Perception)={s['sp_score']:.3f} | S_R (Retrieval)={s['sr_score']:.3f} | S_U (Utilization)={s['su_score']:.3f}")
            print(f"Detection:     Predicted Label={'HALLUCINATED' if s['predicted_label']==1 else 'FAITHFUL'} (Prob={s['hallucination_probability']:.3f})")
            if is_fail:
                print(f"Attribution:   Cause={s['attributed_cause'].upper()} (Confidence={s['cause_confidence']:.3f})")
                print(f"Cause Probs:   P(Perception)={s['cause_probabilities']['perception']:.3f} | P(Retrieval)={s['cause_probabilities']['retrieval']:.3f} | P(Utilization)={s['cause_probabilities']['utilization']:.3f}")

    print_section("FAITHFUL (GROUNDED)", selected_faithful, is_fail=False)
    print_section("PERCEPTION FAILURES", selected_perception, is_fail=True)
    print_section("RETRIEVAL FAILURES", selected_retrieval, is_fail=True)
    print_section("UTILIZATION FAILURES", selected_utilization, is_fail=True)

    out_file = args.output_file or f"artifacts/reports/tier_{tier}_{canonical_model}_proof_samples.json"
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    output_payload = {
        "tier": tier,
        "model": canonical_model,
        "counts": {
            "total_samples": len(samples_data),
            "faithful_pool": len(faithful),
            "perception_failures_pool": len(perception_fails),
            "retrieval_failures_pool": len(retrieval_fails),
            "utilization_failures_pool": len(utilization_fails),
        },
        "faithful_samples": selected_faithful,
        "perception_failure_samples": selected_perception,
        "retrieval_failure_samples": selected_retrieval,
        "utilization_failure_samples": selected_utilization,
    }
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(output_payload, f, indent=2)
    print(f"\n[Saved full qualitative JSON to: {out_file}]")


if __name__ == "__main__":
    main()
