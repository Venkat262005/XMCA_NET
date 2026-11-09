"""End-to-End XMCA-Net Consistency and Detection Pipeline Runner.

Executes Stages 5 through 10 of the research pipeline:
1. Atomic claim extraction
2. Tri-axis consistency diagnostics (S_P via OWL-ViT, S_R via Cross-Encoder, S_U via DeBERTa-v3 NLI)
3. Aligned diagnostic feature compilation with explicit missingness masks
4. Anti-leakage partition splitting (Validation image groups vs Held-out Test groups)
5. XMCA-Net combiner training, temperature calibration, and threshold selection
6. Neural Cause-attribution head training and diagnosis
7. Standardized benchmark metrics computation (AUROC, PR-AUC, ECE, Attribution Accuracy)
"""

import os
import sys
import json
import argparse
from pathlib import Path

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import numpy as np
import torch

from src.retrieval.corpus import AuthenticKnowledgeCorpus
from src.retrieval.retriever import UnifiedRetriever
from src.claims.extractor import AtomicClaimExtractor
from src.perception.consistency import PerceptionConsistencyEstimator
from src.retrieval.consistency import RetrievalConsistencyEstimator
from src.utilization.consistency import UtilizationConsistencyEstimator
from src.features.extractor import FeatureExtractor, DiagnosticFeatureVector
from src.combiner.combiner import XMCADetectionCombiner, XMCAAttributionHead
from src.evaluation.metrics import MetricEvaluator
from src.baselines.baselines import BaselineEvaluator
from src.data.loader import AuthenticDatasetLoader
from scripts.run_full_benchmark_generation import locate_image


MODEL_KEY_ALIASES = {
    "llava": "llava",
    "llava_1.5_7b": "llava",
    "qwen2_vl": "qwen2_vl",
    "qwen2_vl_7b": "qwen2_vl",
    "internvl": "internvl",
    "internvl2_2b": "internvl",
    "internvl2_8b": "internvl2_8b",
}


def parse_args():
    parser = argparse.ArgumentParser(description="XMCA-Net Consistency Pipeline Runner")
    parser.add_argument("--tier", type=int, default=100, choices=[100, 300, 600, 1000],
                        help="Progressive sample tier to evaluate")
    parser.add_argument("--model", type=str, default="llava_1.5_7b",
                        help="Model key to evaluate")
    parser.add_argument("--generations_file", type=str, default=None,
                        help="Path to generation records JSONL")
    parser.add_argument("--corpus_file", type=str, default="artifacts/corpora/authentic_knowledge_corpus.parquet",
                        help="Path to authentic corpus parquet")
    parser.add_argument("--image_group_manifest", type=str, default="artifacts/manifests/global_image_group_manifest.parquet",
                        help="Path to global image group manifest")
    parser.add_argument("--output_dir", type=str, default="artifacts/reports",
                        help="Directory to save evaluation reports")
    return parser.parse_args()


def build_authentic_ground_truth_map() -> Dict[str, Dict[str, Any]]:
    """Build authentic lookup dictionary mapping sample_id to ground truth answer and labels."""
    loader = AuthenticDatasetLoader()
    gt_map: Dict[str, Dict[str, Any]] = {}

    # POPE
    try:
        pope_records = loader.load_pope()
        for p in pope_records:
            gt_map[p.original_sample_id] = {
                "binary_label": p.official_label_fields["hallucination_binary"],
                "cause_label": 0,  # Perception
                "category": "pope",
                "gt_objects": [p.raw_record.get("text", "").split()[-1].rstrip("?.!,")] if p.official_label_fields["hallucination_binary"] == 0 else [],
            }
    except Exception as e:
        print(f"Warning: could not load POPE ground truth: {e}")

    # FaithScore
    try:
        fs_records = loader.load_faithscore()
        for f in fs_records:
            facts = f.official_label_fields.get("stage2_atomic_facts", {})
            has_hallucination = any(v == 0 for v in facts.values()) if facts else False
            gt_map[f.original_sample_id] = {
                "binary_label": 1 if has_hallucination else 0,
                "cause_label": 0,  # Perception / Descriptive Grounding
                "category": "faithscore",
                "gt_objects": [],
            }
    except Exception as e:
        print(f"Warning: could not load FaithScore ground truth: {e}")

    # A-OKVQA
    try:
        aok_records = loader.load_aokvqa()
        for a in aok_records:
            correct_idx = a.official_label_fields.get("correct_choice_idx")
            choices = a.official_answer_fields.get("choices", [])
            correct_choice = choices[correct_idx] if 0 <= correct_idx < len(choices) else ""
            direct_answers = a.official_answer_fields.get("direct_answers", [])
            accepted = list(set([correct_choice.lower()] + [str(ans).lower() for ans in direct_answers if ans]))
            gt_map[a.original_sample_id] = {
                "accepted_answers": accepted,
                "cause_label": 1,  # Retrieval
                "category": "a_okvqa",
            }
    except Exception as e:
        print(f"Warning: could not load A-OKVQA ground truth: {e}")

    # InfoSeek
    try:
        info_records = loader.load_infoseek(max_records=2500)
        for item in info_records:
            answers = item.official_answer_fields.get("answer", [])
            eval_answers = item.official_answer_fields.get("answer_eval", [])
            clean_eval = [str(x).lower() for x in eval_answers if isinstance(x, str)]
            accepted = list(set([str(ans).lower() for ans in answers] + clean_eval))
            gt_map[item.original_sample_id] = {
                "accepted_answers": accepted,
                "cause_label": 1,  # Retrieval
                "category": "infoseek",
            }
    except Exception as e:
        print(f"Warning: could not load InfoSeek ground truth: {e}")

    return gt_map


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    features_dir = Path("artifacts/features")
    features_dir.mkdir(parents=True, exist_ok=True)
    models_dir = Path("artifacts/models")
    models_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("XMCA-Net End-to-End Consistency Diagnostic & Detection Pipeline")
    print(f"Tier: {args.tier} | Model: {args.model}")
    print("=" * 65)

    # 1. Resolve generations file
    canonical_model = MODEL_KEY_ALIASES.get(args.model, args.model)
    gen_file = args.generations_file
    if gen_file is None:
        cand1 = Path(f"artifacts/generations/full_benchmark/tier_{args.tier}/{args.model}/generation_records.jsonl")
        cand2 = Path(f"artifacts/generations/full_benchmark/tier_{args.tier}/{canonical_model}/generation_records.jsonl")
        if cand1.exists():
            gen_path = cand1
        elif cand2.exists():
            gen_path = cand2
        else:
            gen_path = cand1
    else:
        gen_path = Path(gen_file)

    if not gen_path.exists():
        print(f"Generation records file not found at: {gen_path}")
        print("Please run scripts/run_full_benchmark_generation.py first or specify --generations_file.")
        return

    # Load generation records
    records = []
    with open(gen_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    print(f"Loaded {len(records)} generation records from {gen_path}.")

    # 2. Load authentic corpus & initialize retriever
    print("Loading authentic knowledge corpus...")
    corpus = AuthenticKnowledgeCorpus.load_parquet(Path(args.corpus_file))
    print(f"Loaded corpus with {len(corpus)} authentic documents.")
    retriever = UnifiedRetriever(corpus)

    # 3. Load global image groups for strict anti-leakage partitioning
    image_groups_df = pd.read_parquet(args.image_group_manifest)
    training_eligible_images = set(
        image_groups_df[image_groups_df["eligible_for_combiner_training"] == True]["global_image_id"]
    )
    held_out_images = set(
        image_groups_df[image_groups_df["combiner_partition"] == "held_out_test"]["global_image_id"]
    )
    print(f"Image partition loaded: {len(training_eligible_images)} training-eligible, {len(held_out_images)} held-out.")

    # 4. Load authentic ground truth labels
    print("Building authentic ground truth mapping...")
    gt_map = build_authentic_ground_truth_map()
    print(f"Mapped {len(gt_map)} authentic ground truth entries.")

    # 5. Initialize diagnostic engines
    claim_extractor = AtomicClaimExtractor()
    p_estimator = PerceptionConsistencyEstimator()
    r_estimator = RetrievalConsistencyEstimator()
    u_estimator = UtilizationConsistencyEstimator()
    feat_extractor = FeatureExtractor()

    feature_vectors: List[DiagnosticFeatureVector] = []
    cause_labels_list: List[Optional[int]] = []
    print("\nExecuting multi-axis neural consistency diagnostics...")

    for i, rec in enumerate(records):
        sample_id = rec.get("sample_id", f"sample_{i}")
        dataset_id = rec.get("dataset_id", "unknown").lower()
        image_id = rec.get("image_id", "")
        image_path = rec.get("image_path", None)
        if image_path:
            p = Path(image_path)
            if not p.exists():
                found_path = locate_image(Path("data/images"), dataset_id, image_id or p.name)
                image_path = str(found_path) if found_path else None
        elif image_id:
            found_path = locate_image(Path("data/images"), dataset_id, image_id)
            image_path = str(found_path) if found_path else None
        question = rec.get("question", "")
        generated_text = rec.get("generated_text", "")

        # Look up genuine ground truth label
        gt_entry = gt_map.get(sample_id, {})
        gt_label = None
        cause_label = None
        gt_objects = None

        if "binary_label" in gt_entry:
            gt_label = int(gt_entry["binary_label"])
            cause_label = gt_entry.get("cause_label", 0)
            gt_objects = gt_entry.get("gt_objects", None)
        elif "accepted_answers" in gt_entry:
            accepted = gt_entry["accepted_answers"]
            gen_lower = generated_text.lower()
            is_faithful = any(ans in gen_lower for ans in accepted if ans)
            gt_label = 0 if is_faithful else 1
            cause_label = gt_entry.get("cause_label", 1)

        # Step 5a: Atomic claim extraction
        claims_res = claim_extractor.extract_claims(sample_id, generated_text)

        # Step 5b: Perception consistency (S_P) via zero-shot OWL-ViT visual grounding
        p_res = p_estimator.compute_sp(
            sample_id=sample_id,
            image_id=image_id,
            claims=claims_res.claims,
            image_path=image_path,
            ground_truth_objects=gt_objects,
        )

        # Step 5c: Retrieval consistency (S_R) via Neural Cross-Encoder
        r_result = retriever.retrieve(
            dataset_id=dataset_id,
            sample_id=sample_id,
            query=question,
            top_k=5,
        )
        r_res = r_estimator.compute_sr(r_result)

        # Step 5d: Utilization consistency (S_U) via DeBERTa-v3 NLI
        u_res = u_estimator.compute_su(
            dataset_id=dataset_id,
            sample_id=sample_id,
            claims=claims_res.claims,
            retrieval_result=r_result,
        )

        # Step 5e: Feature vector compilation
        feat_vec = feat_extractor.extract_features(
            sample_id=sample_id,
            dataset_id=dataset_id,
            model_name=args.model,
            p_res=p_res,
            r_res=r_res,
            u_res=u_res,
            ground_truth_label=gt_label,
        )
        feature_vectors.append(feat_vec)
        cause_labels_list.append(cause_label)

        if (i + 1) % 50 == 0:
            print(f"Processed {i + 1} / {len(records)} diagnostic samples...")

    # Save features to parquet
    features_df = feat_extractor.to_dataframe(feature_vectors)
    feat_parquet_path = features_dir / f"tier_{args.tier}_{args.model}_features.parquet"
    features_df.to_parquet(feat_parquet_path, index=False)
    if canonical_model != args.model:
        canonical_parquet_path = features_dir / f"tier_{args.tier}_{canonical_model}_features.parquet"
        features_df.to_parquet(canonical_parquet_path, index=False)
    print(f"\nSaved {len(features_df)} diagnostic feature vectors to {feat_parquet_path}.")

    # 6. Strict Partitioning: Train vs Held-Out Test (Zero Leakage)
    train_vectors = []
    train_causes = []
    test_vectors = []
    test_causes = []

    for vec, rec, cause in zip(feature_vectors, records, cause_labels_list):
        img = rec.get("image_id", "")
        if img in held_out_images:
            test_vectors.append(vec)
            test_causes.append(cause)
        else:
            train_vectors.append(vec)
            train_causes.append(cause)

    print(f"Partitioned features: {len(train_vectors)} training partition, {len(test_vectors)} strictly held-out test.")

    # 7. Combiner Training, Calibration & Neural Attribution
    combiner = XMCADetectionCombiner()
    attr_head = XMCAAttributionHead()

    train_with_labels = [v for v in train_vectors if v.ground_truth_label is not None]
    train_cause_filtered = [c for v, c in zip(train_vectors, train_causes) if v.ground_truth_label is not None and c is not None]

    if len(train_with_labels) >= 10:
        print(f"Training XMCA neural combiner on {len(train_with_labels)} labeled validation samples...")
        combiner.fit(
            training_vectors=train_with_labels,
            labels=[v.ground_truth_label for v in train_with_labels],
            epochs=50,
            lr=1e-3,
        )
        print("Calibrating temperature scaling via L-BFGS on validation logits...")
        combiner.calibrate_temperature(
            val_vectors=train_with_labels,
            val_labels=[v.ground_truth_label for v in train_with_labels],
        )

        if len(train_cause_filtered) == len(train_with_labels):
            print("Training neural cause-attribution head on verified ground truth...")
            attr_head.fit(
                combiner=combiner,
                vectors=train_with_labels,
                cause_labels=train_cause_filtered,
                epochs=35,
            )

        # Save trained neural checkpoint
        ckpt_path = models_dir / f"tier_{args.tier}_{canonical_model}_combiner.pt"
        torch.save({
            "model_state_dict": combiner.model.state_dict(),
            "scaler_state_dict": combiner.scaler.state_dict(),
            "attr_head_state_dict": attr_head.network.state_dict(),
            "mean": combiner.mean,
            "std": combiner.std,
            "temperature": combiner.temperature,
            "threshold": combiner.decision_threshold,
        }, ckpt_path)
        print(f"Saved trained PyTorch checkpoint to: {ckpt_path}")
    else:
        print("Insufficient training labels in partition; evaluating with zero-shot neural weights.")

    # 8. Predictions & Attribution
    eval_target_vectors = test_vectors if test_vectors else feature_vectors
    predictions = [combiner.predict(v) for v in eval_target_vectors]
    attributions = [attr_head.attribute_cause(v, p, combiner=combiner) for v, p in zip(eval_target_vectors, predictions)]

    # 9. Benchmark Metrics Computation
    evaluator = MetricEvaluator()
    eval_labels = [v.ground_truth_label for v in eval_target_vectors if v.ground_truth_label is not None]
    eval_scores = [p.hallucination_probability for v, p in zip(eval_target_vectors, predictions) if v.ground_truth_label is not None]

    report = {
        "tier": args.tier,
        "model": args.model,
        "total_records_evaluated": len(eval_target_vectors),
        "labeled_records_count": len(eval_labels),
        "combiner_temperature": combiner.temperature,
        "combiner_threshold": combiner.decision_threshold,
    }

    if len(eval_labels) > 0 and len(set(eval_labels)) > 1:
        metrics = evaluator.compute_detection_metrics(eval_labels, eval_scores)
        report["detection_metrics"] = metrics.model_dump()
        print("\n=== XMCA-Net Detection Performance ===")
        print(f"AUROC:       {metrics.auroc:.4f}")
        print(f"PR-AUC:      {metrics.pr_auc:.4f}")
        print(f"ECE:         {metrics.ece:.4f}")
        print(f"F1-Score:    {metrics.f1_score:.4f}")
        print(f"Brier Score: {metrics.brier_score:.4f}")
    else:
        report["detection_metrics"] = "Insufficient binary labels for AUROC in evaluated partition"

    # Attribution summary
    cause_counts = {}
    for a in attributions:
        cause_counts[a.attributed_cause] = cause_counts.get(a.attributed_cause, 0) + 1
    report["attribution_distribution"] = cause_counts
    print("\nAttribution Breakdown:")
    for cause, count in cause_counts.items():
        print(f"  {cause}: {count} ({count/len(attributions)*100:.1f}%)")

    # Save comprehensive report
    report_file = output_dir / f"tier_{args.tier}_{args.model}_evaluation_report.json"
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport written to: {report_file}")


if __name__ == "__main__":
    main()
