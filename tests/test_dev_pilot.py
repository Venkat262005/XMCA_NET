"""Unit test verifying the development-only baseline pilot runner."""

import os
import pandas as pd
import numpy as np
import pytest

from scripts.run_dev_baseline_pilot import run_development_baseline_pilot


def test_development_baseline_pilot_execution(tmp_path):
    # Create synthetic features parquet for dev cohort
    cohort = pd.read_parquet("artifacts/manifests/cohort_manifest.parquet")
    N = len(cohort)

    data = {
        "sample_id": cohort["original_sample_id"].values,
        "ground_truth_label": np.random.randint(0, 2, size=N),
        "sp_visual_sim": np.random.randn(N),
        "sp_object_iou": np.random.randn(N),
        "sp_color_sim": np.random.randn(N),
        "token_entropy": np.random.randn(N),
        "su_nli_entailment": np.random.randn(N),
        "su_nli_contradiction": np.random.randn(N),
        "sr_bm25_score": np.random.randn(N),
    }
    df = pd.DataFrame(data)
    feat_path = str(tmp_path / "mock_features.parquet")
    df.to_parquet(feat_path)

    outdir = str(tmp_path / "dev_pilot_out")
    ckptdir = str(tmp_path / "ckpts")

    run_development_baseline_pilot(
        features_parquet=feat_path,
        model_name="mock_vlm",
        output_dir=outdir,
        checkpoints_dir=ckptdir,
    )

    report_file = os.path.join(outdir, "dev_pilot_report_mock_vlm.json")
    assert os.path.exists(report_file)
    assert os.path.exists(os.path.join(ckptdir, "halp_mock_vlm_probe.pt"))
    import json
    with open(report_file) as f:
        rep = json.load(f)
    assert rep["baselines_evaluated"]["HalLoc"]["status"] == "STOPPED_MISSING_TOKEN_ANNOTATIONS"
