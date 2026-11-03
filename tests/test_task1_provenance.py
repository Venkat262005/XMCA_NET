"""Mandatory provenance, integrity, and authenticity tests for XMCA-Net Task 1.

Validates the actual downloaded source records, image manifests, checksums,
split preservation, and strict non-fabrication contracts.
"""

import json
import hashlib
from pathlib import Path
import pytest
import pandas as pd

from src.data.loader import AuthenticDatasetLoader, NormalizedRecord
from src.utils.config import load_datasets_config


@pytest.fixture(scope="module")
def raw_loader():
    return AuthenticDatasetLoader(data_raw_dir=Path("data/raw"))


@pytest.fixture(scope="module")
def manifests():
    manifest_dir = Path("artifacts/manifests")
    return {
        "source": json.loads((manifest_dir / "source_manifest.json").read_text(encoding="utf-8")),
        "integrity": json.loads((manifest_dir / "data_integrity_report.json").read_text(encoding="utf-8")),
        "split": json.loads((manifest_dir / "split_compatibility_report.json").read_text(encoding="utf-8")),
        "failures": json.loads((manifest_dir / "download_failures.json").read_text(encoding="utf-8")),
        "cohort": pd.read_parquet(manifest_dir / "candidate_cohort_manifest.parquet"),
        "inventory": pd.read_parquet(manifest_dir / "dataset_inventory.parquet"),
        "images": pd.read_parquet(manifest_dir / "image_manifest.parquet"),
    }


def test_authentic_source_id_verification(raw_loader):
    """Verify authentic source IDs from original releases."""
    pope = raw_loader.load_pope()
    assert len(pope) == 3000
    assert pope[0].original_sample_id.isdigit()
    assert pope[0].original_image_id.startswith("COCO_val2014_")

    faith = raw_loader.load_faithscore()
    assert len(faith) == 174
    assert faith[0].original_sample_id.startswith("000000")
    assert faith[0].original_image_id.startswith("COCO_val2014_")

    aokvqa = raw_loader.load_aokvqa()
    assert len(aokvqa) == 1145
    assert len(aokvqa[0].original_sample_id) > 0
    assert aokvqa[0].original_image_id.endswith(".jpg")

    infoseek = raw_loader.load_infoseek(max_records=100)
    assert len(infoseek) == 100
    assert infoseek[0].original_sample_id.startswith("infoseek_val_")
    assert infoseek[0].original_image_id.startswith("oven_")


def test_sha256_verification(manifests):
    """Verify SHA-256 hashes of all 4 downloaded raw files against configuration and source manifest."""
    cfg = load_datasets_config()
    for ds_key, ds_meta in cfg.datasets.items():
        expected_hash = ds_meta.source.expected_sha256
        assert expected_hash is not None
        actual_file = Path(ds_meta.source.data_file)
        assert actual_file.exists()

        hasher = hashlib.sha256()
        with open(actual_file, "rb") as f:
            while chunk := f.read(65536):
                hasher.update(chunk)
        calculated_hash = hasher.hexdigest()
        assert calculated_hash == expected_hash
        assert manifests["source"][ds_key]["sha256"] == expected_hash


def test_original_split_preservation(raw_loader, manifests):
    """Verify official train/val/test split membership is preserved without leakage."""
    for r in raw_loader.load_pope():
        assert r.original_split == "test"

    for r in raw_loader.load_faithscore():
        assert r.original_split == "test"

    for r in raw_loader.load_aokvqa():
        assert r.original_split == "val"

    for r in raw_loader.load_infoseek(max_records=100):
        assert "val" in r.original_split.lower()

    # In candidate cohort manifest
    df_cohort = manifests["cohort"]
    assert set(df_cohort[df_cohort["dataset_id"] == "pope"]["original_split"]) == {"test"}
    assert set(df_cohort[df_cohort["dataset_id"] == "faithscore"]["original_split"]) == {"test"}
    assert set(df_cohort[df_cohort["dataset_id"] == "a_okvqa"]["original_split"]) == {"val"}


def test_sample_count_verification(manifests):
    """Verify sample counts match verified pools and documented limits."""
    df_cohort = manifests["cohort"]
    pope_count = len(df_cohort[df_cohort["dataset_id"] == "pope"])
    faith_count = len(df_cohort[df_cohort["dataset_id"] == "faithscore"])
    aokvqa_count = len(df_cohort[df_cohort["dataset_id"] == "a_okvqa"])
    infoseek_count = len(df_cohort[df_cohort["dataset_id"] == "infoseek"])

    assert pope_count == 1000
    assert faith_count == 174 # Verified official ceiling (deduplicated from 180 total lines)
    assert aokvqa_count == 1000
    assert infoseek_count == 1000

    # Progressive tiers
    assert len(df_cohort[(df_cohort["dataset_id"] == "pope") & (df_cohort["in_tier_100"])]) == 100
    assert len(df_cohort[(df_cohort["dataset_id"] == "faithscore") & (df_cohort["in_tier_100"])]) == 100
    assert len(df_cohort[(df_cohort["dataset_id"] == "a_okvqa") & (df_cohort["in_tier_100"])]) == 100
    assert len(df_cohort[(df_cohort["dataset_id"] == "infoseek") & (df_cohort["in_tier_100"])]) == 100


def test_duplicate_detection(manifests):
    """Verify zero duplicate sample IDs exist in cohort and inventory."""
    df_cohort = manifests["cohort"]
    for ds_id in ["pope", "faithscore", "a_okvqa", "infoseek"]:
        ds_subset = df_cohort[df_cohort["dataset_id"] == ds_id]
        assert ds_subset["original_sample_id"].nunique() == len(ds_subset)
        assert ds_subset["source_record_hash"].nunique() == len(ds_subset)


def test_no_synthetic_data_acceptance(manifests):
    """Assert zero synthetic or fabricated labels entered the project."""
    integrity = manifests["integrity"]
    assert integrity["synthetic_records_allowed"] is False
    assert integrity["synthetic_records_found"] == 0
    assert integrity["duplicate_sample_ids_detected"] == 0


def test_preflight_image_verification(manifests):
    """Verify materialized preflight images and failure logging."""
    df_images = manifests["images"]
    assert len(df_images) == 100 # 25 per dataset

    # Check verified COCO images for POPE, FaithScore, A-OKVQA
    verified_pope = df_images[(df_images["dataset_id"] == "pope") & (df_images["status"] == "verified")]
    verified_faith = df_images[(df_images["dataset_id"] == "faithscore") & (df_images["status"] == "verified")]
    verified_aokvqa = df_images[(df_images["dataset_id"] == "a_okvqa") & (df_images["status"] == "verified")]

    assert len(verified_pope) == 25
    assert len(verified_faith) == 25
    assert len(verified_aokvqa) == 25

    for row in verified_pope.itertuples():
        img_path = Path(row.path)
        assert img_path.exists()
        assert row.size_bytes > 0
        assert row.width > 0
        assert row.height > 0
        assert len(row.sha256) == 64


def test_failed_download_reporting(manifests):
    """Verify failed/restricted image downloads are recorded without silent fallbacks."""
    failures = manifests["failures"]
    assert len(failures) == 25 # InfoSeek gated images recorded
    for failure in failures:
        assert failure["dataset_id"] == "infoseek"
        assert failure["status"] == "access_restricted"
        assert "Hugging Face" in failure["error"]


def test_cross_dataset_image_overlap_analysis(manifests):
    """Verify physical image overlap across COCO-derived cohorts is audited."""
    split_rep = manifests["split"]
    assert "pope_faithscore_shared_coco_images" in split_rep
    assert "pope_aokvqa_shared_coco_images" in split_rep
    assert "leakage_risk_assessment" in split_rep
    assert split_rep["pope_aokvqa_shared_coco_images"] >= 0
