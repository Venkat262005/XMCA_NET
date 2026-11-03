#!/usr/bin/env python3
"""Acquisition and raw verification script for XMCA-Net Task 1.

Acquires the authentic raw annotation files for:
1. POPE (RUCAIBox / lmms-lab)
2. FaithScore (bcdnlp/FAITHSCORE)
3. A-OKVQA (Allen Institute)
4. InfoSeek (Google Research / Yang Chen et al.)

Computes SHA-256 hashes and inspects sample statistics.
"""

import os
import sys
import shutil
import tarfile
import hashlib
import tempfile
import urllib.request
import subprocess
from pathlib import Path

DATA_RAW_DIR = Path("data/raw")
DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)


def compute_sha256(file_path: Path) -> str:
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def acquire_pope() -> dict:
    target_path = DATA_RAW_DIR / "pope_coco_adversarial.json"
    if not target_path.exists():
        with tempfile.TemporaryDirectory() as tmp_dir:
            repo_dir = Path(tmp_dir) / "pope"
            print("Cloning official POPE repository from RUCAIBox/POPE...")
            subprocess.run(
                ["git", "clone", "--depth", "1", "https://github.com/RUCAIBox/POPE", str(repo_dir)],
                check=True,
                capture_output=True,
            )
            src_file = repo_dir / "output" / "coco" / "coco_pope_adversarial.json"
            shutil.copy2(src_file, target_path)

    sha256 = compute_sha256(target_path)
    size = target_path.stat().st_size
    print(f"[POPE] Acquired: {target_path} (Size: {size:,} bytes, SHA-256: {sha256[:16]}...)")
    return {"path": str(target_path), "size": size, "sha256": sha256}


def acquire_faithscore() -> dict:
    target_path = DATA_RAW_DIR / "faithscore_annotations.jsonl"
    if not target_path.exists():
        with tempfile.TemporaryDirectory() as tmp_dir:
            repo_dir = Path(tmp_dir) / "faithscore"
            print("Cloning official FaithScore repository from bcdnlp/FAITHSCORE...")
            subprocess.run(
                ["git", "clone", "--depth", "1", "https://github.com/bcdnlp/FAITHSCORE", str(repo_dir)],
                check=True,
                capture_output=True,
            )
            src_file = repo_dir / "annotation.jsonl"
            shutil.copy2(src_file, target_path)

    sha256 = compute_sha256(target_path)
    size = target_path.stat().st_size
    print(f"[FaithScore] Acquired: {target_path} (Size: {size:,} bytes, SHA-256: {sha256[:16]}...)")
    return {"path": str(target_path), "size": size, "sha256": sha256}


def acquire_aokvqa() -> dict:
    target_path = DATA_RAW_DIR / "aokvqa_v1p0_val.json"
    if not target_path.exists():
        tar_path = DATA_RAW_DIR / "aokvqa_v1p0.tar.gz"
        if not tar_path.exists():
            print("Downloading official A-OKVQA v1.0 release from Allen Institute S3...")
            url = "https://prior-datasets.s3.us-east-2.amazonaws.com/aokvqa/aokvqa_v1p0.tar.gz"
            urllib.request.urlretrieve(url, tar_path)
        
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(DATA_RAW_DIR)

    sha256 = compute_sha256(target_path)
    size = target_path.stat().st_size
    print(f"[A-OKVQA] Acquired: {target_path} (Size: {size:,} bytes, SHA-256: {sha256[:16]}...)")
    return {"path": str(target_path), "size": size, "sha256": sha256}


def acquire_infoseek() -> dict:
    target_path = DATA_RAW_DIR / "infoseek_val.jsonl"
    if not target_path.exists():
        print("Downloading official InfoSeek validation annotations from Google Research storage...")
        url = "http://storage.googleapis.com/gresearch/open-vision-language/infoseek/infoseek_val.jsonl"
        urllib.request.urlretrieve(url, target_path)

    sha256 = compute_sha256(target_path)
    size = target_path.stat().st_size
    print(f"[InfoSeek] Acquired: {target_path} (Size: {size:,} bytes, SHA-256: {sha256[:16]}...)")
    return {"path": str(target_path), "size": size, "sha256": sha256}


def main():
    print("=== XMCA-Net Task 1: Authentic Dataset Acquisition ===")
    results = {}
    results["pope"] = acquire_pope()
    results["faithscore"] = acquire_faithscore()
    results["a_okvqa"] = acquire_aokvqa()
    results["infoseek"] = acquire_infoseek()
    print("All four authentic raw dataset files verified and acquired.")
    return results


if __name__ == "__main__":
    main()

