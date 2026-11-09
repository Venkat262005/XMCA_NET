"""InfoSeek Image Acquisition Utility for XMCA-Net Benchmark.

Downloads the exact physical images for the InfoSeek cohort from the official
gated Hugging Face repository (ychenNLP/oven) using streaming, avoiding large 294 GB downloads.

Prerequisites:
  1. Accept user agreement at https://huggingface.co/datasets/ychenNLP/oven
  2. Set HF_TOKEN environment variable (or pass --token).
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Set

# Ensure project root in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description="InfoSeek Image Downloader")
    parser.add_argument("--tier", type=int, default=300, choices=[100, 300, 600, 1000],
                        help="Benchmark progressive tier")
    parser.add_argument("--cohort_manifest", type=str, default="artifacts/manifests/cohort_manifest.parquet",
                        help="Path to cohort manifest parquet")
    parser.add_argument("--output_dir", type=str, default="data/images/infoseek",
                        help="Target output directory for InfoSeek images")
    parser.add_argument("--token", type=str, default=None,
                        help="Hugging Face access token (defaults to HF_TOKEN env var)")
    return parser.parse_args()


def get_target_image_ids(manifest_path: str, tier: int) -> Set[str]:
    """Extract required image IDs for InfoSeek in the target tier."""
    df = pd.read_parquet(manifest_path)
    tier_col = f"in_tier_{tier}"
    info_df = df[(df["dataset_id"] == "infoseek") & (df[tier_col] == True)]
    target_ids = set(info_df["original_image_id"].unique())
    return target_ids


def main():
    args = parse_args()
    token = args.token or os.environ.get("HF_TOKEN")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    target_ids = get_target_image_ids(args.cohort_manifest, args.tier)
    print("=" * 65)
    print("XMCA-Net InfoSeek Image Acquisition")
    print(f"Target Tier:         {args.tier}")
    print(f"Required Images:     {len(target_ids)} unique OVEN entity images")
    print(f"Output Directory:    {out_dir}")
    print("=" * 65)

    # Check already downloaded images
    existing_ids = set()
    for p in out_dir.glob("oven_*"):
        stem = p.stem
        if stem in target_ids:
            existing_ids.add(stem)

    missing_ids = target_ids - existing_ids
    print(f"Already cached:      {len(existing_ids)} images")
    print(f"Remaining to fetch:  {len(missing_ids)} images")

    if not missing_ids:
        print("\nAll required InfoSeek images are already downloaded and verified on disk!")
        return

    if not token:
        print("\n" + "!" * 65)
        print("ERROR: Hugging Face authentication token is required to access ychenNLP/oven.")
        print("Steps to resolve:")
        print("  1. Visit https://huggingface.co/datasets/ychenNLP/oven and click 'Agree and access'")
        print("  2. Copy your HF access token from https://huggingface.co/settings/tokens")
        print("  3. Set HF_TOKEN environment variable in Kaggle:")
        print("     import os; os.environ['HF_TOKEN'] = 'hf_your_token_here'")
        print("  4. Re-run this script: !python scripts/acquire_infoseek_images.py --tier " + str(args.tier))
        print("!" * 65)
        sys.exit(1)

    print("\nConnecting to ychenNLP/oven in streaming mode via Hugging Face...")
    try:
        from datasets import load_dataset
        ds = load_dataset("ychenNLP/oven", split="train", streaming=True, token=token)
    except Exception as e:
        print(f"Failed to connect to ychenNLP/oven: {e}", file=sys.stderr)
        sys.exit(1)

    saved_count = 0
    print("Streaming and extracting target InfoSeek images...")

    try:
        for item in ds:
            img_id = item.get("image_id")
            if img_id in missing_ids:
                img = item.get("image")
                if img is not None:
                    target_file = out_dir / f"{img_id}.jpg"
                    img.convert("RGB").save(target_file, "JPEG")
                    missing_ids.remove(img_id)
                    saved_count += 1
                    if saved_count % 25 == 0 or len(missing_ids) == 0:
                        print(f"  Downloaded {saved_count} images (Remaining: {len(missing_ids)})...")
                if not missing_ids:
                    break
    except Exception as e:
        print(f"Stream error encountered: {e}", file=sys.stderr)

    print(f"\nInfoSeek acquisition completed. Successfully saved {saved_count} images to {out_dir}.")


if __name__ == "__main__":
    main()

