"""Build authentic knowledge corpus for XMCA-Net RAG evaluation.

Extracts authentic evidence documents from:
- A-OKVQA train rationales (corpus_split_eligibility: train_only)
- A-OKVQA val rationales (corpus_split_eligibility: eval_held_out)
- InfoSeek entity records (corpus_split_eligibility: eval_held_out)

Saves authentic_knowledge_corpus.parquet and corpus_provenance_report.json.
"""

import os
import sys
import json
from pathlib import Path

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.retrieval.corpus import CorpusBuilder, compute_sha256


def main():
    raw_dir = Path("data/raw")
    output_dir = Path("artifacts/corpora")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Building authentic knowledge corpus from raw dataset files...")
    corpus = CorpusBuilder.build_from_raw_datasets(
        aokvqa_train_path=raw_dir / "aokvqa_v1p0_train.json",
        aokvqa_val_path=raw_dir / "aokvqa_v1p0_val.json",
        infoseek_val_path=raw_dir / "infoseek_val.jsonl",
        max_infoseek_records=None,
    )

    parquet_path = output_dir / "authentic_knowledge_corpus.parquet"
    print(f"Saving {len(corpus)} authentic documents to {parquet_path}...")
    corpus.save_parquet(parquet_path)

    # Compute provenance metadata
    split_counts = {}
    origin_counts = {}
    for doc in corpus.documents:
        split_counts[doc.corpus_split_eligibility] = split_counts.get(doc.corpus_split_eligibility, 0) + 1
        origin_counts[doc.dataset_origin] = origin_counts.get(doc.dataset_origin, 0) + 1

    report = {
        "total_documents": len(corpus),
        "split_counts": split_counts,
        "origin_counts": origin_counts,
        "parquet_path": str(parquet_path),
        "non_rag_datasets": ["pope", "faithscore"],
        "leakage_protection": {
            "train_only_isolation_enforced": True,
            "held_out_validation_isolation_enforced": True,
        },
    }

    report_path = output_dir / "corpus_provenance_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("Corpus build complete:")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

