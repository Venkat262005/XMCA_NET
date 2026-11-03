"""Authentic dataset loaders and normalization interfaces for XMCA-Net.

Preserves all original fields while creating a standardized metadata representation
without altering original dataset semantics or inventing missing values.
"""

import os
import json
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class NormalizedRecord(BaseModel):
    """Standardized representation of a benchmark sample."""

    dataset_id: str
    original_sample_id: str
    original_image_id: str
    original_split: str
    question: str
    official_answer_fields: Dict[str, Any]
    official_label_fields: Dict[str, Any]
    source_revision: str
    source_record_hash: str

    # Image metadata fields (populated only when materialized)
    image_path: Optional[str] = None
    image_sha256: Optional[str] = None
    image_download_status: str = "pending" # pending | verified | missing | failed

    # Raw preservation
    raw_record: Dict[str, Any] = Field(default_factory=dict)


def compute_dict_hash(d: Dict[str, Any]) -> str:
    serialized = json.dumps(d, sort_keys=True, default=str)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


class AuthenticDatasetLoader:
    """Loads and standardizes authentic dataset files."""

    def __init__(self, data_raw_dir: Optional[Path] = None):
        self.data_raw_dir = Path(data_raw_dir or "data/raw")

    def load_pope(self, filename: str = "pope_coco_adversarial.json") -> List[NormalizedRecord]:
        file_path = self.data_raw_dir / filename
        if not file_path.exists():
            raise FileNotFoundError(f"Authentic POPE file not found: {file_path}")

        records: List[NormalizedRecord] = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line_str = line.strip()
                if not line_str:
                    continue
                item = json.loads(line_str)
                rec = NormalizedRecord(
                    dataset_id="pope",
                    original_sample_id=str(item["question_id"]),
                    original_image_id=item["image"],
                    original_split="test",
                    question=item["text"],
                    official_answer_fields={"label": item["label"]},
                    official_label_fields={
                        "target_present": item["label"].strip().lower() == "yes",
                        "hallucination_binary": 0 if item["label"].strip().lower() == "yes" else 1,
                    },
                    source_revision="RUCAIBox/POPE@7e24b45",
                    source_record_hash=compute_dict_hash(item),
                    raw_record=item,
                )
                records.append(rec)
        return records

    def load_faithscore(self, filename: str = "faithscore_annotations.jsonl", deduplicate: bool = True) -> List[NormalizedRecord]:
        file_path = self.data_raw_dir / filename
        if not file_path.exists():
            raise FileNotFoundError(f"Authentic FaithScore file not found: {file_path}")

        records: List[NormalizedRecord] = []
        seen_hashes = set()
        with open(file_path, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                line_str = line.strip()
                if not line_str:
                    continue
                item = json.loads(line_str)
                rec_hash = compute_dict_hash(item)
                if deduplicate and rec_hash in seen_hashes:
                    continue
                seen_hashes.add(rec_hash)

                image_filename = f"COCO_val2014_{str(item['id']).zfill(12)}.jpg"
                rec = NormalizedRecord(
                    dataset_id="faithscore",
                    original_sample_id=f"{str(item['id'])}_eval_{len(records):03d}",
                    original_image_id=image_filename,
                    original_split="test",
                    question="Describe the image in detail.",
                    official_answer_fields={"answer": item.get("answer", "")},
                    official_label_fields={
                        "stage1_descriptive": item.get("stage 1", {}),
                        "stage2_atomic_facts": item.get("stage 2", {}),
                    },
                    source_revision="bcdnlp/FAITHSCORE@c2a5fa2",
                    source_record_hash=rec_hash,
                    raw_record=item,
                )
                records.append(rec)
        return records

    def load_aokvqa(self, filename: str = "aokvqa_v1p0_val.json") -> List[NormalizedRecord]:
        file_path = self.data_raw_dir / filename
        if not file_path.exists():
            raise FileNotFoundError(f"Authentic A-OKVQA file not found: {file_path}")

        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        records: List[NormalizedRecord] = []
        for item in data:
            image_filename = f"{int(item['image_id']):012d}.jpg"
            rec = NormalizedRecord(
                dataset_id="a_okvqa",
                original_sample_id=str(item["question_id"]),
                original_image_id=image_filename,
                original_split=item.get("split", "val"),
                question=item["question"],
                official_answer_fields={
                    "choices": item["choices"],
                    "correct_choice_idx": item["correct_choice_idx"],
                    "direct_answers": item.get("direct_answers", []),
                    "rationales": item.get("rationales", []),
                },
                official_label_fields={
                    "correct_choice_idx": item["correct_choice_idx"],
                    "difficult_direct_answer": item.get("difficult_direct_answer", False),
                },
                source_revision="AllenInstitute/aokvqa_v1.0@release",
                source_record_hash=compute_dict_hash(item),
                raw_record=item,
            )
            records.append(rec)
        return records

    def load_infoseek(self, filename: str = "infoseek_val.jsonl", max_records: Optional[int] = None) -> List[NormalizedRecord]:
        file_path = self.data_raw_dir / filename
        if not file_path.exists():
            raise FileNotFoundError(f"Authentic InfoSeek file not found: {file_path}")

        records: List[NormalizedRecord] = []
        count = 0
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line_str = line.strip()
                if not line_str:
                    continue
                item = json.loads(line_str)
                rec = NormalizedRecord(
                    dataset_id="infoseek",
                    original_sample_id=str(item["data_id"]),
                    original_image_id=item["image_id"],
                    original_split=item.get("data_split", "val"),
                    question=item["question"],
                    official_answer_fields={
                        "answer": item.get("answer", []),
                        "answer_eval": item.get("answer_eval", []),
                    },
                    official_label_fields={
                        "answer": item.get("answer", []),
                    },
                    source_revision="GoogleResearch/infoseek_val@emnlp2023",
                    source_record_hash=compute_dict_hash(item),
                    raw_record=item,
                )
                records.append(rec)
                count += 1
                if max_records and count >= max_records:
                    break
        return records

    def load_all(self) -> Dict[str, List[NormalizedRecord]]:
        return {
            "pope": self.load_pope(),
            "faithscore": self.load_faithscore(),
            "a_okvqa": self.load_aokvqa(),
            "infoseek": self.load_infoseek(),
        }


# Backwards compatibility alias
DatasetLoader = AuthenticDatasetLoader
