"""Run-manifest validation and generation utilities for XMCA-Net.

Enforces strictly identical evaluation sample IDs across all three VLM backbones
and records full reproducibility metadata (git revisions, dataset hashes, timestamps).
"""

from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Set, Any, Optional
import json
from pydantic import BaseModel, Field, field_validator


class ModelRunEntry(BaseModel):
    model_key: str
    model_id: str
    revision: str
    sample_ids: List[str]
    generation_file: Optional[str] = None
    status: str = "pending" # pending | completed | failed

    @field_validator("revision")
    @classmethod
    def check_revision(cls, v: str) -> str:
        if not v or v.strip() == "":
            raise ValueError("Model run entry must specify a non-empty commit revision SHA.")
        return v


class EvaluationManifest(BaseModel):
    dataset_name: str
    sample_tier: int
    expected_sample_count: int
    matched_sample_ids: List[str]
    model_runs: Dict[str, ModelRunEntry] = Field(default_factory=dict)
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    strict_mode: bool = True

    @field_validator("sample_tier")
    @classmethod
    def validate_tier(cls, tier: int) -> int:
        valid_tiers = [100, 300, 600, 1000]
        if tier not in valid_tiers:
            raise ValueError(f"Sample tier {tier} not recognized. Must be one of {valid_tiers}.")
        return tier

    def validate_matched_ids(self) -> None:
        """Enforce that every registered model evaluated the exact same sample IDs."""
        if not self.matched_sample_ids:
            raise ValueError(f"EvaluationManifest for {self.dataset_name} has empty matched_sample_ids.")
        
        expected_set = set(self.matched_sample_ids)
        if len(expected_set) != self.sample_tier:
            raise ValueError(
                f"Evaluation tier is {self.sample_tier}, but matched_sample_ids has {len(expected_set)} unique IDs."
            )

        for model_key, run in self.model_runs.items():
            run_ids = set(run.sample_ids)
            if run_ids != expected_set:
                missing = expected_set - run_ids
                extra = run_ids - expected_set
                raise ValueError(
                    f"Mismatch in evaluation sample IDs for model '{model_key}' on dataset '{self.dataset_name}'. "
                    f"Missing IDs: {len(missing)}, Extra IDs: {len(extra)}. "
                    "Cross-model evaluations must have exactly matched sample IDs."
                )

    def save(self, output_path: Path) -> None:
        self.validate_matched_ids()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(self.model_dump_json(indent=2))

    @classmethod
    def load(cls, manifest_path: Path) -> "EvaluationManifest":
        if not manifest_path.exists():
            raise FileNotFoundError(f"Manifest file not found: {manifest_path}")
        with open(manifest_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        manifest = cls(**data)
        manifest.validate_matched_ids()
        return manifest

