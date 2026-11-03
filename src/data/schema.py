"""Dataset sample schemas and strict validation models for XMCA-Net.

Covers the four research datasets: POPE, FaithScore, A-OKVQA, InfoSeek.
Strictly disallows synthetic or fabricated data fields.
"""

from typing import List, Optional, Union, Dict, Any
from pydantic import BaseModel, Field, field_validator


class POPESample(BaseModel):
    sample_id: str = Field(..., min_length=1)
    image_id: str = Field(..., min_length=1)
    image_path: str = Field(..., min_length=1)
    question: str = Field(..., min_length=1)
    ground_truth_label: str = Field(...) # "yes" or "no"
    category: str = Field(..., min_length=1)

    @field_validator("ground_truth_label")
    @classmethod
    def validate_gt(cls, v: str) -> str:
        clean = v.strip().lower()
        if clean not in {"yes", "no"}:
            raise ValueError(f"POPE ground_truth_label must be 'yes' or 'no', got '{v}'")
        return clean


class FaithScoreAtomicFact(BaseModel):
    fact_id: Optional[str] = None
    statement: str = Field(..., min_length=1)
    is_hallucinated: bool


class FaithScoreSample(BaseModel):
    sample_id: str = Field(..., min_length=1)
    image_id: str = Field(..., min_length=1)
    image_path: str = Field(..., min_length=1)
    question: str = Field(..., min_length=1)
    atomic_facts: List[Union[FaithScoreAtomicFact, str]] = Field(..., min_length=1)
    ground_truth_label: Union[int, str, bool]


class AOKVQASample(BaseModel):
    sample_id: str = Field(..., min_length=1)
    image_id: str = Field(..., min_length=1)
    image_path: str = Field(..., min_length=1)
    question: str = Field(..., min_length=1)
    choices: List[str] = Field(..., min_length=2)
    correct_choice_idx: int = Field(..., ge=0)
    rationales: List[str] = Field(default_factory=list)

    @field_validator("correct_choice_idx")
    @classmethod
    def validate_idx(cls, v: int, info) -> int:
        choices = info.data.get("choices")
        if choices and v >= len(choices):
            raise ValueError(f"correct_choice_idx {v} is out of bounds for choices of length {len(choices)}")
        return v


class InfoSeekSample(BaseModel):
    sample_id: str = Field(..., min_length=1)
    image_id: str = Field(..., min_length=1)
    image_path: str = Field(..., min_length=1)
    question: str = Field(..., min_length=1)
    entity_id: str = Field(..., min_length=1)
    golden_answers: List[str] = Field(..., min_length=1)


SCHEMA_MAP: Dict[str, Any] = {
    "pope": POPESample,
    "faithscore": FaithScoreSample,
    "a_okvqa": AOKVQASample,
    "infoseek": InfoSeekSample,
}


def validate_sample_schema(dataset_key: str, data: Dict[str, Any]) -> BaseModel:
    """Validate a single raw sample dictionary against the dataset's strict schema."""
    key = dataset_key.lower().replace("-", "_")
    if key not in SCHEMA_MAP:
        raise KeyError(
            f"Unknown dataset key '{dataset_key}'. Must be one of: {list(SCHEMA_MAP.keys())}"
        )
    return SCHEMA_MAP[key](**data)
