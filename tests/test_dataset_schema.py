"""Tests for dataset schema validation across POPE, FaithScore, A-OKVQA, InfoSeek."""

import pytest
from pydantic import ValidationError

from src.data.schema import (
    POPESample,
    FaithScoreSample,
    AOKVQASample,
    InfoSeekSample,
    validate_sample_schema,
)


def test_valid_pope_sample():
    sample = {
        "sample_id": "pope_001",
        "image_id": "coco_0001",
        "image_path": "data/images/coco/0001.jpg",
        "question": "Is there a dog in the image?",
        "ground_truth_label": "yes",
        "category": "adversarial",
    }
    validated = validate_sample_schema("pope", sample)
    assert isinstance(validated, POPESample)
    assert validated.ground_truth_label == "yes"


def test_invalid_pope_label():
    sample = {
        "sample_id": "pope_001",
        "image_id": "coco_0001",
        "image_path": "data/images/coco/0001.jpg",
        "question": "Is there a dog in the image?",
        "ground_truth_label": "maybe", # Must be 'yes' or 'no'
        "category": "adversarial",
    }
    with pytest.raises(ValidationError, match="must be 'yes' or 'no'"):
        validate_sample_schema("pope", sample)


def test_valid_faithscore_sample():
    sample = {
        "sample_id": "faith_001",
        "image_id": "coco_0002",
        "image_path": "data/images/coco/0002.jpg",
        "question": "Describe the scene.",
        "atomic_facts": [
            {"statement": "A red car is parked on the street.", "is_hallucinated": False},
            {"statement": "A cat is sitting on the hood.", "is_hallucinated": True},
        ],
        "ground_truth_label": 0,
    }
    validated = validate_sample_schema("faithscore", sample)
    assert isinstance(validated, FaithScoreSample)
    assert len(validated.atomic_facts) == 2


def test_valid_a_okvqa_sample():
    sample = {
        "sample_id": "aokvqa_001",
        "image_id": "coco_0003",
        "image_path": "data/images/coco/0003.jpg",
        "question": "What kind of sporting event is this?",
        "choices": ["baseball", "soccer", "tennis", "basketball"],
        "correct_choice_idx": 1,
        "rationales": ["The players are kicking a black-and-white ball towards a net."],
    }
    validated = validate_sample_schema("a_okvqa", sample)
    assert isinstance(validated, AOKVQASample)
    assert validated.choices[validated.correct_choice_idx] == "soccer"


def test_invalid_a_okvqa_choice_idx():
    sample = {
        "sample_id": "aokvqa_001",
        "image_id": "coco_0003",
        "image_path": "data/images/coco/0003.jpg",
        "question": "What kind of sporting event is this?",
        "choices": ["baseball", "soccer"],
        "correct_choice_idx": 5, # Out of bounds
        "rationales": [],
    }
    with pytest.raises(ValidationError, match="out of bounds"):
        validate_sample_schema("a_okvqa", sample)


def test_valid_infoseek_sample():
    sample = {
        "sample_id": "infoseek_001",
        "image_id": "oven_0004",
        "image_path": "data/images/oven/0004.jpg",
        "question": "Who designed this architectural landmark?",
        "entity_id": "Q12345",
        "golden_answers": ["Frank Lloyd Wright", "F. L. Wright"],
    }
    validated = validate_sample_schema("infoseek", sample)
    assert isinstance(validated, InfoSeekSample)
    assert validated.entity_id == "Q12345"


def test_unknown_dataset_schema():
    with pytest.raises(KeyError, match="Unknown dataset key"):
        validate_sample_schema("unsupported_dataset", {})
