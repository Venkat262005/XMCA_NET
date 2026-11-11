"""Unit tests for atomic claim extraction module."""

import pytest
from src.claims.extractor import AtomicClaimExtractor, ClaimExtractionResult


def test_claim_extractor_object_presence():
    extractor = AtomicClaimExtractor()
    text = "There is a dog sitting on the grass. There is a red car."
    res = extractor.extract_claims("sample_01", text)

    assert res.total_claims >= 2
    types = [c.claim_type for c in res.claims]
    assert "object_presence" in types or "location" in types


def test_claim_extractor_attribute_and_count():
    extractor = AtomicClaimExtractor()
    text = "The dog is brown. There are two birds in the sky."
    res = extractor.extract_claims("sample_02", text)

    assert res.total_claims >= 2
    claim_texts = [c.claim_text for c in res.claims]
    assert any("brown" in t for t in claim_texts)
    assert any("two" in t or "birds" in t for t in claim_texts)


def test_claim_extractor_empty_string():
    extractor = AtomicClaimExtractor()
    res = extractor.extract_claims("sample_03", "")
    assert res.total_claims == 0
    assert len(res.claims) == 0


def test_claim_extractor_binary_pope_vqa():
    extractor = AtomicClaimExtractor()
    # Question asks about a dog, answer says Yes
    res_yes = extractor.extract_claims("pope_1", "Yes, there is a dog.", question="Is there a dog in the image?")
    assert res_yes.total_claims >= 1
    presence_claims = [c for c in res_yes.claims if c.claim_type == "object_presence"]
    assert len(presence_claims) >= 1
    assert presence_claims[0].subject == "dog"

    # Question asks about a car, answer says No
    res_no = extractor.extract_claims("pope_2", "No.", question="Is there a car in the image?")
    assert res_no.total_claims == 1
    absence_claims = [c for c in res_no.claims if c.claim_type == "object_absence"]
    assert len(absence_claims) == 1
    assert absence_claims[0].subject == "car"

    # Verbose POPE answer: should ONLY extract the binary absence claim, ignoring filler
    res_verbose = extractor.extract_claims(
        "pope_3",
        "No, there is no car in the image. Instead, there is a large wooden dining table with a chair.",
        question="Is there a car in the image?",
    )
    assert res_verbose.total_claims == 1
    assert res_verbose.claims[0].claim_type == "object_absence"
    assert res_verbose.claims[0].subject == "car"

