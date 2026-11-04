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
