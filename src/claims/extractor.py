"""Atomic claim extraction module for XMCA-Net.

Decomposes generated VLM answers into atomic verifiable factual claims:
- Object presence claims (e.g. "There is a dog")
- Attribute claims (e.g. "The dog is brown")
- Action claims (e.g. "The dog is running")
- Relationship claims (e.g. "The dog is next to the ball")
- Count claims (e.g. "There are three birds")
- Location claims (e.g. "The car is parked on the street")
- External factual claims (e.g. "Norman architecture originated in Normandy")

Provides structured claim representation and verification state tracking.
"""

import re
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class AtomicClaim(BaseModel):
    """Structured atomic factual claim extracted from a generated answer."""

    claim_id: str
    claim_text: str
    claim_type: str  # 'object_presence' | 'attribute' | 'action' | 'relationship' | 'count' | 'location' | 'external_fact'
    subject: str
    predicate: Optional[str] = None
    object_or_value: Optional[str] = None
    verification_state: str = "unverified"  # 'supported' | 'contradicted' | 'insufficient_evidence' | 'unverifiable' | 'not_applicable'
    confidence: float = 1.0
    evidence_doc_id: Optional[str] = None
    bounding_box: Optional[List[float]] = None  # [ymin, xmin, ymax, xmax] if visually grounded


class ClaimExtractionResult(BaseModel):
    """Collection of atomic claims extracted from a generated answer."""

    sample_id: str
    original_text: str
    claims: List[AtomicClaim] = Field(default_factory=list)
    total_claims: int = 0
    claim_types: Dict[str, int] = Field(default_factory=dict)


class AtomicClaimExtractor:
    """Extracts atomic claims using syntactic and linguistic rule decomposition."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}

    def extract_claims(self, sample_id: str, generated_text: str, question: Optional[str] = None) -> ClaimExtractionResult:
        """Decompose generated answer text into atomic claims."""
        text = generated_text.strip()
        if not text:
            return ClaimExtractionResult(sample_id=sample_id, original_text="", claims=[], total_claims=0)

        # Clean markdown formatting and conversational wrappers
        clean_text = re.sub(r"\*\*|__", "", text)
        sentences = [s.strip() for s in re.split(r"[.!?\n]+", clean_text) if s.strip()]

        claims: List[AtomicClaim] = []
        claim_counter = 0
        binary_vqa_resolved = False

        # Special handling for binary VQA queries (e.g. POPE: "Is there a <object> in the image?")
        # When resolved, ONLY the target binary claim is emitted — general sentence decomposition
        # is deliberately skipped to prevent verbose answers from polluting S_P with unverifiable
        # extra claims (e.g. "golden retriever", "bench", "park") that OWL-ViT cannot detect.
        if question:
            q_lower = question.lower().strip()
            pope_match = re.search(r"is there (?:a |an |any )?([a-zA-Z\s]+?)(?: in the image|\son the\b|\sin the\b|\son\b|\sin\b|\?)", q_lower)
            if pope_match:
                target_obj = pope_match.group(1).strip()
                text_lower = text.lower().strip()
                is_no = bool(re.search(r"\b(?:no|not|none|isn't|is not|aren't|are not)\b", text_lower))
                is_yes = bool(re.search(r"\b(?:yes|yeah|there is|there are|exists)\b", text_lower))
                if is_no:
                    claims.append(
                        AtomicClaim(
                            claim_id=f"{sample_id}_c{claim_counter}",
                            claim_text=f"There is no {target_obj} in the image",
                            claim_type="object_absence",
                            subject=target_obj,
                            predicate="absent",
                            object_or_value=None,
                        )
                    )
                    claim_counter += 1
                    binary_vqa_resolved = True
                elif is_yes or len(sentences) <= 1:
                    claims.append(
                        AtomicClaim(
                            claim_id=f"{sample_id}_c{claim_counter}",
                            claim_text=f"There is a {target_obj} in the image",
                            claim_type="object_presence",
                            subject=target_obj,
                            predicate="exists",
                            object_or_value=None,
                        )
                    )
                    claim_counter += 1
                    binary_vqa_resolved = True

        # General sentence decomposition — skipped when binary VQA is resolved to prevent
        # noise claims from elaborate model responses dragging down S_P
        if not binary_vqa_resolved:
            for sentence in sentences:
                sub_clauses = [c.strip() for c in re.split(r",\s*(?:and|but|while|whereas)\s*|;\s*", sentence) if c.strip()]
                for clause in sub_clauses:
                    extracted = self._decompose_clause(sample_id, claim_counter, clause)
                    for c in extracted:
                        claims.append(c)
                        claim_counter += 1

        # If no clause matched specific patterns, capture the raw sentence as a single claim
        if not claims:
            for s_idx, sentence in enumerate(sentences):
                claims.append(
                    AtomicClaim(
                        claim_id=f"{sample_id}_claim_{s_idx}",
                        claim_text=sentence,
                        claim_type="external_fact",
                        subject=sentence.split()[0] if sentence.split() else "unknown",
                        predicate="states",
                        object_or_value=sentence,
                    )
                )

        type_counts: Dict[str, int] = {}
        for c in claims:
            type_counts[c.claim_type] = type_counts.get(c.claim_type, 0) + 1

        return ClaimExtractionResult(
            sample_id=sample_id,
            original_text=generated_text,
            claims=claims,
            total_claims=len(claims),
            claim_types=type_counts,
        )

    def _decompose_clause(self, sample_id: str, claim_idx: int, clause: str) -> List[AtomicClaim]:
        """Decompose a single clause into structured atomic claims."""
        results: List[AtomicClaim] = []
        clause_lower = clause.lower().strip()

        # 1. Existence / Object presence: "There is/are a [X]"
        exist_match = re.search(r"there (?:is|are|exists|exist) (?:a |an |some |several )?([a-zA-Z\s]+)", clause_lower)
        if exist_match:
            obj = exist_match.group(1).split(" in ")[0].split(" on ")[0].split(" with ")[0].strip()
            results.append(
                AtomicClaim(
                    claim_id=f"{sample_id}_c{claim_idx}",
                    claim_text=f"There is {obj}",
                    claim_type="object_presence",
                    subject=obj,
                    predicate="exists",
                    object_or_value=None,
                )
            )
            return results

        # 2. Count claims: "There are [number/two/three] [X]" or "[number] [X] are [Y]"
        count_match = re.search(r"(?:there (?:is|are) )?(\b\d+\b|one|two|three|four|five|six|seven|eight|nine|ten)\s+([a-zA-Z]+)", clause_lower)
        if count_match:
            count_val = count_match.group(1)
            obj = count_match.group(2)
            results.append(
                AtomicClaim(
                    claim_id=f"{sample_id}_c{claim_idx}",
                    claim_text=f"Count of {obj} is {count_val}",
                    claim_type="count",
                    subject=obj,
                    predicate="has_count",
                    object_or_value=count_val,
                )
            )
            return results

        # 3. Attribute claims: "The [subject] is [attribute]"
        attr_match = re.search(r"(?:the |a |an )?([a-zA-Z]+) (?:is|are|seems|looks) ([a-zA-Z]+)", clause_lower)
        if attr_match:
            subj = attr_match.group(1)
            attr = attr_match.group(2)
            results.append(
                AtomicClaim(
                    claim_id=f"{sample_id}_c{claim_idx}",
                    claim_text=f"{subj} has attribute {attr}",
                    claim_type="attribute",
                    subject=subj,
                    predicate="is",
                    object_or_value=attr,
                )
            )
            return results

        # 4. Location / Spatial relationship: "[subject] is on/in/near/at [location]"
        loc_match = re.search(r"([a-zA-Z\s]+) (?:is|are|located|standing|sitting|parked) (in|on|at|near|by|under|above) ([a-zA-Z\s]+)", clause_lower)
        if loc_match:
            subj = loc_match.group(1).strip()
            prep = loc_match.group(2).strip()
            loc = loc_match.group(3).strip()
            results.append(
                AtomicClaim(
                    claim_id=f"{sample_id}_c{claim_idx}",
                    claim_text=f"{subj} is {prep} {loc}",
                    claim_type="location",
                    subject=subj,
                    predicate=prep,
                    object_or_value=loc,
                )
            )
            return results

        # Fallback to general external factual statement
        results.append(
            AtomicClaim(
                claim_id=f"{sample_id}_c{claim_idx}",
                claim_text=clause,
                claim_type="external_fact",
                subject=clause.split()[0] if clause.split() else "unknown",
                predicate="states",
                object_or_value=clause,
            )
        )
        return results

