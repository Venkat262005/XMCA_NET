"""Unified RAG retriever for XMCA-Net.

Supports BM25 lexical retrieval and dense semantic embeddings.
Enforces partition eligibility (train-only vs held-out) to prevent data leakage.
Explicitly flags POPE and FaithScore as NOT_APPLICABLE.
"""

from typing import List, Dict, Any, Optional, Set, Tuple
from pydantic import BaseModel, Field

from src.retrieval.corpus import AuthenticKnowledgeCorpus, CorpusDocument


class RetrievedPassage(BaseModel):
    """Passage retrieved from the knowledge corpus with diagnostic scores."""

    doc_id: str
    passage_text: str
    relevance_score: float
    rank: int
    source_url: str
    entity_association: Optional[str] = None
    corpus_split_eligibility: str


class RetrievalResult(BaseModel):
    """Full diagnostic result of a retrieval operation."""

    dataset_id: str
    sample_id: str
    is_applicable: bool
    query: str
    retrieved_passages: List[RetrievedPassage] = Field(default_factory=list)
    top_score: float = 0.0
    mean_score: float = 0.0
    score_margin: float = 0.0
    evidence_coverage: float = 0.0
    missingness_mask: float = 0.0  # 1.0 = RAG is NOT applicable, 0.0 = RAG is active


# Non-RAG datasets that must never have synthetic documents or fabricated retrieval
NON_RAG_DATASETS = {"pope", "faithscore"}


class UnifiedRetriever:
    """Unified retrieval engine supporting lexical BM25 and dense similarity."""

    def __init__(self, corpus: Optional[AuthenticKnowledgeCorpus] = None):
        self.corpus = corpus

    def set_corpus(self, corpus: AuthenticKnowledgeCorpus):
        self.corpus = corpus

    def retrieve(
        self,
        dataset_id: str,
        sample_id: str,
        query: str,
        top_k: int = 5,
        allowed_eligibility: Optional[Set[str]] = None,
    ) -> RetrievalResult:
        """Execute retrieval with strict applicability checks."""
        # Rule: POPE and FaithScore are strictly non-RAG
        if dataset_id.lower() in NON_RAG_DATASETS:
            return RetrievalResult(
                dataset_id=dataset_id,
                sample_id=sample_id,
                is_applicable=False,
                query=query,
                retrieved_passages=[],
                top_score=0.0,
                mean_score=0.0,
                score_margin=0.0,
                evidence_coverage=0.0,
                missingness_mask=1.0,  # Missing/inapplicable
            )

        if self.corpus is None or len(self.corpus) == 0:
            return RetrievalResult(
                dataset_id=dataset_id,
                sample_id=sample_id,
                is_applicable=True,
                query=query,
                retrieved_passages=[],
                top_score=0.0,
                mean_score=0.0,
                score_margin=0.0,
                evidence_coverage=0.0,
                missingness_mask=1.0,
            )

        # Execute search in authentic corpus
        ranked_hits = self.corpus.search(query, top_k=top_k, allowed_eligibility=allowed_eligibility)

        if not ranked_hits:
            return RetrievalResult(
                dataset_id=dataset_id,
                sample_id=sample_id,
                is_applicable=True,
                query=query,
                retrieved_passages=[],
                top_score=0.0,
                mean_score=0.0,
                score_margin=0.0,
                evidence_coverage=0.0,
                missingness_mask=0.0,
            )

        passages: List[RetrievedPassage] = []
        scores = []
        for rank, (doc, score) in enumerate(ranked_hits, start=1):
            scores.append(score)
            passages.append(
                RetrievedPassage(
                    doc_id=doc.doc_id,
                    passage_text=doc.passage_text,
                    relevance_score=float(score),
                    rank=rank,
                    source_url=doc.source_url,
                    entity_association=doc.entity_association,
                    corpus_split_eligibility=doc.corpus_split_eligibility,
                )
            )

        top_score = scores[0] if scores else 0.0
        mean_score = sum(scores) / len(scores) if scores else 0.0
        score_margin = (scores[0] - scores[1]) if len(scores) > 1 else top_score

        # Query token coverage in top retrieved passage
        query_words = set(query.lower().split())
        top_words = set(passages[0].passage_text.lower().split()) if passages else set()
        coverage = len(query_words.intersection(top_words)) / max(1, len(query_words))

        return RetrievalResult(
            dataset_id=dataset_id,
            sample_id=sample_id,
            is_applicable=True,
            query=query,
            retrieved_passages=passages,
            top_score=top_score,
            mean_score=mean_score,
            score_margin=score_margin,
            evidence_coverage=coverage,
            missingness_mask=0.0,
        )
