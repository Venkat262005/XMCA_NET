"""Unit tests for authentic knowledge corpus and unified retriever."""

import pytest
from pathlib import Path
from src.retrieval.corpus import (
    CorpusDocument,
    BM25Index,
    AuthenticKnowledgeCorpus,
    compute_sha256,
)
from src.retrieval.retriever import UnifiedRetriever, NON_RAG_DATASETS


def test_corpus_document_schema():
    doc = CorpusDocument(
        doc_id="test_doc_01",
        source_url="https://example.org/test#01",
        passage_text="The Golden Gate Bridge is an iconic suspension bridge located in San Francisco.",
        passage_boundaries=(0, 77),
        entity_association="Golden Gate Bridge",
        sha256=compute_sha256("The Golden Gate Bridge is an iconic suspension bridge located in San Francisco."),
        corpus_split_eligibility="train_only",
        dataset_origin="a_okvqa",
    )
    assert doc.doc_id == "test_doc_01"
    assert doc.corpus_split_eligibility == "train_only"
    assert len(doc.sha256) == 64


def test_bm25_search_and_split_isolation():
    docs = [
        CorpusDocument(
            doc_id="doc_train",
            source_url="https://example.org/train",
            passage_text="Norman architecture is characterized by round arches and massive stone walls.",
            sha256=compute_sha256("Norman architecture is characterized by round arches and massive stone walls."),
            corpus_split_eligibility="train_only",
            dataset_origin="a_okvqa",
        ),
        CorpusDocument(
            doc_id="doc_eval",
            source_url="https://example.org/eval",
            passage_text="Gothic architecture is characterized by pointed arches and ribbed vaults.",
            sha256=compute_sha256("Gothic architecture is characterized by pointed arches and ribbed vaults."),
            corpus_split_eligibility="eval_held_out",
            dataset_origin="a_okvqa",
        ),
    ]

    index = BM25Index()
    index.build(docs)

    # Search without restriction
    results = index.search("arches architecture", top_k=5)
    assert len(results) == 2

    # Search with split isolation: allowed only 'train_only'
    isolated_results = index.search("arches architecture", top_k=5, allowed_eligibility={"train_only"})
    assert len(isolated_results) == 1
    assert isolated_results[0][0].doc_id == "doc_train"


def test_retriever_strictly_blocks_synthetic_rag_for_pope_and_faithscore():
    corpus = AuthenticKnowledgeCorpus()
    retriever = UnifiedRetriever(corpus)

    # POPE must be flagged NOT_APPLICABLE with missingness_mask = 1.0
    res_pope = retriever.retrieve(
        dataset_id="pope",
        sample_id="pope_sample_01",
        query="Is there a cat in the image?",
    )
    assert res_pope.is_applicable is False
    assert res_pope.missingness_mask == 1.0
    assert len(res_pope.retrieved_passages) == 0

    # FaithScore must be flagged NOT_APPLICABLE with missingness_mask = 1.0
    res_fs = retriever.retrieve(
        dataset_id="faithscore",
        sample_id="fs_sample_01",
        query="Describe the image in detail.",
    )
    assert res_fs.is_applicable is False
    assert res_fs.missingness_mask == 1.0
    assert len(res_fs.retrieved_passages) == 0
