"""Tests for the eval harness's metrics and for the dataset's labels."""

from __future__ import annotations

from evals.common import (
    EvalItem,
    check_labels,
    first_relevant_rank,
    load_dataset,
    parse_citations,
    percentile,
    reciprocal_rank,
)
from rag.index import SearchResult
from rag.ingest import Chunk, chunk_documents, load_documents
from rag.pipeline import DOCS_DIR

ITEM = EvalItem(
    id="t",
    category="factual",
    question="q",
    expected_sources=("security",),
    evidence=("locked for 30 minutes",),
    reference_answer="",
)


def result(source: str, text: str) -> SearchResult:
    return SearchResult(chunk=Chunk(id=f"{source}#0", source=source, text=text), score=0.5)


def test_relevance_needs_both_source_and_evidence_and_ignores_case_and_whitespace():
    relevant = result("security", "an account is LOCKED  for\n30 minutes.")
    wrong_source = result("pricing", "locked for 30 minutes")
    wrong_chunk = result("security", "2FA is available on all plans")
    results = [wrong_source, wrong_chunk, relevant]
    assert first_relevant_rank(results, ITEM) == 3
    assert reciprocal_rank(results, ITEM) == 1 / 3
    assert reciprocal_rank([wrong_chunk], ITEM) == 0.0


def test_parse_citations_ignores_non_source_brackets():
    text = "It's 30 minutes [security][pricing]. See [the docs](http://x). [answer truncated]"
    assert parse_citations(text) == {"security", "pricing"}


def test_percentile():
    assert percentile([], 50) == 0.0
    assert percentile([5.0], 95) == 5.0
    assert percentile([1.0, 2.0, 3.0, 4.0, 5.0], 50) == 3.0


def test_dataset_loads_with_off_corpus_items():
    items = load_dataset()
    assert len(items) >= 25
    assert any(not i.answerable for i in items)


def test_every_evidence_phrase_matches_a_real_chunk():
    # A typo in dataset.jsonl would otherwise score as a retrieval miss
    # on every backend and look like a model problem.
    chunks = chunk_documents(load_documents(DOCS_DIR))
    assert check_labels(load_dataset(), chunks) == []
