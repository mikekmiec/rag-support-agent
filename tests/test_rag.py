"""Unit tests for the pipeline. No network, no models: fakes stand in for
the index, reranker, and Claude client."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from rag.generate import REFUSAL_MESSAGE, build_user_prompt, generate_answer
from rag.index import SearchResult, TfidfIndex
from rag.ingest import Chunk, Document, chunk_documents
from rag.pipeline import RAGPipeline
from rag.retrieve import Retriever


def make_result(chunk_id: str, score: float, source: str = "doc") -> SearchResult:
    return SearchResult(chunk=Chunk(id=chunk_id, source=source, text=chunk_id), score=score)


# --- ingest ---------------------------------------------------------------


def test_chunks_overlap_by_requested_word_count():
    doc = Document(source="d", text=" ".join(str(i) for i in range(10)))
    chunks = chunk_documents([doc], chunk_size=4, overlap=1)
    assert [c.text for c in chunks] == ["0 1 2 3", "3 4 5 6", "6 7 8 9"]
    assert [c.id for c in chunks] == ["d#0", "d#1", "d#2"]


@pytest.mark.parametrize("chunk_size,overlap", [(4, 4), (4, 5), (0, 0), (4, -1)])
def test_invalid_chunking_params_raise_instead_of_looping(chunk_size, overlap):
    doc = Document(source="d", text="a b c d e f")
    with pytest.raises(ValueError):
        chunk_documents([doc], chunk_size=chunk_size, overlap=overlap)


# --- retrieval + reranking --------------------------------------------------


class FakeIndex:
    def __init__(self, results: list[SearchResult]):
        self.chunks = [r.chunk for r in results]
        self._results = results
        self.last_k: int | None = None

    def search(self, query: str, k: int = 3) -> list[SearchResult]:
        self.last_k = k
        return self._results[:k]


class FakeReranker:
    """Rescores by a fixed table and sorts, like the real cross-encoder."""

    def __init__(self, scores: dict[str, float]):
        self.scores = scores

    def rerank(self, query, results, top_k=3):
        rescored = [SearchResult(r.chunk, self.scores[r.chunk.id]) for r in results]
        return sorted(rescored, key=lambda r: r.score, reverse=True)[:top_k]


def test_retriever_drops_results_below_min_score():
    index = FakeIndex([make_result("a", 0.9), make_result("b", 0.2)])
    assert [r.chunk.id for r in Retriever(index, min_score=0.5).retrieve("q")] == ["a"]


def test_pipeline_without_reranker_returns_final_k():
    index = FakeIndex([make_result(c, 0.9) for c in "abcde"])
    pipeline = RAGPipeline(retriever=Retriever(index, min_score=0.0), final_k=3)
    assert len(pipeline.retrieve("q")) == 3
    assert index.last_k == 3


def test_pipeline_reranks_wide_pool_down_to_final_k():
    index = FakeIndex([make_result(c, 0.9) for c in "abcde"])
    reranker = FakeReranker({"a": 0.1, "b": 0.2, "c": 0.3, "d": 0.4, "e": 0.99})
    pipeline = RAGPipeline(
        retriever=Retriever(index, min_score=0.0),
        reranker=reranker,
        final_k=2,
        candidate_k=5,
        rerank_min_score=0.0,
    )
    assert [r.chunk.id for r in pipeline.retrieve("q")] == ["e", "d"]
    assert index.last_k == 5  # fetched the wide pool, not just final_k


def test_pipeline_applies_cutoff_after_reranking():
    # Regression test: the first-stage scores all clear the index cutoff,
    # but the reranker scores everything as irrelevant. Before the fix,
    # all of these went to the model anyway.
    index = FakeIndex([make_result(c, 0.9) for c in "abc"])
    reranker = FakeReranker({"a": 0.001, "b": 0.002, "c": 0.6})
    pipeline = RAGPipeline(
        retriever=Retriever(index, min_score=0.0),
        reranker=reranker,
        rerank_min_score=0.05,
    )
    assert [r.chunk.id for r in pipeline.retrieve("q")] == ["c"]


def test_tfidf_index_ranks_keyword_match_first():
    chunks = [
        Chunk(id="x#0", source="x", text="accounts lock after failed login attempts"),
        Chunk(id="y#0", source="y", text="annual billing discount for team plans"),
    ]
    results = TfidfIndex(chunks).search("annual discount", k=2)
    assert results[0].chunk.id == "y#0"
    assert results[0].score > results[1].score


# --- generation -------------------------------------------------------------


def fake_client(stop_reason: str, content: list) -> SimpleNamespace:
    response = SimpleNamespace(
        stop_reason=stop_reason,
        content=content,
        model="claude-opus-5-5",
        usage=SimpleNamespace(input_tokens=100, output_tokens=20),
    )
    create = lambda **kwargs: response  # noqa: E731
    return SimpleNamespace(beta=SimpleNamespace(messages=SimpleNamespace(create=create)))


def test_generate_joins_text_blocks_and_skips_others():
    client = fake_client(
        "end_turn",
        [
            SimpleNamespace(type="thinking", thinking=""),
            SimpleNamespace(type="text", text="Locked for 30 minutes "),
            SimpleNamespace(type="text", text="[security]"),
        ],
    )
    answer = generate_answer(client, "q", [])
    assert answer.text == "Locked for 30 minutes [security]"
    assert answer.input_tokens == 100 and not answer.refused


def test_generate_handles_refusal_with_no_text_block():
    # Before the fix this raised StopIteration.
    answer = generate_answer(fake_client("refusal", []), "q", [])
    assert answer.refused
    assert answer.text == REFUSAL_MESSAGE


def test_generate_flags_truncated_answers():
    client = fake_client("max_tokens", [SimpleNamespace(type="text", text="Partial")])
    assert "truncated" in generate_answer(client, "q", []).text


def test_prompt_labels_each_chunk_with_its_source():
    prompt = build_user_prompt("q?", [make_result("p#0", 0.5, source="pricing")])
    assert '<excerpt source="pricing">\np#0\n</excerpt>' in prompt
    assert prompt.endswith("Question: q?")
    assert "no relevant context" in build_user_prompt("q?", [])
