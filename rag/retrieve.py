"""Stage 3: Retrieval — the query-time layer on top of the index.

Kept separate from TfidfIndex on purpose: the index is a dumb data structure
(vectors in, ranked results out), while the retriever owns query-time policy
— how many results to fetch, what counts as "relevant enough" to bother
showing the model. That's where a reranker, a metadata filter (e.g. "only
docs tagged security"), or a minimum-score cutoff would plug in without
touching the indexing code.
"""

from __future__ import annotations

from rag.index import Index, SearchResult


class Retriever:
    def __init__(self, index: Index, min_score: float = 0.05):
        # min_score guards against feeding the model irrelevant chunks when
        # the corpus genuinely has no good match for the query. The right
        # cutoff is backend-dependent (TF-IDF scores unrelated text at ~0,
        # dense embeddings at ~0.3+), so callers should pass the backend's
        # calibrated value from rag.backends.DEFAULT_MIN_SCORES — the 0.05
        # default here only suits TF-IDF.
        self.index = index
        self.min_score = min_score

    def retrieve(self, query: str, k: int = 3) -> list[SearchResult]:
        results = self.index.search(query, k=k)
        return [r for r in results if r.score >= self.min_score]
