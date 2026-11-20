"""Stage 2: Indexing — turn chunks into vectors you can search over.

This uses TF-IDF (term frequency-inverse document frequency) instead of a
neural embedding model. TF-IDF represents each chunk as a sparse vector over
the vocabulary: a word's weight in the vector is high if it appears often in
that chunk (term frequency) but rarely across the whole corpus (inverse
document frequency) — so common words like "the" or "storage" (if every doc
mentions storage) get down-weighted, while distinctive words like "lockout"
or "failover" get up-weighted. It's a bag-of-words model: word order and
meaning are ignored, so it can't tell "dog bites man" from "man bites dog",
and it will miss a query like "How long until my data is gone?" matching a
chunk about "purge" (no shared vocabulary at all — that's what neural
embeddings are for). What it's good at: fast, free, no external API call,
fully interpretable (you can inspect exactly which words drove a match),
and a strong baseline for keyword-heavy queries. This is also the "sparse
retrieval" half of what production systems call hybrid search, typically
paired with a dense embedding model to catch semantic matches TF-IDF misses.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from rag.ingest import Chunk


@dataclass(frozen=True)
class SearchResult:
    chunk: Chunk
    score: float  # cosine similarity in [0, 1], higher = more relevant


class Index(Protocol):
    """The shape every index backend must have: hold chunks, answer search().

    This is a `typing.Protocol` (structural typing), not an abstract base
    class — `TfidfIndex`, `LocalDenseIndex`, and `VoyageIndex` don't inherit
    from it or from each other. Each just happens to have a `chunks`
    attribute and a matching `search()` method, which is enough for
    `Retriever` (retrieve.py) to work with any of them interchangeably. This
    is the concrete mechanism behind the README's "swap TF-IDF for a real
    vector DB without touching retrieval code" claim: three encoding
    strategies (sparse keyword, local dense, hosted dense) share one
    interface, so `generate.py` and `retrieve.py` never need to know which
    one is plugged in.
    """

    chunks: list[Chunk]

    def search(self, query: str, k: int = 3) -> list[SearchResult]: ...


class TfidfIndex:
    """An in-memory vector index. A real deployment would swap this class
    for a vector database (pgvector, Pinecone, Weaviate, ...) so the index
    persists, scales past memory, and supports incremental updates — but the
    interface (fit chunks once, then `search(query, k)`) stays the same.
    """

    def __init__(self, chunks: list[Chunk]):
        if not chunks:
            raise ValueError("Cannot build an index with zero chunks")
        self.chunks = chunks
        self.vectorizer = TfidfVectorizer(stop_words="english")
        # fit_transform learns the corpus vocabulary/IDF weights AND
        # vectorizes every chunk in one pass — this is the "build the index"
        # step, done once up front.
        self.matrix = self.vectorizer.fit_transform([c.text for c in chunks])

    def search(self, query: str, k: int = 3) -> list[SearchResult]:
        """Vectorize the query with the SAME fitted vocabulary, then rank
        every chunk by cosine similarity to the query vector."""
        query_vector = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vector, self.matrix)[0]
        top_indices = scores.argsort()[::-1][:k]
        return [SearchResult(chunk=self.chunks[i], score=float(scores[i])) for i in top_indices]
