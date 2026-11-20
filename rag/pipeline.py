"""The retrieval half of the pipeline, wired up once for every entry point.

`main.py`, `demo.py`, `compare_backends.py`, and the eval harness all need
the same "retrieve, optionally rerank, apply the relevance cutoff" logic.
Keeping it in one place means a fix to that logic (like applying the
cutoff *after* reranking, which the per-script copies used to skip) lands
everywhere at once.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from rag.backends import DEFAULT_MIN_SCORES, build_index
from rag.index import Index, SearchResult
from rag.ingest import chunk_documents, load_documents
from rag.retrieve import Retriever

if TYPE_CHECKING:
    # Type hint only — sentence-transformers stays an optional dependency.
    from rag.rerank import Reranker

DOCS_DIR = Path(__file__).resolve().parent.parent / "data" / "docs"

FINAL_K = 3

# When reranking is on, retrieve a wider candidate pool from the index
# (cheap) so the cross-encoder (expensive) has more than FINAL_K options to
# choose from — reranking a pool the same size as what you'd already show
# the user can only ever reorder it, never surface a better chunk the
# initial retrieval ranked just outside the final top-k.
RERANK_CANDIDATES = 8

# Cutoff on the reranker's sigmoid score, applied after reranking (the
# index's cutoff runs before it, on a different scale). Defaults to 0 —
# off — because the eval showed no safe value for this model: the MS MARCO
# cross-encoder scores some correct answers to paraphrased questions at
# ~0.0001 ("card gets declined" vs. the "failed payment" chunk), while
# some off-corpus questions reach 0.085. Any cutoff that rejects off-corpus
# questions also throws away real answers. Raise it only if
# `python -m evals.retrieval_eval` shows a clean gap for your reranker.
RERANK_MIN_SCORE = 0.0


@dataclass
class RAGPipeline:
    retriever: Retriever
    reranker: Reranker | None = None
    final_k: int = FINAL_K
    candidate_k: int = RERANK_CANDIDATES
    rerank_min_score: float = RERANK_MIN_SCORE

    @property
    def index(self) -> Index:
        return self.retriever.index

    @property
    def label(self) -> str:
        rerank = " + reranking" if self.reranker else ""
        return f"{type(self.index).__name__}{rerank}"

    def retrieve(self, query: str) -> list[SearchResult]:
        """Return up to `final_k` chunks that clear every relevance cutoff."""
        if self.reranker is None:
            return self.retriever.retrieve(query, k=self.final_k)
        candidates = self.retriever.retrieve(query, k=self.candidate_k)
        reranked = self.reranker.rerank(query, candidates, top_k=self.final_k)
        # The reranker's scores are a different scale from the index's, so
        # they need their own cutoff. Without this, an off-corpus question
        # still sends its top chunks (scored ~0.001) to the model.
        return [r for r in reranked if r.score >= self.rerank_min_score]


def build_reranker(enabled: bool | None = None) -> Reranker | None:
    """Build the cross-encoder reranker if enabled (default: `RAG_RERANK=1`)."""
    if enabled is None:
        enabled = os.environ.get("RAG_RERANK") == "1"
    if not enabled:
        return None
    from rag.rerank import Reranker  # lazy: keeps sentence-transformers optional

    return Reranker()


def build_pipeline(
    backend: str | None = None,
    rerank: bool | None = None,
    docs_dir: Path = DOCS_DIR,
) -> RAGPipeline:
    """Build the pipeline from arguments, falling back to `RAG_BACKEND` /
    `RAG_RERANK` env vars, then to TF-IDF with no reranking."""
    backend = backend or os.environ.get("RAG_BACKEND", "tfidf")
    chunks = chunk_documents(load_documents(docs_dir))
    index = build_index(chunks, backend=backend)
    retriever = Retriever(index, min_score=DEFAULT_MIN_SCORES[backend])
    return RAGPipeline(retriever=retriever, reranker=build_reranker(rerank))
