"""Stage 3.5: Reranking — a second, more accurate pass over retrieved candidates.

Every backend in this repo (TfidfIndex, LocalDenseIndex, VoyageIndex) is a
*bi-encoder*: the query and each chunk are encoded independently into
vectors, and relevance is just the cosine similarity between two vectors
that were never compared while being computed. That's what makes bi-encoder
search fast — chunk vectors are precomputed once, and a query only needs
one new vector to compare against all of them — but it caps how much
signal the model can use: it never sees the query and a candidate chunk
together, so it can't reason about whether the chunk actually *answers* the
query versus merely shares a topic with it.

A *cross-encoder* removes that cap by taking the (query, chunk) pair as
joint input to a transformer, attending across both simultaneously, and
outputting one relevance score per pair. This is strictly more accurate —
and strictly more expensive, since nothing can be precomputed: every
candidate has to be scored at query time, which does not scale to
searching a full corpus. The standard pattern (and the reason this is a
separate stage bolted onto retrieval rather than a replacement for it) is
two-phase: use a cheap bi-encoder/TF-IDF index to pull a wide candidate net
(recall), then use a cross-encoder to rerank just those few candidates down
to the top few that actually go in the prompt (precision).

Measured with `python -m evals.retrieval_eval`: reranking the local
backend's top 8 raises hit@1 from 68% to 88% and hit@3 from 92% to 100%.
It's a strong *ranker*. It is not a reliable *relevance detector*, though:
it scores some correct answers to paraphrased questions near 0 (e.g.
"card gets declined" vs. the chunk about failed payments: ~0.0001) while
some off-corpus questions reach ~0.085, so no score cutoff separates the
two. That's why rag/pipeline.py's post-rerank cutoff defaults to off, and
why the system prompt's "say you don't know" rule still does the real work
on off-corpus questions.
"""

from __future__ import annotations

import numpy as np
from sentence_transformers import CrossEncoder

from rag.index import SearchResult

# A small (~80MB), fast cross-encoder trained on MS MARCO passage ranking —
# the standard baseline reranker, good enough to clearly demonstrate the
# precision gain over bi-encoder cosine similarity without a heavy model.
DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class Reranker:  # pylint: disable=too-few-public-methods
    """Takes retrieval candidates from ANY backend and rescores them by how
    well each one actually answers the query, not just how topically close
    it is. Backend-agnostic on purpose: it only ever sees (query, chunk
    text) pairs, never the vectors that produced the candidate set, so it
    composes with TfidfIndex, LocalDenseIndex, or VoyageIndex identically.
    """

    def __init__(self, model_name: str = DEFAULT_MODEL):
        self._model = CrossEncoder(model_name)

    def rerank(self, query: str, results: list[SearchResult], top_k: int = 3) -> list[SearchResult]:
        """Rescore `results` against `query`; return the best `top_k`."""
        if not results:
            return results
        pairs = [(query, r.chunk.text) for r in results]
        # This model outputs an unbounded regression logit, not a cosine
        # similarity — sigmoid maps it into (0, 1) so `.score` keeps the
        # same "higher = more relevant, roughly comparable across stages"
        # meaning everywhere else in this codebase relies on (e.g.
        # Retriever's min_score cutoff).
        raw_scores = self._model.predict(pairs)
        scores = 1 / (1 + np.exp(-raw_scores))
        reranked = sorted(zip(results, scores, strict=True), key=lambda pair: pair[1], reverse=True)
        return [
            SearchResult(chunk=result.chunk, score=float(score))
            for result, score in reranked[:top_k]
        ]
