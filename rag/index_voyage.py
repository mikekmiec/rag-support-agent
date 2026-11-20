"""Alternate indexing backend: Voyage AI's hosted embedding API.

Anthropic doesn't ship its own embedding endpoint — Voyage AI is the
embedding provider Anthropic recommends pairing with Claude for RAG. Unlike
LocalDenseIndex (index_dense.py), no model runs on your machine: every
`embed()` call is a network request to Voyage's API, billed per token, using
a much larger/more capable model than anything practical to run locally.

Two details that only apply to a hosted retrieval-tuned API like this one:

1. `input_type`. Voyage's embedding models are trained asymmetrically — a
   document chunk and a question about it aren't phrased the same way, so
   the model embeds them slightly differently depending on which role you
   tell it. We pass `input_type="document"` when indexing chunks and
   `input_type="query"` when embedding the search query. Get this backwards
   (or leave it unset) and retrieval quality degrades — this is a common
   real-world bug with hosted retrieval embedding APIs, not just a Voyage
   quirk. TF-IDF and the local sentence-transformers model have no
   equivalent concept: the same encoding function runs on both sides.
2. Cost and latency. Every `search()` call is a network round trip; every
   chunk in the corpus was billed at index-build time. A production system
   embeds chunks once and persists the vectors (a vector DB), rather than
   re-embedding the whole corpus on every process start the way this toy
   example does for simplicity.
3. Rate limits and retries. Any network API can fail transiently — a rate
   limit, a momentary 5xx, a timeout — and a well-behaved client retries
   those with *exponential backoff*: wait a bit, retry, and if it fails
   again wait longer next time, so a burst of calls doesn't hammer a
   struggling or throttling server. `voyageai.Client` already implements
   this correctly (via `tenacity`), but ships with it OFF by default
   (`max_retries=0`) — you have to opt in. This is worth knowing generally,
   not just for Voyage: check whether an SDK already has retry/backoff
   before writing your own; duplicating it is easy to get subtly wrong
   (e.g. retrying non-idempotent calls, or backing off linearly instead of
   exponentially and hammering a rate limit harder each time).

Requires `VOYAGE_API_KEY` in the environment (from https://voyage.ai) — the
`voyageai.Client()` call below reads it automatically, the same pattern as
`Anthropic()` reading `ANTHROPIC_API_KEY`.
"""

from __future__ import annotations

import numpy as np
import voyageai
from sklearn.metrics.pairwise import cosine_similarity

from rag.index import SearchResult
from rag.ingest import Chunk

# voyage-4 is Voyage's current general-purpose embedding model — a solid
# default for text retrieval. voyage-4-lite trades some quality for lower
# cost/latency; voyage-4-large is the higher-quality/higher-cost option;
# voyage-code-3 exists for code-heavy corpora.
DEFAULT_MODEL = "voyage-4"

# voyageai.Client retries RateLimitError / ServiceUnavailableError / Timeout
# with exponential backoff + jitter (1s, up to 16s between attempts) — but
# only if max_retries > 0. New Voyage accounts without a payment method on
# file are capped at 3 requests/minute, which a single index build (one
# batch call) plus a few queries can exceed in seconds; this makes those
# transient throttles retry-and-recover instead of crashing the process.
DEFAULT_MAX_RETRIES = 5


class VoyageIndex:
    """Same shape as TfidfIndex: fit chunks once, then `search(query, k)`."""

    def __init__(
        self,
        chunks: list[Chunk],
        model: str = DEFAULT_MODEL,
        max_retries: int = DEFAULT_MAX_RETRIES,
    ):
        if not chunks:
            raise ValueError("Cannot build an index with zero chunks")
        self.chunks = chunks
        self.model = model
        # reads VOYAGE_API_KEY from the environment
        self._client = voyageai.Client(max_retries=max_retries)
        result = self._client.embed([c.text for c in chunks], model=model, input_type="document")
        self.embeddings = np.array(result.embeddings)

    def search(self, query: str, k: int = 3) -> list[SearchResult]:
        result = self._client.embed([query], model=self.model, input_type="query")
        query_vector = np.array(result.embeddings)
        scores = cosine_similarity(query_vector, self.embeddings)[0]
        top_indices = scores.argsort()[::-1][:k]
        return [SearchResult(chunk=self.chunks[i], score=float(scores[i])) for i in top_indices]
