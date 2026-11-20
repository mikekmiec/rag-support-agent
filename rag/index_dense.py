"""Alternate indexing backend: local neural (dense) embeddings.

TfidfIndex (index.py) represents chunks as sparse bag-of-words vectors —
zero notion of meaning, only shared vocabulary. This backend instead runs
every chunk through a small pretrained neural network (a "sentence
embedding" model) that maps text to a dense vector — typically a few hundred
floats — positioned so that texts with *similar meaning* land near each
other, even with zero shared words. That's what fixes TF-IDF's blind spot:
a query like "How long until my data is gone?" can now match a chunk about
"permanently purged" because the model learned during training that those
phrases mean similar things, not because any word overlaps.

Everything downstream of the embedding step — cosine similarity, top-k
ranking — is identical to TfidfIndex. Only the vectorization step changes.
That symmetry is the point: diff this file against index.py and the only
real difference is `_encode()`.

Tradeoffs vs TF-IDF: much better semantic recall, but slower to build
(runs a neural net over every chunk), needs a model download (~90MB for the
default model) and a heavier dependency (PyTorch), and is no longer
human-interpretable — you can't point to "which words" drove a match the
way you can with TF-IDF weights.

Tradeoffs vs VoyageIndex (index_voyage.py): runs fully offline after the
one-time model download, no per-call cost, no API key, no network latency.
But it's a much smaller/older model than what a hosted embeddings API
offers, so retrieval quality is generally lower, and every chunk/query is
encoded on your own CPU (or GPU, if available) instead of purpose-built
serving infrastructure.
"""

from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from rag.index import SearchResult
from rag.ingest import Chunk

# all-MiniLM-L6-v2: a small (~90MB), fast, widely-used sentence embedding
# model. Not state of the art, but a standard baseline for "local dense
# retrieval" demos and good enough to clearly beat TF-IDF on paraphrased
# queries. Swap in a larger model (e.g. "all-mpnet-base-v2") for quality at
# the cost of speed.
DEFAULT_MODEL = "all-MiniLM-L6-v2"


class LocalDenseIndex:  # pylint: disable=too-few-public-methods
    """Same shape as TfidfIndex: fit chunks once, then `search(query, k)`."""

    def __init__(self, chunks: list[Chunk], model_name: str = DEFAULT_MODEL):
        if not chunks:
            raise ValueError("Cannot build an index with zero chunks")
        self.chunks = chunks
        # Downloads the model to a local cache on first use (~90MB), then
        # loads from cache on every run after that.
        self._model = SentenceTransformer(model_name)
        self.embeddings = self._encode([c.text for c in chunks])

    def _encode(self, texts: list[str]) -> np.ndarray:
        # normalize_embeddings=True makes every vector unit length, which is
        # what most sentence-embedding models are trained/tuned to be
        # compared with — cosine similarity is the standard metric here.
        # encode() already returns a numpy array by default (convert_to_numpy
        # defaults to True), but its type hints are a union that includes
        # torch.Tensor. np.asarray makes the return type match at no cost.
        return np.asarray(
            self._model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
        )

    def search(self, query: str, k: int = 3) -> list[SearchResult]:
        """Return the k chunks whose embeddings are most similar to the query."""
        query_vector = self._encode([query])
        scores = cosine_similarity(query_vector, self.embeddings)[0]
        top_indices = scores.argsort()[::-1][:k]
        return [SearchResult(chunk=self.chunks[i], score=float(scores[i])) for i in top_indices]
