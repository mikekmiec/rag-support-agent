"""Backend selection: build any of the three index implementations by name.

Each backend's heavy dependency (PyTorch for LocalDenseIndex, the voyageai
client for VoyageIndex) is imported *inside* its branch, not at module load
time. That means `import rag.backends` — and therefore `python demo.py`
with the default "tfidf" backend — never pulls in PyTorch or the voyageai
package, even though the code to use them exists in the repo. Only the
backend you actually select pays its import cost.
"""

from __future__ import annotations

from rag.index import Index
from rag.ingest import Chunk

BACKENDS = ("tfidf", "local", "voyage")

# Minimum cosine similarity for a chunk to count as relevant, per backend.
# One shared number doesn't work: TF-IDF scores exactly 0 for chunks with no
# shared vocabulary, so a small floor filters well, but dense embeddings
# score even unrelated text around 0.3-0.4, so a floor of 0.05 never
# filters anything. Calibrated with `python -m evals.retrieval_eval`
# against evals/dataset.jsonl, choosing the highest cutoff that never
# discards an answerable question (a missed answer is worse than extra
# context, since the prompt still tells the model to abstain):
#   tfidf  0.05  real answers' top score >= 0.10; off-corpus up to 0.21
#   local  0.30  real answers' top score >= 0.33; rejects 3 of 6 off-corpus
#   voyage 0.0   not yet measured; run the eval with VOYAGE_API_KEY set
# Re-run the eval if you change the corpus, chunking, or embedding model.
DEFAULT_MIN_SCORES = {
    "tfidf": 0.05,
    "local": 0.30,
    "voyage": 0.0,
}


def build_index(chunks: list[Chunk], backend: str = "tfidf") -> Index:
    if backend == "tfidf":
        from rag.index import TfidfIndex

        return TfidfIndex(chunks)
    elif backend == "local":
        from rag.index_dense import LocalDenseIndex

        return LocalDenseIndex(chunks)
    elif backend == "voyage":
        from rag.index_voyage import VoyageIndex

        return VoyageIndex(chunks)
    else:
        raise ValueError(f"Unknown backend {backend!r}, expected one of {BACKENDS}")
