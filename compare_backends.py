"""Retrieval-only comparison across all three indexing backends, with and
without cross-encoder reranking.

Runs the same style of questions as demo.py through TfidfIndex,
LocalDenseIndex, and VoyageIndex, printing only the retrieved chunks and
their scores — no Claude calls, so this is free and fast to re-run.
Generation (rag/generate.py) is identical regardless of backend or
reranking; this script isolates the two things that actually differ: which
chunks retrieval surfaces, how confidently, and how a cross-encoder pass
(rag/rerank.py) reorders each backend's own candidates.

What to look for:
  - The off-corpus question (mobile app) should score noticeably lower than
    the on-corpus questions on every backend — ideally with a wide gap.
    Compare that gap pre- vs. post-rerank: the cross-encoder should push it
    much lower (see rag/rerank.py's docstring for the measured numbers).
  - Try editing a question below to use different words than the source
    docs (e.g. "delete permanently" instead of "purge"). TF-IDF's score
    should drop sharply while the embedding backends barely notice — that
    gap IS semantic retrieval, and it's the whole reason dense embeddings
    exist over bag-of-words.
  - Reranking can also reorder which chunk comes out on top, not just
    rescale the score — that's the "precision" half of the two-stage
    retrieve-then-rerank pattern.

Backends and the reranker that aren't installed or configured are skipped
with a note instead of crashing the whole comparison — this only requires
the deps and API keys for whichever piece(s) you actually want to compare.

Run with: python compare_backends.py
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from rag.backends import BACKENDS, DEFAULT_MIN_SCORES, build_index
from rag.index import SearchResult
from rag.ingest import chunk_documents, load_documents
from rag.pipeline import DOCS_DIR, RAGPipeline
from rag.retrieve import Retriever

if TYPE_CHECKING:
    from rag.rerank import Reranker

# Wider than the final display width so reranking has room to promote a
# chunk the bi-encoder/TF-IDF ranked just outside the top FINAL_K.
CANDIDATE_K = 5
FINAL_K = 2

QUESTIONS = [
    "How many failed login attempts trigger an account lockout, and for how long?",
    "What happens to my files 30 days after I delete my account?",
    "Do I get a discount for paying annually?",
    "Does Nimbus have a native mobile app for iOS?",  # deliberately off-corpus
]


def build_reranker() -> Reranker | None:
    try:
        from rag.rerank import Reranker

        return Reranker()
    except ImportError as e:
        print(f"skipping reranking: dependency not installed ({e})")
        return None


def format_results(results: list[SearchResult]) -> str:
    return ", ".join(f"{r.chunk.source}={r.score:.3f}" for r in results)


def main() -> None:
    docs = load_documents(DOCS_DIR)
    chunks = chunk_documents(docs)

    indexes = {}
    for backend in BACKENDS:
        try:
            indexes[backend] = build_index(chunks, backend=backend)
        except ImportError as e:
            print(f"skipping {backend!r}: dependency not installed ({e})")
        except Exception as e:
            # Broad on purpose: this is a best-effort environment probe, not
            # production code. voyageai raises its own exception types for a
            # missing/invalid VOYAGE_API_KEY or a network failure, and we
            # want any of those to degrade to "skip this backend" rather
            # than aborting the whole comparison.
            print(f"skipping {backend!r}: {e}")

    if not indexes:
        raise SystemExit(
            "No backends available. Install at least one set of deps — "
            "see requirements.txt / requirements-optional.txt."
        )

    reranker = build_reranker()

    label = f"{', '.join(indexes)} (reranker: {'on' if reranker else 'off'})"
    print(f"\nComparing backends: {label}\n{'=' * 70}")

    for question in QUESTIONS:
        print(f"\nQ: {question}")
        for name, index in indexes.items():
            candidates = index.search(question, k=CANDIDATE_K)
            print(f"  {name:>10} raw top-{FINAL_K}: {format_results(candidates[:FINAL_K])}")
            if reranker:
                # Same retrieve -> rerank -> cutoff path the app uses, so a
                # question the pipeline would reject prints as "(none)".
                pipeline = RAGPipeline(
                    retriever=Retriever(index, min_score=DEFAULT_MIN_SCORES[name]),
                    reranker=reranker,
                    final_k=FINAL_K,
                    candidate_k=CANDIDATE_K,
                )
                reranked = pipeline.retrieve(question)
                print(f"  {name:>10} reranked:    {format_results(reranked) or '(none)'}")


if __name__ == "__main__":
    main()
