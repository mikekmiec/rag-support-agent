"""Interactive RAG demo. Run with: python main.py

Wires the pipeline stages together:
  ingest (load + chunk) -> index (backend-selectable) -> retrieve (top-k)
    -> [optional] rerank (cross-encoder) -> generate (Claude)

Set RAG_BACKEND to pick the indexing backend (default: tfidf):
  RAG_BACKEND=tfidf   python main.py   # sparse keyword vectors, no extra deps
  RAG_BACKEND=local   python main.py   # local neural embeddings (sentence-transformers)
  RAG_BACKEND=voyage  python main.py   # Voyage AI hosted embeddings (needs VOYAGE_API_KEY)
See rag/index.py, rag/index_dense.py, rag/index_voyage.py for what each does.

Set RAG_RERANK=1 to add the cross-encoder reranking stage on top of any
backend (needs sentence-transformers, same as RAG_BACKEND=local):
  RAG_RERANK=1 RAG_BACKEND=voyage python main.py
See rag/rerank.py for what it does and why it's a separate stage.
"""

from __future__ import annotations

import anthropic

from rag.client import make_client
from rag.display import printable
from rag.generate import generate_answer
from rag.pipeline import DOCS_DIR, build_pipeline


def main() -> None:
    """Run an interactive question-answering loop over the corpus."""
    client = make_client()  # the current user's own key; see rag/client.py
    pipeline = build_pipeline()

    print(f"Backend: {pipeline.label}")
    print(f"Indexed {len(pipeline.index.chunks)} chunks from {DOCS_DIR}")
    print("Ask a question about Nimbus Cloud Storage (Ctrl+C to quit)\n")

    while True:
        try:
            question = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not question:
            continue

        results = pipeline.retrieve(question)
        print("  retrieved:")
        if not results:
            print("    (nothing above the relevance threshold)")
        for r in results:
            preview = r.chunk.text[:70].replace("\n", " ")
            print(f"    [{r.score:.3f}] {r.chunk.source}: {preview}...")

        try:
            answer = generate_answer(client, question, results)
        except anthropic.APIError as e:
            print(f"\n  API error: {e}\n")
            continue
        print(f"\n{printable(answer.text)}\n")


if __name__ == "__main__":
    main()
