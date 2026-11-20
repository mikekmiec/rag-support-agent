"""Non-interactive demo: run a fixed set of questions through the pipeline.

Good for a first test run and for walkthroughs — each question is chosen so
the answer requires a specific fact that only exists in the corpus, which
is the clearest way to show retrieval is actually doing work (a model
answering from general "knowledge" alone would have to guess).

Run with: python demo.py
Or pick a backend: RAG_BACKEND=local python demo.py   (see main.py for options)
Or add reranking:  RAG_RERANK=1 python demo.py         (see rag/rerank.py)

For measured quality numbers rather than eyeballing answers, see evals/.
"""

from __future__ import annotations

from anthropic import Anthropic

from rag.generate import generate_answer
from rag.pipeline import build_pipeline

QUESTIONS = [
    "How many failed login attempts trigger an account lockout, and for how long?",
    "What happens to my files 30 days after I delete my account?",
    "What's the rate limit on the file upload API, and how do I avoid hitting it for bulk uploads?",
    "Do I get a discount for paying annually?",
    "What caused the March outage and how long did it last?",
    "Does Nimbus have a native mobile app for iOS?",  # deliberately off-corpus
]


def main() -> None:
    """Answer each demo question and print what was retrieved."""
    pipeline = build_pipeline()
    client = Anthropic()

    print(f"Backend: {pipeline.label}\n")
    for question in QUESTIONS:
        results = pipeline.retrieve(question)
        print(f"Q: {question}")
        if results:
            for r in results:
                print(f"   retrieved [{r.score:.3f}] {r.chunk.source}")
        else:
            print("   retrieved: (nothing above threshold)")
        answer = generate_answer(client, question, results)
        print(f"A: {answer.text}\n{'-' * 70}")


if __name__ == "__main__":
    main()
