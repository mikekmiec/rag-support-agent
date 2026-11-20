"""Retrieval eval: score every backend (with and without reranking) against
the labeled questions in evals/dataset.jsonl. No Claude calls — free and
fast to re-run after any change to chunking, backends, or thresholds.

Run from the repo root:
    python -m evals.retrieval_eval                      # every installed backend
    python -m evals.retrieval_eval --backends tfidf local
    python -m evals.retrieval_eval --save               # also write JSON to evals/results/

Metrics (answerable questions):
  hit@1 / hit@3  fraction where a relevant chunk is ranked 1st / anywhere in
                 the final top-3 the model would see
  MRR            mean reciprocal rank of the first relevant chunk (1.0 =
                 always first, 0.5 = always second, 0 = never retrieved)
  pool hit       reranking only: fraction where a relevant chunk was in the
                 candidate pool at all. The reranker can't recover a chunk
                 the first stage never fetched, so this caps its hit@3.
  false reject   fraction where the cutoff removed *everything*, so the model
                 is told there's no context even though the corpus has the answer

Metrics (off-corpus questions):
  rejected       fraction where nothing cleared the cutoff, so the model gets
                 no context and has nothing it could misuse

The "calibration" table shows how well each config's raw top-1 score
separates real answers from off-corpus questions, which is what the
thresholds in rag/backends.py and rag/pipeline.py are tuned from.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone

from evals.common import (
    RESULTS_DIR,
    EvalItem,
    check_labels,
    first_relevant_rank,
    load_dataset,
    mean,
    percentile,
    reciprocal_rank,
)
from rag.backends import BACKENDS, DEFAULT_MIN_SCORES, build_index
from rag.index import Index
from rag.ingest import chunk_documents, load_documents
from rag.pipeline import DOCS_DIR, RAGPipeline, build_reranker
from rag.retrieve import Retriever


@dataclass
class ConfigResult:
    config: str
    hit_at_1: float
    hit_at_3: float
    mrr: float
    pool_hit: float | None
    false_reject: float
    off_corpus_rejected: float
    latency_p50_ms: float
    latency_p95_ms: float
    hit_at_3_by_category: dict[str, float]
    # Calibration: raw top-1 scores (no cutoff applied).
    answerable_top1_min: float
    off_corpus_top1_max: float
    misses: list[str] = field(default_factory=list)


def evaluate(name: str, pipeline: RAGPipeline, items: list[EvalItem]) -> ConfigResult:
    off_corpus = [i for i in items if not i.answerable]

    hits1, hits3, rrs, pool_hits, false_rejects, latencies = [], [], [], [], [], []
    by_category: dict[str, list[float]] = {}
    answerable_top1: list[float] = []
    off_top1: list[float] = []
    rejected = []
    misses = []

    for item in items:
        start = time.perf_counter()
        results = pipeline.retrieve(item.question)
        latencies.append((time.perf_counter() - start) * 1000)

        # Raw scores with no cutoff, for the calibration table. With
        # reranking that's the reranker's score of its best candidate.
        raw = pipeline.index.search(item.question, k=pipeline.candidate_k)
        if pipeline.reranker:
            raw = pipeline.reranker.rerank(item.question, raw, top_k=1)
        top1 = raw[0].score if raw else 0.0

        if not item.answerable:
            rejected.append(1.0 if not results else 0.0)
            off_top1.append(top1)
            continue

        rank = first_relevant_rank(results, item)
        hits1.append(1.0 if rank == 1 else 0.0)
        hit3 = 1.0 if rank is not None and rank <= 3 else 0.0
        hits3.append(hit3)
        rrs.append(reciprocal_rank(results, item))
        false_rejects.append(1.0 if not results else 0.0)
        by_category.setdefault(item.category, []).append(hit3)
        if not hit3:
            misses.append(item.id)
        if first_relevant_rank(raw, item) == 1:
            answerable_top1.append(top1)
        if pipeline.reranker:
            pool = pipeline.index.search(item.question, k=pipeline.candidate_k)
            pool_hits.append(1.0 if first_relevant_rank(pool, item) else 0.0)

    return ConfigResult(
        config=name,
        hit_at_1=mean(hits1),
        hit_at_3=mean(hits3),
        mrr=mean(rrs),
        pool_hit=mean(pool_hits) if pipeline.reranker else None,
        false_reject=mean(false_rejects),
        off_corpus_rejected=mean(rejected) if off_corpus else 0.0,
        latency_p50_ms=percentile(latencies, 50),
        latency_p95_ms=percentile(latencies, 95),
        hit_at_3_by_category={k: mean(v) for k, v in sorted(by_category.items())},
        answerable_top1_min=min(answerable_top1) if answerable_top1 else 0.0,
        off_corpus_top1_max=max(off_top1) if off_top1 else 0.0,
        misses=misses,
    )


def build_indexes(backends: list[str], chunks) -> dict[str, Index]:
    indexes = {}
    for backend in backends:
        try:
            indexes[backend] = build_index(chunks, backend=backend)
        except ImportError as e:
            print(f"skipping {backend!r}: dependency not installed ({e})")
        except Exception as e:  # pylint: disable=broad-except
            # e.g. a missing VOYAGE_API_KEY or network failure: skip the
            # backend rather than abort the whole eval.
            print(f"skipping {backend!r}: {e}")
    return indexes


def print_report(results: list[ConfigResult]) -> None:
    def pct(x: float | None) -> str:
        return "   -" if x is None else f"{x * 100:4.0f}%"

    header = (
        f"{'config':<22} {'hit@1':>6} {'hit@3':>6} {'MRR':>5} {'pool':>6} "
        f"{'f.rej':>6} {'off-rej':>7} {'p50ms':>6} {'p95ms':>6}"
    )
    print("\nRetrieval quality")
    print(header)
    print("-" * len(header))
    for r in results:
        print(
            f"{r.config:<22} {pct(r.hit_at_1):>6} {pct(r.hit_at_3):>6} {r.mrr:5.2f} "
            f"{pct(r.pool_hit):>6} {pct(r.false_reject):>6} {pct(r.off_corpus_rejected):>7} "
            f"{r.latency_p50_ms:6.1f} {r.latency_p95_ms:6.1f}"
        )

    categories = sorted({c for r in results for c in r.hit_at_3_by_category})
    print("\nhit@3 by question type")
    print(f"{'config':<22} " + " ".join(f"{c:>11}" for c in categories))
    for r in results:
        cells = " ".join(f"{pct(r.hit_at_3_by_category.get(c)):>11}" for c in categories)
        print(f"{r.config:<22} {cells}")

    print("\nCalibration (raw top-1 score, before any cutoff)")
    print(f"{'config':<22} {'lowest real hit':>16} {'highest off-corpus':>19} {'separable':>10}")
    for r in results:
        separable = "yes" if r.answerable_top1_min > r.off_corpus_top1_max else "NO"
        print(
            f"{r.config:<22} {r.answerable_top1_min:16.3f} "
            f"{r.off_corpus_top1_max:19.3f} {separable:>10}"
        )

    print("\nMissed (no relevant chunk in final top-3)")
    for r in results:
        print(f"  {r.config:<20} {', '.join(r.misses) or '(none)'}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", maxsplit=1)[0])
    parser.add_argument("--backends", nargs="+", choices=BACKENDS, default=list(BACKENDS))
    parser.add_argument("--no-rerank", action="store_true", help="skip the reranked configs")
    parser.add_argument("--save", action="store_true", help="write results JSON to evals/results/")
    parser.add_argument(
        "--min-hit-at-3",
        type=float,
        default=0.0,
        help="exit non-zero if any config's hit@3 is below this (e.g. 0.85 in CI)",
    )
    args = parser.parse_args()

    items = load_dataset()
    chunks = chunk_documents(load_documents(DOCS_DIR))
    unmatched = check_labels(items, chunks)
    if unmatched:
        raise SystemExit(f"Evidence phrases match no chunk in the corpus for: {unmatched}")

    indexes = build_indexes(args.backends, chunks)
    if not indexes:
        raise SystemExit("No backends available.")

    reranker = None
    if not args.no_rerank:
        try:
            reranker = build_reranker(enabled=True)
        except ImportError as e:
            print(f"skipping reranked configs: dependency not installed ({e})")

    results = []
    for name, index in indexes.items():
        retriever = Retriever(index, min_score=DEFAULT_MIN_SCORES[name])
        results.append(evaluate(name, RAGPipeline(retriever=retriever), items))
        if reranker:
            pipeline = RAGPipeline(retriever=retriever, reranker=reranker)
            results.append(evaluate(f"{name} + rerank", pipeline, items))

    n_answerable = sum(i.answerable for i in items)
    n_off = len(items) - n_answerable
    print(f"{len(items)} questions ({n_answerable} answerable, {n_off} off-corpus)")
    print_report(results)

    # Regression gate for CI: fail if any config drops below the floor.
    failing = [r.config for r in results if r.hit_at_3 < args.min_hit_at_3]
    if failing:
        raise SystemExit(f"hit@3 below {args.min_hit_at_3:.0%} for: {', '.join(failing)}")

    if args.save:
        RESULTS_DIR.mkdir(exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        path = RESULTS_DIR / f"retrieval-{stamp}.json"
        path.write_text(json.dumps([asdict(r) for r in results], indent=2))
        print(f"\nSaved {path}")


if __name__ == "__main__":
    main()
