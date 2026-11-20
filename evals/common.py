"""Dataset loading and metric functions shared by the eval scripts.

Relevance is labeled by *source document + evidence phrase*, not by chunk
ID. A chunk counts as relevant if it comes from one of the expected source
documents AND contains one of the evidence phrases. Chunk IDs like
`security#2` change whenever chunk_size/overlap change; evidence phrases
don't, so the labels survive re-chunking. Requiring the phrase (not just
the source) matters because every document has several chunks, and only
the one holding the fact actually answers the question.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from rag.index import SearchResult
from rag.ingest import Chunk

DATASET_PATH = Path(__file__).resolve().parent / "dataset.jsonl"
RESULTS_DIR = Path(__file__).resolve().parent / "results"


@dataclass(frozen=True)
class EvalItem:
    id: str
    category: str  # factual | paraphrase | multi_hop | off_corpus
    question: str
    expected_sources: tuple[str, ...]
    evidence: tuple[str, ...]
    reference_answer: str

    @property
    def answerable(self) -> bool:
        return bool(self.expected_sources)


def load_dataset(path: Path = DATASET_PATH) -> list[EvalItem]:
    items = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        raw = json.loads(line)
        item = EvalItem(
            id=raw["id"],
            category=raw["category"],
            question=raw["question"],
            expected_sources=tuple(raw["expected_sources"]),
            evidence=tuple(raw["evidence"]),
            reference_answer=raw["reference_answer"],
        )
        if item.answerable and not item.evidence:
            raise ValueError(f"{path}:{line_no}: answerable item {item.id!r} has no evidence")
        items.append(item)
    ids = [i.id for i in items]
    if len(ids) != len(set(ids)):
        raise ValueError(f"{path}: duplicate item ids")
    return items


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def is_relevant(chunk: Chunk, item: EvalItem) -> bool:
    if chunk.source not in item.expected_sources:
        return False
    text = _normalize(chunk.text)
    return any(_normalize(e) in text for e in item.evidence)


def check_labels(items: list[EvalItem], chunks: list[Chunk]) -> list[str]:
    """Return the ids of answerable items that no chunk in the corpus
    satisfies — a typo in an evidence phrase would otherwise silently score
    as a retrieval miss on every backend."""
    return [i.id for i in items if i.answerable and not any(is_relevant(c, i) for c in chunks)]


def first_relevant_rank(results: list[SearchResult], item: EvalItem) -> int | None:
    """1-based rank of the first relevant result, or None if none is relevant."""
    for rank, r in enumerate(results, start=1):
        if is_relevant(r.chunk, item):
            return rank
    return None


def reciprocal_rank(results: list[SearchResult], item: EvalItem) -> float:
    rank = first_relevant_rank(results, item)
    return 1.0 / rank if rank else 0.0


_CITATION = re.compile(r"\[([a-z0-9_]+)\]")


def parse_citations(answer_text: str) -> set[str]:
    """Source names the answer cites in [brackets], e.g. "[pricing]".

    Only lowercase identifiers count, which skips Markdown links and
    bracketed prose like "[answer truncated...]".
    """
    return set(_CITATION.findall(answer_text))


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, round(pct / 100 * (len(ordered) - 1)))
    return ordered[index]
