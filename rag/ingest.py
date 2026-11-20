"""Stage 1: Ingestion — load raw documents and split them into chunks.

Chunking matters because retrieval scores a whole chunk against the query.
Too large, and a chunk mixes unrelated facts together, diluting relevance
scores and making it hard to cite a precise source. Too small, and you lose
surrounding context the model needs to answer well. A sliding window with
overlap is a standard middle ground: it keeps chunks focused while avoiding
hard cuts in the middle of a fact that spans a chunk boundary.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Document:
    source: str  # filename stem, used for citations
    text: str


@dataclass(frozen=True)
class Chunk:
    id: str  # "<source>#<chunk_index>", unique within the corpus
    source: str
    text: str


def load_documents(directory: Path) -> list[Document]:
    """Read every .txt file in `directory` into a Document."""
    docs = []
    for path in sorted(directory.glob("*.txt")):
        text = path.read_text(encoding="utf-8").strip()
        docs.append(Document(source=path.stem, text=text))
    if not docs:
        raise FileNotFoundError(f"No .txt documents found in {directory}")
    return docs


def chunk_documents(docs: list[Document], chunk_size: int = 120, overlap: int = 20) -> list[Chunk]:
    """Split each document into overlapping word-count windows.

    `chunk_size` and `overlap` are in words, not tokens or characters — good
    enough for a toy corpus. Production ingestion pipelines usually chunk by
    token count (to match the embedding model's tokenizer) and often split on
    sentence or paragraph boundaries instead of a raw sliding window.
    """
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")
    if not 0 <= overlap < chunk_size:
        # overlap >= chunk_size would make `start` stop advancing (or move
        # backwards) and loop forever.
        raise ValueError(
            f"overlap must be in [0, chunk_size); got overlap={overlap}, chunk_size={chunk_size}"
        )
    chunks: list[Chunk] = []
    for doc in docs:
        words = doc.text.split()
        start = 0
        index = 0
        while start < len(words):
            end = start + chunk_size
            chunk_text = " ".join(words[start:end])
            chunks.append(Chunk(id=f"{doc.source}#{index}", source=doc.source, text=chunk_text))
            index += 1
            if end >= len(words):
                break
            start = end - overlap  # step back so chunks overlap
    return chunks
