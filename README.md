# Simple RAG Example

A minimal, from-scratch Retrieval-Augmented Generation pipeline: a toy
support-doc corpus, three swappable retrieval backends (TF-IDF, local
neural embeddings, Voyage AI hosted embeddings), an optional cross-encoder
reranking pass, and Claude for generation. Built as a study project — the
code is intentionally small enough to read end-to-end and explain in an
interview.

## What RAG is, in one paragraph

An LLM only knows what was in its training data (plus whatever's in the
prompt). RAG closes that gap at request time: given a question, you
*retrieve* the most relevant passages from your own document collection,
*augment* the prompt with them as context, and ask the model to *generate*
an answer grounded in that context instead of its parametric memory. This
is cheaper and faster to update than fine-tuning (swap a file, not retrain
a model), and it lets the model cite sources and say "I don't know" when
nothing relevant is retrieved — which plain fine-tuning can't give you.

## Pipeline

```text
data/docs/*.txt
      │
      ▼
┌─────────────┐   chunk_documents()   ┌───────────────┐   TfidfVectorizer   ┌────────────┐
│  ingest.py  │ ────────────────────► │   index.py    │ ──────────────────► │  in-memory │
│ load + split│                       │  fit TF-IDF    │                     │  matrix    │
└─────────────┘                       └───────┬────────┘                    └─────┬──────┘
                                               │                                   │
                                               ▼                                   ▼
                                       ┌────────────────┐   query vector   ┌───────────────┐
                     question ───────► │  retrieve.py   │ ───────────────► │ cosine sim,   │
                                       │  top-k + filter│ ◄─────────────── │ rank, top-k   │
                                       └───────┬────────┘                  └───────────────┘
                                               │ retrieved chunks (wider pool if reranking)
                                               ▼
                                    [optional] rag/rerank.py — cross-encoder
                                    rescoring, narrows back down to top-k
                                               │
                                               ▼
                                       ┌────────────────┐
                                       │  generate.py   │  builds augmented prompt,
                                       │  Claude API     │  calls client.messages.create
                                       └────────────────┘
```

Four required stages, plus one optional stage, one file each:

| File                                                   | Stage                                | What it does                                                    |
| ------------------------------------------------------ | ------------------------------------ | --------------------------------------------------------------- |
| `rag/ingest.py`                                        | Ingestion                            | Load `.txt` files, split into overlapping word-window chunks    |
| `rag/index.py` (+ `index_dense.py`, `index_voyage.py`) | Indexing                             | Turn chunks into vectors, per backend; cosine-similarity search |
| `rag/retrieve.py`                                      | Retrieval                            | Query-time policy: top-k, minimum-relevance cutoff              |
| `rag/rerank.py`                                        | Reranking (optional, `RAG_RERANK=1`) | Cross-encoder rescoring of retrieved candidates for precision   |
| `rag/generate.py`                                      | Augmentation + Generation            | Format retrieved chunks into the prompt, call Claude            |
| `rag/pipeline.py`                                      | Wiring                               | Retrieve, rerank, apply cutoffs; shared by every entry point    |

## Setup

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=your-key-here   # or `ant auth login`
```

That's enough for the default backend (TF-IDF) with no reranking. The two
alternate retrieval backends, and reranking, need extra, optional
dependencies — see [Retrieval backends](#retrieval-backends) below.

## Run it

```bash
python demo.py              # runs a fixed set of questions, non-interactive — start here
python main.py               # interactive: ask your own questions about the corpus
python compare_backends.py   # retrieval-only, no Claude calls — compares all installed backends, with and without reranking
```

Both `demo.py` and `main.py` default to the TF-IDF backend with no
reranking. Change either with an env var:

```bash
RAG_BACKEND=local  python demo.py   # local neural embeddings (sentence-transformers)
RAG_BACKEND=voyage python demo.py   # Voyage AI hosted embeddings
RAG_RERANK=1        python demo.py   # add cross-encoder reranking on top of any backend
```

The corpus (`data/docs/`) is six documents about a fictional company,
"Nimbus Cloud Storage" — pricing, security, API rate limits, data retention,
support SLAs, and outage history. Each has specific numbers (lockout
thresholds, rate limits, discount percentages) that a model can't guess —
which is exactly what makes it easy to *tell* retrieval is working: try
asking `demo.py`'s questions with the corpus deleted and watch the model
have nothing to cite.

## Evaluation

`evals/dataset.jsonl` holds 31 labeled questions: 25 answerable (factual
lookups, paraphrases that share few words with the source, and multi-hop
questions) and 6 the corpus can't answer. Each answerable question names
its source document, a short evidence phrase the right chunk must contain,
and a reference answer. Labeling by evidence phrase instead of chunk ID
keeps the labels valid if you change the chunking.

```bash
python -m evals.retrieval_eval    # free: retrieval only, every installed backend, +/- reranking
python -m evals.answer_eval       # paid (~$0.80): full pipeline + LLM judge
python -m pytest                  # unit tests (pip install -r requirements-dev.txt)
```

**Retrieval** (`retrieval_eval`), measured on this corpus:

| Config           | hit@1 | hit@3 | MRR  | Paraphrase hit@3 | Off-corpus rejected |
| ---------------- | ----- | ----- | ---- | ---------------- | ------------------- |
| `tfidf`          | 60%   | 88%   | 0.73 | 57%              | 0%                  |
| `tfidf + rerank` | 80%   | 92%   | 0.86 | 71%              | 0%                  |
| `local`          | 68%   | 92%   | 0.79 | 86%              | 50%                 |
| `local + rerank` | 88%   | 100%  | 0.94 | 100%             | 50%                 |

hit@k is the share of answerable questions with a relevant chunk in the top
k; MRR is the mean of 1/rank of the first relevant chunk. Three things the
numbers show:

- **Dense embeddings fix paraphrases.** TF-IDF misses 3 of 7 reworded
  questions ("card gets declined" never matches "failed payment"); local
  embeddings miss 1.
- **Reranking is the biggest single gain.** It moves the right chunk to
  first place far more often (hit@1 68% → 88% on `local`).
- **No score cutoff cleanly rejects off-corpus questions.** On every
  config, some off-corpus question scores higher than some real answer
  (the eval's calibration table shows the overlap). The cross-encoder is
  the worst offender: it scores some correct paraphrase matches near 0.
  The cutoffs are therefore set to never discard a real answer, and the
  system prompt's "say you don't know" rule is the real defense.
  `answer_eval` measures how well that rule holds.

**Answers** (`answer_eval`) runs each question through retrieval and
Claude, then grades the answer with a second model (`claude-sonnet-5-5`)
against the reference answer and the retrieved context. It reports
accuracy, faithfulness (every claim supported by the retrieved context),
abstention on off-corpus questions, citation validity (every `[source]`
cited was actually retrieved), latency, and cost. It prints the judge's
reasoning for every failure, so you can check the grader as well as the
pipeline.

## Retrieval backends

Three interchangeable implementations of the same `Index` interface
(`chunks`, `search(query, k) -> list[SearchResult]` — see `rag/index.py`).
`rag/backends.py` builds whichever one you ask for; nothing else in the
pipeline (`retrieve.py`, `generate.py`) knows or cares which is active.

| Backend  | File                  | How it represents text                               | Extra setup                                                              |
| -------- | --------------------- | ---------------------------------------------------- | ------------------------------------------------------------------------ |
| `tfidf`  | `rag/index.py`        | Sparse bag-of-words (keyword overlap)                | None — this is the default                                               |
| `local`  | `rag/index_dense.py`  | Dense neural embedding, computed on your machine     | `pip install sentence-transformers` (pulls in PyTorch, a real download)  |
| `voyage` | `rag/index_voyage.py` | Dense neural embedding, computed via Voyage AI's API | `pip install voyageai` + `VOYAGE_API_KEY` env var (get one at voyage.ai) |

Run `python compare_backends.py` after installing whichever backends you
want to try — it builds each available one, runs the same questions through
all of them, and prints retrieved sources + scores side by side with no
Claude calls (so it's free to iterate on). The comparison is the fastest
way to *see* the TF-IDF vs. neural-embedding tradeoff discussed below
instead of just reading about it: edit a question to paraphrase away from
the source docs' exact wording and watch TF-IDF's score collapse while the
embedding backends barely move.

### Reranking (optional fourth stage)

`rag/rerank.py` adds a cross-encoder pass on top of *any* of the three
backends above — it's backend-agnostic, since it only ever looks at
`(query, chunk text)` pairs, never the vectors that produced them. This is
the standard **retrieve-then-rerank** pattern: the index does a cheap,
approximate first pass over the whole corpus (recall), and the
cross-encoder does an expensive, accurate second pass over just the
resulting candidates (precision).

```bash
pip install sentence-transformers   # same dependency as RAG_BACKEND=local — reranking reuses it
RAG_RERANK=1 python demo.py
```

`compare_backends.py` shows both the raw retrieval scores and the
reranked scores for every backend, so you can see reranking's effect
directly. It tends to converge different backends onto the *same* final
answer even when their raw top-2 rankings disagreed, because the reranker
only cares whether a candidate actually answers the question, not which
backend proposed it. See [Evaluation](#evaluation) for the measured gain,
and for why its scores don't work as an off-corpus cutoff.

## What to look at, in order

1. **`rag/ingest.py`** — chunking strategy (sliding window with overlap, why chunk size is a tradeoff).
2. **`rag/index.py`** — what TF-IDF actually computes, and why it's called "sparse retrieval." Then **`rag/index_dense.py`** and **`rag/index_voyage.py`** for the two dense (neural embedding) alternatives — same `search()` shape, different vectorization.
3. **`rag/retrieve.py`** — why retrieval policy (top-k, score threshold) is kept separate from the index itself.
4. **`rag/rerank.py`** — why a cross-encoder is a *different architecture*, not just a bigger embedding model, and why it's a second pass over a few candidates rather than a replacement for the index.
5. **`rag/generate.py`** — the augmented prompt is just string formatting; the system prompt is what makes the model refuse to answer ungrounded questions.
6. **`rag/pipeline.py`** — how the stages connect, and where each relevance cutoff applies.
7. Run `demo.py` and watch the last question (deliberately off-corpus) — the model should say it doesn't know rather than hallucinate.
8. **`evals/`** — run `python -m evals.retrieval_eval` to see the numbers behind every claim above.

## Interview talking points

**Why TF-IDF here instead of neural embeddings?** TF-IDF is a sparse,
keyword-based representation — fast, free, fully interpretable, no external
API call. It fails on synonymy/paraphrase (a query about "cost" won't match
a chunk that only says "price" if IDF has down-weighted both, and it has
zero notion that "purge" and "delete permanently" are related). Neural
embedding models encode meaning into dense vectors and catch semantic
matches TF-IDF misses, at the cost of an extra API call or a loaded model —
this repo actually implements both alternatives so you can compare them
directly instead of taking it on faith: `rag/index_dense.py` runs a small
model (`sentence-transformers`) locally, `rag/index_voyage.py` calls
Voyage AI's hosted API (Anthropic's recommended embedding partner, since
Anthropic doesn't ship its own embedding endpoint). Run
`python compare_backends.py` and watch the score gap on a paraphrased
query — that gap is the entire pitch for dense retrieval in one number.
Production systems often run **hybrid search**: sparse (TF-IDF/BM25) +
dense (embeddings), combined via reciprocal rank fusion, because each
catches cases the other misses.

**Chunking tradeoffs.** Small chunks → precise retrieval, less irrelevant
context, but each chunk lacks surrounding context and you need more of them
to cover an idea. Large chunks → more context per hit, but mixed topics
dilute the relevance score and blow through context budget faster. Overlap
softens hard cuts that would otherwise split a fact across a chunk
boundary. Real systems often chunk semantically (by heading/paragraph) or
by token count matched to the embedding model's context window, not raw
word count.

**Why a separate retriever from the index?** The index is just a data
structure (vectors → nearest neighbors). Query-time policy — how many
results, minimum relevance, metadata filters, reranking — changes
independently of how the vectors are stored. Keeping them separate is what
lets you swap TF-IDF for a real vector DB (pgvector, Pinecone, Weaviate)
without touching retrieval policy.

**Evaluation.** Implemented in `evals/` (see [Evaluation](#evaluation)):
hit@k and MRR for retrieval, then end-to-end accuracy, *faithfulness* (did
the answer only use retrieved content?), abstention on off-corpus
questions, and citation validity, graded by an LLM judge against labeled
reference answers. The judge prints its reasoning so it can be audited.

**Failure modes this demo makes visible.** Ask a question with no good
match in the corpus. The system prompt instructs the model to say it
doesn't know rather than fill the gap from its own training data — that's
the generation-side defense, and the eval shows it has to carry most of
the weight. A retrieval-side score cutoff looks like it should catch
off-corpus questions, but on every backend the score ranges overlap: dense
embeddings score unrelated text around 0.2–0.45, and the best cutoff for
the local backend (0.30) rejects only half of the off-corpus questions
without discarding a real answer. The overlap only shows up when you score
a labeled set of real and off-topic questions. Testing a few questions by
hand won't reveal it.

**Reranking.** A common two-stage pattern: retrieve a larger candidate set
cheaply (top 20-50 by TF-IDF or embedding cosine similarity), then rerank
with a more expensive but more accurate cross-encoder model down to the
top 3-5 that actually go in the prompt — implemented here in
`rag/rerank.py`, composable with any of the three backends. The mechanism
is the key thing to be able to explain: a bi-encoder embeds the query and
each chunk *independently*, so it never lets the model reason about the
query and a specific chunk together — it can only compare two vectors that
were each computed in isolation. A cross-encoder feeds `(query, chunk)` in
as one joint input and attends across both, which is strictly more
accurate (it can judge "does this chunk answer the question," not just
"is this chunk about a similar topic") and strictly more expensive (no
precomputing — every candidate has to be scored at query time, so it only
works over a small candidate set, not the whole corpus). Measured here:
reranking the local backend's top 8 raises hit@1 from 68% to 88%. The
catch is that a small MS MARCO cross-encoder is a good *ranker* but a poor
*relevance detector*. It scores some correct paraphrase matches near 0, so
its scores can't double as an "is anything relevant?" cutoff.

**RAG vs. long context vs. fine-tuning.** With ~1M-token context windows
now available, "just put everything in the prompt" is viable for smaller
corpora — but it costs more per request (you pay for every token every
time) and can suffer from the model attending less well to content buried
in the middle of a huge context. RAG scales to corpora far larger than any
context window and keeps per-request cost proportional to what's actually
relevant. Fine-tuning is a different axis entirely — it changes model
*behavior/style*, not *knowledge access*; it's the wrong tool for "answer
questions about docs that change weekly."

## Extending this

- ~~Swap TF-IDF for real embeddings~~ — done, see
  [Retrieval backends](#retrieval-backends): `rag/index_dense.py` (local)
  and `rag/index_voyage.py` (hosted) both implement the same `Index`
  interface as `TfidfIndex`, selected via `RAG_BACKEND`.
- Add **hybrid search**: run TF-IDF and an embedding backend in parallel,
  merge their rankings with reciprocal rank fusion. `compare_backends.py`
  already computes both sets of scores side by side — fusing them instead
  of just printing them is the natural next step.
- ~~Add a reranking pass between `retrieve()` and `generate_answer()`~~ —
  done, see `rag/rerank.py` and `RAG_RERANK=1`.
- Swap the in-memory matrix for a persistent vector DB so the index
  survives restarts and scales past what fits in memory.
- Add citations validation: parse the `[source]` tags the model outputs
  and verify each one was actually in the retrieved set.
