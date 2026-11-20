"""End-to-end answer eval: run each labeled question through the full
pipeline (retrieve -> generate), then grade the answer with an LLM judge
plus deterministic checks. This one calls the Claude API (two calls per
question, one to answer and one to judge), so it costs money; it shows an
estimate and asks before running.

Run from the repo root:
    python -m evals.answer_eval                         # TF-IDF, no reranking
    python -m evals.answer_eval --backend local --rerank
    python -m evals.answer_eval --limit 5               # quick smoke test
    python -m evals.answer_eval --save                  # write per-question JSON

What it measures:
  accuracy        answerable questions the judge grades fully correct
                  against the reference answer
  faithfulness    answers whose every claim is supported by the retrieved
                  context (catches the model filling gaps from training data)
  abstention      off-corpus questions where the model correctly says it
                  doesn't know; the complement is the hallucination rate
  citation valid  answers whose [source] citations all name a document that
                  was actually retrieved (deterministic, no judge)
  latency / cost  per-question wall-clock for retrieval + generation, and
                  total spend from the API's reported token usage

Every per-question verdict is printed (and saved with --save) so you can
read the judge's reasoning. Spot-check it: an eval is only as trustworthy
as its grader.
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Literal

from anthropic import Anthropic
from pydantic import BaseModel, Field

from evals.common import (
    RESULTS_DIR,
    EvalItem,
    load_dataset,
    mean,
    parse_citations,
    percentile,
)
from rag.backends import BACKENDS
from rag.client import make_client
from rag.generate import MODEL as ANSWER_MODEL
from rag.generate import build_user_prompt, generate_answer
from rag.pipeline import RAGPipeline, build_pipeline
from rag.pricing import cost_usd

# A cheaper model than the one being graded is standard for judges; grading
# an answer against a reference and a few excerpts is a much easier task
# than producing it.
JUDGE_MODEL = "claude-sonnet-5-5"
JUDGE_EFFORT = "medium"

# Rough per-question token usage for the up-front estimate only (the
# report uses the real numbers the API returns).
EST_ANSWER_TOKENS = (1_000, 600)
EST_JUDGE_TOKENS = (1_500, 800)

JUDGE_SYSTEM = """You grade answers from a retrieval-augmented support assistant.

You get the question, the context excerpts the assistant was given, a
reference answer written by a human, and the assistant's answer. Grade two
things independently:

correctness, against the reference answer:
- "correct": states the key facts in the reference (numbers, limits,
  timeframes, yes/no) with nothing that contradicts it. Extra accurate
  detail is fine.
- "partially_correct": some key facts right, some missing or wrong.
- "incorrect": the key facts are wrong or missing, or it answers a
  different question.
- "abstained": the assistant says the information isn't available rather
  than answering. Use this whenever it declines, even if declining was
  the right call.

faithful, against the context excerpts only (ignore the reference here):
true if every factual claim in the answer is supported by the excerpts. A
correct-sounding claim that isn't in the excerpts is NOT faithful. An
answer that only says the information isn't available is faithful.

List each unsupported claim, if any. Explain your reasoning briefly first."""


class Verdict(BaseModel):
    reasoning: str = Field(description="Brief explanation, written before the grades.")
    correctness: Literal["correct", "partially_correct", "incorrect", "abstained"]
    faithful: bool
    unsupported_claims: list[str]


@dataclass
class QuestionResult:
    id: str
    category: str
    answerable: bool
    question: str
    retrieved: list[str]  # chunk ids
    answer: str
    answer_model: str
    citations: list[str]
    citations_valid: bool
    cited_anything: bool
    correctness: str
    faithful: bool
    unsupported_claims: list[str]
    judge_reasoning: str
    latency_ms: float
    cost_usd: float
    error: str | None = None  # set when the question couldn't be run at all


def judge(client: Anthropic, item: EvalItem, context: str, answer: str) -> tuple[Verdict, float]:
    prompt = (
        f"<question>{item.question}</question>\n\n"
        f"<context>\n{context}\n</context>\n\n"
        f"<reference_answer>{item.reference_answer}</reference_answer>\n\n"
        f"<assistant_answer>\n{answer}\n</assistant_answer>"
    )
    response = client.messages.parse(
        model=JUDGE_MODEL,
        max_tokens=16000,
        system=JUDGE_SYSTEM,
        messages=[{"role": "user", "content": prompt}],
        output_config={"effort": JUDGE_EFFORT},
        output_format=Verdict,
    )
    spent = cost_usd(JUDGE_MODEL, response.usage.input_tokens, response.usage.output_tokens)
    if response.parsed_output is None:
        # Refusal or truncation: record it rather than crash the run.
        verdict = Verdict(
            reasoning=f"judge returned no verdict (stop_reason={response.stop_reason})",
            correctness="incorrect",
            faithful=False,
            unsupported_claims=[],
        )
        return verdict, spent
    return response.parsed_output, spent


def run_one(client: Anthropic, pipeline: RAGPipeline, item: EvalItem) -> QuestionResult:
    start = time.perf_counter()
    results = pipeline.retrieve(item.question)
    answer = generate_answer(client, item.question, results)
    latency_ms = (time.perf_counter() - start) * 1000

    retrieved_sources = {r.chunk.source for r in results}
    citations = parse_citations(answer.text)
    # The judge sees exactly the context the model saw (minus the question).
    context = build_user_prompt(item.question, results).split("\n\nQuestion:")[0]
    verdict, judge_cost = judge(client, item, context, answer.text)

    return QuestionResult(
        id=item.id,
        category=item.category,
        answerable=item.answerable,
        question=item.question,
        retrieved=[r.chunk.id for r in results],
        answer=answer.text,
        answer_model=answer.model,
        citations=sorted(citations),
        citations_valid=citations <= retrieved_sources,
        cited_anything=bool(citations),
        correctness=verdict.correctness,
        faithful=verdict.faithful,
        unsupported_claims=verdict.unsupported_claims,
        judge_reasoning=verdict.reasoning,
        latency_ms=latency_ms,
        cost_usd=cost_usd(answer.model, answer.input_tokens, answer.output_tokens) + judge_cost,
    )


def run_one_safely(client: Anthropic, pipeline: RAGPipeline, item: EvalItem) -> QuestionResult:
    """Like run_one, but an API error becomes an error row instead of
    aborting the whole run (and discarding results already paid for)."""
    try:
        return run_one(client, pipeline, item)
    except Exception as e:  # noqa: BLE001 - recorded and reported, not swallowed
        return QuestionResult(
            id=item.id,
            category=item.category,
            answerable=item.answerable,
            question=item.question,
            retrieved=[],
            answer="",
            answer_model="",
            citations=[],
            citations_valid=False,
            cited_anything=False,
            correctness="error",
            faithful=False,
            unsupported_claims=[],
            judge_reasoning="",
            latency_ms=0.0,
            cost_usd=0.0,
            error=f"{type(e).__name__}: {e}",
        )


def summarize(all_results: list[QuestionResult]) -> dict[str, float]:
    # Errored questions are excluded from the quality rates and counted
    # separately, so an outage can't masquerade as a model regression.
    results = [r for r in all_results if r.error is None]
    answerable = [r for r in results if r.answerable]
    off_corpus = [r for r in results if not r.answerable]
    answered = [r for r in results if r.correctness != "abstained"]

    def rate(rows: list[QuestionResult], pred) -> float:
        return mean([1.0 if pred(r) else 0.0 for r in rows])

    latencies = [r.latency_ms for r in results]
    return {
        "accuracy": rate(answerable, lambda r: r.correctness == "correct"),
        "partially_correct": rate(answerable, lambda r: r.correctness == "partially_correct"),
        "wrongly_abstained": rate(answerable, lambda r: r.correctness == "abstained"),
        "off_corpus_abstention": rate(off_corpus, lambda r: r.correctness == "abstained"),
        "faithfulness": rate(results, lambda r: r.faithful),
        "citation_valid": rate(results, lambda r: r.citations_valid),
        "answered_without_citation": rate(answered, lambda r: not r.cited_anything),
        "latency_p50_ms": percentile(latencies, 50),
        "latency_p95_ms": percentile(latencies, 95),
        "total_cost_usd": sum(r.cost_usd for r in results),
        "errors": float(len(all_results) - len(results)),
    }


def print_report(label: str, results: list[QuestionResult], summary: dict[str, float]) -> None:
    print(f"\nPer question ({label})")
    for r in results:
        if r.error:
            print(f"  {r.id:<28} ERROR {r.error}")
            continue
        flags = []
        if not r.faithful:
            flags.append("UNFAITHFUL")
        if not r.citations_valid:
            flags.append(f"BAD CITATION {r.citations}")
        expected = "answer" if r.answerable else "abstain"
        print(f"  {r.id:<28} expected={expected:<7} got={r.correctness:<17} {' '.join(flags)}")
        if r.correctness not in ("correct", "abstained") or flags:
            print(f"      judge: {r.judge_reasoning}")

    def pct(key: str) -> str:
        return f"{summary[key] * 100:.0f}%"

    print(f"\nSummary ({label}, {len(results)} questions)")
    print(
        f"  accuracy (answerable)        {pct('accuracy')}  "
        f"(+{pct('partially_correct')} partial, {pct('wrongly_abstained')} wrongly abstained)"
    )
    print(f"  off-corpus abstention        {pct('off_corpus_abstention')}")
    print(f"  faithfulness                 {pct('faithfulness')}")
    print(f"  citations valid              {pct('citation_valid')}")
    print(f"  answered without a citation  {pct('answered_without_citation')}")
    print(
        f"  latency p50 / p95            {summary['latency_p50_ms']:.0f} / "
        f"{summary['latency_p95_ms']:.0f} ms"
    )
    print(f"  total cost                   ${summary['total_cost_usd']:.3f}")
    if summary["errors"]:
        print(f"  ERRORS                       {summary['errors']:.0f} questions failed to run")
        print("                               (excluded from the rates above)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", maxsplit=1)[0])
    parser.add_argument("--backend", choices=BACKENDS, default="tfidf")
    parser.add_argument("--rerank", action="store_true")
    parser.add_argument("--limit", type=int, help="only run the first N questions")
    parser.add_argument("--ids", nargs="+", help="only run these question ids")
    parser.add_argument("--workers", type=int, default=4, help="questions run in parallel")
    parser.add_argument("--save", action="store_true", help="write results JSON to evals/results/")
    parser.add_argument("--yes", action="store_true", help="skip the cost confirmation")
    args = parser.parse_args()

    items = load_dataset()
    if args.ids:
        unknown = set(args.ids) - {i.id for i in items}
        if unknown:
            raise SystemExit(f"Unknown ids: {sorted(unknown)}")
        items = [i for i in items if i.id in args.ids]
    if args.limit:
        items = items[: args.limit]

    estimate = len(items) * (
        cost_usd(ANSWER_MODEL, *EST_ANSWER_TOKENS) + cost_usd(JUDGE_MODEL, *EST_JUDGE_TOKENS)
    )
    print(f"{len(items)} questions x 2 API calls ({ANSWER_MODEL} answers, {JUDGE_MODEL} judges)")
    print(f"Estimated cost: ~${estimate:.2f}")
    if not args.yes and input("Continue? [y/N] ").strip().lower() != "y":
        raise SystemExit("Aborted.")

    client = make_client()  # before building the index: fail fast without a key
    pipeline = build_pipeline(backend=args.backend, rerank=args.rerank)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(lambda item: run_one_safely(client, pipeline, item), items))

    summary = summarize(results)
    print_report(pipeline.label, results, summary)

    if args.save:
        RESULTS_DIR.mkdir(exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        rerank = "-rerank" if args.rerank else ""
        path = RESULTS_DIR / f"answers-{args.backend}{rerank}-{stamp}.json"
        payload = {
            "config": pipeline.label,
            "answer_model": ANSWER_MODEL,
            "judge_model": JUDGE_MODEL,
            "summary": summary,
            "questions": [asdict(r) for r in results],
        }
        path.write_text(json.dumps(payload, indent=2))
        print(f"\nSaved {path}")


if __name__ == "__main__":
    main()
