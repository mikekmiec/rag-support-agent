"""Interactive support agent. Run with: python agent.py

Unlike main.py (one retrieval, one answer), Claude drives the tools here:
it decides when and what to search, can read whole documents, and can open
a support ticket, which you're asked to approve first. Tool calls print as
they happen; each turn ends with a trace line (model calls, tokens, cost,
latency).

Uses the same RAG_BACKEND / RAG_RERANK settings as main.py.
Try:
  > Does the Team plan get faster support than Starter, and what does it cost?
  > Is there a Linux desktop client?
  > yes, please open a ticket asking for one
"""

from __future__ import annotations

import json
from pathlib import Path

import anthropic

from rag.agent import SupportAgent, TicketStore, ToolCall
from rag.client import make_client
from rag.display import printable
from rag.ingest import load_documents
from rag.pipeline import DOCS_DIR, build_pipeline

TICKETS_PATH = Path(__file__).parent / "data" / "tickets.jsonl"


def show_tool_call(call: ToolCall) -> None:
    status = " (error)" if call.is_error else ""
    if call.approved is False:
        status = " (declined)"
    print(f"  -> {call.name}({json.dumps(call.input)}){status}")


def confirm_action(call: ToolCall) -> bool:
    # The values come from the model, so strip control characters: what you
    # approve has to be exactly what you see.
    print(f"\n  The agent wants to run {call.name}:")
    for key, value in call.input.items():
        print(f"    {key}: {printable(str(value))}")
    try:
        return input("  Approve? [y/N] ").strip().lower() == "y"
    except (EOFError, KeyboardInterrupt):
        return False


def main() -> None:
    client = make_client()  # before building the index: fail fast without a key
    pipeline = build_pipeline()
    documents = {d.source: d.text for d in load_documents(DOCS_DIR)}
    agent = SupportAgent(
        client=client,
        pipeline=pipeline,
        documents=documents,
        tickets=TicketStore(TICKETS_PATH),
        confirm=confirm_action,
        on_tool_call=show_tool_call,
    )

    print(f"Support agent ({pipeline.label}). Tickets are saved to {TICKETS_PATH}.")
    print("Ask about Nimbus Cloud Storage (Ctrl+C to quit)\n")
    while True:
        try:
            question = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not question:
            continue
        try:
            turn = agent.ask(question)
        except anthropic.APIError as e:
            # The agent rolls back the failed turn, so the conversation can
            # continue; ask again or try another question.
            print(f"\n  API error, nothing was done: {e}\n")
            continue
        print(f"\n{printable(turn.text)}\n")
        print(
            f"  [{turn.model_calls} model calls, {len(turn.tool_calls)} tool calls, "
            f"{turn.input_tokens + turn.output_tokens} tokens, ${turn.cost_usd:.4f}, "
            f"{turn.latency_ms / 1000:.1f}s]\n"
        )


if __name__ == "__main__":
    main()
