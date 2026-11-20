"""Agent loop tests. A scripted fake client plays the model's side, so these
check the orchestration (tool dispatch, approval, errors, step budget,
history) without network calls."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from rag.agent import SupportAgent, TicketStore, tool_definitions
from rag.display import printable
from rag.generate import REFUSAL_MESSAGE
from rag.index import TfidfIndex
from rag.ingest import Chunk
from rag.pipeline import RAGPipeline
from rag.retrieve import Retriever

DOCS = {
    "security": "After 5 consecutive failed login attempts, an account is locked for 30 minutes.",
    "pricing": "The Team plan is $29 per month and includes 2TB of storage.",
}


def text(t):
    return SimpleNamespace(type="text", text=t)


def tool_use(id_, name, **input_):
    return SimpleNamespace(type="tool_use", id=id_, name=name, input=input_)


def response(stop_reason, *content):
    return SimpleNamespace(
        stop_reason=stop_reason,
        content=list(content),
        model="claude-opus-5-5",
        usage=SimpleNamespace(input_tokens=1000, output_tokens=100),
    )


class ScriptedClient:
    """Returns the scripted responses in order; records every request."""

    def __init__(self, *responses):
        self.responses = list(responses)
        self.requests = []
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        # Snapshot messages: the agent keeps appending to the same list.
        self.requests.append({**kwargs, "messages": list(kwargs["messages"])})
        return self.responses.pop(0)


@pytest.fixture
def pipeline():
    chunks = [Chunk(id=f"{s}#0", source=s, text=t) for s, t in DOCS.items()]
    return RAGPipeline(retriever=Retriever(TfidfIndex(chunks), min_score=0.05))


def make_agent(client, pipeline, tmp_path, **kwargs):
    return SupportAgent(
        client=client,
        pipeline=pipeline,
        documents=DOCS,
        tickets=TicketStore(tmp_path / "tickets.jsonl"),
        **kwargs,
    )


def tool_results(request):
    """The tool_result blocks in a request's last message."""
    return request["messages"][-1]["content"]


def test_searches_then_answers(pipeline, tmp_path):
    client = ScriptedClient(
        response("tool_use", tool_use("t1", "search_docs", query="failed login lockout")),
        response("end_turn", text("Locked for 30 minutes [security].")),
    )
    turn = make_agent(client, pipeline, tmp_path).ask("How long is the lockout?")

    assert turn.text == "Locked for 30 minutes [security]."
    assert [c.name for c in turn.tool_calls] == ["search_docs"]
    assert turn.model_calls == 2
    assert turn.input_tokens == 2000
    (result,) = tool_results(client.requests[1])
    assert result["tool_use_id"] == "t1"
    assert 'source="security"' in result["content"]
    assert "locked for 30 minutes" in result["content"]


def test_parallel_tool_calls_return_in_one_message(pipeline, tmp_path):
    client = ScriptedClient(
        response(
            "tool_use",
            tool_use("a", "search_docs", query="lockout"),
            tool_use("b", "read_document", source="pricing"),
        ),
        response("end_turn", text("done")),
    )
    make_agent(client, pipeline, tmp_path).ask("q")
    assert [r["tool_use_id"] for r in tool_results(client.requests[1])] == ["a", "b"]


def test_ticket_requires_approval_and_is_created_when_approved(pipeline, tmp_path):
    seen = []
    client = ScriptedClient(
        response(
            "tool_use",
            tool_use(
                "t1",
                "create_support_ticket",
                subject="Linux client",
                severity="low",
                description="Customer wants a Linux desktop client.",
            ),
        ),
        response("end_turn", text("Opened NIM-0001.")),
    )
    agent = make_agent(client, pipeline, tmp_path, confirm=lambda c: seen.append(c) or True)
    turn = agent.ask("Please open a ticket")

    assert len(seen) == 1 and seen[0].input["subject"] == "Linux client"
    assert turn.tool_calls[0].approved is True
    assert [t["id"] for t in agent.tickets.all()] == ["NIM-0001"]
    assert "Created ticket NIM-0001" in tool_results(client.requests[1])[0]["content"]


def test_declined_ticket_is_not_created_and_model_is_told(pipeline, tmp_path):
    client = ScriptedClient(
        response(
            "tool_use",
            tool_use("t1", "create_support_ticket", subject="s", severity="low", description="d"),
        ),
        response("end_turn", text("Okay, no ticket.")),
    )
    agent = make_agent(client, pipeline, tmp_path, confirm=lambda c: False)
    turn = agent.ask("q")

    assert agent.tickets.all() == []
    assert turn.tool_calls[0].approved is False
    result = tool_results(client.requests[1])[0]
    assert "declined" in result["content"] and "is_error" not in result


def test_actions_are_denied_by_default(pipeline, tmp_path):
    client = ScriptedClient(
        response(
            "tool_use",
            tool_use("t1", "create_support_ticket", subject="s", severity="low", description="d"),
        ),
        response("end_turn", text("ok")),
    )
    agent = make_agent(client, pipeline, tmp_path)  # no confirm callback
    agent.ask("q")
    assert agent.tickets.all() == []


def test_search_tool_never_asks_for_approval(pipeline, tmp_path):
    def confirm(call):
        raise AssertionError("read-only tools must not prompt")

    client = ScriptedClient(
        response("tool_use", tool_use("t1", "search_docs", query="team plan")),
        response("end_turn", text("ok")),
    )
    turn = make_agent(client, pipeline, tmp_path, confirm=confirm).ask("q")
    assert turn.tool_calls[0].approved is None


def test_tool_errors_go_back_to_the_model(pipeline, tmp_path):
    client = ScriptedClient(
        response(
            "tool_use",
            tool_use("t1", "read_document", source="nonexistent"),
            tool_use("t2", "delete_everything"),
        ),
        response("end_turn", text("Sorry, I couldn't find that.")),
    )
    turn = make_agent(client, pipeline, tmp_path).ask("q")

    results = tool_results(client.requests[1])
    assert all(r["is_error"] for r in results)
    assert "unknown document" in results[0]["content"]
    assert "unknown tool" in results[1]["content"]
    assert turn.text == "Sorry, I couldn't find that."


def test_step_limit_stops_a_looping_model(pipeline, tmp_path):
    looping = [response("tool_use", tool_use(f"t{i}", "search_docs", query="x")) for i in range(3)]
    client = ScriptedClient(*looping)
    turn = make_agent(client, pipeline, tmp_path, max_steps=3).ask("q")

    assert turn.hit_step_limit
    assert turn.model_calls == 3
    assert "wasn't able to finish" in turn.text


def test_refusal_returns_message_and_skips_empty_assistant_turn(pipeline, tmp_path):
    client = ScriptedClient(response("refusal"), response("end_turn", text("Hi again")))
    agent = make_agent(client, pipeline, tmp_path)

    assert agent.ask("q").text == REFUSAL_MESSAGE
    agent.ask("follow-up")
    # No empty assistant message was appended between the two user turns.
    assert [m["role"] for m in client.requests[1]["messages"]] == ["user", "user"]


def test_history_carries_across_turns(pipeline, tmp_path):
    client = ScriptedClient(
        response("end_turn", text("The Team plan is $29 [pricing].")),
        response("end_turn", text("Yes, I can open a ticket.")),
    )
    agent = make_agent(client, pipeline, tmp_path)
    agent.ask("How much is Team?")
    agent.ask("Can you open a ticket about it?")

    roles = [m["role"] for m in client.requests[1]["messages"]]
    assert roles == ["user", "assistant", "user"]


def test_tool_schemas_are_strict_and_list_documents():
    tools = {t["name"]: t for t in tool_definitions(["pricing", "security"])}
    assert all(t["strict"] for t in tools.values())
    assert all(t["input_schema"]["additionalProperties"] is False for t in tools.values())
    assert tools["read_document"]["input_schema"]["properties"]["source"]["enum"] == [
        "pricing",
        "security",
    ]


@pytest.mark.parametrize(
    "kwargs,error",
    [
        ({"subject": "s", "severity": "urgent", "description": "d"}, "severity"),
        ({"subject": "", "severity": "low", "description": "d"}, "subject"),
        ({"subject": "x" * 201, "severity": "low", "description": "d"}, "subject"),
        ({"subject": "s", "severity": "low", "description": "x" * 5001}, "description"),
    ],
)
def test_ticket_store_rejects_invalid_input(tmp_path, kwargs, error):
    store = TicketStore(tmp_path / "t.jsonl")
    with pytest.raises(ValueError, match=error):
        store.create(**kwargs)
    assert store.all() == []


def test_printable_strips_terminal_escapes_but_keeps_newlines():
    hidden = "Subject: refund\x1b[2K\x1b[1A\rSubject: harmless\nline two\t."
    assert printable(hidden) == "Subject: refund[2K[1ASubject: harmless\nline two\t."


class FailingClient(ScriptedClient):
    """Scripted responses, then raises on the next call."""

    def _create(self, **kwargs):
        if not self.responses:
            raise RuntimeError("API unavailable")
        return super()._create(**kwargs)


def test_failed_turn_is_rolled_back_so_history_stays_valid(pipeline, tmp_path):
    client = FailingClient(
        response("end_turn", text("first answer")),
        response("tool_use", tool_use("t1", "search_docs", query="x")),
        # the call after the tool result fails
    )
    agent = make_agent(client, pipeline, tmp_path)
    agent.ask("first question")
    history_before = list(agent.messages)

    with pytest.raises(RuntimeError):
        agent.ask("second question")
    assert agent.messages == history_before  # no dangling user msg or tool_use


def test_tool_call_cut_off_by_max_tokens_gets_an_error_result(pipeline, tmp_path):
    client = ScriptedClient(
        response("max_tokens", text("Let me check"), tool_use("t1", "search_docs", query="x")),
    )
    agent = make_agent(client, pipeline, tmp_path)
    turn = agent.ask("q")

    assert "truncated" in turn.text
    assert turn.tool_calls == []  # the cut-off call was not executed
    last = agent.messages[-1]
    assert last["role"] == "user"
    assert last["content"][0]["tool_use_id"] == "t1"
    assert last["content"][0]["is_error"] is True


def test_requests_enable_prompt_caching_and_fallbacks(pipeline, tmp_path):
    client = ScriptedClient(response("end_turn", text("ok")))
    make_agent(client, pipeline, tmp_path).ask("q")
    request = client.requests[0]
    assert request["cache_control"] == {"type": "ephemeral"}
    assert request["fallbacks"] == "default"
