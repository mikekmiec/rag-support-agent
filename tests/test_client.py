"""Credential handling: the repo never holds a key, so the client must come
from the current user's environment, CLI profile, or a hidden prompt."""

from __future__ import annotations

import pytest

from rag import client as client_mod


@pytest.fixture
def no_credentials(monkeypatch, tmp_path):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    monkeypatch.delenv("ANTHROPIC_PROFILE", raising=False)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))  # no `ant` profile dir


def test_uses_environment_key(monkeypatch, no_credentials):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-key-from-env")
    assert client_mod.make_client().api_key == "fake-key-from-env"


def test_prompts_with_hidden_input_when_interactive(monkeypatch, no_credentials):
    prompts = []
    monkeypatch.setattr(client_mod.sys.stdin, "isatty", lambda: True, raising=False)
    monkeypatch.setattr(client_mod.getpass, "getpass", lambda p: prompts.append(p) or "fake-typed-key")
    assert client_mod.make_client().api_key == "fake-typed-key"
    assert len(prompts) == 1


def test_exits_with_instructions_when_no_key_and_not_interactive(monkeypatch, no_credentials):
    monkeypatch.setattr(client_mod.sys.stdin, "isatty", lambda: False, raising=False)
    with pytest.raises(SystemExit, match="ANTHROPIC_API_KEY"):
        client_mod.make_client()


def test_blank_prompt_input_exits(monkeypatch, no_credentials):
    monkeypatch.setattr(client_mod.sys.stdin, "isatty", lambda: True, raising=False)
    monkeypatch.setattr(client_mod.getpass, "getpass", lambda p: "  ")
    with pytest.raises(SystemExit):
        client_mod.make_client()
