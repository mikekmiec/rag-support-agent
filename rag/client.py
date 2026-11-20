"""Create the Anthropic client from the *current user's* credentials.

This repo never contains an API key. Whoever runs the code supplies their
own, in one of three ways, checked in order:

1. `ANTHROPIC_API_KEY` (or `ANTHROPIC_AUTH_TOKEN`) in the environment
2. a profile from `ant auth login`, which the SDK finds on its own
3. typing it at a hidden prompt. It is kept in memory for this process
   only, and never written to disk, logged, or echoed.

If none of those is available (e.g. a non-interactive run with no
environment variable), it exits with setup instructions instead of the
SDK's lower-level authentication error on the first request.
"""

from __future__ import annotations

import getpass
import os
import sys
from pathlib import Path

from anthropic import Anthropic

SETUP_HELP = (
    "No Anthropic API key found. Create one at https://console.anthropic.com, then either:\n"
    "  export ANTHROPIC_API_KEY=...   (in your shell, not in a file in this repo)\n"
    "  ant auth login                 (stores a profile outside the repo)\n"
    "or run this command in an interactive terminal to enter a key for this session only."
)


def _has_env_credentials() -> bool:
    return bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"))


def _has_cli_profile() -> bool:
    # `ant auth login` stores profiles under ~/.config/anthropic/; the SDK
    # reads them automatically when no environment variable is set.
    config_home = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))
    return bool(os.environ.get("ANTHROPIC_PROFILE")) or (config_home / "anthropic").is_dir()


def make_client() -> Anthropic:
    """Return a client using the current user's credentials (see module docstring)."""
    if _has_env_credentials() or _has_cli_profile():
        return Anthropic()
    if sys.stdin.isatty():
        key = getpass.getpass("Anthropic API key (hidden, used for this session only): ").strip()
        if key:
            return Anthropic(api_key=key)
    raise SystemExit(SETUP_HELP)
