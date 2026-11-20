"""Safe terminal output for untrusted text."""

from __future__ import annotations

import re

_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b-\x1f\x7f-\x9f]")


def printable(text: str) -> str:
    """Strip terminal control characters (keeping newlines and tabs).

    Model output and tool inputs are untrusted text. An ANSI escape
    sequence in them could clear the screen, recolor, or overwrite lines,
    for example to hide part of a ticket in the approval prompt. Apply
    this to anything printed to a terminal.
    """
    return _CONTROL_CHARS.sub("", text)
