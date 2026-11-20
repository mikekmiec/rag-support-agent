"""Model prices, for the cost figures in agent traces and eval reports.

USD per million tokens (input, output), standard API rates. Prompt-cache
reads and writes are billed differently, so with caching on these figures
are an upper bound. Update when pricing changes.
"""

from __future__ import annotations

PRICES: dict[str, tuple[float, float]] = {
    "claude-opus-5-5": (4.00, 20.00),
    "claude-sonnet-5-5": (2.00, 10.00),
    "claude-opus-5": (5.00, 25.00),
    "claude-opus-4-8": (5.00, 25.00),
}

DEFAULT_MODEL = "claude-opus-5-5"


def cost_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    """Cost of one call. Unknown models (e.g. a fallback) use the default's price."""
    in_price, out_price = PRICES.get(model, PRICES[DEFAULT_MODEL])
    return (input_tokens * in_price + output_tokens * out_price) / 1_000_000
