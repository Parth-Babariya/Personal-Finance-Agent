"""Runtime check that money figures in an LLM reply come from tool results.

The system prompt asks the model not to invent numbers; this enforces it. A
figure counts as grounded if it matches (within rounding) a number from this
turn's tool results, from the conversation, or a sum/difference of two tool
numbers (e.g. "₹1,300 less than last month").
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from itertools import combinations
from typing import Any

_ANY_NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?")
_SCALE = {"k": 1e3, "l": 1e5, "lakh": 1e5, "lakhs": 1e5, "lac": 1e5, "cr": 1e7, "crore": 1e7, "crores": 1e7}
_MAX_PAIR_BASE = 150  # cap pairwise derivations for very large tool results


@dataclass
class GroundingReport:
    checked: int = 0
    unverified: list[float] = field(default_factory=list)
    retried: bool = False

    @property
    def verified(self) -> bool:
        return not self.unverified

    def to_dict(self) -> dict[str, Any]:
        return {"checked": self.checked, "verified": self.verified, "unverified": self.unverified,
                "retried": self.retried}


def _money_pattern(currency: str) -> re.Pattern[str]:
    symbols = "|".join(re.escape(s) for s in {currency, "₹", "$", "Rs.", "Rs", "INR"})
    return re.compile(
        rf"(?:{symbols})\s?(\d[\d,]*(?:\.\d+)?)\s?(k|lakhs?|lac|l|crores?|cr)?\b", re.IGNORECASE
    )


def extract_money(text: str, currency: str) -> list[tuple[float, bool]]:
    """Return (value, abbreviated) for each currency-marked amount in ``text``."""
    out = []
    for raw, scale in _money_pattern(currency).findall(text):
        value = float(raw.replace(",", ""))
        if scale:
            value *= _SCALE[scale.lower()]
        out.append((value, bool(scale)))
    return out


def _numbers_in(obj: Any) -> list[float]:
    if isinstance(obj, bool):
        return []
    if isinstance(obj, int | float):
        return [float(obj)]
    if isinstance(obj, dict):
        return [n for v in obj.values() for n in _numbers_in(v)]
    if isinstance(obj, list | tuple):
        return [n for v in obj for n in _numbers_in(v)]
    return []


def allowed_numbers(tool_results: list[dict[str, Any]], context_texts: list[str]) -> set[float]:
    base = {abs(round(n, 2)) for result in tool_results for n in _numbers_in(result)}
    allowed = set(base)
    largest = sorted(base, reverse=True)[:_MAX_PAIR_BASE]
    for a, b in combinations(largest, 2):
        allowed.add(round(a + b, 2))
        allowed.add(round(abs(a - b), 2))
    for text in context_texts:
        allowed.update(float(n.replace(",", "")) for n in _ANY_NUMBER.findall(text))
    return allowed


def check_grounding(
    reply: str, tool_results: list[dict[str, Any]], context_texts: list[str], currency: str
) -> GroundingReport:
    figures = extract_money(reply, currency)
    if not figures:
        return GroundingReport()
    allowed = allowed_numbers(tool_results, context_texts)
    unverified = [value for value, abbreviated in figures if not _matches(value, abbreviated, allowed)]
    return GroundingReport(checked=len(figures), unverified=unverified)


def _matches(value: float, abbreviated: bool, allowed: set[float]) -> bool:
    # Allow rounding to whole currency units; "₹45k"-style figures get 5% slack.
    return any(abs(a - value) <= (max(0.51, 0.05 * a) if abbreviated else 0.51) for a in allowed)
