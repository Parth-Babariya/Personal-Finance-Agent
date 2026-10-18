"""Mask sensitive identifiers before text reaches the LLM provider or the database.

Patterns are deliberately conservative so ordinary amounts are not masked:
card numbers must pass the Luhn check, and bare account numbers are only
masked when introduced by an "account"/"a/c" keyword.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

_CARD = re.compile(r"(?<!\d)(?:\d[ -]?){12,18}\d(?!\d)")
_AADHAAR = re.compile(r"(?<!\d)[2-9]\d{3}[ -]?\d{4}[ -]?\d{4}(?!\d)")
_PAN = re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b")
_ACCOUNT = re.compile(r"(?i)\b(account|acct|acc|a/c)(\s*(?:no\.?|number|#)?\s*[:\-]?\s*)(\d{9,18})\b")
_PHONE = re.compile(r"(?<![\d+])(?:\+91[ -]?|0)?[6-9]\d{4}[ -]?\d{5}(?!\d)")
_EMAIL_OR_UPI = re.compile(r"\b[\w.+-]+@[A-Za-z][\w-]*(?:\.[\w-]+)*\b")


def _luhn_ok(digits: str) -> bool:
    total, parity = 0, len(digits) % 2
    for i, ch in enumerate(digits):
        d = int(ch)
        if i % 2 == parity:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


@dataclass
class RedactionResult:
    text: str
    found: list[str] = field(default_factory=list)  # kinds that were masked

    @property
    def redacted(self) -> bool:
        return bool(self.found)


def redact(text: str) -> RedactionResult:
    found: list[str] = []

    def card(match: re.Match[str]) -> str:
        digits = re.sub(r"\D", "", match.group(0))
        if 13 <= len(digits) <= 19 and _luhn_ok(digits):
            found.append("card")
            return f"[CARD ••••{digits[-4:]}]"
        return match.group(0)

    def tagged(kind: str, replacement: str):
        def _sub(match: re.Match[str]) -> str:
            found.append(kind)
            return replacement
        return _sub

    def account(match: re.Match[str]) -> str:
        found.append("account")
        return f"{match.group(1)}{match.group(2)}[ACCOUNT ••••{match.group(3)[-4:]}]"

    text = _CARD.sub(card, text)
    text = _ACCOUNT.sub(account, text)
    text = _AADHAAR.sub(tagged("aadhaar", "[AADHAAR]"), text)
    text = _PAN.sub(tagged("pan", "[PAN]"), text)
    text = _PHONE.sub(tagged("phone", "[PHONE]"), text)
    text = _EMAIL_OR_UPI.sub(tagged("email_or_upi", "[EMAIL/UPI]"), text)
    return RedactionResult(text, sorted(set(found)))
