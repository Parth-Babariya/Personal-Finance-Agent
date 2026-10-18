"""Deterministic keyword planner used when no LLM is configured or the LLM fails.

It maps a message to tool calls with simple pattern matching, runs them through
the same ``ToolRegistry`` the LLM uses, and renders a templated reply.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from src.core.types import CATEGORIES
from src.tools import ToolContext, ToolOutcome, ToolRegistry

# Ordered longest-first so "last 7 days" wins over "last".
_PERIOD_PHRASES: list[tuple[str, str]] = [
    ("last 30 days", "last_30_days"),
    ("past 30 days", "last_30_days"),
    ("last 7 days", "last_7_days"),
    ("past 7 days", "last_7_days"),
    ("previous month", "last_month"),
    ("last month", "last_month"),
    ("this month", "this_month"),
    ("past week", "last_7_days"),
    ("last week", "last_7_days"),
    ("this week", "this_week"),
    ("this year", "this_year"),
    ("last year", "last_year"),
    ("all time", "all_time"),
    ("ever", "all_time"),
    ("today", "today"),
]

_CATEGORY_SYNONYMS: dict[str, tuple[str, ...]] = {
    "food": ("food", "restaurant", "dining", "eating out", "swiggy", "zomato", "lunch", "dinner", "coffee"),
    "groceries": ("grocery", "groceries", "supermarket", "blinkit", "zepto", "bigbasket"),
    "transport": ("transport", "uber", "ola", "cab", "taxi", "fuel", "petrol", "metro", "bus", "commute"),
    "shopping": ("shopping", "amazon", "flipkart", "clothes", "myntra"),
    "entertainment": ("entertainment", "movie", "movies", "concert", "games"),
    "utilities": ("utilities", "electricity", "water bill", "internet", "wifi", "phone bill", "recharge"),
    "rent": ("rent",),
    "health": ("health", "medical", "medicine", "doctor", "pharmacy", "gym"),
    "education": ("education", "course", "books", "tuition"),
    "travel": ("travel", "flight", "hotel", "trip", "vacation"),
    "subscriptions": ("subscription", "subscriptions", "spotify", "netflix"),
    "salary": ("salary", "paycheck"),
    "freelance": ("freelance",),
}

_ADD_PATTERN = re.compile(
    r"\b(?:spent|paid|bought|received|earned|got)\b\s*(?:rs\.?|inr|₹|\$)?\s*(\d+(?:[.,]\d+)*)",
    re.IGNORECASE,
)
_FINANCE_WORDS = (
    "spend", "spent", "expense", "money", "income", "earn", "save", "saving", "budget",
    "cost", "how much", "transaction", "summary", "balance", "net", "paid",
)
_FOLLOW_UP = re.compile(r"^\s*(what about|how about|and|same for|what of)\b", re.IGNORECASE)

HELP_TEXT = (
    "I can answer questions about your spending and income, for example: "
    "'How much did I spend on food this month?', 'Compare this month with last month', "
    "'Show my biggest expenses', or 'I spent 450 on groceries'."
)


@dataclass
class _Parsed:
    periods: list[str]
    category: str | None


def _find_periods(text: str) -> list[str]:
    hits: list[tuple[int, str]] = []
    taken: list[range] = []
    for phrase, period in _PERIOD_PHRASES:
        for match in re.finditer(rf"\b{re.escape(phrase)}\b", text):
            span = range(match.start(), match.end())
            if any(span.start in r or r.start in span for r in taken):
                continue
            taken.append(span)
            hits.append((match.start(), period))
    return [period for _, period in sorted(hits)]


def _find_category(text: str) -> str | None:
    for category, words in _CATEGORY_SYNONYMS.items():
        if any(re.search(rf"\b{re.escape(w)}\b", text) for w in words):
            return category
    for category in CATEGORIES:
        if re.search(rf"\b{category}\b", text):
            return category
    return None


class KeywordFallbackPlanner:
    def __init__(self, registry: ToolRegistry, currency: str):
        self._registry = registry
        self._cur = currency

    def plan(self, message: str, history: list[dict[str, Any]] | None = None) -> list[tuple[str, dict[str, Any]]]:
        text = message.lower()
        parsed = _Parsed(_find_periods(text), _find_category(text))

        # Follow-ups like "what about last month?" inherit context from the last user turn.
        if _FOLLOW_UP.search(text) and history:
            previous = next((m["content"] for m in reversed(history) if m["role"] == "user"), "")
            prior_plan = self.plan(previous) if previous else []
            if prior_plan and prior_plan[0][0] != "add_transaction":
                name, args = prior_plan[0]
                period_key = "period_a" if name == "compare_periods" else "period"
                return [(name, _with(
                    args,
                    **{period_key: parsed.periods[0] if parsed.periods else None},
                    category=parsed.category,
                ))]

        add = _ADD_PATTERN.search(text)
        if add:
            amount = float(add.group(1).replace(",", ""))
            is_income = re.search(r"\b(received|earned|got)\b", text) is not None
            category = parsed.category or ("salary" if is_income else "other")
            return [("add_transaction", {
                "amount": amount,
                "type": "income" if is_income else "expense",
                "category": category,
                "description": message.strip()[:200],
            })]

        period = parsed.periods[0] if parsed.periods else None
        category = parsed.category

        if re.search(r"\b(compare|vs|versus|compared|difference|than)\b", text):
            a = parsed.periods[0] if parsed.periods else "this_month"
            b = parsed.periods[1] if len(parsed.periods) > 1 else ("last_month" if a != "last_month" else "this_month")
            return [("compare_periods", _with({"period_a": a, "period_b": b}, category=category))]

        feature = self._feature_plan(text, period, category)
        if feature:
            return feature

        if re.search(r"\b(top|biggest|largest|highest|most expensive|big)\b", text):
            return [("get_top_expenses", _with({"period": period or "this_month"}, category=category))]

        if re.search(r"\b(list|show|find|search|transactions|history)\b", text):
            return [("search_transactions", _with({"period": period or "all_time", "limit": 10}, category=category))]

        if re.search(r"\b(insight|insights|advice|tips|habits|analy[sz]e|analysis)\b", text):
            return [("get_insights", {"period": period or "this_month"})]

        if not (period or category or any(w in text for w in _FINANCE_WORDS)):
            return []

        plan = [("get_spending_summary", _with({"period": period or "this_month"}, category=category))]
        if category is None:
            plan.append(("get_insights", {"period": period or "this_month"}))
        return plan

    def _feature_plan(
        self, text: str, period: str | None, category: str | None
    ) -> list[tuple[str, dict[str, Any]]]:
        """Budgets, analytics and goals; only planned when the tool is registered."""
        asks_amount = re.search(r"\b(how much|spend|spent|spending)\b", text)
        candidates: list[tuple[bool, str, dict[str, Any]]] = [
            (bool(re.search(r"\bbudgets?\b", text)), "check_budget", _with({}, category=category)),
            (bool(re.search(r"\b(recurring|regular payments|regular bills|bills)\b", text)
                  or (re.search(r"\bsubscriptions?\b", text) and not asks_amount)),
             "find_recurring_payments", {}),
            (bool(re.search(r"\b(unusual|suspicious|anomal\w*|strange|odd|out of the ordinary)\b", text)),
             "find_unusual_spending", {"period": period or "last_30_days"}),
            (bool(re.search(r"\b(trend|trends|over time|month by month|each month|monthly|lately)\b", text)),
             "get_monthly_trend", _with({"months": 6}, category=category)),
            (bool(re.search(r"\b(goal|goals|saving for|save for)\b", text)), "get_savings_goals", {}),
        ]
        for matched, name, args in candidates:
            if matched and self._registry.get(name) is not None:
                return [(name, args)]
        return []

    async def respond(
        self, message: str, ctx: ToolContext, history: list[dict[str, Any]] | None = None
    ) -> tuple[str, list[ToolOutcome]]:
        plan = self.plan(message, history)
        if not plan:
            return HELP_TEXT, []
        outcomes = [await self._registry.execute(name, args, ctx) for name, args in plan]
        parts = [self._render(o) for o in outcomes]
        return "\n\n".join(p for p in parts if p), outcomes

    # --- rendering ---

    def _m(self, amount: float) -> str:
        return f"{self._cur}{amount:,.2f}"

    def _render(self, o: ToolOutcome) -> str:
        r = o.result
        if not o.ok:
            return f"Sorry, I couldn't complete that: {r.get('error', 'unknown error')}"
        if o.name == "get_spending_summary":
            scope = f" on {r['category']}" if r.get("category") else ""
            lines = [f"You spent {self._m(r['total_expense'])}{scope} ({r['period']}), "
                     f"across {_plural(r['transaction_count'], 'transaction')}."]
            if r["total_income"]:
                lines.append(f"Income: {self._m(r['total_income'])}, net: {self._m(r['net'])}.")
            if not r.get("category") and r["by_category"]:
                top = list(r["by_category"].items())[:5]
                lines.append("By category: " + ", ".join(f"{c} {self._m(v)}" for c, v in top) + ".")
            return " ".join(lines)
        if o.name == "compare_periods":
            a, b = r["a"], r["b"]
            direction = "more" if r["expense_change"] > 0 else "less"
            pct = f" ({abs(r['expense_change_pct']):.1f}%)" if r["expense_change_pct"] is not None else ""
            movers = sorted(r["by_category"].items(), key=lambda kv: abs(kv[1]["change"]), reverse=True)[:3]
            mover_text = ", ".join(
                f"{c} {'+' if v['change'] >= 0 else '-'}{self._m(abs(v['change']))}" for c, v in movers
            )
            text = (f"{a['period'].replace('_', ' ').capitalize()}: {self._m(a['total_expense'])} vs "
                    f"{b['period'].replace('_', ' ')}: {self._m(b['total_expense'])} — "
                    f"{self._m(abs(r['expense_change']))} {direction}{pct}.")
            return text + (f" Biggest changes: {mover_text}." if mover_text else "")
        if o.name in {"search_transactions", "get_top_expenses"}:
            rows = r["transactions"]
            if not rows:
                return f"No matching transactions ({r['period']})."
            head = (f"Found {r['match_count']} transactions totalling {self._m(r['total_amount'])} ({r['period']}):"
                    if o.name == "search_transactions" else f"Your largest expenses ({r['period']}):")
            body = "\n".join(f"- {t['date']}: {self._m(t['amount'])} · {t['category']}"
                             + (f" · {t['description']}" if t["description"] else "") for t in rows)
            return f"{head}\n{body}"
        if o.name == "get_insights":
            return "\n".join(f"- {i}" for i in r["insights"])
        if o.name == "add_transaction":
            t = r.get("transaction") or r["added"]
            what = f"{t['type']} of {self._m(t['amount'])} in '{t['category']}' on {t['date']}"
            dup = " Note: a matching transaction already exists that day." if r.get("possible_duplicates") else ""
            if r.get("status") == "pending_confirmation":
                return f"I'll record an {what}.{dup} Reply 'yes' to confirm or 'no' to cancel."
            return f"Recorded {what}.{dup}"
        if o.name == "check_budget":
            if not r["budgets"]:
                return r["note"] + " You can set one in the Budgets view."
            lines = []
            for b in r["budgets"]:
                line = (f"- {b['category']}: {self._m(b['spent'])} of {self._m(b['limit'])} "
                        f"({b['used_pct']:.0f}%, {b['state']})")
                if b.get("projected_state") == "over" and b["state"] != "over":
                    line += f"; on pace for {self._m(b['projected_month_end'])} by month end"
                lines.append(line)
            return f"Budgets for {r['budgets'][0]['month']}:\n" + "\n".join(lines)
        if o.name == "get_monthly_trend":
            months = ", ".join(f"{m['month']} {self._m(m['expense'])}" + (" (so far)" if m["partial"] else "")
                               for m in r["months"])
            scope = f" on {r['category']}" if r.get("category") else ""
            avg = r["average_monthly_expense"]
            return f"Monthly spending{scope}: {months}." + (f" Average: {self._m(avg)}." if avg is not None else "")
        if o.name == "find_unusual_spending":
            if not r["unusual"]:
                return f"Nothing unusual ({r['period']})."
            body = "\n".join(f"- {t['date']}: {self._m(t['amount'])} · {t['description'] or t['category']} "
                             f"({t['reason']})" for t in r["unusual"])
            return f"Unusual expenses ({r['period']}):\n{body}"
        if o.name == "find_recurring_payments":
            if not r["recurring"]:
                return "I didn't find any recurring payments yet (needs at least 3 months of history)."
            body = "\n".join(f"- {g['description']}: {self._m(g['typical_amount'])}/month, next around "
                             f"{g['next_expected']}" for g in r["recurring"])
            return f"Recurring payments (about {self._m(r['estimated_monthly_total'])}/month):\n{body}"
        if o.name == "get_savings_goals":
            if not r["goals"]:
                return r["note"]
            lines = []
            for g in r["goals"]:
                line = f"- {g['name']}: {self._m(g['saved_amount'])} of {self._m(g['target_amount'])} " \
                       f"({g['progress_pct']:.0f}%)"
                if g["completed"]:
                    line += ", completed"
                elif g["required_per_month"] is not None:
                    status = "on track" if g["on_track"] else "behind"
                    line += f", needs {self._m(g['required_per_month'])}/month, {status}"
                lines.append(line)
            return "Savings goals:\n" + "\n".join(lines)
        return ""


def _plural(n: int, word: str) -> str:
    return f"{n} {word}" + ("" if n == 1 else "s")


def _with(args: dict[str, Any], **extra: Any) -> dict[str, Any]:
    merged = dict(args)
    merged.update({k: v for k, v in extra.items() if v is not None})
    return merged
