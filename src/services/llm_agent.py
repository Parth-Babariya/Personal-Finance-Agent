"""LLM-orchestrated finance agent with deterministic fallback routing."""

from __future__ import annotations

import json
import os
from typing import Any

from .insight_generator import SimpleInsightGenerator
from .memory_manager import SessionMemory
from .sql_transaction_service import SQLTransactionService
from .tools import CategoryBreakdownTool, SpendingSummaryTool


class FinanceLLMAgent:
    """Routes user requests to tools and summarizes results.

    If OpenAI credentials are configured, a model can choose actions.
    Otherwise, a deterministic keyword planner is used.
    """

    def __init__(
        self,
        tx_service: SQLTransactionService,
        memory: SessionMemory,
        insight_gen: SimpleInsightGenerator,
    ) -> None:
        self._tx_service = tx_service
        self._memory = memory
        self._insight_gen = insight_gen
        self._summary_tool = SpendingSummaryTool(tx_service)
        self._breakdown_tool = CategoryBreakdownTool(tx_service)
        self._model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    async def respond(self, message: str, user_id: str) -> dict[str, Any]:
        """Generate an agent response with tool orchestration."""
        actions = self._plan_actions(message)

        insights: list[str] = []
        evidence: list[str] = []

        transactions = self._tx_service.get_transactions(limit=0, user_id=user_id)

        for action in actions:
            if action == "insights":
                insights = await self._insight_gen.generate_insights(transactions)
                if insights:
                    evidence.extend(insights)
            elif action == "spending_summary":
                summary = await self._summary_tool.execute(period="month", user_id=user_id)
                evidence.append(summary)
            elif action == "category_breakdown":
                breakdown = await self._breakdown_tool.execute(user_id=user_id)
                evidence.append(breakdown)
            elif action == "memory_search":
                mem = self._memory.search(message, limit=3, namespace=user_id)
                if mem:
                    evidence.append(f"Memory matches: {len(mem)}")

        if not evidence:
            # Safe fallback if planner produced no actions.
            insights = await self._insight_gen.generate_insights(transactions)
            evidence = insights.copy()

        reply = self._compose_reply(len(transactions), evidence)
        return {
            "reply": reply,
            "insights": insights,
            "actions": actions,
        }

    def _compose_reply(self, transaction_count: int, evidence: list[str]) -> str:
        """Build concise final response from tool evidence."""
        trimmed = evidence[:3]
        if not trimmed:
            return f"Based on {transaction_count} transactions, no additional insights were generated."
        return f"Based on {transaction_count} transactions: " + " | ".join(trimmed)

    def _plan_actions(self, message: str) -> list[str]:
        """Plan actions with a low-latency hybrid strategy.

        For clear finance intents, prefer deterministic keyword planning.
        For ambiguous prompts, attempt LLM planning and then fallback.
        """
        keyword_actions = self._keyword_plan_actions(message)
        if not self._is_ambiguous_prompt(message):
            return keyword_actions

        planned = self._llm_plan_actions(message)
        if planned:
            return planned
        return keyword_actions

    def _is_ambiguous_prompt(self, message: str) -> bool:
        """Detect prompts that likely benefit from LLM planning."""
        text = message.lower()
        finance_markers = [
            "summary",
            "total",
            "overall",
            "spending",
            "category",
            "breakdown",
            "where",
            "split",
            "remember",
            "before",
            "previous",
            "last",
            "insight",
        ]
        return not any(marker in text for marker in finance_markers)

    def _keyword_plan_actions(self, message: str) -> list[str]:
        """Deterministic planner for environments without LLM credentials."""
        text = message.lower()
        actions: list[str] = []

        if any(k in text for k in ["summary", "total", "overall", "spending"]):
            actions.append("spending_summary")
        if any(k in text for k in ["category", "breakdown", "where", "split"]):
            actions.append("category_breakdown")
        if any(k in text for k in ["remember", "before", "previous", "last"]):
            actions.append("memory_search")

        # Always include insights so chat remains useful.
        if "insights" not in actions:
            actions.append("insights")

        # Preserve order and remove duplicates.
        dedup: list[str] = []
        for action in actions:
            if action not in dedup:
                dedup.append(action)
        return dedup

    def _llm_plan_actions(self, message: str) -> list[str]:
        """Optionally use OpenAI to choose actions in strict JSON.

        Returns empty list on any error to guarantee deterministic fallback.
        """
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            return []

        try:
            from openai import OpenAI

            client = OpenAI(api_key=api_key)
            completion = client.chat.completions.create(
                model=self._model,
                temperature=0,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a planner for a personal finance agent. "
                            "Choose actions from: spending_summary, category_breakdown, memory_search, insights. "
                            "Return strict JSON: {\"actions\": [..]}."
                        ),
                    },
                    {"role": "user", "content": message},
                ],
                response_format={"type": "json_object"},
            )
            content = completion.choices[0].message.content or "{}"
            parsed = json.loads(content)
            actions = parsed.get("actions", [])
            allowed = {"spending_summary", "category_breakdown", "memory_search", "insights"}
            clean = [a for a in actions if a in allowed]
            return clean[:4]
        except Exception:
            return []
