"""Tests for the LLM-orchestrated finance agent fallback behavior."""

import pytest

from src.services.insight_generator import SimpleInsightGenerator
from src.services.llm_agent import FinanceLLMAgent
from src.services.memory_manager import SessionMemory
from src.services.transaction_service import MockTransactionService


class TestFinanceLLMAgent:
    """Validate deterministic planning and response generation."""

    def setup_method(self):
        self.tx = MockTransactionService()
        self.memory = SessionMemory()
        self.insight_gen = SimpleInsightGenerator()
        self.agent = FinanceLLMAgent(self.tx, self.memory, self.insight_gen)

        self.tx.add_transaction({"amount": 120.0, "category": "food"}, user_id="u1")
        self.tx.add_transaction({"amount": 80.0, "category": "transport"}, user_id="u1")
        self.memory.store("u1:chat_0", {"message": "last summary", "insights": ["demo"]})

    def test_keyword_planner_selects_expected_actions(self):
        actions = self.agent._keyword_plan_actions("show category breakdown and summary")
        assert "spending_summary" in actions
        assert "category_breakdown" in actions
        assert "insights" in actions

    @pytest.mark.asyncio
    async def test_respond_generates_reply_and_insights(self):
        result = await self.agent.respond("How is my spending?", user_id="u1")
        assert "reply" in result
        assert isinstance(result["insights"], list)
        assert len(result["insights"]) >= 1

    @pytest.mark.asyncio
    async def test_respond_uses_memory_search_action(self):
        result = await self.agent.respond("What did I ask before?", user_id="u1")
        assert "reply" in result
        assert "actions" in result
        assert "memory_search" in result["actions"]
