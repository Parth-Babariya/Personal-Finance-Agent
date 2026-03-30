"""Unit tests for agent tools built on top of the transaction service."""

import pytest

from src.services.tools import CategoryBreakdownTool, SpendingSummaryTool
from src.services.transaction_service import MockTransactionService


class TestAgentTools:
    """Verify formatted outputs from concrete tool implementations."""

    def setup_method(self):
        self.tx = MockTransactionService()
        self.summary_tool = SpendingSummaryTool(self.tx)
        self.breakdown_tool = CategoryBreakdownTool(self.tx)

    def test_tool_metadata_properties(self):
        """Tool names and descriptions should be stable and user-friendly."""
        assert self.summary_tool.name == "spending_summary"
        assert "summary" in self.summary_tool.description.lower()
        assert self.breakdown_tool.name == "category_breakdown"
        assert "breakdown" in self.breakdown_tool.description.lower()

    @pytest.mark.asyncio
    async def test_spending_summary_tool_formats_values(self):
        """Summary tool should format period, total amount, and count."""
        self.tx.add_transaction({"amount": 100.0, "category": "food"})
        self.tx.add_transaction({"amount": 40.0, "category": "transport"})

        output = await self.summary_tool.execute(period="week")

        assert "Period: week" in output
        assert "Total: $140.00" in output
        assert "Transactions: 2" in output

    @pytest.mark.asyncio
    async def test_category_breakdown_tool_empty_data(self):
        """Breakdown tool should return a friendly empty-state message."""
        output = await self.breakdown_tool.execute()
        assert output == "No spending data available."

    @pytest.mark.asyncio
    async def test_category_breakdown_tool_sorted_output(self):
        """Breakdown tool should list categories alphabetically."""
        self.tx.add_transaction({"amount": 30.0, "category": "transport"})
        self.tx.add_transaction({"amount": 70.0, "category": "food"})

        output = await self.breakdown_tool.execute()

        assert output.startswith("Spending by category:\n")
        food_index = output.find("food: $70.00")
        transport_index = output.find("transport: $30.00")
        assert food_index != -1
        assert transport_index != -1
        assert food_index < transport_index
