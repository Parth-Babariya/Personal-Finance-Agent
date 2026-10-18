"""Tool registry: schemas, validation, execution, and user scoping."""

import pytest

from src.agent.tools import ToolContext
from tests.conftest import TODAY


@pytest.fixture
def ctx(seeded) -> ToolContext:
    return ToolContext(user_id=seeded, today=TODAY)


def test_schemas_are_openai_function_specs(registry):
    schemas = registry.schemas()
    assert {s["function"]["name"] for s in schemas} == set(registry.names)
    for spec in schemas:
        assert spec["type"] == "function"
        params = spec["function"]["parameters"]
        assert params["type"] == "object"
        assert "user_id" not in params["properties"]  # identity never comes from the model


async def test_spending_summary(registry, ctx):
    out = await registry.execute("get_spending_summary", '{"period": "this_month", "category": "food"}', ctx)
    assert out.ok
    assert out.result["total_expense"] == 800.0
    assert out.result["period"] == "2026-03-01 to 2026-03-18"


async def test_custom_date_range_overrides_period(registry, ctx):
    out = await registry.execute(
        "get_spending_summary", {"period": "today", "start_date": "2026-02-01", "end_date": "2026-02-28"}, ctx
    )
    assert out.result["total_expense"] == 18300.0


async def test_compare_periods(registry, ctx):
    out = await registry.execute("compare_periods", {"period_a": "this_month", "period_b": "last_month"}, ctx)
    r = out.result
    assert r["a"]["total_expense"] == 17000.0 and r["b"]["total_expense"] == 18300.0
    assert r["expense_change"] == -1300.0
    assert r["expense_change_pct"] == -7.1
    assert r["by_category"]["shopping"] == {"a": 0.0, "b": 2500.0, "change": -2500.0}


async def test_search_and_top_expenses(registry, ctx):
    found = await registry.execute("search_transactions", {"category": "food", "limit": 2}, ctx)
    assert found.result["match_count"] == 3
    assert found.result["total_amount"] == 1600.0
    assert len(found.result["transactions"]) == 2 and found.result["truncated"] is True

    top = await registry.execute("get_top_expenses", {"period": "last_month", "limit": 2}, ctx)
    assert [t["amount"] for t in top.result["transactions"]] == [15000.0, 2500.0]


async def test_add_transaction_and_future_date_rejected(registry, ctx, tx_service):
    ok = await registry.execute("add_transaction", {"amount": 250, "category": "transport"}, ctx)
    assert ok.ok and ok.result["added"]["date"] == "2026-03-18"
    assert len(tx_service.list_transactions(ctx.user_id, category="transport")) == 1

    future = await registry.execute("add_transaction", {"amount": 10, "date": "2026-04-01"}, ctx)
    assert not future.ok and "future" in future.result["error"]


@pytest.mark.parametrize(
    ("name", "args"),
    [
        ("get_spending_summary", '{"period": "fortnight"}'),
        ("get_spending_summary", '{"category": "crypto"}'),
        ("get_spending_summary", '{"user_id": "someone-else"}'),
        ("add_transaction", '{"amount": -5}'),
        ("search_transactions", '{"limit": 1000}'),
        ("search_transactions", '{"start_date": "2026-03-10", "end_date": "2026-03-01"}'),
        ("get_spending_summary", "not json"),
        ("get_spending_summary", "[1, 2]"),
    ],
)
async def test_invalid_arguments_are_returned_as_errors(registry, ctx, name, args):
    out = await registry.execute(name, args, ctx)
    assert not out.ok
    assert "Invalid arguments" in out.result["error"]


async def test_unknown_tool(registry, ctx):
    out = await registry.execute("drop_tables", "{}", ctx)
    assert not out.ok and "Unknown tool" in out.result["error"]


async def test_tools_are_scoped_to_context_user(registry, seeded, other_user_id):
    other = ToolContext(user_id=other_user_id, today=TODAY)
    out = await registry.execute("get_spending_summary", {"period": "all_time"}, other)
    assert out.result["transaction_count"] == 0
