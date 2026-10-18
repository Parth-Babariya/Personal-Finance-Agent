"""Transaction service, auth service, conversation store and insights."""

from datetime import date

import pytest

from src.services.auth import AuthService
from src.services.insights import SimpleInsightGenerator


class TestSQLTransactionService:
    def test_add_defaults_date_to_today_and_rounds_money(self, tx_service, user_id):
        saved = tx_service.add_transaction({"amount": 10.005, "category": "food"}, user_id)
        assert saved["date"] == "2026-03-18"
        assert saved["amount"] == 10.01
        assert saved["type"] == "expense"

    def test_decimal_sums_are_exact(self, tx_service, user_id):
        for _ in range(10):
            tx_service.add_transaction({"amount": 0.1}, user_id)
        assert tx_service.get_summary(user_id)["total_expense"] == 1.0

    def test_summary_splits_income_and_expense(self, tx_service, seeded):
        s = tx_service.get_summary(seeded, start=date(2026, 3, 1), end=date(2026, 3, 18))
        assert s["total_expense"] == 17000.0
        assert s["total_income"] == 60000.0
        assert s["net"] == 43000.0
        assert s["transaction_count"] == 5
        assert s["by_category"] == {"rent": 15000.0, "groceries": 1200.0, "food": 800.0}
        assert "salary" not in s["by_category"]

    def test_summary_category_filter(self, tx_service, seeded):
        s = tx_service.get_summary(seeded, start=date(2026, 2, 1), end=date(2026, 2, 28), category="food")
        assert s["total_expense"] == 800.0
        assert s["transaction_count"] == 1

    def test_list_filters(self, tx_service, seeded):
        food = tx_service.list_transactions(seeded, category="food")
        assert [t["amount"] for t in food] == [300.0, 500.0, 800.0]  # newest first
        assert len(tx_service.list_transactions(seeded, query="RENT")) == 2
        assert len(tx_service.list_transactions(seeded, min_amount=1000, max_amount=3000)) == 2
        assert len(tx_service.list_transactions(seeded, type="income")) == 1
        largest = tx_service.list_transactions(seeded, type="expense", order_by="amount", limit=2)
        assert [t["amount"] for t in largest] == [15000.0, 15000.0]

    def test_pagination(self, tx_service, seeded):
        page = tx_service.list_transactions(seeded, limit=3, offset=2)
        assert len(page) == 3

    def test_update_and_delete(self, tx_service, user_id):
        saved = tx_service.add_transaction({"amount": 50, "category": "food"}, user_id)
        updated = tx_service.update_transaction(saved["id"], {"amount": 75, "category": "groceries"}, user_id)
        assert updated["amount"] == 75.0 and updated["category"] == "groceries"
        assert tx_service.delete_transaction(saved["id"], user_id) is True
        assert tx_service.get_transaction(saved["id"], user_id) is None
        assert tx_service.delete_transaction(saved["id"], user_id) is False

    def test_user_isolation(self, tx_service, seeded, other_user_id):
        tx = tx_service.list_transactions(seeded)[0]
        assert tx_service.list_transactions(other_user_id) == []
        assert tx_service.get_summary(other_user_id)["transaction_count"] == 0
        assert tx_service.get_transaction(tx["id"], other_user_id) is None
        assert tx_service.update_transaction(tx["id"], {"amount": 1}, other_user_id) is None
        assert tx_service.delete_transaction(tx["id"], other_user_id) is False


class TestAuthService:
    def test_register_authenticate(self, session_factory):
        auth = AuthService(session_factory)
        user = auth.register_user("carol", "StrongPass123")
        assert auth.get_user_by_id(user.id).username == "carol"
        assert auth.authenticate_user("carol", "StrongPass123").id == user.id
        assert auth.authenticate_user("carol", "wrong") is None
        assert auth.authenticate_user("nobody", "StrongPass123") is None

    def test_duplicate_username(self, session_factory):
        auth = AuthService(session_factory)
        auth.register_user("dave", "StrongPass123")
        with pytest.raises(ValueError, match="already exists"):
            auth.register_user("dave", "AnotherPass456")


class TestConversationStore:
    def test_recent_is_chronological_and_limited(self, conversations, user_id):
        for i in range(5):
            conversations.add_message(user_id, "user", f"msg {i}")
        recent = conversations.recent(user_id, limit=3)
        assert [m["content"] for m in recent] == ["msg 2", "msg 3", "msg 4"]

    def test_search_and_clear_are_user_scoped(self, conversations, user_id, other_user_id):
        conversations.add_message(user_id, "user", "food budget question")
        conversations.add_message(other_user_id, "user", "food for bob")
        assert len(conversations.search(user_id, "FOOD")) == 1
        assert conversations.clear(user_id) == 1
        assert conversations.recent(user_id) == []
        assert len(conversations.recent(other_user_id)) == 1


class TestInsights:
    def test_empty(self):
        assert SimpleInsightGenerator().generate_insights([]) == ["No transactions to analyze."]

    def test_rules(self, tx_service, seeded):
        rows = tx_service.list_transactions(seeded, start=date(2026, 3, 1))
        insights = SimpleInsightGenerator("₹").generate_insights(rows)
        joined = " ".join(insights)
        assert "'rent'" in joined and "₹15,000.00" in joined
        assert "You saved 72%" in joined
        assert "Total spending: ₹17,000.00 across 4 expenses." in insights

    def test_overspending(self):
        rows = [
            {"amount": 100.0, "type": "income", "category": "salary", "date": "2026-03-01"},
            {"amount": 150.0, "type": "expense", "category": "food", "date": "2026-03-02"},
        ]
        assert any("more than you earned" in i for i in SimpleInsightGenerator("$").generate_insights(rows))
