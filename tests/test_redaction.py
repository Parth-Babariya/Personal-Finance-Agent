import pytest

from src.agent.redaction import redact


@pytest.mark.parametrize(
    ("text", "kind", "must_not_contain"),
    [
        ("my card 4111 1111 1111 1111 was charged", "card", "4111 1111 1111 1111"),
        ("card 5500-0000-0000-0004", "card", "5500-0000-0000-0004"),
        ("aadhaar 2345 6789 0123", "aadhaar", "2345 6789 0123"),
        ("PAN is ABCDE1234F", "pan", "ABCDE1234F"),
        ("account no: 123456789012 at HDFC", "account", "123456789012"),
        ("a/c 987654321 debited", "account", "987654321"),
        ("call me on +91 98765 43210", "phone", "98765 43210"),
        ("paid to rahul@okicici via UPI", "email_or_upi", "rahul@okicici"),
        ("mail me at a.b@example.com", "email_or_upi", "a.b@example.com"),
    ],
)
def test_masks_sensitive_identifiers(text, kind, must_not_contain):
    result = redact(text)
    assert kind in result.found
    assert must_not_contain not in result.text


def test_card_keeps_last_four_for_context():
    assert "[CARD ••••1111]" in redact("4111111111111111").text


@pytest.mark.parametrize(
    "text",
    [
        "I spent 1250000 on a car",
        "How much did I spend in 2026?",
        "spent ₹45,006.00 across 22 transactions",
        "transaction 1234567890123 failed",  # 13 digits but fails Luhn
        "my salary is 65000 and rent 15000",
    ],
)
def test_leaves_ordinary_text_alone(text):
    result = redact(text)
    assert result.text == text and not result.redacted
