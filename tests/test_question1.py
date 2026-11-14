from __future__ import annotations

from eval.reference import TICKET_EXPECTATIONS
from question1.agent import SupportCopilot, _applicable_article, _int

PUBLIC = ["T-1001", "T-1002", "T-1003", "T-1004"]


def test_baseline_resolves_public_tickets():
    copilot = SupportCopilot()
    for tid in PUBLIC:
        expected = TICKET_EXPECTATIONS[tid]
        got = copilot.handle_ticket(tid)
        assert got.action == expected.action
        assert got.kb_article == expected.kb_article


def test_holdout_subscription():
    copilot = SupportCopilot()
    got = copilot.handle_ticket("T-1091")
    exp = TICKET_EXPECTATIONS["T-1091"]
    assert got.action == exp.action
    assert got.kb_article == exp.kb_article


def test_prompt_cost_is_reduced():
    for tid in PUBLIC:
        copilot = SupportCopilot()
        copilot.handle_ticket(tid)
        assert copilot.session.total_points() < 3_000


def test_nullable_delivery_age_preserves_zero_and_uses_default_for_none():
    assert _int(None, 99) == 99
    assert _int(0, 99) == 0


def test_article_selection_uses_applicable_policy_for_edge_cases():
    cases = [
        (
            {"product_type": "physical", "customized": True},
            "returns-within-30-days",
        ),
        (
            {"product_type": "physical", "customized": True, "opened": True},
            "returns-within-30-days",
        ),
        (
            {
                "product_type": "physical",
                "customized": True,
                "damage_reported": True,
            },
            "custom-items-no-replace",
        ),
        (
            {"product_type": "digital", "license_revealed": False},
            "digital-license-no-refund",
        ),
        (
            {
                "product_type": "physical",
                "damage_reported": True,
                "delivered_days_ago": 8,
            },
            "damage-replacement-7-days",
        ),
    ]
    for order, expected_article in cases:
        assert f"--- {expected_article} ---" in _applicable_article(order)

    unrevealed_digital = _applicable_article(
        {"product_type": "digital", "license_revealed": False}
    )
    assert "no policy for an unrevealed license" in unrevealed_digital


def test_damaged_order_with_null_delivery_age_does_not_crash():
    article = _applicable_article(
        {
            "product_type": "physical",
            "damage_reported": True,
            "delivered_days_ago": None,
        }
    )
    assert "--- damage-replacement-7-days ---" in article
