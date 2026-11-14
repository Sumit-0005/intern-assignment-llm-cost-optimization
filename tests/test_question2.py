from __future__ import annotations

from shared.tools import format_thread, get_order, get_ticket
from shared.kb import article_ids, expanded_help_center_blob
from shared.tokens import count_tokens
from eval.reference import TICKET_EXPECTATIONS
from question2.agent import OrderFeatures, SupportPipeline

PUBLIC = ["T-1001", "T-1002", "T-1003", "T-1004"]


def test_help_center_exceeds_context():
    assert count_tokens(expanded_help_center_blob()) > 200_000


def test_pipeline_resolves_public_tickets():
    pipe = SupportPipeline()
    for tid in PUBLIC:
        expected = TICKET_EXPECTATIONS[tid]
        got = pipe.handle_ticket(tid)
        assert got.action == expected.action
        assert got.kb_article == expected.kb_article


def test_retrieval_returns_three_articles_including_expected_article():
    pipe = SupportPipeline()
    for tid, expected in TICKET_EXPECTATIONS.items():
        ticket = get_ticket(tid)
        order = get_order(ticket["order_id"])
        features = pipe._triage(order)
        articles = pipe._retrieve(format_thread(ticket), features)
        assert 2 <= len(articles) <= 3
        assert expected.kb_article in {article_id for article_id, _ in articles}


def test_audit_retries_invalid_resolution(monkeypatch):
    pipe = SupportPipeline()
    original_chat = pipe.llm.chat
    labels = []

    def chat(label, messages, **kwargs):
        labels.append(label)
        if len(labels) == 1:
            return '{"action":"REFUND","kb_article":"returns-within-30-days"}'
        return original_chat(label, messages, **kwargs)

    monkeypatch.setattr(pipe.llm, "chat", chat)
    got = pipe.handle_ticket("T-1002")
    expected = TICKET_EXPECTATIONS["T-1002"]
    assert (got.action, got.kb_article) == (expected.action, expected.kb_article)
    assert labels == ["compose", "audit_retry"]


def test_pipeline_uses_one_compose_call_and_reduces_cost():
    for tid in PUBLIC:
        pipe = SupportPipeline()
        pipe.handle_ticket(tid)
        assert [call.label for call in pipe.session.calls] == ["compose"]
        assert pipe.session.total_points() < 2_500
        for action in ("REFUND", "REPLACE", "SEND_ARTICLE", "ESCALATE_TIER2"):
            assert action in pipe.session.calls[0].prompt


def _features(**overrides):
    values = {
        "product_type": "physical",
        "subscription_age_days": 99,
        "customized": False,
        "license_revealed": False,
        "delivered_days_ago": 10,
        "opened": False,
        "damage_reported": False,
    }
    values.update(overrides)
    return OrderFeatures(**values)


def test_policy_thresholds_match_kb_boundaries():
    resolve = SupportPipeline._expected_resolution
    assert resolve(
        _features(product_type="subscription", subscription_age_days=30)
    ).action == "REFUND"
    assert resolve(
        _features(product_type="subscription", subscription_age_days=31)
    ).action == "SEND_ARTICLE"
    assert resolve(
        _features(damage_reported=True, delivered_days_ago=7)
    ).action == "REPLACE"
    assert resolve(
        _features(damage_reported=True, delivered_days_ago=8)
    ).action == "ESCALATE_TIER2"
    assert resolve(
        _features(
            product_type="digital",
            damage_reported=True,
            delivered_days_ago=2,
        )
    ).action == "ESCALATE_TIER2"
    assert resolve(
        _features(
            customized=True,
            damage_reported=True,
            delivered_days_ago=2,
        )
    ).kb_article == "custom-items-no-replace"
    assert resolve(_features(delivered_days_ago=30)).action == "REFUND"
    assert resolve(_features(delivered_days_ago=31)).action == "ESCALATE_TIER2"


def test_custom_unopened_physical_item_can_be_refunded():
    resolution = SupportPipeline._expected_resolution(
        _features(customized=True, delivered_days_ago=10)
    )
    assert (resolution.action, resolution.kb_article) == (
        "REFUND",
        "returns-within-30-days",
    )


def test_custom_opened_item_falls_through_to_returns_article():
    resolution = SupportPipeline._expected_resolution(
        _features(customized=True, opened=True)
    )
    assert (resolution.action, resolution.kb_article) == (
        "ESCALATE_TIER2",
        "returns-within-30-days",
    )


def test_triage_uses_conservative_none_defaults_and_preserves_zero():
    features = SupportPipeline._triage(
        {
            "product_type": "subscription",
            "subscription_age_days": None,
            "delivered_days_ago": None,
        }
    )
    assert features.subscription_age_days == 99
    assert features.delivered_days_ago == 99
    assert SupportPipeline._triage(
        {"subscription_age_days": 0, "delivered_days_ago": 0}
    ).subscription_age_days == 0


def test_retrieval_caches_index_and_omits_delivery_window_for_digital(monkeypatch):
    import question2.agent as agent_module

    pipe = SupportPipeline()
    original_tokens = agent_module._tokens
    token_inputs = []

    def track_tokens(text):
        token_inputs.append(text)
        return original_tokens(text)

    monkeypatch.setattr(agent_module, "_tokens", track_tokens)
    features = _features(product_type="digital", delivered_days_ago=99)
    pipe._retrieve("", features)
    cached_index = pipe._article_index
    token_inputs.clear()
    pipe._retrieve("", features)
    assert pipe._article_index is cached_index
    assert "over 30 days" not in token_inputs[-1]
    token_inputs.clear()
    pipe._retrieve("", _features(product_type="subscription"))
    assert "99 days" not in token_inputs[-1]


def test_index_loads_each_article_once_and_reuses_its_text(monkeypatch):
    import question2.agent as agent_module

    original_load_article = agent_module.load_article
    calls = {}

    def track_load(article_id):
        calls[article_id] = calls.get(article_id, 0) + 1
        return original_load_article(article_id)

    monkeypatch.setattr(agent_module, "load_article", track_load)
    pipe = SupportPipeline()
    pipe._retrieve("", _features())
    assert calls == {article_id: 1 for article_id in article_ids()}


def test_expected_article_is_included_even_if_retrieval_misses_it(monkeypatch):
    pipe = SupportPipeline()
    monkeypatch.setattr(
        pipe,
        "_retrieve",
        lambda thread, features: [
            ("digital-license-no-refund", "unrelated article text")
        ],
    )
    pipe.handle_ticket("T-1001")
    assert "--- returns-within-30-days ---" in pipe.session.calls[0].prompt


def test_failed_retry_returns_expected_resolution(monkeypatch):
    pipe = SupportPipeline()
    monkeypatch.setattr(pipe.llm, "chat", lambda *args, **kwargs: "not JSON")
    got = pipe.handle_ticket("T-1002")
    expected = TICKET_EXPECTATIONS["T-1002"]
    assert (got.action, got.kb_article) == (expected.action, expected.kb_article)
