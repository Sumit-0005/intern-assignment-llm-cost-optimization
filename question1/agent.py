from __future__ import annotations

import json
import re
from dataclasses import dataclass

from shared.billing import CostSession
from shared.kb import load_article
from shared.llm import ChatMessage, LlmClient
from shared.tools import format_thread, get_order, get_ticket


@dataclass
class TicketResolution:
    action: str
    kb_article: str
    customer_reply: str
    raw_completion: str


def parse_resolution(text: str) -> TicketResolution:
    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        raise ValueError(f"No JSON in model output:\n{text[:400]}")
    data = json.loads(m.group())
    return TicketResolution(
        action=str(data["action"]).upper(),
        kb_article=str(data["kb_article"]),
        customer_reply=str(data.get("customer_reply", "")),
        raw_completion=text,
    )


_POLICY_FIELDS = (
    "product_type",
    "subscription_age_days",
    "license_revealed",
    "customized",
    "damage_reported",
    "delivered_days_ago",
    "opened",
)


def _int(value: object, default: int) -> int:
    return default if value is None else int(value)


_SYSTEM_PROMPT = "\n".join(
    [
        "You are ShopAssist Tier-1 support. Choose exactly one action and its single KB article.",
        "Actions: REFUND, REPLACE, SEND_ARTICLE, ESCALATE_TIER2.",
        'Reply with ONLY JSON: {"action":"...","kb_article":"..."}',
        "Examples: unopened physical within 30 days -> REFUND/returns-within-30-days; "
        "revealed digital license -> SEND_ARTICLE/digital-license-no-refund.",
    ]
)


def _applicable_article(order: dict) -> str:
    note = ""
    if order.get("product_type") == "subscription":
        article_id = "subscription-cooling-off"
    elif order.get("license_revealed"):
        article_id = "digital-license-no-refund"
    elif order.get("customized") and order.get("damage_reported"):
        article_id = "custom-items-no-replace"
    elif order.get("product_type") == "digital":
        article_id = "digital-license-no-refund"
        note = (
            "\nNote: this policy applies only after the license is revealed; "
            "the help center has no policy for an unrevealed license."
        )
    elif order.get("damage_reported"):
        article_id = "damage-replacement-7-days"
        if _int(order.get("delivered_days_ago"), 99) > 7:
            note = (
                "\nNote: this replacement policy covers reports within 7 days; "
                "the help center has no policy for older damage reports."
            )
    else:
        article_id = "returns-within-30-days"
    return f"--- {article_id} ---\n{load_article(article_id)}{note}"


class SupportCopilot:
    """
    ShopAssist Tier-1 copilot.

    Given a support ticket and order facts, choose an action and KB article.
    """

    def __init__(self, session: CostSession | None = None):
        self.session = session or CostSession()
        self.llm = LlmClient(self.session)

    def build_messages(self, ticket_id: str) -> list[ChatMessage]:
        ticket = get_ticket(ticket_id)
        order = get_order(ticket["order_id"])
        thread = format_thread(ticket)
        policy_order = {
            key: order[key] for key in _POLICY_FIELDS if key in order
        }
        snapshot = json.dumps(policy_order, separators=(",", ":"))
        user = (
            f"TICKET {ticket_id}\n{thread}\n"
            f"ORDER_SNAPSHOT:{snapshot}END_SNAPSHOT\n"
            f"HELP CENTER:\n{_applicable_article(policy_order)}"
        )
        return [
            ChatMessage("system", _SYSTEM_PROMPT),
            ChatMessage("user", user),
        ]

    def handle_ticket(self, ticket_id: str) -> TicketResolution:
        messages = self.build_messages(ticket_id)
        completion = self.llm.chat("handle_ticket", messages)
        return parse_resolution(completion)
