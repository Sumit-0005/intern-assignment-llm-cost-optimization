from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import asdict, dataclass

from shared.billing import CostSession
from shared.kb import article_ids, load_article
from shared.llm import ChatMessage, LlmClient
from shared.tools import format_thread, get_order, get_ticket

from question1.agent import parse_resolution


@dataclass
class TicketResolution:
    action: str
    kb_article: str


@dataclass(frozen=True)
class OrderFeatures:
    product_type: str
    subscription_age_days: int
    customized: bool
    license_revealed: bool
    delivered_days_ago: int
    opened: bool
    damage_reported: bool


_ACTIONS = {"REFUND", "REPLACE", "SEND_ARTICLE", "ESCALATE_TIER2"}
_MAX_ARTICLES = 3
_UNKNOWN_AGE_DAYS = 99
# From subscription-cooling-off: prorated refunds are available within 30 days.
_SUBSCRIPTION_REFUND_DAYS = 30
# From damage-replacement-7-days: standard damaged items qualify within 7 days.
_DAMAGE_REPLACEMENT_DAYS = 7
# From returns-within-30-days: resalable physical items qualify within 30 days.
_PHYSICAL_RETURN_DAYS = 30
_TOKEN_RE = re.compile(r"[a-z0-9]+")
_COMPOSE_SYSTEM = (
    "You are ShopAssist Tier-1 support. Use the order facts and supplied help-center "
    "articles. Actions: REFUND, REPLACE, SEND_ARTICLE, ESCALATE_TIER2. "
    'Choose one action and one kb_article. Reply with ONLY JSON: '
    '{"action":"...","kb_article":"..."}'
)


def _tokens(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def _int_or_default(value: object, default: int) -> int:
    return default if value is None else int(value)


def _parse_resolution(text: str) -> TicketResolution | None:
    try:
        parsed = parse_resolution(text)
    except (KeyError, TypeError, ValueError):
        return None
    return TicketResolution(parsed.action, parsed.kb_article)


class SupportPipeline:
    """ShopAssist pipeline with code-based triage, retrieval, and auditing."""

    def __init__(self, session: CostSession | None = None):
        self.session = session or CostSession()
        self.llm = LlmClient(self.session)
        self._article_ids = tuple(article_ids())
        self._article_id_set = frozenset(self._article_ids)
        self._article_index: list[tuple[str, str, list[str]]] | None = None
        self._document_frequency: Counter[str] = Counter()
        self._average_article_length = 0.0

    def handle_ticket(self, ticket_id: str) -> TicketResolution:
        ticket = get_ticket(ticket_id)
        order = get_order(ticket["order_id"])
        thread = format_thread(ticket)
        features = self._triage(order)
        articles = self._retrieve(thread, features)
        expected = self._expected_resolution(features)
        if expected.kb_article not in {article_id for article_id, _ in articles}:
            self._ensure_article_index()
            article_texts = {
                article_id: text
                for article_id, text, _ in self._article_index or []
            }
            expected_text = article_texts.get(expected.kb_article)
            if expected_text is None:
                expected_text = load_article(expected.kb_article)
            articles = [
                (expected.kb_article, expected_text),
                *articles[: _MAX_ARTICLES - 1],
            ]
        raw = self._compose(ticket_id, thread, features, articles)
        resolution = _parse_resolution(raw)
        resolution = self._audit(
            resolution,
            ticket_id,
            thread,
            features,
            articles,
        )
        return TicketResolution(resolution.action, resolution.kb_article)

    @staticmethod
    def _triage(order: dict) -> OrderFeatures:
        return OrderFeatures(
            product_type=str(order.get("product_type", "")),
            # Unknown age must not be treated as refund-eligible.
            subscription_age_days=_int_or_default(
                order.get("subscription_age_days"), _UNKNOWN_AGE_DAYS
            ),
            customized=bool(order.get("customized", False)),
            license_revealed=bool(order.get("license_revealed", False)),
            delivered_days_ago=_int_or_default(
                order.get("delivered_days_ago"), _UNKNOWN_AGE_DAYS
            ),
            opened=bool(order.get("opened", False)),
            damage_reported=bool(order.get("damage_reported", False)),
        )

    def _retrieve(
        self,
        thread: str,
        features: OrderFeatures,
    ) -> list[tuple[str, str]]:
        self._ensure_article_index()
        documents = self._article_index
        if not documents:
            return []

        feature_terms = [features.product_type]
        if features.product_type == "physical":
            feature_terms.extend(["physical item", "return", "refund"])
        elif features.product_type == "subscription":
            feature_terms.append("subscription cancellation")
            if features.subscription_age_days != _UNKNOWN_AGE_DAYS:
                feature_terms.append(f"{features.subscription_age_days} days")
        elif features.product_type == "digital":
            feature_terms.extend(["digital product", "license"])

        if features.customized:
            feature_terms.extend(["custom", "customized", "personalized"])
        elif features.product_type == "physical":
            feature_terms.extend(["standard", "stock"])
        if features.license_revealed:
            feature_terms.extend(["digital", "license", "revealed"])
        if features.damage_reported:
            feature_terms.extend(["damage", "damaged", "replacement"])
        if features.product_type == "physical":
            if features.delivered_days_ago <= _DAMAGE_REPLACEMENT_DAYS:
                feature_terms.extend(["within 7 days"])
            elif features.delivered_days_ago <= _PHYSICAL_RETURN_DAYS:
                feature_terms.extend(["within 30 days"])
            else:
                feature_terms.extend(["over 30 days"])
            feature_terms.extend(
                ["opened"] if features.opened else ["unopened", "resalable"]
            )

        query = Counter(_tokens(f"{thread} {' '.join(feature_terms)}"))
        scores: list[tuple[float, str]] = []
        for article_id, _, terms in documents:
            term_frequency = Counter(terms)
            score = 0.0
            for token, query_frequency in query.items():
                frequency = term_frequency[token]
                if not frequency:
                    continue
                df = self._document_frequency[token]
                inverse_frequency = math.log(
                    1 + (len(documents) - df + 0.5) / (df + 0.5)
                )
                length_norm = 1.5 * (
                    1 - 0.75 + 0.75 * len(terms) / self._average_article_length
                )
                score += query_frequency * inverse_frequency * (
                    frequency * 2.5 / (frequency + length_norm)
                )
            scores.append((score, article_id))

        selected_ids = [
            article_id
            for _, article_id in sorted(scores, key=lambda item: (-item[0], item[1]))[
                :_MAX_ARTICLES
            ]
        ]
        article_texts = {article_id: text for article_id, text, _ in documents}
        return [(article_id, article_texts[article_id]) for article_id in selected_ids]

    def _ensure_article_index(self) -> None:
        if self._article_index is not None:
            return

        index = []
        for article_id in self._article_ids:
            text = load_article(article_id)
            index.append(
                (article_id, text, _tokens(f"{article_id} {text}"))
            )
        self._article_index = index
        self._document_frequency = Counter(
            token
            for _, _, terms in index
            for token in set(terms)
        )
        if index:
            self._average_article_length = sum(
                len(terms) for _, _, terms in index
            ) / len(index)

    def _compose(
        self,
        ticket_id: str,
        thread: str,
        features: OrderFeatures,
        articles: list[tuple[str, str]],
        correction: str | None = None,
        label: str = "compose",
    ) -> str:
        system = _COMPOSE_SYSTEM
        if correction:
            system = f"{system}\n{correction}"
        snapshot = json.dumps(asdict(features), separators=(",", ":"))
        article_text = "\n\n".join(
            f"--- {article_id} ---\n{text}" for article_id, text in articles
        )
        user = (
            f"TICKET {ticket_id}\nTHREAD:\n{thread}\n"
            f"ORDER_SNAPSHOT:{snapshot}END_SNAPSHOT\n"
            f"HELP CENTER:\n{article_text}"
        )
        return self.llm.chat(
            label,
            [ChatMessage("system", system), ChatMessage("user", user)],
        )

    def _audit(
        self,
        resolution: TicketResolution | None,
        ticket_id: str,
        thread: str,
        features: OrderFeatures,
        articles: list[tuple[str, str]],
    ) -> TicketResolution:
        errors = self._audit_errors(resolution, features)
        if resolution is not None and not errors:
            return resolution

        repaired = _parse_resolution(
            self._compose(
                ticket_id,
                thread,
                features,
                articles,
                correction=(
                    "The previous output failed validation: "
                    f"{'; '.join(errors)}. Re-check the supplied facts and articles; "
                    "return valid JSON only."
                ),
                label="audit_retry",
            )
        )
        retry_errors = self._audit_errors(repaired, features)
        structural_errors = [
            error
            for error in retry_errors
            if not error.startswith("resolution conflicts")
        ]
        if repaired is None or structural_errors:
            return self._expected_resolution(features)
        return repaired

    def _audit_errors(
        self,
        resolution: TicketResolution | None,
        features: OrderFeatures,
    ) -> list[str]:
        if resolution is None:
            return ["response is not valid JSON with action and kb_article"]

        errors = []
        if resolution.action not in _ACTIONS:
            errors.append(f"unsupported action {resolution.action!r}")
        if resolution.kb_article not in self._article_id_set:
            errors.append(f"unknown KB article {resolution.kb_article!r}")

        expected = self._expected_resolution(features)
        if (resolution.action, resolution.kb_article) != (
            expected.action,
            expected.kb_article,
        ):
            errors.append(
                "resolution conflicts with order policy "
                f"(expected {expected.action}/{expected.kb_article})"
            )
        return errors

    @staticmethod
    def _expected_resolution(features: OrderFeatures) -> TicketResolution:
        if features.product_type == "subscription":
            # subscription-cooling-off permits prorated refunds through day 30.
            action = (
                "REFUND"
                if features.subscription_age_days <= _SUBSCRIPTION_REFUND_DAYS
                else "SEND_ARTICLE"
            )
            return TicketResolution(action, "subscription-cooling-off")

        if features.license_revealed:
            return TicketResolution("SEND_ARTICLE", "digital-license-no-refund")

        if features.customized and features.damage_reported:
            # custom-items-no-replace directs custom merchandise to Tier 2.
            return TicketResolution("ESCALATE_TIER2", "custom-items-no-replace")

        if features.product_type == "digital":
            return TicketResolution("ESCALATE_TIER2", "digital-license-no-refund")

        if features.damage_reported:
            # damage-replacement-7-days only guarantees replacement through day 7.
            if features.delivered_days_ago <= _DAMAGE_REPLACEMENT_DAYS:
                return TicketResolution("REPLACE", "damage-replacement-7-days")
            return TicketResolution("ESCALATE_TIER2", "damage-replacement-7-days")

        if (
            features.product_type == "physical"
            and not features.opened
            and features.delivered_days_ago <= _PHYSICAL_RETURN_DAYS
        ):
            # returns-within-30-days covers resalable physical items within 30 days.
            return TicketResolution("REFUND", "returns-within-30-days")

        # Unmatched cases lack an explicit automatic remedy; escalate conservatively.
        return TicketResolution("ESCALATE_TIER2", "returns-within-30-days")
