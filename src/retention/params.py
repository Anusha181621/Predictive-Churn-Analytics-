"""Assumptions and economics for the retention layer.

Every number here is either a **business policy** (what a campaign costs, where the value bands sit)
or an **unvalidated assumption** (how likely an intervention is to work). The distinction matters
enough to be structural rather than a comment, so :attr:`RetentionParams.assumptions` returns the
unvalidated ones with the reasoning behind each, and every output that depends on them carries a
flag saying so.

Why retention propensity is an assumption and must stay labelled
---------------------------------------------------------------
Retention propensity is *the probability that contacting a customer changes their behaviour*.
Measuring it needs intervention history: campaigns sent, and what happened to a comparable group
that was not contacted. This dataset has no campaign log and no holdout, so propensity **cannot be
learned from it** -- there is nothing in four CSVs of transactions that identifies a causal effect.

The honest options were to omit the score, or to state an assumption openly and propagate it. The
brief asks for the latter, so:

* the base rate is one configurable number, not a fitted-looking artefact;
* the behavioural multipliers below are *directional* judgements with a stated rationale, not
  estimates -- they are deliberately coarse (0.6, 1.4) so nobody mistakes them for measurements;
* everything derived from propensity is marked as assumption-dependent in the outputs.

Read the retention opportunity score as a *ranking* device that is defensible under a stated
assumption, not as a forecast. The moment real campaign results exist, replace
:attr:`base_retention_propensity` with a measured uplift and drop the multipliers.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.config.settings import get_settings

__all__ = ["RetentionParams", "PRIORITY_BANDS"]

#: Priority bands, from most to least urgent, applied to the retention opportunity percentile.
PRIORITY_BANDS = ("Critical", "High", "Medium", "Low")


@dataclass(frozen=True)
class RetentionParams:
    """Economics, thresholds and assumptions for the retention decision layer."""

    # ------------------------------------------------------------------ expected future revenue
    #: Horizon for expected future revenue, in days. Matches the churn horizon by default so that
    #: "revenue at risk" is the revenue at risk *over the window the churn probability describes*.
    #: Mixing a 90-day probability with a 365-day revenue figure would overstate exposure fourfold.
    revenue_horizon_days: int = 90

    #: Weight on recent (last 365 days) behaviour versus lifetime behaviour when projecting orders
    #: and order value. Recent behaviour is the better predictor of the next order, so it leads --
    #: but not exclusively, because a single quiet quarter should not erase three years of history.
    recent_behaviour_weight: float = 0.6

    #: Below this tenure a customer's rate cannot be extrapolated safely, so their projection is
    #: capped at what they have actually spent. This is the lesson from the interim estimate in
    #: Section 3, where annualising 24 days of history made a one-order customer look like the most
    #: valuable account in the book.
    min_tenure_days_for_projection: int = 180

    #: Hard ceiling on projected revenue as a multiple of observed lifetime revenue, so no
    #: projection can run away from the evidence supporting it.
    max_projection_multiple: float = 2.0

    # ------------------------------------------------------------------ retention propensity
    #: ASSUMPTION. Baseline probability that contacting an at-risk customer changes their behaviour.
    #: 25% is a common planning figure for win-back email in retail; it is a placeholder for a
    #: measured uplift, not an estimate derived from this data.
    base_retention_propensity: float = 0.25

    #: ASSUMPTION. Multiplier where the customer demonstrably responds to discounts, and the
    #: recommended action is a discount. Directional: someone who buys mainly on promotion is more
    #: movable by a promotion.
    discount_responsive_multiplier: float = 1.4

    #: ASSUMPTION. Multiplier where the customer has been silent for longer than a year. Recovery
    #: rates fall sharply with absence; this is coarse and deliberately so.
    long_absence_multiplier: float = 0.6

    #: ASSUMPTION. Multiplier for customers still transacting recently, where an intervention is
    #: nudging an existing habit rather than restarting a dead one.
    still_active_multiplier: float = 1.2

    #: ASSUMPTION. Multiplier for heavy returners: winning the order back does not mean winning the
    #: margin back, so the effective success of an intervention is discounted.
    high_return_multiplier: float = 0.7

    #: Bounds so no combination of multipliers produces an implausible propensity.
    min_retention_propensity: float = 0.02
    max_retention_propensity: float = 0.75

    # ------------------------------------------------------------------ campaign economics
    #: Cost of one outbound contact (email, push, SMS blended), in the dataset currency. A policy
    #: input, not an assumption: the business knows what its channels cost.
    communication_cost: float = 1.50

    #: Fixed overhead attributed per targeted customer for creative, list handling and QA.
    campaign_overhead_per_customer: float = 0.50

    #: Cost of a loyalty reward or free-shipping incentive, where the action uses one.
    free_shipping_cost: float = 5.00
    loyalty_reward_cost: float = 10.00

    #: Discount incentives cost a share of the revenue they unlock, so their cost is computed from
    #: the offered depth rather than fixed here.

    #: Cap on the discount depth the engine will ever offer, as a percentage. Prevents the engine
    #: from proposing a 50% giveaway just because a customer once took one.
    max_offer_discount_pct: float = 25.0

    #: Floor on offered discount depth, so an offer is worth making at all.
    min_offer_discount_pct: float = 10.0

    #: Expected ROI at or below which a customer is not targeted. Zero means "must at least pay for
    #: itself"; raise it to demand a margin.
    min_expected_roi: float = 0.0

    # ------------------------------------------------------------------ churn risk bands
    #: Churn-probability edges of the Medium and High risk bands. These default to the platform's
    #: configured thresholds (``RISK_THRESHOLD_MEDIUM`` / ``RISK_THRESHOLD_HIGH``) rather than to
    #: literals, so the segments band a customer exactly where the model's own ``risk_level`` puts
    #: them. Restating the numbers here was the bug that let a CRM manager move the bands in `.env`
    #: and see the risk levels change while "High-Value At Risk" kept using the old edge.
    risk_medium_threshold: float = field(
        default_factory=lambda: get_settings().risk_threshold_medium
    )
    risk_high_threshold: float = field(
        default_factory=lambda: get_settings().risk_threshold_high
    )

    # ------------------------------------------------------------------ segmentation thresholds
    #: Value percentile at or above which a customer counts as high value for segmentation.
    high_value_percentile: float = 0.80
    #: Value percentile below which a customer counts as low value.
    low_value_percentile: float = 0.40

    #: Orders in the last 365 days at or above which a customer counts as frequent.
    frequent_orders_365d: int = 6
    #: Lifetime orders at or above which a customer counts as loyal (a repeat relationship).
    loyal_min_orders: int = 3

    #: Return rate at or above which a customer is a high-return customer.
    high_return_rate: float = 0.40

    #: Churn probability below which a customer needs no intervention at all. The brief lists
    #: "already highly engaged" as a legitimate reason not to target, and it is a materially
    #: different reason from "uneconomic" -- conflating the two makes the suppression list
    #: uninterpretable.
    already_engaged_max_churn: float = 0.15

    #: Discount-dependency *percentile* at or above which a customer counts as discount-driven for
    #: segmentation. A percentile rather than an absolute score because 50.6% of this dataset's order
    #: lines carry a discount, so any absolute threshold catches most of the book and the segment
    #: stops distinguishing anybody: the first version labelled 526 of 1,000 customers
    #: "Discount-Driven At Risk", which is not a segment, it is a description of the brand.
    discount_driven_percentile: float = 0.70

    #: Recency beyond which a customer is treated as lost rather than dormant: two full annual
    #: cycles without an order.
    lost_recency_days: int = 730

    #: Revenue growth at or below which a customer counts as declining.
    declining_revenue_growth: float = -0.30

    #: Days from a seasonal customer's buying window within which a seasonal campaign is timely.
    seasonal_campaign_lead_days: int = 60

    #: Category diversity below which a customer is concentrated enough to be worth cross-selling.
    cross_sell_max_diversity: float = 0.45
    #: Category diversity above which a customer is broad enough to pitch new arrivals to.
    new_collection_min_diversity: float = 0.70

    #: Full-price order rate at or above which discounting is inappropriate -- the brief is explicit
    #: that premium customers should not be handed unnecessary discounts.
    full_price_buyer_rate: float = 0.70

    #: Purchase regularity above which a replenishment reminder is credible.
    replenishment_min_regularity: float = 0.55

    # ------------------------------------------------------------------ priority banding
    #: Retention-opportunity percentiles splitting the priority bands.
    critical_priority_percentile: float = 0.95
    high_priority_percentile: float = 0.85
    medium_priority_percentile: float = 0.60

    #: Recorded for provenance in the outputs.
    currency: str = "EUR"

    def validate(self) -> None:
        """Raise ``ValueError`` if the parameters are mutually inconsistent."""
        if self.revenue_horizon_days <= 0:
            raise ValueError("revenue_horizon_days must be positive")
        if not 0.0 <= self.recent_behaviour_weight <= 1.0:
            raise ValueError("recent_behaviour_weight must lie in [0, 1]")
        if not 0.0 < self.base_retention_propensity < 1.0:
            raise ValueError("base_retention_propensity must lie strictly in (0, 1)")
        if not 0.0 < self.min_retention_propensity <= self.max_retention_propensity < 1.0:
            raise ValueError(
                "propensity bounds must satisfy 0 < min <= max < 1, got "
                f"min={self.min_retention_propensity}, max={self.max_retention_propensity}"
            )
        if self.max_projection_multiple < 1.0:
            raise ValueError("max_projection_multiple must be at least 1")
        if not 0.0 < self.low_value_percentile < self.high_value_percentile < 1.0:
            raise ValueError("value percentiles must satisfy 0 < low < high < 1")
        if not 0.0 < self.risk_medium_threshold < self.risk_high_threshold < 1.0:
            raise ValueError(
                "risk band thresholds must satisfy 0 < medium < high < 1, got "
                f"medium={self.risk_medium_threshold}, high={self.risk_high_threshold}"
            )
        if not (
            0.0
            < self.medium_priority_percentile
            < self.high_priority_percentile
            < self.critical_priority_percentile
            < 1.0
        ):
            raise ValueError("priority percentiles must be strictly increasing and inside (0, 1)")
        if not 0.0 <= self.min_offer_discount_pct <= self.max_offer_discount_pct <= 100.0:
            raise ValueError("offer discount bounds must satisfy 0 <= min <= max <= 100")
        if self.communication_cost < 0 or self.campaign_overhead_per_customer < 0:
            raise ValueError("costs cannot be negative")

    @property
    def contact_cost(self) -> float:
        """Cost of reaching one customer, before any incentive."""
        return self.communication_cost + self.campaign_overhead_per_customer

    def assumptions(self) -> dict[str, dict[str, object]]:
        """The unvalidated assumptions, with the reasoning, for the outputs and the report.

        Deliberately separate from the policy inputs: a reader must be able to see at a glance which
        numbers came from the business and which were invented in the absence of data.
        """
        return {
            "base_retention_propensity": {
                "value": self.base_retention_propensity,
                "kind": "ASSUMPTION",
                "why": (
                    "Measuring intervention uplift needs a campaign log and an untreated control "
                    "group. This dataset has neither, so propensity cannot be learned from it. "
                    "25% is a planning placeholder for retail win-back email."
                ),
                "replace_with": "measured uplift from a holdout test once campaigns have run",
            },
            "discount_responsive_multiplier": {
                "value": self.discount_responsive_multiplier,
                "kind": "ASSUMPTION",
                "why": "Directional: a customer who buys mainly on promotion is more movable by one.",
            },
            "long_absence_multiplier": {
                "value": self.long_absence_multiplier,
                "kind": "ASSUMPTION",
                "why": "Recovery rates fall sharply with absence beyond a year.",
            },
            "still_active_multiplier": {
                "value": self.still_active_multiplier,
                "kind": "ASSUMPTION",
                "why": "Nudging a live habit is easier than restarting a dead one.",
            },
            "high_return_multiplier": {
                "value": self.high_return_multiplier,
                "kind": "ASSUMPTION",
                "why": "Winning the order back does not win the margin back for a heavy returner.",
            },
        }

    def policy_inputs(self) -> dict[str, object]:
        """The business policy numbers, which are inputs rather than assumptions."""
        return {
            "currency": self.currency,
            "revenue_horizon_days": self.revenue_horizon_days,
            "risk_medium_threshold": self.risk_medium_threshold,
            "risk_high_threshold": self.risk_high_threshold,
            "communication_cost": self.communication_cost,
            "campaign_overhead_per_customer": self.campaign_overhead_per_customer,
            "free_shipping_cost": self.free_shipping_cost,
            "loyalty_reward_cost": self.loyalty_reward_cost,
            "max_offer_discount_pct": self.max_offer_discount_pct,
            "min_offer_discount_pct": self.min_offer_discount_pct,
            "min_expected_roi": self.min_expected_roi,
        }
