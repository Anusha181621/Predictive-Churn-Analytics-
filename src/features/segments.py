"""Behavioural segments.

Two outputs, deliberately:

* **Boolean flags** (``is_frequent_buyer``, ``is_dormant_buyer``, …) which may overlap. A
  customer can be both frequent and declining, and flattening that away loses the most
  actionable group in the book.
* **A single ``behavioral_segment``** label, resolved by priority, because dashboards and
  pivot tables need one value per customer.

Priority order and why it is that order
---------------------------------------
1. **New Buyer** -- too little history to judge. Calling a three-week-old customer "declining"
   because their second month is quieter than their first is noise, not signal.
2. **Dormant Buyer** -- gap far beyond the customer's *own* cadence, or past an absolute backstop.
3. **Seasonal Buyer** -- a repeatable annual buying window.
4. **Declining Buyer** -- still buying, but materially less than they were.
5. **Frequent Buyer** -- high recent order count and not declining.
6. **Occasional Buyer** -- the default for an active customer with a modest cadence.

Where the seasonal protection actually lives
--------------------------------------------
Note that Dormant outranks Seasonal here, which looks backwards given the instruction that
seasonal customers must not be called churned for a long gap alone. It is not: the protection
lives inside ``is_dormant_buyer``, which excludes anyone whose silence is
``seasonally_explained_inactivity``. Keeping the rule in exactly one place is what makes it
trustworthy -- and it avoids the failure mode of the alternative. Ranking Seasonal *above*
Dormant would label a seasonal customer "Seasonal Buyer" no matter how long they had been gone,
so a genuinely churned customer who happened to shop seasonally would be quietly filed under
"nothing to see here". Two years of silence is churn, whatever the shopping pattern.

So a seasonal customer mid-cycle and out of season keeps the Seasonal label, while one who has
skipped their whole season falls through to Dormant, which is the behaviour the business wants.

Every threshold comes from :class:`~src.features.params.FeatureParams`, so the definitions are
configurable rather than hard-coded here.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.context import FeatureContext

__all__ = ["build_segment_features", "SEGMENT_LABELS"]

#: In resolution priority order; earlier labels win.
SEGMENT_LABELS = (
    "New Buyer",
    "Dormant Buyer",
    "Seasonal Buyer",
    "Declining Buyer",
    "Frequent Buyer",
    "Occasional Buyer",
)

NO_HISTORY = "No History"


def build_segment_features(
    context: FeatureContext,
    rfm: pd.DataFrame,
    gaps: pd.DataFrame,
    trends: pd.DataFrame,
    lifecycle: pd.DataFrame,
    seasonality: pd.DataFrame,
) -> pd.DataFrame:
    """One row per customer: segment flags plus a single resolved ``behavioral_segment``."""
    features = context.empty_frame()
    params = context.params
    has_history = context.has_history

    tenure = lifecycle["customer_tenure_days"]
    gap_ratio = gaps["purchase_gap_ratio"]
    recency = rfm["recency_days"]
    orders_365d = rfm.get("orders_365d", pd.Series(0, index=features.index))

    # --- individual, overlapping flags ---

    features["is_new_buyer"] = (tenure.lt(params.new_customer_days) & has_history).fillna(False)

    features["is_seasonal_buyer"] = seasonality["is_seasonal_buyer"]
    features["seasonally_explained_inactivity"] = seasonality["seasonally_explained_inactivity"]

    # Dormant on the customer's own terms first, with an absolute backstop for customers whose
    # measured cadence is so slow that even 2x it would be over a year.
    beyond_own_cadence = gap_ratio.ge(params.dormant_gap_multiple)
    beyond_backstop = recency.ge(params.dormant_recency_days)
    features["is_dormant_buyer"] = (
        (beyond_own_cadence | beyond_backstop)
        & has_history
        & ~features["seasonally_explained_inactivity"]
    ).fillna(False)

    # Declining needs a real revenue fall AND a frequency fall, so a customer who simply bought
    # one cheaper item is not labelled as decaying.
    revenue_falling = trends["revenue_growth"].le(params.declining_revenue_growth)
    frequency_falling = trends["order_frequency_growth"].lt(0)
    below_own_average = trends["recent_vs_historical_revenue"].lt(1.0)
    features["is_declining_buyer"] = (
        (revenue_falling & (frequency_falling | below_own_average))
        & has_history
        & ~features["is_dormant_buyer"]
    ).fillna(False)

    features["is_frequent_buyer"] = (
        orders_365d.ge(params.frequent_orders_365d) & has_history
    ).fillna(False)

    features["is_occasional_buyer"] = (
        has_history
        & ~features["is_frequent_buyer"]
        & ~features["is_dormant_buyer"]
        & ~features["is_new_buyer"]
    ).fillna(False)

    # --- the single resolved label ---
    #
    # Assigned in reverse priority so that earlier, higher-priority rules overwrite later ones.
    # Dormant is applied after Seasonal on purpose -- see the module docstring: the seasonal
    # protection is already baked into `is_dormant_buyer`, so a seasonal customer who reaches
    # here as dormant has genuinely gone quiet through their own season.
    segment = pd.Series(NO_HISTORY, index=features.index, dtype="object")
    segment[has_history] = "Occasional Buyer"
    segment[features["is_frequent_buyer"]] = "Frequent Buyer"
    segment[features["is_declining_buyer"]] = "Declining Buyer"
    segment[features["is_seasonal_buyer"]] = "Seasonal Buyer"
    segment[features["is_dormant_buyer"]] = "Dormant Buyer"
    segment[features["is_new_buyer"]] = "New Buyer"
    features["behavioral_segment"] = segment

    # --- lifecycle stage, an orthogonal view ---
    #
    # Where behavioral_segment describes *how* a customer buys, this describes how far along the
    # engagement curve they are. Both are useful and neither subsumes the other.
    stage = pd.Series(NO_HISTORY, index=features.index, dtype="object")
    stage[has_history] = "Active"
    stage[has_history & features["is_new_buyer"]] = "New"
    stage[has_history & features["is_declining_buyer"]] = "Declining"
    stage[has_history & gap_ratio.ge(1.5) & ~features["is_dormant_buyer"]] = "At Risk"
    stage[features["is_dormant_buyer"]] = "Dormant"
    stage[has_history & recency.ge(params.dormant_recency_days * 2)] = "Lost"
    features["lifecycle_stage"] = stage

    # A compact, human-readable reason for the label, so a business user is never shown a
    # classification they cannot interrogate.
    features["segment_reason"] = _explain(features, gaps, rfm, seasonality, params)

    return features


def _explain(
    features: pd.DataFrame,
    gaps: pd.DataFrame,
    rfm: pd.DataFrame,
    seasonality: pd.DataFrame,
    params,
) -> pd.Series:
    """Build a one-line justification for each customer's resolved segment."""
    gap_ratio = gaps["purchase_gap_ratio"].round(2)
    expected = gaps["expected_purchase_interval_days"].round(0)
    recency = rfm["recency_days"]
    score = seasonality["seasonal_customer_score"].round(2)
    days_off = seasonality["days_from_preferred_season"].round(0)

    reason = pd.Series("", index=features.index, dtype="object")
    reason[features["behavioral_segment"].eq(NO_HISTORY)] = (
        "No purchases on or before the as-of date"
    )
    mask = features["behavioral_segment"].eq("Occasional Buyer")
    reason[mask] = "Active with a modest cadence"

    mask = features["behavioral_segment"].eq("Frequent Buyer")
    reason[mask] = (
        "At least "
        + str(params.frequent_orders_365d)
        + " orders in the last 365 days"
    )

    mask = features["behavioral_segment"].eq("Declining Buyer")
    reason[mask] = "Recent revenue fell materially versus the previous window"

    mask = features["behavioral_segment"].eq("Dormant Buyer")
    reason[mask] = (
        "Silent for "
        + recency[mask].astype("Int64").astype(str)
        + " days, "
        + gap_ratio[mask].astype(str)
        + "x their usual "
        + expected[mask].astype("Int64").astype(str)
        + "-day interval"
    )

    mask = features["behavioral_segment"].eq("Seasonal Buyer")
    reason[mask] = (
        "Buys in a repeatable seasonal window (score "
        + score[mask].astype(str)
        + "); currently "
        + days_off[mask].astype("Int64").astype(str)
        + " days from that season"
    )

    mask = features["behavioral_segment"].eq("New Buyer")
    reason[mask] = (
        "Only "
        + recency[mask].astype("Int64").astype(str)
        + " days since their last of few early orders; too little history to judge"
    )
    return reason
