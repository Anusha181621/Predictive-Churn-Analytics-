"""Trend features: is this customer's behaviour improving or decaying?

Two complementary comparisons, because they answer different questions:

* **Window versus preceding window** (``*_growth``) -- "last quarter versus the quarter before".
  Sensitive to a recent change, and the natural home of the brief's ``spend_decline_pct``.
* **Recent versus lifetime average** (``recent_vs_historical_*``) -- "is the last quarter above
  or below this customer's own long-run rate". Robust when a single window happens to be quiet.

A customer who spent nothing in the baseline window has an *undefined* growth rate, not an
infinite one, so those come back as NaN. The gradient-boosted models in Section 3 handle NaN
natively, and inventing a finite number here would fabricate a trend that does not exist.
"""

from __future__ import annotations

import pandas as pd

from src.features.context import FeatureContext, growth, safe_divide

__all__ = ["build_trend_features"]


def _window_aggregates(orders: pd.DataFrame, index: pd.Index, suffix: str) -> pd.DataFrame:
    grouped = orders.groupby("customer_id", observed=True)
    frame = pd.DataFrame(index=index)
    frame[f"orders_{suffix}"] = grouped.size().reindex(index, fill_value=0)
    frame[f"revenue_{suffix}"] = grouped["order_value"].sum().reindex(index, fill_value=0.0)
    frame[f"units_{suffix}"] = grouped["units"].sum().reindex(index, fill_value=0)
    frame[f"aov_{suffix}"] = grouped["order_value"].mean().reindex(index)
    return frame


def build_trend_features(context: FeatureContext) -> pd.DataFrame:
    """One row per customer: growth rates and recent-versus-historical ratios."""
    features = context.empty_frame()
    window = context.params.trend_window_days

    recent = _window_aggregates(context.orders_within(window), features.index, "recent")
    previous = _window_aggregates(
        context.orders_between(window * 2, window), features.index, "previous"
    )

    # --- window versus preceding window ---
    features["revenue_growth"] = growth(recent["revenue_recent"], previous["revenue_previous"])
    features["order_frequency_growth"] = growth(
        recent["orders_recent"].astype(float), previous["orders_previous"].astype(float)
    )
    features["quantity_growth"] = growth(
        recent["units_recent"].astype(float), previous["units_previous"].astype(float)
    )
    features["aov_growth"] = growth(recent["aov_recent"], previous["aov_previous"])

    # The brief names these as explicit decline percentages. A positive number reads as
    # "declined by this much", which is what a business user expects to see in a driver list.
    features["spend_decline_pct"] = -features["revenue_growth"]
    features["order_frequency_decline_pct"] = -features["order_frequency_growth"]
    features["aov_decline_pct"] = -features["aov_growth"]

    # --- recent versus the customer's own lifetime rate ---
    #
    # The baseline is the customer's lifetime average per window of the same length, so the
    # comparison is like-for-like regardless of how long they have been a customer.
    first_purchase = (
        context.orders.groupby("customer_id", observed=True)["purchase_date"]
        .min()
        .reindex(features.index)
    )
    observed_days = (context.as_of - first_purchase).dt.days.add(1)
    # Fewer than one full window of history cannot support a "historical average".
    windows_of_history = (observed_days / window).where(observed_days.ge(window))

    grouped = context.orders.groupby("customer_id", observed=True)
    lifetime_revenue = grouped["order_value"].sum().reindex(features.index, fill_value=0.0)
    lifetime_orders = grouped.size().reindex(features.index, fill_value=0).astype(float)

    historical_revenue_rate = safe_divide(lifetime_revenue, windows_of_history)
    historical_order_rate = safe_divide(lifetime_orders, windows_of_history)

    features["recent_vs_historical_revenue"] = safe_divide(
        recent["revenue_recent"], historical_revenue_rate
    )
    features["recent_vs_historical_frequency"] = safe_divide(
        recent["orders_recent"].astype(float), historical_order_rate
    )

    # --- the raw window figures, kept because explanations need them ---
    #
    # "Revenue fell 48%" is only convincing next to "from EUR 820 to EUR 425".
    features["revenue_recent_window"] = recent["revenue_recent"].round(2)
    features["revenue_previous_window"] = previous["revenue_previous"].round(2)
    features["orders_recent_window"] = recent["orders_recent"]
    features["orders_previous_window"] = previous["orders_previous"]
    features["aov_recent_window"] = recent["aov_recent"]
    features["aov_previous_window"] = previous["aov_previous"]
    features["trend_window_days"] = window

    # Deliberately NOT rounded: this is the modelling table, and truncating ratios to a
    # few decimals would discard real signal. Presentation rounding happens at CSV-write
    # time in scripts/build_features.py. Monetary columns are rounded to cents above,
    # where the rounding is part of the definition rather than cosmetic.
    return features
