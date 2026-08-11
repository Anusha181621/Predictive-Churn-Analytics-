"""Recency, frequency and monetary features.

All windows are half-open, ``(as_of - days, as_of]``, and every input row already satisfies
``purchase_date <= as_of`` because it came from :class:`~src.features.context.FeatureContext`.
"""

from __future__ import annotations

import pandas as pd

from src.features.context import FeatureContext, safe_divide

__all__ = ["build_rfm_features"]


def build_rfm_features(context: FeatureContext) -> pd.DataFrame:
    """One row per customer: recency, order/unit counts, revenue, and rolling windows."""
    features = context.empty_frame()
    orders = context.orders
    by_customer = orders.groupby("customer_id", observed=True)

    # --- recency ---
    last_purchase = by_customer["purchase_date"].max()
    features["recency_days"] = (context.as_of - last_purchase).dt.days.reindex(features.index)

    # --- frequency ---
    features["total_orders"] = by_customer.size().reindex(features.index, fill_value=0)
    features["total_lines"] = by_customer["lines"].sum().reindex(features.index, fill_value=0)
    features["total_units"] = by_customer["units"].sum().reindex(features.index, fill_value=0)

    # --- monetary ---
    features["lifetime_revenue"] = (
        by_customer["order_value"].sum().reindex(features.index, fill_value=0.0).round(2)
    )
    features["lifetime_gross_revenue"] = (
        by_customer["gross_value"].sum().reindex(features.index, fill_value=0.0).round(2)
    )
    features["average_order_value"] = by_customer["order_value"].mean().reindex(features.index)
    features["max_order_value"] = by_customer["order_value"].max().reindex(features.index)
    features["min_order_value"] = by_customer["order_value"].min().reindex(features.index)

    # Average value of a single item, which is not the same as the average order value: it
    # divides by units rather than by orders.
    features["average_item_value"] = safe_divide(
        features["lifetime_revenue"], features["total_units"]
    )
    features["average_units_per_order"] = safe_divide(
        features["total_units"], features["total_orders"]
    )

    # --- rolling windows ---
    for days in context.params.windows:
        window = context.orders_within(days)
        grouped = window.groupby("customer_id", observed=True)
        features[f"orders_{days}d"] = grouped.size().reindex(features.index, fill_value=0)
        features[f"revenue_{days}d"] = (
            grouped["order_value"].sum().reindex(features.index, fill_value=0.0).round(2)
        )
        features[f"units_{days}d"] = (
            grouped["units"].sum().reindex(features.index, fill_value=0)
        )

    # Share of lifetime revenue earned in the most recent year: a compact "is this customer
    # still worth what they used to be" signal.
    if 365 in context.params.windows:
        features["revenue_share_last_365d"] = safe_divide(
            features["revenue_365d"], features["lifetime_revenue"]
        )

    # Deliberately NOT rounded: this is the modelling table, and truncating ratios to a
    # few decimals would discard real signal. Presentation rounding happens at CSV-write
    # time in scripts/build_features.py. Monetary columns are rounded to cents above,
    # where the rounding is part of the definition rather than cosmetic.
    return features
