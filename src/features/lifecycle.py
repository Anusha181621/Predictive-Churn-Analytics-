"""Lifecycle features: how long has this customer been around, and how active were they?

``active_months`` counts calendar months containing at least one order; ``inactive_months`` is
the remainder of the customer's observable life. Both are counted only up to the as-of date, so
a customer's inactive month count never includes months that have not happened yet.
"""

from __future__ import annotations

import pandas as pd

from src.features.context import FeatureContext, safe_divide

__all__ = ["build_lifecycle_features"]


def _month_index(dates: pd.Series) -> pd.Series:
    """Map timestamps to a monotonic month number, so month arithmetic spans year boundaries."""
    return dates.dt.year * 12 + dates.dt.month


def build_lifecycle_features(context: FeatureContext) -> pd.DataFrame:
    """One row per customer: tenure, first/last purchase, active and inactive months."""
    features = context.empty_frame()
    orders = context.orders
    grouped = orders.groupby("customer_id", observed=True)

    registration = context.customers.set_index("customer_id")["registration_date"].reindex(
        features.index
    )
    first_purchase = grouped["purchase_date"].min().reindex(features.index)
    last_purchase = grouped["purchase_date"].max().reindex(features.index)

    features["registration_date"] = registration
    features["first_purchase_date"] = first_purchase
    features["last_purchase_date"] = last_purchase

    # Tenure runs from the first observed purchase, not from registration. In this dataset the
    # two coincide for every customer, but keeping them separate means the feature stays correct
    # if a future data drop ever separates sign-up from first order.
    features["customer_tenure_days"] = (context.as_of - first_purchase).dt.days
    features["days_since_first_purchase"] = features["customer_tenure_days"]
    features["days_since_registration"] = (context.as_of - registration).dt.days
    features["first_to_last_purchase_days"] = (last_purchase - first_purchase).dt.days

    # --- active / inactive months ---
    if orders.empty:
        active_months = pd.Series(0, index=features.index, dtype="int64")
    else:
        months = orders.assign(month=_month_index(orders["purchase_date"]))
        active_months = (
            months.groupby("customer_id", observed=True)["month"]
            .nunique()
            .reindex(features.index, fill_value=0)
        )
    features["active_months"] = active_months

    as_of_month = context.as_of.year * 12 + context.as_of.month
    first_month = _month_index(first_purchase)
    observable_months = (as_of_month - first_month + 1).where(first_purchase.notna())
    features["observable_months"] = observable_months
    features["inactive_months"] = (observable_months - active_months).clip(lower=0)
    features["active_month_rate"] = safe_divide(active_months, observable_months)

    # --- early versus late behaviour ---
    #
    # Orders in the customer's first 90 days versus their most recent 90: a customer who started
    # strongly and has gone quiet looks very different from one who was always slow, even when
    # both have the same lifetime order count.
    if not orders.empty:
        with_first = orders.merge(
            first_purchase.rename("first_purchase").reset_index(), on="customer_id", how="left"
        )
        days_since_first = (with_first["purchase_date"] - with_first["first_purchase"]).dt.days
        early = with_first[days_since_first.le(90)]
        features["orders_first_90d"] = (
            early.groupby("customer_id", observed=True)
            .size()
            .reindex(features.index, fill_value=0)
        )
        features["revenue_first_90d"] = (
            early.groupby("customer_id", observed=True)["order_value"]
            .sum()
            .reindex(features.index, fill_value=0.0)
            .round(2)
        )
    else:
        features["orders_first_90d"] = 0
        features["revenue_first_90d"] = 0.0

    recent = context.orders_within(90)
    features["orders_recent_90d"] = (
        recent.groupby("customer_id", observed=True)
        .size()
        .reindex(features.index, fill_value=0)
    )

    # Only meaningful once the two 90-day windows no longer overlap, otherwise the same orders
    # sit on both sides of the ratio.
    non_overlapping = features["customer_tenure_days"].ge(180)
    features["early_vs_recent_order_ratio"] = safe_divide(
        features["orders_recent_90d"].astype(float), features["orders_first_90d"].astype(float)
    ).where(non_overlapping)

    features["is_repeat_customer"] = features.index.isin(
        grouped.size()[grouped.size().ge(2)].index
    )
    features["is_one_time_buyer"] = features.index.isin(
        grouped.size()[grouped.size().eq(1)].index
    )

    return features
