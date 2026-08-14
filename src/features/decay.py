"""Intensity-decay features: is this customer's buying winding down, and how fast?

``trends.py`` already compares a recent window against the one before it. That answers "what
changed lately" and it is the right question for a frequent buyer, but it has a blind spot that
matters here: a customer who buys three times a year has 0 orders in most 90-day windows, so the
window-on-window comparison is 0 against 0 -- undefined -- for most of the cohort most of the time.
The decline is real and spread across two years; the comparison cannot see it.

This module measures the same thing over the customer's **whole history** instead of two windows,
which is stable at low order counts:

* **Halves.** Orders in the first versus the second half of the customer's own tenure. One number,
  no fitting, defined for anybody with a tenure -- and already enough to separate a customer whose
  buying stopped early from one still going.
* **Slope.** Orders bucketed into quarters and regressed on the bucket index, normalised by the
  customer's mean bucket count so it reads as a proportional decline per quarter rather than an
  absolute one. Reported alongside ``order_intensity_r2`` so the model can tell a clean trend from
  a line drawn through scatter -- a slope without its goodness of fit is an overconfident number.
* **Recency shares.** The fraction of a customer's lifetime orders that fall in the last 180 and
  365 days. Lifetime-normalised, so they are comparable between a customer with 4 orders and one
  with 40, which the raw ``orders_180d`` counters are not.

Why the halves and the slope are both here
------------------------------------------
They fail differently. The halves split is robust but coarse -- it cannot distinguish a steady
decline from a cliff at the midpoint. The slope resolves the shape but needs enough buckets to fit,
and returns NaN below ``min_buckets_for_decay`` rather than a fitted line through two points. Each
covers the other's gap, and NaN is a signal the tree models read natively.

All windows are anchored on the as-of date and every count comes from
:class:`~src.features.context.FeatureContext`, so nothing here can see past the prediction date.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.context import FeatureContext, safe_divide

__all__ = ["build_decay_features"]


def _bucketed_slope(
    orders: pd.DataFrame,
    index: pd.Index,
    as_of: pd.Timestamp,
    bucket_days: int,
    min_buckets: int,
) -> pd.DataFrame:
    """Least-squares slope of orders-per-bucket against bucket index, per customer.

    Buckets run forward from each customer's own first purchase, so bucket 0 is always their first
    period and the slope describes their personal history rather than the calendar. Empty buckets
    inside the span count as genuine zeros -- a quarter with no orders is the strongest evidence of
    decay there is, and dropping it would fit the trend only through the quarters that went well.
    """
    columns = ["order_intensity_slope", "order_intensity_r2", "decay_buckets"]
    if orders.empty:
        return pd.DataFrame(columns=columns, index=index, dtype="float64")

    grouped = orders.groupby("customer_id", observed=True)["purchase_date"]
    first = grouped.min()
    # Number of whole buckets between the first purchase and the as-of date, at least one.
    span_buckets = (
        ((as_of - first).dt.days // bucket_days).astype("int64").add(1).clip(lower=1)
    )

    working = orders[["customer_id", "purchase_date"]].copy()
    working["bucket"] = (
        (working["purchase_date"] - working["customer_id"].map(first)).dt.days // bucket_days
    ).astype("int64")
    counts = working.groupby(["customer_id", "bucket"], observed=True).size()

    results: dict[str, list[float]] = {name: [] for name in columns}
    customers: list[object] = []
    for customer, n_buckets in span_buckets.items():
        customers.append(customer)
        if n_buckets < min_buckets:
            for name in columns[:2]:
                results[name].append(np.nan)
            results["decay_buckets"].append(float(n_buckets))
            continue

        # Dense series over the whole span, so silent buckets contribute their zeros.
        y = np.zeros(int(n_buckets), dtype="float64")
        try:
            observed = counts.loc[customer]
        except KeyError:  # pragma: no cover - a customer in the span index always has orders
            observed = pd.Series(dtype="float64")
        for bucket, count in observed.items():
            position = int(bucket)
            if 0 <= position < len(y):
                y[position] = float(count)

        x = np.arange(len(y), dtype="float64")
        mean_y = y.mean()
        slope, intercept = np.polyfit(x, y, 1)
        # Normalised by the customer's own mean bucket count, so -0.5 reads as "losing half an
        # average quarter's worth of orders per quarter" for a heavy and a light buyer alike.
        results["order_intensity_slope"].append(
            float(slope / mean_y) if mean_y > 0 else np.nan
        )

        total_variance = float(((y - mean_y) ** 2).sum())
        residual = float(((y - (slope * x + intercept)) ** 2).sum())
        results["order_intensity_r2"].append(
            float(1.0 - residual / total_variance) if total_variance > 0 else np.nan
        )
        results["decay_buckets"].append(float(n_buckets))

    return pd.DataFrame(results, index=pd.Index(customers, name=index.name)).reindex(index)


def build_decay_features(context: FeatureContext) -> pd.DataFrame:
    """One row per customer: tenure-halves comparison, fitted decay slope and recency shares."""
    features = context.empty_frame()
    params = context.params
    orders = context.orders

    if orders.empty:
        first_purchase = pd.Series(index=features.index, dtype="datetime64[ns]")
        lifetime_orders = pd.Series(0.0, index=features.index, dtype="float64")
    else:
        grouped = orders.groupby("customer_id", observed=True)
        first_purchase = grouped["purchase_date"].min().reindex(features.index)
        lifetime_orders = (
            grouped.size().astype("float64").reindex(features.index).fillna(0.0)
        )

    # --- 1. first half of tenure versus second half ---
    midpoint = first_purchase + (context.as_of - first_purchase) / 2
    if orders.empty:
        first_half = pd.Series(0.0, index=features.index, dtype="float64")
        second_half = pd.Series(0.0, index=features.index, dtype="float64")
    else:
        customer_midpoint = orders["customer_id"].map(midpoint)
        in_first = orders["purchase_date"].le(customer_midpoint)
        first_half = (
            orders[in_first]
            .groupby("customer_id", observed=True)
            .size()
            .astype("float64")
            .reindex(features.index)
            .fillna(0.0)
        )
        second_half = (
            orders[~in_first]
            .groupby("customer_id", observed=True)
            .size()
            .astype("float64")
            .reindex(features.index)
            .fillna(0.0)
        )

    features["orders_first_half_tenure"] = first_half
    features["orders_second_half_tenure"] = second_half
    # Laplace-smoothed, so a customer who bought 4 times early and 0 times lately gets a defined
    # ratio (0.2) rather than a zero that a plain division would render as an undefined 0/4 for
    # the reverse case. Below 1 means the second half was quieter.
    features["tenure_half_order_ratio"] = (second_half + 1.0) / (first_half + 1.0)
    # Share of lifetime orders that landed in the second half. 0.5 is flat, near 0 is a customer
    # who stopped, near 1 is one who is accelerating or simply new.
    features["second_half_order_share"] = safe_divide(second_half, lifetime_orders)

    # --- 2. the fitted slope over quarterly buckets ---
    slope = _bucketed_slope(
        orders,
        features.index,
        context.as_of,
        int(params.decay_bucket_days),
        int(params.min_buckets_for_decay),
    )
    for column in slope.columns:
        features[column] = slope[column]

    # --- 3. lifetime-normalised recency shares ---
    #
    # Deliberately shares rather than counts: `rfm.py` already provides orders_180d/orders_365d,
    # and what those cannot say is whether 2 orders in the last year is this customer's whole
    # history or a collapse from 20.
    for window in (180, 365):
        recent = (
            context.orders_within(window)
            .groupby("customer_id", observed=True)
            .size()
            .astype("float64")
            .reindex(features.index)
            .fillna(0.0)
        )
        features[f"order_share_last_{window}d"] = safe_divide(recent, lifetime_orders)

    # Silence as a fraction of the customer's whole history. Unlike raw recency this is
    # scale-free: 90 days quiet is routine at three years' tenure and terminal at four months'.
    last_purchase = (
        orders.groupby("customer_id", observed=True)["purchase_date"].max().reindex(features.index)
        if not orders.empty
        else pd.Series(index=features.index, dtype="datetime64[ns]")
    )
    recency_days = (context.as_of - last_purchase).dt.days.astype("float64")
    tenure_days = (context.as_of - first_purchase).dt.days.astype("float64").add(1.0)
    features["recency_share_of_tenure"] = safe_divide(recency_days, tenure_days)

    # Deliberately NOT rounded: this is the modelling table, and truncating ratios to a
    # few decimals would discard real signal. Presentation rounding happens at CSV-write
    # time in scripts/build_features.py. Monetary columns are rounded to cents above,
    # where the rounding is part of the definition rather than cosmetic.
    return features
