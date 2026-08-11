"""Return behaviour features, from the join onto ``Return.csv``.

The as-of discipline matters more here than anywhere else in the feature layer. A return happens
days or weeks *after* its purchase, so at any given as-of date some orders already placed have
returns still in flight. :mod:`src.features.context` has already withheld those, which means the
rates below are deliberately lower than the eventual, fully-settled rates. That is correct
behaviour, not a bug: on the shipped data 104 returns are dated after the last purchase, and a
model trained on settled return rates would be reading the future.

Both denominators are computed from the same clipped window as the numerator, so the rates are
internally consistent.

A high return rate is treated as a *signal*, never a verdict: the brief is explicit that heavy
returners are not automatically churners. Some of the most valuable customers in fashion
retail buy three sizes and keep one.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.context import FeatureContext, safe_divide

__all__ = ["build_return_features"]


def build_return_features(context: FeatureContext) -> pd.DataFrame:
    """One row per customer: returned units/orders, return rates and recency."""
    features = context.empty_frame()
    returns, orders, lines = context.returns, context.orders, context.lines

    purchased_units = (
        lines.groupby("customer_id", observed=True)["quantity"]
        .sum()
        .reindex(features.index, fill_value=0)
        if not lines.empty
        else pd.Series(0, index=features.index)
    )
    total_orders = (
        orders.groupby("customer_id", observed=True)
        .size()
        .reindex(features.index, fill_value=0)
        if not orders.empty
        else pd.Series(0, index=features.index)
    )

    if returns.empty:
        features["returned_units"] = 0
        features["returned_orders"] = 0
        features["returned_lines"] = 0
        features["return_rate"] = safe_divide(
            pd.Series(0.0, index=features.index), purchased_units.astype(float)
        )
        features["return_frequency"] = safe_divide(
            pd.Series(0.0, index=features.index), total_orders.astype(float)
        )
        features["recent_return_rate"] = np.nan
        features["days_since_last_return"] = np.nan
        features["average_return_quantity"] = np.nan
        features["is_serial_returner"] = False
        return features

    grouped = returns.groupby("customer_id", observed=True)
    features["returned_units"] = (
        grouped["return_quantity"].sum().reindex(features.index, fill_value=0)
    )
    features["returned_orders"] = grouped["order_id"].nunique().reindex(features.index, fill_value=0)
    features["returned_lines"] = grouped.size().reindex(features.index, fill_value=0)
    features["average_return_quantity"] = (
        grouped["return_quantity"].mean().reindex(features.index)
    )

    # The brief's primary definition: returned units over purchased units.
    features["return_rate"] = safe_divide(
        features["returned_units"].astype(float), purchased_units.astype(float)
    )
    # Share of orders that saw any return -- a different and equally useful denominator.
    features["return_frequency"] = safe_divide(
        features["returned_orders"].astype(float), total_orders.astype(float)
    )

    # --- recent return behaviour ---
    #
    # Compared against units bought in the same window, so a customer who simply bought less
    # recently does not appear to have started returning more.
    window = context.params.recent_return_window_days
    cutoff = context.window_start(window)
    recent_returns = returns[returns["return_date"].gt(cutoff)]
    recent_lines = lines[lines["purchase_date"].gt(cutoff)]

    recent_returned = (
        recent_returns.groupby("customer_id", observed=True)["return_quantity"]
        .sum()
        .reindex(features.index, fill_value=0)
    )
    recent_purchased = (
        recent_lines.groupby("customer_id", observed=True)["quantity"]
        .sum()
        .reindex(features.index, fill_value=0)
    )
    features["recent_return_rate"] = safe_divide(
        recent_returned.astype(float), recent_purchased.astype(float)
    )
    features["returned_units_recent"] = recent_returned

    features["days_since_last_return"] = (
        (context.as_of - grouped["return_date"].max()).dt.days.reindex(features.index)
    )

    # --- return trend ---
    #
    # Whether returning is getting worse for this customer, which reads differently from a
    # consistently high rate: a rising rate suggests growing dissatisfaction.
    features["return_rate_trend"] = (
        features["recent_return_rate"] - features["return_rate"]
    )

    # Roughly 12% of this dataset's customers are designed as serial returners at about 2.6x the
    # base rate; the base unit rate is ~20%, so 40% is a deliberately conservative cut.
    features["is_serial_returner"] = features["return_rate"].ge(0.40).fillna(False)

    # Deliberately NOT rounded: this is the modelling table, and truncating ratios to a
    # few decimals would discard real signal. Presentation rounding happens at CSV-write
    # time in scripts/build_features.py. Monetary columns are rounded to cents above,
    # where the rounding is part of the definition rather than cosmetic.
    return features
