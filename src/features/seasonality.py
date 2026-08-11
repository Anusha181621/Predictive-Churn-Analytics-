"""Seasonality features.

This module exists to satisfy one explicit instruction: *do not classify seasonal customers as
churned merely because of a long purchase gap*. A customer who buys coats every October and
nothing else looks, every June, exactly like a churner to a recency rule. The features here give
the model and the segment logic a way to tell the two apart.

Why circular statistics rather than a month histogram
-----------------------------------------------------
The obvious approach -- concentration across the twelve calendar months -- has a flaw that
matters for a fashion retailer: December and January are adjacent in the year but maximally far
apart as bucket labels. A customer who reliably shops the Christmas-and-January-sales window
would score as *unseasonal* under a plain histogram, which is precisely backwards.

So purchase dates are mapped onto a circle (day of year to an angle) and the resultant vector
length ``R`` measures how tightly clustered they are, with December and January correctly
adjacent. ``R`` also yields a mean *direction* -- the middle of the customer's buying season --
which is what makes ``in_preferred_season`` possible.

Bias correction
---------------
Raw ``R`` is 1.0 for a single purchase and stays high for two, so an unseasonal customer with
little history would look perfectly seasonal by accident. ``seasonal_customer_score`` therefore
uses the bias-corrected form ``(n·R² − 1) / (n − 1)``, which is ~0 for randomly scattered dates
at any ``n``, and additionally requires a minimum number of orders spread over at least two
calendar years -- one burst in one December is not a season.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.context import FeatureContext

__all__ = ["build_seasonality_features"]

#: Mean days per year, so the angle mapping does not drift across leap years.
DAYS_PER_YEAR = 365.25


def _circular_statistics(orders: pd.DataFrame, index: pd.Index) -> pd.DataFrame:
    """Resultant length, mean day-of-year and order count per customer."""
    columns = ["circular_r", "preferred_day_of_year", "seasonal_order_count", "seasonal_years"]
    if orders.empty:
        return pd.DataFrame(columns=columns, index=index, dtype="float64")

    day_of_year = orders["purchase_date"].dt.dayofyear
    angle = 2.0 * np.pi * (day_of_year - 1) / DAYS_PER_YEAR
    working = orders.assign(
        cos_angle=np.cos(angle),
        sin_angle=np.sin(angle),
        year=orders["purchase_date"].dt.year,
    )
    grouped = working.groupby("customer_id", observed=True)
    mean_cos = grouped["cos_angle"].mean()
    mean_sin = grouped["sin_angle"].mean()

    resultant = np.sqrt(mean_cos**2 + mean_sin**2)
    mean_angle = np.arctan2(mean_sin, mean_cos) % (2.0 * np.pi)

    stats = pd.DataFrame(
        {
            "circular_r": resultant,
            "preferred_day_of_year": (mean_angle / (2.0 * np.pi) * DAYS_PER_YEAR) + 1.0,
            "seasonal_order_count": grouped.size(),
            "seasonal_years": grouped["year"].nunique(),
        }
    )
    return stats.reindex(index)


def _month_concentration(orders: pd.DataFrame, index: pd.Index, period: str) -> pd.Series:
    """Normalised Herfindahl concentration of orders across months or quarters.

    0 means orders are spread evenly across every bucket; 1 means they all fall in one. Reported
    because the brief asks for it by name -- but note it inherits the December/January adjacency
    problem described in the module docstring, which is why ``seasonal_customer_score`` uses the
    circular measure instead.
    """
    if orders.empty:
        return pd.Series(index=index, dtype="float64")

    buckets = 12 if period == "month" else 4
    key = (
        orders["purchase_date"].dt.month
        if period == "month"
        else orders["purchase_date"].dt.quarter
    )
    counts = orders.assign(bucket=key).groupby(["customer_id", "bucket"], observed=True).size()
    totals = counts.groupby(level="customer_id").transform("sum")
    share = counts / totals
    herfindahl = (share**2).groupby(level="customer_id").sum()
    uniform = 1.0 / buckets
    return ((herfindahl - uniform) / (1.0 - uniform)).reindex(index)


def _mode_bucket(orders: pd.DataFrame, index: pd.Index, period: str) -> pd.Series:
    """The customer's busiest month or quarter, ties broken by the earlier bucket."""
    if orders.empty:
        return pd.Series(index=index, dtype="float64")
    key = (
        orders["purchase_date"].dt.month
        if period == "month"
        else orders["purchase_date"].dt.quarter
    )
    counts = (
        orders.assign(bucket=key)
        .groupby(["customer_id", "bucket"], observed=True)
        .size()
        .rename("orders")
        .reset_index()
        .sort_values(["customer_id", "orders", "bucket"], ascending=[True, False, True])
    )
    return counts.drop_duplicates("customer_id").set_index("customer_id")["bucket"].reindex(index)


def _circular_distance_days(day_a: pd.Series, day_b: float) -> pd.Series:
    """Shortest distance in days between two days of the year, going either way round."""
    raw = (day_a - day_b).abs() % DAYS_PER_YEAR
    return np.minimum(raw, DAYS_PER_YEAR - raw)


def build_seasonality_features(context: FeatureContext) -> pd.DataFrame:
    """One row per customer: seasonal concentration, preferred period, and in-season status."""
    features = context.empty_frame()
    orders = context.orders
    params = context.params

    features["preferred_purchase_month"] = _mode_bucket(orders, features.index, "month")
    features["preferred_purchase_quarter"] = _mode_bucket(orders, features.index, "quarter")
    features["seasonal_purchase_concentration"] = _month_concentration(
        orders, features.index, "month"
    )
    features["quarterly_purchase_concentration"] = _month_concentration(
        orders, features.index, "quarter"
    )

    stats = _circular_statistics(orders, features.index)
    features["circular_concentration"] = stats["circular_r"]
    features["preferred_day_of_year"] = stats["preferred_day_of_year"]
    features["purchase_years_spanned"] = stats["seasonal_years"]

    # --- bias-corrected seasonality score ---
    count = stats["seasonal_order_count"]
    resultant = stats["circular_r"]
    corrected = (count * resultant**2 - 1.0) / (count - 1.0)
    score = corrected.clip(lower=0.0, upper=1.0)

    # Only trust the score where there is enough evidence for it to mean anything.
    enough_evidence = count.ge(params.min_orders_for_seasonality) & stats["seasonal_years"].ge(
        params.min_years_for_seasonality
    )
    features["seasonal_customer_score"] = score.where(enough_evidence)
    features["is_seasonal_buyer"] = (
        features["seasonal_customer_score"].ge(params.seasonal_score_threshold).fillna(False)
    )

    # --- is the customer in their season right now? ---
    as_of_day = float(context.as_of.dayofyear)
    distance = _circular_distance_days(features["preferred_day_of_year"], as_of_day)
    features["days_from_preferred_season"] = distance
    features["in_preferred_season"] = distance.le(params.season_halfwidth_days).fillna(False)

    # --- how many whole buying cycles have gone by unused? ---
    #
    # A season recurs annually, so a silence longer than a year means the customer's season came
    # round and they did not buy. That is a churn signal in its own right, and it is what stops
    # the shield below from excusing an absence of any length.
    last_purchase = (
        context.orders.groupby("customer_id", observed=True)["purchase_date"]
        .max()
        .reindex(features.index)
    )
    recency = (context.as_of - last_purchase).dt.days
    features["annual_cycles_missed"] = (recency / DAYS_PER_YEAR).apply(
        lambda value: float(int(value)) if pd.notna(value) else value
    )
    features["missed_full_season"] = features["annual_cycles_missed"].ge(1.0).fillna(False)

    # --- the feature that stops a seasonal buyer being mistaken for a churner ---
    #
    # Read as: "this customer is quiet, and quiet is exactly what we should expect from them
    # right now." The segment logic and the churn label consult this instead of applying a flat
    # recency rule. Two deliberate asymmetries keep it from becoming a blanket excuse:
    #
    #   * A seasonal customer silent *during* their own season is genuinely worrying, so being
    #     in season removes the shield.
    #   * A seasonal customer who has been silent for over a year has already skipped a whole
    #     season, so `missed_full_season` removes the shield too. Without this, a customer two
    #     years absent would be excused indefinitely for being "out of season" -- which is how a
    #     well-meant seasonality rule turns into a blind spot.
    features["seasonally_explained_inactivity"] = (
        features["is_seasonal_buyer"]
        & ~features["in_preferred_season"]
        & ~features["missed_full_season"]
    )

    # Deliberately NOT rounded: this is the modelling table, and truncating ratios to a
    # few decimals would discard real signal. Presentation rounding happens at CSV-write
    # time in scripts/build_features.py. Monetary columns are rounded to cents above,
    # where the rounding is part of the definition rather than cosmetic.
    return features
