"""Inter-purchase gap features.

Gaps are measured between consecutive *orders*, not order lines: three lines of one basket are
one purchase occasion, and treating them as three would collapse every gap to zero.

The point of this module is ``purchase_gap_ratio``. A flat "no purchase in 180 days" rule
punishes a twice-a-year buyer and lets a weekly buyer go three weeks unnoticed. Dividing the
current gap by the customer's *own* historical cadence normalises that away, which is what the
brief asks for when it distinguishes true churn risk from normal inactivity.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.context import FeatureContext, safe_divide

__all__ = ["build_gap_features"]


def _gap_statistics(orders: pd.DataFrame) -> pd.DataFrame:
    """Mean / median / max / std of the day gaps between consecutive orders per customer."""
    if orders.empty:
        return pd.DataFrame(
            columns=["average_purchase_gap", "median_purchase_gap", "maximum_purchase_gap",
                     "minimum_purchase_gap", "purchase_gap_std", "observed_gaps"]
        )

    ordered = orders.sort_values(["customer_id", "purchase_date"])
    gaps = (
        ordered.groupby("customer_id", observed=True)["purchase_date"]
        .diff()
        .dt.days
    )
    ordered = ordered.assign(gap_days=gaps)
    # The first order of each customer has no preceding order, so no gap.
    measured = ordered.dropna(subset=["gap_days"])

    grouped = measured.groupby("customer_id", observed=True)["gap_days"]
    return pd.DataFrame(
        {
            "average_purchase_gap": grouped.mean(),
            "median_purchase_gap": grouped.median(),
            "maximum_purchase_gap": grouped.max(),
            "minimum_purchase_gap": grouped.min(),
            "purchase_gap_std": grouped.std(),
            "observed_gaps": grouped.size(),
        }
    )


def build_gap_features(context: FeatureContext) -> pd.DataFrame:
    """One row per customer: gap statistics, the current gap, and the ratio between them."""
    features = context.empty_frame()
    stats = _gap_statistics(context.orders).reindex(features.index)

    for column in stats.columns:
        features[column] = stats[column]
    features["observed_gaps"] = features["observed_gaps"].fillna(0).astype("int64")

    # --- the current gap ---
    last_purchase = (
        context.orders.groupby("customer_id", observed=True)["purchase_date"]
        .max()
        .reindex(features.index)
    )
    features["current_purchase_gap"] = (context.as_of - last_purchase).dt.days

    # --- the customer's own expected cadence ---
    #
    # Median rather than mean: a single holiday-season binge or one long dormant stretch would
    # drag the mean far from the customer's typical rhythm, and the ratio below would inherit
    # that distortion. Customers with only one order have no measurable cadence, so they fall
    # back to a configured default and are flagged.
    expected = features["median_purchase_gap"].copy()
    features["has_measurable_cadence"] = expected.notna()
    expected = expected.fillna(float(context.params.default_expected_interval_days))
    # A same-day repeat buyer would otherwise divide by zero.
    features["expected_purchase_interval_days"] = expected.clip(lower=1.0)

    features["purchase_gap_ratio"] = safe_divide(
        features["current_purchase_gap"], features["expected_purchase_interval_days"]
    )

    # How unusual is the current silence compared with the longest gap the customer has ever
    # come back from? Above 1 means they have never before been away this long.
    features["gap_vs_max_gap_ratio"] = safe_divide(
        features["current_purchase_gap"], features["maximum_purchase_gap"]
    )

    # Regularity: a low coefficient of variation means a predictable rhythm, which makes the gap
    # ratio trustworthy. High variation means the customer was always erratic, so a long silence
    # says less about them. Mapped to (0, 1] so that 1.0 is perfectly clockwork.
    features["purchase_gap_cv"] = safe_divide(
        features["purchase_gap_std"], features["average_purchase_gap"]
    )
    features["purchase_regularity"] = 1.0 / (1.0 + features["purchase_gap_cv"])

    # Deliberately NOT rounded: this is the modelling table, and truncating ratios to a
    # few decimals would discard real signal. Presentation rounding happens at CSV-write
    # time in scripts/build_features.py. Monetary columns are rounded to cents above,
    # where the rounding is part of the definition rather than cosmetic.
    return features
