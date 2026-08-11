"""Customer value features.

``annualized_revenue`` is the load-bearing one: Section 5 multiplies it by churn probability to
get revenue at risk, so an inflated value here becomes an inflated business case there. The
denominator is floored at a configurable minimum tenure, because dividing a new customer's first
EUR 200 by three days of history would annualise to EUR 24,000 and make them look like the most
valuable account in the book.

``customer_value_segment`` is deliberately *relative* -- quantiles of lifetime revenue within the
cohort observed at this as-of date. High value means high value compared with this brand's other
customers, which is the comparison a CRM manager actually makes. It follows that the segment of a
given customer can change between as-of dates even if their own spending did not, and that is
intended.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.context import FeatureContext, safe_divide

__all__ = ["build_value_features"]

#: Label used where a customer has no purchase history to value.
NO_HISTORY = "No History"


def build_value_features(context: FeatureContext, rfm: pd.DataFrame, lifecycle: pd.DataFrame) -> pd.DataFrame:
    """One row per customer: annualised revenue and a relative value segment.

    Takes the already-computed RFM and lifecycle frames rather than recomputing revenue and
    tenure, so there is exactly one definition of each in the codebase.
    """
    features = context.empty_frame()
    params = context.params

    lifetime_revenue = rfm["lifetime_revenue"]
    tenure_days = lifecycle["customer_tenure_days"]
    has_history = context.has_history

    features["lifetime_revenue"] = lifetime_revenue
    features["average_order_value"] = rfm["average_order_value"]

    # --- annualised revenue ---
    effective_tenure = tenure_days.clip(lower=params.min_tenure_days_for_annualisation)
    features["annualized_revenue"] = safe_divide(
        lifetime_revenue * 365.25, effective_tenure.astype(float)
    ).round(2)
    # Flag the customers whose annualisation was floored, so downstream code can discount the
    # figure rather than trust an extrapolation from a handful of days.
    features["annualisation_floored"] = (
        tenure_days.lt(params.min_tenure_days_for_annualisation) & has_history
    ).fillna(False)

    features["revenue_per_order"] = rfm["average_order_value"]
    features["revenue_per_active_month"] = safe_divide(
        lifetime_revenue, lifecycle["active_months"].astype(float)
    )

    # --- relative value segment ---
    purchasers = lifetime_revenue.where(has_history & lifetime_revenue.gt(0)).dropna()
    if purchasers.empty:
        # Nobody has spent anything yet, so there is no cohort to rank against.
        features["customer_value_segment"] = NO_HISTORY
        features["value_percentile"] = np.nan
        return features

    high_cut = purchasers.quantile(params.high_value_quantile)
    medium_cut = purchasers.quantile(params.medium_value_quantile)

    segment = pd.Series(NO_HISTORY, index=features.index, dtype="object")
    scored = has_history & lifetime_revenue.notna()
    segment[scored] = "Low Value"
    segment[scored & lifetime_revenue.ge(medium_cut)] = "Medium Value"
    segment[scored & lifetime_revenue.ge(high_cut)] = "High Value"
    features["customer_value_segment"] = segment

    # Percentile rank within the purchasing cohort: a continuous companion to the three bands,
    # useful for ranking and for the prioritisation score in Section 5.
    features["value_percentile"] = (
        lifetime_revenue.where(scored).rank(pct=True).round(4)
    )
    features["high_value_threshold"] = round(float(high_cut), 2)
    features["medium_value_threshold"] = round(float(medium_cut), 2)

    # Deliberately NOT rounded: this is the modelling table, and truncating ratios to a
    # few decimals would discard real signal. Presentation rounding happens at CSV-write
    # time in scripts/build_features.py. Monetary columns are rounded to cents above,
    # where the rounding is part of the definition rather than cosmetic.
    return features
