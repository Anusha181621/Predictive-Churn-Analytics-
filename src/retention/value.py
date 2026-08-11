"""Expected future revenue.

The brief asks for a projection built from historical order frequency, average order value, recent
behaviour, customer tenure and historical annual revenue. All five are here, combined as a
frequency x value model rather than a single ratio, because they answer different questions:

    expected orders    = blend(lifetime order rate, recent order rate)
    expected value     = blend(lifetime AOV,        recent AOV)
    expected revenue   = expected orders x expected value, pro-rated to the horizon

**Recent behaviour leads the blend** (60/40 by default) because the next order resembles the last
few far more than the lifetime average -- but it does not dominate, because a single quiet quarter
should not erase three years of history.

**Tenure governs how far the projection may reach.** This is the part that matters most, and it is
the lesson from Section 3's interim estimate: annualising a customer with one EUR 780 order and 24
days of history implied EUR 9,500 a year and put them top of the revenue-at-risk table. Two guards
now prevent that. Customers below ``min_tenure_days_for_projection`` have their rate computed against
that floor rather than their actual tenure, and *every* projection is capped at a multiple of
observed lifetime revenue, so no figure can outrun the evidence behind it.

**Historical annual revenue** is carried as a cross-check rather than an input to the blend: the
projection is reported alongside it with the ratio between them, so a reviewer can see when the
frequency x value model disagrees with the simple annualisation and why.

Nothing here uses a future transaction. Every input is a feature computed as of the prediction date
by :mod:`src.features`, which clipped transactions *and* returns to that date.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.retention.params import RetentionParams
from src.utils.logging_config import get_logger

__all__ = ["build_expected_revenue"]

logger = get_logger(__name__)

DAYS_PER_YEAR = 365.25


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return numerator / denominator.where(denominator.gt(0))


def build_expected_revenue(
    features: pd.DataFrame, params: RetentionParams | None = None
) -> pd.DataFrame:
    """Project each customer's revenue over the next ``revenue_horizon_days``.

    Parameters
    ----------
    features:
        The customer feature table, indexed or keyed by ``customer_id``.

    Returns
    -------
    A frame indexed by ``customer_id`` carrying the projection and every intermediate quantity, so
    the number is auditable rather than a black box: a CRM manager who asks "why is this customer
    worth EUR 400 to us" gets expected orders, expected order value and the cap that applied.
    """
    params = params or RetentionParams()
    params.validate()

    frame = features.set_index("customer_id") if "customer_id" in features.columns else features
    out = pd.DataFrame(index=frame.index)

    horizon = float(params.revenue_horizon_days)
    horizon_years = horizon / DAYS_PER_YEAR

    tenure = pd.to_numeric(frame["customer_tenure_days"], errors="coerce").fillna(0.0)
    total_orders = pd.to_numeric(frame["total_orders"], errors="coerce").fillna(0.0)
    lifetime_revenue = pd.to_numeric(frame["lifetime_revenue"], errors="coerce").fillna(0.0)
    orders_365d = pd.to_numeric(frame.get("orders_365d", 0), errors="coerce").fillna(0.0)
    revenue_365d = pd.to_numeric(frame.get("revenue_365d", 0), errors="coerce").fillna(0.0)

    # --- 1. historical order frequency, per year ---
    #
    # The denominator is floored at the minimum projection tenure, so a three-week-old customer's
    # two orders do not imply 35 orders a year.
    effective_tenure = tenure.clip(lower=float(params.min_tenure_days_for_projection))
    out["observed_tenure_days"] = tenure
    out["effective_tenure_days"] = effective_tenure
    out["tenure_floored"] = tenure.lt(params.min_tenure_days_for_projection)

    lifetime_order_rate = _safe_divide(total_orders, effective_tenure / DAYS_PER_YEAR)
    out["lifetime_orders_per_year"] = lifetime_order_rate.fillna(0.0)

    # --- 2. recent behaviour ---
    #
    # For a customer with a full year of history, orders_365d *is* the annual rate. For a younger
    # customer there is no complete recent year, so their lifetime rate stands in rather than
    # inventing a partial-year extrapolation.
    has_full_year = tenure.ge(DAYS_PER_YEAR)
    recent_order_rate = orders_365d.where(has_full_year, lifetime_order_rate)
    out["recent_orders_per_year"] = recent_order_rate.fillna(0.0)

    weight = float(params.recent_behaviour_weight)
    expected_orders_per_year = (
        weight * recent_order_rate.fillna(0.0) + (1.0 - weight) * lifetime_order_rate.fillna(0.0)
    )
    out["expected_orders_per_year"] = expected_orders_per_year

    # --- 3. average order value, lifetime and recent ---
    lifetime_aov = _safe_divide(lifetime_revenue, total_orders)
    recent_aov = _safe_divide(revenue_365d, orders_365d)
    # Where there is no recent order there is no recent AOV, so the lifetime figure carries it.
    blended_aov = (
        weight * recent_aov.fillna(lifetime_aov) + (1.0 - weight) * lifetime_aov
    ).fillna(0.0)
    out["lifetime_average_order_value"] = lifetime_aov.fillna(0.0)
    out["recent_average_order_value"] = recent_aov.fillna(0.0)
    out["expected_average_order_value"] = blended_aov

    # --- 4. the projection ---
    raw_annual = (expected_orders_per_year * blended_aov).fillna(0.0).clip(lower=0.0)
    out["projected_annual_revenue"] = raw_annual.round(2)

    raw_horizon = raw_annual * horizon_years

    # --- 5. the cap that keeps a projection tethered to the evidence ---
    #
    # Never claim more over the horizon than a multiple of what the customer has actually spent in
    # their whole life with the brand.
    ceiling = lifetime_revenue * float(params.max_projection_multiple)
    capped = np.minimum(raw_horizon, ceiling)
    out["projection_capped"] = raw_horizon.gt(ceiling) & lifetime_revenue.gt(0)
    out["expected_future_revenue"] = pd.Series(capped, index=out.index).clip(lower=0.0).round(2)

    # --- 6. the cross-check ---
    historical_annual = pd.to_numeric(
        frame.get("annualized_revenue", np.nan), errors="coerce"
    )
    out["historical_annual_revenue"] = historical_annual.round(2)
    out["projection_vs_historical_ratio"] = _safe_divide(
        out["projected_annual_revenue"], historical_annual
    ).round(4)

    out["revenue_horizon_days"] = params.revenue_horizon_days

    logger.info(
        "Expected future revenue over %d days: total %s %s, mean %s %s "
        "(%d projection(s) capped at %.1fx lifetime revenue, %d tenure denominator(s) floored)",
        params.revenue_horizon_days,
        params.currency,
        f"{out['expected_future_revenue'].sum():,.2f}",
        params.currency,
        f"{out['expected_future_revenue'].mean():,.2f}",
        int(out["projection_capped"].sum()),
        params.max_projection_multiple,
        int(out["tenure_floored"].sum()),
    )
    return out
