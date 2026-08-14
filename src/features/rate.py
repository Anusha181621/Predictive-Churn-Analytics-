"""Latent purchase-rate features: how often does this customer buy, and is the silence odd?

Every other feature group describes *what the customer did*. This one estimates the parameter that
generated it -- a per-customer arrival rate -- and then asks the only question the label actually
poses: **given that rate, how surprising is the current silence, and how likely is another order
before the horizon closes?**

Why this is worth a module of its own
-------------------------------------
``rfm.py`` counts orders in fixed windows and ``gaps.py`` measures the current gap against the
customer's own median gap. Both are good, and both degrade in the same place: the low-frequency
customers. A window count of 0 is equally consistent with "buys twice a year, currently fine" and
"buys twice a year, gone for good"; a median gap needs at least two orders to exist at all and is
noise at three. Those customers are precisely where prediction error concentrates, because the
high-frequency ones are easy from any angle.

Rates are estimated three ways, because they answer different questions:

``lifetime_orders_per_year``
    Orders over tenure. The unbiased estimate, and unusable on its own for short tenures -- one
    order three weeks after registering annualises to 17 orders a year.
``shrunk_order_rate``
    The same figure pulled toward a fixed prior (see :class:`~src.features.params.FeatureParams`).
    This is the one downstream features are built on. The prior is a constant, not a cohort mean:
    a cohort mean is recomputed per as-of date and would fingerprint the snapshot.
``orders_per_active_year``
    Orders over the span from first to *last* purchase -- the rate **while the customer was still
    buying**, which deliberately ignores any trailing silence.

The last two together are what separate a genuinely low-rate customer from a lapsed one, and their
ratio is the point of ``active_span_share_of_tenure``: a customer whose buying span covers 95% of
their tenure is current, one whose span stops at 40% of it left months ago. A single rate estimate
cannot express that difference.

From rate to the label
----------------------
Treating orders as arrivals at rate ``r`` gives two directly interpretable columns:

* ``implied_repurchase_probability = 1 - exp(-r * horizon / 365)`` -- an analytic prior on the
  label itself, before the model has learned anything.
* ``silence_survival_probability = exp(-r * recency / 365)`` -- the probability of a silence at
  least this long from a customer who buys at rate ``r``. Small values mean the silence is hard to
  explain away, which is the churn signal in its cleanest form.

These are *priors offered as features*, not predictions. They use no information after the as-of
date: ``horizon`` is a fixed forward window length, not anything observed inside it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.context import FeatureContext, safe_divide

__all__ = ["build_rate_features"]

#: Mean days per year, matching :mod:`src.features.seasonality` so annualisation agrees across
#: the feature layer.
DAYS_PER_YEAR = 365.25


def _per_customer_dates(context: FeatureContext, index: pd.Index) -> pd.DataFrame:
    """First purchase, last purchase and order count per customer, reindexed to every customer."""
    if context.orders.empty:
        return pd.DataFrame(
            {
                "first_purchase": pd.Series(dtype="datetime64[ns]"),
                "last_purchase": pd.Series(dtype="datetime64[ns]"),
                "orders": pd.Series(dtype="float64"),
                "revenue": pd.Series(dtype="float64"),
                "units": pd.Series(dtype="float64"),
            },
            index=index,
        )

    grouped = context.orders.groupby("customer_id", observed=True)
    return pd.DataFrame(
        {
            "first_purchase": grouped["purchase_date"].min(),
            "last_purchase": grouped["purchase_date"].max(),
            "orders": grouped.size().astype("float64"),
            "revenue": grouped["order_value"].sum(),
            "units": grouped["units"].sum(),
        }
    ).reindex(index)


def build_rate_features(context: FeatureContext) -> pd.DataFrame:
    """One row per customer: rate estimates and the horizon/silence probabilities they imply."""
    features = context.empty_frame()
    params = context.params
    facts = _per_customer_dates(context, features.index)

    orders = facts["orders"]
    # A customer with no orders has a rate of zero, not an unknown one -- the absence is the
    # observation. Tenure, by contrast, stays NaN for them: there is no first purchase to date it
    # from, and inventing one would fabricate history.
    orders_observed = orders.fillna(0.0)

    # --- tenure, measured from the first purchase ---
    #
    # From the first purchase rather than the registration date, so the denominator covers only the
    # period the customer could actually have been observed buying. `lifecycle.py` reports both
    # spans; this module needs the purchasing one.
    tenure_days = (context.as_of - facts["first_purchase"]).dt.days.astype("float64")
    # Inclusive of the first day, so a customer who bought once today has one day of tenure rather
    # than zero and cannot divide by it.
    tenure_days = tenure_days.add(1.0)
    tenure_years = tenure_days / DAYS_PER_YEAR

    features["purchasing_tenure_days"] = tenure_days

    # --- 1. the raw annualised rate, withheld where tenure is too short to support it ---
    long_enough = tenure_days.ge(float(params.min_tenure_days_for_rate))
    features["lifetime_orders_per_year"] = safe_divide(orders_observed, tenure_years).where(
        long_enough
    )
    features["revenue_per_tenure_year"] = safe_divide(
        facts["revenue"].fillna(0.0), tenure_years
    ).where(long_enough)
    features["units_per_tenure_year"] = safe_divide(
        facts["units"].fillna(0.0), tenure_years
    ).where(long_enough)

    # --- 2. the shrunk rate: defined for everyone, and the basis for everything below ---
    #
    # (orders + a) / (tenure_years + b). Short tenures sit near the prior a/b and long ones
    # converge on the raw figure, so one column is usable across the whole cohort without the
    # short-tenure blow-up that makes the raw rate unusable.
    prior_orders = float(params.rate_prior_orders)
    prior_years = float(params.rate_prior_years)
    shrunk = (orders_observed + prior_orders) / (tenure_years.fillna(0.0) + prior_years)
    features["shrunk_order_rate"] = shrunk
    features["rate_implied_interval_days"] = DAYS_PER_YEAR / shrunk

    # How much the shrinkage moved the estimate -- large where the evidence is thin, ~0 where the
    # customer has enough history to speak for themselves. Lets the model discount the rate
    # features exactly where they are least trustworthy, instead of trusting them uniformly.
    features["rate_shrinkage_pull"] = (
        shrunk - features["lifetime_orders_per_year"]
    ).abs()

    # --- 3. the rate while still active, and the span it covers ---
    active_span_days = (
        (facts["last_purchase"] - facts["first_purchase"]).dt.days.astype("float64").add(1.0)
    )
    features["active_span_days"] = active_span_days
    # Needs a second order for the span to mean anything; one order is a point, not a span.
    measurable_span = orders_observed.ge(2.0) & active_span_days.gt(0.0)
    features["orders_per_active_year"] = safe_divide(
        orders_observed, active_span_days / DAYS_PER_YEAR
    ).where(measurable_span)

    # The wall-detector. 1.0 means the customer was still buying at the as-of date; a low value
    # means their buying stopped partway through their tenure and never resumed. This is the
    # cleanest available read on "did this relationship already end", and unlike raw recency it is
    # normalised by how long the customer has been around -- three months of silence means
    # something very different at six months' tenure than at three years'.
    features["active_span_share_of_tenure"] = safe_divide(active_span_days, tenure_days)

    # Rate while active versus rate over the whole tenure. Above 1 means the trailing silence is
    # dragging the lifetime figure down, i.e. the customer used to buy faster than they do now.
    features["active_vs_lifetime_rate_ratio"] = safe_divide(
        features["orders_per_active_year"], features["lifetime_orders_per_year"]
    )

    # --- 4. project the rate onto the outcome window and onto the observed silence ---
    horizon_years = float(params.outcome_horizon_days) / DAYS_PER_YEAR
    expected_in_horizon = shrunk * horizon_years
    features["expected_orders_in_horizon"] = expected_in_horizon
    # 1 - exp(-lambda): the chance of at least one arrival in the window. This is a prior on the
    # *complement* of the churn label, computed from history alone.
    features["implied_repurchase_probability"] = 1.0 - np.exp(-expected_in_horizon)

    recency_days = (context.as_of - facts["last_purchase"]).dt.days.astype("float64")
    expected_since_last = shrunk * (recency_days / DAYS_PER_YEAR)
    # "How many orders should have arrived since the last one, and did not." A customer who buys
    # ten times a year and has been quiet nine months is missing seven and a half orders; a
    # once-a-year buyer quiet the same nine months is missing three quarters of one.
    features["missed_expected_orders"] = expected_since_last
    # The survival function of that silence. Near 1 = unremarkable, near 0 = very hard to explain
    # for a customer who buys at this rate.
    #
    # `missed_expected_orders` is also, by construction, the current silence measured in
    # rate-implied intervals -- `rate x recency / 365` and `recency / (365 / rate)` are the same
    # number. So there is deliberately no separate `recency_vs_rate_interval` column here: it
    # would be an exact duplicate under a different name, and duplicated columns split a tree's
    # importance between two identical splits for no gain. This is the rate-based counterpart to
    # `gaps.purchase_gap_ratio`, which uses the observed median gap and is undefined for
    # single-order customers where this one still works.
    features["silence_survival_probability"] = np.exp(-expected_since_last)

    # Deliberately NOT rounded: this is the modelling table, and truncating ratios to a
    # few decimals would discard real signal. Presentation rounding happens at CSV-write
    # time in scripts/build_features.py. Monetary columns are rounded to cents above,
    # where the rounding is part of the definition rather than cosmetic.
    return features
