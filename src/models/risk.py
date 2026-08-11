"""Churn risk bands and revenue at risk.

Bands follow the brief's defaults and stay configurable through the existing risk-threshold
settings, so a CRM manager can move the boundaries without touching code::

    Low       churn probability < 0.30
    Medium               0.30 <= p < 0.60
    High                 0.60 <= p < 0.80
    Critical             0.80 <= p

Boundaries are inclusive at the lower edge, so a probability of exactly 0.60 is High. The brief
writes the bands as "30-60%" and "60-80%", which leaves the boundary ambiguous; making the choice
explicit here means the dashboard and the model agree on where a customer sits.

Revenue at risk
---------------
``revenue_at_risk = churn probability x expected revenue over the horizon``, with expected revenue
computed by :func:`expected_horizon_revenue` as the customer's observed spend rate applied to the
horizon -- and crucially never extrapolated past the history that exists. See that function for why
the obvious annualised version had to be abandoned.

This is an interim estimate. Section 5 replaces it with a fuller expected-future-revenue model using
order frequency, recent behaviour and tenure; the simple version lives here so the predictions file
the brief asks for is complete and internally consistent on its own.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config.settings import Settings, get_settings

__all__ = [
    "RISK_LEVELS",
    "assign_risk_level",
    "expected_horizon_revenue",
    "revenue_at_risk",
    "risk_distribution",
]

#: Ordered from least to most severe, for consistent sorting and chart ordering.
RISK_LEVELS = ("Low", "Medium", "High", "Critical")


def assign_risk_level(
    probability: pd.Series, settings: Settings | None = None
) -> pd.Series:
    """Map churn probabilities to the configured risk bands.

    Returns an ordered categorical, so sorting and grouping put Critical last rather than
    alphabetically between High and Low.
    """
    settings = settings or get_settings()
    edges = [
        -np.inf,
        settings.risk_threshold_medium,
        settings.risk_threshold_high,
        settings.risk_threshold_critical,
        np.inf,
    ]
    # right=False makes the lower edge inclusive: p == 0.60 lands in High.
    bands = pd.cut(
        pd.to_numeric(probability, errors="coerce"),
        bins=edges,
        labels=list(RISK_LEVELS),
        right=False,
        ordered=True,
    )
    return bands.astype(pd.CategoricalDtype(categories=list(RISK_LEVELS), ordered=True))


def risk_distribution(risk_level: pd.Series) -> pd.DataFrame:
    """Customer counts and shares per risk band, in severity order."""
    counts = risk_level.value_counts().reindex(list(RISK_LEVELS), fill_value=0)
    total = int(counts.sum())
    return pd.DataFrame(
        {
            "customers": counts.astype("int64"),
            "share": (counts / total).round(6) if total else 0.0,
        }
    )


def expected_horizon_revenue(
    lifetime_revenue: pd.Series, tenure_days: pd.Series, horizon_days: int
) -> pd.Series:
    """Revenue the customer can be expected to spend over the next ``horizon_days``.

    ``lifetime_revenue x horizon / max(tenure, horizon)``.

    The ``max(tenure, horizon)`` denominator is the important part: it refuses to extrapolate
    beyond what has actually been observed. Annualising instead -- the obvious first approach --
    ranked a customer with a single EUR 780 order and 24 days of history as the single most valuable
    account in the book, because 24 days of spend scaled to a year implies EUR 9,500 a year. Under
    this formula a customer with less than one horizon of history is credited with no more than they
    have actually spent, which is the most that can honestly be claimed about them.

    For customers with a year or more of history the two agree, so nothing is lost where the
    extrapolation was sound in the first place.
    """
    revenue = pd.to_numeric(lifetime_revenue, errors="coerce").fillna(0.0).clip(lower=0.0)
    tenure = pd.to_numeric(tenure_days, errors="coerce").fillna(0.0)
    denominator = tenure.clip(lower=float(horizon_days))
    return (revenue * float(horizon_days) / denominator).round(2)


def revenue_at_risk(
    probability: pd.Series,
    lifetime_revenue: pd.Series,
    tenure_days: pd.Series,
    horizon_days: int,
) -> pd.Series:
    """``churn probability x expected revenue over the horizon``.

    Missing or negative revenue becomes zero exposure rather than propagating NaN into a figure the
    business will sum.
    """
    expected = expected_horizon_revenue(lifetime_revenue, tenure_days, horizon_days)
    return (pd.to_numeric(probability, errors="coerce").fillna(0.0) * expected).round(2)
