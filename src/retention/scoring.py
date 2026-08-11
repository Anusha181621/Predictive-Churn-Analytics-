"""Revenue at risk, retention propensity, the opportunity score, and campaign ROI.

    revenue at risk              = churn probability x expected future revenue
    retention propensity         = base assumption x behavioural multipliers   [ASSUMPTION]
    retention opportunity score  = churn probability x expected future revenue x propensity
                                 = revenue at risk x propensity
    expected retained revenue    = revenue at risk x propensity
    expected ROI                 = (expected retained revenue - campaign cost) / campaign cost

The propensity term is what turns "how much is at stake" into "how much is *winnable*", and it is the
reason the priority list is not simply the revenue-at-risk list. A customer two years gone with EUR
2,000 at risk ranks below one who lapsed last month with EUR 900 at risk, because the second is far
more likely to come back — which is the decision a retention team actually faces.

Every propensity number is an unvalidated assumption. This dataset has no campaign log and no
control group, so intervention uplift cannot be estimated from it; see
:mod:`src.retention.params`. Consequently:

* every propensity-derived column is accompanied by ``propensity_is_assumption = True``;
* the multipliers applied to each customer are recorded in ``propensity_basis``, so the number can be
  taken apart rather than trusted;
* ``revenue_at_risk`` is kept free of the assumption. It depends only on the model's probability and
  the observed-revenue projection, so a business that rejects the propensity figures can still use
  the exposure number.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.retention.params import PRIORITY_BANDS, RetentionParams
from src.utils.logging_config import get_logger

__all__ = ["assign_priority", "build_retention_propensity", "build_scores"]

logger = get_logger(__name__)


def _numeric(frame: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(default, index=frame.index, dtype="float64")
    return pd.to_numeric(frame[column], errors="coerce")


def _boolean(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(False, index=frame.index, dtype="bool")
    return frame[column].fillna(False).astype("bool")


def build_retention_propensity(
    scored: pd.DataFrame, params: RetentionParams
) -> pd.DataFrame:
    """Estimate the assumed probability that an intervention changes behaviour.

    Returns the propensity plus a human-readable ``propensity_basis`` naming every multiplier that
    fired, so the figure can be audited instead of taken on faith.
    """
    out = pd.DataFrame(index=scored.index)
    propensity = pd.Series(params.base_retention_propensity, index=scored.index, dtype="float64")

    recency = _numeric(scored, "recency_days", np.nan)
    return_rate = _numeric(scored, "return_rate")
    is_discount_driven = _boolean(scored, "is_discount_driven")

    long_absence = recency.ge(365).fillna(False)
    still_active = recency.lt(90).fillna(False)
    heavy_returner = return_rate.ge(params.high_return_rate)

    basis: list[list[str]] = [
        [f"base {params.base_retention_propensity:.0%} (assumption)"] for _ in range(len(scored))
    ]
    positions = {customer: index for index, customer in enumerate(scored.index)}

    def apply(mask: pd.Series, multiplier: float, label: str) -> None:
        nonlocal propensity
        propensity = propensity.where(~mask, propensity * multiplier)
        for customer in scored.index[mask]:
            basis[positions[customer]].append(f"{label} x{multiplier:g}")

    apply(is_discount_driven, params.discount_responsive_multiplier, "discount-responsive")
    apply(long_absence, params.long_absence_multiplier, "absent over a year")
    apply(still_active, params.still_active_multiplier, "still active")
    apply(heavy_returner, params.high_return_multiplier, "heavy returner")

    out["retention_propensity"] = propensity.clip(
        lower=params.min_retention_propensity, upper=params.max_retention_propensity
    ).round(4)
    out["propensity_basis"] = ["; ".join(reasons) for reasons in basis]
    out["propensity_is_assumption"] = True
    return out


def assign_priority(opportunity: pd.Series, params: RetentionParams) -> pd.Series:
    """Band the retention opportunity score into Critical / High / Medium / Low.

    Percentile-based rather than absolute-threshold-based, because the useful question for a team
    with finite capacity is "who are my top few hundred", not "who crosses an arbitrary euro line"
    that shifts every time the book grows.

    Customers with no opportunity at all (nothing at stake) are banded Low regardless of percentile,
    so a percentile rank cannot promote a zero.
    """
    ranked = opportunity.rank(pct=True, method="average")
    bands = pd.Series(PRIORITY_BANDS[-1], index=opportunity.index, dtype="object")
    bands[ranked.ge(params.medium_priority_percentile)] = "Medium"
    bands[ranked.ge(params.high_priority_percentile)] = "High"
    bands[ranked.ge(params.critical_priority_percentile)] = "Critical"
    bands[opportunity.le(0) | opportunity.isna()] = PRIORITY_BANDS[-1]
    return pd.Series(
        pd.Categorical(bands, categories=list(PRIORITY_BANDS), ordered=True),
        index=opportunity.index,
    )


def build_scores(
    scored: pd.DataFrame,
    expected_revenue: pd.DataFrame,
    params: RetentionParams | None = None,
) -> pd.DataFrame:
    """Compute revenue at risk, propensity, the opportunity score and its priority band.

    Campaign cost and ROI are *not* computed here: the incentive cost depends on the recommended
    action, so they belong to :mod:`src.retention.recommendations`, which decides the action. Keeping
    them apart avoids a circular dependency and stops a cost estimate being quoted before the thing
    it prices has been chosen.
    """
    params = params or RetentionParams()
    params.validate()

    out = pd.DataFrame(index=scored.index)
    churn = _numeric(scored, "churn_probability").clip(lower=0.0, upper=1.0)
    future_revenue = expected_revenue["expected_future_revenue"].reindex(scored.index).fillna(0.0)

    out["churn_probability"] = churn.round(6)
    out["expected_future_revenue"] = future_revenue.round(2)

    # --- revenue at risk: deliberately assumption-free ---
    out["revenue_at_risk"] = (churn * future_revenue).round(2)

    # --- propensity and everything downstream of it ---
    propensity_frame = build_retention_propensity(scored, params)
    out = out.join(propensity_frame)

    out["expected_retained_revenue"] = (
        out["revenue_at_risk"] * out["retention_propensity"]
    ).round(2)
    # The brief's formula, which is arithmetically the same as revenue at risk x propensity. Kept as
    # its own column because it is the number the priority list is sorted on.
    out["retention_opportunity_score"] = out["expected_retained_revenue"]

    out["priority"] = assign_priority(out["retention_opportunity_score"], params)
    out["retention_opportunity_percentile"] = (
        out["retention_opportunity_score"].rank(pct=True).round(4)
    )

    logger.info(
        "Revenue at risk: %s %s across %d customers; expected retained revenue %s %s "
        "under the assumed propensity (mean %.1f%%)",
        params.currency,
        f"{out['revenue_at_risk'].sum():,.2f}",
        len(out),
        params.currency,
        f"{out['expected_retained_revenue'].sum():,.2f}",
        100.0 * float(out["retention_propensity"].mean()),
    )
    return out
