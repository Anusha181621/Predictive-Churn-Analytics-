"""The twelve business segments.

Two outputs, and the reason for both is the brief's own instruction to *allow customers to belong to
multiple analytical dimensions rather than forcing every customer into one rigid segment*:

* **Twelve boolean flags**, which overlap freely. A customer can be a High-Return Customer *and*
  High-Value At Risk, and those two facts drive different parts of a retention decision -- one says
  act now, the other says be careful what you offer. Collapsing them loses the second.
* **One ``primary_segment``**, resolved by priority, because a dashboard tile and a pivot table need
  a single value per customer.

Priority answers a specific question: *what is the single most decision-relevant thing about this
customer?* Not "which rule is most specific", which is why the order is not simply narrowest-first:

1. **Lost** -- beyond recovery, so nothing else about them changes what to do (nothing).
2. **High-Value At Risk** -- the largest sums in play, and still recoverable.
3. **Frequent but Declining** -- an early warning on a good customer, catchable before it becomes 2.
4. **Discount-Driven At Risk** -- at risk, and the *kind* of at-risk that dictates the offer.
5. **High-Return Customers** -- winning the order back may not win the margin, so this qualifies
   any action and must not be hidden behind a value label.
6. **Dormant** -- inactive but not yet unrecoverable.
7. **Champions** / 8. **Loyal Customers** -- positive labels, ranked below the at-risk ones because
   they call for maintenance rather than intervention. By construction a Champion cannot also be
   High-Value At Risk (one requires low churn risk, the other high), so this ordering never demotes
   a genuinely at-risk customer to a comfortable label.
9. **Seasonal** / 10. **New** / 11. **One-Time Buyers** -- context labels: true, and useful, but they
    describe the relationship rather than a required action.
12. **Low-Value At Risk** -- at risk, but the economics rarely justify contact.

Every threshold comes from :class:`~src.retention.params.RetentionParams`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.retention.params import RetentionParams
from src.utils.logging_config import get_logger

__all__ = ["SEGMENTS", "SEGMENT_FLAGS", "build_segments"]

logger = get_logger(__name__)

#: In primary-resolution priority order. See the module docstring for why this order.
SEGMENTS: tuple[str, ...] = (
    "Lost Customers",
    "High-Value At Risk",
    "Frequent but Declining",
    "Discount-Driven At Risk",
    "High-Return Customers",
    "Dormant Customers",
    "Champions",
    "Loyal Customers",
    "Seasonal Customers",
    "New Customers",
    "One-Time Buyers",
    "Low-Value At Risk",
)

#: Column name carrying each segment's boolean flag.
SEGMENT_FLAGS: dict[str, str] = {
    segment: "is_" + segment.lower().replace(" ", "_").replace("-", "_") for segment in SEGMENTS
}

FALLBACK_SEGMENT = "Steady Customers"


def _numeric(frame: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(default, index=frame.index, dtype="float64")
    return pd.to_numeric(frame[column], errors="coerce")


def _boolean(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(False, index=frame.index, dtype="bool")
    return frame[column].fillna(False).astype("bool")


def build_segments(
    scored: pd.DataFrame, params: RetentionParams | None = None
) -> pd.DataFrame:
    """Assign the twelve segments.

    ``scored`` must carry the customer features joined to the churn prediction -- specifically
    ``churn_probability``, ``value_percentile`` and the behavioural features from
    :mod:`src.features`.
    """
    params = params or RetentionParams()
    params.validate()

    frame = scored
    out = pd.DataFrame(index=frame.index)

    churn = _numeric(frame, "churn_probability")
    value_percentile = _numeric(frame, "value_percentile")
    recency = _numeric(frame, "recency_days", np.nan)
    orders_365d = _numeric(frame, "orders_365d")
    total_orders = _numeric(frame, "total_orders")
    return_rate = _numeric(frame, "return_rate")
    revenue_growth = _numeric(frame, "revenue_growth", np.nan)
    recent_vs_historical = _numeric(frame, "recent_vs_historical_revenue", np.nan)
    tenure = _numeric(frame, "customer_tenure_days", np.nan)

    has_history = _boolean(frame, "has_purchase_history")
    is_seasonal = _boolean(frame, "is_seasonal_buyer")
    is_dormant = _boolean(frame, "is_dormant_buyer")
    is_new = _boolean(frame, "is_new_buyer")
    is_one_time = _boolean(frame, "is_one_time_buyer")
    is_discount_driven = _boolean(frame, "is_discount_driven")
    # Ranked within the cohort, for the reason given on `discount_driven_percentile`.
    discount_dependency = _numeric(frame, "discount_dependency_score")
    heavily_discount_dependent = discount_dependency.rank(pct=True).ge(
        params.discount_driven_percentile
    ).fillna(False)

    # Risk bands come from the model's own configured thresholds, carried on the prediction.
    high_risk = churn.ge(params.risk_high_threshold)
    medium_risk = churn.ge(params.risk_medium_threshold)
    low_risk = churn.lt(params.risk_medium_threshold)

    high_value = value_percentile.ge(params.high_value_percentile)
    low_value = value_percentile.lt(params.low_value_percentile)

    # --- the twelve flags, overlapping by design ---

    # Two full annual cycles of silence. Distinguished from Dormant because the action differs
    # completely: dormant customers are worth a win-back, lost ones are worth a suppression list.
    lost = has_history & recency.ge(params.lost_recency_days)

    dormant = has_history & (is_dormant | recency.ge(365)) & ~lost

    high_value_at_risk = has_history & high_value & high_risk & ~lost

    # "Was frequent, is fading" -- the group worth catching earliest. Requires evidence of a real
    # habit (so a two-order customer does not qualify) and evidence it is decaying.
    was_frequent = (orders_365d.ge(params.frequent_orders_365d)) | (
        total_orders.ge(params.frequent_orders_365d)
    )
    declining = revenue_growth.le(params.declining_revenue_growth) | recent_vs_historical.lt(0.7)
    frequent_but_declining = has_history & was_frequent & declining.fillna(False) & ~lost

    discount_driven_at_risk = (
        has_history & is_discount_driven & heavily_discount_dependent & medium_risk & ~lost
    )

    high_return = has_history & return_rate.ge(params.high_return_rate)

    # Champions: the top of the book, still engaged. Low risk is part of the definition, which is
    # what makes Champions and High-Value At Risk mutually exclusive.
    champions = (
        has_history
        & high_value
        & orders_365d.ge(params.frequent_orders_365d)
        & low_risk
        & ~dormant
        & ~lost
    )

    loyal = (
        has_history
        & total_orders.ge(params.loyal_min_orders)
        & ~high_risk
        & ~dormant
        & ~lost
        & ~champions
    )

    seasonal = has_history & is_seasonal & ~lost
    new_customers = has_history & (is_new | tenure.lt(90).fillna(False))
    one_time = has_history & is_one_time & ~new_customers
    low_value_at_risk = has_history & low_value & high_risk & ~lost

    flags = {
        "Lost Customers": lost,
        "High-Value At Risk": high_value_at_risk,
        "Frequent but Declining": frequent_but_declining,
        "Discount-Driven At Risk": discount_driven_at_risk,
        "High-Return Customers": high_return,
        "Dormant Customers": dormant,
        "Champions": champions,
        "Loyal Customers": loyal,
        "Seasonal Customers": seasonal,
        "New Customers": new_customers,
        "One-Time Buyers": one_time,
        "Low-Value At Risk": low_value_at_risk,
    }
    for segment, mask in flags.items():
        out[SEGMENT_FLAGS[segment]] = mask.fillna(False).astype("bool")

    # --- the resolved primary label ---
    #
    # Assigned in reverse priority so earlier, higher-priority rules overwrite later ones.
    primary = pd.Series(FALLBACK_SEGMENT, index=frame.index, dtype="object")
    primary[~has_history] = "No History"
    for segment in reversed(SEGMENTS):
        primary[out[SEGMENT_FLAGS[segment]]] = segment
    out["primary_segment"] = primary

    # --- the multi-label view, for the dashboard ---
    flag_columns = [SEGMENT_FLAGS[segment] for segment in SEGMENTS]
    out["segment_count"] = out[flag_columns].sum(axis=1).astype("int64")
    out["all_segments"] = [
        "; ".join(segment for segment in SEGMENTS if row[SEGMENT_FLAGS[segment]]) or FALLBACK_SEGMENT
        for _, row in out.iterrows()
    ]

    counts = out["primary_segment"].value_counts()
    logger.info(
        "Segments: %s",
        ", ".join(f"{segment}={count}" for segment, count in counts.items()),
    )
    logger.info(
        "Multi-membership: %d customers carry more than one segment (mean %.2f each)",
        int(out["segment_count"].gt(1).sum()),
        float(out["segment_count"].mean()),
    )
    return out
