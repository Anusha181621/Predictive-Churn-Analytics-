"""The churn label.

The single most important design decision in this section, so the reasoning is set out in full.

Forward-looking, not retrospective
----------------------------------
The brief defines churn as "no purchase within 180 days after the customer's last purchase". Taken
literally that is a *retrospective* rule applied to whoever happens to look quiet today, and it is
the rule the brief elsewhere warns against, because it cannot tell a churned customer from a
seasonal one.

This module operationalises it differently: at a prediction date ``as_of``, a customer is churned
if they made **no purchase in the window ``(as_of, as_of + horizon]``**. The two agree on what
churn means -- a horizon of silence following the last purchase -- but the forward-looking form is
what makes supervised learning possible and honest:

* Features come from ``<= as_of``; the label comes from ``> as_of``. The two never touch, so the
  split is leakage-free by construction.
* **It cannot mislabel a seasonal customer**, because it does not infer churn from inactivity at
  all. It observes what the customer actually did next. A retrospective recency rule labels a
  quiet January-only buyer as churned every June; this label asks whether they came back, and if
  they did, they are not churned. The mislabelling problem the brief warns about is a property of
  retrospective rules, and choosing a forward-looking label removes it rather than patching it.

The residual seasonal risk, and the adaptive horizon
----------------------------------------------------
One real problem survives. A loyal annual buyer whose next purchase lands at ``as_of + 200`` days
is labelled churned by a 180-day horizon even though nothing is wrong. The fixed horizon is
uniform, comparable and matches the stated default, so it stays the default -- but
:data:`LabelMode.ADAPTIVE` gives every customer a horizon scaled to *their own* cadence
(``2 x expected interval``, floored and capped), so a frequent buyer is judged over 90 days and an
annual buyer over a year. :func:`compare_label_modes` quantifies the disagreement, and the build
report shows how many seasonal customers it affects, so the residual risk is measured rather than
assumed away.

Observability, or why some rows have no label at all
---------------------------------------------------
A label needs its whole outcome window to have happened. With data ending 2025-12-31 and a 180-day
horizon, the latest date that can be labelled is 2025-07-04; anything later is **right-censored**
and gets ``NA``, never ``0``. Silently treating an unfinished window as "did not churn" would
train the model that recent customers never leave, which is the most damaging error available here
and the easiest one to make by accident. Censored rows are dropped from training and counted in
the report.

Eligibility
-----------
A customer with no purchase on or before ``as_of`` is not yet a customer and cannot churn, so they
are ineligible rather than labelled 0. Customers whose first purchase is very recent are labelled
but flagged, because judging someone who joined last week is unreliable evidence either way.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from enum import StrEnum

import numpy as np
import pandas as pd

from src.data.csv_loader import Datasets
from src.utils.logging_config import get_logger

__all__ = [
    "LabelMode",
    "LabelParams",
    "ChurnLabels",
    "build_churn_labels",
    "latest_labelable_as_of",
    "compare_label_modes",
]

logger = get_logger(__name__)


class LabelMode(StrEnum):
    """How the outcome window length is chosen."""

    #: One horizon for everybody -- the brief's stated default.
    FIXED = "fixed"
    #: A horizon scaled to each customer's own expected purchase interval.
    ADAPTIVE = "adaptive"


@dataclass(frozen=True)
class LabelParams:
    """Configuration for the churn label."""

    #: Outcome window length in days for :attr:`LabelMode.FIXED`. The brief's default is 180 and
    #: it is configurable through ``CHURN_INACTIVITY_DAYS``.
    horizon_days: int = 180

    mode: LabelMode = LabelMode.FIXED

    #: Adaptive mode: horizon = multiple x the customer's expected purchase interval...
    adaptive_multiple: float = 2.0
    #: ...clamped to this range. The floor stops a weekly buyer being judged over a fortnight;
    #: the cap keeps outcome windows observable, since a 400-day window costs 400 days of usable
    #: history at the end of the dataset.
    adaptive_min_days: int = 90
    adaptive_max_days: int = 365

    #: Customers whose first purchase is within this many days of the as-of date are labelled but
    #: flagged ``is_new_at_as_of``: a week of history is thin evidence in either direction.
    new_customer_days: int = 90

    def validate(self) -> None:
        if self.horizon_days <= 0:
            raise ValueError(f"horizon_days must be positive, got {self.horizon_days}")
        if self.adaptive_multiple <= 0:
            raise ValueError("adaptive_multiple must be positive")
        if not 0 < self.adaptive_min_days <= self.adaptive_max_days:
            raise ValueError(
                "adaptive horizon bounds must satisfy 0 < min <= max, got "
                f"min={self.adaptive_min_days}, max={self.adaptive_max_days}"
            )


@dataclass(frozen=True)
class ChurnLabels:
    """Labels for one as-of date, one row per customer, plus the diagnostics."""

    as_of: pd.Timestamp
    labels: pd.DataFrame
    params: LabelParams
    data_end: pd.Timestamp

    @property
    def trainable(self) -> pd.DataFrame:
        """Rows usable for supervised learning: eligible and fully observed."""
        return self.labels[self.labels["label_usable"]]

    def summary(self) -> dict[str, object]:
        labels = self.labels
        usable = self.trainable
        positives = int(usable["churned"].sum()) if len(usable) else 0
        return {
            "as_of_date": self.as_of.date().isoformat(),
            "horizon_days": int(self.params.horizon_days)
            if self.params.mode is LabelMode.FIXED
            else None,
            "mode": str(self.params.mode),
            "customers": len(labels),
            "eligible": int(labels["label_eligible"].sum()),
            "observed": int(labels["label_observable"].sum()),
            "usable": len(usable),
            "churned": positives,
            "churn_rate": round(positives / len(usable), 6) if len(usable) else None,
            "censored": int((labels["label_eligible"] & ~labels["label_observable"]).sum()),
            "new_customers": int(labels["is_new_at_as_of"].sum()),
        }


def latest_labelable_as_of(
    data: Datasets, params: LabelParams | None = None
) -> pd.Timestamp:
    """The last as-of date whose fixed-horizon outcome window fits inside the data.

    Anything after this is right-censored. Useful for choosing an evaluation grid without
    accidentally including dates whose outcome has not happened yet.
    """
    params = params or LabelParams()
    data_end = pd.Timestamp(data.transactions["purchase_date"].max()).normalize()
    return data_end - pd.Timedelta(days=params.horizon_days)


def _expected_interval(orders: pd.DataFrame, index: pd.Index, params: LabelParams) -> pd.Series:
    """Each customer's median inter-purchase gap, for the adaptive horizon.

    Median rather than mean, for the same reason as in the feature layer: one holiday binge or one
    long dormant stretch would drag the mean away from the customer's real rhythm.
    """
    if orders.empty:
        return pd.Series(np.nan, index=index, dtype="float64")
    ordered = orders.sort_values(["customer_id", "purchase_date"])
    gaps = ordered.groupby("customer_id", observed=True)["purchase_date"].diff().dt.days
    median = gaps.groupby(ordered["customer_id"]).median()
    return median.reindex(index)


def build_churn_labels(
    data: Datasets,
    as_of_date: str | date | datetime | pd.Timestamp,
    params: LabelParams | None = None,
) -> ChurnLabels:
    """Build the churn label for every customer at ``as_of_date``.

    Returns one row per Customer ID with these columns:

    ``horizon_days``
        The outcome window length applied to this customer.
    ``outcome_window_end``
        ``as_of + horizon_days``.
    ``purchases_in_window``
        Orders strictly after ``as_of`` and up to the window end.
    ``churned``
        ``1`` if no purchase in the window, ``0`` if there was one, ``NA`` when the row is
        ineligible or the window is not fully observed. Never a silent ``0``.
    ``label_eligible``
        The customer had bought at least once by ``as_of``.
    ``label_observable``
        The whole outcome window falls inside the data.
    ``label_usable``
        Both of the above -- the flag the panel builder filters on.
    ``is_new_at_as_of``
        First purchase within ``new_customer_days`` of ``as_of``; thin evidence, kept but flagged.
    ``exclusion_reason``
        Why a row is unusable, for the report.
    """
    params = params or LabelParams()
    params.validate()
    as_of = pd.Timestamp(as_of_date).normalize()

    transactions = data.transactions
    data_end = pd.Timestamp(transactions["purchase_date"].max()).normalize()
    index = pd.Index(data.customers["customer_id"], name="customer_id")

    # --- history strictly on or before the as-of date ---
    past = transactions[transactions["purchase_date"].le(as_of)]
    past_orders = past.drop_duplicates("order_id")[["customer_id", "purchase_date"]]

    first_purchase = past_orders.groupby("customer_id", observed=True)["purchase_date"].min()
    last_purchase = past_orders.groupby("customer_id", observed=True)["purchase_date"].max()

    frame = pd.DataFrame(index=index)
    frame["last_purchase_before_as_of"] = last_purchase.reindex(index)
    frame["label_eligible"] = frame["last_purchase_before_as_of"].notna()

    tenure = (as_of - first_purchase.reindex(index)).dt.days
    frame["is_new_at_as_of"] = tenure.lt(params.new_customer_days).fillna(False)

    # --- the horizon ---
    if params.mode is LabelMode.FIXED:
        horizon = pd.Series(float(params.horizon_days), index=index)
    else:
        interval = _expected_interval(past_orders, index, params)
        # A customer with one order has no measurable cadence, so they fall back to the fixed
        # horizon rather than to an invented interval.
        scaled = interval * params.adaptive_multiple
        horizon = scaled.clip(
            lower=params.adaptive_min_days, upper=params.adaptive_max_days
        ).fillna(float(params.horizon_days))
    frame["horizon_days"] = horizon
    frame["outcome_window_end"] = as_of + pd.to_timedelta(horizon, unit="D")

    # --- observability: the window must have finished ---
    #
    # NA, never 0. Treating an unfinished window as "did not churn" would teach the model that
    # recent customers never leave.
    frame["label_observable"] = frame["outcome_window_end"].le(data_end)

    # --- the outcome ---
    future = transactions[transactions["purchase_date"].gt(as_of)]
    future_orders = future.drop_duplicates("order_id")[["customer_id", "purchase_date"]]
    if future_orders.empty:
        purchases = pd.Series(0, index=index, dtype="int64")
    else:
        joined = future_orders.merge(
            frame["outcome_window_end"].rename("window_end").reset_index(),
            on="customer_id",
            how="inner",
        )
        in_window = joined[joined["purchase_date"].le(joined["window_end"])]
        purchases = (
            in_window.groupby("customer_id", observed=True)
            .size()
            .reindex(index, fill_value=0)
            .astype("int64")
        )
    frame["purchases_in_window"] = purchases

    frame["label_usable"] = frame["label_eligible"] & frame["label_observable"]
    churned = (~frame["purchases_in_window"].gt(0)).astype("Int8")
    frame["churned"] = churned.where(frame["label_usable"], pd.NA)

    # Days until the next purchase, whenever it comes. Not a feature -- it is future information
    # -- but invaluable for auditing the label, and it is what makes the seasonal diagnostic below
    # possible.
    if future_orders.empty:
        frame["days_to_next_purchase"] = np.nan
    else:
        next_purchase = future_orders.groupby("customer_id", observed=True)["purchase_date"].min()
        frame["days_to_next_purchase"] = (
            (next_purchase.reindex(index) - as_of).dt.days.astype("float64")
        )

    reason = pd.Series("", index=index, dtype="object")
    reason[~frame["label_eligible"]] = "no purchase history at the as-of date"
    reason[frame["label_eligible"] & ~frame["label_observable"]] = (
        f"outcome window extends past the end of the data ({data_end.date()})"
    )
    frame["exclusion_reason"] = reason

    result = ChurnLabels(as_of=as_of, labels=frame, params=params, data_end=data_end)
    summary = result.summary()
    logger.info(
        "Labels at %s (%s, horizon %s): %d usable of %d customers, churn rate %s "
        "(%d ineligible, %d censored)",
        as_of.date(),
        params.mode,
        summary["horizon_days"] or "per-customer",
        summary["usable"],
        summary["customers"],
        f"{summary['churn_rate']:.1%}" if summary["churn_rate"] is not None else "n/a",
        summary["customers"] - summary["eligible"],
        summary["censored"],
    )
    return result


def compare_label_modes(
    data: Datasets,
    as_of_date: str | date | datetime | pd.Timestamp,
    params: LabelParams | None = None,
) -> dict[str, object]:
    """Quantify how the fixed and adaptive horizons disagree.

    The interesting number is ``seasonal_rescued``: customers the fixed 180-day horizon calls
    churned but who do buy again within their own, longer, personal horizon. Those are precisely
    the loyal-but-slow customers a uniform window mislabels, so measuring them keeps the residual
    risk visible instead of asserting it away.
    """
    base = params or LabelParams()
    fixed = build_churn_labels(data, as_of_date, LabelParams(**{**base.__dict__, "mode": LabelMode.FIXED}))
    adaptive = build_churn_labels(
        data, as_of_date, LabelParams(**{**base.__dict__, "mode": LabelMode.ADAPTIVE})
    )

    both = fixed.labels["label_usable"] & adaptive.labels["label_usable"]
    fixed_label = fixed.labels.loc[both, "churned"].astype("float64")
    adaptive_label = adaptive.labels.loc[both, "churned"].astype("float64")

    rescued = (fixed_label.eq(1) & adaptive_label.eq(0))
    caught = (fixed_label.eq(0) & adaptive_label.eq(1))

    return {
        "as_of_date": pd.Timestamp(as_of_date).date().isoformat(),
        "comparable_customers": int(both.sum()),
        "fixed_churn_rate": round(float(fixed_label.mean()), 6) if both.any() else None,
        "adaptive_churn_rate": round(float(adaptive_label.mean()), 6) if both.any() else None,
        "agreement": round(float(fixed_label.eq(adaptive_label).mean()), 6) if both.any() else None,
        "rescued_by_adaptive": int(rescued.sum()),
        "caught_by_adaptive": int(caught.sum()),
        "adaptive_censored": int(
            (adaptive.labels["label_eligible"] & ~adaptive.labels["label_observable"]).sum()
        ),
    }
