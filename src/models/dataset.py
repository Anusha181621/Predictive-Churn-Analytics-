"""Build the modelling panel: features and labels stacked across historical as-of dates.

One row is a **(customer, as-of date)** pair: the customer's features as they stood on that date
and whether they went on to churn. Stacking several dates is what makes time-based validation
possible, and it is only sound because both halves are already date-disciplined --
:mod:`src.features.context` clips features to ``<= as_of`` and :mod:`src.models.labels` takes the
outcome from ``> as_of``.

The two obvious mistakes this module is built to avoid:

1. **Labelling an unfinished window.** Rows whose outcome window runs past the end of the data are
   dropped, not defaulted to "did not churn".
2. **Leaking the period.** Nothing that identifies *which* snapshot a row came from may reach the
   model, or it can memorise "rows from late 2024 churn less" instead of learning behaviour. The
   as-of date is kept as metadata for splitting and reporting, and excluded from the feature matrix
   by :mod:`src.models.preprocessing`.

The same customer appears at several as-of dates, which is intended: this is panel data, and the
production question is "will *this* customer churn from *today*", asked repeatedly. Adjacent
snapshots of one customer are correlated, so a random split across rows would be far too
optimistic; :mod:`src.models.splits` splits on time instead.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, datetime

import pandas as pd

from src.data.csv_loader import Datasets, load_all
from src.features.builder import build_customer_features
from src.features.params import FeatureParams
from src.models.labels import ChurnLabels, LabelParams, build_churn_labels, latest_labelable_as_of
from src.utils.logging_config import get_logger

__all__ = ["ModellingPanel", "build_panel", "monthly_as_of_grid", "quarterly_as_of_grid"]

logger = get_logger(__name__)

#: Column carrying the as-of date. Metadata for splitting, never a feature.
AS_OF_COLUMN = "as_of_date"
#: The supervised target.
TARGET_COLUMN = "churned"


@dataclass(frozen=True)
class ModellingPanel:
    """Stacked (customer, as-of date) rows with features, target and metadata."""

    frame: pd.DataFrame
    as_of_dates: list[pd.Timestamp]
    label_params: LabelParams
    feature_params: FeatureParams
    snapshot_summaries: list[dict[str, object]]

    @property
    def target(self) -> pd.Series:
        return self.frame[TARGET_COLUMN].astype("int8")

    def summary(self) -> dict[str, object]:
        churn_rate = float(self.frame[TARGET_COLUMN].mean()) if len(self.frame) else None
        return {
            "rows": len(self.frame),
            "customers": int(self.frame["customer_id"].nunique()),
            "as_of_dates": [d.date().isoformat() for d in self.as_of_dates],
            "churn_rate": round(churn_rate, 6) if churn_rate is not None else None,
            "snapshots": self.snapshot_summaries,
        }

    def churn_rate_by(self, column: str) -> pd.DataFrame:
        """Churn rate broken down by a categorical column -- used to audit the label."""
        grouped = self.frame.groupby(column, observed=True)[TARGET_COLUMN]
        return pd.DataFrame({"rows": grouped.size(), "churn_rate": grouped.mean()}).sort_values(
            "rows", ascending=False
        )


def _month_ends(start: pd.Timestamp, end: pd.Timestamp, step_months: int) -> list[pd.Timestamp]:
    dates = pd.date_range(start=start, end=end, freq="ME")
    return [d.normalize() for d in dates[::step_months]]


def monthly_as_of_grid(
    data: Datasets,
    label_params: LabelParams | None = None,
    *,
    start: str | pd.Timestamp | None = None,
    step_months: int = 1,
) -> list[pd.Timestamp]:
    """Month-end as-of dates from ``start`` up to the last date that can still be labelled.

    The upper bound is the point of this helper: it is ``data_end - horizon``, so no date whose
    outcome window is unfinished can slip into an evaluation grid by accident.
    """
    label_params = label_params or LabelParams()
    first_purchase = pd.Timestamp(data.transactions["purchase_date"].min()).normalize()
    # Six months of runway, so the earliest snapshots have some history to describe.
    default_start = (first_purchase + pd.Timedelta(days=180)).normalize()
    begin = pd.Timestamp(start).normalize() if start is not None else default_start
    return _month_ends(begin, latest_labelable_as_of(data, label_params), step_months)


def quarterly_as_of_grid(
    data: Datasets,
    label_params: LabelParams | None = None,
    *,
    start: str | pd.Timestamp | None = None,
) -> list[pd.Timestamp]:
    """As :func:`monthly_as_of_grid`, every third month."""
    return monthly_as_of_grid(data, label_params, start=start, step_months=3)


def build_snapshot(
    data: Datasets,
    as_of_date: str | date | datetime | pd.Timestamp,
    label_params: LabelParams | None = None,
    feature_params: FeatureParams | None = None,
) -> tuple[pd.DataFrame, ChurnLabels]:
    """Features joined to labels for a single as-of date, filtered to usable rows."""
    features = build_customer_features(
        data, as_of_date=as_of_date, params=feature_params
    ).features
    labels = build_churn_labels(data, as_of_date, label_params)

    label_columns = [
        TARGET_COLUMN,
        "horizon_days",
        "outcome_window_end",
        "purchases_in_window",
        "days_to_next_purchase",
        "is_new_at_as_of",
        "label_usable",
    ]
    joined = features.merge(
        labels.labels[label_columns].reset_index(), on="customer_id", how="inner", validate="1:1"
    )
    usable = joined[joined["label_usable"]].drop(columns=["label_usable"]).copy()
    usable[TARGET_COLUMN] = usable[TARGET_COLUMN].astype("int8")
    return usable, labels


def build_panel(
    data: Datasets | None = None,
    as_of_dates: Sequence[str | date | datetime | pd.Timestamp] | None = None,
    label_params: LabelParams | None = None,
    feature_params: FeatureParams | None = None,
) -> ModellingPanel:
    """Build the stacked panel over ``as_of_dates``.

    Defaults to a quarterly grid ending at the last labelable date. Each snapshot costs a full
    feature build, so a quarterly grid is the sensible default and monthly is available when more
    rows matter more than build time.
    """
    data = data if data is not None else load_all()
    label_params = label_params or LabelParams()
    feature_params = feature_params or FeatureParams()

    if as_of_dates is None:
        as_of_dates = quarterly_as_of_grid(data, label_params)
    resolved = sorted({pd.Timestamp(d).normalize() for d in as_of_dates})
    if not resolved:
        raise ValueError("no as-of dates to build a panel from")

    horizon_limit = latest_labelable_as_of(data, label_params)
    too_late = [d for d in resolved if d > horizon_limit]
    if too_late and label_params.mode.value == "fixed":
        # Loud, because a silently unlabelable snapshot would just vanish and make the panel
        # smaller than the caller expects for no visible reason.
        logger.warning(
            "%d as-of date(s) are past the last labelable date %s and will yield no rows: %s",
            len(too_late),
            horizon_limit.date(),
            [d.date().isoformat() for d in too_late],
        )

    frames: list[pd.DataFrame] = []
    summaries: list[dict[str, object]] = []
    for as_of in resolved:
        snapshot, labels = build_snapshot(data, as_of, label_params, feature_params)
        summaries.append(labels.summary())
        if snapshot.empty:
            logger.warning("Snapshot %s produced no usable rows", as_of.date())
            continue
        frames.append(snapshot)

    if not frames:
        raise ValueError(
            "every as-of date produced zero usable rows; check the horizon against the data range"
        )

    panel = pd.concat(frames, ignore_index=True)
    panel[AS_OF_COLUMN] = pd.to_datetime(panel[AS_OF_COLUMN])

    logger.info(
        "Panel: %d rows over %d as-of dates (%s to %s), %d unique customers, churn rate %.1f%%",
        len(panel),
        len(frames),
        resolved[0].date(),
        resolved[-1].date(),
        panel["customer_id"].nunique(),
        100.0 * panel[TARGET_COLUMN].mean(),
    )
    return ModellingPanel(
        frame=panel,
        as_of_dates=[pd.Timestamp(d) for d in sorted(panel[AS_OF_COLUMN].unique())],
        label_params=label_params,
        feature_params=feature_params,
        snapshot_summaries=summaries,
    )
