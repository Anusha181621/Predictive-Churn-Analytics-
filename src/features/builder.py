"""The customer feature store builder.

Turns the four CSV files into ``customer_features``: exactly one row per Customer ID, computed
strictly as of a prediction date.

The contract, in order of importance:

1. **One row per Customer ID, always.** Every customer in ``Customer.csv`` appears, including
   those with no orders on or before the as-of date. Zero-history customers are not silently
   dropped; they are flagged by ``has_purchase_history`` and
   ``registered_at_as_of`` so the model step can decide what to do with them. Dropping them here
   would hide a real cohort and quietly change the row count between as-of dates.
2. **No leakage.** Every module reads from :class:`~src.features.context.FeatureContext`, which
   clipped transactions *and* returns to the as-of date once. See that module for why the return
   clipping is the subtle half.
3. **Reusable.** ``build_customer_features(data, as_of_date=...)`` is callable repeatedly with
   different dates, which is exactly what Section 3's time-based validation needs: build features
   at several historical as-of dates and stack them.

The output is an analytical artefact. The CSVs remain the source of truth; nothing here writes to
``data/``.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime

import pandas as pd

from src.data.csv_loader import Datasets, load_all
from src.features.affinity import build_affinity_features
from src.features.context import FeatureContext, build_context
from src.features.discount import build_discount_features
from src.features.gaps import build_gap_features
from src.features.lifecycle import build_lifecycle_features
from src.features.params import FeatureParams
from src.features.returns import build_return_features
from src.features.rfm import build_rfm_features
from src.features.seasonality import build_seasonality_features
from src.features.segments import build_segment_features
from src.features.trends import build_trend_features
from src.features.value import build_value_features
from src.utils.logging_config import get_logger

__all__ = ["build_customer_features", "FeatureBuildResult", "FEATURE_GROUPS"]

logger = get_logger(__name__)

#: Documentation of what each group contributes, used by the build report.
FEATURE_GROUPS = (
    "identity",
    "rfm",
    "gaps",
    "trends",
    "lifecycle",
    "affinity",
    "discount",
    "returns",
    "seasonality",
    "value",
    "segments",
)


@dataclass(frozen=True)
class FeatureBuildResult:
    """The feature table plus the metadata needed to report on and audit the build."""

    features: pd.DataFrame
    as_of_date: pd.Timestamp
    params: FeatureParams
    group_columns: dict[str, list[str]]
    issues: list[str]

    @property
    def customer_count(self) -> int:
        return len(self.features)

    @property
    def feature_names(self) -> list[str]:
        """Feature columns, excluding the ``customer_id`` key itself."""
        return [column for column in self.features.columns if column != "customer_id"]

    @property
    def feature_count(self) -> int:
        return len(self.feature_names)

    def summary(self) -> dict[str, object]:
        return {
            "as_of_date": self.as_of_date.date().isoformat(),
            "customers": self.customer_count,
            "features": self.feature_count,
            "groups": {name: len(columns) for name, columns in self.group_columns.items()},
            "issues": self.issues,
        }


def _identity_features(context: FeatureContext) -> pd.DataFrame:
    """The customer attributes carried through from ``Customer.csv``, plus cohort flags."""
    features = context.empty_frame()
    attributes = context.customers.set_index("customer_id")

    for column in ("age", "customer_gender", "city", "country", "acquisition_channel"):
        features[column] = attributes[column].reindex(features.index)

    features["as_of_date"] = context.as_of
    features["registered_at_as_of"] = context.registered
    features["has_purchase_history"] = context.has_history

    # Age bands, because the dashboard slices risk by age group and a consistent banding has to
    # come from one place.
    features["age_band"] = pd.cut(
        pd.to_numeric(features["age"], errors="coerce"),
        bins=[0, 24, 34, 44, 54, 200],
        labels=["18-24", "25-34", "35-44", "45-54", "55+"],
        right=True,
    ).astype("object")

    return features


def _detect_issues(features: pd.DataFrame, context: FeatureContext) -> list[str]:
    """Collect calculation caveats worth reporting rather than hiding."""
    issues: list[str] = []
    total = len(features)

    no_history = int((~features["has_purchase_history"]).sum())
    if no_history:
        issues.append(
            f"{no_history} of {total} customers have no orders on or before "
            f"{context.as_of.date()}; their behavioural features are null by construction and "
            "they carry has_purchase_history=False."
        )

    unregistered = int((~features["registered_at_as_of"]).sum())
    if unregistered:
        issues.append(
            f"{unregistered} customers had not registered by {context.as_of.date()} and did not "
            "exist as customers yet; exclude them when training."
        )

    single_order = int((features["total_orders"] == 1).sum())
    if single_order:
        issues.append(
            f"{single_order} customers have exactly one order, so no inter-purchase gap can be "
            "measured; average/median/maximum_purchase_gap are null and "
            "expected_purchase_interval_days falls back to "
            f"{context.params.default_expected_interval_days} days "
            "(has_measurable_cadence=False)."
        )

    no_baseline = int(features["revenue_growth"].isna().sum())
    if no_baseline:
        issues.append(
            f"{no_baseline} customers have no revenue in the previous "
            f"{context.params.trend_window_days}-day window, so *_growth features are null "
            "rather than infinite. Tree models handle these natively."
        )

    unscored_seasonality = int(features["seasonal_customer_score"].isna().sum())
    if unscored_seasonality:
        issues.append(
            f"{unscored_seasonality} customers do not meet the evidence bar for a seasonality "
            f"score (>= {context.params.min_orders_for_seasonality} orders across "
            f">= {context.params.min_years_for_seasonality} calendar years); "
            "seasonal_customer_score is null for them, which is intentional - with one or two "
            "orders any customer looks perfectly seasonal by accident."
        )

    floored = int(features.get("annualisation_floored", pd.Series(dtype=bool)).sum())
    if floored:
        issues.append(
            f"{floored} customers have tenure below "
            f"{context.params.min_tenure_days_for_annualisation} days, so annualized_revenue "
            "used a floored denominator and is an upper bound, not an estimate."
        )

    withheld = int(
        features["returned_units"].sum() if "returned_units" in features else 0
    )
    logger.debug("Observed returned units at as-of date: %s", withheld)

    duplicated = int(features.index.duplicated().sum())
    if duplicated:
        issues.append(f"BUG: {duplicated} duplicated customer_id rows in the feature table.")

    return issues


def build_customer_features(
    data: Datasets | None = None,
    as_of_date: str | date | datetime | pd.Timestamp | None = None,
    params: FeatureParams | None = None,
) -> FeatureBuildResult:
    """Build the customer feature table as of ``as_of_date``.

    Parameters
    ----------
    data:
        Loaded CSVs. Defaults to :func:`~src.data.csv_loader.load_all`.
    as_of_date:
        The prediction date. Only transactions and returns on or before it are used. Defaults to
        the maximum purchase date in the data, so a build is reproducible.
    params:
        Feature thresholds. Defaults to :class:`~src.features.params.FeatureParams`.

    Returns
    -------
    :class:`FeatureBuildResult` holding the table, the resolved as-of date and any calculation
    caveats worth reporting.
    """
    data = data if data is not None else load_all()
    context = build_context(data, as_of_date=as_of_date, params=params)

    # Order matters only where a later group reuses an earlier one's definitions -- value and
    # segments read RFM/lifecycle/trends rather than recomputing revenue or tenure, so there is
    # a single definition of each quantity in the codebase.
    identity = _identity_features(context)
    rfm = build_rfm_features(context)
    gaps = build_gap_features(context)
    trends = build_trend_features(context)
    lifecycle = build_lifecycle_features(context)
    affinity = build_affinity_features(context)
    discount = build_discount_features(context)
    returns = build_return_features(context)
    seasonality = build_seasonality_features(context)
    value = build_value_features(context, rfm=rfm, lifecycle=lifecycle)
    segments = build_segment_features(
        context, rfm=rfm, gaps=gaps, trends=trends, lifecycle=lifecycle, seasonality=seasonality
    )

    groups = {
        "identity": identity,
        "rfm": rfm,
        "gaps": gaps,
        "trends": trends,
        "lifecycle": lifecycle,
        "affinity": affinity,
        "discount": discount,
        "returns": returns,
        "seasonality": seasonality,
        "value": value,
        "segments": segments,
    }

    # A later group re-stating a column an earlier one already produced (value and RFM both
    # expose lifetime_revenue by design) must not create a duplicate column.
    frames: list[pd.DataFrame] = []
    seen: set[str] = set()
    group_columns: dict[str, list[str]] = {}
    for name, frame in groups.items():
        fresh = [column for column in frame.columns if column not in seen]
        group_columns[name] = fresh
        seen.update(fresh)
        frames.append(frame[fresh])

    features = pd.concat(frames, axis=1)

    if features.index.has_duplicates:  # pragma: no cover - guards a structural mistake
        raise RuntimeError("feature table has duplicate customer_id values")
    if len(features) != len(data.customers):  # pragma: no cover
        raise RuntimeError(
            f"feature table has {len(features)} rows for {len(data.customers)} customers; "
            "the one-row-per-customer contract is broken"
        )

    issues = _detect_issues(features, context)
    features = features.reset_index()

    logger.info(
        "Built %d features for %d customers as of %s (%d issue note(s))",
        len(features.columns) - 1,
        len(features),
        context.as_of.date(),
        len(issues),
    )
    for issue in issues:
        logger.info("  note: %s", issue)

    return FeatureBuildResult(
        features=features,
        as_of_date=context.as_of,
        params=context.params,
        group_columns=group_columns,
        issues=issues,
    )
