"""Time-based splitting.

A random split would be badly wrong here for two independent reasons, and it is worth naming both
because they get conflated:

1. **Temporal leakage.** Rows are snapshots in time. A random split trains on 2025 and tests on
   2024, so the model is evaluated on a past it has already seen.
2. **Panel correlation.** The same customer appears at many as-of dates, and adjacent snapshots of
   one customer are nearly identical. A random split puts a customer's March snapshot in train and
   their April snapshot in test, measuring memorisation rather than generalisation.

So everything is split on the as-of date.

The embargo, and where it actually needs to apply
-------------------------------------------------
A row at ``as_of = T`` carries a label drawn from ``(T, T + horizon]``, so its *label* describes a
period that a later row uses for its *features*. Information flows forward across a boundary even
when the feature dates do not overlap, and the fix is an embargo -- a ``horizon``-wide gap. The
question is which boundaries need one, and the answer is not "all of them":

* **Before the test period: mandatory.** Every row used to fit anything -- the model or its
  probability calibrator -- must have its outcome window closed before the test as-of date.
  This is what makes the reported number trustworthy, so it is enforced unconditionally.
* **Before the selection validation: mandatory, but *inside* the training data.** An embargo here
  buys nothing for the test estimate; it exists so that model *selection* is unbiased.

Applying the embargo at both outer boundaries -- the obvious first design -- turns out to be a
mistake on three years of data with a 180-day horizon. It consumes a full year of the 24 usable
months and confines training to the brand's early growth phase, where the churn base rate is 18-33%
against 47% in the test period. The resulting model was measurably *worse than a single feature*:
test ROC-AUC 0.68 versus 0.73 for ``orders_365d`` alone. The cure was worse than the disease.

So the design is three-stage:

    selection : inner_train -> [embargo] -> inner_validation      (pick the model family)
    refit     : all fit dates                                     (use the recent data too)
    calibrate : calibration date                                  (set the probability level)
                                    -> [embargo] -> test          (report once)

Both ``fit`` and ``calibration`` dates sit before the test embargo, so the test estimate stays
clean while training still reaches data close to the prediction regime. Skipping the inner embargo
instead -- training right up to the validation date -- inflates validation scores badly enough to
break selection: LightGBM scored 0.88 validation PR-AUC against 0.66 on test, and would have been
chosen over the linear model that actually generalised better.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import pandas as pd

from src.models.dataset import AS_OF_COLUMN, TARGET_COLUMN, ModellingPanel
from src.utils.logging_config import get_logger

__all__ = ["SplitPlan", "TimeSplit", "plan_model_dates", "make_time_split"]

logger = get_logger(__name__)


@dataclass(frozen=True)
class SplitPlan:
    """As-of dates for every stage, all embargoes already applied."""

    #: Inner split used to choose the model family, embargoed from each other.
    selection_train: list[pd.Timestamp]
    selection_validation: list[pd.Timestamp]
    #: Everything the final model is fitted on (a superset of the selection dates).
    fit: list[pd.Timestamp]
    #: Held out from the fit, used to calibrate probabilities.
    calibration: list[pd.Timestamp]
    #: Touched once, for the reported metrics.
    test: list[pd.Timestamp]
    #: Dates discarded to embargo gaps, with the boundary that consumed them.
    embargoed: dict[str, list[pd.Timestamp]] = field(default_factory=dict)
    horizon_days: int = 180

    def all_dates(self) -> list[pd.Timestamp]:
        """Every date that needs a feature snapshot built."""
        return sorted(set(self.fit) | set(self.calibration) | set(self.test))

    def validate(self) -> None:
        """Assert every embargo the design promises. Raises ``ValueError`` on violation."""
        gap = pd.Timedelta(days=self.horizon_days)

        # The one that protects the reported number: nothing fitted may resolve after test starts.
        fitted = self.fit + self.calibration
        if fitted and self.test:
            latest_outcome = max(fitted) + gap
            if latest_outcome > min(self.test):
                raise ValueError(
                    f"test embargo violated: fitting data at {max(fitted).date()} resolves on "
                    f"{latest_outcome.date()}, after the test period starts {min(self.test).date()}"
                )

        # The one that keeps selection unbiased.
        if self.selection_train and self.selection_validation:
            if max(self.selection_train) + gap > min(self.selection_validation):
                raise ValueError(
                    f"selection embargo violated: inner training at "
                    f"{max(self.selection_train).date()} resolves after the inner validation "
                    f"starts {min(self.selection_validation).date()}"
                )

        if self.calibration and set(self.calibration) & set(self.fit):
            raise ValueError("calibration dates must be held out of the fit")
        if self.test and self.fit and max(self.fit) >= min(self.test):
            raise ValueError("fit dates must precede the test period")

    def summary(self) -> dict[str, object]:
        def dates(values: Sequence[pd.Timestamp]) -> list[str]:
            return [d.date().isoformat() for d in values]

        return {
            "horizon_days": self.horizon_days,
            "selection_train": dates(self.selection_train),
            "selection_validation": dates(self.selection_validation),
            "fit": dates(self.fit),
            "calibration": dates(self.calibration),
            "test": dates(self.test),
            "embargoed": {k: dates(v) for k, v in self.embargoed.items()},
        }


def plan_model_dates(
    as_of_dates: Sequence[pd.Timestamp],
    horizon_days: int,
    *,
    test_periods: int = 1,
    calibration_periods: int = 1,
    selection_validation_periods: int = 3,
) -> SplitPlan:
    """Assign as-of dates to the selection, fit, calibration and test stages.

    Works backwards from the most recent dates, so the test period sits as close to production
    conditions as the data allows.
    """
    ordered = sorted(set(as_of_dates))
    gap = pd.Timedelta(days=horizon_days)
    minimum = test_periods + calibration_periods + selection_validation_periods + 1
    if len(ordered) < minimum:
        raise ValueError(
            f"need at least {minimum} as-of dates for this split, got {len(ordered)}"
        )

    embargoed: dict[str, list[pd.Timestamp]] = {}

    # --- test: the most recent dates ---
    test = ordered[-test_periods:]
    remaining = ordered[:-test_periods]

    # --- everything fitted must resolve before the test period opens ---
    fit_ceiling = min(test) - gap
    usable = [d for d in remaining if d <= fit_ceiling]
    if len(usable) < calibration_periods + selection_validation_periods + 1:
        raise ValueError(
            f"only {len(usable)} as-of date(s) clear the {horizon_days}-day embargo before the "
            f"test period starting {min(test).date()}; widen the grid or shorten the horizon"
        )
    embargoed["before_test"] = [d for d in remaining if d > fit_ceiling]

    # --- calibration: the latest usable dates, held out of the fit ---
    calibration = usable[-calibration_periods:]
    fit = usable[:-calibration_periods]
    if not fit:
        raise ValueError("no as-of dates left to fit on after reserving calibration dates")

    # --- the inner selection split, embargoed inside the fit data ---
    selection_validation = fit[-selection_validation_periods:]
    selection_ceiling = min(selection_validation) - gap
    selection_train = [d for d in fit if d <= selection_ceiling]
    if not selection_train:
        raise ValueError(
            f"no as-of date clears the {horizon_days}-day embargo before the selection validation "
            f"starting {min(selection_validation).date()}; widen the grid or shorten the horizon"
        )
    embargoed["before_selection_validation"] = [
        d for d in fit if d > selection_ceiling and d not in selection_validation
    ]

    plan = SplitPlan(
        selection_train=selection_train,
        selection_validation=selection_validation,
        fit=fit,
        calibration=calibration,
        test=test,
        embargoed=embargoed,
        horizon_days=horizon_days,
    )
    plan.validate()
    return plan


@dataclass(frozen=True)
class TimeSplit:
    """The panel sliced according to a :class:`SplitPlan`."""

    plan: SplitPlan
    selection_train: pd.DataFrame
    selection_validation: pd.DataFrame
    fit: pd.DataFrame
    calibration: pd.DataFrame
    test: pd.DataFrame

    @property
    def horizon_days(self) -> int:
        return self.plan.horizon_days

    def summary(self) -> dict[str, object]:
        def describe(frame: pd.DataFrame) -> dict[str, object]:
            return {
                "rows": len(frame),
                "customers": int(frame["customer_id"].nunique()) if len(frame) else 0,
                "churn_rate": round(float(frame[TARGET_COLUMN].mean()), 6) if len(frame) else None,
            }

        return {
            "plan": self.plan.summary(),
            "selection_train": describe(self.selection_train),
            "selection_validation": describe(self.selection_validation),
            "fit": describe(self.fit),
            "calibration": describe(self.calibration),
            "test": describe(self.test),
        }


def make_time_split(panel: ModellingPanel, plan: SplitPlan) -> TimeSplit:
    """Slice ``panel`` according to ``plan``."""
    frame = panel.frame

    def slice_for(dates: Sequence[pd.Timestamp]) -> pd.DataFrame:
        return frame[frame[AS_OF_COLUMN].isin(list(dates))].copy()

    split = TimeSplit(
        plan=plan,
        selection_train=slice_for(plan.selection_train),
        selection_validation=slice_for(plan.selection_validation),
        fit=slice_for(plan.fit),
        calibration=slice_for(plan.calibration),
        test=slice_for(plan.test),
    )
    logger.info(
        "Split (%d-day embargo): selection %d/%d rows | fit %d rows to %s | calibration %d rows "
        "at %s | test %d rows at %s",
        plan.horizon_days,
        len(split.selection_train),
        len(split.selection_validation),
        len(split.fit),
        max(plan.fit).date(),
        len(split.calibration),
        ", ".join(d.date().isoformat() for d in plan.calibration),
        len(split.test),
        ", ".join(d.date().isoformat() for d in plan.test),
    )
    return split
