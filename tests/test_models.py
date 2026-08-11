"""Tests for the split planning, preprocessing, risk bands and model persistence.

The split tests are the important ones: a mis-ordered split produces plausible-looking but
meaningless scores, which is a failure mode no amount of downstream testing would catch.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.models.preprocessing import EXCLUDED_COLUMNS, feature_matrix, split_feature_columns
from src.models.registry import (
    FeatureMismatchError,
    ModelMetadata,
    SavedModel,
    load_model,
    save_model,
)
from src.models.risk import (
    RISK_LEVELS,
    assign_risk_level,
    expected_horizon_revenue,
    revenue_at_risk,
    risk_distribution,
)
from src.models.splits import plan_model_dates
from src.models.train import IDENTITY_FEATURES, single_feature_baselines

MONTHLY = [pd.Timestamp("2023-07-31") + pd.offsets.MonthEnd(i) for i in range(24)]


# ======================================================================================
# SPLIT PLANNING
# ======================================================================================


def test_plan_is_chronological() -> None:
    plan = plan_model_dates(MONTHLY, 180)
    assert max(plan.selection_train) < min(plan.selection_validation)
    assert max(plan.fit) < min(plan.calibration)
    assert max(plan.calibration) < min(plan.test)


def test_everything_fitted_resolves_before_the_test_period() -> None:
    """The embargo that protects the reported number."""
    horizon = 180
    plan = plan_model_dates(MONTHLY, horizon)
    latest_outcome = max(plan.fit + plan.calibration) + pd.Timedelta(days=horizon)
    assert latest_outcome <= min(plan.test)


def test_selection_split_is_embargoed_from_itself() -> None:
    """The embargo that keeps model selection unbiased."""
    horizon = 180
    plan = plan_model_dates(MONTHLY, horizon)
    assert max(plan.selection_train) + pd.Timedelta(days=horizon) <= min(
        plan.selection_validation
    )


def test_calibration_dates_are_held_out_of_the_fit() -> None:
    plan = plan_model_dates(MONTHLY, 180)
    assert not set(plan.calibration) & set(plan.fit)


def test_selection_dates_are_a_subset_of_the_fit_dates() -> None:
    """The refit must see everything selection saw, plus the held-back recent snapshots."""
    plan = plan_model_dates(MONTHLY, 180)
    assert set(plan.selection_train) <= set(plan.fit)
    assert set(plan.selection_validation) <= set(plan.fit)


def test_plan_validate_catches_a_broken_plan() -> None:
    plan = plan_model_dates(MONTHLY, 180)
    broken = type(plan)(
        selection_train=plan.selection_train,
        selection_validation=plan.selection_validation,
        # Fitting right up against the test date violates the test embargo.
        fit=plan.fit + [min(plan.test) - pd.Timedelta(days=1)],
        calibration=plan.calibration,
        test=plan.test,
        horizon_days=180,
    )
    with pytest.raises(ValueError, match="test embargo violated"):
        broken.validate()


def test_embargoed_dates_are_reported_not_silently_dropped() -> None:
    plan = plan_model_dates(MONTHLY, 180)
    assigned = set(plan.fit) | set(plan.calibration) | set(plan.test)
    accounted = assigned | {d for dates in plan.embargoed.values() for d in dates}
    assert accounted == set(MONTHLY), "some grid dates vanished without being reported"


def test_a_longer_horizon_consumes_more_timeline() -> None:
    short = plan_model_dates(MONTHLY, 90)
    long = plan_model_dates(MONTHLY, 180)
    assert len(short.fit) > len(long.fit)


def test_too_few_dates_is_rejected() -> None:
    with pytest.raises(ValueError, match="need at least"):
        plan_model_dates(MONTHLY[:3], 180)


def test_an_impossible_embargo_is_rejected() -> None:
    """A horizon wider than the grid cannot support a three-stage split."""
    with pytest.raises(ValueError, match="clear the|need at least"):
        plan_model_dates(MONTHLY, 2000)


def test_all_dates_covers_everything_needing_a_snapshot() -> None:
    plan = plan_model_dates(MONTHLY, 180)
    assert set(plan.all_dates()) == set(plan.fit) | set(plan.calibration) | set(plan.test)


# ======================================================================================
# PREPROCESSING: what the model may not see
# ======================================================================================


@pytest.mark.parametrize(
    "column",
    ["customer_id", "churned", "purchases_in_window", "days_to_next_purchase",
     "outcome_window_end", "as_of_date", "high_value_threshold", "medium_value_threshold",
     "registration_date", "first_purchase_date", "last_purchase_date", "segment_reason",
     "is_new_at_as_of"],
)
def test_leaky_and_period_marking_columns_are_excluded(column: str) -> None:
    assert column in EXCLUDED_COLUMNS


def test_a_stray_datetime_column_is_refused() -> None:
    """A datetime that escaped the exclusion list would act as a period marker."""
    frame = pd.DataFrame(
        {"recency_days": [1, 2], "some_new_date": pd.to_datetime(["2024-01-01", "2024-02-01"])}
    )
    with pytest.raises(ValueError, match="period marker"):
        split_feature_columns(frame)


def test_booleans_are_treated_as_numeric() -> None:
    frame = pd.DataFrame({"is_seasonal_buyer": [True, False], "city": ["Berlin", "Vienna"]})
    numeric, categorical = split_feature_columns(frame)
    assert numeric == ["is_seasonal_buyer"]
    assert categorical == ["city"]


def test_feature_matrix_is_deterministically_ordered() -> None:
    frame = pd.DataFrame({"z_num": [1.0], "a_num": [2.0], "z_cat": ["x"], "a_cat": ["y"]})
    assert list(feature_matrix(frame).columns) == ["a_num", "z_num", "a_cat", "z_cat"]


def test_identity_features_are_named_for_exclusion() -> None:
    """These are excluded by the trainer, not the preprocessor, so pin the list."""
    for column in ("age", "city", "country", "preferred_brand", "acquisition_channel"):
        assert column in IDENTITY_FEATURES


# ======================================================================================
# RISK BANDS
# ======================================================================================


@pytest.mark.parametrize(
    ("probability", "expected"),
    [
        (0.0, "Low"),
        (0.29, "Low"),
        (0.30, "Medium"),      # lower edge inclusive
        (0.59, "Medium"),
        (0.60, "High"),
        (0.79, "High"),
        (0.80, "Critical"),
        (1.0, "Critical"),
    ],
)
def test_risk_bands_follow_the_configured_thresholds(probability: float, expected: str) -> None:
    assert assign_risk_level(pd.Series([probability])).iloc[0] == expected


def test_risk_levels_are_ordered_by_severity() -> None:
    """So charts and sorts put Critical last, not alphabetically between High and Low."""
    bands = assign_risk_level(pd.Series([0.9, 0.1, 0.7, 0.4]))
    assert list(bands.sort_values()) == ["Low", "Medium", "High", "Critical"]


def test_risk_bands_respect_reconfigured_thresholds(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RISK_THRESHOLD_MEDIUM", "0.10")
    monkeypatch.setenv("RISK_THRESHOLD_HIGH", "0.20")
    monkeypatch.setenv("RISK_THRESHOLD_CRITICAL", "0.30")
    from src.config.settings import get_settings

    try:
        settings = get_settings(refresh=True)
        assert assign_risk_level(pd.Series([0.25]), settings).iloc[0] == "High"
    finally:
        monkeypatch.undo()
        get_settings(refresh=True)


def test_risk_distribution_covers_every_band() -> None:
    distribution = risk_distribution(assign_risk_level(pd.Series([0.1, 0.1, 0.9])))
    assert list(distribution.index) == list(RISK_LEVELS)
    assert distribution.loc["Low", "customers"] == 2
    assert distribution.loc["High", "customers"] == 0
    assert distribution["share"].sum() == pytest.approx(1.0)


# ======================================================================================
# REVENUE AT RISK
# ======================================================================================


def test_expected_revenue_is_prorated_for_established_customers() -> None:
    """730 days of history and EUR 2,000 spend -> EUR 500 expected over 180 days."""
    expected = expected_horizon_revenue(pd.Series([2000.0]), pd.Series([730.0]), 180)
    assert expected.iloc[0] == pytest.approx(2000.0 * 180 / 730, abs=0.01)


def test_expected_revenue_never_extrapolates_beyond_observed_history() -> None:
    """A 24-day-old customer is credited with what they spent, not an annualised fantasy.

    Annualising 24 days of EUR 780 implies EUR 9,500 a year and made brand-new customers dominate
    the revenue-at-risk ranking. Flooring the denominator at the horizon stops that.
    """
    expected = expected_horizon_revenue(pd.Series([780.0]), pd.Series([24.0]), 180)
    assert expected.iloc[0] == pytest.approx(780.0)


def test_expected_revenue_is_monotone_in_tenure() -> None:
    """Longer history at the same total spend means a lower rate, so lower expected revenue."""
    revenue = pd.Series([1000.0, 1000.0, 1000.0])
    tenure = pd.Series([180.0, 365.0, 730.0])
    expected = expected_horizon_revenue(revenue, tenure, 180)
    assert expected.iloc[0] > expected.iloc[1] > expected.iloc[2]


def test_revenue_at_risk_is_probability_times_expected_revenue() -> None:
    result = revenue_at_risk(pd.Series([0.5]), pd.Series([2000.0]), pd.Series([360.0]), 180)
    assert result.iloc[0] == pytest.approx(0.5 * 1000.0, abs=0.5)


def test_revenue_at_risk_handles_missing_and_negative_revenue() -> None:
    result = revenue_at_risk(
        pd.Series([0.8, 0.8]), pd.Series([np.nan, -50.0]), pd.Series([365.0, 365.0]), 180
    )
    assert (result == 0.0).all()


def test_revenue_at_risk_is_zero_at_zero_probability() -> None:
    result = revenue_at_risk(pd.Series([0.0]), pd.Series([5000.0]), pd.Series([730.0]), 180)
    assert result.iloc[0] == 0.0


# ======================================================================================
# MODEL PERSISTENCE
# ======================================================================================


def _metadata(columns: list[str]) -> ModelMetadata:
    return ModelMetadata(
        model_name="dummy",
        trained_at="2026-01-01T00:00:00+00:00",
        horizon_days=180,
        label_mode="fixed",
        feature_columns=columns,
        numeric_columns=columns,
        categorical_columns=[],
        train_as_of_dates=["2024-01-31"],
        validation_as_of_dates=["2024-12-31"],
        test_as_of_dates=["2025-06-30"],
        train_rows=100,
        train_churn_rate=0.3,
        calibration="isotonic",
        random_seed=42,
    )


class _Dummy:
    """Minimal estimator so persistence can be tested without training anything."""

    def predict_proba(self, x):  # noqa: ANN001, ANN201
        return np.column_stack([np.full(len(x), 0.4), np.full(len(x), 0.6)])


def test_model_round_trips_with_its_metadata(tmp_path: Path) -> None:
    columns = ["recency_days", "total_orders"]
    model_path, metadata_path = save_model(_Dummy(), _metadata(columns), tmp_path)
    assert model_path.is_file() and metadata_path.is_file()

    loaded = load_model(tmp_path)
    assert loaded.metadata.horizon_days == 180
    assert loaded.metadata.feature_columns == columns
    assert loaded.metadata.calibration == "isotonic"


def test_metadata_is_readable_json(tmp_path: Path) -> None:
    _, metadata_path = save_model(_Dummy(), _metadata(["recency_days"]), tmp_path)
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert payload["horizon_days"] == 180
    assert payload["model_name"] == "dummy"


def test_scoring_refuses_a_frame_missing_expected_features(tmp_path: Path) -> None:
    """The contract that stops a stale model returning confident nonsense."""
    save_model(_Dummy(), _metadata(["recency_days", "total_orders"]), tmp_path)
    loaded = load_model(tmp_path)
    with pytest.raises(FeatureMismatchError, match="missing from the frame"):
        loaded.predict_proba(pd.DataFrame({"recency_days": [10]}))


def test_scoring_tolerates_extra_columns(tmp_path: Path) -> None:
    """New features appearing is harmless; expected ones vanishing is not."""
    save_model(_Dummy(), _metadata(["recency_days"]), tmp_path)
    loaded = load_model(tmp_path)
    probability = loaded.predict_proba(
        pd.DataFrame({"recency_days": [10, 20], "brand_new_feature": [1, 2]})
    )
    assert len(probability) == 2
    assert (probability == 0.6).all()


def test_align_reorders_to_the_stored_column_order(tmp_path: Path) -> None:
    save_model(_Dummy(), _metadata(["a", "b"]), tmp_path)
    loaded = load_model(tmp_path)
    aligned = loaded.align(pd.DataFrame({"b": [1], "a": [2]}))
    assert list(aligned.columns) == ["a", "b"]


def test_loading_without_a_trained_model_is_a_clear_error(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="train_model.py"):
        load_model(tmp_path)


def test_loading_without_metadata_is_refused(tmp_path: Path) -> None:
    """A model with no metadata cannot be interpreted, so it must not load silently."""
    save_model(_Dummy(), _metadata(["a"]), tmp_path)
    (tmp_path / "churn_model_metadata.json").unlink()
    with pytest.raises(FileNotFoundError, match="cannot be interpreted"):
        load_model(tmp_path)


# ======================================================================================
# SANITY-FLOOR BASELINES
# ======================================================================================


def test_single_feature_baselines_rank_a_perfect_feature_top() -> None:
    """A feature that separates the classes exactly should score ROC-AUC 1.0."""
    frame = pd.DataFrame(
        {
            "churned": [0, 0, 1, 1],
            "recency_days": [5, 10, 500, 600],
            "orders_365d": [9, 8, 1, 0],
        }
    )
    baselines = single_feature_baselines(frame)
    assert baselines["roc_auc"].max() == pytest.approx(1.0)


def test_single_feature_baselines_skip_absent_columns() -> None:
    frame = pd.DataFrame({"churned": [0, 1], "recency_days": [5, 500]})
    baselines = single_feature_baselines(frame)
    assert set(baselines["baseline"]) == {"recency_days"}
