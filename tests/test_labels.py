"""Tests for the churn label.

The label is the part of this section most able to look fine and be wrong, so these tests target
the three ways it could quietly break:

* labelling an **unfinished** outcome window as "did not churn", which would teach the model that
  recent customers never leave;
* labelling a customer who **was not a customer yet**;
* reading the future into the label window boundaries.

The synthetic dataset is small enough that every expected label can be worked out by hand.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.data.csv_loader import Datasets
from src.models.labels import (
    LabelMode,
    LabelParams,
    build_churn_labels,
    compare_label_modes,
    latest_labelable_as_of,
)

AS_OF = pd.Timestamp("2024-06-30")
HORIZON = 180


def _make(rows: list[tuple[str, str, str]]) -> Datasets:
    """Build a dataset from ``(customer, order, date)`` triples. Data ends 2025-12-31."""
    customers = sorted({customer for customer, _, _ in rows})
    customer_frame = pd.DataFrame(
        {
            "customer_id": pd.array(customers, dtype="string"),
            "age": pd.array([30] * len(customers), dtype="int16"),
            "customer_gender": pd.array(["Female"] * len(customers), dtype="string"),
            "city": pd.array(["Berlin"] * len(customers), dtype="string"),
            "country": pd.array(["Germany"] * len(customers), dtype="string"),
            "acquisition_channel": pd.array(["Referral"] * len(customers), dtype="string"),
            "registration_date": pd.to_datetime(
                [min(d for c, _, d in rows if c == customer) for customer in customers]
            ),
        }
    )
    products = pd.DataFrame(
        {
            "sku_id": pd.array(["P0001"], dtype="string"),
            "category": pd.array(["Apparel"], dtype="string"),
            "subcategory": pd.array(["T-Shirts"], dtype="string"),
            "brand": pd.array(["UrbanEdge"], dtype="string"),
            "product_gender": pd.array(["Men"], dtype="string"),
            "list_price": [50.0],
        }
    )
    # An anchor order fixes the end of the data at 2025-12-31 regardless of the rows supplied.
    anchor = [("__ANCHOR__", "999999", "2025-12-31")]
    customer_frame = pd.concat(
        [
            customer_frame,
            pd.DataFrame(
                {
                    "customer_id": pd.array(["__ANCHOR__"], dtype="string"),
                    "age": pd.array([30], dtype="int16"),
                    "customer_gender": pd.array(["Female"], dtype="string"),
                    "city": pd.array(["Berlin"], dtype="string"),
                    "country": pd.array(["Germany"], dtype="string"),
                    "acquisition_channel": pd.array(["Referral"], dtype="string"),
                    "registration_date": pd.to_datetime(["2025-12-31"]),
                }
            ),
        ],
        ignore_index=True,
    )
    transactions = pd.DataFrame(
        [(c, o, "P0001", d) for c, o, d in rows + anchor],
        columns=["customer_id", "order_id", "sku_id", "purchase_date"],
    )
    transactions["purchase_date"] = pd.to_datetime(transactions["purchase_date"])
    for column in ("customer_id", "order_id", "sku_id"):
        transactions[column] = transactions[column].astype("string")
    transactions["quantity"] = pd.array([1] * len(transactions), dtype="int16")
    transactions["selling_price"] = 50.0
    transactions["discount_pct"] = pd.array([0] * len(transactions), dtype="int16")
    transactions["coupon_used"] = pd.array(["No"] * len(transactions), dtype="string")
    transactions["net_order_value"] = 50.0
    transactions["payment_method"] = pd.array(["PayPal"] * len(transactions), dtype="string")
    returns = pd.DataFrame(
        {
            "customer_id": pd.array([], dtype="string"),
            "order_id": pd.array([], dtype="string"),
            "sku_id": pd.array([], dtype="string"),
            "return_date": pd.to_datetime(pd.Series([], dtype="object")),
            "return_quantity": pd.array([], dtype="int16"),
        }
    )
    return Datasets(
        customers=customer_frame, products=products, transactions=transactions, returns=returns
    )


# --- the core semantics ---------------------------------------------------------------


def test_no_purchase_in_the_window_is_churn() -> None:
    """Bought before the as-of date, nothing within the horizon -> churned."""
    data = _make([("C1", "000001", "2024-01-15")])
    labels = build_churn_labels(data, AS_OF, LabelParams(horizon_days=HORIZON)).labels
    assert labels.loc["C1", "churned"] == 1
    assert labels.loc["C1", "purchases_in_window"] == 0
    assert labels.loc["C1", "outcome_window_end"] == AS_OF + pd.Timedelta(days=HORIZON)


def test_a_purchase_in_the_window_is_not_churn() -> None:
    data = _make([("C1", "000001", "2024-01-15"), ("C1", "000002", "2024-08-01")])
    labels = build_churn_labels(data, AS_OF, LabelParams(horizon_days=HORIZON)).labels
    assert labels.loc["C1", "churned"] == 0
    assert labels.loc["C1", "purchases_in_window"] == 1


def test_a_purchase_just_inside_the_window_counts() -> None:
    """The window is ``(as_of, as_of + horizon]`` -- the final day is included."""
    edge = AS_OF + pd.Timedelta(days=HORIZON)
    data = _make([("C1", "000001", "2024-01-15"), ("C1", "000002", edge.date().isoformat())])
    labels = build_churn_labels(data, AS_OF, LabelParams(horizon_days=HORIZON)).labels
    assert labels.loc["C1", "churned"] == 0


def test_a_purchase_one_day_past_the_window_does_not_count() -> None:
    beyond = AS_OF + pd.Timedelta(days=HORIZON + 1)
    data = _make([("C1", "000001", "2024-01-15"), ("C1", "000002", beyond.date().isoformat())])
    labels = build_churn_labels(data, AS_OF, LabelParams(horizon_days=HORIZON)).labels
    assert labels.loc["C1", "churned"] == 1
    assert labels.loc["C1", "days_to_next_purchase"] == HORIZON + 1


def test_a_purchase_on_the_as_of_date_is_history_not_outcome() -> None:
    """The window opens strictly after the as-of date, so the same day cannot rescue a customer."""
    data = _make([("C1", "000001", AS_OF.date().isoformat())])
    labels = build_churn_labels(data, AS_OF, LabelParams(horizon_days=HORIZON)).labels
    assert labels.loc["C1", "label_eligible"] == True  # noqa: E712
    assert labels.loc["C1", "purchases_in_window"] == 0
    assert labels.loc["C1", "churned"] == 1


# --- censoring: the error that would be most damaging ---------------------------------


def test_an_unfinished_window_is_null_not_zero() -> None:
    """Data ends 2025-12-31, so a window opened at 2025-10-31 has not closed.

    Defaulting an unfinished window to "did not churn" would teach the model that recent customers
    never leave, which is the single most damaging mistake available here.
    """
    data = _make([("C1", "000001", "2025-09-01")])
    result = build_churn_labels(data, "2025-10-31", LabelParams(horizon_days=HORIZON))
    assert pd.isna(result.labels.loc["C1", "churned"])
    assert result.labels.loc["C1", "label_observable"] == False  # noqa: E712
    assert result.labels.loc["C1", "label_usable"] == False  # noqa: E712
    assert "extends past the end of the data" in result.labels.loc["C1", "exclusion_reason"]
    assert result.summary()["censored"] >= 1


def test_censored_rows_are_excluded_from_the_trainable_set() -> None:
    data = _make([("C1", "000001", "2025-09-01")])
    result = build_churn_labels(data, "2025-10-31", LabelParams(horizon_days=HORIZON))
    assert "C1" not in result.trainable.index


def test_latest_labelable_as_of_is_the_data_end_minus_the_horizon() -> None:
    data = _make([("C1", "000001", "2024-01-15")])
    assert latest_labelable_as_of(data, LabelParams(horizon_days=180)) == pd.Timestamp("2025-07-04")
    assert latest_labelable_as_of(data, LabelParams(horizon_days=90)) == pd.Timestamp("2025-10-02")


# --- eligibility ---------------------------------------------------------------------


def test_a_customer_with_no_history_is_ineligible_not_retained() -> None:
    """Someone who has not bought yet cannot churn, so the label is NA rather than 1."""
    data = _make([("C1", "000001", "2025-03-01")])
    labels = build_churn_labels(data, "2024-06-30", LabelParams(horizon_days=HORIZON)).labels
    assert labels.loc["C1", "label_eligible"] == False  # noqa: E712
    assert pd.isna(labels.loc["C1", "churned"])
    assert "no purchase history" in labels.loc["C1", "exclusion_reason"]


def test_new_customers_are_labelled_but_flagged() -> None:
    """Thin evidence is still evidence; it is flagged rather than discarded."""
    data = _make([("C1", "000001", "2024-06-01")])
    labels = build_churn_labels(
        data, AS_OF, LabelParams(horizon_days=HORIZON, new_customer_days=90)
    ).labels
    assert labels.loc["C1", "is_new_at_as_of"] == True  # noqa: E712
    assert labels.loc["C1", "label_usable"] == True  # noqa: E712


def test_every_customer_gets_exactly_one_row() -> None:
    data = _make(
        [("C1", "000001", "2024-01-15"), ("C2", "000002", "2024-02-15"), ("C1", "000003", "2024-03-01")]
    )
    labels = build_churn_labels(data, AS_OF, LabelParams(horizon_days=HORIZON)).labels
    assert len(labels) == len(data.customers)
    assert labels.index.is_unique


# --- the horizon is configurable ------------------------------------------------------


def test_a_shorter_horizon_labels_more_customers_as_churned() -> None:
    """A purchase 200 days out is churn at 180 days and not at 365."""
    data = _make([("C1", "000001", "2024-01-15"), ("C1", "000002", "2025-01-16")])
    at_180 = build_churn_labels(data, AS_OF, LabelParams(horizon_days=180)).labels
    at_365 = build_churn_labels(data, AS_OF, LabelParams(horizon_days=365)).labels
    assert at_180.loc["C1", "churned"] == 1
    assert at_365.loc["C1", "churned"] == 0


# --- adaptive horizon ----------------------------------------------------------------


def test_adaptive_horizon_scales_to_the_customers_own_cadence() -> None:
    """A monthly buyer gets the floor; a slow buyer gets a longer window."""
    frequent = [("C1", f"00000{i}", d) for i, d in enumerate(
        ["2024-01-01", "2024-02-01", "2024-03-01", "2024-04-01"], start=1
    )]
    slow = [("C2", "000010", "2023-01-01"), ("C2", "000011", "2023-09-01"),
            ("C2", "000012", "2024-05-01")]
    data = _make(frequent + slow)
    params = LabelParams(
        horizon_days=180, mode=LabelMode.ADAPTIVE, adaptive_multiple=2.0,
        adaptive_min_days=90, adaptive_max_days=365,
    )
    labels = build_churn_labels(data, AS_OF, params).labels
    # C1's median gap is ~30 days, so 2x30 = 60 is floored to the 90-day minimum.
    assert labels.loc["C1", "horizon_days"] == 90
    # C2's gaps are ~243 days, so 2x that is capped at 365.
    assert labels.loc["C2", "horizon_days"] == 365


def test_adaptive_falls_back_to_the_fixed_horizon_without_a_measurable_cadence() -> None:
    data = _make([("C1", "000001", "2024-01-15")])
    params = LabelParams(horizon_days=180, mode=LabelMode.ADAPTIVE)
    labels = build_churn_labels(data, AS_OF, params).labels
    assert labels.loc["C1", "horizon_days"] == 180


def test_label_mode_comparison_reports_disagreement() -> None:
    data = _make(
        [("C1", "000001", "2024-01-15"), ("C1", "000002", "2024-03-01"),
         ("C2", "000010", "2023-06-01"), ("C2", "000011", "2024-05-01")]
    )
    comparison = compare_label_modes(data, AS_OF, LabelParams(horizon_days=180))
    assert comparison["comparable_customers"] >= 1
    assert 0.0 <= comparison["agreement"] <= 1.0
    assert comparison["rescued_by_adaptive"] >= 0
    assert comparison["caught_by_adaptive"] >= 0


# --- parameter validation ------------------------------------------------------------


@pytest.mark.parametrize(
    ("kwargs", "fragment"),
    [
        ({"horizon_days": 0}, "horizon_days must be positive"),
        ({"adaptive_multiple": 0}, "adaptive_multiple must be positive"),
        ({"adaptive_min_days": 400, "adaptive_max_days": 100}, "min <= max"),
    ],
)
def test_invalid_label_params_are_rejected(kwargs: dict, fragment: str) -> None:
    with pytest.raises(ValueError, match=fragment):
        LabelParams(**kwargs).validate()


# --- against the real data -----------------------------------------------------------


def test_real_data_label_is_leakage_free(data) -> None:
    """Truncating the data at the as-of date must not change any label.

    The label reads only ``(as_of, as_of + horizon]``, so removing rows outside the union of
    history and that window cannot move it.
    """
    as_of = pd.Timestamp("2024-06-30")
    params = LabelParams(horizon_days=180)
    full = build_churn_labels(data, as_of, params).labels
    window_end = as_of + pd.Timedelta(days=180)
    truncated_data = Datasets(
        customers=data.customers,
        products=data.products,
        transactions=data.transactions[data.transactions["purchase_date"].le(window_end)],
        returns=data.returns,
    )
    truncated = build_churn_labels(truncated_data, as_of, params).labels
    pd.testing.assert_series_equal(full["churned"], truncated["churned"])


def test_real_data_churn_rate_is_plausible(data) -> None:
    result = build_churn_labels(data, "2024-06-30", LabelParams(horizon_days=180))
    summary = result.summary()
    assert summary["usable"] == 457          # customers with history at that date
    assert 0.2 < summary["churn_rate"] < 0.5
    assert summary["censored"] == 0


def test_real_data_late_as_of_is_fully_censored(data) -> None:
    """Past 2025-07-04 no 180-day window can close, so nothing is labelable."""
    result = build_churn_labels(data, "2025-10-31", LabelParams(horizon_days=180))
    assert result.summary()["usable"] == 0
    assert result.summary()["censored"] > 0
