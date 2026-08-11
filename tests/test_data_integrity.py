"""Data integrity tests.

Two jobs:

1. Assert that every ``error``-severity check in :mod:`src.data.validation` passes, so a
   corrupted or truncated CSV drop is caught immediately.
2. Pin the three known, intentional properties of the dataset that are easy to misread as
   bugs -- the post-window return dates, the unit-vs-line return rate, and the
   registration/first-purchase equality. Encoding them as tests means a future reader finds a
   documented fact rather than rediscovering a "bug".
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.data import schema as sch
from src.data.csv_loader import Datasets
from src.data.validation import DataQualityReport


# --- the report as a whole -----------------------------------------------------------


def test_no_error_severity_check_fails(report: DataQualityReport) -> None:
    failures = [f"{r.name}: {r.detail}" for r in report.errors]
    assert not failures, "data quality errors:\n  " + "\n  ".join(failures)


def test_no_warnings_against_the_shipped_dataset(report: DataQualityReport) -> None:
    """Row counts and categorical domains should match the data as shipped."""
    warnings = [f"{r.name}: {r.detail}" for r in report.warnings]
    assert not warnings, "data quality warnings:\n  " + "\n  ".join(warnings)


def test_report_renders(report: DataQualityReport) -> None:
    assert "checks passed" in report.to_markdown()
    assert report.to_dict()["summary"]["total"] == len(report.results)


# --- key counts ----------------------------------------------------------------------


def test_key_counts(data: Datasets) -> None:
    assert data.customers["customer_id"].nunique() == sch.EXPECTED_CUSTOMER_COUNT
    assert data.products["sku_id"].nunique() == sch.EXPECTED_SKU_COUNT
    assert data.transactions["order_id"].nunique() == sch.EXPECTED_ORDER_COUNT


def test_transaction_date_range(data: Datasets) -> None:
    purchase_date = data.transactions["purchase_date"]
    assert purchase_date.min() == pd.Timestamp("2023-01-02")
    assert purchase_date.max() == pd.Timestamp("2025-12-31")


# --- referential integrity -----------------------------------------------------------


def test_no_orphan_foreign_keys(data: Datasets) -> None:
    customer_ids = set(data.customers["customer_id"])
    sku_ids = set(data.products["sku_id"])
    order_ids = set(data.transactions["order_id"])

    assert set(data.transactions["customer_id"]) <= customer_ids
    assert set(data.transactions["sku_id"]) <= sku_ids
    assert set(data.returns["customer_id"]) <= customer_ids
    assert set(data.returns["sku_id"]) <= sku_ids
    assert set(data.returns["order_id"]) <= order_ids


def test_order_grain_is_intact(data: Datasets) -> None:
    grouped = data.transactions.groupby("order_id", observed=True)
    for column in ("customer_id", "purchase_date", "payment_method"):
        assert grouped[column].nunique().max() == 1, column
    assert not data.transactions.duplicated(["order_id", "sku_id"]).any()


# --- arithmetic ----------------------------------------------------------------------


def test_net_order_value_matches_its_formula(data: Datasets) -> None:
    txn = data.transactions
    expected = (txn["quantity"] * txn["selling_price"] * (1 - txn["discount_pct"] / 100.0)).round(2)
    assert (expected - txn["net_order_value"]).abs().max() <= 0.011


def test_discount_domain_and_coupon_rule(data: Datasets) -> None:
    txn = data.transactions
    assert set(txn["discount_pct"].unique().tolist()) <= set(sch.ALLOWED_DISCOUNTS)
    assert not ((txn["coupon_used"] == "Yes") & (txn["discount_pct"] == 0)).any()


# --- returns -------------------------------------------------------------------------


def test_returns_never_exceed_what_was_purchased(data: Datasets) -> None:
    merged = data.returns.merge(
        data.transactions[["order_id", "sku_id", "quantity", "purchase_date"]],
        on=["order_id", "sku_id"],
        how="left",
        validate="1:1",
    )
    assert (merged["return_quantity"] <= merged["quantity"]).all()
    assert (merged["return_date"] > merged["purchase_date"]).all()


def test_unit_and_line_return_rates_are_different_numbers(data: Datasets) -> None:
    """The brief's "~20% return rate" is the UNIT rate. The line rate is ~25%.

    Confusing the two mis-states every return feature, so both are pinned here.
    """
    purchased_units = data.transactions["quantity"].sum()
    returned_units = data.returns["return_quantity"].sum()
    unit_rate = returned_units / purchased_units
    line_rate = len(data.returns) / len(data.transactions)

    assert unit_rate == pytest.approx(0.20, abs=1e-4)
    assert line_rate == pytest.approx(0.2524, abs=1e-3)
    assert unit_rate != pytest.approx(line_rate, abs=1e-3)


def test_some_return_dates_fall_after_the_last_purchase_date(data: Datasets) -> None:
    """Intentional: late-December orders are returned the following January.

    Any feature computed "as of" a date must clip returns to that date, or a return that had
    not happened yet leaks into the feature.
    """
    last_purchase = data.transactions["purchase_date"].max()
    beyond = data.returns["return_date"] > last_purchase
    assert beyond.sum() == 104
    assert data.returns["return_date"].max() == pd.Timestamp("2026-01-29")


# --- customer timeline ---------------------------------------------------------------


def test_registration_date_equals_the_first_purchase_date(data: Datasets) -> None:
    """True for all 1,000 customers, so tenure and days-since-first-purchase are one feature."""
    first_purchase = data.transactions.groupby("customer_id", observed=True)["purchase_date"].min()
    registration = data.customers.set_index("customer_id")["registration_date"]
    assert registration.reindex(first_purchase.index).equals(first_purchase)


def test_every_customer_has_at_least_one_transaction(data: Datasets) -> None:
    """So there is no cold-start cohort in this data to exercise a no-history code path."""
    assert set(data.customers["customer_id"]) == set(data.transactions["customer_id"])


def test_no_purchase_predates_registration(data: Datasets) -> None:
    registration = data.customers.set_index("customer_id")["registration_date"]
    txn = data.transactions
    assert (txn["purchase_date"] >= txn["customer_id"].map(registration)).all()


def test_newest_customers_are_right_censored(data: Datasets) -> None:
    """The newest registration is 10 days before the window ends.

    Those customers cannot be labelled churned or retained on any sensible horizon and must be
    flagged or excluded when the label is built.
    """
    registration = data.customers["registration_date"]
    last_purchase = data.transactions["purchase_date"].max()
    assert registration.max() == pd.Timestamp("2025-12-21")
    assert (last_purchase - registration.max()).days == 10


# --- geography -----------------------------------------------------------------------


def test_each_city_belongs_to_exactly_one_country(data: Datasets) -> None:
    per_city = data.customers.groupby("city", observed=True)["country"].nunique()
    assert per_city.max() == 1
    assert per_city.size == 32
