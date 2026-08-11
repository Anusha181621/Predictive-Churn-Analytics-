"""Unit tests for the validation layer itself.

The tests in ``test_data_integrity.py`` prove the validators *pass* on the real CSV files. That
alone is weak evidence: a validator that always returns "PASS" would satisfy it. These tests do
the complementary job — they build a small, internally consistent synthetic dataset, break
exactly one thing, and assert that the specific named check fails.

Nothing here touches the real CSV files.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.data.csv_loader import Datasets
from src.data.validation import (
    ValidationReport,
    compute_return_rate,
    validate_customers,
    validate_datasets,
    validate_products,
    validate_relationships,
    validate_returns,
    validate_transactions,
)


# --------------------------------------------------------------------------------------
# a tiny, internally consistent dataset
#
# 3 customers, 3 SKUs, 3 orders (4 lines), 1 return. Small enough to reason about, complete
# enough that every error-severity check has something to look at.
# --------------------------------------------------------------------------------------


def _customers() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "customer_id": pd.array(["CUST0001", "CUST0002", "CUST0003"], dtype="string"),
            "age": pd.array([30, 45, 22], dtype="int16"),
            "customer_gender": pd.array(["Female", "Male", "Other / Prefer not to say"], dtype="string"),
            "city": pd.array(["Berlin", "Amsterdam", "Vienna"], dtype="string"),
            "country": pd.array(["Germany", "Netherlands", "Austria"], dtype="string"),
            "acquisition_channel": pd.array(["Referral", "Email", "Instagram"], dtype="string"),
            "registration_date": pd.to_datetime(["2023-01-05", "2023-02-10", "2023-03-15"]),
        }
    )


def _products() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sku_id": pd.array(["P0001", "P0002", "P0003"], dtype="string"),
            "category": pd.array(["Apparel", "Footwear", "Accessories"], dtype="string"),
            "subcategory": pd.array(["T-Shirts", "Sneakers", "Bags"], dtype="string"),
            "brand": pd.array(["UrbanEdge", "NovaWear", "LuxeLine"], dtype="string"),
            "product_gender": pd.array(["Men", "Women", "Unisex"], dtype="string"),
            "list_price": [20.00, 80.00, 150.00],
        }
    )


def _transactions() -> pd.DataFrame:
    """Four order lines across three orders. Net order value is arithmetically correct."""
    rows = [
        # customer,   order,    sku,     date,         qty, price, disc, coupon, payment
        ("CUST0001", "000001", "P0001", "2023-01-05", 2, 20.00, 0, "No", "PayPal"),
        ("CUST0001", "000001", "P0002", "2023-01-05", 1, 80.00, 10, "Yes", "PayPal"),
        ("CUST0002", "000002", "P0003", "2023-02-10", 1, 150.00, 20, "Yes", "Credit Card"),
        ("CUST0003", "000003", "P0001", "2023-03-15", 3, 20.00, 0, "No", "Debit Card"),
    ]
    frame = pd.DataFrame(
        rows,
        columns=[
            "customer_id", "order_id", "sku_id", "purchase_date", "quantity",
            "selling_price", "discount_pct", "coupon_used", "payment_method",
        ],
    )
    frame["purchase_date"] = pd.to_datetime(frame["purchase_date"])
    for column in ("customer_id", "order_id", "sku_id", "coupon_used", "payment_method"):
        frame[column] = frame[column].astype("string")
    frame["quantity"] = frame["quantity"].astype("int16")
    frame["discount_pct"] = frame["discount_pct"].astype("int16")
    frame["net_order_value"] = (
        frame["quantity"] * frame["selling_price"] * (1 - frame["discount_pct"] / 100.0)
    ).round(2)
    return frame[[
        "customer_id", "order_id", "sku_id", "purchase_date", "quantity", "selling_price",
        "discount_pct", "coupon_used", "net_order_value", "payment_method",
    ]]


def _returns() -> pd.DataFrame:
    """One return: 1 of the 2 units of P0001 on order 000001, 10 days later."""
    frame = pd.DataFrame(
        {
            "customer_id": pd.array(["CUST0001"], dtype="string"),
            "order_id": pd.array(["000001"], dtype="string"),
            "sku_id": pd.array(["P0001"], dtype="string"),
            "return_date": pd.to_datetime(["2023-01-15"]),
            "return_quantity": pd.array([1], dtype="int16"),
        }
    )
    return frame


@pytest.fixture
def tiny() -> Datasets:
    return Datasets(
        customers=_customers(),
        products=_products(),
        transactions=_transactions(),
        returns=_returns(),
    )


def _errors(report) -> set[str]:
    """The names of the error-severity checks that failed."""
    return {check.name for check in report.errors}


# --------------------------------------------------------------------------------------
# the synthetic dataset must itself be clean, or every negative test below is meaningless
# --------------------------------------------------------------------------------------


def test_the_synthetic_dataset_has_no_errors(tiny: Datasets) -> None:
    report = validate_datasets(tiny)
    assert report.ok, [f"{c.full_name}: {c.detail}" for c in report.errors]


def test_each_validator_is_clean_on_the_synthetic_dataset(tiny: Datasets) -> None:
    assert validate_customers(tiny.customers, transactions=tiny.transactions).ok
    assert validate_products(tiny.products).ok
    assert validate_transactions(
        tiny.transactions, customers=tiny.customers, products=tiny.products
    ).ok
    assert validate_returns(tiny.returns, transactions=tiny.transactions).ok
    assert validate_relationships(tiny).ok


# --------------------------------------------------------------------------------------
# Customer.csv
# --------------------------------------------------------------------------------------


def test_missing_customer_id_column_is_reported(tiny: Datasets) -> None:
    report = validate_customers(tiny.customers.drop(columns=["customer_id"]))
    assert not report.ok
    assert "required columns are present" in _errors(report)
    assert "customer_id" in report.check("required columns are present").detail


def test_duplicate_customer_id_is_caught(tiny: Datasets) -> None:
    customers = pd.concat([tiny.customers, tiny.customers.iloc[[0]]], ignore_index=True)
    report = validate_customers(customers)
    assert "Customer ID is unique" in _errors(report)
    assert report.metrics["duplicate_customers"] == 1
    assert report.metrics["total_customers"] == 4
    assert report.metrics["unique_customers"] == 3


def test_null_customer_id_is_caught(tiny: Datasets) -> None:
    customers = tiny.customers.copy()
    customers.loc[1, "customer_id"] = pd.NA
    report = validate_customers(customers)
    assert "customer_id is not null" in _errors(report)
    assert report.metrics["missing_values"] == {"customer_id": 1}


def test_blank_customer_id_is_caught(tiny: Datasets) -> None:
    customers = tiny.customers.copy()
    customers.loc[1, "customer_id"] = "   "
    report = validate_customers(customers)
    assert "Customer ID is not blank" in _errors(report)


@pytest.mark.parametrize("bad_age", [-1, 200])
def test_impossible_age_is_caught(tiny: Datasets, bad_age: int) -> None:
    customers = tiny.customers.copy()
    customers["age"] = customers["age"].astype("int32")
    customers.loc[0, "age"] = bad_age
    report = validate_customers(customers)
    assert "Age is within the possible range 0-120" in _errors(report)


def test_age_outside_the_documented_range_is_only_a_warning(tiny: Datasets) -> None:
    customers = tiny.customers.copy()
    customers["age"] = customers["age"].astype("int32")
    customers.loc[0, "age"] = 71  # possible, but outside the documented 18-65
    report = validate_customers(customers)
    assert report.ok, "an unusual but possible age must not be an error"
    assert "Age is within the documented range 18-65" in {c.name for c in report.warnings}


def test_non_numeric_age_is_caught(tiny: Datasets) -> None:
    customers = tiny.customers.copy()
    customers["age"] = customers["age"].astype("object")
    customers.loc[0, "age"] = "thirty"
    report = validate_customers(customers)
    assert "Age is numeric" in _errors(report)


def test_unknown_gender_value_is_flagged_as_a_warning(tiny: Datasets) -> None:
    customers = tiny.customers.copy()
    customers.loc[0, "customer_gender"] = "Robot"
    report = validate_customers(customers)
    warning = report.check("customer_gender values are within the documented set")
    assert not warning.passed
    assert warning.severity == "warning"
    assert "Robot" in warning.detail


def test_unparsed_registration_date_is_caught(tiny: Datasets) -> None:
    customers = tiny.customers.copy()
    customers["registration_date"] = customers["registration_date"].astype("string")
    report = validate_customers(customers)
    assert "registration_date parses as a date" in _errors(report)


def test_implausibly_early_registration_date_is_caught(tiny: Datasets) -> None:
    customers = tiny.customers.copy()
    customers.loc[0, "registration_date"] = pd.Timestamp("1899-05-01")
    report = validate_customers(customers)
    assert "registration_date is not implausibly early" in _errors(report)


def test_registration_after_the_last_purchase_is_caught(tiny: Datasets) -> None:
    customers = tiny.customers.copy()
    customers.loc[0, "registration_date"] = pd.Timestamp("2030-01-01")
    report = validate_customers(customers, transactions=tiny.transactions)
    assert "registration_date is not after the last purchase date" in _errors(report)


def test_customer_metrics_include_the_date_range(tiny: Datasets) -> None:
    report = validate_customers(tiny.customers)
    assert report.metrics["registration_date_min"] == pd.Timestamp("2023-01-05")
    assert report.metrics["registration_date_max"] == pd.Timestamp("2023-03-15")
    assert report.metrics["missing_values"] == {}


# --------------------------------------------------------------------------------------
# Product.csv
# --------------------------------------------------------------------------------------


def test_duplicate_sku_is_caught(tiny: Datasets) -> None:
    products = pd.concat([tiny.products, tiny.products.iloc[[2]]], ignore_index=True)
    report = validate_products(products)
    assert "SKU ID is unique" in _errors(report)
    assert report.metrics["duplicate_skus"] == 1


def test_null_sku_is_caught(tiny: Datasets) -> None:
    products = tiny.products.copy()
    products.loc[0, "sku_id"] = pd.NA
    report = validate_products(products)
    assert "sku_id is not null" in _errors(report)


def test_negative_price_is_caught(tiny: Datasets) -> None:
    products = tiny.products.copy()
    products.loc[1, "list_price"] = -5.0
    report = validate_products(products)
    assert "Price is greater than or equal to zero" in _errors(report)


def test_zero_price_is_only_a_warning(tiny: Datasets) -> None:
    products = tiny.products.copy()
    products.loc[1, "list_price"] = 0.0
    report = validate_products(products)
    assert report.ok, "a zero price is suspicious but not structurally invalid"
    assert "Price is greater than zero" in {c.name for c in report.warnings}


@pytest.mark.parametrize("column", ["category", "subcategory"])
def test_blank_category_is_caught(tiny: Datasets, column: str) -> None:
    products = tiny.products.copy()
    products.loc[0, column] = ""
    report = validate_products(products)
    assert f"{column} is populated" in _errors(report)


@pytest.mark.parametrize("column", ["category", "subcategory"])
def test_missing_category_column_is_reported(tiny: Datasets, column: str) -> None:
    report = validate_products(tiny.products.drop(columns=[column]))
    assert not report.ok
    assert column in report.check("required columns are present").detail


# --------------------------------------------------------------------------------------
# Transaction.csv
# --------------------------------------------------------------------------------------


def test_zero_quantity_is_caught(tiny: Datasets) -> None:
    transactions = tiny.transactions.copy()
    transactions.loc[0, "quantity"] = 0
    report = validate_transactions(transactions)
    assert "Quantity is greater than zero" in _errors(report)


def test_negative_quantity_is_caught(tiny: Datasets) -> None:
    transactions = tiny.transactions.copy()
    transactions.loc[0, "quantity"] = -2
    report = validate_transactions(transactions)
    assert "Quantity is greater than zero" in _errors(report)


def test_negative_selling_price_is_caught(tiny: Datasets) -> None:
    transactions = tiny.transactions.copy()
    transactions.loc[0, "selling_price"] = -1.0
    report = validate_transactions(transactions)
    assert "Selling price is greater than or equal to zero" in _errors(report)


@pytest.mark.parametrize("bad_discount", [-10, 101, 150])
def test_discount_outside_zero_to_one_hundred_is_caught(
    tiny: Datasets, bad_discount: int
) -> None:
    transactions = tiny.transactions.copy()
    transactions["discount_pct"] = transactions["discount_pct"].astype("int32")
    transactions.loc[0, "discount_pct"] = bad_discount
    report = validate_transactions(transactions)
    assert "Discount is between 0 and 100 percent" in _errors(report)


def test_undocumented_but_in_range_discount_is_only_a_warning(tiny: Datasets) -> None:
    transactions = tiny.transactions.copy()
    transactions["discount_pct"] = transactions["discount_pct"].astype("int32")
    transactions.loc[0, "discount_pct"] = 7  # in range, but not in the documented domain
    transactions["net_order_value"] = (
        transactions["quantity"] * transactions["selling_price"]
        * (1 - transactions["discount_pct"] / 100.0)
    ).round(2)
    report = validate_transactions(transactions)
    assert report.ok
    assert "Discount is drawn from the documented domain" in {c.name for c in report.warnings}


def test_wrong_net_order_value_is_caught(tiny: Datasets) -> None:
    transactions = tiny.transactions.copy()
    transactions.loc[0, "net_order_value"] = 999.99
    report = validate_transactions(transactions)
    assert "Net order value equals quantity x selling price x (1 - discount)" in _errors(report)


def test_rounding_sized_net_order_value_deviation_is_tolerated(tiny: Datasets) -> None:
    """Half a cent must not be reported: the source file is rounded to 2 decimals."""
    transactions = tiny.transactions.copy()
    transactions.loc[0, "net_order_value"] += 0.005
    report = validate_transactions(transactions)
    assert report.ok


def test_coupon_with_zero_discount_is_caught(tiny: Datasets) -> None:
    transactions = tiny.transactions.copy()
    transactions.loc[0, "coupon_used"] = "Yes"  # row 0 has a 0% discount
    report = validate_transactions(transactions)
    assert "A coupon implies a non-zero discount" in _errors(report)


def test_unknown_customer_id_in_transactions_is_caught(tiny: Datasets) -> None:
    transactions = tiny.transactions.copy()
    transactions.loc[0, "customer_id"] = "CUST9999"
    report = validate_transactions(transactions, customers=tiny.customers)
    failure = report.check("Customer ID exists in Customer.csv")
    assert not failure.passed
    assert "CUST9999" in failure.detail


def test_unknown_sku_id_in_transactions_is_caught(tiny: Datasets) -> None:
    transactions = tiny.transactions.copy()
    transactions.loc[0, "sku_id"] = "P9999"
    report = validate_transactions(transactions, products=tiny.products)
    failure = report.check("SKU ID exists in Product.csv")
    assert not failure.passed
    assert "P9999" in failure.detail


def test_foreign_keys_are_skipped_when_the_other_table_is_absent(tiny: Datasets) -> None:
    """The validator must be usable on one file in isolation."""
    report = validate_transactions(tiny.transactions)
    names = {check.name for check in report.checks}
    assert "Customer ID exists in Customer.csv" not in names
    assert "SKU ID exists in Product.csv" not in names
    assert report.ok


def test_order_id_read_as_an_integer_is_caught(tiny: Datasets) -> None:
    """The zero-padding trap: pandas inferring int64 must be reported, not tolerated."""
    transactions = tiny.transactions.copy()
    transactions["order_id"] = transactions["order_id"].astype("int64")
    report = validate_transactions(transactions)
    failure = report.check("Order ID is a string, preserving zero padding")
    assert not failure.passed
    assert "zero padding" in failure.detail


def test_inconsistent_order_id_width_is_caught(tiny: Datasets) -> None:
    transactions = tiny.transactions.copy()
    transactions.loc[0, "order_id"] = "1"
    report = validate_transactions(transactions)
    assert "Order ID is zero-padded to 6 characters" in _errors(report)


def test_transaction_metrics(tiny: Datasets) -> None:
    report = validate_transactions(tiny.transactions)
    assert report.metrics["total_order_lines"] == 4
    assert report.metrics["distinct_orders"] == 3
    assert report.metrics["purchasing_customers"] == 3
    assert report.metrics["total_units_purchased"] == 7
    assert report.metrics["purchase_date_min"] == pd.Timestamp("2023-01-05")
    assert report.metrics["purchase_date_max"] == pd.Timestamp("2023-03-15")


# --------------------------------------------------------------------------------------
# Return.csv
# --------------------------------------------------------------------------------------


def test_return_with_no_matching_transaction_is_caught(tiny: Datasets) -> None:
    returns = tiny.returns.copy()
    returns.loc[0, "order_id"] = "009999"
    report = validate_returns(returns, transactions=tiny.transactions)
    assert "every return corresponds to an actual transaction line" in _errors(report)


def test_return_of_a_sku_not_in_that_order_is_caught(tiny: Datasets) -> None:
    """Order 000001 contains P0001 and P0002 but not P0003."""
    returns = tiny.returns.copy()
    returns.loc[0, "sku_id"] = "P0003"
    report = validate_returns(returns, transactions=tiny.transactions)
    assert "every return corresponds to an actual transaction line" in _errors(report)


def test_over_return_is_caught(tiny: Datasets) -> None:
    """Only 2 units of P0001 were bought on order 000001."""
    returns = tiny.returns.copy()
    returns.loc[0, "return_quantity"] = 3
    report = validate_returns(returns, transactions=tiny.transactions)
    assert "return quantity does not exceed the purchased quantity" in _errors(report)


def test_returning_exactly_what_was_purchased_is_allowed(tiny: Datasets) -> None:
    returns = tiny.returns.copy()
    returns.loc[0, "return_quantity"] = 2
    report = validate_returns(returns, transactions=tiny.transactions)
    assert report.ok


@pytest.mark.parametrize("bad_quantity", [0, -1])
def test_non_positive_return_quantity_is_caught(tiny: Datasets, bad_quantity: int) -> None:
    returns = tiny.returns.copy()
    returns["return_quantity"] = returns["return_quantity"].astype("int32")
    returns.loc[0, "return_quantity"] = bad_quantity
    report = validate_returns(returns, transactions=tiny.transactions)
    assert "Return quantity is greater than zero" in _errors(report)


@pytest.mark.parametrize("bad_date", ["2023-01-05", "2022-12-01"])
def test_return_date_not_after_purchase_is_caught(tiny: Datasets, bad_date: str) -> None:
    """Same-day and earlier return dates are both invalid; the purchase was 2023-01-05."""
    returns = tiny.returns.copy()
    returns.loc[0, "return_date"] = pd.Timestamp(bad_date)
    report = validate_returns(returns, transactions=tiny.transactions)
    assert "return date is after the purchase date" in _errors(report)


def test_duplicate_return_line_is_caught(tiny: Datasets) -> None:
    returns = pd.concat([tiny.returns, tiny.returns], ignore_index=True)
    report = validate_returns(returns, transactions=tiny.transactions)
    assert "at most one return row per order line" in _errors(report)


def test_return_attributed_to_the_wrong_customer_is_caught(tiny: Datasets) -> None:
    returns = tiny.returns.copy()
    returns.loc[0, "customer_id"] = "CUST0002"  # order 000001 belongs to CUST0001
    report = validate_returns(returns, transactions=tiny.transactions)
    assert "return customer matches the customer who placed the order" in _errors(report)


def test_return_checks_against_transactions_are_skipped_when_absent(tiny: Datasets) -> None:
    report = validate_returns(tiny.returns)
    names = {check.name for check in report.checks}
    assert "every return corresponds to an actual transaction line" not in names
    assert "Return quantity is greater than zero" in names
    assert report.ok


def test_return_metrics(tiny: Datasets) -> None:
    report = validate_returns(tiny.returns, transactions=tiny.transactions)
    assert report.metrics["total_return_lines"] == 1
    assert report.metrics["total_units_returned"] == 1
    assert report.metrics["customers_with_returns"] == 1
    assert report.metrics["return_date_max"] == pd.Timestamp("2023-01-15")


# --------------------------------------------------------------------------------------
# cross-table relationships
# --------------------------------------------------------------------------------------


def test_order_spanning_two_customers_is_caught(tiny: Datasets) -> None:
    transactions = tiny.transactions.copy()
    transactions.loc[1, "customer_id"] = "CUST0002"  # same order 000001, different customer
    report = validate_relationships(
        Datasets(tiny.customers, tiny.products, transactions, tiny.returns)
    )
    assert "one customer per order" in _errors(report)


def test_order_spanning_two_dates_is_caught(tiny: Datasets) -> None:
    transactions = tiny.transactions.copy()
    transactions.loc[1, "purchase_date"] = pd.Timestamp("2023-01-06")
    report = validate_relationships(
        Datasets(tiny.customers, tiny.products, transactions, tiny.returns)
    )
    assert "one purchase date per order" in _errors(report)


def test_order_with_two_payment_methods_is_caught(tiny: Datasets) -> None:
    transactions = tiny.transactions.copy()
    transactions.loc[1, "payment_method"] = "Debit Card"
    report = validate_relationships(
        Datasets(tiny.customers, tiny.products, transactions, tiny.returns)
    )
    assert "one payment method per order" in _errors(report)


def test_duplicate_sku_within_an_order_is_caught(tiny: Datasets) -> None:
    transactions = tiny.transactions.copy()
    transactions.loc[1, "sku_id"] = "P0001"  # order 000001 would hold P0001 twice
    report = validate_relationships(
        Datasets(tiny.customers, tiny.products, transactions, tiny.returns)
    )
    assert "a SKU appears at most once per order" in _errors(report)


def test_purchase_before_registration_is_caught(tiny: Datasets) -> None:
    customers = tiny.customers.copy()
    customers.loc[0, "registration_date"] = pd.Timestamp("2023-06-01")
    report = validate_relationships(
        Datasets(customers, tiny.products, tiny.transactions, tiny.returns)
    )
    assert "no purchase predates the customer's registration" in _errors(report)


def test_registration_not_equal_to_first_purchase_is_caught(tiny: Datasets) -> None:
    customers = tiny.customers.copy()
    customers.loc[0, "registration_date"] = pd.Timestamp("2023-01-01")
    report = validate_relationships(
        Datasets(customers, tiny.products, tiny.transactions, tiny.returns)
    )
    assert "registration date equals the first purchase date" in _errors(report)


def test_city_in_two_countries_is_caught(tiny: Datasets) -> None:
    customers = pd.concat([tiny.customers, tiny.customers.iloc[[0]]], ignore_index=True)
    customers.loc[3, "customer_id"] = "CUST0004"
    customers.loc[3, "country"] = "Austria"  # Berlin already maps to Germany
    report = validate_relationships(
        Datasets(customers, tiny.products, tiny.transactions, tiny.returns)
    )
    assert "each city maps to a single country" in _errors(report)


def test_customer_without_orders_is_only_a_warning(tiny: Datasets) -> None:
    customers = pd.concat([tiny.customers, tiny.customers.iloc[[0]]], ignore_index=True)
    customers.loc[3, "customer_id"] = "CUST0004"
    customers.loc[3, "registration_date"] = pd.Timestamp("2023-04-01")
    report = validate_relationships(
        Datasets(customers, tiny.products, tiny.transactions, tiny.returns)
    )
    assert report.ok
    assert "every customer has at least one transaction" in {c.name for c in report.warnings}


# --------------------------------------------------------------------------------------
# return rate
# --------------------------------------------------------------------------------------


def test_return_rate_is_measured_not_assumed(tiny: Datasets) -> None:
    """7 units purchased, 1 returned -> 1/7, nowhere near 20%."""
    rates = compute_return_rate(tiny.transactions, tiny.returns)
    assert rates["purchased_units"] == 7
    assert rates["returned_units"] == 1
    assert rates["unit_return_rate"] == pytest.approx(1 / 7)
    assert rates["line_return_rate"] == pytest.approx(1 / 4)
    assert rates["order_return_rate"] == pytest.approx(1 / 3)


def test_return_rate_reports_the_three_denominators_separately(tiny: Datasets) -> None:
    rates = compute_return_rate(tiny.transactions, tiny.returns)
    assert rates["unit_return_rate"] != rates["line_return_rate"]
    assert rates["total_order_lines"] == 4
    assert rates["total_orders"] == 3
    assert rates["orders_with_returns"] == 1


def test_return_rate_with_no_returns_is_zero(tiny: Datasets) -> None:
    empty = tiny.returns.iloc[0:0]
    rates = compute_return_rate(tiny.transactions, empty)
    assert rates["returned_units"] == 0
    assert rates["unit_return_rate"] == 0.0


def test_return_rate_does_not_divide_by_zero(tiny: Datasets) -> None:
    rates = compute_return_rate(tiny.transactions.iloc[0:0], tiny.returns.iloc[0:0])
    assert rates["unit_return_rate"] is None
    assert rates["line_return_rate"] is None
    assert rates["order_return_rate"] is None


# --------------------------------------------------------------------------------------
# the report object
# --------------------------------------------------------------------------------------


def test_a_broken_validator_is_reported_not_raised(tiny: Datasets) -> None:
    """Structurally destroyed input must still yield a report, never a traceback."""
    broken = Datasets(
        customers=tiny.customers.drop(columns=["registration_date"]),
        products=tiny.products,
        transactions=tiny.transactions.drop(columns=["purchase_date"]),
        returns=tiny.returns,
    )
    report = validate_datasets(broken)  # must not raise
    assert not report.ok
    assert report.summary()["errors"] > 0


def test_report_json_round_trips(tiny: Datasets) -> None:
    report = validate_datasets(tiny)
    payload = json.loads(report.to_json())
    assert payload["ok"] is True
    assert payload["summary"]["total"] == len(report.checks)
    assert set(payload["tables"]) == {
        "customers", "products", "transactions", "returns", "relationships",
    }
    # Timestamps and numpy scalars must have been made JSON-safe.
    assert isinstance(payload["dataset"]["purchase_date_max"], str)
    assert isinstance(payload["dataset"]["unit_return_rate"], float)
    assert payload["tables"]["customers"]["metrics"]["total_customers"] == 3


def test_report_saves_to_the_requested_path(tiny: Datasets, tmp_path: Path) -> None:
    destination = tmp_path / "nested" / "data_quality_report.json"
    written = validate_datasets(tiny).save(destination)
    assert written == destination
    assert json.loads(destination.read_text(encoding="utf-8"))["ok"] is True


def test_report_markdown_includes_every_table(tiny: Datasets) -> None:
    markdown = validate_datasets(tiny).to_markdown()
    for table in ("customers", "products", "transactions", "returns", "relationships"):
        assert f"## {table}" in markdown
    assert "checks passed" in markdown


def test_lookup_by_name_raises_for_an_unknown_check(tiny: Datasets) -> None:
    report = validate_datasets(tiny)
    with pytest.raises(KeyError):
        report.check("customers: no such check")


def test_severity_partitions_the_checks(tiny: Datasets) -> None:
    report: ValidationReport = validate_datasets(tiny)
    counts = report.summary()
    assert counts["total"] == len(report.checks)
    assert counts["passed"] + len(report.errors) + len(report.warnings) == counts["total"]
