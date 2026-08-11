"""CSV loader tests.

These pin the traps found when the source files were first inspected. The zero-padded
``Order ID`` test in particular is a regression guard: pandas' default type inference reads
that column as ``int64``, which silently turns ``"000001"`` into ``1`` and breaks every string
join against ``Return.csv``.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.data import schema as sch
from src.data.csv_loader import (
    Datasets,
    SchemaError,
    clear_cache,
    load_customers,
    load_products,
    load_returns,
    load_table,
    load_transactions,
)


# --- shape --------------------------------------------------------------------------


def test_row_counts_match_the_shipped_dataset(data: Datasets) -> None:
    assert data.row_counts == {
        "customers": 1_000,
        "products": 500,
        "transactions": 20_000,
        "returns": 5_048,
    }


@pytest.mark.parametrize("table", sorted(sch.TABLES))
def test_columns_match_the_declared_schema(data: Datasets, table: str) -> None:
    frame = data.as_dict()[table]
    assert list(frame.columns) == list(sch.TABLES[table].canonical_columns)


@pytest.mark.parametrize("table", sorted(sch.TABLES))
def test_raw_headers_are_available_when_normalisation_is_off(table: str) -> None:
    frame = load_table(table, normalize_columns=False)
    assert list(frame.columns) == list(sch.TABLES[table].columns)


@pytest.mark.parametrize("table", sorted(sch.TABLES))
def test_no_missing_values_anywhere(data: Datasets, table: str) -> None:
    frame = data.as_dict()[table]
    assert not frame.isna().any().any()


# --- the zero-padded Order ID trap ---------------------------------------------------


def test_order_id_stays_a_zero_padded_string(data: Datasets) -> None:
    order_ids = data.transactions["order_id"]
    assert pd.api.types.is_string_dtype(order_ids), (
        f"order_id came back as {order_ids.dtype}; the zero padding has been destroyed"
    )
    assert (order_ids.str.len() == sch.ORDER_ID_WIDTH).all()
    assert "000001" in set(order_ids)
    assert order_ids.str.fullmatch(r"\d{6}").all()


def test_return_order_id_uses_the_same_padded_representation(data: Datasets) -> None:
    """Both sides of the join must agree, or the merge silently produces nothing."""
    returns_ids = data.returns["order_id"]
    assert pd.api.types.is_string_dtype(returns_ids)
    assert (returns_ids.str.len() == sch.ORDER_ID_WIDTH).all()
    assert set(returns_ids) <= set(data.transactions["order_id"])


def test_transaction_return_join_actually_matches(data: Datasets) -> None:
    merged = data.returns.merge(
        data.transactions[["order_id", "sku_id", "quantity"]],
        on=["order_id", "sku_id"],
        how="left",
        validate="1:1",
    )
    assert not merged["quantity"].isna().any()


@pytest.mark.parametrize(
    ("table", "column"),
    [
        ("customers", "customer_id"),
        ("products", "sku_id"),
        ("transactions", "customer_id"),
        ("transactions", "sku_id"),
        ("transactions", "order_id"),
        ("returns", "customer_id"),
        ("returns", "sku_id"),
        ("returns", "order_id"),
    ],
)
def test_every_id_column_is_a_string(data: Datasets, table: str, column: str) -> None:
    assert pd.api.types.is_string_dtype(data.as_dict()[table][column])


# --- dtypes -------------------------------------------------------------------------


def test_date_columns_are_parsed_as_datetimes(data: Datasets) -> None:
    assert pd.api.types.is_datetime64_any_dtype(data.customers["registration_date"])
    assert pd.api.types.is_datetime64_any_dtype(data.transactions["purchase_date"])
    assert pd.api.types.is_datetime64_any_dtype(data.returns["return_date"])


def test_numeric_columns_are_numeric(data: Datasets) -> None:
    numeric = {
        "customers": ["age"],
        "products": ["list_price"],
        "transactions": ["quantity", "selling_price", "discount_pct", "net_order_value"],
        "returns": ["return_quantity"],
    }
    frames = data.as_dict()
    for table, columns in numeric.items():
        for column in columns:
            assert pd.api.types.is_numeric_dtype(frames[table][column]), f"{table}.{column}"


# --- encoding -----------------------------------------------------------------------


def test_non_ascii_city_names_survive_the_read(data: Datasets) -> None:
    """The file is UTF-8; reading it with the Windows default codec would mangle these."""
    cities = set(data.customers["city"])
    assert "Düsseldorf" in cities
    assert "Liège" in cities


# --- the duplicated Gender column ----------------------------------------------------


def test_gender_columns_are_disambiguated(data: Datasets) -> None:
    assert "customer_gender" in data.customers.columns
    assert "product_gender" in data.products.columns
    assert "gender" not in data.customers.columns
    assert "gender" not in data.products.columns


def test_joining_products_onto_transactions_and_customers_has_no_column_collision(
    data: Datasets,
) -> None:
    merged = (
        data.transactions.merge(data.products, on="sku_id", how="left")
        .merge(data.customers, on="customer_id", how="left")
    )
    collisions = [c for c in merged.columns if c.endswith(("_x", "_y"))]
    assert not collisions, f"column collision on join: {collisions}"
    assert {"customer_gender", "product_gender"} <= set(merged.columns)


# --- loader mechanics ---------------------------------------------------------------


def test_individual_loaders_agree_with_load_all(data: Datasets) -> None:
    pd.testing.assert_frame_equal(load_customers(), data.customers)
    pd.testing.assert_frame_equal(load_products(), data.products)
    pd.testing.assert_frame_equal(load_transactions(), data.transactions)
    pd.testing.assert_frame_equal(load_returns(), data.returns)


def test_the_cache_hands_out_copies_not_the_cached_object() -> None:
    """A caller mutating its frame must not corrupt what the next caller receives."""
    first = load_transactions()
    first.loc[0, "order_id"] = "TAMPERED"
    second = load_transactions()
    assert second.loc[0, "order_id"] != "TAMPERED"


def test_cache_can_be_bypassed_and_cleared() -> None:
    uncached = load_products(use_cache=False)
    clear_cache()
    pd.testing.assert_frame_equal(uncached, load_products())


def test_unknown_table_is_rejected() -> None:
    with pytest.raises(KeyError, match="Unknown table"):
        load_table("orders")


def test_a_column_mismatch_raises_schema_error(tmp_path, monkeypatch) -> None:
    """A renamed or dropped column must fail loudly at load time, not later as a KeyError."""
    broken = tmp_path / "Broken.csv"
    broken.write_text("SKU ID,Category,Subcategory,Brand,Gender\nP0001,Apparel,T,B,Men\n", encoding="utf-8")
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("PRODUCT_FILE", "Broken.csv")
    from src.config.settings import get_settings

    try:
        clear_cache()
        with pytest.raises(SchemaError) as excinfo:
            load_products(settings=get_settings(refresh=True))
        assert "Missing column(s): ['Price']" in str(excinfo.value)
    finally:
        monkeypatch.undo()
        get_settings(refresh=True)
        clear_cache()


def test_datasets_is_iterable_and_convertible(data: Datasets) -> None:
    names = [name for name, _ in data]
    assert names == ["customers", "products", "transactions", "returns"]
    assert set(data.as_dict()) == set(names)
