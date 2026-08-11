"""Tests for the customer feature layer.

Three kinds of test here, in descending order of importance:

1. **Leakage proofs.** Building features at date T from the full dataset must give byte-identical
   results to building them at T from a dataset physically truncated at T. If any feature could
   see the future, those two builds would differ. This is the strongest available statement that
   the as-of discipline holds, and it is checked for transactions and for returns separately --
   returns being the subtle case, since a return is a later event than its purchase.
2. **Hand-computed values.** A purpose-built synthetic dataset whose gaps, windows, revenues and
   month counts can be worked out on paper, so the arithmetic is pinned rather than merely
   self-consistent.
3. **The contract.** Exactly one row per Customer ID, always, including customers with no
   history at the as-of date.

Nothing here touches the real CSV files except the handful of tests explicitly marked as using
them.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.csv_loader import Datasets
from src.features import build_customer_features, resolve_as_of_date
from src.features.context import build_context
from src.features.params import FeatureParams

AS_OF = pd.Timestamp("2025-06-30")


# --------------------------------------------------------------------------------------
# a synthetic dataset built for arithmetic that can be checked on paper
#
# Every customer exists to exercise one behaviour:
#   CUST0001  clockwork      6 orders exactly 30 days apart, EUR 100 each
#   CUST0002  brand new      1 order, 10 days before the as-of date
#   CUST0003  seasonal       mid-January every year, three years running
#   CUST0004  declining      spent well last quarter, barely this one
#   CUST0005  dormant        two orders in early 2023 and nothing since
#   CUST0006  not yet a customer -- registers and buys only AFTER the as-of date
#   CUST0007  returner       one order, one return before the as-of date, one after
# --------------------------------------------------------------------------------------

_CUSTOMERS = [
    # id,        age, gender,   city,        country,       channel,        registration
    ("CUST0001", 30, "Female", "Berlin", "Germany", "Referral", "2025-01-01"),
    ("CUST0002", 41, "Male", "Amsterdam", "Netherlands", "Email", "2025-06-20"),
    ("CUST0003", 35, "Female", "Vienna", "Austria", "Instagram", "2023-01-15"),
    ("CUST0004", 52, "Male", "Brussels", "Belgium", "Paid Search", "2025-01-10"),
    ("CUST0005", 27, "Female", "Hamburg", "Germany", "Facebook", "2023-01-10"),
    ("CUST0006", 22, "Male", "Utrecht", "Netherlands", "Direct", "2025-08-01"),
    ("CUST0007", 33, "Female", "Graz", "Austria", "Organic Search", "2025-05-01"),
]

_PRODUCTS = [
    # sku,     category,      subcategory, brand,        gender,   price
    ("P0001", "Apparel", "T-Shirts", "UrbanEdge", "Men", 50.00),
    ("P0002", "Footwear", "Sneakers", "NovaWear", "Women", 100.00),
    ("P0003", "Accessories", "Bags", "LuxeLine", "Unisex", 200.00),
    ("P0004", "Outerwear", "Parkas", "ModeStreet", "Women", 300.00),
]

_TRANSACTIONS = [
    # customer,  order,    sku,     date,         qty, price, disc, coupon, payment
    # -- clockwork: 30-day cadence, EUR 100 per order --
    ("CUST0001", "000001", "P0001", "2025-01-01", 2, 50.00, 0, "No", "PayPal"),
    ("CUST0001", "000002", "P0001", "2025-01-31", 2, 50.00, 0, "No", "PayPal"),
    ("CUST0001", "000003", "P0001", "2025-03-02", 2, 50.00, 0, "No", "PayPal"),
    ("CUST0001", "000004", "P0001", "2025-04-01", 2, 50.00, 0, "No", "PayPal"),
    ("CUST0001", "000005", "P0001", "2025-05-01", 2, 50.00, 0, "No", "PayPal"),
    ("CUST0001", "000006", "P0001", "2025-05-31", 2, 50.00, 0, "No", "PayPal"),
    # -- brand new --
    ("CUST0002", "000010", "P0002", "2025-06-20", 1, 100.00, 0, "No", "Credit Card"),
    # -- seasonal: mid-January, three years --
    ("CUST0003", "000020", "P0004", "2023-01-15", 1, 300.00, 0, "No", "PayPal"),
    ("CUST0003", "000021", "P0004", "2024-01-15", 1, 300.00, 0, "No", "PayPal"),
    ("CUST0003", "000022", "P0004", "2025-01-15", 1, 300.00, 0, "No", "PayPal"),
    # -- declining: EUR 300 in the previous window, EUR 100 in the recent one --
    ("CUST0004", "000030", "P0003", "2025-01-10", 1, 200.00, 0, "No", "Debit Card"),
    ("CUST0004", "000031", "P0001", "2025-02-10", 2, 50.00, 0, "No", "Debit Card"),
    ("CUST0004", "000032", "P0002", "2025-05-15", 1, 100.00, 0, "No", "Debit Card"),
    # -- dormant: early 2023 only --
    ("CUST0005", "000040", "P0001", "2023-01-10", 1, 50.00, 0, "No", "PayPal"),
    ("CUST0005", "000041", "P0002", "2023-02-10", 1, 100.00, 0, "No", "PayPal"),
    # -- not yet a customer at the as-of date --
    ("CUST0006", "000050", "P0001", "2025-08-01", 1, 50.00, 0, "No", "PayPal"),
    ("CUST0006", "000051", "P0002", "2025-09-01", 1, 100.00, 0, "No", "PayPal"),
    # -- returner: one order, two lines, 7 units total --
    ("CUST0007", "000070", "P0001", "2025-05-01", 4, 50.00, 0, "No", "Credit Card"),
    ("CUST0007", "000070", "P0002", "2025-05-01", 3, 100.00, 0, "No", "Credit Card"),
    # -- a future order for the clockwork customer, to prove transaction clipping --
    ("CUST0001", "000080", "P0003", "2025-08-15", 5, 200.00, 0, "No", "PayPal"),
]

_RETURNS = [
    # customer,  order,    sku,     return date,  qty
    ("CUST0007", "000070", "P0001", "2025-05-10", 1),   # before the as-of date: counts
    ("CUST0007", "000070", "P0002", "2025-07-15", 2),   # AFTER the as-of date: must not count
]


def _frame(rows, columns, dates=(), strings=(), ints=()):
    frame = pd.DataFrame(rows, columns=columns)
    for column in dates:
        frame[column] = pd.to_datetime(frame[column])
    for column in strings:
        frame[column] = frame[column].astype("string")
    for column in ints:
        frame[column] = frame[column].astype("int16")
    return frame


def make_datasets() -> Datasets:
    """The synthetic dataset, including rows that postdate the as-of date."""
    customers = _frame(
        _CUSTOMERS,
        ["customer_id", "age", "customer_gender", "city", "country",
         "acquisition_channel", "registration_date"],
        dates=["registration_date"],
        strings=["customer_id", "customer_gender", "city", "country", "acquisition_channel"],
        ints=["age"],
    )
    products = _frame(
        _PRODUCTS,
        ["sku_id", "category", "subcategory", "brand", "product_gender", "list_price"],
        strings=["sku_id", "category", "subcategory", "brand", "product_gender"],
    )
    transactions = _frame(
        _TRANSACTIONS,
        ["customer_id", "order_id", "sku_id", "purchase_date", "quantity", "selling_price",
         "discount_pct", "coupon_used", "payment_method"],
        dates=["purchase_date"],
        strings=["customer_id", "order_id", "sku_id", "coupon_used", "payment_method"],
        ints=["quantity", "discount_pct"],
    )
    transactions["net_order_value"] = (
        transactions["quantity"] * transactions["selling_price"]
        * (1 - transactions["discount_pct"] / 100.0)
    ).round(2)
    transactions = transactions[[
        "customer_id", "order_id", "sku_id", "purchase_date", "quantity", "selling_price",
        "discount_pct", "coupon_used", "net_order_value", "payment_method",
    ]]
    returns = _frame(
        _RETURNS,
        ["customer_id", "order_id", "sku_id", "return_date", "return_quantity"],
        dates=["return_date"],
        strings=["customer_id", "order_id", "sku_id"],
        ints=["return_quantity"],
    )
    return Datasets(
        customers=customers, products=products, transactions=transactions, returns=returns
    )


@pytest.fixture
def synthetic() -> Datasets:
    return make_datasets()


@pytest.fixture
def built(synthetic: Datasets) -> pd.DataFrame:
    """Features for the synthetic dataset, indexed by customer for easy assertions."""
    return build_customer_features(synthetic, as_of_date=AS_OF).features.set_index("customer_id")


# ======================================================================================
# 1. LEAKAGE PROOFS
# ======================================================================================


def _truncate(data: Datasets, as_of: pd.Timestamp) -> Datasets:
    """Physically remove everything after ``as_of`` from the source frames."""
    return Datasets(
        customers=data.customers,
        products=data.products,
        transactions=data.transactions[data.transactions["purchase_date"].le(as_of)].copy(),
        returns=data.returns[data.returns["return_date"].le(as_of)].copy(),
    )


def test_features_are_identical_whether_or_not_future_data_exists(synthetic: Datasets) -> None:
    """The central leakage proof.

    If any feature reached past the as-of date, deleting the future rows would change it.
    """
    with_future = build_customer_features(synthetic, as_of_date=AS_OF).features
    without_future = build_customer_features(_truncate(synthetic, AS_OF), as_of_date=AS_OF).features
    pd.testing.assert_frame_equal(with_future, without_future)


def test_future_transactions_do_not_leak(synthetic: Datasets) -> None:
    """CUST0001 has a EUR 1,000 order six weeks after the as-of date."""
    features = build_customer_features(synthetic, as_of_date=AS_OF).features.set_index(
        "customer_id"
    )
    assert features.loc["CUST0001", "total_orders"] == 6, "the 2025-08-15 order leaked in"
    assert features.loc["CUST0001", "lifetime_revenue"] == pytest.approx(600.0)
    assert features.loc["CUST0001", "last_purchase_date"] == pd.Timestamp("2025-05-31")


def test_future_returns_do_not_leak(synthetic: Datasets) -> None:
    """The subtle half: a return dated after the as-of date has not happened yet.

    CUST0007 bought 7 units and will eventually return 3, but only 1 unit had been returned by
    the as-of date. The feature must say 1/7, not 3/7.
    """
    features = build_customer_features(synthetic, as_of_date=AS_OF).features.set_index(
        "customer_id"
    )
    assert features.loc["CUST0007", "returned_units"] == 1, "the 2025-07-15 return leaked in"
    assert features.loc["CUST0007", "total_units"] == 7
    assert features.loc["CUST0007", "return_rate"] == pytest.approx(1 / 7)
    assert features.loc["CUST0007", "returned_orders"] == 1


def test_the_leakage_test_is_not_vacuous(synthetic: Datasets) -> None:
    """Guard against the proof above passing because nothing depends on the as-of date."""
    early = build_customer_features(synthetic, as_of_date="2025-03-31").features.set_index(
        "customer_id"
    )
    late = build_customer_features(synthetic, as_of_date=AS_OF).features.set_index("customer_id")
    assert early.loc["CUST0001", "total_orders"] == 3
    assert late.loc["CUST0001", "total_orders"] == 6


def test_a_later_as_of_date_reveals_the_withheld_return(synthetic: Datasets) -> None:
    """Moving the as-of date past the second return should surface it."""
    later = build_customer_features(synthetic, as_of_date="2025-07-31").features.set_index(
        "customer_id"
    )
    assert later.loc["CUST0007", "returned_units"] == 3
    assert later.loc["CUST0007", "return_rate"] == pytest.approx(3 / 7)


def test_context_reports_what_it_withheld(synthetic: Datasets) -> None:
    context = build_context(synthetic, as_of_date=AS_OF)
    assert len(context.lines) == len(_TRANSACTIONS) - 3      # one future order + two CUST0006
    assert len(context.returns) == 1
    assert context.lines["purchase_date"].max() <= AS_OF
    assert context.returns["return_date"].max() <= AS_OF


def test_returns_are_restricted_to_in_window_order_lines(synthetic: Datasets) -> None:
    """A return can only count against an order line the context can actually see."""
    context = build_context(synthetic, as_of_date=AS_OF)
    line_keys = set(map(tuple, context.lines[["order_id", "sku_id"]].to_numpy()))
    return_keys = set(map(tuple, context.returns[["order_id", "sku_id"]].to_numpy()))
    assert return_keys <= line_keys


# ======================================================================================
# 2. THE ONE-ROW-PER-CUSTOMER CONTRACT
# ======================================================================================


def test_exactly_one_row_per_customer(built: pd.DataFrame, synthetic: Datasets) -> None:
    assert len(built) == len(synthetic.customers)
    assert not built.index.duplicated().any()
    assert set(built.index) == set(synthetic.customers["customer_id"])


def test_customers_with_no_history_are_kept_and_flagged(built: pd.DataFrame) -> None:
    """CUST0006 registers and buys only after the as-of date, so did not exist yet."""
    row = built.loc["CUST0006"]
    assert row["has_purchase_history"] is False or row["has_purchase_history"] == False  # noqa: E712
    assert row["registered_at_as_of"] == False  # noqa: E712
    assert row["total_orders"] == 0
    assert row["lifetime_revenue"] == 0.0
    assert pd.isna(row["recency_days"])
    assert row["behavioral_segment"] == "No History"


def test_the_row_count_is_stable_across_as_of_dates(synthetic: Datasets) -> None:
    """A shrinking cohort must not shrink the table, or row counts stop being comparable."""
    for as_of in ("2023-06-30", "2024-06-30", "2025-06-30"):
        result = build_customer_features(synthetic, as_of_date=as_of)
        assert result.customer_count == len(synthetic.customers)


def test_build_is_deterministic(synthetic: Datasets) -> None:
    first = build_customer_features(synthetic, as_of_date=AS_OF).features
    second = build_customer_features(synthetic, as_of_date=AS_OF).features
    pd.testing.assert_frame_equal(first, second)


# ======================================================================================
# 3. RFM ARITHMETIC
# ======================================================================================


def test_recency_and_totals(built: pd.DataFrame) -> None:
    row = built.loc["CUST0001"]
    assert row["recency_days"] == 30           # 2025-05-31 -> 2025-06-30
    assert row["total_orders"] == 6
    assert row["total_units"] == 12            # 6 orders x 2 units
    assert row["lifetime_revenue"] == pytest.approx(600.0)
    assert row["average_order_value"] == pytest.approx(100.0)
    assert row["average_item_value"] == pytest.approx(50.0)


def test_rolling_windows_are_half_open(built: pd.DataFrame) -> None:
    """Windows are ``(as_of - days, as_of]``, so an order exactly N days back is outside.

    CUST0001's last order is exactly 30 days before the as-of date, which places it on the 31st
    day back and therefore outside the 30-day window.
    """
    row = built.loc["CUST0001"]
    assert row["orders_30d"] == 0
    assert row["orders_90d"] == 2               # 2025-05-01 and 2025-05-31
    assert row["orders_180d"] == 5              # everything except 2025-01-01
    assert row["orders_365d"] == 6
    assert row["revenue_90d"] == pytest.approx(200.0)
    assert row["revenue_365d"] == pytest.approx(600.0)


def test_multi_line_order_counts_as_one_order(built: pd.DataFrame) -> None:
    """CUST0007's single basket holds two SKUs: one order, two lines, seven units."""
    row = built.loc["CUST0007"]
    assert row["total_orders"] == 1
    assert row["total_lines"] == 2
    assert row["total_units"] == 7
    assert row["lifetime_revenue"] == pytest.approx(500.0)   # 4x50 + 3x100


# ======================================================================================
# 4. GAP ARITHMETIC
# ======================================================================================


def test_gaps_for_a_clockwork_customer(built: pd.DataFrame) -> None:
    row = built.loc["CUST0001"]
    assert row["observed_gaps"] == 5                    # 6 orders -> 5 gaps
    assert row["average_purchase_gap"] == pytest.approx(30.0)
    assert row["median_purchase_gap"] == pytest.approx(30.0)
    assert row["maximum_purchase_gap"] == pytest.approx(30.0)
    assert row["purchase_gap_std"] == pytest.approx(0.0)
    assert row["current_purchase_gap"] == 30
    assert row["purchase_gap_ratio"] == pytest.approx(1.0)
    assert row["has_measurable_cadence"] == True         # noqa: E712
    assert row["purchase_regularity"] == pytest.approx(1.0)


def test_a_single_order_customer_has_no_measurable_cadence(built: pd.DataFrame) -> None:
    row = built.loc["CUST0002"]
    assert row["observed_gaps"] == 0
    assert pd.isna(row["average_purchase_gap"])
    assert pd.isna(row["median_purchase_gap"])
    assert row["has_measurable_cadence"] == False        # noqa: E712
    # Falls back to the configured default rather than dividing by nothing.
    assert row["expected_purchase_interval_days"] == 90.0
    assert row["current_purchase_gap"] == 10
    assert row["purchase_gap_ratio"] == pytest.approx(10 / 90)


def test_gap_ratio_flags_an_unusually_long_silence(built: pd.DataFrame) -> None:
    """CUST0005: gaps of 31 days, then silent since 2023-02-10."""
    row = built.loc["CUST0005"]
    assert row["median_purchase_gap"] == pytest.approx(31.0)
    expected_recency = (AS_OF - pd.Timestamp("2023-02-10")).days
    assert row["current_purchase_gap"] == expected_recency
    assert row["purchase_gap_ratio"] == pytest.approx(expected_recency / 31.0)
    assert row["purchase_gap_ratio"] > 15


# ======================================================================================
# 5. TREND ARITHMETIC
# ======================================================================================


def test_revenue_growth_compares_adjacent_windows(built: pd.DataFrame) -> None:
    """CUST0001: EUR 200 in the recent 90 days versus EUR 300 in the 90 before that."""
    row = built.loc["CUST0001"]
    assert row["revenue_recent_window"] == pytest.approx(200.0)
    assert row["revenue_previous_window"] == pytest.approx(300.0)
    assert row["revenue_growth"] == pytest.approx(-1 / 3, abs=1e-4)
    assert row["spend_decline_pct"] == pytest.approx(1 / 3, abs=1e-4)
    assert row["orders_recent_window"] == 2
    assert row["orders_previous_window"] == 3
    assert row["order_frequency_growth"] == pytest.approx(-1 / 3, abs=1e-4)


def test_declining_customer_shows_a_steep_fall(built: pd.DataFrame) -> None:
    """CUST0004: EUR 300 in the previous window, EUR 100 in the recent one."""
    row = built.loc["CUST0004"]
    assert row["revenue_previous_window"] == pytest.approx(300.0)
    assert row["revenue_recent_window"] == pytest.approx(100.0)
    assert row["revenue_growth"] == pytest.approx(-2 / 3, abs=1e-4)
    assert row["is_declining_buyer"] == True             # noqa: E712


def test_growth_is_null_not_infinite_without_a_baseline(built: pd.DataFrame) -> None:
    """CUST0002 spent nothing in the previous window, so growth is undefined."""
    row = built.loc["CUST0002"]
    assert row["revenue_previous_window"] == pytest.approx(0.0)
    assert pd.isna(row["revenue_growth"])
    assert not np.isinf(pd.to_numeric([row["revenue_growth"]], errors="coerce")).any()


# ======================================================================================
# 6. LIFECYCLE ARITHMETIC
# ======================================================================================


def test_tenure_and_month_counts(built: pd.DataFrame) -> None:
    """CUST0001 bought in Jan, Mar, Apr and May -- four active months of six observable."""
    row = built.loc["CUST0001"]
    assert row["customer_tenure_days"] == 180           # 2025-01-01 -> 2025-06-30
    assert row["first_purchase_date"] == pd.Timestamp("2025-01-01")
    assert row["last_purchase_date"] == pd.Timestamp("2025-05-31")
    assert row["active_months"] == 4
    assert row["observable_months"] == 6                # January through June inclusive
    assert row["inactive_months"] == 2                  # February and June
    assert row["is_repeat_customer"] == True            # noqa: E712
    assert row["is_one_time_buyer"] == False            # noqa: E712


def test_one_time_buyer_flags(built: pd.DataFrame) -> None:
    row = built.loc["CUST0002"]
    assert row["is_one_time_buyer"] == True             # noqa: E712
    assert row["is_repeat_customer"] == False           # noqa: E712


# ======================================================================================
# 7. PRODUCT AFFINITY
# ======================================================================================


def test_preferred_dimensions(built: pd.DataFrame) -> None:
    row = built.loc["CUST0001"]
    assert row["preferred_category"] == "Apparel"
    assert row["preferred_subcategory"] == "T-Shirts"
    assert row["preferred_brand"] == "UrbanEdge"
    assert row["category_count"] == 1
    assert row["sku_count"] == 1
    assert row["preferred_category_share"] == pytest.approx(1.0)
    # A single-category customer has no diversity at all.
    assert row["category_diversity"] == pytest.approx(0.0)


def test_diversity_rises_with_breadth(built: pd.DataFrame) -> None:
    """CUST0004 bought across three categories, CUST0001 across one."""
    assert built.loc["CUST0004", "category_count"] == 3
    assert built.loc["CUST0004", "category_diversity"] > built.loc["CUST0001", "category_diversity"]


def test_days_since_preferred_category_purchase(built: pd.DataFrame) -> None:
    row = built.loc["CUST0001"]
    assert row["days_since_preferred_category_purchase"] == 30
    assert row["most_recent_category"] == "Apparel"
    assert row["preferred_category_is_latest"] == True   # noqa: E712


def test_preferred_dimension_ties_break_deterministically(synthetic: Datasets) -> None:
    """CUST0007 bought 4 units of Apparel and 3 of Footwear, so Apparel wins on units."""
    features = build_customer_features(synthetic, as_of_date=AS_OF).features.set_index(
        "customer_id"
    )
    assert features.loc["CUST0007", "preferred_category"] == "Apparel"


# ======================================================================================
# 8. DISCOUNT BEHAVIOUR
# ======================================================================================


def test_full_price_customer(built: pd.DataFrame) -> None:
    row = built.loc["CUST0001"]
    assert row["average_discount"] == pytest.approx(0.0)
    assert row["discount_order_rate"] == pytest.approx(0.0)
    assert row["coupon_usage_rate"] == pytest.approx(0.0)
    assert row["full_price_order_rate"] == pytest.approx(1.0)
    assert row["discount_dependency_score"] == pytest.approx(0.0)
    assert row["is_full_price_buyer"] == True            # noqa: E712
    assert row["is_discount_driven"] == False            # noqa: E712


def test_discount_features_respond_to_discounts(synthetic: Datasets) -> None:
    """Half of a customer's orders discounted at 40% should move every discount feature."""
    data = synthetic
    transactions = data.transactions.copy()
    target = transactions["customer_id"].eq("CUST0001") & transactions["order_id"].isin(
        ["000001", "000002", "000003"]
    )
    transactions.loc[target, "discount_pct"] = 40
    transactions.loc[target, "coupon_used"] = "Yes"
    transactions["net_order_value"] = (
        transactions["quantity"] * transactions["selling_price"]
        * (1 - transactions["discount_pct"] / 100.0)
    ).round(2)
    modified = Datasets(data.customers, data.products, transactions, data.returns)

    row = build_customer_features(modified, as_of_date=AS_OF).features.set_index(
        "customer_id"
    ).loc["CUST0001"]
    assert row["discount_order_rate"] == pytest.approx(0.5)
    assert row["coupon_usage_rate"] == pytest.approx(0.5)
    assert row["full_price_order_rate"] == pytest.approx(0.5)
    assert row["average_discount"] == pytest.approx(20.0)          # mean of 40,40,40,0,0,0
    assert row["average_discount_when_discounted"] == pytest.approx(40.0)
    assert row["is_full_price_buyer"] == False                     # noqa: E712


# ======================================================================================
# 9. RETURNS
# ======================================================================================


def test_customer_with_no_returns(built: pd.DataFrame) -> None:
    row = built.loc["CUST0001"]
    assert row["returned_units"] == 0
    assert row["returned_orders"] == 0
    assert row["return_rate"] == pytest.approx(0.0)
    assert row["is_serial_returner"] == False            # noqa: E712


def test_return_rate_uses_units_purchased_in_the_same_window(built: pd.DataFrame) -> None:
    row = built.loc["CUST0007"]
    assert row["return_rate"] == pytest.approx(1 / 7)
    assert row["return_frequency"] == pytest.approx(1.0)   # its only order saw a return
    assert row["days_since_last_return"] == (AS_OF - pd.Timestamp("2025-05-10")).days


# ======================================================================================
# 10. SEASONALITY -- and the protection it provides
# ======================================================================================


def test_seasonal_customer_is_detected(built: pd.DataFrame) -> None:
    """CUST0003 buys mid-January three years running."""
    row = built.loc["CUST0003"]
    assert row["purchase_years_spanned"] == 3
    assert row["preferred_purchase_month"] == 1
    assert row["seasonal_customer_score"] > 0.9
    assert row["is_seasonal_buyer"] == True              # noqa: E712
    # Mid-January is roughly half a year from the end of June.
    assert row["in_preferred_season"] == False           # noqa: E712
    assert row["days_from_preferred_season"] > 150


def test_seasonality_is_not_scored_without_enough_evidence(built: pd.DataFrame) -> None:
    """One order makes any customer look perfectly seasonal, so the score is withheld."""
    row = built.loc["CUST0002"]
    assert pd.isna(row["seasonal_customer_score"])
    assert row["is_seasonal_buyer"] == False             # noqa: E712


def test_a_seasonal_customer_out_of_season_is_not_called_dormant(built: pd.DataFrame) -> None:
    """The requirement, stated as a test.

    CUST0003 has been silent for over five months -- far beyond twice their measured cadence --
    yet must not be filed as dormant, because their buying window is half a year away.
    """
    row = built.loc["CUST0003"]
    assert row["recency_days"] > 150
    assert row["purchase_gap_ratio"] < 1  # a yearly cadence, so the ratio itself stays low
    assert row["seasonally_explained_inactivity"] == True   # noqa: E712
    assert row["is_dormant_buyer"] == False                 # noqa: E712
    assert row["behavioral_segment"] == "Seasonal Buyer"


def test_a_seasonal_customer_who_missed_a_whole_season_is_dormant(synthetic: Datasets) -> None:
    """The other half of the rule, and the reason the shield is not unconditional.

    Judged from mid-2026, CUST0003 has skipped the January they never miss. Being "out of
    season" cannot excuse that: they have been through a full cycle without buying.
    """
    row = build_customer_features(synthetic, as_of_date="2026-06-30").features.set_index(
        "customer_id"
    ).loc["CUST0003"]
    assert row["is_seasonal_buyer"] == True                  # noqa: E712
    assert row["annual_cycles_missed"] >= 1
    assert row["missed_full_season"] == True                 # noqa: E712
    assert row["seasonally_explained_inactivity"] == False   # noqa: E712
    assert row["is_dormant_buyer"] == True                   # noqa: E712
    assert row["behavioral_segment"] == "Dormant Buyer"


def test_seasonal_concentration_of_a_spread_out_customer_is_low(built: pd.DataFrame) -> None:
    """CUST0001 bought across four different months, so is not concentrated."""
    assert built.loc["CUST0001", "seasonal_purchase_concentration"] < 0.5
    assert built.loc["CUST0003", "seasonal_purchase_concentration"] == pytest.approx(1.0)


def test_in_season_customer_is_flagged(synthetic: Datasets) -> None:
    """Judged in mid-January, the January buyer is in their season."""
    row = build_customer_features(synthetic, as_of_date="2025-01-20").features.set_index(
        "customer_id"
    ).loc["CUST0003"]
    assert row["in_preferred_season"] == True            # noqa: E712
    assert row["seasonally_explained_inactivity"] == False  # noqa: E712


# ======================================================================================
# 11. VALUE AND SEGMENTS
# ======================================================================================


def test_annualised_revenue(built: pd.DataFrame) -> None:
    """CUST0001: EUR 600 over 180 days annualises to about EUR 1,218."""
    row = built.loc["CUST0001"]
    assert row["annualized_revenue"] == pytest.approx(600 * 365.25 / 180, abs=0.5)
    assert row["annualisation_floored"] == False          # noqa: E712


def test_annualisation_is_floored_for_very_new_customers(built: pd.DataFrame) -> None:
    """CUST0002 is 10 days old; dividing by 10 days would annualise EUR 100 to EUR 3,652."""
    row = built.loc["CUST0002"]
    assert row["annualisation_floored"] == True           # noqa: E712
    # Floored at 30 days, so about EUR 1,218 rather than EUR 3,652.
    assert row["annualized_revenue"] == pytest.approx(100 * 365.25 / 30, abs=0.5)
    assert row["annualized_revenue"] < 1500


def test_value_segments_are_relative_to_the_cohort(built: pd.DataFrame) -> None:
    segments = set(built["customer_value_segment"])
    assert segments <= {"High Value", "Medium Value", "Low Value", "No History"}
    # The customer with no history cannot be valued.
    assert built.loc["CUST0006", "customer_value_segment"] == "No History"
    # The biggest spender must land in the top band.
    top = built[built["has_purchase_history"]]["lifetime_revenue"].idxmax()
    assert built.loc[top, "customer_value_segment"] == "High Value"


def test_new_buyer_takes_priority(built: pd.DataFrame) -> None:
    """Ten days of history is too little to call anyone declining or dormant."""
    assert built.loc["CUST0002", "is_new_buyer"] == True     # noqa: E712
    assert built.loc["CUST0002", "behavioral_segment"] == "New Buyer"


def test_dormant_customer_is_labelled(built: pd.DataFrame) -> None:
    row = built.loc["CUST0005"]
    assert row["is_dormant_buyer"] == True                   # noqa: E712
    assert row["behavioral_segment"] == "Dormant Buyer"
    assert row["lifecycle_stage"] in {"Dormant", "Lost"}


def test_every_customer_gets_a_segment_and_a_reason(built: pd.DataFrame) -> None:
    assert built["behavioral_segment"].notna().all()
    assert built["segment_reason"].str.len().gt(0).all()


def test_segment_flags_may_overlap(built: pd.DataFrame) -> None:
    """The flags are a multi-label view; only the resolved label is exclusive."""
    flags = ["is_new_buyer", "is_frequent_buyer", "is_declining_buyer",
             "is_dormant_buyer", "is_occasional_buyer", "is_seasonal_buyer"]
    for flag in flags:
        assert built[flag].dtype == bool
    assert built["behavioral_segment"].nunique() >= 4


# ======================================================================================
# 12. AS-OF RESOLUTION AND PARAMETERS
# ======================================================================================


def test_as_of_defaults_to_the_last_purchase_date(synthetic: Datasets) -> None:
    """Derived from the data, not the wall clock, so a build stays reproducible."""
    assert resolve_as_of_date(synthetic) == pd.Timestamp("2025-09-01")
    result = build_customer_features(synthetic)
    assert result.as_of_date == pd.Timestamp("2025-09-01")


def test_as_of_accepts_strings_dates_and_timestamps(synthetic: Datasets) -> None:
    import datetime as dt

    expected = pd.Timestamp("2025-06-30")
    for value in ("2025-06-30", dt.date(2025, 6, 30), pd.Timestamp("2025-06-30 13:45")):
        assert resolve_as_of_date(synthetic, value) == expected


def test_an_unparseable_as_of_date_is_rejected(synthetic: Datasets) -> None:
    with pytest.raises(Exception):
        resolve_as_of_date(synthetic, "the thirty-first of Octember")


def test_params_are_honoured(synthetic: Datasets) -> None:
    """Changing the trend window must change the trend features."""
    default = build_customer_features(synthetic, as_of_date=AS_OF).features.set_index(
        "customer_id"
    )
    wider = build_customer_features(
        synthetic, as_of_date=AS_OF, params=FeatureParams(trend_window_days=30)
    ).features.set_index("customer_id")
    assert default.loc["CUST0001", "trend_window_days"] == 90
    assert wider.loc["CUST0001", "trend_window_days"] == 30
    assert default.loc["CUST0001", "revenue_recent_window"] != wider.loc[
        "CUST0001", "revenue_recent_window"
    ]


def test_invalid_params_are_rejected() -> None:
    with pytest.raises(ValueError, match="strictly increasing|medium"):
        FeatureParams(high_value_quantile=0.4, medium_value_quantile=0.6).validate()
    with pytest.raises(ValueError, match="positive"):
        FeatureParams(trend_window_days=0).validate()


def test_build_result_reports_its_own_shape(synthetic: Datasets) -> None:
    result = build_customer_features(synthetic, as_of_date=AS_OF)
    assert result.feature_count == len(result.feature_names)
    assert "customer_id" not in result.feature_names
    summary = result.summary()
    assert summary["customers"] == len(synthetic.customers)
    assert summary["as_of_date"] == "2025-06-30"
    assert sum(summary["groups"].values()) == result.feature_count


def test_issues_are_reported_for_thin_history(synthetic: Datasets) -> None:
    result = build_customer_features(synthetic, as_of_date=AS_OF)
    joined = " ".join(result.issues)
    assert "no orders on or before" in joined
    assert "one order" in joined


# ======================================================================================
# 13. AGAINST THE REAL CSV FILES
# ======================================================================================


def test_real_data_yields_one_row_per_customer(data) -> None:
    result = build_customer_features(data, as_of_date="2025-12-31")
    assert result.customer_count == 1000
    assert result.features["customer_id"].is_unique


def test_real_data_totals_match_the_source(data) -> None:
    """Aggregating the features back up must reproduce the source figures exactly."""
    features = build_customer_features(data, as_of_date="2025-12-31").features
    assert features["total_orders"].sum() == data.transactions["order_id"].nunique()
    assert features["total_units"].sum() == data.transactions["quantity"].sum()
    assert features["lifetime_revenue"].sum() == pytest.approx(
        data.transactions["net_order_value"].sum(), abs=0.5
    )


def test_real_data_withholds_the_104_future_returns(data) -> None:
    """The dataset's known quirk, as a feature-layer assertion."""
    features = build_customer_features(data, as_of_date="2025-12-31").features
    clipped = data.returns[data.returns["return_date"].le(pd.Timestamp("2025-12-31"))]
    assert features["returned_units"].sum() == clipped["return_quantity"].sum()
    assert features["returned_units"].sum() < data.returns["return_quantity"].sum()


def test_real_data_has_no_leakage(data) -> None:
    """The leakage proof, repeated on the real 20,000-row dataset at a mid-history date."""
    as_of = pd.Timestamp("2024-06-30")
    with_future = build_customer_features(data, as_of_date=as_of).features
    without_future = build_customer_features(_truncate(data, as_of), as_of_date=as_of).features
    pd.testing.assert_frame_equal(with_future, without_future)
