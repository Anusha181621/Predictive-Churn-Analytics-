"""Declarative schema for the four source CSV files.

Single source of truth for column names, dtypes, date columns and allowed categorical values.
The loader, the validator, the inspection script and the tests all read these declarations
rather than repeating literals, so a change to the CSV structure is a one-line change here.

Two things in this schema are defensive rather than cosmetic, and both come from inspecting
the real files:

1. ``Order ID`` is a **zero-padded 6-digit string** (``000001`` ... ``006726``). Pandas' type
   inference reads it as ``int64`` and silently destroys the padding, which then breaks any
   string join against ``Return.csv`` and mis-renders every exported order number. All three
   ID columns are therefore pinned to the ``string`` dtype.
2. ``Gender`` exists in **both** ``Customer.csv`` (Female / Male / Other) and ``Product.csv``
   (Men / Women / Unisex) with different vocabularies. Merging product attributes onto
   transactions and then onto customers would collide into ``Gender_x`` / ``Gender_y``, so the
   rename maps below disambiguate them into ``customer_gender`` and ``product_gender``.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

__all__ = [
    "TableSchema",
    "CUSTOMERS",
    "TRANSACTIONS",
    "RETURNS",
    "PRODUCTS",
    "TABLES",
    "ID_DTYPE",
    "EXPECTED_ROW_COUNTS",
    "EXPECTED_CUSTOMER_COUNT",
    "EXPECTED_SKU_COUNT",
    "EXPECTED_ORDER_COUNT",
    "ALLOWED_DISCOUNTS",
    "ORDER_ID_WIDTH",
]

#: Pandas dtype used for every identifier column, so keys never lose zero padding.
ID_DTYPE = "string"

#: ``Order ID`` is zero-padded to this width in the source files.
ORDER_ID_WIDTH = 6

#: The discount column is a percentage drawn from this fixed domain.
ALLOWED_DISCOUNTS = frozenset({0, 5, 10, 15, 20, 25, 30, 40, 50})


@dataclass(frozen=True)
class TableSchema:
    """Everything the loader needs to know about one CSV file."""

    #: Logical table name, also the key used by ``Settings.csv_path``.
    name: str
    #: Human-readable description for reports.
    description: str
    #: Exact CSV header names, in file order.
    columns: tuple[str, ...]
    #: ``{raw header: pandas dtype}`` for columns that must not be inferred.
    dtypes: Mapping[str, str]
    #: Raw headers to parse as dates.
    date_columns: tuple[str, ...]
    #: Raw headers holding numeric measures (used by the profiler and validator).
    numeric_columns: tuple[str, ...]
    #: ``{raw header: canonical snake_case name}``.
    rename_map: Mapping[str, str]
    #: ``{canonical name: allowed values}`` for low-cardinality categoricals.
    allowed_values: Mapping[str, frozenset[str]]

    @property
    def canonical_columns(self) -> tuple[str, ...]:
        """Column names after :attr:`rename_map` is applied, in file order."""
        return tuple(self.rename_map[column] for column in self.columns)

    def canonical(self, raw_column: str) -> str:
        """Return the canonical name of a raw CSV header."""
        return self.rename_map[raw_column]


CUSTOMERS = TableSchema(
    name="customers",
    description="One row per customer. Registration Date equals the first purchase date.",
    columns=(
        "Customer ID",
        "Age",
        "Gender",
        "City",
        "Country",
        "Customer Acquisition Channel",
        "Registration Date",
    ),
    dtypes=MappingProxyType(
        {
            "Customer ID": ID_DTYPE,
            "Age": "int16",
            "Gender": "string",
            "City": "string",
            "Country": "string",
            "Customer Acquisition Channel": "string",
        }
    ),
    date_columns=("Registration Date",),
    numeric_columns=("Age",),
    rename_map=MappingProxyType(
        {
            "Customer ID": "customer_id",
            "Age": "age",
            # Disambiguated from Product.Gender - see the module docstring.
            "Gender": "customer_gender",
            "City": "city",
            "Country": "country",
            "Customer Acquisition Channel": "acquisition_channel",
            "Registration Date": "registration_date",
        }
    ),
    allowed_values=MappingProxyType(
        {
            "customer_gender": frozenset({"Female", "Male", "Other / Prefer not to say"}),
            "country": frozenset({"Germany", "Netherlands", "Austria", "Belgium"}),
            "acquisition_channel": frozenset(
                {
                    "Organic Search",
                    "Paid Search",
                    "Google Ads",
                    "Instagram",
                    "Facebook",
                    "Influencer",
                    "Referral",
                    "Email",
                    "Direct",
                }
            ),
        }
    ),
)

TRANSACTIONS = TableSchema(
    name="transactions",
    description=(
        "One row per order line (one SKU within one order). Customer ID, Purchase Date and "
        "Payment Method are constant across the lines of an order."
    ),
    columns=(
        "Customer ID",
        "Order ID",
        "SKU ID",
        "Purchase Date",
        "Quantity",
        "Selling Price",
        "Discount",
        "Coupon Used",
        "Net Order Value",
        "Payment Method",
    ),
    dtypes=MappingProxyType(
        {
            "Customer ID": ID_DTYPE,
            # Zero-padded; must stay a string. See the module docstring.
            "Order ID": ID_DTYPE,
            "SKU ID": ID_DTYPE,
            "Quantity": "int16",
            "Selling Price": "float64",
            "Discount": "int16",
            "Coupon Used": "string",
            "Net Order Value": "float64",
            "Payment Method": "string",
        }
    ),
    date_columns=("Purchase Date",),
    numeric_columns=("Quantity", "Selling Price", "Discount", "Net Order Value"),
    rename_map=MappingProxyType(
        {
            "Customer ID": "customer_id",
            "Order ID": "order_id",
            "SKU ID": "sku_id",
            "Purchase Date": "purchase_date",
            "Quantity": "quantity",
            "Selling Price": "selling_price",
            "Discount": "discount_pct",
            "Coupon Used": "coupon_used",
            "Net Order Value": "net_order_value",
            "Payment Method": "payment_method",
        }
    ),
    allowed_values=MappingProxyType(
        {
            "coupon_used": frozenset({"Yes", "No"}),
            "payment_method": frozenset(
                {"Credit Card", "Debit Card", "PayPal", "Buy Now Pay Later"}
            ),
        }
    ),
)

RETURNS = TableSchema(
    name="returns",
    description=(
        "One row per returned order line; at most one row per (Order ID, SKU ID). Return "
        "dates for late-December orders can fall in the following January."
    ),
    columns=("Customer ID", "Order ID", "SKU ID", "Return Date", "Return Quantity"),
    dtypes=MappingProxyType(
        {
            "Customer ID": ID_DTYPE,
            "Order ID": ID_DTYPE,
            "SKU ID": ID_DTYPE,
            "Return Quantity": "int16",
        }
    ),
    date_columns=("Return Date",),
    numeric_columns=("Return Quantity",),
    rename_map=MappingProxyType(
        {
            "Customer ID": "customer_id",
            "Order ID": "order_id",
            "SKU ID": "sku_id",
            "Return Date": "return_date",
            "Return Quantity": "return_quantity",
        }
    ),
    allowed_values=MappingProxyType({}),
)

PRODUCTS = TableSchema(
    name="products",
    description="One row per SKU. Price is the base list price; Selling Price drifts from it.",
    columns=("SKU ID", "Category", "Subcategory", "Brand", "Gender", "Price"),
    dtypes=MappingProxyType(
        {
            "SKU ID": ID_DTYPE,
            "Category": "string",
            "Subcategory": "string",
            "Brand": "string",
            "Gender": "string",
            "Price": "float64",
        }
    ),
    date_columns=(),
    numeric_columns=("Price",),
    rename_map=MappingProxyType(
        {
            "SKU ID": "sku_id",
            "Category": "category",
            "Subcategory": "subcategory",
            "Brand": "brand",
            # Disambiguated from Customer.Gender - see the module docstring.
            "Gender": "product_gender",
            "Price": "list_price",
        }
    ),
    allowed_values=MappingProxyType(
        {
            "category": frozenset(
                {"Apparel", "Footwear", "Activewear", "Outerwear", "Accessories"}
            ),
            "brand": frozenset(
                {"UrbanEdge", "ModeStreet", "TrendAura", "NovaWear", "LuxeLine", "ActiveCore"}
            ),
            "product_gender": frozenset({"Men", "Women", "Unisex"}),
        }
    ),
)

#: All table schemas keyed by logical name, in dependency order.
TABLES: Mapping[str, TableSchema] = MappingProxyType(
    {
        CUSTOMERS.name: CUSTOMERS,
        PRODUCTS.name: PRODUCTS,
        TRANSACTIONS.name: TRANSACTIONS,
        RETURNS.name: RETURNS,
    }
)

# --------------------------------------------------------------------------------------
# Expected shape of the shipped dataset.
#
# These are facts about the current data/*.csv files, verified by inspection. The validator
# reports a mismatch as a warning (the data may legitimately be refreshed) rather than an
# error, but the tests assert them so an accidental truncation or duplication is caught.
# --------------------------------------------------------------------------------------

EXPECTED_ROW_COUNTS: Mapping[str, int] = MappingProxyType(
    {
        "customers": 1_000,
        "products": 500,
        "transactions": 20_000,
        "returns": 5_048,
    }
)

EXPECTED_CUSTOMER_COUNT = 1_000
EXPECTED_SKU_COUNT = 500
EXPECTED_ORDER_COUNT = 6_726
