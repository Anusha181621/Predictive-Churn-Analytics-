"""Read-only data quality validation.

Turns the one-off inspection of the four CSV files into executable checks. Every function here
is pure: it reads the loaded DataFrames and reports, and never mutates, coerces or drops a row.
This is *not* a cleaning step -- the CSVs are the source of truth, and a failure here means the
data changed in a way the platform needs to know about.

Severities
----------
``error``
    A structural or referential invariant the platform relies on. A failure breaks joins or
    arithmetic downstream.
``warning``
    A departure from the shipped dataset's known shape (for example a different row count).
    Legitimate if the data was refreshed, so it does not fail the run.
``info``
    A known, intentional property of the data that is easy to misread. Recorded so it is not
    rediscovered as a bug: the post-window return dates, the unit-vs-line return rate, and the
    deliberate drift between ``Selling Price`` and ``Product.Price``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Callable, Literal

import pandas as pd

from src.data import schema as sch
from src.data.csv_loader import Datasets
from src.utils.logging_config import get_logger

__all__ = ["CheckResult", "DataQualityReport", "run_all_checks", "Severity"]

logger = get_logger(__name__)

Severity = Literal["error", "warning", "info"]

#: ``Net Order Value`` is rounded to 2 decimals in the file, so allow just over half a cent.
MONEY_TOLERANCE = 0.011


@dataclass(frozen=True)
class CheckResult:
    """The outcome of a single data quality check."""

    name: str
    passed: bool
    severity: Severity
    detail: str

    @property
    def status(self) -> str:
        if self.passed:
            return "PASS"
        return "FAIL" if self.severity == "error" else self.severity.upper()

    def as_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "passed": self.passed,
            "severity": self.severity,
            "status": self.status,
            "detail": self.detail,
        }


@dataclass
class DataQualityReport:
    """The full set of check results, plus convenience accessors and renderers."""

    results: list[CheckResult] = field(default_factory=list)

    def add(self, name: str, passed: bool, detail: str, severity: Severity = "error") -> None:
        self.results.append(
            CheckResult(name=name, passed=bool(passed), severity=severity, detail=detail)
        )

    # --- querying -------------------------------------------------------------------

    @property
    def errors(self) -> list[CheckResult]:
        """Failed checks with ``error`` severity -- these should block a pipeline run."""
        return [r for r in self.results if not r.passed and r.severity == "error"]

    @property
    def warnings(self) -> list[CheckResult]:
        return [r for r in self.results if not r.passed and r.severity == "warning"]

    @property
    def ok(self) -> bool:
        """True when no ``error``-severity check failed."""
        return not self.errors

    def summary(self) -> dict[str, int]:
        return {
            "total": len(self.results),
            "passed": sum(1 for r in self.results if r.passed),
            "errors": len(self.errors),
            "warnings": len(self.warnings),
        }

    # --- rendering ------------------------------------------------------------------

    def to_markdown(self) -> str:
        counts = self.summary()
        lines = [
            "| # | Check | Severity | Result | Detail |",
            "|---|---|---|---|---|",
        ]
        for index, result in enumerate(self.results, start=1):
            detail = result.detail.replace("|", "\\|")
            lines.append(
                f"| {index} | {result.name} | {result.severity} | {result.status} | {detail} |"
            )
        lines.append("")
        lines.append(
            f"**{counts['passed']}/{counts['total']} checks passed** - "
            f"{counts['errors']} error(s), {counts['warnings']} warning(s)."
        )
        return "\n".join(lines)

    def to_dict(self) -> dict[str, object]:
        return {"summary": self.summary(), "results": [r.as_dict() for r in self.results]}

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    def __str__(self) -> str:  # pragma: no cover - convenience only
        counts = self.summary()
        head = f"{counts['passed']}/{counts['total']} checks passed"
        body = "\n".join(
            f"  [{r.status:7s}] {r.name}: {r.detail}"
            for r in self.results
            if not r.passed or r.severity == "info"
        )
        return f"{head}\n{body}" if body else head


# --------------------------------------------------------------------------------------
# individual check groups
# --------------------------------------------------------------------------------------


def _check_shape(data: Datasets, report: DataQualityReport) -> None:
    for table, frame in data:
        expected = sch.EXPECTED_ROW_COUNTS[table]
        actual = len(frame)
        report.add(
            f"{table}: row count",
            actual == expected,
            f"{actual:,} rows (expected {expected:,})",
            severity="warning",
        )

    report.add(
        "customers: Customer ID is unique",
        data.customers["customer_id"].is_unique,
        f"{data.customers['customer_id'].nunique():,} unique of {len(data.customers):,} rows",
    )
    report.add(
        "products: SKU ID is unique",
        data.products["sku_id"].is_unique,
        f"{data.products['sku_id'].nunique():,} unique of {len(data.products):,} rows",
    )

    for table, frame in data:
        null_counts = frame.isna().sum()
        offenders = null_counts[null_counts > 0]
        report.add(
            f"{table}: no missing values",
            offenders.empty,
            "no nulls" if offenders.empty else f"nulls in {offenders.to_dict()}",
        )


def _check_key_counts(data: Datasets, report: DataQualityReport) -> None:
    checks = (
        ("unique customers", data.customers["customer_id"].nunique(), sch.EXPECTED_CUSTOMER_COUNT),
        ("unique SKUs", data.products["sku_id"].nunique(), sch.EXPECTED_SKU_COUNT),
        ("distinct orders", data.transactions["order_id"].nunique(), sch.EXPECTED_ORDER_COUNT),
    )
    for label, actual, expected in checks:
        report.add(
            f"dataset: {label}",
            actual == expected,
            f"{actual:,} (expected {expected:,})",
            severity="warning",
        )


def _check_id_formats(data: Datasets, report: DataQualityReport) -> None:
    """Guard the zero-padded Order ID, the dataset's easiest column to corrupt."""
    order_ids = data.transactions["order_id"]
    is_string = pd.api.types.is_string_dtype(order_ids)
    report.add(
        "transactions: Order ID is a string dtype",
        is_string,
        f"dtype={order_ids.dtype} "
        "(an integer dtype means the zero padding has been destroyed)",
    )
    if is_string:
        widths = order_ids.str.len().unique()
        report.add(
            f"transactions: Order ID is zero-padded to {sch.ORDER_ID_WIDTH} characters",
            set(widths) == {sch.ORDER_ID_WIDTH},
            f"observed lengths {sorted(widths.tolist())}",
        )

    for label, frame, column, prefix, width in (
        ("customers", data.customers, "customer_id", "CUST", 8),
        ("products", data.products, "sku_id", "P", 5),
    ):
        series = frame[column]
        conforms = bool(series.str.fullmatch(rf"{prefix}\d+").all()) and set(
            series.str.len().unique()
        ) == {width}
        report.add(
            f"{label}: {column} matches {prefix}<digits> and is {width} characters",
            conforms,
            f"e.g. {series.iloc[0]!r}",
        )


def _check_referential_integrity(data: Datasets, report: DataQualityReport) -> None:
    customer_ids = set(data.customers["customer_id"])
    sku_ids = set(data.products["sku_id"])

    orphan_checks = (
        ("transactions.customer_id -> customers", set(data.transactions["customer_id"]) - customer_ids),
        ("transactions.sku_id -> products", set(data.transactions["sku_id"]) - sku_ids),
        ("returns.customer_id -> customers", set(data.returns["customer_id"]) - customer_ids),
        ("returns.sku_id -> products", set(data.returns["sku_id"]) - sku_ids),
        ("returns.order_id -> transactions", set(data.returns["order_id"]) - set(data.transactions["order_id"])),
    )
    for label, orphans in orphan_checks:
        report.add(
            f"integrity: {label}",
            not orphans,
            "no orphans" if not orphans else f"{len(orphans)} orphan key(s), e.g. {sorted(orphans)[:3]}",
        )

    # A return must reference an (Order ID, SKU ID) pair that actually exists as an order line.
    line_keys = set(map(tuple, data.transactions[["order_id", "sku_id"]].to_numpy()))
    return_keys = set(map(tuple, data.returns[["order_id", "sku_id"]].to_numpy()))
    missing_lines = return_keys - line_keys
    report.add(
        "integrity: returns (order_id, sku_id) -> transaction line",
        not missing_lines,
        "every return matches an order line"
        if not missing_lines
        else f"{len(missing_lines)} return(s) reference a line that does not exist",
    )
    report.add(
        "returns: (order_id, sku_id) is unique",
        len(return_keys) == len(data.returns),
        f"{len(return_keys):,} unique pairs of {len(data.returns):,} rows",
    )

    # The return's customer must be the customer who placed the order.
    order_owner = data.transactions.drop_duplicates("order_id").set_index("order_id")["customer_id"]
    expected_owner = data.returns["order_id"].map(order_owner)
    mismatched = int((expected_owner != data.returns["customer_id"]).sum())
    report.add(
        "returns: customer matches the order's customer",
        mismatched == 0,
        f"{mismatched} mismatch(es)",
    )


def _check_order_grain(data: Datasets, report: DataQualityReport) -> None:
    """One Order ID = one customer, one date, one payment method, unique SKUs."""
    grouped = data.transactions.groupby("order_id", observed=True)
    for column, label in (
        ("customer_id", "customer"),
        ("purchase_date", "purchase date"),
        ("payment_method", "payment method"),
    ):
        violations = int((grouped[column].nunique() > 1).sum())
        report.add(
            f"grain: one {label} per order",
            violations == 0,
            f"{violations} order(s) with more than one {label}",
        )

    duplicate_lines = int(data.transactions.duplicated(["order_id", "sku_id"]).sum())
    report.add(
        "grain: a SKU appears at most once per order",
        duplicate_lines == 0,
        f"{duplicate_lines} duplicate (order_id, sku_id) row(s)",
    )

    lines_per_order = grouped.size()
    report.add(
        "grain: lines per order",
        True,
        f"min={lines_per_order.min()}, max={lines_per_order.max()}, "
        f"mean={lines_per_order.mean():.2f}",
        severity="info",
    )


def _check_transaction_arithmetic(data: Datasets, report: DataQualityReport) -> None:
    txn = data.transactions
    expected = (
        txn["quantity"] * txn["selling_price"] * (1 - txn["discount_pct"] / 100.0)
    ).round(2)
    mismatches = int((expected - txn["net_order_value"]).abs().gt(MONEY_TOLERANCE).sum())
    report.add(
        "transactions: net_order_value = quantity x selling_price x (1 - discount)",
        mismatches == 0,
        f"{mismatches} mismatch(es) beyond {MONEY_TOLERANCE} tolerance",
    )

    report.add(
        "transactions: quantity is positive",
        bool(txn["quantity"].gt(0).all()),
        f"min={txn['quantity'].min()}, max={txn['quantity'].max()}",
    )
    report.add(
        "transactions: selling_price is positive",
        bool(txn["selling_price"].gt(0).all()),
        f"min={txn['selling_price'].min():.2f}, max={txn['selling_price'].max():.2f}",
    )
    report.add(
        "transactions: net_order_value is positive",
        bool(txn["net_order_value"].gt(0).all()),
        f"min={txn['net_order_value'].min():.2f}, max={txn['net_order_value'].max():.2f}",
    )

    observed_discounts = set(txn["discount_pct"].unique().tolist())
    report.add(
        "transactions: discount_pct within the allowed domain",
        observed_discounts <= set(sch.ALLOWED_DISCOUNTS),
        f"observed {sorted(observed_discounts)}",
    )

    coupon_without_discount = int(
        ((txn["coupon_used"] == "Yes") & (txn["discount_pct"] == 0)).sum()
    )
    report.add(
        "transactions: coupon_used = Yes implies discount_pct > 0",
        coupon_without_discount == 0,
        f"{coupon_without_discount} coupon row(s) with a zero discount",
    )


def _check_returns(data: Datasets, report: DataQualityReport) -> None:
    txn = data.transactions
    rtn = data.returns

    line_keys = ["order_id", "sku_id"]
    # No `validate=` here: duplicate keys are reported by their own checks above, and this
    # function must still produce a report rather than raise when the data is malformed.
    merged = rtn.merge(txn[line_keys + ["quantity", "purchase_date"]], on=line_keys, how="left")

    over_returned = int(merged["return_quantity"].gt(merged["quantity"]).sum())
    report.add(
        "returns: return_quantity <= purchased quantity",
        over_returned == 0,
        f"{over_returned} row(s) returning more units than were bought",
    )
    report.add(
        "returns: return_quantity is positive",
        bool(rtn["return_quantity"].gt(0).all()),
        f"min={rtn['return_quantity'].min()}, max={rtn['return_quantity'].max()}",
    )

    lag_days = (merged["return_date"] - merged["purchase_date"]).dt.days
    report.add(
        "returns: return_date is after purchase_date",
        bool(lag_days.gt(0).all()),
        f"lag in days: min={lag_days.min()}, max={lag_days.max()}, mean={lag_days.mean():.1f}",
    )

    purchased_units = int(txn["quantity"].sum())
    returned_units = int(rtn["return_quantity"].sum())
    report.add(
        "returns: unit return rate",
        True,
        f"{returned_units:,} of {purchased_units:,} units = "
        f"{100.0 * returned_units / purchased_units:.4f}% "
        f"(line-level rate is a DIFFERENT number: {len(rtn):,}/{len(txn):,} = "
        f"{100.0 * len(rtn) / len(txn):.2f}%)",
        severity="info",
    )

    # Returns of late-December orders land in the following January, so return dates extend
    # past the last purchase date. Anything computing features "as of" a date must clip
    # returns to that date or it leaks the future.
    last_purchase = txn["purchase_date"].max()
    beyond = int(rtn["return_date"].gt(last_purchase).sum())
    report.add(
        "returns: return dates extending past the last purchase date",
        True,
        f"{beyond} return(s) dated after {last_purchase.date()} "
        f"(latest {rtn['return_date'].max().date()}); clip returns to the as-of date when "
        "building features to avoid leakage",
        severity="info",
    )


def _check_customer_timeline(data: Datasets, report: DataQualityReport) -> None:
    txn = data.transactions
    first_purchase = txn.groupby("customer_id", observed=True)["purchase_date"].min()
    registration = data.customers.set_index("customer_id")["registration_date"]

    customers_without_orders = int(registration.index.difference(first_purchase.index).size)
    report.add(
        "customers: every customer has at least one transaction",
        customers_without_orders == 0,
        f"{customers_without_orders} customer(s) with no transactions",
        severity="warning",
    )

    aligned = registration.reindex(first_purchase.index)
    mismatched = int((aligned != first_purchase).sum())
    report.add(
        "customers: registration_date equals the first purchase date",
        mismatched == 0,
        f"{mismatched} customer(s) where the two differ",
    )

    before_registration = int(
        (txn["purchase_date"] < txn["customer_id"].map(registration)).sum()
    )
    report.add(
        "transactions: no purchase predates the customer's registration",
        before_registration == 0,
        f"{before_registration} transaction(s) before registration",
    )

    # Customers registered close to the end of the window have almost no observable history
    # and must be excluded or flagged when a churn label is assigned.
    last_purchase = txn["purchase_date"].max()
    tenure_days = (last_purchase - registration).dt.days
    thin = int(tenure_days.lt(30).sum())
    report.add(
        "customers: right-censored cohort",
        True,
        f"{thin} customer(s) registered within 30 days of {last_purchase.date()} "
        f"(newest registration {registration.max().date()}); flag or exclude them when "
        "labelling churn",
        severity="info",
    )


def _check_categoricals(data: Datasets, report: DataQualityReport) -> None:
    for table, frame in data:
        for column, allowed in sch.TABLES[table].allowed_values.items():
            observed = set(frame[column].dropna().unique().tolist())
            unexpected = observed - set(allowed)
            report.add(
                f"{table}: {column} values are within the known set",
                not unexpected,
                f"{len(observed)} distinct value(s)"
                if not unexpected
                else f"unexpected: {sorted(unexpected)}",
                severity="warning",
            )

    # Every city must belong to exactly one country.
    per_city = data.customers.groupby("city", observed=True)["country"].nunique()
    conflicts = per_city[per_city > 1]
    report.add(
        "customers: each city maps to a single country",
        conflicts.empty,
        f"{per_city.size} cities, no conflicts"
        if conflicts.empty
        else f"conflicting cities: {list(conflicts.index)}",
    )


def _check_price_drift(data: Datasets, report: DataQualityReport) -> None:
    """Selling Price intentionally drifts from Product.Price -- report, never flag."""
    merged = data.transactions[["sku_id", "selling_price"]].merge(
        data.products[["sku_id", "list_price"]], on="sku_id", how="left"
    )
    ratio = merged["selling_price"] / merged["list_price"]
    exact = int(ratio.sub(1.0).abs().lt(1e-9).sum())
    report.add(
        "transactions: selling_price vs product list_price",
        True,
        f"ratio min={ratio.min():.4f}, max={ratio.max():.4f}; "
        f"{exact:,} of {len(merged):,} lines match the list price exactly - the drift is "
        "intentional pricing variation, not an error",
        severity="info",
    )


def _check_date_coverage(data: Datasets, report: DataQualityReport) -> None:
    txn = data.transactions
    report.add(
        "dataset: transaction date range",
        True,
        f"{txn['purchase_date'].min().date()} to {txn['purchase_date'].max().date()}",
        severity="info",
    )
    orders = txn.drop_duplicates("order_id")
    orders_per_year = orders.groupby(orders["purchase_date"].dt.year).size().to_dict()
    report.add(
        "dataset: orders per year",
        True,
        ", ".join(f"{year}: {count:,}" for year, count in sorted(orders_per_year.items()))
        + " - volume grows year over year, so a time-based split trains on less data than it "
        "tests on",
        severity="info",
    )


_CHECK_GROUPS: tuple[Callable[[Datasets, DataQualityReport], None], ...] = (
    _check_shape,
    _check_key_counts,
    _check_id_formats,
    _check_referential_integrity,
    _check_order_grain,
    _check_transaction_arithmetic,
    _check_returns,
    _check_customer_timeline,
    _check_categoricals,
    _check_price_drift,
    _check_date_coverage,
)


def run_all_checks(data: Datasets) -> DataQualityReport:
    """Run every data quality check and return the report.

    A check group that raises is recorded as a failed check rather than propagating, so badly
    malformed data still produces a full, readable report instead of a traceback.
    """
    report = DataQualityReport()
    for group in _CHECK_GROUPS:
        try:
            group(data, report)
        except Exception as exc:  # noqa: BLE001 - a broken check must not hide the others
            name = group.__name__.lstrip("_")
            logger.exception("Check group %s raised", name)
            report.add(
                f"{name}: check group failed to run",
                False,
                f"{type(exc).__name__}: {exc}",
            )
    counts = report.summary()
    logger.info(
        "Data quality: %d/%d checks passed (%d error(s), %d warning(s))",
        counts["passed"],
        counts["total"],
        counts["errors"],
        counts["warnings"],
    )
    return report
