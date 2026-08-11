"""Reusable, read-only CSV validation layer.

The four CSV files are the source of truth. Nothing in this module writes to, coerces, drops
rows from or otherwise alters them: a validator reads DataFrames and *reports*. A failure here
means the data changed in a way the platform needs to know about, not that the data should be
quietly repaired.

Structure
---------
One validator per source file, each usable on its own::

    validate_customers(customers)
    validate_products(products)
    validate_transactions(transactions, customers=..., products=...)
    validate_returns(returns, transactions=...)

plus :func:`validate_relationships` for the cross-table invariants and
:func:`validate_datasets` which runs everything and returns a single
:class:`ValidationReport`. :func:`compute_return_rate` reports the *measured* return rate --
it is never assumed to be 20%.

Each validator returns a :class:`TableReport` carrying both its checks and a dict of summary
metrics (total rows, unique keys, missing values, duplicates, date range), so a dashboard can
render the numbers without re-deriving them.

Validators accept frames using the canonical snake_case column names produced by
:mod:`src.data.csv_loader`. They are defensive about structure: a missing column is reported as
a failed check and the checks that depend on it are skipped, rather than raising.

Severities
----------
``error``
    A structural, referential or arithmetic invariant the platform relies on.
``warning``
    A departure from the dataset's documented shape (a different row count, an unseen category
    value). Legitimate if the data was refreshed, so it does not fail the run.
``info``
    A known, intentional property that is easy to misread as a bug -- the post-window return
    dates, the unit-vs-line return rate, the deliberate selling-price drift.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import pandas as pd

from src.data import schema as sch
from src.data.csv_loader import Datasets
from src.utils.logging_config import get_logger
from src.utils.paths import ensure_dir

__all__ = [
    "CheckResult",
    "Severity",
    "TableReport",
    "ValidationReport",
    "compute_return_rate",
    "validate_customers",
    "validate_products",
    "validate_transactions",
    "validate_returns",
    "validate_relationships",
    "validate_datasets",
]

logger = get_logger(__name__)

Severity = Literal["error", "warning", "info"]

#: ``Net Order Value`` is rounded to 2 decimals in the file, so allow just over half a cent.
MONEY_TOLERANCE = 0.011

#: Ages outside this range are impossible and treated as an error.
AGE_HARD_BOUNDS = (0, 120)
#: Ages outside this range contradict the data dictionary and are treated as a warning.
AGE_DOCUMENTED_BOUNDS = (18, 65)

#: A date before this is a data entry error rather than history.
EARLIEST_PLAUSIBLE_DATE = pd.Timestamp("2000-01-01")


# --------------------------------------------------------------------------------------
# result containers
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class CheckResult:
    """The outcome of a single validation check."""

    table: str
    name: str
    passed: bool
    severity: Severity
    detail: str

    @property
    def full_name(self) -> str:
        return f"{self.table}: {self.name}"

    @property
    def status(self) -> str:
        if self.passed:
            return "PASS"
        return "FAIL" if self.severity == "error" else self.severity.upper()

    def as_dict(self) -> dict[str, object]:
        return {
            "table": self.table,
            "check": self.name,
            "passed": self.passed,
            "severity": self.severity,
            "status": self.status,
            "detail": self.detail,
        }


@dataclass
class TableReport:
    """Checks and summary metrics for one source file (or the cross-table group)."""

    table: str
    metrics: dict[str, object] = field(default_factory=dict)
    checks: list[CheckResult] = field(default_factory=list)

    # --- building -------------------------------------------------------------------

    def add(self, name: str, passed: bool, detail: str, severity: Severity = "error") -> None:
        self.checks.append(
            CheckResult(
                table=self.table,
                name=name,
                passed=bool(passed),
                severity=severity,
                detail=detail,
            )
        )

    def measure(self, **metrics: object) -> None:
        self.metrics.update(metrics)

    # --- querying -------------------------------------------------------------------

    @property
    def errors(self) -> list[CheckResult]:
        return [c for c in self.checks if not c.passed and c.severity == "error"]

    @property
    def warnings(self) -> list[CheckResult]:
        return [c for c in self.checks if not c.passed and c.severity == "warning"]

    @property
    def ok(self) -> bool:
        return not self.errors

    def check(self, name: str) -> CheckResult:
        """Return the check with this name. Raises ``KeyError`` if absent."""
        for candidate in self.checks:
            if candidate.name == name:
                return candidate
        raise KeyError(f"No check named {name!r} in table {self.table!r}")

    def as_dict(self) -> dict[str, object]:
        return {
            "table": self.table,
            "ok": self.ok,
            "metrics": _jsonable(self.metrics),
            "checks": [c.as_dict() for c in self.checks],
        }


@dataclass
class ValidationReport:
    """The full validation outcome: per-table reports plus dataset-level metrics."""

    tables: dict[str, TableReport] = field(default_factory=dict)
    dataset: dict[str, object] = field(default_factory=dict)
    generated_at: str = ""

    def __post_init__(self) -> None:
        if not self.generated_at:
            self.generated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")

    def add_table(self, report: TableReport) -> None:
        self.tables[report.table] = report

    # --- querying -------------------------------------------------------------------

    @property
    def checks(self) -> list[CheckResult]:
        return [c for report in self.tables.values() for c in report.checks]

    @property
    def errors(self) -> list[CheckResult]:
        """Failed ``error``-severity checks -- these should block a pipeline run."""
        return [c for c in self.checks if not c.passed and c.severity == "error"]

    @property
    def warnings(self) -> list[CheckResult]:
        return [c for c in self.checks if not c.passed and c.severity == "warning"]

    @property
    def ok(self) -> bool:
        """True when no ``error``-severity check failed."""
        return not self.errors

    def check(self, full_name: str) -> CheckResult:
        """Look a check up by its ``"table: name"`` full name."""
        for candidate in self.checks:
            if candidate.full_name == full_name:
                return candidate
        raise KeyError(f"No check named {full_name!r}")

    def summary(self) -> dict[str, int]:
        checks = self.checks
        return {
            "total": len(checks),
            "passed": sum(1 for c in checks if c.passed),
            "errors": len(self.errors),
            "warnings": len(self.warnings),
        }

    # --- rendering ------------------------------------------------------------------

    def to_dict(self) -> dict[str, object]:
        return {
            "generated_at": self.generated_at,
            "ok": self.ok,
            "summary": self.summary(),
            "dataset": _jsonable(self.dataset),
            "tables": {name: report.as_dict() for name, report in self.tables.items()},
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    def save(self, path: str | Path) -> Path:
        """Write the report as JSON. The only write this module performs, and never to data/."""
        destination = Path(path)
        ensure_dir(destination.parent)
        destination.write_text(self.to_json(), encoding="utf-8")
        logger.info("Wrote validation report to %s", destination)
        return destination

    def to_markdown(self) -> str:
        counts = self.summary()
        lines = [
            f"Generated: {self.generated_at}",
            "",
            "## Dataset summary",
            "",
            "| Metric | Value |",
            "|---|---|",
        ]
        lines += [f"| {_label(k)} | {_render(v)} |" for k, v in self.dataset.items()]

        for name, report in self.tables.items():
            lines += ["", f"## {name}", ""]
            if report.metrics:
                lines += ["| Metric | Value |", "|---|---|"]
                lines += [f"| {_label(k)} | {_render(v)} |" for k, v in report.metrics.items()]
                lines.append("")
            lines += ["| Check | Severity | Result | Detail |", "|---|---|---|---|"]
            for check in report.checks:
                detail = check.detail.replace("|", "\\|")
                lines.append(
                    f"| {check.name} | {check.severity} | {check.status} | {detail} |"
                )

        lines += [
            "",
            f"**{counts['passed']}/{counts['total']} checks passed** - "
            f"{counts['errors']} error(s), {counts['warnings']} warning(s).",
        ]
        return "\n".join(lines)

    def __str__(self) -> str:  # pragma: no cover - convenience only
        counts = self.summary()
        head = f"{counts['passed']}/{counts['total']} checks passed"
        body = "\n".join(
            f"  [{c.status:7s}] {c.full_name}: {c.detail}"
            for c in self.checks
            if not c.passed or c.severity == "info"
        )
        return f"{head}\n{body}" if body else head


# --------------------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------------------


def _jsonable(value):
    """Convert pandas / numpy scalars into JSON-serialisable Python values."""
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if pd.api.types.is_scalar(value):
        if pd.isna(value):
            return None
        item = getattr(value, "item", None)
        return item() if callable(item) else value
    return str(value)


def _label(key: str) -> str:
    return key.replace("_", " ").capitalize()


def _render(value: object) -> str:
    if isinstance(value, float):
        return f"{value:,.4f}" if abs(value) < 1 else f"{value:,.2f}"
    if isinstance(value, int):
        return f"{value:,}"
    if isinstance(value, dict):
        return ", ".join(f"{k}={_render(v)}" for k, v in value.items()) or "none"
    if isinstance(value, (list, tuple)):
        return ", ".join(str(v) for v in value) or "none"
    return str(value)


def _require_columns(
    frame: pd.DataFrame, report: TableReport, required: Sequence[str]
) -> list[str]:
    """Report which required columns are present, and return the missing ones.

    Callers skip any check that depends on a missing column, so validation degrades into a
    useful report instead of raising on a structurally broken file.
    """
    missing = [column for column in required if column not in frame.columns]
    report.add(
        "required columns are present",
        not missing,
        f"all {len(required)} present" if not missing else f"missing {missing}",
    )
    return missing


def _check_not_null(frame: pd.DataFrame, report: TableReport, columns: Iterable[str]) -> None:
    for column in columns:
        if column not in frame.columns:
            continue
        nulls = int(frame[column].isna().sum())
        report.add(
            f"{column} is not null",
            nulls == 0,
            "no nulls" if nulls == 0 else f"{nulls:,} null value(s)",
        )


def _missing_value_counts(frame: pd.DataFrame) -> dict[str, int]:
    counts = frame.isna().sum()
    return {str(column): int(count) for column, count in counts.items() if count}


def _date_metrics(series: pd.Series) -> dict[str, object]:
    valid = series.dropna()
    if valid.empty:
        return {"min": None, "max": None}
    return {"min": valid.min(), "max": valid.max()}


def _check_dates_are_valid(
    frame: pd.DataFrame,
    report: TableReport,
    column: str,
    *,
    not_after: pd.Timestamp | None = None,
    not_after_label: str = "",
) -> None:
    """A date column must parse, be non-null, and sit inside a plausible window."""
    if column not in frame.columns:
        return
    series = frame[column]

    is_datetime = pd.api.types.is_datetime64_any_dtype(series)
    report.add(
        f"{column} parses as a date",
        is_datetime,
        f"dtype={series.dtype}"
        + ("" if is_datetime else " (expected datetime64; the column did not parse)"),
    )
    if not is_datetime:
        return

    nulls = int(series.isna().sum())
    report.add(
        f"{column} has no unparsed values",
        nulls == 0,
        "all values parsed" if nulls == 0 else f"{nulls:,} value(s) failed to parse",
    )

    too_early = int(series.lt(EARLIEST_PLAUSIBLE_DATE).sum())
    report.add(
        f"{column} is not implausibly early",
        too_early == 0,
        f"{too_early} value(s) before {EARLIEST_PLAUSIBLE_DATE.date()}",
    )

    if not_after is not None:
        beyond = int(series.gt(not_after).sum())
        report.add(
            f"{column} is not after {not_after_label or not_after.date()}",
            beyond == 0,
            f"{beyond} value(s) after {not_after.date()}",
        )


# --------------------------------------------------------------------------------------
# Customer.csv
# --------------------------------------------------------------------------------------


def validate_customers(
    customers: pd.DataFrame, *, transactions: pd.DataFrame | None = None
) -> TableReport:
    """Validate ``Customer.csv``.

    Checks the customer key exists, is unique and non-null; that age, gender and the
    registration date are valid. Reports total and unique customers, missing values,
    duplicates and the registration date range.

    ``transactions`` is optional; when given, the registration date is also checked against
    the end of the transaction window.
    """
    report = TableReport("customers")
    required = ["customer_id", "age", "customer_gender", "city", "country",
                "acquisition_channel", "registration_date"]
    missing = _require_columns(customers, report, required)

    duplicates = 0
    unique_customers = 0
    if "customer_id" not in missing:
        key = customers["customer_id"]
        unique_customers = int(key.nunique(dropna=True))
        duplicates = int(key.duplicated().sum())
        report.add(
            "Customer ID is unique",
            duplicates == 0,
            f"{unique_customers:,} unique of {len(customers):,} rows"
            + ("" if duplicates == 0 else f"; {duplicates} duplicate row(s)"),
        )
        blank = int(key.fillna("").astype("string").str.strip().eq("").sum())
        report.add(
            "Customer ID is not blank",
            blank == 0,
            "no blank keys" if blank == 0 else f"{blank} blank key(s)",
        )
        report.add(
            "Customer ID count matches the documented dataset",
            unique_customers == sch.EXPECTED_CUSTOMER_COUNT,
            f"{unique_customers:,} (expected {sch.EXPECTED_CUSTOMER_COUNT:,})",
            severity="warning",
        )

    _check_not_null(customers, report, ["customer_id", "age", "customer_gender", "city",
                                       "country", "acquisition_channel", "registration_date"])

    # --- age ---
    if "age" not in missing:
        age = pd.to_numeric(customers["age"], errors="coerce")
        unparsed = int(age.isna().sum() - customers["age"].isna().sum())
        report.add(
            "Age is numeric",
            unparsed == 0,
            "all values numeric" if unparsed == 0 else f"{unparsed} non-numeric value(s)",
        )
        low, high = AGE_HARD_BOUNDS
        impossible = int(((age < low) | (age > high)).sum())
        report.add(
            f"Age is within the possible range {low}-{high}",
            impossible == 0,
            f"min={age.min()}, max={age.max()}"
            + ("" if impossible == 0 else f"; {impossible} impossible value(s)"),
        )
        doc_low, doc_high = AGE_DOCUMENTED_BOUNDS
        outside = int(((age < doc_low) | (age > doc_high)).sum())
        report.add(
            f"Age is within the documented range {doc_low}-{doc_high}",
            outside == 0,
            f"{outside} value(s) outside the documented range",
            severity="warning",
        )
        report.measure(age_min=age.min(), age_max=age.max(), age_mean=round(float(age.mean()), 1))

    # --- categoricals ---
    _check_allowed_values(customers, report, sch.CUSTOMERS, missing)

    # --- registration date ---
    horizon = None
    if transactions is not None and "purchase_date" in transactions.columns:
        horizon = transactions["purchase_date"].max()
    _check_dates_are_valid(
        customers,
        report,
        "registration_date",
        not_after=horizon,
        not_after_label="the last purchase date",
    )

    dates = (
        _date_metrics(customers["registration_date"])
        if "registration_date" not in missing
        else {"min": None, "max": None}
    )
    report.measure(
        total_customers=len(customers),
        unique_customers=unique_customers,
        duplicate_customers=duplicates,
        missing_values=_missing_value_counts(customers),
        registration_date_min=dates["min"],
        registration_date_max=dates["max"],
    )
    return report


# --------------------------------------------------------------------------------------
# Product.csv
# --------------------------------------------------------------------------------------


def validate_products(products: pd.DataFrame) -> TableReport:
    """Validate ``Product.csv``.

    Checks the SKU key exists, is unique and non-null; that price is non-negative; and that
    category and subcategory are populated.
    """
    report = TableReport("products")
    required = ["sku_id", "category", "subcategory", "brand", "product_gender", "list_price"]
    missing = _require_columns(products, report, required)

    duplicates = 0
    unique_skus = 0
    if "sku_id" not in missing:
        key = products["sku_id"]
        unique_skus = int(key.nunique(dropna=True))
        duplicates = int(key.duplicated().sum())
        report.add(
            "SKU ID is unique",
            duplicates == 0,
            f"{unique_skus:,} unique of {len(products):,} rows"
            + ("" if duplicates == 0 else f"; {duplicates} duplicate row(s)"),
        )
        blank = int(key.fillna("").astype("string").str.strip().eq("").sum())
        report.add(
            "SKU ID is not blank",
            blank == 0,
            "no blank keys" if blank == 0 else f"{blank} blank key(s)",
        )
        report.add(
            "SKU count matches the documented dataset",
            unique_skus == sch.EXPECTED_SKU_COUNT,
            f"{unique_skus:,} (expected {sch.EXPECTED_SKU_COUNT:,})",
            severity="warning",
        )

    _check_not_null(products, report, required)

    # --- price ---
    if "list_price" not in missing:
        price = pd.to_numeric(products["list_price"], errors="coerce")
        unparsed = int(price.isna().sum() - products["list_price"].isna().sum())
        report.add(
            "Price is numeric",
            unparsed == 0,
            "all values numeric" if unparsed == 0 else f"{unparsed} non-numeric value(s)",
        )
        negative = int(price.lt(0).sum())
        report.add(
            "Price is greater than or equal to zero",
            negative == 0,
            f"min={price.min():.2f}, max={price.max():.2f}"
            + ("" if negative == 0 else f"; {negative} negative value(s)"),
        )
        zero = int(price.eq(0).sum())
        report.add(
            "Price is greater than zero",
            zero == 0,
            f"{zero} SKU(s) priced at zero",
            severity="warning",
        )
        report.measure(
            price_min=round(float(price.min()), 2),
            price_max=round(float(price.max()), 2),
            price_mean=round(float(price.mean()), 2),
        )

    # --- category / subcategory populated ---
    for column in ("category", "subcategory"):
        if column in missing:
            continue
        blank = int(products[column].fillna("").astype("string").str.strip().eq("").sum())
        report.add(
            f"{column} is populated",
            blank == 0,
            f"{products[column].nunique():,} distinct value(s)"
            if blank == 0
            else f"{blank} blank value(s)",
        )

    _check_allowed_values(products, report, sch.PRODUCTS, missing)

    report.measure(
        total_products=len(products),
        unique_skus=unique_skus,
        duplicate_skus=duplicates,
        missing_values=_missing_value_counts(products),
        categories=int(products["category"].nunique()) if "category" not in missing else 0,
        subcategories=int(products["subcategory"].nunique()) if "subcategory" not in missing else 0,
        brands=int(products["brand"].nunique()) if "brand" not in missing else 0,
    )
    return report


# --------------------------------------------------------------------------------------
# Transaction.csv
# --------------------------------------------------------------------------------------


def validate_transactions(
    transactions: pd.DataFrame,
    *,
    customers: pd.DataFrame | None = None,
    products: pd.DataFrame | None = None,
) -> TableReport:
    """Validate ``Transaction.csv``.

    Checks that the keys exist, the purchase date is valid, quantity is positive, selling price
    is non-negative, discount lies between 0 and 100%, and that net order value equals
    ``quantity x selling_price x (1 - discount/100)``.

    When ``customers`` / ``products`` are supplied, the foreign keys
    ``Transaction.Customer ID -> Customer.Customer ID`` and
    ``Transaction.SKU ID -> Product.SKU ID`` are verified too.
    """
    report = TableReport("transactions")
    required = ["customer_id", "order_id", "sku_id", "purchase_date", "quantity",
                "selling_price", "discount_pct", "coupon_used", "net_order_value",
                "payment_method"]
    missing = _require_columns(transactions, report, required)

    _check_not_null(transactions, report, required)

    for column in ("customer_id", "order_id", "sku_id"):
        if column in missing:
            continue
        blank = int(transactions[column].fillna("").astype("string").str.strip().eq("").sum())
        report.add(
            f"{column} is not blank",
            blank == 0,
            "no blank keys" if blank == 0 else f"{blank} blank key(s)",
        )

    # The zero padding on Order ID is the easiest thing in this dataset to destroy.
    if "order_id" not in missing:
        order_ids = transactions["order_id"]
        is_string = pd.api.types.is_string_dtype(order_ids)
        report.add(
            "Order ID is a string, preserving zero padding",
            is_string,
            f"dtype={order_ids.dtype}"
            + ("" if is_string else " (an integer dtype has destroyed the zero padding)"),
        )
        if is_string:
            widths = sorted(order_ids.str.len().dropna().unique().tolist())
            report.add(
                f"Order ID is zero-padded to {sch.ORDER_ID_WIDTH} characters",
                widths == [sch.ORDER_ID_WIDTH],
                f"observed lengths {widths}",
            )

    _check_dates_are_valid(transactions, report, "purchase_date")

    # --- quantity ---
    if "quantity" not in missing:
        quantity = pd.to_numeric(transactions["quantity"], errors="coerce")
        non_positive = int(quantity.le(0).sum())
        report.add(
            "Quantity is greater than zero",
            non_positive == 0,
            f"min={quantity.min()}, max={quantity.max()}"
            + ("" if non_positive == 0 else f"; {non_positive} non-positive value(s)"),
        )

    # --- selling price ---
    if "selling_price" not in missing:
        price = pd.to_numeric(transactions["selling_price"], errors="coerce")
        negative = int(price.lt(0).sum())
        report.add(
            "Selling price is greater than or equal to zero",
            negative == 0,
            f"min={price.min():.2f}, max={price.max():.2f}"
            + ("" if negative == 0 else f"; {negative} negative value(s)"),
        )

    # --- discount ---
    if "discount_pct" not in missing:
        discount = pd.to_numeric(transactions["discount_pct"], errors="coerce")
        out_of_range = int(((discount < 0) | (discount > 100)).sum())
        report.add(
            "Discount is between 0 and 100 percent",
            out_of_range == 0,
            f"min={discount.min()}, max={discount.max()}"
            + ("" if out_of_range == 0 else f"; {out_of_range} value(s) outside 0-100"),
        )
        observed = set(discount.dropna().astype(int).unique().tolist())
        unexpected = observed - set(sch.ALLOWED_DISCOUNTS)
        report.add(
            "Discount is drawn from the documented domain",
            not unexpected,
            f"observed {sorted(observed)}"
            if not unexpected
            else f"unexpected value(s) {sorted(unexpected)}",
            severity="warning",
        )

    # --- net order value ---
    if not {"quantity", "selling_price", "discount_pct", "net_order_value"} & set(missing):
        quantity = pd.to_numeric(transactions["quantity"], errors="coerce")
        price = pd.to_numeric(transactions["selling_price"], errors="coerce")
        discount = pd.to_numeric(transactions["discount_pct"], errors="coerce")
        actual = pd.to_numeric(transactions["net_order_value"], errors="coerce")
        expected = (quantity * price * (1 - discount / 100.0)).round(2)
        deviation = (expected - actual).abs()
        mismatches = int(deviation.gt(MONEY_TOLERANCE).sum())
        report.add(
            "Net order value equals quantity x selling price x (1 - discount)",
            mismatches == 0,
            f"{mismatches} mismatch(es) beyond a {MONEY_TOLERANCE} tolerance"
            + ("" if mismatches == 0 else f"; largest deviation {deviation.max():.2f}"),
        )
        negative = int(actual.lt(0).sum())
        report.add(
            "Net order value is greater than or equal to zero",
            negative == 0,
            f"min={actual.min():.2f}, max={actual.max():.2f}"
            + ("" if negative == 0 else f"; {negative} negative value(s)"),
        )

    # --- coupon consistency ---
    if not {"coupon_used", "discount_pct"} & set(missing):
        discount = pd.to_numeric(transactions["discount_pct"], errors="coerce")
        offenders = int(((transactions["coupon_used"] == "Yes") & discount.eq(0)).sum())
        report.add(
            "A coupon implies a non-zero discount",
            offenders == 0,
            f"{offenders} coupon row(s) with a zero discount",
        )

    _check_allowed_values(transactions, report, sch.TRANSACTIONS, missing)

    # --- foreign keys ---
    if customers is not None and "customer_id" not in missing:
        known = set(customers["customer_id"].dropna())
        orphans = set(transactions["customer_id"].dropna()) - known
        report.add(
            "Customer ID exists in Customer.csv",
            not orphans,
            "every customer key resolves"
            if not orphans
            else f"{len(orphans)} unknown key(s), e.g. {sorted(orphans)[:3]}",
        )
    if products is not None and "sku_id" not in missing:
        known = set(products["sku_id"].dropna())
        orphans = set(transactions["sku_id"].dropna()) - known
        report.add(
            "SKU ID exists in Product.csv",
            not orphans,
            "every SKU key resolves"
            if not orphans
            else f"{len(orphans)} unknown key(s), e.g. {sorted(orphans)[:3]}",
        )

    # --- metrics ---
    metrics: dict[str, object] = {
        "total_order_lines": len(transactions),
        "missing_values": _missing_value_counts(transactions),
    }
    if "order_id" not in missing:
        metrics["distinct_orders"] = int(transactions["order_id"].nunique())
    if "customer_id" not in missing:
        metrics["purchasing_customers"] = int(transactions["customer_id"].nunique())
    if "sku_id" not in missing:
        metrics["skus_sold"] = int(transactions["sku_id"].nunique())
    if "quantity" not in missing:
        metrics["total_units_purchased"] = int(
            pd.to_numeric(transactions["quantity"], errors="coerce").sum()
        )
    if "net_order_value" not in missing:
        metrics["total_net_revenue"] = round(
            float(pd.to_numeric(transactions["net_order_value"], errors="coerce").sum()), 2
        )
    if "purchase_date" not in missing:
        dates = _date_metrics(transactions["purchase_date"])
        metrics["purchase_date_min"] = dates["min"]
        metrics["purchase_date_max"] = dates["max"]
    report.measure(**metrics)
    return report


# --------------------------------------------------------------------------------------
# Return.csv
# --------------------------------------------------------------------------------------


def validate_returns(
    returns: pd.DataFrame, *, transactions: pd.DataFrame | None = None
) -> TableReport:
    """Validate ``Return.csv``.

    Checks the keys exist, the return date is valid and the return quantity is positive. When
    ``transactions`` is supplied, also checks that every return corresponds to a real order
    line, that the returned quantity never exceeds what was purchased, and that the return date
    falls after the purchase date.
    """
    report = TableReport("returns")
    required = ["customer_id", "order_id", "sku_id", "return_date", "return_quantity"]
    missing = _require_columns(returns, report, required)

    _check_not_null(returns, report, required)

    for column in ("customer_id", "order_id", "sku_id"):
        if column in missing:
            continue
        blank = int(returns[column].fillna("").astype("string").str.strip().eq("").sum())
        report.add(
            f"{column} is not blank",
            blank == 0,
            "no blank keys" if blank == 0 else f"{blank} blank key(s)",
        )

    _check_dates_are_valid(returns, report, "return_date")

    if "return_quantity" not in missing:
        quantity = pd.to_numeric(returns["return_quantity"], errors="coerce")
        non_positive = int(quantity.le(0).sum())
        report.add(
            "Return quantity is greater than zero",
            non_positive == 0,
            f"min={quantity.min()}, max={quantity.max()}"
            + ("" if non_positive == 0 else f"; {non_positive} non-positive value(s)"),
        )

    line_keys = ["order_id", "sku_id"]
    if not set(line_keys) & set(missing):
        duplicated = int(returns.duplicated(line_keys).sum())
        report.add(
            "at most one return row per order line",
            duplicated == 0,
            f"{duplicated} duplicate (order_id, sku_id) row(s)",
        )

    # --- against the transactions that were actually placed ---
    if transactions is not None and not set(line_keys) & set(missing):
        available = [c for c in ("quantity", "purchase_date") if c in transactions.columns]
        merged = returns.merge(
            transactions[line_keys + available].drop_duplicates(line_keys),
            on=line_keys,
            how="left",
        )

        if available:
            unmatched = int(merged[available[0]].isna().sum())
            report.add(
                "every return corresponds to an actual transaction line",
                unmatched == 0,
                "every return matches an order line"
                if unmatched == 0
                else f"{unmatched} return(s) reference an order line that does not exist",
            )

        if "quantity" in available and "return_quantity" not in missing:
            purchased = pd.to_numeric(merged["quantity"], errors="coerce")
            returned = pd.to_numeric(merged["return_quantity"], errors="coerce")
            over = int(returned.gt(purchased).sum())
            report.add(
                "return quantity does not exceed the purchased quantity",
                over == 0,
                "no over-returns"
                if over == 0
                else f"{over} row(s) returning more units than were bought",
            )

        if "purchase_date" in available and "return_date" not in missing:
            lag = (merged["return_date"] - merged["purchase_date"]).dt.days
            not_after = int(lag.le(0).sum())
            report.add(
                "return date is after the purchase date",
                not_after == 0,
                f"lag in days: min={lag.min()}, max={lag.max()}, mean={lag.mean():.1f}"
                if not_after == 0
                else f"{not_after} return(s) dated on or before the purchase",
            )

        if "customer_id" in transactions.columns and "customer_id" not in missing:
            owner = transactions.drop_duplicates("order_id").set_index("order_id")["customer_id"]
            expected_owner = returns["order_id"].map(owner)
            mismatched = int((expected_owner != returns["customer_id"]).sum())
            report.add(
                "return customer matches the customer who placed the order",
                mismatched == 0,
                f"{mismatched} mismatch(es)",
            )

    metrics: dict[str, object] = {
        "total_return_lines": len(returns),
        "missing_values": _missing_value_counts(returns),
    }
    if "return_quantity" not in missing:
        metrics["total_units_returned"] = int(
            pd.to_numeric(returns["return_quantity"], errors="coerce").sum()
        )
    if "customer_id" not in missing:
        metrics["customers_with_returns"] = int(returns["customer_id"].nunique())
    if "order_id" not in missing:
        metrics["orders_with_returns"] = int(returns["order_id"].nunique())
    if "return_date" not in missing:
        dates = _date_metrics(returns["return_date"])
        metrics["return_date_min"] = dates["min"]
        metrics["return_date_max"] = dates["max"]
    report.measure(**metrics)
    return report


# --------------------------------------------------------------------------------------
# return rate
# --------------------------------------------------------------------------------------


def compute_return_rate(
    transactions: pd.DataFrame, returns: pd.DataFrame
) -> dict[str, object]:
    """Compute the **measured** return rate.

    The primary definition is unit-based::

        return rate = total returned quantity / total purchased quantity

    The line-based and order-based rates are reported alongside it because they are different
    numbers and are easy to mistake for one another. Nothing here assumes 20%.
    """
    purchased_units = int(pd.to_numeric(transactions["quantity"], errors="coerce").sum())
    returned_units = int(pd.to_numeric(returns["return_quantity"], errors="coerce").sum())

    total_lines = len(transactions)
    returned_lines = len(returns)
    total_orders = int(transactions["order_id"].nunique()) if total_lines else 0
    returned_orders = int(returns["order_id"].nunique()) if returned_lines else 0

    def rate(numerator: int, denominator: int) -> float | None:
        return round(numerator / denominator, 6) if denominator else None

    return {
        "purchased_units": purchased_units,
        "returned_units": returned_units,
        "unit_return_rate": rate(returned_units, purchased_units),
        "total_order_lines": total_lines,
        "returned_order_lines": returned_lines,
        "line_return_rate": rate(returned_lines, total_lines),
        "total_orders": total_orders,
        "orders_with_returns": returned_orders,
        "order_return_rate": rate(returned_orders, total_orders),
    }


# --------------------------------------------------------------------------------------
# cross-table relationships
# --------------------------------------------------------------------------------------


def _check_allowed_values(
    frame: pd.DataFrame,
    report: TableReport,
    table_schema: sch.TableSchema,
    missing: Sequence[str],
) -> None:
    """Report categorical values outside the documented set, as a warning."""
    for column, allowed in table_schema.allowed_values.items():
        if column in missing or column not in frame.columns:
            continue
        observed = set(frame[column].dropna().unique().tolist())
        unexpected = observed - set(allowed)
        report.add(
            f"{column} values are within the documented set",
            not unexpected,
            f"{len(observed)} distinct value(s)"
            if not unexpected
            else f"unexpected value(s) {sorted(unexpected)}",
            severity="warning",
        )


def validate_relationships(data: Datasets) -> TableReport:
    """Validate invariants that span more than one file."""
    report = TableReport("relationships")
    txn, rtn, customers, products = (
        data.transactions,
        data.returns,
        data.customers,
        data.products,
    )

    # --- order grain: one Order ID = one customer, one date, one payment method ---
    grouped = txn.groupby("order_id", observed=True)
    for column, label in (
        ("customer_id", "customer"),
        ("purchase_date", "purchase date"),
        ("payment_method", "payment method"),
    ):
        violations = int((grouped[column].nunique() > 1).sum())
        report.add(
            f"one {label} per order",
            violations == 0,
            f"{violations} order(s) with more than one {label}",
        )
    duplicate_lines = int(txn.duplicated(["order_id", "sku_id"]).sum())
    report.add(
        "a SKU appears at most once per order",
        duplicate_lines == 0,
        f"{duplicate_lines} duplicate (order_id, sku_id) row(s)",
    )
    lines_per_order = grouped.size()
    report.add(
        "lines per order",
        True,
        f"min={lines_per_order.min()}, max={lines_per_order.max()}, "
        f"mean={lines_per_order.mean():.2f}",
        severity="info",
    )

    # --- customer timeline ---
    first_purchase = txn.groupby("customer_id", observed=True)["purchase_date"].min()
    registration = customers.set_index("customer_id")["registration_date"]

    without_orders = int(registration.index.difference(first_purchase.index).size)
    report.add(
        "every customer has at least one transaction",
        without_orders == 0,
        f"{without_orders} customer(s) with no transactions",
        severity="warning",
    )
    aligned = registration.reindex(first_purchase.index)
    mismatched = int((aligned != first_purchase).sum())
    report.add(
        "registration date equals the first purchase date",
        mismatched == 0,
        f"{mismatched} customer(s) where the two differ",
    )
    before_registration = int((txn["purchase_date"] < txn["customer_id"].map(registration)).sum())
    report.add(
        "no purchase predates the customer's registration",
        before_registration == 0,
        f"{before_registration} transaction(s) before registration",
    )

    # --- geography ---
    per_city = customers.groupby("city", observed=True)["country"].nunique()
    conflicts = per_city[per_city > 1]
    report.add(
        "each city maps to a single country",
        conflicts.empty,
        f"{per_city.size} cities, no conflicts"
        if conflicts.empty
        else f"conflicting cities: {list(conflicts.index)}",
    )

    # --- unsold inventory ---
    never_sold = set(products["sku_id"]) - set(txn["sku_id"])
    report.add(
        "SKUs that were never sold",
        True,
        f"{len(never_sold)} SKU(s) with no transactions",
        severity="info",
    )

    # --- known-and-intentional properties, recorded so they are not mistaken for bugs ---
    last_purchase = txn["purchase_date"].max()
    beyond = int(rtn["return_date"].gt(last_purchase).sum())
    report.add(
        "return dates extending past the last purchase date",
        True,
        f"{beyond} return(s) dated after {last_purchase.date()} "
        f"(latest {rtn['return_date'].max().date()}) because late-December orders are returned "
        "in January; clip returns to the as-of date when building features to avoid leakage",
        severity="info",
    )

    tenure_days = (last_purchase - registration).dt.days
    thin = int(tenure_days.lt(30).sum())
    report.add(
        "right-censored cohort",
        True,
        f"{thin} customer(s) registered within 30 days of {last_purchase.date()} "
        f"(newest registration {registration.max().date()}); flag or exclude them when "
        "labelling churn",
        severity="info",
    )

    merged = txn[["sku_id", "selling_price"]].merge(
        products[["sku_id", "list_price"]], on="sku_id", how="left"
    )
    ratio = merged["selling_price"] / merged["list_price"]
    exact = int(ratio.sub(1.0).abs().lt(1e-9).sum())
    report.add(
        "selling price versus product list price",
        True,
        f"ratio min={ratio.min():.4f}, max={ratio.max():.4f}; {exact:,} of {len(merged):,} "
        "lines match the list price exactly - the drift is intentional pricing variation, "
        "not an error",
        severity="info",
    )

    orders = txn.drop_duplicates("order_id")
    per_year = orders.groupby(orders["purchase_date"].dt.year).size().to_dict()
    report.add(
        "orders per year",
        True,
        ", ".join(f"{year}: {count:,}" for year, count in sorted(per_year.items()))
        + " - volume grows year over year, so a time-based split trains on less data than it "
        "tests on",
        severity="info",
    )

    report.measure(
        distinct_orders=int(txn["order_id"].nunique()),
        orders_per_year={str(y): int(c) for y, c in sorted(per_year.items())},
        skus_never_sold=len(never_sold),
        customers_without_orders=without_orders,
        returns_after_last_purchase=beyond,
    )
    return report


# --------------------------------------------------------------------------------------
# entry point
# --------------------------------------------------------------------------------------


def validate_datasets(data: Datasets) -> ValidationReport:
    """Run every validator over the four loaded tables and return one report.

    A validator that raises is recorded as a failed check rather than propagating, so badly
    malformed data still produces a complete, readable report instead of a traceback.
    """
    report = ValidationReport()

    validators = (
        ("customers", lambda: validate_customers(data.customers, transactions=data.transactions)),
        ("products", lambda: validate_products(data.products)),
        (
            "transactions",
            lambda: validate_transactions(
                data.transactions, customers=data.customers, products=data.products
            ),
        ),
        ("returns", lambda: validate_returns(data.returns, transactions=data.transactions)),
        ("relationships", lambda: validate_relationships(data)),
    )
    for name, run in validators:
        try:
            report.add_table(run())
        except Exception as exc:  # noqa: BLE001 - one broken validator must not hide the rest
            logger.exception("Validator for %s raised", name)
            failed = TableReport(name)
            failed.add(
                "validator ran to completion", False, f"{type(exc).__name__}: {exc}"
            )
            report.add_table(failed)

    # --- dataset-level summary ---
    #
    # Every entry is guarded: the summary is a convenience for the dashboard, so a structurally
    # broken file must degrade it to `None` rather than lose the whole report.
    def safe(compute):
        try:
            return compute()
        except Exception:  # noqa: BLE001
            return None

    try:
        return_rate = compute_return_rate(data.transactions, data.returns)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Return rate computation raised")
        return_rate = {"error": f"{type(exc).__name__}: {exc}"}

    txn, rtn = data.transactions, data.returns
    report.dataset = {
        "row_counts": data.row_counts,
        "unique_customers": safe(lambda: int(data.customers["customer_id"].nunique())),
        "unique_skus": safe(lambda: int(data.products["sku_id"].nunique())),
        "distinct_orders": safe(lambda: int(txn["order_id"].nunique())),
        "purchase_date_min": safe(lambda: txn["purchase_date"].min()),
        "purchase_date_max": safe(lambda: txn["purchase_date"].max()),
        "return_date_min": safe(lambda: rtn["return_date"].min()),
        "return_date_max": safe(lambda: rtn["return_date"].max()),
        "total_net_revenue": safe(lambda: round(float(txn["net_order_value"].sum()), 2)),
        **return_rate,
    }

    counts = report.summary()
    rate = return_rate.get("unit_return_rate")
    logger.info(
        "Validation: %d/%d checks passed (%d error(s), %d warning(s)); "
        "measured unit return rate %s",
        counts["passed"],
        counts["total"],
        counts["errors"],
        counts["warnings"],
        f"{rate:.4%}" if isinstance(rate, float) else "n/a",
    )
    return report
