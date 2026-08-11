"""Profile the source CSV files and run the data quality checks.

Makes the initial data inspection reproducible instead of a one-off: loads the four CSVs
through the shared loader, prints a profile (shape, dtypes, null counts, key counts, date
ranges, numeric summaries, categorical value counts), runs every check in
:mod:`src.data.validation`, and writes the results to ``outputs/``.

Usage::

    python scripts/inspect_data.py
    python scripts/inspect_data.py --quiet --out outputs
    python scripts/inspect_data.py --no-write

Exits non-zero if any ``error``-severity check fails, so it is usable as a gate in a future
pipeline or CI job.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow `python scripts/inspect_data.py` from the project root without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from src.config.settings import ConfigError, get_settings  # noqa: E402
from src.data import schema as sch  # noqa: E402
from src.data.csv_loader import Datasets, SchemaError, load_all  # noqa: E402
from src.data.validation import ValidationReport, validate_datasets  # noqa: E402
from src.utils.logging_config import configure_logging, get_logger  # noqa: E402
from src.utils.paths import ensure_dir  # noqa: E402

logger = get_logger("scripts.inspect_data")

#: Categorical columns worth listing in full, keyed by table.
CATEGORICAL_COLUMNS: dict[str, tuple[str, ...]] = {
    "customers": ("customer_gender", "country", "acquisition_channel"),
    "products": ("category", "brand", "product_gender"),
    "transactions": ("coupon_used", "payment_method", "discount_pct"),
    "returns": (),
}


def _rule(title: str, char: str = "=") -> str:
    return f"\n{char * 78}\n{title}\n{char * 78}"


def _profile_table(name: str, frame: pd.DataFrame) -> list[str]:
    table_schema = sch.TABLES[name]
    lines = [_rule(f"{name.upper()}  ({table_schema.description})", "-")]
    lines.append(f"shape: {frame.shape[0]:,} rows x {frame.shape[1]} columns")
    lines.append("")
    lines.append(f"{'column':<24} {'dtype':<16} {'nulls':>7} {'unique':>10}  example")
    for column in frame.columns:
        series = frame[column]
        example = "" if series.empty else str(series.iloc[0])
        lines.append(
            f"{column:<24} {str(series.dtype):<16} {series.isna().sum():>7,} "
            f"{series.nunique():>10,}  {example}"
        )

    for column in table_schema.date_columns:
        canonical = table_schema.canonical(column)
        series = frame[canonical]
        lines.append("")
        lines.append(f"{canonical}: {series.min().date()} to {series.max().date()}")

    numeric = [table_schema.canonical(c) for c in table_schema.numeric_columns]
    if numeric:
        lines.append("")
        lines.append("numeric summary:")
        summary = frame[numeric].describe().T[["min", "mean", "50%", "max", "std"]]
        lines.append(summary.to_string(float_format=lambda v: f"{v:,.2f}"))

    categoricals = CATEGORICAL_COLUMNS.get(name, ())
    if categoricals:
        lines.append("")
        lines.append("categorical value counts:")
        for column in categoricals:
            counts = frame[column].value_counts().sort_index()
            rendered = ", ".join(f"{value}={count:,}" for value, count in counts.items())
            lines.append(f"  {column}: {rendered}")

    return lines


def _profile_relationships(data: Datasets) -> list[str]:
    txn, rtn = data.transactions, data.returns
    purchased_units = int(txn["quantity"].sum())
    returned_units = int(rtn["return_quantity"].sum())
    orders = txn.drop_duplicates("order_id")
    orders_per_customer = txn.groupby("customer_id", observed=True)["order_id"].nunique()

    lines = [_rule("CROSS-TABLE FACTS", "-")]
    facts = [
        ("unique customers", f"{data.customers['customer_id'].nunique():,}"),
        ("unique SKUs", f"{data.products['sku_id'].nunique():,}"),
        ("order lines", f"{len(txn):,}"),
        ("distinct orders", f"{txn['order_id'].nunique():,}"),
        ("purchase date range", f"{txn['purchase_date'].min().date()} to {txn['purchase_date'].max().date()}"),
        ("return date range", f"{rtn['return_date'].min().date()} to {rtn['return_date'].max().date()}"),
        ("units purchased", f"{purchased_units:,}"),
        ("units returned", f"{returned_units:,}"),
        ("unit return rate", f"{100.0 * returned_units / purchased_units:.4f}%"),
        ("line return rate", f"{100.0 * len(rtn) / len(txn):.2f}%"),
        ("gross sales", f"{(txn['quantity'] * txn['selling_price']).sum():,.2f}"),
        ("net sales", f"{txn['net_order_value'].sum():,.2f}"),
        ("average order value (net)", f"{txn.groupby('order_id', observed=True)['net_order_value'].sum().mean():,.2f}"),
        ("average orders per customer", f"{orders_per_customer.mean():.2f}"),
        ("one-order customers", f"{int(orders_per_customer.eq(1).sum()):,}"),
        ("customers with returns", f"{rtn['customer_id'].nunique():,}"),
        ("orders per year", ", ".join(
            f"{year}: {count:,}"
            for year, count in sorted(orders.groupby(orders["purchase_date"].dt.year).size().items())
        )),
    ]
    width = max(len(label) for label, _ in facts)
    lines.extend(f"{label:<{width}} : {value}" for label, value in facts)
    return lines


def _build_markdown(data: Datasets, report: ValidationReport, settings) -> str:
    txn = data.transactions
    lines = [
        "# Data profile",
        "",
        "Generated by `scripts/inspect_data.py`. The four CSV files are the source of truth; "
        "this report is read-only and never modifies them.",
        "",
        f"- Data directory: `{settings.data_path}`",
        f"- Currency: {settings.currency}",
        "",
        "## Files",
        "",
        "| Table | File | Rows | Columns |",
        "|---|---|---|---|",
    ]
    for table, frame in data:
        lines.append(
            f"| {table} | `{settings.table_files[table]}` | {len(frame):,} | {frame.shape[1]} |"
        )

    lines += [
        "",
        "## Columns",
        "",
    ]
    for table, frame in data:
        lines.append(f"### {table}")
        lines.append("")
        lines.append("| Column | Dtype | Nulls | Unique |")
        lines.append("|---|---|---|---|")
        for column in frame.columns:
            series = frame[column]
            lines.append(
                f"| `{column}` | {series.dtype} | {series.isna().sum():,} | {series.nunique():,} |"
            )
        lines.append("")

    purchased = int(txn["quantity"].sum())
    returned = int(data.returns["return_quantity"].sum())
    lines += [
        "## Headline figures",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Unique customers | {data.customers['customer_id'].nunique():,} |",
        f"| Unique SKUs | {data.products['sku_id'].nunique():,} |",
        f"| Order lines | {len(txn):,} |",
        f"| Distinct orders | {txn['order_id'].nunique():,} |",
        f"| Transaction date range | {txn['purchase_date'].min().date()} to {txn['purchase_date'].max().date()} |",
        f"| Return date range | {data.returns['return_date'].min().date()} to {data.returns['return_date'].max().date()} |",
        f"| Units purchased | {purchased:,} |",
        f"| Units returned | {returned:,} |",
        f"| Unit return rate | {100.0 * returned / purchased:.4f}% |",
        f"| Line-level return rate | {100.0 * len(data.returns) / len(txn):.2f}% |",
        f"| Net sales | {settings.currency} {txn['net_order_value'].sum():,.2f} |",
        "",
        "## Data quality checks",
        "",
        report.to_markdown(),
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--out",
        default=None,
        help="Directory for the generated report files (default: OUTPUTS_DIR, i.e. outputs/).",
    )
    parser.add_argument("--no-write", action="store_true", help="Print only; write no files.")
    parser.add_argument("--quiet", action="store_true", help="Suppress the console profile.")
    args = parser.parse_args(argv)

    configure_logging()
    settings = get_settings()

    try:
        settings.validate_files()
        data = load_all(settings=settings)
    except (ConfigError, SchemaError) as exc:
        logger.error("%s", exc)
        return 2

    report = validate_datasets(data)

    if not args.quiet:
        out = [_rule("SOURCE CSV FILES")]
        for table, frame in data:
            out.append(f"  {settings.csv_path(table)}  ({len(frame):,} rows)")
        for table, frame in data:
            out.extend(_profile_table(table, frame))
        out.extend(_profile_relationships(data))
        out.append(_rule("DATA QUALITY CHECKS"))
        for result in report.checks:
            out.append(f"  [{result.status:7s}] {result.full_name}: {result.detail}")
        counts = report.summary()
        out.append(
            f"\n  {counts['passed']}/{counts['total']} checks passed - "
            f"{counts['errors']} error(s), {counts['warnings']} warning(s)."
        )
        # `errors="replace"` keeps a cp1252 console from choking on Duesseldorf / Liege.
        text = "\n".join(out)
        sys.stdout.buffer.write(text.encode("utf-8", errors="replace") + b"\n")

    if not args.no_write:
        out_dir = ensure_dir(args.out or settings.outputs_dir)
        profile_path = out_dir / "data_profile.md"
        profile_path.write_text(_build_markdown(data, report, settings), encoding="utf-8")
        logger.info("Wrote %s", profile_path)
        # Same report object, same writer as scripts/validate_data.py.
        report.save(out_dir / "data_quality_report.json")

    if not report.ok:
        for failure in report.errors:
            logger.error("FAILED %s: %s", failure.name, failure.detail)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
