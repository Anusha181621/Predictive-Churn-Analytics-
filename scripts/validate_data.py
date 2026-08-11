"""Validate the four source CSV files and write the data quality report.

Reads the CSVs directly, runs every validator in :mod:`src.data.validation`, prints the result
and writes ``outputs/data_quality_report.json`` for the dashboard to display later.

The source CSV files are never modified.

Usage::

    python scripts/validate_data.py
    python scripts/validate_data.py --table customers --table returns
    python scripts/validate_data.py --no-write --quiet
    python scripts/validate_data.py --json-out outputs/data_quality_report.json

Exit codes: ``0`` all good, ``1`` at least one error-severity check failed, ``2`` the data
could not be loaded at all.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config.settings import ConfigError, get_settings  # noqa: E402
from src.data.csv_loader import SchemaError, load_all  # noqa: E402
from src.data.validation import (  # noqa: E402
    ValidationReport,
    compute_return_rate,
    validate_customers,
    validate_datasets,
    validate_products,
    validate_relationships,
    validate_returns,
    validate_transactions,
)
from src.utils.logging_config import configure_logging, get_logger  # noqa: E402

logger = get_logger("scripts.validate_data")

TABLE_CHOICES = ("customers", "products", "transactions", "returns", "relationships")

#: Report file name; Section 1 of the brief names this path explicitly.
REPORT_FILENAME = "data_quality_report.json"


def _write(text: str) -> None:
    """Write UTF-8 to stdout regardless of the console code page."""
    sys.stdout.buffer.write(text.encode("utf-8", errors="replace") + b"\n")


def _selected_report(data, tables: list[str]) -> ValidationReport:
    """Run only the requested validators, so a single file can be checked in isolation."""
    report = ValidationReport()
    builders = {
        "customers": lambda: validate_customers(data.customers, transactions=data.transactions),
        "products": lambda: validate_products(data.products),
        "transactions": lambda: validate_transactions(
            data.transactions, customers=data.customers, products=data.products
        ),
        "returns": lambda: validate_returns(data.returns, transactions=data.transactions),
        "relationships": lambda: validate_relationships(data),
    }
    for table in tables:
        report.add_table(builders[table]())
    report.dataset = {
        "row_counts": data.row_counts,
        **compute_return_rate(data.transactions, data.returns),
    }
    return report


def _render(report: ValidationReport) -> str:
    lines: list[str] = []
    rule = "=" * 78

    lines += [rule, "DATA QUALITY VALIDATION", rule, ""]

    for name, table in report.tables.items():
        lines.append("-" * 78)
        lines.append(f"{name.upper()}")
        lines.append("-" * 78)
        if table.metrics:
            width = max(len(key) for key in table.metrics)
            for key, value in table.metrics.items():
                lines.append(f"  {key:<{width}} : {_value(value)}")
            lines.append("")
        for check in table.checks:
            lines.append(f"  [{check.status:7s}] {check.name}: {check.detail}")
        lines.append("")

    lines += ["-" * 78, "RETURN RATE (measured, not assumed)", "-" * 78]
    for key in (
        "purchased_units",
        "returned_units",
        "unit_return_rate",
        "total_order_lines",
        "returned_order_lines",
        "line_return_rate",
        "total_orders",
        "orders_with_returns",
        "order_return_rate",
    ):
        if key in report.dataset:
            value = report.dataset[key]
            rendered = f"{value:.4%}" if key.endswith("_rate") and value is not None else _value(value)
            lines.append(f"  {key:<22} : {rendered}")

    counts = report.summary()
    lines += [
        "",
        rule,
        f"{counts['passed']}/{counts['total']} checks passed - "
        f"{counts['errors']} error(s), {counts['warnings']} warning(s).",
        rule,
    ]
    return "\n".join(lines)


def _value(value: object) -> str:
    if isinstance(value, dict):
        return ", ".join(f"{k}={v}" for k, v in value.items()) if value else "none"
    if isinstance(value, float):
        # Rates are sub-unit and need the extra places; money and averages read better at 2.
        return f"{value:,.6f}" if abs(value) < 1 else f"{value:,.2f}"
    if isinstance(value, int) and not isinstance(value, bool):
        return f"{value:,}"
    if hasattr(value, "date"):
        return str(value.date())
    return str(value)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate the source CSV files.")
    parser.add_argument(
        "--table",
        action="append",
        choices=TABLE_CHOICES,
        dest="tables",
        help="Validate only this table; repeatable. Defaults to all of them.",
    )
    parser.add_argument(
        "--json-out",
        default=None,
        help=f"Where to write the JSON report (default: OUTPUTS_DIR/{REPORT_FILENAME}).",
    )
    parser.add_argument("--no-write", action="store_true", help="Print only; write no files.")
    parser.add_argument("--quiet", action="store_true", help="Suppress the console report.")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Also exit non-zero when a warning-severity check fails.",
    )
    args = parser.parse_args(argv)

    configure_logging()
    settings = get_settings()

    try:
        settings.validate_files()
        data = load_all(settings=settings)
    except (ConfigError, SchemaError) as exc:
        logger.error("%s", exc)
        return 2

    report = _selected_report(data, args.tables) if args.tables else validate_datasets(data)

    if not args.quiet:
        _write(_render(report))

    if not args.no_write:
        destination = Path(args.json_out) if args.json_out else settings.outputs_path / REPORT_FILENAME
        report.save(destination)

    for failure in report.errors:
        logger.error("FAILED %s: %s", failure.full_name, failure.detail)
    for warning in report.warnings:
        logger.warning("%s: %s", warning.full_name, warning.detail)

    if not report.ok:
        return 1
    if args.strict and report.warnings:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
