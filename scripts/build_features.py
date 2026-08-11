"""Build the customer feature table from the CSV files and write it to ``outputs/``.

Reads the four CSVs directly, computes one row per Customer ID as of a prediction date, and
writes ``outputs/customer_features.csv``. That output is an analytical artefact -- the CSV files
under ``data/`` remain the source of truth and are never modified.

Usage::

    python scripts/build_features.py
    python scripts/build_features.py --as-of 2025-06-30
    python scripts/build_features.py --as-of 2024-12-31 --out outputs/features_2024.csv
    python scripts/build_features.py --list-features

Exit codes: ``0`` success, ``2`` the data could not be loaded.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from src.config.settings import ConfigError, get_settings  # noqa: E402
from src.data.csv_loader import SchemaError, load_all  # noqa: E402
from src.features.builder import build_customer_features  # noqa: E402
from src.features.params import FeatureParams  # noqa: E402
from src.utils.logging_config import configure_logging, get_logger  # noqa: E402
from src.utils.paths import ensure_dir  # noqa: E402

logger = get_logger("scripts.build_features")

OUTPUT_FILENAME = "customer_features.csv"


def _write(text: str) -> None:
    sys.stdout.buffer.write(text.encode("utf-8", errors="replace") + b"\n")


def _report(result, destination: Path | None) -> str:
    rule = "=" * 78
    features = result.features
    lines = [rule, "CUSTOMER FEATURE BUILD", rule, ""]
    lines.append(f"  as-of date          : {result.as_of_date.date()}")
    lines.append(f"  customers (rows)    : {result.customer_count:,}")
    lines.append(f"  features (columns)  : {result.feature_count}")
    lines.append(f"  with purchase history: {int(features['has_purchase_history'].sum()):,}")
    if destination is not None:
        lines.append(f"  written to          : {destination}")
    lines.append("")

    lines.append("-" * 78)
    lines.append("FEATURES BY GROUP")
    lines.append("-" * 78)
    for group, columns in result.group_columns.items():
        lines.append(f"  {group} ({len(columns)}):")
        # Wrap the names so a 60-feature group stays readable in a terminal.
        current = "    "
        for column in columns:
            if len(current) + len(column) + 2 > 76:
                lines.append(current.rstrip())
                current = "    "
            current += f"{column}, "
        lines.append(current.rstrip().rstrip(","))
        lines.append("")

    lines.append("-" * 78)
    lines.append("SEGMENT DISTRIBUTION")
    lines.append("-" * 78)
    for column in ("behavioral_segment", "lifecycle_stage", "customer_value_segment"):
        counts = features[column].value_counts(dropna=False)
        rendered = ", ".join(f"{value}={count:,}" for value, count in counts.items())
        lines.append(f"  {column}: {rendered}")
    lines.append("")

    lines.append("-" * 78)
    lines.append("SPOT CHECKS")
    lines.append("-" * 78)
    observed = features[features["has_purchase_history"]]
    for column, fmt in (
        ("recency_days", "{:,.0f}"),
        ("total_orders", "{:,.2f}"),
        ("lifetime_revenue", "{:,.2f}"),
        ("average_order_value", "{:,.2f}"),
        ("purchase_gap_ratio", "{:,.2f}"),
        ("return_rate", "{:,.4f}"),
        ("discount_dependency_score", "{:,.4f}"),
        ("seasonal_customer_score", "{:,.4f}"),
        ("annualized_revenue", "{:,.2f}"),
    ):
        series = pd.to_numeric(observed[column], errors="coerce")
        lines.append(
            f"  {column:<26} mean={fmt.format(series.mean())}"
            f"  min={fmt.format(series.min())}  max={fmt.format(series.max())}"
            f"  null={int(series.isna().sum()):,}"
        )
    lines.append("")

    if result.issues:
        lines.append("-" * 78)
        lines.append("CALCULATION NOTES")
        lines.append("-" * 78)
        for issue in result.issues:
            lines.append(f"  - {issue}")
        lines.append("")

    lines.append(rule)
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the customer feature table.")
    parser.add_argument(
        "--as-of",
        default=None,
        help="Prediction date (YYYY-MM-DD). Defaults to AS_OF_DATE, else the maximum "
        "purchase date in the data.",
    )
    parser.add_argument(
        "--out",
        default=None,
        help=f"Output CSV path (default: OUTPUTS_DIR/{OUTPUT_FILENAME}).",
    )
    parser.add_argument("--no-write", action="store_true", help="Compute only; write no file.")
    parser.add_argument("--quiet", action="store_true", help="Suppress the console report.")
    parser.add_argument(
        "--list-features",
        action="store_true",
        help="Print the feature names one per line and exit.",
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

    as_of = args.as_of if args.as_of is not None else settings.as_of_date
    result = build_customer_features(data, as_of_date=as_of, params=FeatureParams())

    if args.list_features:
        _write("\n".join(result.feature_names))
        return 0

    destination = None
    if not args.no_write:
        if args.out:
            destination = Path(args.out)
            ensure_dir(destination.parent)
        else:
            destination = ensure_dir(settings.outputs_dir) / OUTPUT_FILENAME
        # utf-8-sig so the non-ASCII city names survive a double-click into Excel, which is
        # where a CRM manager will open this.
        #
        # `%.6g` is presentation rounding, applied only here: the in-memory table returned by
        # build_customer_features keeps full precision for modelling, while the CSV stays
        # readable (0.111111 rather than 0.1111111111111111) without losing useful signal.
        result.features.to_csv(
            destination, index=False, encoding="utf-8-sig", float_format="%.6g"
        )
        logger.info("Wrote %s (%d rows)", destination, len(result.features))

    if not args.quiet:
        _write(_report(result, destination))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
