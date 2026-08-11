"""Build the retention decision layer and write the scores and recommendations.

Runs straight from the CSV files::

    data/*.csv -> features -> churn model -> expected revenue -> segments
               -> revenue at risk -> opportunity score -> recommendations

Usage::

    python scripts/retention.py
    python scripts/retention.py --propensity 0.15          # test a different assumption
    python scripts/retention.py --min-roi 0.5              # demand a 50% margin before contacting
    python scripts/retention.py --top 20
    python scripts/retention.py --customer CUST0234

Writes ``outputs/customer_retention_scores.csv``, ``outputs/retention_recommendations.csv`` and
``outputs/retention_assumptions.json``.

Exit codes: ``0`` success, ``2`` the data or the model could not be loaded.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from src.config.settings import ConfigError, get_settings  # noqa: E402
from src.data.csv_loader import SchemaError, load_all  # noqa: E402
from src.retention.params import RetentionParams  # noqa: E402
from src.retention.pipeline import (  # noqa: E402
    RetentionResult,
    build_retention_layer,
    write_retention_outputs,
)
from src.retention.segments import SEGMENTS  # noqa: E402
from src.utils.logging_config import configure_logging, get_logger  # noqa: E402

logger = get_logger("scripts.retention")


def _write(text: str) -> None:
    sys.stdout.buffer.write(text.encode("utf-8", errors="replace") + b"\n")


def _customer_block(result: RetentionResult, customer_id: str) -> str:
    scores = result.scores[result.scores["Customer ID"].eq(customer_id)]
    recommendations = result.recommendations[
        result.recommendations["Customer ID"].eq(customer_id)
    ]
    if scores.empty:
        return f"No retention record for {customer_id}."
    score, recommendation = scores.iloc[0], recommendations.iloc[0]
    currency = result.params.currency
    roi = recommendation["Expected ROI"]
    return "\n".join(
        [
            f"Customer: {customer_id}",
            f"Churn probability: {score['Churn probability']:.0%}   "
            f"Risk: {score['Risk level']}   Priority: {score['Priority']}",
            f"Segments: {score['All segments']}",
            "",
            f"Lifetime revenue          : {currency} {score['Lifetime revenue']:,.2f}",
            f"Expected future revenue   : {currency} {score['Expected future revenue']:,.2f} "
            f"(next {result.params.revenue_horizon_days} days)",
            f"Revenue at risk           : {currency} {score['Revenue at risk']:,.2f}",
            f"Retention propensity      : {score['Retention propensity (ASSUMED)']:.0%}  "
            "[ASSUMPTION]",
            f"  basis                   : {score['Propensity basis (ASSUMED)']}",
            f"Expected retained revenue : {currency} {score['Expected retained revenue']:,.2f}",
            "",
            f"Recommended action  : {recommendation['Recommended action']}",
            f"Channel             : {recommendation['Recommended channel']}",
            f"Category            : {recommendation['Recommended category']}",
            f"Product             : {recommendation['Recommended product/SKU']} "
            f"{recommendation['Recommended product']}",
            f"Offer               : {recommendation['Recommended offer']}",
            f"Campaign cost       : {currency} {recommendation['Campaign cost']:,.2f}",
            f"Expected ROI        : {roi:+.0%}" if pd.notna(roi) else "Expected ROI        : n/a",
            f"Why                 : {recommendation['Reason']}",
        ]
    )


def _report(result: RetentionResult, written: dict[str, Path], top: int) -> str:
    rule = "=" * 78
    summary = result.summary()
    currency = result.params.currency
    lines = [rule, "RETENTION DECISION LAYER", rule, ""]
    lines += [
        f"  prediction date          : {summary['as_of_date']}",
        f"  customers                : {summary['customers']:,}",
        f"  revenue horizon          : {summary['revenue_horizon_days']} days",
        "",
        f"  expected future revenue  : {currency} {summary['total_expected_future_revenue']:>14,.2f}",
        f"  revenue at risk          : {currency} {summary['total_revenue_at_risk']:>14,.2f}",
        f"  expected retained revenue: {currency} "
        f"{summary['total_expected_retained_revenue']:>14,.2f}   [depends on the assumption below]",
        "",
    ]

    lines += ["-" * 78, "ASSUMPTIONS (not learned from this data)", "-" * 78]
    for name, detail in summary["assumptions"].items():
        lines.append(f"  {name} = {detail['value']}   [{detail['kind']}]")
        lines.append(f"      {detail['why']}")
        if "replace_with" in detail:
            lines.append(f"      Replace with: {detail['replace_with']}")
    lines.append("")
    lines.append(
        f"  Mean applied propensity: {summary['mean_retention_propensity']:.1%}. "
        "Revenue at risk is deliberately free of this assumption; expected retained revenue, "
        "the opportunity score and every ROI figure are not."
    )
    lines.append("")

    lines += ["-" * 78, "SEGMENTS (primary label; customers can carry several)", "-" * 78]
    counts = summary["primary_segment_counts"]
    for segment in list(SEGMENTS) + [s for s in counts if s not in SEGMENTS]:
        if segment in counts:
            flag_total = int(result.scores[segment].sum()) if segment in result.scores else 0
            lines.append(
                f"  {segment:<26} primary for {counts[segment]:4d}   "
                + (f"flagged for {flag_total:4d}" if segment in result.scores else "")
            )
    lines.append("")

    lines += ["-" * 78, "RECOMMENDED ACTIONS", "-" * 78]
    action_counts = summary["action_counts"]
    grouped = result.recommendations.groupby("Recommended action", observed=True).agg(
        customers=("Customer ID", "size"),
        cost=("Campaign cost", "sum"),
        expected_return=("Expected retained revenue", "sum"),
    )
    for action, row in grouped.sort_values("customers", ascending=False).iterrows():
        roi = (
            (row["expected_return"] - row["cost"]) / row["cost"] if row["cost"] > 0 else float("nan")
        )
        lines.append(
            f"  {str(action):<28} {int(row['customers']):4d} customers   "
            f"cost {currency} {row['cost']:>9,.2f}   return {currency} "
            f"{row['expected_return']:>10,.2f}   "
            + (f"ROI {roi:+7.0%}" if roi == roi else "ROI     n/a")
        )
    lines.append("")

    # A suppression list is only actionable if you can see *why* each customer is on it. "Already
    # engaged" and "unrecoverable" call for completely different follow-up, and lumping them into one
    # Do Not Target count hides that.
    suppressed = result.recommendations[
        result.recommendations["Recommended action"].eq("Do Not Target")
    ]
    if not suppressed.empty:
        lines += ["-" * 78, "WHY CUSTOMERS ARE NOT TARGETED", "-" * 78]
        reasons = [
            ("already engaged", "already highly engaged, not at risk"),
            ("beyond two full", "silent beyond two buying cycles, unrecoverable"),
            ("usual buying season", "seasonal and out of season — wait, do not discount"),
            ("never purchased", "no purchase history to win back"),
            ("does not cover", "expected return does not cover the campaign cost"),
        ]
        for fragment, label in reasons:
            count = int(suppressed["Reason"].str.contains(fragment, case=False, na=False).sum())
            if count:
                lines.append(f"  {label:<52} {count:4d}")
        lines.append("")

    lines += ["-" * 78, "CAMPAIGN ECONOMICS", "-" * 78]
    lines += [
        f"  customers targeted   : {summary['customers_targeted']:,}",
        f"  customers suppressed : {summary['customers_suppressed']:,}",
        f"  total campaign cost  : {currency} {summary['total_campaign_cost']:>12,.2f}",
        f"  expected return      : {currency} {summary['campaign_expected_return']:>12,.2f}",
        f"  blended ROI          : "
        + (f"{summary['campaign_roi']:+.0%}" if summary["campaign_roi"] is not None else "n/a"),
        "",
    ]

    lines += ["-" * 78, "PRIORITY BANDS", "-" * 78]
    for band, count in summary["priority_counts"].items():
        lines.append(f"  {str(band):<10} {count:4d} customers")
    lines.append("")

    if top:
        lines += ["-" * 78, f"TOP {top} RETENTION OPPORTUNITIES", "-" * 78]
        merged = result.scores.merge(
            result.recommendations[
                ["Customer ID", "Recommended action", "Recommended offer", "Expected ROI"]
            ],
            on="Customer ID",
        )
        head = merged.nlargest(top, "Retention opportunity score")[
            [
                "Customer ID", "Primary segment", "Churn probability", "Revenue at risk",
                "Retention opportunity score", "Priority", "Recommended action",
            ]
        ]
        lines.append(head.to_string(index=False))
        lines.append("")

    if written:
        lines += ["-" * 78, "FILES WRITTEN", "-" * 78]
        for name, path in written.items():
            lines.append(f"  {name:<30} {path}")
    lines += ["", rule]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the retention decision layer.")
    parser.add_argument("--as-of", default=None, help="Prediction date (YYYY-MM-DD).")
    parser.add_argument(
        "--propensity",
        type=float,
        default=None,
        help="Override the assumed base retention propensity (0-1).",
    )
    parser.add_argument(
        "--min-roi",
        type=float,
        default=None,
        help="Minimum expected ROI before a customer is targeted (default 0.0).",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=None,
        help="Revenue horizon in days. Defaults to the model's churn horizon.",
    )
    parser.add_argument("--model-dir", default=None)
    parser.add_argument("--no-write", action="store_true", help="Compute only; write no files.")
    parser.add_argument("--quiet", action="store_true", help="Suppress the console report.")
    parser.add_argument("--top", type=int, default=15, help="Top opportunities to print.")
    parser.add_argument("--customer", default=None, help="Print one customer's block and exit.")
    args = parser.parse_args(argv)

    configure_logging()
    settings = get_settings()

    try:
        settings.validate_files()
        data = load_all(settings=settings)
    except (ConfigError, SchemaError) as exc:
        logger.error("%s", exc)
        return 2

    overrides: dict[str, object] = {}
    if args.propensity is not None:
        overrides["base_retention_propensity"] = args.propensity
    if args.min_roi is not None:
        overrides["min_expected_roi"] = args.min_roi
    if args.horizon is not None:
        overrides["revenue_horizon_days"] = args.horizon

    try:
        params = RetentionParams(**overrides) if overrides else None
        result = build_retention_layer(
            data,
            as_of_date=args.as_of,
            model_dir=args.model_dir,
            settings=settings,
            params=params,
        )
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        return 2
    except ValueError as exc:
        logger.error("%s", exc)
        return 2

    if args.customer:
        _write(_customer_block(result, args.customer))
        return 0

    written: dict[str, Path] = {}
    if not args.no_write:
        written = write_retention_outputs(result, settings)

    if not args.quiet:
        _write(_report(result, written, args.top))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
