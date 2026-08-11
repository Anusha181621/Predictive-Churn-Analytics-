"""Explain churn predictions with SHAP and write the explanation artefacts.

Runs straight from the CSV files::

    data/*.csv -> features -> saved model -> probabilities -> SHAP -> readable drivers

Usage::

    python scripts/explain.py
    python scripts/explain.py --customer CUST0234
    python scripts/explain.py --top-k 3 --risk-level High --risk-level Critical
    python scripts/explain.py --as-of 2025-06-30

Writes ``outputs/customer_churn_explanations.csv`` and, under ``outputs/explainability/``, the
global feature importance, SHAP summary, dependence curves and a readable driver summary.

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
from src.explainability.pipeline import (  # noqa: E402
    ExplainabilityResult,
    explain_churn,
    write_explainability_outputs,
)
from src.models.risk import RISK_LEVELS  # noqa: E402
from src.utils.logging_config import configure_logging, get_logger  # noqa: E402

logger = get_logger("scripts.explain")


def _write(text: str) -> None:
    sys.stdout.buffer.write(text.encode("utf-8", errors="replace") + b"\n")


def _report(result: ExplainabilityResult, written: dict[str, Path], examples: int) -> str:
    rule = "=" * 78
    summary = result.summary()
    lines = [rule, "CHURN EXPLAINABILITY (SHAP)", rule, ""]
    lines += [
        f"  model               : {summary['model']}",
        f"  churn definition    : no purchase within {summary['horizon_days']} days",
        f"  prediction date     : {summary['as_of_date']}",
        f"  customers explained : {summary['customers_explained']:,}",
        f"  drivers per customer: {summary['drivers_per_customer']}",
        f"  explanation rows    : {summary['explanation_rows']:,}",
        f"  features explained  : {summary['features_explained']}",
        f"  base value          : {summary['base_value']:.4f} ({summary['shap_scale']})",
        "",
    ]

    lines += ["-" * 78, "GLOBAL FEATURE IMPORTANCE (mean |SHAP|) AND DIRECTION OF IMPACT", "-" * 78]
    top = result.global_explanation.summary.head(15)
    for row in top.to_dict(orient="records"):
        lines.append(
            f"  {row['rank']:>2}. {row['feature']:<40} {row['mean_abs_shap']:.4f} "
            f"({row['importance_share']:>5.1%})  {row['direction']}"
        )

    mixed = result.global_explanation.summary["direction"].eq("mixed / non-monotone").sum()
    lines += [
        "",
        f"  {mixed} of {len(result.global_explanation.summary)} features have a non-monotone "
        "value-to-risk relationship and are reported as mixed rather than being forced into a "
        "direction.",
        "",
    ]

    lines += ["-" * 78, "MOST COMMON TOP DRIVER ACROSS CUSTOMERS", "-" * 78]
    for feature, count in summary["most_common_top_driver"].items():
        lines.append(f"  {feature:<44} top driver for {count:,} customers")
    lines.append("")

    if examples:
        lines += ["-" * 78, f"EXAMPLE EXPLANATIONS ({examples} highest-risk customers)", "-" * 78, ""]
        ranked = (
            result.explanations[result.explanations["Driver rank"].eq(1)]
            .sort_values("Churn probability", ascending=False)
            .head(examples)
        )
        for customer_id in ranked["Customer ID"]:
            lines.append(result.narrative_for(customer_id))
            lines.append("")

    lines += ["-" * 78, "FILES WRITTEN", "-" * 78]
    for name, path in written.items():
        lines.append(f"  {name:<28} {path}")
    lines += ["", rule]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Explain churn predictions with SHAP.")
    parser.add_argument("--as-of", default=None, help="Prediction date (YYYY-MM-DD).")
    parser.add_argument(
        "--top-k", type=int, default=5, help="Drivers per customer (the brief asks for 3-5)."
    )
    parser.add_argument(
        "--risk-level",
        action="append",
        choices=list(RISK_LEVELS),
        dest="risk_levels",
        help="Restrict to these risk bands; repeatable. Defaults to every customer.",
    )
    parser.add_argument("--model-dir", default=None)
    parser.add_argument("--out", default=None, help="Path for the per-customer explanations CSV.")
    parser.add_argument("--no-write", action="store_true", help="Compute only; write no files.")
    parser.add_argument("--quiet", action="store_true", help="Suppress the console report.")
    parser.add_argument(
        "--examples", type=int, default=3, help="How many example explanations to print."
    )
    parser.add_argument(
        "--customer",
        default=None,
        help="Print one customer's explanation block and exit.",
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

    try:
        result = explain_churn(
            data,
            as_of_date=args.as_of,
            model_dir=args.model_dir,
            settings=settings,
            top_k=args.top_k,
            risk_levels=tuple(args.risk_levels) if args.risk_levels else None,
        )
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        return 2

    if args.customer:
        _write(result.narrative_for(args.customer))
        return 0

    written: dict[str, Path] = {}
    if not args.no_write:
        written = write_explainability_outputs(result, settings, explanations_path=args.out)

    if not args.quiet:
        _write(_report(result, written, args.examples))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
