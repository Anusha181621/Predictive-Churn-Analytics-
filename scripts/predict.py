"""Score every customer with the saved model and write ``outputs/customer_churn_predictions.csv``.

Runs straight from the CSV files -- no database, no ingestion step, no manual preprocessing::

    data/*.csv -> features (as of the prediction date) -> saved model -> predictions

Usage::

    python scripts/predict.py
    python scripts/predict.py --as-of 2025-06-30
    python scripts/predict.py --out outputs/predictions_june.csv
    python scripts/predict.py --top 20

The default prediction date is the **latest** date in the data, not the latest labelable one:
training needs a settled outcome window, scoring wants the freshest view precisely because the
outcome has not happened yet.

Exit codes: ``0`` success, ``2`` the data or the model could not be loaded.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from src.config.settings import ConfigError, get_settings  # noqa: E402
from src.data.csv_loader import SchemaError, load_all  # noqa: E402
from src.models.predict import (  # noqa: E402
    PREDICTION_FILENAME,
    prediction_summary,
    score_customers,
    write_predictions,
)
from src.models.registry import load_model  # noqa: E402
from src.models.risk import RISK_LEVELS, risk_distribution  # noqa: E402
from src.utils.logging_config import configure_logging, get_logger  # noqa: E402

logger = get_logger("scripts.predict")


def _write(text: str) -> None:
    sys.stdout.buffer.write(text.encode("utf-8", errors="replace") + b"\n")


def _report(predictions: pd.DataFrame, model, destination: Path | None, top: int) -> str:
    rule = "=" * 78
    summary = prediction_summary(predictions)
    currency = get_settings().currency
    lines = [rule, "CHURN PREDICTIONS", rule, ""]
    lines += [
        f"  model              : {model.metadata.model_name} "
        f"(trained {model.metadata.trained_at})",
        f"  churn definition   : no purchase within {model.metadata.horizon_days} days",
        f"  prediction date    : {summary['prediction_date']}",
        f"  customers scored   : {summary['customers_scored']:,}",
        f"  mean churn prob.   : {summary['mean_churn_probability']:.4f}",
    ]
    if destination is not None:
        lines.append(f"  written to         : {destination}")

    lines += ["", "-" * 78, "RISK DISTRIBUTION", "-" * 78]
    distribution = risk_distribution(predictions["Risk level"])
    revenue = predictions.groupby("Risk level", observed=False)["Revenue at risk"].sum()
    for level in RISK_LEVELS:
        row = distribution.loc[level]
        lines.append(
            f"  {level:<10} {int(row['customers']):5d} customers  {row['share']:6.1%}   "
            f"revenue at risk {currency} {revenue.get(level, 0.0):>12,.2f}"
        )
    lines += [
        "",
        f"  total revenue at risk               {currency} "
        f"{summary['total_revenue_at_risk']:>14,.2f}",
        f"  of which High + Critical            {currency} "
        f"{summary['revenue_at_risk_high_and_critical']:>14,.2f}",
    ]

    if "Behavioural segment" in predictions.columns:
        lines += ["", "-" * 78, "MEAN CHURN PROBABILITY BY BEHAVIOURAL SEGMENT", "-" * 78]
        grouped = (
            predictions.groupby("Behavioural segment", observed=True)
            .agg(
                customers=("Customer ID", "size"),
                mean_probability=("Churn probability", "mean"),
                revenue_at_risk=("Revenue at risk", "sum"),
            )
            .sort_values("mean_probability", ascending=False)
        )
        for segment, row in grouped.iterrows():
            lines.append(
                f"  {str(segment):<20} {int(row['customers']):5d} customers   "
                f"mean p {row['mean_probability']:.3f}   "
                f"{currency} {row['revenue_at_risk']:>12,.2f}"
            )

    if top:
        lines += ["", "-" * 78, f"TOP {top} CUSTOMERS BY REVENUE AT RISK", "-" * 78]
        columns = [
            "Customer ID", "Churn probability", "Risk level", "Customer value",
            "Lifetime revenue", "Recency", "Frequency", "Revenue at risk",
        ]
        head = predictions.nlargest(top, "Revenue at risk")[columns]
        lines.append(head.to_string(index=False))

    lines += ["", rule]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Score customers for churn risk.")
    parser.add_argument(
        "--as-of",
        default=None,
        help="Prediction date (YYYY-MM-DD). Defaults to the latest date in the data.",
    )
    parser.add_argument(
        "--out", default=None, help=f"Output CSV path (default: OUTPUTS_DIR/{PREDICTION_FILENAME})."
    )
    parser.add_argument("--model-dir", default=None, help="Where to load the model from.")
    parser.add_argument("--no-write", action="store_true", help="Score only; write no file.")
    parser.add_argument("--quiet", action="store_true", help="Suppress the console report.")
    parser.add_argument(
        "--top", type=int, default=15, help="How many top customers to print (0 to skip)."
    )
    parser.add_argument(
        "--summary-json", default=None, help="Also write the summary to this JSON path."
    )
    args = parser.parse_args(argv)

    configure_logging()
    settings = get_settings()

    try:
        settings.validate_files()
        data = load_all(settings=settings)
        model = load_model(args.model_dir or settings.models_dir)
    except (ConfigError, SchemaError, FileNotFoundError) as exc:
        logger.error("%s", exc)
        return 2

    predictions = score_customers(
        data, as_of_date=args.as_of, model=model, settings=settings
    )

    destination = None
    if not args.no_write:
        destination = write_predictions(predictions, args.out, settings)

    if args.summary_json:
        path = Path(args.summary_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(prediction_summary(predictions), indent=2, default=str), encoding="utf-8"
        )
        logger.info("Wrote %s", path)

    if not args.quiet:
        _write(_report(predictions, model, destination, args.top))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
