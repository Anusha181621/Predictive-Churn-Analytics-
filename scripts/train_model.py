"""Train the churn model from the CSV files and save it under ``models/``.

Runs the whole path with no manual preprocessing step::

    data/*.csv -> features at historical as-of dates -> labels -> select -> refit -> calibrate

Usage::

    python scripts/train_model.py
    python scripts/train_model.py --horizon 90
    python scripts/train_model.py --label-mode adaptive
    python scripts/train_model.py --no-save --quiet

Exit codes: ``0`` success, ``2`` the data could not be loaded. With ``--strict``, ``1`` when the
trained model fails its sanity floor against a single-feature heuristic.
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
from src.models.labels import LabelMode, LabelParams, compare_label_modes  # noqa: E402
from src.models.train import TrainingResult, train_churn_model  # noqa: E402
from src.utils.logging_config import configure_logging, get_logger  # noqa: E402
from src.utils.paths import ensure_dir  # noqa: E402

logger = get_logger("scripts.train_model")

METRICS_FILENAME = "model_metrics.json"


def _write(text: str) -> None:
    sys.stdout.buffer.write(text.encode("utf-8", errors="replace") + b"\n")


def _report(result: TrainingResult, label_comparison: dict | None) -> str:
    rule = "=" * 78
    lines = [rule, "CHURN MODEL TRAINING", rule, ""]
    meta = result.metadata
    plan = result.split.plan

    lines += [
        f"  model selected      : {meta.model_name}",
        f"  selection rationale : {meta.selection_rationale}",
        f"  churn definition    : no purchase within {meta.horizon_days} days after the as-of "
        f"date ({meta.label_mode} horizon)",
        f"  calibration         : {meta.calibration}",
        f"  feature columns     : {len(meta.feature_columns)}",
        "",
        "-" * 78,
        "TIME-BASED SPLIT",
        "-" * 78,
        f"  selection train  : {len(result.split.selection_train):5d} rows  "
        f"{plan.selection_train[0].date()} .. {plan.selection_train[-1].date()}",
        f"  selection valid. : {len(result.split.selection_validation):5d} rows  "
        f"{', '.join(d.date().isoformat() for d in plan.selection_validation)}",
        f"  fit (refit)      : {len(result.split.fit):5d} rows  "
        f"{plan.fit[0].date()} .. {plan.fit[-1].date()}",
        f"  calibration      : {len(result.split.calibration):5d} rows  "
        f"{', '.join(d.date().isoformat() for d in plan.calibration)}",
        f"  test             : {len(result.split.test):5d} rows  "
        f"{', '.join(d.date().isoformat() for d in plan.test)}",
        "",
        "  churn base rate by stage: "
        + ", ".join(f"{k}={v:.1%}" for k, v in meta.metrics["base_rates"].items()),
        "",
        "-" * 78,
        "MODEL COMPARISON (embargoed inner split, ranked by PR-AUC)",
        "-" * 78,
        result.leaderboard().to_string(index=False),
        "",
        "-" * 78,
        "TEST METRICS (single look, after calibration)",
        "-" * 78,
    ]

    m = result.test_eval.metrics
    lines += [
        f"  rows {result.test_eval.n}, churn base rate {result.test_eval.base_rate:.1%}",
        "",
        f"  ROC-AUC   {m['roc_auc']:.4f}      PR-AUC    {m['pr_auc']:.4f}",
        f"  Precision {m['precision']:.4f}      Recall    {m['recall']:.4f}      F1 {m['f1']:.4f}",
        "",
        "  Accuracy:",
        f"    at the tuned threshold {result.decision_threshold:.3f}   {m['accuracy']:.4f}",
        f"    at the conventional 0.5                {m['accuracy_at_0.5']:.4f}",
        f"    predicting the majority class always   {m['majority_class_accuracy']:.4f}"
        f"   <- the number to beat",
        f"    lift over that baseline                {m['accuracy_lift_over_majority']:+.4f}",
        "",
        "  Calibration:",
        f"    Brier {m['brier']:.4f}   ECE {m['ece']:.4f}   "
        f"mean predicted {m['mean_predicted']:.3f} vs observed {m['mean_observed']:.3f} "
        f"(bias {m['calibration_bias']:+.4f})",
        "",
        "  Business value (top decile of the ranked list):",
        f"    lift {m['lift_top_decile']:.2f}x random   precision {m['precision_top_decile']:.3f}"
        f"   recall {m['recall_top_decile']:.3f}",
    ]
    for key, value in result.test_eval.business.items():
        lines.append(f"    {key:<42} {value:,.4f}")
    confusion = result.test_eval.confusion
    lines += [
        "",
        f"  Confusion matrix @ {result.decision_threshold:.3f}: TN {confusion['tn']}  "
        f"FP {confusion['fp']}  FN {confusion['fn']}  TP {confusion['tp']}",
        "",
        "  Reliability (predicted vs observed churn rate per probability decile):",
    ]
    for row in result.test_eval.reliability.to_dict(orient="records"):
        lines.append(
            f"    {str(row['bin']):<22} predicted {row['predicted']:.3f}  "
            f"observed {row['observed']:.3f}  n={int(row['n'])}"
        )

    lines += [
        "",
        "-" * 78,
        "SANITY FLOOR: single-feature heuristics on the same test period",
        "-" * 78,
        result.baselines.to_string(index=False),
        "",
        "-" * 78,
        "TOP PREDICTIVE FEATURES (permutation importance, held-out period)",
        "-" * 78,
    ]
    top = result.top_features.head(15)
    for row in top.to_dict(orient="records"):
        lines.append(
            f"  {row['feature']:<42} {row['importance']:+.5f} "
            f"({row['importance_share']:.1%} of positive importance)"
        )

    if label_comparison:
        lines += [
            "",
            "-" * 78,
            "LABEL SENSITIVITY: fixed versus per-customer adaptive horizon",
            "-" * 78,
            f"  comparable customers    : {label_comparison['comparable_customers']}",
            f"  fixed churn rate        : {label_comparison['fixed_churn_rate']}",
            f"  adaptive churn rate     : {label_comparison['adaptive_churn_rate']}",
            f"  agreement               : {label_comparison['agreement']}",
            f"  comparison date         : {label_comparison['as_of_date']}",
            f"  rescued by adaptive     : {label_comparison['rescued_by_adaptive']} "
            "(the fixed horizon calls them churned, but they do buy again within their own "
            "expected interval -- the loyal-but-slow customers a uniform window mislabels)",
            f"  caught by adaptive      : {label_comparison['caught_by_adaptive']} "
            "(frequent buyers flagged sooner by their shorter personal horizon)",
            f"  censored under adaptive : {label_comparison['adaptive_censored']} "
            "(longer personal windows extend past the end of the data)",
        ]

    lines += ["", "-" * 78, "NOTES", "-" * 78]
    for note in result.notes:
        lines.append(f"  - {note}")
    lines += ["", rule]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train the churn prediction model.")
    parser.add_argument(
        "--horizon",
        type=int,
        default=None,
        help="Churn horizon in days (default: CHURN_INACTIVITY_DAYS, i.e. 180).",
    )
    parser.add_argument(
        "--label-mode",
        choices=[mode.value for mode in LabelMode],
        default=LabelMode.FIXED.value,
        help="fixed = one horizon for everybody; adaptive = scaled to each customer's cadence.",
    )
    parser.add_argument("--test-periods", type=int, default=1)
    parser.add_argument("--calibration-periods", type=int, default=1)
    parser.add_argument("--selection-validation-periods", type=int, default=3)
    parser.add_argument(
        "--max-features",
        type=int,
        default=None,
        help="Trim the feature matrix to the N most important columns, ranked on the embargoed "
        "inner validation split. Default: keep every column.",
    )
    parser.add_argument(
        "--tune",
        action="store_true",
        help="Random-search the selected model's hyperparameters against the inner validation "
        "split before refitting. Slow: multiplies training time by roughly --tuning-iterations.",
    )
    parser.add_argument("--tuning-iterations", type=int, default=40)
    parser.add_argument("--model-dir", default=None, help="Where to save the model.")
    parser.add_argument("--no-save", action="store_true", help="Train but do not persist.")
    parser.add_argument("--quiet", action="store_true", help="Suppress the console report.")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero if the model fails to beat a single-feature heuristic.",
    )
    parser.add_argument(
        "--skip-label-comparison",
        action="store_true",
        help="Skip the fixed-versus-adaptive label sensitivity check (saves a feature build).",
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

    label_params = LabelParams(
        horizon_days=args.horizon if args.horizon is not None else settings.churn_inactivity_days,
        mode=LabelMode(args.label_mode),
    )

    result = train_churn_model(
        data,
        label_params=label_params,
        settings=settings,
        test_periods=args.test_periods,
        calibration_periods=args.calibration_periods,
        selection_validation_periods=args.selection_validation_periods,
        max_features=args.max_features,
        tune=args.tune,
        tuning_iterations=args.tuning_iterations,
        model_dir=args.model_dir,
        save=not args.no_save,
    )

    comparison = None
    if not args.skip_label_comparison:
        # Quantify the residual seasonal risk in the fixed horizon rather than asserting it away.
        #
        # The comparison date must be early enough that the *longest* adaptive window is still
        # observable. Run it at the test date and the slow-cadence customers -- exactly the ones a
        # uniform 180-day horizon can mislabel -- are censored out of the comparison, so it would
        # report "0 rescued" for a purely structural reason and look like evidence of safety.
        data_end = data.transactions["purchase_date"].max().normalize()
        comparison_date = data_end - pd.Timedelta(days=label_params.adaptive_max_days)
        logger.info(
            "Comparing label modes at %s, the latest date where a %d-day adaptive window is "
            "fully observed",
            comparison_date.date(),
            label_params.adaptive_max_days,
        )
        comparison = compare_label_modes(data, comparison_date, label_params)

    if not args.no_save:
        metrics_path = ensure_dir(settings.outputs_dir) / METRICS_FILENAME
        payload = result.metadata.as_dict()
        if comparison:
            payload["label_sensitivity"] = comparison
        metrics_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
        )
        logger.info("Wrote %s", metrics_path)

    if not args.quiet:
        _write(_report(result, comparison))

    # A model that cannot beat a one-line heuristic must not pass silently -- but training did
    # succeed, so the default exit code reflects that and the warning carries the quality signal.
    # `--strict` turns it into a build failure for use as a pipeline gate.
    unproven = any(note.startswith("WARNING") for note in result.notes)
    if unproven:
        logger.warning(
            "The trained model did not clear its sanity floor against a single-feature heuristic; "
            "see the WARNING note above."
        )
        if args.strict:
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
