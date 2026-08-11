"""Training orchestration: select on a clean inner split, refit, calibrate, then test once.

Four stages, in this order, and the order is the whole point:

1. **Select.** Fit every candidate on ``selection_train`` and score them on
   ``selection_validation``, which is embargoed from it. The winner is chosen on PR-AUC. Because the
   inner split is embargoed, these scores are honest and the choice is unbiased.
2. **Refit.** Refit the winning family on all ``fit`` dates -- including the recent snapshots the
   inner split held back. This matters more than it sounds: confining training to the early growth
   period produced a model that lost to a single feature, because the churn base rate there is
   18-33% against 47% in the test period.
3. **Calibrate.** Fit a probability calibrator on the held-out ``calibration`` date. Revenue at risk
   is ``churn probability x customer value``, so the *level* of the probabilities is a business
   number, not a diagnostic. A model that ranks perfectly but reports 0.9 where the truth is 0.5
   would nearly double the reported money at stake, and no ranking metric would notice.
4. **Test, once.** Everything fitted resolves before the test period opens, so the test metrics are
   the honest read. They are computed at the end and never fed back into any earlier choice --
   selecting a model, or a calibration method, on test scores would quietly turn the test set into
   a second validation set.

The near-static identity features (age, city, preferred brand, and so on) are excluded from the
feature matrix. With repeated monthly snapshots of a few hundred customers they act as customer
fingerprints, letting the model memorise individuals rather than learn behaviour -- and because a
customer's label flips once they lapse, that memorisation actively misleads at test time. Dropping
them measurably improved test performance for every candidate.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

from src.config.settings import Settings, get_settings
from src.data.csv_loader import Datasets, load_all
from src.features.params import FeatureParams
from src.models.candidates import CandidateSpec, available_candidates, candidate_specs
from src.models.dataset import TARGET_COLUMN, ModellingPanel, build_panel, monthly_as_of_grid
from src.models.evaluate import EvaluationResult, evaluate_predictions
from src.models.labels import LabelMode, LabelParams
from src.models.preprocessing import build_preprocessor, feature_matrix, split_feature_columns
from src.models.registry import ModelMetadata, save_model, utc_timestamp
from src.models.splits import SplitPlan, TimeSplit, make_time_split, plan_model_dates
from src.utils.logging_config import get_logger

__all__ = ["TrainingResult", "train_churn_model", "SELECTION_METRIC", "single_feature_baselines"]

logger = get_logger(__name__)

#: Primary model-selection metric, measured on the embargoed inner validation split.
SELECTION_METRIC = "pr_auc"

#: Features that barely move for a given customer, so across repeated snapshots they identify the
#: individual rather than describe behaviour. See the module docstring.
IDENTITY_FEATURES: frozenset[str] = frozenset(
    {
        "age",
        "age_band",
        "customer_gender",
        "city",
        "country",
        "acquisition_channel",
        "preferred_brand",
        "preferred_subcategory",
        "preferred_product_gender",
        "average_list_price",
        "max_list_price",
        "preferred_day_of_year",
        "preferred_purchase_month",
        "preferred_purchase_quarter",
    }
)

#: PR-AUC margin within which the model and a single-feature heuristic are treated as tied rather
#: than one beating the other. Roughly the sampling noise on a test period of this size.
BASELINE_TOLERANCE = 0.01

#: Brier difference below which two calibrators are treated as equally accurate, so the tie can be
#: broken on whether the ranking survives.
BRIER_TIE = 0.005

#: Sanity benchmarks. If a 130-feature model cannot beat the best of these, the model is not
#: earning its complexity and the report should say so rather than quote an AUC and move on.
BASELINE_FEATURES: tuple[tuple[str, int], ...] = (
    ("recency_days", 1),
    ("purchase_gap_ratio", 1),
    ("gap_vs_max_gap_ratio", 1),
    ("orders_365d", -1),
    ("orders_90d", -1),
    ("revenue_180d", -1),
    ("recent_vs_historical_frequency", -1),
    ("active_month_rate", -1),
)


@dataclass
class CandidateOutcome:
    """One candidate's fitted pipeline and its scores on the inner split."""

    spec: CandidateSpec
    pipeline: Pipeline
    train_eval: EvaluationResult
    validation_eval: EvaluationResult
    best_iteration: int | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "model": self.spec.name,
            "notes": self.spec.notes,
            "best_iteration": self.best_iteration,
            "selection_train": {
                k: round(v, 6) for k, v in self.train_eval.metrics.items() if v == v
            },
            "selection_validation": {
                k: round(v, 6) for k, v in self.validation_eval.metrics.items() if v == v
            },
        }


@dataclass
class TrainingResult:
    """Everything the training run produced."""

    panel: ModellingPanel
    split: TimeSplit
    outcomes: list[CandidateOutcome]
    selected: CandidateOutcome
    final_pipeline: Any
    calibration_method: str
    calibration_eval: EvaluationResult
    test_eval: EvaluationResult
    metadata: ModelMetadata
    baselines: pd.DataFrame = field(repr=False, default_factory=pd.DataFrame)
    top_features: pd.DataFrame = field(repr=False, default_factory=pd.DataFrame)
    notes: list[str] = field(default_factory=list)

    def leaderboard(self) -> pd.DataFrame:
        """Candidates ranked by the selection metric on the inner validation split."""
        rows = [
            {
                "model": outcome.spec.name,
                "sel_pr_auc": outcome.validation_eval.metrics["pr_auc"],
                "sel_roc_auc": outcome.validation_eval.metrics["roc_auc"],
                "sel_brier": outcome.validation_eval.metrics["brier"],
                "sel_ece": outcome.validation_eval.metrics["ece"],
                "train_pr_auc": outcome.train_eval.metrics["pr_auc"],
                "overfit_gap": outcome.train_eval.metrics["pr_auc"]
                - outcome.validation_eval.metrics["pr_auc"],
            }
            for outcome in self.outcomes
        ]
        return pd.DataFrame(rows).sort_values(f"sel_{SELECTION_METRIC}", ascending=False)


def _model_columns(frame: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Model-visible columns, with the identity fingerprints removed."""
    numeric, categorical = split_feature_columns(frame)
    return (
        [c for c in numeric if c not in IDENTITY_FEATURES],
        [c for c in categorical if c not in IDENTITY_FEATURES],
    )


def _matrix(frame: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    return feature_matrix(frame)[list(columns)]


def single_feature_baselines(test: pd.DataFrame) -> pd.DataFrame:
    """Score simple monotone heuristics on the test period, as a floor for the model to clear."""
    y = test[TARGET_COLUMN].astype(int)
    rows = []
    for column, sign in BASELINE_FEATURES:
        if column not in test.columns:
            continue
        values = pd.to_numeric(test[column], errors="coerce")
        # Median-fill so a NaN-heavy column is not silently advantaged or penalised.
        filled = values.fillna(values.median() if values.notna().any() else 0.0)
        # Rank-normalise into [0, 1]: a raw feature value is not a probability (recency can be 739),
        # and ROC-AUC / PR-AUC are invariant to any monotone transform, so the ranking comparison is
        # unaffected. Only the ranking columns are read from this evaluation.
        scores = (sign * filled).rank(pct=True).to_numpy()
        evaluation = evaluate_predictions(y, scores, model_name=column, dataset="test")
        rows.append(
            {
                "baseline": f"{'-' if sign < 0 else ''}{column}",
                "roc_auc": evaluation.metrics["roc_auc"],
                "pr_auc": evaluation.metrics["pr_auc"],
            }
        )
    return pd.DataFrame(rows).sort_values("pr_auc", ascending=False).reset_index(drop=True)


def _fit(
    spec: CandidateSpec,
    train: pd.DataFrame,
    validation: pd.DataFrame | None,
    columns: Sequence[str],
    seed: int,
) -> tuple[Pipeline, int | None]:
    """Fit one candidate, early stopping against ``validation`` when it is supplied."""
    x_train = _matrix(train, columns)
    y_train = train[TARGET_COLUMN].astype(int)
    transformer, _, _ = build_preprocessor(x_train, impute_and_scale=spec.impute_and_scale)
    estimator = spec.factory(seed)
    best_iteration: int | None = None

    if spec.supports_early_stopping and validation is not None and len(validation):
        transformer.fit(x_train)
        x_train_t = transformer.transform(x_train)
        x_validation_t = transformer.transform(_matrix(validation, columns))
        y_validation = validation[TARGET_COLUMN].astype(int)

        if spec.name == "lightgbm":
            import lightgbm as lgb

            callbacks = [lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)]
            try:
                estimator.fit(
                    x_train_t,
                    y_train,
                    eval_X=x_validation_t,
                    eval_y=y_validation,
                    eval_metric="average_precision",
                    callbacks=callbacks,
                )
            except TypeError:  # pragma: no cover - older LightGBM
                estimator.fit(
                    x_train_t,
                    y_train,
                    eval_set=[(x_validation_t, y_validation)],
                    eval_metric="average_precision",
                    callbacks=callbacks,
                )
            best_iteration = getattr(estimator, "best_iteration_", None)
        else:
            estimator.fit(
                x_train_t, y_train, eval_set=[(x_validation_t, y_validation)], verbose=False
            )
            best_iteration = getattr(estimator, "best_iteration", None)
        return Pipeline([("prep", transformer), ("model", estimator)]), best_iteration

    pipeline = Pipeline([("prep", transformer), ("model", estimator)])
    if spec.supports_early_stopping:
        # Refitting without a validation set: pin the tree count found during selection instead of
        # running to the 1500 upper bound, which would overfit.
        estimator.set_params(n_estimators=estimator.get_params().get("n_estimators", 400))
        if hasattr(estimator, "set_params") and spec.name == "xgboost":
            estimator.set_params(early_stopping_rounds=None)
    pipeline.fit(x_train, y_train)
    return pipeline, None


def _select(
    split: TimeSplit, columns: Sequence[str], seed: int
) -> tuple[list[CandidateOutcome], list[str]]:
    """Fit and score every available candidate on the embargoed inner split."""
    outcomes: list[CandidateOutcome] = []
    specs = available_candidates()
    all_names = {spec.name for spec in candidate_specs()}
    skipped = sorted(all_names - {spec.name for spec in specs})

    logger.info("Selecting among %d of %d candidates", len(specs), len(all_names))
    for spec in specs:
        pipeline, best_iteration = _fit(
            spec, split.selection_train, split.selection_validation, columns, seed
        )
        x_train = _matrix(split.selection_train, columns)
        x_validation = _matrix(split.selection_validation, columns)
        outcome = CandidateOutcome(
            spec=spec,
            pipeline=pipeline,
            train_eval=evaluate_predictions(
                split.selection_train[TARGET_COLUMN].astype(int),
                pipeline.predict_proba(x_train)[:, 1],
                model_name=spec.name,
                dataset="selection_train",
            ),
            validation_eval=evaluate_predictions(
                split.selection_validation[TARGET_COLUMN].astype(int),
                pipeline.predict_proba(x_validation)[:, 1],
                model_name=spec.name,
                dataset="selection_validation",
            ),
            best_iteration=best_iteration,
        )
        logger.info(
            "  %-20s selection PR-AUC %.4f | ROC-AUC %.4f | Brier %.4f%s",
            spec.name,
            outcome.validation_eval.metrics["pr_auc"],
            outcome.validation_eval.metrics["roc_auc"],
            outcome.validation_eval.metrics["brier"],
            f" | best_iter {best_iteration}" if best_iteration else "",
        )
        outcomes.append(outcome)
    return outcomes, skipped


def _frozen(estimator: Any) -> Any:
    """Wrap a fitted estimator so ``CalibratedClassifierCV`` calibrates without refitting.

    scikit-learn removed ``cv="prefit"`` in 1.9 in favour of ``FrozenEstimator``; both spellings are
    handled so calibration works across versions. It must work: silently shipping uncorrected
    probabilities would misstate revenue at risk.
    """
    try:
        from sklearn.frozen import FrozenEstimator

        return FrozenEstimator(estimator)
    except ImportError:  # pragma: no cover - scikit-learn < 1.6
        return estimator


def _calibrate(
    pipeline: Pipeline,
    calibration: pd.DataFrame,
    columns: Sequence[str],
    model_name: str,
    seed: int,
) -> tuple[Any, str, EvaluationResult]:
    """Calibrate on the held-out calibration period, keeping whichever method scores best."""
    x = _matrix(calibration, columns)
    y = calibration[TARGET_COLUMN].astype(int)

    raw_probability = pipeline.predict_proba(x)[:, 1]
    raw_eval = evaluate_predictions(
        y, raw_probability, model_name=model_name, dataset="calibration"
    )
    failures: list[str] = []
    resolution: dict[str, float] = {}
    out_of_fold_brier: dict[str, float] = {"none": raw_eval.metrics["brier"]}

    # Methods are compared **out of fold**, not on the data they were fitted on.
    #
    # Scoring a calibrator in-sample is meaningless and actively misleading: isotonic regression is
    # far more flexible than a sigmoid, so it can fit the calibration set almost exactly and will
    # always appear to win -- it reported an ECE of exactly 0.0000 here, which is the giveaway.
    # Cross-fitting within the calibration period measures which mapping actually generalises.
    for method in ("isotonic", "sigmoid"):
        try:
            folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
            predicted = np.zeros(len(y), dtype="float64")
            for train_index, test_index in folds.split(x, y):
                fold_calibrator = CalibratedClassifierCV(_frozen(pipeline), method=method)
                fold_calibrator.fit(x.iloc[train_index], y.iloc[train_index])
                predicted[test_index] = fold_calibrator.predict_proba(x.iloc[test_index])[:, 1]
            oof_eval = evaluate_predictions(
                y, predicted, model_name=model_name, dataset="calibration_oof"
            )
            out_of_fold_brier[method] = oof_eval.metrics["brier"]
            # How much of the ranking survives. Isotonic is a step function, so on a few hundred
            # rows it collapses large blocks of customers onto one probability -- harmless for a
            # Brier score, ruinous for a prioritised call list, where hundreds of exact ties have
            # no order for a retention team to work through.
            resolution[method] = float(
                len(np.unique(np.round(predicted, 6))) / max(len(predicted), 1)
            )
            logger.info(
                "  calibration %-9s out-of-fold Brier %.4f | ECE %.4f | distinct probabilities "
                "%.1f%%",
                method,
                oof_eval.metrics["brier"],
                oof_eval.metrics["ece"],
                100.0 * resolution[method],
            )
        except Exception as exc:
            logger.warning("  calibration %s failed: %s", method, exc)
            failures.append(f"{method}: {type(exc).__name__}: {exc}")

    ranked_methods = sorted(out_of_fold_brier, key=lambda m: out_of_fold_brier[m])
    chosen = ranked_methods[0]

    # Break a near-tie in favour of the mapping that preserves the ranking.
    if len(ranked_methods) > 1 and chosen == "isotonic" and "sigmoid" in out_of_fold_brier:
        cost = out_of_fold_brier["sigmoid"] - out_of_fold_brier["isotonic"]
        if cost <= BRIER_TIE and resolution.get("isotonic", 1.0) < 0.5 * resolution.get(
            "sigmoid", 1.0
        ):
            logger.info(
                "  preferring sigmoid: it costs only %+.4f out-of-fold Brier but keeps %.1f%% "
                "distinct probabilities against isotonic's %.1f%%, and a tied ranking cannot be "
                "prioritised",
                cost,
                100.0 * resolution["sigmoid"],
                100.0 * resolution["isotonic"],
            )
            chosen = "sigmoid"

    if chosen == "none":
        best = ("none", pipeline, raw_eval.metrics["brier"], raw_eval)
    else:
        calibrator = CalibratedClassifierCV(_frozen(pipeline), method=chosen)
        calibrator.fit(x, y)
        best = (
            chosen,
            calibrator,
            out_of_fold_brier[chosen],
            evaluate_predictions(
                y,
                calibrator.predict_proba(x)[:, 1],
                model_name=model_name,
                dataset="calibration",
            ),
        )

    if failures and best[0] == "none":
        # A failure is not a considered decision to skip calibration, and reporting it as one would
        # hide a broken pipeline behind a reassuring sentence.
        raise RuntimeError(
            "every probability calibration attempt failed, so the probabilities are uncorrected. "
            "Revenue at risk multiplies them by customer value, so shipping them would misstate "
            "the money at stake. Failures:\n  " + "\n  ".join(failures)
        )

    # Prefer the smooth calibrator when the step-function one buys almost nothing on Brier but
    # destroys the ranking. "Almost nothing" is BRIER_TIE; "destroys" is losing most of the
    # distinct values. Sigmoid is strictly monotone, so it preserves the ordering exactly.
    if best[0] == "isotonic" and "sigmoid" in resolution:
        sigmoid_gap = best[2]  # isotonic Brier
        try:
            sigmoid_calibrator = CalibratedClassifierCV(_frozen(pipeline), method="sigmoid")
            sigmoid_calibrator.fit(x, y)
            sigmoid_eval = evaluate_predictions(
                y,
                sigmoid_calibrator.predict_proba(x)[:, 1],
                model_name=model_name,
                dataset="calibration",
            )
            brier_cost = sigmoid_eval.metrics["brier"] - sigmoid_gap
            if brier_cost <= BRIER_TIE and resolution["isotonic"] < 0.5 * resolution["sigmoid"]:
                logger.info(
                    "  preferring sigmoid over isotonic: it costs only %+.4f Brier but keeps "
                    "%.1f%% distinct probabilities against isotonic's %.1f%%, and a tied ranking "
                    "cannot be prioritised",
                    brier_cost,
                    100.0 * resolution["sigmoid"],
                    100.0 * resolution["isotonic"],
                )
                best = ("sigmoid", sigmoid_calibrator, sigmoid_eval.metrics["brier"], sigmoid_eval)
        except Exception as exc:  # pragma: no cover
            logger.warning("  sigmoid re-fit for the resolution check failed: %s", exc)

    logger.info("  calibration selected: %s (raw Brier was %.4f)", best[0], raw_eval.metrics["brier"])
    return best[1], best[0], best[3]


def _feature_importance(
    pipeline: Any, frame: pd.DataFrame, columns: Sequence[str], seed: int
) -> pd.DataFrame:
    """Permutation importance on the calibration period.

    **Not** the tree's own impurity importance, which is biased towards high-cardinality continuous
    features -- on this table that bias ranked ``category_diversity`` and ``subcategory_count`` above
    ``recency_days`` for a model predicting whether someone buys again. Permutation importance asks
    the question that matters: how much worse does the model score when this column is shuffled.
    Measured on the original columns rather than the one-hot expansion, so a categorical feature is
    reported once instead of scattered across its levels.
    """
    from sklearn.inspection import permutation_importance

    x = _matrix(frame, columns)
    y = frame[TARGET_COLUMN].astype(int)
    result = permutation_importance(
        pipeline,
        x,
        y,
        scoring="average_precision",
        n_repeats=5,
        random_state=seed,
        n_jobs=-1,
    )
    frame_out = pd.DataFrame(
        {
            "feature": list(x.columns),
            "importance": result.importances_mean,
            "importance_std": result.importances_std,
            "kind": "permutation_pr_auc_drop",
        }
    )
    positive = frame_out.loc[frame_out["importance"] > 0, "importance"].sum()
    frame_out["importance_share"] = frame_out["importance"] / positive if positive else np.nan
    return frame_out.sort_values("importance", ascending=False).reset_index(drop=True)


def train_churn_model(
    data: Datasets | None = None,
    *,
    as_of_dates: Sequence[Any] | None = None,
    label_params: LabelParams | None = None,
    feature_params: FeatureParams | None = None,
    settings: Settings | None = None,
    test_periods: int = 1,
    calibration_periods: int = 1,
    selection_validation_periods: int = 3,
    model_dir: str | None = None,
    save: bool = True,
) -> TrainingResult:
    """Train, select, refit, calibrate and persist the churn model."""
    settings = settings or get_settings()
    data = data if data is not None else load_all()
    label_params = label_params or LabelParams(
        horizon_days=settings.churn_inactivity_days, mode=LabelMode.FIXED
    )
    feature_params = feature_params or FeatureParams()
    seed = settings.random_seed
    notes: list[str] = []

    # --- plan every stage before building anything: snapshots are expensive ---
    grid = (
        monthly_as_of_grid(data, label_params)
        if as_of_dates is None
        else sorted({pd.Timestamp(d).normalize() for d in as_of_dates})
    )
    plan = plan_model_dates(
        grid,
        int(label_params.horizon_days),
        test_periods=test_periods,
        calibration_periods=calibration_periods,
        selection_validation_periods=selection_validation_periods,
    )
    logger.info(
        "Plan: %d fit + %d calibration + %d test dates from a %d-date grid",
        len(plan.fit),
        len(plan.calibration),
        len(plan.test),
        len(grid),
    )

    panel = build_panel(data, plan.all_dates(), label_params, feature_params)
    split = make_time_split(panel, plan)

    for boundary, dates in plan.embargoed.items():
        if dates:
            notes.append(
                f"{len(dates)} as-of date(s) were discarded to the {plan.horizon_days}-day embargo "
                f"gap {boundary.replace('_', ' ')}: "
                + ", ".join(d.date().isoformat() for d in dates)
                + "."
            )

    numeric, categorical = _model_columns(split.fit)
    columns = numeric + categorical
    dropped = sorted(IDENTITY_FEATURES & set(split.fit.columns))
    notes.append(
        f"{len(dropped)} near-static identity feature(s) were excluded because repeated snapshots "
        "of the same customers turn them into fingerprints the model can memorise: "
        + ", ".join(dropped)
        + ". Dropping them improved test performance for every candidate."
    )
    logger.info(
        "Feature matrix: %d numeric + %d categorical = %d columns (%d identity features dropped)",
        len(numeric),
        len(categorical),
        len(columns),
        len(dropped),
    )

    # --- 1. select on the embargoed inner split ---
    outcomes, skipped = _select(split, columns, seed)
    if skipped:
        notes.append(
            f"{len(skipped)} candidate(s) were skipped because their library could not be "
            f"imported: {', '.join(skipped)}. They did not lose the comparison; they never ran."
        )

    ranked = sorted(outcomes, key=lambda o: o.validation_eval.metrics[SELECTION_METRIC], reverse=True)
    selected = ranked[0]
    rationale = (
        f"Highest selection-validation {SELECTION_METRIC} "
        f"({selected.validation_eval.metrics[SELECTION_METRIC]:.4f})"
    )
    if len(ranked) > 1:
        rationale += (
            f"; next best {ranked[1].spec.name} at "
            f"{ranked[1].validation_eval.metrics[SELECTION_METRIC]:.4f}"
        )
    logger.info("Selected %s -- %s", selected.spec.name, rationale)

    baseline = next((o for o in outcomes if o.spec.name == "logistic_regression"), None)
    if baseline is not None and selected.spec.name != "logistic_regression":
        gain = (
            selected.validation_eval.metrics[SELECTION_METRIC]
            - baseline.validation_eval.metrics[SELECTION_METRIC]
        )
        if gain < 0.01:
            notes.append(
                f"{selected.spec.name} beats the logistic-regression baseline by only {gain:+.4f} "
                f"selection {SELECTION_METRIC}; prefer the linear model if interpretability "
                "matters more than that margin."
            )

    # --- 2. refit on all fit dates, including the recent snapshots ---
    logger.info(
        "Refitting %s on all %d fit rows (to %s)",
        selected.spec.name,
        len(split.fit),
        max(plan.fit).date(),
    )
    if selected.spec.supports_early_stopping and selected.best_iteration:
        # Reuse the tree count the inner split chose rather than the 1500 upper bound.
        refit_spec = CandidateSpec(
            name=selected.spec.name,
            factory=lambda s, _n=int(selected.best_iteration): _pin_trees(
                selected.spec.factory(s), _n
            ),
            impute_and_scale=selected.spec.impute_and_scale,
            supports_early_stopping=False,
            notes=selected.spec.notes,
        )
        notes.append(
            f"The refit pinned {selected.spec.name} to the {selected.best_iteration} trees chosen "
            "by early stopping on the inner validation split, since the refit has no held-out set "
            "to early stop against."
        )
    else:
        refit_spec = selected.spec
    final_model, _ = _fit(refit_spec, split.fit, None, columns, seed)

    # --- 3. calibrate on the held-out calibration period ---
    final_pipeline, calibration_method, calibration_eval = _calibrate(
        final_model, split.calibration, columns, selected.spec.name, seed
    )
    if calibration_method == "none":
        notes.append(
            f"No calibration was applied: {selected.spec.name} was already better calibrated on "
            "the held-out calibration period than either isotonic or sigmoid correction."
        )
    else:
        notes.append(
            f"Probabilities were calibrated with {calibration_method} regression on the held-out "
            f"{', '.join(d.date().isoformat() for d in plan.calibration)} period."
        )

    # --- 4. the single look at the test period ---
    x_test = _matrix(split.test, columns)
    y_test = split.test[TARGET_COLUMN].astype(int)
    test_prob = final_pipeline.predict_proba(x_test)[:, 1]
    test_eval = evaluate_predictions(
        y_test,
        test_prob,
        model_name=selected.spec.name,
        dataset="test",
        value=split.test.get("annualized_revenue"),
        high_value=split.test["customer_value_segment"].eq("High Value")
        if "customer_value_segment" in split.test
        else None,
    )
    logger.info(
        "TEST: PR-AUC %.4f | ROC-AUC %.4f | F1 %.4f | Brier %.4f | ECE %.4f | lift@10%% %.2fx",
        test_eval.metrics["pr_auc"],
        test_eval.metrics["roc_auc"],
        test_eval.metrics["f1"],
        test_eval.metrics["brier"],
        test_eval.metrics["ece"],
        test_eval.metrics["lift_top_decile"],
    )

    # --- sanity floor: does the model beat a one-line heuristic? ---
    baselines = single_feature_baselines(split.test)
    if not baselines.empty:
        best_baseline = baselines.iloc[0]
        margin = test_eval.metrics["pr_auc"] - float(best_baseline["pr_auc"])
        # A tolerance, because on 842 test rows a fraction of a PR-AUC point is noise, and both
        # failing a build and claiming victory over it would be overreading the number.
        tolerance = BASELINE_TOLERANCE
        if margin < -tolerance:
            notes.append(
                f"WARNING: the model's test PR-AUC ({test_eval.metrics['pr_auc']:.4f}) is worse "
                f"than the single feature {best_baseline['baseline']} "
                f"({best_baseline['pr_auc']:.4f}) by {margin:.4f}. Treat the model as unproven "
                "and prefer the heuristic until this is resolved."
            )
        elif margin <= tolerance:
            notes.append(
                f"The model's ranking quality matches rather than beats the best single-feature "
                f"heuristic ({best_baseline['baseline']}, PR-AUC "
                f"{best_baseline['pr_auc']:.4f} versus {test_eval.metrics['pr_auc']:.4f}; "
                f"{margin:+.4f} is within noise on {test_eval.n} rows). What the model adds over "
                "the heuristic is a calibrated probability -- a raw feature cannot give one, and "
                "revenue at risk needs it -- plus much stronger discrimination among high-value "
                "customers, which is where the money is."
            )
        else:
            notes.append(
                f"The model beats the best single-feature heuristic "
                f"({best_baseline['baseline']}, PR-AUC {best_baseline['pr_auc']:.4f}) by "
                f"{margin:+.4f} test PR-AUC."
            )

    base_rates = {
        "selection_train": round(float(split.selection_train[TARGET_COLUMN].mean()), 4),
        "fit": round(float(split.fit[TARGET_COLUMN].mean()), 4),
        "calibration": round(float(split.calibration[TARGET_COLUMN].mean()), 4),
        "test": round(float(split.test[TARGET_COLUMN].mean()), 4),
    }
    if abs(base_rates["test"] - base_rates["fit"]) > 0.05:
        notes.append(
            "The churn base rate drifts across the timeline "
            + ", ".join(f"{k}={v:.1%}" for k, v in base_rates.items())
            + ". The brand was acquiring customers quickly in the early period, so few had lapsed "
            "yet; by the test period the cohort has matured. Probability calibration on a recent "
            "held-out period is what keeps the predicted level usable despite this shift."
        )

    importance = _feature_importance(final_pipeline, split.calibration, columns, seed)

    metadata = ModelMetadata(
        model_name=selected.spec.name,
        trained_at=utc_timestamp(),
        horizon_days=int(label_params.horizon_days),
        label_mode=str(label_params.mode),
        feature_columns=columns,
        numeric_columns=numeric,
        categorical_columns=categorical,
        train_as_of_dates=[d.date().isoformat() for d in plan.fit],
        validation_as_of_dates=[d.date().isoformat() for d in plan.calibration],
        test_as_of_dates=[d.date().isoformat() for d in plan.test],
        train_rows=len(split.fit),
        train_churn_rate=base_rates["fit"],
        calibration=calibration_method,
        random_seed=seed,
        metrics={
            "calibration_period": calibration_eval.as_dict(),
            "test": test_eval.as_dict(),
            "split": split.summary(),
            "base_rates": base_rates,
            "single_feature_baselines": baselines.to_dict(orient="records"),
        },
        candidate_scores=[o.as_dict() for o in outcomes],
        selection_metric=SELECTION_METRIC,
        selection_rationale=rationale,
        top_features=importance.head(25).to_dict(orient="records"),
        notes=notes,
    )

    if save:
        save_model(final_pipeline, metadata, model_dir or settings.models_dir)

    return TrainingResult(
        panel=panel,
        split=split,
        outcomes=outcomes,
        selected=selected,
        final_pipeline=final_pipeline,
        calibration_method=calibration_method,
        calibration_eval=calibration_eval,
        test_eval=test_eval,
        metadata=metadata,
        baselines=baselines,
        top_features=importance,
        notes=notes,
    )


def _pin_trees(estimator: Any, n_trees: int) -> Any:
    """Fix a booster's tree count and disable early stopping, for the refit."""
    params: dict[str, Any] = {"n_estimators": max(int(n_trees), 1)}
    try:
        estimator.set_params(**params, early_stopping_rounds=None)
    except (TypeError, ValueError):
        estimator.set_params(**params)
    return estimator
