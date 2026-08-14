"""Hyperparameter search on the embargoed inner validation split.

The hyperparameters in :mod:`src.models.candidates` were hand-picked to be conservative and were
never actually searched. That was a reasonable starting point and it left a real problem behind:
with the enlarged feature table LightGBM's early stopping fires after **four** trees, so the
"gradient-boosted model" being shipped is a handful of stumps. Nobody would have noticed from the
metrics alone -- four stumps still score respectably -- which is exactly why the search belongs in
the codebase rather than in someone's notebook.

Where the search is scored, and why it matters
----------------------------------------------
Candidates are trained on ``selection_train`` and scored on ``selection_validation``, the same
embargoed inner split :func:`src.models.train.train_churn_model` uses to pick the model family.
That is the only defensible scoreboard available: the calibration period is reserved for the
probability mapping and the decision threshold, and the test period is looked at exactly once, at
the end. Searching against test -- or against calibration, which then also sets the threshold --
would quietly convert a held-out set into a training set and inflate every number downstream.

Random search rather than a grid: with eight interacting parameters a grid coarse enough to finish
tests a worse set of values than random sampling of the same size, and the budget here is a fixed
number of fits so a run's cost is predictable.

Class weighting is included in the search space deliberately, against the reasoning in
``candidates.py``. That module leaves reweighting off to protect the probability *level*, because
revenue at risk multiplies those probabilities by customer value. The objection is sound but the
pipeline already answers it: probabilities are recalibrated on a held-out period afterwards, which
restores the level whatever the weighting did to it. So the ranking gain is worth measuring rather
than assuming away -- and if it does not help, the search will simply not select it.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from src.models.candidates import CandidateSpec
from src.models.dataset import TARGET_COLUMN
from src.models.evaluate import evaluate_predictions
from src.utils.logging_config import get_logger

__all__ = ["TuningResult", "search_hyperparameters", "SEARCH_SPACES"]

logger = get_logger(__name__)

#: Sampling ranges per candidate. Values are either a list (sampled uniformly) or a
#: ``(low, high, kind)`` triple sampled uniformly on a linear or log scale.
SEARCH_SPACES: dict[str, dict[str, Any]] = {
    "lightgbm": {
        "num_leaves": [7, 15, 31, 63, 127],
        "max_depth": [3, 4, 5, 6, 8, -1],
        "min_child_samples": [5, 10, 20, 30, 50, 80],
        "learning_rate": (0.01, 0.2, "log"),
        "subsample": (0.6, 1.0, "linear"),
        "colsample_bytree": (0.4, 1.0, "linear"),
        "reg_alpha": (1e-3, 10.0, "log"),
        "reg_lambda": (1e-3, 10.0, "log"),
        "class_weight": [None, "balanced"],
    },
    "xgboost": {
        "max_depth": [2, 3, 4, 5, 6, 8],
        "min_child_weight": [1, 5, 10, 20, 40],
        "learning_rate": (0.01, 0.2, "log"),
        "subsample": (0.6, 1.0, "linear"),
        "colsample_bytree": (0.4, 1.0, "linear"),
        "reg_alpha": (1e-3, 10.0, "log"),
        "reg_lambda": (1e-3, 10.0, "log"),
        "scale_pos_weight": (0.5, 4.0, "log"),
    },
    "random_forest": {
        "n_estimators": [200, 400, 800],
        "max_depth": [4, 6, 8, 12, None],
        "min_samples_leaf": [1, 5, 10, 20, 40],
        "max_features": ["sqrt", "log2", 0.3, 0.5],
        "class_weight": [None, "balanced"],
    },
    "logistic_regression": {
        "C": (1e-3, 10.0, "log"),
        "class_weight": [None, "balanced"],
    },
}


@dataclass
class TuningResult:
    """The winning parameters for one candidate, plus the full trial log."""

    model: str
    best_params: dict[str, Any]
    best_score: float
    baseline_score: float
    metric: str
    trials: list[dict[str, Any]] = field(default_factory=list)
    seconds: float = 0.0

    @property
    def improvement(self) -> float:
        return self.best_score - self.baseline_score

    def as_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "metric": self.metric,
            "baseline_score": round(self.baseline_score, 6),
            "best_score": round(self.best_score, 6),
            "improvement": round(self.improvement, 6),
            "best_params": {k: _jsonable(v) for k, v in self.best_params.items()},
            "trials_run": len(self.trials),
            "seconds": round(self.seconds, 1),
            # Only the leaders are persisted: a full log of 60 trials would dominate the metrics
            # file without telling a reader anything the top few do not.
            "top_trials": [
                {"score": round(t["score"], 6), "params": {k: _jsonable(v) for k, v in t["params"].items()}}
                for t in sorted(self.trials, key=lambda t: -t["score"])[:5]
            ],
        }


def _jsonable(value: Any) -> Any:
    """Coerce numpy scalars so the metadata round-trips through JSON."""
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value


def _sample(space: dict[str, Any], rng: np.random.Generator) -> dict[str, Any]:
    """Draw one parameter set from ``space``."""
    drawn: dict[str, Any] = {}
    for name, spec in space.items():
        if isinstance(spec, list):
            drawn[name] = spec[int(rng.integers(len(spec)))]
        else:
            low, high, kind = spec
            if kind == "log":
                drawn[name] = float(np.exp(rng.uniform(np.log(low), np.log(high))))
            else:
                drawn[name] = float(rng.uniform(low, high))
    return drawn


def search_hyperparameters(
    spec: CandidateSpec,
    split: Any,
    columns: Sequence[str],
    *,
    seed: int,
    iterations: int = 40,
    metric: str = "pr_auc",
) -> TuningResult:
    """Random-search ``spec``'s hyperparameters against the inner validation split.

    Returns the hand-set defaults untouched if no sampled configuration beats them, so a search
    that finds nothing degrades to the previous behaviour rather than shipping a worse model.
    """
    # Imported here rather than at module scope to avoid a circular import: `train` imports this
    # module for the search, and this module needs `train`'s fitting helper.
    from src.models.train import _fit

    space = SEARCH_SPACES.get(spec.name)
    if not space:
        raise ValueError(f"no search space defined for candidate {spec.name!r}")

    y_validation = split.selection_validation[TARGET_COLUMN].astype(int)
    started = time.time()

    def evaluate(overrides: dict[str, Any]) -> float:
        tuned = CandidateSpec(
            name=spec.name,
            factory=lambda s, _o=overrides: _apply(spec.factory(s), _o),
            impute_and_scale=spec.impute_and_scale,
            supports_early_stopping=spec.supports_early_stopping,
            notes=spec.notes,
        )
        try:
            pipeline, _ = _fit(tuned, split.selection_train, split.selection_validation, columns, seed)
        except Exception as exc:  # a bad corner of the space must not abort the search
            logger.debug("  trial failed (%s): %s", type(exc).__name__, exc)
            return float("-inf")
        probability = pipeline.predict_proba(_matrix_for(split.selection_validation, columns))[:, 1]
        result = evaluate_predictions(
            y_validation, probability, model_name=spec.name, dataset="selection_validation"
        )
        score = result.metrics[metric]
        return float(score) if score == score else float("-inf")

    baseline = evaluate({})
    logger.info(
        "Tuning %s: %d trials against selection-validation %s (baseline %.4f)",
        spec.name,
        iterations,
        metric,
        baseline,
    )

    rng = np.random.default_rng(seed)
    trials: list[dict[str, Any]] = [{"params": {}, "score": baseline}]
    best_params: dict[str, Any] = {}
    best_score = baseline

    for trial in range(iterations):
        params = _sample(space, rng)
        score = evaluate(params)
        trials.append({"params": params, "score": score})
        if score > best_score:
            best_score, best_params = score, params
            logger.info("  trial %2d: %s %.4f (new best)", trial + 1, metric, score)

    result = TuningResult(
        model=spec.name,
        best_params=best_params,
        best_score=best_score,
        baseline_score=baseline,
        metric=metric,
        trials=trials,
        seconds=time.time() - started,
    )
    if not best_params:
        logger.info(
            "  no sampled configuration beat the hand-set defaults (%.4f); keeping them",
            baseline,
        )
    else:
        logger.info(
            "  best %s %.4f (%+.4f over the defaults) in %.0fs",
            metric,
            best_score,
            result.improvement,
            result.seconds,
        )
    return result


def _apply(estimator: Any, overrides: dict[str, Any]) -> Any:
    """Set ``overrides`` on an estimator, ignoring parameters it does not accept."""
    if not overrides:
        return estimator
    accepted = set(estimator.get_params().keys())
    usable = {k: v for k, v in overrides.items() if k in accepted}
    estimator.set_params(**usable)
    return estimator


def _matrix_for(frame: Any, columns: Sequence[str]) -> Any:
    from src.models.train import _matrix

    return _matrix(frame, columns)
