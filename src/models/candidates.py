"""The candidate models.

Four estimators, each with a reason to be in the comparison rather than for the sake of a longer
table:

* **Logistic regression** -- the linear baseline. If a gradient-boosted ensemble cannot beat it, the
  extra complexity is not earning its keep and should be dropped.
* **Random forest** -- captures interactions with almost no tuning, and its probability estimates
  are usually better calibrated out of the box than a boosted model's.
* **LightGBM** and **XGBoost** -- the strong learners. Both handle NaN natively, which matters here
  because NaN is meaningful in this feature table.

Hyperparameters are deliberately conservative. The training panel is a few thousand rows against
~130 features, so the failure mode is overfitting, not underfitting: depth is capped, leaf minimums
are high, and both boosters subsample rows and columns. LightGBM and XGBoost also get early
stopping against the validation period, which is the honest way to pick tree count.

Class weighting is left off by default. The churn rate on this data is moderate rather than
extreme, so reweighting buys little ranking quality while actively distorting the predicted
probabilities -- and those probabilities are multiplied by customer value downstream to produce
revenue at risk, so their calibration is a business requirement, not a nicety.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from src.utils.logging_config import get_logger

__all__ = ["CandidateSpec", "candidate_specs", "available_candidates"]

logger = get_logger(__name__)


@dataclass(frozen=True)
class CandidateSpec:
    """A model to evaluate."""

    name: str
    #: Builds the bare estimator. Called with the random seed.
    factory: Callable[[int], Any]
    #: Whether the estimator needs imputation and scaling (see :mod:`src.models.preprocessing`).
    impute_and_scale: bool
    #: Whether it supports ``eval_set`` early stopping.
    supports_early_stopping: bool = False
    notes: str = ""
    extra: dict[str, Any] = field(default_factory=dict)


def _logistic(seed: int) -> Any:
    return LogisticRegression(
        # Strong regularisation: ~130 correlated features against a few thousand rows.
        # `penalty` is left at its default L2 -- passing it explicitly is deprecated from
        # scikit-learn 1.8.
        C=0.1,
        solver="lbfgs",
        max_iter=2000,
        random_state=seed,
    )


def _random_forest(seed: int) -> Any:
    return RandomForestClassifier(
        n_estimators=400,
        max_depth=8,
        min_samples_leaf=20,
        max_features="sqrt",
        n_jobs=-1,
        random_state=seed,
    )


def _lightgbm(seed: int) -> Any:
    from lightgbm import LGBMClassifier

    return LGBMClassifier(
        n_estimators=1500,          # an upper bound; early stopping chooses the real count
        learning_rate=0.03,
        num_leaves=15,
        max_depth=5,
        min_child_samples=30,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.7,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=seed,
        n_jobs=-1,
        verbosity=-1,
    )


def _xgboost(seed: int) -> Any:
    from xgboost import XGBClassifier

    return XGBClassifier(
        n_estimators=1500,
        learning_rate=0.03,
        max_depth=4,
        min_child_weight=10,
        subsample=0.8,
        colsample_bytree=0.7,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=seed,
        n_jobs=-1,
        eval_metric="aucpr",
        early_stopping_rounds=100,
        tree_method="hist",
    )


_SPECS: tuple[CandidateSpec, ...] = (
    CandidateSpec(
        name="logistic_regression",
        factory=_logistic,
        impute_and_scale=True,
        notes="Linear baseline. If nothing beats this, the complexity is not paying for itself.",
    ),
    CandidateSpec(
        name="random_forest",
        factory=_random_forest,
        impute_and_scale=True,
        notes="Bagged trees; typically well-calibrated without post-hoc correction.",
    ),
    CandidateSpec(
        name="lightgbm",
        factory=_lightgbm,
        impute_and_scale=False,
        supports_early_stopping=True,
        notes="Gradient boosting, native NaN handling, early stopped on the validation period.",
    ),
    CandidateSpec(
        name="xgboost",
        factory=_xgboost,
        impute_and_scale=False,
        supports_early_stopping=True,
        notes="Gradient boosting, native NaN handling, early stopped on the validation period.",
    ),
)


def candidate_specs() -> tuple[CandidateSpec, ...]:
    """Every candidate, whether or not its library is installed."""
    return _SPECS


def available_candidates() -> list[CandidateSpec]:
    """Candidates whose libraries import successfully.

    XGBoost and LightGBM are optional at runtime: the comparison degrades to whatever is installed
    rather than failing outright, and the report records what was skipped so a missing library never
    masquerades as a model that simply lost.
    """
    usable: list[CandidateSpec] = []
    for spec in _SPECS:
        try:
            spec.factory(0)
        except ImportError as exc:
            logger.warning("Skipping candidate %s: %s", spec.name, exc)
            continue
        usable.append(spec)
    return usable
