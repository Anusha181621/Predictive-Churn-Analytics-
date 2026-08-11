"""SHAP value computation for the saved churn model.

Three problems have to be solved before SHAP values are usable here, and each is easy to get
subtly wrong.

**Unwrapping.** The persisted model is a ``CalibratedClassifierCV`` wrapping a ``FrozenEstimator``
wrapping a ``Pipeline(preprocessor, LGBMClassifier)``. ``TreeExplainer`` needs the booster itself and
a matrix in the booster's own feature space, so the wrappers are peeled off and the preprocessor is
applied explicitly.

**The calibration layer.** SHAP explains the *uncalibrated* model, because that is where the trees
are. Contributions are therefore on the log-odds scale of the margin before calibration, while the
probability reported to the business is after it. This does not invalidate the explanation:
calibration here is a monotone transform, so it cannot reorder contributions or flip a sign — the
*ranking* and *direction* of every driver carry over exactly. What does not carry over is the
arithmetic: the contributions sum to the uncalibrated margin, not to the calibrated probability. The
outputs say so rather than implying a false additivity.

**One-hot fragmentation.** The preprocessor expands ``preferred_category`` into one column per
category, so raw SHAP output would report five weak drivers instead of one real one. Contributions
are summed back onto the original feature, which is valid because SHAP values are additive.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from src.models.registry import SavedModel
from src.utils.logging_config import get_logger

__all__ = ["ShapResult", "compute_shap_values", "unwrap_pipeline"]

logger = get_logger(__name__)


def unwrap_pipeline(model: Any) -> tuple[Any, Any]:
    """Peel calibration/freezing wrappers off and return ``(preprocessor, tree_model)``.

    Raises ``TypeError`` if no tree model can be reached, rather than falling back to a slow
    model-agnostic explainer that would silently take minutes and produce approximate values.
    """
    current = model
    for _ in range(6):  # generous bound; the real chain is three deep
        if hasattr(current, "named_steps"):
            steps = current.named_steps
            if "prep" in steps and "model" in steps:
                return steps["prep"], steps["model"]
        if hasattr(current, "calibrated_classifiers_") and current.calibrated_classifiers_:
            current = current.calibrated_classifiers_[0].estimator
            continue
        if hasattr(current, "estimator"):
            current = current.estimator
            continue
        break
    raise TypeError(
        f"could not reach a Pipeline(prep, model) inside {type(model).__name__}; "
        "TreeExplainer needs the booster and its own feature space"
    )


def _map_expanded_to_original(
    expanded: list[str], original: list[str]
) -> dict[str, str]:
    """Map post-transform column names back to the feature they came from.

    Numeric columns pass through with their own name; ``OneHotEncoder`` emits
    ``<column>_<category>``. Longest-prefix matching handles both, and preferring the longest match
    stops ``category`` from claiming ``category_diversity``'s columns.
    """
    by_length = sorted(original, key=len, reverse=True)
    mapping: dict[str, str] = {}
    for name in expanded:
        if name in original:
            mapping[name] = name
            continue
        match = next((column for column in by_length if name.startswith(f"{column}_")), None)
        mapping[name] = match if match is not None else name
    return mapping


@dataclass
class ShapResult:
    """SHAP contributions aggregated onto the original feature columns."""

    #: One row per customer, one column per original feature. Log-odds scale.
    contributions: pd.DataFrame
    #: The feature values the contributions explain, same index and columns.
    values: pd.DataFrame
    #: Model output for the average customer, on the same log-odds scale.
    base_value: float
    #: Contributions before one-hot aggregation, for auditing.
    expanded: pd.DataFrame = field(repr=False, default_factory=pd.DataFrame)
    #: How the expanded columns were folded together.
    expansion_map: dict[str, str] = field(repr=False, default_factory=dict)

    @property
    def customer_ids(self) -> pd.Index:
        return self.contributions.index

    def global_importance(self) -> pd.Series:
        """Mean absolute contribution per feature -- the standard SHAP importance."""
        return self.contributions.abs().mean().sort_values(ascending=False)

    def mean_contribution(self) -> pd.Series:
        """Mean signed contribution: does this feature push the cohort towards churn on average?"""
        return self.contributions.mean()

    def direction(self) -> pd.Series:
        """Spearman correlation between each feature's value and its own contribution.

        Positive means higher values push towards churn. Spearman rather than Pearson because the
        relationship a tree learns is monotone-ish but rarely linear, and rank correlation captures
        that without assuming a shape.
        """
        out: dict[str, float] = {}
        for column in self.contributions.columns:
            values = pd.to_numeric(self.values[column], errors="coerce")
            if values.notna().sum() < 3 or values.nunique(dropna=True) < 2:
                # Categorical or constant: a value-versus-contribution correlation is meaningless.
                out[column] = float("nan")
                continue
            out[column] = float(values.corr(self.contributions[column], method="spearman"))
        return pd.Series(out)


def compute_shap_values(
    model: SavedModel, features: pd.DataFrame, *, id_column: str = "customer_id"
) -> ShapResult:
    """Compute per-customer SHAP contributions for ``features``.

    ``features`` is the full feature table from :func:`~src.features.build_customer_features`; the
    model's stored feature contract selects and orders the columns it was trained on.
    """
    import shap

    preprocessor, tree_model = unwrap_pipeline(model.pipeline)
    matrix = model.align(features)
    index = pd.Index(features[id_column], name=id_column)

    transformed = preprocessor.transform(matrix)
    expanded_names = list(preprocessor.get_feature_names_out())

    explainer = shap.TreeExplainer(tree_model)
    raw = explainer.shap_values(transformed)

    # LightGBM binary classifiers have returned both a single array and a two-element list across
    # versions; normalise to the positive class either way.
    array = np.asarray(raw)
    if array.ndim == 3:
        array = array[:, :, 1] if array.shape[2] == 2 else array[:, :, 0]
    elif isinstance(raw, list):  # pragma: no cover - older shap
        array = np.asarray(raw[1] if len(raw) == 2 else raw[0])

    base = np.asarray(explainer.expected_value).ravel()
    base_value = float(base[-1] if base.size > 1 else base[0])

    expanded = pd.DataFrame(array, index=index, columns=expanded_names)

    original = list(matrix.columns)
    mapping = _map_expanded_to_original(expanded_names, original)
    # Additivity is what makes this fold valid: a feature's total contribution is the sum of its
    # one-hot columns' contributions.
    contributions = expanded.T.groupby(pd.Series(mapping)).sum().T
    contributions = contributions.reindex(columns=original, fill_value=0.0)

    values = matrix.copy()
    values.index = index

    folded = len(expanded_names) - len(original)
    logger.info(
        "SHAP: %d customers x %d features (folded %d one-hot columns back onto their source "
        "features); base value %.4f on the uncalibrated log-odds scale",
        len(contributions),
        contributions.shape[1],
        folded,
        base_value,
    )
    return ShapResult(
        contributions=contributions,
        values=values,
        base_value=base_value,
        expanded=expanded,
        expansion_map=mapping,
    )
