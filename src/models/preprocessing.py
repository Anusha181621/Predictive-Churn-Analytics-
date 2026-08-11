"""Feature-matrix construction: what the model is allowed to see, and how it is encoded.

Three categories of column are excluded, and the second is the one that matters most.

**Identifiers and label columns.** ``customer_id``, the target, and the label bookkeeping
(``purchases_in_window``, ``days_to_next_purchase``, ``outcome_window_end``) are all derived from
data *after* the as-of date. They exist for auditing; letting any of them reach the model would be
direct target leakage.

**Period markers.** Anything that says *which snapshot* a row came from is excluded, even though it
contains no future information. ``as_of_date`` is the obvious one, but the subtle ones are
``high_value_threshold`` and ``medium_value_threshold``: they are cohort-wide constants recomputed
at every as-of date, so they act as a fingerprint of the period. A tree can split on
"threshold > 1400" to identify the 2025 snapshots and then apply that period's base churn rate,
which scores beautifully in validation and is worthless in production. Raw dates
(``first_purchase_date``, ``last_purchase_date``, ``registration_date``) are excluded for the same
reason -- their information already survives as ``recency_days``, ``customer_tenure_days`` and
``days_since_registration``, which are relative to the as-of date and therefore portable across
periods.

**Free text.** ``segment_reason`` is a human-readable sentence, not a feature.

Everything else -- 130-odd behavioural features -- goes in. Numeric columns are imputed and scaled
only where the estimator needs it; categoricals are one-hot encoded with ``handle_unknown="ignore"``
so a category first seen in the test period cannot crash scoring.

On missing values
-----------------
NaN is meaningful in this feature table, not dirt: ``revenue_growth`` is NaN precisely when a
customer had no baseline window, and ``seasonal_customer_score`` is NaN when there is not enough
history to score. Gradient-boosted trees read NaN natively and learn from it, so
:func:`build_preprocessor` leaves it alone for them and imputes only for the estimators that cannot
cope (logistic regression), where the median plus a missing-indicator column preserves the fact
that the value was absent.
"""

from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.models.dataset import AS_OF_COLUMN, TARGET_COLUMN
from src.utils.logging_config import get_logger

__all__ = [
    "EXCLUDED_COLUMNS",
    "split_feature_columns",
    "build_preprocessor",
    "feature_matrix",
    "expanded_feature_names",
]

logger = get_logger(__name__)

#: Columns never offered to the model, with the reason grouped by kind.
EXCLUDED_COLUMNS: frozenset[str] = frozenset(
    {
        # identifiers
        "customer_id",
        # the target and its bookkeeping -- all derived from after the as-of date
        TARGET_COLUMN,
        "purchases_in_window",
        "days_to_next_purchase",
        "outcome_window_end",
        "horizon_days",
        # Label-module bookkeeping. Not leakage (it is computed from history), but it only exists
        # on the training panel, never on a scoring frame -- and the feature layer already carries
        # the same signal as `is_new_buyer`. Including it would train a model that cannot score.
        "is_new_at_as_of",
        # period markers: no future information, but they fingerprint the snapshot
        AS_OF_COLUMN,
        "high_value_threshold",
        "medium_value_threshold",
        "trend_window_days",
        # raw dates, superseded by as-of-relative equivalents
        "registration_date",
        "first_purchase_date",
        "last_purchase_date",
        # free text
        "segment_reason",
    }
)


def split_feature_columns(frame: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Return ``(numeric, categorical)`` feature column names, in deterministic order.

    Booleans are treated as numeric: 0/1 is exactly what every estimator here wants, and one-hot
    encoding them would double the width for no gain.
    """
    candidates = [column for column in frame.columns if column not in EXCLUDED_COLUMNS]

    numeric: list[str] = []
    categorical: list[str] = []
    for column in candidates:
        series = frame[column]
        if pd.api.types.is_bool_dtype(series):
            numeric.append(column)
        elif pd.api.types.is_numeric_dtype(series):
            numeric.append(column)
        elif pd.api.types.is_datetime64_any_dtype(series):
            # A datetime that escaped EXCLUDED_COLUMNS is a bug rather than a feature: it would be
            # a period marker. Refuse it loudly instead of silently casting it to an integer.
            raise ValueError(
                f"column {column!r} is a datetime and would act as a period marker; add it to "
                "EXCLUDED_COLUMNS or convert it to an as-of-relative number"
            )
        else:
            categorical.append(column)

    return sorted(numeric), sorted(categorical)


def build_preprocessor(
    frame: pd.DataFrame, *, impute_and_scale: bool
) -> tuple[ColumnTransformer, list[str], list[str]]:
    """Build the column transformer for a feature frame.

    Parameters
    ----------
    impute_and_scale:
        ``True`` for estimators that cannot handle NaN and are scale-sensitive (logistic
        regression). ``False`` for tree ensembles, which read NaN natively -- and for which
        imputing would destroy the signal that a value was missing.
    """
    numeric, categorical = split_feature_columns(frame)

    if impute_and_scale:
        numeric_pipeline: Pipeline | str = Pipeline(
            [
                # add_indicator keeps "this was missing" as its own column, so imputation does not
                # erase the information that NaN carried.
                ("impute", SimpleImputer(strategy="median", add_indicator=True)),
                ("scale", StandardScaler()),
            ]
        )
    else:
        numeric_pipeline = "passthrough"

    categorical_pipeline = Pipeline(
        [
            ("impute", SimpleImputer(strategy="constant", fill_value="__missing__")),
            # handle_unknown="ignore" so a category first seen at scoring time cannot crash a run.
            ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False, min_frequency=5)),
        ]
    )

    transformer = ColumnTransformer(
        [
            ("numeric", numeric_pipeline, numeric),
            ("categorical", categorical_pipeline, categorical),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return transformer, numeric, categorical


def feature_matrix(frame: pd.DataFrame) -> pd.DataFrame:
    """The model-visible columns of ``frame``, in deterministic order."""
    numeric, categorical = split_feature_columns(frame)
    ordered = numeric + categorical
    matrix = frame[ordered].copy()
    # Object-dtype categoricals must be strings for OneHotEncoder; pandas NA becomes the
    # sentinel the imputer expects.
    for column in categorical:
        matrix[column] = matrix[column].astype("object").where(matrix[column].notna(), None)
    for column in numeric:
        if pd.api.types.is_bool_dtype(matrix[column]):
            matrix[column] = matrix[column].astype("float64")
    return matrix


def expanded_feature_names(transformer: ColumnTransformer) -> list[str]:
    """Post-transform feature names, for feature-importance reporting."""
    try:
        return list(transformer.get_feature_names_out())
    except Exception:  # pragma: no cover - depends on sklearn internals
        logger.warning("Could not recover expanded feature names from the transformer")
        return []
