"""Customer-level feature engineering, computed directly from the CSV files.

The entry point is :func:`~src.features.builder.build_customer_features`, which returns exactly
one row per Customer ID as of a prediction date::

    from src.features import build_customer_features

    result = build_customer_features(as_of_date="2025-12-31")
    result.features        # one row per customer
    result.feature_count
    result.issues          # calculation caveats worth reporting

Leakage control lives in :mod:`src.features.context`, which clips transactions *and* returns to
the as-of date once; every feature module reads only from that clipped context. Thresholds live
in :mod:`src.features.params`.
"""

from src.features.builder import (
    FEATURE_GROUPS,
    FeatureBuildResult,
    build_customer_features,
)
from src.features.context import FeatureContext, build_context, resolve_as_of_date
from src.features.params import FeatureParams

__all__ = [
    "FEATURE_GROUPS",
    "FeatureBuildResult",
    "FeatureContext",
    "FeatureParams",
    "build_context",
    "build_customer_features",
    "resolve_as_of_date",
]
