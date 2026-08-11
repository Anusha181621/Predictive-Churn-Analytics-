"""Retention decision layer: revenue at risk, segmentation, prioritisation and recommendations.

Entry point::

    from src.retention import build_retention_layer, write_retention_outputs

    result = build_retention_layer()
    result.scores            # per-customer revenue at risk and opportunity score
    result.recommendations   # per-customer action, channel, category, SKU, offer, reason
    write_retention_outputs(result)

One thing to read before using the numbers. **Retention propensity is an assumption, not an
estimate.** Measuring intervention uplift needs a campaign log and an untreated control group, and
this dataset has neither, so it cannot be learned here. The base rate and its behavioural
multipliers are stated openly in :class:`~src.retention.params.RetentionParams`, every derived
column is flagged as assumption-dependent, and ``outputs/retention_assumptions.json`` ships
alongside the CSVs. ``revenue_at_risk`` is deliberately kept free of the assumption, so a business
that rejects the propensity figures can still use the exposure number.
"""

from src.retention.params import PRIORITY_BANDS, RetentionParams
from src.retention.pipeline import (
    RECOMMENDATIONS_FILENAME,
    RETENTION_SCORES_FILENAME,
    RetentionResult,
    build_retention_layer,
    write_retention_outputs,
)
from src.retention.recommendations import (
    ACTIONS,
    RecommendationInputs,
    build_recommendation_inputs,
    build_recommendations,
)
from src.retention.scoring import assign_priority, build_retention_propensity, build_scores
from src.retention.segments import SEGMENT_FLAGS, SEGMENTS, build_segments
from src.retention.value import build_expected_revenue

__all__ = [
    "ACTIONS",
    "PRIORITY_BANDS",
    "RECOMMENDATIONS_FILENAME",
    "RETENTION_SCORES_FILENAME",
    "RecommendationInputs",
    "RetentionParams",
    "RetentionResult",
    "SEGMENTS",
    "SEGMENT_FLAGS",
    "assign_priority",
    "build_expected_revenue",
    "build_recommendation_inputs",
    "build_recommendations",
    "build_retention_layer",
    "build_retention_propensity",
    "build_scores",
    "build_segments",
    "write_retention_outputs",
]
