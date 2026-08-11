"""One call from the CSV files to retention scores and recommendations.

    data/*.csv -> features (as of a date) -> churn model -> expected revenue
               -> segments -> revenue at risk -> opportunity score -> recommendations

No database, no ingestion step, no manual preprocessing. Two artefacts:

``outputs/customer_retention_scores.csv``
    One row per customer: expected future revenue, revenue at risk, retention propensity, the
    opportunity score, its priority band, and the twelve segment flags.
``outputs/retention_recommendations.csv``
    One row per customer: action, channel, category, SKU, offer, reason, priority and the economics.

The transactions used to derive bestsellers and complementary categories are clipped to the
prediction date, for the same reason the features are: a recommendation built from next month's
bestsellers would be a leak wearing a different hat.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path

import pandas as pd

from src.config.settings import Settings, get_settings
from src.data.csv_loader import Datasets, load_all
from src.features.builder import build_customer_features
from src.features.params import FeatureParams
from src.models.predict import score_customers
from src.models.registry import SavedModel, load_model
from src.retention.params import RetentionParams
from src.retention.recommendations import build_recommendation_inputs, build_recommendations
from src.retention.scoring import build_scores
from src.retention.segments import SEGMENT_FLAGS, SEGMENTS, build_segments
from src.retention.value import build_expected_revenue
from src.utils.logging_config import get_logger
from src.utils.paths import ensure_dir

__all__ = [
    "RetentionResult",
    "RETENTION_SCORES_FILENAME",
    "RECOMMENDATIONS_FILENAME",
    "build_retention_layer",
    "write_retention_outputs",
]

logger = get_logger(__name__)

RETENTION_SCORES_FILENAME = "customer_retention_scores.csv"
RECOMMENDATIONS_FILENAME = "retention_recommendations.csv"

#: Columns of the scores artefact, in a business-readable order.
SCORE_COLUMNS = [
    "Customer ID",
    "Prediction date",
    "Churn probability",
    "Risk level",
    "Primary segment",
    "All segments",
    "Customer value",
    "Lifetime revenue",
    "Expected future revenue",
    "Revenue at risk",
    "Retention propensity (ASSUMED)",
    "Expected retained revenue",
    "Retention opportunity score",
    "Priority",
]

#: Columns of the recommendations artefact, in the order the brief lists them.
RECOMMENDATION_COLUMNS = [
    "Customer ID",
    "Prediction date",
    "Churn probability",
    "Risk level",
    "Primary segment",
    "Recommended action",
    "Recommended channel",
    "Recommended category",
    "Recommended product/SKU",
    "Recommended product",
    "Recommended offer",
    "Reason",
    "Priority",
    "Revenue at risk",
    "Expected retained revenue",
    "Campaign cost",
    "Expected ROI",
]


@dataclass
class RetentionResult:
    """Everything the retention layer produced."""

    as_of: pd.Timestamp
    params: RetentionParams
    scores: pd.DataFrame
    recommendations: pd.DataFrame
    detail: pd.DataFrame = field(repr=False, default_factory=pd.DataFrame)

    def summary(self) -> dict[str, object]:
        scores, recommendations = self.scores, self.recommendations
        targeted = recommendations[recommendations["Recommended action"].ne("Do Not Target")]
        return {
            "as_of_date": self.as_of.date().isoformat(),
            "customers": len(scores),
            "currency": self.params.currency,
            "revenue_horizon_days": self.params.revenue_horizon_days,
            "total_expected_future_revenue": round(
                float(scores["Expected future revenue"].sum()), 2
            ),
            "total_revenue_at_risk": round(float(scores["Revenue at risk"].sum()), 2),
            "total_expected_retained_revenue": round(
                float(scores["Expected retained revenue"].sum()), 2
            ),
            "mean_retention_propensity": round(
                float(scores["Retention propensity (ASSUMED)"].mean()), 4
            ),
            "customers_targeted": len(targeted),
            "customers_suppressed": len(recommendations) - len(targeted),
            "total_campaign_cost": round(float(targeted["Campaign cost"].sum()), 2),
            "campaign_expected_return": round(
                float(targeted["Expected retained revenue"].sum()), 2
            ),
            "campaign_roi": round(
                (
                    float(targeted["Expected retained revenue"].sum())
                    - float(targeted["Campaign cost"].sum())
                )
                / float(targeted["Campaign cost"].sum()),
                4,
            )
            if float(targeted["Campaign cost"].sum()) > 0
            else None,
            "primary_segment_counts": scores["Primary segment"].value_counts().to_dict(),
            "action_counts": recommendations["Recommended action"].value_counts().to_dict(),
            "priority_counts": scores["Priority"].value_counts().to_dict(),
            "assumptions": self.params.assumptions(),
            "policy_inputs": self.params.policy_inputs(),
        }


def build_retention_layer(
    data: Datasets | None = None,
    as_of_date: str | date | datetime | pd.Timestamp | None = None,
    *,
    model: SavedModel | None = None,
    model_dir: str | None = None,
    settings: Settings | None = None,
    feature_params: FeatureParams | None = None,
    params: RetentionParams | None = None,
    predictions: pd.DataFrame | None = None,
) -> RetentionResult:
    """Score, segment, value and recommend for every customer."""
    settings = settings or get_settings()
    data = data if data is not None else load_all()
    model = model or load_model(model_dir or settings.models_dir)
    params = params or RetentionParams(revenue_horizon_days=model.metadata.horizon_days)
    params.validate()

    as_of = (
        pd.Timestamp(as_of_date).normalize()
        if as_of_date is not None
        else pd.Timestamp(data.transactions["purchase_date"].max()).normalize()
    )

    features = build_customer_features(data, as_of_date=as_of, params=feature_params).features
    if predictions is None:
        predictions = score_customers(
            data, as_of_date=as_of, model=model, settings=settings, feature_params=feature_params
        )

    # --- join the features to the prediction, keyed on customer ---
    prediction_frame = predictions.rename(
        columns={
            "Customer ID": "customer_id",
            "Churn probability": "churn_probability",
            "Risk level": "risk_level",
            "Customer value": "customer_value_segment_predicted",
        }
    )[["customer_id", "churn_probability", "risk_level", "Prediction date"]]
    joined = features.merge(prediction_frame, on="customer_id", how="inner", validate="1:1")
    joined = joined.set_index("customer_id")

    # --- expected future revenue, from frequency x value with tenure guards ---
    expected_revenue = build_expected_revenue(features, params)

    # --- segments (need the churn probability, so after the join) ---
    segments = build_segments(joined, params)
    joined = joined.join(segments)

    # --- revenue at risk, propensity, opportunity score, priority ---
    scores = build_scores(joined, expected_revenue, params)
    joined = joined.join(scores[[c for c in scores.columns if c not in joined.columns]])

    # --- recommendations, from transactions clipped to the prediction date ---
    transactions_as_of = data.transactions[data.transactions["purchase_date"].le(as_of)]
    recommendation_inputs = build_recommendation_inputs(data.products, transactions_as_of)
    recommendations = build_recommendations(joined, recommendation_inputs, params)
    joined = joined.join(recommendations[["expected_roi", "campaign_cost", "recommended_action"]])

    prediction_date = as_of.date().isoformat()

    scores_out = pd.DataFrame(
        {
            "Customer ID": joined.index,
            "Prediction date": prediction_date,
            "Churn probability": joined["churn_probability"].round(6).to_numpy(),
            "Risk level": joined["risk_level"].to_numpy(),
            "Primary segment": joined["primary_segment"].to_numpy(),
            "All segments": joined["all_segments"].to_numpy(),
            "Customer value": joined["customer_value_segment"].to_numpy(),
            "Lifetime revenue": joined["lifetime_revenue"].round(2).to_numpy(),
            "Expected future revenue": joined["expected_future_revenue"].to_numpy(),
            "Revenue at risk": joined["revenue_at_risk"].to_numpy(),
            "Retention propensity (ASSUMED)": joined["retention_propensity"].to_numpy(),
            "Expected retained revenue": joined["expected_retained_revenue"].to_numpy(),
            "Retention opportunity score": joined["retention_opportunity_score"].to_numpy(),
            "Priority": joined["priority"].astype(str).to_numpy(),
        }
    )
    # Diagnostics after the headline columns: the projection's working, so the number is auditable.
    for column in (
        "expected_orders_per_year",
        "expected_average_order_value",
        "projected_annual_revenue",
        "historical_annual_revenue",
        "projection_vs_historical_ratio",
        "projection_capped",
        "tenure_floored",
    ):
        scores_out[column.replace("_", " ").capitalize()] = (
            expected_revenue[column].reindex(joined.index).to_numpy()
        )
    scores_out["Propensity basis (ASSUMED)"] = joined["propensity_basis"].to_numpy()
    scores_out["Retention opportunity percentile"] = joined[
        "retention_opportunity_percentile"
    ].to_numpy()
    for segment in SEGMENTS:
        scores_out[segment] = joined[SEGMENT_FLAGS[segment]].to_numpy()

    recommendations_out = pd.DataFrame(
        {
            "Customer ID": joined.index,
            "Prediction date": prediction_date,
            "Churn probability": joined["churn_probability"].round(6).to_numpy(),
            "Risk level": joined["risk_level"].to_numpy(),
            "Primary segment": joined["primary_segment"].to_numpy(),
            "Recommended action": recommendations["recommended_action"].to_numpy(),
            "Recommended channel": recommendations["recommended_channel"].to_numpy(),
            "Recommended category": recommendations["recommended_category"].to_numpy(),
            "Recommended product/SKU": recommendations["recommended_sku"].to_numpy(),
            "Recommended product": recommendations["recommended_product"].to_numpy(),
            "Recommended offer": recommendations["recommended_offer"].to_numpy(),
            "Reason": recommendations["reason"].to_numpy(),
            "Priority": joined["priority"].astype(str).to_numpy(),
            "Revenue at risk": joined["revenue_at_risk"].to_numpy(),
            "Expected retained revenue": recommendations["expected_retained_revenue"].to_numpy(),
            "Campaign cost": recommendations["campaign_cost"].to_numpy(),
            "Expected ROI": recommendations["expected_roi"].to_numpy(),
        }
    )
    recommendations_out["Suppressed action"] = recommendations["suppressed_action"].to_numpy()
    recommendations_out["ROI depends on an assumption"] = True

    return RetentionResult(
        as_of=as_of,
        params=params,
        scores=scores_out,
        recommendations=recommendations_out,
        detail=joined,
    )


def write_retention_outputs(
    result: RetentionResult,
    settings: Settings | None = None,
    *,
    scores_path: str | Path | None = None,
    recommendations_path: str | Path | None = None,
) -> dict[str, Path]:
    """Write both artefacts. Analytical outputs; ``data/`` is never touched."""
    settings = settings or get_settings()
    outputs = ensure_dir(settings.outputs_dir)

    targets = {
        "customer_retention_scores": Path(scores_path)
        if scores_path
        else outputs / RETENTION_SCORES_FILENAME,
        "retention_recommendations": Path(recommendations_path)
        if recommendations_path
        else outputs / RECOMMENDATIONS_FILENAME,
    }
    frames = {
        "customer_retention_scores": result.scores,
        "retention_recommendations": result.recommendations,
    }
    for name, path in targets.items():
        ensure_dir(path.parent)
        frames[name].to_csv(path, index=False, encoding="utf-8-sig", float_format="%.6g")
        logger.info("Wrote %s (%d rows)", path, len(frames[name]))

    # The assumption manifest travels with the numbers, so nobody has to take the propensity
    # figures on trust or go hunting through code for them.
    import json

    manifest = outputs / "retention_assumptions.json"
    manifest.write_text(
        json.dumps(result.summary(), indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    targets["retention_assumptions"] = manifest
    logger.info("Wrote %s", manifest)
    return targets
