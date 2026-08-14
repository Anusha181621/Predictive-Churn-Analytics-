"""Scoring: features at an as-of date, through the saved model, out to predictions.

The whole path runs from the CSV files with no manual preprocessing step in between::

    data/*.csv -> build_customer_features(as_of) -> saved pipeline -> predictions

The default as-of date is the **latest** date in the data, not the latest *labelable* date. That
distinction is the point of scoring: training needs a settled outcome window, so it stops 180 days
short of the end, whereas prediction wants the freshest possible view precisely because the outcome
has not happened yet. Scoring rows therefore have no label, and none is invented for them.
"""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path

import pandas as pd

from src.config.settings import Settings, get_settings
from src.data.csv_loader import Datasets, load_all
from src.features.builder import build_customer_features
from src.features.params import FeatureParams
from src.models.registry import SavedModel, load_model
from src.models.risk import (
    assign_risk_level,
    expected_horizon_revenue,
    revenue_at_risk,
    risk_distribution,
)
from src.utils.logging_config import get_logger
from src.utils.paths import ensure_dir

__all__ = ["PREDICTION_FILENAME", "score_customers", "write_predictions"]

logger = get_logger(__name__)

PREDICTION_FILENAME = "customer_churn_predictions.csv"

#: Output columns, in the order the brief lists them. Business-readable names, because a CRM
#: manager opens this file directly.
OUTPUT_COLUMNS = [
    "Customer ID",
    "Prediction date",
    "Churn probability",
    "Risk level",
    "Customer value",
    "Lifetime revenue",
    "Recent revenue",
    "Recency",
    "Frequency",
    "Revenue at risk",
]


def score_customers(
    data: Datasets | None = None,
    as_of_date: str | date | datetime | pd.Timestamp | None = None,
    *,
    model: SavedModel | None = None,
    model_dir: str | None = None,
    settings: Settings | None = None,
    feature_params: FeatureParams | None = None,
    include_diagnostics: bool = True,
) -> pd.DataFrame:
    """Score every customer at ``as_of_date`` and return the prediction table.

    Customers with no purchase history at the as-of date are scored too -- the model has features
    for them and the dashboard needs a row per customer -- but they are flagged
    ``has_purchase_history=False`` so a retention list can exclude people who never bought.
    """
    settings = settings or get_settings()
    data = data if data is not None else load_all()
    model = model or load_model(model_dir or settings.models_dir)

    as_of = (
        pd.Timestamp(as_of_date).normalize()
        if as_of_date is not None
        else pd.Timestamp(data.transactions["purchase_date"].max()).normalize()
    )

    features = build_customer_features(data, as_of_date=as_of, params=feature_params).features
    probability = model.predict_proba(features)

    horizon = model.metadata.horizon_days
    predictions = pd.DataFrame(
        {
            "Customer ID": features["customer_id"],
            "Prediction date": as_of.date().isoformat(),
            "Churn probability": probability.round(6).to_numpy(),
            "Customer value": features["customer_value_segment"],
            "Lifetime revenue": features["lifetime_revenue"].round(2),
            "Recent revenue": features["revenue_180d"].round(2),
            "Recency": features["recency_days"],
            "Frequency": features["total_orders"],
        }
    )
    predictions["Risk level"] = assign_risk_level(
        predictions["Churn probability"], settings
    ).to_numpy()
    predictions["Revenue at risk"] = revenue_at_risk(
        predictions["Churn probability"],
        features["lifetime_revenue"],
        features["customer_tenure_days"],
        horizon,
    ).to_numpy()

    predictions = predictions[OUTPUT_COLUMNS]

    if include_diagnostics:
        # Carried alongside the required columns because the downstream sections need them and
        # recomputing features just to recover them would be wasteful.
        predictions["Horizon days"] = horizon
        predictions["Annualized revenue"] = features["annualized_revenue"].to_numpy()
        predictions["Expected horizon revenue"] = expected_horizon_revenue(
            features["lifetime_revenue"], features["customer_tenure_days"], horizon
        ).to_numpy()
        predictions["Customer tenure days"] = features["customer_tenure_days"].to_numpy()
        predictions["Behavioural segment"] = features["behavioral_segment"].to_numpy()
        predictions["Lifecycle stage"] = features["lifecycle_stage"].to_numpy()
        predictions["Purchase gap ratio"] = features["purchase_gap_ratio"].round(4).to_numpy()
        predictions["Seasonal customer score"] = (
            features["seasonal_customer_score"].round(4).to_numpy()
        )
        predictions["Seasonally explained inactivity"] = features[
            "seasonally_explained_inactivity"
        ].to_numpy()
        predictions["Has purchase history"] = features["has_purchase_history"].to_numpy()
        # The binary call at the threshold the model's reported precision, recall and accuracy were
        # measured at -- taken from the metadata rather than re-derived here, so the confusion
        # matrix in the metrics file and the flag in this CSV can never disagree. It is distinct
        # from "Risk level", which bands the same probability for prioritisation rather than
        # answering the yes/no question the model was scored on.
        predictions["Predicted churn"] = (
            predictions["Churn probability"] >= float(model.metadata.decision_threshold)
        ).to_numpy()
        predictions["Decision threshold"] = float(model.metadata.decision_threshold)
        predictions["Model"] = model.metadata.model_name

    logger.info(
        "Scored %d customers as of %s with %s (mean churn probability %.4f)",
        len(predictions),
        as_of.date(),
        model.metadata.model_name,
        float(predictions["Churn probability"].mean()),
    )
    return predictions


def write_predictions(
    predictions: pd.DataFrame,
    destination: str | Path | None = None,
    settings: Settings | None = None,
) -> Path:
    """Write the predictions CSV. An analytical artefact; ``data/`` is never touched."""
    settings = settings or get_settings()
    if destination is None:
        target = ensure_dir(settings.outputs_dir) / PREDICTION_FILENAME
    else:
        target = Path(destination)
        ensure_dir(target.parent)
    predictions.to_csv(target, index=False, encoding="utf-8-sig", float_format="%.6g")
    logger.info("Wrote %s (%d rows)", target, len(predictions))
    return target


def prediction_summary(predictions: pd.DataFrame) -> dict[str, object]:
    """Risk distribution and revenue exposure, for the run report and the dashboard."""
    distribution = risk_distribution(predictions["Risk level"])
    return {
        "customers_scored": len(predictions),
        "prediction_date": str(predictions["Prediction date"].iloc[0]) if len(predictions) else None,
        "mean_churn_probability": round(float(predictions["Churn probability"].mean()), 6),
        "risk_distribution": {
            str(level): int(row["customers"]) for level, row in distribution.iterrows()
        },
        "total_revenue_at_risk": round(float(predictions["Revenue at risk"].sum()), 2),
        "revenue_at_risk_high_and_critical": round(
            float(
                predictions.loc[
                    predictions["Risk level"].isin(["High", "Critical"]), "Revenue at risk"
                ].sum()
            ),
            2,
        ),
    }
