"""One call from the CSV files to explained predictions.

    data/*.csv -> features (as of a date) -> saved model -> probabilities -> SHAP -> sentences

No database, no ingestion step, no manual preprocessing. The same feature build that produced the
prediction produces the explanation, so a driver sentence can never describe a different snapshot
from the probability it accompanies -- which is the failure mode of explaining from a separately
recomputed feature table.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path

import pandas as pd

from src.config.settings import Settings, get_settings
from src.data.csv_loader import Datasets, load_all
from src.explainability.customer_explanations import (
    build_customer_explanations,
    explanation_for,
    write_customer_explanations,
)
from src.explainability.global_explanations import (
    GlobalExplanation,
    build_global_explanation,
    write_global_explanation,
)
from src.explainability.narratives import NarrativeBuilder
from src.explainability.shap_values import ShapResult, compute_shap_values
from src.features.builder import build_customer_features
from src.features.params import FeatureParams
from src.models.predict import score_customers
from src.models.registry import SavedModel, load_model
from src.utils.logging_config import get_logger

__all__ = ["ExplainabilityResult", "explain_churn", "EXPLAINABILITY_DIR"]

logger = get_logger(__name__)

#: Sub-directory of ``outputs/`` for the global artefacts, as the brief specifies.
EXPLAINABILITY_DIR = "explainability"


@dataclass
class ExplainabilityResult:
    """Everything one explainability run produced."""

    as_of: pd.Timestamp
    model: SavedModel
    predictions: pd.DataFrame
    shap_result: ShapResult
    global_explanation: GlobalExplanation
    explanations: pd.DataFrame
    narratives: NarrativeBuilder = field(repr=False)
    top_k: int

    def narrative_for(self, customer_id: str) -> str:
        """The readable driver block for one customer."""
        return explanation_for(self.explanations, customer_id)

    def summary(self) -> dict[str, object]:
        drivers = self.explanations
        return {
            "as_of_date": self.as_of.date().isoformat(),
            "model": self.model.metadata.model_name,
            "horizon_days": self.model.metadata.horizon_days,
            "calibration": self.model.metadata.calibration,
            "shap_scale": "uncalibrated log-odds (margin before probability calibration)",
            "customers_explained": int(drivers["Customer ID"].nunique()) if len(drivers) else 0,
            "drivers_per_customer": self.top_k,
            "explanation_rows": len(drivers),
            "base_value": round(self.shap_result.base_value, 6),
            "features_explained": int(self.shap_result.contributions.shape[1]),
            "top_global_drivers": self.global_explanation.summary.head(10)[
                ["rank", "feature", "label", "mean_abs_shap", "direction"]
            ].to_dict(orient="records"),
            "most_common_top_driver": (
                drivers[drivers["Driver rank"].eq(1)]["Feature"].value_counts().head(5).to_dict()
                if len(drivers)
                else {}
            ),
        }


def explain_churn(
    data: Datasets | None = None,
    as_of_date: str | date | datetime | pd.Timestamp | None = None,
    *,
    model: SavedModel | None = None,
    model_dir: str | None = None,
    settings: Settings | None = None,
    feature_params: FeatureParams | None = None,
    top_k: int = 5,
    risk_levels: tuple[str, ...] | None = None,
    predictions: pd.DataFrame | None = None,
) -> ExplainabilityResult:
    """Score every customer and explain each one's churn risk.

    Parameters
    ----------
    top_k:
        Drivers per customer; the brief asks for 3-5 and the default is 5.
    risk_levels:
        Restrict explanations to these bands. ``None`` explains everybody, so the dashboard can
        interrogate a Low-risk customer too.
    predictions:
        Reuse an existing scored table instead of re-scoring. It must come from the same as-of date.
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
    if predictions is None:
        predictions = score_customers(
            data, as_of_date=as_of, model=model, settings=settings, feature_params=feature_params
        )
    elif str(predictions["Prediction date"].iloc[0]) != as_of.date().isoformat():
        # A mismatch here would pair one snapshot's probability with another's drivers, which is
        # worse than useless: it would look plausible and be wrong.
        raise ValueError(
            f"supplied predictions are dated {predictions['Prediction date'].iloc[0]} but the "
            f"explanation is being built as of {as_of.date()}; they must match"
        )

    shap_result = compute_shap_values(model, features)
    narratives = NarrativeBuilder(shap_result.values, currency=settings.currency)

    global_explanation = build_global_explanation(shap_result, narratives)
    explanations = build_customer_explanations(
        shap_result,
        predictions,
        narratives=narratives,
        top_k=top_k,
        risk_levels=risk_levels,
    )

    return ExplainabilityResult(
        as_of=as_of,
        model=model,
        predictions=predictions,
        shap_result=shap_result,
        global_explanation=global_explanation,
        explanations=explanations,
        narratives=narratives,
        top_k=top_k,
    )


def write_explainability_outputs(
    result: ExplainabilityResult,
    settings: Settings | None = None,
    *,
    explanations_path: str | Path | None = None,
) -> dict[str, Path]:
    """Write the per-customer CSV and the global artefacts under ``outputs/explainability/``."""
    settings = settings or get_settings()
    written: dict[str, Path] = {
        "customer_explanations": write_customer_explanations(
            result.explanations, explanations_path, settings.outputs_dir
        )
    }
    written.update(
        write_global_explanation(
            result.global_explanation,
            Path(settings.outputs_path) / EXPLAINABILITY_DIR,
            metadata=result.summary(),
        )
    )
    return written
