"""Churn model: labelling, time-based validation, training, selection and scoring.

The whole path runs from the CSV files with no manual preprocessing step::

    data/*.csv -> features (as of a date) -> churn model -> predictions

Entry points::

    from src.models import train_churn_model, score_customers

    result = train_churn_model()          # builds the panel, splits on time, selects, calibrates
    predictions = score_customers()       # scores every customer at the latest date

Two modules carry the design weight. :mod:`src.models.labels` defines churn as a *forward-looking*
outcome -- no purchase in ``(as_of, as_of + horizon]`` -- which is what makes the label
leakage-free and prevents a seasonal customer being mislabelled from inactivity alone.
:mod:`src.models.splits` splits on the as-of date with an embargo, so no training row's outcome
window overlaps the next period's feature window.
"""

from src.models.candidates import CandidateSpec, available_candidates, candidate_specs
from src.models.dataset import ModellingPanel, build_panel, monthly_as_of_grid, quarterly_as_of_grid
from src.models.evaluate import EvaluationResult, evaluate_predictions
from src.models.labels import (
    ChurnLabels,
    LabelMode,
    LabelParams,
    build_churn_labels,
    compare_label_modes,
    latest_labelable_as_of,
)
from src.models.predict import prediction_summary, score_customers, write_predictions
from src.models.registry import ModelMetadata, SavedModel, load_model, save_model
from src.models.risk import (
    RISK_LEVELS,
    assign_risk_level,
    expected_horizon_revenue,
    revenue_at_risk,
    risk_distribution,
)
from src.models.splits import SplitPlan, TimeSplit, make_time_split, plan_model_dates
from src.models.train import TrainingResult, train_churn_model

__all__ = [
    "CandidateSpec",
    "ChurnLabels",
    "EvaluationResult",
    "LabelMode",
    "LabelParams",
    "ModelMetadata",
    "ModellingPanel",
    "RISK_LEVELS",
    "SavedModel",
    "SplitPlan",
    "TimeSplit",
    "TrainingResult",
    "assign_risk_level",
    "available_candidates",
    "build_churn_labels",
    "build_panel",
    "candidate_specs",
    "compare_label_modes",
    "evaluate_predictions",
    "expected_horizon_revenue",
    "latest_labelable_as_of",
    "load_model",
    "make_time_split",
    "monthly_as_of_grid",
    "plan_model_dates",
    "prediction_summary",
    "quarterly_as_of_grid",
    "revenue_at_risk",
    "risk_distribution",
    "save_model",
    "score_customers",
    "train_churn_model",
    "write_predictions",
]
