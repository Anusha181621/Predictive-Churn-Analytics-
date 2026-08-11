"""Explainable churn prediction with SHAP: why is this customer likely to churn?

Entry point::

    from src.explainability import explain_churn

    result = explain_churn()                 # scores, explains, returns everything
    result.explanations                      # long-format per-customer drivers
    result.global_explanation.summary        # model-level SHAP summary
    print(result.narrative_for("CUST0234"))  # the readable block for one customer

Two things worth knowing before reading the numbers. SHAP explains the model *before* probability
calibration, because that is where the trees are; calibration is monotone, so driver ranking and
direction carry over unchanged, but the contributions are on the uncalibrated log-odds scale and do
not sum to the reported probability. And the sentences in
:mod:`src.explainability.narratives` are composed at runtime from each customer's own values and
contributions -- a phrase grammar per feature, not a fixed set of explanations.
"""

from src.explainability.customer_explanations import (
    EXPLANATION_COLUMNS,
    EXPLANATION_FILENAME,
    build_customer_explanations,
    explanation_for,
    write_customer_explanations,
)
from src.explainability.global_explanations import (
    GlobalExplanation,
    build_global_explanation,
    write_global_explanation,
)
from src.explainability.narratives import VOCABULARY, NarrativeBuilder, Phrase, format_value
from src.explainability.pipeline import ExplainabilityResult, explain_churn
from src.explainability.shap_values import ShapResult, compute_shap_values, unwrap_pipeline

__all__ = [
    "EXPLANATION_COLUMNS",
    "EXPLANATION_FILENAME",
    "ExplainabilityResult",
    "GlobalExplanation",
    "NarrativeBuilder",
    "Phrase",
    "ShapResult",
    "VOCABULARY",
    "build_customer_explanations",
    "build_global_explanation",
    "compute_shap_values",
    "explain_churn",
    "explanation_for",
    "format_value",
    "unwrap_pipeline",
    "write_customer_explanations",
    "write_global_explanation",
]
