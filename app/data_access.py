"""Cached readers for every artefact the dashboard displays.

The dashboard is a *reader*. It owns no business logic: every number it shows was computed by
the pipeline under ``src/`` and written to ``data/``, ``outputs/`` or ``models/``. The one
exception is the What-If simulator, which re-runs the real retention layer rather than
reimplementing its arithmetic -- see :mod:`app.views.what_if`.

Three things this module is responsible for.

**One master frame.** Five artefacts describe the same 1,000 customers on five different key
spellings. :func:`load_customer_master` joins them once, validated ``1:1`` so a silent fan-out
becomes an exception rather than a wrong total, and renames everything into one ``snake_case``
namespace. Pages then read one frame instead of re-deriving the join and drifting apart.

**One name for revenue at risk.** Two different figures exist and they are both correct:

===========================  =====================================================  ==========
Column                       Definition                                             Total
===========================  =====================================================  ==========
``revenue_at_risk``          churn x expected future revenue (frequency x value)     EUR 125,129
``model_revenue_at_risk``    churn x lifetime x horizon / max(tenure, horizon)       EUR 162,302
===========================  =====================================================  ==========

The brief defines revenue at risk as *churn probability x expected future revenue*, which is the
decision layer's figure, so ``revenue_at_risk`` is the one every business page shows. The model's
own estimate is kept under an explicit name and surfaced only on the Model Performance page. They
are never added together and never shown as though they were the same quantity.

**Refresh without a restart.** Cache keys include each file's modification time and size, so
re-running any pipeline script and reloading the browser shows the new numbers.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import streamlit as st

from src.config.settings import Settings, get_settings
from src.data.csv_loader import Datasets, load_all
from src.utils.logging_config import get_logger

logger = get_logger("app.data_access")

__all__ = [
    "ARTEFACTS",
    "ACTIVE_RECENCY_DAYS",
    "AT_RISK_PROBABILITY",
    "MissingArtefact",
    "campaign_summary",
    "load_customer_master",
    "load_data_quality",
    "load_explanations",
    "load_global_importance",
    "load_model_metrics",
    "load_retention_assumptions",
    "load_shap_dependence",
    "load_shap_summary",
    "load_source_data",
    "prediction_date",
    "require",
]

#: A customer counts as active when they bought within this many days of the prediction date.
#: 180 days is the definition used in ``DATA_DICTIONARY_AND_VALIDATION.md`` (598 customers on the
#: shipped data), so the dashboard's "active" agrees with the project's own documentation rather
#: than inventing a third definition.
ACTIVE_RECENCY_DAYS = 180

#: At or above this churn probability a customer is counted as at risk. It is the Low/Medium
#: band edge, so "at risk" means exactly "not in the Low band" -- and it is read from
#: ``RISK_THRESHOLD_MEDIUM`` rather than restated here, because a literal would let the KPI
#: disagree with the very bands the pipeline assigned. Resolved once at import: the dashboard is
#: a fresh process per ``streamlit run``, so a configuration change arrives with the restart.
AT_RISK_PROBABILITY = get_settings().risk_threshold_medium


@dataclass(frozen=True)
class Artefact:
    """One generated file, and the command that produces it."""

    key: str
    filename: str
    command: str
    directory: str = "outputs"

    def path(self, settings: Settings) -> Path:
        base = settings.models_path if self.directory == "models" else settings.outputs_path
        return base / self.filename


ARTEFACTS: dict[str, Artefact] = {
    a.key: a
    for a in (
        Artefact("features", "customer_features.csv", "python scripts/build_features.py"),
        Artefact("predictions", "customer_churn_predictions.csv", "python scripts/predict.py"),
        Artefact("explanations", "customer_churn_explanations.csv", "python scripts/explain.py"),
        Artefact("scores", "customer_retention_scores.csv", "python scripts/retention.py"),
        Artefact("recommendations", "retention_recommendations.csv", "python scripts/retention.py"),
        Artefact("assumptions", "retention_assumptions.json", "python scripts/retention.py"),
        Artefact("metrics", "model_metrics.json", "python scripts/train_model.py"),
        Artefact("quality", "data_quality_report.json", "python scripts/validate_data.py"),
        Artefact(
            "shap_summary", "explainability/shap_summary.csv", "python scripts/explain.py"
        ),
        Artefact(
            "shap_importance",
            "explainability/global_feature_importance.csv",
            "python scripts/explain.py",
        ),
        Artefact(
            "shap_dependence", "explainability/shap_dependence.csv", "python scripts/explain.py"
        ),
        Artefact("model", "churn_model.joblib", "python scripts/train_model.py", "models"),
    )
}


#: What each generated file is called when a reader has to be told it is missing. A filename is
#: an implementation detail; "the churn predictions" is the thing they are actually waiting for.
DISPLAY_NAMES: dict[str, str] = {
    "features": "Customer profiles",
    "predictions": "Churn predictions",
    "explanations": "Churn driver explanations",
    "scores": "Retention scores",
    "recommendations": "Retention recommendations",
    "assumptions": "Campaign assumptions",
    "metrics": "Model performance results",
    "quality": "Source data quality checks",
    "shap_summary": "Churn driver summary",
    "shap_importance": "Churn driver ranking",
    "shap_dependence": "Churn driver detail",
    "model": "The trained churn model",
}


class MissingArtefact(RuntimeError):
    """Raised when a page needs a generated file that has not been produced yet."""


# --------------------------------------------------------------------------------------
# low-level cached readers
# --------------------------------------------------------------------------------------


def _stamp(path: Path) -> tuple[float, int]:
    """File identity for the cache key, so regenerating an artefact invalidates it."""
    stat = path.stat()
    return (stat.st_mtime, stat.st_size)


def _resolve(key: str) -> Path:
    settings = get_settings()
    path = ARTEFACTS[key].path(settings)
    if not path.exists():
        raise MissingArtefact(key)
    return path


@st.cache_data(show_spinner=False)
def _read_csv(path_str: str, stamp: tuple[float, int]) -> pd.DataFrame:
    # utf-8-sig because the pipeline writes a BOM so the CSVs open cleanly in Excel.
    return pd.read_csv(path_str, encoding="utf-8-sig")


@st.cache_data(show_spinner=False)
def _read_json(path_str: str, stamp: tuple[float, int]) -> dict:
    return json.loads(Path(path_str).read_text(encoding="utf-8-sig"))


def _csv(key: str) -> pd.DataFrame:
    path = _resolve(key)
    return _read_csv(str(path), _stamp(path))


def _json(key: str) -> dict:
    path = _resolve(key)
    return _read_json(str(path), _stamp(path))


def missing(*keys: str) -> list[Artefact]:
    """Return the artefacts among ``keys`` that do not exist yet."""
    settings = get_settings()
    return [ARTEFACTS[k] for k in keys if not ARTEFACTS[k].path(settings).exists()]


def require(*keys: str) -> None:
    """Stop the page with plain-language guidance if any required artefact is absent.

    ``outputs/`` and ``models/`` are git-ignored, so a fresh clone has neither. What the reader
    needs is what is missing and who can restore it -- not a shell command. The command that
    produces each file stays on :class:`Artefact` for whoever operates the pipeline, and is
    logged rather than rendered, so an administrator reading the log still gets it directly.
    """
    absent = missing(*keys)
    if not absent:
        return

    logger.error(
        "Page blocked: %s missing. Regenerate with: %s",
        ", ".join(f"{a.directory}/{a.filename}" for a in absent),
        "; ".join(sorted({a.command for a in absent})),
    )
    names = "\n".join(f"- {DISPLAY_NAMES.get(a.key, a.key.replace('_', ' ').capitalize())}"
                      for a in absent)
    st.error(
        "**This view is waiting on the latest analysis.**\n\n"
        f"Not available yet:\n{names}\n\n"
        "The analysis is refreshed as a scheduled job. Ask whoever administers this dashboard "
        "to run the refresh, then reload the page."
    )
    st.stop()


# --------------------------------------------------------------------------------------
# source CSVs
# --------------------------------------------------------------------------------------


@st.cache_resource(show_spinner=False)
def load_source_data() -> Datasets:
    """The four source CSVs, through the project's own loader.

    Cached as a resource rather than as data: the frames are read-only here and shared, so there
    is no reason to pay for a per-session copy of 20,000 transaction rows.
    """
    return load_all()


# --------------------------------------------------------------------------------------
# generated artefacts
# --------------------------------------------------------------------------------------

#: Columns lifted from the predictions artefact, mapped into the master frame's namespace.
_PREDICTION_COLUMNS = {
    "Customer ID": "customer_id",
    "Prediction date": "prediction_date",
    "Churn probability": "churn_probability",
    "Risk level": "risk_level",
    "Recent revenue": "recent_revenue",
    "Revenue at risk": "model_revenue_at_risk",
    "Expected horizon revenue": "model_expected_horizon_revenue",
    "Horizon days": "horizon_days",
    "Model": "model_name",
}

_SCORE_COLUMNS = {
    "Customer ID": "customer_id",
    "Primary segment": "primary_segment",
    "All segments": "all_segments",
    "Expected future revenue": "expected_future_revenue",
    "Revenue at risk": "revenue_at_risk",
    "Retention propensity (ASSUMED)": "retention_propensity",
    "Expected retained revenue": "expected_retained_revenue",
    "Retention opportunity score": "retention_opportunity_score",
    "Priority": "priority",
    "Retention opportunity percentile": "retention_opportunity_percentile",
    "Propensity basis (ASSUMED)": "propensity_basis",
    "Expected orders per year": "expected_orders_per_year",
    "Expected average order value": "expected_average_order_value",
    "Projected annual revenue": "projected_annual_revenue",
    "Historical annual revenue": "historical_annual_revenue",
    "Projection capped": "projection_capped",
}

_RECOMMENDATION_COLUMNS = {
    "Customer ID": "customer_id",
    "Recommended action": "recommended_action",
    "Recommended channel": "recommended_channel",
    "Recommended category": "recommended_category",
    "Recommended product/SKU": "recommended_sku",
    "Recommended product": "recommended_product",
    "Recommended offer": "recommended_offer",
    "Reason": "reason",
    "Expected ROI": "expected_roi",
    "Campaign cost": "campaign_cost",
    "Suppressed action": "suppressed_action",
    "ROI depends on an assumption": "roi_depends_on_assumption",
}

#: The twelve business segments carried as flag columns on the scores artefact. A customer can
#: be flagged for several at once -- that multi-membership is the point of the design, so the
#: flags are kept alongside the single ``primary_segment``.
SEGMENT_FLAG_COLUMNS = (
    "Champions",
    "Loyal Customers",
    "High-Value At Risk",
    "Frequent but Declining",
    "Discount-Driven At Risk",
    "Seasonal Customers",
    "New Customers",
    "One-Time Buyers",
    "Dormant Customers",
    "Lost Customers",
    "High-Return Customers",
    "Low-Value At Risk",
)


def _rename_subset(frame: pd.DataFrame, mapping: dict[str, str]) -> pd.DataFrame:
    """Select the columns of ``mapping`` that exist, renamed. Tolerates an older artefact."""
    present = {old: new for old, new in mapping.items() if old in frame.columns}
    return frame[list(present)].rename(columns=present)


@st.cache_data(show_spinner="Loading customer data...")
def load_customer_master() -> pd.DataFrame:
    """One row per customer, joining features, predictions, scores, recommendations and drivers.

    Every join is validated ``1:1``. If an artefact were regenerated at a different as-of date, or
    for a different customer set, the merge raises instead of silently producing a frame with
    duplicated or missing rows that every downstream total would then get wrong.
    """
    features = _csv("features")
    predictions = _rename_subset(_csv("predictions"), _PREDICTION_COLUMNS)
    scores_raw = _csv("scores")
    scores = _rename_subset(scores_raw, _SCORE_COLUMNS)
    recommendations = _rename_subset(_csv("recommendations"), _RECOMMENDATION_COLUMNS)

    flags = [c for c in SEGMENT_FLAG_COLUMNS if c in scores_raw.columns]
    if flags:
        scores = scores.join(scores_raw[flags].astype(bool))

    master = features.merge(predictions, on="customer_id", how="inner", validate="1:1")
    master = master.merge(scores, on="customer_id", how="inner", validate="1:1")
    master = master.merge(recommendations, on="customer_id", how="inner", validate="1:1")

    # The strongest driver for each customer, for the action centre's "main churn driver".
    explanations = _csv("explanations")
    top = explanations[explanations["Driver rank"] == 1]
    top = top[["Customer ID", "Feature label", "Human-readable explanation"]].rename(
        columns={
            "Customer ID": "customer_id",
            "Feature label": "top_driver",
            "Human-readable explanation": "top_driver_explanation",
        }
    )
    master = master.merge(top, on="customer_id", how="left", validate="1:1")

    master["is_active"] = master["recency_days"].le(ACTIVE_RECENCY_DAYS)
    master["is_at_risk"] = master["churn_probability"].ge(AT_RISK_PROBABILITY)
    master["is_targeted"] = master["recommended_action"].ne("Do Not Target")
    return master


@st.cache_data(show_spinner=False)
def load_explanations() -> pd.DataFrame:
    """All five drivers per customer, long format."""
    return _csv("explanations")


@st.cache_data(show_spinner=False)
def load_shap_summary() -> pd.DataFrame:
    return _csv("shap_summary")


@st.cache_data(show_spinner=False)
def load_global_importance() -> pd.DataFrame:
    return _csv("shap_importance")


@st.cache_data(show_spinner=False)
def load_shap_dependence() -> pd.DataFrame:
    return _csv("shap_dependence")


def load_model_metrics() -> dict:
    return _json("metrics")


def load_data_quality() -> dict:
    return _json("quality")


def load_retention_assumptions() -> dict:
    return _json("assumptions")


# --------------------------------------------------------------------------------------
# derived summaries shared by more than one page
# --------------------------------------------------------------------------------------


def prediction_date(master: pd.DataFrame) -> str:
    """The as-of date the artefacts were generated for."""
    values = master["prediction_date"].dropna().unique()
    return str(values[0]) if len(values) else "unknown"


def campaign_summary(frame: pd.DataFrame) -> dict[str, float]:
    """Campaign economics over ``frame``, counting only customers actually targeted.

    Suppressed customers carry no cost and no expected return, so including them would dilute
    the ROI toward zero and misstate what the campaign costs to run.
    """
    targeted = frame[frame["is_targeted"]]
    cost = float(targeted["campaign_cost"].sum())
    retained = float(targeted["expected_retained_revenue"].sum())
    return {
        "targeted": int(len(targeted)),
        "suppressed": int(len(frame) - len(targeted)),
        "cost": cost,
        "expected_retained": retained,
        # Expected *customers* retained: each targeted customer is retained with probability
        # (churn x propensity) -- the chance they would have left, times the chance contact works.
        "customers_retained": float(
            (targeted["churn_probability"] * targeted["retention_propensity"]).sum()
        ),
        "roi": (retained - cost) / cost if cost > 0 else float("nan"),
    }
