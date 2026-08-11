"""Global model explanation: which features drive churn across the whole book, and which way.

Four artefacts, written under ``outputs/explainability/``:

``global_feature_importance.csv``
    Mean absolute SHAP contribution per feature -- how much the model leans on each one.
``shap_summary.csv``
    The beeswarm plot as data: per feature, the importance, the mean signed contribution, the
    direction of impact, and the contribution spread. Emitted as data rather than a PNG on purpose,
    so the Streamlit dashboard in Section 6 can render it interactively with Plotly and filter it,
    which a static image cannot do. It also avoids adding matplotlib as a dependency for one chart.
``shap_dependence.csv``
    Binned feature value against mean contribution for the top drivers -- the shape of each
    relationship, so "higher recency raises risk" can be shown as a curve rather than asserted.
``top_churn_drivers.md``
    The same content as a readable summary, so the findings survive outside a notebook.

Direction of impact is measured, not assumed: it is the rank correlation between a feature's value
and its own SHAP contribution across customers. A tree can learn a non-monotone relationship, and
where it has, the correlation is near zero and the summary says the direction is mixed rather than
inventing one.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.explainability.narratives import NarrativeBuilder
from src.explainability.shap_values import ShapResult
from src.utils.logging_config import get_logger
from src.utils.paths import ensure_dir

__all__ = ["GlobalExplanation", "build_global_explanation", "write_global_explanation"]

logger = get_logger(__name__)

#: Below this absolute rank correlation the relationship is called mixed rather than directional.
DIRECTION_THRESHOLD = 0.15


def _direction_label(correlation: float) -> str:
    if pd.isna(correlation):
        return "not applicable"
    if correlation >= DIRECTION_THRESHOLD:
        return "higher values raise churn risk"
    if correlation <= -DIRECTION_THRESHOLD:
        return "higher values lower churn risk"
    return "mixed / non-monotone"


@dataclass
class GlobalExplanation:
    """Model-level SHAP summary."""

    summary: pd.DataFrame
    dependence: pd.DataFrame
    base_value: float
    customers: int

    @property
    def top_drivers(self) -> pd.DataFrame:
        return self.summary.head(20)

    def to_markdown(self) -> str:
        lines = [
            "# Global churn model explanation",
            "",
            f"- Customers explained: **{self.customers:,}**",
            f"- Base value (average customer, uncalibrated log-odds): **{self.base_value:.4f}**",
            "",
            "SHAP explains the model *before* probability calibration, because that is where the "
            "trees are. Calibration is a monotone transform, so the ranking and direction of every "
            "driver carry over to the reported probability unchanged; the contribution magnitudes "
            "are on the uncalibrated log-odds scale and do not sum to the calibrated probability.",
            "",
            "## Top churn drivers",
            "",
            "| # | Feature | What it means | Mean \\|SHAP\\| | Share | Direction of impact |",
            "|---|---|---|---|---|---|",
        ]
        for rank, (_, row) in enumerate(self.top_drivers.iterrows(), start=1):
            lines.append(
                f"| {rank} | `{row['feature']}` | {row['label']} | {row['mean_abs_shap']:.4f} | "
                f"{row['importance_share']:.1%} | {row['direction']} |"
            )
        lines += [
            "",
            "## How to read the direction column",
            "",
            "Direction is the Spearman correlation between a feature's value and its own SHAP "
            f"contribution across customers. Beyond ±{DIRECTION_THRESHOLD} it is reported as "
            "directional; inside that band the model has learned a non-monotone relationship and "
            "the column says so rather than forcing a direction that is not there.",
            "",
        ]
        mixed = self.summary[self.summary["direction"].eq("mixed / non-monotone")]
        if not mixed.empty:
            lines.append(
                f"{len(mixed)} of {len(self.summary)} features are non-monotone; the strongest are "
                + ", ".join(f"`{f}`" for f in mixed.head(5)["feature"])
                + "."
            )
        return "\n".join(lines)


def build_global_explanation(
    shap_result: ShapResult, narratives: NarrativeBuilder, *, dependence_top: int = 12
) -> GlobalExplanation:
    """Aggregate per-customer SHAP contributions into a model-level summary."""
    importance = shap_result.global_importance()
    mean_signed = shap_result.mean_contribution()
    direction = shap_result.direction()

    total = importance.sum()
    summary = pd.DataFrame(
        {
            "feature": importance.index,
            "label": [narratives.label_for(f) for f in importance.index],
            "mean_abs_shap": importance.to_numpy(),
            "importance_share": (importance / total).to_numpy() if total else np.nan,
            "mean_shap": mean_signed.reindex(importance.index).to_numpy(),
            "shap_std": shap_result.contributions.std().reindex(importance.index).to_numpy(),
            "shap_min": shap_result.contributions.min().reindex(importance.index).to_numpy(),
            "shap_max": shap_result.contributions.max().reindex(importance.index).to_numpy(),
            "value_shap_correlation": direction.reindex(importance.index).to_numpy(),
        }
    ).reset_index(drop=True)
    summary["direction"] = [_direction_label(c) for c in summary["value_shap_correlation"]]
    summary["rank"] = np.arange(1, len(summary) + 1)

    dependence = _build_dependence(shap_result, summary.head(dependence_top)["feature"].tolist())

    logger.info(
        "Global explanation: %d features ranked; top 5 = %s",
        len(summary),
        ", ".join(summary.head(5)["feature"]),
    )
    return GlobalExplanation(
        summary=summary,
        dependence=dependence,
        base_value=shap_result.base_value,
        customers=len(shap_result.contributions),
    )


def _build_dependence(
    shap_result: ShapResult, features: list[str], bins: int = 10
) -> pd.DataFrame:
    """Binned value-versus-contribution curves for the top features.

    Quantile bins, so each bucket holds a comparable number of customers; equal-width bins on a
    skewed feature leave near-empty buckets whose noise then dominates the curve.
    """
    frames: list[pd.DataFrame] = []
    for feature in features:
        values = pd.to_numeric(shap_result.values[feature], errors="coerce")
        contributions = shap_result.contributions[feature]
        if values.notna().sum() < bins or values.nunique(dropna=True) < 2:
            continue
        try:
            buckets = pd.qcut(values, q=bins, duplicates="drop")
        except (ValueError, IndexError):  # pragma: no cover - degenerate distribution
            continue
        grouped = pd.DataFrame({"bucket": buckets, "value": values, "shap": contributions})
        aggregated = (
            grouped.groupby("bucket", observed=True)
            .agg(
                customers=("shap", "size"),
                value_min=("value", "min"),
                value_max=("value", "max"),
                value_mean=("value", "mean"),
                mean_shap=("shap", "mean"),
            )
            .reset_index()
        )
        aggregated["feature"] = feature
        aggregated["bucket"] = aggregated["bucket"].astype(str)
        frames.append(aggregated)
    if not frames:  # pragma: no cover
        return pd.DataFrame(
            columns=["feature", "bucket", "customers", "value_min", "value_max", "value_mean",
                     "mean_shap"]
        )
    combined = pd.concat(frames, ignore_index=True)
    return combined[
        ["feature", "bucket", "customers", "value_min", "value_max", "value_mean", "mean_shap"]
    ]


def write_global_explanation(
    explanation: GlobalExplanation, directory: str | Path, *, metadata: dict | None = None
) -> dict[str, Path]:
    """Write the global artefacts and return the paths written."""
    target = ensure_dir(directory)
    written: dict[str, Path] = {}

    importance_path = target / "global_feature_importance.csv"
    explanation.summary[
        ["rank", "feature", "label", "mean_abs_shap", "importance_share", "direction"]
    ].to_csv(importance_path, index=False, encoding="utf-8-sig", float_format="%.6g")
    written["global_feature_importance"] = importance_path

    summary_path = target / "shap_summary.csv"
    explanation.summary.to_csv(
        summary_path, index=False, encoding="utf-8-sig", float_format="%.6g"
    )
    written["shap_summary"] = summary_path

    dependence_path = target / "shap_dependence.csv"
    explanation.dependence.to_csv(
        dependence_path, index=False, encoding="utf-8-sig", float_format="%.6g"
    )
    written["shap_dependence"] = dependence_path

    markdown_path = target / "top_churn_drivers.md"
    markdown_path.write_text(explanation.to_markdown(), encoding="utf-8")
    written["top_churn_drivers"] = markdown_path

    if metadata is not None:
        metadata_path = target / "explainability_metadata.json"
        metadata_path.write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
        )
        written["metadata"] = metadata_path

    for name, path in written.items():
        logger.info("Wrote %s -> %s", name, path)
    return written
