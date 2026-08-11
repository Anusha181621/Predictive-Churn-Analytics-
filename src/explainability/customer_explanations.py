"""Per-customer churn explanations: the top 3-5 drivers, in words, for every customer.

Which drivers are chosen
------------------------
Drivers are ranked by **absolute** SHAP contribution, then written out with their sign. Ranking by
signed contribution would be wrong in a way that is easy to miss: it would hide the strongest
*protective* factor, and a retention manager looking at a customer needs to know that their weekly
ordering habit is the one thing still holding them, not only what is going wrong.

Each customer therefore gets their top ``k`` drivers by magnitude, so a mostly-safe customer's list
correctly reads as reasons they are safe, and an at-risk customer's list reads as reasons they are
leaving. The ``Direction`` column carries the sign, so a dashboard can split them.

Long format, on purpose
-----------------------
One row per (customer, driver) rather than five columns of driver text. That is what the brief's
column list implies, and it is what a dashboard actually wants: filtering to "every customer whose
top driver is a widening purchase gap" is a one-line query on this shape and awkward on the wide one.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.explainability.narratives import NarrativeBuilder, driver_group, format_value
from src.explainability.shap_values import ShapResult
from src.utils.logging_config import get_logger
from src.utils.paths import ensure_dir

__all__ = [
    "EXPLANATION_COLUMNS",
    "EXPLANATION_FILENAME",
    "build_customer_explanations",
    "write_customer_explanations",
    "explanation_for",
]

logger = get_logger(__name__)

EXPLANATION_FILENAME = "customer_churn_explanations.csv"

#: The columns the brief specifies, in order. Diagnostics are appended after these.
EXPLANATION_COLUMNS = [
    "Customer ID",
    "Churn probability",
    "Risk level",
    "Driver rank",
    "Feature",
    "Feature value",
    "Contribution",
    "Direction",
    "Human-readable explanation",
]


def build_customer_explanations(
    shap_result: ShapResult,
    predictions: pd.DataFrame,
    *,
    narratives: NarrativeBuilder | None = None,
    top_k: int = 5,
    risk_levels: tuple[str, ...] | None = None,
    group_drivers: bool = True,
) -> pd.DataFrame:
    """Build the long-format explanation table.

    Parameters
    ----------
    shap_result:
        Per-customer contributions from :func:`~src.explainability.shap_values.compute_shap_values`.
    predictions:
        The scored table from :func:`~src.models.predict.score_customers`, joined on Customer ID.
    top_k:
        Drivers per customer. The brief asks for 3-5; the default is 5.
    risk_levels:
        Restrict to these risk bands. ``None`` covers everybody, which is what the dashboard needs
        so that a Low-risk customer can still be interrogated.
    group_drivers:
        Collapse near-duplicate features onto one concept each, keeping the strongest contributor,
        so the driver list carries distinct reasons rather than restatements.
    """
    if not 1 <= top_k <= 25:
        raise ValueError(f"top_k must be between 1 and 25, got {top_k}")

    narratives = narratives or NarrativeBuilder(shap_result.values)
    contributions = shap_result.contributions

    scored = predictions.set_index("Customer ID")
    if risk_levels is not None:
        keep = scored.index[scored["Risk level"].isin(list(risk_levels))]
        # `Index.intersection` drops the name when the two differ ("customer_id" vs "Customer ID"),
        # and the unnamed index then makes `stack().reset_index()` emit `level_0`, breaking the
        # groupby downstream. Restore it explicitly rather than relying on pandas' naming.
        selected = contributions.index.intersection(keep).rename(contributions.index.name)
        contributions = contributions.loc[selected]

    if contributions.empty:  # pragma: no cover - only if a filter excludes everyone
        return pd.DataFrame(columns=EXPLANATION_COLUMNS)

    # Rank every (customer, feature) cell by |contribution| and keep the top k per customer. Done
    # as one vectorised melt rather than a per-customer loop: 1,000 customers x 127 features is
    # 127,000 cells, and looping would be a hundred times slower for no benefit.
    long = (
        contributions.stack()
        .rename("contribution")
        .reset_index()
        .rename(columns={"level_1": "feature"})
    )
    long["abs_contribution"] = long["contribution"].abs()

    if group_drivers:
        # Keep only each concept's strongest contributor before ranking, so five slots buy five
        # distinct reasons rather than several restatements of one. Without this, `cadence` alone
        # spent two of the five on "typically 49 days between orders" and "typically orders every
        # 49 days" -- the same number twice, by construction.
        long["driver_group"] = long["feature"].map(driver_group)
        long = (
            long.sort_values(["customer_id", "driver_group", "abs_contribution"], ascending=[True, True, False])
            .drop_duplicates(["customer_id", "driver_group"])
            .copy()
        )
    else:
        long["driver_group"] = long["feature"]

    long["driver_rank"] = (
        long.groupby("customer_id", observed=True)["abs_contribution"]
        .rank(method="first", ascending=False)
        .astype("int64")
    )
    drivers = long[long["driver_rank"].le(top_k)].sort_values(["customer_id", "driver_rank"])

    # Share of the customer's total absolute contribution, so a reader can see whether the top
    # driver dominates or the risk is diffuse.
    totals = contributions.abs().sum(axis=1)
    drivers = drivers.assign(
        contribution_share=drivers["abs_contribution"]
        / drivers["customer_id"].map(totals).replace(0, np.nan)
    )

    records: list[dict[str, object]] = []
    for row in drivers.itertuples(index=False):
        customer_id = row.customer_id
        feature = row.feature
        contribution = float(row.contribution)
        phrase_kind = _kind_for(feature)
        raw_value = (
            shap_result.values.at[customer_id, feature]
            if feature in shap_result.values.columns
            else None
        )
        records.append(
            {
                "Customer ID": customer_id,
                "Churn probability": float(scored.at[customer_id, "Churn probability"]),
                "Risk level": str(scored.at[customer_id, "Risk level"]),
                "Driver rank": int(row.driver_rank),
                "Feature": feature,
                "Feature value": format_value(raw_value, phrase_kind, narratives.currency),
                "Contribution": round(contribution, 6),
                "Direction": "increases risk" if contribution > 0 else "reduces risk",
                "Human-readable explanation": narratives.sentence(
                    customer_id, feature, contribution
                ),
                # --- diagnostics beyond the required columns ---
                "Feature label": narratives.label_for(feature),
                "Raw feature value": raw_value,
                "Driver group": row.driver_group,
                "Contribution share": round(float(row.contribution_share), 6)
                if pd.notna(row.contribution_share)
                else None,
                "Prediction date": scored.at[customer_id, "Prediction date"],
            }
        )

    frame = pd.DataFrame.from_records(records)
    ordered = EXPLANATION_COLUMNS + [
        column for column in frame.columns if column not in EXPLANATION_COLUMNS
    ]
    frame = frame[ordered]

    logger.info(
        "Explanations: %d rows for %d customers (top %d drivers each)",
        len(frame),
        frame["Customer ID"].nunique(),
        top_k,
    )
    return frame


def _kind_for(feature: str) -> str:
    from src.explainability.narratives import VOCABULARY

    phrase = VOCABULARY.get(feature)
    return phrase.kind if phrase is not None else "number"


def explanation_for(explanations: pd.DataFrame, customer_id: str) -> str:
    """Render one customer's explanation as the block the brief illustrates.

    This is the Customer 360 view's text, produced from the same table the dashboard reads, so the
    two can never drift apart.
    """
    rows = explanations[explanations["Customer ID"].eq(customer_id)].sort_values("Driver rank")
    if rows.empty:
        return f"No explanation available for {customer_id}."
    head = rows.iloc[0]
    lines = [
        f"Customer: {customer_id}",
        f"Churn probability: {head['Churn probability']:.0%}",
        f"Risk level: {head['Risk level']}",
        "",
        "Top drivers:",
        "",
    ]
    for position, (_, row) in enumerate(rows.iterrows(), start=1):
        # Arrow rather than colour, so the block reads correctly in a terminal, a CSV cell and a
        # dashboard alike.
        arrow = "^" if row["Direction"] == "increases risk" else "v"
        lines.append(f"{position}. [{arrow}] {row['Human-readable explanation']}")
    return "\n".join(lines)


def write_customer_explanations(
    explanations: pd.DataFrame, destination: str | Path | None = None, outputs_dir: str = "outputs"
) -> Path:
    """Write the explanations CSV. An analytical artefact; ``data/`` is never touched."""
    if destination is None:
        target = ensure_dir(outputs_dir) / EXPLANATION_FILENAME
    else:
        target = Path(destination)
        ensure_dir(target.parent)
    explanations.to_csv(target, index=False, encoding="utf-8-sig", float_format="%.6g")
    logger.info("Wrote %s (%d rows)", target, len(explanations))
    return target
