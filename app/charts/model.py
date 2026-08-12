"""Model-performance and explainability charts.

The SHAP artefacts were written as CSV rather than PNG precisely so this page could render them
interactively; these functions are the intended consumer.

One honest constraint runs through the SHAP charts: contributions are on the model's
**uncalibrated** log-odds scale, because that is where the trees are. Calibration is monotone, so
the ranking and the direction of every driver carry over to the reported probability unchanged --
but the magnitudes do not add up to it. The axis labels say "log-odds" rather than implying an
additivity that is not there.
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

from app.theme import (
    AXIS,
    DIVERGING_HIGH,
    DIVERGING_LOW,
    DIVERGING_MID,
    INK,
    INK_MUTED,
    SEQUENTIAL,
    SURFACE,
    categorical,
)

__all__ = [
    "reliability_curve",
    "confusion_matrix",
    "feature_importance",
    "permutation_importance",
    "dependence_curve",
    "customer_drivers",
]

BAR_LINE = dict(color=SURFACE, width=2)

_DIRECTION_LABELS = {
    "raise": "Higher values raise churn risk",
    "lower": "Higher values lower churn risk",
    "mixed": "Mixed / non-monotone",
}


def _direction_key(text: object) -> str:
    value = str(text).lower()
    if "raise" in value or "increase" in value:
        return "raise"
    if "lower" in value or "reduce" in value or "decrease" in value:
        return "lower"
    return "mixed"


_DIRECTION_COLOURS = {
    "raise": DIVERGING_HIGH,
    "lower": DIVERGING_LOW,
    "mixed": DIVERGING_MID,
}


def reliability_curve(bins: list[dict], *, height: int = 360) -> go.Figure:
    """Predicted probability against what actually happened, per calibration bin.

    The 45-degree line is perfect calibration. It is drawn in the de-emphasis grey as a
    *reference*, not as a second data series, so the eye goes to the model's own curve and reads
    the gap. Both are named in the legend, so identity never rests on colour alone.
    """
    if not bins:
        return go.Figure()

    frame = pd.DataFrame(bins)
    figure = go.Figure()

    figure.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="Perfect calibration",
            line=dict(color=AXIS, width=2),
            hoverinfo="skip",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=frame["predicted"],
            y=frame["observed"],
            mode="lines+markers",
            name="Model",
            line=dict(color=categorical(0), width=2),
            marker=dict(
                size=10,
                color=categorical(0),
                line=dict(color=SURFACE, width=2),  # surface ring keeps overlapping dots legible
            ),
            customdata=frame[["n", "gap"]].to_numpy(),
            hovertemplate=(
                "Predicted %{x:.1%}<br>Observed %{y:.1%}<br>"
                "%{customdata[0]:,} customers · gap %{customdata[1]:.3f}<extra></extra>"
            ),
        )
    )

    figure.update_layout(
        height=height,
        xaxis_title="Predicted churn probability",
        yaxis_title="Observed churn rate",
    )
    figure.update_xaxes(tickformat=".0%", range=[0, 1])
    figure.update_yaxes(tickformat=".0%", range=[0, 1])
    return figure


def confusion_matrix(confusion: dict, *, height: int = 330) -> go.Figure:
    """The 2x2 outcome table at the 0.5 decision threshold.

    A count grid is magnitude, so it takes the one-hue sequential ramp. Cell labels flip between
    ink and white by the depth of their own fill, so every number clears contrast against the
    cell it sits in.
    """
    tn = int(confusion.get("tn", 0))
    fp = int(confusion.get("fp", 0))
    fn = int(confusion.get("fn", 0))
    tp = int(confusion.get("tp", 0))

    z = [[tn, fp], [fn, tp]]
    y_labels = ["Actually stayed", "Actually churned"]
    x_labels = ["Predicted stay", "Predicted churn"]

    peak = max(tn, fp, fn, tp) or 1
    text_colours = [
        ["#ffffff" if value / peak > 0.55 else INK for value in row] for row in z
    ]

    figure = go.Figure(
        go.Heatmap(
            z=z,
            x=x_labels,
            y=y_labels,
            colorscale=[[i / (len(SEQUENTIAL) - 1), c] for i, c in enumerate(SEQUENTIAL)],
            showscale=False,
            xgap=2,  # the 2px surface gap, so touching cells read as separate marks
            ygap=2,
            hovertemplate="%{y} · %{x}<br>%{z:,} customers<extra></extra>",
        )
    )
    for row, (label, colours) in enumerate(zip(z, text_colours)):
        for col, value in enumerate(label):
            figure.add_annotation(
                x=x_labels[col],
                y=y_labels[row],
                text=f"<b>{value:,}</b>",
                showarrow=False,
                font=dict(size=17, color=colours[col]),
            )

    figure.update_layout(height=height, xaxis_title=None, yaxis_title=None)
    figure.update_xaxes(showgrid=False, ticks="")
    figure.update_yaxes(showgrid=False, ticks="")
    return figure


def feature_importance(
    frame: pd.DataFrame, *, top_n: int = 15, height: int = 460
) -> go.Figure:
    """Mean absolute SHAP per feature, coloured by the direction of the effect.

    Length carries importance; colour carries polarity, which is a genuinely separate fact and so
    earns the second channel. Features whose relationship the model learned as non-monotone are
    drawn in the neutral grey rather than being forced into a direction they do not have.
    """
    top = frame.nsmallest(top_n, "rank").sort_values("mean_abs_shap")
    keys = [_direction_key(d) for d in top["direction"]]

    figure = go.Figure()
    for key in ("raise", "lower", "mixed"):
        mask = [k == key for k in keys]
        if not any(mask):
            continue
        subset = top[mask]
        figure.add_bar(
            y=subset["label"].astype(str),
            x=subset["mean_abs_shap"],
            orientation="h",
            name=_DIRECTION_LABELS[key],
            marker=dict(color=_DIRECTION_COLOURS[key], cornerradius=4, line=BAR_LINE),
            customdata=subset[["importance_share"]].to_numpy(),
            hovertemplate="%{y}<br>Mean |SHAP| %{x:.4f}<br>"
            "%{customdata[0]:.1%} of total importance<extra></extra>",
        )

    figure.update_layout(
        height=max(height, 30 * len(top) + 110),
        xaxis_title="Mean |SHAP| (uncalibrated log-odds)",
        yaxis_title=None,
        barmode="overlay",
        bargap=0.35,
    )
    return figure


def permutation_importance(
    features: list[dict], *, top_n: int = 15, height: int = 440
) -> go.Figure:
    """Permutation importance from training: the PR-AUC drop when a feature is shuffled.

    One measure over nominal features, so one colour. Permutation importance is reported instead
    of the trees' own impurity importance because impurity importance inflates high-cardinality
    features regardless of whether they predict anything.
    """
    if not features:
        return go.Figure()
    frame = pd.DataFrame(features).head(top_n).sort_values("importance")

    figure = go.Figure(
        go.Bar(
            y=frame["feature"].astype(str),
            x=frame["importance"],
            orientation="h",
            marker=dict(color=categorical(0), cornerradius=4, line=BAR_LINE),
            error_x=dict(
                type="data",
                array=frame.get("importance_std", pd.Series([0] * len(frame))),
                color=AXIS,
                thickness=1,
                width=3,
            ),
            customdata=frame[["importance_share"]].to_numpy()
            if "importance_share" in frame
            else None,
            hovertemplate="%{y}<br>PR-AUC drop %{x:.4f}<extra></extra>",
        )
    )
    figure.update_layout(
        height=max(height, 30 * len(frame) + 100),
        xaxis_title="PR-AUC drop when shuffled",
        yaxis_title=None,
        showlegend=False,
    )
    return figure


def customer_drivers(frame: pd.DataFrame, *, height: int = 300) -> go.Figure:
    """One customer's top drivers, as a diverging bar around zero.

    Drivers are ranked by *absolute* contribution, so the strongest protective factor appears
    beside the strongest risk factor. A retention manager needs to know what is still holding a
    customer, not only what is pushing them away — hiding the protective side would make a
    steady weekly buyer look like a pure risk.
    """
    ordered = frame.sort_values("Contribution")
    colours = [
        DIVERGING_HIGH if value > 0 else DIVERGING_LOW for value in ordered["Contribution"]
    ]

    figure = go.Figure(
        go.Bar(
            y=ordered["Feature label"].astype(str),
            x=ordered["Contribution"],
            orientation="h",
            marker=dict(color=colours, cornerradius=4, line=BAR_LINE),
            customdata=ordered[["Direction", "Feature value"]].to_numpy(),
            hovertemplate="%{y}<br>Contribution %{x:+.4f} (%{customdata[0]})"
            "<br>Value: %{customdata[1]}<extra></extra>",
        )
    )
    figure.add_vline(x=0, line_width=1, line_color=AXIS)
    figure.update_layout(
        height=max(height, 42 * len(ordered) + 90),
        xaxis_title="Contribution to churn risk (log-odds) — right raises, left lowers",
        yaxis_title=None,
        showlegend=False,
        bargap=0.4,
    )
    return figure


def dependence_curve(frame: pd.DataFrame, feature_label: str, *, height: int = 340) -> go.Figure:
    """How a feature's contribution changes across its own value range.

    A zero line marks where the feature stops pushing the prediction either way; above it the
    feature is pushing towards churn, below it away.
    """
    ordered = frame.sort_values("value_mean")
    figure = go.Figure(
        go.Scatter(
            x=ordered["value_mean"],
            y=ordered["mean_shap"],
            mode="lines+markers",
            name=feature_label,
            line=dict(color=categorical(0), width=2),
            marker=dict(size=9, color=categorical(0), line=dict(color=SURFACE, width=2)),
            customdata=ordered[["customers"]].to_numpy(),
            hovertemplate="Value %{x:,.3f}<br>Mean SHAP %{y:+.4f}<br>"
            "%{customdata[0]:,} customers<extra></extra>",
        )
    )
    figure.add_hline(y=0, line_width=1, line_color=AXIS)
    figure.update_layout(
        height=height,
        xaxis_title=feature_label,
        yaxis_title="Mean SHAP (log-odds)",
        showlegend=False,
    )
    return figure
