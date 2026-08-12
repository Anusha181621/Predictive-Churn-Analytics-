"""Distribution charts: how churn probability and risk are spread across the customer base.

Mark conventions shared with the other chart modules: bars carry a 4px rounded data-end and a
2px ring in the surface colour, which is what separates touching marks -- a contrasting border
would add ink that is not data. Grids are solid hairlines, never dashed, because a dashed rule
reads as a threshold or a projection when it is only a grid.
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

from app.theme import (
    AXIS,
    INK_MUTED,
    RISK_COLOURS,
    RISK_ORDER,
    SURFACE,
    categorical,
)

__all__ = ["churn_probability_distribution", "risk_level_distribution", "probability_by_group"]

BAR_LINE = dict(color=SURFACE, width=2)


def churn_probability_distribution(
    frame: pd.DataFrame,
    *,
    thresholds: dict[str, float] | None = None,
    height: int = 340,
) -> go.Figure:
    """Histogram of churn probability, with the risk-band edges marked.

    One series, so no legend: the title says what is plotted. The band edges are annotated
    because the bands are the thing a reader acts on, and without them the histogram is just a
    shape.
    """
    figure = go.Figure(
        go.Histogram(
            x=frame["churn_probability"],
            nbinsx=40,
            marker=dict(color=categorical(0), cornerradius=3, line=BAR_LINE),
            hovertemplate="Churn probability %{x}<br>%{y} customers<extra></extra>",
            name="Customers",
        )
    )

    for label, edge in (thresholds or {}).items():
        figure.add_vline(
            x=edge,
            line_width=1,
            line_color=INK_MUTED,
            annotation_text=label,
            annotation_position="top",
            annotation_font=dict(size=11, color=INK_MUTED),
        )

    figure.update_layout(
        height=height,
        xaxis_title="Churn probability",
        yaxis_title="Customers",
        bargap=0.06,  # a histogram reads as a continuous shape, so the bins sit close together
        showlegend=False,
    )
    figure.update_xaxes(tickformat=".0%", range=[0, 1])
    return figure


def risk_level_distribution(
    frame: pd.DataFrame, *, height: int = 340, horizontal: bool = False
) -> go.Figure:
    """Customers per risk band.

    Risk is a severity scale, so the bars wear the reserved status palette rather than
    categorical slots -- green through red carries the meaning directly. With only four bars,
    every one is direct-labelled; that stays readable where labelling a dense chart would not.
    """
    counts = (
        frame["risk_level"].value_counts().reindex(RISK_ORDER, fill_value=0).astype(int)
    )
    total = int(counts.sum())
    colours = [RISK_COLOURS[level] for level in RISK_ORDER]
    shares = [(c / total if total else 0) for c in counts]
    # Headroom for the outside labels. Without an explicit range Plotly fits the axis to the
    # bars, and the count sitting past the tallest bar gets clipped by the plot edge.
    headroom = [0, (int(counts.max()) or 1) * 1.18]

    if horizontal:
        order = list(reversed(RISK_ORDER))
        values = counts.reindex(order)
        figure = go.Figure(
            go.Bar(
                y=order,
                x=values,
                orientation="h",
                width=0.45,
                marker=dict(color=[RISK_COLOURS[l] for l in order], cornerradius=4, line=BAR_LINE),
                text=[f"{v:,}" for v in values],
                textposition="outside",
                customdata=[[v / total if total else 0] for v in values],
                hovertemplate="%{y}: %{x:,} customers (%{customdata[0]:.1%})<extra></extra>",
            )
        )
        figure.update_layout(height=height, xaxis_title="Customers", yaxis_title=None)
        figure.update_xaxes(range=headroom)
    else:
        figure = go.Figure(
            go.Bar(
                x=list(RISK_ORDER),
                y=counts,
                # A quarter of the category slot. With only four bands a full-width bar becomes a
                # thick saturated block; the leftover slot is meant to be air.
                width=0.25,
                marker=dict(color=colours, cornerradius=4, line=BAR_LINE),
                text=[f"{v:,}" for v in counts],
                textposition="outside",
                customdata=[[s] for s in shares],
                hovertemplate="%{x}: %{y:,} customers (%{customdata[0]:.1%})<extra></extra>",
            )
        )
        figure.update_layout(height=height, xaxis_title=None, yaxis_title="Customers")
        figure.update_yaxes(range=headroom)

    figure.update_layout(showlegend=False, uniformtext=dict(mode="hide", minsize=10))
    figure.update_traces(textfont=dict(color=INK_MUTED, size=11))
    return figure


def probability_by_group(
    frame: pd.DataFrame,
    group: str,
    *,
    title_value: str = "Mean churn probability",
    height: int = 380,
    top_n: int | None = None,
) -> go.Figure:
    """Mean churn probability per category, as a horizontal bar chart.

    One measure over nominal categories, so every bar takes the same colour. Shading them by
    value would double-encode the length as hue and spend the only free channel restating what
    the bar already shows.
    """
    summary = (
        frame.groupby(group, observed=True)
        .agg(value=("churn_probability", "mean"), customers=("customer_id", "size"))
        .sort_values("value")
    )
    if top_n:
        summary = summary.tail(top_n)

    figure = go.Figure(
        go.Bar(
            y=summary.index.astype(str),
            x=summary["value"],
            orientation="h",
            marker=dict(color=categorical(0), cornerradius=4, line=BAR_LINE),
            text=[f"{v:.0%}" for v in summary["value"]],
            textposition="outside",
            customdata=summary[["customers"]].to_numpy(),
            hovertemplate="%{y}<br>" + title_value + ": %{x:.1%}<br>"
            "%{customdata[0]:,} customers<extra></extra>",
        )
    )
    figure.update_layout(
        height=max(height, 34 * len(summary) + 90),
        xaxis_title=title_value,
        yaxis_title=None,
        showlegend=False,
    )
    figure.update_xaxes(tickformat=".0%", range=[0, min(1.0, float(summary["value"].max()) * 1.25)])
    figure.update_traces(textfont=dict(color=INK_MUTED, size=11))
    figure.update_yaxes(linecolor=AXIS)
    return figure
