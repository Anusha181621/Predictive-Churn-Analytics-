"""Breakdown charts: a measure split across segments, geography, channel or category.

Two forms live here and the distinction matters:

*   :func:`risk_mix` answers "**what is this group made of**" -- risk bands within a category. It
    is a stacked bar in the status palette, showing absolute counts so a 241-customer segment
    does not look the same size as a 3-customer one.
*   :func:`measure_by_group` answers "**how much**" -- one measure across nominal categories. Every
    bar is the same colour. Shading bars by their own value is a value-ramp on nominal
    categories: it double-encodes length as hue and burns the free channel on nothing.
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

from app.theme import INK_MUTED, RISK_COLOURS, RISK_ORDER, SURFACE, categorical

__all__ = ["risk_mix", "measure_by_group", "segment_counts"]

BAR_LINE = dict(color=SURFACE, width=2)


def _height(rows: int, base: int, per_row: int = 34) -> int:
    return max(base, rows * per_row + 96)


def risk_mix(
    frame: pd.DataFrame,
    group: str,
    *,
    height: int = 380,
    top_n: int | None = 12,
    order_by_risk: bool = True,
) -> go.Figure:
    """Customers per risk band within each category, as a horizontal stacked bar.

    Ordered by the share of High + Critical rather than by total size, so the categories that
    need attention rise to the top instead of merely the biggest ones.
    """
    pivot = (
        frame.pivot_table(
            index=group, columns="risk_level", values="customer_id", aggfunc="size", observed=True
        )
        .reindex(columns=list(RISK_ORDER))
        .fillna(0)
        .astype(int)
    )
    if pivot.empty:
        return go.Figure()

    totals = pivot.sum(axis=1)
    if order_by_risk:
        exposure = (pivot.get("High", 0) + pivot.get("Critical", 0)) / totals.replace(0, 1)
        pivot = pivot.loc[exposure.sort_values().index]
    else:
        pivot = pivot.loc[totals.sort_values().index]

    if top_n:
        pivot = pivot.tail(top_n)
    totals = pivot.sum(axis=1)

    figure = go.Figure()
    for level in RISK_ORDER:
        values = pivot[level]
        figure.add_bar(
            y=pivot.index.astype(str),
            x=values,
            name=level,
            orientation="h",
            marker=dict(color=RISK_COLOURS[level], line=BAR_LINE),
            customdata=(values / totals.replace(0, 1)).to_numpy().reshape(-1, 1),
            hovertemplate=f"%{{y}}<br>{level}: %{{x:,}} customers "
            "(%{customdata[0]:.1%} of the group)<extra></extra>",
        )

    figure.update_layout(
        barmode="stack",
        height=_height(len(pivot), height),
        xaxis_title="Customers",
        yaxis_title=None,
        bargap=0.35,
        legend=dict(traceorder="normal"),
    )
    return figure


def measure_by_group(
    frame: pd.DataFrame,
    group: str,
    measure: str,
    *,
    label: str,
    height: int = 380,
    top_n: int | None = 12,
    money: bool = True,
    colour_slot: int = 0,
    colour_by: dict[str, str] | None = None,
    order: tuple[str, ...] | None = None,
) -> go.Figure:
    """Sum of ``measure`` per category, as a horizontal bar chart in a single colour.

    ``colour_by`` is the one exception to the single-colour rule, for categories that carry
    *meaning* rather than mere identity -- risk bands, which take the status palette. It is not a
    way to shade bars by their own value.
    """
    summary = (
        frame.groupby(group, observed=True)
        .agg(value=(measure, "sum"), customers=("customer_id", "size"))
        .sort_values("value")
    )
    summary = summary[summary["value"] != 0]
    if order:
        # A severity scale reads in its own order, not sorted by size.
        summary = summary.reindex([o for o in reversed(order) if o in summary.index])
    if top_n and not order:
        summary = summary.tail(top_n)
    if summary.empty:
        return go.Figure()

    fill = (
        [colour_by.get(str(name), categorical(colour_slot)) for name in summary.index]
        if colour_by
        else categorical(colour_slot)
    )

    fmt = ",.0f"
    text = [f"{v:,.0f}" for v in summary["value"]]
    hover = "%{y}<br>" + label + ": %{x:,.0f}<br>%{customdata[0]:,} customers<extra></extra>"

    figure = go.Figure(
        go.Bar(
            y=summary.index.astype(str),
            x=summary["value"],
            orientation="h",
            marker=dict(color=fill, cornerradius=4, line=BAR_LINE),
            text=text,
            textposition="outside",
            customdata=summary[["customers"]].to_numpy(),
            hovertemplate=hover,
        )
    )
    figure.update_layout(
        height=_height(len(summary), height),
        xaxis_title=label,
        yaxis_title=None,
        showlegend=False,
    )
    figure.update_xaxes(
        tickformat=fmt,
        range=[0, float(summary["value"].max()) * 1.22],
    )
    figure.update_traces(textfont=dict(color=INK_MUTED, size=11))
    return figure


def segment_counts(
    counts: pd.Series,
    *,
    label: str = "Customers",
    height: int = 420,
    colour_slot: int = 0,
    secondary: pd.Series | None = None,
    secondary_label: str = "Also flagged",
) -> go.Figure:
    """Customers per segment.

    When ``secondary`` is supplied the chart becomes two grouped series -- customers whose
    *primary* segment this is, against everyone carrying the flag at all. The gap between them is
    the point of the multi-label design, so both are drawn rather than described in prose.
    """
    ordered = counts.sort_values()
    figure = go.Figure()
    figure.add_bar(
        y=ordered.index.astype(str),
        x=ordered.to_numpy(),
        orientation="h",
        name=label,
        marker=dict(color=categorical(colour_slot), cornerradius=4, line=BAR_LINE),
        hovertemplate="%{y}<br>" + label + ": %{x:,}<extra></extra>",
    )
    if secondary is not None:
        aligned = secondary.reindex(ordered.index).fillna(0).astype(int)
        figure.add_bar(
            y=ordered.index.astype(str),
            x=aligned.to_numpy(),
            orientation="h",
            name=secondary_label,
            marker=dict(color=categorical(1), cornerradius=4, line=BAR_LINE),
            hovertemplate="%{y}<br>" + secondary_label + ": %{x:,}<extra></extra>",
        )

    figure.update_layout(
        height=_height(len(ordered), height, per_row=38),
        xaxis_title="Customers",
        yaxis_title=None,
        barmode="group",
        bargap=0.32,
        bargroupgap=0.12,
        showlegend=secondary is not None,
    )
    return figure
