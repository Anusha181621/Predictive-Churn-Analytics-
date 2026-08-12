"""Customer Segmentation -- twelve business segments, and the fact that customers carry several.

The segments are deliberately **multi-label**. A customer who is both a High-Return Customer and
High-Value At Risk needs both facts: one says act now, the other says be careful what you offer.
So this page shows the primary segment *and* the full flag membership, and the gap between the two
counts is the point rather than an inconsistency.

A thirteenth value, ``Steady Customers``, appears as the primary segment for customers who match
none of the twelve. It is a fallback, not a business segment, and is labelled as such.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from app.charts.breakdowns import measure_by_group, risk_mix, segment_counts
from app.components.filters import render_filters
from app.components.kpi import Kpi, kpi_row
from app.components.layout import chart_card, page_header, section
from app.components.tables import data_table
from app.data_access import (
    SEGMENT_FLAG_COLUMNS,
    load_customer_master,
    prediction_date,
    require,
)
from app.formatting import integer, money, percent, ratio

FALLBACK_SEGMENT = "Steady Customers"

CUSTOMER_COLUMNS = [
    "customer_id",
    "primary_segment",
    "all_segments",
    "churn_probability",
    "risk_level",
    "customer_value_segment",
    "lifetime_revenue",
    "revenue_at_risk",
    "recommended_action",
    "priority",
]


def render() -> None:
    require("features", "predictions", "scores", "recommendations", "explanations")

    master = load_customer_master()
    as_of = prediction_date(master)

    page_header(
        "Customer Segmentation",
        "Value, risk and behaviour — as overlapping dimensions rather than one rigid label",
        as_of=as_of,
    )

    frame, _ = render_filters(master, namespace="seg")
    if frame.empty:
        st.warning("No customers match the current filters. Clear one to see results.")
        return

    flags = [c for c in SEGMENT_FLAG_COLUMNS if c in frame.columns]
    membership = frame[flags].sum().astype(int) if flags else pd.Series(dtype=int)
    per_customer = frame[flags].sum(axis=1) if flags else pd.Series(0, index=frame.index)
    multi = int((per_customer > 1).sum())

    kpi_row(
        [
            Kpi("Customers", integer(len(frame))),
            Kpi(
                "Segments in use",
                integer(frame["primary_segment"].nunique()),
                "As a primary label",
            ),
            Kpi(
                "Carry more than one segment",
                integer(multi),
                percent(multi / len(frame)) + " of customers",
            ),
            Kpi(
                "Mean segments per customer",
                ratio(float(per_customer.mean()), 2),
                "Across the twelve flags",
            ),
            Kpi(
                "Unsegmented",
                integer(int(frame["primary_segment"].eq(FALLBACK_SEGMENT).sum())),
                f"Fall back to “{FALLBACK_SEGMENT}”",
            ),
        ]
    )

    section(
        "Primary segment against full membership",
        "The left bar is where a customer was finally placed; the right is everyone who qualifies. "
        "The difference is the multi-label design doing its job.",
    )
    primary = frame["primary_segment"].value_counts()
    if flags:
        ordering = membership.sort_values(ascending=False).index.tolist()
        primary_aligned = primary.reindex(ordering).fillna(0).astype(int)
        st.plotly_chart(
            segment_counts(
                primary_aligned,
                label="Primary segment",
                secondary=membership.reindex(ordering),
                secondary_label="Also flagged",
                height=460,
            ),
            width="stretch",
            key="seg_membership",
        )
    else:
        st.plotly_chart(
            segment_counts(primary, label="Customers", height=440),
            width="stretch",
            key="seg_primary",
        )

    section("What each segment looks like")
    profile = _segment_profile(frame)
    st.dataframe(
        profile,
        column_config={
            "Segment": st.column_config.Column("Segment"),
            "Customers": st.column_config.NumberColumn("Customers", format="localized"),
            "Mean churn probability": st.column_config.NumberColumn(
                "Mean churn probability", format="percent"
            ),
            "Revenue at risk": st.column_config.NumberColumn("Revenue at risk", format="euro"),
            "Lifetime revenue": st.column_config.NumberColumn("Lifetime revenue", format="euro"),
            "Mean recency (days)": st.column_config.NumberColumn(
                "Mean recency (days)", format="localized"
            ),
            "Mean orders": st.column_config.NumberColumn("Mean orders", format="%.1f"),
            "Targeted": st.column_config.NumberColumn("Targeted", format="localized"),
        },
        hide_index=True,
        width="stretch",
    )

    left, right = st.columns(2, gap="medium")
    with left:
        chart_card("Risk mix by segment")
        st.plotly_chart(
            risk_mix(frame, "primary_segment", height=440, top_n=14),
            width="stretch",
            key="seg_riskmix",
        )
    with right:
        chart_card("Revenue at risk by segment")
        st.plotly_chart(
            measure_by_group(
                frame,
                "primary_segment",
                "revenue_at_risk",
                label="Revenue at risk",
                height=440,
                top_n=14,
            ),
            width="stretch",
            key="seg_rar",
        )

    section("Customers")
    data_table(
        frame.sort_values("revenue_at_risk", ascending=False)[CUSTOMER_COLUMNS],
        CUSTOMER_COLUMNS,
        download_name="customer_segments.csv",
        key="seg_table",
        height=440,
        caption="`All segments` lists every flag a customer carries, not only the primary one.",
    )


def _segment_profile(frame: pd.DataFrame) -> pd.DataFrame:
    """One row per primary segment, with the numbers that distinguish them."""
    grouped = frame.groupby("primary_segment", observed=True)
    profile = pd.DataFrame(
        {
            "Customers": grouped.size(),
            "Mean churn probability": grouped["churn_probability"].mean(),
            "Revenue at risk": grouped["revenue_at_risk"].sum(),
            "Lifetime revenue": grouped["lifetime_revenue"].sum(),
            "Mean recency (days)": grouped["recency_days"].mean().round(0),
            "Mean orders": grouped["total_orders"].mean(),
            "Targeted": grouped["is_targeted"].sum().astype(int),
        }
    )
    return (
        profile.sort_values("Revenue at risk", ascending=False)
        .reset_index()
        .rename(columns={"primary_segment": "Segment"})
    )
