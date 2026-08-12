"""Churn Risk -- the distribution of risk, and how it varies across the business.

All seven filters scope the whole page: every chart below re-renders against the same slice, so
two charts under one heading can never describe two different populations.
"""

from __future__ import annotations

import streamlit as st

from app.charts.breakdowns import risk_mix
from app.charts.distributions import (
    churn_probability_distribution,
    probability_by_group,
    risk_level_distribution,
)
from app.components.filters import filter_caption, render_filters
from app.components.kpi import Kpi, kpi_row
from app.components.layout import chart_card, page_header, section
from app.components.tables import data_table
from app.data_access import load_customer_master, prediction_date, require
from app.formatting import integer, money, percent
from src.config.settings import get_settings

TABLE_COLUMNS = [
    "customer_id",
    "churn_probability",
    "risk_level",
    "customer_value_segment",
    "primary_segment",
    "country",
    "city",
    "acquisition_channel",
    "preferred_category",
    "recency_days",
    "total_orders",
    "lifetime_revenue",
    "revenue_at_risk",
]


def render() -> None:
    require("features", "predictions", "scores", "recommendations", "explanations")

    master = load_customer_master()
    as_of = prediction_date(master)

    page_header(
        "Churn Risk",
        "How churn probability is distributed, and where it concentrates",
        as_of=as_of,
    )

    frame, selections = render_filters(master, namespace="risk")
    if frame.empty:
        st.warning("No customers match the current filters. Clear one to see results.")
        return

    st.caption(f"Showing **{len(frame):,}** customers · {filter_caption(selections)}")

    kpi_row(
        [
            Kpi("Customers", integer(len(frame)), f"{percent(len(frame) / len(master))} of the book"),
            Kpi(
                "Mean churn probability",
                percent(float(frame["churn_probability"].mean())),
                f"Whole book: {percent(float(master['churn_probability'].mean()))}",
            ),
            Kpi(
                "High + Critical",
                integer(int(frame["risk_level"].isin(["High", "Critical"]).sum())),
                percent(float(frame["risk_level"].isin(["High", "Critical"]).mean())),
            ),
            Kpi(
                "Revenue at risk",
                money(float(frame["revenue_at_risk"].sum())),
                "Churn probability × expected future revenue",
            ),
        ]
    )

    settings = get_settings()
    thresholds = {
        "Medium": settings.risk_threshold_medium,
        "High": settings.risk_threshold_high,
        "Critical": settings.risk_threshold_critical,
    }

    section("Distribution")
    left, right = st.columns(2, gap="medium")
    with left:
        chart_card(
            "Churn probability distribution",
            "Isotonic calibration maps customers onto a limited set of distinct probabilities, "
            "which is why the shape is stepped rather than smooth.",
        )
        st.plotly_chart(
            churn_probability_distribution(frame, thresholds=thresholds),
            width="stretch",
            key="cr_hist",
        )
    with right:
        chart_card("Risk-level distribution", "The same customers, cut into the four bands.")
        st.plotly_chart(
            risk_level_distribution(frame, horizontal=True), width="stretch", key="cr_bands"
        )

    section("Where risk concentrates", "Each chart is ordered by High + Critical share.")

    left, right = st.columns(2, gap="medium")
    with left:
        chart_card("Risk by segment")
        st.plotly_chart(
            risk_mix(frame, "primary_segment", height=430), width="stretch", key="cr_segment"
        )
    with right:
        chart_card("Risk by acquisition channel")
        st.plotly_chart(
            risk_mix(frame, "acquisition_channel", height=430), width="stretch", key="cr_channel"
        )

    left, right = st.columns(2, gap="medium")
    with left:
        chart_card("Risk by geography", "Country of the billing address.")
        st.plotly_chart(risk_mix(frame, "country", height=300), width="stretch", key="cr_country")
    with right:
        chart_card("Risk by product category", "The category the customer buys from most.")
        st.plotly_chart(
            risk_mix(frame, "preferred_category", height=300), width="stretch", key="cr_category"
        )

    chart_card(
        "Mean churn probability by city",
        "Twelve highest cities in the current slice. Cities carry few customers each, so read "
        "these as indicative rather than precise.",
    )
    st.plotly_chart(
        probability_by_group(frame, "city", top_n=12, height=420),
        width="stretch",
        key="cr_city",
    )

    section("Customers in this slice")
    data_table(
        frame.sort_values("churn_probability", ascending=False)[TABLE_COLUMNS],
        TABLE_COLUMNS,
        download_name="churn_risk_customers.csv",
        key="cr_table",
        height=420,
        caption="Sorted by churn probability. Every value is also downloadable at full precision.",
    )
