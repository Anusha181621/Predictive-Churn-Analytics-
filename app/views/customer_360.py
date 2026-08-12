"""Customer 360 -- everything the platform knows about one customer, on one screen.

This is the only page that reads the source transactions directly, because a purchase history is
the one thing the aggregated feature table cannot show. Those transactions are clipped to the
prediction date, for the same reason the features are: a history that includes orders placed after
the prediction would not be the history the model saw.
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app.charts.model import customer_drivers
from app.components.kpi import Kpi, kpi_row, pill
from app.components.layout import chart_card, page_header, section
from app.components.tables import data_table
from app.data_access import (
    load_customer_master,
    load_explanations,
    load_source_data,
    prediction_date,
    require,
)
from app.formatting import days, integer, money, percent, ratio, signed_percent
from app.theme import SURFACE, categorical

ORDER_COLUMNS = ["Order date", "Category", "Brand", "SKU", "Units", "Discount %", "Net value"]


def render() -> None:
    require("features", "predictions", "scores", "recommendations", "explanations")

    master = load_customer_master()
    as_of = prediction_date(master)

    page_header(
        "Customer 360",
        "Profile, history, risk, the reasons behind it, and the recommended action",
        as_of=as_of,
    )

    ordered = master.sort_values("retention_opportunity_score", ascending=False)
    options = ordered["customer_id"].tolist()
    st.sidebar.markdown("### Customer")
    st.sidebar.caption("Ordered by retention opportunity, highest first.")
    customer_id = st.sidebar.selectbox("Customer ID", options, index=0, key="c360_pick")

    row = master.loc[master["customer_id"] == customer_id].iloc[0]

    _profile(row)
    _behaviour(row)
    _risk_and_drivers(row, customer_id)
    _recommendation(row)
    _history(customer_id, as_of)


def _profile(row: pd.Series) -> None:
    # Both badges are labelled: risk and priority use the same four band names, so two bare pills
    # reading "Critical" beside each other say nothing about which is which.
    st.markdown(
        f"## {row['customer_id']} &nbsp; "
        f"{pill(row['risk_level'], label='Risk')} &nbsp; "
        f"{pill(row['priority'], kind='priority', label='Priority')}",
        unsafe_allow_html=True,
    )
    st.caption(
        f"{row.get('primary_segment', '—')} · all segments: {row.get('all_segments', '—')}"
    )

    kpi_row(
        [
            Kpi("Age", integer(row.get("age")), str(row.get("age_band", ""))),
            Kpi("Gender", str(row.get("customer_gender", "—"))),
            Kpi("Location", f"{row.get('city', '—')}", str(row.get("country", ""))),
            Kpi("Acquired via", str(row.get("acquisition_channel", "—"))),
            Kpi(
                "Tenure",
                days(row.get("customer_tenure_days")),
                f"First order {str(row.get('first_purchase_date', ''))[:10]}",
            ),
        ]
    )


def _behaviour(row: pd.Series) -> None:
    section("Purchase behaviour")
    kpi_row(
        [
            Kpi("Recency", days(row.get("recency_days")), "Since the last order"),
            Kpi("Frequency", integer(row.get("total_orders")), f"{integer(row.get('total_units'))} units"),
            Kpi("Lifetime revenue", money(row.get("lifetime_revenue")), "Net of discounts"),
            Kpi("Average order value", money(row.get("average_order_value"))),
            Kpi(
                "Return rate",
                percent(row.get("return_rate")),
                f"{integer(row.get('returned_units'))} units returned",
            ),
        ]
    )
    kpi_row(
        [
            Kpi("Preferred category", str(row.get("preferred_category", "—"))),
            Kpi("Preferred brand", str(row.get("preferred_brand", "—"))),
            Kpi("Preferred subcategory", str(row.get("preferred_subcategory", "—"))),
            Kpi(
                "Typical gap between orders",
                days(row.get("median_purchase_gap")),
                f"Current gap {days(row.get('current_purchase_gap'))}",
            ),
            Kpi(
                "Discount dependency",
                ratio(row.get("discount_dependency_score")),
                f"Average discount {percent((row.get('average_discount') or 0) / 100)}",
            ),
        ]
    )


def _risk_and_drivers(row: pd.Series, customer_id: str) -> None:
    section("Churn risk and why")

    kpi_row(
        [
            Kpi(
                "Churn probability",
                percent(row.get("churn_probability")),
                "No purchase in the next 180 days",
            ),
            Kpi("Risk level", str(row.get("risk_level", "—"))),
            Kpi(
                "Revenue at risk",
                money(row.get("revenue_at_risk")),
                "Churn probability × expected future revenue",
            ),
            Kpi(
                "Expected future revenue",
                money(row.get("expected_future_revenue")),
                "Next 180 days",
            ),
        ]
    )

    explanations = load_explanations()
    drivers = explanations[explanations["Customer ID"] == customer_id].sort_values("Driver rank")
    if drivers.empty:
        st.info("No SHAP drivers were recorded for this customer.")
        return

    left, right = st.columns([3, 4], gap="medium")
    with left:
        chart_card(
            "Driver contributions",
            "Ranked by absolute size, so protective factors are shown as well as risks.",
        )
        st.plotly_chart(customer_drivers(drivers), width="stretch", key="c360_drivers")
    with right:
        chart_card(
            "Top churn drivers",
            "Each sentence is composed at run time from this customer's own values and "
            "contributions.",
        )
        for _, driver in drivers.iterrows():
            arrow = "▲" if driver["Contribution"] > 0 else "▼"
            st.markdown(
                f"**{int(driver['Driver rank'])}. {arrow} {driver['Feature label']}** — "
                f"{driver['Human-readable explanation']}"
            )
        st.caption(
            "Contributions are on the model's uncalibrated log-odds scale. Calibration is "
            "monotone, so the ranking and direction carry over to the reported probability, but "
            "the values do not sum to it."
        )


def _recommendation(row: pd.Series) -> None:
    section("Recommended retention action")

    targeted = row.get("recommended_action") != "Do Not Target"
    kpi_row(
        [
            Kpi("Action", str(row.get("recommended_action", "—"))),
            Kpi("Channel", str(row.get("recommended_channel", "—"))),
            Kpi("Category", str(row.get("recommended_category", "—")) or "—"),
            Kpi("Offer", str(row.get("recommended_offer", "—")) or "—"),
        ]
    )
    kpi_row(
        [
            Kpi(
                "Recommended product",
                str(row.get("recommended_sku", "—")) or "—",
                str(row.get("recommended_product", "")),
            ),
            Kpi("Campaign cost", money(row.get("campaign_cost")) if targeted else "—"),
            Kpi(
                "Expected revenue retained",
                money(row.get("expected_retained_revenue")),
                "ASSUMPTION-dependent",
            ),
            Kpi(
                "Expected ROI",
                signed_percent(row.get("expected_roi")) if targeted else "—",
                "ASSUMPTION-dependent",
                tone="good" if targeted and (row.get("expected_roi") or 0) > 0 else "neutral",
            ),
        ]
    )
    st.markdown(f"**Reason.** {row.get('reason', '—')}")
    if not targeted and str(row.get("suppressed_action", "")).strip():
        st.caption(
            f"Originally proposed: **{row['suppressed_action']}** — overridden by the ROI guardrail."
        )
    st.caption(f"Retention propensity basis: {row.get('propensity_basis', '—')}")


def _history(customer_id: str, as_of: str) -> None:
    section("Purchase history")

    data = load_source_data()
    cutoff = pd.Timestamp(as_of)
    lines = data.transactions[
        (data.transactions["customer_id"] == customer_id)
        & (data.transactions["purchase_date"] <= cutoff)
    ]
    if lines.empty:
        st.info("This customer has no orders on or before the prediction date.")
        return

    lines = lines.merge(
        data.products[["sku_id", "category", "subcategory", "brand"]], on="sku_id", how="left"
    )

    monthly = (
        lines.assign(month=lines["purchase_date"].dt.to_period("M").dt.to_timestamp())
        .groupby("month", as_index=False)
        .agg(revenue=("net_order_value", "sum"), orders=("order_id", "nunique"))
    )

    left, right = st.columns([3, 2], gap="medium")
    with left:
        chart_card("Net revenue by month", "Clipped to the prediction date.")
        figure = go.Figure(
            go.Bar(
                x=monthly["month"],
                y=monthly["revenue"],
                marker=dict(
                    color=categorical(0), cornerradius=3, line=dict(color=SURFACE, width=2)
                ),
                customdata=monthly[["orders"]].to_numpy(),
                hovertemplate="%{x|%b %Y}<br>%{y:,.2f}<br>%{customdata[0]} order(s)<extra></extra>",
            )
        )
        figure.update_layout(
            height=320, xaxis_title=None, yaxis_title="Net revenue", showlegend=False
        )
        st.plotly_chart(figure, width="stretch", key="c360_monthly")
    with right:
        chart_card("Summary")
        kpi_row(
            [
                Kpi("Orders", integer(lines["order_id"].nunique())),
                Kpi("Order lines", integer(len(lines))),
            ]
        )
        kpi_row(
            [
                Kpi("Units", integer(int(lines["quantity"].sum()))),
                Kpi("Categories", integer(lines["category"].nunique())),
            ]
        )
        st.markdown("")
        top = (
            lines.groupby("category", as_index=False)["net_order_value"]
            .sum()
            .sort_values("net_order_value", ascending=False)
        )
        st.markdown("**Spend by category**")
        for _, item in top.iterrows():
            st.markdown(f"- {item['category']}: {money(item['net_order_value'], 2)}")

    display = (
        lines.assign(
            **{
                "Order date": lines["purchase_date"].dt.date,
                "Category": lines["category"],
                "Brand": lines["brand"],
                "SKU": lines["sku_id"],
                "Units": lines["quantity"],
                "Discount %": lines["discount_pct"],
                "Net value": lines["net_order_value"],
            }
        )
        .sort_values("purchase_date", ascending=False)[ORDER_COLUMNS]
    )
    data_table(
        display,
        ORDER_COLUMNS,
        download_name=f"{customer_id}_orders.csv",
        key="c360_orders",
        height=320,
    )
