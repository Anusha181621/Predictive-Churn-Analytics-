"""Revenue at Risk -- the money behind the probabilities.

Revenue at risk is ``churn probability × expected future revenue``, computed per customer and then
summed. Expected future revenue is a frequency × value projection over the same horizon the churn
probability describes -- ``CHURN_INACTIVITY_DAYS`` -- because mixing a 90-day probability with an
annual revenue figure would overstate the exposure roughly fourfold.

Two guards in that projection are worth knowing when reading these totals. The rate denominator is
floored at 180 days (a separate setting, and deliberately longer than the churn horizon) and every
projection is capped at twice observed lifetime revenue, so a customer with one large order and
three weeks of history cannot be extrapolated into the most valuable account in the book.
"""

from __future__ import annotations

import streamlit as st

from app.charts.breakdowns import measure_by_group
from app.components.filters import render_filters
from app.components.kpi import Kpi, hero, kpi_row
from app.components.layout import chart_card, page_header, section
from app.components.tables import data_table
from app.data_access import load_customer_master, prediction_date, require
from app.formatting import horizon_phrase, integer, money, money_compact, percent
from app.theme import RISK_COLOURS, RISK_ORDER

TABLE_COLUMNS = [
    "customer_id",
    "revenue_at_risk",
    "churn_probability",
    "risk_level",
    "expected_future_revenue",
    "lifetime_revenue",
    "customer_value_segment",
    "primary_segment",
    "country",
    "acquisition_channel",
    "preferred_category",
]


def render() -> None:
    require("features", "predictions", "scores", "recommendations", "explanations")

    master = load_customer_master()
    as_of = prediction_date(master)

    page_header(
        "Revenue at Risk",
        "Exposure in euro, and which parts of the business carry it",
        as_of=as_of,
    )

    frame, _ = render_filters(master, namespace="rar")
    if frame.empty:
        st.warning(
            "No customers match the current filters. Use **Clear all** above, or open "
            "**Filters** and widen one."
        )
        return

    exposure = float(frame["revenue_at_risk"].sum())
    expected = float(frame["expected_future_revenue"].sum())
    book = float(master["revenue_at_risk"].sum())
    top_decile_cut = frame["revenue_at_risk"].quantile(0.9)
    top_decile = float(frame.loc[frame["revenue_at_risk"] >= top_decile_cut, "revenue_at_risk"].sum())

    hero(
        "Total revenue at risk",
        money(exposure),
        f"{percent(exposure / book) if book else '—'} of the {money(book)} across the whole book",
    )
    st.markdown("")

    kpi_row(
        [
            Kpi(
                "Expected future revenue",
                money_compact(expected),
                f"Projected over the {horizon_phrase()}",
            ),
            Kpi(
                "Share at risk",
                percent(exposure / expected) if expected else "—",
                "Revenue at risk ÷ expected future revenue",
            ),
            Kpi(
                "In High + Critical",
                money_compact(
                    float(
                        frame.loc[
                            frame["risk_level"].isin(["High", "Critical"]), "revenue_at_risk"
                        ].sum()
                    )
                ),
                f"{integer(int(frame['risk_level'].isin(['High', 'Critical']).sum()))} customers",
            ),
            Kpi(
                "Top decile of customers",
                money_compact(top_decile),
                f"{percent(top_decile / exposure) if exposure else '—'} of the exposure",
            ),
        ]
    )

    section("Where the exposure sits")

    left, right = st.columns(2, gap="medium")
    with left:
        chart_card("Revenue at risk by segment")
        st.plotly_chart(
            measure_by_group(
                frame, "primary_segment", "revenue_at_risk", label="Revenue at risk", height=420
            ),
            width="stretch",
            key="rar_segment",
        )
    with right:
        chart_card("Revenue at risk by acquisition channel")
        st.plotly_chart(
            measure_by_group(
                frame,
                "acquisition_channel",
                "revenue_at_risk",
                label="Revenue at risk",
                height=420,
            ),
            width="stretch",
            key="rar_channel",
        )

    left, right = st.columns(2, gap="medium")
    with left:
        chart_card("Revenue at risk by geography", "Country of the billing address.")
        st.plotly_chart(
            measure_by_group(
                frame, "country", "revenue_at_risk", label="Revenue at risk", height=280
            ),
            width="stretch",
            key="rar_country",
        )
    with right:
        chart_card(
            "Revenue at risk by product category",
            "Attributed to the category each customer buys from most.",
        )
        st.plotly_chart(
            measure_by_group(
                frame,
                "preferred_category",
                "revenue_at_risk",
                label="Revenue at risk",
                height=280,
            ),
            width="stretch",
            key="rar_category",
        )

    chart_card(
        "Revenue at risk by risk level",
        "Ordered by severity rather than by size, so the bands read in their natural order.",
    )
    st.plotly_chart(
        measure_by_group(
            frame,
            "risk_level",
            "revenue_at_risk",
            label="Revenue at risk",
            colour_by=RISK_COLOURS,
            order=RISK_ORDER,
            height=300,
        ),
        width="stretch",
        key="rar_band",
    )

    section("Customers by exposure", "Sorted by revenue at risk, highest first.")
    data_table(
        frame.sort_values("revenue_at_risk", ascending=False)[TABLE_COLUMNS],
        TABLE_COLUMNS,
        download_name="revenue_at_risk_customers.csv",
        key="rar_table",
        height=460,
        caption=(
            "Exports the current slice exactly as filtered. Revenue at risk here is the decision "
            "layer's figure — churn probability × expected future revenue."
        ),
    )
