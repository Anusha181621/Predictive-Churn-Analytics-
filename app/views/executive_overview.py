"""Executive Overview -- the board-level read of the customer base.

Deliberately unfiltered. This is the page someone quotes in a meeting, so it always describes the
whole book; the filtered views live on Churn Risk and Revenue at Risk. Every figure here is read
from the pipeline's artefacts, and the two that depend on the retention-propensity assumption are
labelled as such rather than presented alongside the measured ones without comment.
"""

from __future__ import annotations

import streamlit as st

from app.charts.breakdowns import measure_by_group, risk_mix
from app.charts.distributions import risk_level_distribution
from app.components.kpi import Kpi, hero, kpi_row
from app.components.layout import assumption_notice, chart_card, page_header, section
from app.data_access import (
    ACTIVE_RECENCY_DAYS,
    AT_RISK_PROBABILITY,
    campaign_summary,
    load_customer_master,
    load_data_quality,
    missing,
    prediction_date,
    require,
)
from app.formatting import integer, money, money_compact, percent, signed_percent
from app.theme import RISK_COLOURS, RISK_ORDER
from src.config.settings import get_settings


def render() -> None:
    require("features", "predictions", "scores", "recommendations", "explanations")

    settings = get_settings()
    master = load_customer_master()
    as_of = prediction_date(master)
    summary = campaign_summary(master)

    page_header(
        "Executive Overview",
        "Churn exposure and the retention opportunity across the whole customer base",
        as_of=as_of,
    )

    total = len(master)
    high = int(master["risk_level"].eq("High").sum())
    critical = int(master["risk_level"].eq("Critical").sum())
    at_risk = int(master["is_at_risk"].sum())
    active = int(master["is_active"].sum())
    revenue_at_risk = float(master["revenue_at_risk"].sum())
    expected_future = float(master["expected_future_revenue"].sum())

    hero(
        "Revenue at risk over the next 180 days",
        money(revenue_at_risk),
        f"{percent(revenue_at_risk / expected_future) if expected_future else '—'} of the "
        f"{money(expected_future)} expected future revenue. "
        "Churn probability × expected future revenue, per customer.",
    )

    st.markdown("")

    kpi_row(
        [
            Kpi("Total customers", integer(total), "One row per Customer ID"),
            Kpi(
                "Active customers",
                integer(active),
                f"Bought within {ACTIVE_RECENCY_DAYS} days · {percent(active / total)}",
            ),
            Kpi(
                "At-risk customers",
                integer(at_risk),
                f"Churn probability ≥ {AT_RISK_PROBABILITY:.0%} · {percent(at_risk / total)}",
                tone="bad" if at_risk / total > 0.5 else "neutral",
            ),
            Kpi(
                "High risk",
                integer(high),
                f"{settings.risk_threshold_high:.0%} ≤ churn probability "
                f"< {settings.risk_threshold_critical:.0%}",
            ),
            Kpi(
                "Critical risk",
                integer(critical),
                f"Churn probability ≥ {settings.risk_threshold_critical:.0%}",
                tone="bad",
            ),
        ]
    )

    st.markdown("")

    kpi_row(
        [
            Kpi(
                "Predicted churn rate",
                percent(float(master["churn_probability"].mean())),
                "Mean calibrated churn probability",
            ),
            Kpi(
                "Revenue at risk",
                money_compact(revenue_at_risk),
                f"{money_compact(float(master.loc[master['risk_level'].isin(['High', 'Critical']), 'revenue_at_risk'].sum()))} "
                "in the High and Critical bands",
            ),
            Kpi(
                "Expected retained revenue",
                money_compact(summary["expected_retained"]),
                f"Across {integer(summary['targeted'])} targeted customers · ASSUMPTION-dependent",
            ),
            Kpi(
                "Expected ROI",
                signed_percent(summary["roi"]),
                f"{money(summary['cost'])} campaign cost · ASSUMPTION-dependent",
                tone="good" if summary["roi"] > 0 else "bad",
            ),
        ]
    )

    assumption_notice(
        "<b>Expected retained revenue and ROI rest on an assumption, not a measurement.</b> "
        "Retention propensity — the chance that contacting a customer changes their behaviour — "
        "needs a campaign log and an untreated control group to estimate, and this dataset has "
        "neither. The base rate is a configurable 25%. <b>Revenue at risk is deliberately free of "
        "that assumption</b>, so it stands on its own."
    )

    # ----------------------------------------------------------------- charts
    section("Where the risk sits")

    left, right = st.columns(2, gap="medium")
    with left:
        chart_card(
            "Customers by risk level",
            "Bands are configurable; the lower edge is inclusive, so a probability of exactly "
            f"{settings.risk_threshold_high:.2f} is High.",
        )
        st.plotly_chart(
            risk_level_distribution(master), width="stretch", key="exec_risk_levels"
        )
    with right:
        chart_card(
            "Revenue at risk by risk level",
            "Exposure follows value as well as probability, so the biggest band is not "
            "automatically the biggest number.",
        )
        st.plotly_chart(
            measure_by_group(
                master,
                "risk_level",
                "revenue_at_risk",
                label="Revenue at risk",
                colour_by=RISK_COLOURS,
                order=RISK_ORDER,
                height=340,
            ),
            width="stretch",
            key="exec_rar_by_risk",
        )

    left, right = st.columns(2, gap="medium")
    with left:
        chart_card(
            "Risk by segment",
            "Ordered by the share of High and Critical customers, not by segment size.",
        )
        st.plotly_chart(
            risk_mix(master, "primary_segment", height=420), width="stretch", key="exec_by_segment"
        )
    with right:
        chart_card(
            "Risk by acquisition channel",
            "The channel that acquired the customer, from Customer.csv.",
        )
        st.plotly_chart(
            risk_mix(master, "acquisition_channel", height=420),
            width="stretch",
            key="exec_by_channel",
        )

    _data_quality_panel()


def _data_quality_panel() -> None:
    """The validation report, which exists specifically for the dashboard to display.

    Kept collapsed: it matters when it fails, and should not compete with the business numbers
    when it passes.
    """
    if missing("quality"):
        return
    report = load_data_quality()
    totals = report.get("summary", {})
    ok = report.get("ok", False)
    label = "passing" if ok else "FAILING"

    with st.expander(
        f"Source data quality — {totals.get('passed', 0)}/{totals.get('total', 0)} checks {label}",
        expanded=not ok,
    ):
        st.markdown(
            f"Validated directly against the four source CSVs: "
            f"**{totals.get('errors', 0)}** errors, **{totals.get('warnings', 0)}** warnings. "
            "The validators only ever report — nothing in this project rewrites `data/`."
        )
        dataset = report.get("dataset", {})
        rate = dataset.get("unit_return_rate")
        if rate is not None:
            st.markdown(
                f"Measured return rate: **{percent(rate, 2)}** of purchased units "
                f"({integer(dataset.get('returned_units'))} of "
                f"{integer(dataset.get('purchased_units'))}). The line-level rate is a different "
                "number and is easy to confuse with it."
            )
        failures = [
            check
            for table in report.get("tables", {}).values()
            for check in table.get("checks", [])
            if not check.get("passed", True)
        ]
        if failures:
            st.error(f"{len(failures)} check(s) failed — the numbers above may not be reliable.")
            for check in failures[:10]:
                st.markdown(f"- **{check.get('table')}: {check.get('check')}** — {check.get('detail')}")
