"""Retention Action Center -- the working list, ordered by what is worth recovering.

Sorted by retention opportunity score (``revenue at risk × retention propensity``) rather than by
churn probability, because the most likely churner is not the most valuable one to save.

The suppression list is shown alongside the targets and broken out by *reason*. "Already highly
engaged" and "not economic to contact" are both reasons not to send a campaign, but they call for
completely different follow-up, and a single count of "221 not targeted" hides that.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from app.charts.breakdowns import segment_counts
from app.components.filters import render_filters
from app.components.kpi import Kpi, kpi_row
from app.components.layout import assumption_notice, chart_card, page_header, section
from app.components.tables import data_table
from app.data_access import campaign_summary, load_customer_master, prediction_date, require
from app.formatting import integer, money, percent, signed_percent

#: The columns the brief asks for, in its order, followed by the economics.
ACTION_COLUMNS = [
    "customer_id",
    "risk_level",
    "churn_probability",
    "revenue_at_risk",
    "customer_value_segment",
    "top_driver",
    "recommended_action",
    "recommended_channel",
    "recommended_offer",
    "expected_roi",
    "priority",
]

DETAIL_COLUMNS = [
    "customer_id",
    "primary_segment",
    "recommended_action",
    "recommended_category",
    "recommended_sku",
    "recommended_product",
    "recommended_offer",
    "campaign_cost",
    "expected_retained_revenue",
    "expected_roi",
    "reason",
]

SUPPRESSED_COLUMNS = [
    "customer_id",
    "churn_probability",
    "risk_level",
    "primary_segment",
    "revenue_at_risk",
    "suppressed_action",
    "reason",
]


def render() -> None:
    require("features", "predictions", "scores", "recommendations", "explanations")

    master = load_customer_master()
    as_of = prediction_date(master)

    page_header(
        "Retention Action Center",
        "Who to contact first, with what, and what it is expected to return",
        as_of=as_of,
    )

    frame, _ = render_filters(master, namespace="ac")
    if frame.empty:
        st.warning("No customers match the current filters. Clear one to see results.")
        return

    summary = campaign_summary(frame)
    targeted = frame[frame["is_targeted"]].sort_values(
        "retention_opportunity_score", ascending=False
    )
    suppressed = frame[~frame["is_targeted"]]

    kpi_row(
        [
            Kpi("Customers targeted", integer(summary["targeted"]), f"of {integer(len(frame))} in view"),
            Kpi("Campaign cost", money(summary["cost"]), "Contact + incentive"),
            Kpi(
                "Expected revenue retained",
                money(summary["expected_retained"]),
                "ASSUMPTION-dependent",
            ),
            Kpi(
                "Expected ROI",
                signed_percent(summary["roi"]),
                "ASSUMPTION-dependent",
                tone="good" if summary["roi"] > 0 else "bad",
            ),
            Kpi(
                "Customers retained",
                integer(summary["customers_retained"]),
                "Σ churn probability × propensity",
            ),
        ]
    )

    assumption_notice(
        "<b>Expected ROI depends on the assumed retention propensity (base 25%).</b> It ranks "
        "customers defensibly under a stated assumption; it is not a forecast. "
        "<code>Revenue at risk</code> and <code>Main churn driver</code> do not depend on it."
    )

    section(
        "Priority list",
        "Sorted by retention opportunity — revenue at risk weighted by how movable the customer is.",
    )
    data_table(
        targeted[ACTION_COLUMNS],
        ACTION_COLUMNS,
        download_name="retention_action_list.csv",
        key="ac_main",
        height=520,
        caption=(
            "Every action, channel, offer and reason is derived from that customer's own "
            "behaviour — nothing here is a fixed house default."
        ),
    )

    left, right = st.columns([3, 2], gap="medium")
    with left:
        chart_card(
            "Recommended actions",
            "Nine distinct actions are in use across the book; a single blanket action would be "
            "the hardcoding the brief rules out.",
        )
        counts = frame["recommended_action"].value_counts()
        st.plotly_chart(
            segment_counts(counts, label="Customers", height=380),
            width="stretch",
            key="ac_actions",
        )
    with right:
        chart_card("Offers in use", "Discount depth is derived from what each customer responds to.")
        offers = (
            frame.loc[frame["is_targeted"], "recommended_offer"]
            .fillna("—")
            .value_counts()
            .head(8)
        )
        st.plotly_chart(
            segment_counts(offers, label="Customers", height=320, colour_slot=2),
            width="stretch",
            key="ac_offers",
        )

    section("Recommendation detail", "The full recommendation behind each row above.")
    data_table(
        targeted[DETAIL_COLUMNS],
        DETAIL_COLUMNS,
        download_name="retention_recommendation_detail.csv",
        key="ac_detail",
        height=380,
    )

    _suppression_panel(suppressed)


def _suppression_panel(suppressed: pd.DataFrame) -> None:
    """Not-targeted customers, split by why."""
    section(
        f"Not targeted — {len(suppressed):,} customers",
        "Grouped by reason, because 'already engaged' and 'uneconomic' need opposite follow-up.",
    )
    if suppressed.empty:
        st.info("Every customer in this slice is being targeted.")
        return

    reasons = suppressed["reason"].fillna("Unspecified")
    buckets = {
        "Already highly engaged": reasons.str.contains("engaged", case=False, na=False),
        "Uneconomic to contact": reasons.str.contains(
            "roi|economic|cost", case=False, na=False, regex=True
        ),
        "Seasonal and out of season": reasons.str.contains("season", case=False, na=False),
        "Unrecoverable": reasons.str.contains("lost|unrecoverable", case=False, na=False),
    }
    assigned = pd.Series(False, index=suppressed.index)
    tiles = []
    for label, mask in buckets.items():
        fresh = mask & ~assigned
        assigned |= fresh
        if int(fresh.sum()):
            tiles.append(Kpi(label, integer(int(fresh.sum()))))
    remainder = int((~assigned).sum())
    if remainder:
        tiles.append(Kpi("Other reasons", integer(remainder)))
    if tiles:
        kpi_row(tiles)

    st.markdown("")
    data_table(
        suppressed.sort_values("revenue_at_risk", ascending=False)[SUPPRESSED_COLUMNS],
        SUPPRESSED_COLUMNS,
        download_name="retention_suppressed.csv",
        key="ac_suppressed",
        height=320,
        caption=(
            "`Suppressed action` records what was proposed before the ROI guardrail overrode it, "
            "so a rejected recommendation is still auditable."
        ),
    )
