"""Tables and their download buttons.

Every chart in this dashboard has a table twin. That is not a nicety: three of the palette's
categorical slots sit below 3:1 contrast against the chart surface, so the palette validator
requires that values also be reachable without relying on colour. A sortable table plus a CSV
export is that route, and it is what a retention manager wants anyway.

Columns are formatted through one registry rather than per page, so "Revenue at risk" is a euro
figure with the same precision wherever it appears. Values stay **numeric** in the grid --
formatting is display-only -- so sorting a money column orders by amount rather than
alphabetically by its rendered string.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from app.formatting import currency_symbol, horizon_phrase

__all__ = ["COLUMN_LABELS", "data_table", "download_button"]


def _money(label: str) -> "st.column_config.NumberColumn":
    return st.column_config.NumberColumn(label, format="euro")


def _rate(label: str, help_text: str | None = None) -> "st.column_config.NumberColumn":
    return st.column_config.NumberColumn(label, format="percent", help=help_text)


def _count(label: str) -> "st.column_config.NumberColumn":
    return st.column_config.NumberColumn(label, format="localized")


def _decimal(label: str, places: int = 2) -> "st.column_config.NumberColumn":
    return st.column_config.NumberColumn(label, format=f"%.{places}f")


#: Display label for every master-frame column the pages put in a table.
COLUMN_LABELS: dict[str, str] = {
    "customer_id": "Customer",
    "churn_probability": "Churn probability",
    "risk_level": "Risk",
    "customer_value_segment": "Customer value",
    "primary_segment": "Segment",
    "all_segments": "All segments",
    "revenue_at_risk": "Revenue at risk",
    "model_revenue_at_risk": "Revenue at risk (model estimate)",
    "expected_future_revenue": "Expected future revenue",
    "expected_retained_revenue": "Expected revenue retained",
    "retention_opportunity_score": "Retention opportunity",
    "retention_propensity": "Retention propensity (ASSUMED)",
    "priority": "Priority",
    "top_driver": "Main churn driver",
    "top_driver_explanation": "Why",
    "recommended_action": "Recommended action",
    "recommended_channel": "Channel",
    "recommended_category": "Category",
    "recommended_sku": "SKU",
    "recommended_product": "Product",
    "recommended_offer": "Offer",
    "reason": "Reason",
    "expected_roi": "Expected ROI",
    "campaign_cost": "Campaign cost",
    "suppressed_action": "Suppressed action",
    "lifetime_revenue": "Lifetime revenue",
    "recent_revenue": "Recent revenue",
    "average_order_value": "AOV",
    "recency_days": "Recency (days)",
    "total_orders": "Orders",
    "total_units": "Units",
    "return_rate": "Return rate",
    "preferred_category": "Preferred category",
    "preferred_brand": "Preferred brand",
    "country": "Country",
    "city": "City",
    "acquisition_channel": "Acquisition channel",
    "customer_tenure_days": "Tenure (days)",
    "age": "Age",
}


#: Verbose free-text columns. Left at their natural width they push the decision columns --
#: expected ROI, priority -- off the right edge of a wide table, where nobody scrolls to find
#: them. Constraining them keeps the numbers that drive an action on screen.
_NARROW_TEXT = {
    "recommended_channel",
    "recommended_offer",
    "recommended_product",
    "reason",
    "top_driver",
    "top_driver_explanation",
    "all_segments",
    "suppressed_action",
    "recommended_action",
}


def _config_for(column: str, label: str) -> object:
    """Pick the display format for a column from its meaning, not its dtype."""
    money = {
        "revenue_at_risk",
        "model_revenue_at_risk",
        "expected_future_revenue",
        "expected_retained_revenue",
        "retention_opportunity_score",
        "lifetime_revenue",
        "recent_revenue",
        "average_order_value",
        "campaign_cost",
        "projected_annual_revenue",
        "historical_annual_revenue",
        "expected_average_order_value",
    }
    rates = {
        "churn_probability": f"Likelihood of no purchase in the {horizon_phrase()}.",
        "return_rate": None,
        "retention_propensity": "ASSUMED, not measured: this dataset has no campaign log.",
        "expected_roi": "(expected revenue retained − campaign cost) / campaign cost.",
    }
    counts = {"total_orders", "total_units", "recency_days", "customer_tenure_days", "age"}

    if column in money:
        return _money(label)
    if column in rates:
        return _rate(label, rates[column])
    if column in counts:
        return _count(label)
    if column in _NARROW_TEXT:
        return st.column_config.Column(label, width="small")
    return None


def data_table(
    frame: pd.DataFrame,
    columns: list[str],
    *,
    download_name: str,
    key: str | None = None,
    height: int | None = None,
    caption: str | None = None,
) -> None:
    """Render ``columns`` of ``frame`` as a sortable grid with a CSV export beneath it."""
    present = [c for c in columns if c in frame.columns]
    view = frame[present]

    config: dict[str, object] = {}
    for column in present:
        label = COLUMN_LABELS.get(column, column.replace("_", " ").capitalize())
        spec = _config_for(column, label)
        config[column] = spec if spec is not None else st.column_config.Column(label)

    st.dataframe(
        view,
        column_config=config,
        hide_index=True,
        width="stretch",
        height=height,
        key=key,
    )
    if caption:
        st.caption(caption)
    download_button(view, download_name, key=f"{key or download_name}_dl")


def download_button(
    frame: pd.DataFrame, filename: str, *, key: str | None = None, label: str | None = None
) -> None:
    """Export exactly the rows and columns on screen, at full precision.

    The button is labelled for the reader, not for the file system: a business user downloading a
    customer list does not need to be told its filename before they click. ``filename`` still
    names the file they receive.

    utf-8-sig for the same reason the pipeline writes it: the city names include Düsseldorf and
    Liège, and Excel on Windows mangles them without a BOM.
    """
    st.download_button(
        label or "Download for Excel",
        data=frame.to_csv(index=False).encode("utf-8-sig"),
        file_name=filename,
        mime="text/csv",
        key=key,
        width="content",
    )


def currency_note() -> str:
    return f"Amounts in {currency_symbol()}."
