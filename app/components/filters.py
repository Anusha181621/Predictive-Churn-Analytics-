"""Sidebar filters, shared by every page that shows a customer population.

Two properties matter more than the widgets themselves.

**One filter set scopes a whole page.** Filters live in the sidebar, never inside a chart card,
so every chart on the page re-renders against the same slice. Two charts showing different
populations under the same heading is the fastest way to make a dashboard untrustworthy.

**Domains come from the unfiltered data.** ``options`` for every widget are computed from the
full customer master, so narrowing one filter never removes options from another and -- more
importantly -- never shifts a colour assignment. Colour follows the entity, not its current rank.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import streamlit as st

from app.theme import RISK_ORDER

__all__ = ["FilterSpec", "CUSTOMER_FILTERS", "render_filters", "filter_caption"]


@dataclass(frozen=True)
class FilterSpec:
    """One multi-select filter over a column of the customer master."""

    key: str
    label: str
    column: str
    order: tuple[str, ...] | None = None  # fixed display order, where one exists


#: The seven filters the brief asks for, in the order they appear in the sidebar.
CUSTOMER_FILTERS: tuple[FilterSpec, ...] = (
    FilterSpec("country", "Country", "country"),
    FilterSpec("city", "City", "city"),
    FilterSpec("channel", "Acquisition channel", "acquisition_channel"),
    FilterSpec("segment", "Customer segment", "primary_segment"),
    FilterSpec("risk", "Risk level", "risk_level", RISK_ORDER),
    FilterSpec(
        "value", "Customer value", "customer_value_segment",
        ("High Value", "Medium Value", "Low Value"),
    ),
    FilterSpec("category", "Preferred category", "preferred_category"),
)


def _options(master: pd.DataFrame, spec: FilterSpec) -> list[str]:
    values = master[spec.column].dropna().astype(str).unique().tolist()
    if spec.order:
        ranked = [v for v in spec.order if v in values]
        return ranked + sorted(v for v in values if v not in spec.order)
    return sorted(values)


def render_filters(
    master: pd.DataFrame,
    specs: tuple[FilterSpec, ...] = CUSTOMER_FILTERS,
    *,
    namespace: str = "flt",
) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    """Draw the filters and return ``(filtered_frame, active_selections)``.

    An empty selection means "no constraint" rather than "nothing", which is what a reader
    expects from a blank filter and avoids the dead-end of an accidentally empty page.
    """
    st.sidebar.markdown("### Filters")
    selections: dict[str, list[str]] = {}
    filtered = master

    # Widget keys carry a "flt_" prefix so they can never collide with a chart or table key on
    # the same page. Streamlit's duplicate-key check spans every element in a run, not just
    # widgets, so a filter named after a dimension would otherwise clash with the chart that
    # breaks that same dimension down.
    def widget_key(spec: FilterSpec) -> str:
        return f"flt_{namespace}_{spec.key}"

    for spec in specs:
        if spec.column not in master.columns:
            continue
        chosen = st.sidebar.multiselect(
            spec.label,
            options=_options(master, spec),
            default=[],
            key=widget_key(spec),
            placeholder="All",
        )
        if chosen:
            selections[spec.label] = chosen
            filtered = filtered[filtered[spec.column].astype(str).isin(chosen)]

    if selections and st.sidebar.button(
        "Clear filters", width="stretch", key=f"flt_{namespace}_clear"
    ):
        for spec in specs:
            st.session_state.pop(widget_key(spec), None)
        st.rerun()

    st.sidebar.caption(f"{len(filtered):,} of {len(master):,} customers match")
    return filtered, selections


def filter_caption(selections: dict[str, list[str]]) -> str:
    """A one-line description of the active slice, for a chart subtitle or an export note."""
    if not selections:
        return "All customers"
    return " · ".join(f"{label}: {', '.join(values)}" for label, values in selections.items())
