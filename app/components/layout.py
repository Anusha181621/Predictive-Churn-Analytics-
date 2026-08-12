"""Page scaffolding: headers, section rules and the recurring assumption notice."""

from __future__ import annotations

import streamlit as st

__all__ = ["page_header", "section", "assumption_notice", "chart_card"]


def page_header(title: str, subtitle: str, *, as_of: str | None = None) -> None:
    """Title, one line of orientation, and the as-of date every number is computed at."""
    st.title(title)
    # <strong>, not markdown asterisks: this string is emitted as raw HTML, where "**" would be
    # rendered literally rather than as emphasis.
    stamp = f" &nbsp;·&nbsp; as of <strong>{as_of}</strong>" if as_of else ""
    st.markdown(
        f'<p class="section-note">{subtitle}{stamp}</p>',
        unsafe_allow_html=True,
    )


def section(title: str, note: str = "") -> None:
    """A labelled break between groups of charts."""
    st.markdown("")
    st.subheader(title, anchor=False)
    if note:
        st.markdown(f'<p class="section-note">{note}</p>', unsafe_allow_html=True)


def assumption_notice(text: str) -> None:
    """Flag a number that depends on the retention-propensity assumption.

    The pipeline names its assumption-dependent columns explicitly and ships
    ``outputs/retention_assumptions.json`` beside the CSVs. The dashboard keeps that visible
    rather than letting a stated assumption quietly become a reported fact once it is on a chart.
    """
    st.markdown(f'<div class="assumption">{text}</div>', unsafe_allow_html=True)


def chart_card(title: str, note: str = "") -> None:
    """A caption above a chart: what it shows, and any caveat about how to read it."""
    st.markdown(f"**{title}**")
    if note:
        st.markdown(f'<p class="section-note">{note}</p>', unsafe_allow_html=True)
