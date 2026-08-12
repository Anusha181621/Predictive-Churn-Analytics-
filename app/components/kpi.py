"""KPI cards and the hero figure.

A single headline number is a *figure*, not a chart -- a one-bar bar chart of "total customers"
carries no more information than the number itself and costs a reader far more effort. These are
the forms for that case.
"""

from __future__ import annotations

from dataclasses import dataclass

import streamlit as st

from app.theme import RISK_COLOURS, STATUS

__all__ = ["Kpi", "kpi_row", "hero", "pill"]


@dataclass(frozen=True)
class Kpi:
    """One stat tile.

    ``note`` carries the supporting detail -- a share, a comparison, a definition -- so the
    value itself stays a clean number. ``tone`` colours the note only, never the value: a
    coloured headline number reads as a status signal even when it is just a count.
    """

    label: str
    value: str
    note: str = ""
    tone: str = "neutral"  # neutral | good | bad


def _note_class(tone: str) -> str:
    return {"good": "kpi-note kpi-good", "bad": "kpi-note kpi-bad"}.get(tone, "kpi-note")


def kpi_row(items: list[Kpi], columns: int | None = None) -> None:
    """Render stat tiles in an evenly-spaced responsive row."""
    if not items:
        return
    width = columns or len(items)
    for start in range(0, len(items), width):
        chunk = items[start : start + width]
        cols = st.columns(len(chunk), gap="small")
        for col, item in zip(cols, chunk):
            note = (
                f'<div class="{_note_class(item.tone)}">{item.note}</div>' if item.note else ""
            )
            col.markdown(
                f'<div class="kpi-card">'
                f'<div class="kpi-label">{item.label}</div>'
                f'<div class="kpi-value">{item.value}</div>'
                f"{note}</div>",
                unsafe_allow_html=True,
            )


def hero(label: str, value: str, note: str = "") -> None:
    """The single number a page leads with. Exactly one per view."""
    detail = f'<div class="kpi-note">{note}</div>' if note else ""
    st.markdown(
        f'<div class="hero-label">{label}</div>'
        f'<div class="hero-value">{value}</div>{detail}',
        unsafe_allow_html=True,
    )


def pill(value: str, kind: str = "risk", label: str | None = None) -> str:
    """An inline coloured badge for a risk level or priority band.

    Returns markup rather than rendering, so it can be embedded in a larger block. The band name
    is always inside the pill -- the colour reinforces the label, it never replaces it. The colour
    is chosen from ``value`` alone, so prefixing a label cannot change it.
    """
    palette = RISK_COLOURS if kind == "risk" else {
        "Critical": STATUS["critical"],
        "High": STATUS["serious"],
        "Medium": STATUS["warning"],
        "Low": STATUS["good"],
    }
    colour = palette.get(str(value), STATUS["warning"])
    text = f"{label}: {value}" if label else str(value)
    return f'<span class="pill" style="background:{colour}">{text}</span>'
