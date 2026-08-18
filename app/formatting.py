"""Number and label formatting shared by every page.

Kept in one module so a euro sign, a thousands separator or a percentage never appears in two
different shapes on two different pages.
"""

from __future__ import annotations

import math

from src.config.settings import get_settings

__all__ = [
    "currency_symbol",
    "money",
    "money_compact",
    "integer",
    "percent",
    "signed_percent",
    "ratio",
    "days",
    "horizon_days",
    "horizon_phrase",
]

_SYMBOLS = {"EUR": "€", "USD": "$", "GBP": "£"}


def currency_symbol() -> str:
    """The configured currency's symbol, falling back to the ISO code itself."""
    code = get_settings().currency
    return _SYMBOLS.get(code.upper(), f"{code} ")


def _blank(value: object) -> bool:
    return value is None or (isinstance(value, float) and math.isnan(value))


def money(value: float | None, decimals: int = 0) -> str:
    """``EUR 125,129``. Full precision -- for totals a reader may want to reconcile."""
    if _blank(value):
        return "—"
    return f"{currency_symbol()}{value:,.{decimals}f}"


def money_compact(value: float | None) -> str:
    """``EUR 125.1k``. For KPI cards, where the exact cent is noise."""
    if _blank(value):
        return "—"
    magnitude = abs(float(value))
    if magnitude >= 1_000_000:
        return f"{currency_symbol()}{value / 1_000_000:,.1f}M"
    if magnitude >= 1_000:
        return f"{currency_symbol()}{value / 1_000:,.1f}k"
    return f"{currency_symbol()}{value:,.0f}"


def integer(value: float | None) -> str:
    if _blank(value):
        return "—"
    return f"{value:,.0f}"


def percent(value: float | None, decimals: int = 1) -> str:
    """A rate held as a fraction (0.4555) rendered as ``45.6%``."""
    if _blank(value):
        return "—"
    return f"{value * 100:,.{decimals}f}%"


def signed_percent(value: float | None, decimals: int = 0) -> str:
    """A ratio rendered with an explicit sign, as ROI is conventionally quoted: ``+437%``."""
    if _blank(value):
        return "—"
    return f"{value * 100:+,.{decimals}f}%"


def ratio(value: float | None, decimals: int = 2) -> str:
    if _blank(value):
        return "—"
    return f"{value:,.{decimals}f}"


def days(value: float | None) -> str:
    if _blank(value):
        return "—"
    count = int(round(float(value)))
    return f"{count:,} day" + ("" if count == 1 else "s")


def horizon_days() -> int:
    """The churn horizon every probability on screen describes, in days.

    Read rather than restated. The horizon moved from 180 days to 90 when the model was retrained,
    and the pages that had spelled "180 days" into their copy went on saying so -- a wrong number
    presented with the same confidence as a right one. Nothing on a page should name this figure
    from memory.
    """
    return int(get_settings().churn_inactivity_days)


def horizon_phrase() -> str:
    """``next 90 days``, for embedding mid-sentence."""
    return f"next {horizon_days()} days"
