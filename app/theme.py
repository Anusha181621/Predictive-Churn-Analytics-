"""Visual system for the dashboard: palette, Plotly template and card styling.

Every colour used anywhere in the app comes from this module, so the eight pages read as
one system rather than eight independently-styled screens.

Two rules from the palette design are load-bearing and easy to undo by accident:

*   **Risk levels wear the status palette, everything else wears the categorical one.**
    Low / Medium / High / Critical mean good / warning / serious / critical, so they use the
    reserved status colours. A behavioural segment or an acquisition channel means *identity*,
    not severity, so it takes a categorical slot. Mixing the two makes a status colour
    impersonate a series, and the reader stops being able to trust that red means bad.
*   **Categorical hues are assigned in a fixed order and never cycled.** ``categorical()``
    hands out slots 1..8 positionally; past eight it raises rather than inventing a ninth hue,
    because a generated hue is indistinguishable from an existing one under colour-blindness.

The palette was checked with the validator rather than by eye: on the light surface used here
all eight slots sit inside the lightness band, clear the chroma floor, and the worst adjacent
pair separates by 9.1 delta-E under protanopia (target 8) and 19.6 for normal vision (floor 15).
Three slots fall below 3:1 contrast against the surface, which obliges every chart to carry
visible labels or a table view -- both of which the pages ship.
"""

from __future__ import annotations

import plotly.graph_objects as go
import plotly.io as pio

__all__ = [
    "SURFACE",
    "PAGE",
    "INK",
    "INK_SECONDARY",
    "INK_MUTED",
    "GRID",
    "AXIS",
    "CATEGORICAL",
    "SEQUENTIAL",
    "DIVERGING_LOW",
    "DIVERGING_MID",
    "DIVERGING_HIGH",
    "STATUS",
    "RISK_COLOURS",
    "RISK_ORDER",
    "PRIORITY_COLOURS",
    "PRIORITY_ORDER",
    "TEMPLATE",
    "CSS",
    "categorical",
    "colour_map",
    "register_template",
]

# --------------------------------------------------------------------------------------
# surfaces and ink
# --------------------------------------------------------------------------------------

SURFACE = "#fcfcfb"  # the chart surface; the palette was validated against this exact value
PAGE = "#f9f9f7"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
BORDER = "rgba(11,11,11,0.10)"

# --------------------------------------------------------------------------------------
# the four colour jobs
# --------------------------------------------------------------------------------------

#: Identity. Fixed order -- slot N always means "the Nth entity", never "the Nth largest".
CATEGORICAL = (
    "#2a78d6",  # blue
    "#eb6834",  # orange
    "#1baf7a",  # aqua
    "#eda100",  # yellow
    "#e87ba4",  # magenta
    "#008300",  # green
    "#4a3aa7",  # violet
    "#e34948",  # red
)

#: Magnitude. One hue, light to dark -- used for heatmaps and other continuous scales.
SEQUENTIAL = (
    "#cde2fb",
    "#9ec5f4",
    "#6da7ec",
    "#3987e5",
    "#2a78d6",
    "#256abf",
    "#1c5cab",
    "#184f95",
    "#104281",
)

#: Polarity. Two hues that read as opposite, with a neutral -- never a hue -- in the middle.
DIVERGING_LOW = "#2a78d6"
DIVERGING_MID = "#c9c8c2"
DIVERGING_HIGH = "#d03b3b"

#: State. Reserved: never reused as a series colour.
STATUS = {
    "good": "#0ca30c",
    "warning": "#fab219",
    "serious": "#ec835a",
    "critical": "#d03b3b",
}

RISK_ORDER = ("Low", "Medium", "High", "Critical")

#: Risk is a severity scale, so it takes the status palette rather than categorical slots.
RISK_COLOURS = {
    "Low": STATUS["good"],
    "Medium": STATUS["warning"],
    "High": STATUS["serious"],
    "Critical": STATUS["critical"],
}

PRIORITY_ORDER = ("Critical", "High", "Medium", "Low")

PRIORITY_COLOURS = {
    "Critical": STATUS["critical"],
    "High": STATUS["serious"],
    "Medium": STATUS["warning"],
    "Low": STATUS["good"],
}

FONT_FAMILY = 'system-ui, -apple-system, "Segoe UI", sans-serif'


def categorical(index: int) -> str:
    """Return categorical slot ``index`` (0-based).

    Raises rather than wrapping around: a ninth categorical hue cannot be told apart from an
    existing one under colour-blindness, so the caller must fold its tail into "Other" or
    facet into small multiples instead.
    """
    if not 0 <= index < len(CATEGORICAL):
        raise IndexError(
            f"categorical slot {index} is out of range (0..{len(CATEGORICAL) - 1}). "
            "Fold the tail into 'Other' or facet rather than generating a new hue."
        )
    return CATEGORICAL[index]


def colour_map(values: list[str]) -> dict[str, str]:
    """Map entity names to fixed categorical slots, in the order given.

    Callers pass a *stable* ordering (the full domain, not the filtered one), so that removing
    a series from a filter never repaints the survivors.
    """
    if len(values) > len(CATEGORICAL):
        head = list(values)[: len(CATEGORICAL) - 1]
        mapping = {name: CATEGORICAL[i] for i, name in enumerate(head)}
        mapping["Other"] = INK_MUTED
        return mapping
    return {name: CATEGORICAL[i] for i, name in enumerate(values)}


# --------------------------------------------------------------------------------------
# the Plotly template
# --------------------------------------------------------------------------------------

_AXIS_STYLE = dict(
    showgrid=True,
    gridcolor=GRID,
    gridwidth=1,
    griddash="solid",  # never dashed: dashing reads as "projection" when it is just a grid
    zeroline=False,
    linecolor=AXIS,
    linewidth=1,
    ticks="outside",
    ticklen=4,
    tickcolor=AXIS,
    tickfont=dict(color=INK_MUTED, size=12),
    title=dict(font=dict(color=INK_SECONDARY, size=12)),
    automargin=True,
)

TEMPLATE = go.layout.Template(
    layout=dict(
        colorway=list(CATEGORICAL),
        font=dict(family=FONT_FAMILY, color=INK_SECONDARY, size=13),
        paper_bgcolor=SURFACE,
        plot_bgcolor=SURFACE,
        title=dict(font=dict(color=INK, size=15), x=0, xanchor="left", pad=dict(b=12)),
        xaxis=dict(**_AXIS_STYLE),
        yaxis=dict(**_AXIS_STYLE),
        margin=dict(l=8, r=16, t=48, b=8),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
            font=dict(color=INK_SECONDARY, size=12),
            bgcolor="rgba(0,0,0,0)",
            title=dict(text=""),
        ),
        hoverlabel=dict(
            bgcolor=SURFACE,
            bordercolor=AXIS,
            font=dict(family=FONT_FAMILY, color=INK, size=12),
        ),
        colorscale=dict(sequential=[[i / (len(SEQUENTIAL) - 1), c] for i, c in enumerate(SEQUENTIAL)]),
        bargap=0.42,  # leaves air in the band rather than letting bars fill it
    )
)


def register_template() -> None:
    """Install the template as Plotly's default for this process."""
    pio.templates["churn"] = TEMPLATE
    pio.templates.default = "churn"


# --------------------------------------------------------------------------------------
# page styling
# --------------------------------------------------------------------------------------

CSS = f"""
<style>
  .block-container {{ padding-top: 2.4rem; padding-bottom: 3rem; max-width: 1500px; }}

  /* KPI cards ------------------------------------------------------------------ */
  .kpi-card {{
    background: {SURFACE};
    border: 1px solid {BORDER};
    border-radius: 10px;
    padding: 14px 16px 15px 16px;
    height: 100%;
    display: flex; flex-direction: column; gap: 4px;
  }}
  .kpi-label {{
    font-size: 0.78rem; color: {INK_MUTED}; font-weight: 500;
    letter-spacing: .01em; line-height: 1.25;
  }}
  .kpi-value {{
    font-size: 1.65rem; font-weight: 600; color: {INK}; line-height: 1.15;
  }}
  .kpi-note {{ font-size: 0.75rem; color: {INK_SECONDARY}; line-height: 1.3; }}
  .kpi-good {{ color: #006300; font-weight: 600; }}
  .kpi-bad  {{ color: {STATUS["critical"]}; font-weight: 600; }}

  /* hero figure ---------------------------------------------------------------- */
  .hero-value {{
    font-size: 3rem; font-weight: 600; color: {INK}; line-height: 1.05; margin: 0;
  }}
  .hero-label {{ font-size: 0.85rem; color: {INK_MUTED}; margin-bottom: 2px; }}

  /* risk / priority pills ------------------------------------------------------ */
  .pill {{
    display: inline-block; padding: 2px 10px; border-radius: 999px;
    font-size: 0.75rem; font-weight: 600; color: #fff;
  }}

  /* assumption banner ---------------------------------------------------------- */
  .assumption {{
    border-left: 3px solid {STATUS["warning"]};
    background: rgba(250,178,25,0.08);
    padding: 9px 13px; border-radius: 0 6px 6px 0;
    font-size: 0.8rem; color: {INK_SECONDARY}; margin: 4px 0 14px 0;
  }}

  .section-note {{ font-size: 0.8rem; color: {INK_MUTED}; margin: -6px 0 12px 0; }}

  div[data-testid="stMetricValue"] {{ font-size: 1.5rem; }}
  section[data-testid="stSidebar"] {{ border-right: 1px solid {BORDER}; }}
</style>
"""
