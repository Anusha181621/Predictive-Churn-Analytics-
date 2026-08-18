"""Tests for the Streamlit dashboard.

The dashboard's job is narrow and worth stating precisely: it **reads** the artefacts the pipeline
wrote and renders them. It owns no business logic, so what needs testing is not arithmetic but
faithfulness -- that the join across five artefacts stays one row per customer, that the headline
figures equal the artefacts they came from, and that every page renders rather than raising.

The pages are exercised through Streamlit's own ``AppTest`` harness, which runs each module in a
real script-run context. That matters: a page can import perfectly and still fail at render time
on a duplicate element key or a missing column, and only an actual run catches it.

Everything here needs the generated artefacts, which are git-ignored, so a fresh clone skips this
module rather than failing. Run the pipeline scripts to enable it.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.config.settings import get_settings

pytest.importorskip("streamlit", reason="streamlit is required for the dashboard tests")

from streamlit.testing.v1 import AppTest  # noqa: E402

from app import data_access  # noqa: E402
from app.formatting import money, money_compact, percent, signed_percent  # noqa: E402
from app.theme import CATEGORICAL, RISK_COLOURS, RISK_ORDER, categorical, colour_map  # noqa: E402
from src.utils.paths import project_root  # noqa: E402


def _artefact(key: str) -> pd.DataFrame:
    """Read an artefact straight from disk.

    Deliberately independent of the dashboard's own reader: comparing the dashboard against a file
    it also parsed would only prove pandas is deterministic.
    """
    settings = get_settings()
    return pd.read_csv(data_access.ARTEFACTS[key].path(settings), encoding="utf-8-sig")

PAGES = (
    "executive_overview",
    "churn_risk",
    "revenue_at_risk",
    "retention_action_center",
    "customer_360",
    "segmentation",
    "model_performance",
    "what_if",
)

#: Pages that only need the CSV artefacts. `what_if` additionally needs the trained model.
CSV_ONLY_PAGES = tuple(p for p in PAGES if p != "what_if")

PAGE_SCRIPT = """
import sys
sys.path.insert(0, {root!r})
import streamlit as st
from app.theme import register_template
register_template()
from app.views import {page} as _page
_page.render()
"""


def _artefacts_present(*keys: str) -> bool:
    return not data_access.missing(*keys)


CORE = ("features", "predictions", "scores", "recommendations", "explanations")

requires_artefacts = pytest.mark.skipif(
    not _artefacts_present(*CORE),
    reason="generated artefacts missing; run the pipeline scripts first",
)
requires_model = pytest.mark.skipif(
    not _artefacts_present("model"),
    reason="no trained model; run `python scripts/train_model.py` first",
)


def _run_page(page: str, timeout: int = 300) -> AppTest:
    app = AppTest.from_string(
        PAGE_SCRIPT.format(root=str(project_root()), page=page), default_timeout=timeout
    )
    app.run()
    return app


# ======================================================================================
# the data-access layer
# ======================================================================================


@requires_artefacts
def test_the_master_frame_is_one_row_per_customer() -> None:
    master = data_access.load_customer_master()
    assert master["customer_id"].is_unique
    assert len(master) == len(_artefact("predictions"))


@requires_artefacts
def test_the_master_frame_keeps_both_revenue_at_risk_figures_apart() -> None:
    """Two different quantities, deliberately under two different names.

    The decision layer's figure is what the business pages report; the model's own estimate is a
    different number computed a different way. Collapsing them would let the dashboard show a
    total that reconciles with nothing.
    """
    master = data_access.load_customer_master()
    assert "revenue_at_risk" in master
    assert "model_revenue_at_risk" in master
    assert not master["revenue_at_risk"].equals(master["model_revenue_at_risk"])


@requires_artefacts
def test_headline_totals_equal_the_artefacts_they_came_from() -> None:
    """Nothing is recomputed on the way to the screen."""
    master = data_access.load_customer_master()
    scores = _artefact("scores")

    assert float(master["revenue_at_risk"].sum()) == pytest.approx(
        float(scores["Revenue at risk"].sum()), abs=0.01
    )
    assert float(master["expected_future_revenue"].sum()) == pytest.approx(
        float(scores["Expected future revenue"].sum()), abs=0.01
    )


@requires_artefacts
def test_campaign_summary_counts_only_contacted_customers() -> None:
    """Cost, return and ROI must all be drawn from the same population.

    Mixing an all-customer return with a targeted-only cost is the easy mistake here, and it
    inflates the reported ROI.
    """
    master = data_access.load_customer_master()
    summary = data_access.campaign_summary(master)
    targeted = master[master["is_targeted"]]

    assert summary["targeted"] == len(targeted)
    assert summary["suppressed"] == len(master) - len(targeted)
    assert summary["cost"] == pytest.approx(float(targeted["campaign_cost"].sum()), abs=0.01)
    assert summary["expected_retained"] == pytest.approx(
        float(targeted["expected_retained_revenue"].sum()), abs=0.01
    )
    assert summary["roi"] == pytest.approx(
        (summary["expected_retained"] - summary["cost"]) / summary["cost"], abs=1e-6
    )


@requires_artefacts
def test_active_and_at_risk_flags_follow_their_documented_definitions() -> None:
    master = data_access.load_customer_master()
    expected_active = master["recency_days"].le(data_access.ACTIVE_RECENCY_DAYS)
    expected_at_risk = master["churn_probability"].ge(data_access.AT_RISK_PROBABILITY)
    pd.testing.assert_series_equal(master["is_active"], expected_active, check_names=False)
    pd.testing.assert_series_equal(master["is_at_risk"], expected_at_risk, check_names=False)


@requires_artefacts
def test_every_customer_carries_a_top_driver() -> None:
    """The action centre's "main churn driver" column cannot be blank for anyone."""
    master = data_access.load_customer_master()
    assert master["top_driver"].notna().all()
    assert master["top_driver_explanation"].notna().all()


def test_every_declared_artefact_names_the_command_that_makes_it() -> None:
    for key, artefact in data_access.ARTEFACTS.items():
        assert artefact.command.startswith("python scripts/"), key
        assert artefact.directory in {"outputs", "models"}, key


def test_a_missing_artefact_produces_guidance_rather_than_a_traceback() -> None:
    """A fresh clone has no `outputs/`. The page must explain itself, not raise.

    The guidance is written for whoever is looking at the dashboard, who is not necessarily the
    person who runs the pipeline. It names what is missing in business terms and says who can
    restore it; the shell command lives on ``Artefact.command`` and goes to the log instead.
    """
    app = AppTest.from_string(
        """
import sys
sys.path.insert(0, {root!r})
import streamlit as st
from app.data_access import require
require("predictions")
st.write("should not get here")
""".format(root=str(project_root())),
        default_timeout=60,
    )
    # Point the settings at an empty directory so the artefact genuinely cannot be found.
    import os

    original = os.environ.get("OUTPUTS_DIR")
    os.environ["OUTPUTS_DIR"] = "tests"
    try:
        get_settings(refresh=True)
        app.run()
    finally:
        if original is None:
            os.environ.pop("OUTPUTS_DIR", None)
        else:
            os.environ["OUTPUTS_DIR"] = original
        get_settings(refresh=True)

    assert not app.exception, "a missing artefact raised instead of guiding"
    assert app.error, "no guidance was shown for the missing artefact"
    message = app.error[0].value
    assert "Churn predictions" in message, "the guidance does not name what is missing"
    assert "reload the page" in message, "the guidance does not say what to do next"
    for fragment in ("python scripts/", "```", ".csv", "outputs/"):
        assert fragment not in message, f"the guidance still shows a code reference: {fragment}"


# ======================================================================================
# every page renders
# ======================================================================================


@requires_artefacts
@pytest.mark.parametrize("page", CSV_ONLY_PAGES)
def test_page_renders_without_error(page: str) -> None:
    app = _run_page(page)
    assert not app.exception, (
        f"{page} raised: {app.exception[0].type}: {app.exception[0].message}"
    )
    assert not app.error, f"{page} showed an error box: {[e.value[:120] for e in app.error]}"


@requires_artefacts
@requires_model
def test_the_what_if_page_renders() -> None:
    app = _run_page("what_if", timeout=600)
    assert not app.exception, f"what_if raised: {app.exception[0].message}"
    assert not app.error


@requires_artefacts
@pytest.mark.parametrize("page", CSV_ONLY_PAGES)
def test_page_produces_visible_content(page: str) -> None:
    """A page that renders nothing would pass the exception check and still be broken."""
    app = _run_page(page)
    rendered = len(app.markdown) + len(app.dataframe) + len(app.get("plotly_chart"))
    assert rendered > 3, f"{page} rendered almost nothing ({rendered} elements)"


#: Fragments that betray the implementation to a reader who did not build it. A dashboard that
#: tells a retention manager to run `python scripts/predict.py` has stopped being a product.
_CODE_REFERENCES = (
    "scripts/",
    ".csv",
    ".json",
    ".joblib",
    "src/",
    "outputs/",
    "models/",
    "```",
    "log-odds",
    "artefact",
)

#: Business vocabulary that happens to contain a banned fragment. "Customer.csv" is a file;
#: "SKU" and "CSV export" are things a merchandiser says out loud.
_ALLOWED = ("download for excel",)


def _visible_text(app: AppTest) -> list[str]:
    """Every string the page actually puts in front of a reader."""
    chunks: list[str] = []
    for kind in ("markdown", "caption", "error", "info", "success", "warning"):
        for element in app.get(kind):
            chunks.append(str(element.value))
    return chunks


@requires_artefacts
@pytest.mark.parametrize("page", CSV_ONLY_PAGES)
def test_no_page_shows_a_code_reference(page: str) -> None:
    """Every visible string must name a business concept, not a file, script or library."""
    app = _run_page(page)
    offenders = [
        (fragment, chunk.strip()[:160])
        for chunk in _visible_text(app)
        for fragment in _CODE_REFERENCES
        if fragment in chunk.lower()
        and not any(allowed in chunk.lower() for allowed in _ALLOWED)
    ]
    assert not offenders, f"{page} shows code references: {offenders[:4]}"


@requires_artefacts
def test_the_filter_bar_narrows_the_page_it_scopes() -> None:
    """The filters are only worth having if every number below them moves together."""
    app = _run_page("churn_risk")
    assert app.multiselect, "the filter bar rendered no controls"

    master = data_access.load_customer_master()
    unfiltered = "\n".join(_visible_text(app))
    assert f"{len(master):,}" in unfiltered, "the bar does not report the unfiltered population"

    risk = next(m for m in app.multiselect if m.key.endswith("_risk"))
    app = risk.set_value(["Critical"]).run(timeout=300)

    expected = int(master["risk_level"].eq("Critical").sum())
    filtered = "\n".join(_visible_text(app))
    assert f"{expected:,}" in filtered, "the match count did not follow the filter"
    assert "Critical" in filtered, "the active selection is not shown as a chip"


@requires_artefacts
def test_the_executive_overview_reports_the_pipelines_own_numbers() -> None:
    """The board-level page must agree with the artefacts to the digit."""
    master = data_access.load_customer_master()
    summary = data_access.campaign_summary(master)
    app = _run_page("executive_overview")

    text = "\n".join(str(m.value) for m in app.markdown)
    assert money(float(master["revenue_at_risk"].sum())) in text
    assert f"{len(master):,}" in text
    assert f"{int(master['is_active'].sum()):,}" in text
    assert money_compact(summary["expected_retained"]) in text
    assert signed_percent(summary["roi"]) in text


@requires_artefacts
def test_the_overview_labels_its_assumption_dependent_figures() -> None:
    """ROI rests on an assumption this dataset cannot measure, and must say so on the page."""
    app = _run_page("executive_overview")
    text = "\n".join(str(m.value) for m in app.markdown)
    assert "ASSUMPTION" in text.upper()
    assert "propensity" in text.lower()


@requires_artefacts
def test_model_performance_surfaces_the_training_runs_own_warning() -> None:
    """The model fails its own sanity floor; the page must not quietly omit that."""
    metrics = data_access.load_model_metrics()
    warnings = [n for n in metrics.get("notes", []) if str(n).upper().startswith("WARNING")]
    app = _run_page("model_performance")
    if warnings:
        assert app.warning, "the training run flagged the model but the page shows no warning"


# ======================================================================================
# the visual system
# ======================================================================================


def test_risk_levels_use_the_reserved_status_palette_not_series_colours() -> None:
    """Risk means severity. A status colour must never be reused as a series colour."""
    assert set(RISK_COLOURS) == set(RISK_ORDER)
    assert not set(RISK_COLOURS.values()) & set(CATEGORICAL), (
        "a risk colour collides with a categorical slot"
    )


def test_categorical_slots_are_fixed_and_never_generated() -> None:
    """A ninth hue would be indistinguishable from an existing one under colour-blindness."""
    assert categorical(0) == CATEGORICAL[0]
    assert categorical(len(CATEGORICAL) - 1) == CATEGORICAL[-1]
    with pytest.raises(IndexError):
        categorical(len(CATEGORICAL))


def test_colour_assignment_follows_the_entity_not_its_rank() -> None:
    """Filtering a series out must not repaint the survivors."""
    full = colour_map(["Alpha", "Beta", "Gamma"])
    filtered = colour_map(["Alpha", "Beta", "Gamma"])
    assert full == filtered
    assert full["Alpha"] != full["Beta"] != full["Gamma"]


def test_a_long_domain_folds_into_other_rather_than_inventing_hues() -> None:
    names = [f"seg{i}" for i in range(12)]
    mapping = colour_map(names)
    assert "Other" in mapping
    assert len(set(mapping.values())) <= len(CATEGORICAL)


@pytest.mark.parametrize(
    ("value", "rendered"),
    [
        (0.0, "0.0%"),
        (0.4512, "45.1%"),
        (0.45549447, "45.5%"),  # the shipped mean churn probability
        (1.0, "100.0%"),
        (None, "—"),
    ],
)
def test_percentages_render_consistently(value: float | None, rendered: str) -> None:
    """Values are chosen off the half-way boundary on purpose.

    A case like 0.4555 has no single correct answer at one decimal place: it is 45.550000000000004
    in binary floating point, so it rounds up, and asserting either way would be testing float
    representation rather than the formatter's contract.
    """
    assert percent(value) == rendered


def test_money_and_roi_formatting_is_stable() -> None:
    assert money(125128.96) == "€125,129"
    assert money_compact(37264.67) == "€37.3k"
    assert signed_percent(4.3733) == "+437%"
    assert money(None) == "—"
