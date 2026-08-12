"""What-If Simulator -- stress-test the assumptions the retention plan rests on.

**This page re-runs the real retention layer.** It builds a modified
:class:`~src.retention.params.RetentionParams` and calls
:func:`~src.retention.pipeline.build_retention_layer`, exactly as ``scripts/retention.py`` does,
rather than rescaling the shipped numbers in the browser.

That distinction is not pedantry. Propensity and cost both feed the ROI guardrail that decides
*who gets contacted at all*: raise the contact cost and customers drop out of the campaign
entirely, which changes the targeted count, the total cost and the blended ROI in a way no linear
rescale of the existing columns can reproduce. Re-running is the only way to get an answer that
matches what the pipeline would actually do.

The two threshold controls work differently and deliberately so. Customer value and risk
thresholds do not change any recommendation — they narrow *who you choose to contact* from the
plan the engine produced — so they are applied as a filter afterwards and respond instantly.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from app.components.kpi import Kpi, kpi_row
from app.components.layout import assumption_notice, page_header, section
from app.components.tables import data_table
from app.data_access import (
    campaign_summary,
    load_customer_master,
    load_source_data,
    prediction_date,
    require,
)
from app.formatting import integer, money, percent, signed_percent

# `src.retention` and `src.models` pull in scikit-learn, LightGBM and SHAP, which cost around
# twenty seconds to import from a network share. `dashboard.py` imports every view module at
# start-up, so importing them here would make *all eight pages* wait for machine-learning
# libraries that only this one page uses. They are therefore imported inside the functions that
# need them: the cost is paid once, on the first visit to this page, by the person who asked for
# it.

SCENARIO_COLUMNS = [
    "customer_id",
    "churn_probability",
    "risk_level",
    "expected_future_revenue",
    "revenue_at_risk",
    "retention_propensity",
    "recommended_action",
    "recommended_offer",
    "campaign_cost",
    "expected_retained_revenue",
    "expected_roi",
]


@st.cache_resource(show_spinner="Loading the churn model...")
def _model():
    from src.config.settings import get_settings
    from src.models.registry import load_model

    return load_model(get_settings().models_path)


@st.cache_data(show_spinner="Re-running the retention layer with your assumptions...")
def _scenario(
    propensity: float,
    communication_cost: float,
    max_discount: float,
    min_roi: float,
    as_of: str,
) -> pd.DataFrame:
    """Run the genuine decision layer under modified parameters.

    The churn probabilities are held fixed at the shipped predictions: the model is not being
    retrained, only the retention economics are being varied. ``revenue_horizon_days`` is pinned
    to the model's own horizon so that changing an unrelated slider cannot silently move the
    revenue projection onto a different window.
    """
    from src.retention.params import RetentionParams
    from src.retention.pipeline import build_retention_layer

    model = _model()
    data = load_source_data()
    params = RetentionParams(
        revenue_horizon_days=model.metadata.horizon_days,
        base_retention_propensity=propensity,
        communication_cost=communication_cost,
        max_offer_discount_pct=max_discount,
        min_expected_roi=min_roi,
    )
    result = build_retention_layer(data, as_of_date=as_of, model=model, params=params)

    scores = result.scores.rename(
        columns={
            "Customer ID": "customer_id",
            "Churn probability": "churn_probability",
            "Risk level": "risk_level",
            "Expected future revenue": "expected_future_revenue",
            "Revenue at risk": "revenue_at_risk",
            "Retention propensity (ASSUMED)": "retention_propensity",
            "Expected retained revenue": "expected_retained_revenue",
        }
    )[
        [
            "customer_id",
            "churn_probability",
            "risk_level",
            "expected_future_revenue",
            "revenue_at_risk",
            "retention_propensity",
            "expected_retained_revenue",
        ]
    ]
    recommendations = result.recommendations.rename(
        columns={
            "Customer ID": "customer_id",
            "Recommended action": "recommended_action",
            "Recommended offer": "recommended_offer",
            "Campaign cost": "campaign_cost",
            "Expected ROI": "expected_roi",
        }
    )[
        [
            "customer_id",
            "recommended_action",
            "recommended_offer",
            "campaign_cost",
            "expected_roi",
        ]
    ]

    merged = scores.merge(recommendations, on="customer_id", validate="1:1")
    merged["is_targeted"] = merged["recommended_action"].ne("Do Not Target")
    return merged


def render() -> None:
    require("features", "predictions", "scores", "recommendations", "model")

    from src.retention.params import RetentionParams

    master = load_customer_master()
    as_of = prediction_date(master)
    baseline = campaign_summary(master)
    defaults = RetentionParams()

    page_header(
        "What-If Simulator",
        "Vary the campaign economics and the propensity assumption, and see the plan change",
        as_of=as_of,
    )

    assumption_notice(
        "The <b>intervention success rate</b> below is the assumption the whole retention "
        "business case rests on, and this dataset cannot measure it — that needs a campaign log "
        "and an untreated control group. Sweeping it here is the honest way to use it: see how "
        "much the answer depends on a number nobody has measured."
    )

    controls = _controls(defaults)

    scenario = _scenario(
        controls["propensity"],
        controls["communication_cost"],
        controls["max_discount"],
        controls["min_roi"],
        as_of,
    )

    targeted = scenario[
        scenario["is_targeted"]
        & scenario["expected_future_revenue"].ge(controls["value_threshold"])
        & scenario["churn_probability"].ge(controls["risk_threshold"])
    ]

    cost = float(targeted["campaign_cost"].sum())
    retained_revenue = float(targeted["expected_retained_revenue"].sum())
    retained_customers = float(
        (targeted["churn_probability"] * targeted["retention_propensity"]).sum()
    )
    roi = (retained_revenue - cost) / cost if cost > 0 else float("nan")

    section("Outcome", "Compared against the shipped plan at its default assumptions.")
    kpi_row(
        [
            Kpi(
                "Customers targeted",
                integer(len(targeted)),
                _delta(len(targeted), baseline["targeted"], integer),
                tone=_tone(len(targeted) - baseline["targeted"]),
            ),
            Kpi(
                "Campaign cost",
                money(cost),
                _delta(cost, baseline["cost"], money),
                tone=_tone(baseline["cost"] - cost),
            ),
            Kpi(
                "Expected customers retained",
                integer(retained_customers),
                _delta(retained_customers, baseline["customers_retained"], integer),
                tone=_tone(retained_customers - baseline["customers_retained"]),
            ),
            Kpi(
                "Expected revenue retained",
                money(retained_revenue),
                _delta(retained_revenue, baseline["expected_retained"], money),
                tone=_tone(retained_revenue - baseline["expected_retained"]),
            ),
            Kpi(
                "ROI",
                signed_percent(roi),
                _delta(roi, baseline["roi"], lambda v: signed_percent(v, 0)),
                tone=_tone(roi - baseline["roi"]),
            ),
        ]
    )

    if _is_baseline(controls, defaults):
        st.success(
            "These controls are at the pipeline's default assumptions, so the figures above "
            "reproduce the shipped plan exactly — the simulator and `scripts/retention.py` agree."
        )

    net = retained_revenue - cost
    st.markdown(
        f"**Net expected gain: {money(net)}** — {money(retained_revenue)} expected revenue "
        f"retained against {money(cost)} of campaign cost, across {integer(len(targeted))} "
        f"contacted customers."
    )

    section("Who gets contacted under these assumptions")
    left, right = st.columns([2, 3], gap="medium")
    with left:
        actions = targeted["recommended_action"].value_counts()
        st.markdown("**Actions in the plan**")
        for name, count in actions.items():
            st.markdown(f"- {name}: **{count:,}**")
        dropped = int(scenario["is_targeted"].sum()) - len(targeted)
        if dropped:
            st.caption(
                f"{dropped:,} customers the engine would target are excluded by your value and "
                "risk thresholds."
            )
    with right:
        st.markdown("**Sensitivity: ROI against the assumed success rate**")
        st.caption(
            "Every other control stays where you set it. Each point is a *full* re-run of the "
            "decision layer — about 5 seconds each, so the sweep is opt-in. Results are cached, "
            "so re-running it after changing one slider only recomputes what actually moved."
        )
        if st.button("Run sensitivity sweep", key="wi_sweep"):
            st.session_state["wi_sweep_done"] = True
        if st.session_state.get("wi_sweep_done"):
            st.plotly_chart(
                _sensitivity_chart(controls, as_of),
                width="stretch",
                key="wi_sensitivity",
            )
        else:
            st.info(
                "Not run yet. The single most useful thing this dashboard can tell a CRM "
                "manager is how hard the plan leans on an unmeasured number."
            )

    section("Scenario detail")
    data_table(
        targeted.sort_values("expected_retained_revenue", ascending=False)[SCENARIO_COLUMNS],
        SCENARIO_COLUMNS,
        download_name="what_if_scenario.csv",
        key="wi_table",
        height=400,
        caption="Produced by the same code path as `python scripts/retention.py`.",
    )


def _controls(defaults: RetentionParams) -> dict[str, float]:
    """The five inputs the brief asks for, in the sidebar."""
    st.sidebar.markdown("### Campaign assumptions")
    st.sidebar.caption("These re-run the decision layer.")

    propensity = st.sidebar.slider(
        "Intervention success rate (ASSUMED)",
        min_value=0.05,
        max_value=0.60,
        value=float(defaults.base_retention_propensity),
        step=0.01,
        format="%.0f%%",
        help="Probability that contacting a customer changes their behaviour. "
        "Cannot be measured from this dataset.",
    )
    communication_cost = st.sidebar.slider(
        "Communication cost per contact",
        min_value=0.0,
        max_value=15.0,
        value=float(defaults.communication_cost),
        step=0.25,
        help=f"A further {defaults.campaign_overhead_per_customer:.2f} of overhead is added per "
        "targeted customer.",
    )
    max_discount = st.sidebar.slider(
        "Maximum discount depth offered (%)",
        min_value=10.0,
        max_value=50.0,
        value=float(defaults.max_offer_discount_pct),
        step=5.0,
        help="Caps how deep an offer the engine may propose. The depth actually offered still "
        "comes from what each customer has responded to.",
    )
    min_roi = st.sidebar.slider(
        "Minimum expected ROI to contact",
        min_value=-0.5,
        max_value=2.0,
        value=float(defaults.min_expected_roi),
        step=0.05,
        format="%.0f%%",
        help="Customers below this are downgraded to Do Not Target.",
    )

    st.sidebar.markdown("### Targeting thresholds")
    st.sidebar.caption("These filter the plan; they do not change any recommendation.")
    value_threshold = st.sidebar.slider(
        "Minimum expected future revenue",
        min_value=0.0,
        max_value=2000.0,
        value=0.0,
        step=50.0,
    )
    risk_threshold = st.sidebar.slider(
        "Minimum churn probability",
        min_value=0.0,
        max_value=1.0,
        value=0.0,
        step=0.05,
        format="%.0f%%",
    )

    return {
        "propensity": propensity,
        "communication_cost": communication_cost,
        "max_discount": max_discount,
        "min_roi": min_roi,
        "value_threshold": value_threshold,
        "risk_threshold": risk_threshold,
    }


def _is_baseline(controls: dict[str, float], defaults: RetentionParams) -> bool:
    return (
        controls["propensity"] == defaults.base_retention_propensity
        and controls["communication_cost"] == defaults.communication_cost
        and controls["max_discount"] == defaults.max_offer_discount_pct
        and controls["min_roi"] == defaults.min_expected_roi
        and controls["value_threshold"] == 0.0
        and controls["risk_threshold"] == 0.0
    )


def _sensitivity_chart(controls: dict[str, float], as_of: str):
    """ROI across a sweep of the propensity assumption, holding everything else fixed."""
    import plotly.graph_objects as go

    from app.theme import AXIS, SURFACE, categorical

    points = [0.05, 0.15, 0.25, 0.35, 0.50]
    xs, ys = [], []
    for value in points:
        frame = _scenario(
            value,
            controls["communication_cost"],
            controls["max_discount"],
            controls["min_roi"],
            as_of,
        )
        chosen = frame[
            frame["is_targeted"]
            & frame["expected_future_revenue"].ge(controls["value_threshold"])
            & frame["churn_probability"].ge(controls["risk_threshold"])
        ]
        cost = float(chosen["campaign_cost"].sum())
        if cost <= 0:
            continue
        xs.append(value)
        ys.append((float(chosen["expected_retained_revenue"].sum()) - cost) / cost)

    figure = go.Figure(
        go.Scatter(
            x=xs,
            y=ys,
            mode="lines+markers",
            name="ROI",
            line=dict(color=categorical(0), width=2),
            marker=dict(size=9, color=categorical(0), line=dict(color=SURFACE, width=2)),
            hovertemplate="Success rate %{x:.0%}<br>ROI %{y:+.0%}<extra></extra>",
        )
    )
    figure.add_hline(y=0, line_width=1, line_color=AXIS, annotation_text="Breaks even")
    figure.update_layout(
        height=300,
        xaxis_title="Assumed intervention success rate",
        yaxis_title="Blended ROI",
        showlegend=False,
    )
    figure.update_xaxes(tickformat=".0%")
    figure.update_yaxes(tickformat="+.0%")
    return figure


def _delta(value: float, reference: float, formatter) -> str:
    """A signed comparison against the shipped baseline."""
    if reference is None or (isinstance(reference, float) and pd.isna(reference)):
        return ""
    difference = value - reference
    if abs(difference) < 1e-9:
        return "same as the shipped plan"
    sign = "+" if difference > 0 else "−"
    return f"{sign}{formatter(abs(difference))} vs shipped plan"


def _tone(direction: float) -> str:
    if direction > 1e-9:
        return "good"
    if direction < -1e-9:
        return "bad"
    return "neutral"
