"""Model Performance -- how good the churn model actually is, including where it is not.

The headline metrics come from the **test** period, which the training run scored exactly once
after everything -- model *and* calibrator -- had been fitted on strictly earlier data with a
horizon-wide embargo in between. That is what makes them an estimate of future performance rather
than a description of the training set.

This page also surfaces the warnings the training run recorded about itself. The pipeline scores
eight one-line heuristics as a sanity floor on every run and says so when the model fails to clear
one; hiding that here would defeat the point of measuring it.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from app.charts.model import (
    confusion_matrix,
    dependence_curve,
    feature_importance,
    permutation_importance,
    reliability_curve,
)
from app.components.kpi import Kpi, kpi_row
from app.components.layout import chart_card, page_header, section
from app.components.tables import data_table
from app.data_access import (
    load_customer_master,
    load_global_importance,
    load_model_metrics,
    load_shap_dependence,
    load_shap_summary,
    missing,
    require,
)
from app.formatting import integer, money, percent, ratio

SHAP_TABLE_COLUMNS = [
    "rank",
    "label",
    "feature",
    "mean_abs_shap",
    "importance_share",
    "mean_shap",
    "direction",
]


def render() -> None:
    require("metrics")
    metrics = load_model_metrics()

    page_header(
        "Model Performance",
        f"{metrics.get('model_name', 'model')} · {metrics.get('horizon_days', '?')}-day churn "
        f"horizon · {metrics.get('calibration', 'no')} calibration",
        as_of=str(metrics.get("trained_at", ""))[:10],
    )

    periods = metrics.get("metrics", {})
    test = periods.get("test", {})
    scores = test.get("metrics", {})

    _headline(test, scores)
    _notes(metrics)
    _calibration(test, periods.get("calibration_period", {}))
    _importance(metrics)
    _shap()
    _split(metrics)


def _headline(test: dict, scores: dict) -> None:
    section(
        "Test period",
        f"Scored once on {', '.join(test.get('as_of_dates', [])) or 'the held-out period'} · "
        f"{integer(test.get('n'))} customers · base rate {percent(test.get('base_rate'))}.",
    )
    kpi_row(
        [
            Kpi("ROC-AUC", ratio(scores.get("roc_auc"), 4), "Ranking quality"),
            Kpi("PR-AUC", ratio(scores.get("pr_auc"), 4), "Ranking quality on the positive class"),
            Kpi("Precision", ratio(scores.get("precision"), 3), "At the 0.5 threshold"),
            Kpi("Recall", ratio(scores.get("recall"), 3), "At the 0.5 threshold"),
            Kpi("F1", ratio(scores.get("f1"), 3), "At the 0.5 threshold"),
        ]
    )
    kpi_row(
        [
            Kpi("Brier score", ratio(scores.get("brier"), 4), "Lower is better"),
            Kpi("ECE", ratio(scores.get("ece"), 4), "Expected calibration error"),
            Kpi(
                "Calibration bias",
                ratio(scores.get("calibration_bias"), 4),
                f"Predicted {percent(scores.get('mean_predicted'))} vs observed "
                f"{percent(scores.get('mean_observed'))}",
            ),
            Kpi(
                "Lift, top decile",
                ratio(scores.get("lift_top_decile"), 2) + "×",
                "Against random targeting",
            ),
            Kpi(
                "ROC-AUC, High Value",
                ratio(test.get("business", {}).get("high_value_roc_auc"), 4),
                "Where the money is",
                tone="good",
            ),
        ]
    )


def _notes(metrics: dict) -> None:
    """The training run's own caveats, including any failed sanity floor."""
    notes = metrics.get("notes") or []
    if not notes:
        return
    warnings = [n for n in notes if str(n).upper().startswith("WARNING")]
    others = [n for n in notes if n not in warnings]

    if warnings:
        st.warning(
            "**The training run flagged this model.**\n\n"
            + "\n\n".join(f"- {note}" for note in warnings)
        )
    if others:
        with st.expander(f"Training notes ({len(others)})"):
            for note in others:
                st.markdown(f"- {note}")


def _calibration(test: dict, calibration_period: dict) -> None:
    section(
        "Calibration and outcomes",
        "A calibrated probability is what revenue at risk needs — a raw ranking score cannot be "
        "multiplied by customer value and still mean anything.",
    )
    left, right = st.columns(2, gap="medium")
    with left:
        chart_card(
            "Reliability — predicted against observed",
            "Points on the reference line mean the predicted probability matched what happened.",
        )
        st.plotly_chart(
            reliability_curve(test.get("reliability", [])), width="stretch", key="mp_reliability"
        )
    with right:
        chart_card(
            "Confusion matrix",
            "At the default 0.5 threshold. The team works a ranked list rather than a cut-off, "
            "so read this alongside the lift figure rather than instead of it.",
        )
        st.plotly_chart(
            confusion_matrix(test.get("confusion", {})), width="stretch", key="mp_confusion"
        )

    business = test.get("business", {})
    if business:
        kpi_row(
            [
                Kpi(
                    "Revenue at risk in the test period",
                    money(business.get("at_risk_revenue_total")),
                    "The model's own exposure estimate for that period",
                ),
                Kpi(
                    "Captured by the top decile",
                    money(business.get("at_risk_revenue_captured_top_decile")),
                    percent(business.get("revenue_capture_rate_top_decile")) + " of the total",
                ),
                Kpi(
                    "Revenue lift, top decile",
                    ratio(business.get("revenue_lift_top_decile"), 2) + "×",
                    "Against random targeting",
                ),
                Kpi(
                    "PR-AUC, High Value",
                    ratio(business.get("high_value_pr_auc"), 4),
                    "Restricted to high-value customers",
                ),
            ]
        )

    if calibration_period:
        st.caption(
            "The calibrator itself was chosen out-of-fold on a separate held-out period "
            f"({', '.join(calibration_period.get('as_of_dates', [])) or 'earlier'}), because "
            "scoring a calibrator in-sample always flatters it."
        )


def _importance(metrics: dict) -> None:
    section(
        "What drives the model",
        "Permutation importance — the drop in PR-AUC when a feature is shuffled — rather than the "
        "trees' impurity importance, which inflates high-cardinality features whether or not they "
        "predict anything.",
    )
    st.plotly_chart(
        permutation_importance(metrics.get("top_features", [])),
        width="stretch",
        key="mp_permutation",
    )


def _shap() -> None:
    if missing("shap_importance", "shap_summary"):
        st.info(
            "SHAP artefacts are not present. Run `python scripts/explain.py` to generate the "
            "global explainability outputs."
        )
        return

    section(
        "SHAP summary",
        "Written as data rather than images so it can be sorted, filtered and read here rather "
        "than squinted at.",
    )

    importance = load_global_importance()
    left, right = st.columns([3, 2], gap="medium")
    with left:
        chart_card(
            "Global feature importance",
            "Length is importance; colour is the direction the feature pushes risk. Features the "
            "model learned as non-monotone are shown as mixed rather than forced into a direction.",
        )
        st.plotly_chart(
            feature_importance(importance, top_n=15), width="stretch", key="mp_shap_importance"
        )
    with right:
        chart_card("Dependence", "How one feature's contribution changes across its own range.")
        dependence = load_shap_dependence()
        features = dependence["feature"].dropna().unique().tolist()
        if features:
            labels = (
                importance.set_index("feature")["label"].to_dict()
                if "label" in importance
                else {}
            )
            chosen = st.selectbox(
                "Feature",
                features,
                format_func=lambda f: labels.get(f, f),
                key="mp_dep_pick",
            )
            st.plotly_chart(
                dependence_curve(
                    dependence[dependence["feature"] == chosen], labels.get(chosen, chosen)
                ),
                width="stretch",
                key="mp_dependence",
            )

    summary = load_shap_summary()
    present = [c for c in SHAP_TABLE_COLUMNS if c in summary.columns]
    st.markdown("**Per-feature SHAP summary**")
    st.dataframe(
        summary[present].sort_values("rank"),
        column_config={
            "rank": st.column_config.NumberColumn("Rank", format="localized"),
            "label": st.column_config.Column("Feature"),
            "feature": st.column_config.Column("Column"),
            "mean_abs_shap": st.column_config.NumberColumn("Mean |SHAP|", format="%.4f"),
            "importance_share": st.column_config.NumberColumn("Share", format="percent"),
            "mean_shap": st.column_config.NumberColumn("Mean signed SHAP", format="%.4f"),
            "direction": st.column_config.Column("Direction"),
        },
        hide_index=True,
        width="stretch",
        height=320,
    )
    st.download_button(
        "Download shap_summary.csv",
        data=summary.to_csv(index=False).encode("utf-8-sig"),
        file_name="shap_summary.csv",
        mime="text/csv",
        key="mp_shap_dl",
        width="content",
    )
    st.caption(
        "Contributions are on the model's uncalibrated log-odds scale. Calibration is monotone, "
        "so ranking and direction carry over to the reported probability exactly; the magnitudes "
        "do not sum to it."
    )


def _split(metrics: dict) -> None:
    section(
        "How the model was validated",
        "Time-based, not a random split. A row's label describes a period a later row uses for "
        "its features, so an embargo sits before the test period and before the inner selection "
        "split.",
    )
    plan = metrics.get("metrics", {}).get("split", {}).get("plan", {})
    if not plan:
        plan = metrics.get("split", {}).get("plan", {})
    if not plan:
        st.info("No split plan was recorded in the metrics file.")
        return

    embargoed = plan.get("embargoed", {}) or {}
    rows = [
        ("Selection train", plan.get("selection_train", [])),
        ("Embargo (before selection validation)", embargoed.get("before_selection_validation", [])),
        ("Selection validation", plan.get("selection_validation", [])),
        ("Fit (refit)", plan.get("fit", [])),
        ("Calibration", plan.get("calibration", [])),
        ("Embargo (before test)", embargoed.get("before_test", [])),
        ("Test", plan.get("test", [])),
    ]
    table = pd.DataFrame(
        [
            {
                "Stage": name,
                "Periods": len(dates),
                "From": dates[0] if dates else "—",
                "To": dates[-1] if dates else "—",
            }
            for name, dates in rows
        ]
    )
    st.dataframe(table, hide_index=True, width="stretch")

    left, right = st.columns(2)
    left.markdown(
        f"**Training rows:** {integer(metrics.get('train_rows'))} · "
        f"**churn rate in training:** {percent(metrics.get('train_churn_rate'))}"
    )
    right.markdown(
        f"**Features used:** {integer(len(metrics.get('feature_columns', [])))} · "
        f"**seed:** {metrics.get('random_seed')}"
    )

    if not missing("predictions"):
        master = load_customer_master()
        st.caption(
            "For reference, the model's own revenue-at-risk estimate across the current scoring "
            f"date totals {money(float(master['model_revenue_at_risk'].sum()))}. The business "
            "pages instead report "
            f"{money(float(master['revenue_at_risk'].sum()))}, the decision layer's figure — "
            "churn probability × expected future revenue, which is the definition the retention "
            "brief uses. They are different quantities and are never added together."
        )
