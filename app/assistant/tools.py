"""The tools the assistant answers from.

Every figure the assistant reports has to come back from one of these functions. That is the whole
design: the model chooses *which* questions to ask of the data and how to phrase the answer, but it
never supplies a number itself. An answer is therefore reproducible -- each figure can be traced to
a tool call, and each tool call to a column of the same master frame the pages render.

Three rules shaped these signatures:

**Return what a sentence needs, not what a dataframe holds.** A tool that dumped forty columns
would spend the context window on noise and invite the model to quote a field it does not
understand. Each function returns a small, named set of figures.

**Filters use the vocabulary the dashboard already uses.** ``risk_level``, ``segment``, ``country``,
``channel`` and ``category`` are the same dimensions as the filter bar (`CUSTOMER_FILTERS` in
``app/components/filters.py``), so an answer can always be checked against a page.

**An unknown input is an answer, not an exception.** Asking for a customer who does not exist, or a
dimension that is not carried, returns a message naming what *is* available. A raised exception
would end the turn; a message lets the model correct itself and try again.
"""

from __future__ import annotations

import json
from typing import Any

import pandas as pd

from app.data_access import (
    campaign_summary,
    load_customer_master,
    load_explanations,
    load_global_importance,
    load_model_metrics,
    prediction_date,
)
from src.config.settings import get_settings

__all__ = [
    "ORDERINGS",
    "DIMENSIONS",
    "MEASURES",
    "book_summary",
    "rank_customers",
    "customer_detail",
    "aggregate",
    "churn_drivers",
    "model_summary",
    "TOOL_FUNCTIONS",
]

#: Hard ceiling on rows returned by :func:`rank_customers`. A "top customers" question is answered
#: by a readable list; a request for hundreds is a request for the CSV export, not for a sentence.
MAX_ROWS = 50

#: What ``rank_customers`` can sort by, mapped to the master frame's columns. Every one of these is
#: descending-is-worse, so the ordering never needs a direction argument.
ORDERINGS: dict[str, str] = {
    "revenue_at_risk": "revenue_at_risk",
    "churn_probability": "churn_probability",
    "retention_opportunity": "retention_opportunity_score",
    "lifetime_revenue": "lifetime_revenue",
    "expected_future_revenue": "expected_future_revenue",
}

#: What ``aggregate`` can group by.
DIMENSIONS: dict[str, str] = {
    "segment": "primary_segment",
    "risk_level": "risk_level",
    "priority": "priority",
    "channel": "acquisition_channel",
    "country": "country",
    "city": "city",
    "category": "preferred_category",
    "customer_value": "customer_value_segment",
}

#: What ``aggregate`` can measure. ``customers`` is a count; the rest are column reductions.
MEASURES: dict[str, tuple[str, str]] = {
    "customers": ("customer_id", "count"),
    "mean_churn_probability": ("churn_probability", "mean"),
    "revenue_at_risk": ("revenue_at_risk", "sum"),
    "lifetime_revenue": ("lifetime_revenue", "sum"),
    "expected_future_revenue": ("expected_future_revenue", "sum"),
}

#: Filter arguments shared by ``rank_customers``, mapped to their columns.
_FILTERS: dict[str, str] = {
    "risk_level": "risk_level",
    "segment": "primary_segment",
    "country": "country",
    "channel": "acquisition_channel",
    "category": "preferred_category",
}


def _dump(payload: Any) -> str:
    """Serialise a tool result.

    Compact separators and no indentation: this text is going into the context window, and pretty
    printing a fifty-row table costs tokens that buy the model nothing.
    """
    return json.dumps(payload, default=str, separators=(",", ":"))


def _problem(message: str, **context: Any) -> str:
    """A tool result that says what went wrong and what would work instead."""
    return _dump({"error": message, **context})


def _round(value: Any, places: int = 2) -> Any:
    """Round for display, leaving anything non-numeric alone.

    numpy scalars are unwrapped first. ``json.dumps(default=str)`` would otherwise render an
    ``int64`` as the *string* ``"5"``, and a number quoted as text invites the model to treat it
    as a label rather than a quantity.
    """
    if hasattr(value, "item") and not isinstance(value, (str, bytes)):
        try:
            value = value.item()
        except (AttributeError, ValueError):  # pragma: no cover - not a numpy scalar after all
            return value
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return round(value, places)
    return value


def _money(value: Any) -> Any:
    return _round(value, 2)


def book_summary() -> str:
    """Summarise the whole customer base: size, churn risk, revenue exposure and campaign totals.

    Call this first when a question is about the business as a whole rather than about particular
    customers, and to get your bearings before a more specific query.
    """
    master = load_customer_master()
    settings = get_settings()
    campaign = campaign_summary(master)
    bands = master["risk_level"].value_counts().to_dict()

    return _dump(
        {
            "as_of_date": prediction_date(master),
            "currency": settings.currency,
            "churn_horizon_days": settings.churn_inactivity_days,
            "customers": int(len(master)),
            "mean_churn_probability": _round(float(master["churn_probability"].mean()), 4),
            "customers_by_risk_level": {k: int(v) for k, v in bands.items()},
            "revenue_at_risk": _money(float(master["revenue_at_risk"].sum())),
            "expected_future_revenue": _money(float(master["expected_future_revenue"].sum())),
            "lifetime_revenue": _money(float(master["lifetime_revenue"].sum())),
            "customers_targeted": campaign["targeted"],
            "customers_suppressed": campaign["suppressed"],
            "campaign_cost_ASSUMED": _money(campaign["cost"]),
            "expected_retained_revenue_ASSUMED": _money(campaign["expected_retained"]),
            "note": (
                "Figures with the ASSUMED suffix depend on the retention-propensity assumption, "
                "which this dataset cannot measure. Revenue at risk does not depend on it."
            ),
        }
    )


def rank_customers(
    order_by: str = "revenue_at_risk",
    limit: int = 10,
    risk_level: str = "",
    segment: str = "",
    country: str = "",
    channel: str = "",
    category: str = "",
) -> str:
    """List the customers that rank highest on a measure, optionally narrowed to a group.

    Args:
        order_by: One of revenue_at_risk, churn_probability, retention_opportunity,
            lifetime_revenue, expected_future_revenue. Always ranked highest first.
        limit: How many customers to return, at most 50.
        risk_level: Optional. Keep only this risk band: Low, Medium, High or Critical.
        segment: Optional. Keep only this behavioural segment, e.g. "Champions".
        country: Optional. Keep only customers in this country.
        channel: Optional. Keep only customers acquired through this channel.
        category: Optional. Keep only customers whose preferred product category is this.
    """
    if order_by not in ORDERINGS:
        return _problem(
            f"Unknown order_by {order_by!r}.", valid_order_by=sorted(ORDERINGS)
        )

    master = load_customer_master()
    frame = master
    applied: dict[str, str] = {}

    for name, value in (
        ("risk_level", risk_level),
        ("segment", segment),
        ("country", country),
        ("channel", channel),
        ("category", category),
    ):
        if not value:
            continue
        column = _FILTERS[name]
        if column not in frame.columns:
            continue
        matched = frame[frame[column].astype(str).str.casefold() == value.casefold()]
        if matched.empty:
            # Naming the available values lets the model retry with a real one instead of
            # reporting an empty result as though the business had no such customers.
            return _problem(
                f"No customers matched {name}={value!r}.",
                available=sorted(master[column].dropna().astype(str).unique().tolist())[:25],
            )
        frame = matched
        applied[name] = value

    limit = max(1, min(int(limit), MAX_ROWS))
    column = ORDERINGS[order_by]
    top = frame.sort_values(column, ascending=False).head(limit)

    columns = [
        "customer_id",
        "churn_probability",
        "risk_level",
        "revenue_at_risk",
        "lifetime_revenue",
        "primary_segment",
        "country",
        "acquisition_channel",
        "recommended_action",
        "top_driver",
    ]
    rows = [
        {k: _round(v, 4) if k == "churn_probability" else _round(v) for k, v in row.items()}
        for row in top[[c for c in columns if c in top.columns]].to_dict(orient="records")
    ]

    return _dump(
        {
            "ordered_by": order_by,
            "filters_applied": applied or "none",
            "matching_customers": int(len(frame)),
            "returned": len(rows),
            "customers": rows,
        }
    )


def customer_detail(customer_id: str) -> str:
    """Look up one customer: their risk, value, segment, recommended action and churn drivers.

    Args:
        customer_id: The customer identifier, e.g. "CUST0234".
    """
    master = load_customer_master()
    matched = master[master["customer_id"].astype(str).str.casefold() == customer_id.casefold()]
    if matched.empty:
        return _problem(
            f"No customer with id {customer_id!r}.",
            hint="Ids look like CUST0234. Use rank_customers to find real ones.",
        )

    row = matched.iloc[0]
    fields = {
        "customer_id": "customer_id",
        "churn_probability": "churn_probability",
        "risk_level": "risk_level",
        "priority": "priority",
        "primary_segment": "primary_segment",
        "all_segments": "all_segments",
        "customer_value_segment": "customer_value_segment",
        "country": "country",
        "city": "city",
        "acquisition_channel": "acquisition_channel",
        "preferred_category": "preferred_category",
        "total_orders": "total_orders",
        "lifetime_revenue": "lifetime_revenue",
        "recency_days": "recency_days",
        "customer_tenure_days": "customer_tenure_days",
        "expected_future_revenue": "expected_future_revenue",
        "revenue_at_risk": "revenue_at_risk",
        "recommended_action": "recommended_action",
        "recommended_offer": "recommended_offer",
        "recommended_channel": "recommended_channel",
        "reason": "reason",
    }
    profile = {
        name: _round(row[column], 4 if name == "churn_probability" else 2)
        for name, column in fields.items()
        if column in matched.columns and pd.notna(row[column])
    }

    drivers: list[dict[str, Any]] = []
    explanations = load_explanations()
    theirs = explanations[
        explanations["Customer ID"].astype(str).str.casefold() == customer_id.casefold()
    ].sort_values("Driver rank")
    for _, driver in theirs.iterrows():
        drivers.append(
            {
                "rank": int(driver["Driver rank"]),
                "driver": driver.get("Feature label"),
                "direction": driver.get("Direction"),
                "explanation": driver.get("Human-readable explanation"),
            }
        )

    return _dump(
        {
            "customer": profile,
            "churn_drivers": drivers,
            "note": (
                "Churn probability is the modelled likelihood of no purchase within the horizon, "
                "not a statement about what this individual will do."
            ),
        }
    )


def aggregate(dimension: str, measure: str = "customers") -> str:
    """Break a measure down by a dimension, to compare groups against each other.

    Args:
        dimension: One of segment, risk_level, priority, channel, country, city, category,
            customer_value.
        measure: One of customers, mean_churn_probability, revenue_at_risk, lifetime_revenue,
            expected_future_revenue.
    """
    if dimension not in DIMENSIONS:
        return _problem(f"Unknown dimension {dimension!r}.", valid_dimensions=sorted(DIMENSIONS))
    if measure not in MEASURES:
        return _problem(f"Unknown measure {measure!r}.", valid_measures=sorted(MEASURES))

    master = load_customer_master()
    column = DIMENSIONS[dimension]
    if column not in master.columns:
        return _problem(f"The data does not carry {dimension!r}.")

    value_column, how = MEASURES[measure]
    grouped = master.groupby(column)[value_column].agg(how).sort_values(ascending=False)

    places = 4 if measure == "mean_churn_probability" else 2
    return _dump(
        {
            "dimension": dimension,
            "measure": measure,
            "groups": [
                {"group": str(name), measure: _round(value, places)}
                for name, value in grouped.items()
            ],
        }
    )


def churn_drivers(limit: int = 10) -> str:
    """List what drives churn across the whole customer base, strongest first.

    Use this for "why are customers leaving?" at the level of the business. For one customer's own
    reasons, use customer_detail instead.

    Args:
        limit: How many drivers to return, at most 25.
    """
    importance = load_global_importance()
    limit = max(1, min(int(limit), 25))
    top = importance.sort_values("rank").head(limit)

    return _dump(
        {
            "drivers": [
                {
                    "rank": int(row["rank"]),
                    "driver": row.get("label", row.get("feature")),
                    "share_of_importance": _round(row.get("importance_share"), 4),
                    "direction": row.get("direction"),
                }
                for _, row in top.iterrows()
            ],
            "note": (
                "Direction is measured from the data. 'mixed / non-monotone' means the model did "
                "not learn a single direction for that driver, and it should not be described as "
                "raising or lowering risk."
            ),
        }
    )


def model_summary() -> str:
    """Report how accurate the churn model is, with the caveats the training run recorded.

    Use this for any question about whether the model can be trusted, how it was validated, or how
    good the predictions are.
    """
    metrics = load_model_metrics()
    test = metrics.get("metrics", {}).get("test", {})
    scores = test.get("metrics", {})

    return _dump(
        {
            "model": metrics.get("model_name"),
            "trained_at": metrics.get("trained_at"),
            "churn_horizon_days": metrics.get("horizon_days"),
            "tested_on_dates": metrics.get("test_as_of_dates"),
            "test_customers_scored": test.get("n"),
            "accuracy": _round(scores.get("accuracy"), 4),
            "accuracy_predicting_the_majority_class_always": _round(
                scores.get("majority_class_accuracy"), 4
            ),
            "accuracy_lift_over_that_baseline": _round(
                scores.get("accuracy_lift_over_majority"), 4
            ),
            "roc_auc": _round(scores.get("roc_auc"), 4),
            "pr_auc": _round(scores.get("pr_auc"), 4),
            "precision": _round(scores.get("precision"), 4),
            "recall": _round(scores.get("recall"), 4),
            "calibration_bias": _round(scores.get("calibration_bias"), 4),
            "caveats_recorded_by_the_training_run": metrics.get("notes", []),
            "note": (
                "Accuracy on its own flatters any model when one class dominates. Quote it "
                "against the majority-class baseline, and pass on the caveats above rather than "
                "reporting the headline figure alone."
            ),
        }
    )


#: The callables the agent exposes, in the order they are offered to the model. Kept as plain
#: functions here -- ``app.assistant.agent`` decorates them -- so every one can be called directly
#: from a test with no client, no key and no network.
TOOL_FUNCTIONS = (
    book_summary,
    rank_customers,
    customer_detail,
    aggregate,
    churn_drivers,
    model_summary,
)
