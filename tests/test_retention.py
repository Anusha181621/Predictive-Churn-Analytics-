"""Tests for the retention decision layer.

The properties worth pinning are the ones that make a retention list *safe to act on*:

* a projection never claims more revenue than the history supporting it;
* revenue at risk stays free of the propensity assumption, so a business that rejects the
  assumption can still use the exposure figure;
* the brief's two guardrails hold structurally -- premium customers are never handed a discount, and
  nobody is contacted at a negative expected ROI;
* a discount is costed against the revenue it recovers, not against everything the customer might
  ever spend;
* every recommendation field varies with the customer.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.retention.params import PRIORITY_BANDS, RetentionParams
from src.retention.recommendations import (
    ACTIONS,
    build_recommendation_inputs,
    build_recommendations,
)
from src.retention.scoring import assign_priority, build_retention_propensity, build_scores
from src.retention.segments import SEGMENT_FLAGS, SEGMENTS, build_segments
from src.retention.value import build_expected_revenue


# --------------------------------------------------------------------------------------
# fixtures: a small cohort whose economics can be worked out by hand
# --------------------------------------------------------------------------------------

CUSTOMERS = ["C_CHAMP", "C_HVAR", "C_DISCOUNT", "C_LOST", "C_NEW", "C_SEASONAL", "C_RETURNER"]


@pytest.fixture
def cohort() -> pd.DataFrame:
    """Seven archetypes, one per rule path through the engine."""
    frame = pd.DataFrame(
        {
            "customer_id": CUSTOMERS,
            # value and volume
            "lifetime_revenue": [8000.0, 6000.0, 1200.0, 900.0, 200.0, 3000.0, 1500.0],
            "total_orders": [40.0, 20.0, 8.0, 4.0, 1.0, 9.0, 10.0],
            "orders_365d": [12.0, 4.0, 3.0, 0.0, 1.0, 2.0, 4.0],
            "revenue_365d": [2400.0, 900.0, 400.0, 0.0, 200.0, 600.0, 500.0],
            "customer_tenure_days": [1000.0, 900.0, 700.0, 800.0, 20.0, 900.0, 700.0],
            "average_order_value": [200.0, 300.0, 150.0, 225.0, 200.0, 333.0, 150.0],
            "average_item_value": [60.0, 80.0, 40.0, 55.0, 70.0, 90.0, 45.0],
            "annualized_revenue": [2920.0, 2430.0, 625.0, 410.0, 3650.0, 1215.0, 780.0],
            "value_percentile": [0.99, 0.90, 0.45, 0.30, 0.10, 0.75, 0.50],
            # risk
            "churn_probability": [0.05, 0.85, 0.55, 0.90, 0.40, 0.70, 0.65],
            "risk_level": ["Low", "Critical", "Medium", "Critical", "Medium", "High", "High"],
            "recency_days": [15.0, 200.0, 120.0, 800.0, 10.0, 250.0, 150.0],
            "purchase_gap_ratio": [0.5, 2.5, 1.4, 9.0, 0.2, 1.1, 1.8],
            "expected_purchase_interval_days": [30.0, 80.0, 85.0, 90.0, 90.0, 220.0, 84.0],
            "purchase_regularity": [0.9, 0.4, 0.30, 0.2, np.nan, 0.5, 0.45],
            "revenue_growth": [0.1, -0.6, -0.4, np.nan, np.nan, -0.2, -0.1],
            "recent_vs_historical_revenue": [1.1, 0.4, 0.6, 0.0, np.nan, 0.8, 0.9],
            # discount behaviour
            "full_price_order_rate": [0.9, 0.2, 0.05, 0.3, 0.5, 0.8, 0.4],
            "discount_order_rate": [0.1, 0.8, 0.95, 0.7, 0.5, 0.2, 0.6],
            "discount_dependency_score": [0.05, 0.70, 0.92, 0.60, 0.40, 0.10, 0.55],
            "average_discount_when_discounted": [10.0, 20.0, 25.0, 30.0, 15.0, 10.0, 20.0],
            "average_discount": [5.0, 16.0, 24.0, 21.0, 8.0, 2.0, 12.0],
            "is_discount_driven": [False, True, True, True, False, False, True],
            # returns
            "return_rate": [0.05, 0.10, 0.15, 0.20, 0.0, 0.05, 0.55],
            # product affinity
            "preferred_category": ["Apparel", "Footwear", "Apparel", "Apparel", "Apparel",
                                   "Outerwear", "Footwear"],
            "preferred_product_gender": ["Women", "Women", "Men", "Women", "Men", "Women", "Men"],
            "category_diversity": [0.80, 0.30, 0.35, 0.20, 0.0, 0.75, 0.40],
            "category_count": [5.0, 2.0, 2.0, 1.0, 1.0, 4.0, 2.0],
            # seasonality
            "is_seasonal_buyer": [False, False, False, False, False, True, False],
            "seasonally_explained_inactivity": [False, False, False, False, False, True, False],
            "days_from_preferred_season": [np.nan, np.nan, np.nan, np.nan, np.nan, 150.0, np.nan],
            # lifecycle
            "is_dormant_buyer": [False, False, False, True, False, False, False],
            "is_new_buyer": [False, False, False, False, True, False, False],
            "is_one_time_buyer": [False, False, False, False, True, False, False],
            "has_purchase_history": [True] * 7,
            "acquisition_channel": ["Referral", "Instagram", "Google Ads", "Email", "Facebook",
                                     "Direct", "Influencer"],
            "age": [45.0, 28.0, 34.0, 52.0, 22.0, 39.0, 31.0],
            "customer_value_segment": ["High Value", "High Value", "Medium Value", "Low Value",
                                        "Low Value", "Medium Value", "Medium Value"],
        }
    )
    return frame


@pytest.fixture
def params() -> RetentionParams:
    return RetentionParams()


@pytest.fixture
def indexed(cohort: pd.DataFrame) -> pd.DataFrame:
    return cohort.set_index("customer_id")


@pytest.fixture
def products() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sku_id": ["P0001", "P0002", "P0003", "P0004", "P0005"],
            "category": ["Apparel", "Apparel", "Footwear", "Outerwear", "Accessories"],
            "subcategory": ["T-Shirts", "Jeans", "Sneakers", "Parkas", "Bags"],
            "brand": ["UrbanEdge", "ModeStreet", "NovaWear", "LuxeLine", "TrendAura"],
            "product_gender": ["Men", "Women", "Unisex", "Women", "Unisex"],
            "list_price": [25.0, 60.0, 95.0, 250.0, 45.0],
        }
    )


@pytest.fixture
def transactions() -> pd.DataFrame:
    rows = [
        ("C_CHAMP", "P0001", 5), ("C_CHAMP", "P0005", 3),
        ("C_HVAR", "P0003", 4), ("C_DISCOUNT", "P0001", 2),
        ("C_LOST", "P0002", 1), ("C_NEW", "P0001", 1),
        ("C_SEASONAL", "P0004", 2), ("C_RETURNER", "P0003", 3),
    ]
    frame = pd.DataFrame(rows, columns=["customer_id", "sku_id", "quantity"])
    frame["purchase_date"] = pd.Timestamp("2025-01-01")
    return frame


# ======================================================================================
# EXPECTED FUTURE REVENUE
# ======================================================================================


def test_projection_blends_recent_and_lifetime_behaviour(
    cohort: pd.DataFrame, params: RetentionParams
) -> None:
    projected = build_expected_revenue(cohort, params)
    champion = projected.loc["C_CHAMP"]
    # 40 orders over 1,000 days is ~14.6/year; 12 in the last year. The blend must sit between.
    assert 12.0 <= champion["expected_orders_per_year"] <= 14.7
    assert champion["expected_average_order_value"] > 0


def test_projection_never_exceeds_a_multiple_of_lifetime_revenue(
    cohort: pd.DataFrame, params: RetentionParams
) -> None:
    """The cap that keeps a projection tethered to the evidence behind it."""
    projected = build_expected_revenue(cohort, params)
    ceiling = cohort.set_index("customer_id")["lifetime_revenue"] * params.max_projection_multiple
    assert (projected["expected_future_revenue"] <= ceiling + 0.01).all()


def test_a_very_new_customer_is_not_extrapolated_into_a_whale(
    cohort: pd.DataFrame, params: RetentionParams
) -> None:
    """C_NEW spent EUR 200 in 20 days. Annualising that implies EUR 3,650 a year.

    This is the failure the interim Section 3 estimate actually exhibited, so it is pinned here.
    """
    projected = build_expected_revenue(cohort, params)
    new = projected.loc["C_NEW"]
    assert new["tenure_floored"] == True  # noqa: E712
    assert new["expected_future_revenue"] <= 200.0 * params.max_projection_multiple
    # And they must not outrank a genuine high-value customer.
    assert new["expected_future_revenue"] < projected.loc["C_CHAMP", "expected_future_revenue"]


def test_projection_carries_its_own_working(cohort: pd.DataFrame, params: RetentionParams) -> None:
    """A CRM manager asking "why is this customer worth that" must get the components."""
    projected = build_expected_revenue(cohort, params)
    for column in (
        "expected_orders_per_year",
        "expected_average_order_value",
        "projected_annual_revenue",
        "historical_annual_revenue",
        "projection_vs_historical_ratio",
        "projection_capped",
        "tenure_floored",
    ):
        assert column in projected.columns


def test_a_dormant_customer_projects_less_than_an_active_one(
    cohort: pd.DataFrame, params: RetentionParams
) -> None:
    projected = build_expected_revenue(cohort, params)
    assert (
        projected.loc["C_LOST", "expected_future_revenue"]
        < projected.loc["C_CHAMP", "expected_future_revenue"]
    )


def test_horizon_scales_the_projection(cohort: pd.DataFrame) -> None:
    short = build_expected_revenue(cohort, RetentionParams(revenue_horizon_days=90))
    long = build_expected_revenue(cohort, RetentionParams(revenue_horizon_days=365))
    assert (
        long["expected_future_revenue"].sum() > short["expected_future_revenue"].sum()
    )


# ======================================================================================
# SEGMENTS
# ======================================================================================


def test_each_archetype_lands_in_its_intended_segment(
    indexed: pd.DataFrame, params: RetentionParams
) -> None:
    segments = build_segments(indexed, params)
    assert segments.loc["C_CHAMP", "primary_segment"] == "Champions"
    assert segments.loc["C_HVAR", "primary_segment"] == "High-Value At Risk"
    assert segments.loc["C_LOST", "primary_segment"] == "Lost Customers"
    assert segments.loc["C_RETURNER", SEGMENT_FLAGS["High-Return Customers"]] == True  # noqa: E712
    assert segments.loc["C_SEASONAL", SEGMENT_FLAGS["Seasonal Customers"]] == True  # noqa: E712
    assert segments.loc["C_NEW", SEGMENT_FLAGS["New Customers"]] == True  # noqa: E712


def test_all_twelve_segments_exist_as_flags(
    indexed: pd.DataFrame, params: RetentionParams
) -> None:
    segments = build_segments(indexed, params)
    assert len(SEGMENTS) == 12
    for segment in SEGMENTS:
        assert SEGMENT_FLAGS[segment] in segments.columns


def test_customers_may_carry_several_segments(
    indexed: pd.DataFrame, params: RetentionParams
) -> None:
    """The brief asks for multi-dimensional membership, not one rigid label."""
    segments = build_segments(indexed, params)
    assert segments["segment_count"].max() > 1
    assert segments["all_segments"].str.contains(";").any()


def test_champions_and_high_value_at_risk_are_mutually_exclusive(
    indexed: pd.DataFrame, params: RetentionParams
) -> None:
    """One requires low churn risk and the other high, so no customer can be both."""
    segments = build_segments(indexed, params)
    both = segments[SEGMENT_FLAGS["Champions"]] & segments[SEGMENT_FLAGS["High-Value At Risk"]]
    assert not both.any()


def test_lost_customers_are_excluded_from_recoverable_segments(
    indexed: pd.DataFrame, params: RetentionParams
) -> None:
    """Nothing about an unrecoverable customer should imply an action."""
    segments = build_segments(indexed, params)
    lost = segments[SEGMENT_FLAGS["Lost Customers"]]
    for segment in ("Dormant Customers", "High-Value At Risk", "Champions", "Loyal Customers"):
        assert not (lost & segments[SEGMENT_FLAGS[segment]]).any()


def test_discount_driven_uses_a_cohort_percentile(
    indexed: pd.DataFrame, params: RetentionParams
) -> None:
    """An absolute threshold labelled over half the real book, which is not a segment.

    Ranking within the cohort keeps the segment discriminating whatever the brand's overall
    promotional intensity happens to be.
    """
    segments = build_segments(indexed, params)
    flagged = segments[SEGMENT_FLAGS["Discount-Driven At Risk"]]
    assert flagged.sum() < len(segments) / 2
    # The most discount-dependent at-risk customer must be in it.
    assert flagged.loc["C_DISCOUNT"] == True  # noqa: E712


def test_a_customer_with_no_history_gets_no_segment(params: RetentionParams) -> None:
    frame = pd.DataFrame(
        {"churn_probability": [0.5], "has_purchase_history": [False], "value_percentile": [np.nan]},
        index=pd.Index(["C_NONE"], name="customer_id"),
    )
    segments = build_segments(frame, params)
    assert segments.loc["C_NONE", "primary_segment"] == "No History"
    assert segments.loc["C_NONE", "segment_count"] == 0


# ======================================================================================
# SCORING
# ======================================================================================


def test_revenue_at_risk_is_probability_times_expected_revenue(
    indexed: pd.DataFrame, cohort: pd.DataFrame, params: RetentionParams
) -> None:
    projected = build_expected_revenue(cohort, params)
    scores = build_scores(indexed, projected, params)
    expected = (
        indexed["churn_probability"] * projected["expected_future_revenue"]
    ).round(2)
    pd.testing.assert_series_equal(
        scores["revenue_at_risk"], expected, check_names=False, atol=0.01
    )


def test_revenue_at_risk_is_free_of_the_propensity_assumption(
    indexed: pd.DataFrame, cohort: pd.DataFrame
) -> None:
    """A business that rejects the assumption must still be able to use the exposure figure."""
    low = build_scores(indexed, build_expected_revenue(cohort), RetentionParams(base_retention_propensity=0.05))
    high = build_scores(indexed, build_expected_revenue(cohort), RetentionParams(base_retention_propensity=0.60))
    pd.testing.assert_series_equal(low["revenue_at_risk"], high["revenue_at_risk"])
    # But everything downstream of the assumption must move.
    assert low["expected_retained_revenue"].sum() < high["expected_retained_revenue"].sum()


def test_propensity_is_flagged_as_an_assumption(
    indexed: pd.DataFrame, params: RetentionParams
) -> None:
    propensity = build_retention_propensity(indexed, params)
    assert propensity["propensity_is_assumption"].all()
    assert propensity["propensity_basis"].str.contains("assumption").all()


def test_propensity_basis_names_every_multiplier_that_fired(
    indexed: pd.DataFrame, params: RetentionParams
) -> None:
    """The number must be auditable rather than taken on faith."""
    propensity = build_retention_propensity(indexed, params)
    assert "discount-responsive" in propensity.loc["C_DISCOUNT", "propensity_basis"]
    assert "absent over a year" in propensity.loc["C_LOST", "propensity_basis"]
    assert "still active" in propensity.loc["C_CHAMP", "propensity_basis"]
    assert "heavy returner" in propensity.loc["C_RETURNER", "propensity_basis"]


def test_propensity_stays_inside_its_bounds(
    indexed: pd.DataFrame, params: RetentionParams
) -> None:
    propensity = build_retention_propensity(indexed, params)["retention_propensity"]
    assert (propensity >= params.min_retention_propensity).all()
    assert (propensity <= params.max_retention_propensity).all()


def test_opportunity_score_is_revenue_at_risk_times_propensity(
    indexed: pd.DataFrame, cohort: pd.DataFrame, params: RetentionParams
) -> None:
    scores = build_scores(indexed, build_expected_revenue(cohort, params), params)
    expected = (scores["revenue_at_risk"] * scores["retention_propensity"]).round(2)
    pd.testing.assert_series_equal(
        scores["retention_opportunity_score"], expected, check_names=False, atol=0.01
    )


def test_priority_bands_are_ordered_and_complete(params: RetentionParams) -> None:
    opportunity = pd.Series(np.linspace(1, 100, 100))
    bands = assign_priority(opportunity, params)
    assert list(bands.cat.categories) == list(PRIORITY_BANDS)
    assert bands.iloc[-1] == "Critical"
    assert bands.iloc[0] == "Low"


def test_zero_opportunity_is_never_promoted_by_its_percentile(params: RetentionParams) -> None:
    """Ranking alone would put a zero in a high band whenever most of the cohort is also zero."""
    opportunity = pd.Series([0.0] * 90 + [10.0] * 10)
    bands = assign_priority(opportunity, params)
    assert (bands[opportunity.eq(0)] == "Low").all()


# ======================================================================================
# RECOMMENDATIONS
# ======================================================================================


@pytest.fixture
def recommended(
    indexed: pd.DataFrame,
    cohort: pd.DataFrame,
    products: pd.DataFrame,
    transactions: pd.DataFrame,
    params: RetentionParams,
) -> pd.DataFrame:
    projected = build_expected_revenue(cohort, params)
    segments = build_segments(indexed, params)
    scores = build_scores(indexed, projected, params)
    joined = indexed.join(segments).join(
        scores[[c for c in scores.columns if c not in indexed.columns]]
    )
    inputs = build_recommendation_inputs(products, transactions)
    return build_recommendations(joined, inputs, params)


def test_every_action_is_one_of_the_briefs_actions(recommended: pd.DataFrame) -> None:
    assert set(recommended["recommended_action"]) <= set(ACTIONS)


def test_a_premium_customer_is_never_offered_a_discount(recommended: pd.DataFrame) -> None:
    """The brief's guardrail, enforced structurally rather than hoped for."""
    champion = recommended.loc["C_CHAMP"]
    assert "discount" not in champion["recommended_offer"].lower()
    assert champion["recommended_action"] not in {"Targeted Discount", "Win-Back"}


def test_a_discount_driven_customer_at_risk_gets_a_discount(recommended: pd.DataFrame) -> None:
    assert recommended.loc["C_DISCOUNT", "recommended_action"] == "Targeted Discount"


def test_a_regular_buyer_gets_a_reminder_before_a_discount(
    indexed: pd.DataFrame,
    cohort: pd.DataFrame,
    products: pd.DataFrame,
    transactions: pd.DataFrame,
    params: RetentionParams,
) -> None:
    """Cheap levers before expensive ones: a predictable buyer barely into their interval needs a
    nudge, not margin. Giving the same customer clockwork cadence must flip the recommendation."""
    regular = cohort.copy()
    regular.loc[regular["customer_id"].eq("C_DISCOUNT"), "purchase_regularity"] = 0.8
    indexed_regular = regular.set_index("customer_id")
    projected = build_expected_revenue(regular, params)
    joined = indexed_regular.join(build_segments(indexed_regular, params))
    scores = build_scores(indexed_regular, projected, params)
    joined = joined.join(scores[[c for c in scores.columns if c not in joined.columns]])
    result = build_recommendations(joined, build_recommendation_inputs(products, transactions), params)
    assert result.loc["C_DISCOUNT", "recommended_action"] == "Replenishment Reminder"
    assert result.loc["C_DISCOUNT", "incentive_cost"] == 0.0


def test_the_offer_depth_matches_the_customers_own_behaviour(
    recommended: pd.DataFrame,
) -> None:
    """A house-standard "15% off" for everybody would be the hardcoding the brief forbids."""
    discount_offer = recommended.loc["C_DISCOUNT", "recommended_offer"]
    # C_DISCOUNT responds at 25%, and the policy cap is 25%.
    assert "25%" in discount_offer
    assert "20%" in recommended.loc["C_HVAR", "recommended_offer"]


def test_a_lost_customer_is_suppressed(recommended: pd.DataFrame) -> None:
    row = recommended.loc["C_LOST"]
    assert row["recommended_action"] == "Do Not Target"
    assert row["campaign_cost"] == 0.0
    assert "recovery is implausible" in row["reason"]


def test_a_seasonal_customer_out_of_season_is_left_alone(recommended: pd.DataFrame) -> None:
    """The seasonality guardrail carried all the way through to the action.

    Discounting a customer who is merely between seasons trains them to wait for a discount.
    """
    row = recommended.loc["C_SEASONAL"]
    assert row["recommended_action"] == "Do Not Target"
    assert "season" in row["reason"].lower()


def test_the_recommended_sku_is_real_and_unowned(
    recommended: pd.DataFrame, transactions: pd.DataFrame, products: pd.DataFrame
) -> None:
    owned = transactions.groupby("customer_id")["sku_id"].agg(set).to_dict()
    valid = set(products["sku_id"])
    for customer_id, row in recommended.iterrows():
        sku = row["recommended_sku"]
        if not sku:
            continue
        assert sku in valid, f"{customer_id}: {sku} is not a real product"
        assert sku not in owned.get(customer_id, set()), f"{customer_id} already owns {sku}"


def test_the_recommended_category_reflects_affinity(recommended: pd.DataFrame) -> None:
    assert recommended.loc["C_DISCOUNT", "recommended_category"] == "Apparel"
    assert recommended.loc["C_HVAR", "recommended_category"] == "Footwear"


def test_the_channel_reflects_how_the_customer_was_acquired(
    recommended: pd.DataFrame,
) -> None:
    assert "Instagram" in recommended.loc["C_HVAR", "recommended_channel"]
    assert "Email" in recommended.loc["C_DISCOUNT", "recommended_channel"]


def test_reasons_cite_the_customers_own_numbers(recommended: pd.DataFrame) -> None:
    """A reason with no numbers in it is a slogan, not an explanation."""
    for customer_id, row in recommended.iterrows():
        reason = row["reason"]
        assert len(reason) > 30, f"{customer_id}: reason too thin"
        assert any(character.isdigit() for character in reason), f"{customer_id}: {reason}"


def test_reasons_differ_between_customers(recommended: pd.DataFrame) -> None:
    assert recommended["reason"].nunique() == len(recommended)


# ======================================================================================
# CAMPAIGN ECONOMICS
# ======================================================================================


def test_a_discount_is_costed_against_the_revenue_it_recovers(
    recommended: pd.DataFrame,
) -> None:
    """The bug this test exists to prevent.

    Costing the discount against *all* expected future revenue made 315 of 1,000 discounts look
    uneconomic and suppressed six of the top ten opportunities: the cost scaled with
    ``EFR x depth`` while the benefit was only ``EFR x churn x propensity``, roughly three times
    smaller. A coupon is redeemed on the order the intervention produces.
    """
    row = recommended.loc["C_DISCOUNT"]
    retained = row["expected_retained_revenue"]
    # 25% of the recovered revenue, plus the contact cost.
    assert row["incentive_cost"] == pytest.approx(retained * 0.25, abs=0.02)
    assert row["campaign_cost"] < retained, "a discount must not cost more than it recovers here"
    assert row["expected_roi"] > 0


def test_no_customer_is_contacted_at_a_negative_expected_roi(
    recommended: pd.DataFrame, params: RetentionParams
) -> None:
    """The brief's economic guardrail."""
    targeted = recommended[recommended["recommended_action"].ne("Do Not Target")]
    assert (targeted["expected_roi"] > params.min_expected_roi).all()


def test_suppressed_customers_cost_nothing(recommended: pd.DataFrame) -> None:
    suppressed = recommended[recommended["recommended_action"].eq("Do Not Target")]
    assert (suppressed["campaign_cost"] == 0.0).all()
    assert suppressed["expected_roi"].isna().all()


def test_an_economically_suppressed_action_is_recorded(
    indexed: pd.DataFrame,
    cohort: pd.DataFrame,
    products: pd.DataFrame,
    transactions: pd.DataFrame,
) -> None:
    """Demanding an implausible ROI must suppress everyone and say what it dropped."""
    strict = RetentionParams(min_expected_roi=1000.0)
    projected = build_expected_revenue(cohort, strict)
    joined = indexed.join(build_segments(indexed, strict))
    joined = joined.join(
        build_scores(indexed, projected, strict)[["revenue_at_risk", "retention_propensity",
                                                   "expected_retained_revenue", "priority",
                                                   "expected_future_revenue"]]
    )
    inputs = build_recommendation_inputs(products, transactions)
    result = build_recommendations(joined, inputs, strict)
    assert (result["recommended_action"] == "Do Not Target").all()
    dropped = result[result["suppressed_action"].ne("")]
    assert not dropped.empty
    assert dropped["reason"].str.contains("uneconomic").all()


def test_raising_the_assumed_propensity_targets_more_customers(
    indexed: pd.DataFrame, cohort: pd.DataFrame, products: pd.DataFrame, transactions: pd.DataFrame
) -> None:
    """A visible, auditable consequence of the assumption, which is the point of labelling it."""
    inputs = build_recommendation_inputs(products, transactions)

    def targeted(propensity: float) -> int:
        params = RetentionParams(base_retention_propensity=propensity)
        projected = build_expected_revenue(cohort, params)
        joined = indexed.join(build_segments(indexed, params))
        scores = build_scores(indexed, projected, params)
        joined = joined.join(scores[[c for c in scores.columns if c not in joined.columns]])
        result = build_recommendations(joined, inputs, params)
        return int(result["recommended_action"].ne("Do Not Target").sum())

    assert targeted(0.40) >= targeted(0.03)


# ======================================================================================
# PARAMETER VALIDATION
# ======================================================================================


@pytest.mark.parametrize(
    ("kwargs", "fragment"),
    [
        ({"revenue_horizon_days": 0}, "revenue_horizon_days must be positive"),
        ({"recent_behaviour_weight": 1.5}, "recent_behaviour_weight"),
        ({"base_retention_propensity": 0.0}, "base_retention_propensity"),
        ({"max_projection_multiple": 0.5}, "max_projection_multiple"),
        ({"low_value_percentile": 0.9, "high_value_percentile": 0.5}, "value percentiles"),
        ({"communication_cost": -1.0}, "costs cannot be negative"),
        ({"min_offer_discount_pct": 40.0, "max_offer_discount_pct": 20.0}, "offer discount bounds"),
    ],
)
def test_invalid_params_are_rejected(kwargs: dict, fragment: str) -> None:
    with pytest.raises(ValueError, match=fragment):
        RetentionParams(**kwargs).validate()


def test_assumptions_are_separated_from_policy_inputs(params: RetentionParams) -> None:
    """A reader must be able to see which numbers came from the business and which were invented."""
    assumptions = params.assumptions()
    policy = params.policy_inputs()
    assert "base_retention_propensity" in assumptions
    assert "communication_cost" in policy
    assert not set(assumptions) & set(policy)
    for detail in assumptions.values():
        assert detail["kind"] == "ASSUMPTION"
        assert len(str(detail["why"])) > 30
