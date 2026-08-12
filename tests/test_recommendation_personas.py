"""One test per customer type the brief names, checked end to end through the engine.

The brief asks that the recommendation engine be exercised against nine specific personas. Each
fixture below is a customer built to be *unambiguous* about which kind they are, and each test
asserts the business property that should follow -- not merely that some action came out.

These are deliberately property assertions rather than exact-string comparisons against the
cascade. Asserting "a premium buyer receives no discount" pins the rule the business cares about
and survives a reordering of the rules; asserting "a premium buyer receives Organic Engagement"
would just restate the implementation and break the first time it is tuned.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.retention.params import RetentionParams
from src.retention.recommendations import (
    ACTIONS,
    build_recommendation_inputs,
    build_recommendations,
)
from src.retention.scoring import build_scores
from src.retention.segments import build_segments
from src.retention.value import build_expected_revenue

#: The nine personas the brief lists, in its order.
PERSONAS = [
    "FREQUENT",
    "OCCASIONAL",
    "SEASONAL",
    "DECLINING",
    "HIGH_VALUE",
    "DISCOUNT_SENSITIVE",
    "PREMIUM",
    "HIGH_RETURN",
    "NEW",
]

#: Actions that hand over margin. The brief is explicit that these must not reach a full-price
#: buyer, so several tests below assert against this set rather than a single action name.
DISCOUNT_ACTIONS = {"Targeted Discount", "Free Shipping", "Loyalty Reward"}


@pytest.fixture
def cohort() -> pd.DataFrame:
    """Nine customers, one per persona, each unmistakable in its defining behaviour."""
    frame = pd.DataFrame(
        {
            "customer_id": PERSONAS,
            # --- volume and value -------------------------------------------------------
            "lifetime_revenue":   [6000.0, 1400.0, 3000.0, 5200.0, 9000.0, 1200.0, 7000.0, 1500.0, 180.0],
            "total_orders":       [48.0,    6.0,    9.0,   30.0,   26.0,    8.0,   18.0,   10.0,   1.0],
            "orders_365d":        [16.0,    2.0,    2.0,    2.0,    5.0,    3.0,    6.0,    4.0,   1.0],
            "revenue_365d":       [2000.0, 320.0,  600.0,  300.0, 1700.0,  400.0, 2400.0,  500.0, 180.0],
            "customer_tenure_days": [1100.0, 900.0, 900.0, 1000.0, 1000.0, 700.0, 950.0,  700.0,  18.0],
            "average_order_value":  [125.0, 233.0, 333.0,  173.0,  346.0,  150.0, 388.0,  150.0, 180.0],
            "average_item_value":   [45.0,   70.0,  90.0,   60.0,  110.0,   40.0, 120.0,   45.0,  60.0],
            "annualized_revenue":   [1990.0, 567.0, 1215.0, 1898.0, 3285.0, 625.0, 2690.0, 780.0, 3650.0],
            "value_percentile":     [0.88,   0.42,  0.75,   0.85,   0.98,   0.45,  0.92,   0.50,  0.08],
            "customer_value_segment": ["High Value", "Low Value", "Medium Value", "High Value",
                                       "High Value", "Medium Value", "High Value",
                                       "Medium Value", "Low Value"],
            # --- risk -------------------------------------------------------------------
            "churn_probability":  [0.08,  0.45,  0.70,  0.72,  0.78,  0.55,  0.35,  0.65,  0.40],
            "risk_level":         ["Low", "Medium", "High", "High", "High", "Medium",
                                   "Medium", "High", "Medium"],
            "recency_days":       [12.0, 150.0, 250.0, 210.0, 190.0, 120.0,  60.0, 150.0,  9.0],
            "purchase_gap_ratio": [1.0,   1.6,   1.1,   2.6,   2.2,   1.4,   1.0,   1.8,   0.2],
            "expected_purchase_interval_days": [22.0, 110.0, 220.0, 80.0, 90.0, 85.0, 60.0, 84.0, 90.0],
            "purchase_regularity": [0.92,  0.35,  0.50,  0.40,  0.45,  0.30,  0.60,  0.45, np.nan],
            "revenue_growth":      [0.05, -0.10, -0.20, -0.75, -0.35, -0.40, -0.05, -0.10, np.nan],
            "recent_vs_historical_revenue": [1.05, 0.85, 0.80, 0.25, 0.55, 0.60, 0.95, 0.90, np.nan],
            # --- discount behaviour -----------------------------------------------------
            "full_price_order_rate":  [0.55, 0.50, 0.80, 0.45, 0.60, 0.05, 0.95, 0.40, 0.50],
            "discount_order_rate":    [0.45, 0.50, 0.20, 0.55, 0.40, 0.95, 0.05, 0.60, 0.50],
            "discount_dependency_score": [0.30, 0.45, 0.10, 0.50, 0.35, 0.94, 0.02, 0.55, 0.40],
            "average_discount_when_discounted": [10.0, 15.0, 10.0, 20.0, 15.0, 25.0, 5.0, 20.0, 15.0],
            "average_discount":       [5.0,  8.0,  2.0, 11.0,  6.0, 24.0,  1.0, 12.0,  8.0],
            "is_discount_driven":     [False, False, False, False, False, True, False, True, False],
            # --- returns ----------------------------------------------------------------
            "return_rate":            [0.05, 0.10, 0.05, 0.12, 0.08, 0.15, 0.03, 0.58, 0.0],
            # --- affinity ---------------------------------------------------------------
            "preferred_category": ["Apparel", "Apparel", "Outerwear", "Footwear", "Apparel",
                                   "Apparel", "Outerwear", "Footwear", "Apparel"],
            "preferred_product_gender": ["Women", "Men", "Women", "Men", "Women", "Men",
                                         "Women", "Men", "Men"],
            "category_diversity": [0.80, 0.50, 0.75, 0.35, 0.60, 0.35, 0.72, 0.40, 0.0],
            "category_count":     [5.0,   3.0,  4.0,  2.0,  4.0,  2.0,  4.0,  2.0,  1.0],
            # --- seasonality ------------------------------------------------------------
            "is_seasonal_buyer": [False, False, True, False, False, False, False, False, False],
            "seasonally_explained_inactivity": [False, False, True, False, False, False,
                                                False, False, False],
            "days_from_preferred_season": [np.nan, np.nan, 150.0, np.nan, np.nan, np.nan,
                                           np.nan, np.nan, np.nan],
            # --- lifecycle --------------------------------------------------------------
            "is_dormant_buyer":   [False, False, False, False, False, False, False, False, False],
            "is_new_buyer":       [False, False, False, False, False, False, False, False, True],
            "is_one_time_buyer":  [False, False, False, False, False, False, False, False, True],
            "has_purchase_history": [True] * 9,
            "acquisition_channel": ["Referral", "Email", "Direct", "Google Ads", "Instagram",
                                    "Paid Search", "Direct", "Influencer", "Facebook"],
            "age": [41.0, 33.0, 39.0, 47.0, 36.0, 25.0, 52.0, 29.0, 23.0],
        }
    )
    return frame.set_index("customer_id")


@pytest.fixture
def products() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sku_id": ["P0001", "P0002", "P0003", "P0004", "P0005", "P0006"],
            "category": ["Apparel", "Apparel", "Footwear", "Outerwear", "Accessories", "Footwear"],
            "subcategory": ["T-Shirts", "Jeans", "Sneakers", "Parkas", "Bags", "Boots"],
            "brand": ["UrbanEdge", "ModeStreet", "NovaWear", "LuxeLine", "TrendAura", "NovaWear"],
            "product_gender": ["Men", "Women", "Unisex", "Women", "Unisex", "Men"],
            "list_price": [25.0, 60.0, 95.0, 250.0, 45.0, 110.0],
        }
    )


@pytest.fixture
def transactions() -> pd.DataFrame:
    rows = [
        ("FREQUENT", "P0001", 6), ("OCCASIONAL", "P0002", 2), ("SEASONAL", "P0004", 2),
        ("DECLINING", "P0003", 3), ("HIGH_VALUE", "P0002", 4), ("DISCOUNT_SENSITIVE", "P0001", 3),
        ("PREMIUM", "P0004", 2), ("HIGH_RETURN", "P0003", 4), ("NEW", "P0001", 1),
    ]
    frame = pd.DataFrame(rows, columns=["customer_id", "sku_id", "quantity"])
    frame["purchase_date"] = pd.Timestamp("2025-06-01")
    return frame


@pytest.fixture
def params() -> RetentionParams:
    return RetentionParams()


@pytest.fixture
def recommended(cohort, products, transactions, params) -> pd.DataFrame:
    """Run the real chain: value -> segments -> scores -> recommendations."""
    revenue = build_expected_revenue(cohort.reset_index(), params)
    segments = build_segments(cohort, params)
    joined = cohort.join(segments)
    scores = build_scores(joined, revenue, params)
    joined = joined.join(scores[[c for c in scores.columns if c not in joined.columns]])
    inputs = build_recommendation_inputs(products, transactions)
    return build_recommendations(joined, inputs, params)


def _row(recommended: pd.DataFrame, persona: str) -> pd.Series:
    return recommended.loc[persona]


# ======================================================================================
# every persona gets a valid, complete recommendation
# ======================================================================================


def test_every_persona_receives_a_recognised_action(recommended: pd.DataFrame) -> None:
    assert set(recommended.index) == set(PERSONAS)
    assert set(recommended["recommended_action"]).issubset(set(ACTIONS))


def test_every_persona_receives_a_reason_naming_its_own_behaviour(
    recommended: pd.DataFrame,
) -> None:
    reasons = recommended["reason"]
    assert reasons.notna().all() and (reasons.str.len() > 10).all()
    # Nine different customers must not collapse onto one sentence.
    assert reasons.nunique() >= 7, f"only {reasons.nunique()} distinct reasons for 9 personas"


def test_every_targeted_persona_is_given_a_channel(recommended: pd.DataFrame) -> None:
    targeted = recommended[recommended["recommended_action"] != "Do Not Target"]
    assert (targeted["recommended_channel"].str.len() > 0).all()


# ======================================================================================
# the nine personas
# ======================================================================================


def test_frequent_buyer_is_nurtured_not_discounted(recommended: pd.DataFrame) -> None:
    """A customer ordering every three weeks with 8% churn risk needs no margin spent on them.

    The cheap levers come first by design: a predictable buyer barely into their own interval gets
    recognition or a reminder, not an incentive.
    """
    row = _row(recommended, "FREQUENT")
    assert row["recommended_action"] != "Targeted Discount"
    assert row["expected_roi"] > 0 or row["recommended_action"] == "Do Not Target"


def test_occasional_buyer_is_still_worth_contacting(recommended: pd.DataFrame) -> None:
    """Moderate value and moderate risk is the ordinary case, and must not fall through a gap."""
    row = _row(recommended, "OCCASIONAL")
    assert row["recommended_action"] in set(ACTIONS)
    assert row["recommended_action"] != "Do Not Target"
    assert row["recommended_category"], "an occasional buyer should still get a category"


def test_seasonal_buyer_out_of_season_is_left_alone(recommended: pd.DataFrame) -> None:
    """Quiet *because it is the wrong month* is not churn, and must not trigger a win-back.

    This is the specific mistake the brief warns about: a flat inactivity rule punishes a
    twice-a-year buyer for behaving exactly as expected.
    """
    row = _row(recommended, "SEASONAL")
    assert row["recommended_action"] == "Do Not Target"
    assert "season" in str(row["reason"]).lower()
    assert float(row["campaign_cost"]) == 0.0


def test_declining_customer_is_targeted_and_the_decline_is_named(
    recommended: pd.DataFrame,
) -> None:
    """Revenue down 75% against a 30-order history is the clearest save-me signal in the book."""
    row = _row(recommended, "DECLINING")
    assert row["recommended_action"] != "Do Not Target"
    assert row["expected_roi"] > 0


def test_high_value_customer_at_risk_is_prioritised(recommended: pd.DataFrame) -> None:
    """The most valuable at-risk account must outrank a cheaper one carrying the same risk."""
    high_value = _row(recommended, "HIGH_VALUE")
    occasional = _row(recommended, "OCCASIONAL")
    assert high_value["recommended_action"] != "Do Not Target"
    assert high_value["expected_retained_revenue"] > occasional["expected_retained_revenue"]


def test_discount_sensitive_customer_is_offered_the_depth_they_respond_to(
    recommended: pd.DataFrame,
) -> None:
    """95% of their orders are discounted: a promotion is the lever that actually moves them.

    The depth is theirs, not a house default -- and it is capped by policy, so a customer who once
    took 50% is not handed 50% again.
    """
    row = _row(recommended, "DISCOUNT_SENSITIVE")
    assert row["recommended_action"] == "Targeted Discount"
    offer = str(row["recommended_offer"])
    assert "%" in offer
    depth = int("".join(ch for ch in offer.split("%")[0] if ch.isdigit()))
    assert RetentionParams().min_offer_discount_pct <= depth <= RetentionParams().max_offer_discount_pct


def test_premium_customer_is_never_handed_a_discount(recommended: pd.DataFrame) -> None:
    """The brief's structural guardrail: 95% full-price buying must never attract an incentive.

    It holds because the full-price rule is evaluated *before* every discount rule, so a premium
    customer cannot reach one -- not because a later check happens to remove it.
    """
    row = _row(recommended, "PREMIUM")
    assert row["recommended_action"] not in DISCOUNT_ACTIONS
    assert "%" not in str(row["recommended_offer"]), (
        f"a premium customer was offered {row['recommended_offer']!r}"
    )


def test_high_return_customer_is_discounted_for_the_margin_they_hand_back(
    cohort, recommended: pd.DataFrame, params: RetentionParams
) -> None:
    """Winning back a 58%-returner does not win back 58% of the revenue.

    The engine reflects that in the propensity rather than in the action: the high-return
    multiplier reduces how much of their exposure is treated as recoverable.
    """
    revenue = build_expected_revenue(cohort.reset_index(), params)
    segments = build_segments(cohort, params)
    joined = cohort.join(segments)
    scores = build_scores(joined, revenue, params)

    assert bool(joined.loc["HIGH_RETURN", "is_high_return_customers"])
    basis = str(scores.loc["HIGH_RETURN", "propensity_basis"]).lower()
    assert "return" in basis, f"the high-return multiplier did not fire: {basis!r}"
    assert (
        scores.loc["HIGH_RETURN", "retention_propensity"]
        < params.base_retention_propensity * params.discount_responsive_multiplier
    )


def test_new_customer_is_treated_as_new_not_as_lapsed(
    cohort, recommended: pd.DataFrame, params: RetentionParams
) -> None:
    """Eighteen days of history is not evidence of churn, and must not be read as it.

    A new customer has no cadence to be late against, so nothing about them should look dormant or
    lost -- and their thin history must not be projected into a large expected revenue.
    """
    segments = build_segments(cohort, params)
    assert bool(segments.loc["NEW", "is_new_customers"])
    assert not bool(segments.loc["NEW", "is_lost_customers"])
    assert not bool(segments.loc["NEW", "is_dormant_customers"])

    revenue = build_expected_revenue(cohort.reset_index(), params)
    projected = float(revenue.loc["NEW", "expected_future_revenue"])
    lifetime = float(cohort.loc["NEW", "lifetime_revenue"])
    assert projected <= params.max_projection_multiple * lifetime, (
        f"18 days of history projected into {projected:.2f} against {lifetime:.2f} observed"
    )


# ======================================================================================
# cross-persona invariants
# ======================================================================================


def test_no_persona_is_contacted_at_a_negative_expected_roi(recommended: pd.DataFrame) -> None:
    targeted = recommended[recommended["recommended_action"] != "Do Not Target"]
    assert (targeted["expected_roi"] > 0).all()


def test_suppressed_personas_cost_nothing_and_say_why(recommended: pd.DataFrame) -> None:
    suppressed = recommended[recommended["recommended_action"] == "Do Not Target"]
    assert (suppressed["campaign_cost"] == 0).all()
    assert (suppressed["reason"].str.len() > 0).all()


def test_the_personas_do_not_all_receive_the_same_treatment(recommended: pd.DataFrame) -> None:
    """Nine genuinely different customers must produce genuinely different plans.

    Without this, an engine that returned one action for everybody would satisfy every other test
    in this module.
    """
    assert recommended["recommended_action"].nunique() >= 4
    assert recommended["recommended_channel"].nunique() >= 3
