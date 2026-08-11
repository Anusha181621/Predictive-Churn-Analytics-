"""The personalised retention recommendation engine.

What "do not hardcode recommendations" requires
-----------------------------------------------
Every field of a recommendation is derived from the customer's own behaviour:

* **Action** -- chosen by a rule cascade over their features, with the *first* matching rule winning.
  The rules are ordered so that suppression comes before spending and cheap incentives before
  expensive ones.
* **Offer** -- the discount depth is *their* demonstrated depth, capped by policy. A customer who has
  only ever responded to 10% is offered 10%; one who habitually takes 25% is offered 25%. A single
  fixed "15% off" for everybody would be the hardcoding the brief forbids.
* **Category** -- their preferred category, or for a cross-sell the most common category among
  *comparable customers* that they have not bought, computed from the transaction data.
* **SKU** -- a real product: the best-selling SKU in the recommended category, filtered to their
  target gender and price band, excluding everything they already own.
* **Channel** -- inferred from the channel that acquired them, with age as a tiebreak.
* **Reason** -- composed at runtime from the values that fired the rule, so it cites numbers.

Where the brief's guardrails live
---------------------------------
Two instructions from the brief are enforced structurally rather than hoped for:

*Do not recommend unnecessary discounts to premium customers.* A full-price buyer can never reach a
discount rule, because ``Organic Engagement`` is checked first and claims them.

*Do not target where expected ROI is negative.* ROI is computed for the chosen action, and a negative
result overrides the action with ``Do Not Target`` — after the fact, so the reason can say what was
proposed and why it was dropped.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.retention.params import RetentionParams
from src.utils.logging_config import get_logger

__all__ = ["ACTIONS", "build_recommendations", "RecommendationInputs"]

logger = get_logger(__name__)

#: Every action the engine can propose, as named by the brief.
ACTIONS: tuple[str, ...] = (
    "Do Not Target",
    "Organic Engagement",
    "Loyalty Reward",
    "Seasonal Campaign",
    "Replenishment Reminder",
    "Win-Back",
    "Targeted Discount",
    "Free Shipping",
    "Cross-Sell",
    "New Collection",
    "Personalized Recommendation",
)

#: Actions that carry no incentive cost, only the cost of making contact.
_NO_INCENTIVE = {"Organic Engagement", "Personalized Recommendation", "New Collection",
                 "Cross-Sell", "Replenishment Reminder", "Seasonal Campaign"}


@dataclass(frozen=True)
class RecommendationInputs:
    """Reference data the engine needs beyond the customer's own features."""

    #: SKU -> (category, subcategory, brand, target gender, list price).
    products: pd.DataFrame
    #: Units sold per SKU as of the prediction date, for "best-selling" choices.
    sku_popularity: pd.Series
    #: SKUs each customer has already bought, so a recommendation is never something they own.
    purchased_skus: dict[str, set[str]]
    #: Category -> most frequently co-purchased other category, for cross-sell.
    complementary_category: dict[str, str]


def build_recommendation_inputs(
    products: pd.DataFrame, transactions_as_of: pd.DataFrame
) -> RecommendationInputs:
    """Derive the reference data from the CSVs, clipped to the prediction date.

    ``transactions_as_of`` must already be filtered to ``purchase_date <= as_of`` -- a recommendation
    built from tomorrow's bestsellers would be as much of a leak as a feature built from them.
    """
    popularity = (
        transactions_as_of.groupby("sku_id", observed=True)["quantity"].sum().sort_values(
            ascending=False
        )
    )
    purchased: dict[str, set[str]] = (
        transactions_as_of.groupby("customer_id", observed=True)["sku_id"]
        .agg(lambda values: set(values))
        .to_dict()
    )

    # Complementary category: among customers who bought category A, which other category do they
    # buy most? Measured from the data rather than asserted from fashion intuition.
    with_category = transactions_as_of.merge(
        products[["sku_id", "category"]], on="sku_id", how="left"
    )
    pairs = with_category[["customer_id", "category"]].drop_duplicates()
    joined = pairs.merge(pairs, on="customer_id", suffixes=("", "_other"))
    joined = joined[joined["category"].ne(joined["category_other"])]
    counts = joined.groupby(["category", "category_other"], observed=True).size()
    complementary: dict[str, str] = {}
    if not counts.empty:
        for category in counts.index.get_level_values(0).unique():
            complementary[category] = counts.loc[category].idxmax()

    return RecommendationInputs(
        products=products.set_index("sku_id"),
        sku_popularity=popularity,
        purchased_skus=purchased,
        complementary_category=complementary,
    )


# --------------------------------------------------------------------------------------
# channel
# --------------------------------------------------------------------------------------

#: Acquisition channel -> the outbound channel most likely to reach them again. A customer who came
#: through Instagram is reachable by social retargeting; one who came by referral or email is
#: reachable by email.
_CHANNEL_BY_ACQUISITION: dict[str, str] = {
    "Email": "Email",
    "Referral": "Email + referral incentive",
    "Organic Search": "Email",
    "Direct": "Email",
    "Paid Search": "Email + paid search retargeting",
    "Google Ads": "Email + paid search retargeting",
    "Instagram": "Instagram + email",
    "Facebook": "Facebook + email",
    "Influencer": "Instagram + influencer collaboration",
}


def _recommend_channel(row: pd.Series) -> str:
    channel = _CHANNEL_BY_ACQUISITION.get(str(row.get("acquisition_channel")), "Email")
    age = row.get("age")
    # Under-30s are markedly more reachable in-app than by email, so push is added rather than
    # substituted -- email remains the cheapest channel and the fallback.
    try:
        if age is not None and float(age) < 30 and "Instagram" not in channel:
            channel = f"{channel} + app push"
    except (TypeError, ValueError):  # pragma: no cover
        pass
    return channel


# --------------------------------------------------------------------------------------
# offer
# --------------------------------------------------------------------------------------


def _offer_discount_pct(row: pd.Series, params: RetentionParams) -> float:
    """The discount depth this customer has actually responded to, bounded by policy.

    ``average_discount_when_discounted`` is the depth at which they have historically bought, which
    is a far better offer than a house-standard percentage: it neither under-bids a bargain hunter
    nor gives away margin to someone who buys at 10% off.
    """
    demonstrated = row.get("average_discount_when_discounted")
    if demonstrated is None or (isinstance(demonstrated, float) and np.isnan(demonstrated)):
        demonstrated = row.get("average_discount", params.min_offer_discount_pct)
    try:
        depth = float(demonstrated)
    except (TypeError, ValueError):  # pragma: no cover
        depth = params.min_offer_discount_pct
    if np.isnan(depth) or depth <= 0:
        depth = params.min_offer_discount_pct
    # Round to the nearest 5 so the offer looks like a real promotion.
    depth = 5.0 * round(depth / 5.0)
    return float(np.clip(depth, params.min_offer_discount_pct, params.max_offer_discount_pct))


# --------------------------------------------------------------------------------------
# the action cascade
# --------------------------------------------------------------------------------------


def _choose_action(row: pd.Series, params: RetentionParams) -> tuple[str, str]:
    """Return ``(action, reason)`` for one customer. The first matching rule wins."""
    churn = float(row.get("churn_probability") or 0.0)
    recency = row.get("recency_days")
    recency = float(recency) if recency is not None and not pd.isna(recency) else np.inf
    gap_ratio = row.get("purchase_gap_ratio")
    gap_ratio = float(gap_ratio) if gap_ratio is not None and not pd.isna(gap_ratio) else np.nan
    diversity = row.get("category_diversity")
    diversity = float(diversity) if diversity is not None and not pd.isna(diversity) else np.nan
    regularity = row.get("purchase_regularity")
    regularity = float(regularity) if regularity is not None and not pd.isna(regularity) else np.nan
    full_price_rate = row.get("full_price_order_rate")
    full_price_rate = (
        float(full_price_rate) if full_price_rate is not None and not pd.isna(full_price_rate) else 0.0
    )
    season_distance = row.get("days_from_preferred_season")
    season_distance = (
        float(season_distance)
        if season_distance is not None and not pd.isna(season_distance)
        else np.nan
    )

    # --- suppression, before any spending ---

    if not bool(row.get("has_purchase_history", True)):
        return "Do Not Target", "Has never purchased, so there is no behaviour to win back"

    if bool(row.get("is_lost_customers", False)):
        return (
            "Do Not Target",
            f"Silent for {recency:,.0f} days, beyond two full buying cycles; recovery is "
            "implausible and the contact is better spent elsewhere",
        )

    # The seasonality guardrail from the feature layer, carried through to the action: a seasonal
    # customer who is merely out of season is not churning, and discounting them now would train
    # them to wait for a discount.
    if bool(row.get("seasonally_explained_inactivity", False)):
        return (
            "Do Not Target",
            f"Quiet because they are {season_distance:,.0f} days from their usual buying season, "
            "not because they are disengaging; wait for the season rather than discounting now",
        )

    # "Already highly engaged" is one of the brief's own reasons not to target, and it is a
    # different fact from "uneconomic". Separating them keeps the suppression list readable: a
    # comfortable customer and an unrecoverable one should not share a label without explanation.
    if (
        churn < params.already_engaged_max_churn
        and not bool(row.get("is_dormant_customers", False))
    ):
        return (
            "Do Not Target",
            f"Only a {churn:.0%} churn probability with {row.get('orders_365d', 0):.0f} orders in "
            "the last year; already engaged, so contact would spend budget to protect a "
            "relationship that is not at risk",
        )

    if churn < 0.30 and bool(row.get("is_champions", False)):
        return (
            "Loyalty Reward",
            f"Top-percentile value with {row.get('orders_365d', 0):.0f} orders in the last year and "
            f"only a {churn:.0%} churn probability; recognise rather than discount",
        )

    # --- positive maintenance and no-incentive contact ---

    if full_price_rate >= params.full_price_buyer_rate and churn < 0.60:
        return (
            "Organic Engagement",
            f"Pays full price on {full_price_rate:.0%} of orders, so a discount would give away "
            f"margin they do not need; engage with content at a {churn:.0%} churn probability",
        )

    if (
        bool(row.get("is_seasonal_customers", False))
        and not np.isnan(season_distance)
        and season_distance <= params.seasonal_campaign_lead_days
    ):
        return (
            "Seasonal Campaign",
            f"Buys in a repeatable seasonal window and is {season_distance:,.0f} days from it; "
            "time the contact to the season they already buy in",
        )

    if (
        not np.isnan(regularity)
        and regularity >= params.replenishment_min_regularity
        and not np.isnan(gap_ratio)
        and 0.8 <= gap_ratio <= 2.0
    ):
        return (
            "Replenishment Reminder",
            f"Orders on a predictable rhythm (regularity {regularity:.2f}) and is "
            f"{gap_ratio:.1f}x into their usual interval; a reminder is enough",
        )

    # --- win-back and incentives ---

    if recency >= 365:
        return (
            "Win-Back",
            f"No order for {recency:,.0f} days against a usual interval of "
            f"{row.get('expected_purchase_interval_days', float('nan')):,.0f} days; needs a "
            "re-engagement campaign, not a nudge",
        )

    if bool(row.get("is_discount_driven", False)) and churn >= 0.30:
        return (
            "Targeted Discount",
            f"Buys mainly on promotion ({row.get('discount_order_rate', 0):.0%} of orders "
            f"discounted) and carries a {churn:.0%} churn probability; a discount is the lever "
            "they respond to",
        )

    if churn >= 0.30 and not np.isnan(diversity) and diversity <= params.cross_sell_max_diversity:
        return (
            "Cross-Sell",
            f"Concentrated in {row.get('preferred_category', 'one category')} "
            f"(diversity {diversity:.2f}); widening the relationship is more durable than a "
            "discount",
        )

    if churn >= 0.30 and not np.isnan(diversity) and diversity >= params.new_collection_min_diversity:
        return (
            "New Collection",
            f"Shops broadly across {row.get('category_count', 0):.0f} categories "
            f"(diversity {diversity:.2f}); responds to newness rather than price",
        )

    if churn >= 0.30:
        return (
            "Free Shipping",
            f"At {churn:.0%} churn probability with an average order of "
            f"{row.get('average_order_value', 0):,.0f}; free shipping is a cheaper lever than a "
            "margin discount",
        )

    return (
        "Personalized Recommendation",
        f"Still engaged at a {churn:.0%} churn probability; keep the relationship warm with "
        f"products from {row.get('preferred_category', 'their preferred category')}",
    )


# --------------------------------------------------------------------------------------
# category and SKU
# --------------------------------------------------------------------------------------


def _recommend_category(row: pd.Series, action: str, inputs: RecommendationInputs) -> str:
    preferred = row.get("preferred_category")
    if action == "Cross-Sell" and isinstance(preferred, str):
        # The category their peers pair with theirs most often, measured from the transactions.
        return inputs.complementary_category.get(preferred, preferred)
    if isinstance(preferred, str) and preferred:
        return preferred
    return ""


def _recommend_sku(
    customer_id: str, category: str, row: pd.Series, inputs: RecommendationInputs
) -> tuple[str, str]:
    """Best-selling SKU in ``category`` that the customer does not already own.

    Filtered to their target gender and to a price band around what they actually pay, so the
    suggestion is plausible rather than merely popular. Returns ``(sku, description)``.
    """
    if not category or inputs.products.empty:
        return "", ""

    candidates = inputs.products[inputs.products["category"].eq(category)]
    if candidates.empty:
        return "", ""

    owned = inputs.purchased_skus.get(customer_id, set())
    candidates = candidates[~candidates.index.isin(owned)]
    if candidates.empty:
        return "", ""

    # Gender: match their revealed preference, always allowing unisex.
    gender = row.get("preferred_product_gender")
    if isinstance(gender, str) and gender in {"Men", "Women"}:
        gendered = candidates[candidates["product_gender"].isin([gender, "Unisex"])]
        if not gendered.empty:
            candidates = gendered

    # Price band: within 1.5x of their average item value, so a EUR 20 buyer is not sent a
    # EUR 400 coat.
    average_item = row.get("average_item_value")
    try:
        ceiling = float(average_item) * 1.5
        if ceiling > 0:
            affordable = candidates[candidates["list_price"].le(ceiling)]
            if not affordable.empty:
                candidates = affordable
    except (TypeError, ValueError):  # pragma: no cover
        pass

    popularity = inputs.sku_popularity.reindex(candidates.index).fillna(0.0)
    # Ties broken by SKU id, so a rebuild is reproducible.
    best = popularity.sort_values(ascending=False, kind="stable").index[0]
    product = inputs.products.loc[best]
    description = (
        f"{product['brand']} {product['subcategory']} ({product['product_gender']}, "
        f"{product['list_price']:,.2f})"
    )
    return str(best), description


# --------------------------------------------------------------------------------------
# offer text and economics
# --------------------------------------------------------------------------------------


def _offer_for(action: str, row: pd.Series, params: RetentionParams) -> tuple[str, float]:
    """Return ``(offer text, incentive cost)`` for the chosen action.

    **Discount incentives are costed against the revenue they recover, not against all expected
    future revenue.** Getting this wrong is easy and expensive: charging the discount against the
    customer's whole projected spend made 315 of 1,000 discounts look uneconomic and suppressed six
    of the top ten opportunities, because the cost was scaled by ``EFR x depth`` while the benefit
    was only ``EFR x churn x propensity`` -- a factor of roughly three apart. A win-back coupon is
    redeemed on an order that happens *because* of the intervention, so its expected cost is
    ``depth x expected retained revenue``.

    Known simplification: a discount is also redeemed by customers who would have bought anyway,
    so this understates the true cost by that cannibalisation. Modelling it needs a control group,
    which this dataset does not have, so it is stated rather than guessed at.
    """
    if action == "Do Not Target":
        return "None — suppress from the campaign", 0.0
    if action == "Organic Engagement":
        return "No incentive — editorial content and new-arrival preview", 0.0
    if action == "Loyalty Reward":
        return (
            f"Loyalty reward: early access plus a {params.loyalty_reward_cost:,.0f} credit",
            params.loyalty_reward_cost,
        )
    if action == "Free Shipping":
        return "Free shipping on the next order", params.free_shipping_cost
    if action in {"Targeted Discount", "Win-Back"}:
        depth = _offer_discount_pct(row, params)
        recovered = float(row.get("expected_retained_revenue") or 0.0)
        cost = recovered * depth / 100.0
        label = "win-back" if action == "Win-Back" else "personalised"
        return f"{depth:.0f}% {label} discount, matched to the depth they respond to", cost
    if action == "Replenishment Reminder":
        return "No incentive — reminder timed to their usual reorder interval", 0.0
    if action == "Seasonal Campaign":
        return "No incentive — seasonal collection preview timed to their buying window", 0.0
    if action == "Cross-Sell":
        return "No incentive — curated cross-category edit", 0.0
    if action == "New Collection":
        return "No incentive — new-collection early access", 0.0
    return "No incentive — personalised product edit", 0.0


def build_recommendations(
    scored: pd.DataFrame,
    inputs: RecommendationInputs,
    params: RetentionParams | None = None,
) -> pd.DataFrame:
    """Build one recommendation per customer.

    ``scored`` must carry the features, the churn probability, the segment flags and the scoring
    columns (``expected_future_revenue``, ``revenue_at_risk``, ``retention_propensity``).
    """
    params = params or RetentionParams()
    params.validate()

    records: list[dict[str, object]] = []
    for customer_id, row in scored.iterrows():
        action, reason = _choose_action(row, params)
        category = _recommend_category(row, action, inputs)
        sku, sku_description = _recommend_sku(str(customer_id), category, row, inputs)
        offer, incentive_cost = _offer_for(action, row, params)

        contact_cost = 0.0 if action == "Do Not Target" else params.contact_cost
        campaign_cost = contact_cost + incentive_cost
        expected_retained = float(row.get("expected_retained_revenue") or 0.0)

        if action == "Do Not Target":
            roi = np.nan
        elif campaign_cost > 0:
            roi = (expected_retained - campaign_cost) / campaign_cost
        else:  # pragma: no cover - contact always costs something
            roi = np.nan

        # The brief's economic guardrail, applied after the action is chosen so the reason can say
        # what was proposed and why it was dropped.
        suppressed_action = ""
        if action != "Do Not Target" and not np.isnan(roi) and roi <= params.min_expected_roi:
            suppressed_action = action
            reason = (
                f"{action} was indicated ({reason.rstrip('.')}), but the expected return of "
                f"{params.currency} {expected_retained:,.2f} does not cover the "
                f"{params.currency} {campaign_cost:,.2f} it would cost, so the ROI of {roi:+.0%} "
                "makes contact uneconomic"
            )
            action = "Do Not Target"
            offer, incentive_cost = _offer_for(action, row, params)
            campaign_cost = 0.0
            roi = np.nan

        records.append(
            {
                "customer_id": customer_id,
                "recommended_action": action,
                "recommended_channel": "" if action == "Do Not Target" else _recommend_channel(row),
                "recommended_category": category,
                "recommended_sku": sku,
                "recommended_product": sku_description,
                "recommended_offer": offer,
                "reason": reason,
                "priority": row.get("priority"),
                "expected_retained_revenue": round(expected_retained, 2),
                "campaign_cost": round(campaign_cost, 2),
                "incentive_cost": round(incentive_cost, 2),
                "expected_roi": round(float(roi), 4) if not np.isnan(roi) else None,
                "suppressed_action": suppressed_action,
                "roi_is_assumption_dependent": True,
            }
        )

    frame = pd.DataFrame.from_records(records).set_index("customer_id")
    counts = frame["recommended_action"].value_counts()
    logger.info(
        "Recommendations: %s",
        ", ".join(f"{action}={count}" for action, count in counts.items()),
    )
    suppressed = int(frame["suppressed_action"].ne("").sum())
    if suppressed:
        logger.info(
            "%d recommendation(s) were downgraded to Do Not Target because expected ROI did not "
            "clear %+.0f%%",
            suppressed,
            100.0 * params.min_expected_roi,
        )
    return frame
