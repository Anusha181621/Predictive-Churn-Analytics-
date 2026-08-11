"""Discount and coupon behaviour features.

The business purpose is asymmetric, and worth stating plainly: these features exist as much to
decide who *not* to discount as who to discount. The brief is explicit -- *do not recommend
unnecessary discounts to premium customers* -- so a customer who reliably pays full price must be
distinguishable from one who only ever buys on sale.

Rates are order-grained, not line-grained: "the share of this customer's orders that used a
coupon" is the business question, and computing it over lines would over-weight big baskets.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.context import FeatureContext, safe_divide

__all__ = ["build_discount_features"]


def build_discount_features(context: FeatureContext) -> pd.DataFrame:
    """One row per customer: discount depth, coupon use, and a dependency score."""
    features = context.empty_frame()
    lines, orders = context.lines, context.orders

    if lines.empty:
        for column in (
            "average_discount", "max_discount", "discount_order_rate", "coupon_usage_rate",
            "full_price_order_rate", "discounted_revenue_share", "discount_dependency_score",
            "average_discount_when_discounted", "discounted_line_rate",
        ):
            features[column] = np.nan
        features["revenue_from_discounted_orders"] = 0.0
        features["is_full_price_buyer"] = False
        features["is_discount_driven"] = False
        return features

    line_group = lines.groupby("customer_id", observed=True)
    order_group = orders.groupby("customer_id", observed=True)

    # --- discount depth ---
    features["average_discount"] = line_group["discount_pct"].mean().reindex(features.index)
    features["max_discount"] = line_group["discount_pct"].max().reindex(features.index)
    features["discounted_line_rate"] = (
        lines.assign(flag=lines["discount_pct"].gt(0))
        .groupby("customer_id", observed=True)["flag"]
        .mean()
        .reindex(features.index)
    )

    # Average depth *given* that a discount was applied. Separates "always buys at 50% off" from
    # "occasionally catches a 5% voucher"; the plain mean conflates the two.
    discounted_lines = lines[lines["discount_pct"].gt(0)]
    features["average_discount_when_discounted"] = (
        discounted_lines.groupby("customer_id", observed=True)["discount_pct"]
        .mean()
        .reindex(features.index)
    )

    # --- order-level rates ---
    features["discount_order_rate"] = (
        order_group["any_discount"].mean().reindex(features.index)
    )
    features["coupon_usage_rate"] = order_group["used_coupon"].mean().reindex(features.index)
    features["full_price_order_rate"] = order_group["full_price"].mean().reindex(features.index)

    # --- revenue exposure ---
    discounted_revenue = (
        orders[orders["any_discount"]]
        .groupby("customer_id", observed=True)["order_value"]
        .sum()
        .reindex(features.index, fill_value=0.0)
    )
    total_revenue = order_group["order_value"].sum().reindex(features.index)
    features["revenue_from_discounted_orders"] = discounted_revenue.round(2)
    features["discounted_revenue_share"] = safe_divide(discounted_revenue, total_revenue)

    # --- dependency score ---
    #
    # A single 0-1 number blending three views, because any one alone is misleading: how often
    # they need a discount, how much of their spend depends on one, and how deep those discounts
    # run (normalised by the 50% maximum in the data). Averaging keeps every component's
    # contribution visible and the result interpretable, unlike a fitted weighting.
    depth = (features["average_discount"] / 50.0).clip(upper=1.0)
    features["discount_dependency_score"] = (
        features["discount_order_rate"].fillna(0.0)
        + features["discounted_revenue_share"].fillna(0.0)
        + depth.fillna(0.0)
    ) / 3.0

    # --- personas ---
    #
    # Deliberately not mutually exclusive, and both are conservative: a customer must be clearly
    # one or the other, so the ambiguous middle is left unlabelled rather than guessed at.
    features["is_full_price_buyer"] = (
        features["full_price_order_rate"].ge(0.7) & features["coupon_usage_rate"].le(0.1)
    ).fillna(False)
    features["is_discount_driven"] = (
        features["discount_order_rate"].ge(0.7) | features["discounted_revenue_share"].ge(0.8)
    ).fillna(False)

    # Deliberately NOT rounded: this is the modelling table, and truncating ratios to a
    # few decimals would discard real signal. Presentation rounding happens at CSV-write
    # time in scripts/build_features.py. Monetary columns are rounded to cents above,
    # where the rounding is part of the definition rather than cosmetic.
    return features
