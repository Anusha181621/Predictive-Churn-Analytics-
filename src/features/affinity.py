"""Product affinity features, from the join onto ``Product.csv``.

Beyond "what do they buy", this module answers "have they stopped buying it", via
``days_since_preferred_category_purchase``. That is the feature behind one of the brief's named
churn drivers -- *the customer has not purchased from their preferred category in 8 months* --
and it is also what lets the recommendation engine in Section 5 suggest the right category
rather than a generic discount.

Ties are broken deterministically (by unit count, then revenue, then name) so a rebuild produces
byte-identical output.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.context import FeatureContext, safe_divide

__all__ = ["build_affinity_features"]


def _preferred(lines: pd.DataFrame, dimension: str, index: pd.Index) -> pd.Series:
    """The customer's top value of ``dimension`` by units, then revenue, then name."""
    if lines.empty:
        return pd.Series(index=index, dtype="object")
    totals = (
        lines.groupby(["customer_id", dimension], observed=True)
        .agg(units=("quantity", "sum"), revenue=("net_order_value", "sum"))
        .reset_index()
    )
    # Ascending name as the final tiebreak keeps the result stable across runs.
    totals = totals.sort_values(
        ["customer_id", "units", "revenue", dimension],
        ascending=[True, False, False, True],
    )
    top = totals.drop_duplicates("customer_id").set_index("customer_id")[dimension]
    return top.reindex(index)


def _normalised_entropy(lines: pd.DataFrame, dimension: str, index: pd.Index) -> pd.Series:
    """Shannon entropy of the customer's spend across ``dimension``, scaled to [0, 1].

    0 means everything came from one value; 1 means spend was spread evenly across every value
    available. Normalising by ``log(k)`` where ``k`` is the number of values *that customer* used
    would make a two-category customer look as diverse as a five-category one, so the divisor is
    instead the number of distinct values observed across all customers at this as-of date.
    """
    if lines.empty:
        return pd.Series(index=index, dtype="float64")

    catalogue_size = lines[dimension].nunique()
    if catalogue_size <= 1:
        return pd.Series(0.0, index=index)

    units = lines.groupby(["customer_id", dimension], observed=True)["quantity"].sum()
    totals = units.groupby(level="customer_id").transform("sum")
    share = units / totals
    entropy = -(share * np.log(share)).groupby(level="customer_id").sum()
    return (entropy / np.log(catalogue_size)).reindex(index)


def build_affinity_features(context: FeatureContext) -> pd.DataFrame:
    """One row per customer: preferred category/subcategory/brand, breadth and diversity."""
    features = context.empty_frame()
    lines = context.lines
    grouped = lines.groupby("customer_id", observed=True) if not lines.empty else None

    # --- preferences ---
    features["preferred_category"] = _preferred(lines, "category", features.index)
    features["preferred_subcategory"] = _preferred(lines, "subcategory", features.index)
    features["preferred_brand"] = _preferred(lines, "brand", features.index)
    features["preferred_product_gender"] = _preferred(lines, "product_gender", features.index)

    # --- breadth ---
    for column, dimension in (
        ("category_count", "category"),
        ("subcategory_count", "subcategory"),
        ("brand_count", "brand"),
        ("sku_count", "sku_id"),
    ):
        features[column] = (
            grouped[dimension].nunique().reindex(features.index, fill_value=0)
            if grouped is not None
            else 0
        )

    # --- concentration ---
    features["category_diversity"] = _normalised_entropy(lines, "category", features.index)
    features["brand_diversity"] = _normalised_entropy(lines, "brand", features.index)

    if not lines.empty:
        units_by_customer = grouped["quantity"].sum()
        preferred_units = (
            lines.merge(
                features["preferred_category"].rename("preferred_category").reset_index(),
                on="customer_id",
                how="left",
            )
            .query("category == preferred_category")
            .groupby("customer_id", observed=True)["quantity"]
            .sum()
        )
        features["preferred_category_share"] = safe_divide(
            preferred_units.reindex(features.index, fill_value=0).astype(float),
            units_by_customer.reindex(features.index).astype(float),
        )
    else:
        features["preferred_category_share"] = np.nan

    # --- most recent category, and whether the preferred one has gone quiet ---
    if not lines.empty:
        latest = lines.sort_values(["customer_id", "purchase_date"]).drop_duplicates(
            "customer_id", keep="last"
        )
        features["most_recent_category"] = (
            latest.set_index("customer_id")["category"].reindex(features.index)
        )

        preferred_lines = lines.merge(
            features["preferred_category"].rename("preferred_category").reset_index(),
            on="customer_id",
            how="left",
        ).query("category == preferred_category")
        last_preferred = preferred_lines.groupby("customer_id", observed=True)[
            "purchase_date"
        ].max()
        features["days_since_preferred_category_purchase"] = (
            (context.as_of - last_preferred).dt.days.reindex(features.index)
        )
        # Has the customer drifted away from what they used to buy? True when their latest
        # basket came from some other category. Whether that drift is *significant* depends on
        # the customer's own cadence, so the comparison against
        # `expected_purchase_interval_days` is left to the explanation layer, which has both.
        features["preferred_category_is_latest"] = (
            features["most_recent_category"].eq(features["preferred_category"])
            & features["most_recent_category"].notna()
        )
    else:
        features["most_recent_category"] = pd.NA
        features["days_since_preferred_category_purchase"] = np.nan
        features["preferred_category_is_latest"] = False

    # --- price tier of what they buy ---
    #
    # Average list price of the SKUs a customer chooses indicates whether they shop the premium
    # end. Section 5 uses this to avoid pushing discounts at customers who buy at full price.
    if not lines.empty:
        features["average_list_price"] = grouped["list_price"].mean().reindex(features.index)
        features["max_list_price"] = grouped["list_price"].max().reindex(features.index)
    else:
        features["average_list_price"] = np.nan
        features["max_list_price"] = np.nan

    # Deliberately NOT rounded: this is the modelling table, and truncating ratios to a
    # few decimals would discard real signal. Presentation rounding happens at CSV-write
    # time in scripts/build_features.py. Monetary columns are rounded to cents above,
    # where the rounding is part of the definition rather than cosmetic.
    return features
