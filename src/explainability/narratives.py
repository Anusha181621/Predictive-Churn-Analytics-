"""Turning SHAP contributions into sentences a CRM manager can act on.

What "do not hardcode explanations" means here
----------------------------------------------
The brief forbids hardcoded explanations, and it is worth being precise about what that rules out,
because natural language cannot appear from nowhere.

What is forbidden is an explanation whose *content* is fixed: a generic "the model predicts churn",
or a canned top-five list that reads the same for every customer. What is required is that **which**
drivers appear, **in what order**, and **every number in the sentence** all come from that
customer's own SHAP contributions and feature values.

That is what this module does. Each feature contributes a *phrase grammar* -- a template plus a
formatter and, where useful, a companion feature to compare against. The sentence is composed at
runtime from the customer's actual value, their position in the cohort, and the sign of their
contribution, so the same feature produces "Last purchased 12 days ago, more recently than 78% of
customers" for one customer and "Last purchased 419 days ago, longer ago than 96% of customers" for
another. No sentence exists until a customer's data is fed through it.

Two further design points:

* **Every feature can be explained.** A feature with no vocabulary entry falls back to a generic
  composed sentence rather than being dropped or given a placeholder, so the driver list is never
  silently truncated to whatever happens to have nice wording. Vocabulary entries improve phrasing;
  they are not a prerequisite for it.
* **Comparisons are chosen to be meaningful, not merely available.** Where a feature is already
  self-relative (``purchase_gap_ratio`` is a multiple of the customer's own average) the sentence
  says so, because "2.8x their own historical average" is far more actionable than a cohort
  percentile. Where it is an absolute quantity, the cohort percentile supplies the missing context.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

__all__ = [
    "DRIVER_GROUPS",
    "NarrativeBuilder",
    "Phrase",
    "VOCABULARY",
    "driver_group",
    "format_value",
]


# --------------------------------------------------------------------------------------
# formatting
# --------------------------------------------------------------------------------------


def format_value(value: object, kind: str, currency: str = "EUR") -> str:
    """Render a raw feature value the way a business user would expect to read it."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "not available"
    if isinstance(value, (bool, np.bool_)):
        return "yes" if bool(value) else "no"
    if kind == "text":
        return str(value)

    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)

    if kind == "days":
        return f"{number:,.0f} day" + ("" if abs(number) == 1 else "s")
    if kind == "money":
        return f"{currency} {number:,.2f}"
    if kind == "share":            # a 0-1 fraction shown as a percentage
        return f"{number:.1%}"
    if kind == "percent":         # already expressed in percent, e.g. a 20% discount
        return f"{number:,.0f}%"
    if kind == "ratio":
        return f"{number:,.2f}x"
    if kind == "count":
        return f"{number:,.0f}"
    if kind == "months":
        return f"{number:,.0f} month" + ("" if abs(number) == 1 else "s")
    return f"{number:,.2f}"


@dataclass(frozen=True)
class Phrase:
    """A phrase grammar for one feature.

    ``template`` is composed at runtime and may reference ``{value}``, ``{percentile}``,
    ``{cohort_median}``, ``{companion}``, ``{change}`` and any column named in ``context``.
    """

    label: str
    template: str
    kind: str = "number"
    #: A paired feature, exposed to the template as ``{companion}``; the relative change between
    #: the two becomes ``{change}``.
    companion: str | None = None
    #: Extra feature columns exposed to the template by name.
    context: tuple[str, ...] = ()
    #: Formatter for the companion, when it differs from ``kind``.
    companion_kind: str | None = None
    #: Suppress the cohort comparison clause where the template already carries its own context.
    self_relative: bool = False
    #: Alternative template used when the value falls below :attr:`low_threshold`, or when a boolean
    #: feature is false. Without this a template asserts its premise regardless of the value -- a
    #: seasonality template read "buys in a repeatable seasonal window (score 0.00)" for a customer
    #: whose score of zero means precisely the opposite. A sentence that contradicts its own number
    #: is worse than no sentence.
    low_template: str | None = None
    low_threshold: float | None = None


#: Phrase grammars, keyed by feature. Anything absent falls back to a generic composed sentence.
VOCABULARY: dict[str, Phrase] = {
    # --- recency and cadence: the strongest churn signals ---
    "recency_days": Phrase(
        "Days since last purchase", "Last purchased {value} ago", "days"
    ),
    "current_purchase_gap": Phrase(
        "Current purchase gap", "Has now gone {value} without ordering", "days"
    ),
    "purchase_gap_ratio": Phrase(
        "Purchase gap vs own average",
        "Current purchase gap is {value} their own typical interval of "
        "{expected_purchase_interval_days}",
        "ratio",
        context=("expected_purchase_interval_days",),
        self_relative=True,
    ),
    "gap_vs_max_gap_ratio": Phrase(
        "Gap vs longest previous gap",
        "The current silence is {value} the longest gap they have ever returned from",
        "ratio",
        self_relative=True,
    ),
    "expected_purchase_interval_days": Phrase(
        "Expected purchase interval", "Typically orders every {value}", "days"
    ),
    "average_purchase_gap": Phrase(
        "Average purchase gap", "Averages {value} between orders", "days"
    ),
    "median_purchase_gap": Phrase(
        "Median purchase gap", "Typically {value} between orders", "days"
    ),
    "maximum_purchase_gap": Phrase(
        "Longest purchase gap", "Their longest previous gap was {value}", "days"
    ),
    "purchase_gap_std": Phrase(
        "Purchase gap variability",
        "Their order timing varies by {value} around their average",
        "days",
    ),
    "purchase_regularity": Phrase(
        "Purchase regularity", "Order timing regularity scores {value} out of 1", "number"
    ),
    # --- frequency ---
    "total_orders": Phrase("Lifetime orders", "Has placed {value} orders in total", "count"),
    "total_units": Phrase("Lifetime units", "Has bought {value} items in total", "count"),
    "total_lines": Phrase(
        "Lifetime order lines", "Has bought across {value} order lines", "count"
    ),
    "average_units_per_order": Phrase(
        "Items per order", "Buys {value} items per order on average", "number"
    ),
    "average_item_value": Phrase(
        "Average item value", "Pays {value} per item on average", "money"
    ),
    "max_order_value": Phrase("Largest order", "Their largest order was {value}", "money"),
    "min_order_value": Phrase("Smallest order", "Their smallest order was {value}", "money"),
    "observable_months": Phrase(
        "Observable months", "Has been observable for {value}", "months"
    ),
    "brand_diversity": Phrase(
        "Brand diversity", "Spreads spending across brands at {value} of 1", "number",
        self_relative=True,
    ),
    "value_percentile": Phrase(
        "Value percentile", "Sits at the {value} percentile of customer value", "share",
        self_relative=True,
    ),
    "orders_30d": Phrase("Orders, last 30 days", "{value} orders in the last 30 days", "count"),
    "orders_90d": Phrase("Orders, last 90 days", "{value} orders in the last 90 days", "count"),
    "orders_180d": Phrase("Orders, last 180 days", "{value} orders in the last 180 days", "count"),
    "orders_365d": Phrase("Orders, last 12 months", "{value} orders in the last 12 months", "count"),
    "order_frequency_growth": Phrase(
        "Order frequency trend",
        "Order frequency moved by {value} versus the previous 90 days "
        "({orders_recent_window} orders against {orders_previous_window})",
        "share",
        context=("orders_recent_window", "orders_previous_window"),
        self_relative=True,
    ),
    "recent_vs_historical_frequency": Phrase(
        "Recent vs historical frequency",
        "Recent ordering runs at {value} of their own long-run rate",
        "ratio",
        self_relative=True,
    ),
    # --- monetary ---
    "lifetime_revenue": Phrase("Lifetime revenue", "Has spent {value} in total", "money"),
    "revenue_90d": Phrase("Revenue, last 90 days", "Spent {value} in the last 90 days", "money"),
    "revenue_180d": Phrase("Revenue, last 180 days", "Spent {value} in the last 180 days", "money"),
    "revenue_365d": Phrase(
        "Revenue, last 12 months", "Spent {value} in the last 12 months", "money"
    ),
    "average_order_value": Phrase("Average order value", "Averages {value} per order", "money"),
    "revenue_growth": Phrase(
        "Revenue trend",
        "Spending moved by {value} versus the previous 90 days ({revenue_recent_window} "
        "against {revenue_previous_window})",
        "share",
        context=("revenue_recent_window", "revenue_previous_window"),
        self_relative=True,
    ),
    "recent_vs_historical_revenue": Phrase(
        "Recent vs historical revenue",
        "Recent spending runs at {value} of their own long-run rate",
        "ratio",
        self_relative=True,
    ),
    "revenue_share_last_365d": Phrase(
        "Share of revenue in the last year",
        "{value} of their lifetime spend came in the last 12 months",
        "share",
        self_relative=True,
    ),
    "annualized_revenue": Phrase(
        "Annualised revenue", "Spending annualises to {value}", "money"
    ),
    # --- lifecycle ---
    "customer_tenure_days": Phrase(
        "Customer tenure", "Has been a customer for {value}", "days"
    ),
    "active_months": Phrase(
        "Active months", "Ordered in {value} of their observable months", "months"
    ),
    "inactive_months": Phrase(
        "Inactive months", "Went {value} without ordering at all", "months"
    ),
    "active_month_rate": Phrase(
        "Share of months active", "Ordered in {value} of the months since joining", "share",
        self_relative=True,
    ),
    "is_one_time_buyer": Phrase(
        "One-time buyer",
        "Has only ever placed a single order",
        "text",
        low_template="Has ordered more than once",
    ),
    # --- product affinity ---
    "days_since_preferred_category_purchase": Phrase(
        "Days since preferred category",
        "Has not bought from their preferred category ({preferred_category}) for {value}",
        "days",
        context=("preferred_category",),
    ),
    "category_diversity": Phrase(
        "Category diversity",
        "Spreads spending across {category_count} categories (diversity {value} of 1)",
        "number",
        context=("category_count",),
        self_relative=True,
    ),
    "category_count": Phrase(
        "Categories bought", "Has bought from {value} different categories", "count"
    ),
    "subcategory_count": Phrase(
        "Subcategories bought", "Has bought from {value} different subcategories", "count"
    ),
    "brand_count": Phrase("Brands bought", "Has bought {value} different brands", "count"),
    "sku_count": Phrase("Distinct products bought", "Has bought {value} distinct products", "count"),
    "preferred_category_share": Phrase(
        "Concentration in preferred category",
        "{value} of their units come from {preferred_category}",
        "share",
        context=("preferred_category",),
        self_relative=True,
    ),
    # --- discount behaviour ---
    "average_discount": Phrase(
        "Average discount", "Buys at an average discount of {value}", "percent"
    ),
    "max_discount": Phrase("Deepest discount taken", "Deepest discount taken was {value}", "percent"),
    "discount_order_rate": Phrase(
        "Share of orders discounted", "{value} of their orders carried a discount", "share",
        self_relative=True,
    ),
    "coupon_usage_rate": Phrase(
        "Coupon usage rate", "Used a coupon on {value} of their orders", "share", self_relative=True
    ),
    "full_price_order_rate": Phrase(
        "Full-price order rate", "Paid full price on {value} of their orders", "share",
        self_relative=True,
    ),
    "discount_dependency_score": Phrase(
        "Discount dependency", "Discount dependency scores {value} out of 1", "number"
    ),
    "discounted_revenue_share": Phrase(
        "Revenue from discounted orders",
        "{value} of their spend came on discounted orders",
        "share",
        self_relative=True,
    ),
    "revenue_from_discounted_orders": Phrase(
        "Discounted revenue", "{value} of their spend came on discounted orders", "money"
    ),
    "average_discount_when_discounted": Phrase(
        "Discount depth when discounted",
        "When they do use a discount it averages {value}",
        "percent",
    ),
    # --- returns ---
    "return_rate": Phrase("Return rate", "Returns {value} of the units they buy", "share"),
    "recent_return_rate": Phrase(
        "Recent return rate", "Returned {value} of recently bought units", "share"
    ),
    "return_frequency": Phrase(
        "Share of orders returned", "{value} of their orders included a return", "share",
        self_relative=True,
    ),
    "returned_units": Phrase("Units returned", "Has returned {value} units", "count"),
    "days_since_last_return": Phrase(
        "Days since last return", "Last returned something {value} ago", "days"
    ),
    # --- seasonality ---
    "seasonal_customer_score": Phrase(
        "Seasonality score",
        "Buys in a repeatable seasonal window (score {value} of 1); currently "
        "{days_from_preferred_season} from that season",
        "number",
        context=("days_from_preferred_season",),
        self_relative=True,
        # Below the seasonality threshold the premise is false, so the sentence must not assert it.
        low_threshold=0.35,
        low_template="Shows no repeatable seasonal buying pattern (score {value} of 1)",
    ),
    "days_from_preferred_season": Phrase(
        "Distance from buying season", "Currently {value} away from their usual buying season",
        "days",
    ),
    "in_preferred_season": Phrase(
        "In buying season",
        "Is currently inside their usual buying season",
        "text",
        low_template="Is currently outside their usual buying season",
    ),
    "seasonally_explained_inactivity": Phrase(
        "Inactivity explained by season",
        "Their quiet spell is explained by being out of season, not by disengagement",
        "text",
        low_template="Their inactivity is not explained by seasonality",
    ),
    "annual_cycles_missed": Phrase(
        "Annual cycles missed", "Has missed {value} full buying cycles", "count"
    ),
    "seasonal_purchase_concentration": Phrase(
        "Seasonal concentration", "Purchases concentrate at {value} of 1 across the calendar",
        "number",
    ),
    # --- value and identity ---
    "value_percentile": Phrase(
        "Value percentile", "Sits at the {value} percentile of customer value", "share",
        self_relative=True,
    ),
    "customer_value_segment": Phrase("Value segment", "Falls in the {value} band", "text"),
    "behavioral_segment": Phrase("Behavioural segment", "Behaves as a {value}", "text"),
    "lifecycle_stage": Phrase("Lifecycle stage", "Sits at the {value} lifecycle stage", "text"),
    "most_recent_category": Phrase(
        "Most recent category", "Their latest purchase was in {value}", "text"
    ),
    "preferred_category": Phrase("Preferred category", "Buys mostly {value}", "text"),
}


# --------------------------------------------------------------------------------------
# composition
# --------------------------------------------------------------------------------------


@dataclass
class NarrativeBuilder:
    """Composes driver sentences from feature values and cohort context.

    The cohort percentiles are computed once from the scored population, so a sentence can say where
    this customer sits relative to everyone else without recomputing anything per row.
    """

    values: pd.DataFrame
    currency: str = "EUR"
    percentiles: pd.DataFrame = field(init=False, repr=False)

    def __post_init__(self) -> None:
        numeric = self.values.select_dtypes(include="number")
        # rank(pct=True) gives each customer's position in [0, 1] for every numeric feature.
        self.percentiles = numeric.rank(pct=True)

    # --- helpers -------------------------------------------------------------------

    def _formatted(self, customer_id, feature: str, kind: str) -> str:
        if feature not in self.values.columns:
            return "not available"
        return format_value(self.values.at[customer_id, feature], kind, self.currency)

    def _cohort_clause(self, customer_id, feature: str) -> str:
        """"...higher than 92% of customers", or empty when there is no useful comparison."""
        if feature not in self.percentiles.columns:
            return ""
        percentile = self.percentiles.at[customer_id, feature]
        if pd.isna(percentile):
            return ""
        share = float(percentile)
        # Inside the middle of the distribution the comparison adds nothing, so it is left out
        # rather than padding every sentence with "higher than 51% of customers".
        if 0.35 <= share <= 0.65:
            return ""
        if share > 0.65:
            return f", higher than {share:.0%} of customers"
        return f", lower than {1 - share:.0%} of customers"

    def _resolve_template(self, customer_id, feature: str, phrase: Phrase) -> str:
        """Pick the template whose premise matches this customer's value."""
        if phrase.low_template is None or feature not in self.values.columns:
            return phrase.template
        value = self.values.at[customer_id, feature]
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return phrase.template
        if isinstance(value, (bool, np.bool_)):
            return phrase.template if bool(value) else phrase.low_template
        if phrase.low_threshold is not None:
            try:
                return phrase.template if float(value) >= phrase.low_threshold else phrase.low_template
            except (TypeError, ValueError):
                return phrase.template
        # No threshold given: treat it as a 0/1 flag.
        try:
            return phrase.template if float(value) else phrase.low_template
        except (TypeError, ValueError):
            return phrase.template

    def label_for(self, feature: str) -> str:
        phrase = VOCABULARY.get(feature)
        if phrase is not None:
            return phrase.label
        return feature.replace("_", " ").capitalize()

    # --- the sentence ---------------------------------------------------------------

    def sentence(self, customer_id, feature: str, contribution: float) -> str:
        """Compose the explanation for one driver of one customer."""
        phrase = VOCABULARY.get(feature)
        effect = "raising churn risk" if contribution > 0 else "lowering churn risk"

        if phrase is None:
            # Generic composition, so a feature without a hand-written grammar still gets a real,
            # value-bearing sentence instead of a placeholder.
            value = self._formatted(customer_id, feature, "number")
            clause = self._cohort_clause(customer_id, feature)
            return f"{self.label_for(feature)} is {value}{clause} — {effect}"

        template = self._resolve_template(customer_id, feature, phrase)
        substitutions: dict[str, str] = {
            "value": self._formatted(customer_id, feature, phrase.kind),
            "percentile": self._cohort_clause(customer_id, feature).lstrip(", "),
        }
        for extra in phrase.context:
            substitutions[extra] = self._formatted(
                customer_id, extra, VOCABULARY.get(extra, Phrase("", "", "number")).kind
            )
        if phrase.companion:
            substitutions["companion"] = self._formatted(
                customer_id, phrase.companion, phrase.companion_kind or phrase.kind
            )

        try:
            body = template.format(**substitutions)
        except (KeyError, IndexError):  # pragma: no cover - a malformed template
            body = f"{phrase.label} is {substitutions['value']}"

        clause = "" if phrase.self_relative else self._cohort_clause(customer_id, feature)
        return f"{body}{clause} — {effect}"


# --------------------------------------------------------------------------------------
# driver concept groups
# --------------------------------------------------------------------------------------
#
# The feature table deliberately carries several views of the same underlying behaviour --
# `median_purchase_gap` and `expected_purchase_interval_days` are the same number by construction,
# and `orders_90d`/`orders_180d`/`orders_365d` are three windows on one habit. SHAP ranks them
# independently, so an ungrouped top-five list happily spent two slots saying "typically 49 days
# between orders" and "typically orders every 49 days".
#
# Collapsing features onto concepts and keeping only each concept's strongest contributor means five
# slots buy five *distinct* reasons. Anything unmapped becomes its own group, so a new feature is
# never silently merged into an unrelated concept.

DRIVER_GROUPS: dict[str, str] = {}


def _register(group: str, *features: str) -> None:
    for feature in features:
        DRIVER_GROUPS[feature] = group


_register(
    "recency",
    "recency_days", "current_purchase_gap", "purchase_gap_ratio", "gap_vs_max_gap_ratio",
)
_register(
    "cadence",
    "average_purchase_gap", "median_purchase_gap", "maximum_purchase_gap", "minimum_purchase_gap",
    "purchase_gap_std", "purchase_gap_cv", "purchase_regularity",
    "expected_purchase_interval_days", "has_measurable_cadence", "observed_gaps",
)
_register(
    "order volume",
    "total_orders", "total_lines", "total_units", "average_units_per_order",
)
_register(
    "recent activity",
    "orders_30d", "orders_90d", "orders_180d", "orders_365d",
    "revenue_30d", "revenue_90d", "revenue_180d", "revenue_365d",
    "units_30d", "units_90d", "units_180d", "units_365d",
    "orders_recent_window", "orders_previous_window", "orders_recent_90d",
    "orders_first_90d", "revenue_first_90d", "revenue_share_last_365d",
)
_register(
    "spend level",
    "lifetime_revenue", "lifetime_gross_revenue", "average_order_value", "average_item_value",
    "max_order_value", "min_order_value", "annualized_revenue", "revenue_per_order",
    "revenue_per_active_month", "value_percentile", "customer_value_segment",
    "annualisation_floored", "high_value_threshold", "medium_value_threshold",
)
_register(
    "trend",
    "revenue_growth", "order_frequency_growth", "quantity_growth", "aov_growth",
    "spend_decline_pct", "order_frequency_decline_pct", "aov_decline_pct",
    "recent_vs_historical_revenue", "recent_vs_historical_frequency",
    "revenue_recent_window", "revenue_previous_window", "aov_recent_window",
    "aov_previous_window", "early_vs_recent_order_ratio",
)
_register(
    "product breadth",
    "category_count", "subcategory_count", "brand_count", "sku_count",
    "category_diversity", "brand_diversity",
)
_register(
    "category affinity",
    "preferred_category", "preferred_subcategory", "preferred_brand", "preferred_product_gender",
    "most_recent_category", "preferred_category_share", "preferred_category_is_latest",
    "days_since_preferred_category_purchase", "average_list_price", "max_list_price",
)
_register(
    "discount behaviour",
    "average_discount", "max_discount", "discount_order_rate", "coupon_usage_rate",
    "full_price_order_rate", "discount_dependency_score", "discounted_revenue_share",
    "revenue_from_discounted_orders", "average_discount_when_discounted", "discounted_line_rate",
    "is_discount_driven", "is_full_price_buyer",
)
_register(
    "returns",
    "return_rate", "recent_return_rate", "return_frequency", "return_rate_trend",
    "returned_units", "returned_orders", "returned_lines", "returned_units_recent",
    "average_return_quantity", "days_since_last_return", "is_serial_returner",
)
_register(
    "seasonality",
    "seasonal_customer_score", "seasonal_purchase_concentration",
    "quarterly_purchase_concentration", "circular_concentration", "preferred_day_of_year",
    "preferred_purchase_month", "preferred_purchase_quarter", "purchase_years_spanned",
    "days_from_preferred_season", "in_preferred_season", "seasonally_explained_inactivity",
    "missed_full_season", "annual_cycles_missed", "is_seasonal_buyer",
)
_register(
    "lifecycle",
    "customer_tenure_days", "days_since_first_purchase", "days_since_registration",
    "first_to_last_purchase_days", "active_months", "inactive_months", "observable_months",
    "active_month_rate", "is_repeat_customer", "is_one_time_buyer",
    "behavioral_segment", "lifecycle_stage", "is_new_buyer", "is_dormant_buyer",
    "is_declining_buyer", "is_frequent_buyer", "is_occasional_buyer",
    "registered_at_as_of", "has_purchase_history",
)


def driver_group(feature: str) -> str:
    """The concept a feature belongs to. Unmapped features form their own group."""
    return DRIVER_GROUPS.get(feature, feature)
