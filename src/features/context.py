"""The as-of boundary: the one place data is clipped to the prediction date.

This module is the leakage guard for the whole feature layer. It clips transactions and returns
to the as-of date **once**, and every feature module then works exclusively from the resulting
:class:`FeatureContext`. No feature module receives the raw, unclipped frames, so no feature
module can look into the future by accident.

Two clipping rules, both load-bearing:

1. ``purchase_date <= as_of`` -- a transaction that has not happened yet cannot inform a
   prediction made today.
2. ``return_date <= as_of`` -- and this is the subtle one. A return is a *separate later event*
   from its purchase. Filtering only on the purchase date would let a return that has not
   happened yet count against an order that has, which is exactly the leak the shipped data
   invites: 104 of its returns are dated after the last purchase date, the latest a full month
   later. Return features are therefore built only from returns already observed at the as-of
   date, which means they legitimately understate the eventual return rate. That is the point:
   the model may only know what was knowable.

Everything downstream is derived here too -- the product-joined order lines and the order-level
frame -- so the expensive joins happen once per build rather than once per feature module.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime

import numpy as np
import pandas as pd

from src.data.csv_loader import Datasets
from src.features.params import FeatureParams
from src.utils.logging_config import get_logger

__all__ = ["FeatureContext", "build_context", "resolve_as_of_date"]

logger = get_logger(__name__)


def resolve_as_of_date(
    data: Datasets, as_of_date: str | date | datetime | pd.Timestamp | None = None
) -> pd.Timestamp:
    """Resolve the prediction date, defaulting to the last purchase date in the data.

    Deriving the default from the data rather than from the wall clock keeps a build
    reproducible: running it tomorrow gives the same answer.
    """
    if as_of_date is None:
        resolved = pd.Timestamp(data.transactions["purchase_date"].max()).normalize()
        logger.info("as_of_date not supplied; derived %s from the data", resolved.date())
        return resolved
    resolved = pd.Timestamp(as_of_date).normalize()
    if pd.isna(resolved):
        raise ValueError(f"as_of_date {as_of_date!r} could not be parsed as a date")
    return resolved


@dataclass(frozen=True)
class FeatureContext:
    """Everything the feature modules are allowed to see, already clipped to ``as_of``.

    Attributes
    ----------
    as_of:
        The prediction date. No row in this context postdates it.
    customers:
        All customers, unfiltered -- the output must carry one row per Customer ID even for
        customers with no history yet. Use :attr:`registered` to tell them apart.
    lines:
        Order lines with ``purchase_date <= as_of``, joined to product attributes.
    orders:
        One row per order, aggregated from :attr:`lines`.
    returns:
        Returns with ``return_date <= as_of``, restricted to in-window order lines.
    """

    as_of: pd.Timestamp
    params: FeatureParams
    customers: pd.DataFrame
    lines: pd.DataFrame
    orders: pd.DataFrame
    returns: pd.DataFrame
    customer_ids: pd.Index = field(repr=False)

    # --- cohort membership ----------------------------------------------------------

    @property
    def registered(self) -> pd.Series:
        """Whether each customer had registered on or before the as-of date."""
        registration = self.customers.set_index("customer_id")["registration_date"]
        return registration.le(self.as_of).reindex(self.customer_ids, fill_value=False)

    @property
    def has_history(self) -> pd.Series:
        """Whether each customer has at least one order on or before the as-of date."""
        purchasers = set(self.orders["customer_id"])
        return pd.Series(
            [customer in purchasers for customer in self.customer_ids],
            index=self.customer_ids,
            dtype="bool",
        )

    # --- helpers used by the feature modules ----------------------------------------

    def window_start(self, days: int) -> pd.Timestamp:
        """Left edge of a rolling window. Windows are ``(start, as_of]``."""
        return self.as_of - pd.Timedelta(days=days)

    def orders_within(self, days: int) -> pd.DataFrame:
        """Orders in the last ``days`` days, i.e. ``as_of - days < date <= as_of``."""
        return self.orders[self.orders["purchase_date"].gt(self.window_start(days))]

    def orders_between(self, start_days_ago: int, end_days_ago: int) -> pd.DataFrame:
        """Orders in ``(as_of - start_days_ago, as_of - end_days_ago]``.

        Used by the trend features to isolate the window preceding the recent one.
        """
        if start_days_ago <= end_days_ago:
            raise ValueError("start_days_ago must be greater than end_days_ago")
        dates = self.orders["purchase_date"]
        return self.orders[
            dates.gt(self.window_start(start_days_ago)) & dates.le(self.window_start(end_days_ago))
        ]

    def empty_frame(self) -> pd.DataFrame:
        """An empty frame indexed by every customer, for feature modules to populate."""
        return pd.DataFrame(index=self.customer_ids.copy())


def _aggregate_orders(lines: pd.DataFrame) -> pd.DataFrame:
    """Collapse order lines into one row per order.

    Several features are order-grained rather than line-grained -- order counts, average order
    value, inter-purchase gaps, "share of orders that used a coupon" -- and computing them off
    the line grain would silently weight multi-line orders more heavily.
    """
    if lines.empty:
        return pd.DataFrame(
            {
                "order_id": pd.Series(dtype="string"),
                "customer_id": pd.Series(dtype="string"),
                "purchase_date": pd.Series(dtype="datetime64[ns]"),
                "order_value": pd.Series(dtype="float64"),
                "units": pd.Series(dtype="int64"),
                "lines": pd.Series(dtype="int64"),
                "gross_value": pd.Series(dtype="float64"),
                "max_discount_pct": pd.Series(dtype="float64"),
                "mean_discount_pct": pd.Series(dtype="float64"),
                "used_coupon": pd.Series(dtype="bool"),
                "any_discount": pd.Series(dtype="bool"),
                "full_price": pd.Series(dtype="bool"),
            }
        )

    working = lines.assign(
        gross_value=lines["quantity"] * lines["selling_price"],
        coupon_flag=lines["coupon_used"].eq("Yes"),
        discount_flag=lines["discount_pct"].gt(0),
    )
    orders = working.groupby("order_id", observed=True).agg(
        customer_id=("customer_id", "first"),
        purchase_date=("purchase_date", "first"),
        order_value=("net_order_value", "sum"),
        units=("quantity", "sum"),
        lines=("order_id", "size"),
        gross_value=("gross_value", "sum"),
        max_discount_pct=("discount_pct", "max"),
        mean_discount_pct=("discount_pct", "mean"),
        used_coupon=("coupon_flag", "any"),
        any_discount=("discount_flag", "any"),
    )
    orders["full_price"] = ~orders["any_discount"]
    return orders.reset_index()


def build_context(
    data: Datasets,
    as_of_date: str | date | datetime | pd.Timestamp | None = None,
    params: FeatureParams | None = None,
) -> FeatureContext:
    """Clip the four tables to ``as_of_date`` and prepare the shared frames.

    This is the only function that sees the unclipped data. Everything downstream reads the
    returned context.
    """
    params = params or FeatureParams()
    params.validate()
    as_of = resolve_as_of_date(data, as_of_date)

    # --- rule 1: transactions up to and including the as-of date ---
    transactions = data.transactions
    lines = transactions[transactions["purchase_date"].le(as_of)].copy()

    # Product attributes come from a dimension table with no time component, so joining them is
    # not a leak: a SKU's category today was its category then.
    lines = lines.merge(
        data.products[["sku_id", "category", "subcategory", "brand", "product_gender", "list_price"]],
        on="sku_id",
        how="left",
        validate="m:1",
    )
    lines["gross_value"] = lines["quantity"] * lines["selling_price"]

    orders = _aggregate_orders(lines)

    # --- rule 2: returns observed by the as-of date, on in-window order lines ---
    returns = data.returns[data.returns["return_date"].le(as_of)].copy()
    if not returns.empty and not lines.empty:
        in_window = lines[["order_id", "sku_id"]].drop_duplicates()
        returns = returns.merge(in_window, on=["order_id", "sku_id"], how="inner")
    elif not returns.empty:
        returns = returns.iloc[0:0]

    customer_ids = pd.Index(data.customers["customer_id"], name="customer_id")

    dropped_transactions = len(transactions) - len(lines)
    dropped_returns = len(data.returns) - len(returns)
    logger.info(
        "Feature context at %s: %d/%d order lines, %d orders, %d/%d returns "
        "(withheld %d future transaction line(s) and %d future/out-of-window return(s))",
        as_of.date(),
        len(lines),
        len(transactions),
        len(orders),
        len(returns),
        len(data.returns),
        dropped_transactions,
        dropped_returns,
    )

    return FeatureContext(
        as_of=as_of,
        params=params,
        customers=data.customers,
        lines=lines,
        orders=orders,
        returns=returns,
        customer_ids=customer_ids,
    )


# --------------------------------------------------------------------------------------
# shared numeric helpers
# --------------------------------------------------------------------------------------


def safe_divide(
    numerator: pd.Series, denominator: pd.Series | float, fill: float = np.nan
) -> pd.Series:
    """Divide, mapping a zero or missing denominator to ``fill`` instead of raising or inf."""
    if isinstance(denominator, pd.Series):
        safe = denominator.where(denominator.ne(0) & denominator.notna())
    else:
        safe = denominator if denominator else np.nan
    result = numerator / safe
    return result.fillna(fill) if not np.isnan(fill) else result


def growth(recent: pd.Series, previous: pd.Series) -> pd.Series:
    """Fractional change from ``previous`` to ``recent``.

    Returns NaN where there is no baseline to compare against. A customer who spent nothing in
    the prior window has an undefined growth rate, not an infinite one, and the gradient-boosted
    models used later handle NaN natively -- so inventing a number here would be worse than
    admitting ignorance.
    """
    baseline = previous.where(previous.gt(0))
    return (recent - baseline) / baseline
