"""Tunable parameters for feature engineering.

Every threshold the feature layer uses lives here rather than being scattered as magic numbers,
so a business user can change the definition of "frequent" or "dormant" in one place and the
churn model in Section 3 can sweep them.

Defaults are grounded in the shipped dataset (median inter-purchase gap, three years of
history); they are not arbitrary round numbers picked for looks.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["FeatureParams"]


@dataclass(frozen=True)
class FeatureParams:
    """Parameters controlling how customer features are derived."""

    # --- rolling windows, in days ---
    #: Windows for the orders_*/revenue_* counters.
    windows: tuple[int, ...] = (30, 90, 180, 365)

    #: The "recent" window used by every trend feature. Trends compare this window against the
    #: immediately preceding window of the same length, so 90 days means "last quarter versus
    #: the quarter before".
    trend_window_days: int = 90

    #: Window for `recent_return_rate`.
    recent_return_window_days: int = 180

    # --- lifecycle ---
    #: A customer whose tenure is below this has too little history to judge; they are labelled
    #: "New Buyer" rather than being called dormant or declining on thin evidence.
    new_customer_days: int = 90

    #: Floor on the denominator of `annualized_revenue`. Without it, a customer who registered
    #: three days ago and spent EUR 200 would show an annualised revenue of EUR 24,000.
    min_tenure_days_for_annualisation: int = 30

    #: Fallback expected inter-purchase interval for customers with fewer than two orders,
    #: where no gap can be measured.
    default_expected_interval_days: int = 90

    # --- the outcome window the model is asked about ---
    #: Length of the forward window the churn label covers. The rate and seasonality features
    #: project a customer's history *onto this window* -- "how many orders should they place in
    #: the next N days", "does their buying season fall inside it" -- so the number has to match
    #: the label horizon or the projection answers a question nobody asked. Callers that change
    #: `LabelParams.horizon_days` must change this with it; `train_churn_model` does so
    #: automatically.
    outcome_horizon_days: int = 180

    # --- latent purchase-rate estimation ---
    #: Empirical-Bayes prior for the per-customer order rate, expressed as a pseudo-observation:
    #: `prior_orders` orders seen over `prior_years` of pseudo-tenure. Shrinkage matters because
    #: the raw rate `orders / tenure_years` is wildly unstable for the short-tenure, low-count
    #: customers who carry most of the prediction error -- one order in three weeks reads as
    #: 17 orders/year. Deliberately a fixed constant rather than a cohort mean: a cohort mean is
    #: recomputed at every as-of date and would fingerprint the snapshot, exactly the failure
    #: `src.models.preprocessing` documents for `high_value_threshold`.
    rate_prior_orders: float = 2.0
    rate_prior_years: float = 0.75

    #: Minimum tenure before an unshrunk annualised rate is reported at all. Below this the raw
    #: figure is arithmetic noise, so `lifetime_orders_per_year` is null and only the shrunk
    #: estimate is offered.
    min_tenure_days_for_rate: int = 60

    # --- intensity decay ---
    #: Width of the buckets the order-intensity trend is regressed over. 90 days is a quarter,
    #: which is long enough that a typical customer has a non-zero count in most buckets and
    #: short enough to resolve a decline within a year of history.
    decay_bucket_days: int = 90

    #: Minimum buckets of history before a decay slope is fitted. A slope through two points is
    #: not evidence of a trend.
    min_buckets_for_decay: int = 3

    # --- behavioural segments ---
    #: Orders in the last 365 days at or above which a customer counts as a frequent buyer.
    frequent_orders_365d: int = 6

    #: A customer is dormant once their current gap exceeds this multiple of their own expected
    #: interval -- a personalised threshold, not one fixed number of days for everybody.
    dormant_gap_multiple: float = 2.0

    #: Absolute recency backstop: beyond this many days a customer is dormant regardless of
    #: their historical cadence.
    dormant_recency_days: int = 365

    #: Revenue growth at or below this (i.e. a 30% fall) marks a declining customer.
    declining_revenue_growth: float = -0.30

    # --- seasonality ---
    #: Minimum orders before a seasonality score is meaningful. With one or two orders any
    #: customer looks perfectly "concentrated" purely by accident.
    min_orders_for_seasonality: int = 3

    #: Seasonality also requires the pattern to repeat across at least this many calendar
    #: years; a single burst in one December is not a season.
    min_years_for_seasonality: int = 2

    #: Bias-corrected circular concentration at or above which a customer is seasonal.
    seasonal_score_threshold: float = 0.35

    #: Half-width, in days, of a customer's buying season around their circular mean purchase
    #: date. Used to decide whether they are currently *in* season.
    season_halfwidth_days: int = 45

    # --- customer value ---
    #: Lifetime-revenue quantiles splitting High / Medium / Low value, computed within the
    #: cohort observed at the as-of date.
    high_value_quantile: float = 0.80
    medium_value_quantile: float = 0.50

    def validate(self) -> None:
        """Raise ``ValueError`` if the parameters are mutually inconsistent."""
        if not self.windows or any(w <= 0 for w in self.windows):
            raise ValueError(f"windows must all be positive, got {self.windows}")
        if self.trend_window_days <= 0:
            raise ValueError("trend_window_days must be positive")
        if not 0 < self.medium_value_quantile < self.high_value_quantile < 1:
            raise ValueError(
                "value quantiles must satisfy 0 < medium < high < 1, got "
                f"medium={self.medium_value_quantile}, high={self.high_value_quantile}"
            )
        if self.min_orders_for_seasonality < 2:
            raise ValueError("min_orders_for_seasonality must be at least 2")
        if not 0 <= self.seasonal_score_threshold <= 1:
            raise ValueError("seasonal_score_threshold must lie in [0, 1]")
        if self.dormant_gap_multiple <= 0:
            raise ValueError("dormant_gap_multiple must be positive")
        if self.declining_revenue_growth >= 0:
            raise ValueError("declining_revenue_growth describes a decline, so it must be < 0")
        if self.outcome_horizon_days <= 0:
            raise ValueError("outcome_horizon_days must be positive")
        if self.rate_prior_orders < 0 or self.rate_prior_years <= 0:
            raise ValueError(
                "the rate prior needs a non-negative pseudo-count over a positive pseudo-tenure, "
                f"got {self.rate_prior_orders} orders over {self.rate_prior_years} years"
            )
        if self.decay_bucket_days <= 0:
            raise ValueError("decay_bucket_days must be positive")
        if self.min_buckets_for_decay < 3:
            raise ValueError("min_buckets_for_decay must be at least 3 to fit a meaningful slope")
