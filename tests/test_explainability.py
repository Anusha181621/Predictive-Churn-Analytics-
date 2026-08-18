"""Tests for the SHAP explainability layer.

The properties worth pinning are not "SHAP returns numbers" but the ones that make an explanation
*trustworthy*:

* the contributions are folded back onto the original features, so one-hot expansion does not
  fragment a single real driver into five weak ones;
* a sentence never contradicts the value it quotes;
* the drivers shown are distinct concepts, not the same number restated;
* every number in a sentence comes from that customer's own data, which is what "do not hardcode
  explanations" actually requires.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.explainability.customer_explanations import (
    EXPLANATION_COLUMNS,
    build_customer_explanations,
    explanation_for,
)
from src.explainability.global_explanations import build_global_explanation
from src.explainability.narratives import (
    DRIVER_GROUPS,
    VOCABULARY,
    NarrativeBuilder,
    Phrase,
    driver_group,
    format_value,
)
from src.explainability.shap_values import ShapResult, _map_expanded_to_original, unwrap_pipeline


# --------------------------------------------------------------------------------------
# a hand-built SHAP result, so the expected sentences can be reasoned about exactly
# --------------------------------------------------------------------------------------


@pytest.fixture
def shap_result() -> ShapResult:
    index = pd.Index(["CUST0001", "CUST0002", "CUST0003"], name="customer_id")
    values = pd.DataFrame(
        {
            "recency_days": [10.0, 200.0, 400.0],
            "purchase_gap_ratio": [0.3, 2.5, 8.0],
            "expected_purchase_interval_days": [30.0, 80.0, 50.0],
            "orders_365d": [12.0, 3.0, 0.0],
            "seasonal_customer_score": [0.0, 0.78, np.nan],
            "days_from_preferred_season": [120.0, 2.0, 90.0],
            "return_rate": [0.0, 0.25, 0.60],
            "preferred_category": ["Apparel", "Footwear", "Accessories"],
            "median_purchase_gap": [30.0, 80.0, 50.0],
            "is_one_time_buyer": [False, False, True],
        },
        index=index,
    )
    contributions = pd.DataFrame(
        {
            "recency_days": [-0.90, 0.40, 1.20],
            "purchase_gap_ratio": [-0.50, 0.30, 0.80],
            "expected_purchase_interval_days": [0.05, 0.10, 0.02],
            "orders_365d": [-0.70, 0.05, 0.60],
            "seasonal_customer_score": [0.10, 0.55, 0.01],
            "days_from_preferred_season": [0.02, 0.03, 0.01],
            "return_rate": [-0.01, 0.20, 0.45],
            "preferred_category": [0.03, -0.02, 0.04],
            "median_purchase_gap": [0.04, 0.09, 0.03],
            "is_one_time_buyer": [-0.02, -0.01, 0.35],
        },
        index=index,
    )
    return ShapResult(contributions=contributions, values=values, base_value=-0.9)


@pytest.fixture
def predictions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Customer ID": ["CUST0001", "CUST0002", "CUST0003"],
            "Prediction date": ["2025-12-31"] * 3,
            "Churn probability": [0.05, 0.62, 0.94],
            "Risk level": ["Low", "High", "Critical"],
        }
    )


# ======================================================================================
# one-hot folding
# ======================================================================================


def test_one_hot_columns_fold_back_onto_their_source_feature() -> None:
    expanded = [
        "recency_days",
        "preferred_category_Apparel",
        "preferred_category_Footwear",
        "category_diversity",
    ]
    original = ["recency_days", "preferred_category", "category_diversity"]
    mapping = _map_expanded_to_original(expanded, original)
    assert mapping["preferred_category_Apparel"] == "preferred_category"
    assert mapping["preferred_category_Footwear"] == "preferred_category"
    assert mapping["recency_days"] == "recency_days"


def test_longest_prefix_wins_so_similar_names_do_not_collide() -> None:
    """``category_diversity`` must not be captured by ``category``."""
    expanded = ["category_diversity", "category_Apparel"]
    original = ["category", "category_diversity"]
    mapping = _map_expanded_to_original(expanded, original)
    assert mapping["category_diversity"] == "category_diversity"
    assert mapping["category_Apparel"] == "category"


def test_missing_indicator_columns_fold_onto_the_column_they_describe() -> None:
    """``SimpleImputer(add_indicator=True)`` prefixes rather than suffixes the source column.

    Prefix matching cannot see a prefix that is on the front, so these columns used to map to
    themselves and were then dropped when the folded frame was reindexed onto the original
    features. That silently deleted real explanation mass and broke SHAP additivity -- the folded
    contributions no longer summed to the margin -- without any error. The flag says "we did not
    know this customer's value for that feature", so it belongs to that feature.
    """
    expanded = [
        "purchase_gap_std",
        "gap_vs_max_gap_ratio",
        "missingindicator_purchase_gap_std",
        "missingindicator_gap_vs_max_gap_ratio",
    ]
    original = ["purchase_gap_std", "gap_vs_max_gap_ratio"]
    mapping = _map_expanded_to_original(expanded, original)
    assert mapping["missingindicator_purchase_gap_std"] == "purchase_gap_std"
    assert mapping["missingindicator_gap_vs_max_gap_ratio"] == "gap_vs_max_gap_ratio"
    # Every expanded column must land on a real feature, or the reindex discards it.
    assert set(mapping.values()) <= set(original)


def test_an_unmeasurable_feature_says_so_instead_of_reporting_a_blank() -> None:
    """A one-time buyer has no purchase gap, so no regularity, gap ratio or intensity slope.

    Those features still arrive here as ranked drivers: the model imputes the value and flags the
    gap, and since the missing-indicator contributions are folded back onto their source feature,
    the flag carries real weight. The sentence has to say the absence *is* the signal -- writing
    "order timing regularity is not available" reports a data problem instead of a finding.
    """
    index = pd.Index([f"C{i}" for i in range(4)], name="customer_id")
    values = pd.DataFrame({"purchase_regularity": [0.4, 0.6, 0.8, np.nan]}, index=index)
    builder = NarrativeBuilder(values)

    sentence = builder.sentence("C3", "purchase_regularity", 0.2)
    assert "not available" not in sentence
    assert "cannot be measured" in sentence
    assert "raising churn risk" in sentence

    # A customer who does have the value still gets the ordinary, value-bearing sentence.
    assert "cannot be measured" not in builder.sentence("C0", "purchase_regularity", 0.2)


def test_an_unplaceable_column_still_maps_to_itself() -> None:
    """The fold must not invent a home for a column it genuinely cannot place.

    Mapping it to itself is what lets ``compute_shap_values`` notice it is unplaced and warn,
    rather than folding an unknown transformer's output onto an arbitrary feature.
    """
    mapping = _map_expanded_to_original(["pca_0"], ["recency_days"])
    assert mapping["pca_0"] == "pca_0"


def test_unwrapping_a_non_pipeline_is_a_clear_error() -> None:
    """Better than silently falling back to a slow, approximate explainer."""
    with pytest.raises(TypeError, match="TreeExplainer needs"):
        unwrap_pipeline(object())


# ======================================================================================
# global explanation
# ======================================================================================


def test_global_importance_is_mean_absolute_contribution(shap_result: ShapResult) -> None:
    importance = shap_result.global_importance()
    assert importance.index[0] == "recency_days"      # |-0.9|, |0.4|, |1.2| is the largest mean
    assert importance.is_monotonic_decreasing


def test_direction_is_measured_from_value_versus_contribution(shap_result: ShapResult) -> None:
    """Recency rises with its own contribution, so higher recency must read as raising risk."""
    direction = shap_result.direction()
    assert direction["recency_days"] == pytest.approx(1.0)
    assert direction["orders_365d"] == pytest.approx(-1.0)


def test_direction_is_not_applicable_for_categoricals(shap_result: ShapResult) -> None:
    assert pd.isna(shap_result.direction()["preferred_category"])


def test_global_summary_labels_non_monotone_features_as_mixed() -> None:
    """A feature whose contribution does not track its value must not be given a direction."""
    index = pd.Index(["a", "b", "c", "d"], name="customer_id")
    # U-shaped: extreme values push one way, middle values the other, so no direction exists.
    values = pd.DataFrame({"noisy": [1.0, 2.0, 3.0, 4.0]}, index=index)
    contributions = pd.DataFrame({"noisy": [0.5, -0.5, -0.5, 0.5]}, index=index)
    result = ShapResult(contributions=contributions, values=values, base_value=0.0)
    summary = build_global_explanation(result, NarrativeBuilder(values)).summary
    assert summary.loc[0, "direction"] == "mixed / non-monotone"


def test_global_summary_is_ranked_and_shares_sum_to_one(shap_result: ShapResult) -> None:
    explanation = build_global_explanation(shap_result, NarrativeBuilder(shap_result.values))
    assert list(explanation.summary["rank"]) == list(range(1, len(explanation.summary) + 1))
    assert explanation.summary["importance_share"].sum() == pytest.approx(1.0)


def test_global_markdown_states_the_calibration_caveat(shap_result: ShapResult) -> None:
    """The scale mismatch must be stated, not implied away."""
    markdown = build_global_explanation(
        shap_result, NarrativeBuilder(shap_result.values)
    ).to_markdown()
    assert "probability calibration" in markdown
    assert "uncalibrated log-odds" in markdown


# ======================================================================================
# narratives
# ======================================================================================


def test_sentences_carry_the_customers_own_numbers(shap_result: ShapResult) -> None:
    """The same feature must produce different sentences for different customers."""
    builder = NarrativeBuilder(shap_result.values)
    first = builder.sentence("CUST0001", "recency_days", -0.9)
    third = builder.sentence("CUST0003", "recency_days", 1.2)
    assert "10 days" in first
    assert "400 days" in third
    assert first != third


def test_direction_wording_follows_the_contribution_sign(shap_result: ShapResult) -> None:
    builder = NarrativeBuilder(shap_result.values)
    assert "lowering churn risk" in builder.sentence("CUST0001", "recency_days", -0.9)
    assert "raising churn risk" in builder.sentence("CUST0003", "recency_days", 1.2)


def test_a_self_relative_feature_quotes_the_customers_own_baseline(
    shap_result: ShapResult,
) -> None:
    """"2.5x their own typical interval of 80 days" beats any cohort percentile here."""
    builder = NarrativeBuilder(shap_result.values)
    sentence = builder.sentence("CUST0002", "purchase_gap_ratio", 0.3)
    assert "2.50x" in sentence
    assert "80 days" in sentence


def test_a_sentence_never_contradicts_its_own_value(shap_result: ShapResult) -> None:
    """A seasonality score of 0 must not claim a repeatable seasonal window.

    This was a real defect: the template asserted its premise regardless of the number it quoted.
    """
    builder = NarrativeBuilder(shap_result.values)
    unseasonal = builder.sentence("CUST0001", "seasonal_customer_score", 0.10)
    seasonal = builder.sentence("CUST0002", "seasonal_customer_score", 0.55)
    assert "no repeatable seasonal" in unseasonal
    assert "repeatable seasonal window" in seasonal
    assert "0.00" in unseasonal


def test_boolean_features_choose_the_matching_phrasing(shap_result: ShapResult) -> None:
    builder = NarrativeBuilder(shap_result.values)
    assert "only ever placed a single order" in builder.sentence(
        "CUST0003", "is_one_time_buyer", 0.35
    )
    assert "ordered more than once" in builder.sentence("CUST0001", "is_one_time_buyer", -0.02)


def test_context_features_are_interpolated(shap_result: ShapResult) -> None:
    """The preferred-category sentence must name the actual category."""
    values = shap_result.values.copy()
    values["days_since_preferred_category_purchase"] = [10.0, 200.0, 673.0]
    builder = NarrativeBuilder(values)
    sentence = builder.sentence("CUST0003", "days_since_preferred_category_purchase", 0.5)
    assert "Accessories" in sentence
    assert "673 days" in sentence


def test_a_sentence_never_prints_not_available_inside_its_own_wording() -> None:
    """Feature selection can trim away a column another feature's grammar leans on.

    ``category_diversity``'s template reads "spreads spending across {category_count} categories",
    but the 20-column feature matrix keeps the diversity score and drops the count. Composing the
    template regardless put "spreads spending across not available categories" in front of a CRM
    manager. The generic composition needs nothing but the feature's own value, so it is the right
    fallback -- a weaker sentence, never a broken one.
    """
    index = pd.Index([f"C{i}" for i in range(5)], name="customer_id")
    values = pd.DataFrame({"category_diversity": np.linspace(0.1, 0.9, 5)}, index=index)
    builder = NarrativeBuilder(values)
    sentence = builder.sentence("C4", "category_diversity", -0.3)
    assert "not available" not in sentence
    assert "Category diversity" in sentence
    assert "0.90" in sentence
    assert "lowering churn risk" in sentence

    # With the companion column present, the richer grammar is still the one that gets used.
    values["category_count"] = [1, 2, 3, 4, 5]
    richer = NarrativeBuilder(values).sentence("C4", "category_diversity", -0.3)
    assert "Spreads spending across 5 categories" in richer


def test_a_feature_without_vocabulary_still_gets_a_real_sentence(
    shap_result: ShapResult,
) -> None:
    """No feature is dropped or given a placeholder for want of hand-written wording."""
    values = shap_result.values.copy()
    values["some_brand_new_feature"] = [1.0, 5.0, 9.0]
    builder = NarrativeBuilder(values)
    sentence = builder.sentence("CUST0003", "some_brand_new_feature", 0.4)
    assert "Some brand new feature" in sentence
    assert "9.00" in sentence
    assert "raising churn risk" in sentence


def test_cohort_clause_is_omitted_in_the_middle_of_the_distribution() -> None:
    """Padding every sentence with "higher than 51% of customers" would be noise."""
    index = pd.Index([f"C{i}" for i in range(11)], name="customer_id")
    values = pd.DataFrame({"return_rate": np.linspace(0, 1, 11)}, index=index)
    builder = NarrativeBuilder(values)
    middle = builder.sentence("C5", "return_rate", 0.1)
    extreme = builder.sentence("C10", "return_rate", 0.1)
    assert "of customers" not in middle
    assert "higher than" in extreme


@pytest.mark.parametrize(
    ("value", "kind", "expected"),
    [
        (145.0, "days", "145 days"),
        (1.0, "days", "1 day"),
        (1234.5, "money", "EUR 1,234.50"),
        (0.452, "share", "45.2%"),
        (20.0, "percent", "20%"),
        (2.8, "ratio", "2.80x"),
        (13.0, "count", "13"),
        (True, "text", "yes"),
        (False, "text", "no"),
        (np.nan, "number", "not available"),
        (None, "number", "not available"),
    ],
)
def test_value_formatting(value: object, kind: str, expected: str) -> None:
    assert format_value(value, kind) == expected


def test_vocabulary_templates_only_use_resolvable_placeholders() -> None:
    """A template referencing an unavailable name would silently degrade to a bare label."""
    import string

    for feature, phrase in VOCABULARY.items():
        allowed = {"value", "percentile", "companion", "change", "cohort_median"}
        allowed |= set(phrase.context)
        if phrase.companion:
            allowed.add("companion")
        for template in (phrase.template, phrase.low_template):
            if not template:
                continue
            used = {
                name for _, name, _, _ in string.Formatter().parse(template) if name
            }
            assert used <= allowed, f"{feature}: template uses {used - allowed}"


# ======================================================================================
# driver grouping
# ======================================================================================


def test_equivalent_features_share_a_concept_group() -> None:
    """These two are the same number by construction, so they must not take two slots."""
    assert driver_group("median_purchase_gap") == driver_group("expected_purchase_interval_days")
    assert driver_group("orders_90d") == driver_group("orders_365d")


def test_distinct_concepts_stay_separate() -> None:
    assert driver_group("recency_days") != driver_group("return_rate")
    assert driver_group("category_diversity") != driver_group("discount_dependency_score")


def test_an_unmapped_feature_forms_its_own_group() -> None:
    """So a new feature is never silently merged into an unrelated concept."""
    assert driver_group("a_feature_added_next_week") == "a_feature_added_next_week"
    assert "a_feature_added_next_week" not in DRIVER_GROUPS


def test_grouping_removes_duplicate_concepts_from_the_driver_list(
    shap_result: ShapResult, predictions: pd.DataFrame
) -> None:
    grouped = build_customer_explanations(shap_result, predictions, top_k=5, group_drivers=True)
    ungrouped = build_customer_explanations(shap_result, predictions, top_k=8, group_drivers=False)

    for customer in predictions["Customer ID"]:
        groups = grouped[grouped["Customer ID"].eq(customer)]["Driver group"]
        assert groups.is_unique, f"{customer} has two drivers from one concept"

    # Without grouping, cadence claims two slots for CUST0002 (median gap and expected interval).
    raw = ungrouped[ungrouped["Customer ID"].eq("CUST0002")]["Feature"].tolist()
    assert {"median_purchase_gap", "expected_purchase_interval_days"} <= set(raw)


# ======================================================================================
# per-customer explanation table
# ======================================================================================


def test_required_columns_are_present_and_first(
    shap_result: ShapResult, predictions: pd.DataFrame
) -> None:
    frame = build_customer_explanations(shap_result, predictions, top_k=5)
    assert list(frame.columns)[: len(EXPLANATION_COLUMNS)] == EXPLANATION_COLUMNS


def test_every_customer_gets_exactly_top_k_drivers(
    shap_result: ShapResult, predictions: pd.DataFrame
) -> None:
    for top_k in (3, 4, 5):
        frame = build_customer_explanations(shap_result, predictions, top_k=top_k)
        counts = frame.groupby("Customer ID").size()
        assert (counts == top_k).all()
        assert set(frame["Driver rank"]) == set(range(1, top_k + 1))


def test_drivers_are_ranked_by_absolute_contribution(
    shap_result: ShapResult, predictions: pd.DataFrame
) -> None:
    """Absolute, so the strongest *protective* factor is not hidden."""
    frame = build_customer_explanations(shap_result, predictions, top_k=5)
    safe = frame[frame["Customer ID"].eq("CUST0001")]
    magnitudes = safe["Contribution"].abs().tolist()
    assert magnitudes == sorted(magnitudes, reverse=True)
    # CUST0001's strongest driver is protective; ranking on the signed value would have buried it.
    assert safe.iloc[0]["Direction"] == "reduces risk"
    assert safe.iloc[0]["Feature"] == "recency_days"


def test_direction_column_matches_the_contribution_sign(
    shap_result: ShapResult, predictions: pd.DataFrame
) -> None:
    frame = build_customer_explanations(shap_result, predictions, top_k=5)
    increases = frame["Direction"].eq("increases risk")
    assert (frame.loc[increases, "Contribution"] > 0).all()
    assert (frame.loc[~increases, "Contribution"] <= 0).all()


def test_probability_and_risk_level_come_from_the_predictions(
    shap_result: ShapResult, predictions: pd.DataFrame
) -> None:
    frame = build_customer_explanations(shap_result, predictions, top_k=3)
    critical = frame[frame["Customer ID"].eq("CUST0003")]
    assert (critical["Risk level"] == "Critical").all()
    assert critical["Churn probability"].eq(0.94).all()


def test_risk_level_filter_restricts_the_output(
    shap_result: ShapResult, predictions: pd.DataFrame
) -> None:
    frame = build_customer_explanations(
        shap_result, predictions, top_k=3, risk_levels=("High", "Critical")
    )
    assert set(frame["Customer ID"]) == {"CUST0002", "CUST0003"}


def test_contribution_share_is_relative_to_the_full_contribution(
    shap_result: ShapResult, predictions: pd.DataFrame
) -> None:
    """The denominator must be all features, not just the surviving drivers."""
    frame = build_customer_explanations(shap_result, predictions, top_k=3)
    for customer in predictions["Customer ID"]:
        share = frame[frame["Customer ID"].eq(customer)]["Contribution share"].sum()
        assert 0 < share < 1


def test_top_k_is_validated(shap_result: ShapResult, predictions: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="top_k must be between"):
        build_customer_explanations(shap_result, predictions, top_k=0)


def test_narrative_block_renders_for_one_customer(
    shap_result: ShapResult, predictions: pd.DataFrame
) -> None:
    frame = build_customer_explanations(shap_result, predictions, top_k=5)
    block = explanation_for(frame, "CUST0003")
    assert "Customer: CUST0003" in block
    assert "Churn probability: 94%" in block
    assert "Risk level: Critical" in block
    assert block.count("\n1.") == 1
    assert "5." in block


def test_narrative_block_handles_an_unknown_customer(
    shap_result: ShapResult, predictions: pd.DataFrame
) -> None:
    frame = build_customer_explanations(shap_result, predictions, top_k=3)
    assert "No explanation available" in explanation_for(frame, "CUST9999")


def test_no_explanation_is_generic(shap_result: ShapResult, predictions: pd.DataFrame) -> None:
    """The failure mode the brief forbids: a sentence that says nothing about the customer."""
    frame = build_customer_explanations(shap_result, predictions, top_k=5)
    banned = ["the model predicts", "likely to churn because of the model", "high risk customer"]
    for row in frame.itertuples(index=False):
        feature, value, sentence = row[4], row[5], row[8]
        assert not any(phrase in sentence.lower() for phrase in banned)
        if value in {"yes", "no", "not available"}:
            # A boolean states its meaning in prose ("has only ever placed a single order") rather
            # than quoting "yes", which reads far better. The value-conditional template test
            # covers that both branches are reachable and correct.
            assert len(sentence) > 25, f"{feature}: {sentence!r} is too thin to be informative"
            continue
        # The precise property: every other sentence quotes the value the row itself reports, so it
        # cannot have been written independently of this customer's data.
        assert value in sentence, f"{feature}: {sentence!r} does not quote {value!r}"


def test_explanations_differ_between_customers(
    shap_result: ShapResult, predictions: pd.DataFrame
) -> None:
    frame = build_customer_explanations(shap_result, predictions, top_k=5)
    per_customer = frame.groupby("Customer ID")["Human-readable explanation"].apply(tuple)
    assert per_customer.nunique() == len(per_customer)
