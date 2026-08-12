"""The full CSV-to-recommendation chain, exercised against the real data and the trained model.

The other suites test units in isolation against small synthetic fixtures, which is what makes
them fast and precise. This one runs the actual pipeline modules over the shipped CSVs, because
some properties only exist at the seams: that scoring produces probabilities and not scores, that
SHAP contributions line up with the columns the model was trained on, and that the retention
layer's economics reconcile with the recommendations it emitted.

Everything here needs a trained model in ``models/``, which is git-ignored, so a fresh clone skips
this module rather than failing. Run ``python scripts/train_model.py`` to enable it. The stages are
module-scoped fixtures so the chain is built once rather than once per assertion -- scoring,
explaining and recommending each take several seconds.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.config.settings import get_settings
from src.data.csv_loader import load_all
from src.explainability.pipeline import explain_churn
from src.explainability.shap_values import compute_shap_values
from src.models.evaluate import evaluate_predictions
from src.models.predict import score_customers
from src.models.registry import FeatureMismatchError, load_model
from src.models.risk import RISK_LEVELS
from src.retention.params import RetentionParams
from src.retention.pipeline import build_retention_layer

AS_OF = "2025-12-31"


def _model_or_skip():
    settings = get_settings()
    if not (settings.models_path / "churn_model.joblib").exists():
        pytest.skip("no trained model; run `python scripts/train_model.py` first")
    return load_model(settings.models_path)


@pytest.fixture(scope="module")
def model():
    return _model_or_skip()


@pytest.fixture(scope="module")
def data():
    return load_all()


@pytest.fixture(scope="module")
def predictions(data, model) -> pd.DataFrame:
    return score_customers(data, as_of_date=AS_OF, model=model)


@pytest.fixture(scope="module")
def explained(data, model, predictions):
    return explain_churn(data, as_of_date=AS_OF, model=model, predictions=predictions)


@pytest.fixture(scope="module")
def retention(data, model, predictions):
    return build_retention_layer(
        data,
        as_of_date=AS_OF,
        model=model,
        params=RetentionParams(revenue_horizon_days=model.metadata.horizon_days),
        predictions=predictions,
    )


# ======================================================================================
# ML: probability range, risk categories, reproducibility, metrics
# ======================================================================================


def test_every_score_is_a_probability(predictions: pd.DataFrame) -> None:
    """Not a decision-function score: revenue at risk multiplies by this, so it must be a rate."""
    probability = predictions["Churn probability"]
    assert probability.notna().all(), "a customer was scored NaN"
    assert probability.between(0.0, 1.0).all(), (
        f"probabilities outside [0, 1]: min {probability.min()}, max {probability.max()}"
    )


def test_every_customer_is_scored_exactly_once(data, predictions: pd.DataFrame) -> None:
    assert len(predictions) == len(data.customers)
    assert predictions["Customer ID"].is_unique
    assert set(predictions["Customer ID"]) == set(data.customers["customer_id"])


def test_risk_levels_are_drawn_from_the_configured_bands(predictions: pd.DataFrame) -> None:
    assert set(predictions["Risk level"]).issubset(set(RISK_LEVELS))


def test_risk_level_agrees_with_the_probability_that_produced_it(
    predictions: pd.DataFrame,
) -> None:
    """The band is a function of the probability, so it cannot disagree with it."""
    settings = get_settings()
    edges = {
        "Low": (0.0, settings.risk_threshold_medium),
        "Medium": (settings.risk_threshold_medium, settings.risk_threshold_high),
        "High": (settings.risk_threshold_high, settings.risk_threshold_critical),
        "Critical": (settings.risk_threshold_critical, 1.0 + 1e-9),
    }
    for level, (low, high) in edges.items():
        band = predictions.loc[predictions["Risk level"] == level, "Churn probability"]
        if band.empty:
            continue
        assert band.min() >= low, f"{level} contains a probability below {low}"
        assert band.max() < high, f"{level} contains a probability at or above {high}"


def test_scoring_is_reproducible(data, model, predictions: pd.DataFrame) -> None:
    """Same data, same model, same answer -- to the bit.

    Reproducibility is what lets two people compare a number, and what lets a rerun be a check
    rather than a new opinion.
    """
    again = score_customers(data, as_of_date=AS_OF, model=model)
    pd.testing.assert_frame_equal(
        predictions.reset_index(drop=True), again.reset_index(drop=True)
    )


def test_reloading_the_model_from_disk_gives_the_same_scores(data, predictions) -> None:
    """Persistence round-trips exactly: the serialised model is the model."""
    reloaded = load_model(get_settings().models_path)
    again = score_customers(data, as_of_date=AS_OF, model=reloaded)
    np.testing.assert_allclose(
        again["Churn probability"].to_numpy(),
        predictions["Churn probability"].to_numpy(),
        rtol=0,
        atol=0,
    )


def test_the_model_records_the_seed_it_was_trained_with(model) -> None:
    assert model.metadata.random_seed is not None


def test_scoring_refuses_features_the_model_was_not_trained_on(model, data) -> None:
    """The feature contract fails loudly rather than scoring a misaligned frame."""
    frame = pd.DataFrame({"not_a_real_feature": [1.0, 2.0]})
    with pytest.raises(FeatureMismatchError):
        model.predict_proba(frame)


def test_evaluation_metrics_are_correct_on_a_known_case() -> None:
    """A perfect ranking scores 1.0; an inverted one scores 0.0.

    Pinning the metric implementation against hand-checkable inputs, so a silently swapped
    argument order or a probability/label mix-up cannot pass.
    """
    labels = np.array([0, 0, 1, 1])
    perfect = np.array([0.1, 0.2, 0.8, 0.9])

    good = evaluate_predictions(labels, perfect, model_name="t", dataset="d")
    assert good.metrics["roc_auc"] == pytest.approx(1.0)
    assert good.metrics["pr_auc"] == pytest.approx(1.0)
    assert good.metrics["precision"] == pytest.approx(1.0)
    assert good.metrics["recall"] == pytest.approx(1.0)
    assert good.metrics["f1"] == pytest.approx(1.0)
    assert good.confusion["tp"] == 2 and good.confusion["tn"] == 2
    assert good.confusion["fp"] == 0 and good.confusion["fn"] == 0

    inverted = evaluate_predictions(labels, 1 - perfect, model_name="t", dataset="d")
    assert inverted.metrics["roc_auc"] == pytest.approx(0.0)

    # Brier is the mean squared error of the probability, so a confidently wrong call is 1.0.
    certain_and_wrong = evaluate_predictions(
        np.array([1, 1]), np.array([0.0, 0.0]), model_name="t", dataset="d"
    )
    assert certain_and_wrong.metrics["brier"] == pytest.approx(1.0)


def test_calibration_bias_is_predicted_minus_observed() -> None:
    labels = np.array([0, 0, 0, 1])
    probabilities = np.array([0.5, 0.5, 0.5, 0.5])
    result = evaluate_predictions(labels, probabilities, model_name="t", dataset="d")
    assert result.metrics["mean_predicted"] == pytest.approx(0.5)
    assert result.metrics["mean_observed"] == pytest.approx(0.25)
    assert result.metrics["calibration_bias"] == pytest.approx(0.25)


# ======================================================================================
# Explainability: the real SHAP path against the calibrated model
# ======================================================================================


def test_shap_runs_against_the_real_calibrated_model(data, model) -> None:
    """TreeSHAP has to reach the trees through the calibration wrapper to run at all.

    The unit tests cover the folding and direction logic on a hand-built result. This covers the
    part they cannot: that the wrapper is actually unwrappable and the explainer accepts what
    comes out.
    """
    from src.features import build_customer_features

    features = build_customer_features(data, as_of_date=AS_OF).features
    sample = features.head(40)

    result = compute_shap_values(model, sample)

    assert len(result.contributions) == len(sample)
    assert not result.contributions.isna().any().any(), "SHAP produced NaN contributions"

    # Contributions are folded back onto the source features, so nothing one-hot survives: every
    # output column is a column the model was trained on, not an expanded `category_Footwear`.
    trained = set(model.metadata.feature_columns)
    assert set(result.contributions.columns).issubset(trained), (
        "an expanded one-hot column leaked into the folded SHAP output: "
        f"{sorted(set(result.contributions.columns) - trained)[:5]}"
    )
    assert list(result.contributions.index) == list(sample["customer_id"])


def test_shap_contributions_are_not_all_zero(data, model) -> None:
    """A model that explains nothing would produce a table of zeros and look fine otherwise."""
    from src.features import build_customer_features

    sample = build_customer_features(data, as_of_date=AS_OF).features.head(40)
    result = compute_shap_values(model, sample)
    assert float(result.contributions.abs().to_numpy().sum()) > 0


def test_folding_preserves_the_total_contribution(data, model) -> None:
    """SHAP values are additive, which is what makes folding one-hot columns legitimate.

    Summing the expanded contributions and the folded ones must give the same total per customer;
    if it did not, the aggregation would be inventing or destroying explanation.
    """
    from src.features import build_customer_features

    sample = build_customer_features(data, as_of_date=AS_OF).features.head(25)
    result = compute_shap_values(model, sample)
    np.testing.assert_allclose(
        result.contributions.sum(axis=1).to_numpy(),
        result.expanded.sum(axis=1).to_numpy(),
        rtol=1e-9,
        atol=1e-9,
    )


def test_explanations_cover_every_customer_with_the_briefs_columns(explained) -> None:
    frame = explained.explanations
    required = [
        "Customer ID",
        "Churn probability",
        "Risk level",
        "Driver rank",
        "Feature",
        "Feature value",
        "Contribution",
        "Direction",
        "Human-readable explanation",
    ]
    assert list(frame.columns[: len(required)]) == required
    assert frame["Customer ID"].nunique() == 1000
    assert set(frame["Driver rank"]) == {1, 2, 3, 4, 5}


def test_no_explanation_sentence_is_reused_wholesale(explained) -> None:
    """Explanations are composed per customer, not selected from a canned list."""
    sentences = explained.explanations["Human-readable explanation"]
    assert sentences.notna().all() and (sentences.str.len() > 0).all()
    # With 5,000 driver rows a fixed template set would collapse to a handful of strings.
    assert sentences.nunique() > 500, f"only {sentences.nunique()} distinct sentences"


def test_direction_always_matches_the_sign_of_the_contribution(explained) -> None:
    frame = explained.explanations
    positive = frame[frame["Contribution"] > 0]["Direction"].str.lower()
    negative = frame[frame["Contribution"] < 0]["Direction"].str.lower()
    assert positive.str.contains("increase|raise", regex=True).all()
    assert negative.str.contains("decrease|reduce|lower", regex=True).all()


# ======================================================================================
# Financial calculations, end to end
# ======================================================================================


def test_revenue_at_risk_is_probability_times_expected_future_revenue(retention) -> None:
    scores = retention.scores
    expected = scores["Churn probability"] * scores["Expected future revenue"]
    np.testing.assert_allclose(
        scores["Revenue at risk"].to_numpy(), expected.to_numpy(), rtol=1e-6, atol=0.02
    )


def test_expected_retained_revenue_is_revenue_at_risk_times_propensity(retention) -> None:
    scores = retention.scores
    expected = scores["Revenue at risk"] * scores["Retention propensity (ASSUMED)"]
    np.testing.assert_allclose(
        scores["Expected retained revenue"].to_numpy(),
        expected.to_numpy(),
        rtol=1e-6,
        atol=0.02,
    )


def test_no_projection_exceeds_twice_observed_lifetime_revenue(retention) -> None:
    """The guard against extrapolating a short history into a large number."""
    scores = retention.scores
    established = scores[scores["Lifetime revenue"] > 0]
    ratio = established["Expected future revenue"] / established["Lifetime revenue"]
    assert ratio.max() <= 2.0 + 1e-9


def test_campaign_cost_is_charged_only_to_contacted_customers(retention) -> None:
    recommendations = retention.recommendations
    suppressed = recommendations[recommendations["Recommended action"] == "Do Not Target"]
    assert (suppressed["Campaign cost"].fillna(0) == 0).all()

    targeted = recommendations[recommendations["Recommended action"] != "Do Not Target"]
    assert (targeted["Campaign cost"] > 0).all(), "a contacted customer costs something"


def test_roi_is_the_return_over_the_cost_that_produced_it(retention) -> None:
    """``ROI = (expected retained revenue - campaign cost) / campaign cost``.

    Checked to a hundredth rather than exactly. Both inputs are published rounded to the cent, and
    the cheapest campaign costs EUR 2.00, so half a cent of rounding on the numerator moves the
    recomputed ratio by up to ~0.006. Demanding more precision than the published columns carry
    would be testing the rounding, not the formula.
    """
    targeted = retention.recommendations[
        retention.recommendations["Recommended action"] != "Do Not Target"
    ]
    computed = (
        targeted["Expected retained revenue"] - targeted["Campaign cost"]
    ) / targeted["Campaign cost"]
    np.testing.assert_allclose(
        targeted["Expected ROI"].to_numpy(), computed.to_numpy(), rtol=0, atol=0.01
    )


def test_the_two_artefacts_agree_on_expected_retained_revenue(retention) -> None:
    """The scores and recommendations files must not disagree about the same quantity."""
    merged = retention.recommendations.merge(
        retention.scores[["Customer ID", "Expected retained revenue"]],
        on="Customer ID",
        suffixes=("_rec", "_score"),
        validate="1:1",
    )
    np.testing.assert_allclose(
        merged["Expected retained revenue_rec"].to_numpy(),
        merged["Expected retained revenue_score"].to_numpy(),
        rtol=0,
        atol=1e-9,
    )


def test_nobody_is_contacted_at_a_loss(retention) -> None:
    targeted = retention.recommendations[
        retention.recommendations["Recommended action"] != "Do Not Target"
    ]
    assert (targeted["Expected ROI"] > 0).all(), "a targeted customer has non-positive ROI"


def test_the_campaign_totals_reconcile_with_the_per_customer_rows(retention) -> None:
    """The headline the dashboard shows is the sum of the rows beneath it."""
    summary = retention.summary()
    targeted = retention.recommendations[
        retention.recommendations["Recommended action"] != "Do Not Target"
    ]
    assert summary["customers_targeted"] == len(targeted)
    assert summary["total_campaign_cost"] == pytest.approx(
        float(targeted["Campaign cost"].sum()), abs=0.01
    )
    blended = (
        summary["campaign_expected_return"] - summary["total_campaign_cost"]
    ) / summary["total_campaign_cost"]
    assert summary["campaign_roi"] == pytest.approx(blended, abs=1e-3)


def test_revenue_at_risk_never_exceeds_the_revenue_it_is_drawn_from(retention) -> None:
    scores = retention.scores
    assert (scores["Revenue at risk"] <= scores["Expected future revenue"] + 0.01).all()
    assert (scores["Revenue at risk"] >= -1e-9).all()


# ======================================================================================
# The whole chain
# ======================================================================================


def test_the_four_csvs_become_a_prioritised_recommendation_for_every_customer(
    data, predictions, explained, retention
) -> None:
    """CSV -> features -> churn -> SHAP -> revenue at risk -> segments -> action, for all 1,000.

    The brief's final requirement, asserted as one chain: every customer that exists in
    ``Customer.csv`` comes out of the far end with a probability, a reason and an action.
    """
    customers = set(data.customers["customer_id"])

    assert set(predictions["Customer ID"]) == customers
    assert set(explained.explanations["Customer ID"]) == customers
    assert set(retention.scores["Customer ID"]) == customers
    assert set(retention.recommendations["Customer ID"]) == customers

    recommendations = retention.recommendations
    assert recommendations["Recommended action"].notna().all()
    assert (recommendations["Reason"].str.len() > 0).all()
    assert retention.scores["Primary segment"].notna().all()
