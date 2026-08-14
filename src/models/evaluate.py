"""Model evaluation: statistical quality, calibration, and business value.

Accuracy is deliberately absent as a headline. With a churn rate near a half it is nearly
uninformative, and at any other rate it rewards predicting the majority class. The metrics that
matter here fall into three groups.

**Ranking** -- ROC-AUC and PR-AUC. PR-AUC is the primary selection metric because the operational
question is "who are the top few hundred customers worth contacting", which is a precision question
on the positive class; ROC-AUC weights performance on the negatives that nobody will act on.

**Threshold quality** -- precision, recall, F1 and the confusion matrix at the operating threshold,
plus precision/recall at the top decile, since a retention team works a ranked list of fixed length
rather than a probability cut-off.

**Calibration** -- Brier score, expected calibration error and a reliability curve. This is not
academic. Revenue at risk is ``churn probability x expected future revenue``, so a model that
ranks perfectly but reports 0.9 where the truth is 0.5 will overstate the money at stake by
almost double and send the retention budget after a number that does not exist. Ranking metrics are
completely blind to that error.

**Business value** -- lift over random targeting, the share of at-risk revenue captured in the top
decile, and precision among high-value customers. A model that finds cheap churners while missing
expensive ones is worth little regardless of its AUC.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.utils.logging_config import get_logger

__all__ = ["EvaluationResult", "evaluate_predictions", "expected_calibration_error",
           "reliability_table", "top_decile_metrics", "best_accuracy_threshold"]

logger = get_logger(__name__)

#: Default probability cut-off for the confusion-matrix style metrics.
DEFAULT_THRESHOLD = 0.5


def best_accuracy_threshold(
    y_true: pd.Series | np.ndarray, y_prob: pd.Series | np.ndarray
) -> tuple[float, float]:
    """The probability cut-off that maximises accuracy, and the accuracy it achieves.

    Returns ``(threshold, accuracy)``.

    0.5 is the right cut-off only when the cost of a false positive equals that of a false negative
    *and* the probabilities are perfectly calibrated to the population being scored. Neither holds
    here: the churn base rate drifts from 25% to 47% across the timeline, so a model calibrated on
    one period systematically under-calls churn on a later, churnier one -- which is precisely the
    shape of the shipped model's errors, where recall sat at 0.32 against a precision of 0.69.

    Every midpoint between adjacent distinct predicted probabilities is a candidate, since accuracy
    is a step function of the threshold and can only change as the cut-off crosses a predicted
    value. Ties are broken toward 0.5: on a few hundred rows a wide plateau of thresholds is
    genuinely equal-scoring, and picking the middle of that plateau generalises better than picking
    whichever end the sort happened to visit first.

    **This must be fitted on held-out data that is not the test set.** Choosing it on the data it is
    then scored on turns a reported accuracy into an upper bound achievable only in hindsight;
    :func:`src.models.train.train_churn_model` fits it on the calibration period.
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype="float64")
    if len(y_true) == 0:
        return DEFAULT_THRESHOLD, float("nan")

    distinct = np.unique(y_prob)
    if len(distinct) == 1:
        return DEFAULT_THRESHOLD, float(((y_prob >= DEFAULT_THRESHOLD).astype(int) == y_true).mean())

    midpoints = (distinct[:-1] + distinct[1:]) / 2.0
    candidates = np.concatenate(([distinct[0] - 1e-6], midpoints, [distinct[-1] + 1e-6]))
    # (n_candidates, n_rows) is fine at these sizes and avoids a Python loop over the grid.
    accuracies = ((y_prob[None, :] >= candidates[:, None]).astype(int) == y_true[None, :]).mean(axis=1)

    best = accuracies.max()
    plateau = candidates[accuracies >= best - 1e-12]
    chosen = float(plateau[np.argmin(np.abs(plateau - DEFAULT_THRESHOLD))])
    return chosen, float(best)


def expected_calibration_error(
    y_true: np.ndarray, y_prob: np.ndarray, bins: int = 10
) -> float:
    """Weighted mean gap between predicted probability and observed frequency.

    Quantile bins rather than equal-width ones, so every bin carries a comparable number of
    customers; equal-width bins on a skewed probability distribution leave some bins nearly empty
    and their noise then dominates the score.
    """
    if len(y_true) == 0:
        return float("nan")
    frame = pd.DataFrame({"y": y_true, "p": y_prob})
    try:
        frame["bin"] = pd.qcut(frame["p"], q=bins, duplicates="drop")
    except (ValueError, IndexError):  # pragma: no cover - degenerate probability distribution
        return float(np.abs(frame["p"].mean() - frame["y"].mean()))
    grouped = frame.groupby("bin", observed=True).agg(
        predicted=("p", "mean"), observed=("y", "mean"), n=("y", "size")
    )
    return float((grouped["n"] / len(frame) * (grouped["predicted"] - grouped["observed"]).abs()).sum())


def reliability_table(y_true: np.ndarray, y_prob: np.ndarray, bins: int = 10) -> pd.DataFrame:
    """Predicted versus observed churn rate per probability bin -- the calibration curve as data."""
    frame = pd.DataFrame({"y": y_true, "p": y_prob})
    try:
        frame["bin"] = pd.qcut(frame["p"], q=bins, duplicates="drop")
    except (ValueError, IndexError):  # pragma: no cover
        return pd.DataFrame(columns=["predicted", "observed", "n", "gap"])
    grouped = frame.groupby("bin", observed=True).agg(
        predicted=("p", "mean"), observed=("y", "mean"), n=("y", "size")
    )
    grouped["gap"] = grouped["predicted"] - grouped["observed"]
    return grouped.reset_index().assign(bin=lambda f: f["bin"].astype(str))


def top_decile_metrics(
    y_true: np.ndarray, y_prob: np.ndarray, *, fraction: float = 0.10
) -> dict[str, float]:
    """Precision, recall and lift in the highest-scoring ``fraction`` of customers.

    This is how the model will actually be used -- work the top N of a ranked list -- so it is
    measured directly rather than inferred from a probability threshold.
    """
    n = len(y_true)
    if n == 0:
        return {"precision": float("nan"), "recall": float("nan"), "lift": float("nan"), "n": 0}
    k = max(1, int(round(n * fraction)))
    order = np.argsort(-y_prob, kind="stable")[:k]
    selected = y_true[order]
    base_rate = float(y_true.mean())
    precision = float(selected.mean())
    return {
        "precision": precision,
        "recall": float(selected.sum() / y_true.sum()) if y_true.sum() else float("nan"),
        "lift": precision / base_rate if base_rate > 0 else float("nan"),
        "n": int(k),
    }


@dataclass(frozen=True)
class EvaluationResult:
    """Every metric for one model on one dataset."""

    model_name: str
    dataset: str
    n: int
    positives: int
    base_rate: float
    threshold: float
    metrics: dict[str, float]
    confusion: dict[str, int]
    reliability: pd.DataFrame = field(repr=False)
    business: dict[str, float] = field(default_factory=dict)

    @property
    def pr_auc(self) -> float:
        return self.metrics["pr_auc"]

    @property
    def roc_auc(self) -> float:
        return self.metrics["roc_auc"]

    def as_dict(self) -> dict[str, object]:
        return {
            "model": self.model_name,
            "dataset": self.dataset,
            "n": self.n,
            "positives": self.positives,
            "base_rate": round(self.base_rate, 6),
            "threshold": self.threshold,
            "metrics": {k: (round(v, 6) if v == v else None) for k, v in self.metrics.items()},
            "confusion": self.confusion,
            "business": {k: (round(v, 6) if v == v else None) for k, v in self.business.items()},
            "reliability": self.reliability.to_dict(orient="records"),
        }


def _business_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    *,
    value: pd.Series | None,
    high_value: pd.Series | None,
) -> dict[str, float]:
    """Revenue-aware metrics: does the model find the churners that cost money?"""
    out: dict[str, float] = {}
    n = len(y_true)
    if n == 0:
        return out

    if value is not None:
        revenue = value.to_numpy(dtype="float64")
        at_risk_total = float(revenue[y_true == 1].sum())
        k = max(1, int(round(n * 0.10)))
        order = np.argsort(-y_prob, kind="stable")[:k]
        captured = float(revenue[order][y_true[order] == 1].sum())
        out["at_risk_revenue_total"] = at_risk_total
        out["at_risk_revenue_captured_top_decile"] = captured
        out["revenue_capture_rate_top_decile"] = (
            captured / at_risk_total if at_risk_total > 0 else float("nan")
        )
        # Random targeting of the same list length would capture proportionally.
        out["revenue_lift_top_decile"] = (
            (captured / at_risk_total) / (k / n) if at_risk_total > 0 else float("nan")
        )

    if high_value is not None:
        mask = high_value.to_numpy(dtype="bool")
        if mask.any():
            sub_true, sub_prob = y_true[mask], y_prob[mask]
            k = max(1, int(round(len(sub_true) * 0.10)))
            order = np.argsort(-sub_prob, kind="stable")[:k]
            out["high_value_precision_top_decile"] = float(sub_true[order].mean())
            if len(np.unique(sub_true)) > 1:
                out["high_value_pr_auc"] = float(average_precision_score(sub_true, sub_prob))
                out["high_value_roc_auc"] = float(roc_auc_score(sub_true, sub_prob))
    return out


def evaluate_predictions(
    y_true: pd.Series | np.ndarray,
    y_prob: pd.Series | np.ndarray,
    *,
    model_name: str,
    dataset: str,
    threshold: float = DEFAULT_THRESHOLD,
    value: pd.Series | None = None,
    high_value: pd.Series | None = None,
) -> EvaluationResult:
    """Compute the full metric set for one model on one dataset."""
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype="float64")
    n = len(y_true)
    positives = int(y_true.sum())
    base_rate = float(y_true.mean()) if n else float("nan")

    y_pred = (y_prob >= threshold).astype(int)

    # A degenerate single-class dataset makes the ranking metrics undefined; report NaN rather
    # than raising, so one bad slice cannot abort a whole comparison.
    both_classes = len(np.unique(y_true)) > 1
    metrics: dict[str, float] = {
        "roc_auc": float(roc_auc_score(y_true, y_prob)) if both_classes else float("nan"),
        "pr_auc": float(average_precision_score(y_true, y_prob)) if both_classes else float("nan"),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "accuracy": float((y_pred == y_true).mean()) if n else float("nan"),
        "brier": float(brier_score_loss(y_true, y_prob)) if n else float("nan"),
        "log_loss": float(log_loss(y_true, np.clip(y_prob, 1e-9, 1 - 1e-9)))
        if both_classes
        else float("nan"),
        "ece": expected_calibration_error(y_true, y_prob),
        "mean_predicted": float(y_prob.mean()) if n else float("nan"),
        "mean_observed": base_rate,
    }
    # A single number for "is the overall level right", separate from the per-bin ECE.
    metrics["calibration_bias"] = metrics["mean_predicted"] - metrics["mean_observed"]

    decile = top_decile_metrics(y_true, y_prob)
    metrics["precision_top_decile"] = decile["precision"]
    metrics["recall_top_decile"] = decile["recall"]
    metrics["lift_top_decile"] = decile["lift"]

    if n and both_classes:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    else:  # pragma: no cover - degenerate slice
        tn = fp = fn = tp = 0
    confusion = {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}

    return EvaluationResult(
        model_name=model_name,
        dataset=dataset,
        n=n,
        positives=positives,
        base_rate=base_rate,
        threshold=threshold,
        metrics=metrics,
        confusion=confusion,
        reliability=reliability_table(y_true, y_prob),
        business=_business_metrics(y_true, y_prob, value=value, high_value=high_value),
    )
