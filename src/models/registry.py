"""Model persistence.

A saved model is stored alongside the metadata needed to interpret it later: the as-of dates it was
trained on, the churn horizon, the feature columns in order, and the metrics it achieved. Without
that, a ``.pkl`` on disk is unusable six months later -- you cannot tell what it predicts, over what
window, or whether the feature table it expects still looks the same.

The feature list is the load-bearing part. If the feature layer gains or loses a column, a model
trained on the old set will silently receive a misaligned matrix, so :func:`load_model` compares the
stored column list against what it is handed and refuses to score on a mismatch rather than
returning confident nonsense.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from src.utils.logging_config import get_logger
from src.utils.paths import ensure_dir

__all__ = ["ModelMetadata", "save_model", "load_model", "SavedModel", "MODEL_FILENAME",
           "METADATA_FILENAME"]

logger = get_logger(__name__)

MODEL_FILENAME = "churn_model.joblib"
METADATA_FILENAME = "churn_model_metadata.json"


class FeatureMismatchError(RuntimeError):
    """Raised when a frame does not carry the columns the saved model was trained on."""


@dataclass
class ModelMetadata:
    """Everything needed to interpret and safely reuse a saved model."""

    model_name: str
    trained_at: str
    #: Churn horizon in days -- what "churn" means for this model's output.
    horizon_days: int
    label_mode: str
    #: Feature columns in the exact order the pipeline expects.
    feature_columns: list[str]
    numeric_columns: list[str]
    categorical_columns: list[str]
    train_as_of_dates: list[str]
    validation_as_of_dates: list[str]
    test_as_of_dates: list[str]
    train_rows: int
    train_churn_rate: float
    calibration: str
    random_seed: int
    metrics: dict[str, Any] = field(default_factory=dict)
    candidate_scores: list[dict[str, Any]] = field(default_factory=list)
    selection_metric: str = "pr_auc"
    selection_rationale: str = ""
    top_features: list[dict[str, Any]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SavedModel:
    """A loaded estimator plus its metadata."""

    pipeline: Any
    metadata: ModelMetadata

    def predict_proba(self, frame: pd.DataFrame) -> pd.Series:
        """Churn probability for each row, with the feature contract enforced first."""
        matrix = self.align(frame)
        probabilities = self.pipeline.predict_proba(matrix)[:, 1]
        return pd.Series(probabilities, index=frame.index, name="churn_probability")

    def align(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Return ``frame`` restricted and reordered to the stored feature columns.

        Raises :class:`FeatureMismatchError` if any expected column is absent. Extra columns are
        dropped silently -- new features appearing is harmless; expected ones vanishing is not.
        """
        expected = self.metadata.feature_columns
        missing = [column for column in expected if column not in frame.columns]
        if missing:
            raise FeatureMismatchError(
                f"the saved model {self.metadata.model_name!r} expects "
                f"{len(expected)} feature columns; {len(missing)} are missing from the frame "
                f"(first few: {missing[:5]}). The feature layer has changed since training -- "
                "retrain rather than scoring on a misaligned matrix."
            )
        return frame[expected]


def save_model(
    pipeline: Any,
    metadata: ModelMetadata,
    directory: str | Path,
    *,
    filename: str = MODEL_FILENAME,
) -> tuple[Path, Path]:
    """Persist the pipeline and its metadata. Returns ``(model path, metadata path)``."""
    target = ensure_dir(directory)
    model_path = target / filename
    metadata_path = target / METADATA_FILENAME

    joblib.dump(pipeline, model_path)
    metadata_path.write_text(
        json.dumps(metadata.as_dict(), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    logger.info("Saved model to %s (%.1f KB)", model_path, model_path.stat().st_size / 1024)
    logger.info("Saved metadata to %s", metadata_path)
    return model_path, metadata_path


def load_model(
    directory: str | Path, *, filename: str = MODEL_FILENAME
) -> SavedModel:
    """Load a persisted model and its metadata."""
    source = Path(directory)
    model_path = source / filename
    metadata_path = source / METADATA_FILENAME

    if not model_path.is_file():
        raise FileNotFoundError(
            f"no trained model at {model_path}. Run `python scripts/train_model.py` first."
        )
    if not metadata_path.is_file():
        raise FileNotFoundError(
            f"model at {model_path} has no metadata at {metadata_path}; it cannot be interpreted "
            "safely, so retrain to regenerate both."
        )

    pipeline = joblib.load(model_path)
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata = ModelMetadata(**payload)
    logger.info(
        "Loaded %s trained %s on a %d-day horizon (%d features)",
        metadata.model_name,
        metadata.trained_at,
        metadata.horizon_days,
        len(metadata.feature_columns),
    )
    return SavedModel(pipeline=pipeline, metadata=metadata)


def utc_timestamp() -> str:
    """ISO timestamp for metadata."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")
