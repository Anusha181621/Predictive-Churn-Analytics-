"""Fashion churn platform - source packages.

The four CSV files under ``data/`` are the single source of truth. There is no database,
no ingestion pipeline and no ETL step: every module here reads the CSVs directly through
:mod:`src.data.csv_loader`.

Sub-packages
------------
config
    Environment-driven configuration and path resolution.
data
    CSV schema declarations, the reusable CSV loader, and read-only data validation.
features
    Customer-level feature engineering: 148 features, computed strictly as of a prediction date.
models
    Churn labelling, time-based validation with embargoes, training, calibration and scoring.
explainability
    SHAP contributions and the phrase grammar that turns them into per-customer sentences.
segmentation
    Empty. The value/risk/behaviour segmentation described in the brief is implemented in
    :mod:`src.retention.segments`, alongside the scoring that consumes it.
retention
    Expected future revenue, revenue at risk, the twelve segments, prioritisation and the
    recommendation engine.
utils
    Cross-cutting helpers: path resolution and logging configuration.

The Streamlit dashboard that reads the artefacts these packages produce lives in ``app/``.
"""

__version__ = "1.0.0"
