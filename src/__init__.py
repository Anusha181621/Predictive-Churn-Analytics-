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
    Customer-level feature engineering (next implementation step).
models
    Churn model training, evaluation and prediction (not implemented yet).
explainability
    SHAP and human-readable churn driver explanations (not implemented yet).
segmentation
    Value/risk/behaviour segmentation (not implemented yet).
retention
    Revenue at risk, retention ROI and the recommendation engine (not implemented yet).
utils
    Cross-cutting helpers: path resolution and logging configuration.
"""

__version__ = "0.1.0"
