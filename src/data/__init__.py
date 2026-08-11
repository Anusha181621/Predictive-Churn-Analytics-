"""CSV access layer.

The CSV files are the source of truth. This package declares their schema
(:mod:`src.data.schema`), loads them with explicit dtypes (:mod:`src.data.csv_loader`) and
runs read-only quality checks over them (:mod:`src.data.validation`). Nothing here writes to
or mutates the source files, and there is deliberately no database, ingestion or ETL code.
"""

from src.data.csv_loader import Datasets, SchemaError, load_all
from src.data.validation import (
    ValidationReport,
    compute_return_rate,
    validate_customers,
    validate_datasets,
    validate_products,
    validate_relationships,
    validate_returns,
    validate_transactions,
)

__all__ = [
    "Datasets",
    "SchemaError",
    "load_all",
    "ValidationReport",
    "compute_return_rate",
    "validate_customers",
    "validate_datasets",
    "validate_products",
    "validate_relationships",
    "validate_returns",
    "validate_transactions",
]
