"""Shared pytest fixtures."""

from __future__ import annotations

import pytest

from src.config.settings import Settings, get_settings
from src.data.csv_loader import Datasets, load_all
from src.data.validation import ValidationReport, validate_datasets


@pytest.fixture(scope="session")
def settings() -> Settings:
    return get_settings()


@pytest.fixture(scope="session")
def data() -> Datasets:
    """The four source tables, parsed once for the whole test session."""
    return load_all()


@pytest.fixture(scope="session")
def report(data: Datasets) -> ValidationReport:
    """The full validation report over the real CSVs, computed once for the session."""
    return validate_datasets(data)
