"""Shared pytest fixtures."""

from __future__ import annotations

import pytest

from src.config.settings import Settings, get_settings
from src.data.csv_loader import Datasets, load_all
from src.data.validation import DataQualityReport, run_all_checks


@pytest.fixture(scope="session")
def settings() -> Settings:
    return get_settings()


@pytest.fixture(scope="session")
def data() -> Datasets:
    """The four source tables, parsed once for the whole test session."""
    return load_all()


@pytest.fixture(scope="session")
def report(data: Datasets) -> DataQualityReport:
    """The full data quality report, computed once for the whole test session."""
    return run_all_checks(data)
