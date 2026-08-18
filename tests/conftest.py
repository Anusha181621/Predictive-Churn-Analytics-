"""Shared pytest fixtures."""

from __future__ import annotations

import pytest

from src.config.settings import Settings, get_settings
from src.data.csv_loader import Datasets, load_all
from src.data.validation import ValidationReport, validate_datasets


@pytest.fixture(autouse=True)
def _settings_cache_is_not_shared_between_tests() -> object:
    """Rebuild the settings cache after every test.

    ``get_settings`` is ``lru_cache``d, which is right for an application and a trap for a test
    suite: a test that patches an environment variable and then refreshes leaves the *patched*
    settings cached, and ``monkeypatch`` undoing the variable afterwards cannot undo that. Every
    later test then reads whatever directory the earlier one pointed at.

    That is not hypothetical -- it happened. A test pinning ``OUTPUTS_DIR`` at a temporary
    directory left the cache aimed there, and every artefact-reading test that ran before the next
    refresh failed with a missing file that was sitting on disk the whole time. Refreshing here,
    after ``monkeypatch`` has restored the environment, makes the isolation structural instead of
    something each test has to remember.
    """
    yield
    get_settings(refresh=True)


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
