"""Configuration and path-resolution tests.

The point of these is portability: nothing may depend on an absolute path or on the current
working directory, because the project has to keep working when moved to another machine.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.config.settings import ConfigError, Settings, get_settings
from src.utils.paths import project_root, resolve_under_root


def test_project_root_contains_the_expected_markers() -> None:
    root = project_root()
    assert (root / "src").is_dir()
    assert (root / "requirements.txt").is_file()


def test_relative_paths_resolve_under_the_project_root() -> None:
    assert resolve_under_root("data") == project_root() / "data"


def test_absolute_paths_pass_through_unchanged(tmp_path: Path) -> None:
    assert resolve_under_root(tmp_path) == tmp_path


def test_path_resolution_is_independent_of_the_working_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    assert resolve_under_root("data") == project_root() / "data"


def test_defaults_point_at_the_four_csv_files(settings: Settings) -> None:
    assert settings.data_path.is_dir()
    for path in (
        settings.customer_path,
        settings.transaction_path,
        settings.return_path,
        settings.product_path,
    ):
        assert path.is_file(), path
    # No absolute path may be baked into the configured values themselves.
    assert not Path(settings.data_dir).is_absolute()


def test_csv_path_matches_the_named_properties(settings: Settings) -> None:
    assert settings.csv_path("customers") == settings.customer_path
    assert settings.csv_path("transactions") == settings.transaction_path
    assert settings.csv_path("returns") == settings.return_path
    assert settings.csv_path("products") == settings.product_path


def test_csv_path_rejects_an_unknown_table(settings: Settings) -> None:
    with pytest.raises(ConfigError, match="Unknown table"):
        settings.csv_path("orders")


def test_environment_variables_override_the_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DATA_DIR", "some/other/dir")
    monkeypatch.setenv("CUSTOMER_FILE", "Clients.csv")
    monkeypatch.setenv("CHURN_INACTIVITY_DAYS", "90")
    monkeypatch.setenv("AS_OF_DATE", "2025-06-30")
    try:
        overridden = get_settings(refresh=True)
        assert overridden.customer_path == project_root() / "some/other/dir/Clients.csv"
        assert overridden.churn_inactivity_days == 90
        assert overridden.as_of_date is not None
        assert overridden.as_of_date.isoformat() == "2025-06-30"
    finally:
        # Undo the module-level cache so later tests see the real configuration.
        monkeypatch.undo()
        get_settings(refresh=True)


def test_a_missing_csv_raises_a_config_error_naming_the_resolved_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CUSTOMER_FILE", "DoesNotExist.csv")
    try:
        broken = get_settings(refresh=True)
        with pytest.raises(ConfigError) as excinfo:
            broken.validate_files()
        message = str(excinfo.value)
        assert "DoesNotExist.csv" in message
        assert "customers" in message
        assert "DATA_DIR resolved to" in message
    finally:
        monkeypatch.undo()
        get_settings(refresh=True)


def test_blank_as_of_date_means_derive_from_the_data(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AS_OF_DATE", "   ")
    try:
        assert get_settings(refresh=True).as_of_date is None
    finally:
        monkeypatch.undo()
        get_settings(refresh=True)


@pytest.mark.parametrize(
    ("variable", "value", "fragment"),
    [
        ("CHURN_INACTIVITY_DAYS", "not-a-number", "not a valid integer"),
        ("RISK_THRESHOLD_HIGH", "high", "not a valid number"),
        ("AS_OF_DATE", "31-12-2025", "not a valid ISO date"),
        ("RISK_THRESHOLD_MEDIUM", "0.9", "strictly increasing"),
        ("RISK_THRESHOLD_CRITICAL", "1.5", "between 0 and 1"),
        ("CHURN_INACTIVITY_DAYS", "0", "must be positive"),
    ],
)
def test_invalid_configuration_is_rejected_with_a_clear_message(
    monkeypatch: pytest.MonkeyPatch, variable: str, value: str, fragment: str
) -> None:
    monkeypatch.setenv(variable, value)
    try:
        with pytest.raises(ConfigError, match=fragment):
            get_settings(refresh=True)
    finally:
        monkeypatch.undo()
        get_settings(refresh=True)


def test_risk_bands_are_ordered(settings: Settings) -> None:
    assert 0 < settings.risk_threshold_medium < settings.risk_threshold_high
    assert settings.risk_threshold_high < settings.risk_threshold_critical < 1
