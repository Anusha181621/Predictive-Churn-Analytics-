"""Environment-driven configuration.

Every configurable value in the platform is declared here once, read from the environment (or
an optional ``.env`` file) and exposed through :func:`get_settings`.

Design rules:

* Every setting has a working default, so the project runs correctly with **no** ``.env``.
* Paths are stored as configured (normally relative) and resolved against the project root on
  access, so nothing absolute is ever baked in and the project stays portable.
* Nothing here reads a CSV or touches pandas -- configuration must be cheap and import-safe.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import date
from functools import lru_cache
from pathlib import Path

from src.utils.paths import project_root, resolve_under_root

__all__ = ["ConfigError", "Settings", "get_settings"]


class ConfigError(RuntimeError):
    """Raised when configuration is unusable, e.g. a configured CSV file is missing."""


# --------------------------------------------------------------------------------------
# environment parsing helpers
# --------------------------------------------------------------------------------------


def _env_str(key: str, default: str) -> str:
    value = os.environ.get(key)
    if value is None:
        return default
    value = value.strip()
    return value if value else default


def _env_int(key: str, default: int) -> int:
    raw = _env_str(key, str(default))
    try:
        return int(raw)
    except ValueError as exc:
        raise ConfigError(f"{key}={raw!r} is not a valid integer") from exc


def _env_float(key: str, default: float) -> float:
    raw = _env_str(key, str(default))
    try:
        return float(raw)
    except ValueError as exc:
        raise ConfigError(f"{key}={raw!r} is not a valid number") from exc


def _env_date(key: str) -> date | None:
    """Parse an optional ISO date. Blank or unset means "derive it from the data"."""
    raw = os.environ.get(key, "").strip()
    if not raw:
        return None
    try:
        return date.fromisoformat(raw)
    except ValueError as exc:
        raise ConfigError(f"{key}={raw!r} is not a valid ISO date (expected YYYY-MM-DD)") from exc


def _load_dotenv_if_available() -> None:
    """Load ``<root>/.env`` when python-dotenv is installed.

    Kept optional so that configuration -- and therefore the CSV loader -- still imports in a
    bare environment. Real environment variables always win over the file.
    """
    try:
        from dotenv import load_dotenv
    except ModuleNotFoundError:  # pragma: no cover - only when deps are not installed
        return
    load_dotenv(project_root() / ".env", override=False)


# --------------------------------------------------------------------------------------
# settings
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Settings:
    """Resolved platform configuration."""

    # --- CSV data (the source of truth; there is no database) ---
    data_dir: str = "data"
    customer_file: str = "Customer.csv"
    transaction_file: str = "Transaction.csv"
    return_file: str = "Return.csv"
    product_file: str = "Product.csv"

    # --- output locations ---
    models_dir: str = "models"
    outputs_dir: str = "outputs"
    log_dir: str = "logs"

    # --- logging ---
    log_level: str = "INFO"

    # --- churn definition (consumed from the feature/model step onwards) ---
    churn_inactivity_days: int = 180
    as_of_date: date | None = None

    # --- churn risk bands ---
    risk_threshold_medium: float = 0.30
    risk_threshold_high: float = 0.60
    risk_threshold_critical: float = 0.80

    # --- reproducibility & presentation ---
    random_seed: int = 42
    currency: str = "EUR"

    # Populated in __post_init__; not read from the environment.
    table_files: dict[str, str] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "table_files",
            {
                "customers": self.customer_file,
                "transactions": self.transaction_file,
                "returns": self.return_file,
                "products": self.product_file,
            },
        )
        self._validate_thresholds()

    def _validate_thresholds(self) -> None:
        thresholds = (
            self.risk_threshold_medium,
            self.risk_threshold_high,
            self.risk_threshold_critical,
        )
        if not all(0.0 < t < 1.0 for t in thresholds):
            raise ConfigError(
                "Risk thresholds must each lie strictly between 0 and 1, got "
                f"medium={self.risk_threshold_medium}, high={self.risk_threshold_high}, "
                f"critical={self.risk_threshold_critical}"
            )
        if not thresholds[0] < thresholds[1] < thresholds[2]:
            raise ConfigError(
                "Risk thresholds must be strictly increasing "
                "(medium < high < critical), got "
                f"medium={self.risk_threshold_medium}, high={self.risk_threshold_high}, "
                f"critical={self.risk_threshold_critical}"
            )
        if self.churn_inactivity_days <= 0:
            raise ConfigError(
                f"CHURN_INACTIVITY_DAYS must be positive, got {self.churn_inactivity_days}"
            )

    # --- resolved paths -------------------------------------------------------------

    @property
    def project_root(self) -> Path:
        return project_root()

    @property
    def data_path(self) -> Path:
        return resolve_under_root(self.data_dir)

    @property
    def customer_path(self) -> Path:
        return self.data_path / self.customer_file

    @property
    def transaction_path(self) -> Path:
        return self.data_path / self.transaction_file

    @property
    def return_path(self) -> Path:
        return self.data_path / self.return_file

    @property
    def product_path(self) -> Path:
        return self.data_path / self.product_file

    @property
    def models_path(self) -> Path:
        return resolve_under_root(self.models_dir)

    @property
    def outputs_path(self) -> Path:
        return resolve_under_root(self.outputs_dir)

    @property
    def log_path(self) -> Path:
        return resolve_under_root(self.log_dir)

    def csv_path(self, table: str) -> Path:
        """Return the resolved path of one logical table.

        ``table`` is one of ``customers``, ``transactions``, ``returns``, ``products``.
        """
        try:
            filename = self.table_files[table]
        except KeyError as exc:
            known = ", ".join(sorted(self.table_files))
            raise ConfigError(f"Unknown table {table!r}; expected one of: {known}") from exc
        return self.data_path / filename

    # --- validation -----------------------------------------------------------------

    def missing_files(self) -> dict[str, Path]:
        """Return ``{table: resolved path}`` for every configured CSV that is not present."""
        return {
            table: path
            for table in self.table_files
            if not (path := self.csv_path(table)).is_file()
        }

    def validate_files(self) -> None:
        """Raise :class:`ConfigError` naming every missing CSV and where it was looked for.

        This produces an actionable message instead of a bare ``FileNotFoundError`` surfacing
        from inside pandas several frames deeper.
        """
        missing = self.missing_files()
        if not missing:
            return
        details = "\n".join(f"  - {table}: {path}" for table, path in sorted(missing.items()))
        raise ConfigError(
            f"{len(missing)} configured CSV file(s) were not found.\n"
            f"{details}\n"
            f"DATA_DIR resolved to: {self.data_path}\n"
            f"Project root: {self.project_root}\n"
            "Check DATA_DIR and the *_FILE settings in your .env (see .env.example)."
        )


def _build_settings() -> Settings:
    _load_dotenv_if_available()
    return Settings(
        data_dir=_env_str("DATA_DIR", "data"),
        customer_file=_env_str("CUSTOMER_FILE", "Customer.csv"),
        transaction_file=_env_str("TRANSACTION_FILE", "Transaction.csv"),
        return_file=_env_str("RETURN_FILE", "Return.csv"),
        product_file=_env_str("PRODUCT_FILE", "Product.csv"),
        models_dir=_env_str("MODELS_DIR", "models"),
        outputs_dir=_env_str("OUTPUTS_DIR", "outputs"),
        log_dir=_env_str("LOG_DIR", "logs"),
        log_level=_env_str("LOG_LEVEL", "INFO").upper(),
        churn_inactivity_days=_env_int("CHURN_INACTIVITY_DAYS", 180),
        as_of_date=_env_date("AS_OF_DATE"),
        risk_threshold_medium=_env_float("RISK_THRESHOLD_MEDIUM", 0.30),
        risk_threshold_high=_env_float("RISK_THRESHOLD_HIGH", 0.60),
        risk_threshold_critical=_env_float("RISK_THRESHOLD_CRITICAL", 0.80),
        random_seed=_env_int("RANDOM_SEED", 42),
        currency=_env_str("CURRENCY", "EUR"),
    )


@lru_cache(maxsize=1)
def _cached_settings() -> Settings:
    return _build_settings()


def get_settings(*, refresh: bool = False) -> Settings:
    """Return the cached :class:`Settings`.

    Pass ``refresh=True`` to re-read the environment -- used by tests that monkeypatch
    environment variables, and useful in a long-running app after a config change.
    """
    if refresh:
        _cached_settings.cache_clear()
    return _cached_settings()
