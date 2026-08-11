"""The reusable CSV loading utility.

The four CSV files under ``data/`` are the source of truth for this platform. There is no
database, no ingestion step and no ETL pipeline: this module is the *only* place that reads
them, and every other module goes through it.

What it guarantees:

* **Explicit dtypes, never inference.** In particular ``Order ID`` stays the zero-padded
  6-character string it is in the file. Pandas would otherwise infer ``int64`` and turn
  ``"000001"`` into ``1``, silently breaking joins against ``Return.csv`` and every exported
  order number.
* **Explicit UTF-8.** ``Customer.csv`` contains non-ASCII city names (Duesseldorf, Liege);
  reading it with the platform default encoding on Windows would corrupt them.
* **Schema enforcement.** A missing or unexpected column raises :class:`SchemaError` naming
  the offending columns, rather than surfacing as a ``KeyError`` deep in feature code.
* **Canonical snake_case names by default**, which also disambiguates the ``Gender`` column
  that exists in both ``Customer.csv`` and ``Product.csv``.
* **Read-only.** Nothing here writes to or mutates the source files.

Typical use::

    from src.data.csv_loader import load_all

    data = load_all()
    data.transactions["order_id"].iloc[0]   # -> '000001'
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import pandas as pd

from src.config.settings import Settings, get_settings
from src.data import schema as sch
from src.utils.logging_config import get_logger

__all__ = [
    "Datasets",
    "SchemaError",
    "load_table",
    "load_customers",
    "load_transactions",
    "load_returns",
    "load_products",
    "load_all",
    "clear_cache",
]

logger = get_logger(__name__)

#: Read with a BOM-tolerant UTF-8 codec: the shipped files have no BOM, but a spreadsheet
#: round-trip commonly adds one and it would otherwise corrupt the first column name.
ENCODING = "utf-8-sig"

#: Only these strings become NaN. The default pandas list treats values such as "NA" and "None"
#: as missing, which would be wrong for free-text fields like City or Brand.
NA_VALUES = ["", "#N/A", "N/A", "NaN", "nan", "null", "NULL"]


class SchemaError(ValueError):
    """Raised when a CSV file does not match its declared schema."""


@dataclass(frozen=True)
class Datasets:
    """The four source tables, loaded.

    Attribute names match the logical table names used by :mod:`src.data.schema` and
    :class:`src.config.settings.Settings`.
    """

    customers: pd.DataFrame
    products: pd.DataFrame
    transactions: pd.DataFrame
    returns: pd.DataFrame

    def __iter__(self) -> Iterator[tuple[str, pd.DataFrame]]:
        yield "customers", self.customers
        yield "products", self.products
        yield "transactions", self.transactions
        yield "returns", self.returns

    @property
    def row_counts(self) -> dict[str, int]:
        return {name: len(frame) for name, frame in self}

    def as_dict(self) -> dict[str, pd.DataFrame]:
        return dict(self)


# --------------------------------------------------------------------------------------
# caching
#
# Keyed on (path, mtime, size, normalize_columns) so an edited CSV is always re-read, while
# repeated calls within a run -- and later, Streamlit's re-run-on-every-interaction model --
# do not re-parse 20,000 rows. Deliberately hand-rolled instead of `st.cache_data` so that
# `src/` stays free of any UI framework dependency.
# --------------------------------------------------------------------------------------

_CacheKey = tuple[str, int, int, bool]
_cache: dict[_CacheKey, pd.DataFrame] = {}


def clear_cache() -> None:
    """Drop every cached DataFrame. Mainly useful in tests."""
    _cache.clear()


def _cache_key(path: Path, normalize_columns: bool) -> _CacheKey:
    stat = path.stat()
    return (str(path), stat.st_mtime_ns, stat.st_size, normalize_columns)


# --------------------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------------------


def _check_columns(frame: pd.DataFrame, table: sch.TableSchema, path: Path) -> None:
    found = list(frame.columns)
    expected = set(table.columns)
    missing = [column for column in table.columns if column not in found]
    unexpected = [column for column in found if column not in expected]
    if missing or unexpected:
        parts = [f"{path.name} does not match the declared schema for table {table.name!r}."]
        if missing:
            parts.append(f"Missing column(s): {missing}")
        if unexpected:
            parts.append(f"Unexpected column(s): {unexpected}")
        parts.append(f"Expected exactly: {list(table.columns)}")
        parts.append(f"Found: {found}")
        raise SchemaError(" ".join(parts))


def _strip_string_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Trim stray whitespace from string columns, in place, and return the frame."""
    for column in frame.columns:
        if pd.api.types.is_string_dtype(frame[column]):
            frame[column] = frame[column].str.strip()
    return frame


def load_table(
    table: str,
    *,
    settings: Settings | None = None,
    normalize_columns: bool = True,
    use_cache: bool = True,
) -> pd.DataFrame:
    """Load one logical table from its CSV file.

    Parameters
    ----------
    table:
        ``"customers"``, ``"products"``, ``"transactions"`` or ``"returns"``.
    settings:
        Configuration override; defaults to :func:`~src.config.settings.get_settings`.
    normalize_columns:
        Rename the raw headers to canonical snake_case names (the default, and what all
        downstream code expects). Pass ``False`` to keep the original headers -- useful when
        displaying the file as-is.
    use_cache:
        Serve from the in-process cache when the file is unchanged.

    Returns
    -------
    A copy of the loaded frame, so callers can mutate it without corrupting the cache.
    """
    settings = settings or get_settings()
    if table not in sch.TABLES:
        known = ", ".join(sorted(sch.TABLES))
        raise KeyError(f"Unknown table {table!r}; expected one of: {known}")

    table_schema = sch.TABLES[table]
    path = settings.csv_path(table)
    if not path.is_file():
        # Delegate to the config layer so the error names every missing file at once and
        # explains which settings to change.
        settings.validate_files()

    key = _cache_key(path, normalize_columns)
    if use_cache and key in _cache:
        logger.debug("Cache hit for %s (%s)", table, path.name)
        return _cache[key].copy()

    logger.info("Reading %s from %s", table, path)
    frame = pd.read_csv(
        path,
        dtype=dict(table_schema.dtypes),
        parse_dates=list(table_schema.date_columns) or None,
        encoding=ENCODING,
        na_values=NA_VALUES,
        keep_default_na=False,
    )

    _check_columns(frame, table_schema, path)
    # Reorder to the declared file order so downstream column positions are deterministic.
    # `.copy()` because the following in-place strip must not write through to a view.
    frame = frame[list(table_schema.columns)].copy()
    _strip_string_columns(frame)

    if normalize_columns:
        frame = frame.rename(columns=dict(table_schema.rename_map))

    logger.info("Loaded %s: %d rows x %d columns", table, len(frame), frame.shape[1])

    if use_cache:
        _cache[key] = frame
        return frame.copy()
    return frame


def load_customers(**kwargs) -> pd.DataFrame:
    """Load ``Customer.csv`` (1,000 rows: one per customer)."""
    return load_table("customers", **kwargs)


def load_transactions(**kwargs) -> pd.DataFrame:
    """Load ``Transaction.csv`` (20,000 rows: one per order line)."""
    return load_table("transactions", **kwargs)


def load_returns(**kwargs) -> pd.DataFrame:
    """Load ``Return.csv`` (5,048 rows: one per returned order line)."""
    return load_table("returns", **kwargs)


def load_products(**kwargs) -> pd.DataFrame:
    """Load ``Product.csv`` (500 rows: one per SKU)."""
    return load_table("products", **kwargs)


def load_all(
    *,
    settings: Settings | None = None,
    normalize_columns: bool = True,
    use_cache: bool = True,
) -> Datasets:
    """Load all four tables and return them as a :class:`Datasets` bundle."""
    settings = settings or get_settings()
    settings.validate_files()
    kwargs = {
        "settings": settings,
        "normalize_columns": normalize_columns,
        "use_cache": use_cache,
    }
    return Datasets(
        customers=load_table("customers", **kwargs),
        products=load_table("products", **kwargs),
        transactions=load_table("transactions", **kwargs),
        returns=load_table("returns", **kwargs),
    )
