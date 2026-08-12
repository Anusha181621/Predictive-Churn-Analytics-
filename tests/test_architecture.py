"""The Critical Architecture Check, as executable tests.

The brief's central constraint is repeated in every section: the four CSV files are the source of
truth, and there must be no database, no ETL pipeline, no ingestion service and no API. A
constraint that is only stated in prose drifts the moment somebody adds a convenient dependency,
so it is asserted here instead.

Each test corresponds to one line of the brief's architecture checklist. The most important is
:func:`test_running_the_pipeline_does_not_modify_the_source_csvs`, which hashes the source files,
runs the feature build over them, and hashes them again -- the difference between *believing* the
CSVs are read-only and *demonstrating* it.
"""

from __future__ import annotations

import ast
import hashlib
import re
from pathlib import Path

import pytest

from src.config.settings import get_settings
from src.data.csv_loader import load_all
from src.data.validation import validate_datasets
from src.features import build_customer_features
from src.utils.paths import project_root

# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------

#: The application's own source. Tests are excluded: this module necessarily *names* the
#: forbidden libraries in order to search for them, and would otherwise match itself.
SOURCE_DIRS = ("src", "app", "scripts")


def _python_files() -> list[Path]:
    root = project_root()
    files: list[Path] = []
    for directory in SOURCE_DIRS:
        files.extend(
            path
            for path in (root / directory).rglob("*.py")
            if "__pycache__" not in path.parts
        )
    assert files, "no source files found - the layout assumption is wrong"
    return files


def _imported_modules(path: Path) -> set[str]:
    """Top-level module names imported by ``path``, via AST rather than a text search.

    Parsing means a library named in a docstring or a comment -- as several are, in the very
    explanations of why they are absent -- cannot produce a false failure.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            modules.add(node.module.split(".")[0])
    return modules


def _all_imports() -> dict[str, set[str]]:
    return {str(path): _imported_modules(path) for path in _python_files()}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _csv_fingerprint() -> dict[str, tuple[str, int]]:
    """Content hash and byte size of every source CSV."""
    settings = get_settings()
    return {
        name: (_sha256(settings.csv_path(name)), settings.csv_path(name).stat().st_size)
        for name in ("customers", "transactions", "returns", "products")
    }


# --------------------------------------------------------------------------------------
# no database
# --------------------------------------------------------------------------------------

DATABASE_MODULES = {
    "sqlalchemy",
    "sqlite3",
    "psycopg2",
    "psycopg",
    "pymysql",
    "MySQLdb",
    "duckdb",
    "pymongo",
    "redis",
    "cassandra",
    "clickhouse_driver",
    "snowflake",
    "pyodbc",
    "asyncpg",
    "alembic",
    "peewee",
    "tortoise",
}


def test_no_database_driver_or_orm_is_imported() -> None:
    offenders = {
        path: sorted(modules & DATABASE_MODULES)
        for path, modules in _all_imports().items()
        if modules & DATABASE_MODULES
    }
    assert not offenders, f"database dependencies found: {offenders}"


def test_no_sql_is_executed_anywhere() -> None:
    """No DDL, no DML and no cursor execution.

    Searched as text rather than AST because SQL would arrive as a string literal, which is
    exactly what a text search is good at.
    """
    patterns = re.compile(
        r"\bCREATE\s+TABLE\b|\bINSERT\s+INTO\b|\bDROP\s+TABLE\b|\bALTER\s+TABLE\b"
        r"|\bcreate_engine\s*\(|\.execute(many)?\s*\(|\bcursor\s*\(",
        re.IGNORECASE,
    )
    offenders = {
        str(path): sorted({m.group(0).strip() for m in patterns.finditer(path.read_text("utf-8"))})
        for path in _python_files()
        if patterns.search(path.read_text("utf-8"))
    }
    assert not offenders, f"SQL or cursor usage found: {offenders}"


def test_no_database_dependency_is_declared() -> None:
    text = (project_root() / "requirements.txt").read_text(encoding="utf-8")
    declared = {
        re.split(r"[<>=!\[]", line.strip())[0].strip().lower()
        for line in text.splitlines()
        if line.strip() and not line.strip().startswith("#")
    }
    forbidden = declared & {m.lower() for m in DATABASE_MODULES}
    assert not forbidden, f"requirements.txt declares database packages: {sorted(forbidden)}"


# --------------------------------------------------------------------------------------
# no ETL pipeline, no ingestion service, no API
# --------------------------------------------------------------------------------------

ORCHESTRATION_MODULES = {"airflow", "prefect", "dagster", "luigi", "celery", "kafka", "pyspark"}
API_MODULES = {"fastapi", "flask", "django", "starlette", "uvicorn", "gunicorn", "aiohttp", "tornado"}


def test_no_etl_or_orchestration_framework_is_imported() -> None:
    offenders = {
        path: sorted(modules & ORCHESTRATION_MODULES)
        for path, modules in _all_imports().items()
        if modules & ORCHESTRATION_MODULES
    }
    assert not offenders, f"ETL/orchestration dependencies found: {offenders}"


def test_no_api_or_server_framework_is_imported() -> None:
    """No API layer, and no long-running ingestion service.

    Streamlit serves the dashboard, but it is a UI over files that already exist -- it accepts no
    inbound data and is not an ingestion endpoint.
    """
    offenders = {
        path: sorted(modules & API_MODULES)
        for path, modules in _all_imports().items()
        if modules & API_MODULES
    }
    assert not offenders, f"API/server dependencies found: {offenders}"


def test_every_script_is_a_batch_cli_not_a_service() -> None:
    """Each entry point runs, reports and exits -- nothing daemonises or listens on a socket."""
    for path in (project_root() / "scripts").glob("*.py"):
        source = path.read_text(encoding="utf-8")
        assert "def main(" in source, f"{path.name} has no main() entry point"
        assert "SystemExit(main(" in source or "sys.exit(main(" in source, (
            f"{path.name} does not exit with its main() status"
        )
        for forbidden in ("while True", "serve_forever", "app.run(", "socket."):
            assert forbidden not in source, f"{path.name} looks like a service: {forbidden!r}"


# --------------------------------------------------------------------------------------
# the CSVs are the source of truth, and they are read-only
# --------------------------------------------------------------------------------------


#: Names through which a *source* CSV path is obtained. Anything reading one of these is reading
#: the source of truth; anything else reading a CSV is reading a generated artefact.
SOURCE_PATH_NAMES = re.compile(
    r"(?<![A-Za-z0-9_])(data_path|data_dir|customer_path|transaction_path|return_path"
    r"|product_path|csv_path|CUSTOMER_FILE|TRANSACTION_FILE|RETURN_FILE|PRODUCT_FILE)"
)


def test_only_the_loader_reads_the_source_csv_files() -> None:
    """Every consumer of the *source* data goes through the loader.

    A second reader of ``data/`` would silently reintroduce the traps the loader exists to
    prevent -- most notably ``Order ID`` losing its zero padding to an inferred ``int64``, which
    breaks every string join against ``Return.csv``.

    Reading a *generated* artefact is a different act and is allowed: the dashboard reads
    ``outputs/*.csv``, which the loader could not parse anyway because its schema is bound to the
    four source tables. That those artefacts are the only other files read is asserted separately,
    by :func:`test_the_dashboard_reads_only_from_data_outputs_and_models`.
    """
    reads_csv = re.compile(r"\b(?:pd|pandas)\.read_csv\s*\(")
    loader = "src/data/csv_loader.py"

    readers = {
        path.relative_to(project_root()).as_posix(): path.read_text("utf-8")
        for path in _python_files()
        if reads_csv.search(path.read_text("utf-8"))
    }
    assert loader in readers, "the loader should be the module that reads the source CSVs"

    for name, source in readers.items():
        if name == loader:
            continue
        assert not SOURCE_PATH_NAMES.search(source), (
            f"{name} reads a CSV using a source-data path; it must go through {loader}"
        )


def test_no_code_path_writes_into_the_data_directory() -> None:
    """Writes go to outputs/, models/ and logs/ -- never to the source data."""
    write_call = re.compile(
        r"\.to_csv\s*\(|\.to_json\s*\(|\.to_parquet\s*\(|\.write_text\s*\(|\.write_bytes\s*\("
        r"|\bopen\s*\([^)]*['\"][wa]|joblib\.dump\s*\("
    )
    # SOURCE_PATH_NAMES is anchored on a non-word character so that `metadata_path` -- which
    # contains the literal substring "data_path" and points at outputs/ or models/ -- is not
    # mistaken for the source data directory.
    offenders = []
    for path in _python_files():
        for number, line in enumerate(path.read_text("utf-8").splitlines(), start=1):
            if write_call.search(line) and SOURCE_PATH_NAMES.search(line):
                offenders.append(f"{path.name}:{number}: {line.strip()}")
    assert not offenders, f"a write targets the source data directory: {offenders}"


def test_running_the_pipeline_does_not_modify_the_source_csvs() -> None:
    """Hash the CSVs, exercise the pipeline over them, hash again.

    This is the check that actually earns the claim. Reading with pandas, validating, and building
    the full feature table must leave every source byte where it was.
    """
    before = _csv_fingerprint()

    data = load_all()
    validate_datasets(data)
    build_customer_features(data, as_of_date="2025-06-30")

    after = _csv_fingerprint()
    assert after == before, (
        "the source CSVs changed while the pipeline ran: "
        f"{[name for name in before if before[name] != after[name]]}"
    )


def test_the_loader_never_opens_a_source_file_for_writing() -> None:
    source = (project_root() / "src" / "data" / "csv_loader.py").read_text(encoding="utf-8")
    assert "read_csv" in source
    for forbidden in ("to_csv", "write_text", "write_bytes", '"w"', "'w'", '"a"', "'a'"):
        assert forbidden not in source, f"csv_loader.py contains {forbidden!r}"


# --------------------------------------------------------------------------------------
# portability: relative paths, no machine-specific configuration
# --------------------------------------------------------------------------------------


def test_no_absolute_path_is_hardcoded() -> None:
    """The project must work after being copied to another machine."""
    absolute = re.compile(r"['\"](?:[A-Za-z]:[\\/]|/(?:home|Users|mnt|opt|var)/)")
    offenders = []
    for path in _python_files():
        for number, line in enumerate(path.read_text("utf-8").splitlines(), start=1):
            if absolute.search(line) and "#" not in line.split("'")[0].split('"')[0]:
                offenders.append(f"{path.name}:{number}: {line.strip()}")
    assert not offenders, f"hardcoded absolute paths: {offenders}"


def test_relative_paths_resolve_against_the_project_root(monkeypatch: pytest.MonkeyPatch) -> None:
    """The portability guarantee: a relative setting is anchored to the project, not the CWD.

    Pinned with explicit values rather than whatever the ambient environment happens to hold, so
    the test states the property instead of describing the machine it runs on.
    """
    for variable in ("DATA_DIR", "OUTPUTS_DIR", "MODELS_DIR", "LOG_DIR"):
        monkeypatch.delenv(variable, raising=False)
    monkeypatch.setenv("DATA_DIR", "data")
    monkeypatch.setenv("OUTPUTS_DIR", "outputs")
    settings = get_settings(refresh=True)
    root = project_root()

    for path in (settings.data_path, settings.outputs_path):
        assert path.is_absolute(), f"{path} should resolve to an absolute path"
        assert root in path.parents, f"{path} is not anchored to the project root"

    get_settings(refresh=True)


def test_an_absolute_path_is_honoured_as_given(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Pointing a directory at shared storage is supported, and documented as supported.

    Relative paths are what make the project portable; absolute ones are an explicit escape hatch,
    so resolution must not quietly re-anchor them under the project root.
    """
    monkeypatch.setenv("OUTPUTS_DIR", str(tmp_path))
    settings = get_settings(refresh=True)
    assert settings.outputs_path == tmp_path
    get_settings(refresh=True)


# --------------------------------------------------------------------------------------
# the dashboard is a reader over the CSV-derived artefacts
# --------------------------------------------------------------------------------------


def test_the_dashboard_entry_point_exists_and_is_a_single_command() -> None:
    entry = project_root() / "app" / "dashboard.py"
    assert entry.is_file(), "streamlit run app/dashboard.py needs app/dashboard.py"
    assert "st.navigation" in entry.read_text(encoding="utf-8")


def test_the_dashboard_reads_only_from_data_outputs_and_models() -> None:
    """No hidden third source: everything the pages show comes through the data-access layer."""
    from app import data_access

    for key, artefact in data_access.ARTEFACTS.items():
        assert artefact.directory in {"outputs", "models"}, (
            f"artefact {key} points at {artefact.directory}, which is not a generated location"
        )
        assert artefact.command.startswith("python scripts/"), (
            f"artefact {key} has no reproducible command that produces it"
        )


def test_the_dashboard_does_not_import_a_database_or_api_client() -> None:
    forbidden = DATABASE_MODULES | API_MODULES | ORCHESTRATION_MODULES
    offenders = {
        path: sorted(modules & forbidden)
        for path, modules in _all_imports().items()
        if "app" in Path(path).parts and modules & forbidden
    }
    assert not offenders, f"the dashboard reaches outside the CSV architecture: {offenders}"


@pytest.mark.parametrize(
    "package", ["src.config", "src.data", "src.features", "src.models", "src.retention"]
)
def test_core_packages_import_without_side_effects(package: str) -> None:
    """Importing a package must not read, write or connect to anything."""
    before = _csv_fingerprint()
    __import__(package)
    assert _csv_fingerprint() == before
