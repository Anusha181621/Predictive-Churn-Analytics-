"""Logging configuration.

One idempotent entry point, :func:`configure_logging`, so that scripts, tests and (later) the
Streamlit app all produce the same log format without stacking duplicate handlers when they
re-import or re-run.

Note the explicit UTF-8 stream: the customer data contains non-ASCII city names
(``Dusseldorf`` with an umlaut, ``Liege`` with a grave accent) and the default Windows console
encoding is cp1252, which would raise ``UnicodeEncodeError`` when those values are logged.
"""

from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

from src.utils.paths import ensure_dir

__all__ = ["configure_logging", "get_logger"]

LOG_FORMAT = "%(asctime)s %(levelname)-8s %(name)s %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

_MAX_BYTES = 2 * 1024 * 1024  # 2 MB per log file
_BACKUP_COUNT = 3

# Marks handlers this module installed, so repeated calls replace our own handlers instead of
# appending to them (Streamlit re-runs a script top-to-bottom on every interaction).
_HANDLER_TAG = "fashion_churn_platform"


def _tagged(handler: logging.Handler) -> logging.Handler:
    handler.set_name(_HANDLER_TAG)
    return handler


def _remove_our_handlers(logger: logging.Logger) -> None:
    for handler in list(logger.handlers):
        if handler.get_name() == _HANDLER_TAG:
            logger.removeHandler(handler)
            handler.close()


def configure_logging(
    level: int | str | None = None,
    log_file: str | Path | None = None,
    *,
    to_console: bool = True,
) -> logging.Logger:
    """Configure the root logger and return it.

    Parameters
    ----------
    level:
        Log level as a name (``"INFO"``) or numeric value. Defaults to the configured
        ``LOG_LEVEL``.
    log_file:
        Destination log file. Relative paths resolve under the project root. Defaults to
        ``<LOG_DIR>/fashion_churn_platform.log``. Pass ``False``-y values only via
        ``to_console`` -- file logging is always on so runs are auditable.
    to_console:
        Also stream to stderr. Turn this off for quiet batch runs.

    Safe to call more than once; each call replaces the handlers it previously installed.
    """
    # Imported lazily: src.config imports nothing from this module, but keeping the import
    # local documents that logging works even if configuration fails to load.
    from src.config.settings import get_settings

    settings = get_settings()

    if level is None:
        level = settings.log_level
    if isinstance(level, str):
        level = logging.getLevelNamesMapping().get(level.strip().upper(), logging.INFO)

    if log_file is None:
        log_file = ensure_dir(settings.log_dir) / "fashion_churn_platform.log"
    else:
        log_file = Path(log_file)
        ensure_dir(log_file.parent)

    formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)

    root = logging.getLogger()
    _remove_our_handlers(root)
    root.setLevel(level)

    file_handler = RotatingFileHandler(
        log_file, maxBytes=_MAX_BYTES, backupCount=_BACKUP_COUNT, encoding="utf-8"
    )
    file_handler.setFormatter(formatter)
    root.addHandler(_tagged(file_handler))

    if to_console:
        stream = logging.StreamHandler(stream=sys.stderr)
        stream.setFormatter(formatter)
        root.addHandler(_tagged(stream))
        # Non-ASCII city names must not blow up on a cp1252 console.
        reconfigure = getattr(sys.stderr, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):  # pragma: no cover - depends on the host terminal
                pass

    return root


def get_logger(name: str) -> logging.Logger:
    """Return a module logger. Call :func:`configure_logging` once at your entry point."""
    return logging.getLogger(name)
