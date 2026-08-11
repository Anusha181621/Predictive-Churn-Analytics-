"""Project path resolution.

This is the only module in the project that knows how to turn a configured path into a real
filesystem location. Keeping that logic in one place is what makes the project portable: no
absolute path is hard-coded anywhere, so moving the project to another machine (or another
directory on the same machine) needs no code or configuration change.
"""

from __future__ import annotations

from pathlib import Path

__all__ = ["project_root", "resolve_under_root", "ensure_dir"]


def project_root() -> Path:
    """Return the project root directory.

    Derived from this file's own location (``<root>/src/utils/paths.py``), so it is correct
    no matter what the current working directory is when the code runs.
    """
    return Path(__file__).resolve().parents[2]


def resolve_under_root(path: str | Path) -> Path:
    """Resolve ``path`` against the project root.

    An absolute path is returned unchanged, which lets a user point ``DATA_DIR`` at a shared
    location if they want to. A relative path -- the normal, portable case -- is anchored to
    the project root rather than to the current working directory.
    """
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate
    return (project_root() / candidate).resolve()


def ensure_dir(path: str | Path) -> Path:
    """Create a directory (and its parents) if it does not exist, and return it."""
    resolved = resolve_under_root(path)
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved
