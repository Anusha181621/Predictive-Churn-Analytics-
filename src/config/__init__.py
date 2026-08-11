"""Configuration management.

Exposes :func:`src.config.settings.get_settings`, the single entry point for every
configurable value. All paths are resolved relative to the project root so the project
remains portable across machines.
"""

from src.config.settings import ConfigError, Settings, get_settings

__all__ = ["ConfigError", "Settings", "get_settings"]
