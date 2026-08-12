"""One module per dashboard page, each exposing a ``render()`` function.

**Why ``views/`` and not ``pages/``.** ``pages/`` is a magic directory name in Streamlit: a folder
of that name beside the entry script triggers automatic multi-page navigation during start-up,
before the entry script runs. That would duplicate every page in the sidebar -- once from
``st.navigation`` and once auto-discovered -- and, because the automatic scan builds a ``Page``
from the *resolved path* of the entry script, it fails outright when the project sits on a Windows
network share: ``st.Page`` rejects UNC paths by design, so that resolving one cannot open an SMB
connection and disclose the server process's credentials.

These are therefore plain modules, imported and wired into ``st.navigation`` by ``dashboard.py``
as callables. Callables carry no filesystem path, so navigation works identically from a local
disk and from a mapped network drive.
"""
