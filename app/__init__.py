"""Streamlit retention dashboard.

    streamlit run app/dashboard.py

``dashboard.py`` is the entry point; ``data_access.py`` holds the cached readers and the joined
customer master frame; ``theme.py`` and ``formatting.py`` hold the visual system; reusable widgets
live in ``components/``, Plotly figure builders in ``charts/``, and one module per page in
``views/`` (named ``views`` rather than ``pages`` on purpose -- see ``app/views/__init__.py``).
"""
