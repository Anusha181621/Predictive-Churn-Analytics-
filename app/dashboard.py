"""Entry point for the retention dashboard.

    streamlit run app/dashboard.py

Reads ``data/*.csv``, ``outputs/*.csv`` and ``models/*``. There is no database, no ingestion step
and no API: the dashboard displays artefacts the pipeline under ``src/`` has already produced, and
computes nothing of its own except the What-If simulator, which re-runs the real retention layer.

Navigation is declared explicitly with :func:`streamlit.navigation`, and the page modules live in
``app/views/`` rather than ``app/pages/``. Both details are deliberate.

``pages/`` is a magic directory name: when a folder of that name sits beside the entry script,
Streamlit builds *automatic* multi-page navigation from it during start-up, before the script runs.
That has two consequences here. Every page would appear twice -- once properly, once as an empty
auto-discovered script. And the automatic scan builds a ``Page`` from the resolved path of the
entry script, which **fails outright when the project lives on a Windows network share**:
``st.Page`` rejects UNC paths deliberately, to avoid an SMB connection disclosing the server
process's credentials (see ``streamlit/navigation/page.py``). Naming the folder ``views``
sidesteps the magic directory entirely, so navigation is exactly what is declared below and the
app runs from a network drive as happily as from a local one.
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

# The project root, so `import src...` and `import app...` work regardless of where streamlit was
# launched from. Mirrors what the scripts under scripts/ do.
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from app.theme import CSS, register_template  # noqa: E402

st.set_page_config(
    page_title="Customer Churn & Retention",
    page_icon=":material/insights:",
    layout="wide",
    initial_sidebar_state="expanded",
)

register_template()
st.markdown(CSS, unsafe_allow_html=True)

from app.views import (  # noqa: E402
    churn_risk,
    customer_360,
    executive_overview,
    model_performance,
    retention_action_center,
    revenue_at_risk,
    segmentation,
    what_if,
)

PAGES = [
    # The default page is served at "/" and has no separate path of its own. Giving it a
    # `url_path` as well would leave that URL 404-ing into a "Page not found" dialog.
    st.Page(
        executive_overview.render,
        title="Executive Overview",
        icon=":material/dashboard:",
        default=True,
    ),
    st.Page(
        churn_risk.render, title="Churn Risk", icon=":material/warning:", url_path="churn-risk"
    ),
    st.Page(
        revenue_at_risk.render,
        title="Revenue at Risk",
        icon=":material/euro:",
        url_path="revenue-at-risk",
    ),
    st.Page(
        retention_action_center.render,
        title="Retention Action Center",
        icon=":material/campaign:",
        url_path="action-center",
    ),
    st.Page(
        customer_360.render,
        title="Customer 360",
        icon=":material/person_search:",
        url_path="customer-360",
    ),
    # st.Page(
    #     segmentation.render,
    #     title="Customer Segmentation",
    #     icon=":material/donut_large:",
    #     url_path="segmentation",
    # ),
    # st.Page(
    #     what_if.render, title="What-If Simulator", icon=":material/tune:", url_path="what-if"
    # ),
    # st.Page(
    #     model_performance.render,
    #     title="Model Performance",
    #     icon=":material/analytics:",
    #     url_path="model-performance",
    # ),
]


def main() -> None:
    page = st.navigation(PAGES, position="sidebar")
    page.run()

    # Drawn after the page so it sits below the page's own filters rather than above them.
    with st.sidebar:
        st.divider()
        st.caption(
            "Source of truth: the four CSVs in `data/`. "
            "Every figure is read from the pipeline's own artefacts."
        )


main()
