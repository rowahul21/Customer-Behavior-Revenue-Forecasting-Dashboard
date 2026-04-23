"""
app.py  ·  CustomerIQ Dashboard
================================
Run with:  streamlit run app.py
"""

import streamlit as st
from utils.styling import set_page_config, inject_css

set_page_config("Overview")
inject_css()

# ── Sidebar navigation ─────────────────────────────────────────────────────────
st.markdown("""
    <style>
        /* Hide default Streamlit menu (pages) */
        [data-testid="stSidebarNav"] {
            display: none;
        }
        /* Hide hamburger menu */
        #MainMenu {visibility: hidden;}
        /* Hide footer */
        footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

st.sidebar.markdown("""
<div style="text-align:center; padding: 16px 0 24px 0;">
    <div style="font-size:1.2rem; font-weight:800; color:#F1F5F9;">CustomerIQ</div>
    <div style="font-size:0.72rem; color:#64748B; letter-spacing:0.1em;">
        CUSTOMER INTELLIGENCE PLATFORM
    </div>
</div>
""", unsafe_allow_html=True)

page = st.sidebar.radio(
    "Navigation",
    [
        "Overview",
        "Customer Insights",
        "Churn Analysis",
        "Revenue Forecasting",
        "Experiment Lab",
    ],
    label_visibility="collapsed",
)

st.sidebar.markdown("---")
st.sidebar.markdown("""
<div style="font-size:0.7rem; color:#475569; padding-top:8px;">
    Dataset: 25K orders · 8K customers<br>
    Period: Jan 2020 – Mar 2026<br>
    Built with Streamlit + Scikit-learn
</div>
""", unsafe_allow_html=True)

# ── Page routing ───────────────────────────────────────────────────────────────
if   "Overview"          in page: exec(open("pages/01_overview.py", encoding="utf-8").read())
elif "Customer Insights" in page: exec(open("pages/02_customer_insights.py", encoding="utf-8").read())
elif "Churn"             in page: exec(open("pages/03_churn_analysis.py", encoding="utf-8").read())
elif "Revenue"           in page: exec(open("pages/04_revenue_forecasting.py", encoding="utf-8").read())
elif "Experiment"        in page: exec(open("pages/05_experiment_lab.py", encoding="utf-8").read())
