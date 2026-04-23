"""
styling.py
----------
Shared theme tokens, metric card renderer, and Plotly defaults.
Import this at the top of every page to keep visual consistency.
"""

import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

# ── Brand palette ──────────────────────────────────────────────────────────────
COLORS = {
    "primary":    "#4F46E5",   # indigo
    "secondary":  "#7C3AED",   # violet
    "accent":     "#06B6D4",   # cyan
    "success":    "#10B981",   # emerald
    "warning":    "#F59E0B",   # amber
    "danger":     "#EF4444",   # red
    "neutral":    "#64748B",   # slate
    "bg_card":    "#1E293B",   # dark slate card
    "bg_chart":   "rgba(0,0,0,0)",
}

SEGMENT_COLORS = {
    "Champions":          "#10B981",
    "Loyal Customers":    "#4F46E5",
    "Promising":          "#06B6D4",
    "Potential Loyalist": "#8B5CF6",
    "At Risk":            "#F59E0B",
    "Cant Lose Them":     "#EF4444",
    "Needs Attention":    "#64748B",
    "Lost":               "#F472B6",
}

PLOTLY_TEMPLATE = "plotly_dark"


# ── Page config ────────────────────────────────────────────────────────────────
def set_page_config(title: str, icon: str = "🛒"):
    st.set_page_config(
        page_title=f"{title} | CustomerIQ",
        page_icon=icon,
        layout="wide",
        initial_sidebar_state="expanded",
    )


# ── Global CSS ─────────────────────────────────────────────────────────────────
def inject_css():
    st.markdown("""
    <style>
    /* ── Main background ── */
    .stApp { background-color: #0F172A; }

    /* ── Metric cards ── */
    .metric-card {
        background: linear-gradient(135deg, #1E293B 0%, #0F172A 100%);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 20px 24px;
        text-align: center;
    }
    .metric-label {
        color: #94A3B8;
        font-size: 0.78rem;
        font-weight: 600;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin-bottom: 6px;
    }
    .metric-value {
        color: #F1F5F9;
        font-size: 1.9rem;
        font-weight: 700;
        line-height: 1;
    }
    .metric-delta {
        font-size: 0.78rem;
        margin-top: 6px;
        font-weight: 600;
    }
    .delta-up   { color: #10B981; }
    .delta-down { color: #EF4444; }
    .delta-flat { color: #94A3B8; }

    /* ── Section headers ── */
    .section-header {
        color: #E2E8F0;
        font-size: 1.15rem;
        font-weight: 700;
        border-left: 4px solid #4F46E5;
        padding-left: 12px;
        margin: 24px 0 16px 0;
    }

    /* ── Sidebar ── */
    section[data-testid="stSidebar"] { background-color: #1E293B; }
    section[data-testid="stSidebar"] * { color: #CBD5E1 !important; }

    /* ── Hide Streamlit branding ── */
    #MainMenu, footer, header { visibility: hidden; }

    /* ── Tabs ── */
    .stTabs [data-baseweb="tab"] { color: #94A3B8; font-weight: 600; }
    .stTabs [aria-selected="true"] { color: #4F46E5 !important; border-bottom: 2px solid #4F46E5; }

    /* ── DataFrame ── */
    .dataframe { border: 1px solid #334155 !important; }
    </style>
    """, unsafe_allow_html=True)


# ── Metric card HTML ───────────────────────────────────────────────────────────
def metric_card(label: str, value: str, delta: str = "", delta_dir: str = "flat"):
    delta_class = {"up": "delta-up", "down": "delta-down", "flat": "delta-flat"}.get(delta_dir, "delta-flat")
    delta_icon  = {"up": "▲", "down": "▼", "flat": ""}.get(delta_dir, "")
    delta_html  = f'<div class="metric-delta {delta_class}">{delta_icon} {delta}</div>' if delta else ""
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        {delta_html}
    </div>
    """, unsafe_allow_html=True)


def section_header(text: str):
    st.markdown(f'<div class="section-header">{text}</div>', unsafe_allow_html=True)


# ── Plotly layout defaults ─────────────────────────────────────────────────────
def apply_chart_style(fig: go.Figure, height: int = 380) -> go.Figure:
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color="#CBD5E1"),
        margin=dict(l=20, r=20, t=40, b=20),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=11)),
        xaxis=dict(gridcolor="#1E293B", zeroline=False),
        yaxis=dict(gridcolor="#1E293B", zeroline=False),
    )
    return fig
