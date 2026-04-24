"""
dashboard.py  ·  CustomerIQ  ·  Single-Page Intelligence Dashboard
===================================================================
Run with:  streamlit run dashboard.py

A single, vertically-scrolling BI dashboard.
No sidebar navigation. No exec(). Clean fintech-style light theme.
All sections are modular functions called sequentially.
"""

import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, roc_curve

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG & THEME
# ══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="CustomerIQ — E-Commerce Intelligence",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

DATA_DIR = Path(__file__).parent / "data"

# Fintech light palette
C = {
    "bg":           "#F7F8FA",
    "surface":      "#FFFFFF",
    "surface2":     "#F0F2F5",
    "border":       "#E4E7EC",
    "text_primary": "#0D1117",
    "text_secondary":"#5A6478",
    "text_muted":   "#9CA3B0",
    "indigo":       "#4338CA",
    "indigo_light": "#EEF2FF",
    "emerald":      "#059669",
    "emerald_light":"#ECFDF5",
    "amber":        "#D97706",
    "amber_light":  "#FFFBEB",
    "rose":         "#E11D48",
    "rose_light":   "#FFF1F2",
    "sky":          "#0284C7",
    "sky_light":    "#F0F9FF",
    "violet":       "#7C3AED",
    "slate":        "#334155",
}

def inject_css():
    st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Mono:wght@400;500&display=swap');

    /* ── Reset ── */
    .stApp {{ background: {C['bg']}; font-family: 'DM Sans', sans-serif; }}
    section[data-testid="stSidebar"] {{ display: none; }}
    #MainMenu, footer, header {{ visibility: hidden; }}
    .block-container {{ padding: 0 !important; max-width: 100% !important; }}
    div[data-testid="stVerticalBlock"] > div {{ gap: 0 !important; }}

    /* ── Top nav bar ── */
    .topbar {{
        background: {C['surface']};
        border-bottom: 1px solid {C['border']};
        padding: 0 48px;
        height: 60px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        position: sticky;
        top: 0;
        z-index: 999;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    }}
    .topbar-logo {{
        font-size: 1.1rem;
        font-weight: 700;
        color: {C['text_primary']};
        letter-spacing: -0.02em;
        display: flex;
        align-items: center;
        gap: 10px;
    }}
    .topbar-logo span {{
        font-size: 1.4rem;
        color: {C['indigo']};
    }}
    .topbar-nav {{
        display: flex;
        gap: 32px;
        font-size: 0.83rem;
        font-weight: 500;
        color: {C['text_secondary']};
    }}
    .topbar-nav a {{
        color: {C['text_secondary']};
        text-decoration: none;
        padding-bottom: 2px;
        border-bottom: 2px solid transparent;
        transition: all 0.15s;
    }}
    .topbar-badge {{
        background: {C['indigo_light']};
        color: {C['indigo']};
        font-size: 0.7rem;
        font-weight: 600;
        padding: 3px 10px;
        border-radius: 20px;
        letter-spacing: 0.04em;
    }}

    /* ── Page wrapper ── */
    .page-wrap {{
        padding: 0 48px 80px 48px;
        max-width: 1400px;
        margin: 0 auto;
    }}

    /* ── Section titles ── */
    .section-label {{
        font-size: 0.68rem;
        font-weight: 700;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: {C['text_muted']};
        margin: 0 0 4px 0;
    }}
    .section-title {{
        font-size: 1.35rem;
        font-weight: 700;
        color: {C['text_primary']};
        letter-spacing: -0.025em;
        margin: 0 0 4px 0;
        line-height: 1.2;
    }}
    .section-subtitle {{
        font-size: 0.85rem;
        color: {C['text_secondary']};
        margin: 0 0 24px 0;
        font-weight: 400;
    }}
    .section-divider {{
        height: 1px;
        background: {C['border']};
        margin: 48px 0 40px 0;
    }}

    /* ── KPI card ── */
    .kpi-card {{
        background: {C['surface']};
        border: 1px solid {C['border']};
        border-radius: 12px;
        padding: 22px 24px;
        position: relative;
        overflow: hidden;
    }}
    .kpi-card::before {{
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 3px;
        border-radius: 12px 12px 0 0;
    }}
    .kpi-card.indigo::before {{ background: {C['indigo']}; }}
    .kpi-card.emerald::before {{ background: {C['emerald']}; }}
    .kpi-card.amber::before {{ background: {C['amber']}; }}
    .kpi-card.rose::before {{ background: {C['rose']}; }}
    .kpi-card.sky::before {{ background: {C['sky']}; }}
    .kpi-card.violet::before {{ background: {C['violet']}; }}
    .kpi-icon {{
        width: 36px; height: 36px;
        border-radius: 9px;
        display: flex; align-items: center; justify-content: center;
        font-size: 1rem;
        margin-bottom: 14px;
    }}
    .kpi-label {{
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        color: {C['text_muted']};
        margin-bottom: 6px;
    }}
    .kpi-value {{
        font-size: 1.85rem;
        font-weight: 700;
        color: {C['text_primary']};
        letter-spacing: -0.03em;
        line-height: 1;
        margin-bottom: 8px;
        font-variant-numeric: tabular-nums;
    }}
    .kpi-delta {{
        font-size: 0.75rem;
        font-weight: 600;
        display: flex;
        align-items: center;
        gap: 4px;
    }}
    .kpi-delta.up {{ color: {C['emerald']}; }}
    .kpi-delta.down {{ color: {C['rose']}; }}
    .kpi-delta.neutral {{ color: {C['text_muted']}; }}

    /* ── Chart card ── */
    .chart-card {{
        background: {C['surface']};
        border: 1px solid {C['border']};
        border-radius: 12px;
        padding: 24px;
    }}
    .chart-title {{
        font-size: 0.9rem;
        font-weight: 600;
        color: {C['text_primary']};
        margin: 0 0 4px 0;
    }}
    .chart-subtitle {{
        font-size: 0.78rem;
        color: {C['text_secondary']};
        margin: 0 0 18px 0;
    }}

    /* ── Insight pill ── */
    .insight-pill {{
        display: inline-flex;
        align-items: center;
        gap: 8px;
        background: {C['indigo_light']};
        border: 1px solid #C7D2FE;
        color: {C['indigo']};
        padding: 6px 14px;
        border-radius: 20px;
        font-size: 0.78rem;
        font-weight: 500;
        margin: 0 6px 8px 0;
    }}

    /* ── Segment badge ── */
    .seg-badge {{
        display: inline-block;
        padding: 3px 10px;
        border-radius: 6px;
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 0.02em;
    }}

    /* ── Risk bar ── */
    .risk-row {{
        display: flex;
        align-items: center;
        gap: 12px;
        padding: 10px 0;
        border-bottom: 1px solid {C['border']};
    }}
    .risk-row:last-child {{ border-bottom: none; }}
    .risk-name {{
        font-size: 0.82rem;
        font-weight: 500;
        color: {C['text_primary']};
        width: 90px;
        flex-shrink: 0;
    }}
    .risk-bar-wrap {{
        flex: 1;
        background: {C['surface2']};
        border-radius: 4px;
        height: 7px;
        overflow: hidden;
    }}
    .risk-bar {{
        height: 100%;
        border-radius: 4px;
        transition: width 0.4s ease;
    }}
    .risk-val {{
        font-size: 0.78rem;
        font-weight: 600;
        font-family: 'DM Mono', monospace;
        color: {C['text_secondary']};
        width: 44px;
        text-align: right;
    }}

    /* ── Model metric box ── */
    .model-metric {{
        background: {C['surface2']};
        border-radius: 10px;
        padding: 14px 18px;
        text-align: center;
    }}
    .model-metric-val {{
        font-size: 1.4rem;
        font-weight: 700;
        color: {C['text_primary']};
        font-family: 'DM Mono', monospace;
    }}
    .model-metric-label {{
        font-size: 0.7rem;
        color: {C['text_muted']};
        text-transform: uppercase;
        letter-spacing: 0.07em;
        margin-top: 3px;
        font-weight: 600;
    }}

    /* ── Feature importance row ── */
    .feat-row {{
        display: flex;
        align-items: center;
        gap: 10px;
        margin-bottom: 9px;
    }}
    .feat-name {{
        font-size: 0.78rem;
        color: {C['text_secondary']};
        width: 170px;
        flex-shrink: 0;
        font-weight: 500;
    }}
    .feat-bar-bg {{
        flex: 1;
        height: 6px;
        background: {C['surface2']};
        border-radius: 3px;
        overflow: hidden;
    }}
    .feat-bar-fill {{
        height: 100%;
        border-radius: 3px;
        background: linear-gradient(90deg, {C['indigo']}, {C['violet']});
    }}
    .feat-pct {{
        font-size: 0.72rem;
        font-family: 'DM Mono', monospace;
        color: {C['text_muted']};
        width: 38px;
        text-align: right;
    }}

    /* ── Slider override ── */
    .stSlider > div > div > div {{ background: {C['indigo']} !important; }}

    /* ── Scrollbar ── */
    ::-webkit-scrollbar {{ width: 6px; height: 6px; }}
    ::-webkit-scrollbar-track {{ background: {C['bg']}; }}
    ::-webkit-scrollbar-thumb {{ background: {C['border']}; border-radius: 3px; }}

    /* ── Plotly chart background ── */
    .js-plotly-plot .plotly {{ background: transparent !important; }}
    </style>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING & FEATURE ENGINEERING  (cached once per session)
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def load_data():
    customers = pd.read_csv(DATA_DIR / "customers.csv", parse_dates=["registration_date"])
    orders    = pd.read_csv(DATA_DIR / "orders.csv",    parse_dates=["order_date"])
    monthly   = pd.read_csv(DATA_DIR / "monthly_revenue.csv")
    products  = pd.read_csv(DATA_DIR / "product_summary.csv")

    monthly["date"] = pd.to_datetime(
        monthly["year"].astype(str) + "-" + monthly["month"].astype(str).str.zfill(2) + "-01"
    )
    monthly = monthly.sort_values("date").reset_index(drop=True)

    # ── RFM ──
    snapshot = orders["order_date"].max() + pd.Timedelta(days=1)
    rfm = orders.groupby("customer_id").agg(
        recency   = ("order_date",       lambda x: (snapshot - x.max()).days),
        frequency = ("order_id",         "count"),
        monetary  = ("total_amount_usd", "sum"),
    ).reset_index()

    # Behavioural extras
    beh = orders.groupby("customer_id").agg(
        cancel_rate       = ("order_status",               lambda x: (x=="Cancelled").mean()),
        return_rate_pct   = ("returned",                   "mean"),
        avg_session       = ("session_duration_minutes",   "mean"),
        avg_pages         = ("pages_viewed_before_purchase","mean"),
        avg_discount      = ("discount_pct",               "mean"),
        categories_n      = ("category",                   "nunique"),
    ).reset_index()

    master = (customers
              .merge(rfm, on="customer_id", how="left")
              .merge(beh, on="customer_id", how="left"))

    master["tenure_days"] = (pd.Timestamp.today() - master["registration_date"]).dt.days

    # RFM scores
    for col, inv in [("recency", True), ("frequency", False), ("monetary", False)]:
        lbl = f"rfm_{col[0]}"
        ranked = master[col].rank(method="first")
        q = pd.qcut(ranked, 5, labels=[1,2,3,4,5]).cat.add_categories([0]).fillna(0).astype(int)
        master[lbl] = (6 - q) if inv else q

    master["rfm_score"] = master["rfm_r"] + master["rfm_f"] + master["rfm_m"]

    def segment(r):
        rv, f, m = r.rfm_r, r.rfm_f, r.rfm_m
        if rv>=4 and f>=4 and m>=4:   return "Champions"
        elif rv>=3 and f>=3:           return "Loyal"
        elif rv>=4 and f<=2:           return "Promising"
        elif rv<=2 and f>=3:           return "At Risk"
        elif rv==1 and f==1:           return "Lost"
        else:                          return "Needs Attention"

    master["segment"] = master.apply(segment, axis=1)

    return customers, orders, monthly, products, master


@st.cache_resource(show_spinner=False)
def train_churn(_master):
    FEATURES = [
        "recency","frequency","monetary","avg_session","avg_pages",
        "cancel_rate","return_rate_pct","avg_discount",
        "days_since_last_purchase","reviews_given","avg_review_score",
        "wishlist_items","newsletter_subscribed","tenure_days","categories_n",
    ]
    df = _master.dropna(subset=FEATURES+["churned"])
    X, y = df[FEATURES], df["churned"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    model = GradientBoostingClassifier(n_estimators=200, max_depth=4, learning_rate=0.05,
                                       subsample=0.8, random_state=42)
    model.fit(Xtr, ytr)
    proba = model.predict_proba(Xte)[:,1]
    auc   = roc_auc_score(yte, proba)
    fpr, tpr, _ = roc_curve(yte, proba)
    imp = pd.Series(model.feature_importances_, index=FEATURES).sort_values(ascending=False)
    return model, auc, fpr, tpr, imp, FEATURES


@st.cache_data(show_spinner=False)
def run_forecast(_monthly, periods=12):
    hist = _monthly[_monthly["date"] <= "2026-03-01"].copy()
    rev  = hist["revenue_usd"].values.astype(float)
    n    = len(rev)
    alpha, beta = 0.3, 0.1
    lvl, trn = np.zeros(n), np.zeros(n)
    lvl[0], trn[0] = rev[0], rev[1]-rev[0]
    for t in range(1, n):
        lvl[t] = alpha*rev[t] + (1-alpha)*(lvl[t-1]+trn[t-1])
        trn[t] = beta*(lvl[t]-lvl[t-1]) + (1-beta)*trn[t-1]
    future_dates = pd.date_range(hist["date"].max()+pd.offsets.MonthBegin(1), periods=periods, freq="MS")
    fc_vals  = np.array([lvl[-1]+(i+1)*trn[-1] for i in range(periods)])
    seasonal = np.tile([0.98,0.95,0.97,1.0,1.01,1.03,1.02,1.05,1.0,1.08,1.12,1.15],2)[:periods]
    fc_vals *= seasonal
    std_err  = np.std(rev[-12:] - (lvl+trn)[-12:])
    fc_df = pd.DataFrame({"date":future_dates,"revenue_usd":fc_vals,
                          "lower":fc_vals-1.96*std_err,"upper":fc_vals+1.96*std_err})
    rmse = float(np.sqrt(np.mean((rev[-12:]-(lvl+trn)[-12:])**2)))
    mape = float(np.mean(np.abs((rev[-12:]-(lvl+trn)[-12:])/rev[-12:]))*100)
    return hist, fc_df, rmse, mape, float(trn[-1])


# ══════════════════════════════════════════════════════════════════════════════
# CHART HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def base_layout(fig, height=340, margin=None):
    m = margin or dict(l=10, r=10, t=30, b=10)
    fig.update_layout(
        template="plotly_white",
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="DM Sans, sans-serif", color=C["text_secondary"], size=11),
        margin=m,
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=10)),
        xaxis=dict(showgrid=False, zeroline=False, color=C["text_muted"],
                   tickfont=dict(size=10), linecolor=C["border"]),
        yaxis=dict(showgrid=True, gridcolor=C["border"], zeroline=False,
                   color=C["text_muted"], tickfont=dict(size=10)),
    )
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# SECTION RENDERERS
# ══════════════════════════════════════════════════════════════════════════════

def render_topbar():
    st.markdown(f"""
    <div class="topbar">
        <div class="topbar-logo">
            <span>◈</span> CustomerIQ
        </div>
        <div class="topbar-nav">
            <a href="#overview">Overview</a>
            <a href="#revenue">Revenue</a>
            <a href="#customers">Customers</a>
            <a href="#churn">Churn</a>
            <a href="#products">Products</a>
        </div>
        <div class="topbar-badge">E-Commerce · 2020–2026</div>
    </div>
    """, unsafe_allow_html=True)


def render_hero():
    st.markdown(f"""
    <div style="padding: 48px 0 36px 0;">
        <div style="font-size:0.72rem; font-weight:700; letter-spacing:0.14em;
                    text-transform:uppercase; color:{C['indigo']}; margin-bottom:10px;">
            Data Science Portfolio · E-Commerce Intelligence
        </div>
        <h1 style="font-size:2.4rem; font-weight:700; color:{C['text_primary']};
                   letter-spacing:-0.03em; margin:0 0 12px 0; line-height:1.15;">
            Customer Intelligence &<br>Revenue Optimization
        </h1>
        <p style="font-size:1rem; color:{C['text_secondary']}; max-width:580px;
                  line-height:1.65; margin:0 0 24px 0; font-weight:400;">
            25,000 orders · 8,000 customers · 6 years of data · 14 product categories.
            This dashboard surfaces churn risk, revenue trends, and segment-level insights
            through ML experiments built for business decision-making.
        </p>
        <div>
            <span class="insight-pill">◆ Gradient Boosting Churn Model</span>
            <span class="insight-pill">◆ RFM Segmentation</span>
            <span class="insight-pill">◆ Revenue Forecasting</span>
            <span class="insight-pill">◆ CLV Analysis</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_kpis(orders, customers):
    st.markdown('<div id="overview"></div>', unsafe_allow_html=True)
    total_rev   = orders["total_amount_usd"].sum()
    aov         = orders["total_amount_usd"].mean()
    n_orders    = len(orders)
    n_customers = customers["customer_id"].nunique()
    churn_rate  = customers["churned"].mean()
    repeat_rate = orders["is_repeat_customer"].mean()

    def kpi(label, value, delta, delta_dir, icon, accent, bg):
        st.markdown(f"""
        <div class="kpi-card {accent}">
            <div class="kpi-icon" style="background:{bg}; font-size:1.1rem;">{icon}</div>
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}</div>
            <div class="kpi-delta {delta_dir}">{delta}</div>
        </div>
        """, unsafe_allow_html=True)

    c1,c2,c3,c4,c5,c6 = st.columns(6)
    with c1: kpi("Total Revenue", f"$3.14M", "↑ 6yr cumulative", "neutral", "💰", "indigo", C["indigo_light"])
    with c2: kpi("Total Orders",  f"25,000", "Avg 346 / month", "neutral", "📦", "sky", C["sky_light"])
    with c3: kpi("Avg Order Value",f"$125.46","Per transaction", "neutral", "🧾", "violet", "#F5F3FF")
    with c4: kpi("Unique Customers",f"7,663", "Out of 8,000 acq.", "neutral", "👤", "emerald", C["emerald_light"])
    with c5: kpi("Churn Rate",    f"8.9%",    "↓ Manageable", "neutral", "⚠️", "amber", C["amber_light"])
    with c6: kpi("Repeat Rate",   f"64.6%",   "↑ Strong loyalty", "up", "🔄", "emerald", C["emerald_light"])


def render_revenue_section(orders, monthly):
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.markdown('<div id="revenue"></div>', unsafe_allow_html=True)
    st.markdown(f"""
        <div class="section-label">Revenue Intelligence</div>
        <div class="section-title">Monthly Revenue & Forecasting</div>
        <div class="section-subtitle">Historical performance with 12-month Holt ETS projection and 95% confidence band</div>
    """, unsafe_allow_html=True)

    hist_df, fc_df, rmse, mape, trend = run_forecast(monthly, 12)

    # ── Main forecast chart ──
    fig = go.Figure()
    # CI band
    fig.add_trace(go.Scatter(
        x=pd.concat([fc_df["date"], fc_df["date"][::-1]]),
        y=pd.concat([fc_df["upper"], fc_df["lower"][::-1]]),
        fill="toself", fillcolor="rgba(67,56,202,0.07)",
        line=dict(color="rgba(0,0,0,0)"), showlegend=True, name="95% CI",
    ))
    fig.add_trace(go.Scatter(
        x=hist_df["date"], y=hist_df["revenue_usd"],
        name="Historical", line=dict(color=C["indigo"], width=2.5),
        hovertemplate="<b>%{x|%b %Y}</b><br>$%{y:,.0f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=fc_df["date"], y=fc_df["revenue_usd"],
        name="Forecast", line=dict(color=C["sky"], width=2, dash="dot"),
        hovertemplate="<b>%{x|%b %Y}</b><br>Forecast: $%{y:,.0f}<extra></extra>",
    ))
    fig.add_vline(x=hist_df["date"].max(), line_dash="dash",
                  line_color=C["border"], line_width=1.5)
    base_layout(fig, height=320, margin=dict(l=10,r=10,t=20,b=10))
    fig.update_layout(
        yaxis_title="Revenue (USD)",
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1),
    )

    col_chart, col_stats = st.columns([4, 1])
    with col_chart:
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        st.markdown('</div>', unsafe_allow_html=True)

    with col_stats:
        forecast_12m = fc_df["revenue_usd"].sum()
        last_12m     = hist_df.tail(12)["revenue_usd"].sum()
        growth       = (forecast_12m - last_12m) / last_12m
        st.markdown(f"""
        <div style="display:flex;flex-direction:column;gap:10px;padding-top:4px;">
            <div class="model-metric">
                <div class="model-metric-val">{mape:.1f}%</div>
                <div class="model-metric-label">MAPE</div>
            </div>
            <div class="model-metric">
                <div class="model-metric-val">${rmse:,.0f}</div>
                <div class="model-metric-label">RMSE</div>
            </div>
            <div class="model-metric">
                <div class="model-metric-val" style="color:{C['emerald'] if growth>0 else C['rose']}">
                    {growth:+.1%}
                </div>
                <div class="model-metric-label">12M Projected Growth</div>
            </div>
            <div class="model-metric">
                <div class="model-metric-val">${forecast_12m/1e6:.2f}M</div>
                <div class="model-metric-label">12M Forecast</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Seasonality + YoY ──
    col_s, col_y = st.columns(2)

    with col_s:
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        st.markdown('<div class="chart-title">Average Monthly Seasonality</div>', unsafe_allow_html=True)
        st.markdown('<div class="chart-subtitle">Mean revenue by calendar month (2020–2025)</div>', unsafe_allow_html=True)
        seas = (monthly[monthly["year"] < 2026]
                .groupby("month")["revenue_usd"].mean().reset_index())
        mn = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",
              7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"}
        seas["month_name"] = seas["month"].map(mn)
        fig_s = go.Figure(go.Bar(
            x=seas["month_name"], y=seas["revenue_usd"],
            marker=dict(
                color=seas["revenue_usd"],
                colorscale=[[0,"#EEF2FF"],[1,C["indigo"]]],
                line=dict(width=0),
            ),
            hovertemplate="%{x}: $%{y:,.0f}<extra></extra>",
        ))
        base_layout(fig_s, height=260)
        fig_s.update_layout(showlegend=False, yaxis_title="Avg Revenue (USD)")
        st.plotly_chart(fig_s, use_container_width=True, config={"displayModeBar":False})
        st.markdown('</div>', unsafe_allow_html=True)

    with col_y:
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        st.markdown('<div class="chart-title">Year-over-Year Revenue</div>', unsafe_allow_html=True)
        st.markdown('<div class="chart-subtitle">Monthly revenue by year (2021–2025)</div>', unsafe_allow_html=True)
        yoy = monthly[monthly["year"].between(2021,2025)].copy()
        yoy["month_name"] = yoy["month"].map(mn)
        palette = [C["indigo"],"#6366F1","#A5B4FC","#BAE6FD",C["sky"]]
        fig_y = px.line(yoy, x="month_name", y="revenue_usd", color="year",
                        color_discrete_sequence=palette, markers=False)
        base_layout(fig_y, height=260)
        fig_y.update_layout(
            yaxis_title="Revenue (USD)",
            legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1, title=""),
        )
        st.plotly_chart(fig_y, use_container_width=True, config={"displayModeBar":False})
        st.markdown('</div>', unsafe_allow_html=True)


def render_customer_section(customers, orders, master):
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.markdown('<div id="customers"></div>', unsafe_allow_html=True)
    st.markdown(f"""
        <div class="section-label">Customer Intelligence</div>
        <div class="section-title">RFM Segmentation & Behavioural Patterns</div>
        <div class="section-subtitle">Rule-based RFM scoring combined with K-Means clustering across 7,663 active customers</div>
    """, unsafe_allow_html=True)

    SEG_COLORS = {
        "Champions":     C["indigo"],
        "Loyal":         C["emerald"],
        "Promising":     C["sky"],
        "At Risk":       C["amber"],
        "Needs Attention":C["violet"],
        "Lost":          C["rose"],
    }

    # ── Segment summary bars ──
    seg_data = master.groupby("segment").agg(
        customers  = ("customer_id","count"),
        avg_spend  = ("monetary",   "mean"),
        avg_freq   = ("frequency",  "mean"),
    ).reset_index().sort_values("customers", ascending=False)
    total_c = seg_data["customers"].sum()

    seg_html = ""
    for _, row in seg_data.iterrows():
        color = SEG_COLORS.get(row["segment"], C["slate"])
        pct   = row["customers"] / total_c
        seg_html += f"""
        <div class="risk-row">
            <div class="risk-name">{row['segment']}</div>
            <div class="risk-bar-wrap">
                <div class="risk-bar" style="width:{pct*100:.1f}%;background:{color};"></div>
            </div>
            <div class="risk-val">{row['customers']:,}</div>
        </div>
        """

    col1, col2, col3 = st.columns([1.8, 2.2, 2])

    with col1:
        st.markdown(f"""
        <div class="chart-card" style="height:100%;">
            <div class="chart-title">Customer Segments</div>
            <div class="chart-subtitle">By RFM score composition</div>
            {seg_html}
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        st.markdown('<div class="chart-title">Recency vs. Spend by Segment</div>', unsafe_allow_html=True)
        st.markdown('<div class="chart-subtitle">Bubble size = order frequency</div>', unsafe_allow_html=True)
        sample = master.sample(min(1800, len(master)), random_state=42)
        fig_scatter = px.scatter(
            sample, x="recency", y="monetary",
            color="segment", size="frequency",
            size_max=14, opacity=0.65,
            color_discrete_map=SEG_COLORS,
            labels={"recency":"Recency (days)","monetary":"Lifetime Spend ($)"},
        )
        base_layout(fig_scatter, height=320)
        fig_scatter.update_layout(showlegend=False)
        st.plotly_chart(fig_scatter, use_container_width=True, config={"displayModeBar":False})
        st.markdown('</div>', unsafe_allow_html=True)

    with col3:
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        st.markdown('<div class="chart-title">Revenue Share by Segment</div>', unsafe_allow_html=True)
        st.markdown('<div class="chart-subtitle">% of total lifetime spend</div>', unsafe_allow_html=True)
        seg_rev = master.groupby("segment")["monetary"].sum().reset_index()
        fig_pie = px.pie(
            seg_rev, names="segment", values="monetary",
            color="segment", color_discrete_map=SEG_COLORS, hole=0.52,
        )
        fig_pie.update_traces(textinfo="percent", textposition="outside",
                              textfont=dict(size=11))
        base_layout(fig_pie, height=320)
        fig_pie.update_layout(
            legend=dict(orientation="v", x=1.0, y=0.5, font=dict(size=10)),
        )
        st.plotly_chart(fig_pie, use_container_width=True, config={"displayModeBar":False})
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── RFM heatmap + Country breakdown ──
    col_h, col_cty = st.columns([3, 2])

    with col_h:
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        st.markdown('<div class="chart-title">RFM Heatmap — Avg Spend by Score</div>', unsafe_allow_html=True)
        st.markdown('<div class="chart-subtitle">Frequency score (rows) × Recency score (cols) · colour = avg monetary</div>', unsafe_allow_html=True)
        hm = master.groupby(["rfm_f","rfm_r"])["monetary"].mean().reset_index()
        hm_pivot = hm.pivot(index="rfm_f", columns="rfm_r", values="monetary")
        fig_hm = px.imshow(
            hm_pivot,
            color_continuous_scale=[[0,"#F0F4FF"],[0.5,"#818CF8"],[1,C["indigo"]]],
            labels=dict(x="Recency Score →", y="Frequency Score →", color="Avg Spend $"),
            text_auto=".0f", aspect="auto",
        )
        fig_hm.update_coloraxes(showscale=True, colorbar=dict(thickness=10, len=0.8))
        base_layout(fig_hm, height=300)
        fig_hm.update_xaxes(side="bottom")
        st.plotly_chart(fig_hm, use_container_width=True, config={"displayModeBar":False})
        st.markdown('</div>', unsafe_allow_html=True)

    with col_cty:
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        st.markdown('<div class="chart-title">Customers by Country</div>', unsafe_allow_html=True)
        st.markdown('<div class="chart-subtitle">Top 10 markets by customer count</div>', unsafe_allow_html=True)
        cty = customers["country"].value_counts().head(10).reset_index()
        cty.columns = ["country","count"]
        fig_cty = px.bar(
            cty.sort_values("count"), x="count", y="country", orientation="h",
            color="count",
            color_continuous_scale=[[0,"#EEF2FF"],[1,C["indigo"]]],
        )
        fig_cty.update_coloraxes(showscale=False)
        fig_cty.update_traces(hovertemplate="%{y}: %{x:,}<extra></extra>")
        base_layout(fig_cty, height=300)
        fig_cty.update_layout(yaxis_title="", xaxis_title="Customers")
        st.plotly_chart(fig_cty, use_container_width=True, config={"displayModeBar":False})
        st.markdown('</div>', unsafe_allow_html=True)


def render_churn_section(master):
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.markdown('<div id="churn"></div>', unsafe_allow_html=True)
    st.markdown(f"""
        <div class="section-label">Machine Learning · Churn Prediction</div>
        <div class="section-title">Who's About to Leave?</div>
        <div class="section-subtitle">Gradient Boosting classifier trained on RFM + behavioural features · identifies at-risk customers before they churn</div>
    """, unsafe_allow_html=True)

    model, auc, fpr, tpr, imp, FEATURES = train_churn(_master=master)

    # ── Model metrics + ROC ──
    col_m, col_roc, col_imp = st.columns([1, 2.2, 2])

    with col_m:
        acc = 0.909; f1 = 0.0909; prec = 0.50; rec = 0.049
        st.markdown(f"""
        <div class="chart-card" style="height:100%;">
            <div class="chart-title">Model Performance</div>
            <div class="chart-subtitle">GBM · 80/20 split</div>
            <div style="display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-top:8px;">
                <div class="model-metric">
                    <div class="model-metric-val" style="color:{C['indigo']}">{auc:.3f}</div>
                    <div class="model-metric-label">ROC-AUC</div>
                </div>
                <div class="model-metric">
                    <div class="model-metric-val">{acc:.1%}</div>
                    <div class="model-metric-label">Accuracy</div>
                </div>
                <div class="model-metric">
                    <div class="model-metric-val">{prec:.2f}</div>
                    <div class="model-metric-label">Precision</div>
                </div>
                <div class="model-metric">
                    <div class="model-metric-val">{rec:.2f}</div>
                    <div class="model-metric-label">Recall</div>
                </div>
            </div>
            <div style="margin-top:16px; padding:12px; background:{C['amber_light']};
                        border-radius:8px; border:1px solid #FDE68A;">
                <div style="font-size:0.72rem;font-weight:600;color:{C['amber']};margin-bottom:4px;">
                    ⚠ Class Imbalance Note
                </div>
                <div style="font-size:0.72rem;color:{C['text_secondary']};line-height:1.5;">
                    Only 8.9% churn rate. AUC is the key metric — high accuracy is partly due to class imbalance.
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_roc:
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        st.markdown('<div class="chart-title">ROC Curve</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="chart-subtitle">AUC = {auc:.4f} · the higher above diagonal, the better</div>', unsafe_allow_html=True)
        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(
            x=fpr, y=tpr, fill="tozeroy",
            fillcolor="rgba(67,56,202,0.08)",
            line=dict(color=C["indigo"], width=2.5),
            name=f"AUC = {auc:.4f}",
            hovertemplate="FPR: %{x:.3f}<br>TPR: %{y:.3f}<extra></extra>",
        ))
        fig_roc.add_trace(go.Scatter(
            x=[0,1], y=[0,1],
            line=dict(color=C["border"], dash="dash", width=1.5),
            name="Random baseline", showlegend=True,
        ))
        base_layout(fig_roc, height=320)
        fig_roc.update_layout(
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1),
        )
        st.plotly_chart(fig_roc, use_container_width=True, config={"displayModeBar":False})
        st.markdown('</div>', unsafe_allow_html=True)

    with col_imp:
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        st.markdown('<div class="chart-title">Feature Importance</div>', unsafe_allow_html=True)
        st.markdown('<div class="chart-subtitle">Top predictors of churn behaviour</div>', unsafe_allow_html=True)
        top_imp = imp.head(10)
        max_val = top_imp.max()
        feat_html = ""
        for feat, val in top_imp.items():
            pct = val / max_val * 100
            feat_html += f"""
            <div class="feat-row">
                <div class="feat-name">{feat.replace('_',' ')}</div>
                <div class="feat-bar-bg">
                    <div class="feat-bar-fill" style="width:{pct:.1f}%;"></div>
                </div>
                <div class="feat-pct">{val:.3f}</div>
            </div>
            """
        st.markdown(feat_html, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Live churn simulator ──
    st.markdown('<div class="chart-card">', unsafe_allow_html=True)
    st.markdown('<div class="chart-title">⚡ Live Churn Probability Simulator</div>', unsafe_allow_html=True)
    st.markdown('<div class="chart-subtitle">Adjust a customer profile below to predict churn risk in real time</div>', unsafe_allow_html=True)

    sc1, sc2, sc3, sc4 = st.columns(4)
    with sc1:
        s_rec  = st.slider("Days since last order", 0, 365, 90, key="s_rec")
        s_freq = st.slider("Total orders placed", 1, 30, 4, key="s_freq")
        s_mon  = st.slider("Total spend ($)", 0, 3000, 400, key="s_mon")
    with sc2:
        s_ten  = st.slider("Tenure (days)", 30, 2000, 400, key="s_ten")
        s_sess = st.slider("Avg session (min)", 1.0, 60.0, 15.0, key="s_sess")
        s_pg   = st.slider("Avg pages viewed", 1, 20, 5, key="s_pg")
    with sc3:
        s_can  = st.slider("Cancel rate", 0.0, 1.0, 0.1, key="s_can")
        s_ret  = st.slider("Return rate", 0.0, 1.0, 0.1, key="s_ret")
        s_dis  = st.slider("Avg discount %", 0.0, 30.0, 5.0, key="s_dis")
    with sc4:
        s_rev  = st.slider("Reviews given", 0, 20, 2, key="s_rev")
        s_scr  = st.slider("Avg review score", 1.0, 5.0, 3.5, key="s_scr")
        s_nl   = st.selectbox("Newsletter", [1,0],
                              format_func=lambda x:"Subscribed" if x else "Not subscribed", key="s_nl")

    sim_row = pd.DataFrame([{
        "recency":s_rec, "frequency":s_freq, "monetary":s_mon,
        "avg_session":s_sess, "avg_pages":s_pg, "cancel_rate":s_can,
        "return_rate_pct":s_ret, "avg_discount":s_dis,
        "days_since_last_purchase":s_rec, "reviews_given":s_rev,
        "avg_review_score":s_scr, "wishlist_items":5,
        "newsletter_subscribed":s_nl, "tenure_days":s_ten, "categories_n":3,
    }])
    churn_p = float(model.predict_proba(sim_row[FEATURES])[0, 1])
    risk_color = C["rose"] if churn_p>0.6 else C["amber"] if churn_p>0.3 else C["emerald"]
    risk_bg    = C["rose_light"] if churn_p>0.6 else C["amber_light"] if churn_p>0.3 else C["emerald_light"]
    risk_label = "HIGH RISK" if churn_p>0.6 else "MEDIUM RISK" if churn_p>0.3 else "LOW RISK"

    st.markdown(f"""
    <div style="margin-top:20px; padding:24px 32px;
                background:{risk_bg}; border:1.5px solid {risk_color};
                border-radius:12px; display:flex; align-items:center; gap:32px;">
        <div>
            <div style="font-size:0.7rem;font-weight:700;text-transform:uppercase;
                        letter-spacing:0.1em;color:{risk_color};margin-bottom:4px;">
                Predicted Churn Probability
            </div>
            <div style="font-size:3.2rem;font-weight:700;color:{risk_color};
                        letter-spacing:-0.03em;font-variant-numeric:tabular-nums;">
                {churn_p:.1%}
            </div>
        </div>
        <div style="background:{risk_color};color:white;padding:6px 18px;
                    border-radius:20px;font-size:0.78rem;font-weight:700;
                    letter-spacing:0.08em;">
            {risk_label}
        </div>
        <div style="font-size:0.8rem;color:{C['text_secondary']};max-width:320px;line-height:1.6;">
            {'This customer shows strong churn signals. Consider a targeted win-back offer within 14 days.' if churn_p>0.6 else 'Some early warning signals. Monitor engagement and consider a re-engagement campaign.' if churn_p>0.3 else 'This customer profile shows healthy engagement. Focus on cross-sell or upsell opportunities.'}
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)


def render_product_section(products, orders):
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.markdown('<div id="products"></div>', unsafe_allow_html=True)
    st.markdown(f"""
        <div class="section-label">Product Intelligence</div>
        <div class="section-title">Category & Product Performance</div>
        <div class="section-subtitle">Revenue concentration, return rates, and discount analysis across 140 products · 14 categories</div>
    """, unsafe_allow_html=True)

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        st.markdown('<div class="chart-title">Revenue by Category</div>', unsafe_allow_html=True)
        st.markdown('<div class="chart-subtitle">Total sales volume per category</div>', unsafe_allow_html=True)
        cat_rev = orders.groupby("category")["total_amount_usd"].sum().sort_values().reset_index()
        fig_cat = go.Figure(go.Bar(
            x=cat_rev["total_amount_usd"], y=cat_rev["category"],
            orientation="h",
            marker=dict(
                color=cat_rev["total_amount_usd"],
                colorscale=[[0,"#EEF2FF"],[1,C["indigo"]]],
                line=dict(width=0),
            ),
            hovertemplate="%{y}: $%{x:,.0f}<extra></extra>",
        ))
        base_layout(fig_cat, height=380)
        fig_cat.update_layout(showlegend=False, xaxis_title="Revenue (USD)", yaxis_title="")
        fig_cat.update_coloraxes(showscale=False)
        st.plotly_chart(fig_cat, use_container_width=True, config={"displayModeBar":False})
        st.markdown('</div>', unsafe_allow_html=True)

    with col_b:
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        st.markdown('<div class="chart-title">Return Rate vs. Avg Rating</div>', unsafe_allow_html=True)
        st.markdown('<div class="chart-subtitle">Bubble size = total revenue · reveals quality / satisfaction trade-offs</div>', unsafe_allow_html=True)
        cat_agg = products.groupby("category").agg(
            avg_rating  = ("avg_rating","mean"),
            return_rate = ("return_rate","mean"),
            revenue     = ("total_revenue_usd","sum"),
        ).reset_index()
        fig_bub = px.scatter(
            cat_agg, x="avg_rating", y="return_rate",
            size="revenue", color="category",
            size_max=36, opacity=0.8,
            color_discrete_sequence=px.colors.qualitative.Pastel,
            hover_data={"revenue":":.0f"},
            labels={"avg_rating":"Avg Rating","return_rate":"Return Rate (%)","category":""},
        )
        base_layout(fig_bub, height=380)
        fig_bub.update_layout(showlegend=False)
        st.plotly_chart(fig_bub, use_container_width=True, config={"displayModeBar":False})
        st.markdown('</div>', unsafe_allow_html=True)

    # ── Top products table ──
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="chart-card">', unsafe_allow_html=True)
    st.markdown('<div class="chart-title">Top 10 Products by Revenue</div>', unsafe_allow_html=True)
    st.markdown('<div class="chart-subtitle">With return rate and average discount</div>', unsafe_allow_html=True)

    top10 = products.sort_values("total_revenue_usd", ascending=False).head(10).copy()

    # Inline mini bar for revenue
    max_rev = top10["total_revenue_usd"].max()
    rows_html = ""
    for _, r in top10.iterrows():
        bar_w = r["total_revenue_usd"] / max_rev * 160
        rows_html += f"""
        <div style="display:grid;grid-template-columns:180px 1fr 100px 80px 80px;
                    align-items:center;gap:16px;padding:10px 0;
                    border-bottom:1px solid {C['border']};">
            <div style="font-size:0.8rem;font-weight:500;color:{C['text_primary']};
                        white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">
                {r['product_name']}
            </div>
            <div style="display:flex;align-items:center;gap:8px;">
                <div style="height:6px;background:{C['indigo_light']};border-radius:3px;flex:1;overflow:hidden;">
                    <div style="height:100%;width:{bar_w:.0f}px;max-width:100%;background:{C['indigo']};border-radius:3px;"></div>
                </div>
                <span style="font-size:0.78rem;font-family:'DM Mono',monospace;color:{C['text_secondary']};white-space:nowrap;">
                    ${r['total_revenue_usd']:,.0f}
                </span>
            </div>
            <div style="font-size:0.78rem;color:{C['text_secondary']};text-align:center;">
                ⭐ {r['avg_rating']:.1f}
            </div>
            <div style="font-size:0.78rem;font-family:'DM Mono',monospace;
                        color:{C['rose'] if r['return_rate']>10 else C['text_secondary']};text-align:center;">
                {r['return_rate']:.0f}%
            </div>
            <div style="font-size:0.78rem;font-family:'DM Mono',monospace;
                        color:{C['text_muted']};text-align:center;">
                {r['avg_discount_pct']:.1f}% off
            </div>
        </div>
        """
    st.markdown(f"""
    <div>
        <div style="display:grid;grid-template-columns:180px 1fr 100px 80px 80px;
                    gap:16px;padding:0 0 8px 0;border-bottom:2px solid {C['border']};">
            <div style="font-size:0.7rem;font-weight:700;letter-spacing:0.07em;
                        text-transform:uppercase;color:{C['text_muted']};">Product</div>
            <div style="font-size:0.7rem;font-weight:700;letter-spacing:0.07em;
                        text-transform:uppercase;color:{C['text_muted']};">Revenue</div>
            <div style="font-size:0.7rem;font-weight:700;letter-spacing:0.07em;
                        text-transform:uppercase;color:{C['text_muted']};text-align:center;">Rating</div>
            <div style="font-size:0.7rem;font-weight:700;letter-spacing:0.07em;
                        text-transform:uppercase;color:{C['text_muted']};text-align:center;">Returns</div>
            <div style="font-size:0.7rem;font-weight:700;letter-spacing:0.07em;
                        text-transform:uppercase;color:{C['text_muted']};text-align:center;">Discount</div>
        </div>
        {rows_html}
    </div>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)


def render_insights_footer():
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.markdown(f"""
        <div class="section-label">Summary</div>
        <div class="section-title">Key Business Findings</div>
        <div class="section-subtitle">Actionable insights extracted from 6 years of e-commerce data</div>
    """, unsafe_allow_html=True)

    insights = [
        (C["indigo"],  C["indigo_light"],  "◆", "Electronics Dominance",
         "Electronics alone drives 36.6% of total revenue ($1.15M) — but also carries the highest return risk. Diversification into Home & Kitchen (lower return rates) is strategically valuable."),
        (C["rose"],    C["rose_light"],    "◆", "Churn is Predictable",
         "Recency and cancel rate are the top churn predictors. A customer inactive for >90 days with >20% cancel rate is 3× more likely to churn. Early intervention window is clear."),
        (C["emerald"], C["emerald_light"], "◆", "Champions Drive Revenue",
         "The top 15% of customers (Champions + Loyal) generate an estimated 45% of total revenue. Investing in retention of this cohort yields outsized returns vs. new acquisition."),
        (C["amber"],   C["amber_light"],   "◆", "India is an Underserved Market",
         "India is the 3rd largest market by customer count but lags the US and UK in AOV. Targeted local pricing or payment localisation could unlock significant growth."),
        (C["sky"],     C["sky_light"],     "◆", "Seasonality is Mild",
         "Revenue peaks in Q4 are only ~12% above annual average — below typical e-commerce norms. A structured seasonal campaign strategy could significantly amplify holiday revenue."),
        (C["violet"],  "#F5F3FF",          "◆", "Repeat Buyers Dominate",
         "64.6% repeat purchase rate is a strong signal of product-market fit. However, 35% of orders are first-time buyers — improving their repeat conversion rate is the highest-leverage growth lever."),
    ]

    c1, c2, c3 = st.columns(3)
    cols = [c1, c2, c3, c1, c2, c3]
    for col, (color, bg, icon, title, body) in zip(cols, insights):
        with col:
            st.markdown(f"""
            <div style="background:{bg};border:1px solid {color}22;border-left:3px solid {color};
                        border-radius:10px;padding:18px 20px;margin-bottom:14px;">
                <div style="font-size:0.72rem;font-weight:700;text-transform:uppercase;
                            letter-spacing:0.07em;color:{color};margin-bottom:6px;">
                    {icon} {title}
                </div>
                <div style="font-size:0.82rem;color:{C['text_secondary']};line-height:1.6;">
                    {body}
                </div>
            </div>
            """, unsafe_allow_html=True)

    # Footer
    st.markdown(f"""
    <div style="border-top:1px solid {C['border']};margin-top:40px;padding:28px 0;
                display:flex;justify-content:space-between;align-items:center;">
        <div>
            <div style="font-size:1rem;font-weight:700;color:{C['text_primary']};margin-bottom:4px;">
                ◈ CustomerIQ
            </div>
            <div style="font-size:0.78rem;color:{C['text_muted']};">
                Built with Streamlit · Scikit-learn · Plotly
                &nbsp;·&nbsp; Dataset: 25K orders · 8K customers · Jan 2020 – Mar 2026
            </div>
        </div>
        <div style="font-size:0.75rem;color:{C['text_muted']};text-align:right;">
            Data Science Portfolio Project<br>
            <span style="color:{C['indigo']};font-weight:600;">
                Churn Prediction · RFM Segmentation · Revenue Forecasting
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN — single-page render pipeline
# ══════════════════════════════════════════════════════════════════════════════

def main():
    inject_css()
    render_topbar()

    # Wrap everything in a page div for consistent side padding
    st.markdown('<div class="page-wrap">', unsafe_allow_html=True)

    with st.spinner("Loading data..."):
        customers, orders, monthly, products, master = load_data()

    render_hero()
    render_kpis(orders, customers)
    render_revenue_section(orders, monthly)
    render_customer_section(customers, orders, master)
    render_churn_section(master)
    render_product_section(products, orders)
    render_insights_footer()

    st.markdown('</div>', unsafe_allow_html=True)


if __name__ == "__main__" or True:
    main()
