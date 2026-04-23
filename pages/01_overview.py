"""
pages/01_overview.py  ·  Overview Page
"""

import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import pandas as pd

from utils.data_loader import load_raw, load_orders, kpis
from utils.styling import (
    metric_card, section_header, apply_chart_style,
    COLORS, PLOTLY_TEMPLATE
)

# ── Load data ──────────────────────────────────────────────────────────────────
customers, orders, monthly, products = load_raw()

# ── Sidebar filters ────────────────────────────────────────────────────────────
st.sidebar.markdown("### Filters")

all_countries = sorted(orders["country"].unique()) if "country" in orders.columns else []
countries = customers["country"].unique().tolist()
sel_countries = st.sidebar.multiselect("Country", sorted(countries), default=[])

all_cats = sorted(orders["category"].unique().tolist())
sel_cats = st.sidebar.multiselect("Category", all_cats, default=[])

years = sorted(orders["year"].unique().tolist())
sel_years = st.sidebar.multiselect("Year", years, default=[])

# Apply filters
o = orders.copy()
c = customers.copy()

if sel_countries:
    c = c[c["country"].isin(sel_countries)]
    o = o[o["customer_id"].isin(c["customer_id"])]
if sel_cats:
    o = o[o["category"].isin(sel_cats)]
if sel_years:
    o = o[o["year"].isin(sel_years)]

# ── Page header ────────────────────────────────────────────────────────────────
st.markdown("""
<div style="padding:24px 0 8px 0;">
    <h1 style="color:#F1F5F9; font-size:2rem; font-weight:800; margin:0;">
        Business Overview
    </h1>
    <p style="color:#64748B; font-size:0.9rem; margin-top:4px;">
        Headline KPIs and performance trends across your e-commerce operation
    </p>
</div>
""", unsafe_allow_html=True)

# ── KPI scorecards ─────────────────────────────────────────────────────────────
kpi_vals = kpis(o, c)

cols = st.columns(4)
with cols[0]:
    metric_card("Total Revenue", f"${kpi_vals['total_revenue']:,.0f}", "", "flat")
with cols[1]:
    metric_card("Total Orders", f"{kpi_vals['total_orders']:,}", "", "flat")
with cols[2]:
    metric_card("Avg Order Value", f"${kpi_vals['aov']:.2f}", "", "flat")
with cols[3]:
    metric_card("Unique Customers", f"{kpi_vals['total_customers']:,}", "", "flat")

st.markdown("<div style='margin-top:16px;'></div>", unsafe_allow_html=True)
cols2 = st.columns(4)
with cols2[0]:
    metric_card("Churn Rate", f"{kpi_vals['churn_rate']:.1%}", "", "flat")
with cols2[1]:
    metric_card("Return Rate", f"{kpi_vals['return_rate']:.1%}", "", "flat")
with cols2[2]:
    metric_card("Avg Delivery Days", f"{kpi_vals['avg_delivery']:.1f}d", "", "flat")
with cols2[3]:
    metric_card("Repeat Purchase Rate", f"{kpi_vals['repeat_rate']:.1%}", "", "flat")

st.markdown("<br>", unsafe_allow_html=True)

# ── Revenue over time ──────────────────────────────────────────────────────────
section_header("Revenue Trend")

rev_monthly = (
    o.groupby(["year", "month"])["total_amount_usd"]
    .sum()
    .reset_index()
)
rev_monthly["date"] = pd.to_datetime(
    rev_monthly["year"].astype(str) + "-" + rev_monthly["month"].astype(str).str.zfill(2) + "-01"
)

fig_rev = go.Figure()
fig_rev.add_trace(go.Scatter(
    x=rev_monthly["date"], y=rev_monthly["total_amount_usd"],
    fill="tozeroy",
    line=dict(color=COLORS["primary"], width=2.5),
    fillcolor="rgba(79,70,229,0.15)",
    name="Monthly Revenue",
    hovertemplate="<b>%{x|%b %Y}</b><br>Revenue: $%{y:,.0f}<extra></extra>",
))
apply_chart_style(fig_rev, height=320)
fig_rev.update_layout(showlegend=False, yaxis_title="Revenue (USD)", xaxis_title="")
st.plotly_chart(fig_rev, use_container_width=True)

# ── Two-column breakdown ───────────────────────────────────────────────────────
col_a, col_b = st.columns(2)

with col_a:
    section_header("Revenue by Category")
    cat_rev = (
        o.groupby("category")["total_amount_usd"]
        .sum()
        .sort_values(ascending=True)
        .reset_index()
    )
    fig_cat = px.bar(
        cat_rev, x="total_amount_usd", y="category",
        orientation="h",
        color="total_amount_usd",
        color_continuous_scale=[[0,"#1E293B"],[1,COLORS["primary"]]],
        labels={"total_amount_usd":"Revenue (USD)", "category":""},
    )
    fig_cat.update_coloraxes(showscale=False)
    apply_chart_style(fig_cat, height=420)
    st.plotly_chart(fig_cat, use_container_width=True)

with col_b:
    section_header("Orders by Country (Top 10)")
    country_orders = (
        o.merge(c[["customer_id","country"]], on="customer_id", how="left")
        if "country" not in o.columns
        else o
    )
    # Use customer country
    ctry = (
        c.groupby("country")["customer_id"].count()
        .sort_values(ascending=False)
        .head(10)
        .reset_index()
        .rename(columns={"customer_id":"customers"})
    )
    fig_ctry = px.bar(
        ctry, x="customers", y="country",
        orientation="h",
        color="customers",
        color_continuous_scale=[[0,"#1E293B"],[1,COLORS["accent"]]],
        labels={"customers":"Customers","country":""},
    )
    fig_ctry.update_coloraxes(showscale=False)
    apply_chart_style(fig_ctry, height=420)
    st.plotly_chart(fig_ctry, use_container_width=True)

# ── Order status + device ──────────────────────────────────────────────────────
col_c, col_d = st.columns(2)

with col_c:
    section_header("Order Status Breakdown")
    status_counts = o["order_status"].value_counts().reset_index()
    status_counts.columns = ["status", "count"]
    colors_status = [COLORS["success"], COLORS["danger"], COLORS["warning"], COLORS["accent"]]
    fig_status = px.pie(
        status_counts, names="status", values="count",
        color_discrete_sequence=colors_status,
        hole=0.55,
    )
    fig_status.update_traces(textposition="outside", textinfo="percent+label")
    apply_chart_style(fig_status, height=320)
    st.plotly_chart(fig_status, use_container_width=True)

with col_d:
    section_header("Payment Method Mix")
    pay = o["payment_method"].value_counts().reset_index()
    pay.columns = ["method", "count"]
    fig_pay = px.bar(
        pay, x="method", y="count",
        color="count",
        color_continuous_scale=[[0,"#1E293B"],[1,COLORS["secondary"]]],
        labels={"count":"Orders","method":""},
    )
    fig_pay.update_coloraxes(showscale=False)
    apply_chart_style(fig_pay, height=320)
    st.plotly_chart(fig_pay, use_container_width=True)

# ── Key insights callout ───────────────────────────────────────────────────────
section_header("Key Business Insights")
st.markdown("""
<div style="display:grid; grid-template-columns:1fr 1fr; gap:12px;">
    <div style="background:#1E293B; border-left:4px solid #10B981; border-radius:8px; padding:14px;">
        <div style="color:#10B981; font-weight:700; font-size:0.85rem;">Strong Repeat Purchase</div>
        <div style="color:#94A3B8; font-size:0.82rem; margin-top:4px;">
            Majority of orders come from returning customers, indicating healthy product-market fit.
        </div>
    </div>
    <div style="background:#1E293B; border-left:4px solid #F59E0B; border-radius:8px; padding:14px;">
        <div style="color:#F59E0B; font-weight:700; font-size:0.85rem;">Low but Trackable Churn</div>
        <div style="color:#94A3B8; font-size:0.82rem; margin-top:4px;">
            ~8.9% churn rate. Even a 2-point reduction would retain ~160 customers and ~$20K revenue annually.
        </div>
    </div>
    <div style="background:#1E293B; border-left:4px solid #4F46E5; border-radius:8px; padding:14px;">
        <div style="color:#4F46E5; font-weight:700; font-size:0.85rem;">Electronics Leads Revenue</div>
        <div style="color:#94A3B8; font-size:0.82rem; margin-top:4px;">
            Electronics and Clothing dominate orders. Upsell opportunities exist in Home & Kitchen.
        </div>
    </div>
    <div style="background:#1E293B; border-left:4px solid #06B6D4; border-radius:8px; padding:14px;">
        <div style="color:#06B6D4; font-weight:700; font-size:0.85rem;">US Dominates, India Growing</div>
        <div style="color:#94A3B8; font-size:0.82rem; margin-top:4px;">
            US accounts for ~31% of customers. India is the 3rd largest market — a high-growth opportunity.
        </div>
    </div>
</div>
""", unsafe_allow_html=True)
