"""
pages/04_revenue_forecasting.py  ·  Revenue Trends & Forecasting
"""

import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import pandas as pd
import numpy as np

from utils.data_loader import load_monthly, load_raw
from utils.styling import (
    section_header, apply_chart_style, metric_card,
    COLORS, PLOTLY_TEMPLATE
)
from models.ml_models import forecast_revenue

# ── Load ───────────────────────────────────────────────────────────────────────
monthly = load_monthly()
customers, orders, _, _ = load_raw()

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="padding:24px 0 8px 0;">
    <h1 style="color:#F1F5F9; font-size:2rem; font-weight:800; margin:0;">
        Revenue Trends & Forecasting
    </h1>
    <p style="color:#64748B; font-size:0.9rem; margin-top:4px;">
        Historical performance analysis · Holt's double exponential smoothing forecast
    </p>
</div>
""", unsafe_allow_html=True)

# ── Forecast controls ──────────────────────────────────────────────────────────
col_ctrl1, col_ctrl2 = st.columns([3, 1])
with col_ctrl2:
    forecast_months = st.select_slider(
        "Forecast horizon",
        options=[3, 6, 9, 12, 18, 24],
        value=12,
        help="Number of months to forecast",
    )

# ── Generate forecast ──────────────────────────────────────────────────────────
with st.spinner("Generating revenue forecast..."):
    hist_df, forecast_df, fc_metrics = forecast_revenue(monthly, periods=forecast_months)

# ── Forecast KPIs ──────────────────────────────────────────────────────────────
section_header("Forecast Summary")
forecast_total = forecast_df["revenue_usd"].sum()
hist_last12    = hist_df.tail(12)["revenue_usd"].sum()
yoy_growth     = (forecast_total - hist_last12) / hist_last12

cols = st.columns(4)
with cols[0]:
    metric_card("Forecast Total Revenue", f"${forecast_total:,.0f}", "", "flat")
with cols[1]:
    metric_card("Last 12-Month Revenue", f"${hist_last12:,.0f}", "", "flat")
with cols[2]:
    metric_card("Projected Growth", f"{yoy_growth:+.1%}", "", "up" if yoy_growth > 0 else "down")
with cols[3]:
    metric_card("Monthly Trend", f"${fc_metrics['trend_monthly']:+,.0f}", "", "flat")

st.markdown("<br>", unsafe_allow_html=True)

# ── Combined chart ─────────────────────────────────────────────────────────────
section_header("Revenue Forecast with 95% Confidence Interval")

fig_fc = go.Figure()

# Historical
fig_fc.add_trace(go.Scatter(
    x=hist_df["date"], y=hist_df["revenue_usd"],
    name="Historical Revenue",
    line=dict(color=COLORS["primary"], width=2.5),
    hovertemplate="<b>%{x|%b %Y}</b><br>Revenue: $%{y:,.0f}<extra></extra>",
))

# Confidence band
fig_fc.add_trace(go.Scatter(
    x=pd.concat([forecast_df["date"], forecast_df["date"][::-1]]),
    y=pd.concat([forecast_df["upper"], forecast_df["lower"][::-1]]),
    fill="toself",
    fillcolor="rgba(6,182,212,0.12)",
    line=dict(color="rgba(0,0,0,0)"),
    showlegend=True,
    name="95% CI",
))

# Forecast line
fig_fc.add_trace(go.Scatter(
    x=forecast_df["date"], y=forecast_df["revenue_usd"],
    name="Forecast",
    line=dict(color=COLORS["accent"], width=2.5, dash="dot"),
    hovertemplate="<b>%{x|%b %Y}</b><br>Forecast: $%{y:,.0f}<extra></extra>",
))

# Divider line at last historical point
last_hist_date = hist_df["date"].max()

# vertical line
fig_fc.add_shape(
    type="line",
    x0=last_hist_date,
    x1=last_hist_date,
    y0=0,
    y1=1,
    xref="x",
    yref="paper",
    line=dict(
        color=COLORS["neutral"],
        dash="dash",
        width=2
    )
)

# annotation
fig_fc.add_annotation(
    x=last_hist_date,
    y=1,
    xref="x",
    yref="paper",
    text="Forecast Start",
    showarrow=False,
    xanchor="left",
    yanchor="bottom",
    font=dict(color=COLORS["neutral"])
)

apply_chart_style(fig_fc, height=420)
fig_fc.update_layout(yaxis_title="Revenue (USD)", xaxis_title="")
st.plotly_chart(fig_fc, use_container_width=True)

# ── Forecast table ─────────────────────────────────────────────────────────────
col_a, col_b = st.columns([2, 1])

with col_a:
    section_header("Forecast by Month")
    fc_table = forecast_df[["date","revenue_usd","lower","upper"]].copy()
    fc_table["date"]        = fc_table["date"].dt.strftime("%b %Y")
    fc_table["revenue_usd"] = fc_table["revenue_usd"].map("${:,.0f}".format)
    fc_table["lower"]       = fc_table["lower"].map("${:,.0f}".format)
    fc_table["upper"]       = fc_table["upper"].map("${:,.0f}".format)
    st.dataframe(
        fc_table.rename(columns={
            "date":"Month","revenue_usd":"Forecast Revenue",
            "lower":"Lower 95% CI","upper":"Upper 95% CI",
        }),
        use_container_width=True, hide_index=True,
    )

with col_b:
    section_header("Model Metrics")
    st.markdown(f"""
    <div style="background:#1E293B; border-radius:10px; padding:20px;">
        <div style="color:#94A3B8; font-size:0.72rem; font-weight:700;
                    text-transform:uppercase; margin-bottom:12px;">Holt's ETS Model</div>
        <div style="margin-bottom:14px;">
            <div style="color:#64748B; font-size:0.75rem;">RMSE (last 12 months)</div>
            <div style="color:#F1F5F9; font-size:1.3rem; font-weight:700;">${fc_metrics['rmse']:,.2f}</div>
        </div>
        <div style="margin-bottom:14px;">
            <div style="color:#64748B; font-size:0.75rem;">MAPE (last 12 months)</div>
            <div style="color:#F1F5F9; font-size:1.3rem; font-weight:700;">{fc_metrics['mape']:.2f}%</div>
        </div>
        <div style="margin-bottom:14px;">
            <div style="color:#64748B; font-size:0.75rem;">Monthly Trend</div>
            <div style="color:{'#10B981' if fc_metrics['trend_monthly'] > 0 else '#EF4444'};
                        font-size:1.3rem; font-weight:700;">
                {'▲' if fc_metrics['trend_monthly'] > 0 else '▼'} ${abs(fc_metrics['trend_monthly']):,.2f}
            </div>
        </div>
        <div style="border-top:1px solid #334155; margin-top:8px; padding-top:12px;
                    color:#64748B; font-size:0.73rem; line-height:1.5;">
            Model: Holt's Double Exponential Smoothing<br>
            α = 0.30 · β = 0.10<br>
            Trained on {len(hist_df)} months of data
        </div>
    </div>
    """, unsafe_allow_html=True)

# ── Seasonality analysis ───────────────────────────────────────────────────────
section_header("Monthly Seasonality Pattern")

seasonal = (
    monthly[monthly["year"] < 2026]  # exclude incomplete year
    .groupby("month")["revenue_usd"]
    .mean()
    .reset_index()
)
month_names = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",
               7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"}
seasonal["month_name"] = seasonal["month"].map(month_names)

fig_seasonal = px.bar(
    seasonal, x="month_name", y="revenue_usd",
    color="revenue_usd",
    color_continuous_scale=[[0,"#1E293B"],[0.5,"#4F46E5"],[1,"#06B6D4"]],
    labels={"revenue_usd":"Avg Monthly Revenue (USD)","month_name":"Month"},
    text_auto=".0f",
)
fig_seasonal.update_coloraxes(showscale=False)
apply_chart_style(fig_seasonal, height=300)
st.plotly_chart(fig_seasonal, use_container_width=True)

# ── YoY comparison ─────────────────────────────────────────────────────────────
section_header("Year-over-Year Revenue Comparison")

yoy = monthly[monthly["year"].between(2021, 2025)].copy()
yoy["month_name"] = yoy["month"].map(month_names)

fig_yoy = px.line(
    yoy, x="month_name", y="revenue_usd",
    color="year",
    color_discrete_sequence=px.colors.sequential.Plasma_r[:5],
    labels={"revenue_usd":"Revenue (USD)","month_name":"Month"},
    markers=True,
)
apply_chart_style(fig_yoy, height=340)
st.plotly_chart(fig_yoy, use_container_width=True)
