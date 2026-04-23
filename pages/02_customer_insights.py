"""
pages/02_customer_insights.py  ·  Customer Segmentation & RFM Analysis
"""

import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import pandas as pd
import numpy as np

from utils.data_loader import build_master, load_raw
from utils.styling import (
    section_header, apply_chart_style, metric_card,
    COLORS, SEGMENT_COLORS, PLOTLY_TEMPLATE
)
from models.ml_models import train_segmentation

# ── Load ───────────────────────────────────────────────────────────────────────
master = build_master()
customers, orders, _, _ = load_raw()

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="padding:24px 0 8px 0;">
    <h1 style="color:#F1F5F9; font-size:2rem; font-weight:800; margin:0;">
        Customer Insights
    </h1>
    <p style="color:#64748B; font-size:0.9rem; margin-top:4px;">
        RFM analysis, behavioural segmentation, and lifetime value distribution
    </p>
</div>
""", unsafe_allow_html=True)

# ── RFM segment summary cards ──────────────────────────────────────────────────
section_header("RFM Segment Distribution")

seg_counts = master["rfm_segment"].fillna("Others").replace("", "Others").value_counts().reset_index()
seg_counts.columns = ["segment", "count"]
seg_counts["pct"] = seg_counts["count"] / seg_counts["count"].sum()

cols = st.columns(len(seg_counts))
for i, row in seg_counts.iterrows():
    color = SEGMENT_COLORS.get(row["segment"], COLORS["neutral"])
    with cols[i % len(cols)]:
        st.markdown(f"""
        <div style="background:#1E293B; border-top:3px solid {color};
                    border-radius:8px; padding:14px; text-align:center; margin-bottom:8px;">
            <div style="color:{color}; font-weight:700; font-size:0.78rem; 
                        text-transform:uppercase; letter-spacing:0.06em;">{row['segment']}</div>
            <div style="color:#F1F5F9; font-size:1.5rem; font-weight:800;">{row['count']:,}</div>
            <div style="color:#64748B; font-size:0.75rem;">{row['pct']:.1%} of customers</div>
        </div>
        """, unsafe_allow_html=True)

# ── RFM scatter ────────────────────────────────────────────────────────────────
col_a, col_b = st.columns([3, 2])

with col_a:
    section_header("Recency vs Monetary Value by Segment")
    plot_df = master.sample(min(2000, len(master)), random_state=42).copy()
    plot_df["frequency"] = (
    plot_df["frequency"]
    .replace([float("inf"), -float("inf")], None)
    .fillna(1)
)
    fig_scatter = px.scatter(
        plot_df,
        x="recency", y="monetary",
        color="rfm_segment",
        size="frequency",
        size_max=18,
        color_discrete_map=SEGMENT_COLORS,
        hover_data={"customer_id": True, "frequency": True, "monetary": ":.0f"},
        labels={"recency": "Recency (days since last order)", "monetary": "Total Spend (USD)"},
        opacity=0.75,
    )
    apply_chart_style(fig_scatter, height=420)
    st.plotly_chart(fig_scatter, use_container_width=True)

with col_b:
    section_header("Segment Revenue Share")
    seg_rev = master.groupby("rfm_segment")["monetary"].sum().reset_index()
    seg_rev.columns = ["segment", "revenue"]
    fig_pie = px.pie(
        seg_rev, names="segment", values="revenue",
        color="segment",
        color_discrete_map=SEGMENT_COLORS,
        hole=0.5,
    )
    fig_pie.update_traces(textinfo="percent+label", textposition="outside")
    # pindahin legend ke bawah
    fig_pie.update_layout(
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.2,
            xanchor="center",
            x=0.5,
            font=dict(size=10),
        ),
        margin=dict(t=40, b=80)
    )
    apply_chart_style(fig_pie, height=420)
    st.plotly_chart(fig_pie, use_container_width=True)

# ── RFM heatmap ───────────────────────────────────────────────────────────────
section_header("RFM Score Heatmap (Frequency × Recency)")

heatmap_data = master.groupby(["rfm_r", "rfm_f"])["monetary"].mean().reset_index()
heatmap_pivot = heatmap_data.pivot(index="rfm_f", columns="rfm_r", values="monetary")

fig_heat = px.imshow(
    heatmap_pivot,
    color_continuous_scale=[[0,"#1E293B"],[0.5,"#4F46E5"],[1,"#06B6D4"]],
    labels=dict(x="Recency Score (5=Most Recent)", y="Frequency Score (5=Most Frequent)", color="Avg Spend $"),
    text_auto=".0f",
    aspect="auto",
)
apply_chart_style(fig_heat, height=340)
st.plotly_chart(fig_heat, use_container_width=True)

master[["rfm_r", "rfm_f"]].describe()

master[
    (master["rfm_r"] == 0) | 
    (master["rfm_f"] == 0)
]

# ── K-Means clustering ─────────────────────────────────────────────────────────
section_header("K-Means Cluster Analysis")

n_clusters = st.slider("Number of clusters", min_value=3, max_value=8, value=5, step=1)
cluster_df, inertias, sil_scores, final_sil, profile = train_segmentation(master, n_clusters)

col_c, col_d = st.columns([2, 1])

with col_c:
    seg_merged = cluster_df.merge(
        master[["customer_id","rfm_segment"]], on="customer_id", how="left"
    )
    seg_sample = seg_merged.sample(min(2000, len(seg_merged)), random_state=42).copy()

    seg_sample["frequency"] = (
        seg_sample["frequency"]
        .replace([float("inf"), -float("inf")], None)
        .fillna(1)
    )
    
    cluster_colors = px.colors.qualitative.Set2
    fig_clust = px.scatter(
        seg_sample,
        x="recency", y="monetary",
        color="label",
        size="frequency",
        size_max=16,
        color_discrete_sequence=cluster_colors,
        labels={"recency":"Recency (days)", "monetary":"Spend (USD)"},
        opacity=0.8,
    )
    apply_chart_style(fig_clust, height=380)
    st.plotly_chart(fig_clust, use_container_width=True)

with col_d:
    st.markdown(f"""
    <div style="background:#1E293B; border-radius:10px; padding:20px; margin-top:40px;">
        <div style="color:#94A3B8; font-size:0.75rem; text-transform:uppercase; font-weight:700;">
            Silhouette Score
        </div>
        <div style="color:#10B981; font-size:2.2rem; font-weight:800;">{final_sil:.3f}</div>
        <div style="color:#64748B; font-size:0.78rem; margin-top:4px;">
            Closer to 1.0 = better defined clusters
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    section_header("Cluster Profiles")
    st.dataframe(
        profile[["label", "recency", "frequency", "monetary"]]
        .rename(columns={"label":"Segment","recency":"Avg Recency","frequency":"Avg Orders","monetary":"Avg Spend $"}),
        use_container_width=True,
        hide_index=True,
    )

# ── Membership tier breakdown ──────────────────────────────────────────────────
section_header("Membership Tier Breakdown")

tier_stats = master.groupby("membership_tier").agg(
    customers   = ("customer_id","count"),
    avg_spend   = ("monetary",   "mean"),
    churn_rate  = ("churned",    "mean"),
).reset_index()

col_e, col_f = st.columns(2)

with col_e:
    fig_tier = px.bar(
        tier_stats, x="membership_tier", y="avg_spend",
        color="membership_tier",
        color_discrete_sequence=[COLORS["neutral"], COLORS["accent"], COLORS["warning"], COLORS["secondary"]],
        labels={"avg_spend":"Avg Lifetime Spend (USD)","membership_tier":"Tier"},
        text_auto=".0f",
    )
    apply_chart_style(fig_tier, height=300)
    st.plotly_chart(fig_tier, use_container_width=True)

with col_f:
    fig_churn_tier = px.bar(
        tier_stats, x="membership_tier", y="churn_rate",
        color="membership_tier",
        color_discrete_sequence=[COLORS["neutral"], COLORS["accent"], COLORS["warning"], COLORS["secondary"]],
        labels={"churn_rate":"Churn Rate","membership_tier":"Tier"},
        text_auto=".1%",
    )
    apply_chart_style(fig_churn_tier, height=300)
    st.plotly_chart(fig_churn_tier, use_container_width=True)
