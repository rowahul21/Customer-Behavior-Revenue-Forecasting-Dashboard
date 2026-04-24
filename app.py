"""
app.py  ·  CustomerIQ Dashboard
================================
Run with:  streamlit run app.py
"""

import streamlit as st
from utils.styling import set_page_config, inject_css

# set_page_config("Overview")
# inject_css()

st.set_page_config(
    page_title="CustomerBehavior Dashboard",
    layout="wide",
    initial_sidebar_state="collapsed"
)

import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve

from utils.data_loader import load_raw, load_orders, kpis, build_master, load_monthly, load_products
from utils.styling import (
    metric_card, section_header, apply_chart_style,
    COLORS, PLOTLY_TEMPLATE, SEGMENT_COLORS
)
from models.ml_models import train_segmentation, train_churn_model, predict_churn_proba, CHURN_FEATURES, forecast_revenue

# ── Load data ──────────────────────────────────────────────────────────────────
customers, orders, monthly, products = load_raw()
master = build_master()
monthly_processed = load_monthly()

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center; padding:24px 0 8px 0;">
    <h1 style="color:#F1F5F9; font-size:3rem; font-weight:800; margin:0;">
        CustomerBehavior Dashboard
    </h1>
    <p style="color:#64748B; font-size:1rem; margin-top:4px;">
        Customer Intelligence Platform · Built with Streamlit + Scikit-learn
    </p>
    <div style="font-size:0.8rem; color:#475569; margin-top:8px;">
        Dataset: 25K orders · 8K customers · Period: Jan 2020 – Mar 2026
    </div>
</div>
""", unsafe_allow_html=True)

# ── Filters ────────────────────────────────────────────────────────────────────
st.markdown("### Global Filters")
col_f1, col_f2, col_f3, col_f4 = st.columns(4)

with col_f1:
    all_countries = sorted(orders["country"].unique()) if "country" in orders.columns else []
    countries = customers["country"].unique().tolist()
    sel_countries = st.multiselect("Country", sorted(countries), default=[], key="countries")

with col_f2:
    all_cats = sorted(orders["category"].unique().tolist())
    sel_cats = st.multiselect("Category", all_cats, default=[], key="categories")

with col_f3:
    years = sorted(orders["year"].unique().tolist())
    sel_years = st.multiselect("Year", years, default=[], key="years")

with col_f4:
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("Clear All Filters", type="secondary"):
        st.rerun()

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

# ── Tabs ───────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Overview",
    "Customer Insights",
    "Churn Analysis",
    "Revenue Forecasting",
    "Experiment Lab",
])

# ════════════════════════════════════════════════════════════════════════════════
# TAB 1: Overview
# ════════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown("""
    <div style="padding:24px 0 8px 0;">
        <h2 style="color:#F1F5F9; font-size:2rem; font-weight:800; margin:0;">
            Business Overview
        </h2>
        <p style="color:#64748B; font-size:0.9rem; margin-top:4px;">
            Headline KPIs and performance trends across your e-commerce operation
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ── KPI scorecards ─────────────────────────────────────────────────────────
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

    # ── Revenue over time ──────────────────────────────────────────────────────
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

    # ── Two-column breakdown ───────────────────────────────────────────────────
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

    # ── Order status + device ──────────────────────────────────────────────────
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

    # ── Key insights callout ───────────────────────────────────────────────────
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

# ════════════════════════════════════════════════════════════════════════════════
# TAB 2: Customer Insights
# ════════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("""
    <div style="padding:24px 0 8px 0;">
        <h2 style="color:#F1F5F9; font-size:2rem; font-weight:800; margin:0;">
            Customer Insights
        </h2>
        <p style="color:#64748B; font-size:0.9rem; margin-top:4px;">
            RFM analysis, behavioural segmentation, and lifetime value distribution
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ── RFM segment summary cards ──────────────────────────────────────────────
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

    # ── RFM scatter ────────────────────────────────────────────────────────────
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

    # ── RFM heatmap ───────────────────────────────────────────────────────────
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

    # ── K-Means clustering ─────────────────────────────────────────────────────
    section_header("K-Means Cluster Analysis")

    n_clusters = st.slider("Number of clusters", min_value=3, max_value=8, value=5, step=1, key="n_clusters")
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

    # ── Membership tier breakdown ──────────────────────────────────────────────
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

# ════════════════════════════════════════════════════════════════════════════════
# TAB 3: Churn Analysis
# ════════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("""
    <div style="padding:24px 0 8px 0;">
        <h2 style="color:#F1F5F9; font-size:2rem; font-weight:800; margin:0;">
            Churn Prediction
        </h2>
        <p style="color:#64748B; font-size:0.9rem; margin-top:4px;">
            Gradient Boosting classifier · identify at-risk customers before they leave
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ── Model performance cards ────────────────────────────────────────────────
    section_header("Model Performance")

    with st.spinner("Training Gradient Boosting model..."):
        model, X_test, y_test, y_proba, importances, metrics = train_churn_model(master)

    cols = st.columns(5)
    card_data = [
        ("ROC-AUC",   f"{metrics['auc']:.4f}",      "primary"),
        ("Accuracy",  f"{metrics['accuracy']:.1%}",  "success"),
        ("Precision", f"{metrics['precision']:.4f}", "accent"),
        ("Recall",    f"{metrics['recall']:.4f}",    "warning"),
        ("F1 Score",  f"{metrics['f1']:.4f}",        "secondary"),
    ]
    for col, (label, val, color_key) in zip(cols, card_data):
        color = COLORS[color_key]
        with col:
            st.markdown(f"""
            <div style="background:#1E293B; border-top:3px solid {color};
                        border-radius:8px; padding:16px; text-align:center;">
                <div style="color:#94A3B8; font-size:0.72rem; font-weight:700;
                            text-transform:uppercase; letter-spacing:0.08em;">{label}</div>
                <div style="color:#F1F5F9; font-size:1.6rem; font-weight:800; margin-top:4px;">{val}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── ROC curve + Feature importance ────────────────────────────────────────
    col_a, col_b = st.columns(2)

    with col_a:
        section_header("ROC Curve")
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(
            x=fpr, y=tpr,
            fill="tozeroy",
            line=dict(color=COLORS["primary"], width=2.5),
            fillcolor="rgba(79,70,229,0.15)",
            name=f"AUC = {metrics['auc']:.4f}",
        ))
        fig_roc.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            line=dict(color=COLORS["neutral"], dash="dash", width=1.5),
            name="Random Classifier",
            showlegend=True,
        ))
        fig_roc.update_layout(
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
        )
        apply_chart_style(fig_roc, height=360)
        st.plotly_chart(fig_roc, use_container_width=True)

    with col_b:
        section_header("Feature Importance (Top 12)")
        top_imp = importances.tail(12)
        fig_imp = px.bar(
            x=top_imp.values,
            y=top_imp.index,
            orientation="h",
            color=top_imp.values,
            color_continuous_scale=[[0,"#1E293B"],[1,COLORS["primary"]]],
            labels={"x":"Importance", "y":""},
        )
        fig_imp.update_coloraxes(showscale=False)
        apply_chart_style(fig_imp, height=360)
        st.plotly_chart(fig_imp, use_container_width=True)

    # ── Confusion matrix ───────────────────────────────────────────────────────
    col_c, col_d = st.columns(2)

    with col_c:
        section_header("Confusion Matrix")
        cm = metrics["conf_matrix"]
        labels = ["Not Churned", "Churned"]
        fig_cm = px.imshow(
            cm,
            x=labels, y=labels,
            color_continuous_scale=[[0,"#1E293B"],[0.5,"#4F46E5"],[1,"#06B6D4"]],
            text_auto=True, aspect="equal",
            labels=dict(x="Predicted", y="Actual"),
        )
        apply_chart_style(fig_cm, height=320)
        st.plotly_chart(fig_cm, use_container_width=True)

    with col_d:
        section_header("Churn Risk Distribution")
        # Tag predictions onto test set
        test_df = X_test.copy()
        test_df["churn_proba"] = y_proba
        test_df["actual"] = y_test.values

        fig_dist = go.Figure()
        fig_dist.add_trace(go.Histogram(
            x=test_df[test_df["actual"] == 0]["churn_proba"],
            name="Not Churned", opacity=0.75,
            marker_color=COLORS["success"], nbinsx=30,
        ))
        fig_dist.add_trace(go.Histogram(
            x=test_df[test_df["actual"] == 1]["churn_proba"],
            name="Churned", opacity=0.75,
            marker_color=COLORS["danger"], nbinsx=30,
        ))
        fig_dist.update_layout(barmode="overlay")
        apply_chart_style(fig_dist, height=320)
        fig_dist.update_layout(xaxis_title="Predicted Churn Probability", yaxis_title="Count")
        st.plotly_chart(fig_dist, use_container_width=True)

    # ── High-risk customer table ───────────────────────────────────────────────
    section_header("Highest-Risk Customers (Simulation)")
    master_copy = master.dropna(subset=CHURN_FEATURES).copy()
    master_copy["churn_probability"] = model.predict_proba(master_copy[CHURN_FEATURES])[:, 1]
    master_copy["risk_tier"] = pd.cut(
        master_copy["churn_probability"],
        bins=[0, 0.3, 0.6, 1.0],
        labels=["Low", "Medium", "High"],
    )

    top_risk = (
        master_copy[master_copy["churned"] == 0]  # not yet churned but at-risk
        .sort_values("churn_probability", ascending=False)
        .head(15)[["customer_id","country","membership_tier","monetary",
                   "recency","frequency","churn_probability","risk_tier"]]
    )
    top_risk["churn_probability"] = top_risk["churn_probability"].map("{:.1%}".format)
    top_risk["monetary"] = top_risk["monetary"].map("${:,.0f}".format)

    st.dataframe(
        top_risk.rename(columns={
            "customer_id":"Customer","country":"Country",
            "membership_tier":"Tier","monetary":"Lifetime Value",
            "recency":"Days Since Order","frequency":"Total Orders",
            "churn_probability":"Churn Risk","risk_tier":"Risk Level",
        }),
        use_container_width=True,
        hide_index=True,
    )

    # ── Scenario simulator ─────────────────────────────────────────────────────
    section_header("Churn Probability Simulator")
    st.markdown(
        "<p style='color:#64748B; font-size:0.85rem;'>Adjust customer attributes to simulate churn risk in real time.</p>",
        unsafe_allow_html=True
    )

    sim_col1, sim_col2, sim_col3 = st.columns(3)
    with sim_col1:
        sim_recency   = st.slider("Days since last order", 0, 365, 90, key="sim_recency")
        sim_frequency = st.slider("Total orders", 1, 50, 5, key="sim_frequency")
        sim_monetary  = st.slider("Total spend ($)", 0, 5000, 500, key="sim_monetary")
        sim_tenure    = st.slider("Tenure (days)", 30, 2000, 365, key="sim_tenure")
        sim_cancel    = st.slider("Cancel rate", 0.0, 1.0, 0.1, key="sim_cancel")
    with sim_col2:
        sim_return    = st.slider("Return rate", 0.0, 1.0, 0.1, key="sim_return")
        sim_session   = st.slider("Avg session (min)", 1.0, 60.0, 15.0, key="sim_session")
        sim_pages     = st.slider("Avg pages viewed", 1, 20, 5, key="sim_pages")
        sim_discount  = st.slider("Avg discount %", 0.0, 30.0, 5.0, key="sim_discount")
        sim_reviews   = st.slider("Reviews given", 0, 20, 2, key="sim_reviews")
    with sim_col3:
        sim_rev_score = st.slider("Avg review score", 1.0, 5.0, 3.5, key="sim_rev_score")
        sim_wishlist  = st.slider("Wishlist items", 0, 50, 5, key="sim_wishlist")
        sim_newsletter= st.selectbox("Newsletter subscribed", [0, 1], format_func=lambda x: "Yes" if x else "No", key="sim_newsletter")
        sim_days_last = st.slider("Days since last purchase", 0, 365, 60, key="sim_days_last")
        sim_cat_div   = st.slider("Categories purchased", 1, 10, 3, key="sim_cat_div")

    sim_input = {
        "recency":                  sim_recency,
        "frequency":                sim_frequency,
        "monetary":                 sim_monetary,
        "avg_session_minutes":      sim_session,
        "avg_pages_viewed":         sim_pages,
        "cancel_rate":              sim_cancel,
        "return_rate_pct":          sim_return,
        "avg_discount_pct":         sim_discount,
        "days_since_last_purchase": sim_days_last,
        "reviews_given":            sim_reviews,
        "avg_review_score":         sim_rev_score,
        "wishlist_items":           sim_wishlist,
        "newsletter_subscribed":    sim_newsletter,
        "tenure_days":              sim_tenure,
        "categories_purchased":     sim_cat_div,
    }

    churn_prob = predict_churn_proba(model, sim_input)
    risk_color = COLORS["danger"] if churn_prob > 0.6 else COLORS["warning"] if churn_prob > 0.3 else COLORS["success"]
    risk_label = "HIGH RISK" if churn_prob > 0.6 else "MEDIUM RISK" if churn_prob > 0.3 else "LOW RISK"

    st.markdown(f"""
    <div style="background:#1E293B; border: 2px solid {risk_color}; border-radius:12px;
                padding:28px; text-align:center; margin-top:16px;">
        <div style="color:#94A3B8; font-size:0.8rem; font-weight:700; text-transform:uppercase;
                    letter-spacing:0.1em;">Predicted Churn Probability</div>
        <div style="color:{risk_color}; font-size:4rem; font-weight:900; margin:8px 0;">{churn_prob:.1%}</div>
        <div style="background:{risk_color}22; color:{risk_color}; font-size:0.85rem; font-weight:700;
                    padding:6px 20px; border-radius:20px; display:inline-block;">{risk_label}</div>
    </div>
    """, unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════════
# TAB 4: Revenue Forecasting
# ════════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown("""
    <div style="padding:24px 0 8px 0;">
        <h2 style="color:#F1F5F9; font-size:2rem; font-weight:800; margin:0;">
            Revenue Trends & Forecasting
        </h2>
        <p style="color:#64748B; font-size:0.9rem; margin-top:4px;">
            Historical performance analysis · Holt's double exponential smoothing forecast
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ── Forecast controls ──────────────────────────────────────────────────────
    col_ctrl1, col_ctrl2 = st.columns([3, 1])
    with col_ctrl2:
        forecast_months = st.select_slider(
            "Forecast horizon",
            options=[3, 6, 9, 12, 18, 24],
            value=12,
            key="forecast_months",
            help="Number of months to forecast",
        )

    # ── Generate forecast ──────────────────────────────────────────────────────
    with st.spinner("Generating revenue forecast..."):
        hist_df, forecast_df, fc_metrics = forecast_revenue(monthly_processed, periods=forecast_months)

    # ── Forecast KPIs ──────────────────────────────────────────────────────────
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

    # ── Combined chart ─────────────────────────────────────────────────────────
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

    # ── Forecast table ─────────────────────────────────────────────────────────
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

    # ── Seasonality analysis ───────────────────────────────────────────────────
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

    # ── YoY comparison ─────────────────────────────────────────────────────────
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

# ════════════════════════════════════════════════════════════════════════════════
# TAB 5: Experiment Lab
# ════════════════════════════════════════════════════════════════════════════════
with tab5:
    st.markdown("""
    <div style="padding:24px 0 8px 0;">
        <h2 style="color:#F1F5F9; font-size:2rem; font-weight:800; margin:0;">
            Experiment Lab
        </h2>
        <p style="color:#64748B; font-size:0.9rem; margin-top:4px;">
            Interactive what-if scenarios · Product analytics · Advanced insights
        </p>
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs([
        "Revenue Simulator",
        "Product Intelligence",
        "CLV Explorer",
        "Data Profiler",
    ])

    # ════════════════════════════════════════════════════════════════════════════
    # SUBTAB 1: Revenue Simulator — "What if?"
    # ════════════════════════════════════════════════════════════════════════════
    with tab1:
        section_header("What-If Revenue Simulator")
        st.markdown(
            "<p style='color:#64748B;font-size:0.85rem;'>Simulate how business levers affect annual revenue.</p>",
            unsafe_allow_html=True
        )

        baseline_rev  = orders["total_amount_usd"].sum()
        baseline_aov  = orders["total_amount_usd"].mean()
        baseline_cust = orders["customer_id"].nunique()
        churn_rate    = customers["churned"].mean()

        col1, col2 = st.columns(2)
        with col1:
            delta_aov        = st.slider("Increase Average Order Value by (%)", -20, 50, 0, step=1, key="delta_aov")
            delta_orders     = st.slider("Increase Order Volume by (%)", -30, 100, 0, step=1, key="delta_orders")
            delta_churn_red  = st.slider("Reduce Churn Rate by (pp)", 0.0, 8.0, 0.0, step=0.1, key="delta_churn_red")
        with col2:
            delta_discount   = st.slider("Reduce Average Discount by (pp)", -5.0, 10.0, 0.0, step=0.5, key="delta_discount")
            delta_return_red = st.slider("Reduce Return Rate by (pp)", 0.0, 10.0, 0.0, step=0.5, key="delta_return_red")
            new_cust_acq     = st.slider("New Customers Acquired", 0, 2000, 0, step=50, key="new_cust_acq")

        # Calculate simulated revenue
        sim_aov      = baseline_aov * (1 + delta_aov / 100)
        sim_orders   = len(orders)  * (1 + delta_orders / 100)
        sim_rev_base = sim_aov * sim_orders

        # Saved churn revenue
        avg_spend_per_cust   = master["monetary"].mean()
        saved_churn_rev      = (delta_churn_red / 100) * customers["churned"].sum() * avg_spend_per_cust

        # Discount recovery (applied to total revenue)
        discount_recovery    = (delta_discount / 100) * baseline_rev * 0.12  # ~12% of orders have discounts

        # Return reduction savings
        avg_return_cost = 35  # avg cost per return (shipping + restocking)
        return_savings  = (delta_return_red / 100) * orders["returned"].sum() * avg_return_cost

        # New customer revenue
        new_cust_rev = new_cust_acq * avg_spend_per_cust

        simulated_total = sim_rev_base + saved_churn_rev + discount_recovery + return_savings + new_cust_rev
        delta_pct       = (simulated_total - baseline_rev) / baseline_rev

        # Display
        st.markdown("<br>", unsafe_allow_html=True)
        res_cols = st.columns(4)
        with res_cols[0]:
            metric_card("Baseline Revenue", f"${baseline_rev:,.0f}", "", "flat")
        with res_cols[1]:
            metric_card("Simulated Revenue", f"${simulated_total:,.0f}", "", "flat")
        with res_cols[2]:
            dir_ = "up" if delta_pct > 0 else "down"
            metric_card("Revenue Impact", f"{delta_pct:+.1%}", f"${simulated_total - baseline_rev:+,.0f}", dir_)
        with res_cols[3]:
            metric_card("Churn Recovery Value", f"${saved_churn_rev:,.0f}", "", "flat")

        # Waterfall breakdown
        st.markdown("<br>", unsafe_allow_html=True)
        section_header("Revenue Impact Waterfall")

        waterfall_items = {
            "Baseline":           baseline_rev,
            "AOV Improvement":    sim_rev_base - baseline_rev,
            "Churn Recovery":     saved_churn_rev,
            "Discount Savings":   discount_recovery,
            "Return Reduction":   return_savings,
            "New Customers":      new_cust_rev,
        }

        measures = ["absolute"] + ["relative"] * (len(waterfall_items) - 1)
        fig_wf = go.Figure(go.Waterfall(
            orientation="v",
            measure=measures,
            x=list(waterfall_items.keys()),
            y=list(waterfall_items.values()),
            connector={"line":{"color":COLORS["neutral"]}},
            increasing={"marker":{"color":COLORS["success"]}},
            decreasing={"marker":{"color":COLORS["danger"]}},
            totals={"marker":{"color":COLORS["primary"]}},
            textposition="outside",
            text=[f"${v:,.0f}" for v in waterfall_items.values()],
        ))
        apply_chart_style(fig_wf, height=380)
        st.plotly_chart(fig_wf, use_container_width=True)

    # ════════════════════════════════════════════════════════════════════════════
    # SUBTAB 2: Product Intelligence
    # ════════════════════════════════════════════════════════════════════════════
    with tab2:
        section_header("Top Products by Revenue")

        sel_category = st.selectbox(
            "Filter by category",
            ["All"] + sorted(products["category"].unique().tolist()),
            key="sel_category",
        )

        prod_data = products if sel_category == "All" else products[products["category"] == sel_category]
        top_prods = prod_data.sort_values("total_revenue_usd", ascending=False).head(20)

        fig_prods = px.bar(
            top_prods,
            x="total_revenue_usd", y="product_name",
            orientation="h",
            color="avg_rating",
            color_continuous_scale=[[0,"#EF4444"],[0.5,"#F59E0B"],[1,"#10B981"]],
            hover_data={"total_orders":True, "return_rate":True, "avg_discount_pct":True},
            labels={"total_revenue_usd":"Revenue (USD)","product_name":"","avg_rating":"Avg Rating"},
        )
        apply_chart_style(fig_prods, height=520)
        st.plotly_chart(fig_prods, use_container_width=True)

        # Return rate vs Rating bubble chart
        section_header("Return Rate vs Rating (Bubble = Revenue)")
        fig_bubble = px.scatter(
            prod_data,
            x="avg_rating", y="return_rate",
            size="total_revenue_usd",
            color="category",
            hover_data={"product_name":True, "total_orders":True},
            labels={"avg_rating":"Average Rating","return_rate":"Return Rate (%)"},
            size_max=40,
        )
        apply_chart_style(fig_bubble, height=400)
        st.plotly_chart(fig_bubble, use_container_width=True)

    # ════════════════════════════════════════════════════════════════════════════
    # SUBTAB 3: CLV Explorer
    # ════════════════════════════════════════════════════════════════════════════
    with tab3:
        section_header("Customer Lifetime Value Distribution")

        # Simple CLV = monetary (already computed as total historical spend)
        clv_df = master[["customer_id","monetary","frequency","recency","membership_tier",
                          "rfm_segment","country","churned"]].copy()
        clv_df = clv_df.rename(columns={"monetary":"clv"})

        # CLV histogram
        fig_clv_hist = px.histogram(
            clv_df, x="clv", nbins=50,
            color_discrete_sequence=[COLORS["primary"]],
            labels={"clv":"Customer Lifetime Value (USD)"},
            opacity=0.8,
        )
        apply_chart_style(fig_clv_hist, height=300)
        st.plotly_chart(fig_clv_hist, use_container_width=True)

        col_clv1, col_clv2 = st.columns(2)

        with col_clv1:
            section_header("CLV by Membership Tier")
            clv_tier = clv_df.groupby("membership_tier")["clv"].agg(["mean","median","sum"]).reset_index()
            fig_box_tier = px.box(
                clv_df, x="membership_tier", y="clv",
                color="membership_tier",
                color_discrete_sequence=[COLORS["neutral"], COLORS["accent"], COLORS["warning"], COLORS["secondary"]],
                labels={"clv":"CLV (USD)", "membership_tier":"Tier"},
            )
            apply_chart_style(fig_box_tier, height=340)
            st.plotly_chart(fig_box_tier, use_container_width=True)

        with col_clv2:
            section_header("CLV by RFM Segment")
            clv_seg = (
                clv_df.groupby("rfm_segment")["clv"]
                .agg(["mean","count"])
                .reset_index()
                .rename(columns={"mean":"avg_clv","count":"customers"})
                .sort_values("avg_clv", ascending=False)
            )
            fig_clv_seg = px.bar(
                clv_seg, x="rfm_segment", y="avg_clv",
                color="avg_clv",
                color_continuous_scale=[[0,"#1E293B"],[1,COLORS["primary"]]],
                text_auto=".0f",
                labels={"avg_clv":"Avg CLV (USD)","rfm_segment":"Segment"},
            )
            fig_clv_seg.update_coloraxes(showscale=False)
            apply_chart_style(fig_clv_seg, height=340)
            st.plotly_chart(fig_clv_seg, use_container_width=True)

        # Top CLV customers
        section_header("Top 20 Customers by Lifetime Value")
        top_clv = clv_df.nlargest(20,"clv")[["customer_id","clv","frequency","rfm_segment","membership_tier","country"]]
        top_clv["clv"] = top_clv["clv"].map("${:,.0f}".format)
        st.dataframe(
            top_clv.rename(columns={
                "customer_id":"Customer","clv":"Lifetime Value",
                "frequency":"Total Orders","rfm_segment":"Segment",
                "membership_tier":"Tier","country":"Country",
            }),
            use_container_width=True, hide_index=True,
        )

    # ════════════════════════════════════════════════════════════════════════════
    # SUBTAB 4: Data Profiler
    # ════════════════════════════════════════════════════════════════════════════
    with tab4:
        section_header("Dataset Overview")

        col_p1, col_p2, col_p3, col_p4 = st.columns(4)
        with col_p1: metric_card("Customers", f"{len(customers):,}", "", "flat")
        with col_p2: metric_card("Orders", f"{len(orders):,}", "", "flat")
        with col_p3: metric_card("Products", f"{len(products):,}", "", "flat")
        with col_p4: metric_card("Categories", f"{orders['category'].nunique()}", "", "flat")

        st.markdown("<br>", unsafe_allow_html=True)
        sel_table = st.selectbox("Explore table", ["Customers", "Orders", "Products"], key="sel_table")

        if sel_table == "Customers":
            st.dataframe(customers.head(100), use_container_width=True)
        elif sel_table == "Orders":
            st.dataframe(orders.head(100), use_container_width=True)
        else:
            st.dataframe(products, use_container_width=True)
