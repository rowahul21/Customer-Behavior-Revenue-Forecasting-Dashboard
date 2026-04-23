"""
pages/05_experiment_lab.py  ·  Experiment Lab
Interactive ML insights, what-if scenarios, and product analytics
"""

import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import pandas as pd
import numpy as np

from utils.data_loader import build_master, load_raw, load_products
from utils.styling import (
    section_header, apply_chart_style, metric_card,
    COLORS, PLOTLY_TEMPLATE
)

# ── Load ───────────────────────────────────────────────────────────────────────
master   = build_master()
products = load_products()
customers, orders, _, _ = load_raw()

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="padding:24px 0 8px 0;">
    <h1 style="color:#F1F5F9; font-size:2rem; font-weight:800; margin:0;">
        Experiment Lab
    </h1>
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

# ════════════════════════════════════════════════════════════════════════════════
# TAB 1: Revenue Simulator — "What if?"
# ════════════════════════════════════════════════════════════════════════════════
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
        delta_aov        = st.slider("Increase Average Order Value by (%)", -20, 50, 0, step=1)
        delta_orders     = st.slider("Increase Order Volume by (%)", -30, 100, 0, step=1)
        delta_churn_red  = st.slider("Reduce Churn Rate by (pp)", 0.0, 8.0, 0.0, step=0.1)
    with col2:
        delta_discount   = st.slider("Reduce Average Discount by (pp)", -5.0, 10.0, 0.0, step=0.5)
        delta_return_red = st.slider("Reduce Return Rate by (pp)", 0.0, 10.0, 0.0, step=0.5)
        new_cust_acq     = st.slider("New Customers Acquired", 0, 2000, 0, step=50)

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


# ════════════════════════════════════════════════════════════════════════════════
# TAB 2: Product Intelligence
# ════════════════════════════════════════════════════════════════════════════════
with tab2:
    section_header("Top Products by Revenue")

    sel_category = st.selectbox(
        "Filter by category",
        ["All"] + sorted(products["category"].unique().tolist()),
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


# ════════════════════════════════════════════════════════════════════════════════
# TAB 3: CLV Explorer
# ════════════════════════════════════════════════════════════════════════════════
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


# ════════════════════════════════════════════════════════════════════════════════
# TAB 4: Data Profiler
# ════════════════════════════════════════════════════════════════════════════════
with tab4:
    section_header("Dataset Overview")

    col_p1, col_p2, col_p3, col_p4 = st.columns(4)
    with col_p1: metric_card("Customers", f"{len(customers):,}", "", "flat")
    with col_p2: metric_card("Orders", f"{len(orders):,}", "", "flat")
    with col_p3: metric_card("Products", f"{len(products):,}", "", "flat")
    with col_p4: metric_card("Categories", f"{orders['category'].nunique()}", "", "flat")

    st.markdown("<br>", unsafe_allow_html=True)
    sel_table = st.selectbox("Explore table", ["Customers", "Orders", "Products"])

    if sel_table == "Customers":
        st.dataframe(customers.head(100), use_container_width=True)
    elif sel_table == "Orders":
        st.dataframe(orders.head(100), use_container_width=True)
    else:
        st.dataframe(products, use_container_width=True)
