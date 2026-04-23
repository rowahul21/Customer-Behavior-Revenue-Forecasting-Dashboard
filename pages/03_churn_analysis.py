"""
pages/03_churn_analysis.py  ·  Churn Prediction & Risk Analysis
"""

import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve

from utils.data_loader import build_master
from utils.styling import (
    section_header, apply_chart_style, metric_card,
    COLORS, PLOTLY_TEMPLATE
)
from models.ml_models import train_churn_model, predict_churn_proba, CHURN_FEATURES

# ── Load & train ───────────────────────────────────────────────────────────────
master = build_master()

with st.spinner("🤖 Training Gradient Boosting model..."):
    model, X_test, y_test, y_proba, importances, metrics = train_churn_model(master)

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="padding:24px 0 8px 0;">
    <h1 style="color:#F1F5F9; font-size:2rem; font-weight:800; margin:0;">
        Churn Prediction
    </h1>
    <p style="color:#64748B; font-size:0.9rem; margin-top:4px;">
        Gradient Boosting classifier · identify at-risk customers before they leave
    </p>
</div>
""", unsafe_allow_html=True)

# ── Model performance cards ────────────────────────────────────────────────────
section_header("Model Performance")

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

# ── ROC curve + Feature importance ────────────────────────────────────────────
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

# ── Confusion matrix ───────────────────────────────────────────────────────────
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

# ── High-risk customer table ───────────────────────────────────────────────────
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

# ── Scenario simulator ─────────────────────────────────────────────────────────
section_header("Churn Probability Simulator")
st.markdown(
    "<p style='color:#64748B; font-size:0.85rem;'>Adjust customer attributes to simulate churn risk in real time.</p>",
    unsafe_allow_html=True
)

sim_col1, sim_col2, sim_col3 = st.columns(3)
with sim_col1:
    sim_recency   = st.slider("Days since last order", 0, 365, 90)
    sim_frequency = st.slider("Total orders", 1, 50, 5)
    sim_monetary  = st.slider("Total spend ($)", 0, 5000, 500)
    sim_tenure    = st.slider("Tenure (days)", 30, 2000, 365)
    sim_cancel    = st.slider("Cancel rate", 0.0, 1.0, 0.1)
with sim_col2:
    sim_return    = st.slider("Return rate", 0.0, 1.0, 0.1)
    sim_session   = st.slider("Avg session (min)", 1.0, 60.0, 15.0)
    sim_pages     = st.slider("Avg pages viewed", 1, 20, 5)
    sim_discount  = st.slider("Avg discount %", 0.0, 30.0, 5.0)
    sim_reviews   = st.slider("Reviews given", 0, 20, 2)
with sim_col3:
    sim_rev_score = st.slider("Avg review score", 1.0, 5.0, 3.5)
    sim_wishlist  = st.slider("Wishlist items", 0, 50, 5)
    sim_newsletter= st.selectbox("Newsletter subscribed", [0, 1], format_func=lambda x: "Yes" if x else "No")
    sim_days_last = st.slider("Days since last purchase", 0, 365, 60)
    sim_cat_div   = st.slider("Categories purchased", 1, 10, 3)

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
