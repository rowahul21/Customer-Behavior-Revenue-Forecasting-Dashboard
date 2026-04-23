"""
data_loader.py
--------------
Centralised data loading, cleaning, and feature engineering.
All pages import from here — data is cached via st.cache_data so
it is only loaded once per session.
"""

import pandas as pd
import numpy as np
import streamlit as st
from pathlib import Path

# ── Path config ────────────────────────────────────────────────────────────────
DATA_DIR = Path(__file__).parent.parent / "data"


# ── Raw loaders ────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_raw():
    """Load all four CSVs and parse dates."""
    customers = pd.read_csv(DATA_DIR / "customers.csv", parse_dates=["registration_date"])
    orders    = pd.read_csv(DATA_DIR / "orders.csv",    parse_dates=["order_date", "delivery_date"])
    monthly   = pd.read_csv(DATA_DIR / "monthly_revenue.csv")
    products  = pd.read_csv(DATA_DIR / "product_summary.csv")
    return customers, orders, monthly, products


# ── Feature engineering ────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def build_master():
    """
    Merge all tables and engineer key features.
    Returns a single master DataFrame indexed by customer.
    """
    customers, orders, monthly, products = load_raw()

    # ── RFM metrics (from orders) ──
    snapshot_date = orders["order_date"].max() + pd.Timedelta(days=1)
    rfm_raw = (
        orders.groupby("customer_id")
        .agg(
            recency   = ("order_date",       lambda x: (snapshot_date - x.max()).days),
            frequency = ("order_id",         "count"),
            monetary  = ("total_amount_usd", "sum"),
        )
        .reset_index()
    )

    # ── Order-level behavioural stats ──
    order_stats = (
        orders.groupby("customer_id")
        .agg(
            avg_session_minutes  = ("session_duration_minutes",   "mean"),
            avg_pages_viewed     = ("pages_viewed_before_purchase","mean"),
            avg_delivery_days    = ("delivery_days",               "mean"),
            avg_discount_pct     = ("discount_pct",                "mean"),
            cancelled_orders     = ("order_status", lambda x: (x == "Cancelled").sum()),
            returned_orders      = ("returned",                    "sum"),
            repeat_rate          = ("is_repeat_customer",          "mean"),
        )
        .reset_index()
    )

    # ── Category diversity ──
    cat_diversity = (
        orders.groupby("customer_id")["category"]
        .nunique()
        .reset_index()
        .rename(columns={"category": "categories_purchased"})
    )

    # ── Merge all into master ──
    master = (
        customers
        .merge(rfm_raw,      on="customer_id", how="left")
        .merge(order_stats,  on="customer_id", how="left")
        .merge(cat_diversity,on="customer_id", how="left")
    )

    # ── Derived features ──
    master["cancel_rate"]     = master["cancelled_orders"] / master["frequency"].clip(lower=1)
    master["return_rate_pct"] = master["returned_orders"]  / master["frequency"].clip(lower=1)
    master["tenure_days"]     = (pd.Timestamp.today() - master["registration_date"]).dt.days

    # ── RFM scoring (1–5) ──
    for col, ascending in [("recency", False), ("frequency", True), ("monetary", True)]:
        label = f"rfm_{col[0]}"
        ranked = master[col].rank(method="first")
        master[label] = pd.qcut(
            ranked, q=5, labels=[1, 2, 3, 4, 5]
        ).cat.add_categories([0]).fillna(0).astype(int)
        if not ascending:
            master[label] = master[label].apply(lambda v: 6 - v if v > 0 else 0)

    master["rfm_score"] = master["rfm_r"] + master["rfm_f"] + master["rfm_m"]

    # ── Customer segment label ──
    def segment_label(row):
        r, f, m = row["rfm_r"], row["rfm_f"], row["rfm_m"]
        if r >= 4 and f >= 4 and m >= 4:   return "Champions"
        elif r >= 3 and f >= 3:             return "Loyal Customers"
        elif r >= 4 and f <= 2:             return "Promising"
        elif r >= 3 and f <= 2:             return "Potential Loyalist"
        elif r <= 2 and f >= 3:             return "At Risk"
        elif r <= 2 and f <= 2 and m >= 3:  return "Cant Lose Them"
        elif r == 1 and f == 1:             return "Lost"
        else:                               return "Needs Attention"

    master["rfm_segment"] = master.apply(segment_label, axis=1)

    return master


# ── Monthly revenue (for forecasting) ─────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_monthly():
    _, _, monthly, _ = load_raw()
    monthly["date"] = pd.to_datetime(
        monthly["year"].astype(str) + "-" + monthly["month"].astype(str).str.zfill(2) + "-01"
    )
    return monthly.sort_values("date").reset_index(drop=True)


# ── Helpers ────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_orders():
    _, orders, _, _ = load_raw()
    return orders


@st.cache_data(show_spinner=False)
def load_products():
    _, _, _, products = load_raw()
    return products


def kpis(orders_df, customers_df):
    """Return a dict of headline KPIs for the overview page."""
    return {
        "total_revenue":  orders_df["total_amount_usd"].sum(),
        "total_orders":   len(orders_df),
        "total_customers":customers_df["customer_id"].nunique(),
        "aov":            orders_df["total_amount_usd"].mean(),
        "churn_rate":     customers_df["churned"].mean(),
        "return_rate":    (orders_df["returned"].sum() / len(orders_df)),
        "avg_delivery":   orders_df["delivery_days"].mean(),
        "repeat_rate":    orders_df["is_repeat_customer"].mean(),
    }
