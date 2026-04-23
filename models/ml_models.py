"""
ml_models.py
------------
Self-contained ML experiments.  Each function trains (or loads) a model
and returns predictions + artefacts for the dashboard pages.

All heavy models are cached so they only train once per session.
"""

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.metrics import (
    classification_report, roc_auc_score, confusion_matrix,
    silhouette_score
)
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings("ignore")


# ══════════════════════════════════════════════════════════════════════════════
# 1. CHURN PREDICTION
# ══════════════════════════════════════════════════════════════════════════════

CHURN_FEATURES = [
    "recency", "frequency", "monetary",
    "avg_session_minutes", "avg_pages_viewed",
    "cancel_rate", "return_rate_pct",
    "avg_discount_pct", "days_since_last_purchase",
    "reviews_given", "avg_review_score",
    "wishlist_items", "newsletter_subscribed",
    "tenure_days", "categories_purchased",
]


@st.cache_resource(show_spinner=False)
def train_churn_model(master_df: pd.DataFrame):
    """
    Train a Gradient Boosting classifier to predict churn.
    Returns: model, X_test, y_test, feature_names, metrics_dict
    """
    df = master_df.dropna(subset=CHURN_FEATURES + ["churned"])
    X = df[CHURN_FEATURES]
    y = df["churned"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = GradientBoostingClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        subsample=0.8, random_state=42
    )
    model.fit(X_train, y_train)

    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    report = classification_report(y_test, y_pred, output_dict=True)
    metrics = {
        "auc":       round(roc_auc_score(y_test, y_proba), 4),
        "accuracy":  round(report["accuracy"], 4),
        "precision": round(report["1"]["precision"], 4),
        "recall":    round(report["1"]["recall"], 4),
        "f1":        round(report["1"]["f1-score"], 4),
        "conf_matrix": confusion_matrix(y_test, y_pred),
    }

    importances = pd.Series(model.feature_importances_, index=CHURN_FEATURES)
    importances = importances.sort_values(ascending=True)

    return model, X_test, y_test, y_proba, importances, metrics


def predict_churn_proba(model, input_dict: dict) -> float:
    """Predict churn probability for a single customer profile dict."""
    row = pd.DataFrame([{f: input_dict.get(f, 0) for f in CHURN_FEATURES}])
    return float(model.predict_proba(row)[0, 1])


# ══════════════════════════════════════════════════════════════════════════════
# 2. CUSTOMER SEGMENTATION (RFM + K-Means)
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner=False)
def train_segmentation(master_df: pd.DataFrame, n_clusters: int = 5):
    """
    K-Means clustering on scaled RFM features.
    Returns: master_df with 'cluster' column, inertia list, silhouette score
    """
    df = master_df[["customer_id", "recency", "frequency", "monetary"]].dropna().copy()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[["recency", "frequency", "monetary"]])

    # Elbow analysis
    inertias = []
    sil_scores = []
    for k in range(2, 9):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X_scaled)
        inertias.append(km.inertia_)
        sil_scores.append(silhouette_score(X_scaled, km.labels_))

    # Final model
    final_km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df["cluster"] = final_km.fit_predict(X_scaled)

    sil = round(silhouette_score(X_scaled, df["cluster"]), 4)

    # Cluster profiles
    profile = (
        df.groupby("cluster")[["recency", "frequency", "monetary"]]
        .mean()
        .round(1)
        .sort_values("monetary", ascending=False)
        .reset_index()
    )

    # Descriptive labels based on RFM means
    def cluster_label(row):
        if row["monetary"] >= profile["monetary"].quantile(0.75) and row["frequency"] >= profile["frequency"].median():
            return "VIP"
        elif row["frequency"] >= profile["frequency"].median():
            return "Regulars"
        elif row["recency"] <= profile["recency"].quantile(0.33):
            return "Recent Buyers"
        elif row["recency"] >= profile["recency"].quantile(0.75):
            return "Inactive"
        else:
            return "Occasionals"

    profile["label"] = profile.apply(cluster_label, axis=1)
    df = df.merge(profile[["cluster", "label"]], on="cluster")

    return df, inertias, sil_scores, sil, profile


# ══════════════════════════════════════════════════════════════════════════════
# 3. REVENUE FORECASTING (Manual Exponential Smoothing + trend)
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def forecast_revenue(monthly_df: pd.DataFrame, periods: int = 12):
    """
    Holt-Winters style double exponential smoothing (no Prophet dependency).
    Returns: historical df + forecast df with confidence intervals.
    """
    hist = monthly_df[monthly_df["date"] <= "2026-03-01"].copy()
    revenue = hist["revenue_usd"].values.astype(float)
    n = len(revenue)

    # Double exponential smoothing (Holt's method)
    alpha, beta = 0.3, 0.1
    level = np.zeros(n)
    trend = np.zeros(n)
    level[0] = revenue[0]
    trend[0] = revenue[1] - revenue[0]

    for t in range(1, n):
        level[t] = alpha * revenue[t] + (1 - alpha) * (level[t-1] + trend[t-1])
        trend[t] = beta * (level[t] - level[t-1]) + (1 - beta) * trend[t-1]

    # In-sample fit
    fitted = level + trend

    # Forecast
    last_date = hist["date"].max()
    future_dates = pd.date_range(start=last_date + pd.offsets.MonthBegin(1), periods=periods, freq="MS")
    forecast_vals = np.array([level[-1] + (i+1)*trend[-1] for i in range(periods)])
    # Add slight seasonality bump (heuristic from data)
    seasonal = np.tile([0.98, 0.95, 0.97, 1.0, 1.01, 1.03, 1.02, 1.05, 1.0, 1.08, 1.12, 1.15], 2)[:periods]
    forecast_vals = forecast_vals * seasonal

    # RMSE on last 12 months as proxy for CI width
    last12_err = revenue[-12:] - fitted[-12:]
    std_err = np.std(last12_err)

    forecast_df = pd.DataFrame({
        "date":        future_dates,
        "revenue_usd": forecast_vals,
        "lower":       forecast_vals - 1.96 * std_err,
        "upper":       forecast_vals + 1.96 * std_err,
        "type":        "Forecast",
    })

    hist["type"]  = "Historical"
    hist["lower"] = np.nan
    hist["upper"] = np.nan

    rmse = np.sqrt(np.mean((revenue[-12:] - fitted[-12:])**2))
    mape = np.mean(np.abs((revenue[-12:] - fitted[-12:]) / revenue[-12:])) * 100

    metrics = {
        "rmse": round(rmse, 2),
        "mape": round(mape, 2),
        "trend_monthly": round(float(trend[-1]), 2),
    }

    return hist, forecast_df, metrics
