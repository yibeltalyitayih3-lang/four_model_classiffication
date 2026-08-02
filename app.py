"""
Teff Grain Filling Period Classifier — Streamlit Deployment App

Deploy on Streamlit Cloud:
    Main file path: app.py

Required files in repo root:
    - app.py
    - requirements.txt
    - teff.xlsx (optional, for data overview)
    - xgb_teff_model.pkl
    - features.pkl
    - label_encoder.pkl
"""

import pickle
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

ROOT = Path(__file__).resolve().parent

st.set_page_config(
    page_title="Teff Grain Filling Classifier",
    page_icon="🌾",
    layout="wide",
)

TARGET_LABELS = {1: "Short", 2: "Medium", 3: "Long"}
CLASS_DESCRIPTIONS = {
    1: "Short grain filling period",
    2: "Medium grain filling period",
    3: "Long grain filling period",
}


@st.cache_resource
def load_artifacts():
    model = joblib.load(ROOT / "xgb_teff_model.pkl")
    label_encoder = joblib.load(ROOT / "label_encoder.pkl")
    with open(ROOT / "features.pkl", "rb") as f:
        features = pickle.load(f)
    return model, label_encoder, features


@st.cache_data
def load_dataset():
    data_path = ROOT / "teff.xlsx"
    if not data_path.exists():
        data_path = ROOT / "data" / "teff.xlsx"
    if not data_path.exists():
        return None
    return pd.read_excel(data_path)


def dms_to_decimal(value):
    import re

    if pd.isna(value):
        return np.nan
    if isinstance(value, (int, float)):
        return float(value)
    s = str(value).strip()
    match = re.match(r"(\d+)-(\d+)-(\d+)-([NSEW])", s, re.IGNORECASE)
    if not match:
        try:
            return float(s)
        except ValueError:
            return np.nan
    deg, minutes, seconds, direction = match.groups()
    decimal = int(deg) + int(minutes) / 60 + int(seconds) / 3600
    if direction.upper() in ("S", "W"):
        decimal = -decimal
    return decimal


def clean_data(df):
    df = df.copy()
    for col in ("Lat", "Long"):
        if col in df.columns:
            df[col] = df[col].apply(dms_to_decimal)
    bio_cols = [c for c in df.columns if c.lower().startswith("bio")]
    numeric_candidates = [
        "Edate", "Hdate", "Mdate", "PL", "NV", "NVc", "Plc",
        "PCMdate", "PCHdate", "SBP", "SC", "Edatec", "Hdatec", "Mdatec",
        "Lat", "Long", "Altitude", "ph_ho_60cm", "ph_kcl_60cm",
    ] + bio_cols
    for col in numeric_candidates:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "Gfpc" in df.columns:
        df = df.dropna(subset=["Gfpc"])
        df["Gfpc"] = df["Gfpc"].astype(int)
    return df


st.sidebar.title("🌾 Teff Classifier")
page = st.sidebar.radio(
    "Navigation",
    ["Home", "Data Overview", "EDA", "Model Info", "Predict"],
)

model, label_encoder, features = load_artifacts()
numeric_features = features["numeric_features"]
df_raw = load_dataset()
df = clean_data(df_raw) if df_raw is not None else None

if page == "Home":
    st.title("🌾 Teff Grain Filling Period Classifier")
    st.markdown(
        "Classification of **grain filling period** for teff accessions "
        "into **Short**, **Medium**, and **Long** classes using phenological, "
        "geographic, soil, and bioclimatic features."
    )
    c1, c2, c3 = st.columns(3)
    if df is not None:
        c1.metric("Samples", f"{len(df):,}")
        c2.metric("Features", len(numeric_features))
        c3.metric("Classes", df["Gfpc"].nunique())
    else:
        c1.metric("Features", len(numeric_features))
        c2.metric("Model", "XGBoost")
        c3.metric("Target", "Gfpc")

    st.success("Best model: **XGBoost** — Accuracy **> 97%** (CV)")

    if df is not None:
        counts = df["Gfpc"].value_counts().sort_index()
        fig, ax = plt.subplots(figsize=(8, 5))
        labels = [TARGET_LABELS[i] for i in counts.index]
        ax.bar(labels, counts.values, color=["#e74c3c", "#3498db", "#2ecc71"])
        ax.set_title("Class Distribution — Grain Filling Period")
        ax.set_ylabel("Count")
        st.pyplot(fig)
        plt.close()

elif page == "Data Overview":
    st.title("📊 Data Overview")
    if df is None:
        st.warning("Upload `teff.xlsx` to the repo root or `data/` folder for data views.")
    else:
        st.markdown(f"**Shape:** {df.shape[0]} rows × {df.shape[1]} columns")
        tab1, tab2, tab3 = st.tabs(["Preview", "Summary", "Missing Values"])
        with tab1:
            st.dataframe(df.head(20), use_container_width=True)
        with tab2:
            st.dataframe(df.describe().T, use_container_width=True)
        with tab3:
            nulls = df.isnull().sum()
            nulls = nulls[nulls > 0].sort_values(ascending=False)
            st.dataframe(
                pd.DataFrame({"Column": nulls.index, "Missing": nulls.values}),
                use_container_width=True,
            )

elif page == "EDA":
    st.title("🔍 Exploratory Data Analysis")
    if df is None:
        st.warning("Dataset not found. Add `teff.xlsx` for EDA visualizations.")
    else:
        tab1, tab2, tab3 = st.tabs(["Class Distribution", "Correlation", "Null Values"])
        with tab1:
            counts = df["Gfpc"].value_counts().sort_index()
            fig, ax = plt.subplots(figsize=(8, 5))
            labels = [TARGET_LABELS[i] for i in counts.index]
            bars = ax.bar(labels, counts.values, color=["#e74c3c", "#3498db", "#2ecc71"])
            for bar, count in zip(bars, counts.values):
                ax.text(
                    bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                    f"{count}\n({count/len(df)*100:.1f}%)", ha="center",
                )
            ax.set_title("Class Distribution")
            st.pyplot(fig)
            plt.close()

        with tab2:
            numeric_df = df[numeric_features + ["Gfpc"]].select_dtypes(include=[np.number])
            corr = numeric_df.corr()
            top = corr["Gfpc"].abs().sort_values(ascending=False).head(16).index
            fig, ax = plt.subplots(figsize=(12, 10))
            sns.heatmap(
                numeric_df[top].corr(), annot=True, fmt=".2f",
                cmap="RdBu_r", center=0, ax=ax,
            )
            ax.set_title("Correlation Matrix (Top Features vs Gfpc)")
            st.pyplot(fig)
            plt.close()

        with tab3:
            nulls = df.isnull().sum()
            nulls = nulls[nulls > 0].sort_values()
            fig, ax = plt.subplots(figsize=(10, max(4, len(nulls) * 0.3)))
            if len(nulls):
                nulls.plot.barh(ax=ax, color="#e67e22")
            ax.set_title("Missing Values per Column")
            st.pyplot(fig)
            plt.close()

elif page == "Model Info":
    st.title("🤖 Model Information")
    st.markdown("""
    | Model | Test Accuracy | CV Accuracy |
    |-------|---------------|-------------|
    | Logistic Regression | 98.1% | 94.5% |
    | **XGBoost** | **98.1%** | **97.1%** |
    | Random Forest | 86.3% | 74.5% |
    | SVM | 86.6% | 80.4% |

    **Deployed model:** XGBoost (best cross-validation performance)

    **Metrics:** Accuracy, Precision, Recall, F1-Score, Confusion Matrix, ROC-AUC
    """)
    artifacts = ROOT / "artifacts"
    if artifacts.exists():
        for img in sorted(artifacts.glob("*.png")):
            st.image(str(img), caption=img.stem.replace("_", " ").title())

elif page == "Predict":
    st.title("🔮 Predict Grain Filling Period")
    st.markdown("Enter phenological and environmental features to classify the grain filling period.")

    input_data = {}
    col1, col2 = st.columns(2)

    pheno = ["Edate", "Hdate", "Mdate", "PL", "NV", "PCMdate", "PCHdate", "SBP", "SC"]
    geo = ["Lat", "Long", "Altitude", "ph_ho_60cm", "ph_kcl_60cm"]

    with col1:
        st.subheader("Phenological")
        for feat in pheno:
            if feat in numeric_features:
                default = float(df[feat].median()) if df is not None else 0.0
                input_data[feat] = st.number_input(feat, value=default, key=feat)

    with col2:
        st.subheader("Geographic & Soil")
        for feat in geo:
            if feat in numeric_features:
                default = float(df[feat].median()) if df is not None else 0.0
                input_data[feat] = st.number_input(feat, value=default, key=feat)

    for feat in numeric_features:
        if feat not in input_data:
            default = float(df[feat].median()) if df is not None else 0.0
            input_data[feat] = default

    if st.button("Predict", type="primary"):
        input_df = pd.DataFrame([input_data])
        pred_encoded = model.predict(input_df)[0]
        probas = model.predict_proba(input_df)[0]
        pred_class = int(label_encoder.inverse_transform([pred_encoded])[0])
        pred_label = TARGET_LABELS.get(pred_class, str(pred_class))

        st.success(f"**Predicted Class:** {pred_class} — {pred_label}")
        st.caption(CLASS_DESCRIPTIONS.get(pred_class, ""))

        prob_df = pd.DataFrame({
            "Class": [TARGET_LABELS.get(int(c), str(c)) for c in label_encoder.classes_],
            "Probability": probas,
        })
        st.bar_chart(prob_df.set_index("Class"))
