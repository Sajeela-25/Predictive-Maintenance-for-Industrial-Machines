# app.py

import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Load model and scaler
model = joblib.load("rf_model.pkl")
scaler = joblib.load("scaler.pkl")

st.set_page_config(page_title="Predictive Maintenance System", layout="wide")

st.title("🔧 Predictive Maintenance for Industrial Machines")

st.write("""
This application predicts whether a machine is **likely to fail** based on sensor data.
Upload a dataset or enter values manually to get predictions.
""")

# -----------------------------
# Upload Data
# -----------------------------
uploaded_file = st.file_uploader("Upload AI4I Sensor Data CSV", type=["csv"])

if uploaded_file:
    data = pd.read_csv(uploaded_file)

    st.subheader("📊 Dataset Preview")
    st.dataframe(data.head())

    # Rename columns
    data.rename(columns={
        "Air temperature [K]": "air_temp",
        "Process temperature [K]": "process_temp",
        "Rotational speed [rpm]": "rpm",
        "Torque [Nm]": "torque",
        "Tool wear [min]": "tool_wear"
    }, inplace=True)

    features = ["air_temp", "process_temp", "rpm", "torque", "tool_wear"]

    X = data[features]
    X_scaled = scaler.transform(X)

    predictions = model.predict(X_scaled)
    data["Predicted Failure"] = predictions

    st.subheader("⚠️ Prediction Results")
    st.dataframe(data[features + ["Predicted Failure"]])

    # -----------------------------
    # Visualizations
    # -----------------------------
    st.subheader("📈 Feature Distributions")
    selected_feature = st.selectbox("Select Feature", features)
    fig, ax = plt.subplots(figsize=(4, 3))
    sns.histplot(data[selected_feature], kde=True, ax=ax)
    col1, col2 = st.columns(2)
    with col1:
        st.pyplot(fig)

    st.subheader("📊 Correlation Heatmap")
    fig2, ax = plt.subplots(figsize=(8,7))
    sns.heatmap(data[features].corr(), annot=True, cmap="coolwarm", ax=ax)
    col1, col2 = st.columns(2)
    with col1:
        st.pyplot(fig2)

    st.subheader("⭐ Feature Importance")
    fig3, ax3 = plt.subplots(figsize=(8,7))
    ax3.bar(features, model.feature_importances_)
    col1, col2 = st.columns(2)
    with col1:
        st.pyplot(fig3)

else:
    st.info("Please upload the AI4I 2020 dataset CSV file to continue.")
