import streamlit as st
import numpy as np
import pandas as pd
import pickle
import json
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title="Investment Risk Prediction", page_icon="📈", layout="wide")

@st.cache_resource
def load_model():
    with open("risk_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("feature_cols.json", "r") as f:
        feature_cols = json.load(f)
    return model, feature_cols

model, feature_cols = load_model()

green = "#0fbf78"
light_green = "#d8ffe6"

sns.set_style("whitegrid")

# --------------------------------------
# SIDEBAR
# --------------------------------------
st.sidebar.title("Input Features")
st.sidebar.write("Enter financial indicators to predict risk")

user_input = {}

for feat in feature_cols:
    val = st.sidebar.number_input(f"{feat}", value=0.0)
    user_input[feat] = val

st.title("Investment Risk Prediction System")
st.write("A classical machine learning system for predicting investment risk levels.")

# --------------------------------------
# PREDICTION SECTION
# --------------------------------------
if st.button("Predict Risk"):
    X = np.array([[user_input[f] for f in feature_cols]])
    pred = model.predict(X)[0]

    risk_map = {
        0: "🟢 LOW RISK",
        1: "🟡 MEDIUM RISK",
        2: "🔴 HIGH RISK",
    }

    st.subheader("Prediction Result")
    st.success(f"Predicted Risk Level: **{risk_map[int(pred)]}**")

st.header("Feature Importance")

if hasattr(model, "feature_importances_"):
    importance_df = pd.DataFrame({
        "Feature": feature_cols,
        "Importance": model.feature_importances_
    }).sort_values("Importance", ascending=False)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.barplot(
        data=importance_df,
        x="Importance",
        y="Feature",
        palette=[green, light_green]
    )
    ax.set_title("Random Forest Feature Importance", color=green)
    st.pyplot(fig)
else:
    st.info("Feature importance is not available for this model.")
