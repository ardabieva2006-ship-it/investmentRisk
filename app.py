import streamlit as st
import pickle
import json
import numpy as np

# ---------------------------
# Load Model and Feature List
# ---------------------------
@st.cache_resource
def load_model():
    with open("risk_model.pkl", "rb") as f:
        model = pickle.load(f)

    with open("feature_cols.json", "r") as f:
        feature_cols = json.load(f)

    return model, feature_cols

model, feature_cols = load_model()

st.title("📈 Investment Risk Prediction App")
st.write("Enter market indicators below to estimate the risk level.")

# ---------------------------
# Input form for all features
# ---------------------------
inputs = {}

st.subheader("Input Features")

for feature in feature_cols:
    inputs[feature] = st.number_input(
        f"{feature}",
        value=0.0,
        format="%.6f"
    )

# Convert to correct order
X = np.array([[inputs[f] for f in feature_cols]])

# ---------------------------
# Make Prediction
# ---------------------------
if st.button("Predict Risk"):
    try:
        pred = model.predict(X)[0]

        risk_map = {
            0: "🟢 LOW RISK",
            1: "🟡 MEDIUM RISK",
            2: "🔴 HIGH RISK"
        }

        st.success(f"Prediction: {risk_map.get(pred, pred)}")

    except Exception as e:
        st.error(f"Error: {e}")
